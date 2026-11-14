"""
DIGM REAL eviction inside vLLM 0.30 (V1 engine, V2 model runner) — prototype, stage 1
=====================================================================================
Research prototype. In-process engine, single GPU. This is the exact script behind the paper's serving results.

After a request's prompt is prefilled, DIGM picks the tokens to keep (layer-l* key norms,
standardised, low = keep, max-pool 5, 4 sinks, recency = 50% of the budget; the question
part of the prompt is always kept), and then, between two engine steps:
  1. KV compaction : in EVERY layer, the kept tokens' K/V are copied to the first k slots of
                     the request's own blocks (one shared selection = fits vLLM's single block
                     table; this is "DIGM-one": layer l*'s choice applied to all layers).
  2. Scheduler     : the request now has k computed tokens; its tail blocks are FREED back to
                     vLLM's block pool (real memory returned to the engine).
  3. Worker        : token count, prefill length, block-table length updated to k.
  4. RoPE fix      : new tokens keep their ORIGINAL positions (n, n+1, ...) via a per-request
                     offset added to positions after slot mapping (slots/lengths use k).

Self-checks printed per question:
  A  Full KV (no eviction)                      -> baseline answer
  B  identity eviction (keep ALL, offset 0)     -> must equal A exactly (machinery is correct)
  C  DIGM-one, keep 25% of the context          -> accuracy + freed blocks
  D  same as C but WITHOUT the RoPE fix (first 5 questions only) -> shows the fix matters

  python3 digm_vllm_v18.py                                       (Llama-3.2-3B AWQ, l*=4)
  python3 digm_vllm_v18.py Qwen/Qwen2.5-3B-Instruct-AWQ 1
  args: model  layer  gpu_util(0.8)  questions_per_task(2)

STAGE 2 (throughput):  python3 digm_vllm_v18.py --throughput [model layer gpu_util]
  32 RULER prompts (~3.7K tokens each) are submitted at once; each generates 128 tokens.
  Full KV vs DIGM (each request evicted to ~25% right after its prefill). Measures wall time,
  generated tokens/s, the most requests vLLM ran at once, and the cost of each eviction.
"""
import os, sys, json, time, random
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

THROUGHPUT = "--throughput" in sys.argv
if THROUGHPUT:
    sys.argv.remove("--throughput")
SERVE = "--serve" in sys.argv          # online serving: Poisson arrivals, TTFT / TPOT / latency at several request rates
if SERVE:
    sys.argv.remove("--serve")
STRESS = "--stress" in sys.argv        # oversubscribed pool: preemption after eviction, naive vs restore, with accuracy
if STRESS:
    sys.argv.remove("--stress")
LONG = "--long" in sys.argv            # eviction DURING chunked prefill: prompts longer than the KV pool (RULER 16K)
if LONG:
    sys.argv.remove("--long")
LONGBENCH = "--longbench" in sys.argv  # accuracy on LongBench (English, kvpress format): Full KV vs eviction at a fixed budget
if LONGBENCH:
    sys.argv.remove("--longbench")
BUDGET = int(os.environ.get("DIGM_BUDGET", "0"))   # >0: keep this many context tokens (fixed) instead of KEEP x context
CHUNK = int(os.environ.get("DIGM_CHUNK", "1024"))  # --long: prefill chunk size

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "AMead10/Llama-3.2-3B-Instruct-AWQ"
LSTAR = int(sys.argv[2]) if len(sys.argv) > 2 else 4
GPU_UTIL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
PER_TASK = int(sys.argv[4]) if len(sys.argv) > 4 else 2
KEEP, N_SINK, RECENT_FRAC, POOL, STRICT_SLACK = float(os.environ.get("DIGM_KEEP", "0.25")), 4, 0.5, 5, 5
DEBUG = os.environ.get("DIGM_DEBUG", "1") == "1"
SEL = os.environ.get("DIGM_SEL", "one")      # "one" = DIGM-one (layer l*) | "<kind>/<fusion>" = consensus, e.g. cos/vote
                                             # | "head:<kind>" = COMPOSITE: every (layer, head) keeps its own top-k, e.g. head:cos
HEAD_SINKS = os.environ.get("DIGM_HEAD_SINKS", "0") == "1"   # per-head mode: also force the first 4 tokens
EAGER = os.environ.get("DIGM_EAGER", "1") == "1"             # 0 = let vLLM use CUDA graphs (production default)
SPLIT = os.environ.get("DIGM_SPLIT", "dev")                  # stage 1: "dev" rows[:PER_TASK] or "heldout" rows[10:10+PER_TASK]
PERM = os.environ.get("DIGM_PERM", "1") == "1"               # stage 1: also run the PERMUTATION check (composite exactness)

cfg = AutoConfig.from_pretrained(MODEL)
H = cfg.num_key_value_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
extra = {"quantization": "awq"} if "awq" in MODEL.lower() else {}
if LONG:
    # a request may be LONGER than the whole KV pool: it is evicted chunk by chunk, so it never needs it all.
    # vLLM refuses such a max_model_len at start-up; relax only that check (in-process engine).
    try:                                                       # vLLM >= 0.33 moved this check to kv_cache_planning
        import vllm.v1.core.kv_cache_planning as _kvu
        _kvu._check_enough_kv_cache_memory
    except (ImportError, AttributeError):                      # vLLM 0.30
        import vllm.v1.core.kv_cache_utils as _kvu
    _orig_check = _kvu._check_enough_kv_cache_memory
    def _relaxed_check(available_memory, get_needed_memory, max_model_len, estimate_max_model_len):
        if available_memory <= 0:
            return _orig_check(available_memory, get_needed_memory, max_model_len, estimate_max_model_len)
    _kvu._check_enough_kv_cache_memory = _relaxed_check
    extra["max_num_batched_tokens"] = CHUNK
if os.environ.get("DIGM_POOL_BLOCKS"):                       # emulate a small GPU (any mode)
    extra["num_gpu_blocks_override"] = int(os.environ["DIGM_POOL_BLOCKS"])
try:
    llm = LLM(model=MODEL, gpu_memory_utilization=GPU_UTIL, enforce_eager=EAGER, max_model_len=int(os.environ.get("DIGM_MAXLEN", "8192")),
              enable_prefix_caching=False, async_scheduling=False, **extra)
except TypeError:
    llm = LLM(model=MODEL, gpu_memory_utilization=GPU_UTIL, enforce_eager=EAGER, max_model_len=int(os.environ.get("DIGM_MAXLEN", "8192")),
              enable_prefix_caching=False, **extra)
engine = llm.llm_engine
core = engine.engine_core.engine_core
runner = core.model_executor.driver_worker.worker.model_runner
if LONG:
    core.scheduler.scheduler_reserve_full_isl = False         # admit a prompt even if it would not fit in full
sched = core.scheduler
kvm = sched.kv_cache_manager
mgr = kvm.coordinator.single_type_managers[0]
pool = kvm.block_pool
kv = runner.kv_caches
L = len(kv)
shp = tuple(kv[0].shape)
assert len(shp) == 4 and shp[1] == H and shp[-1] == 2 * D, f"unexpected KV layout {shp}"
BS = shp[2]
dev = kv[0].device
tok = llm.get_tokenizer()
USE_EVICT_API = hasattr(kvm, "evict_token_range_for_request") and os.environ.get("DIGM_EVICT_API", "1") == "1"
def free_tail_blocks(rid, keep_nb):
    """Free a live request's blocks from index keep_nb on. Returns blocks returned to the pool.
    Uses vLLM's evict_token_range_for_request when the engine has it (PR #53871);
    otherwise frees the tail blocks directly (vLLM 0.30 / 0.33)."""
    blocks = mgr.req_to_blocks[rid]
    before = pool.get_num_free_blocks()
    if USE_EVICT_API:
        a, b, _ = kvm.evict_token_range_for_request(rid, keep_nb * BS, len(blocks) * BS)
        assert a == keep_nb * BS, (a, keep_nb * BS)
    else:
        tail = blocks[keep_nb:]; del blocks[keep_nb:]
        if rid in mgr.num_cached_block:
            mgr.num_cached_block[rid] = min(mgr.num_cached_block[rid], keep_nb)
        pool.free_blocks(reversed(tail))
    return pool.get_num_free_blocks() - before
print(f"\n[vLLM] {MODEL} | layers {L} | kv heads {H} | D {D} | block {BS} | l*={LSTAR} | free blocks {pool.get_num_free_blocks()} | evict API {USE_EVICT_API}")

# ---------------- RoPE fix: per-request position offset, applied after slot mapping ----------------
OFFSET = torch.zeros(runner.req_states.max_num_reqs, dtype=torch.int64, device=dev)
_orig_prepare_attn = runner.prepare_attn
def _prepare_attn_with_offset(input_batch):
    out = _orig_prepare_attn(input_batch)                      # slots + lengths use compact indices
    nr = input_batch.num_reqs
    qsl = input_batch.query_start_loc[: nr + 1].long()
    qlen = qsl[1:] - qsl[:-1]
    tok_req = torch.repeat_interleave(input_batch.idx_mapping[:nr].long(), qlen)
    input_batch.positions[: tok_req.numel()] += OFFSET[tok_req]   # RoPE sees original positions
    return out
runner.prepare_attn = _prepare_attn_with_offset

# ---------------- helpers ----------------
def find_rid(tag):
    return next((r for r in sched.requests if tag in str(r)), None)

def ctx_budget(n_ctx, lo):
    b = min(BUDGET, n_ctx) if BUDGET > 0 else int(n_ctx * KEEP)
    return max(b, lo)

def slots_of(block_ids, idx):
    mx = int(idx.max()) if idx.numel() else 0
    if mx // BS >= len(block_ids):                       # clear error instead of a CUDA index assert
        raise RuntimeError(f"slot out of range: token {mx} needs block {mx // BS}, request has {len(block_ids)} blocks "
                           f"(= {len(block_ids) * BS} slots)")
    b = torch.as_tensor(block_ids, device=dev, dtype=torch.long)
    return b[idx // BS], idx % BS

def digm_keep(block_ids, n_ctx, n):
    """DIGM-one keep set over prompt positions [0, n): context scored at layer l*, question kept."""
    t = torch.arange(n_ctx, device=dev)
    b, o = slots_of(block_ids, t)
    k = kv[LSTAR][b, :, o, :D]                                 # (n_ctx, H, D) keys of layer l*
    nrm = k.float().norm(dim=-1).mean(dim=-1)
    s = -(nrm - nrm.mean()) / (nrm.std() + 1e-6)
    s = F.max_pool1d(s.view(1, 1, -1), POOL, 1, POOL // 2).view(-1)
    budget = ctx_budget(n_ctx, N_SINK + 1)
    s[:N_SINK] = float("inf")
    s[-max(int(budget * RECENT_FRAC), 1):] = float("inf")
    keep_ctx = torch.topk(s, budget).indices
    keep = torch.cat([keep_ctx, torch.arange(n_ctx, n, device=dev)])
    return torch.sort(keep).values

def list_scores(k, v, kind):                                  # identical to digm_consensus_dev.py
    if kind == "cos":
        return -F.cosine_similarity(k, F.normalize(k, p=2, dim=-1).mean(1, keepdim=True), dim=-1)
    if kind == "cnorm":
        return (k - k.mean(1, keepdim=True)).norm(dim=-1)
    if kind == "knorm":
        return -k.norm(dim=-1)
    if kind == "vk":
        return v.norm(dim=-1) / (k.norm(dim=-1) + 1e-6)
    if kind == "cosv":
        zs = lambda x: (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
        return zs(list_scores(k, v, "cos")) + zs(v.norm(dim=-1).clamp_min(1e-6).log())
    raise ValueError(kind)

def perhead_keep(block_ids, n_ctx, n):
    """COMPOSITE selection: each (layer, head) keeps its own top-k context tokens (same k everywhere) + the question.
    Returns (L, H, k_total) prompt positions. Same k in every layer/head -> one block table still fits."""
    kind = SEL.split(":", 1)[1]
    t = torch.arange(n_ctx, device=dev)
    b, o = slots_of(block_ids, t)
    budget = ctx_budget(n_ctx, 1)
    q = torch.arange(n_ctx, n, device=dev)
    out = []
    for l in range(L):
        x = kv[l][b, :, o, :]                                  # (n_ctx, H, 2D)
        s = list_scores(x[..., :D].transpose(0, 1).float(), x[..., D:].transpose(0, 1).float(), kind)   # (H, n_ctx)
        if HEAD_SINKS:
            s[:, :N_SINK] = float("inf")
        if PROXY_ON:
            _proxy_add(l, s)
        idx = s.topk(budget, dim=-1).indices.sort(dim=-1).values
        out.append(torch.cat([idx, q.unsqueeze(0).expand(H, -1)], dim=-1))
    return torch.stack(out)                                    # (L, H, k_total)

# ---- sensitivity probe: query-free statistics of a request's score distribution (logged per request) ----
PROXY_ON = os.environ.get("DIGM_PROXY", "0") == "1"
LAST_PROXY = {}
def _proxy_add(l, s):
    """s: (H, n) scores of one layer (higher = keep). Per-head statistics, averaged; stored per layer group."""
    s = s.float(); n = s.shape[-1]
    if l == 0:
        LAST_PROXY.clear(); LAST_PROXY["n_ctx"] = n; LAST_PROXY["_acc"] = {}
    k = max(n // 4, 1)
    srt = s.sort(dim=-1, descending=True).values
    mu, sd = s.mean(-1), s.std(-1) + 1e-6
    z = (s - mu.unsqueeze(-1)) / sd.unsqueeze(-1)
    sh = s - s.min(-1, keepdim=True).values
    st = {"gap25": ((srt[:, :k].mean(-1) - srt[:, k:].mean(-1)) / sd).mean().item(),          # separation of the kept quarter
          "share25": (sh.sort(-1, descending=True).values[:, :k].sum(-1) / (sh.sum(-1) + 1e-6)).mean().item(),
          "outlier": (z > 2).float().mean().item(),                                           # fraction of very distinctive keys
          "kurt": ((z ** 4).mean(-1) - 3).mean().item(),
          "redund": (-s).mean().item(),                                                      # mean cosine to the mean key (KeyDiff)
          "margin": ((srt[:, k - 1] - srt[:, min(k, n - 1)]) / sd).mean().item()}           # how sharp the cut at 25% is
    grp = "early" if l < L // 3 else ("late" if l >= 2 * L // 3 else "mid")
    for key, v in st.items():
        LAST_PROXY["_acc"].setdefault(key, []).append(v)
        LAST_PROXY["_acc"].setdefault(f"{key}_{grp}", []).append(v)
    if l == L - 1:
        for key, v in LAST_PROXY.pop("_acc").items():
            LAST_PROXY[key] = sum(v) / len(v)

def perhead_scores(block_ids, n_ctx):
    """(L, H, n_ctx) query-free per-head scores of the context tokens currently held (higher = keep)."""
    kind = SEL.split(":", 1)[1]
    b, o = slots_of(block_ids, torch.arange(n_ctx, device=dev))
    out = []
    for l in range(L):
        x = kv[l][b, :, o, :]
        s = list_scores(x[..., :D].transpose(0, 1).float(), x[..., D:].transpose(0, 1).float(), kind)
        if HEAD_SINKS:
            s[:, :N_SINK] = float("inf")
        out.append(s.half() if s.dtype == torch.float32 and not HEAD_SINKS else s)
    return torch.stack(out)

def consensus_keep(block_ids, n_ctx, n):
    """DIGM-C: every (layer, head) scores the context from the paged cache; fused into ONE set."""
    kind, fusion = SEL.split("/")
    t = torch.arange(n_ctx, device=dev)
    b, o = slots_of(block_ids, t)
    budget = ctx_budget(n_ctx, N_SINK + 1)
    sum_z = torch.zeros(n_ctx, device=dev); vote = torch.zeros(n_ctx, device=dev)
    minr = torch.full((n_ctx,), n_ctx, device=dev, dtype=torch.long); M = 0
    ar = torch.arange(n_ctx, device=dev)
    for l in range(L):
        x = kv[l][b, :, o, :]                                  # (n_ctx, H, 2D)
        s = list_scores(x[..., :D].transpose(0, 1).float(), x[..., D:].transpose(0, 1).float(), kind)
        z = (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-6)
        rank = torch.empty_like(s, dtype=torch.long)
        rank.scatter_(-1, s.argsort(-1, descending=True), ar.expand_as(s).contiguous())
        sum_z += z.sum(0); vote += (rank < budget).sum(0); minr = torch.minimum(minr, rank.min(0).values); M += s.shape[0]
    zm = sum_z / M
    f = zm.clone() if fusion == "mean" else (vote + 0.01 * torch.tanh(zm) if fusion == "vote"
                                             else -minr.float() + 0.5 * vote / (M + 1))
    f[:N_SINK] = float("inf")
    keep = torch.cat([f.topk(budget).indices, torch.arange(n_ctx, n, device=dev)])
    return torch.sort(keep).values

def stream_keep(block_ids, n_ctx, n):
    """StreamingLLM baseline: 4 sinks + the most recent context tokens (+ the question)."""
    budget = ctx_budget(n_ctx, N_SINK + 1)
    keep = torch.cat([torch.arange(N_SINK, device=dev), torch.arange(n_ctx - (budget - N_SINK), n, device=dev)])
    return keep

def select_keep(block_ids, n_ctx, n):
    if SEL == "one":
        return digm_keep(block_ids, n_ctx, n)
    if SEL == "stream":
        return stream_keep(block_ids, n_ctx, n)
    return perhead_keep(block_ids, n_ctx, n) if SEL.startswith("head:") else consensus_keep(block_ids, n_ctx, n)

# ---------------- preemption: vLLM V1 recomputes a preempted request from its token ids ----------------
# After eviction those ids are the KEPT tokens only, so a naive recompute silently re-encodes a different,
# shorter text at the wrong positions. "restore" (default) puts the ORIGINAL prompt back at preemption time:
# vLLM then recomputes the full context exactly like an un-evicted request, and the request is evicted again
# after its re-prefill (same keys -> same selection). "naive" leaves it as is (to measure the damage).
PREEMPT_MODE = os.environ.get("DIGM_PREEMPT", "restore")
EVICTED = {}                                                    # request_id -> {"orig", "idx", "preempts", "needs"}
_orig_preempt = sched._preempt_request
def _preempt_with_restore(request, timestamp, *a, **kw):
    st = EVICTED.get(request.request_id)
    if st is not None:
        st["preempts"] += 1
        if st["idx"] is not None:
            OFFSET[st["idx"]] = 0                               # the worker slot is released; new slot starts at 0
            st["idx"] = None
        if PREEMPT_MODE == "restore":
            outs = list(request._output_token_ids)
            request._all_token_ids[:] = list(st["orig"]) + outs
            request.prompt_token_ids = list(st["orig"])
            request.num_prompt_tokens = len(st["orig"])
            st["needs"] = True                                  # evict again after the re-prefill
    return _orig_preempt(request, timestamp, *a, **kw)
sched._preempt_request = _preempt_with_restore

@torch.no_grad()
def evict(rid, keep, n, use_offset=True):
    """keep = prompt positions to keep (1-D shared set, or (L, H, k) composite). Outputs whose KV is already
    computed (e.g. replayed after a preemption) are always kept. All counters move by d = removed tokens."""
    req = sched.requests[rid]
    comp = int(req.num_computed_tokens)
    if comp > n:                                               # keep already-computed outputs, in order
        extra = torch.arange(n, comp, device=dev)
        keep = torch.cat([keep, extra]) if keep.dim() == 1 else \
            torch.cat([keep, extra.view(1, 1, -1).expand(keep.shape[0], keep.shape[1], -1)], dim=-1)
    k = int(keep.shape[-1])
    kp = k - max(comp - n, 0)                                  # kept PROMPT tokens
    block_ids = list(kvm.get_block_ids(rid)[0])
    if rid not in EVICTED:
        EVICTED[rid] = {"orig": list(req.prompt_token_ids), "idx": None, "preempts": 0, "needs": False}
    # 1) KV compaction in every layer (gather first, then write)
    db, do = slots_of(block_ids, torch.arange(k, device=dev))
    if keep.dim() == 1:                                        # one shared set: same tokens in every layer and head
        sb, so = slots_of(block_ids, keep)
        for l in range(L):
            tmp = kv[l][sb, :, so, :].clone()
            kv[l][db, :, do, :] = tmp
        pos0 = keep
    else:                                                      # composite: slot j of head h holds that head's j-th kept token.
        hh = torch.arange(H, device=dev).unsqueeze(1)          # keys are stored after RoPE, so slot identity does not matter
        dbe, doe = db.unsqueeze(0).expand(H, -1), do.unsqueeze(0).expand(H, -1)
        for l in range(L):
            sb, so = slots_of(block_ids, keep[l])              # (H, k)
            tmp = kv[l][sb, hh, so, :].clone()                 # (H, k, 2D)
            kv[l][dbe, hh, doe, :] = tmp
        pos0 = keep[0, 0]                                      # token ids below: any consistent list (not read by the model)
    d = comp - k
    keep_nb = -(-k // BS)
    # 2) scheduler: prompt = kept prompt tokens, outputs unchanged, tail blocks freed
    kept_ids = [req._all_token_ids[i] for i in pos0[:kp].tolist()]
    outs = list(req._output_token_ids)
    req._all_token_ids[:] = kept_ids + outs
    if req.prompt_token_ids is not None:
        req.prompt_token_ids = list(kept_ids)               # rebind: never mutate the caller's list
    req.num_prompt_tokens = kp
    req.num_computed_tokens = comp - d
    freed = free_tail_blocks(rid, keep_nb)
    # 3) worker state: every counter shifts by -d
    rs = runner.req_states
    idx = rs.req_id_to_index[rid]
    if comp == n:                                              # first eviction: exactly the validated stage-1 path
        rs.num_computed_tokens.gpu[idx] = k
        rs.num_computed_tokens_np[idx] = k
        rs.num_computed_prefill_tokens[idx] = k
        rs.prefill_len.np[idx] = k
    else:                                                      # re-eviction after a restored preemption
        rs.num_computed_tokens.gpu[idx] -= d
        rs.num_computed_tokens_np[idx] -= d
        rs.num_computed_prefill_tokens[idx] -= d
        rs.prefill_len.np[idx] -= d
    rs.prefill_len.copy_to_uva()
    rs.prompt_len.np[idx] = kp; rs.prompt_len.copy_to_uva()
    rs.total_len.gpu[idx] -= d
    rs.max_seq_len[idx] -= d
    row = rs.all_token_ids.gpu[idx]
    row[:kp] = torch.as_tensor(kept_ids, dtype=row.dtype, device=row.device)
    if outs:
        row[kp:kp + len(outs)] = torch.as_tensor(outs, dtype=row.dtype, device=row.device)
    bt = runner.block_tables
    bt.num_blocks.np[0, idx] = keep_nb; bt.num_blocks.copy_to_uva()
    # 4) RoPE offset for every token computed from now on (original position = slot + offset)
    OFFSET[idx] = (n - kp) if use_offset else 0
    EVICTED[rid]["idx"] = idx; EVICTED[rid]["needs"] = False
    return {"n": n, "k": k, "freed_blocks": freed, "freed_tokens": freed * BS}

@torch.no_grad()
def evict_mid(rid, keep):
    """Eviction DURING chunked prefill. keep = positions among the `comp` tokens computed so far (all context).
    The not-yet-computed rest of the prompt stays; positions of everything after keep shift by d (offset += d)."""
    req = sched.requests[rid]
    comp = int(req.num_computed_tokens)
    k = int(keep.shape[-1]); d = comp - k
    if d <= 0:
        return 0
    block_ids = list(kvm.get_block_ids(rid)[0])
    if rid not in EVICTED:
        EVICTED[rid] = {"orig": list(req.prompt_token_ids), "idx": None, "preempts": 0, "needs": False}
    db, do = slots_of(block_ids, torch.arange(k, device=dev))
    if keep.dim() == 1:
        sb, so = slots_of(block_ids, keep)
        for l in range(L):
            tmp = kv[l][sb, :, so, :].clone(); kv[l][db, :, do, :] = tmp
        pos0 = keep
    else:
        hh = torch.arange(H, device=dev).unsqueeze(1)
        dbe, doe = db.unsqueeze(0).expand(H, -1), do.unsqueeze(0).expand(H, -1)
        for l in range(L):
            sb, so = slots_of(block_ids, keep[l])
            tmp = kv[l][sb, hh, so, :].clone(); kv[l][dbe, hh, doe, :] = tmp
        pos0 = keep[0, 0]
    kept_ids = [req._all_token_ids[i] for i in pos0.tolist()]
    rest = list(req._all_token_ids[comp:])                     # prompt tokens not computed yet
    req._all_token_ids[:] = kept_ids + rest
    req.prompt_token_ids = kept_ids + rest
    req.num_prompt_tokens = len(kept_ids) + len(rest)
    req.num_computed_tokens = k
    keep_nb = -(-k // BS)
    free_tail_blocks(rid, keep_nb)
    rs = runner.req_states; idx = rs.req_id_to_index[rid]
    rs.num_computed_tokens.gpu[idx] = k
    rs.num_computed_tokens_np[idx] = k
    rs.num_computed_prefill_tokens[idx] = k
    rs.prefill_len.np[idx] -= d; rs.prefill_len.copy_to_uva()
    rs.prompt_len.np[idx] -= d; rs.prompt_len.copy_to_uva()
    rs.total_len.gpu[idx] -= d
    rs.max_seq_len[idx] -= d
    row = rs.all_token_ids.gpu[idx]
    row[:k + len(rest)] = torch.as_tensor(kept_ids + rest, dtype=row.dtype, device=row.device)
    bt = runner.block_tables
    bt.num_blocks.np[0, idx] = keep_nb; bt.num_blocks.copy_to_uva()
    OFFSET[idx] += d                                           # cumulative: original position = slot + offset
    EVICTED[rid]["idx"] = idx
    return d

def run_long(ids, n_ctx, max_new, mode, tag):
    """mode: full | after (evict once, after the whole prompt) | stream (evict after every chunk, then at the end).
    Returns text, peak KV tokens held by this request, number of evictions."""
    engine.add_request(tag, {"prompt_token_ids": list(ids)}, SamplingParams(max_tokens=max_new, temperature=0.0))
    n, Dsum, final, text, peak, nev = len(ids), 0, False, None, 0, 0
    total = pool.num_gpu_blocks if hasattr(pool, "num_gpu_blocks") else None
    free0 = pool.get_num_free_blocks()
    for _ in range(100000):
        outs = engine.step()
        peak = max(peak, (free0 - pool.get_num_free_blocks()) * BS)
        rid = find_rid(tag)
        if rid is not None and rid in sched.requests and rid in runner.req_states.req_id_to_index:
            req = sched.requests[rid]
            comp = int(req.num_computed_tokens)
            ctx_now = n_ctx - Dsum                              # end of the context in current (compacted) coordinates
            if mode == "stream" and not final and req.num_output_tokens == 0 and comp <= ctx_now \
                    and comp >= ctx_budget(ctx_now, 1) + CHUNK and BUDGET > 0:
                keep = select_keep(list(kvm.get_block_ids(rid)[0]), comp, comp)      # all computed tokens are context
                Dsum += evict_mid(rid, keep); nev += 1
            elif mode in ("stream", "after") and not final and comp >= n - Dsum and req.num_output_tokens >= 1:
                keep = select_keep(list(kvm.get_block_ids(rid)[0]), ctx_now, n - Dsum)
                st = evict(rid, keep, n - Dsum)
                idx = runner.req_states.req_id_to_index[rid]
                OFFSET[idx] += Dsum                             # evict() set n' - kept; add what was removed before
                final, nev = True, nev + 1
        for o in outs:
            if o.finished and tag in str(o.request_id):
                text = o.outputs[0].text
        if text is not None:
            break
    rid_done = next((r for r in EVICTED if tag in str(r)), None)
    if rid_done and EVICTED[rid_done]["idx"] is not None:
        OFFSET[EVICTED[rid_done]["idx"]] = 0; EVICTED[rid_done]["idx"] = None
    while engine.has_unfinished_requests():
        engine.step()
    return text, peak, nev

def run(ids, n_ctx, max_new, mode, tag):
    """mode: full | identity | perm | digm | digm_nofix"""
    engine.add_request(tag, {"prompt_token_ids": list(ids)}, SamplingParams(max_tokens=max_new, temperature=0.0))
    n, done, stats, text = len(ids), False, None, None
    for _ in range(max_new + 50):
        outs = engine.step()
        rid = find_rid(tag)
        if rid is not None and stats is None and mode != "full":
            req = sched.requests[rid]
            if req.num_computed_tokens >= n and req.num_output_tokens >= 1:
                if DEBUG:
                    print(f"    [debug] {tag}: n={n} n_ctx={n_ctx} computed={req.num_computed_tokens} outs={req.num_output_tokens} "
                          f"blocks={len(kvm.get_block_ids(rid)[0])} free={pool.get_num_free_blocks()}", flush=True)
                if mode == "identity":
                    keep = torch.arange(n, device=dev)
                elif mode == "perm":                           # every (layer, head) keeps ALL tokens, in its own random order
                    g = torch.Generator(device="cpu").manual_seed(n)
                    keep = torch.stack([torch.stack([torch.randperm(n, generator=g) for _ in range(H)]) for _ in range(L)]).to(dev)
                else:
                    keep = select_keep(list(kvm.get_block_ids(rid)[0]), n_ctx, n)
                stats = evict(rid, keep, n, use_offset=(mode != "digm_nofix"))
                stats["idx"] = runner.req_states.req_id_to_index[rid]
        for o in outs:
            if o.finished and tag in str(o.request_id):
                text, done = o.outputs[0].text, True
        if done:
            break
    if stats is not None:
        OFFSET[stats["idx"]] = 0
    while engine.has_unfinished_requests():
        engine.step()
    return text, stats


# ================= STAGE 2: many requests at once, real throughput =================
PREEMPTED_ALL = []                                              # every preemption vLLM makes (any request)
_inner_preempt = sched._preempt_request
def _count_preempt(request, timestamp, *a, **kw):
    PREEMPTED_ALL.append(request.request_id)
    return _inner_preempt(request, timestamp, *a, **kw)
sched._preempt_request = _count_preempt

def maybe_evict(rid, m, ev_ms):
    """Evict once after the prompt is computed; evict AGAIN after a restored preemption's re-prefill."""
    req = sched.requests[rid]
    st = EVICTED.get(rid)
    if m["ev"] and not (st is not None and st["needs"]):
        return
    if req.num_computed_tokens >= m["n"] and req.num_output_tokens >= 1 and rid in runner.req_states.req_id_to_index:
        torch.cuda.synchronize(); te = time.perf_counter()
        evict(rid, select_keep(list(kvm.get_block_ids(rid)[0]), m["n_ctx"], m["n"]), m["n"])
        torch.cuda.synchronize(); ev_ms.append((time.perf_counter() - te) * 1000)
        m["ev"] = True; m["evictions"] = m.get("evictions", 0) + 1

def release(m):
    st = EVICTED.get(m.get("rid"))
    if st is not None and st["idx"] is not None:
        OFFSET[st["idx"]] = 0; st["idx"] = None

def bench(mode, prompts, max_new=128):
    meta, tag_of = {}, {}
    pre0 = len(PREEMPTED_ALL)
    for j, (ids, n_ctx) in enumerate(prompts):
        tag = f"{mode}{PREEMPT_MODE[0]}B{j}X"
        meta[tag] = {"n": len(ids), "n_ctx": n_ctx, "ev": False, "rid": None, "text": None}
        engine.add_request(tag, {"prompt_token_ids": list(ids)},
                           SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True))
    torch.cuda.synchronize(); t0 = time.perf_counter()
    gen, max_running, ev_ms = 0, 0, []
    while engine.has_unfinished_requests():
        outs = engine.step()
        max_running = max(max_running, len(sched.running))
        for o in outs:
            if o.finished:
                gen += len(o.outputs[0].token_ids)
                tg = next((t for t in meta if t in str(o.request_id)), None)
                if tg:
                    meta[tg]["text"] = o.outputs[0].text; release(meta[tg])
        for rid in list(sched.requests):
            tg = tag_of.get(rid) or next((t for t in meta if t in str(rid)), None)
            if tg is None:
                continue
            tag_of[rid] = tg; meta[tg]["rid"] = rid
            if mode == "digm":
                maybe_evict(rid, meta[tg], ev_ms)
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    OFFSET.zero_()
    pre_ids = PREEMPTED_ALL[pre0:]
    per_req_pre = [sum(1 for x in pre_ids if x == m["rid"]) for m in meta.values()]
    return {"wall_s": wall, "gen_tokens": gen, "tok_s": gen / wall, "max_running": max_running,
            "evict_ms": (sum(ev_ms) / len(ev_ms)) if ev_ms else 0.0, "evictions": len(ev_ms),
            "preemptions": len(pre_ids), "preempted_after_evict": sum(EVICTED[m["rid"]]["preempts"] for m in meta.values() if m["rid"] in EVICTED),
            "texts": [m["text"] for m in meta.values()], "preempted_per_req": per_req_pre,
            "pre_after_evict_per_req": [EVICTED.get(m["rid"], {}).get("preempts", 0) for m in meta.values()]}

# ================= DIGM-elastic: evict only under memory pressure, only as much as needed =================
FLOOR = float(os.environ.get("DIGM_FLOOR", "0.25"))     # never keep less than this fraction of a request's context
STEP = float(os.environ.get("DIGM_STEP", "0.125"))      # shrink one request by this fraction of its context at a time
TRIG_ON = float(os.environ.get("DIGM_TRIG_ON", "0.98"))   # TriAttention defaults: arm at 98% KV usage,
TRIG_OFF = float(os.environ.get("DIGM_TRIG_OFF", "0.90")) # release at 90%
CACHE = os.environ.get("DIGM_CACHE", "1") == "1"         # reuse each request's first scores for later shrinks
BURST = float(os.environ.get("DIGM_BURST", "1.0"))      # arrival burstiness (coefficient of variation; 1 = Poisson)
POLICY = os.environ.get("DIGM_POLICY", "fair")          # "fair": shrink the request holding the largest fraction first
                                                        # "largest": shrink the request holding the most tokens first
def _set_budget(b):
    global BUDGET
    old = BUDGET; BUDGET = b; return old

@torch.no_grad()
def shrink(rid, m, new_kc):
    """Cut a RUNNING request's context from m['kc'] to new_kc kept tokens (per head, query-free scores on the
    tokens currently held). Question tokens and generated tokens are kept. Positions stay original (offset grows)."""
    req = sched.requests[rid]
    kc, q = m["kc"], m["n"] - m["n_ctx"]
    kp = kc + q
    if new_kc >= kc or int(req.num_computed_tokens) < kp:
        return 0
    idx = runner.req_states.req_id_to_index[rid]
    off_old = int(OFFSET[idx])
    block_ids = list(kvm.get_block_ids(rid)[0])
    S = m.get("S")
    if CACHE and SEL.startswith("head:"):                     # score ONCE per request; later shrinks reuse it
        if S is None or S.shape[-1] != kc:
            S = perhead_scores(block_ids, kc); m["scored"] = m.get("scored", 0) + 1
        top = S.float().topk(new_kc, dim=-1).indices.sort(dim=-1).values          # (L, H, new_kc) slots kept
        keep_ctx = top
    else:
        old = _set_budget(new_kc)
        try:
            keep_ctx = select_keep(block_ids, kc, kc)          # context region only, budget new_kc
        finally:
            _set_budget(old)
    qpos = torch.arange(kc, kp, device=dev)
    keep = torch.cat([keep_ctx, qpos]) if keep_ctx.dim() == 1 else \
        torch.cat([keep_ctx, qpos.view(1, 1, -1).expand(keep_ctx.shape[0], keep_ctx.shape[1], -1)], dim=-1)
    free0 = pool.get_num_free_blocks()
    evict(rid, keep, kp)                                       # sets offset = kp - kp'; add what was removed before
    OFFSET[runner.req_states.req_id_to_index[rid]] += off_old
    m["kc"] = int(keep.shape[-1]) - q
    if CACHE and SEL.startswith("head:"):
        m["S"] = torch.gather(S, -1, keep_ctx)                 # scores follow their tokens into the new slots
    return pool.get_num_free_blocks() - free0

WL_SCOPE = os.environ.get("DIGM_WL_SCOPE", "all")        # all = every arrived request | admitted = only requests vLLM has scheduled
WL_RESERVE = os.environ.get("DIGM_WL_RESERVE", "max")    # max = reserve full generation | remaining = only what is still to generate
WL_ALPHA = float(os.environ.get("DIGM_WL_ALPHA", "1.0")) # >1 = size as if the pool were larger (keep more, accept some queueing)
def _gen_left(m, max_new):
    if WL_RESERVE != "remaining" or m["rid"] is None or m["rid"] not in sched.requests:
        return max_new
    return max(max_new - int(sched.requests[m["rid"]].num_output_tokens), 0)
def water_level(meta, max_new, scope=None, reserve=None):
    """Fraction f of its context each ACTIVE request can keep if all of them must fit in the pool at once together
    with their question and generation reservation:  f = (alpha*pool - margin - sum(q + gen)) / sum(ctx).
    elastic2 = scope all + reserve max (v11); elastic3 = scope admitted + reserve remaining."""
    global WL_SCOPE, WL_RESERVE
    s0, r0 = WL_SCOPE, WL_RESERVE
    if scope: WL_SCOPE = scope
    if reserve: WL_RESERVE = reserve
    try:
        act = [m for m in meta.values() if m["added"] and m["fin"] is None]
        if WL_SCOPE == "admitted":
            act = [m for m in act if m["rid"] is not None and m["rid"] in runner.req_states.req_id_to_index]
        if not act:
            return 1.0
        pool_tok = WL_ALPHA * (pool.num_gpu_blocks - 1) * BS
        fixed = sum((m["n"] - m["n_ctx"]) + _gen_left(m, max_new) + 2 * BS for m in act) + 32 * BS
        ctx = sum(m["n_ctx"] for m in act)
        return max(FLOOR, min(1.0, (pool_tok - fixed) / max(ctx, 1)))
    finally:
        WL_SCOPE, WL_RESERVE = s0, r0
def _water_level_v11(meta, max_new):
    act = [m for m in meta.values() if m["added"] and m["fin"] is None]
    if not act:
        return 1.0
    pool_tok = (pool.num_gpu_blocks - 1) * BS
    fixed = sum((m["n"] - m["n_ctx"]) + max_new + 2 * BS for m in act) + 32 * BS
    ctx = sum(m["n_ctx"] for m in act)
    return max(FLOOR, min(1.0, (pool_tok - fixed) / max(ctx, 1)))

SENS_FLOOR = float(os.environ.get("DIGM_SENS_FLOOR", "0.125"))   # sens: tolerant requests may go down to this fraction
def share25(S):
    """Query-free sensitivity of a request: share of its (min-shifted) per-head score mass held by the top 25% of
    tokens, averaged over layers and heads. High = sensitive to eviction (probe: AUC 0.77-0.81)."""
    S = S.float(); n = S.shape[-1]; k = max(n // 4, 1)
    sh = S - S.min(-1, keepdim=True).values
    return (sh.topk(k, dim=-1).values.sum(-1) / (sh.sum(-1) + 1e-6)).mean().item()

def governor(meta_by_rid, stats, fstar=None, sens=False):
    """Before a step: make sure the next step (one decode token per running request + admitting the next waiting
    request's prompt) fits in free blocks; otherwise shrink running requests until it does, down to FLOOR."""
    need = 2
    for r in sched.running:
        if (int(r.num_computed_tokens) + 1) % BS == 1:          # next token opens a new block
            need += 1
    try:
        w = next(iter(sched.waiting))
        need += -(-(len(w.prompt_token_ids) + 1) // BS)
    except (StopIteration, TypeError, AttributeError):
        pass
    free = pool.get_num_free_blocks()
    guard = 0
    while free < need and guard < 64:
        guard += 1
        cands = []
        for rid, m in meta_by_rid.items():
            if rid not in sched.requests or rid not in runner.req_states.req_id_to_index:
                continue
            req = sched.requests[rid]
            if req.num_output_tokens < 1 or req not in sched.running:
                continue
            fl = FLOOR
            if sens:                                           # tolerant half (below the median sensitivity) may go lower
                known = [x.get("sens") for x in meta_by_rid.values() if x.get("sens") is not None and x["fin"] is None]
                med = sorted(known)[len(known) // 2] if known else None
                tol = m.get("sens") is not None and med is not None and m["sens"] < med
                m["_tol"] = tol
                if tol: fl = SENS_FLOOR
            floor = max(int(m["n_ctx"] * fl), 16)
            if m["kc"] > floor:
                cands.append((rid, m, floor))
        if not cands:
            stats["floor_hit"] += 1; break
        key = (lambda c: c[1]["kc"] / c[1]["n_ctx"]) if POLICY == "fair" else (lambda c: c[1]["kc"])
        if sens and any(c[1].get("_tol") for c in cands):      # squeeze tolerant requests first (fairly among them)
            cands = [c for c in cands if c[1].get("_tol")]
        rid, m, floor = max(cands, key=key)
        new_kc = max(floor, int(m["kc"] - STEP * m["n_ctx"]))
        if fstar is not None:                                  # v2: jump to the water level in ONE shrink
            new_kc = max(floor, min(new_kc, int(fstar * m["n_ctx"])))
        torch.cuda.synchronize(); te = time.perf_counter()
        got = shrink(rid, m, new_kc)
        torch.cuda.synchronize(); stats["ms"].append((time.perf_counter() - te) * 1000); stats["shrinks"] += 1
        free += got

SLO_F = float(os.environ.get("DIGM_SLO_F", "0.5"))       # slo: admit only while every active request could keep >= this fraction
SLO_WAIT = float(os.environ.get("DIGM_SLO_WAIT", "2.0")) # ... or once the oldest waiting request has waited this long (seconds)
def WL(meta, max_new, mode):
    if mode == "slo":
        return water_level(meta, max_new, scope="all", reserve="remaining")
    if mode == "elastic3":
        return water_level(meta, max_new, scope="admitted", reserve="remaining")
    return water_level(meta, max_new, scope="all", reserve="max")

def serve(mode, prompts, rate, max_new, seed=0):
    """Online serving: request j arrives at a Poisson time. mode: full | digm (static: evict to KEEP right after
    prefill) | elastic (no eviction until memory runs short; then the governor shrinks running requests).
    TTFT = first generated token - arrival; TPOT = (finish - first token) / (tokens - 1); e2e = finish - arrival."""
    rng = random.Random(seed); t, arr = 0.0, []
    for _ in prompts:
        if BURST > 1.0:                                        # bursty: gamma inter-arrivals with this coefficient of variation
            k = 1.0 / (BURST * BURST); t += rng.gammavariate(k, 1.0 / (rate * k))
        else:
            t += rng.expovariate(rate)
        arr.append(t)
    meta = {}
    for j, (ids, n_ctx) in enumerate(prompts):
        meta[f"{mode}r{int(rate * 1000)}s{seed}S{j}X"] = {"n": len(ids), "n_ctx": n_ctx, "kc": n_ctx, "ids": ids, "arr": arr[j],
                                "first": None, "fin": None, "out": 0, "ev": False, "rid": None, "added": False, "text": None}
    order = list(meta); nxt, ev_ms, max_running, tag_of = 0, [], 0, {}
    gstats = {"shrinks": 0, "ms": [], "floor_hit": 0, "sized": 0}
    trig = {"armed": False}
    pre0 = len(PREEMPTED_ALL)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    pending = []                                               # slo mode: arrived but held back by DIGM's admission control
    def _admit(tg):
        engine.add_request(tg, {"prompt_token_ids": list(meta[tg]["ids"])},
                           SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True))
        meta[tg]["added"] = True
    while nxt < len(order) or engine.has_unfinished_requests() or pending:
        now = time.perf_counter() - t0
        while nxt < len(order) and meta[order[nxt]]["arr"] <= now:
            if mode == "slo":
                pending.append(order[nxt])
            else:
                _admit(order[nxt])
            nxt += 1
        if mode == "slo":                                      # admit FIFO while everyone could still keep >= SLO_F,
            while pending:                                     # or once the head request has waited SLO_WAIT seconds
                tg = pending[0]
                meta[tg]["added"] = True
                f_with = water_level(meta, max_new, scope="all", reserve="remaining")
                meta[tg]["added"] = False
                if f_with >= SLO_F or now - meta[tg]["arr"] >= SLO_WAIT or not engine.has_unfinished_requests():
                    _admit(pending.pop(0))
                else:
                    break
        if not engine.has_unfinished_requests():
            if nxt < len(order):
                time.sleep(max(0.0, meta[order[nxt]]["arr"] - (time.perf_counter() - t0)))
            continue
        if mode in ("elastic", "elastic2", "elastic3", "slo", "sens"):
            by_rid = {m["rid"]: m for m in meta.values() if m["rid"] is not None}
            governor(by_rid, gstats, WL(meta, max_new, mode) if mode not in ("elastic", "sens") else None, sens=(mode == "sens"))
        if mode == "trigger":                                  # baseline: TriAttention-style KV-usage trigger
            usage = 1.0 - pool.get_num_free_blocks() / max(pool.num_gpu_blocks - 1, 1)
            if usage >= TRIG_ON: trig["armed"] = True
            elif usage <= TRIG_OFF: trig["armed"] = False
            if trig["armed"]:                                  # compress EVERY running request to the fixed budget
                for m in meta.values():
                    rid = m["rid"]
                    if rid is None or rid not in sched.requests or rid not in runner.req_states.req_id_to_index:
                        continue
                    rq = sched.requests[rid]
                    fl = max(int(m["n_ctx"] * FLOOR), 16)
                    if rq in sched.running and rq.num_output_tokens >= 1 and m["kc"] > fl:
                        torch.cuda.synchronize(); te = time.perf_counter()
                        shrink(rid, m, fl)
                        torch.cuda.synchronize(); gstats["ms"].append((time.perf_counter() - te) * 1000); gstats["shrinks"] += 1
        outs = engine.step()
        now = time.perf_counter() - t0
        max_running = max(max_running, len(sched.running))
        for rid in list(sched.requests):
            tg = tag_of.get(rid) or next((x for x in meta if x in str(rid)), None)
            if tg is None:
                continue
            tag_of[rid] = tg; req, m = sched.requests[rid], meta[tg]
            m["rid"] = rid
            st = EVICTED.get(rid)
            if mode in ("elastic", "elastic2", "elastic3", "slo", "sens", "trigger") and st is not None and st["needs"]:   # restored after preemption: full again
                st["needs"] = False; m["kc"] = m["n_ctx"]; m["sized"] = False; m["S"] = None
            if mode in ("elastic2", "elastic3", "slo") and not m.get("sized") and req.num_output_tokens >= 1 \
                    and req.num_computed_tokens >= m["n"] and rid in runner.req_states.req_id_to_index:
                m["sized"] = True                              # admission-time sizing: one eviction, at most
                f = WL(meta, max_new, mode)
                if f < 1.0:
                    torch.cuda.synchronize(); te = time.perf_counter()
                    shrink(rid, m, max(int(m["n_ctx"] * FLOOR), int(f * m["n_ctx"])))
                    torch.cuda.synchronize(); gstats["ms"].append((time.perf_counter() - te) * 1000); gstats["sized"] += 1
            if mode == "sens" and m.get("sens") is None and req.num_output_tokens >= 1 \
                    and req.num_computed_tokens >= m["n"] and rid in runner.req_states.req_id_to_index:
                m["S"] = perhead_scores(list(kvm.get_block_ids(rid)[0]), m["n_ctx"]); m["sens"] = share25(m["S"])
            if m["first"] is None and req.num_output_tokens >= 1:
                m["first"] = now
            if mode == "digm":
                maybe_evict(rid, m, ev_ms)
        for o in outs:
            if o.finished:
                tg = next((x for x in meta if x in str(o.request_id)), None)
                if tg:
                    meta[tg]["fin"] = now; meta[tg]["out"] = len(o.outputs[0].token_ids); meta[tg]["text"] = o.outputs[0].text
                    release(meta[tg])
    OFFSET.zero_()
    wall = time.perf_counter() - t0
    done = [m for m in meta.values() if m["fin"] is not None and m["first"] is not None]
    pct = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))] if v else float("nan")
    ttft = [m["first"] - m["arr"] for m in done]; e2e = [m["fin"] - m["arr"] for m in done]
    tpot = [(m["fin"] - m["first"]) / max(m["out"] - 1, 1) * 1000 for m in done]
    kept = [m["kc"] / m["n_ctx"] for m in meta.values()] if mode in ("elastic", "elastic2", "elastic3", "slo", "sens", "trigger") else \
           ([KEEP] * len(meta) if mode == "digm" else [1.0] * len(meta))
    return {"rate": rate, "n": len(done), "wall_s": wall, "tok_s": sum(m["out"] for m in done) / wall, "max_running": max_running,
            "ttft_p50": pct(ttft, .5), "ttft_p99": pct(ttft, .99), "tpot_ms_p50": pct(tpot, .5), "tpot_ms_p99": pct(tpot, .99),
            "e2e_p50": pct(e2e, .5), "e2e_p99": pct(e2e, .99), "evict_ms": sum(ev_ms) / len(ev_ms) if ev_ms else 0.0,
            "preemptions": len(PREEMPTED_ALL) - pre0, "shrinks": gstats["shrinks"] + gstats["sized"], "sized_at_admission": gstats["sized"],
            "shrink_ms": (sum(gstats["ms"]) / len(gstats["ms"])) if gstats["ms"] else 0.0, "floor_hit": gstats["floor_hit"],
            "kept_mean": sum(kept) / len(kept), "texts": [m["text"] for m in meta.values()]}

if THROUGHPUT or SERVE or STRESS:
    from datasets import load_dataset
    ds = load_dataset("simonjegou/ruler", "4096")["test"]
    by_task = {}
    for row in ds:
        by_task.setdefault(row["task"], []).append(row)
    NREQ = int(os.environ.get("DIGM_NREQ", "32"))               # 32 = earlier runs; raise it when Full KV already fits most
    per_task = -(-NREQ // len(by_task))
    rng = random.Random(42); rows = []
    for task in sorted(by_task):
        r = by_task[task]; rng.shuffle(r); rows += r[:per_task]
    def build_t(row):
        q = row["question"]
        full = tok.apply_chat_template([{"role": "user", "content": row["context"] + "\n\n" + q}],
                                       tokenize=False, add_generation_prompt=True) + (row.get("answer_prefix") or "")
        cut = full.find(row["context"]) + len(row["context"])
        return tok(full, add_special_tokens=False).input_ids, max(len(tok(full[:cut], add_special_tokens=False).input_ids) - 2, 8)
    MIX = os.environ.get("DIGM_MIX", "")                        # e.g. "4096:0.5,8192:0.3,16384:0.2" = mixed prompt lengths
    if MIX:
        rows = []
        for part in MIX.split(","):
            ctxl, frac = part.split(":"); want = max(1, round(NREQ * float(frac)))
            byx = {}
            for row in load_dataset("simonjegou/ruler", ctxl)["test"]:
                byx.setdefault(row["task"], []).append(row)
            pt = -(-want // len(byx)); got = []
            for task in sorted(byx):
                r = byx[task]; random.Random(42).shuffle(r); got += r[:pt]
            random.Random(7).shuffle(got); rows += got[:want]
        random.Random(11).shuffle(rows); rows = rows[:NREQ]
        print(f"  mixed workload {MIX}: {len(rows)} requests", flush=True)
    WORKLOAD = os.environ.get("DIGM_WORKLOAD", "ruler")
    GEN = int(os.environ.get("DIGM_GEN", "128"))
    if WORKLOAD == "longbench":                                  # realistic heterogeneous traffic: 10 English LongBench tasks
        import re as _re, string as _string, difflib as _difflib
        from collections import Counter as _Counter
        _LBT = os.environ.get("DIGM_LB_TASKS", "qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,triviaqa,"
                              "passage_retrieval_en,passage_count,lcc,repobench-p").split(",")
        _MAXP = int(os.environ.get("DIGM_MAXLEN", "8192")) - GEN - 64
        rows = []; pt = -(-NREQ // len(_LBT))
        for t in _LBT:
            r = list(load_dataset("Xnhyacinth/LongBench", t)["test"]); random.Random(42).shuffle(r)
            for x in r[20:20 + pt]:                                # rows 0-19 were used for the accuracy table
                x = dict(x); x["answer"] = list(x["answers"]); x["task"] = t; rows.append(x)
        random.Random(11).shuffle(rows); rows = rows[:NREQ]
        def build_t(row):
            q = row["question"]; ap = row.get("answer_prefix") or ""
            cid = tok(row["context"], add_special_tokens=False).input_ids
            room = _MAXP - len(tok(q + ap, add_special_tokens=False).input_ids) - 64; ctx = row["context"]
            if len(cid) > room:
                h = room // 2; ctx = tok.decode(cid[:h]) + tok.decode(cid[-h:])
            full = tok.apply_chat_template([{"role": "user", "content": ctx + "\n\n" + q}], tokenize=False,
                                           add_generation_prompt=True) + ap
            cut = full.find(ctx) + len(ctx)
            return tok(full, add_special_tokens=False).input_ids, max(len(tok(full[:cut], add_special_tokens=False).input_ids) - 2, 8)
        def _norm(t):
            t = t.lower(); t = "".join(ch for ch in t if ch not in set(_string.punctuation))
            t = _re.sub(r"\b(a|an|the)\b", " ", t); return " ".join(t.split())
        def _f1(p_, g):
            pt_, gt = _norm(p_).split(), _norm(g).split(); c = _Counter(pt_) & _Counter(gt); ns = sum(c.values())
            if ns == 0: return 0.0
            pr, rc = ns / len(pt_), ns / len(gt); return 2 * pr * rc / (pr + rc)
        def lb_serve_score(task, pred, gts):
            lines = [l for l in (pred or "").split("\n") if l.strip()]
            if task in ("lcc", "repobench-p"):
                code = [l for l in lines if not l.lstrip().startswith(("`", "#", "//"))]
                return max(_difflib.SequenceMatcher(None, code[0] if code else "", g).ratio() for g in gts)
            first = lines[0] if lines else ""                     # ignore_eos: score the answer line only
            if task in ("passage_retrieval_en", "passage_count"):
                nums = _re.findall(r"\d+", first)
                best = 0.0
                for g in gts:
                    tg = (_re.findall(r"\d+", g) or [g])[0]; best = max(best, (sum(n == tg for n in nums) / len(nums)) if nums else 0.0)
                return best
            return max(_f1(first, g) for g in gts)
        print(f"  LongBench serving workload: {len(rows)} requests over {len(_LBT)} tasks", flush=True)
    prompts = [build_t(r) for r in rows[:NREQ]]
    bench("full", prompts[:2], 8); bench("digm", prompts[:2], 8)        # warm-up
    if STRESS:
        def _strict(p, ans):
            ans = [ans] if isinstance(ans, str) else list(ans); kk = sum(len(a.split()) for a in ans) + STRICT_SLACK
            head = " ".join((p or "").split()[:kk]).lower(); return sum(a.lower() in head for a in ans) / len(ans)
        answers = [r["answer"] for r in rows[:NREQ]]
        out = {}
        print(f"\n vLLM STRESS (preemption) — {MODEL}, selection={SEL}, {NREQ} requests at once, {GEN} generated each, gpu_util {GPU_UTIL}")
        for label, mode, pm in (("Full KV", "full", "restore"), ("evict, naive preemption", "digm", "naive"),
                                ("evict, restore on preemption", "digm", "restore")):
            PREEMPT_MODE = pm
            r = bench(mode, prompts, GEN)
            acc = [_strict(t, a) for t, a in zip(r["texts"], answers)]
            pre = [a for a, k in zip(acc, r["preempted_per_req"]) if k > 0]
            r.update({"acc": sum(acc) / len(acc) * 100, "acc_preempted": (sum(pre) / len(pre) * 100) if pre else None,
                      "n_preempted_reqs": len(pre)})
            out[label] = dict(r)                                        # keep texts: compared token by token offline
            print(f"  {label:<30} acc {r['acc']:5.1f} | preempted requests {len(pre):>3} (acc " +
                  (f"{r['acc_preempted']:5.1f}" if pre else "  -  ") + f") | preemptions {r['preemptions']:>4} | "
                  f"{r['tok_s']:6.1f} tok/s | max running {r['max_running']:>3} | evictions {r['evictions']}", flush=True)
        os.makedirs("logs", exist_ok=True)
        json.dump({"model": MODEL, "sel": SEL, "gen": GEN, "nreq": NREQ, "gpu_util": GPU_UTIL, "stress": out},
                  open(f"logs/vllm_stress_{MODEL.split('/')[-1]}_gen{GEN}_n{NREQ}_{SEL.replace('/', '-').replace(':', '-')}{os.environ.get('DIGM_RUN', '')}.json", "w"), indent=2)
        sys.exit(0)
    if SERVE:
        rs_env = os.environ.get("DIGM_RATES", "auto")
        if rs_env == "auto":                     # rates relative to what Full KV sustains: 0.5x, 1x, 2x its capacity
            cap = bench("full", prompts[:min(16, len(prompts))], GEN)["tok_s"] / GEN
            RATES = [round(cap * f, 3) for f in (0.5, 1.0, 2.0)]
            print(f"  Full KV capacity ~{cap:.3f} requests/s -> rates {RATES}", flush=True)
        else:
            RATES = [float(x) for x in rs_env.split(",")]
        out = {}
        print(f"\n vLLM ONLINE SERVING — {MODEL}, selection={SEL}, {NREQ} requests (Poisson), {GEN} generated each")
        print(f"  {'rate/s':>6} {'mode':<7} {'tok/s':>7} {'running':>8} {'TTFT p50':>9} {'TTFT p99':>9} {'TPOT p50':>9} {'TPOT p99':>9} {'e2e p50':>8} {'e2e p99':>8}")
        answers = [r["answer"] for r in rows[:NREQ]]
        def _acc(texts):
            if WORKLOAD == "longbench":
                return sum(lb_serve_score(r["task"], tx, r["answer"]) for tx, r in zip(texts, rows[:NREQ])) / len(texts) * 100
            sc = []
            for tx, an in zip(texts, answers):
                an = [an] if isinstance(an, str) else list(an); kk = sum(len(a.split()) for a in an) + STRICT_SLACK
                head = " ".join((tx or "").split()[:kk]).lower(); sc.append(sum(a.lower() in head for a in an) / len(an))
            return sum(sc) / len(sc) * 100
        MODES_S = os.environ.get("DIGM_SERVE_MODES", "full,digm,elastic,elastic2").split(",")
        for rate in RATES:
            for mode in MODES_S:
                r = serve(mode, prompts, rate, GEN, seed=int(os.environ.get("DIGM_SEED", "0"))); r["acc"] = _acc(r.pop("texts")); out[f"{rate}|{mode}"] = r
                print(f"  {rate:6.2f} {mode:<7} {r['tok_s']:7.1f} {r['max_running']:8d} {r['ttft_p50']:8.2f}s {r['ttft_p99']:8.2f}s "
                      f"{r['tpot_ms_p50']:7.1f}ms {r['tpot_ms_p99']:7.1f}ms {r['e2e_p50']:7.1f}s {r['e2e_p99']:7.1f}s | acc {r['acc']:5.1f} "
                      f"| kept {r['kept_mean']*100:4.0f}% | preempt {r['preemptions']:>4} | shrinks {r['shrinks']:>4} ({r['shrink_ms']:.1f} ms each)", flush=True)
        os.makedirs("logs", exist_ok=True)
        json.dump({"model": MODEL, "sel": SEL, "gen": GEN, "nreq": NREQ, "serve": out},
                  open(f"logs/vllm_serve_{MODEL.split('/')[-1]}_gen{GEN}_n{NREQ}_{SEL.replace('/', '-').replace(':', '-')}{os.environ.get('DIGM_RUN', '')}.json", "w"), indent=2)
        sys.exit(0)
    res = {}
    for mode in ("full", "digm"):
        res[mode] = bench(mode, prompts, GEN)
        print(f"  {mode:<5} wall {res[mode]['wall_s']:6.1f} s | {res[mode]['tok_s']:6.1f} generated tok/s | "
              f"max running {res[mode]['max_running']:>3} | eviction {res[mode]['evict_ms']:5.1f} ms/request | "
              f"preempted after eviction {res[mode]['preempted_after_evict']}", flush=True)
    print("\n" + "=" * 92)
    print(f" vLLM THROUGHPUT — {MODEL}, selection={SEL}, l*={LSTAR}, {NREQ} requests x ~{sum(len(p[0]) for p in prompts)//NREQ} prompt tokens, {GEN} generated each")
    print("=" * 92)
    print(f"  Full KV : {res['full']['tok_s']:6.1f} tok/s, up to {res['full']['max_running']} requests at once, {res['full']['wall_s']:.1f} s total, "
          f"preemptions {res['full']['preemptions']}")
    print(f"  DIGM    : {res['digm']['tok_s']:6.1f} tok/s, up to {res['digm']['max_running']} requests at once, {res['digm']['wall_s']:.1f} s total, "
          f"preemptions {res['digm']['preemptions']} (after eviction {res['digm']['preempted_after_evict']}, restore={PREEMPT_MODE})")
    print(f"  speed-up {res['digm']['tok_s']/res['full']['tok_s']:.2f}x | eviction cost {res['digm']['evict_ms']:.1f} ms per request (once)")
    os.makedirs("logs", exist_ok=True)
    json.dump({"model": MODEL, "layer": LSTAR, "sel": SEL, "gen": GEN, "nreq": NREQ, "eager": EAGER,
               "throughput": {m: {k: v for k, v in r.items() if k != "texts"} for m, r in res.items()}}, open(f"logs/vllm_throughput_{MODEL.split('/')[-1]}_gen{GEN}_n{NREQ}_{SEL.replace('/', '-').replace(':', '-')}"
                                       + ("" if EAGER else "_graphs") + os.environ.get("DIGM_RUN", "") + ".json", "w"), indent=2)
    sys.exit(0)

if LONG:
    from datasets import load_dataset
    CTXL = int(os.environ.get("DIGM_CTX", "16384"))
    dsl = load_dataset("simonjegou/ruler", str(CTXL))["test"]
    byt = {}
    for row in dsl:
        byt.setdefault(row["task"], []).append(row)
    rng = random.Random(42); rows_l = []
    for task in sorted(byt):
        r = byt[task]; rng.shuffle(r); rows_l += r[:PER_TASK]          # DEV rows (same seed/split rule as everywhere)
    def _strict(p, ans):
        ans = [ans] if isinstance(ans, str) else list(ans); kk = sum(len(a.split()) for a in ans) + STRICT_SLACK
        head = " ".join((p or "").split()[:kk]).lower(); return sum(a.lower() in head for a in ans) / len(ans)
    def build_l(row):
        full = tok.apply_chat_template([{"role": "user", "content": row["context"] + "\n\n" + row["question"]}],
                                       tokenize=False, add_generation_prompt=True) + (row.get("answer_prefix") or "")
        cut = full.find(row["context"]) + len(row["context"])
        return tok(full, add_special_tokens=False).input_ids, max(len(tok(full[:cut], add_special_tokens=False).input_ids) - 2, 8)
    cap = (pool.num_gpu_blocks - 1) * BS
    print(f"\n vLLM LONG PROMPTS — {MODEL}, selection={SEL}, RULER {CTXL}, budget {BUDGET} context tokens, chunk {CHUNK}, "
          f"KV pool {cap} tokens", flush=True)
    MODES = ["full", "after", "stream"]
    acc = {m: [] for m in MODES}; peaks = {m: [] for m in MODES}; nofit = {m: 0 for m in MODES}; evs = {m: [] for m in MODES}
    t0 = time.perf_counter()
    for i, row in enumerate(rows_l):
        ids, n_ctx = build_l(row); mx = int(row.get("max_new_tokens", 32))
        line = f"  [{i+1:>2}/{len(rows_l)}] {row['task']:<18} prompt {len(ids):>6} |"
        for mode in MODES:
            if mode in ("full", "after") and len(ids) + mx > cap:
                nofit[mode] += 1; line += f" {mode}: does not fit |"; continue
            text, pk, nev = run_long(ids, n_ctx, mx, mode, f"L{i}{mode}X")
            acc[mode].append(_strict(text, row["answer"])); peaks[mode].append(pk); evs[mode].append(nev)
            line += f" {mode} {acc[mode][-1]:.2f} peak {pk:>6} |"
        print(line, flush=True)
    print("\n" + "=" * 100)
    print(f" LONG PROMPTS in vLLM — {MODEL}, RULER {CTXL}, {len(rows_l)} dev questions, KV pool {cap} tokens, "
          f"budget {BUDGET} + chunk {CHUNK}, selection={SEL}")
    print("=" * 100)
    for mode, name in (("full", "Full KV"), ("after", "evict after prefill"), ("stream", "evict DURING prefill")):
        a = acc[mode]
        print(f"  {name:<22} answered {len(a):>3} | does not fit {nofit[mode]:>3} | strict " +
              (f"{sum(a)/len(a)*100:5.1f}" if a else "  -  ") +
              (f" | peak KV {sum(peaks[mode])/len(peaks[mode]):7.0f} tokens (max {max(peaks[mode])})" if a else "") +
              (f" | evictions/request {sum(evs[mode])/len(evs[mode]):.1f}" if a and mode != "full" else ""))
    print(f"  time {time.perf_counter()-t0:.0f}s")
    os.makedirs("logs", exist_ok=True)
    json.dump({"model": MODEL, "sel": SEL, "ctx": CTXL, "budget": BUDGET, "chunk": CHUNK, "pool_tokens": cap,
               "acc": acc, "peak": peaks, "nofit": nofit, "evictions": evs},
              open(f"logs/vllm_long_{MODEL.split('/')[-1]}_{CTXL}_pool{cap}_{SEL.replace('/', '-').replace(':', '-')}.json", "w"), indent=2)
    sys.exit(0)

def as_list(a): return [a] if isinstance(a, str) else list(a)
def strict(p, ans):
    ans = as_list(ans); kk = sum(len(a.split()) for a in ans) + STRICT_SLACK
    head = " ".join((p or "").split()[:kk]).lower(); return sum(a.lower() in head for a in ans) / len(ans)

if LONGBENCH:
    import re, string, difflib
    from collections import Counter
    from datasets import load_dataset
    LB_TASKS = os.environ.get("DIGM_LB_TASKS", "qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,triviaqa,"
                              "passage_retrieval_en,passage_count,lcc,repobench-p").split(",")
    LB_N = int(os.environ.get("DIGM_LB_N", "20"))
    LB_MODES = os.environ.get("DIGM_LB_MODES", "full,digm").split(",")
    MAXP = int(os.environ.get("DIGM_MAXLEN", "8192")) - 600        # prompt budget (generation <= 512 + margin)
    def _norm(t):
        t = t.lower(); t = "".join(ch for ch in t if ch not in set(string.punctuation))
        t = re.sub(r"\b(a|an|the)\b", " ", t); return " ".join(t.split())
    def _f1(p, g):
        pt, gt = _norm(p).split(), _norm(g).split()
        c = Counter(pt) & Counter(gt); ns = sum(c.values())
        if ns == 0: return 0.0
        pr, rc = ns / len(pt), ns / len(gt); return 2 * pr * rc / (pr + rc)
    def _code(p, g):
        lines = [l for l in p.lstrip("\n").split("\n") if l.strip() and not l.lstrip().startswith(("`", "#", "//"))]
        return difflib.SequenceMatcher(None, lines[0] if lines else "", g).ratio()
    def lb_score(task, pred, gts):
        pred = pred or ""
        if task == "triviaqa": pred = pred.lstrip("\n").split("\n")[0]
        best = 0.0
        for g in gts:
            if task in ("passage_retrieval_en", "passage_count"):
                nums = re.findall(r"\d+", pred); tgt = re.findall(r"\d+", g)[0] if re.findall(r"\d+", g) else g
                sc = (sum(n == tgt for n in nums) / len(nums)) if nums else 0.0
            elif task in ("lcc", "repobench-p"):
                sc = _code(pred, g)
            else:
                sc = _f1(pred, g)
            best = max(best, sc)
        return best
    def lb_build(row):
        q = row["question"]; ap = row.get("answer_prefix") or ""
        ctx_ids = tok(row["context"], add_special_tokens=False).input_ids
        room = MAXP - len(tok(q + ap, add_special_tokens=False).input_ids) - 64
        ctx = row["context"]
        if len(ctx_ids) > room:                                     # LongBench rule: truncate the MIDDLE of the context
            h = room // 2; ctx = tok.decode(ctx_ids[:h]) + tok.decode(ctx_ids[-h:])
        full = tok.apply_chat_template([{"role": "user", "content": ctx + "\n\n" + q}], tokenize=False,
                                       add_generation_prompt=True) + ap
        cut = full.find(ctx) + len(ctx)
        ids = tok(full, add_special_tokens=False).input_ids
        n_ctx = len(tok(full[:cut], add_special_tokens=False).input_ids) - 2
        return ids, max(n_ctx, 8)
    res = {m: {} for m in LB_MODES}; lens = []; t0 = time.perf_counter(); LB_PROXY = []
    for task in LB_TASKS:
        rows = list(load_dataset("Xnhyacinth/LongBench", task)["test"])
        random.Random(42).shuffle(rows); rows = rows[:LB_N]
        for m in LB_MODES: res[m][task] = []
        for i, row in enumerate(rows):
            ids, n_ctx = lb_build(row); lens.append(len(ids))
            mx = int(row.get("max_new_tokens") or 64)
            for m in LB_MODES:
                txt, _ = run(ids, n_ctx, mx, m, f"lb{task}{i}{m}")
                res[m][task].append(lb_score(task, txt, row["answers"]))
                if m == "digm" and PROXY_ON:
                    LB_PROXY.append(dict(LAST_PROXY, task=task))
        print(f"  {task:<22} " + " | ".join(f"{m} {sum(res[m][task]) / len(res[m][task]) * 100:5.1f}" for m in LB_MODES), flush=True)
    def macro(m): return sum(sum(v) / len(v) for v in res[m].values()) / len(res[m]) * 100
    def lb_ci(a, b, B=2000):
        rr = random.Random(0); vals = []
        for _ in range(B):
            tot = 0.0
            for t in res[a]:
                ix = [rr.randrange(len(res[a][t])) for _ in res[a][t]]
                tot += sum(res[b][t][j] - res[a][t][j] for j in ix) / len(ix)
            vals.append(tot / len(res[a]) * 100)
        vals.sort(); return vals[int(.025 * B)], vals[int(.975 * B)]
    print("\n" + "=" * 92)
    print(f" LONGBENCH in vLLM — {MODEL}, selection={SEL}, keep {KEEP:.0%} of the context, {len(LB_TASKS)} English tasks x {LB_N},"
          f" prompts <= {MAXP} tokens (mean {sum(lens) // max(len(lens), 1)})")
    print("=" * 92)
    print("  macro score  " + " | ".join(f"{m} {macro(m):5.1f}" for m in LB_MODES))
    for m in LB_MODES[1:]:
        lo, hi = lb_ci(LB_MODES[0], m)
        print(f"  {m} minus {LB_MODES[0]}: {macro(m) - macro(LB_MODES[0]):+.1f} points, 95% CI [{lo:+.1f}, {hi:+.1f}]")
    print(f"  time {time.perf_counter() - t0:.0f}s")
    os.makedirs("logs", exist_ok=True)
    json.dump({"model": MODEL, "sel": SEL, "keep": KEEP, "n": LB_N, "res": res, "proxy": LB_PROXY},
              open(f"logs/vllm_longbench_{MODEL.split('/')[-1]}_{SEL.replace(':', '-')}_k{KEEP}.json", "w"), indent=2)
    sys.exit(0)

# ---------------- RULER dev questions (same split as all dev runs) ----------------
from datasets import load_dataset
ds = load_dataset("simonjegou/ruler", "4096")["test"]
by_task = {}
for row in ds:
    by_task.setdefault(row["task"], []).append(row)
rng = random.Random(42); samples = []
for task in sorted(by_task):
    rows = by_task[task]; rng.shuffle(rows)
    samples += rows[:PER_TASK] if SPLIT == "dev" else rows[10:10 + PER_TASK]

def build(row):
    q = row["question"]
    full = tok.apply_chat_template([{"role": "user", "content": row["context"] + "\n\n" + q}],
                                   tokenize=False, add_generation_prompt=True) + (row.get("answer_prefix") or "")
    cut = full.find(row["context"]) + len(row["context"])     # end of the context (the question may repeat inside it)
    ids = tok(full, add_special_tokens=False).input_ids
    n_ctx = len(tok(full[:cut], add_special_tokens=False).input_ids) - 2       # everything after it (question) is kept
    return ids, max(n_ctx, 8)

res = {"full": [], "identity": [], "digm": [], "digm_nofix": []}
same_identity, same_perm, freed, t0, tasks = 0, 0, [], time.perf_counter(), []
for i, row in enumerate(samples):
    ids, n_ctx = build(row)
    mx = int(row.get("max_new_tokens", 32))
    a, _ = run(ids, n_ctx, mx, "full", f"q{i}full")
    b, _ = run(ids, n_ctx, mx, "identity", f"q{i}id")
    c, st = run(ids, n_ctx, mx, "digm", f"q{i}digm")
    if PROXY_ON:                                               # probe + the same question at a tighter budget
        res.setdefault("proxy", []).append(dict(LAST_PROXY))
        _k0 = KEEP; KEEP = float(os.environ.get("DIGM_KEEP2", "0.125"))
        c2, _ = run(ids, n_ctx, mx, "digm", f"q{i}digm2")
        KEEP = _k0
        res.setdefault("digm_k2", []).append(strict(c2, row["answer"]))
    same_identity += int(a == b); tasks.append(row["task"])
    if PERM:
        pz, _ = run(ids, n_ctx, mx, "perm", f"q{i}perm"); same_perm += int(a == pz)
    res["full"].append(strict(a, row["answer"])); res["identity"].append(strict(b, row["answer"]))
    res["digm"].append(strict(c, row["answer"])); freed.append(st)
    line = f"  [{i+1:>2}/{len(samples)}] {row['task']:<18} full {res['full'][-1]:.2f} | identity==full {a == b}" + (f" | perm==full {a == pz}" if PERM else "") + f" | DIGM {res['digm'][-1]:.2f} | kept {st['k']}/{st['n']} | freed {st['freed_blocks']} blocks"
    if i < 5:
        e, _ = run(ids, n_ctx, mx, "digm_nofix", f"q{i}nofix")
        res["digm_nofix"].append(strict(e, row["answer"]))
        line += f" | no-RoPE-fix {res['digm_nofix'][-1]:.2f}"
        if i < 2:
            line += f"\n        full: {a[:70]!r}\n        DIGM: {c[:70]!r}\n        nofix: {e[:70]!r}"
    print(line, flush=True)

mean = lambda v: sum(v) / max(len(v), 1) * 100
def boot_diff(a, b, B=2000):                                   # paired bootstrap over questions, macro over tasks
    rr = random.Random(0); idx_by = {}
    for j, t in enumerate(tasks): idx_by.setdefault(t, []).append(j)
    vals = []
    for _ in range(B):
        tot = 0.0
        for t, ix in idx_by.items():
            sm = [rr.choice(ix) for _ in ix]; tot += sum(b[j] - a[j] for j in sm) / len(sm)
        vals.append(tot / len(idx_by) * 100)
    vals.sort(); return vals[int(.025 * B)], vals[int(.975 * B)]
print("\n" + "=" * 92)
print(f" DIGM REAL EVICTION IN vLLM — {MODEL}, selection={SEL}, l*={LSTAR}, keep 25% of the context ({len(samples)} RULER {SPLIT} questions"
      + (", CUDA graphs ON" if not EAGER else "") + ")")
print("=" * 92)
print(f"  identity eviction reproduces Full KV exactly : {same_identity}/{len(samples)}   (must be all)")
if PERM:
    print(f"  per-head PERMUTED slots reproduce Full KV    : {same_perm}/{len(samples)}   (composite packing; rare misses = float summation order)")
lo, hi = boot_diff(res["full"], res["digm"])
print(f"  evicted minus Full KV: {mean(res['digm']) - mean(res['full']):+.1f} points, 95% CI [{lo:+.1f}, {hi:+.1f}]")
print(f"  strict accuracy  Full KV {mean(res['full']):5.1f} | {SEL} evicted {mean(res['digm']):5.1f} | "
      f"without RoPE fix (first 5) {mean(res['digm_nofix']):5.1f} vs DIGM {mean(res['digm'][:5]):5.1f}")
avg_n = sum(s['n'] for s in freed) / len(freed); avg_k = sum(s['k'] for s in freed) / len(freed)
print(f"  KV tokens per request {avg_n:.0f} -> {avg_k:.0f} | blocks returned to vLLM per request {sum(s['freed_blocks'] for s in freed)/len(freed):.0f} "
      f"({sum(s['freed_tokens'] for s in freed)/len(freed):.0f} token slots)")
print(f"  time {time.perf_counter()-t0:.0f}s. Stage 1 = correctness. Stage 2 (next) = many requests at once -> throughput.")
os.makedirs("logs", exist_ok=True)
json.dump({"model": MODEL, "layer": LSTAR, "sel": SEL, "split": SPLIT, "eager": EAGER, "tasks": tasks, "res": res,
           "same_identity": same_identity, "same_perm": same_perm if PERM else None, "freed": freed},
          open(f"logs/vllm_stage1_{MODEL.split('/')[-1]}_{SEL.replace('/', '-').replace(':', '-')}_{SPLIT}_n{len(samples)}"
               + ("" if EAGER else "_graphs") + ".json", "w"), indent=2)
