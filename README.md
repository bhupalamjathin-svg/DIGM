<h1 align="center">DIGM</h1>
<p align="center"><b>Dynamic Importance-Gated Memory</b><br>
KV-cache eviction for LLM serving, built on vLLM</p>

<p align="center">
<img alt="vLLM 0.30.0" src="https://img.shields.io/badge/vLLM-0.30.0-blue">
<img alt="status: research prototype" src="https://img.shields.io/badge/status-research%20prototype-orange">
<img alt="patent pending" src="https://img.shields.io/badge/patent-pending-lightgrey">
<img alt="licence: PolyForm Noncommercial 1.0.0" src="https://img.shields.io/badge/licence-PolyForm%20Noncommercial%201.0.0-green">
</p>

<p align="center">
<a href="#about">About</a> ·
<a href="#results">Results</a> ·
<a href="#getting-started">Getting started</a> ·
<a href="#reproducing-the-paper">Reproducing the paper</a> ·
<a href="#contributing">Contributing</a> ·
<a href="#citation">Citation</a>
</p>

<p align="center">
<img src="docs/overview.svg" alt="With a full KV cache, four requests fill GPU memory and two wait. With DIGM, each request keeps only its important tokens, so six requests run in the same memory." width="900">
</p>

---

## About

An LLM server keeps a key-value (KV) cache entry for every token of every request. On a busy GPU this cache fills the memory and limits how many requests can run at once.

DIGM is an eviction layer for vLLM. It drops the least important cached tokens of a request, only when memory runs short, and it starts with the requests that tolerate eviction best.

On three 7–8B models it raises throughput by 1.4–1.8× and shortens time to first token by 17–41%, at a cost of 2–3 LongBench accuracy points. End-to-end latency gets longer; see [Results](#results).

## How it works

DIGM runs inside a stock vLLM engine, with no kernel changes.

| Part | What it does |
|---|---|
| **Per-head scoring** | Each attention head scores every cached token from its keys and keeps its own top tokens. No query is needed. |
| **Packing** | Kept tokens are copied to the front of the request's own blocks. The freed tail blocks go back to vLLM's block pool, so the memory is really returned. |
| **Position offset** | New tokens keep their original positions, so the model's position encoding stays correct after eviction. |
| **Preemption and chunked prefill** | An evicted request can still be preempted safely, and a long prompt can be evicted while it is being prefilled. |
| **Governor** | Eviction happens only when memory runs short. Requests shrink in steps of 12.5% of their context, never below 25%. |
| **Sensitivity** | A statistic of each request's scores predicts how well it tolerates eviction. Tolerant requests shrink first. |

When DIGM keeps every token, the output is identical to full-cache vLLM.

## Results

Three 7–8B models (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B), AWQ 4-bit, RTX 4080 16 GB. Full details are in the paper.

**Accuracy when keeping 25% of the context**

| Model | LongBench, full | LongBench, DIGM 25% | RULER-4K, full | RULER-4K, DIGM 25% |
|---|---|---|---|---|
| Llama-3.1-8B | 48.5 | 46.2 | 84.3 | 73.8 |
| Qwen2.5-7B | 48.9 | 46.9 | 84.7 | 65.4 |
| Mistral-7B | 47.0 | 43.6 | 82.1 | 68.1 |

**Serving**

| Measure | Effect of eviction |
|---|---|
| Throughput | 1.4–1.8× higher |
| Time to first token | 17–41% shorter |
| Median end-to-end latency | 15–60% longer, because the freed memory is spent on running more requests at once |
| Sensitivity-aware governor vs plain governor | better in 37 of 56 paired runs, worse in 3, by up to 7 accuracy points |

RULER is a needle-retrieval benchmark and is the hardest case for eviction. LongBench is closer to ordinary long-context tasks.

## Getting started

**Requirements**

- An NVIDIA GPU. The 7–8B runs used an RTX 4080 (16 GB); 3B models fit on 8 GB.
- vLLM 0.30.0 exactly. DIGM reaches into vLLM internals, so other versions will likely break.

```bash
git clone https://github.com/bhupalamjathin-svg/DIGM.git && cd DIGM
python3 -m venv ~/vllm_env && source ~/vllm_env/bin/activate
pip install -r requirements.txt
```

Models and datasets download from Hugging Face on first run (`simonjegou/ruler`, `Xnhyacinth/LongBench`).

**Quick check (3B model)**

```bash
cd paper_artifact
DIGM_SEL=head:cos python3 digm_vllm_v18.py
```

`DIGM_SEL=head:cos` selects the per-head scoring used in the paper. Without it the script falls back to an older single-layer selection (`one`), kept only for comparison.

The script loads `AMead10/Llama-3.2-3B-Instruct-AWQ` and runs each RULER test question four ways:

| Check | What it does | What to expect |
|---|---|---|
| A | Full KV cache, no eviction | baseline answer |
| B | "Eviction" that keeps every token | **must equal A exactly** |
| C | DIGM keeps 25% of the context | answer + number of freed blocks |
| D | Same as C without the position offset (first 5 questions) | shows why the offset is needed |

On an RTX 4070 laptop this takes under two minutes once the model has downloaded, and ends with:

```
identity eviction reproduces Full KV exactly : 26/26   (must be all)
per-head PERMUTED slots reproduce Full KV    : 26/26
evicted minus Full KV: -15.9 points, 95% CI [-22.1, -9.7]
```

If the first line is not N out of N, the vLLM version is wrong or something is broken. Stop there.
The accuracy drop is large here because this check uses RULER on a 3B model. Use `--longbench` for LongBench.

## Reproducing the paper

One serving run: sensitivity-aware governor with a 25% floor, mixed 4K/8K/16K prompts, bursty arrivals, at 1× and 2× of full-cache capacity, Llama-3.1-8B.

```bash
cd paper_artifact
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
DIGM_SEL=head:cos DIGM_DEBUG=0 \
DIGM_MIX=4096:0.5,8192:0.3,16384:0.2 DIGM_BURST=2.0 DIGM_MAXLEN=24576 DIGM_NREQ=40 DIGM_GEN=256 \
DIGM_SEED=0 DIGM_SENS_FLOOR=0.25 DIGM_RATES=0.564,1.128 DIGM_SERVE_MODES=sens \
python3 digm_vllm_v18.py --serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 4 0.85
```

The request rates (`DIGM_RATES`) were measured on the RTX 4080. On another GPU the capacity differs, so the same rates will not mean 1× and 2×.

All runs: `bash run_lab_tue.sh`, then `bash run_lab_wed.sh`. They take many hours, skip any run that already finished, and expect the environment at `~/vllm_env`. The summaries they produced for the paper are in `paper_artifact/results/`.

**Modes**

| Flag | What it runs |
|---|---|
| (none) | the quick check above |
| `--throughput` | all requests submitted at once: full cache vs eviction |
| `--serve` | timed arrivals: time to first token, per-token time, latency, accuracy |
| `--stress` | small memory pool: preemption after eviction, with and without restore |
| `--long` | eviction during chunked prefill, for prompts longer than the memory pool |
| `--longbench` | LongBench accuracy at a fixed budget |

Settings are `DIGM_*` environment variables. They are commented near the top of the script and where each mode starts.

## Repository layout

```
docs/overview.svg      the diagram at the top of this page
paper_artifact/
  digm_vllm_v18.py     the script behind the paper's serving results (system + benchmarks in one file)
  run_lab_tue.sh       governor vs trigger vs sensitivity-aware governor; LongBench serving
  run_lab_wed.sh       sensitivity-aware governor with a 25% floor (final policy); more seeds
  results/             raw summaries from those two scripts
```

`digm_vllm_v18.py` is one file on purpose: it is the exact code that produced the numbers.

## Limitations

- vLLM 0.30.0 only.
- Single GPU, in-process engine, prefix caching off, async scheduling off.
- The paper's runs used eager mode (`DIGM_EAGER=1`, the default here).
- Other figures in the paper came from earlier versions of this script. They are not in this repo yet.

## Roadmap

- Split the script into a small library (core, benchmark, scoring).
- Port to the current vLLM release.
- Larger models and multi-GPU.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report a bug, what to check before a pull request, and how contributions are licensed.

## Citation

Will be added with the preprint.

## Patent and licence

Patent pending: Indian Patent Application No. 202641128952.

Licence: [PolyForm Noncommercial 1.0.0](LICENSE.md). You may use, change and share this code for research, study and other non-commercial purposes. For commercial use, please open an issue to contact the authors.

Required Notice: Copyright (c) 2026 Sai Jathin Bhupalam and Naveen N C

## Acknowledgements

DIGM is built on [vLLM](https://github.com/vllm-project/vllm). Accuracy is measured on RULER and LongBench.

The idea, experiments and results are the authors'. An AI assistant (Claude) was used heavily to write the code, the scripts and this README. The authors ran every experiment and checked the results.
