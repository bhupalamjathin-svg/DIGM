#!/bin/bash
# DIGM LAB Tuesday (RTX 4080 16 GB), 11:00 -> overnight. Resumable; most important first.
#   bash run_lab_tue.sh        Results: cat logs/TUE.txt
D="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; cd "$D"; mkdir -p "$D/logs"
LOG="$D/logs/tue.log"
step() { echo -e "\n===== $(date '+%H:%M:%S')  $1 =====" | tee -a "$LOG"; }
grep -q "DIGM_WORKLOAD" "$D/digm_vllm_v18.py" 2>/dev/null || { echo "  !! digm_vllm_v18.py missing or wrong version" | tee -a "$LOG"; exit 1; }
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
P="python3 $D/digm_vllm_v18.py"; F="grep --line-buffered -v IndexKernel"; U=0.85
LL=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4; QW=Qwen/Qwen2.5-7B-Instruct-AWQ; MI=solidrust/Mistral-7B-Instruct-v0.3-AWQ
declare -A M=( [llama]=$LL [qwen]=$QW [mistral]=$MI )
declare -A RM=( [llama]=0.564,1.128 [qwen]=0.705,1.41 [mistral]=0.51,1.02 )   # mixed: 1x, 2x of Full KV capacity (Friday)
declare -A R2=( [llama]=1.622 [qwen]=2.143 [mistral]=1.441 )               # 4K: 2x (Thursday)
S="DIGM_SEL=head:cos DIGM_DEBUG=0"
MIX="DIGM_MIX=4096:0.5,8192:0.3,16384:0.2 DIGM_BURST=2.0 DIGM_MAXLEN=24576 DIGM_NREQ=40 DIGM_GEN=256"
LBW="DIGM_WORKLOAD=longbench DIGM_BURST=2.0 DIGM_MAXLEN=16384 DIGM_NREQ=60 DIGM_GEN=128"
R() { local name=$1 n=$2; shift 2
  if [ -s "$D/logs/$name.txt" ] && ! grep -qE "Traceback|Error:" "$D/logs/$name.txt"; then echo "  skip $name (done)" | tee -a "$LOG"; return; fi
  step "$name"; env "$@" 2>&1 | $F | tail -$n | tee "$D/logs/$name.txt" | tee -a "$LOG"; }
source ~/vllm_env/bin/activate

# A) THE DECIDING RUN: sensitivity-aware governor vs plain governor vs trigger, mixed workload, 1x and 2x, seeds 0-2
for sd in 0 1 2; do for m in llama qwen mistral; do
  R t_${m}_mixed_s$sd 10 $S $MIX DIGM_SEED=$sd DIGM_RUN=_t$sd DIGM_RATES=${RM[$m]} DIGM_SERVE_MODES=trigger,elastic,sens $P --serve ${M[$m]} 4 $U
done; done
# B) realistic heterogeneous traffic: LongBench prompts (10 tasks) served with bursty arrivals, 0.5x/1x/2x (auto rates)
for m in llama qwen mistral; do
  R t_${m}_lbserve_s0 20 $S $LBW DIGM_SEED=0 DIGM_RUN=_lb0 DIGM_SERVE_MODES=full,digm,trigger,elastic,sens $P --serve ${M[$m]} 4 $U
done
# C) uniform 4K at 2x: sens vs governor (sens should be neutral here)
for sd in 0 1 2; do for m in llama qwen mistral; do
  R t_${m}_4k2x_s$sd 6 $S DIGM_NREQ=48 DIGM_GEN=256 DIGM_SEED=$sd DIGM_RUN=_t4k$sd DIGM_RATES=${R2[$m]} DIGM_SERVE_MODES=elastic,sens $P --serve ${M[$m]} 4 $U
done; done
# D) overnight: more seeds (mixed 3-4, LongBench serving 1-2) + trigger thresholds on mixed
for sd in 3 4; do for m in llama qwen mistral; do
  R t_${m}_mixed_s$sd 10 $S $MIX DIGM_SEED=$sd DIGM_RUN=_t$sd DIGM_RATES=${RM[$m]} DIGM_SERVE_MODES=trigger,elastic,sens $P --serve ${M[$m]} 4 $U
done; done
for sd in 1 2; do for m in llama qwen mistral; do
  R t_${m}_lbserve_s$sd 20 $S $LBW DIGM_SEED=$sd DIGM_RUN=_lb$sd DIGM_SERVE_MODES=full,digm,trigger,elastic,sens $P --serve ${M[$m]} 4 $U
done; done
for m in llama qwen mistral; do for t in "0.90 0.80" "0.80 0.70"; do set -- $t
  R t_${m}_mixed_trig$1 6 $S $MIX DIGM_SEED=0 DIGM_RUN=_tt$1 DIGM_RATES=${RM[$m]} DIGM_TRIG_ON=$1 DIGM_TRIG_OFF=$2 DIGM_SERVE_MODES=trigger $P --serve ${M[$m]} 4 $U
done; done
deactivate
step "ALL DONE"
{ for f in $(ls "$D"/logs/t_*.txt | sort); do echo "## $(basename $f .txt)"; grep -E "capacity|^ +[0-9.]+ +(full|digm|trigger|elastic|sens) " "$f"; done
  echo; echo "######## errors"; grep -i -E "Traceback|Error:|Killed|out of memory" "$LOG" | tail -20; } > "$D"/logs/TUE.txt
echo "Done: cat $D/logs/TUE.txt" | tee -a "$LOG"
