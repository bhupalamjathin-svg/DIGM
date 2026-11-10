#!/bin/bash
# DIGM LAB follow-up (RTX 4080 16 GB). Resumable; most important first. ~1.5 h for part A.
# Tests the fix for sens losing at 2x on Llama/Mistral: same sens ordering, but tolerant floor 25% (no deeper squeeze).
#   bash run_lab_wed.sh        Results: cat logs/WED.txt
D="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; cd "$D"; mkdir -p "$D/logs"
LOG="$D/logs/wed.log"
step() { echo -e "\n===== $(date '+%H:%M:%S')  $1 =====" | tee -a "$LOG"; }
grep -q "DIGM_SENS_FLOOR" "$D/digm_vllm_v18.py" 2>/dev/null || { echo "  !! digm_vllm_v18.py missing or wrong version" | tee -a "$LOG"; exit 1; }
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
P="python3 $D/digm_vllm_v18.py"; F="grep --line-buffered -v IndexKernel"; U=0.85
LL=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4; QW=Qwen/Qwen2.5-7B-Instruct-AWQ; MI=solidrust/Mistral-7B-Instruct-v0.3-AWQ
declare -A M=( [llama]=$LL [qwen]=$QW [mistral]=$MI )
declare -A RM=( [llama]=0.564,1.128 [qwen]=0.705,1.41 [mistral]=0.51,1.02 )
S="DIGM_SEL=head:cos DIGM_DEBUG=0"
MIX="DIGM_MIX=4096:0.5,8192:0.3,16384:0.2 DIGM_BURST=2.0 DIGM_MAXLEN=24576 DIGM_NREQ=40 DIGM_GEN=256"
LBW="DIGM_WORKLOAD=longbench DIGM_BURST=2.0 DIGM_MAXLEN=16384 DIGM_NREQ=60 DIGM_GEN=128"
R() { local name=$1 n=$2; shift 2
  if [ -s "$D/logs/$name.txt" ] && ! grep -qE "Traceback|Error:" "$D/logs/$name.txt"; then echo "  skip $name (done)" | tee -a "$LOG"; return; fi
  step "$name"; env "$@" 2>&1 | $F | tail -$n | tee "$D/logs/$name.txt" | tee -a "$LOG"; }
source ~/vllm_env/bin/activate
# A) sens with a 25% floor for everyone (ordering only), mixed workload, 1x and 2x, seeds 0-4 (same seeds/rates as Tuesday)
for sd in 0 1 2 3 4; do for m in llama mistral qwen; do
  R w_${m}_mixed_s$sd 6 $S $MIX DIGM_SEED=$sd DIGM_RUN=_w$sd DIGM_SENS_FLOOR=0.25 DIGM_RATES=${RM[$m]} DIGM_SERVE_MODES=sens $P --serve ${M[$m]} 4 $U
done; done
# B) LongBench serving, more seeds (Tuesday's s1 runs were stopped)
for sd in 1 2; do for m in llama mistral qwen; do
  R t_${m}_lbserve_s$sd 20 $S $LBW DIGM_SEED=$sd DIGM_RUN=_lb$sd DIGM_SERVE_MODES=full,digm,trigger,elastic,sens $P --serve ${M[$m]} 4 $U
done; done
deactivate
step "ALL DONE"
{ for f in $(ls "$D"/logs/w_*.txt "$D"/logs/t_*_lbserve_s*.txt | sort); do echo "## $(basename $f .txt)"; grep -E "capacity|^ +[0-9.]+ +(full|digm|trigger|elastic|sens) " "$f"; done
  echo; echo "######## errors"; grep -i -E "Traceback|Error:|Killed|out of memory" "$LOG" | tail -20; } > "$D"/logs/WED.txt
echo "Done: cat $D/logs/WED.txt" | tee -a "$LOG"
