#!/bin/bash
# EvalPlus (HumanEval+ / MBPP+) evaluation for masked diffusion models (DiffuCoder / Dream-Coder).
# Generation uses the Dream sampling pipeline in ../dream (evalplus_dream_gen.py), then
# evalplus.sanitize + evalplus.evaluate.
#
# Usage: bash run_evalplus_diffusion.sh <model_path_or_hf_id> [...]
#   e.g. bash run_evalplus_diffusion.sh apple/DiffuCoder-7B-Instruct ../../diffusion_rl/experiment_result/diffucoder-7b/hard/semantic
#
# Env: DATASETS (default "humaneval mbpp"), RESULTS_DIR (default ./results_diffusion),
#      GEN_ARGS (extra args for evalplus_dream_gen.py, e.g. "--steps 256 --max-gen-length 256")
set -uo pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { sed -n 2,10p "$0"; exit 1; }
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"   # makes `import dream` work

read -r -a DATASETS <<< "${DATASETS:-humaneval mbpp}"
RESULTS_DIR=${RESULTS_DIR:-./results_diffusion}
mkdir -p "$RESULTS_DIR/logs"
SUMMARY="$RESULTS_DIR/summary.txt"
FAILED=()

for MODEL in "$@"; do
  NAME=$(echo "${MODEL%/}" | tr '/' '.')
  for DATASET in "${DATASETS[@]}"; do
    RUN="${DATASET}_${NAME}"
    mkdir -p "$RESULTS_DIR/$DATASET"
    SAMPLES="$RESULTS_DIR/$DATASET/${RUN}_samples.jsonl"
    SANITIZED="${SAMPLES%.jsonl}-sanitized.jsonl"
    LOG="$RESULTS_DIR/logs/${RUN}.log"
    echo ">>> $MODEL on $DATASET (log: $LOG)"
    {
      set -e
      [ -s "$SAMPLES" ] || python evalplus_dream_gen.py --model "$MODEL" --dataset "$DATASET" --out "$SAMPLES" ${GEN_ARGS:-}
      [ -s "$SANITIZED" ] || evalplus.sanitize --samples "$SAMPLES"
      evalplus.evaluate --dataset "$DATASET" --samples "$SANITIZED"
    } &> "$LOG"
    if [ $? -eq 0 ]; then
      { echo "=== $RUN ==="; grep -A5 -i "pass@1" "$LOG"; echo; } >> "$SUMMARY"
    else
      echo "FAILED: $RUN (see $LOG)"; FAILED+=("$RUN")
    fi
  done
done
echo "Summary: $SUMMARY"
[ ${#FAILED[@]} -eq 0 ] || { printf 'Failed: %s\n' "${FAILED[@]}"; exit 1; }
