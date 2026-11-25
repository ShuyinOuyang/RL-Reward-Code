#!/bin/bash
# EvalPlus (HumanEval+ / MBPP+) evaluation for autoregressive models, greedy decoding via vLLM.
#
# Usage: bash run_evalplus_ar.sh <model_path_or_hf_id> [<model_path_or_hf_id> ...]
#   e.g. bash run_evalplus_ar.sh Qwen/Qwen2.5-Coder-7B-Instruct ../../ar_rl/experiment_result/qwen2.5-coder-7b/hard/semantic
#
# Env: DATASETS (default "humaneval mbpp"), RESULTS_DIR (default ./results_ar)
set -uo pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { sed -n 2,7p "$0"; exit 1; }

read -r -a DATASETS <<< "${DATASETS:-humaneval mbpp}"
RESULTS_DIR=${RESULTS_DIR:-./results_ar}
mkdir -p "$RESULTS_DIR/logs"
SUMMARY="$RESULTS_DIR/summary.tsv"
[ -f "$SUMMARY" ] || printf "model\tdataset\tbase_pass@1\tplus_pass@1\n" > "$SUMMARY"

for MODEL in "$@"; do
  NAME=$(echo "${MODEL%/}" | tr '/' '_')
  for DATASET in "${DATASETS[@]}"; do
    LOG="$RESULTS_DIR/logs/${NAME}_${DATASET}.log"
    echo ">>> $MODEL on $DATASET (log: $LOG)"
    evalplus.evaluate --model "$MODEL" --dataset "$DATASET" --backend vllm --greedy \
      --root "$RESULTS_DIR" > "$LOG" 2>&1
    BASE=$(grep -A1 -i "$DATASET (base tests)" "$LOG" | grep -i "pass@1" | awk '{print $NF}' | tr -d '\r\n')
    PLUS=$(grep -A1 -i "$DATASET+ (base + extra tests)" "$LOG" | grep -i "pass@1" | awk '{print $NF}' | tr -d '\r\n')
    printf "%s\t%s\t%s\t%s\n" "$MODEL" "$DATASET" "${BASE:-ERROR}" "${PLUS:-ERROR}" | tee -a "$SUMMARY"
  done
done
echo "Summary: $SUMMARY"
