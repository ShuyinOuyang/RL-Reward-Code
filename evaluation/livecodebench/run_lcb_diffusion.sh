#!/bin/bash
# LiveCodeBench (release_v6, code generation, greedy pass@1) for masked diffusion models.
# Generation uses the Dream sampling pipeline in ../dream; scoring uses lcb_runner's codegen_metrics.
#
# Usage: bash run_lcb_diffusion.sh <model_path_or_hf_id> [...]
#   e.g. bash run_lcb_diffusion.sh apple/DiffuCoder-7B-Instruct ../../diffusion_rl/experiment_result/diffucoder-7b/hard/semantic
#
# Env: RELEASE_VERSION (default release_v6), RESULTS_DIR (default ./lcb_results),
#      NUM_PROCESS_EVALUATE (default 16), EVAL_TIMEOUT (default 6)
set -uo pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { sed -n 2,9p "$0"; exit 1; }
export PYTHONPATH="$(cd .. && pwd):$(pwd):${PYTHONPATH:-}"   # `import dream` and `import lcb_runner`
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy-key-for-local-model}

RELEASE_VERSION=${RELEASE_VERSION:-release_v6}
RESULTS_DIR=${RESULTS_DIR:-./lcb_results}
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/pass1_summary.csv"
[ -f "$SUMMARY" ] || echo "model,pass@1" > "$SUMMARY"

for MODEL in "$@"; do
  SAFE_NAME=$(echo "${MODEL%/}" | tr '/' '.')
  OUT="$RESULTS_DIR/lcb_${RELEASE_VERSION}_${SAFE_NAME}_codelist.json"
  echo ">>> $MODEL"
  python livecodebench_dream_gen.py --model "$MODEL" --release-version "$RELEASE_VERSION" \
      --out "$OUT" --n-samples 1 --temperature 0.0 \
      --num-process-evaluate "${NUM_PROCESS_EVALUATE:-16}" --eval-timeout "${EVAL_TIMEOUT:-6}" --evaluate
  PASS1=$(python -c "import json;print(json.load(open('${OUT%.json}-metrics.json')).get('pass@1','NA'))" 2>/dev/null || echo NA)
  echo "$MODEL,$PASS1" | tee -a "$SUMMARY"
done
column -s, -t "$SUMMARY"
