#!/bin/bash
# LiveCodeBench (release_v6, code generation, greedy pass@1) for autoregressive models via vLLM.
# Must be run from this directory (lcb_runner reads few-shot files by relative path).
#
# Usage: bash run_lcb_ar.sh <base_model_id> <model_path> [<model_path> ...]
#   <base_model_id>  Qwen/Qwen2.5-Coder-7B-Instruct | ByteDance-Seed/Seed-Coder-8B-Instruct
#                    (selects the prompt template / lm_style registered in lcb_runner/lm_styles.py)
#   <model_path>     local checkpoint dir (or the HF id itself to evaluate the base model)
#   e.g. bash run_lcb_ar.sh Qwen/Qwen2.5-Coder-7B-Instruct ../../ar_rl/experiment_result/qwen2.5-coder-7b/hard/semantic
#
# Env: RELEASE_VERSION (default release_v6), RESULTS_ROOT (default ./eval_results)
set -uo pipefail
cd "$(dirname "$0")"
[ $# -ge 2 ] || { sed -n 2,11p "$0"; exit 1; }
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy-key-for-local-model}   # lcb_runner imports openai unconditionally

BASE_MODEL=$1; shift
RELEASE_VERSION=${RELEASE_VERSION:-release_v6}
RESULTS_ROOT=${RESULTS_ROOT:-./eval_results}
mkdir -p "$RESULTS_ROOT"

for LOCAL_PATH in "$@"; do
  RUN_NAME=$(echo "${LOCAL_PATH%/}" | rev | cut -d'/' -f1-3 | rev | tr '/' '_')
  OUT_DIR="$RESULTS_ROOT/$RUN_NAME"; mkdir -p "$OUT_DIR"
  echo ">>> $RUN_NAME (style: $BASE_MODEL, path: $LOCAL_PATH)"
  if [ -s "$OUT_DIR/run.log" ]; then echo "    exists, skipping"; continue; fi
  LP_ARGS=(); [ -e "$LOCAL_PATH" ] && LP_ARGS=(--local_model_path "$LOCAL_PATH")
  python -m lcb_runner.runner.main \
      --model "$BASE_MODEL" "${LP_ARGS[@]}" \
      --scenario codegeneration --evaluate \
      --release_version "$RELEASE_VERSION" \
      --temperature 0.0 --n 1 2>&1 | tee "$OUT_DIR/run.log"
done
echo "Results (per-run logs) under $RESULTS_ROOT; raw generations/metrics under ./output"
