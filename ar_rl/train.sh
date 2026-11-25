#!/bin/bash
# GRPO training of autoregressive (AR) code LLMs with code-specific rewards.
#
# Usage:
#   bash train.sh <model> <difficulty> <reward_set> [extra TRL/GRPO CLI overrides...]
#
#   <model>       qwen2.5-coder-7b | seed-coder-8b
#   <difficulty>  easy | medium | hard   (AceCode-87K split; see ../data/README.md to download)
#                 example                (32-problem sample in ../data/example.jsonl, for testing)
#   <reward_set>  single reward : format | syntax | static | static_semantic | similarity | semantic
#                 combinations  : reward2 | reward2_static | reward2_ss | reward3 | reward4 | reward4_ss |
#                                 reward4_sim | reward5 | reward5_ss   (or any "a+b+c" list, e.g. static+semantic)
#                 curriculum    : dynamic3 | dynamic4      (threshold via $DYNAMIC_THRESHOLD, default 0.8)
#
# Examples:
#   bash train.sh qwen2.5-coder-7b hard semantic
#   bash train.sh seed-coder-8b hard static
#   DYNAMIC_THRESHOLD=0.5 bash train.sh qwen2.5-coder-7b hard dynamic3
#
# Required env vars: E2B_API_KEY (code execution sandbox for the semantic reward)
# Optional env vars: DATA_FILE (override the training .jsonl), WANDB_API_KEY, HF_TOKEN, NUM_GPUS (default 1), ROUTER_PORT (default 8001),
#                    START_ROUTER (default 1), OUTPUT_ROOT (default ./experiment_result)
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${1:?model required}; DIFFICULTY=${2:?difficulty required}; REWARD_SET=${3:?reward_set required}
shift 3

NUM_GPUS=${NUM_GPUS:-1}
ROUTER_PORT=${ROUTER_PORT:-8001}
START_ROUTER=${START_ROUTER:-1}
OUTPUT_ROOT=${OUTPUT_ROOT:-./experiment_result}
DYNAMIC_THRESHOLD=${DYNAMIC_THRESHOLD:-0.8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
: "${E2B_API_KEY:?Please export E2B_API_KEY (https://e2b.dev)}"

# ---- reward sets ----
# Short names -> reward functions in src/open_r1/rewards.py
declare -A FN=(
  [format]=code_format_correctness
  [syntax]=code_syntax_correctness
  [static]=code_static_checking_reward
  [static_semantic]=code_static_semantic_checking_reward
  [similarity]=code_similarity_checking_reward
  [semantic]=code_semantic_reward
)
# Named presets (as in the paper). Any other "+"-joined list of short names also works,
# e.g. "static+semantic"; prefix with "dynamic:" for curriculum scheduling in that order.
declare -A PRESET=(
  [reward2]=format+semantic
  [reward2_static]=static+semantic
  [reward2_ss]=static_semantic+semantic
  [reward3]=format+syntax+semantic
  [reward4]=format+syntax+static+semantic
  [reward4_ss]=format+syntax+static_semantic+semantic
  [reward4_sim]=format+syntax+similarity+semantic
  [reward5]=format+syntax+static+similarity+semantic
  [reward5_ss]=format+syntax+static_semantic+similarity+semantic
  [dynamic3]=dynamic:format+syntax+semantic
  [dynamic4]=dynamic:format+syntax+static_semantic+semantic
)
SPEC=${PRESET[$REWARD_SET]:-$REWARD_SET}
DYNAMIC=0
if [[ "$SPEC" == dynamic:* ]]; then DYNAMIC=1; SPEC=${SPEC#dynamic:}; fi
REWARDS=()
IFS='+' read -r -a PARTS <<< "$SPEC"
for p in "${PARTS[@]}"; do
  [ -n "${FN[$p]:-}" ] || { echo "Unknown reward '$p' in reward_set '$REWARD_SET' (valid: ${!FN[*]})"; exit 1; }
  REWARDS+=("${FN[$p]}")
done

WEIGHTS=()
for i in "${!REWARDS[@]}"; do
  if [ "$DYNAMIC" = 1 ] && [ "$i" -gt 0 ]; then WEIGHTS+=(0.0); else WEIGHTS+=(1.0); fi
done

if [ "$DYNAMIC" = 1 ]; then
  [ "$MODEL" = qwen2.5-coder-7b ] || echo "NOTE: the paper only runs dynamic rewards with qwen2.5-coder-7b; using its config for $MODEL." >&2
  CONFIG=recipes/qwen2.5-coder-7b_dynamic.yaml
  ENTRY=src/open_r1/grpo_dynamic.py
  RUN_NAME="${REWARD_SET//[+:]/_}_${DYNAMIC_THRESHOLD}"
  EXTRA=(--use_dynamic_rewards true --dynamic_reward_threshold "$DYNAMIC_THRESHOLD")
  [ "$MODEL" != qwen2.5-coder-7b ] && EXTRA+=(--model_name_or_path "$(grep '^model_name_or_path' recipes/$MODEL.yaml | awk '{print $2}' | tr -d '"')")
else
  CONFIG=recipes/$MODEL.yaml
  ENTRY=src/open_r1/grpo.py
  RUN_NAME="${REWARD_SET//[+:]/_}"
  EXTRA=()
fi
[ -f "$CONFIG" ] || { echo "Config not found: $CONFIG"; exit 1; }

if [ "$DIFFICULTY" = example ]; then
  DATA_FILE=${DATA_FILE:-../data/example.jsonl}
else
  DATA_FILE=${DATA_FILE:-../data/acecode_${DIFFICULTY}_with_inference_all_1.jsonl}
fi
[ -f "$DATA_FILE" ] || { echo "Training data not found: $DATA_FILE"; echo "See ../data/README.md for how to download AceCode-87K and build the splits."; exit 1; }

OUT_DIR="$OUTPUT_ROOT/$MODEL/$DIFFICULTY/$RUN_NAME"
mkdir -p "$OUT_DIR" logs

if [ "$START_ROUTER" = 1 ] && ! (echo > /dev/tcp/127.0.0.1/$ROUTER_PORT) 2>/dev/null; then
  echo "Starting E2B router on port $ROUTER_PORT ..."
  nohup python scripts/e2b_router.py --port "$ROUTER_PORT" > "logs/e2b_router_$ROUTER_PORT.log" 2>&1 &
  sleep 10
fi

[ -n "${WANDB_API_KEY:-}" ] && export WANDB_RUN_NAME="${MODEL}_${DIFFICULTY}_${RUN_NAME}"

echo "Model=$MODEL Difficulty=$DIFFICULTY Rewards=${REWARDS[*]} Weights=${WEIGHTS[*]} -> $OUT_DIR"
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes "$NUM_GPUS" --main_process_port $((29500 + ROUTER_PORT % 100)) \
  "$ENTRY" --config "$CONFIG" \
  --dataset_name "$DATA_FILE" \
  --output_dir "$OUT_DIR" \
  --e2b_router_url "0.0.0.0:$ROUTER_PORT" \
  --reward_funcs "${REWARDS[@]}" \
  --reward_weights "${WEIGHTS[@]}" \
  "${EXTRA[@]}" "$@" 2>&1 | tee "logs/${MODEL}_${DIFFICULTY}_${RUN_NAME}.log"
