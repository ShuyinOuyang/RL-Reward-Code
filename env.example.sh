#!/bin/bash
# Copy to env.sh (git-ignored), fill in, and `source env.sh` before training/evaluation.
export E2B_API_KEY="<your-e2b-api-key>"       # required: sandboxed execution for code_semantic_reward
export WANDB_API_KEY="<your-wandb-api-key>"   # optional: Weights & Biases logging
export HF_TOKEN="<your-hf-token>"             # optional: gated models / Hub uploads
# export HF_HOME=/path/to/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
