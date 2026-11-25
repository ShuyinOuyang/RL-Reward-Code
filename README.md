# A Study of Reinforcement Learning Reward for Code Generation

Official code for **A Study of Reinforcement Learning Reward for Code Generation** (EACL 2027).

We study how code-specific reward signals (format, syntax, static analysis, similarity to a reference solution, and execution-based correctness) affect reinforcement learning (GRPO) for code generation in two paradigms:

* **Autoregressive (AR) LLMs:** Qwen2.5-Coder-7B-Instruct and Seed-Coder-8B-Instruct, trained with GRPO built on [open-r1](https://github.com/huggingface/open-r1).
* **Masked diffusion LLMs:** DiffuCoder-7B-Instruct and Dream-Coder-v0-Instruct-7B, trained with Coupled-GRPO built on [DiffuCoder / ml-diffucoder](https://github.com/apple/ml-diffucoder).

Models are evaluated on **HumanEval(+) / MBPP(+)** via [EvalPlus](https://github.com/evalplus/evalplus) and on **LiveCodeBench** (release_v6) via [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).

---

## Repository layout

```
.
├── data/                      # example.jsonl (32 problems for testing) + script to build the AceCode splits
├── ar_rl/                     # GRPO for autoregressive models (fork of huggingface/open-r1)
│   ├── src/open_r1/
│   │   ├── grpo.py            #   static-weight multi-reward GRPO
│   │   ├── grpo_dynamic.py    #   dynamic (curriculum) reward scheduling
│   │   ├── rewards.py         #   code reward functions (see "Reward functions")
│   │   └── configs.py
│   ├── recipes/               #   per-model YAML configs + accelerate/DeepSpeed configs
│   ├── scripts/e2b_router.py  #   E2B sandbox router used by the execution reward
│   └── train.sh               #   single entry point for all AR experiments
├── diffusion_rl/              # Coupled-GRPO for diffusion models (fork of apple/ml-diffucoder)
│   ├── src/open_r1/
│   │   ├── grpo.py / grpo_dynamic.py
│   │   ├── coupled_grpo.py    #   DiffuGRPOTrainer (Coupled-GRPO)
│   │   └── rewards.py
│   ├── recipes/
│   └── train.sh               #   single entry point for all diffusion experiments
├── evaluation/
│   ├── dream/                 # Dream/DiffuCoder sampling code shared by the diffusion evaluators
│   ├── evalplus/              # HumanEval+/MBPP+ (AR: vLLM greedy; diffusion: evalplus_dream_gen.py)
│   └── livecodebench/         # fork of LiveCodeBench (lcb_runner) + diffusion generator
└── envs/                      # frozen pip environments used for the paper
```

## Installation

We used Python 3.11, CUDA 12.x and PyTorch 2.11. Every component has its own environment. The two training packages are both named `open_r1`, so they cannot share one environment.

```bash
# 1) AR RL
conda create -n rl-ar python=3.11 -y && conda activate rl-ar
pip install torch==2.11.0 vllm==0.23.0
cd ar_rl && pip install -e ".[dev]" && cd ..
pip install pylint==4.0.6 fastapi uvicorn   # static rewards call pylint; the E2B router needs fastapi/uvicorn
# (exact versions: envs/requirements_ar_rl.frozen.txt)

# 2) Diffusion RL (same pins as AR RL: trl@1bca495, transformers@acdbe62)
conda create -n rl-diffusion python=3.11 -y && conda activate rl-diffusion
pip install torch==2.11.0 vllm==0.23.0
cd diffusion_rl && pip install -e ".[dev]" && cd ..
pip install pylint==4.0.6 fastapi uvicorn
# (exact versions: envs/requirements_diffusion_rl.frozen.txt)
# Note: transformers must be the pinned 4.52-dev commit. With transformers>=4.56 and
# accelerate 1.4.0, GRPOConfig fails with "NameError: ParallelismConfig is not defined".

# 3) EvalPlus
conda create -n evalplus python=3.11 -y && conda activate evalplus
pip install "evalplus[vllm] @ git+https://github.com/evalplus/evalplus@26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
# AR evaluation:        envs/requirements_evalplus_ar.frozen.txt
# Diffusion evaluation: envs/requirements_evalplus_diffusion.frozen.txt (transformers 4.57, jinja2, termcolor)

# 4) LiveCodeBench
conda create -n lcb python=3.11 -y && conda activate lcb
cd evaluation/livecodebench && pip install -e . && cd ../..
# (exact versions: envs/requirements_lcb_ar.frozen.txt / envs/requirements_lcb_diffusion.frozen.txt)
```

`setup.py` pins `torch==2.6.0` (inherited from upstream open-r1). We ran with `torch==2.11.0`; the upstream pins should also work on GPUs supported by torch 2.6.

### Credentials

Copy `env.example.sh`, fill it in, and `source` it before training:

| Variable | Needed for |
|---|---|
| `E2B_API_KEY` | **Required for training.** The execution reward (`code_semantic_reward`) runs unit tests in [E2B](https://e2b.dev) sandboxes. |
| `WANDB_API_KEY` | Optional, for Weights & Biases logging (`report_to: wandb`). |
| `HF_TOKEN` | Optional, for gated models or pushing to the Hub. |

## Data

Training uses the easy/medium/hard splits of [TIGER-Lab/AceCode-87K](https://huggingface.co/datasets/TIGER-Lab/AceCode-87K) (MIT). The full splits are not included: [data/README.md](data/README.md) explains how to download and build them with `data/process_data.py`. For a quick test, `data/example.jsonl` holds 32 problems; train on it with `bash train.sh <model> example <reward_set>`.

## Reward functions

All rewards are implemented in `*/src/open_r1/rewards.py`:

| Name in config (`reward_funcs`) | `train.sh` short name | Signal |
|---|---|---|
| `code_format_correctness` | `format` | Output follows the required format and contains a fenced Python code block |
| `code_syntax_correctness` | `syntax` | Extracted code parses (`ast.parse`) |
| `code_static_checking_reward` | `static` | Normalised pylint score |
| `code_static_semantic_checking_reward` | `static_semantic` | Normalised pylint score with only warning/error/fatal (W, E, F) checks enabled |
| `code_similarity_checking_reward` | `similarity` | Max token + AST similarity to the reference solutions |
| `code_semantic_reward` | `semantic` | Fraction of unit tests passed (executed in E2B) |

## Training

`train.sh` starts the E2B router if it isn't already running, selects the YAML recipe, and overrides the dataset split, reward set, and output directory from the command line.

```bash
source env.example.sh

# Autoregressive (1 GPU per run)
cd ar_rl
bash train.sh qwen2.5-coder-7b hard semantic
bash train.sh seed-coder-8b   hard static
DYNAMIC_THRESHOLD=0.8 bash train.sh qwen2.5-coder-7b hard dynamic4

# Diffusion (1 GPU per run)
cd ../diffusion_rl
bash train.sh diffucoder-7b  hard semantic
bash train.sh dream-coder-7b hard similarity
DYNAMIC_THRESHOLD=0.5 bash train.sh diffucoder-7b hard dynamic3
```

Checkpoints are written to `<ar_rl|diffusion_rl>/experiment_result/<model>/<difficulty>/<reward_set>/`. Any extra arguments after the three positional ones are forwarded to the trainer, for example `--num_train_epochs 2`.

### Experiment grid

| Experiment | Model | Difficulty | `reward_set` |
|---|---|---|---|
| Single reward | qwen2.5-coder-7b | easy, medium, hard | `format` `syntax` `static` `similarity` `semantic` (+ `static_semantic` on hard) |
| Single reward | seed-coder-8b | hard | `format` `syntax` `static` `similarity` `semantic` |
| Single reward | diffucoder-7b | easy, medium, hard | `format` `syntax` `static` `static_semantic` `similarity` `semantic` |
| Single reward | dream-coder-7b | hard | `format` `syntax` `static` `static_semantic` `similarity` `semantic` |
| Reward combination | qwen2.5-coder-7b | hard | `reward2` `reward3` `reward4` `reward5` |
| Reward combination | diffucoder-7b | easy, medium, hard | `reward2` `reward2_static` `reward3` `reward4` `reward4_ss` `reward4_sim` `reward5` `reward5_ss` (+ `reward2_ss` on hard) |
| Reward combination | dream-coder-7b | hard | `reward2` `reward3` `reward4` `reward4_ss` `reward4_sim` `reward5` `reward5_ss` |
| Dynamic / curriculum | qwen2.5-coder-7b, diffucoder-7b | hard | `dynamic3` `dynamic4`, each with `DYNAMIC_THRESHOLD` ∈ {0.5, 0.8} |

Reward-set presets:

| Preset | Rewards |
|---|---|
| `reward2` | format + semantic |
| `reward2_static` | static + semantic |
| `reward2_ss` | static_semantic + semantic |
| `reward3` | format + syntax + semantic |
| `reward4` | format + syntax + static + semantic |
| `reward4_ss` | format + syntax + static_semantic + semantic |
| `reward4_sim` | format + syntax + similarity + semantic |
| `reward5` | format + syntax + static + similarity + semantic |
| `reward5_ss` | format + syntax + static_semantic + similarity + semantic |
| `dynamic3` | format → syntax → semantic |
| `dynamic4` | format → syntax → static → semantic (AR: format → syntax → static_semantic → semantic) |

Any other combination can be given directly as a `+`-joined list of short names, for example `bash train.sh diffucoder-7b hard static+semantic`. Prefix the list with `dynamic:` to schedule the rewards in that order, for example `dynamic:format+semantic`.

In the dynamic runs, all of the weight starts on the first reward. When that reward's batch mean reaches the threshold, the weight moves entirely to the next reward (`grpo_dynamic.py`).

Example sweep:

```bash
cd ar_rl
for d in easy medium hard; do for r in format syntax static similarity semantic; do
  bash train.sh qwen2.5-coder-7b $d $r
done; done
```

## Evaluation

### EvalPlus (HumanEval+ / MBPP+)

```bash
cd evaluation/evalplus
# AR models: greedy decoding with vLLM
bash run_evalplus_ar.sh Qwen/Qwen2.5-Coder-7B-Instruct ../../ar_rl/experiment_result/qwen2.5-coder-7b/hard/semantic
# Diffusion models: greedy, 256 diffusion steps, 256 max new tokens
bash run_evalplus_diffusion.sh apple/DiffuCoder-7B-Instruct ../../diffusion_rl/experiment_result/diffucoder-7b/hard/semantic
```

### LiveCodeBench (release_v6, code generation, pass@1, greedy)

```bash
cd evaluation/livecodebench
# AR models: the first argument selects the prompt style registered in lcb_runner/lm_styles.py
bash run_lcb_ar.sh Qwen/Qwen2.5-Coder-7B-Instruct      ../../ar_rl/experiment_result/qwen2.5-coder-7b/hard/semantic
bash run_lcb_ar.sh ByteDance-Seed/Seed-Coder-8B-Instruct ../../ar_rl/experiment_result/seed-coder-8b/hard/semantic
# Diffusion models
bash run_lcb_diffusion.sh apple/DiffuCoder-7B-Instruct ../../diffusion_rl/experiment_result/diffucoder-7b/hard/semantic
```

Changes compared with upstream LiveCodeBench:
* Added Seed-Coder-8B-Instruct to `lm_styles.py` and the code-generation prompts.
* Added `livecodebench_dream_gen.py` for diffusion-model generation; scoring uses the unchanged `codegen_metrics`.

## Acknowledgements

This repository builds on:
* [huggingface/open-r1](https://github.com/huggingface/open-r1) (Apache-2.0)
* [apple/ml-diffucoder](https://github.com/apple/ml-diffucoder) (Apple license)
* [Gen-Verse/dLLM-RL](https://github.com/Gen-Verse/dLLM-RL), [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) and [DreamLM/Dream](https://github.com/DreamLM/Dream) (Apache-2.0), for the diffusion sampling code used in evaluation
* [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) (MIT)
* [EvalPlus](https://github.com/evalplus/evalplus) (Apache-2.0)
* [AceCode-87K](https://huggingface.co/datasets/TIGER-Lab/AceCode-87K) (MIT)

Each component keeps the license of the project it is derived from. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for which license covers which files, and for the data and model licenses.
