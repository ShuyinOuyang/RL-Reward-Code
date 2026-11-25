# Adapted from Gen-Verse/dLLM-RL (sample/dream_sample.py), https://github.com/Gen-Verse/dLLM-RL,
# licensed under the Apache License, Version 2.0 (see ../LICENSE-APACHE-2.0). Modified for this project.
"""
Generate LiveCodeBench completions using the Dream-model multi-GPU sampling
pipeline (_sample / DreamGenerationConfig), and export either:

  (a) a scored JSON (pass@1/pass@5/... via lcb_runner's codegen_metrics), or
  (b) a code_list JSON ready for `python -m lcb_runner.runner.custom_evaluator`

This is a direct adaptation of an EvalPlus (HumanEval+/MBPP+) generation
script. All diffusion sampling internals (_sample, sample_tokens, the
multi-GPU worker + OOM backoff) are UNCHANGED -- only the benchmark-facing
pieces differ: dataset loading, prompting, code extraction bookkeeping,
and output/scoring.

Usage:
    python livecodebench_dream_gen.py --model apple/DiffuCoder-7B-Instruct \
        --release-version release_v6 --start-date 2025-02-01 --evaluate

    python livecodebench_dream_gen.py --model apple/DiffuCoder-7B-Instruct \
        --release-version release_v6 --out lcb_v6_codelist.json
    # then:
    python -m lcb_runner.runner.custom_evaluator --custom_output_file lcb_v6_codelist.json
"""

from __future__ import annotations
import argparse
import json
import os
import re
import types
import random
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.distributions as dists
from jinja2 import Template
from termcolor import cprint
from tqdm import tqdm
from transformers.utils import ModelOutput

from dream import DreamTokenizer
from dream.modeling_dream import DreamModel
from dream.generation_utils_block import DreamGenerationMixin, DreamGenerationConfig

# --- LiveCodeBench imports (replace evalplus.data) ---------------------
from lcb_runner.benchmarks.code_generation import (
    CodeGenerationProblem,
    load_code_generation_dataset,
)


# ---------------------------------------------------------------------------
# Sampling internals -- UNCHANGED from the EvalPlus script
# ---------------------------------------------------------------------------

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, tar=None):
    logits = logits.float()

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    dist = dists.Categorical(logits=logits)
    x0 = dist.sample()
    probs = dist.probs

    if temperature > 0:
        target = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        target, x0 = probs.max(dim=-1)

    if tar == "confidence":
        return target, x0

    if tar == "margin_confidence":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        target = top1_probs - top2_probs

    if tar == "neg_entropy":
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        target = torch.sum(probs * log_probs, dim=-1)

    return target, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


@torch.no_grad()
def _sample(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    block_length: Optional[int] = 32,
    use_cache: bool = False,
    further_horizon: int = 128,
    mask_token_id: int = 151666,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    pad_target_penalty: float = 1.0,
    unmask_threshold: float = 0.9,
) -> Union[DreamModelOutput, torch.LongTensor]:
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    steps = generation_config.steps
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    tar = generation_config.tar
    alg_temp = generation_config.alg_temp
    cgws = further_horizon

    histories = [] if (return_dict_in_generate and output_history) else None

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    gen_length = max_length - input_ids.shape[1]

    if block_length is None:
        block_length = gen_length

    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length

    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    timesteps = [
        torch.linspace(1, generation_config.eps, spb + 1, device=x.device)
        for spb in steps_per_block
    ]

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        attention_mask = torch.where(attention_mask, torch.tensor(0.0, device=attention_mask.device), torch.tensor(float("-inf"), device=attention_mask.device))
    else:
        tok_idx = None
        attention_mask = "full"

    past_key_values = None

    for num_block in range(num_blocks):
        current_block_start = input_ids.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        if cgws is not None:
            window_end = max_length if cgws is None else min(current_block_end + cgws, max_length)
            window_slice = slice(current_block_start, window_end)

        if use_cache:
            model_output = model(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :current_block_start, :],)
            past_key_values = new_past_key_values
        else:
            model_output = model(x, attention_mask, tok_idx, use_cache=False)

        logits = model_output.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        _, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        x[:, current_block_start] = x0[:, current_block_start]
        if histories is not None:
            histories.append(x.clone().cpu())

        spb = steps_per_block[num_block]
        i = 1
        while True:
            if cgws is not None:
                mask_index = (x[:, window_slice] == mask_token_id)
            else:
                mask_index = (x[:, current_block_start:] == mask_token_id)

            if attention_mask != "full":
                if cgws is not None:
                    current_attention_mask = attention_mask[:, :, window_slice, :window_end]
                else:
                    current_attention_mask = attention_mask[:, :, current_block_start:, :]
            else:
                current_attention_mask = attention_mask

            if use_cache:
                if cgws is not None:
                    model_output = model(x[:, window_slice], current_attention_mask,
                                    tok_idx[:, window_slice] if tok_idx is not None else None,
                                    past_key_values=past_key_values, use_cache=True)
                else:
                    model_output = model(x[:, current_block_start:], current_attention_mask,
                                    tok_idx[:, current_block_start:] if tok_idx is not None else None,
                                    past_key_values=past_key_values, use_cache=True)
                logits = model_output.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            else:
                model_output = model(x, attention_mask, tok_idx, use_cache=False)
                logits = model_output.logits
                logits = logits[:, current_block_start:]
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break

            mask_index[:, block_length:] = False
            mask_logits = logits[mask_index]
            target, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, tar=tar)

            _pad_target_divisor = pad_target_penalty
            _pad_mask_flat = (x0 == pad_token_id)
            if _pad_mask_flat.any():
                target = target.clone()
                target[_pad_mask_flat] = target[_pad_mask_flat] / _pad_target_divisor

            if cgws is not None:
                full_target = torch.full_like(x[:, window_slice], -torch.inf, device=model.device, dtype=logits.dtype)
            else:
                full_target = torch.full_like(x[:, current_block_start:], -torch.inf, device=model.device, dtype=logits.dtype)
            full_target = full_target.float()
            full_target[mask_index] = target
            full_target[:, block_length:] = -torch.inf

            if unmask_threshold is None:
                if i >= spb - 1:
                    if cgws is not None:
                        xwin = x[:, window_slice]
                    else:
                        xwin = x[:, current_block_start:]
                    x_candidates = torch.full_like(xwin, mask_token_id, dtype=torch.long)
                    x_candidates[mask_index] = x0
                    xwin[mask_index] = x_candidates[mask_index]
                else:
                    num_mask_token = mask_index.sum() / mask_index.shape[0]
                    t = timesteps[num_block][i]
                    s = timesteps[num_block][i + 1]
                    number_transfer_tokens = int(num_mask_token * (1 - s / t))

                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_target, number_transfer_tokens)
                        else:
                            full_target = full_target / alg_temp
                            full_target = F.softmax(full_target, dim=-1)
                            transfer_index = torch.multinomial(full_target, num_samples=number_transfer_tokens)

                        if cgws is not None:
                            x_ = torch.zeros_like(x[:, window_slice], device=model.device, dtype=torch.long) + mask_token_id
                        else:
                            x_ = torch.zeros_like(x[:, current_block_start:], device=model.device, dtype=torch.long) + mask_token_id
                        x_[mask_index] = x0.clone()
                        row_indices = torch.arange(x.size(0), device=model.device).unsqueeze(1).expand_as(transfer_index)

                        if cgws is not None:
                            x[:, window_slice][row_indices, transfer_index] = x_[row_indices, transfer_index]
                        else:
                            x[:, current_block_start:][row_indices, transfer_index] = x_[row_indices, transfer_index]
            else:
                if cgws is not None:
                    xwin = x[:, window_slice]
                else:
                    xwin = x[:, current_block_start:]

                selected_map = torch.zeros_like(xwin, dtype=torch.bool)
                selected_map[mask_index] = (target >= unmask_threshold)
                no_sel = ~selected_map.any(dim=-1)
                no_sel = no_sel & mask_index.any(dim=-1)

                if no_sel.any():
                    masked_scores = full_target.masked_fill(~mask_index, float("-inf"))
                    best_idx = torch.argmax(masked_scores, dim=-1)
                    selected_rows = torch.nonzero(no_sel, as_tuple=False).squeeze(-1)
                    selected_map[selected_rows, best_idx[selected_rows]] = True

                selected_map &= mask_index
                x_candidates = torch.full_like(xwin, mask_token_id, dtype=torch.long)
                x_candidates[mask_index] = x0

                xwin[selected_map] = x_candidates[selected_map]

            if histories is not None:
                histories.append(x.clone().cpu())

            i += 1
            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break

        block_all_pad = torch.all(x[:, current_block_start:current_block_end] == pad_token_id)
        if block_all_pad:
            if current_block_end < x.size(1):
                x[:, current_block_end:] = pad_token_id
            if histories is not None:
                histories.append(x.clone().cpu())
            break

    if return_dict_in_generate:
        return DreamModelOutput(sequences=x, history=histories)
    else:
        return x


def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return "We can not extract the code in the output. "


def denoise_step_map(history, mask_id: int, sample_idx: int = 0):
    L = history[0].shape[1]
    step_map = torch.zeros(L, dtype=torch.long)
    prev = torch.full((L,), mask_id, dtype=torch.long)
    for t, snap in enumerate(history, start=0):
        cur = snap[sample_idx]
        changed = (prev == mask_id) & (cur != mask_id)
        step_map[changed] = t
        prev = cur
        if (step_map == 0).sum() == 0:
            break
    return step_map


# ---------------------------------------------------------------------------
# LiveCodeBench task loading  (replaces build_evalplus_data / get_prompt)
# ---------------------------------------------------------------------------
#
# LiveCodeBench problems come in two flavors:
#   - "call_based":  starter_code is non-empty (LeetCode-style). The model
#     must return a full class/function definition matching the signature.
#   - "standard_input": no starter_code. The model must write a standalone
#     script that reads from stdin and prints to stdout.
# Both need to be told explicitly which mode they're in, since that changes
# what shape of code the checker (lcb_runner testing_util.py) expects.

SYSTEM_PROMPT_CALL_BASED = (
    "<|im_start|>system\nYou are an expert Python programmer. You will be given "
    "a question (problem specification) and will generate a correct Python "
    "program that matches the specification and passes all tests.<|im_end|>\n"
    "<|im_start|>user\n{{problem}}\n\n"
    "You will use the following starter code to write the solution and enclose "
    "your full solution (including the starter code) within a single Python "
    "code block:\n```python\n{{starter_code}}\n```\n"
    "<|im_end|>\n<|im_start|>assistant\n```python\n"
)

SYSTEM_PROMPT_STDIN = (
    "<|im_start|>system\nYou are an expert Python programmer. You will be given "
    "a question (problem specification) and will generate a correct Python "
    "program that matches the specification and passes all tests.<|im_end|>\n"
    "<|im_start|>user\n{{problem}}\n\n"
    "Read the inputs from stdin and print the output to stdout. Enclose your "
    "full solution within a single Python code block.\n"
    "<|im_end|>\n<|im_start|>assistant\n```python\n"
)


def build_livecodebench_data(
    release_version: str = "release_v6",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list:
    problems: list[CodeGenerationProblem] = load_code_generation_dataset(
        release_version=release_version, start_date=start_date, end_date=end_date
    )

    data = []
    for p in problems:
        is_call_based = bool(p.starter_code and p.starter_code.strip())
        data.append({
            "question_id": p.question_id,
            "question_content": p.question_content,
            "starter_code": p.starter_code or "",
            "is_call_based": is_call_based,
            # keep the actual problem object around so we can build the
            # input_output eval sample later without reloading the dataset
            "_problem": p,
        })
    return data


def get_prompt(item: dict) -> str:
    if item["is_call_based"]:
        template = SYSTEM_PROMPT_CALL_BASED
    else:
        template = SYSTEM_PROMPT_STDIN
    return Template(template).render(
        problem=item["question_content"],
        starter_code=item["starter_code"],
    )


# ---------------------------------------------------------------------------
# Multi-GPU worker -- UNCHANGED sampling logic, fixed rollout defaults
# ---------------------------------------------------------------------------

def run_batch(model_gpu, tokenizer_gpu, args, pad_id, mask_id, eos_id, batch_prompts):
    """Runs one batch through _sample and returns decoded texts + step maps."""
    enc = tokenizer_gpu(batch_prompts, padding=True, return_tensors="pt", padding_side="left")
    prompt_ids = enc["input_ids"].to(model_gpu.device)
    attn_mask = prompt_ids.ne(pad_id).to(device=model_gpu.device)

    further_horizon = None if not args.use_cache else args.further_horizon

    generation_config = DreamGenerationConfig(
        output_history=True,
        return_dict_in_generate=True,
        max_length=args.max_gen_length + prompt_ids.shape[1],
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tar=args.target,
        alg_temp=args.alg_temp,
    )

    unmask_threshold = None if args.remasking_strategy == "low_confidence_static" else args.dynamic_threshold

    generation_ids = _sample(
        model_gpu,
        prompt_ids,
        attention_mask=attn_mask,
        generation_config=generation_config,
        block_length=args.block_size,
        use_cache=args.use_cache,
        further_horizon=further_horizon,
        mask_token_id=mask_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        pad_target_penalty=args.pad_target_penalty,
        unmask_threshold=unmask_threshold,
    )
    generation_ids.sequences = generation_ids.sequences.cpu()

    seq_ids = generation_ids.sequences[:, prompt_ids.shape[1]:].tolist()
    texts = tokenizer_gpu.batch_decode(seq_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)

    step_maps = [
        denoise_step_map(generation_ids.history, mask_id=mask_id, sample_idx=i)[prompt_ids.shape[1]:].tolist()
        for i in range(len(batch_prompts))
    ]

    del generation_ids
    torch.cuda.empty_cache()
    return texts, step_maps


def process_chunk_with_oom_backoff(model_gpu, tokenizer_gpu, args, pad_id, mask_id, eos_id,
                                    prompts_chunk, idx_chunk, seq_dict, step_dict, rank):
    """Tries to run a chunk at its current size. On CUDA OOM, halves the
    chunk and retries each half recursively, so throughput adapts to
    whatever actually fits instead of relying on a manually guessed
    --batch-size."""
    try:
        texts, step_maps = run_batch(model_gpu, tokenizer_gpu, args, pad_id, mask_id, eos_id, prompts_chunk)
        for i, idx in enumerate(idx_chunk):
            seq_dict[idx] = texts[i]
            step_dict[idx] = step_maps[i]
        return len(prompts_chunk)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts_chunk) == 1:
            print(f"[GPU {rank}] OOM on a single-item batch (idx={idx_chunk[0]}); skipping it.")
            seq_dict[idx_chunk[0]] = ""
            step_dict[idx_chunk[0]] = []
            return 1

        mid = len(prompts_chunk) // 2
        print(f"[GPU {rank}] OOM at batch size {len(prompts_chunk)}; splitting into {mid} + {len(prompts_chunk) - mid}")
        size_a = process_chunk_with_oom_backoff(
            model_gpu, tokenizer_gpu, args, pad_id, mask_id, eos_id,
            prompts_chunk[:mid], idx_chunk[:mid], seq_dict, step_dict, rank,
        )
        size_b = process_chunk_with_oom_backoff(
            model_gpu, tokenizer_gpu, args, pad_id, mask_id, eos_id,
            prompts_chunk[mid:], idx_chunk[mid:], seq_dict, step_dict, rank,
        )
        return min(size_a, size_b)


def worker(pretrained_model, rank, prompts, orig_idx, seq_dict, step_dict, args):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model_gpu = (
        DreamModel.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    model_gpu.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model_gpu)
    model_gpu._sample = types.MethodType(DreamGenerationMixin._sample, model_gpu)
    tokenizer_gpu = DreamTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    pad_id = model_gpu.config.pad_token_id
    mask_id = model_gpu.config.mask_token_id
    eos_id = tokenizer_gpu.convert_tokens_to_ids("<|im_end|>")

    current_batch_size = args.batch_size
    start = 0
    pbar = tqdm(total=len(prompts), desc=f"GPU {rank}", position=rank, leave=True)

    while start < len(prompts):
        end = min(start + current_batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        batch_idxs = orig_idx[start:end]

        worked_size = process_chunk_with_oom_backoff(
            model_gpu, tokenizer_gpu, args, pad_id, mask_id, eos_id,
            batch_prompts, batch_idxs, seq_dict, step_dict, rank,
        )

        if worked_size < current_batch_size:
            current_batch_size = max(1, worked_size)
        elif args.grow_batch_size and current_batch_size < args.max_batch_size:
            current_batch_size = min(args.max_batch_size, current_batch_size + max(1, current_batch_size // 4))

        pbar.update(end - start)
        start = end

    pbar.close()


def split_even(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model path/name, e.g. apple/DiffuCoder-7B-Instruct")

    # LiveCodeBench dataset selection (replaces --dataset humaneval/mbpp)
    parser.add_argument("--release-version", default="release_v6",
                         help="LiveCodeBench release_v1..release_v6 (see lcb_runner README)")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD, filters problems by contest date")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD, filters problems by contest date")

    parser.add_argument("--out", default=None, help="Output code_list JSON path")
    parser.add_argument("--results-dir", default="lcb_results", help="Directory for generated outputs")
    parser.add_argument("--n-samples", type=int, default=1, help="Samples per task (for pass@k, k>1)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N tasks (smoke testing)")
    parser.add_argument("--evaluate", action="store_true",
                         help="Score immediately in-process via lcb_runner.evaluation.codegen_metrics")
    parser.add_argument("--num-process-evaluate", type=int, default=16)
    parser.add_argument("--eval-timeout", type=int, default=6, help="Per-test-case timeout (seconds)")

    # Rollout hyperparameters -- LiveCodeBench solutions run much longer than
    # HumanEval completions, so raise max-gen-length/steps accordingly.
    # (See "Impact of Generation Length / Diffusion Steps" ablations for
    # code-diffusion models -- competitive-programming solutions typically
    # need several hundred tokens, not ~256.)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--pad-target-penalty", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--max-gen-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8,
                         help="Starting batch size. LiveCodeBench prompts are longer than "
                              "HumanEval/MBPP, so start lower than the EvalPlus default.")
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--grow-batch-size", dest="grow_batch_size", action="store_true", default=True)
    parser.add_argument("--no-grow-batch-size", dest="grow_batch_size", action="store_false")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--remasking-strategy", default="low_confidence_static")
    parser.add_argument("--dynamic-threshold", type=float, default=0.9)
    parser.add_argument("--target", default="confidence")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--further-horizon", type=int, default=128)
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", default=True)
    parser.add_argument("--no-use-cache", dest="use_cache", action="store_false")

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    if args.out is None:
        args.out = os.path.join(
            args.results_dir,
            f"lcb_{args.release_version}_{args.model.replace('/', '.')}_codelist.json",
        )
    else:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    mp.set_start_method("spawn", force=True)

    data = build_livecodebench_data(args.release_version, args.start_date, args.end_date)
    if args.limit is not None:
        data = data[: args.limit]
    num = len(data)
    cprint(f"loaded {num} LiveCodeBench problems ({args.release_version})", "cyan")

    generation_prompts, prefix_list, index_list = [], [], []
    for i in range(num):
        generation_prompts += [get_prompt(data[i])] * args.n_samples
        # the prompt already ends in "```python\n" (assistant turn opener),
        # so that's the "prefix" we re-attach before regexing for the block,
        # same pattern as the EvalPlus script's prefix_list.
        prefix_list += ["```python\n"] * args.n_samples
        index_list += [i] * args.n_samples
        data[i]["extracted_output"] = []

    cprint("start generation...", "green")

    N = len(generation_prompts)
    shuffled_idx = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [generation_prompts[i] for i in shuffled_idx]

    n_gpu = torch.cuda.device_count()
    prompt_chunks = split_even(shuffled_prompts, n_gpu)
    idx_chunks = split_even(shuffled_idx, n_gpu)

    manager = mp.Manager()
    seq_dict = manager.dict()
    step_dict = manager.dict()
    procs = []

    for rk in range(n_gpu):
        p = mp.Process(
            target=worker,
            args=(args.model, rk, prompt_chunks[rk], idx_chunks[rk], seq_dict, step_dict, args),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    restored_outputs = [seq_dict[i] for i in range(N)]

    cprint("generation job done!", "green")

    for i, full_output in enumerate(restored_outputs):
        extracted_output = extract_code(prefix_list[i] + full_output)
        index_i = index_list[i]
        data[index_i]["extracted_output"].append(extracted_output)

    # --- write LiveCodeBench custom_evaluator-compatible output ---------
    code_list_payload = [
        {"question_id": d["question_id"], "code_list": d["extracted_output"]}
        for d in data
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(code_list_payload, f, indent=2)

    cprint(
        f"wrote {sum(len(d['extracted_output']) for d in data)} samples "
        f"across {len(data)} problems to {args.out}",
        "green",
    )

    if args.evaluate:
        cprint("scoring in-process via lcb_runner.evaluation.codegen_metrics...", "yellow")
        from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

        samples_list = [d["_problem"].get_evaluation_sample() for d in data]
        generations_list = [d["extracted_output"] for d in data]

        metrics, results, final_metadata = codegen_metrics(
            samples_list,
            generations_list,
            k_list=[1] if args.n_samples == 1 else [1, min(5, args.n_samples)],
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.eval_timeout,
        )
        cprint(json.dumps({k: v for k, v in metrics.items() if k != "detail"}, indent=2), "magenta")

        metrics_out = args.out.rsplit(".json", 1)[0] + "-metrics.json"
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in metrics.items() if k != "detail"}, f, indent=2)
        cprint(f"wrote metrics to {metrics_out}", "green")
    else:
        cprint(
            f"next: python -m lcb_runner.runner.custom_evaluator --custom_output_file {args.out}",
            "yellow",
        )


if __name__ == "__main__":
    main()