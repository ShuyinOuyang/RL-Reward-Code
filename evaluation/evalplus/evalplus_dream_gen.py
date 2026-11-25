# Adapted from Gen-Verse/dLLM-RL (sample/dream_sample.py), https://github.com/Gen-Verse/dLLM-RL,
# licensed under the Apache License, Version 2.0 (see ../LICENSE-APACHE-2.0). Modified for this project.
"""
Generate EvalPlus (HumanEval+ / MBPP+) completions using the Dream-model
multi-GPU sampling pipeline (_sample / DreamGenerationConfig), and export a
samples.jsonl ready for evalplus.sanitize / evalplus.evaluate.

Simplified on purpose: the only two things you choose per run are the
model and the benchmark. All rollout/sampling hyperparameters use fixed,
sensible defaults (editable via CLI flags if you ever need to), and all
the config-file/multi-node/dataset-type plumbing from the original
dream_sample.py has been removed since this script is evalplus-only.

Usage:
    python evalplus_dream_gen.py --model apple/DiffuCoder-7B-Instruct --dataset humaneval
    python evalplus_dream_gen.py --model apple/DiffuCoder-7B-Instruct --dataset mbpp --out mbpp_samples.jsonl

Then:
    evalplus.sanitize --samples <out>
    evalplus.evaluate --dataset humaneval --samples <out-sanitized>
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

from evalplus.data import get_human_eval_plus, get_mbpp_plus


# ---------------------------------------------------------------------------
# Sampling internals -- unchanged from the original dream_sample.py
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

            # if unmask_threshold is None:
            #     num_mask_token = mask_index.sum() / mask_index.shape[0]
            #     t = timesteps[num_block][i]
            #     s = timesteps[num_block][i + 1]
            #     number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < spb - 1 else int(num_mask_token)

            #     if number_transfer_tokens > 0:
            #         if alg_temp is None or alg_temp == 0:
            #             _, transfer_index = torch.topk(full_target, number_transfer_tokens)
            #         else:
            #             full_target = full_target / alg_temp
            #             full_target = F.softmax(full_target, dim=-1)
            #             transfer_index = torch.multinomial(full_target, num_samples=number_transfer_tokens)

            #         if cgws is not None:
            #             x_ = torch.zeros_like(x[:, window_slice], device=model.device, dtype=torch.long) + mask_token_id
            #         else:
            #             x_ = torch.zeros_like(x[:, current_block_start:], device=model.device, dtype=torch.long) + mask_token_id
            #         x_[mask_index] = x0.clone()
            #         row_indices = torch.arange(x.size(0), device=model.device).unsqueeze(1).expand_as(transfer_index)

            #         if cgws is not None:
            #             x[:, window_slice][row_indices, transfer_index] = x_[row_indices, transfer_index]
            #         else:
            #             x[:, current_block_start:][row_indices, transfer_index] = x_[row_indices, transfer_index]
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
# EvalPlus task loading
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_FUNCTION = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{{problem}}\n"
    "Place your code within a single Python code block ```python ```. "
    "Do not include more than one code block. <|im_end|>\n"
    "<|im_start|>assistant\n"
)


def normalize_mbpp_tests(raw) -> list:
    """MBPP+'s test_list field is a joined string, not a list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [line.strip() for line in raw.strip().split("\n") if line.strip()]
    return [str(x).strip() for x in raw if str(x).strip()]


def build_evalplus_data(dataset: str) -> list:
    data = []

    if dataset == "humaneval":
        problems = get_human_eval_plus()
        for task_id, p in problems.items():
            prompt = p["prompt"]
            data.append({
                "task_id": task_id,
                "question": f"You need to complete this code:\n{prompt}",
                "prefix": f"Here is the completed code:\n```python\n{prompt}",
            })

    elif dataset == "mbpp":
        problems = get_mbpp_plus()
        for task_id, p in problems.items():
            description = p["prompt"].strip()
            test_list = normalize_mbpp_tests(p.get("test_list") or p.get("assertion"))
            first_test = test_list[0] if test_list else ""
            data.append({
                "task_id": task_id,
                "question": f"{description}\nYour code should pass the test:\n{first_test}\n",
                "prefix": "Here is the completed code:\n```python",
            })

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return data


def get_prompt(item: dict) -> str:
    template = SYSTEM_PROMPT_FUNCTION + item["prefix"]
    return Template(template).render(problem=item["question"])


# ---------------------------------------------------------------------------
# Multi-GPU worker -- unchanged sampling logic, fixed rollout defaults
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
        return len(prompts_chunk)  # size that worked, for tracking the current safe batch size

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts_chunk) == 1:
            # Can't split further -- this single item genuinely doesn't fit
            # (e.g. an unusually long prompt). Record the failure and move on
            # rather than crashing the whole run.
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

    # current_batch_size adapts over time: starts at args.batch_size (which
    # should be set aggressively -- see --batch-size help), only shrinks on
    # OOM, and is allowed to creep back up periodically so a transient large
    # chunk doesn't permanently cap throughput for the rest of the run.
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

        # Back off to whatever size actually succeeded, then let it grow
        # again slowly so we keep probing for the true ceiling.
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
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--out", default=None, help="Output samples.jsonl path (default: evalplus_results/<dataset>_<model>_samples.jsonl)")
    parser.add_argument("--results-dir", default="evalplus_results", help="Directory for generated outputs")
    parser.add_argument("--n-samples", type=int, default=1, help="Samples per task (for pass@k, k>1)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N tasks (smoke testing)")

    # Rollout hyperparameters -- fixed sensible defaults, overridable if needed
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--pad-target-penalty", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--max-gen-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Starting batch size. On a large-memory GPU, "
                              "set this aggressively -- it will auto-shrink on OOM and "
                              "(if --grow-batch-size) creep back up, so it's safe to start high.")
    parser.add_argument("--max-batch-size", type=int, default=256,
                         help="Ceiling for automatic batch-size growth")
    parser.add_argument("--grow-batch-size", dest="grow_batch_size", action="store_true", default=True,
                         help="Allow batch size to grow back up after a successful batch (default: on)")
    parser.add_argument("--no-grow-batch-size", dest="grow_batch_size", action="store_false")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--remasking-strategy", default="low_confidence_static")
    parser.add_argument("--target", default="confidence")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--further-horizon", type=int, default=128)
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", default=True)
    parser.add_argument("--no-use-cache", dest="use_cache", action="store_false")

    args = parser.parse_args()

    dataset_dir = os.path.join(args.results_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    if args.out is None:
        args.out = os.path.join(dataset_dir, f"{args.dataset}_{args.model.replace('/', '.')}_samples.jsonl")
    else:
        # Even with an explicit --out, make sure its parent directory exists.
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    mp.set_start_method("spawn", force=True)

    data = build_evalplus_data(args.dataset)
    if args.limit is not None:
        data = data[: args.limit]
    num = len(data)

    generation_prompts, prefix_list, index_list = [], [], []
    for i in range(num):
        generation_prompts += [get_prompt(data[i])] * args.n_samples
        prefix_list += [data[i]["prefix"]] * args.n_samples
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

    with open(args.out, "w", encoding="utf-8") as f:
        for item in data:
            task_id = item["task_id"]
            for solution in item["extracted_output"]:
                f.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")

    cprint(f"wrote {sum(len(d['extracted_output']) for d in data)} samples to {args.out}", "green")
    cprint(
        f"next: evalplus.sanitize --samples {args.out}  &&  "
        f"evalplus.evaluate --dataset {args.dataset} --samples "
        f"{args.out.rsplit('.jsonl', 1)[0]}-sanitized.jsonl",
        "yellow",
    )


if __name__ == "__main__":
    main()