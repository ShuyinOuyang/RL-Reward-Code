# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
import subprocess
import sys
import ast
import os
import tempfile
from collections import Counter
from functools import partial, update_wrapper
from typing import Callable, Dict, Literal, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils.code_providers import get_provider
from .utils.competitive_programming import (
    SubtaskResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
)
from .utils.competitive_programming import patch_code as cf_patch_code
from .utils.competitive_programming import score_submission as cf_score_submission
from .utils.competitive_programming import score_subtask

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
from .utils.code_providers import get_provider
from .utils.ioi import (
    SubtaskResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
    score_subtask,
)

def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://huggingface.co/papers/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    """Initialize or get the current event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using a specified execution client.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    Args:
        completions: List of model completions to evaluate
        test_batch_size: Evaluate these many test cases in parallel, then check if any of them failed (0 score):
                       if so stop evaluating; otherwise continue with the next batch of test cases.
        provider_type: The execution provider to use (default: "piston"). Supported values: "piston", "morph"
        **kwargs: Additional arguments passed from the dataset
    """
    # Get the appropriate client based on provider_type
    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        # for info on setting up piston workers, see slurm/piston/README.md
        execution_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from {provider_type} worker: {e}")
            return SubtaskResult()

    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                score_subtask(
                    execution_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def cf_code_reward(
    completions,
    test_batch_size: int = 1,
    patch_code: bool = False,
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
    **kwargs,
) -> list[float]:
    """Reward function that evaluates Codeforces problems using Piston+our CF package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/codeforces (verifiable-prompts subset)

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    languages = kwargs["language"] if "language" in kwargs else [None] * len(completions)
    code_snippets = [
        # note: grading is automatically skipped if a problem has no tests
        cf_patch_code(extract_code(completion[-1]["content"], language), language)
        if patch_code
        else extract_code(completion[-1]["content"], language)
        for completion, language in zip(completions, languages)
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return None

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                cf_score_submission(
                    piston_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                    scoring_mode=scoring_mode,
                    submission_language=problem_data.get("language", None),
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return results


def extract_code(completion: str, language: str | None = "python") -> str:
    if language is None:
        return ""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def _run_acecoder_tests(code: str, test_cases: list[str], timeout: float = 10.0) -> float:
    """Run assert-based test cases against code in a subprocess, return pass rate.

    Follows AceCoder's eval_test_cases.py: each assert is executed independently
    so a runtime error in one test does not fail the rest.
    """
    if not test_cases:
        return 0.0

    lines = [code, "", "_passed = 0"]
    for test in test_cases:
        lines += ["try:", f"    {test}", "    _passed += 1", "except Exception:", "    pass"]
    lines.append("print(_passed)")

    try:
        result = subprocess.run(
            [sys.executable, "-c", "\n".join(lines)],
            capture_output=True, text=True, timeout=timeout,
        )
        stdout = result.stdout.strip()
        if not stdout:
            return 0.0
        return int(stdout.split("\n")[-1]) / len(test_cases)
    except (subprocess.TimeoutExpired, ValueError):
        return 0.0


def acecoder_reward(completions, test_cases: list[list[str]], **kwargs) -> list[float]:
    """Pass-rate reward for AceCoder-style data.

    Follows the AceCoder rule-based reward: pass_rate = passed_tests / total_tests.
    Requires a `test_cases` dataset column containing a list of Python assert strings.
    Accepts both raw-code completions and markdown-fenced completions.
    """
    rewards = []
    for completion, tests in zip(completions, test_cases):
        content = completion[-1]["content"]
        code = extract_code(content, language="python") or content.strip()
        rewards.append(_run_acecoder_tests(code, list(tests)))
    return rewards


def binary_code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
        **kwargs,
    )
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """

    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]

    template = evaluation_script_template

    scripts = [
        template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if enforce_same_language:
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("All verification_info must have the same language", verification_info)

    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )

    return execution_provider.execute_scripts(scripts, ["python"] * len(scripts))


def get_code_format_reward(language: str = "python", use_loose_format: bool = False):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """

    def code_format_reward(completions, **kwargs):
        # if there is a language field, use it instead of the default language. This way we can have mixed language training.
        # print(completions)
        languages = kwargs["language"] if "language" in kwargs else [language] * len(completions)
        # print(languages)
        completion_contents = [completion[0]["content"] for completion in completions]
        # print(completion_contents)
        if use_loose_format:
            matches = [
                re.match(
                    rf"<think>\n.*?\n</think>\n*<answer>\n.*?```{sample_language}.*?```.*?\n</answer>",
                    content,
                    re.DOTALL | re.MULTILINE,
                )
                for content, sample_language in zip(completion_contents, languages)
            ]
        else:
            matches = [
                re.match(
                    rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{sample_language}.*?```.*?\n</answer>$",
                    content,
                    re.DOTALL | re.MULTILINE,
                )
                for content, sample_language in zip(completion_contents, languages)
            ]
        
        reward_list = [1.0 if match else 0.0 for match in matches]
        # print(matches)
        print('-------------------code fortamt correctness--------------')
        print('use loose format', use_loose_format)
        print(reward_list)
        return reward_list

    return code_format_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    Reward function that penalizes overlong completions. It is used to penalize overlong completions,
    but not to reward shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)

    Args:
        max_completion_len: Maximum length of the completion
        soft_punish_cache: Minimum length of the completion. If set to 0, no minimum length is applied.
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Reward function that penalizes overlong completions."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward

# TODO: new reward functions
#   1. code_format
#   2. code_syntax
#   3. static analysis (Pylint)
#   4. Text & AST similarity with the inference code example (no execution needed)
#   5. code_semantic (test case pass rate)



# code format reward
def code_format_correctness(completions, language='python', **kwargs):
    pattern = re.compile(
        rf"^"
        r"(?:(?!```)[\s\S])*?"
        rf"```{language}\n"    # match ```language\n
        r"(?:(?!```)[\s\S])*?"         # match any character, but not ```
        rf"```\n$",                     # match ``` and end of string
        re.DOTALL
    )
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content in completion_contents:
        # # First check if the format matches
        # import pdb; pdb.set_trace();
        format_match = pattern.fullmatch(content)
        if not format_match:
            rewards.append(0.0)
            continue

        # Extract code from between code blocks
        code_blocks = re.findall(rf"```{language}\n(.*?)```", content, re.DOTALL)
        if not code_blocks:
            rewards.append(0.0)
            # continue
        else:
            rewards.append(1.0)

    return rewards

# code syntax reward
def code_syntax_correctness(completions, language='python', **kwargs):
    import ast
    use_loose_format = kwargs.get('use_loose_format', False)
    format_rewards = get_code_format_reward(language='python', use_loose_format=use_loose_format)(completions)
    rewards = []
    for i, (reward, completion) in enumerate(zip(format_rewards, completions)):
        # ##made this up##
        # rewards.append(1.0)
        # continue
        # ################
        if reward < 1:
            rewards.append(0)
            continue
        code = extract_code(completion[-1]["content"])
        try:
            ast.parse(code)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
        rewards.append(1.0 if syntax_valid else 0)
    print('-------------------syntax correctness--------------')
    print(rewards)
    print('use_loose_format', use_loose_format)
    return rewards
    # pattern = re.compile(
    #     rf"^"
    #     r"(?:(?!```)[\s\S])*?"
    #     rf"```{language}\n"    # match ```language\n
    #     r"(?:(?!```)[\s\S])*?"         # match any character, but not ```
    #     rf"```\n$",                     # match ``` and end of string
    #     re.DOTALL
    # )
    # completion_contents = [completion[0]["content"] for completion in completions]
    # rewards = []

    # for content in completion_contents:
    #     # # First check if the format matches
    #     # import pdb; pdb.set_trace();
    #     format_match = pattern.fullmatch(content)
    #     if not format_match:
    #         rewards.append(0.0)
    #         continue

    #     # Extract code from between code blocks
    #     code_blocks = re.findall(rf"```{language}\n(.*?)```", content, re.DOTALL)
    #     if not code_blocks:
    #         rewards.append(0.0)
    #         continue

    #     # Get the first code block (in case there are multiple)
    #     code = code_blocks[0].strip()

    #     # Check syntax if it's Python code
    #     try:
    #         ast.parse(code)
    #         syntax_valid = True
    #     except SyntaxError:
    #         syntax_valid = False
    #     rewards.append(1.0 if syntax_valid else 0)





async def run_script_pylint(code, semaphore, only_semantic_flag=False):
    PYLINT_SCORE_RE = re.compile(r"rated at ([0-9.]+)/10")
    if code == '':
        return {
            "score": 0.0,
            "stdout": '',
            "stderr": '',
            "returncode": '',
        }
    async with semaphore:
        # 1. Write code to a temporary file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
        tmp_path = tmp_file.name
        try:
            tmp_file.write(code.encode("utf-8"))
            tmp_file.flush()
        finally:
            tmp_file.close()

        try:
            # 2. Run pylint on the temporary file
            if only_semantic_flag:
                process = await asyncio.create_subprocess_exec(
                    "pylint",
                    "--disable=all",
                    "--enable=W, E, F",
                    tmp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    "pylint",
                    tmp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=5
                )
            except asyncio.TimeoutError:
                # Kill the process if it hangs
                process.kill()
                return {
                    "score": None,
                    "stdout": "",
                    "stderr": "Process timed out",
                    "returncode": -1,
                }

            returncode = process.returncode
            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")
            combined_output = stdout_text + "\n" + stderr_text

            # 3. Parse score from pylint output
            match = PYLINT_SCORE_RE.search(combined_output)
            score: Optional[float] = float(match.group(1)) if match else None

            return {
                "score": score,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "returncode": returncode,
            }

        finally:
            # 4. Clean up the temporary file
            try:
                os.remove(tmp_path)
            except OSError:
                pass

async def pylint(code_list, max_concurrent_copies=128):
    semaphore = asyncio.Semaphore(max_concurrent_copies)
    tasks = [run_script_pylint(code, semaphore) for i, code in enumerate(code_list)]
    results = await asyncio.gather(*tasks)
    return results

# code static check reward
def code_static_checking_reward(completions, **kwargs):
    use_loose_format = kwargs.get('use_loose_format', False)
    format_rewards = code_syntax_correctness(completions, use_loose_format=use_loose_format)
    code_list = []
    for i, (reward, completion) in enumerate(zip(format_rewards, completions)):
        if reward < 1:
            code_list.append('')
            continue
        code = extract_code(completion[-1]["content"])

        code_list.append(code)
    pylint_res_list = asyncio.run(pylint(code_list))
    rewards = []
    for x in pylint_res_list:
        if x['score'] is not None:
            rewards.append(x["score"]/10)
        else:
            rewards.append(0.0)
    return rewards

async def pylint_semantic(code_list, max_concurrent_copies=128):
    semaphore = asyncio.Semaphore(max_concurrent_copies)
    tasks = [run_script_pylint(code, semaphore, True) for i, code in enumerate(code_list)]
    results = await asyncio.gather(*tasks)
    return results

def code_static_semantic_checking_reward(completions, **kwargs):
    use_loose_format = kwargs.get('use_loose_format', False)
    syntax_rewards = code_syntax_correctness(completions, use_loose_format=use_loose_format)
    code_list = []
    for i, (reward, completion) in enumerate(zip(syntax_rewards, completions)):
        if reward < 1:
            code_list.append('')
            continue
        code = extract_code(completion[-1]["content"])

        code_list.append(code)
    pylint_res_list = asyncio.run(pylint_semantic(code_list))
    rewards = []
    for x in pylint_res_list:
        if x['score'] is not None:
            rewards.append(x["score"]/10)
        else:
            rewards.append(0.0)
    return rewards


def simple_code_tokenizer(code):
    # Very simple tokenizer: split on non-word chars
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\S", code)

class ASTFeatureExtractor(ast.NodeVisitor):
    def __init__(self):
        self.node_types = Counter()
        self.parent_child = Counter()
        self.max_depth = 0

    def generic_visit(self, node, depth=0, parent_type=None):
        node_type = type(node).__name__
        self.node_types[node_type] += 1
        if parent_type is not None:
            self.parent_child[(parent_type, node_type)] += 1
        self.max_depth = max(self.max_depth, depth)

        for child in ast.iter_child_nodes(node):
            self.generic_visit(child, depth + 1, node_type)

    def extract(self, code):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # If parsing fails, return an almost-zero feature dict
            return {"parse_failed": 1.0}

        self.node_types.clear()
        self.parent_child.clear()
        self.max_depth = 0

        self.generic_visit(tree, depth=0, parent_type=None)

        features = {}
        # Node type counts
        for k, v in self.node_types.items():
            features[f"node_{k}"] = float(v)
        # Parent-child patterns
        for (p, c), v in self.parent_child.items():
            features[f"edge_{p}->{c}"] = float(v)
        # Depth
        features["max_depth"] = float(self.max_depth)

        return features

def syntax_similarity(candidate_code, golden_codes):
    corpus = [candidate_code] + golden_codes
    docs = [" ".join(simple_code_tokenizer(c)) for c in corpus]
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # char or token n-grams both work
    X = vectorizer.fit_transform(docs)
    cand_vec = X[0]
    golden_vecs = X[1:]

    sims = cosine_similarity(cand_vec, golden_vecs)[0]  # shape (5,)
    return sims.tolist()

def ast_similarity(candidate_code, golden_codes):
    extractor = ASTFeatureExtractor()
    feature_dicts = [extractor.extract(candidate_code)] + [
        extractor.extract(g) for g in golden_codes
    ]

    # Convert dicts to aligned vectors
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(feature_dicts)

    cand_vec = X[0]
    golden_vecs = X[1:]

    sims = cosine_similarity(cand_vec, golden_vecs)[0]
    return sims.tolist()

def code_similarity_checking_reward(completions, alpha=0.3, **kwargs):
    # inferences
    inferences = kwargs.get("inferences", [])
    use_loose_format = kwargs.get('use_loose_format', False)
    syntax_rewards = code_syntax_correctness(completions, use_loose_format=use_loose_format)
    code_list = []
    inference_list = inferences
    rewards = []
    for i, (reward, completion) in enumerate(zip(syntax_rewards, completions)):
        if reward < 1:
            rewards.append(0.0)
            continue
        code = extract_code(completion[-1]["content"])
        # code_list.append(code)
        golden_codes = [x['completion'] for x in inferences[i]]
        syn_sims = syntax_similarity(code, golden_codes)
        ast_sims = ast_similarity(code, golden_codes)

        combined_sims = []
        for s_syn, s_ast in zip(syn_sims, ast_sims):
            combined = alpha * s_syn + (1.0 - alpha) * s_ast
            combined_sims.append(combined)
        best_sim = max(combined_sims)
        rewards.append(best_sim)

    return rewards

def code_semantic_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
    evaluation_script_template = """
import subprocess
import json

def evaluate_code(code, test_cases):
    passed = 0
    total = len(test_cases)
    exec_timeout = 5

    for case in test_cases:
        process = subprocess.run(
            ["python3", "-c", f"{{code}}\\n{{case}}"],
            text=True,
            capture_output=True,
            timeout=exec_timeout
        )

        if process.returncode != 0:  # Error in execution
            continue

        # If we get here, the assertion passed (no error)
        passed += 1

    success_rate = (passed / total)
    return success_rate

code_snippet = {code_literal}
test_cases = {test_cases_literal}
rate = evaluate_code(code_snippet, test_cases)
print("__PASS_RATE__", rate)
"""
    # 1. compute format rewards
    # print(completions)
    use_loose_format = kwargs.get('use_loose_format', False)
    format_rewards = get_code_format_reward(language='python', use_loose_format=use_loose_format)(completions)
    # 2. collect scripts and their indices in the original array
    template = evaluation_script_template
    scripts = []
    valid_indices = []
    for i, (reward, completion) in enumerate(zip(format_rewards, completions)):
        if reward < 1:
            continue
        code = extract_code(completion[-1]["content"])
        tc = kwargs["test_cases"][i]
        scripts.append(
            template.format(
                code_literal=repr(code),
                test_cases_literal=repr(tc),
            )
        )
        valid_indices.append(i)
    # 3. execute scripts in parallel
    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )
    results = execution_provider.execute_scripts(scripts, ["python"] * len(scripts))
    print('=================execute scripts in parallel============', flush=True)
    print(results, flush=True)

    # 4. fill results into a list of the same length as completions, and keep None for reward=0
    final_results = [0.0] * len(completions)
    for idx, res in zip(valid_indices, results):
        final_results[idx] = res

    return final_results

def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        # "code_format_correctness": code_format_correctness,
        "code_format_correctness": get_code_format_reward(language=script_args.code_language, use_loose_format=script_args.use_loose_format),
        "code_syntax_correctness": update_wrapper(
            partial(
                code_syntax_correctness,
                use_loose_format=script_args.use_loose_format
            ),
            code_syntax_correctness,
        ),
        "code_static_checking_reward": update_wrapper(
            partial(
                code_static_checking_reward,
                use_loose_format=script_args.use_loose_format
            ),
            code_static_checking_reward,
        ),
        "code_static_semantic_checking_reward": update_wrapper(
            partial(
                code_static_semantic_checking_reward,
                use_loose_format=script_args.use_loose_format
            ),
            code_static_semantic_checking_reward,
        ),
        "code_similarity_checking_reward": update_wrapper(
            partial(
                code_similarity_checking_reward,
                use_loose_format=script_args.use_loose_format
            ),
            code_similarity_checking_reward,
        ),
        "code_semantic_reward": update_wrapper(
            partial(
                code_semantic_reward,
                use_loose_format=script_args.use_loose_format
            ),
            code_semantic_reward,
        ),
        
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                provider_type=getattr(script_args, "ioi_provider", "piston"),
            ),
            ioi_code_reward,
        ),
        "cf_code": update_wrapper(
            partial(
                cf_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                scoring_mode=script_args.code_eval_scoring_mode,
            ),
            cf_code_reward,
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
