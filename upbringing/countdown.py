"""Countdown task (TinyZero-style): reach `target` using each of `nums` exactly once with + - * /.

The model completes a prompt that already opens a <think> block, so a well-formed completion
looks like `...reasoning...</think> <answer>(3 + 7) * 2</answer>`.
"""
import re
from collections import Counter

PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
    "User: Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic "
    "operations (+, -, *, /) and each number must be used exactly once. Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\n"
    "Assistant: Let me solve this step by step.\n<think>"
)

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ALLOWED = re.compile(r"^[\d+\-*/().\s]+$")


def make_prompt(nums, target):
    return PROMPT.format(nums=list(nums), target=target)


def extract_answer(completion):
    """The first <answer> block after the closing </think>, or None.

    The first, not the last: a base model that is not stopped at </answer> goes on to invent new "User:" puzzles and
    answer those, and grading a later block would score an answer to a question nobody asked.
    """
    if "</think>" not in completion:
        return None
    m = _ANSWER.search(completion.split("</think>", 1)[1])
    return m.group(1).strip() if m else None


def is_correct(expr, nums, target):
    if expr is None or not _ALLOWED.match(expr):
        return False
    if Counter(int(n) for n in re.findall(r"\d+", expr)) != Counter(int(n) for n in nums):
        return False
    try:
        value = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return False
    return abs(value - target) < 1e-5


def correctness_reward(completions, nums, target, **_):
    return [1.0 if is_correct(extract_answer(c), n, t) else 0.0 for c, n, t in zip(completions, nums, target)]


def format_reward(completions, **_):
    return [0.1 if extract_answer(c) is not None else 0.0 for c in completions]
