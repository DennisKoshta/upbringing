"""Brute-force solvability of the held-out Countdown puzzles under the verifier's rule (use every number exactly once,
+ - * /, any parenthesization). Interprets accuracy ceilings: an unsolvable puzzle caps every model's score.

    PYTHONPATH=. uv run python analysis/countdown_solvable.py
"""
import itertools
import json
from fractions import Fraction

from datasets import load_dataset

N_EVAL = 256


def reachable(nums):
    """All values reachable by combining every number exactly once (exact rational arithmetic)."""
    if len(nums) == 1:
        return {nums[0]}
    out = set()
    n = len(nums)
    for i, j in itertools.combinations(range(n), 2):
        a, b = nums[i], nums[j]
        rest = [nums[k] for k in range(n) if k not in (i, j)]
        cands = {a + b, a - b, b - a, a * b}
        if b != 0:
            cands.add(a / b)
        if a != 0:
            cands.add(b / a)
        for c in cands:
            out |= reachable(rest + [c])
    return out


def main():
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    held = ds.select(range(len(ds) - N_EVAL, len(ds)))
    solvable = [Fraction(r["target"]) in reachable([Fraction(x) for x in r["nums"]]) for r in held]
    by_size = {}
    for r, s in zip(held, solvable):
        by_size.setdefault(len(r["nums"]), []).append(s)
    result = {"held_out": N_EVAL, "solvable": sum(solvable), "share": sum(solvable) / N_EVAL,
              "by_size": {k: [sum(v), len(v)] for k, v in sorted(by_size.items())},
              "solvable_mask": solvable}
    print(json.dumps({k: v for k, v in result.items() if k != "solvable_mask"}, indent=1))
    json.dump(result, open("site/data/rlvr/solvable.json", "w"))


if __name__ == "__main__":
    main()
