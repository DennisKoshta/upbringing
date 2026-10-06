"""Per-step behavior frequencies in RLVR rollouts (regex-based; an LLM-judge pass can refine this later).

    uv run python analysis/rlvr_behaviors.py <rollout_dir> [out.json]
"""
import collections
import glob
import gzip
import json
import os
import re
import sys

BEHAVIORS = {
    # re-checking a candidate against the target
    "verification": r"\b(let me (double[- ])?check|let's (double[- ])?check|verify|check(ing)? (if|whether|this)|this (works|is correct)|that (works|is correct))\b",
    # abandoning a line of attack
    "backtracking": r"\b(wait|doesn'?t work|does not work|not (equal|correct|right)|that'?s not|try (again|another|something else)|instead|too (high|low|big|small))\b",
    # breaking the problem into intermediate targets
    "subgoal": r"\b(we need to (get|reach|make)|first,? (let'?s|we|i)|remaining|left with|then we (need|have))\b",
}
PATTERNS = {k: re.compile(v, re.IGNORECASE) for k, v in BEHAVIORS.items()}


def think_part(completion):
    return completion.split("</think>", 1)[0]


def summarize(rows):
    n = len(rows)
    out = {"n": n, "accuracy": sum(r["correct"] for r in rows) / n,
           "answered": sum(r["answer"] is not None for r in rows) / n,
           "mean_chars": sum(len(r["completion"]) for r in rows) / n}
    for name, pat in PATTERNS.items():
        out[name] = sum(bool(pat.search(think_part(r["completion"]))) for r in rows) / n
    return out


def main(rollout_dir, out_path=None):
    by_step = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(os.path.join(rollout_dir, "step-*.jsonl.gz"))):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            by_step[r["split"]][r["step"]].append(r)
    result = {split: {step: summarize(rows) for step, rows in sorted(steps.items())} for split, steps in by_step.items()}
    for split, steps in result.items():
        print(f"== {split}")
        for step, s in steps.items():
            print(f"step {step:5d} n={s['n']:4d} acc={s['accuracy']:.3f} ans={s['answered']:.2f} chars={s['mean_chars']:6.0f} "
                  + " ".join(f"{k}={s[k]:.2f}" for k in BEHAVIORS))
    if out_path:
        json.dump(result, open(out_path, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:])
