"""Old vs corrected Countdown grader on the one logged batch kept from the first (stopped) RLVR run.

The first run did not stop generation at </answer>; the base model kept writing, often inventing new "User:" puzzles
and answering them. The original grader read the LAST <answer> block after </think>; the corrected grader reads the
FIRST (and generation now stops at </answer>). This script re-grades the same 256 rollouts both ways.

    PYTHONPATH=. uv run python analysis/grader_comparison.py [--json out.json]
"""
import gzip
import json
import re
import sys

from upbringing.countdown import extract_answer, is_correct

EVIDENCE = "analysis/evidence/first-run-step-00024.jsonl.gz"
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def old_extract(completion):
    """The original grader: last <answer> block after </think>."""
    if "</think>" not in completion:
        return None
    found = _ANSWER.findall(completion.split("</think>", 1)[1])
    return found[-1].strip() if found else None


def compare(path=EVIDENCE):
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    out = {"step": rows[0]["step"], "rollouts": len(rows), "multiple_answer_blocks": 0, "different_answer_read": 0,
           "old_rewarded": 0, "new_rewarded": 0, "false_positives": [], "false_negatives": []}
    chars_after = chars_total = 0
    for r in rows:
        c = r["completion"]
        post = c.split("</think>", 1)[1] if "</think>" in c else ""
        old, new = old_extract(c), extract_answer(c)
        old_ok, new_ok = is_correct(old, r["nums"], r["target"]), is_correct(new, r["nums"], r["target"])
        assert r["correct"] == (1.0 if old_ok else 0.0), "the logged reward must match the old rule"
        out["multiple_answer_blocks"] += len(_ANSWER.findall(post)) > 1
        out["different_answer_read"] += old != new
        out["old_rewarded"] += old_ok
        out["new_rewarded"] += new_ok
        case = {"nums": r["nums"], "target": r["target"], "first_answer": new, "answer_old_grader_read": old}
        if old_ok and not new_ok:
            out["false_positives"].append(case)
        if new_ok and not old_ok:
            out["false_negatives"].append(case)
        if "</answer>" in c:
            chars_after += len(c.split("</answer>", 1)[1])
        chars_total += len(c)
    out["share_of_text_after_first_answer"] = chars_after / chars_total
    return out


if __name__ == "__main__":
    result = compare()
    print(json.dumps(result, indent=1))
    if "--json" in sys.argv:
        json.dump(result, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
