"""Pull results from the Modal volume and compact them into site/data/ for the static site.

    PYTHONPATH=. uv run python scripts/export_site.py [--skip-download]

Downloads are cached under runs/cache/ (gitignored); rerun any time to refresh with newer checkpoints.
"""
import argparse
import collections
import gzip
import json
import os
import re

import modal

from analysis.rlvr_behaviors import BEHAVIORS, summarize
from upbringing.evals import TIMELAPSE_PROMPTS

CACHE = "runs/cache"
SITE = "site/data"
vol = modal.Volume.from_name("upb-vol")


def pull(remote, local, refresh=False):
    """Copy one volume file into the cache (skipped if present unless refresh)."""
    if os.path.exists(local) and not refresh:
        return True
    os.makedirs(os.path.dirname(local), exist_ok=True)
    try:
        data = b"".join(vol.read_file(remote))
    except Exception:
        return False
    with open(local, "wb") as f:
        f.write(data)
    return True


def ls(remote):
    try:
        return [os.path.basename(e.path) for e in vol.listdir(remote)]
    except Exception:
        return []


def write(path, obj):
    path = os.path.join(SITE, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    return os.path.getsize(path)


# ---------------------------------------------------------------- Act I

def ckpt_order(label):
    if label == "base":
        return (0, 0)
    stage, step = re.match(r"(sft|dpo)-step-(\d+)", label).groups()
    return (1 if stage == "sft" else 2, int(step))


def export_act1(download):
    files = ls("evals/act1")
    if download:
        for f in files:
            # Eval files gain base log-probs after the fact, so always refresh them.
            pull(f"evals/act1/{f}", f"{CACHE}/evals/act1/{f}", refresh=True)
        for run in ("act1-sft", "act1-dpo"):
            pull(f"runs/{run}/metrics.jsonl", f"{CACHE}/{run}/metrics.jsonl", refresh=True)
            pull(f"runs/{run}/schedule.json", f"{CACHE}/{run}/schedule.json", refresh=True)
    evals = {}
    for f in sorted(os.listdir(f"{CACHE}/evals/act1")):
        if f.endswith(".json"):
            evals[f[:-5]] = json.load(open(f"{CACHE}/evals/act1/{f}"))

    refs = {k: evals.pop(k) for k in ("ai2-sft", "ai2-dpo", "ai2-instruct") if k in evals}
    order = sorted(evals, key=ckpt_order)

    def summary(label, e):
        if label.startswith("ai2-"):
            stage, step = "ref", None
        elif label == "base":
            stage, step = "base", 0
        else:
            rank, step = ckpt_order(label)
            stage = "sft" if rank == 1 else "dpo"
        div = e["diversity"]
        return {
            "id": label, "stage": stage, "step": step,
            "ifeval": round(e["ifeval"]["prompt_loose"], 4), "ifeval_strict": round(e["ifeval"]["prompt_strict"], 4),
            "gsm8k": round(e["gsm8k"]["acc"], 4), "gsm8k_tokens": round(e["gsm8k"]["mean_tokens"], 1),
            "number": {"hist": div["number"]["histogram"], "valid": div["number"]["valid"],
                       "distinct": div["number"]["distinct"], "entropy": div["number"]["entropy_bits"]},
            **{k: {"top": div[k]["top"][:8], "top_raw": div[k].get("top_raw", [])[:8], "valid": div[k]["valid"], "distinct": div[k]["distinct"],
                   "entropy": div[k]["entropy_bits"], "examples": div[k]["examples"][:6]}
               for k in ("joke", "animal", "story")},
        }

    ckpts = [summary(k, evals[k]) for k in order]
    references = [summary(k, v) | {"name": {"ai2-sft": "AI2 SFT", "ai2-dpo": "AI2 DPO",
                                            "ai2-instruct": "AI2 Instruct (SFT+DPO+RLVR)"}[k]}
                  for k, v in refs.items()]

    def series(run, keys):
        path = f"{CACHE}/{run}/metrics.jsonl"
        if not os.path.exists(path):
            return []
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if "loss" in r and "train_runtime" not in r]
        return [{"step": r["step"], **{k: round(r[k], 5) for k in keys if k in r}} for r in rows]

    training = {
        "sft": series("act1-sft", ["loss", "learning_rate", "num_tokens", "mean_token_accuracy"]),
        "dpo": series("act1-dpo", ["loss", "rewards/accuracies", "rewards/margins", "rewards/chosen", "rewards/rejected"]),
        "schedules": {run: json.load(open(f"{CACHE}/{run}/schedule.json"))
                      for run in ("act1-sft", "act1-dpo") if os.path.exists(f"{CACHE}/{run}/schedule.json")},
    }
    size = write("act1/checkpoints.json", {"checkpoints": ckpts, "references": references, "training": training,
                                           "prompts": TIMELAPSE_PROMPTS})

    for i, prompt in enumerate(TIMELAPSE_PROMPTS):
        frames = []
        for label in order + list(refs):
            e = evals.get(label) or refs[label]
            t = e["timelapse"][i]
            frames.append({"id": label, "text": t["text"], "tokens": t["tokens"], "lp": t["logprobs"],
                           "blp": t.get("base_logprobs"), "finish": t["finish"]})
        size += write(f"act1/timelapse-{i:02d}.json", {"prompt": prompt, "frames": frames})
    print(f"act1: {len(ckpts)} checkpoints + {len(references)} references, {size / 1e6:.2f} MB")


# ---------------------------------------------------------------- RLVR

def export_rlvr(download, n_traces=10):
    run = "runs/rlvr-3b"
    if download:
        pull(f"{run}/metrics.jsonl", f"{CACHE}/rlvr-3b/metrics.jsonl", refresh=True)
        pull(f"{run}/base_eval.jsonl.gz", f"{CACHE}/rlvr-3b/base_eval.jsonl.gz")
        files = sorted(ls(f"{run}/rollouts"))
        for f in files[:-1]:  # the newest file may still be written to
            pull(f"{run}/rollouts/{f}", f"{CACHE}/rlvr-3b/rollouts/{f}")
    rows_by_step = collections.defaultdict(lambda: collections.defaultdict(list))
    rdir = f"{CACHE}/rlvr-3b/rollouts"
    for f in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
        for line in gzip.open(os.path.join(rdir, f), "rt"):
            r = json.loads(line)
            rows_by_step[r["split"]][r["step"]].append(r)
    base_path = f"{CACHE}/rlvr-3b/base_eval.jsonl.gz"
    if os.path.exists(base_path):
        for line in gzip.open(base_path, "rt"):
            r = json.loads(line)
            rows_by_step["eval"][0].append(r)

    train = [{"step": s, **summarize(rows)} for s, rows in sorted(rows_by_step["train"].items())]
    evals = [{"step": s, **summarize(rows)} for s, rows in sorted(rows_by_step["eval"].items())]
    metrics = []
    mpath = f"{CACHE}/rlvr-3b/metrics.jsonl"
    if os.path.exists(mpath):
        for line in open(mpath):
            r = json.loads(line)
            if "completions/mean_length" in r:
                metrics.append({"step": r["step"], "tokens": round(r["completions/mean_length"], 1),
                                "entropy": round(r.get("entropy", 0), 4)})
    for d in train + evals:
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 4)
    size = write("rlvr/curves.json", {"train": train, "eval": evals, "tokens": metrics,
                                      "behaviors": {k: v for k, v in BEHAVIORS.items()},
                                      "max_steps": 900})

    # Trace scrubber: held-out puzzles followed across eval checkpoints. Prefer puzzles the base model never solved
    # and the latest checkpoint usually does.
    eval_steps = sorted(rows_by_step["eval"])
    puzzles = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in eval_steps:
        for r in rows_by_step["eval"][s]:
            puzzles[(tuple(r["nums"]), r["target"])][s].append(r)
    last = eval_steps[-1] if eval_steps else None

    def interest(key):
        by = puzzles[key]
        first = sum(r["correct"] for r in by.get(0, [])) / max(len(by.get(0, [])), 1)
        final = sum(r["correct"] for r in by.get(last, [])) / max(len(by.get(last, [])), 1)
        return (final - first, len(key[0]), -abs(key[1] - 50))

    chosen = sorted(puzzles, key=interest, reverse=True)[:n_traces]
    index = []
    for i, key in enumerate(chosen):
        frames = [{"step": s, "samples": [{"text": r["completion"], "correct": r["correct"], "answer": r["answer"]}
                                          for r in puzzles[key][s][:4]]} for s in eval_steps if s in puzzles[key]]
        size += write(f"rlvr/trace-{i:02d}.json", {"nums": list(key[0]), "target": key[1], "frames": frames})
        index.append({"file": f"trace-{i:02d}.json", "nums": list(key[0]), "target": key[1]})
    size += write("rlvr/traces.json", {"puzzles": index, "steps": eval_steps})
    print(f"rlvr: {len(train)} train steps, {len(evals)} eval points, {len(index)} traces, {size / 1e6:.2f} MB")


# ---------------------------------------------------------------- costs

def export_costs():
    import datetime
    import subprocess

    end = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()
    out = subprocess.run(["modal", "billing", "report", "--start", "2026-10-06", "--end", end, "--json"],
                         capture_output=True, text=True).stdout
    by_app = collections.Counter()
    for r in json.loads(out or "[]"):
        if r["description"].startswith("upb-"):
            by_app[r["description"]] += float(r["cost"])
    write("costs.json", {"budget": 100, "by_app": {k: round(v, 2) for k, v in by_app.items()},
                         "total": round(sum(by_app.values()), 2), "as_of": datetime.datetime.now().isoformat(timespec="minutes")})
    print(f"costs: ${sum(by_app.values()):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    export_act1(not args.skip_download)
    export_rlvr(not args.skip_download)
    export_costs()
