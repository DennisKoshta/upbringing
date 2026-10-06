"""Act I evaluation sweep on Modal: every time-lapse snapshot plus the base and AI2's official checkpoints.

    uv run modal run train/act1_eval.py::smoke              # one official checkpoint, small limits
    uv run modal run train/act1_eval.py::sweep --stage sft  # all SFT snapshots
"""
import json
import os

import modal

from upbringing.modal_env import eval_image, secrets, vol

app = modal.App("upb-act1-eval")
EVAL_DIR = "/vol/evals/act1"
RUN_DIRS = {"sft": "/vol/runs/act1-sft", "dpo": "/vol/runs/act1-dpo"}
OFFICIAL = {
    "base": ("allenai/OLMo-2-0425-1B", None),
    "ai2-sft": ("allenai/OLMo-2-0425-1B-SFT", None),
    "ai2-dpo": ("allenai/OLMo-2-0425-1B-DPO", None),
    "ai2-instruct": ("allenai/OLMo-2-0425-1B-Instruct", None),
}
fn = dict(image=eval_image, volumes={"/vol": vol}, secrets=secrets, timeout=3600)


@app.function(gpu="L40S", **fn)
def eval_ckpt(label: str, path: str, revision: str | None = None, ifeval_limit: int | None = None,
              gsm8k_limit: int | None = None):
    from upbringing.evals import evaluate_checkpoint

    out = evaluate_checkpoint(path, revision=revision, label=label, ifeval_limit=ifeval_limit, gsm8k_limit=gsm8k_limit)
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(f"{EVAL_DIR}/{label}.json", "w") as f:
        json.dump(out, f)
    vol.commit()
    return {k: out[k] for k in ("label", "ifeval", "gsm8k")} | {
        "diversity": {k: {kk: v[kk] for kk in ("distinct", "entropy_bits", "top")} for k, v in out["diversity"].items()}}


@app.function(gpu="L40S", **fn)
def base_logprobs():
    """Adds base-model log-probs to every eval file's time-lapse tokens."""
    from upbringing.evals import score_under_base

    vol.reload()
    files = sorted(f for f in os.listdir(EVAL_DIR) if f.endswith(".json"))
    data = {f: json.load(open(f"{EVAL_DIR}/{f}")) for f in files}
    scores = score_under_base({f: d["timelapse"] for f, d in data.items()})
    for f, d in data.items():
        for item, lps in zip(d["timelapse"], scores[f]):
            item["base_logprobs"] = lps
        json.dump(d, open(f"{EVAL_DIR}/{f}", "w"))
    vol.commit()
    return len(files)


@app.local_entrypoint()
def smoke():
    out = eval_ckpt.remote("smoke-ai2-sft", "allenai/OLMo-2-0425-1B-SFT", ifeval_limit=40, gsm8k_limit=40)
    print(json.dumps(out, indent=1)[:3000])


@app.local_entrypoint()
def sweep(stage: str = "sft", official: bool = False):
    jobs = []
    if official:
        jobs += [(label, path, rev) for label, (path, rev) in OFFICIAL.items()]
    if stage:
        snap_dir = f"{RUN_DIRS[stage]}/snapshots"
        for entry in vol.listdir(snap_dir.removeprefix("/vol")):
            name = os.path.basename(entry.path)
            jobs.append((f"{stage}-{name}", f"{snap_dir}/{name}", None))
    print(f"evaluating {len(jobs)} checkpoints")
    for out in eval_ckpt.starmap(jobs, order_outputs=False):
        print(out["label"], "ifeval", round(out["ifeval"]["prompt_loose"], 3), "gsm8k", round(out["gsm8k"]["acc"], 3),
              "number top", out["diversity"]["number"]["top"][:3])


@app.local_entrypoint()
def base_logprobs_entry():
    print("files scored:", base_logprobs.remote())
