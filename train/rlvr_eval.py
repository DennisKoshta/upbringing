"""Post-hoc RLVR evaluation with identical settings for the base model and every saved checkpoint.

Held-out set: the last 256 Countdown puzzles (never reached by training). For each checkpoint:
  * 32 samples per puzzle at temperature 1.0 (top-p 1, 1,024-token cap, stop at </answer>), graded with the corrected
    verifier -> per-attempt accuracy (mean over all 8,192 attempts) with a puzzle-level bootstrap CI, and unbiased
    pass@k for k = 1..32 (Chen et al. 2021 estimator);
  * one greedy completion per puzzle -> greedy accuracy.

    uv run modal run train/rlvr_eval.py::sweep
"""
import json
import os

import modal

from upbringing.modal_env import image, secrets, vol

app = modal.App("upb-rlvr-eval")
OUT = "/vol/evals/rlvr"
RUN = "/vol/runs/rlvr-3b"
N_SAMPLES = 32
KS = [1, 2, 4, 8, 16, 32]


def pass_at_k(n, c, k):
    """Unbiased estimate of P(at least one of k samples is correct) from c correct out of n."""
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(n - c + 1, n + 1):
        p *= 1.0 - k / i
    return 1.0 - p


@app.function(gpu="L40S", image=image, volumes={"/vol": vol}, secrets=secrets, timeout=3600,
              env={"VLLM_USE_FLASHINFER_SAMPLER": "0"})
def posthoc(label: str, path: str, step: int):
    import random

    from vllm import LLM, SamplingParams

    from upbringing.countdown import extract_answer, is_correct
    from upbringing.rlvr import build_datasets

    _, eval_ds = build_datasets(n_train=1)
    llm = LLM(model=path, tokenizer="Qwen/Qwen2.5-3B", dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=2048, seed=0)
    stop = dict(stop=["</answer>"], include_stop_str_in_output=True)
    prompts = list(eval_ds["prompt"])
    sampled = llm.generate(prompts, SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0, max_tokens=1024, seed=0, **stop))
    greedy = llm.generate(prompts, SamplingParams(n=1, temperature=0.0, max_tokens=1024, **stop))

    puzzles = []
    for row, s, g in zip(eval_ds, sampled, greedy):
        ok = [is_correct(extract_answer(o.text), row["nums"], row["target"]) for o in s.outputs]
        puzzles.append({
            "nums": row["nums"], "target": row["target"], "correct": sum(ok),
            "greedy": is_correct(extract_answer(g.outputs[0].text), row["nums"], row["target"]),
            "tokens": sum(len(o.token_ids) for o in s.outputs) / len(s.outputs),
        })
    n_p = len(puzzles)
    acc = sum(p["correct"] for p in puzzles) / (n_p * N_SAMPLES)
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(puzzles)["correct"] for _ in range(n_p)) / (n_p * N_SAMPLES) for _ in range(2000))
    out = {
        "label": label, "step": step, "path": path, "n_puzzles": n_p, "samples_per_puzzle": N_SAMPLES,
        "settings": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 1024, "stop": "</answer>", "seed": 0},
        "per_attempt_accuracy": acc, "ci95": [boots[50], boots[1949]],
        "greedy_accuracy": sum(p["greedy"] for p in puzzles) / n_p,
        "pass_at_k": {str(k): sum(pass_at_k(N_SAMPLES, p["correct"], k) for p in puzzles) / n_p for k in KS},
        "mean_tokens": sum(p["tokens"] for p in puzzles) / n_p,
        "puzzles": puzzles,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{label}.json", "w") as f:
        json.dump(out, f)
    vol.commit()
    return {k: out[k] for k in ("label", "per_attempt_accuracy", "ci95", "greedy_accuracy", "pass_at_k", "mean_tokens")}


@app.local_entrypoint()
def sweep(only: str = ""):
    jobs = [("base", "Qwen/Qwen2.5-3B", 0)]
    try:
        for e in vol.listdir(RUN.removeprefix("/vol") + "/snapshots"):
            name = os.path.basename(e.path)
            jobs.append((f"rlvr-{name}", f"{RUN}/snapshots/{name}", int(name.split("-")[1])))
    except Exception:
        pass
    done = set()
    try:
        done = {os.path.basename(e.path).removesuffix(".json") for e in vol.listdir(OUT.removeprefix("/vol"))}
    except Exception:
        pass
    jobs = [j for j in jobs if j[0] not in done and (not only or j[0] in only.split(","))]
    print(f"evaluating {len(jobs)} checkpoints (skipping {len(done)})")
    for r in posthoc.starmap(jobs, order_outputs=False):
        print(r["label"], f"acc {r['per_attempt_accuracy']:.3f} ci {r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}",
              f"greedy {r['greedy_accuracy']:.3f}", "pass@k", {k: round(v, 3) for k, v in r["pass_at_k"].items()},
              f"tokens {r['mean_tokens']:.0f}")
