"""RLVR on Modal: GRPO on Countdown from Qwen2.5-3B base.

    uv run modal run --detach train/rlvr.py::grpo
"""
import modal

from upbringing.modal_env import image, secrets, vol

app = modal.App("upb-rlvr")


@app.function(gpu="H200", cpu=8, memory=98304, image=image, volumes={"/vol": vol}, secrets=secrets,
              timeout=12 * 3600, retries=modal.Retries(max_retries=2, initial_delay=5.0))
def grpo(model: str = "Qwen/Qwen2.5-3B", out_dir: str = "/vol/runs/rlvr-3b", max_steps: int = 900):
    from upbringing.rlvr import run_grpo

    try:
        run_grpo(out_dir, model=model, max_steps=max_steps)
    finally:
        vol.commit()


@app.function(gpu="L40S", image=image, volumes={"/vol": vol}, secrets=secrets, timeout=3600,
              env={"VLLM_USE_FLASHINFER_SAMPLER": "0"})
def base_eval(model: str = "Qwen/Qwen2.5-3B", out_path: str = "/vol/runs/rlvr-3b/base_eval.jsonl.gz"):
    """Step-0 baseline on the held-out puzzles, sampled exactly like the in-training eval (4 samples, T=1, stop at
    </answer>), so the trace scrubber can start from the untrained model."""
    import gzip
    import json

    from vllm import LLM, SamplingParams

    from upbringing.countdown import extract_answer, is_correct
    from upbringing.rlvr import build_datasets

    _, eval_ds = build_datasets(n_train=1)
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048, seed=0)
    params = SamplingParams(n=4, temperature=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True,
                            seed=0)
    res = llm.generate(list(eval_ds["prompt"]), params)
    n_correct = 0
    with gzip.open(out_path, "wt") as f:
        for row, r in zip(eval_ds, res):
            for o in r.outputs:
                ans = extract_answer(o.text)
                ok = 1.0 if is_correct(ans, row["nums"], row["target"]) else 0.0
                n_correct += ok
                f.write(json.dumps({"step": 0, "split": "eval", "nums": row["nums"], "target": row["target"],
                                    "completion": o.text, "answer": ans, "correct": ok}) + "\n")
    vol.commit()
    return n_correct / (4 * len(eval_ds))


@app.local_entrypoint()
def base_eval_entry():
    print("base held-out accuracy:", base_eval.remote())
