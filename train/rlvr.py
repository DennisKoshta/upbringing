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
