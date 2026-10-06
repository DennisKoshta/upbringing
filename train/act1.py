"""Act I on Modal.

    uv run modal run --detach train/act1.py::sft
"""
import modal

from upbringing.modal_env import image, secrets, vol

app = modal.App("upb-act1")
SFT_DIR = "/vol/runs/act1-sft"


@app.function(gpu="H100", cpu=16, memory=65536, image=image, volumes={"/vol": vol}, secrets=secrets,
              timeout=10 * 3600, retries=modal.Retries(max_retries=2, initial_delay=5.0))
def sft(max_steps: int = 6500):
    # 6500 steps x ~48k tokens = ~58% of one epoch of the Tulu mix: a budget cut from AI2's 2 epochs.
    from upbringing.sft import run_sft

    try:
        run_sft(SFT_DIR, max_steps=max_steps)
    finally:
        vol.commit()
