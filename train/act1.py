"""Act I on Modal.

    uv run modal run --detach train/act1.py::sft
    uv run modal run --detach train/act1.py::dpo   # after SFT finishes
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


DPO_DIR = "/vol/runs/act1-dpo"


@app.function(gpu="H100", cpu=16, memory=65536, image=image, volumes={"/vol": vol}, secrets=secrets,
              timeout=8 * 3600, retries=modal.Retries(max_retries=2, initial_delay=5.0))
def dpo(sft_step: int = 6500, out_dir: str = DPO_DIR, n_pairs: int = 60_000, max_steps: int = -1):
    from upbringing.dpo import run_dpo

    try:
        run_dpo(out_dir, sft_path=f"{SFT_DIR}/snapshots/step-{sft_step:05d}", n_pairs=n_pairs, max_steps=max_steps)
    finally:
        vol.commit()
