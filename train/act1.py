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


N_GPUS = 4


@app.function(gpu=f"H100:{N_GPUS}", cpu=32, memory=196608, image=image, volumes={"/vol": vol}, secrets=secrets,
              timeout=8 * 3600, retries=modal.Retries(max_retries=2, initial_delay=5.0))
def dpo(sft_step: int = 6500, out_dir: str = DPO_DIR, n_pairs: int = 60_000, max_steps: int = -1):
    # Data-parallel over N_GPUS processes; per-device 4 pairs x (32 / N_GPUS) accumulation keeps 128 pairs per step.
    import subprocess

    try:
        subprocess.run(["torchrun", "--standalone", f"--nproc_per_node={N_GPUS}", "-m", "upbringing.dpo_cli",
                        "--out-dir", out_dir, "--sft-path", f"{SFT_DIR}/snapshots/step-{sft_step:05d}",
                        "--n-pairs", str(n_pairs), "--max-steps", str(max_steps), "--grad-accum", str(32 // N_GPUS),
                        "--num-proc", "8"], check=True)
    finally:
        vol.commit()
