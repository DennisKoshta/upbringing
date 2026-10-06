"""Command-line entry for DPO so it can run under torchrun (one process per GPU).

    torchrun --nproc_per_node 4 -m upbringing.dpo_cli --out-dir ... --sft-path ... --grad-accum 8
"""
import argparse

from upbringing.dpo import N_PAIRS, run_dpo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sft-path", required=True)
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--num-proc", type=int, default=16)
    a = ap.parse_args()
    run_dpo(a.out_dir, sft_path=a.sft_path, n_pairs=a.n_pairs, max_steps=a.max_steps, grad_accum=a.grad_accum,
            num_proc=a.num_proc)
