"""RLVR: GRPO on Countdown from a base model (TinyZero-style), logging every rollout for the trace scrubber.

Each generation round samples 32 puzzles x 8 completions = 256 rollouts and takes 2 optimizer steps on them
(TinyZero/veRL: batch 256, mini-batch 128). fp32 master weights: at lr 1e-6 most bf16 weight updates would round away.
No KL term (beta 0), as in DAPO and Dr. GRPO.
"""
import glob
import gzip
import json
import os

from upbringing.ckpt import make_callback
from upbringing.countdown import extract_answer, is_correct, make_prompt

DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
N_EVAL = 256
SNAPSHOT_STEPS = [50, 100, 200, 300, 400, 600, 800]


def build_datasets(n_eval=N_EVAL, n_train=None):
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train")
    eval_ds = ds.select(range(len(ds) - n_eval, len(ds)))  # held out: never trained on
    train_ds = ds.select(range(n_train or len(ds) - n_eval))

    def add(split):
        return lambda r: {"prompt": make_prompt(r["nums"], r["target"]), "split": split}

    return train_ds.map(add("train")), eval_ds.map(add("eval"))


def make_rewards(rollout_dir):
    """Correctness (1.0) and format (0.1) rewards; correctness also appends every rollout to a per-step log."""
    os.makedirs(rollout_dir, exist_ok=True)

    def correctness_reward(completions, nums, target, split, trainer_state=None, **_):
        answers = [extract_answer(c) for c in completions]
        rewards = [1.0 if is_correct(a, n, t) else 0.0 for a, n, t in zip(answers, nums, target)]
        step = trainer_state.global_step if trainer_state is not None else -1
        with gzip.open(os.path.join(rollout_dir, f"step-{step:05d}.jsonl.gz"), "at") as f:
            for c, a, n, t, r, s in zip(completions, answers, nums, target, rewards, split):
                f.write(json.dumps({"step": step, "split": s, "nums": n, "target": t, "completion": c,
                                    "answer": a, "correct": r}) + "\n")
        return rewards

    def format_reward(completions, **_):
        return [0.1 if extract_answer(c) is not None else 0.0 for c in completions]

    return [correctness_reward, format_reward]


def run_grpo(out_dir, model="Qwen/Qwen2.5-3B", max_steps=900, prompts_per_round=32, num_generations=8,
             per_device_batch=8, updates_per_round=2, max_completion_length=1024, use_vllm=True,
             vllm_gpu_memory_utilization=0.25, eval_steps=50, save_steps=200, train_ds=None, eval_ds=None, **overrides):
    import torch
    from trl import GRPOConfig, GRPOTrainer

    if train_ds is None:
        train_ds, eval_ds = build_datasets()
    rollouts = prompts_per_round * num_generations
    grad_accum = rollouts // updates_per_round // per_device_batch
    cfg = GRPOConfig(**{**dict(
        output_dir=os.path.join(out_dir, "trainer"), max_steps=max_steps,
        learning_rate=1e-6, lr_scheduler_type="constant", warmup_steps=0, weight_decay=0.0, beta=0.0,
        per_device_train_batch_size=per_device_batch, gradient_accumulation_steps=grad_accum,
        generation_batch_size=rollouts, num_generations=num_generations,
        max_completion_length=max_completion_length, temperature=1.0,
        use_vllm=use_vllm, vllm_mode="colocate", vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        model_init_kwargs={"dtype": torch.float32},
        bf16=True, gradient_checkpointing=True,
        eval_strategy="steps", eval_steps=eval_steps, num_generations_eval=4, per_device_eval_batch_size=64,
        logging_steps=1, save_strategy="steps", save_steps=save_steps, save_total_limit=1,
        seed=0, report_to="none",
    ), **overrides})
    trainer = GRPOTrainer(model=model, reward_funcs=make_rewards(os.path.join(out_dir, "rollouts")), args=cfg,
                          train_dataset=train_ds, eval_dataset=eval_ds,
                          callbacks=[make_callback(out_dir, SNAPSHOT_STEPS, [])])
    resume = sorted(glob.glob(os.path.join(cfg.output_dir, "checkpoint-*")))
    trainer.train(resume_from_checkpoint=resume[-1] if resume else None)
    return trainer
