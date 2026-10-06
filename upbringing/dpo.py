"""Act I, stage 2: DPO on top of our SFT, following AI2's 1B recipe on a random subset of their preference mix.

AI2 (open-instruct docs/olmo2.md): dpo_norm (length-normalized DPO) with beta 5, lr 2.5e-6, linear, 10% warmup,
wd 0, 128 pairs/step, max len 2048, 1 epoch over allenai/olmo-2-0425-1b-preference-mix (378k pairs).
TRL's `sigmoid_norm` is the same objective. Budget deviation: we train on N_PAIRS random pairs, not all 378k.
"""
import glob
import os

from upbringing.chat import load_tokenizer
from upbringing.ckpt import make_callback

DATASET = "allenai/olmo-2-0425-1b-preference-mix"
N_PAIRS = 80_000
EARLY_STEPS = [5, 10, 20, 35, 50, 75, 100, 150, 225, 350, 500]
LATE_FRACTIONS = [0.25, 0.5, 0.75]


def run_dpo(out_dir, sft_path, model=None, dataset=None, n_pairs=N_PAIRS, attn="kernels-community/flash-attn2",
            per_device_batch=8, grad_accum=16, max_length=2048, max_steps=-1, num_proc=16, save_steps=200, **overrides):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM
    from trl import DPOConfig, DPOTrainer

    tok = load_tokenizer()
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(sft_path, dtype=torch.float32, attn_implementation=attn)
    if dataset is None:
        dataset = load_dataset(DATASET, split="train").shuffle(seed=111).select(range(n_pairs))
        dataset = dataset.select_columns(["chosen", "rejected"])

    cfg = DPOConfig(**{**dict(
        output_dir=os.path.join(out_dir, "trainer"),
        num_train_epochs=1, max_steps=max_steps,
        per_device_train_batch_size=per_device_batch, gradient_accumulation_steps=grad_accum,
        learning_rate=2.5e-6, lr_scheduler_type="linear", warmup_steps=0.1, weight_decay=0.0,
        loss_type=["sigmoid_norm"], beta=5.0, max_length=max_length, precompute_ref_log_probs=True,
        bf16=True, gradient_checkpointing=True,
        logging_steps=5, save_strategy="steps", save_steps=save_steps, save_total_limit=1,
        dataset_num_proc=num_proc, seed=111, report_to="none",
    ), **overrides})
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=dataset, processing_class=tok,
                         callbacks=[make_callback(out_dir, EARLY_STEPS, LATE_FRACTIONS)])
    resume = sorted(glob.glob(os.path.join(cfg.output_dir, "checkpoint-*")))
    trainer.train(resume_from_checkpoint=resume[-1] if resume else None)
    return trainer
