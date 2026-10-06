"""Act I, stage 1: SFT of OLMo-2-0425-1B on the Tulu 3 OLMo-2 mixture, following AI2's 1B recipe at 1 epoch.

AI2 (open-instruct docs/olmo2.md): lr 3e-5, linear, 3% warmup, wd 0, 128 seqs/step, max len 4096, 2 epochs, bf16
mixed precision. We keep everything but the epoch count, and pack to 12 x 4096 tokens/step (~ the same tokens per step
as 128 unpacked sequences at ~390 tokens each).
"""
import glob
import os

from upbringing.chat import load_tokenizer
from upbringing.ckpt import make_callback

BASE = "allenai/OLMo-2-0425-1B"
DATASET = "allenai/tulu-3-sft-olmo-2-mixture-0225"


def run_sft(out_dir, model=None, dataset=None, attn="kernels-community/flash-attn2", per_device_batch=4, grad_accum=3,
            max_length=4096, max_steps=-1, gradient_checkpointing=False, num_proc=16, save_steps=1000, **overrides):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    tok = load_tokenizer()
    if model is None:
        # fp32 master weights with bf16 autocast, matching accelerate --mixed_precision bf16.
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, attn_implementation=attn)
    if dataset is None:
        dataset = load_dataset(DATASET, split="train").select_columns(["messages"])

    cfg = SFTConfig(**{**dict(
        output_dir=os.path.join(out_dir, "trainer"),
        num_train_epochs=1, max_steps=max_steps,
        per_device_train_batch_size=per_device_batch, gradient_accumulation_steps=grad_accum,
        learning_rate=3e-5, lr_scheduler_type="linear", warmup_steps=0.03, weight_decay=0.0,
        max_length=max_length, packing=True, assistant_only_loss=True,
        bf16=True, gradient_checkpointing=gradient_checkpointing,
        logging_steps=5, save_strategy="steps", save_steps=save_steps, save_total_limit=1,
        dataset_num_proc=num_proc, seed=1, report_to="none",
    ), **overrides})
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=dataset, processing_class=tok,
                         callbacks=[make_callback(out_dir)])
    resume = sorted(glob.glob(os.path.join(cfg.output_dir, "checkpoint-*")))
    trainer.train(resume_from_checkpoint=resume[-1] if resume else None)
    return trainer
