"""Free local smoke test of upbringing.dpo with a tiny random OLMo-2 and 96 real pairs."""
import os, shutil
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from upbringing.dpo import DATASET, run_dpo
from upbringing.sft import BASE

out = "/tmp/claude-1000/smoke_dpo"
shutil.rmtree(out, ignore_errors=True)
cfg = AutoConfig.from_pretrained(BASE)
cfg.update(dict(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4))
model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="sdpa")
ds = load_dataset(DATASET, split="train[:96]").select_columns(["chosen", "rejected"])
tr = run_dpo(out, sft_path=None, model=model, dataset=ds, attn="sdpa", per_device_batch=4, grad_accum=2,
             max_length=512, num_proc=2, save_steps=6, logging_steps=2)
print("loss types:", tr.args.loss_type, "beta", tr.args.beta)
print("last log:", {k: v for k, v in tr.state.log_history[-2].items() if k.startswith(("loss", "rewards"))})
print("snapshots:", sorted(os.listdir(f"{out}/snapshots")))
