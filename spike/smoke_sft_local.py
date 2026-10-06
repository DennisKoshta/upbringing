"""Free local smoke test of upbringing.sft with a tiny random OLMo-2 and 200 real examples."""
import json, os, shutil
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from upbringing.sft import BASE, DATASET, run_sft
from upbringing.chat import load_tokenizer

out = "/tmp/claude-1000/smoke_sft"
shutil.rmtree(out, ignore_errors=True)
cfg = AutoConfig.from_pretrained(BASE)
cfg.update(dict(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4))
model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="sdpa")
ds = load_dataset(DATASET, split="train[:400]").select_columns(["messages"])
tr = run_sft(out, model=model, dataset=ds, attn="sdpa", per_device_batch=2, grad_accum=2, max_length=1024,
             max_steps=12, num_proc=2, save_steps=6, logging_steps=2)

# The trained tokens in the first packed batch must be assistant text only.
batch = next(iter(tr.get_train_dataloader()))
ids, labels = batch["input_ids"][0], batch["labels"][0]
trained = load_tokenizer().decode(ids[labels != -100][:200])
print("TRAINED SAMPLE:", repr(trained[:300]))
assert "<|user|>" not in trained and "<|assistant|>" not in trained
print("snapshots:", sorted(os.listdir(f"{out}/snapshots")), "| schedule:", json.load(open(f"{out}/schedule.json")))
print("metrics lines:", sum(1 for _ in open(f"{out}/metrics.jsonl")), "| trainer ckpts:", os.listdir(f"{out}/trainer"))
