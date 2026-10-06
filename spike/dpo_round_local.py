"""M0: latency of one Act II round on the local 4060 — generate 16 answer pairs, then a LoRA DPO update on them."""
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL = "allenai/OLMo-2-0425-1B-SFT"  # stand-in for our own SFT checkpoint
PROMPTS = [
    "Explain why the sky is blue.", "Give me a tip for learning to cook.", "What's a good name for a cat?",
    "How do vaccines work?", "Write a two-line poem about rain.", "Should I learn Python or JavaScript first?",
    "What causes inflation?", "Describe a sunset to someone who has never seen one.",
    "How can I sleep better?", "What is a black hole?", "Suggest a weekend activity.", "Why do we dream?",
    "How do I apologize to a friend?", "What's the best way to save money?", "Explain recursion simply.",
    "Is coffee bad for you?",
]

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")

t0 = time.time()
chat = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in PROMPTS]
enc = tok(chat * 2, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
with torch.no_grad():
    ids = model.generate(**enc, max_new_tokens=200, do_sample=True, temperature=0.9, pad_token_id=tok.pad_token_id)
texts = [tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in ids]
gen_s = time.time() - t0

# Simulated "length-lover" labeler: prefers the longer answer of each pair.
a, b = texts[:16], texts[16:]
rows = [{"prompt": [{"role": "user", "content": p}],
         "chosen": [{"role": "assistant", "content": max(x, y, key=len)}],
         "rejected": [{"role": "assistant", "content": min(x, y, key=len)}]} for p, x, y in zip(PROMPTS, a, b)]

cfg = DPOConfig(output_dir="/tmp/dpo_round", num_train_epochs=2, per_device_train_batch_size=4, gradient_accumulation_steps=1,
                learning_rate=5e-5, beta=0.1, max_length=512, bf16=True, logging_steps=2, save_strategy="no", report_to="none")
t1 = time.time()
trainer = DPOTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(rows), processing_class=tok,
                     peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
trainer.train()
train_s = time.time() - t1
print(f"\nGENERATE 32 answers: {gen_s:.1f}s | DPO 2 epochs on 16 pairs: {train_s:.1f}s | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print("final logs:", {k: round(v, 3) for k, v in trainer.state.log_history[-2].items() if isinstance(v, float)})
