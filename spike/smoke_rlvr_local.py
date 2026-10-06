"""Free local smoke test of upbringing.rlvr: tiny random Qwen2, HF generation (no vLLM), a few rounds."""
import glob, gzip, json, os, shutil
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from upbringing.rlvr import build_datasets, run_grpo

out = "/tmp/claude-1000/smoke_rlvr"
shutil.rmtree(out, ignore_errors=True)
cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-3B")
cfg.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                layer_types=["full_attention"] * 2, max_window_layers=2))
tiny = f"{out}/tiny"
AutoModelForCausalLM.from_config(cfg).save_pretrained(tiny)
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B").save_pretrained(tiny)
train_ds, eval_ds = build_datasets(n_eval=8, n_train=64)
tr = run_grpo(out, model=tiny, max_steps=6, prompts_per_round=4, num_generations=4, per_device_batch=4,
              max_completion_length=24, use_vllm=False, eval_steps=4, save_steps=4, train_ds=train_ds, eval_ds=eval_ds,
              per_device_eval_batch_size=8, bf16=False, gradient_checkpointing=False)
files = sorted(glob.glob(f"{out}/rollouts/*.gz"))
rows = [json.loads(l) for f in files for l in gzip.open(f, "rt")]
print("rollout files:", [os.path.basename(f) for f in files])
print("rows:", len(rows), "splits:", {s: sum(r["split"] == s for r in rows) for s in ("train", "eval")})
print("metric keys:", sorted(k for k in tr.state.log_history[-2] if "reward" in k)[:6])
print("eval keys:", [k for h in tr.state.log_history for k in h if k.startswith("eval_")][:5])
