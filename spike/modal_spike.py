"""M0 timing spike on Modal: SFT throughput for the 1B and GRPO step time for the 3B Countdown run.

    uv run modal run spike/modal_spike.py
"""
import json
import time

import modal

app = modal.App("upb-spike")
vol = modal.Volume.from_name("upb-vol")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("trl[vllm]==1.14.2", "kernels", "datasets", "peft")
    .env({"HF_HOME": "/vol/hf", "HF_XET_HIGH_PERFORMANCE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("upbringing")
)
common = dict(image=image, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("upb-hf")], timeout=45 * 60)


def timing_callback(records):
    from transformers import TrainerCallback

    class Timing(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            records.append({"t": time.time(), "step": state.global_step, **(logs or {})})
            print(json.dumps(records[-1]), flush=True)

    return Timing()


@app.function(gpu="H100", **common)
def sft_timing(steps: int = 40):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-SFT")  # carries the Tulu chat template
    ds = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225", split="train[:8000]").select_columns(["messages"])

    attn = None
    for impl in ["kernels-community/flash-attn2", "kernels-community/flash-attn", "sdpa"]:
        try:
            model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B", dtype=torch.bfloat16, attn_implementation=impl)
            attn = impl
            break
        except Exception as e:
            print(f"attn {impl} failed: {e!r}"[:300])
    records = []
    cfg = SFTConfig(
        output_dir="/tmp/sft", max_steps=steps, per_device_train_batch_size=16, gradient_accumulation_steps=1,
        max_length=4096, packing=True, learning_rate=2e-5, lr_scheduler_type="linear", warmup_steps=0.03,
        bf16=True, gradient_checkpointing=True, logging_steps=5, save_strategy="no", report_to="none",
    )
    t0 = time.time()
    SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, callbacks=[timing_callback(records)]).train()
    return {"attn": attn, "wall_s": time.time() - t0, "records": records, "mem_gb": torch.cuda.max_memory_allocated() / 1e9}


@app.function(gpu="H200", **common)
def grpo_timing(steps: int = 12):
    import torch
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    from upbringing.countdown import correctness_reward, format_reward, make_prompt

    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train[:2000]")
    ds = ds.map(lambda r: {"prompt": make_prompt(r["nums"], r["target"])})
    records = []
    cfg = GRPOConfig(
        output_dir="/tmp/grpo", max_steps=steps, learning_rate=1e-6, beta=0.001,
        per_device_train_batch_size=8, gradient_accumulation_steps=16, num_generations=8,  # 16 prompts x 8 = 128 rollouts/step
        max_completion_length=1024, temperature=1.0,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.35,
        bf16=True, gradient_checkpointing=True, logging_steps=1, save_strategy="no", report_to="none",
    )
    trainer = GRPOTrainer(model="Qwen/Qwen2.5-3B", reward_funcs=[correctness_reward, format_reward], args=cfg,
                          train_dataset=ds, callbacks=[timing_callback(records)])
    t0 = time.time()
    trainer.train()
    return {"wall_s": time.time() - t0, "records": records, "mem_gb": torch.cuda.max_memory_allocated() / 1e9}


@app.local_entrypoint()
def main(which: str = "sft,grpo"):
    fns = {"sft": sft_timing, "grpo": grpo_timing}
    calls = {name: fns[name].spawn() for name in which.split(",")}
    for name, call in calls.items():
        try:
            out = call.get()
        except Exception as e:
            out = {"error": repr(e)}
        with open(f"spike/{name}_timing.json", "w") as f:
            json.dump(out, f, indent=1)
        print(name, "done:", {k: v for k, v in out.items() if k != "records"})
