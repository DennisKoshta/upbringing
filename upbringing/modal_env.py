"""Shared Modal images, volume and secrets for every upb-* app."""
import modal

vol = modal.Volume.from_name("upb-vol")
deps = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("trl[vllm]==1.14.2", "kernels", "datasets", "peft", "huggingface_hub")
    .env({"HF_HOME": "/vol/hf", "HF_XET_HIGH_PERFORMANCE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
eval_deps = deps.uv_pip_install("lm_eval[ifeval]==0.4.13", "langdetect", "immutabledict", "nltk").env(
    {"VLLM_USE_FLASHINFER_SAMPLER": "0"}  # FlashInfer JIT-compiles its sampler on Ada GPUs and fails without nvcc
)
image = deps.add_local_python_source("upbringing")
eval_image = eval_deps.add_local_python_source("upbringing")
secrets = [modal.Secret.from_name("upb-hf")]
