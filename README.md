# Upbringing: LLM post-training and evaluation

Small-model post-training runs (SFT, DPO, GRPO) on open models, with the evaluation infrastructure to tell whether
training did anything: checkpoint sweeps, reference models, held-out evaluation, full rollout logging. The project
page presents the results interactively: **[denniskoshta.github.io/upbringing](https://denniskoshta.github.io/upbringing/)**

| | What was done | Model |
|---|---|---|
| **I. SFT → DPO** | AI2's published 1B post-training recipe, run on a reduced budget, with dense checkpoints. Each checkpoint is evaluated on IFEval, GSM8K, diversity probes, and fixed prompts with per-token log-probabilities rescored under the base model. | OLMo-2-0425-1B |
| **II. Preference learning** | Simulated labelers with known rules train an SFT model by LoRA DPO, round by round; a Bradley–Terry fit recovers each labeler's rule from their choices. Visitors can judge pairs and see the same fit applied to themselves. | OLMo-2 1B SFT |
| **III. RLVR** | GRPO on Countdown from a base model with a programmatic verifier. Every rollout is logged; held-out evaluation during training and post-hoc with pass@k at matched sampling budgets. | Qwen2.5-3B |

## Status (2026-10-06)

| Component | Status |
|---|---|
| Act I SFT (6,500 steps) | Running on Modal |
| Act I DPO (60k pairs) | Queued; starts automatically after SFT (`scripts/act1_chain.sh`) |
| Act I evaluation sweep | Base and AI2 references done; our checkpoints run automatically after training |
| Act II replays | Generated with **AI2's SFT model as a stand-in**; to be regenerated from our SFT checkpoint |
| Act II live training | Not built (deferred) |
| Act III RLVR (900 steps) | Running on Modal |
| Act III post-hoc evaluation | Base and early checkpoints done; remaining checkpoints after the run |
| Project page | Live, with data from completed stages |

This table is updated as stages finish. Numbers below are from completed stages only.

## Results so far

**Evaluation harness, validated on AI2's released checkpoints.** IFEval (prompt-level loose) on AI2's DPO model:
67.1% (AI2 reports 67.1); SFT 49.4% (50.5); Instruct 68.4% (70.1). This validates our evaluation of their
checkpoints, not a reproduction of their training results; our budget-cut checkpoints are scored separately.

**RLVR (in progress).** Held-out accuracy is reported per attempt: 256 puzzles never used in training, 4 attempts each at
temperature 1, corrected verifier (1,024 attempts per point). Base model: 1.6%. Final numbers, with confidence intervals
and pass@k at matched sampling budgets, will be added when the run completes.

**A grading bug, found and measured.** The first RLVR run did not stop generation at `</answer>`; the base model kept
writing and often invented new "User:" puzzles and answered them (86% of generated text came after the first answer).
The original grader read the *last* answer block. Re-grading the one logged batch kept from that run (step 24, 256
rollouts; [`analysis/grader_comparison.py`](analysis/grader_comparison.py)):

- the old grader read a different answer than the model's first in 166 of 256 rollouts;
- the batch's only reward under the old grader went to a rollout whose actual answer, `(20 + 85) / 93 = 1.08651`, was
  wrong, because a later invented block happened to contain a correct expression;
- the one truly correct answer in the batch, `20 + 1 * 62` for 82, received no reward because the grader read a later
  block (`4000 + (550 x 10) = 10,500 pounds`).

Fix: stop generation at `</answer>`, grade the first answer block, and test both. This is a grading vulnerability with a
measured effect on rewards in one batch, not demonstrated reward hacking: the run was stopped at step 24, before any
evidence of optimization exploiting it could accumulate.

## Exact configurations

| Stage | Configuration (source of truth: the linked file) |
|---|---|
| SFT ([`upbringing/sft.py`](upbringing/sft.py)) | Base `allenai/OLMo-2-0425-1B`; data `allenai/tulu-3-sft-olmo-2-mixture-0225`; lr 3e-5, linear, warmup 3%, weight decay 0; 4 × 4096-token packed sequences per device × 3 accumulation (≈48k tokens/step); max length 4096; assistant-only loss (chat template with `{% generation %}` markers, tested byte-identical to AI2's); fp32 master weights, bf16 autocast; flash-attention 2; 6,500 steps (≈0.58 epoch; AI2: 2 epochs); seed 1; 1× H100 |
| DPO ([`upbringing/dpo.py`](upbringing/dpo.py)) | Init: our SFT step 6,500; data: 60,000 random pairs (seed 111) of `allenai/olmo-2-0425-1b-preference-mix` (AI2: all 378k); loss `sigmoid_norm` (= open-instruct `dpo_norm`), β = 5; lr 2.5e-6, linear, warmup 10%; 4 pairs/device × 32 accumulation = 128 pairs/step; max length 2048; padding-free; reference log-probs precomputed under bf16 autocast; fp32 master weights; 1 epoch; 1× H100 |
| Act I evals ([`upbringing/evals.py`](upbringing/evals.py)) | vLLM, bf16, chat template for every checkpoint including base; greedy; IFEval (541 prompts, lm-eval checkers, max 1024 tokens); GSM8K test (1,319, 0-shot CoT, max 512); 4 diversity probes × 300 samples at T = 1; 14 time-lapse prompts (max 384) with per-token log-probs, rescored under the base model |
| Act II replays ([`scripts/act2_replay.py`](scripts/act2_replay.py)) | LoRA r 16, α 32, all linear layers; DPO β 0.1, lr 5e-5 constant, 3 epochs per round, batch 4; 16 pairs per round (2 samples per prompt, T = 1, top-p 0.95, max 300 tokens); 8 rounds; reference = start-of-round policy (iterative DPO); laptop RTX 4060 |
| RLVR ([`upbringing/rlvr.py`](upbringing/rlvr.py)) | Base `Qwen/Qwen2.5-3B`; data `Jiayi-Pan/Countdown-Tasks-3to4`, last 256 rows held out; reward 1.0 correct + 0.1 well-formed ([`upbringing/countdown.py`](upbringing/countdown.py)); GRPO, TRL `dapo` loss, group-std advantage scaling, ε 0.2, β 0 (no KL); lr 1e-6 constant; 32 prompts × 8 samples = 256 rollouts per round, 2 optimizer steps per round; T = 1, max 1024 completion tokens, stop at `</answer>`; fp32 master weights, bf16 autocast, gradient checkpointing; vLLM colocated; 900 optimizer steps; seed 0; 1× H200 |
| RLVR post-hoc eval ([`train/rlvr_eval.py`](train/rlvr_eval.py)) | Same 256 held-out puzzles for every checkpoint; 32 samples at T = 1 (top-p 1, max 1024, stop `</answer>`) → per-attempt accuracy with puzzle-bootstrap 95% CI and unbiased pass@k (k = 1…32); plus greedy |

Libraries: Modal images use TRL 1.14.2, transformers 5.19.0, vLLM 0.30.0, torch 2.13.0; local runs use torch 2.14.1
(CUDA 12.6), PEFT 0.21.2. `uv.lock` pins the local environment.

## Tests

`uv run pytest` covers the parts that decide correctness: the chat template renders byte-identically to AI2's and masks
exactly the assistant tokens; the Countdown verifier (number use, safe evaluation, first-answer extraction, and the
documented re-grade of the first run's logged batch); eval answer parsing and diversity normalization; Bradley–Terry
recovery of a known rule; Act II feature tagging.

## Reproduce

```bash
uv sync && uv run pytest
uv run modal run --detach train/act1.py::sft                  # Act I SFT    (1x H100)
uv run modal run --detach train/act1.py::dpo                  # Act I DPO    (1x H100)
uv run modal run train/act1_eval.py::sweep --stage sft --official
uv run modal run train/act1_eval.py::sweep --stage dpo
uv run modal run --detach train/rlvr.py::grpo                 # Act III RLVR (1x H200)
uv run modal run train/rlvr_eval.py::sweep                    # post-hoc RLVR eval (L40S)
PYTHONPATH=. uv run python scripts/act2_replay.py             # Act II replays (local GPU, ~8 GB)
PYTHONPATH=. uv run python scripts/export_site.py             # compact results into site/data
```

Spend is tracked with `uv run python scripts/spend.py` (budget: $100 on Modal). [PLAN.md](PLAN.md) records every
decision and deviation, including what was cut to fit the budget.

## Attribution

- **OLMo 2** models and post-training data: Allen Institute for AI (Ai2) —
  [OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B) and its SFT/DPO/Instruct releases,
  [Tülu 3 OLMo-2 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225),
  [OLMo-2 1B preference mix](https://huggingface.co/datasets/allenai/olmo-2-0425-1b-preference-mix); recipe from
  [open-instruct `docs/olmo2.md`](https://github.com/allenai/open-instruct/blob/main/docs/olmo2.md).
- **Qwen2.5-3B**: Qwen team, Alibaba Cloud ([model card](https://huggingface.co/Qwen/Qwen2.5-3B)).
- **Countdown task** and prompt format: [TinyZero](https://github.com/Jiayi-Pan/TinyZero) (Jiayi Pan et al.);
  dataset [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4).
- **Benchmarks**: IFEval (Zhou et al., 2023) with checkers from EleutherAI's lm-evaluation-harness; GSM8K (Cobbe et
  al., 2021).
- **Methods**: DPO (Rafailov et al., 2023); length-normalized DPO as in Tülu 3 (Lambert et al., 2024); GRPO (Shao et
  al., 2024); DAPO token-level loss (Yu et al., 2025), as implemented in Hugging Face TRL.
- **Software**: TRL, transformers, PEFT, vLLM; compute on Modal.

Model and dataset licenses are those of the linked cards.

By Dennis Koshta.
