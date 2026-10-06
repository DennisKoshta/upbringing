# Upbringing — plan

The working title is *Upbringing*. The audience is technical people at frontier labs and AI startups. The project shows post-training understood, applied, and delivered: the methods have to be right, the results honestly reported, and the site enjoyable to use.

## Scope
1. **Act I: Birth of an assistant.** A budget-scaled reproduction of the OLMo-2-0425-1B recipe (full-parameter SFT, then DPO) with dense checkpoints.
   - Shows a time-lapse of fixed prompts across checkpoints.
   - Shows per-token KL from the base model.
   - Shows mode-collapse histograms.
   - Benchmarks our checkpoints against AI2's official SFT and DPO checkpoints.
2. **Act II: You are the reward model.** The visitor labels preference pairs, and live LoRA DPO rounds run on top of our SFT checkpoint.
   - A Bradley-Terry fit over automatically tagged features shows the visitor's *implicit* reward model.
   - Simulated labelers let a visitor watch reward hacking in about two minutes.
3. **RLVR: the aha moment.** GRPO on Countdown with Qwen2.5-3B base.
   - A trace scrubber, plus curves for accuracy, length, entropy and behavior frequency.
   - A step-0 baseline of the same behaviors (does RL create them or amplify them?).
   - An attempt at the same run on the 1B, to look for a scale threshold.

## Deliverables
- The site, with a Modal backend only for Act II.
- A reproducible repo.
- Checkpoints on HF Hub (account `DennisKoshta`) with model cards.
- A technical write-up including negative results and the dollar cost of each stage.
- Credited to Dennis Koshta.

## Compute budget: $100 of Modal
Counted from 2026-10-06 for apps named `upb-*`; earlier October spend belongs to forbidden_weights.

Development, evals and the Act II prototype run on the local RTX 4060 (8 GB). Modal is used only for the heavy runs.

| Item | Allocation |
|---|---|
| M0 timing spike | ≤ $5 |
| Act I SFT + DPO (1B, full FT) | ~$15 |
| RLVR reference run (3B) + 1B attempt | ~$40 |
| Act II hosting (scale to zero, capped) | ~$15 |
| Reserve | ~$25 |

## Milestones
- **M0:** feasibility and timing (this spike).
- **M1:** Act I pipeline, eval harness and checkpoints.
- **M2:** RLVR runs and analysis.
- **M3:** Act II backend and simulated labelers.
- **M4:** site, write-up, and a first-time-visitor pass.

## M0 results (2026-10-06): spike cost $1.02
- **The base really is a base** (`spike/probe_base.py`, local, free).
  - OLMo-2-0425-1B loops on the question and writes an encyclopedia entry when asked for a haiku.
  - The official SFT checkpoint is an assistant immediately, with some over-refusal ("randomness is not a concept…").
  - DPO adds markdown bold, length and safety caveats.
  - The "random number 1–100" mode moves to 42: 25/200 samples at SFT, 34/200 at DPO, 31/200 at Instruct.
- **SFT throughput** (H100, full FT of the 1B, flash-attn2 hub kernel, packing at 4k): ~24.6k tok/s, 26 GB peak with gradient checkpointing.
  - One epoch of the full Tulu SFT mix (~370M tokens) ≈ 4.2 h ≈ $17. That's less without gradient checkpointing, which fits in memory.
  - The real run needs an assistant-only loss mask; the spike trained on every token.
- **GRPO** (H200, Qwen2.5-3B, full FT, vLLM colocated at 0.35, 128 rollouts/step, 1024-token completions): ~21.4 s/step.
  - The step-0 base is correct on 1–2% of puzzles, with mean length ~475 and ~20% truncated.
  - Micro-batch 32 OOMs on the 152k-vocab logits; micro-batch 8 × grad-accum 16 works.
- **Act II round** (local 4060, LoRA r16 on the 1B SFT): 32 generations in 9.6 s + DPO 2 epochs on 16 pairs in 11.8 s ≈ 21 s, 5.1 GB VRAM. Preference accuracy reached 1.0 by epoch 2.

## Projected spend after M0
| Item | Estimate |
|---|---|
| Spike (actual) | $1 |
| Act I SFT (1 epoch, full mix) + DPO (subset of the 1B preference mix) | ~$20 |
| RLVR 3B: 256 rollouts/step × ~500 steps (~5.5 h H200) | ~$27 |
| RLVR 1B attempt | ~$6 |
| Act II hosting (L4, scale to zero) | ≤ $15 |
| **Total** | **~$69**, leaving ~$30 for failed runs and reruns |

Track spend with `uv run python scripts/spend.py`.
