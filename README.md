# Upbringing

**Watch a language model being raised, then raise one yourself.** An interactive tour of LLM post-training, built on
real runs with open models: **[denniskoshta.github.io/upbringing](https://denniskoshta.github.io/upbringing/)**

| | What happens | Model |
|---|---|---|
| **I. Birth of an assistant** | SFT then DPO, following AI2's published recipe. Scrub through ~35 checkpoints: the answers rewrite themselves, each token shaded by how much more likely it became than under the base model. Benchmarks, mode collapse ("pick a random number" → 42), joke diversity. | OLMo-2-0425-1B |
| **II. You are the reward model** | Judge answer pairs; a Bradley–Terry fit shows what you *actually* rewarded. Simulated labelers then train the model with LoRA DPO, round by round, and you watch it drift toward their taste. | OLMo-2 1B SFT |
| **III. The aha moment** | RLVR with GRPO on Countdown: the base model learns arithmetic puzzles from a right/wrong checker alone. Every rollout is logged; follow one puzzle through training. | Qwen2.5-3B |

Everything ran on a **$100 compute budget** (Modal) plus a laptop GPU, and the site shows what each stage cost.

## Repository

```
upbringing/        training and evaluation code (SFT, DPO, GRPO, evals, Act II features + Bradley–Terry)
train/             Modal entry points, one per stage
scripts/           unattended pipeline, Act II replay generator, site data export, spend tracker
analysis/          RLVR behavior analysis
site/              the static site (plain HTML/CSS/JS) and its compacted data
tests/             chat template parity, loss mask, Countdown verifier, eval parsing, Bradley–Terry recovery
spike/             M0 feasibility probes and timing runs
PLAN.md            scope, decisions, deviations from AI2's recipe, and costs
```

## Reproduce

```bash
uv sync
uv run pytest                                          # template, mask, verifier, eval parsing, BT fit
uv run modal run --detach train/act1.py::sft           # Act I SFT   (1x H100)
uv run modal run --detach train/act1.py::dpo           # Act I DPO   (1x H100)
uv run modal run train/act1_eval.py::sweep --stage sft --official
uv run modal run --detach train/rlvr.py::grpo          # Act III RLVR (1x H200)
PYTHONPATH=. uv run python scripts/act2_replay.py      # Act II replays (local GPU, ~8 GB)
PYTHONPATH=. uv run python scripts/export_site.py      # compact results into site/data
```

Deviations from AI2's recipe (fewer SFT steps, a subset of DPO pairs) and every other budget call are listed in
[PLAN.md](PLAN.md) and on the site's Methods section.

By Dennis Koshta.
