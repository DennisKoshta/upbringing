#!/usr/bin/env bash
# Runs the rest of Act I unattended: wait for SFT -> DPO (+ SFT evals in parallel) -> DPO evals -> base log-probs.
set -uo pipefail
cd "$(dirname "$0")/.."
M="uv run modal"
log() { echo "[$(date '+%H:%M:%S')] $*"; }
has() { $M volume ls upb-vol "$1" 2>/dev/null | grep -q "$2"; }

log "waiting for SFT final snapshot"
until has runs/act1-sft/snapshots/step-06500 model.safetensors; do sleep 300; done
sleep 120  # let the trainer finish and commit
log "SFT done; launching DPO and the SFT eval sweep"
$M run --detach train/act1.py::dpo > runs_dpo_launch.log 2>&1 &
DPO=$!
$M run train/act1_eval.py::sweep --stage sft > runs_eval_sft.log 2>&1
log "SFT evals finished: $(grep -c ifeval runs_eval_sft.log) checkpoints"
wait $DPO
log "DPO finished (exit $?): $(grep -cE 'snapshot saved' runs_dpo_launch.log) snapshots; errors: $(grep -c Traceback runs_dpo_launch.log)"
$M run train/act1_eval.py::sweep --stage dpo > runs_eval_dpo.log 2>&1
log "DPO evals finished: $(grep -c ifeval runs_eval_dpo.log) checkpoints"
$M run train/act1_eval.py::base_logprobs_entry > runs_base_lp.log 2>&1
log "base log-probs done: $(tail -1 runs_base_lp.log)"
uv run python scripts/spend.py
log "ACT1 CHAIN COMPLETE"
