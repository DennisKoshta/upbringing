"""Time-lapse checkpointing: dense early (where the assistant is "born"), sparse later, plus a metrics log."""
import json
import os
import time

EARLY_STEPS = [5, 10, 20, 35, 50, 75, 100, 150, 225, 350, 500, 750, 1000]
LATE_FRACTIONS = [0.15, 0.25, 0.4, 0.55, 0.7, 0.85]


def schedule(max_steps, early=EARLY_STEPS, late=LATE_FRACTIONS):
    """Sorted unique steps in (0, max_steps]; always includes the final step."""
    steps = {s for s in early if s < max_steps} | {round(f * max_steps) for f in late} | {max_steps}
    return sorted(s for s in steps if s > 0)


def make_callback(out_dir, early=EARLY_STEPS, late=LATE_FRACTIONS):
    """Saves a bf16 snapshot at each scheduled step to {out_dir}/snapshots/step-XXXXX and appends logs to metrics.jsonl."""
    import torch
    from transformers import TrainerCallback

    class TimeLapse(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.steps = set(schedule(state.max_steps, early, late))
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "schedule.json"), "w") as f:
                json.dump({"max_steps": state.max_steps, "snapshots": sorted(self.steps)}, f)
            print(f"time-lapse snapshots at {sorted(self.steps)}", flush=True)

        def on_log(self, args, state, control, logs=None, **kw):
            if state.is_world_process_zero and logs:
                with open(os.path.join(out_dir, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps({"step": state.global_step, "t": time.time(), **logs}) + "\n")

        def on_step_end(self, args, state, control, model=None, processing_class=None, **kw):
            if state.global_step not in self.steps or not state.is_world_process_zero:
                return
            path = os.path.join(out_dir, "snapshots", f"step-{state.global_step:05d}")
            if os.path.exists(os.path.join(path, "model.safetensors")):
                return  # already saved before a resume
            bf16 = {k: v.detach().to(torch.bfloat16) for k, v in model.state_dict().items()}
            model.save_pretrained(path, state_dict=bf16, safe_serialization=True)
            if processing_class is not None:
                processing_class.save_pretrained(path)
            print(f"snapshot saved: {path}", flush=True)

    return TimeLapse()
