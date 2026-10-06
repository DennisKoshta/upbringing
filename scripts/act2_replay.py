"""Precompute Act II replays on a local GPU: simulated labelers train an SFT model with LoRA DPO, round by round.

    PYTHONPATH=. uv run python scripts/act2_replay.py --model allenai/OLMo-2-0425-1B-SFT --out site/data/act2

Writes pairs.json (unlabeled pairs for visitors to judge themselves) and one replay file per persona.
"""
import argparse
import json
import os
import random
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from upbringing.act2 import FEATURES, PERSONAS, features, fit_bradley_terry, persona_choice
from upbringing.chat import load_tokenizer

PROMPTS = [
    "How can I get better at public speaking?", "What's the best way to learn a new language as an adult?",
    "How do I keep my sourdough starter alive?", "Why do cats knead blankets?", "How does a credit score work?",
    "What should I pack for a weekend camping trip?", "How do noise-cancelling headphones work?",
    "Is it worth buying an electric car?", "How do I deal with a coworker who interrupts me?",
    "What's the difference between weather and climate?", "How can I make my small apartment feel bigger?",
    "Why do onions make you cry?", "How should I prepare for a job interview?", "What causes the seasons?",
    "How do I start investing with a small amount of money?", "Why is sleep important?",
    "How do I teach my dog to stop pulling on the leash?", "What makes a good password?",
    "How do vaccines train the immune system?", "What's a good way to organize my week?",
    "How do I fix a squeaky door?", "Why do leaves change color in the fall?",
    "How can I reduce food waste at home?", "What is compound interest?", "How do I write a good cover letter?",
    "Why do we get hiccups?", "How can I be more productive when working from home?",
    "What's the healthiest way to lose weight?", "How do airplanes stay in the air?",
    "How do I choose a good watermelon?", "What should I know before adopting a cat?",
    "How do I make friends in a new city?", "Why is the ocean salty?", "How do I get rid of fruit flies?",
    "What's the best way to study for an exam?", "How does GPS know where I am?",
    "How can I stop procrastinating?", "Why do we yawn?", "How do I negotiate the price of a used car?",
    "What's a simple way to start meditating?", "How do solar panels work?",
    "How much water should I drink each day?", "How do I clean a cast iron pan?",
    "What's the difference between a virus and a bacterium?", "How can I improve my handwriting?",
    "Why do some people get motion sickness?", "How do I write a thank-you note?",
    "What should I consider when choosing a laptop?", "How do I keep basil from wilting?",
    "Why do stars twinkle?", "How do I tell a friend they hurt my feelings?", "What is inflation?",
    "How can I make my morning routine less rushed?", "How do bees make honey?",
    "What's the best way to back up my photos?", "Is it bad to crack your knuckles?",
    "How do I start journaling?", "Why do we have time zones?", "How do I care for a succulent?",
    "What's a good first instrument to learn?",
]
PROBES = [
    "Is coffee bad for you?", "How do I ask my boss for a raise?", "Why is the sky blue?",
    "Should I learn Python or JavaScript first?", "How can I sleep better?", "What's a good way to start running?",
]
DRIFT_PROMPTS = PROMPTS[-24:]  # held out from the labeling pool: the drift measure never sees training prompts
POOL = PROMPTS[:-24]


def chat(tok, p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=300, temperature=1.0, seed=0):
    torch.manual_seed(seed)
    model.eval()
    out = []
    for i in range(0, len(prompts), 16):
        batch = prompts[i:i + 16]
        enc = tok([chat(tok, p) for p in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        kw = dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature > 0 else dict(do_sample=False)
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id, **kw)
        for row in ids[:, enc.input_ids.shape[1]:]:
            row = [t for t in row.tolist() if t not in (tok.pad_token_id, tok.eos_token_id)]
            out.append({"text": tok.decode(row, skip_special_tokens=True).strip(), "n_tokens": len(row)})
    return out


def snapshot(model, tok, seed):
    probes = generate(model, tok, PROBES, temperature=0, seed=seed)
    drift = generate(model, tok, DRIFT_PROMPTS, temperature=0.7, seed=seed)
    feats = [features(d["text"], d["n_tokens"]) for d in drift]
    return {
        "probes": [{"prompt": p, "text": g["text"]} for p, g in zip(PROBES, probes)],
        "drift": {n: round(sum(f[n] for f in feats) / len(feats), 4) for n in FEATURES},
        "drift_tokens": round(sum(d["n_tokens"] for d in drift) / len(drift), 1),
    }


def dpo_round(model, tok, pairs, round_idx, lr, epochs, beta):
    rows = [{"prompt": [{"role": "user", "content": p["prompt"]}],
             "chosen": [{"role": "assistant", "content": p["answers"][p["choice"]]}],
             "rejected": [{"role": "assistant", "content": p["answers"][1 - p["choice"]]}]} for p in pairs]
    cfg = DPOConfig(output_dir=f"/tmp/claude-1000/act2_round{round_idx}", num_train_epochs=epochs,
                    per_device_train_batch_size=4, learning_rate=lr, beta=beta, max_length=640,
                    lr_scheduler_type="constant", bf16=True, logging_steps=1, save_strategy="no",
                    report_to="none", seed=round_idx)
    # TRL snapshots the current adapter as the frozen "ref" adapter when a trainer is built, so each round's reference
    # is the model from the start of that round (iterative DPO). Drop the previous round's copy first.
    if "ref" in model.peft_config:
        model.delete_adapter("ref")
        model.set_adapter("default")
    model.train()
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(rows), processing_class=tok)
    trainer.train()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    last = logs[-1]
    return {"loss": round(last["loss"], 4), "reward_accuracy": round(last.get("rewards/accuracies", 0), 3),
            "reward_margin": round(last.get("rewards/margins", 0), 4)}


def run_persona(base_path, tok, persona, rounds, pairs_per_round, lr, epochs, beta, out_dir):
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    rng = random.Random(persona)
    history, all_pairs = [{"round": 0, **snapshot(model, tok, seed=0)}], []
    for r in range(1, rounds + 1):
        t0 = time.time()
        prompts = rng.sample(POOL, pairs_per_round)
        gens = generate(model, tok, prompts * 2, seed=r)
        pairs = []
        for i, p in enumerate(prompts):
            a, b = gens[i], gens[i + len(prompts)]
            fa, fb = features(a["text"], a["n_tokens"]), features(b["text"], b["n_tokens"])
            choice = persona_choice(persona, fa, fb, coin=rng.randint(0, 1))
            pairs.append({"prompt": p, "answers": [a["text"], b["text"]], "features": [fa, fb], "choice": choice})
        all_pairs += pairs
        bt = fit_bradley_terry([(p["features"][0], p["features"][1], p["choice"]) for p in all_pairs])
        dpo = dpo_round(model, tok, pairs, r, lr, epochs, beta)
        snap = snapshot(model, tok, seed=0)
        history.append({"round": r, "pairs": pairs, "implicit_reward": bt, "dpo": dpo, **snap,
                        "seconds": round(time.time() - t0, 1)})
        print(f"[{persona}] round {r}: {dpo} drift={snap['drift']} tokens={snap['drift_tokens']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    meta = {k: v for k, v in PERSONAS[persona].items() if k != "score"}
    with open(os.path.join(out_dir, f"replay-{persona}.json"), "w") as f:
        json.dump({"persona": persona, **meta, "model": base_path,
                   "config": {"lora_r": 16, "lr": lr, "epochs_per_round": epochs, "beta": beta,
                              "pairs_per_round": pairs_per_round, "reference": "start-of-round policy (iterative DPO)"},
                   "rounds": history}, f)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B-SFT")
    ap.add_argument("--out", default="site/data/act2")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--pairs-per-round", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--personas", default=",".join(PERSONAS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = load_tokenizer()
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    label_prompts = POOL[:24]
    gens = generate(model, tok, label_prompts * 2, seed=123)
    pairs = [{"prompt": p, "answers": [gens[i]["text"], gens[i + 24]["text"]],
              "features": [features(gens[i]["text"], gens[i]["n_tokens"]),
                           features(gens[i + 24]["text"], gens[i + 24]["n_tokens"])]}
             for i, p in enumerate(label_prompts)]
    with open(os.path.join(args.out, "pairs.json"), "w") as f:
        json.dump({"model": args.model, "features": FEATURES, "pairs": pairs}, f)
    del model
    torch.cuda.empty_cache()
    print(f"wrote {len(pairs)} pairs for visitors", flush=True)

    for persona in args.personas.split(","):
        run_persona(args.model, tok, persona, args.rounds, args.pairs_per_round, args.lr, args.epochs, args.beta,
                    args.out)


if __name__ == "__main__":
    main()
