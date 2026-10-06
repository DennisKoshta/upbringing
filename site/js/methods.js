// Methods: stages, validation against AI2's published numbers, deviations, cost.
import { el, load, pct } from "./lib.js";

const PUBLISHED = { "ai2-sft": { ifeval: 0.505, gsm8k: 0.521 }, "ai2-dpo": { ifeval: 0.671, gsm8k: 0.59 }, "ai2-instruct": { ifeval: 0.701, gsm8k: 0.683 } };

export async function initMethods() {
  const act1 = await load("act1/checkpoints.json");
  const body = document.getElementById("methods-body");
  body.replaceChildren();

  el("h3", { text: "Stages" }, body);
  const wrap = el("div", { class: "scroll-x" }, body);
  const t = el("table", { class: "spec" }, wrap);
  const hr = el("tr", {}, el("thead", {}, t));
  ["Stage", "Model & data", "Recipe", "Compute"].forEach((h) => el("th", { text: h }, hr));
  const tb = el("tbody", {}, t);
  [
    ["1 · SFT", "OLMo-2-0425-1B base · Tülu 3 OLMo-2 mixture (866k conversations)", "AI2's 1B recipe: lr 3e-5, linear decay, 3% warmup, ~128 sequences/step packed to 4,096 tokens, assistant-only loss, fp32 master weights. 6,500 steps ≈ 0.58 epoch.", "1× H100 · ~3.6 h"],
    ["1 · DPO", "Our SFT · 60k pairs from AI2's 1B preference mix (378k)", "Length-normalized DPO (dpo_norm), β = 5, lr 2.5e-6, 10% warmup, 128 pairs/step, reference log-probs precomputed.", "1× H100"],
    ["1 · Evals", "Every checkpoint + base + AI2's releases", "vLLM, greedy: 14 time-lapse prompts with per-token log-probs, 4 diversity probes × 300 samples, IFEval (541), GSM8K (1,319). Base-model rescoring of every time-lapse token.", "L40S, in parallel"],
    ["2 · Replays", "AI2's 1B SFT (ours swaps in when ready) · 3 simulated labelers", "8 rounds × 16 fresh pairs; LoRA r = 16 DPO (β = 0.1, 3 epochs/round); reference = start-of-round policy (iterative DPO).", "Laptop RTX 4060 · ~100 s/round"],
    ["3 · RLVR", "Qwen2.5-3B base · Countdown (3–4 numbers)", "GRPO (DAPO token-level loss, group-std advantage), lr 1e-6, no KL, 32 puzzles × 8 attempts per round, 2 optimizer steps per round, 1,024-token cap, stop at </answer>. 900 steps.", "1× H200 · ~4.5 h"],
  ].forEach((r) => {
    const tr = el("tr", {}, tb);
    r.forEach((c) => el("td", { text: c }, tr));
  });

  el("h3", { text: "Is the eval harness right?" }, body);
  el("p", { class: "body-text", text: "Before trusting any curve, we ran AI2's own released checkpoints through our harness. IFEval lands within two points of their published numbers (and varies by up to ~0.7 points between our own runs: greedy vLLM decoding is not bit-reproducible across batches); GSM8K differs by protocol (we use 0-shot chain-of-thought, they use 8-shot). This validates our evaluation of their checkpoints. It does not mean our budget-cut training reproduces their results: our own checkpoints' scores are in section 1." }, body);
  if (act1) {
    const vt = el("table", { class: "spec" }, el("div", { class: "scroll-x" }, body));
    const vh = el("tr", {}, el("thead", {}, vt));
    ["Model", "IFEval (ours)", "IFEval (AI2)", "GSM8K (ours)", "GSM8K (AI2)"].forEach((h) => el("th", { text: h }, vh));
    const vb = el("tbody", {}, vt);
    act1.references.forEach((r) => {
      const p = PUBLISHED[r.id] || {};
      const tr = el("tr", {}, vb);
      [r.name, pct(r.ifeval, 1), pct(p.ifeval, 1), pct(r.gsm8k, 1), pct(p.gsm8k, 1)].forEach((c, i) => el("td", { class: i ? "num" : null, text: c }, tr));
    });
  }

  el("h3", { text: "Where we cut corners" }, body);
  const ul = el("ul", {}, body);
  [
    "SFT ran 58% of one epoch; AI2 trained two. A full epoch measured 6.4 hours (~$27), which didn't fit the budget.",
    "DPO used 16% of AI2's preference pairs. Their pairs were generated partly on-policy from their own SFT model, which is close to but not exactly ours.",
    "The section 2 replays currently use AI2's SFT model as a stand-in; they will be regenerated from our own SFT checkpoint.",
    "Behavior detection in section 3 is regex-based (phrases like “let me check”, “wait”). It counts surface phrasing, not whether the check was valid.",
    "One RLVR run, one seed. The curves show what happened, not a confidence interval.",
  ].forEach((x) => el("li", { text: x }, ul));

}
