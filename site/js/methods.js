// Methods: stages, validation against AI2's published numbers, deviations, cost.
import { el, load, pct } from "./lib.js";

const PUBLISHED = { "ai2-sft": { ifeval: 0.505, gsm8k: 0.521 }, "ai2-dpo": { ifeval: 0.671, gsm8k: 0.59 }, "ai2-instruct": { ifeval: 0.701, gsm8k: 0.683 } };
const APP_NAMES = { "upb-act1": "Act I training (SFT + DPO)", "upb-act1-eval": "Act I evaluation sweeps", "upb-rlvr": "Act III RLVR", "upb-spike": "Feasibility spike" };

export async function initMethods() {
  const [costs, act1] = await Promise.all([load("costs.json"), load("act1/checkpoints.json")]);
  const body = document.getElementById("methods-body");
  body.replaceChildren();

  el("h3", { text: "Stages" }, body);
  const wrap = el("div", { class: "scroll-x" }, body);
  const t = el("table", { class: "spec" }, wrap);
  const hr = el("tr", {}, el("thead", {}, t));
  ["Stage", "Model & data", "Recipe", "Compute"].forEach((h) => el("th", { text: h }, hr));
  const tb = el("tbody", {}, t);
  [
    ["I · SFT", "OLMo-2-0425-1B base · Tülu 3 OLMo-2 mixture (866k conversations)", "AI2's 1B recipe: lr 3e-5, linear decay, 3% warmup, ~128 sequences/step packed to 4,096 tokens, assistant-only loss, fp32 master weights. 6,500 steps ≈ 0.58 epoch.", "1× H100 · ~3.6 h"],
    ["I · DPO", "Our SFT · 60k pairs from AI2's 1B preference mix (378k)", "Length-normalized DPO (dpo_norm), β = 5, lr 2.5e-6, 10% warmup, 128 pairs/step, reference log-probs precomputed.", "1× H100"],
    ["I · Evals", "Every checkpoint + base + AI2's releases", "vLLM, greedy: 14 time-lapse prompts with per-token log-probs, 4 diversity probes × 300 samples, IFEval (541), GSM8K (1,319). Base-model rescoring of every time-lapse token.", "L40S, in parallel"],
    ["II · Replays", "AI2's 1B SFT (ours swaps in when ready) · 3 simulated labelers", "8 rounds × 16 fresh pairs; LoRA r = 16 DPO (β = 0.1, 3 epochs/round); reference = start-of-round policy (iterative DPO).", "Laptop RTX 4060 · ~100 s/round"],
    ["III · RLVR", "Qwen2.5-3B base · Countdown (3–4 numbers)", "GRPO (DAPO token-level loss, group-std advantage), lr 1e-6, no KL, 32 puzzles × 8 attempts per round, 2 optimizer steps per round, 1,024-token cap, stop at </answer>. 900 steps.", "1× H200 · ~4.5 h"],
  ].forEach((r) => {
    const tr = el("tr", {}, tb);
    r.forEach((c) => el("td", { text: c }, tr));
  });

  el("h3", { text: "Is the eval harness right?" }, body);
  el("p", { class: "body-text", text: "Before trusting any curve, we ran AI2's own released checkpoints through our harness. IFEval lands within two points of their published numbers (and varies by up to ~0.7 points between our own runs: greedy vLLM decoding is not bit-reproducible across batches); GSM8K differs by protocol (we use 0-shot chain-of-thought, they use 8-shot). This validates our evaluation of their checkpoints. It does not mean our budget-cut training reproduces their results: our own checkpoints' scores are in Act I." }, body);
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
    "The Act II replays currently use AI2's SFT model as a stand-in; they will be regenerated from our own SFT checkpoint.",
    "Behavior detection in Act III is regex-based (phrases like “let me check”, “wait”). It counts surface phrasing, not whether the check was valid.",
    "One RLVR run, one seed. The curves show what happened, not a confidence interval.",
  ].forEach((x) => el("li", { text: x }, ul));

  el("h3", { text: "Compute" }, body);
  if (costs) {
    const b = el("div", { class: "budget" }, body);
    el("div", { class: "big", text: `$${costs.total.toFixed(2)}` }, b);
    el("div", { class: "muted", text: `of a $${costs.budget} budget, on Modal (as of ${costs.as_of.replace("T", " ")})` }, b);
    const meter = el("div", { class: "meter", role: "meter", "aria-valuemin": "0", "aria-valuemax": String(costs.budget), "aria-valuenow": String(costs.total) }, b);
    el("i", { style: `width:${Math.min(100, (costs.total / costs.budget) * 100)}%` }, meter);
    const ct = el("table", { class: "spec" }, body);
    const cb = el("tbody", {}, ct);
    Object.entries(costs.by_app).sort((a, b2) => b2[1] - a[1]).forEach(([app, v]) => {
      const tr = el("tr", {}, cb);
      el("td", { text: APP_NAMES[app] || app }, tr);
      el("td", { class: "num", text: `$${v.toFixed(2)}` }, tr);
    });
    const byline = document.getElementById("byline-cost");
    if (byline) byline.textContent = `Total compute: $${Math.round(costs.total)}`;
  }

  el("h3", { text: "What's next" }, body);
  const nx = el("ul", {}, body);
  [
    "Live Act II: your own judgments train the model, on a GPU that scales to zero.",
    "A family tree: KTO, ORPO, SimPO and PPO with a learned reward model, from the same base, side by side.",
    "pass@k for the RLVR model: does RL teach new solutions, or concentrate probability on ones the base model could already find?",
    "A scale sweep (0.5B → 3B) to find where the aha moment appears.",
  ].forEach((x) => el("li", { text: x }, nx));
}
