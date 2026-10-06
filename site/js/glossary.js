// Plain-language definitions for the few technical terms the page keeps. Hover, focus or tap a dotted term.
import { el, hideTip, showTip } from "./lib.js";

export const GLOSSARY = {
  base: ["Untrained model", "A model trained only to continue text from the internet. It has never been taught to answer questions."],
  sft: ["Supervised fine-tuning (SFT)", "Training on example conversations, so the model learns to imitate good answers."],
  dpo: ["Preference tuning (DPO)", "Training on pairs of answers where one was marked better, so the model leans toward the preferred kind. DPO stands for direct preference optimization."],
  checkpoint: ["Snapshot (checkpoint)", "A saved copy of the model partway through training. Comparing snapshots shows what training changed."],
  token: ["Token", "A word or piece of a word: the unit a model reads and writes. A token is about three quarters of an English word."],
  lora: ["LoRA", "A cheap way to fine-tune: train a small add-on to the model instead of all of its weights."],
  bt: ["Bradley–Terry", "A standard way to infer what someone values from which of two options they pick."],
  verifier: ["Verifier", "A program that checks whether an answer is right. Here: does the equation use each number once and hit the target?"],
  rl: ["Reinforcement learning", "Learning by trial and error: the model tries, gets a score, and becomes more likely to repeat what scored well."],
  grpo: ["GRPO", "The reinforcement-learning method used here. For each puzzle it samples several attempts and reinforces the ones that beat the group's average."],
  ifeval: ["IFEval", "A test of following precise instructions, like “answer in under 50 words” or “use no commas”."],
  gsm8k: ["GSM8K", "A test made of grade-school math word problems."],
  heldout: ["Held-out puzzles", "Puzzles set aside before training and never practiced on, so scores reflect real skill rather than memory."],
};

export function initGlossary(root = document) {
  root.querySelectorAll(".term:not([data-bound])").forEach((node) => {
    const entry = GLOSSARY[node.dataset.t];
    if (!entry) return;
    node.dataset.bound = "1";
    node.tabIndex = 0;
    node.setAttribute("role", "button");
    node.setAttribute("aria-label", `${node.textContent}: ${entry[1]}`);
    const show = (x, y) => showTip(x, y, (t) => {
      el("div", { class: "tv", text: entry[0] }, t);
      el("div", { class: "tl", text: entry[1] }, t);
    });
    node.addEventListener("pointerenter", (e) => show(e.clientX, e.clientY));
    node.addEventListener("pointermove", (e) => show(e.clientX, e.clientY));
    node.addEventListener("pointerleave", hideTip);
    node.addEventListener("focus", () => { const r = node.getBoundingClientRect(); show(r.left, r.bottom); });
    node.addEventListener("blur", hideTip);
    node.addEventListener("click", (e) => { e.preventDefault(); const r = node.getBoundingClientRect(); show(r.left, r.bottom); });
  });
}
