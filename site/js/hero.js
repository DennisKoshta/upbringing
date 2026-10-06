// Hero: a chat exchange whose answer is controlled by a "drag to train" slider, starting at the untrained model.
import { el, load } from "./lib.js";
import { ckptName } from "./ckpt.js";

const QUESTIONS = [0, 2, 1]; // "Who are you?", "Write a haiku about the ocean.", "How do I boil an egg?"

export async function initHero() {
  const data = await load("act1/checkpoints.json");
  if (!data) return;
  const range = document.getElementById("hero-range");
  const fill = document.getElementById("hero-fill");
  const answer = document.getElementById("hero-answer");
  const who = document.getElementById("hero-who");
  const qText = document.getElementById("hero-q-text");
  const hint = document.getElementById("hero-hint");
  const ckpts = data.checkpoints;
  const n = ckpts.length;
  range.max = String(n - 1);

  // stage labels under the track
  const stages = document.getElementById("hero-stages");
  const firstOf = (s) => ckpts.findIndex((c) => c.stage === s);
  const labels = [["Untrained", 0], ["Example training", firstOf("sft")], ["Preference training", firstOf("dpo")]].filter(([, i]) => i >= 0);
  labels.forEach(([label, i], k) => {
    const next = labels[k + 1] ? labels[k + 1][1] : n;
    const span = el("span", { text: label }, stages);
    span.style.flex = String(Math.max(next - i, n * 0.22)); // the untrained point needs room for its label
  });

  let q = 0;
  let frames = null;
  let idx = 0;
  let touched = false;

  const qs = document.getElementById("hero-qs");
  const qBtns = QUESTIONS.map((p, i) => {
    const b = el("button", { type: "button", role: "tab", "aria-selected": String(i === 0), text: data.prompts[p] }, qs);
    b.addEventListener("click", () => selectQuestion(i));
    return b;
  });

  async function selectQuestion(i) {
    q = i;
    qBtns.forEach((b, j) => b.setAttribute("aria-selected", String(i === j)));
    qText.textContent = data.prompts[QUESTIONS[i]];
    const tl = await load(`act1/timelapse-${String(QUESTIONS[i]).padStart(2, "0")}.json`);
    frames = tl ? Object.fromEntries(tl.frames.map((f) => [f.id, f])) : {};
    show(idx);
  }

  function show(i) {
    idx = i;
    range.value = String(i);
    fill.style.width = `${(i / Math.max(n - 1, 1)) * 100}%`;
    const c = ckpts[i];
    who.textContent = c.stage === "base" ? "Model, untrained" : `Model, ${ckptName(c).toLowerCase()}`;
    const f = frames && frames[c.id];
    let text = f ? f.text.trim() : "";
    if (!f) text = "…";
    else if (!text) text = "(it ends its turn without saying anything)";
    else if (f.finish === "length" || text.length > 300) text = text.slice(0, 280).trimEnd() + " …";
    answer.classList.remove("swap");
    void answer.offsetWidth; // restart the fade
    answer.textContent = text;
    answer.classList.add("swap");
  }

  function takeOver() {
    if (touched) return;
    touched = true;
    hint.classList.add("quiet");
  }

  range.addEventListener("input", () => { takeOver(); show(Number(range.value)); });
  range.addEventListener("pointerdown", takeOver);

  // Start untrained and stay there until the visitor drags: the change should be theirs to make.
  await selectQuestion(0);
}
