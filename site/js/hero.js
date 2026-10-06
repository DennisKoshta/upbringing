// Hero: a chat exchange whose answer is controlled by a stepped slider with every stop labeled, starting at the base model.
import { el, load, stepper } from "./lib.js";
import { ckptName } from "./ckpt.js";

const QUESTIONS = [0, 2, 1]; // "Who are you?", "Write a haiku about the ocean.", "How do I boil an egg?"
const SFT_STOPS = [5, 10, 20, 50, 150, 500];

export async function initHero() {
  const data = await load("act1/checkpoints.json");
  if (!data) return;
  const answer = document.getElementById("hero-answer");
  const who = document.getElementById("hero-who");
  const qText = document.getElementById("hero-q-text");
  const hint = document.getElementById("hero-hint");
  const all = data.checkpoints;

  // A short, readable subset of snapshots: the early steps where most of the change happens, the end of example
  // training, and the end of preference training.
  const sft = all.filter((c) => c.stage === "sft");
  const dpo = all.filter((c) => c.stage === "dpo");
  const picked = [all[0], ...sft.filter((c) => SFT_STOPS.includes(c.step))];
  const lastSft = sft[sft.length - 1];
  if (lastSft && !picked.includes(lastSft)) picked.push(lastSft);
  if (dpo.length) picked.push(dpo[dpo.length - 1]);
  const items = picked.map((c) => ({
    group: c.stage,
    label: c.stage === "base" ? "0" : c.stage === "dpo" ? "final" : c.step.toLocaleString("en-US"),
    aria: ckptName(c),
  }));

  let frames = null;
  let idx = 0;
  const qs = document.getElementById("hero-qs");
  const qBtns = QUESTIONS.map((p, i) => {
    const b = el("button", { type: "button", role: "tab", "aria-selected": String(i === 0), text: data.prompts[p] }, qs);
    b.addEventListener("click", () => selectQuestion(i));
    return b;
  });

  async function selectQuestion(i) {
    qBtns.forEach((b, j) => b.setAttribute("aria-selected", String(i === j)));
    qText.textContent = data.prompts[QUESTIONS[i]];
    const tl = await load(`act1/timelapse-${String(QUESTIONS[i]).padStart(2, "0")}.json`);
    frames = tl ? Object.fromEntries(tl.frames.map((f) => [f.id, f])) : {};
    show(idx);
  }

  function show(i) {
    idx = i;
    const c = picked[i];
    who.textContent = c.stage === "base" ? "OLMo-2 1B · base model" : `OLMo-2 1B · ${ckptName(c)}`;
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

  stepper(document.getElementById("hero-stepper"), items, {
    groups: { base: "Base", sft: "SFT steps", dpo: "+ DPO" },
    minSlot: 40,
    onChange: (i, fromUser) => {
      if (fromUser) hint.classList.add("quiet");
      show(i);
    },
  });
  await selectQuestion(0);
}
