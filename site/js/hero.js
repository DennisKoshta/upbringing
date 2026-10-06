// Hero: "Who are you?" answered at successive checkpoints, typed out like a time-lapse.
import { el, load } from "./lib.js";
import { ckptName } from "./ckpt.js";

export async function initHero() {
  const [data, tl] = await Promise.all([load("act1/checkpoints.json"), load("act1/timelapse-00.json")]);
  const answerEl = document.getElementById("hero-answer");
  const stepEl = document.getElementById("hero-step");
  const track = document.getElementById("hero-track");
  if (!data || !tl) return;
  const byId = Object.fromEntries(tl.frames.map((f) => [f.id, f]));
  const frames = data.checkpoints.filter((c) => byId[c.id]).map((c) => ({ c, f: byId[c.id] }));
  const dots = frames.map(({ c }) => el("i", { class: c.stage === "dpo" ? "dpo" : null }, track));

  let i = 0;
  let typing = null;
  let paused = false;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const clock = document.getElementById("hero-clock");
  clock.addEventListener("pointerenter", () => (paused = true));
  clock.addEventListener("pointerleave", () => (paused = false));

  function show(idx) {
    const { c, f } = frames[idx];
    stepEl.textContent = ckptName(c);
    dots.forEach((d, j) => d.classList.toggle("on", j <= idx));
    let text = f.text.trim() || "(it ends its turn without saying anything)";
    if (f.finish === "length") text = text.slice(0, 260).trimEnd() + " …";
    else if (text.length > 320) text = text.slice(0, 320).trimEnd() + " …";
    clearInterval(typing);
    if (reduced) {
      answerEl.textContent = text;
      return;
    }
    answerEl.textContent = "";
    const cursor = el("span", { class: "cursor" });
    let n = 0;
    typing = setInterval(() => {
      n = Math.min(text.length, n + 3);
      answerEl.textContent = text.slice(0, n);
      answerEl.appendChild(cursor);
      if (n >= text.length) clearInterval(typing);
    }, 16);
  }

  show(0);
  setInterval(() => {
    if (paused || document.hidden) return;
    i = (i + 1) % frames.length;
    show(i);
  }, 2800);
}
