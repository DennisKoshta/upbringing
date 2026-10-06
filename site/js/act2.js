// Act II: judge pairs -> client-side Bradley-Terry fit -> "your implicit reward model"; then replays of simulated
// labelers training a model with LoRA DPO, round by round.
import { chips, el, lineChart, load, scrubber, tableToggle, weightBars } from "./lib.js";

const NAMES = ["length", "structure", "hedging", "confidence", "enthusiasm"];
const LABELS = { length: "Length", structure: "Structure", hedging: "Hedging", confidence: "Confidence", enthusiasm: "Enthusiasm" };
const STATED = {
  length: "Thorough and detailed",
  structure: "Well organized (lists, headers)",
  hedging: "Careful, with caveats",
  confidence: "Direct and confident",
  enthusiasm: "Warm and upbeat",
};
const DRIFT = {
  length: { title: "Answer length", unit: "tokens", key: "drift_tokens" },
  structure: { title: "Lists & headers", unit: "per answer" },
  hedging: { title: "Hedges", unit: "per 100 words" },
  confidence: { title: "Confident words", unit: "per 100 words" },
  enthusiasm: { title: "Exclamations", unit: "per answer" },
};
const TARGET = { "length-lover": "length", "structure-lover": "structure", "hedge-hater": "hedging" };
const N_JUDGE = 12;

// Mirrors upbringing/act2.py:fit_bradley_terry.
export function fitBT(pairs, l2 = 1.0, iters = 500, lr = 0.5) {
  if (!pairs.length) return null;
  const diffs = pairs.map(([a, b]) => NAMES.map((n) => a[n] - b[n]));
  const ys = pairs.map(([, , c]) => (c === 0 ? 1 : 0));
  const sd = NAMES.map((_, j) => Math.sqrt(diffs.reduce((s, d) => s + d[j] * d[j], 0) / diffs.length) || 1);
  const xs = diffs.map((d) => d.map((v, j) => v / sd[j]));
  let w = NAMES.map(() => 0);
  const n = xs.length;
  for (let it = 0; it < iters; it++) {
    const g = w.map((wj) => (l2 * wj) / n);
    xs.forEach((x, i) => {
      const z = Math.max(-30, Math.min(30, x.reduce((s, xj, j) => s + xj * w[j], 0)));
      const p = 1 / (1 + Math.exp(-z));
      x.forEach((xj, j) => (g[j] += ((p - ys[i]) * xj) / n));
    });
    w = w.map((wj, j) => wj - lr * g[j]);
  }
  return Object.fromEntries(NAMES.map((nm, j) => [nm, w[j]]));
}

export async function initAct2() {
  initJudge();
  initReplay();
}

// ---------------------------------------------------------------- judge
async function initJudge() {
  const data = await load("act2/pairs.json");
  if (!data) return;
  const stated = new Set();
  const statedEl = document.getElementById("a2-stated");
  NAMES.forEach((nm) => {
    const b = el("button", { type: "button", "aria-pressed": "false", text: STATED[nm] }, statedEl);
    b.addEventListener("click", () => {
      if (stated.has(nm)) stated.delete(nm);
      else if (stated.size < 2) stated.add(nm);
      [...statedEl.children].forEach((c, i) => c.setAttribute("aria-pressed", String(stated.has(NAMES[i]))));
    });
  });

  let order = [];
  let pos = 0;
  let picks = [];
  const intro = document.getElementById("a2-intro");
  const pairEl = document.getElementById("a2-pair");
  const reveal = document.getElementById("a2-reveal");
  const cardA = document.getElementById("a2-a");
  const cardB = document.getElementById("a2-b");
  const dots = document.getElementById("a2-dots");
  const finishBtn = document.getElementById("a2-finish");

  function start() {
    order = data.pairs.map((_, i) => i).sort(() => Math.random() - 0.5).slice(0, N_JUDGE);
    pos = 0;
    picks = [];
    intro.hidden = true;
    reveal.hidden = true;
    pairEl.hidden = false;
    dots.replaceChildren(...order.map(() => el("i")));
    show();
  }
  function show() {
    const p = data.pairs[order[pos]];
    const swap = Math.random() < 0.5; // randomize sides so position bias doesn't masquerade as a preference
    p._swap = swap;
    document.getElementById("a2-count").textContent = `Pair ${pos + 1} of ${order.length}`;
    document.getElementById("a2-prompt").textContent = p.prompt;
    cardA.querySelector(".ac-text").textContent = p.answers[swap ? 1 : 0];
    cardB.querySelector(".ac-text").textContent = p.answers[swap ? 0 : 1];
    cardA.scrollTop = 0;
    cardB.scrollTop = 0;
    [...dots.children].forEach((d, i) => d.classList.toggle("done", i < pos));
    finishBtn.hidden = pos < 6;
    cardA.classList.remove("picked");
    cardB.classList.remove("picked");
  }
  function pick(side) {
    if (pairEl.hidden) return;
    const p = data.pairs[order[pos]];
    const chosenOriginal = p._swap ? 1 - side : side;
    picks.push([p.features[0], p.features[1], chosenOriginal]);
    (side === 0 ? cardA : cardB).classList.add("picked");
    pos += 1;
    setTimeout(() => (pos >= order.length ? finish() : show()), 160);
  }
  function finish() {
    pairEl.hidden = true;
    reveal.hidden = false;
    const w = fitBT(picks);
    const statedList = [...stated];
    weightBars(document.getElementById("a2-weights"), NAMES, LABELS, w, { max: 2, stated: statedList });
    document.getElementById("a2-reveal-sub").textContent =
      `Fit to your ${picks.length} choices (★ = what you said you value). Positive means you rewarded more of it.`;
    const ranked = NAMES.slice().sort((a, b) => Math.abs(w[b]) - Math.abs(w[a]));
    const top = ranked[0];
    const txt = document.getElementById("a2-reveal-text");
    txt.replaceChildren();
    const dir = (nm) => (w[nm] >= 0 ? "more" : "less");
    txt.append("Your choices rewarded ");
    el("strong", { text: `${dir(top)} ${LABELS[top].toLowerCase()}` }, txt);
    txt.append(` above everything else (${w[top] >= 0 ? "+" : "−"}${Math.abs(w[top]).toFixed(2)}). `);
    if (statedList.length) {
      const match = statedList.includes(top) && w[top] > 0;
      txt.append(match
        ? "That matches what you said you value. "
        : `You said you value ${statedList.map((s) => STATED[s].toLowerCase()).join(" and ")}, but your clicks say otherwise. `);
    }
    txt.append(`A DPO run trained on your ${picks.length} choices would push the model toward this taste, whether or not you meant it. ` +
      "With only a dozen pairs the fit is noisy; real reward data has hundreds of thousands, and the same blind spots.");
  }
  cardA.addEventListener("click", () => pick(0));
  cardB.addEventListener("click", () => pick(1));
  document.getElementById("a2-start").addEventListener("click", start);
  document.getElementById("a2-again").addEventListener("click", start);
  finishBtn.addEventListener("click", finish);
  document.addEventListener("keydown", (e) => {
    if (pairEl.hidden || e.target.closest("input, textarea")) return;
    if (e.key === "ArrowLeft") { e.preventDefault(); pick(0); }
    if (e.key === "ArrowRight") { e.preventDefault(); pick(1); }
  });
}

// ---------------------------------------------------------------- replay
async function initReplay() {
  const personas = ["length-lover", "structure-lover", "hedge-hater"];
  const replays = (await Promise.all(personas.map((p) => load(`act2/replay-${p}.json`)))).filter(Boolean);
  if (!replays.length) {
    document.getElementById("a2-drift").textContent = "Replays are still being generated.";
    return;
  }
  let cur = 0;
  let round = 0;
  let probe = 0;
  const tabs = document.getElementById("a2-personas");
  tabs.replaceChildren();
  const tabBtns = replays.map((r, i) => {
    const b = el("button", { type: "button", role: "tab", "aria-selected": String(i === 0) }, tabs);
    el("span", { text: r.name }, b);
    el("small", { text: r.rule }, b);
    b.addEventListener("click", () => {
      cur = i;
      tabBtns.forEach((t, j) => t.setAttribute("aria-selected", String(i === j)));
      buildDrift();
      update();
    });
    return b;
  });
  const nRounds = replays[0].rounds.length;
  const scrub = scrubber({
    range: document.getElementById("a2-range"),
    play: document.getElementById("a2-play"),
    ticks: document.getElementById("a2-ticks"),
    count: nRounds,
    interval: 1300,
    onChange: (i) => { round = i; update(); },
  });
  chips(document.getElementById("a2-probes"), replays[0].rounds[0].probes.map((p) => p.prompt), {
    onSelect: (i) => { probe = i; drawProbe(); },
  });

  const driftEl = document.getElementById("a2-drift");
  let charts = [];
  function buildDrift() {
    driftEl.replaceChildren();
    const r = replays[cur];
    const target = TARGET[r.persona];
    charts = NAMES.map((nm) => {
      const box = el("div", { class: `multiple${nm === target ? " target" : ""}` }, driftEl);
      const h = el("h4", {}, box);
      el("span", { text: `${DRIFT[nm].title}` }, h);
      const val = el("span", { class: "val" }, h);
      const vals = r.rounds.map((rd) => (DRIFT[nm].key ? rd[DRIFT[nm].key] : rd.drift[nm]));
      const lo = Math.min(...vals);
      const hi = Math.max(...vals);
      const pad = (hi - lo) * 0.15 || Math.abs(hi) * 0.1 || 1;
      const chart = lineChart(box, {
        height: 96,
        margin: { top: 6, right: 6, bottom: 18, left: 6 },
        x: { min: 0, max: nRounds - 1, ticks: [{ v: 0, label: "0", anchor: "start" }, { v: nRounds - 1, label: String(nRounds - 1), anchor: "end" }] },
        y: { min: Math.max(0, lo - pad), max: hi + pad, ticks: [], fmt: (v) => v.toFixed(nm === "length" ? 0 : 2) },
        series: [{ id: nm, name: DRIFT[nm].title, color: nm === target ? "var(--a2)" : "var(--gray-mark)", points: vals.map((v, i) => [i, v]), dots: false }],
        xLabel: (x) => (x === 0 ? "before training" : `after round ${x}`),
        cursor: round,
      });
      return { chart, val, vals, nm };
    });
  }
  function update() {
    const r = replays[cur];
    const rd = r.rounds[round];
    document.getElementById("a2-round-label").textContent = round === 0 ? "before training" : `after round ${round}`;
    document.getElementById("a2-round-detail").textContent = round === 0
      ? "the SFT model, untouched"
      : `${round * r.config.pairs_per_round} judged pairs · DPO preference accuracy ${Math.round(rd.dpo.reward_accuracy * 100)}%`;
    charts.forEach(({ chart, val, vals, nm }) => {
      chart.setCursor(round);
      val.textContent = `${vals[round].toFixed(nm === "length" ? 0 : 2)} ${DRIFT[nm].unit === "tokens" ? "tokens" : ""}`.trim();
    });
    const target = TARGET[r.persona];
    document.getElementById("a2-rule").textContent = `Hidden rule: ${r.rule}.`;
    const rec = document.getElementById("a2-recovered");
    if (round === 0) {
      weightBars(rec, NAMES, LABELS, null, { max: 3 });
    } else {
      weightBars(rec, NAMES, LABELS, rd.implicit_reward, { max: 3 });
    }
    [...rec.children].forEach((row, i) => row.classList.toggle("stated", NAMES[i] === target));
    drawProbe();
  }
  function drawProbe() {
    const rd = replays[cur].rounds[round];
    const p = rd.probes[probe];
    document.getElementById("a2-probe-answer").textContent = p ? p.text : "";
    document.getElementById("a2-probe-caption").textContent =
      `${round === 0 ? "Before training" : `After round ${round}`}: the model's answer to “${p ? p.prompt : ""}” (greedy decoding).`;
  }
  tableToggle(document.getElementById("a2-drift").parentElement, () => {
    const r = replays[cur];
    return {
      head: ["Round", ...NAMES.map((n) => DRIFT[n].title)],
      rows: r.rounds.map((rd, i) => [String(i), ...NAMES.map((n) => (DRIFT[n].key ? rd[DRIFT[n].key] : rd.drift[n]).toFixed(2))]),
    };
  });
  buildDrift();
  scrub.set(0);
}
