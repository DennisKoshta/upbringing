// Section 2: judge pairs -> client-side Bradley-Terry fit ("what your choices reveal"); then replays of simulated
// judges training a model with LoRA DPO, one round per click.
import { chips, el, lineChart, load, tableToggle, weightBars } from "./lib.js";
import { initGlossary } from "./glossary.js";

const NAMES = ["length", "structure", "hedging", "confidence", "enthusiasm"];
const LABELS = { length: "Length", structure: "Lists & headers", hedging: "Hedging", confidence: "Confidence", enthusiasm: "Enthusiasm" };
const STATED = {
  length: "Thorough and detailed",
  structure: "Well organized (lists, headers)",
  hedging: "Careful, with caveats",
  confidence: "Direct and confident",
  enthusiasm: "Warm and upbeat",
};
const DRIFT = {
  length: { title: "Answer length", unit: "tokens", key: "drift_tokens", digits: 0 },
  structure: { title: "Lists & headers", unit: "per answer", digits: 1 },
  hedging: { title: "Hedge words", unit: "per 100 words", digits: 2 },
  confidence: { title: "Confident words", unit: "per 100 words", digits: 2 },
  enthusiasm: { title: "Exclamation marks", unit: "per answer", digits: 2 },
};
const JUDGES = {
  "length-lover": { name: "Likes long answers", target: "length" },
  "structure-lover": { name: "Likes lists and headers", target: "structure" },
  "hedge-hater": { name: "Dislikes caveats", target: "hedging" },
};
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
    pairEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  function show() {
    const p = data.pairs[order[pos]];
    const swap = Math.random() < 0.5; // randomize sides so a habit of clicking one side doesn't look like a preference
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
      `Estimated from your ${picks.length} picks. A ★ marks what you said you value.`;
    const ranked = NAMES.slice().sort((a, b) => Math.abs(w[b]) - Math.abs(w[a]));
    const top = ranked[0];
    const txt = document.getElementById("a2-reveal-text");
    txt.replaceChildren();
    txt.append("Your picks leaned most toward ");
    el("strong", { text: `${w[top] >= 0 ? "more" : "less"} ${LABELS[top].toLowerCase()}` }, txt);
    txt.append(". ");
    if (statedList.length) {
      const match = statedList.includes(top) && w[top] > 0;
      txt.append(match
        ? "That matches what you said you value. "
        : `You said you value ${statedList.map((s) => STATED[s].toLowerCase()).join(" and ")}, but your picks point elsewhere. `);
    }
    txt.append(`If a model were trained on your ${picks.length} picks, it would drift toward this taste, whether or not you meant it to. ` +
      `${picks.length} picks is a small sample, so this estimate is rough. Real preference datasets have hundreds of thousands of picks, with the same blind spots.`);
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
  const ids = Object.keys(JUDGES);
  const loaded = await Promise.all(ids.map((p) => load(`act2/replay-${p}.json`)));
  const replays = loaded.filter(Boolean);
  if (!replays.length) {
    document.getElementById("a2-drift").textContent = "These replays are still being generated.";
    return;
  }
  let cur = 0;
  let round = 0;
  let probe = 0;
  const nRounds = replays[0].rounds.length;

  const tabs = document.getElementById("a2-personas");
  tabs.replaceChildren();
  const tabBtns = replays.map((r, i) => {
    const b = el("button", { type: "button", role: "tab", "aria-selected": String(i === 0) }, tabs);
    el("span", { text: JUDGES[r.persona].name }, b);
    el("small", { text: `Always picks ${ruleText(r.persona)}` }, b);
    b.addEventListener("click", () => {
      cur = i;
      tabBtns.forEach((t, j) => t.setAttribute("aria-selected", String(i === j)));
      buildDrift();
      update();
    });
    return b;
  });

  const dotsEl = document.getElementById("a2-round-dots");
  const roundDots = Array.from({ length: nRounds - 1 }, (_, i) => {
    const d = el("button", { type: "button", "aria-label": `Show round ${i + 1}` }, dotsEl);
    d.addEventListener("click", () => { round = i + 1; update(); });
    return d;
  });
  const stepBtn = document.getElementById("a2-step");
  stepBtn.addEventListener("click", () => { if (round < nRounds - 1) { round += 1; update(); } });
  document.getElementById("a2-reset").addEventListener("click", () => { round = 0; update(); });

  chips(document.getElementById("a2-probes"), replays[0].rounds[0].probes.map((p) => p.prompt), {
    onSelect: (i) => { probe = i; drawProbe(); },
  });

  const driftEl = document.getElementById("a2-drift");
  let charts = [];
  function buildDrift() {
    driftEl.replaceChildren();
    const r = replays[cur];
    const target = JUDGES[r.persona].target;
    charts = NAMES.map((nm) => {
      const box = el("div", { class: `multiple${nm === target ? " target" : ""}` }, driftEl);
      const h = el("h4", {}, box);
      const tt = el("span", {}, h);
      tt.append(DRIFT[nm].title + " ");
      if (DRIFT[nm].unit === "tokens") {
        tt.append("(");
        el("span", { class: "term", "data-t": "token", text: "tokens" }, tt);
        tt.append(")");
      }
      const val = el("span", { class: "val" }, h);
      const vals = r.rounds.map((rd) => (DRIFT[nm].key ? rd[DRIFT[nm].key] : rd.drift[nm]));
      const lo = Math.min(...vals);
      const hi = Math.max(...vals);
      const pad = (hi - lo) * 0.15 || Math.abs(hi) * 0.1 || 1;
      const chart = lineChart(box, {
        height: 92,
        margin: { top: 6, right: 6, bottom: 18, left: 6 },
        x: { min: 0, max: nRounds - 1, ticks: [{ v: 0, label: "start", anchor: "start" }, { v: nRounds - 1, label: `round ${nRounds - 1}`, anchor: "end" }] },
        y: { min: Math.max(0, lo - pad), max: hi + pad, ticks: [], fmt: (v) => v.toFixed(DRIFT[nm].digits) },
        series: [{ id: nm, name: DRIFT[nm].title, color: nm === target ? "var(--a2)" : "var(--gray-mark)", points: vals.map((v, i) => [i, v]) }],
        xLabel: (x) => (x === 0 ? "before training" : `after round ${x}`),
        cursor: round,
      });
      return { chart, val, vals, nm };
    });
    initGlossary(driftEl);
  }
  function update() {
    const r = replays[cur];
    const rd = r.rounds[round];
    document.getElementById("a2-round-label").textContent = round === 0
      ? "not trained yet"
      : `after round ${round} of ${nRounds - 1} · ${round * r.config.pairs_per_round} picks so far`;
    stepBtn.disabled = round >= nRounds - 1;
    stepBtn.textContent = round >= nRounds - 1 ? "All rounds done" : round === 0 ? "Train a round" : "Train another round";
    roundDots.forEach((d, i) => d.classList.toggle("done", i < round));
    charts.forEach(({ chart, val, vals, nm }) => {
      chart.setCursor(round);
      val.textContent = vals[round].toFixed(DRIFT[nm].digits);
    });
    const target = JUDGES[r.persona].target;
    document.getElementById("a2-rule").textContent = `This judge always picks ${ruleText(r.persona)}.`;
    const rec = document.getElementById("a2-recovered");
    weightBars(rec, NAMES, LABELS, round === 0 ? null : rd.implicit_reward, { max: 3 });
    [...rec.children].forEach((row, i) => row.classList.toggle("stated", NAMES[i] === target));
    drawProbe();
  }
  function drawProbe() {
    const rd = replays[cur].rounds[round];
    const p = rd.probes[probe];
    document.getElementById("a2-probe-q").textContent = p ? p.prompt : "";
    document.getElementById("a2-probe-who").textContent = round === 0 ? "Model, before training" : `Model, after round ${round}`;
    const a = document.getElementById("a2-probe-answer");
    a.textContent = p ? p.text : "";
    a.classList.remove("swap");
    void a.offsetWidth;
    a.classList.add("swap");
  }
  tableToggle(document.getElementById("a2-drift").parentElement, () => {
    const r = replays[cur];
    return {
      head: ["Round", ...NAMES.map((n) => DRIFT[n].title)],
      rows: r.rounds.map((rd, i) => [String(i), ...NAMES.map((n) => (DRIFT[n].key ? rd[DRIFT[n].key] : rd.drift[n]).toFixed(2))]),
    };
  });
  buildDrift();
  update();
}

function ruleText(persona) {
  return {
    "length-lover": "the longer answer",
    "structure-lover": "the answer with more lists, headers and bold text",
    "hedge-hater": "the answer with fewer hedges and caveats",
  }[persona];
}
