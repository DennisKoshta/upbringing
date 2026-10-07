// Section 3: try-it-yourself with the same checker, training curves (synced crosshair), a trace viewer over held-out
// puzzles, the post-hoc "final exam", and the checker bug.
import { chips, el, fmtInt, lineChart, load, pct, player, stepper, tableToggle } from "./lib.js";
import { initGlossary } from "./glossary.js";

const BEHAVIOR_META = {
  search: { name: "Tries another approach", color: "var(--s1)", example: "“let's try…”, “instead”" },
  reject: { name: "Rejects a candidate", color: "var(--s2)", example: "“not equal to 43”, “too big”" },
  verify: { name: "Verifies", color: "var(--s3)", example: "“let me check”, “this works”" },
  wait: { name: "Says “wait”", color: "var(--s4)", example: "“wait”, “hmm”" },
};
const CHARTED = ["search", "reject", "verify"];

function smooth(points, k = 9) {
  return points.map((p, i) => {
    const lo = Math.max(0, i - Math.floor(k / 2));
    const win = points.slice(lo, lo + k);
    return [p[0], win.reduce((s, q) => s + q[1], 0) / win.length];
  });
}
const stepName = (s) => (s === 0 ? "base model" : `after ${s} training steps`);

// Same rule as upbringing/countdown.py: every number exactly once, + - * / and parentheses, exact target.
export function checkEquation(input, nums, target) {
  const expr = input.replace(/×|x|X/g, "*").replace(/÷/g, "/").replace(/[−–]/g, "-").trim();
  if (!expr) return { ok: false, why: "Type an equation first." };
  if (!/^[\d+\-*/().\s]+$/.test(expr)) return { ok: false, why: "Only numbers, + − × ÷ and parentheses are allowed." };
  const used = (expr.match(/\d+/g) || []).map(Number).sort((a, b) => a - b);
  const want = nums.slice().sort((a, b) => a - b);
  if (used.join(",") !== want.join(",")) {
    return { ok: false, why: `Use each of ${nums.join(", ")} exactly once (you used ${used.length ? used.join(", ") : "none"}).` };
  }
  let value;
  try {
    value = Function(`"use strict"; return (${expr});`)(); // input is restricted to digits, operators and parens above
  } catch {
    return { ok: false, why: "That isn't a complete equation." };
  }
  if (!Number.isFinite(value)) return { ok: false, why: "That divides by zero." };
  if (Math.abs(value - target) < 1e-5) return { ok: true, why: `${expr} = ${target}.` };
  return { ok: false, why: `That makes ${+value.toFixed(4)}, not ${target}.` };
}

export async function initAct3() {
  const [curves, traces, posthoc] = await Promise.all([load("rlvr/curves.json"), load("rlvr/traces.json"), load("rlvr/posthoc.json")]);
  let traceApi = null;
  if (traces && traces.puzzles.length) initTryIt(traces.puzzles, (i) => traceApi && traceApi.select(i));
  if (posthoc && posthoc.checkpoints.length > 1) drawPosthoc(posthoc.checkpoints);
  if (!curves || !curves.eval.length) {
    document.getElementById("a3-curves").textContent = "Training is in progress; results appear after the first evaluation.";
    return;
  }
  const lastStep = Math.max(curves.max_steps, ...curves.train.map((t) => t.step));
  drawKpis(curves);
  drawCurves(curves, lastStep);
  if (traces && traces.puzzles.length) traceApi = initTraces(curves, traces);
  initGlossary(document.getElementById("act3"));
}

function initTryIt(puzzles, onSee) {
  // start on a 3-number puzzle: a fair first try
  let i = Math.max(0, puzzles.findIndex((p) => p.nums.length === 3));
  const puzzleEl = document.getElementById("a3-try-puzzle");
  const input = document.getElementById("a3-try-input");
  const result = document.getElementById("a3-try-result");
  const show = () => {
    const p = puzzles[i];
    puzzleEl.replaceChildren();
    puzzleEl.append("Make ");
    el("b", { text: String(p.target) }, puzzleEl);
    puzzleEl.append(" using ");
    p.nums.forEach((nn, k) => {
      el("span", { class: "num-chip", text: String(nn) }, puzzleEl);
      if (k < p.nums.length - 1) puzzleEl.append(" ");
    });
    input.value = "";
    result.replaceChildren();
  };
  document.getElementById("a3-try-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const p = puzzles[i];
    const r = checkEquation(input.value, p.nums, p.target);
    result.replaceChildren();
    el("span", { class: `verdict ${r.ok ? "ok" : "bad"}`, text: r.ok ? "✓ Reward 1." : "✗ Reward 0." }, result);
    result.append(` ${r.why} `);
    el("span", { class: "muted", text: r.ok
      ? "That number is the entire signal the model receives."
      : "The model would receive only the zero, not the explanation." }, result);
    const see = el("button", { class: "linkish", type: "button", text: "See the model's attempts at this puzzle ↓" }, el("div", {}, result));
    see.addEventListener("click", () => { onSee(i); document.getElementById("a3-traces").scrollIntoView({ behavior: "smooth", block: "center" }); });
  });
  document.getElementById("a3-try-new").addEventListener("click", () => { i = (i + 1) % puzzles.length; show(); });
  show();
}

function drawKpis(c) {
  const k = document.getElementById("a3-kpis");
  k.replaceChildren();
  const first = c.eval[0];
  const last = c.eval[c.eval.length - 1];
  const tok0 = c.tokens[0];
  const tok1 = c.tokens[c.tokens.length - 1];
  const tr = c.train;
  const v0 = tr.slice(0, 5);
  const v1 = tr.slice(-10);
  const avg = (a, key) => a.reduce((s, x) => s + (x[key] || 0), 0) / Math.max(a.length, 1);
  const b = CHARTED.slice().sort((x, y) => (avg(v1, y) - avg(v0, y)) - (avg(v1, x) - avg(v0, x)))[0];
  [
    ["Held-out accuracy", pct(last.accuracy), `from ${pct(first.accuracy, 1)} for the base model`],
    ["Response length", tok1 ? `${Math.round(tok1.tokens)} tokens` : "–", tok0 ? `from ${Math.round(tok0.tokens)} at step 0` : ""],
    [`“${BEHAVIOR_META[b].name}” phrasing`, pct(avg(v1, b)), `of rollouts, from ${pct(avg(v0, b))}`],
    ["Rollouts graded", fmtInt(tr.reduce((s, x) => s + x.n, 0)), `over ${tr[tr.length - 1].step} optimizer steps`],
  ].forEach(([label, value, delta]) => {
    const d = el("div", { class: "kpi" }, k);
    el("div", { class: "kl", text: label }, d);
    el("div", { class: "kv", text: value }, d);
    el("div", { class: "kd", text: delta }, d);
  });
}

function drawCurves(c, lastStep) {
  const box = document.getElementById("a3-curves");
  box.replaceChildren();
  const xTicks = [0, 300, 600, 900].filter((v) => v <= lastStep).map((v) => ({ v, label: String(v) }));
  const x = { min: 0, max: lastStep, ticks: xTicks };
  const xLabel = (s) => (s === 0 ? "base model" : `step ${s}`);
  const charts = [];
  const sync = (src) => (s) => charts.forEach((ch, i) => i !== src && ch.showHover(s));
  const mk = (title, opts) => {
    const cell = el("div", { class: "multiple" }, box);
    const h = el("h4", {}, cell);
    h.append(title);
    return lineChart(cell, { height: 190, x, xLabel, margin: { left: 34, right: 12, bottom: 24 }, ...opts });
  };
  const top = Math.ceil(Math.max(0.2, ...c.eval.map((e) => e.accuracy)) * 10) / 10;
  charts.push(mk("Held-out accuracy (per sample)", {
    legend: "space",
    y: { min: 0, max: top, ticks: [0, top / 2, top].map((v) => +v.toFixed(2)), fmt: (v) => pct(v), tipFmt: (v) => pct(v, 1) },
    series: [{ id: "eval", name: "Held-out accuracy", color: "var(--a3)", points: c.eval.map((e) => [e.step, e.accuracy]), dots: true }],
    onHover: sync(0),
  }));
  const tTop = Math.ceil(Math.max(...c.tokens.map((t) => t.tokens), 100) / 100) * 100;
  charts.push(mk("Response length (tokens)", {
    legend: "space",
    y: { min: 0, max: tTop, ticks: [0, tTop / 2, tTop], fmt: (v) => String(Math.round(v)) },
    series: [{ id: "tokens", name: "Mean tokens per rollout", color: "var(--a3)", points: smooth(c.tokens.map((t) => [t.step, t.tokens]), 5) }],
    onHover: sync(1),
  }));
  const bTop = Math.ceil(Math.max(0.1, ...c.train.flatMap((t) => CHARTED.map((k) => t[k] || 0))) * 10) / 10;
  charts.push(mk("Reasoning phrases (share of rollouts)", {
    legend: true,
    y: { min: 0, max: bTop, ticks: [0, bTop / 2, bTop].map((v) => +v.toFixed(2)), fmt: (v) => pct(v), tipFmt: (v) => pct(v, 1) },
    series: CHARTED.map((k) => ({ id: k, name: BEHAVIOR_META[k].name, color: BEHAVIOR_META[k].color, points: smooth(c.train.map((t) => [t.step, t[k] || 0])) })),
    onHover: sync(2),
  }));
  const first = c.train.slice(0, 5);
  const last = c.train.slice(-10);
  const mean = (a, k) => a.reduce((s, t) => s + (t[k] || 0), 0) / Math.max(a.length, 1);
  const note = el("p", { class: "stat-line" }, box.parentElement);
  note.append("The classic “wait” stays rare (");
  el("b", { text: `${pct(mean(first, "wait"))} → ${pct(mean(last, "wait"))}` }, note);
  note.append(") and up-front planning (“first, we need…”) nearly disappears (");
  el("b", { text: `${pct(mean(first, "subgoal"))} → ${pct(mean(last, "subgoal"))}` }, note);
  note.append("). What grows is explicit trial and error: propose, evaluate, reject, try again. Phrase counts measure how the model " +
    "writes, not whether its checks are valid, and they can't distinguish behavior RL created from behavior it amplified.");
  document.getElementById("a3-curves-sub").textContent =
    "Held-out accuracy: 256 puzzles never used in training, 4 samples each at temperature 1. Length and phrase rates: all training rollouts. Hover to read values.";
  tableToggle(box.parentElement, () => ({
    head: ["Step", "Right (new puzzles)", "Right (practice)", ...Object.values(BEHAVIOR_META).map((m) => m.name), "Plans ahead"],
    rows: c.train.filter((t) => t.step % 50 === 0).map((t) => {
      const e = c.eval.find((v) => v.step === t.step);
      return [String(t.step), e ? pct(e.accuracy, 1) : "", pct(t.accuracy, 1), ...Object.keys(BEHAVIOR_META).map((k) => pct(t[k] || 0, 1)), pct(t.subgoal || 0, 1)];
    }),
  }));
}

function initTraces(curves, traces) {
  const key = document.getElementById("a3-key");
  key.replaceChildren();
  Object.values(BEHAVIOR_META).forEach((m) => {
    const s = el("span", {}, key);
    el("i", { style: `background:${m.color}` }, s);
    s.append(`${m.name} (${m.example})`);
  });
  const patterns = Object.entries(curves.behaviors).filter(([k]) => BEHAVIOR_META[k]).map(([k, src]) => [k, new RegExp(src, "gi")]);
  let puzzle = 0;
  let step = 0;
  const steps = traces.steps;
  const scrub = stepper(document.getElementById("a3-stepper"), steps.map((st) => ({
    group: st === 0 ? "base" : "rl", label: String(st), aria: stepName(st),
  })), {
    groups: { base: "Base", rl: "Training steps" },
    minSlot: 34,
    onChange: (i) => { step = i; draw(); },
  });
  player(document.getElementById("a3-play"), () => scrub, 1200);
  const puzzleChips = chips(document.getElementById("a3-puzzles"), traces.puzzles.map((p) => `Make ${p.target} from ${p.nums.join(", ")}`), {
    onSelect: (i) => { puzzle = i; draw(); },
  });

  async function draw() {
    const meta = traces.puzzles[puzzle];
    const t = await load(`rlvr/${meta.file}`);
    const s = steps[step];
    document.getElementById("a3-step-label").textContent = stepName(s);
    const frame = t && t.frames.find((f) => f.step === s);
    const box = document.getElementById("a3-traces");
    box.replaceChildren();
    if (!frame) return;
    const ok = frame.samples.filter((x) => x.correct).length;
    document.getElementById("a3-step-detail").textContent = `${ok} of ${frame.samples.length} samples correct`;
    frame.samples.forEach((smp, i) => {
      const card = el("div", { class: "trace" }, box);
      const head = el("div", { class: "trace-head" }, card);
      el("span", { text: `Sample ${i + 1}` }, head);
      el("span", { class: `verdict ${smp.correct ? "ok" : "bad"}`, text: smp.correct ? "✓ correct" : smp.answer ? "✗ wrong" : "✗ no answer" }, head);
      const tt = el("div", { class: "trace-text" }, card);
      el("span", { class: "pre", text: "<think>" }, tt);
      renderTrace(tt, smp.text, patterns);
    });
  }
  puzzleChips.select(0);
  return { select: (i) => puzzleChips.select(i) };
}

function renderTrace(node, text, patterns) {
  // Underline behavior phrases in the reasoning and box the final answer. Overlaps keep the earliest match.
  const think = text.split("</think>")[0];
  const spans = [];
  patterns.forEach(([k, re]) => {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(think))) {
      spans.push({ start: m.index, end: m.index + m[0].length, k });
      if (m[0].length === 0) re.lastIndex++;
    }
  });
  const ans = /<answer>[\s\S]*?<\/answer>/.exec(text);
  if (ans) spans.push({ start: ans.index, end: ans.index + ans[0].length, k: "answer" });
  spans.sort((a, b) => a.start - b.start);
  let pos = 0;
  for (const sp of spans) {
    if (sp.start < pos) continue;
    node.append(text.slice(pos, sp.start));
    if (sp.k === "answer") el("span", { class: "ans", text: text.slice(sp.start, sp.end) }, node);
    else el("mark", { style: `--b:${BEHAVIOR_META[sp.k].color}`, title: BEHAVIOR_META[sp.k].name, text: text.slice(sp.start, sp.end) }, node);
    pos = sp.end;
  }
  node.append(text.slice(pos));
}

function drawPosthoc(all) {
  const fig = document.getElementById("a3-posthoc");
  fig.hidden = false;
  const s0 = all[0];
  document.getElementById("a3-posthoc-sub").textContent =
    `Every saved checkpoint evaluated identically: the same ${s0.n_puzzles} held-out puzzles, ${s0.samples_per_puzzle} samples each at ` +
    "temperature 1, scored by the corrected verifier.";
  let subset = "all";
  const seg = el("div", { class: "seg", role: "tablist", "aria-label": "Puzzle size" });
  fig.querySelector(".fig-sub").after(seg);
  [["all", "All puzzles"], ["3", "3 numbers"], ["4", "4 numbers"]].forEach(([v, label]) => {
    const b = el("button", { type: "button", role: "tab", "aria-selected": String(v === "all"), text: label }, seg);
    b.addEventListener("click", () => { subset = v; [...seg.children].forEach((c) => c.setAttribute("aria-selected", String(c === b))); render(); });
  });

  function view(c) {
    if (subset === "all" || !c.by_size || !c.by_size[subset]) return c;
    return { ...c, ...c.by_size[subset] };
  }
  function render() {
    const cks = all.map(view);
    const last = cks[cks.length - 1];
    const steps = cks.map((c) => c.step);
    const maxStep = Math.max(...steps);
    const top = Math.min(1, Math.ceil(Math.max(...cks.map((c) => Math.max(c.ci95[1], c.pass_at_k["32"] || 0))) * 10) / 10);
    const pctTicks = [0, top / 2, top].map((v) => +v.toFixed(2));
    const accBox = document.getElementById("a3-acc-ci");
    accBox.replaceChildren();
    lineChart(accBox, {
      height: 200, margin: { left: 34, right: 10, bottom: 24 },
      x: { min: 0, max: maxStep, ticks: steps.filter((v, i) => i === 0 || i === steps.length - 1 || v % 200 === 0).map((v) => ({ v, label: String(v) })) },
      y: { min: 0, max: top, ticks: pctTicks, fmt: (v) => pct(v), tipFmt: (v) => pct(v, 1) },
      legend: true,
      series: [
        { id: "acc", name: "Sampled (T = 1)", color: "var(--a3)", points: cks.map((c) => [c.step, c.per_attempt_accuracy]), dots: true },
        { id: "greedy", name: "Greedy", color: "var(--gray-mark)", points: cks.map((c) => [c.step, c.greedy_accuracy]), dots: true },
      ],
      intervals: cks.map((c) => ({ x: c.step, y0: c.ci95[0], y1: c.ci95[1], color: "var(--a3)" })),
      xLabel: (xv) => (xv === 0 ? "base model" : `step ${xv}`),
    });
    const ks = Object.keys(s0.pass_at_k).map(Number).sort((a, b) => a - b);
    const mid = cks.length > 2 ? [cks[Math.floor((cks.length - 1) / 2)]] : [];
    const pick = [cks[0], ...mid, last].filter((c, i, a) => a.indexOf(c) === i);
    const pkBox = document.getElementById("a3-passk");
    pkBox.replaceChildren();
    lineChart(pkBox, {
      height: 200, margin: { left: 34, right: 12, bottom: 24 },
      x: { min: 0, max: Math.log2(ks[ks.length - 1]), ticks: ks.map((k) => ({ v: Math.log2(k), label: String(k) })) },
      y: { min: 0, max: top, ticks: pctTicks, fmt: (v) => pct(v), tipFmt: (v) => pct(v, 1) },
      legend: true,
      series: pick.map((c) => ({
        id: c.label,
        name: c.step === 0 ? "Base model" : `Step ${c.step}`,
        color: c.step === 0 ? "var(--gray-mark)" : c === last ? "var(--a3)" : "#86b6ef",
        points: ks.map((k) => [Math.log2(k), c.pass_at_k[String(k)]]),
        dots: true,
      })),
      xLabel: (xv) => `k = ${Math.round(2 ** xv)}`,
    });
    const note = document.getElementById("a3-posthoc-note");
    note.replaceChildren();
    note.append(`Base model: ${pct(cks[0].per_attempt_accuracy, 1)} per sample, pass@32 ${pct(cks[0].pass_at_k["32"])}. Step ${last.step}: `);
    el("b", { text: `${pct(last.per_attempt_accuracy)} per sample` }, note);
    note.append(` (95% CI ${pct(last.ci95[0])}–${pct(last.ci95[1])}), pass@32 ${pct(last.pass_at_k["32"])}. ` +
      "Most of the gain is reliability: per-sample accuracy rises much faster than coverage at k = 32. " +
      "Whether RLVR teaches new solutions or concentrates probability on ones the base model can already sample is an open question; 32 samples is far too few to settle it.");
  }
  render();
  tableToggle(fig, () => {
    const ks = Object.keys(s0.pass_at_k).map(Number).sort((a, b) => a - b);
    return {
      head: ["Checkpoint", "Per sample", "95% CI", "Greedy", ...ks.map((k) => `pass@${k}`), "Tokens"],
      rows: all.map(view).map((c) => [stepName(c.step), pct(c.per_attempt_accuracy, 1), `${pct(c.ci95[0], 1)}–${pct(c.ci95[1], 1)}`,
        pct(c.greedy_accuracy, 1), ...ks.map((k) => pct(c.pass_at_k[String(k)], 1)), String(Math.round(c.mean_tokens))]),
    };
  });
}
