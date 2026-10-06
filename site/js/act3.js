// Act III: RLVR curves (synced crosshair across small multiples), a trace scrubber over held-out puzzles, and the
// verifier bug exhibit.
import { chips, el, fmtInt, lineChart, load, pct, scrubber, tableToggle } from "./lib.js";

const BEHAVIOR_META = {
  search: { name: "Tries another approach", color: "var(--s1)", example: "“let's try”, “instead”, “try another”" },
  reject: { name: "Rejects a candidate", color: "var(--s2)", example: "“not equal to 43”, “too big”, “doesn't work”" },
  verify: { name: "Checks its work", color: "var(--s3)", example: "“let me check”, “this works”" },
  wait: { name: "Says “wait”", color: "var(--s4)", example: "“wait”, “hmm”, “hold on”" },
};
const CHARTED = ["search", "reject", "verify"];

function smooth(points, k = 9) {
  return points.map((p, i) => {
    const lo = Math.max(0, i - Math.floor(k / 2));
    const win = points.slice(lo, lo + k);
    return [p[0], win.reduce((s, q) => s + q[1], 0) / win.length];
  });
}

export async function initAct3() {
  const [curves, traces, bug] = await Promise.all([load("rlvr/curves.json"), load("rlvr/traces.json"), load("rlvr/bug.json")]);
  if (bug) drawBug(bug);
  if (!curves || !curves.eval.length) {
    document.getElementById("a3-curves").textContent = "The run is in progress; curves appear after the first evaluation.";
    return;
  }
  const lastStep = Math.max(curves.max_steps, ...curves.train.map((t) => t.step));
  drawKpis(curves);
  drawCurves(curves, lastStep);
  if (traces && traces.puzzles.length) initTraces(curves, traces);
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
  const avg = (a, key) => a.reduce((s, x) => s + x[key], 0) / Math.max(a.length, 1);
  [
    ["Held-out accuracy", pct(last.accuracy, 1), `from ${pct(first.accuracy, 1)} at step 0`],
    ["Answer length", tok1 ? `${Math.round(tok1.tokens)} tokens` : "–", tok0 ? `from ${Math.round(tok0.tokens)} at the start` : ""],
    (() => {
      const b = CHARTED.slice().sort((x, y) => (avg(v1, y) - avg(v0, y)) - (avg(v1, x) - avg(v0, x)))[0];
      return [BEHAVIOR_META[b].name, pct(avg(v1, b)), `of attempts, from ${pct(avg(v0, b))} at the start`];
    })(),
    ["Attempts graded", fmtInt(tr.reduce((s, x) => s + x.n, 0)), `over ${tr[tr.length - 1].step} training steps`],
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
  const xTicks = [0, 150, 300, 450, 600, 750, 900].filter((v) => v <= lastStep).map((v) => ({ v, label: String(v) }));
  const x = { min: 0, max: lastStep, ticks: xTicks };
  const xLabel = (s) => `step ${s}`;
  const charts = [];
  const sync = (src) => (s) => charts.forEach((ch) => ch !== charts[src] && ch.showHover(s));

  const mk = (title, opts) => {
    const cell = el("div", { class: "multiple" }, box);
    el("h4", { text: title }, cell);
    return lineChart(cell, { height: 200, x, xLabel, margin: { left: 34, right: 12, bottom: 24 }, ...opts });
  };
  const accMax = Math.max(0.2, ...c.eval.map((e) => e.accuracy), ...c.train.map((t) => t.accuracy));
  const top = Math.ceil(accMax * 10) / 10;
  charts.push(mk("Accuracy", {
    legend: true,
    y: { min: 0, max: top, ticks: [0, top / 2, top].map((v) => +v.toFixed(2)), fmt: (v) => pct(v), tipFmt: (v) => pct(v, 1) },
    series: [
      { id: "train", name: "Training puzzles (smoothed)", color: "var(--gray-mark)", points: smooth(c.train.map((t) => [t.step, t.accuracy])) },
      { id: "eval", name: "Held-out puzzles", color: "var(--a3)", points: c.eval.map((e) => [e.step, e.accuracy]), dots: true },
    ],
    onHover: sync(0),
  }));
  const tokMax = Math.max(...c.tokens.map((t) => t.tokens), 100);
  const tTop = Math.ceil(tokMax / 100) * 100;
  charts.push(mk("Answer length (tokens)", {
    y: { min: 0, max: tTop, ticks: [0, tTop / 2, tTop], fmt: (v) => String(Math.round(v)) },
    legend: "space",
    series: [{ id: "tokens", name: "Mean tokens per attempt", color: "var(--a3)", points: smooth(c.tokens.map((t) => [t.step, t.tokens]), 5) }],
    onHover: sync(1),
  }));
  const bMax = Math.max(0.1, ...c.train.flatMap((t) => CHARTED.map((b) => t[b] || 0)));
  const bTop = Math.ceil(bMax * 10) / 10;
  charts.push(mk("Share of attempts showing…", {
    legend: true,
    y: { min: 0, max: bTop, ticks: [0, bTop / 2, bTop].map((v) => +v.toFixed(2)), fmt: (v) => pct(v), tipFmt: (v) => pct(v, 1) },
    series: CHARTED.map((k) => ({ id: k, name: BEHAVIOR_META[k].name, color: BEHAVIOR_META[k].color, points: smooth(c.train.map((t) => [t.step, t[k] || 0])) })),
    onHover: sync(2),
  }));
  const first = c.train.slice(0, 5);
  const last = c.train.slice(-10);
  const mean = (a, k) => a.reduce((s2, t) => s2 + (t[k] || 0), 0) / Math.max(a.length, 1);
  const note = el("p", { class: "stat-line" }, box.parentElement);
  note.append("Not charted: the classic “wait” stays rare (");
  el("b", { text: `${pct(mean(first, "wait"))} → ${pct(mean(last, "wait"))}` }, note);
  note.append(" of attempts), and planning subgoals up front fades (");
  el("b", { text: `${pct(mean(first, "subgoal"))} → ${pct(mean(last, "subgoal"))}` }, note);
  note.append("). Here the model's “aha” is systematic trial and error, not a sudden “wait”. Phrases are matched by regex: they show how it talks, not whether its checks are right.");
  tableToggle(box.parentElement, () => ({
    head: ["Step", "Held-out acc.", "Training acc.", ...Object.values(BEHAVIOR_META).map((m) => m.name), "Plans subgoals"],
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
    const s = el("span", { title: m.example }, key);
    el("i", { style: `background:${m.color}` }, s);
    s.append(`${m.name} (${m.example})`);
  });
  const patterns = Object.entries(curves.behaviors).filter(([k]) => BEHAVIOR_META[k]).map(([k, src]) => [k, new RegExp(src, "gi")]);
  let puzzle = 0;
  let step = 0;
  const steps = traces.steps;
  const scrub = scrubber({
    range: document.getElementById("a3-range"),
    play: document.getElementById("a3-play"),
    ticks: document.getElementById("a3-ticks"),
    count: steps.length,
    interval: 1200,
    onChange: (i) => { step = i; draw(); },
  });
  chips(document.getElementById("a3-puzzles"), traces.puzzles.map((p) => `${p.target} from ${p.nums.join(", ")}`), {
    onSelect: (i) => { puzzle = i; draw(); },
  });

  async function draw() {
    const meta = traces.puzzles[puzzle];
    const t = await load(`rlvr/${meta.file}`);
    const s = steps[step];
    document.getElementById("a3-step-label").textContent = s === 0 ? "step 0 (base model)" : `step ${s}`;
    const frame = t && t.frames.find((f) => f.step === s);
    const box = document.getElementById("a3-traces");
    box.replaceChildren();
    if (!frame) return;
    const ok = frame.samples.filter((x) => x.correct).length;
    document.getElementById("a3-step-detail").textContent = `${ok} of ${frame.samples.length} attempts correct`;
    frame.samples.forEach((smp, i) => {
      const card = el("div", { class: "trace" }, box);
      const head = el("div", { class: "trace-head" }, card);
      el("span", { text: `Attempt ${i + 1}` }, head);
      el("span", { class: `verdict ${smp.correct ? "ok" : "bad"}`, text: smp.correct ? "✓ correct" : smp.answer ? "✗ wrong" : "✗ no answer" }, head);
      const tt = el("div", { class: "trace-text" }, card);
      el("span", { class: "pre", text: "<think>" }, tt);
      renderTrace(tt, smp.text, patterns);
    });
  }
  scrub.set(0);
}

function renderTrace(node, text, patterns) {
  // Mark behavior phrases in the reasoning and box the final answer. Overlaps keep the earliest match.
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

function drawBug(bug) {
  const body = document.getElementById("a3-bug-body");
  body.replaceChildren();
  el("p", { class: "body-text", text:
    "In our first run, the base model didn't stop after answering. It went on to invent new “User:” puzzles and " +
    `answer those too: ${Math.round(bug.share_after_first_answer * 100)}% of everything it generated came after its first answer. ` +
    "Our grader read the last answer in the text, so it could have scored an answer to a question nobody asked. " +
    "It also inflated the length curve, which is exactly the signal an aha-moment chart relies on." }, body);
  const cols = el("div", { class: "cols" }, body);
  const left = el("div", {}, cols);
  el("h4", { class: "fig-sub", text: `A real rollout from that run (puzzle: make ${bug.target} from ${bug.nums.join(", ")})` }, left);
  const tt = el("div", { class: "trace-text" }, left);
  tt.append(bug.before.slice(-420));
  const after = el("span", { class: "after" }, tt);
  bug.after.split(/(User:)/).forEach((part) => (part === "User:" ? el("span", { class: "user", text: part }, after) : after.append(part)));
  if (bug.after_chars > bug.after.length) after.append(`\n… ${(bug.after_chars - bug.after.length).toLocaleString("en-US")} more characters`);
  const right = el("div", {}, cols);
  el("h4", { class: "fig-sub", text: "The fix" }, right);
  const ul = el("ul", {}, right);
  [
    "Stop generation at </answer>, as most RLVR setups do.",
    "Grade the first <answer> after </think>, never the last.",
    "A regression test pins the behavior: an invented follow-up answer must not be graded.",
    "We restarted the run (about $1.50 lost). Every curve above is from the fixed run.",
  ].forEach((t) => el("li", { text: t }, ul));
  el("p", { class: "note", text: "Verifiers are where RL goes wrong first: the model optimizes what the checker measures, not what you meant." }, right);
}
