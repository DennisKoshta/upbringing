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
  const [curves, traces, bug, cmp, posthoc] = await Promise.all([load("rlvr/curves.json"), load("rlvr/traces.json"), load("rlvr/bug.json"), load("rlvr/grader_comparison.json"), load("rlvr/posthoc.json")]);
  if (bug) drawBug(bug, cmp);
  if (posthoc && posthoc.checkpoints.length > 1) drawPosthoc(posthoc.checkpoints);
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
    ["Held-out accuracy (per attempt)", pct(last.accuracy, 1), `from ${pct(first.accuracy, 1)} at step 0 · 1,024 attempts`],
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
  note.append("). By these counts, what grows is explicit trial-and-error language, not a sudden “wait”. " +
    "Phrase counts describe how the model talks, not whether its checks are right, and they can't say whether RL created this search or amplified something the base model could already do; that needs comparisons at matched sampling budgets (pass@k, below).");
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

function drawBug(bug, cmp) {
  const body = document.getElementById("a3-bug-body");
  body.replaceChildren();
  const intro = el("p", { class: "body-text" }, body);
  intro.append("In our first run, generation didn't stop at the answer. The base model kept writing, often inventing new " +
    `“User:” puzzles and answering them: ${Math.round(bug.share_after_first_answer * 100)}% of the text it generated came after its first answer` +
    (cmp ? `, and ${Math.round((cmp.multiple_answer_blocks / cmp.rollouts) * 100)}% of rollouts held more than one answer block` : "") +
    ". The original grader read the last answer block. We kept one logged batch from that run and re-graded it both ways.");
  if (cmp) {
    const kp = el("div", { class: "kpis three" }, body);
    [
      [`${cmp.different_answer_read} of ${cmp.rollouts}`, "rollouts where the old grader read a different answer than the model's first"],
      [String(cmp.old_rewarded), `correct-answer reward${cmp.old_rewarded === 1 ? "" : "s"} in the batch under the old grader…`],
      [String(cmp.old_rewarded - cmp.false_positives.length), "…that went to a rollout whose first answer was actually correct"],
    ].forEach(([v, l]) => {
      const d = el("div", { class: "kpi" }, kp);
      el("div", { class: "kv", text: v }, d);
      el("div", { class: "kd", text: l }, d);
    });
    const wrap = el("div", { class: "scroll-x" }, body);
    const t = el("table", { class: "spec cmp" }, wrap);
    const hr = el("tr", {}, el("thead", {}, t));
    ["Puzzle", "The model's answer", "What the old grader read", "Old grader", "Corrected"].forEach((h) => el("th", { text: h }, hr));
    const tb = el("tbody", {}, t);
    const row = (c, oldOk) => {
      const tr = el("tr", {}, tb);
      el("td", { text: `${c.target} from ${c.nums.join(", ")}` }, tr);
      el("td", { class: "mono", text: c.first_answer ?? "–" }, tr);
      el("td", { class: "mono", text: c.answer_old_grader_read ?? "–" }, tr);
      el("td", { class: oldOk ? "ok" : "bad", text: oldOk ? "✓ rewarded" : "✗ no reward" }, tr);
      el("td", { class: oldOk ? "bad" : "ok", text: oldOk ? "✗ no reward" : "✓ rewarded" }, tr);
    };
    cmp.false_positives.forEach((c) => row(c, true));
    cmp.false_negatives.forEach((c) => row(c, false));
    el("p", { class: "fig-sub", text: `Logged batch at step ${cmp.step} (${cmp.rollouts} rollouts). Reproduce with analysis/grader_comparison.py.` }, body);
  }
  const cols = el("div", { class: "cols" }, body);
  const left = el("div", {}, cols);
  el("h4", { class: "fig-sub", text: `What a rollout looked like (make ${bug.target} from ${bug.nums.join(", ")})` }, left);
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
    "Grade the first <answer> after </think>, never a later one.",
    "Tests pin both: an invented follow-up answer is never graded, and the logged batch re-grades as shown.",
    "We restarted the run (about $1.50 lost). Every other number on this page is from the corrected run.",
  ].forEach((x) => el("li", { text: x }, ul));
  el("p", { class: "note", text: "What this is and isn't: a grading vulnerability with a measured effect on the rewards in one batch. " +
    "It is not demonstrated reward hacking; we stopped the run at step 24, before we could see whether optimization learned to exploit it." }, right);
}

function drawPosthoc(cks) {
  const fig = document.getElementById("a3-posthoc");
  fig.hidden = false;
  const s0 = cks[0];
  document.getElementById("a3-posthoc-sub").textContent =
    `Every checkpoint re-evaluated with identical settings: the same ${s0.n_puzzles} held-out puzzles, ` +
    `${s0.samples_per_puzzle} attempts each at temperature ${s0.settings.temperature}, the corrected checker.`;
  const last = cks[cks.length - 1];
  const steps = cks.map((c) => c.step);
  const maxStep = Math.max(...steps);
  const top = Math.min(1, Math.ceil(Math.max(...cks.map((c) => Math.max(c.ci95[1], c.pass_at_k["32"] || 0))) * 10) / 10);
  const pctTicks = [0, top / 2, top].map((v) => +v.toFixed(2));

  // accuracy with CI: a dot per checkpoint, the interval as a short vertical rule
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
    xLabel: (x) => (x === 0 ? "base model" : `step ${x}`),
  });
  // pass@k: base in gray, trained checkpoints on a one-hue ordinal ramp, final in the accent
  const ks = Object.keys(s0.pass_at_k).map(Number).sort((a, b) => a - b);
  const pick = [cks[0], ...cks.slice(1, -1).filter((_, i, arr) => arr.length <= 2 || i === Math.floor(arr.length / 2)), last].filter((c, i, a) => a.indexOf(c) === i);
  const ramp = ["#86b6ef", "#3987e5", "#1c5cab"];
  const pkBox = document.getElementById("a3-passk");
  pkBox.replaceChildren();
  lineChart(pkBox, {
    height: 200, margin: { left: 34, right: 64, bottom: 24 },
    x: { min: 0, max: Math.log2(ks[ks.length - 1]), ticks: ks.map((k) => ({ v: Math.log2(k), label: String(k) })) },
    y: { min: 0, max: top, ticks: pctTicks, fmt: (v) => pct(v), tipFmt: (v) => pct(v, 1) },
    legend: true, endLabels: true,
    series: pick.map((c, i) => ({
      id: c.label,
      name: c.step === 0 ? "Base model" : `Step ${c.step}`,
      color: c.step === 0 ? "var(--gray-mark)" : c === last ? "var(--a3)" : ramp[Math.min(i - 1, ramp.length - 1)],
      points: ks.map((k) => [Math.log2(k), c.pass_at_k[String(k)]]),
      dots: true,
    })),
    xLabel: (x) => `k = ${Math.round(2 ** x)}`,
  });

  const note = document.getElementById("a3-posthoc-note");
  note.replaceChildren();
  note.append(`Base model: ${pct(s0.per_attempt_accuracy, 1)} per attempt, pass@32 ${pct(s0.pass_at_k["32"], 1)}. `);
  note.append(`Step ${last.step}: `);
  el("b", { text: `${pct(last.per_attempt_accuracy, 1)} per attempt` }, note);
  note.append(` (95% CI ${pct(last.ci95[0], 1)}–${pct(last.ci95[1], 1)}), pass@32 ${pct(last.pass_at_k["32"], 1)}. ` +
    "If RL only sharpened what the base model could already sample, the base curve would close the gap as k grows; " +
    "32 attempts is far below the budgets where such crossovers have been reported, so this bounds the question rather than settling it.");
  tableToggle(fig, () => ({
    head: ["Checkpoint", "Per attempt", "95% CI", "Greedy", ...ks.map((k) => `pass@${k}`), "Tokens"],
    rows: cks.map((c) => [c.step === 0 ? "base" : `step ${c.step}`, pct(c.per_attempt_accuracy, 1), `${pct(c.ci95[0], 1)}–${pct(c.ci95[1], 1)}`,
      pct(c.greedy_accuracy, 1), ...ks.map((k) => pct(c.pass_at_k[String(k)], 1)), String(Math.round(c.mean_tokens))]),
  }));
}
