// Act I: one training clock (the scrubber) drives the answer, the benchmark cursor, the histogram and the jokes.
import { chips, el, hideTip, lineChart, load, pct, scrubber, showTip, tableToggle, tipRow } from "./lib.js";
import { ckptDetail, ckptName, visibleTokens } from "./ckpt.js";

const SHADE_FULL = 8; // nats: a token 3000x more likely than under the base model gets the full shade

export async function initAct1() {
  const data = await load("act1/checkpoints.json");
  if (!data) return;
  const moments = (await load("act1/moments.json")) || [];
  const ckpts = data.checkpoints;
  const refs = data.references;
  const n = ckpts.length;
  const lastSft = Math.max(0, ...ckpts.filter((c) => c.stage === "sft").map((c) => c.step));
  document.getElementById("a1-nckpt").textContent = String(n - 1);

  let prompt = 0;
  let idx = 0;

  // ---------------------------------------------------------- scrubber
  const stages = document.getElementById("a1-stages");
  const firstOf = (s) => ckpts.findIndex((c) => c.stage === s);
  [["base", 0], ["SFT", firstOf("sft")], ["DPO", firstOf("dpo")]].forEach(([label, i]) => {
    if (i < 0) return;
    const x = (i / Math.max(n - 1, 1)) * 100;
    if (label !== "base") el("b", { style: `left:${x}%` }, stages);
    el("span", { style: `left:${label === "base" ? 0 : x + 4}%`, text: label === "base" ? "" : label }, stages);
  });
  const momentIdx = () => moments.filter((m) => m.prompt === prompt).map((m) => ckpts.findIndex((c) => c.id === m.ckpt)).filter((i) => i >= 0);
  let scrub = makeScrub();
  function makeScrub() {
    return scrubber({
      range: document.getElementById("a1-range"),
      play: document.getElementById("a1-play"),
      ticks: document.getElementById("a1-ticks"),
      count: n,
      moments: momentIdx(),
      interval: 1100,
      onChange: (i) => { idx = i; update(); },
    });
  }

  // ---------------------------------------------------------- prompts
  const promptChips = chips(document.getElementById("a1-prompts"), data.prompts, {
    onSelect: (i) => {
      prompt = i;
      const r = document.getElementById("a1-range");
      const clone = r.cloneNode(true); // drop old listeners so ticks/moments rebuild for this prompt
      r.replaceWith(clone);
      const p = document.getElementById("a1-play");
      const pc = p.cloneNode(true);
      p.replaceWith(pc);
      scrub = makeScrub();
      scrub.set(idx);
    },
  });

  // ---------------------------------------------------------- benchmark chart
  const benchEl = document.getElementById("a1-bench-chart");
  const refGap = Math.max(1, n * 0.07);
  const refX = (j) => n + 0.4 + (j + 0.5) * refGap;
  const xTicks = [{ v: 0, label: "base", anchor: "start" }];
  const sftEnd = ckpts.map((c, i) => [c, i]).filter(([c]) => c.stage === "sft").pop();
  const dpoEnd = ckpts.map((c, i) => [c, i]).filter(([c]) => c.stage === "dpo").pop();
  if (sftEnd) xTicks.push({ v: sftEnd[1], label: `SFT ${sftEnd[0].step.toLocaleString("en-US")}` });
  if (dpoEnd) xTicks.push({ v: dpoEnd[1], label: `DPO ${dpoEnd[0].step}` });
  const regions = [];
  if (firstOf("sft") >= 0) regions.push({ x0: firstOf("sft") - 0.5, x1: (sftEnd ? sftEnd[1] : n - 1) + 0.5, label: "SFT" });
  if (firstOf("dpo") >= 0) regions.push({ x0: firstOf("dpo") - 0.5, x1: n - 0.5, label: "DPO" });
  const bench = lineChart(benchEl, {
    height: 240,
    margin: { left: 36, right: 14, bottom: 58 },
    x: { min: -0.5, max: n + 0.4 + refs.length * refGap, ticks: xTicks },
    y: { min: 0, max: 0.8, ticks: [0, 0.2, 0.4, 0.6, 0.8], fmt: (v) => pct(v), tipFmt: (v) => pct(v, 1) },
    regions,
    legend: true,
    series: [
      { id: "ifeval", name: "IFEval (instruction following)", color: "var(--s1)", points: ckpts.map((c, i) => [i, c.ifeval]) },
      { id: "gsm8k", name: "GSM8K (math word problems)", color: "var(--s2)", points: ckpts.map((c, i) => [i, c.gsm8k]) },
    ],
    refs: refs.flatMap((r, j) => [
      { x: refX(j), y: r.ifeval, color: "var(--s1)", label: `${r.name}, IFEval` },
      { x: refX(j), y: r.gsm8k, color: "var(--s2)", label: `${r.name}, GSM8K` },
    ]),
    refLabels: refs.map((r, j) => ({ x: refX(j), label: r.name.replace(" (SFT+DPO+RLVR)", "") })),
    xLabel: (x) => (x < n ? ckptName(ckpts[Math.round(x)]) : refs.find((r, j) => Math.abs(refX(j) - x) < 1e-6)?.name || ""),
    snap: 0.01,
    cursor: 0,
    onClick: (x) => { if (x < n) scrub.set(Math.round(x), true); },
    ariaLabel: "IFEval and GSM8K accuracy at every checkpoint",
  });
  tableToggle(document.getElementById("a1-bench"), () => ({
    head: ["Checkpoint", "IFEval", "GSM8K"],
    rows: ckpts.map((c) => [ckptName(c), pct(c.ifeval, 1), pct(c.gsm8k, 1)]).concat(refs.map((r) => [r.name, pct(r.ifeval, 1), pct(r.gsm8k, 1)])),
  }));

  // ---------------------------------------------------------- histogram
  const histEl = document.getElementById("a1-hist");
  const yMax = Math.max(...ckpts.concat(refs).map((c) => Math.max(0, ...Object.values(c.number.hist || {}))));
  const histTable = tableToggle(document.getElementById("a1-random"), () => {
    const c = ckpts[idx];
    return { head: ["Number", "Times picked"], rows: Object.entries(c.number.hist || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, String(v)]) };
  });

  function drawHist(c) {
    histEl.replaceChildren();
    const width = Math.max(histEl.clientWidth, 260);
    const height = 170;
    const m = { top: 18, right: 6, bottom: 22, left: 6 };
    const W = width - m.left - m.right;
    const H = height - m.top - m.bottom;
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Histogram of numbers picked" }, histEl);
    const bw = W / 100;
    const hist = c.number.hist || {};
    el("line", { x1: m.left, x2: m.left + W, y1: m.top + H, y2: m.top + H, stroke: "var(--axis)" }, svg);
    for (let v = 1; v <= 100; v++) {
      const count = hist[v] || 0;
      const h = (count / yMax) * H;
      const x = m.left + (v - 1) * bw;
      if (count) {
        el("rect", {
          x: x + Math.min(1, bw * 0.15), y: m.top + H - h, width: Math.max(bw - Math.min(2, bw * 0.3), 1), height: h, rx: Math.min(2, bw / 3),
          fill: v === 42 ? "var(--accent)" : "var(--gray-mark)",
        }, svg);
      }
      const hit = el("rect", { x, y: m.top, width: bw, height: H, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, (t) => tipRow(t, null, `${count} ×`, `picked ${v}`)));
      hit.addEventListener("pointerleave", hideTip);
    }
    [1, 25, 50, 75, 100].forEach((v) => el("text", { x: m.left + (v - 0.5) * bw, y: height - 6, "text-anchor": "middle", text: String(v) }, svg));
    const c42 = hist[42] || 0;
    if (c42 > 0) {
      el("text", { x: m.left + 41.5 * bw, y: m.top + H - (c42 / yMax) * H - 5, "text-anchor": "middle", text: `42 ×${c42}`, class: "label-strong" }, svg);
    }
    const top = Object.entries(hist).sort((a, b) => b[1] - a[1])[0];
    const stat = document.getElementById("a1-hist-stat");
    stat.replaceChildren();
    if (!c.number.valid) {
      stat.textContent = "No valid numbers at this checkpoint.";
      return;
    }
    stat.append("Favorite: ");
    el("b", { text: top ? top[0] : "–" }, stat);
    stat.append(` (${top ? Math.round((top[1] / c.number.valid) * 100) : 0}% of answers) · `);
    el("b", { text: String(c.number.distinct) }, stat);
    stat.append(` distinct numbers in ${c.number.valid} valid answers · ${c.number.entropy.toFixed(1)} bits of entropy (uniform would be 6.6)`);
  }
  new ResizeObserver(() => drawHist(ckpts[idx])).observe(histEl);

  // ---------------------------------------------------------- jokes
  function drawJokes(c) {
    const list = document.getElementById("a1-joke-list");
    list.replaceChildren();
    const j = c.joke;
    const raws = j.top_raw || [];
    j.top.slice(0, 5).forEach(([key, count], k) => {
      const row = el("div", { class: "joke" }, list);
      const raw = (raws[k] || "").split("\n").find((l) => l.trim() && !/^(sure|of course|here'?s|okay|certainly)\b/i.test(l.trim())) || key;
      el("span", { class: "jt", text: raw.length > 140 ? raw.slice(0, 140) + "…" : raw, title: raws[k] || key }, row);
      el("span", { class: "jn", text: `${count} of ${j.valid}` }, row);
      const bar = el("span", { class: "jb" }, row);
      el("i", { style: `width:${(count / Math.max(j.valid, 1)) * 100}%` }, bar);
    });
    let summary = list.nextElementSibling;
    if (!summary || !summary.classList.contains("joke-summary")) {
      summary = el("p", { class: "joke-summary" });
      list.after(summary);
    }
    summary.textContent = `${j.distinct} different jokes in ${j.valid} tries.`;
  }

  // ---------------------------------------------------------- answer
  const answerEl = document.getElementById("a1-answer");
  let pinned = null;
  async function drawAnswer() {
    const tl = await load(`act1/timelapse-${String(prompt).padStart(2, "0")}.json`);
    const c = ckpts[idx];
    document.getElementById("a1-prompt-text").textContent = `“${data.prompts[prompt]}”`;
    if (!tl) return;
    const frame = tl.frames.find((f) => f.id === c.id);
    answerEl.replaceChildren();
    if (!frame) return;
    const toks = visibleTokens(frame);
    const hasBase = frame.blp && frame.blp.length;
    if (!frame.text.trim()) {
      el("span", { class: "empty", text: "(It ends the turn without writing anything: no answer at all.)" }, answerEl);
    }
    toks.forEach(({ t, lp, blp }) => {
      const span = el("span", { class: "tok", text: t }, answerEl);
      if (hasBase && blp != null && c.stage !== "base") {
        const r = lp - blp;
        if (r > 0) span.style.backgroundColor = `rgba(var(--shade), ${(Math.min(r / SHADE_FULL, 1) * 0.55).toFixed(3)})`;
        else if (r < -1) span.style.backgroundColor = `rgba(var(--shade-neg), ${(Math.min(-r / SHADE_FULL, 1) * 0.4).toFixed(3)})`;
      }
      const show = (e) => showTip(e.clientX, e.clientY, (tt) => tokenTip(tt, t, lp, blp, c));
      span.addEventListener("pointermove", show);
      span.addEventListener("pointerleave", () => { if (pinned !== span) hideTip(); });
      span.addEventListener("click", (e) => {
        if (pinned) pinned.classList.remove("pinned");
        pinned = pinned === span ? null : span;
        if (pinned) { span.classList.add("pinned"); show(e); } else hideTip();
      });
    });
    if (frame.finish === "length") el("span", { class: "trunc", text: " … (cut off at 384 tokens)" }, answerEl);
    const meta = document.getElementById("a1-answer-meta");
    meta.textContent = `${toks.length} tokens${hasBase ? "" : " · shading appears once base-model scores are in"}`;
    const m = moments.find((mm) => mm.prompt === prompt && mm.ckpt === c.id);
    const mEl = document.getElementById("a1-moment");
    mEl.hidden = !m;
    if (m) mEl.textContent = m.text;
  }

  function update() {
    const c = ckpts[idx];
    document.getElementById("a1-ckpt-label").textContent = ckptName(c);
    document.getElementById("a1-ckpt-detail").textContent = ckptDetail(c, lastSft);
    bench.setCursor(idx);
    drawHist(c);
    histTable.refresh();
    drawJokes(c);
    drawAnswer();
  }

  // ---------------------------------------------------------- under the hood
  const hood = document.getElementById("a1-hood");
  const ul = el("ul", {}, hood);
  [
    "Base model: allenai/OLMo-2-0425-1B. Fully open: weights, pretraining data, and the exact post-training datasets.",
    "SFT: AI2's 1B recipe (lr 3e-5, linear decay, 3% warmup, ~128 sequences per step, 4,096-token context), fp32 master weights with bf16 autocast, loss on assistant tokens only.",
    "DPO: length-normalized DPO (AI2's dpo_norm, β = 5, lr 2.5e-6, 128 pairs per step) on a random subset of AI2's 1B preference mix.",
    "Budget cuts: 58% of one SFT epoch (AI2 trained two) and 16% of the preference pairs.",
    "Every checkpoint is answered greedily through the chat template, so the base model sees exactly what the assistant sees. Token shading is log p(checkpoint) − log p(base) for the token the checkpoint chose.",
    "Benchmarks: IFEval prompt-level loose accuracy and 0-shot chain-of-thought GSM8K. Our harness gives AI2's DPO model 67.1% on IFEval, exactly its published number.",
  ].forEach((t) => el("li", { text: t }, ul));

  promptChips.select(0);
  window.addEventListener("act1:goto", (e) => { promptChips.select(e.detail.prompt || 0); scrub.set(e.detail.idx || 0, true); });
}

function tokenTip(t, tok, lp, blp, c) {
  const shown = tok.replace(/\n/g, "⏎").replace(/ /g, "·");
  const row = el("div", { class: "row" }, t);
  el("span", { class: "tok-shown", text: shown }, row);
  const p = Math.exp(lp);
  el("div", { text: `${ckptName(c)}: p = ${fmtP(p)}` }, t);
  if (blp != null) {
    el("div", { class: "tl", text: `base model: p = ${fmtP(Math.exp(blp))}` }, t);
    const r = lp - blp;
    const ratio = Math.exp(Math.abs(r));
    el("div", { class: "tv", text: r >= 0 ? `${fmtRatio(ratio)}× more likely` : `${fmtRatio(ratio)}× less likely` }, t);
  }
}
const fmtP = (p) => (p >= 0.01 ? p.toFixed(2) : p.toExponential(1));
const fmtRatio = (r) => (r >= 100 ? Math.round(r).toLocaleString("en-US") : r >= 10 ? r.toFixed(0) : r.toFixed(1));
