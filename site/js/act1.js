// Section 1: one training clock (the scrubber) drives the answer, the scorecard, the random-number histogram and the
// jokes. "Ask" buttons draw from the 300 real samples taken at the selected checkpoint.
import { chips, compactStep, el, hideTip, load, pct, player, showTip, stepper, tableToggle, tipRow } from "./lib.js";
import { ckptDetail, ckptName, ckptShort, visibleTokens } from "./ckpt.js";
import { initGlossary } from "./glossary.js";

const REFUSAL = /\b(sorry|can'?t fulfill|cannot fulfill|unable to)\b/i;
const SHADE_FULL = 8; // nats: a token ~3000x more likely than under the base model gets the full shade
const TESTS = [
  { key: "ifeval", name: "Instruction following", term: "ifeval", label: "IFEval" },
  { key: "gsm8k", name: "Grade-school math", term: "gsm8k", label: "GSM8K" },
];

export async function initAct1() {
  const data = await load("act1/checkpoints.json");
  if (!data) return;
  const moments = (await load("act1/moments.json")) || [];
  const ckpts = data.checkpoints;
  const refs = data.references;
  const ai2 = refs.find((r) => r.id === "ai2-dpo") || refs[0];
  const n = ckpts.length;
  const lastSft = Math.max(0, ...ckpts.filter((c) => c.stage === "sft").map((c) => c.step));
  document.getElementById("a1-nckpt").textContent = String(n - 1);

  let prompt = 0;
  let idx = 0;

  // ---------------------------------------------------------- timeline: every snapshot is a labeled stop
  const momentIdx = () => moments.filter((m) => m.prompt === prompt).map((m) => ckpts.findIndex((c) => c.id === m.ckpt)).filter((i) => i >= 0);
  const stepItems = ckpts.map((c) => ({ group: c.stage, label: c.stage === "base" ? "0" : compactStep(c.step), aria: ckptName(c) }));
  let scrub = null;
  function makeScrub() {
    scrub = stepper(document.getElementById("a1-stepper"), stepItems, {
      groups: { base: "Base", sft: "SFT steps", dpo: "DPO steps" },
      marks: momentIdx(),
      minSlot: 34,
      selected: idx,
      onChange: (i) => { idx = i; update(); },
    });
  }
  makeScrub();
  player(document.getElementById("a1-play"), () => scrub, 1100);

  // ---------------------------------------------------------- prompts
  const promptChips = chips(document.getElementById("a1-prompts"), data.prompts, {
    onSelect: (i) => {
      prompt = i;
      makeScrub(); // the marked stops depend on the question
      update();
    },
  });

  // ---------------------------------------------------------- scorecard
  const card = document.getElementById("a1-scorecard");
  const base = ckpts[0];
  const rows = TESTS.map((t) => {
    const row = el("div", { class: "score" }, card);
    const head = el("div", { class: "score-head" }, row);
    const nm = el("span", { class: "score-name" }, head);
    nm.append(t.name + " ");
    el("span", { class: "term", "data-t": t.term, text: t.label }, nm);
    const val = el("span", { class: "score-val" }, head);
    const track = el("div", { class: "score-track" }, row);
    const fill = el("i", { class: "score-fill" }, track);
    const mk = (v, cls, label) => {
      const m = el("span", { class: `score-mark ${cls}`, style: `left:${v * 100}%` }, track);
      el("span", { class: "score-mark-label", text: label }, m);
    };
    mk(base[t.key], "base", `base model ${pct(base[t.key])}`);
    if (ai2) mk(ai2[t.key], "ref", `AI2's model ${pct(ai2[t.key])}`);
    const spark = el("div", { class: "spark", title: "Trend over training (click to jump)" }, row);
    return { t, val, fill, spark };
  });
  initGlossary(card);

  function drawScores() {
    rows.forEach(({ t, val, fill, spark }) => {
      const c = ckpts[idx];
      val.textContent = pct(c[t.key]);
      fill.style.width = `${c[t.key] * 100}%`;
      drawSpark(spark, ckpts.map((k) => k[t.key]), idx, (i) => scrub.set(i, true));
    });
  }
  tableToggle(document.getElementById("a1-scores"), () => ({
    head: ["Snapshot", "Instructions (IFEval)", "Math (GSM8K)"],
    rows: ckpts.map((c) => [ckptName(c), pct(c.ifeval, 1), pct(c.gsm8k, 1)]).concat(refs.map((r) => [r.name, pct(r.ifeval, 1), pct(r.gsm8k, 1)])),
  }));

  // ---------------------------------------------------------- random numbers
  const histEl = document.getElementById("a1-hist");
  const yMax = Math.max(...ckpts.map((c) => Math.max(0, ...Object.values(c.number.hist || {}))));
  let lastDraw = null;
  function drawHist(c) {
    histEl.replaceChildren();
    const width = Math.max(histEl.clientWidth, 260);
    const height = 160;
    const m = { top: 18, right: 4, bottom: 22, left: 4 };
    const W = width - m.left - m.right;
    const H = height - m.top - m.bottom;
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "How often each number was picked" }, histEl);
    const bw = W / 100;
    const hist = c.number.hist || {};
    const fair = c.number.valid / 100;
    el("line", { x1: m.left, x2: m.left + W, y1: m.top + H, y2: m.top + H, stroke: "var(--axis)" }, svg);
    for (let v = 1; v <= 100; v++) {
      const count = hist[v] || 0;
      const h = (count / yMax) * H;
      const x = m.left + (v - 1) * bw;
      const hot = v === 42 || v === lastDraw;
      if (count) {
        el("rect", {
          x: x + Math.min(1, bw * 0.15), y: m.top + H - h, width: Math.max(bw - Math.min(2, bw * 0.3), 1), height: h,
          rx: Math.min(2, bw / 3), fill: hot ? "var(--accent)" : "var(--gray-mark)",
        }, svg);
      }
      const hit = el("rect", { x, y: m.top, width: bw, height: H, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, (t) => tipRow(t, null, `${count} of ${c.number.valid}`, `answers were ${v}`)));
      hit.addEventListener("pointerleave", hideTip);
    }
    // what a fair random picker would give: every number equally often
    const fy = m.top + H - (fair / yMax) * H;
    el("line", { x1: m.left, x2: m.left + W, y1: fy, y2: fy, stroke: "var(--ink-2)", "stroke-width": 1 }, svg);
    el("text", { x: m.left + W, y: fy - 4, "text-anchor": "end", text: "a fair random pick", class: "label-ink" }, svg);
    [1, 25, 50, 75, 100].forEach((v) => el("text", { x: m.left + (v - 0.5) * bw, y: height - 6, "text-anchor": "middle", text: String(v) }, svg));
    const c42 = hist[42] || 0;
    if (c42 > 0) el("text", { x: m.left + 41.5 * bw, y: m.top + H - (c42 / yMax) * H - 5, "text-anchor": "middle", text: "42", class: "label-strong" }, svg);

    const stat = document.getElementById("a1-hist-stat");
    stat.replaceChildren();
    if (!c.number.valid) {
      stat.textContent = "No valid numbers at this checkpoint.";
      return;
    }
    const top = Object.entries(hist).sort((a, b) => b[1] - a[1])[0];
    stat.append("Mode: ");
    el("b", { text: top[0] }, stat);
    stat.append(` (${Math.round((top[1] / c.number.valid) * 100)}% of ${c.number.valid} valid answers), ${c.number.distinct} distinct values. ` +
      "A uniform sampler would put about 1% on each.");
    if (c.number.valid < 280) stat.append(` ${300 - c.number.valid} of 300 samples weren't a number from 1 to 100.`);
  }
  new ResizeObserver(() => drawHist(ckpts[idx])).observe(histEl);
  document.getElementById("a1-ask-number").addEventListener("click", () => {
    const c = ckpts[idx];
    const out = document.getElementById("a1-number-out");
    let r = Math.random() * 300;
    lastDraw = null;
    for (const [v, count] of Object.entries(c.number.hist || {})) {
      r -= count;
      if (r < 0) { lastDraw = Number(v); break; }
    }
    if (lastDraw != null) out.textContent = String(lastDraw);
    else {
      const odd = (c.number.examples || []).filter((s) => !/^\s*\d{1,3}\s*$/.test(s));
      out.textContent = odd.length ? `“${truncate(odd[Math.floor(Math.random() * odd.length)].trim(), 80)}”` : "(not a number)";
    }
    out.classList.remove("pop");
    void out.offsetWidth;
    out.classList.add("pop");
    drawHist(c);
  });

  // ---------------------------------------------------------- jokes
  function jokeLine(raw, key) {
    const line = (raw || "").split("\n").map((l) => l.trim()).find((l) => l && !/^(sure|of course|here'?s|okay|certainly|absolutely)\b/i.test(l));
    return line || key;
  }
  function drawJokes(c) {
    const list = document.getElementById("a1-joke-list");
    list.replaceChildren();
    const j = c.joke;
    const raws = j.top_raw || [];
    j.top.slice(0, 4).forEach(([key, count], k) => {
      const row = el("div", { class: "joke" }, list);
      el("span", { class: "jt", text: truncate(jokeLine(raws[k], key), 110), title: raws[k] || key }, row);
      el("span", { class: "jn", text: `${count}×` }, row);
      const bar = el("span", { class: "jb" }, row);
      el("i", { style: `width:${(count / Math.max(j.valid, 1)) * 100}%` }, bar);
    });
    let summary = list.nextElementSibling;
    if (!summary || !summary.classList.contains("joke-summary")) {
      summary = el("p", { class: "joke-summary" });
      list.after(summary);
    }
    summary.textContent = `Most frequent openings above; ${j.distinct} distinct jokes in ${j.valid} samples.`;
    // Over-refusal: early in SFT the model applies the refusal template from the safety data to harmless requests.
    const refusing = j.top.reduce((acc, [key, count], k) => acc + (REFUSAL.test(raws[k] || key) ? count : 0), 0);
    let note = summary.nextElementSibling;
    if (!note || !note.classList.contains("moment")) {
      note = el("p", { class: "moment" });
      summary.after(note);
    }
    note.hidden = refusing / 300 < 0.05;
    note.textContent = `Over-refusal: at least ${Math.round((refusing / 300) * 100)}% of samples here decline to tell a joke. ` +
      "About 13% of the SFT mixture is safety data teaching refusals. The short, repetitive refusal template is learned " +
      "early, before the model learns when to use it, so it fires on harmless requests too. It fades with more SFT.";
  }
  document.getElementById("a1-ask-joke").addEventListener("click", () => {
    const j = ckpts[idx].joke;
    const raws = j.top_raw || [];
    let r = Math.random() * 300;
    let text = null;
    for (let k = 0; k < j.top.length; k++) {
      r -= j.top[k][1];
      if (r < 0) { text = raws[k] || null; break; } // without a verbatim sample, fall through to a random real one
    }
    if (!text) {
      const ex = j.examples || [];
      text = ex.length ? ex[Math.floor(Math.random() * ex.length)] : "";
    }
    const out = document.getElementById("a1-joke-out");
    out.textContent = truncate(text.trim(), 320) || "(no joke)";
    out.classList.remove("pop");
    void out.offsetWidth;
    out.classList.add("pop");
  });

  // ---------------------------------------------------------- answer
  const answerEl = document.getElementById("a1-answer");
  let pinned = null;
  async function drawAnswer() {
    const tl = await load(`act1/timelapse-${String(prompt).padStart(2, "0")}.json`);
    const c = ckpts[idx];
    document.getElementById("a1-prompt-text").textContent = data.prompts[prompt];
    document.getElementById("a1-who").textContent = c.stage === "base" ? "Base model" : `Model, ${ckptShort(c)}`;
    if (!tl) return;
    const frame = tl.frames.find((f) => f.id === c.id);
    answerEl.replaceChildren();
    if (!frame) return;
    const toks = visibleTokens(frame);
    const hasBase = frame.blp && frame.blp.length;
    if (!frame.text.trim()) el("span", { class: "empty", text: "(empty: it ends its turn immediately)" }, answerEl);
    toks.forEach(({ t, lp, blp }) => {
      const span = el("span", { class: "tok", text: t }, answerEl);
      if (hasBase && blp != null && c.stage !== "base") {
        const r = lp - blp;
        if (r > 0) span.style.backgroundColor = `rgba(var(--shade), ${(Math.min(r / SHADE_FULL, 1) * 0.55).toFixed(3)})`;
      }
      const show = (e) => showTip(e.clientX, e.clientY, (tt) => tokenTip(tt, t, lp, blp));
      span.addEventListener("pointermove", show);
      span.addEventListener("pointerleave", () => { if (pinned !== span) hideTip(); });
      span.addEventListener("click", (e) => {
        if (pinned) pinned.classList.remove("pinned");
        pinned = pinned === span ? null : span;
        if (pinned) { span.classList.add("pinned"); show(e); } else hideTip();
      });
    });
    if (frame.finish === "length") el("span", { class: "trunc", text: " … (cut off)" }, answerEl);
    const m = moments.find((mm) => mm.prompt === prompt && mm.ckpt === c.id);
    const mEl = document.getElementById("a1-moment");
    mEl.hidden = !m;
    if (m) mEl.textContent = m.text;
  }

  function update() {
    const c = ckpts[idx];
    document.getElementById("a1-ckpt-label").textContent = ckptName(c);
    document.getElementById("a1-ckpt-detail").textContent = ckptDetail(c, lastSft);
    drawScores();
    drawHist(c);
    drawJokes(c);
    drawAnswer();
  }

  // ---------------------------------------------------------- under the hood
  const hood = document.getElementById("a1-hood");
  const ul = el("ul", {}, hood);
  [
    "Base model: allenai/OLMo-2-0425-1B. Fully open: weights, pretraining data, and the exact post-training datasets.",
    "Example training (SFT): AI2's 1B recipe (lr 3e-5, linear decay, 3% warmup, ~128 sequences per step, 4,096-token context), fp32 master weights with bf16 autocast, loss on assistant tokens only. 6,500 steps, about 58% of one pass over the data (AI2 trained two passes).",
    "Preference training (DPO): length-normalized DPO (AI2's dpo_norm, β = 5, lr 2.5e-6, 128 pairs per step) on 60,000 of AI2's 378,000 preference pairs.",
    "Every snapshot answers greedily through the chat template, so the base model sees exactly what the assistant sees. Shading is log p(snapshot) − log p(base) for each token the snapshot chose.",
    "Tests: IFEval (prompt-level loose accuracy, 541 prompts) and GSM8K (1,319 problems, 0-shot with step-by-step reasoning). Our harness scores AI2's DPO model at 66.4–67.1% on IFEval across two runs (AI2 reports 67.1%). That checks our evaluation, not our training.",
  ].forEach((t) => el("li", { text: t }, ul));

  promptChips.select(0);
}

function drawSpark(node, values, current, onPick) {
  node.replaceChildren();
  const width = Math.max(node.clientWidth, 200);
  const height = 34;
  const pad = 5;
  const max = Math.max(...values, 0.01);
  const x = (i) => pad + (i / Math.max(values.length - 1, 1)) * (width - 2 * pad);
  const y = (v) => height - pad - (v / max) * (height - 2 * pad);
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, "aria-hidden": "true" }, node);
  el("path", { d: values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(""), fill: "none", stroke: "var(--gray-mark)", "stroke-width": 1.5 }, svg);
  el("circle", { cx: x(current), cy: y(values[current]), r: 4, fill: "var(--accent)", stroke: "var(--surface)", "stroke-width": 2 }, svg);
  const hit = el("rect", { x: 0, y: 0, width, height, fill: "transparent", style: "cursor:pointer" }, svg);
  hit.addEventListener("click", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * width;
    onPick(Math.max(0, Math.min(values.length - 1, Math.round(((px - pad) / (width - 2 * pad)) * (values.length - 1)))));
  });
}

function tokenTip(t, tok, lp, blp) {
  const shown = tok.replace(/\n/g, "⏎").replace(/ /g, "·");
  const row = el("div", { class: "row" }, t);
  el("span", { class: "tok-shown", text: shown }, row);
  el("div", { text: `p at this checkpoint: ${fmtP(Math.exp(lp))}` }, t);
  if (blp != null) {
    el("div", { class: "tl", text: `p under the base model: ${fmtP(Math.exp(blp))}` }, t);
    const r = lp - blp;
    const ratio = Math.exp(Math.abs(r));
    if (ratio >= 1.5) el("div", { class: "tv", text: r >= 0 ? `${fmtRatio(ratio)}× more likely than under the base model` : `${fmtRatio(ratio)}× less likely than under the base model` }, t);
  }
}
const fmtP = (p) => (p >= 0.01 ? `${Math.round(p * 100)}%` : p >= 1e-4 ? `${(p * 100).toFixed(2)}%` : "under 0.01%");
const fmtRatio = (r) => (r >= 100 ? Math.round(r).toLocaleString("en-US") : r >= 10 ? r.toFixed(0) : r.toFixed(1));
function truncate(s, k) {
  return s.length > k ? s.slice(0, k - 1).trimEnd() + "…" : s;
}
