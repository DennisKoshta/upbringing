// Small, dependency-free SVG chart kit. Marks follow the data-viz spec: 2px lines, >=8px dots with a 2px surface
// ring, hairline solid grid, crosshair + one tooltip listing every series, and a table view for every chart.

const NS = "http://www.w3.org/2000/svg";
const cache = new Map();

export async function load(path) {
  if (!cache.has(path)) {
    cache.set(path, fetch(`data/${path}`).then((r) => (r.ok ? r.json() : null)).catch(() => null));
  }
  return cache.get(path);
}

export function el(tag, attrs = {}, parent) {
  const isSvg = ["svg", "g", "line", "path", "rect", "circle", "text", "polyline", "clipPath", "defs"].includes(tag);
  const node = isSvg ? document.createElementNS(NS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "text") node.textContent = v;
    else if (k === "class") node.setAttribute("class", v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const pct = (v, d = 0) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
export const fmtInt = (v) => (v == null ? "–" : Math.round(v).toLocaleString("en-US"));
export function scale(d0, d1, r0, r1) {
  const k = (r1 - r0) / ((d1 - d0) || 1);
  const f = (v) => r0 + (v - d0) * k;
  f.invert = (p) => d0 + (p - r0) / k;
  return f;
}
export function niceMax(v, step) {
  return Math.ceil(v / step) * step || step;
}

// ---------------------------------------------------------------- tooltip

const tip = () => document.getElementById("tooltip");
export function showTip(x, y, build) {
  const t = tip();
  t.replaceChildren();
  build(t);
  t.hidden = false;
  const r = t.getBoundingClientRect();
  let left = x + 14;
  let top = y + 14;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
  t.style.left = `${Math.max(8, left)}px`;
  t.style.top = `${Math.max(8, top)}px`;
}
export function hideTip() {
  tip().hidden = true;
}
export function tipRow(parent, color, value, label) {
  const row = el("div", { class: "row" }, parent);
  if (color) el("span", { class: "key", style: `background:${color}` }, row);
  el("span", { class: "tv", text: value }, row);
  if (label) el("span", { class: "tl", text: label }, row);
  return row;
}

// ---------------------------------------------------------------- table view

export function tableToggle(container, getTable) {
  const btn = el("button", { class: "table-toggle", type: "button", text: "Show data table" }, container);
  const wrap = el("div", { class: "table-wrap", hidden: "" }, container);
  btn.addEventListener("click", () => {
    const open = wrap.hidden;
    wrap.hidden = !open;
    btn.textContent = open ? "Hide data table" : "Show data table";
    if (open) {
      const { head, rows } = getTable();
      wrap.replaceChildren();
      const t = el("table", { class: "data" }, wrap);
      const tr = el("tr", {}, el("thead", {}, t));
      head.forEach((h) => el("th", { text: h }, tr));
      const tb = el("tbody", {}, t);
      rows.forEach((r) => {
        const row = el("tr", {}, tb);
        r.forEach((c) => el("td", { text: c }, row));
      });
    }
  });
  return { refresh() { if (!wrap.hidden) { wrap.hidden = true; btn.click(); } } };
}

// ---------------------------------------------------------------- line chart

/**
 * opts: { height, margin, x:{min,max,ticks:[{v,label}]}, y:{min,max,ticks:[v],fmt},
 *   series:[{id,name,color,points:[[x,y]],faint,dots,width}], regions:[{x0,x1,label}],
 *   refs:[{x,y,color,label}], cursor, xLabel(x), onHover(x|null), onClick(x), legend, endLabels }
 */
export function lineChart(container, opts) {
  const root = el("div", { class: "chart" }, container);
  let state = { ...opts };
  let hoverX = null;
  let svg;
  let geom;

  function render() {
    root.replaceChildren();
    const width = Math.max(root.clientWidth || container.clientWidth || 300, 200);
    const height = state.height || 220;
    const m = { top: 10, right: 12, bottom: 26, left: 36, ...(state.margin || {}) };
    const W = width - m.left - m.right;
    const H = height - m.top - m.bottom;
    const sx = scale(state.x.min, state.x.max, m.left, m.left + W);
    const sy = scale(state.y.min, state.y.max, m.top + H, m.top);
    geom = { sx, sy, m, W, H, width, height };

    if (state.legend === "space") root.appendChild(el("div", { class: "legend" }));
    else if (state.legend) {
      const lg = el("div", { class: "legend" });
      state.series.filter((s) => !s.noLegend).forEach((s) => {
        const sp = el("span", {}, lg);
        el("i", { style: `background:${s.color}` }, sp);
        sp.appendChild(document.createTextNode(s.name));
      });
      root.appendChild(lg);
    }
    svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": state.ariaLabel || "" }, root);

    (state.regions || []).forEach((r) => {
      el("rect", { class: "region", x: sx(r.x0), y: m.top, width: Math.max(sx(r.x1) - sx(r.x0), 0), height: H, rx: 4 }, svg);
      if (r.label) el("text", { x: sx(r.x0) + 6, y: m.top + 12, text: r.label, class: "label-ink" }, svg);
    });
    const grid = el("g", { class: "grid" }, svg);
    (state.y.ticks || []).forEach((v) => {
      el("line", { x1: m.left, x2: m.left + W, y1: sy(v), y2: sy(v) }, grid);
      el("text", { x: m.left - 6, y: sy(v) + 3.5, "text-anchor": "end", text: (state.y.fmt || String)(v) }, svg);
    });
    const axis = el("g", { class: "axis" }, svg);
    el("line", { x1: m.left, x2: m.left + W, y1: m.top + H, y2: m.top + H }, axis);
    (state.x.ticks || []).forEach((t) => {
      el("line", { x1: sx(t.v), x2: sx(t.v), y1: m.top + H, y2: m.top + H + 4 }, axis);
      el("text", { x: sx(t.v), y: m.top + H + 16, "text-anchor": t.anchor || "middle", text: t.label }, svg);
    });

    (state.intervals || []).forEach((iv) => {
      el("line", { class: "ci", x1: sx(iv.x), x2: sx(iv.x), y1: sy(iv.y0), y2: sy(iv.y1), stroke: iv.color, opacity: 0.45 }, svg);
    });
    state.series.forEach((s) => {
      if (!s.points.length) return;
      const d = s.points.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");
      el("path", { d, class: s.faint ? "series-faint" : "series-line", stroke: s.color, "stroke-width": s.width || null }, svg);
      if (s.dots) s.points.forEach(([x, y]) => el("circle", { class: "dot", cx: sx(x), cy: sy(y), r: 4, fill: s.color }, svg));
      if (state.endLabels && !s.faint && !s.noLegend) {
        const [lx, ly] = s.points[s.points.length - 1];
        el("text", { x: sx(lx) + 6, y: sy(ly) + 4, text: s.endLabel || s.name, class: "label-ink" }, svg);
      }
    });
    (state.refs || []).forEach((r) => {
      el("circle", { class: "dot", cx: sx(r.x), cy: sy(r.y), r: 4.5, fill: r.color }, svg);
    });
    (state.refLabels || []).forEach((r) => {
      el("text", { x: sx(r.x) + 3, y: m.top + H + 12, "text-anchor": "end", transform: `rotate(-55 ${sx(r.x) + 3} ${m.top + H + 12})`, text: r.label }, svg);
    });

    if (state.cursor != null) {
      el("line", { class: "cursor-line", x1: sx(state.cursor), x2: sx(state.cursor), y1: m.top, y2: m.top + H }, svg);
    }
    const hover = el("line", { class: "hover-line", y1: m.top, y2: m.top + H, visibility: "hidden" }, svg);
    const hit = el("rect", { x: m.left, y: m.top, width: W, height: H, fill: "transparent", style: "cursor:crosshair" }, svg);

    const xs = [...new Set(state.series.flatMap((s) => s.points.map((p) => p[0])).concat((state.refs || []).map((r) => r.x)))].sort((a, b) => a - b);
    const nearestX = (px) => {
      const v = sx.invert(px);
      let best = xs[0];
      for (const x of xs) if (Math.abs(x - v) < Math.abs(best - v)) best = x;
      return best;
    };
    const showAt = (x, cx, cy) => {
      hover.setAttribute("x1", sx(x));
      hover.setAttribute("x2", sx(x));
      hover.setAttribute("visibility", "visible");
      if (cx == null) return;
      showTip(cx, cy, (t) => {
        el("div", { class: "tl", text: state.xLabel ? state.xLabel(x) : String(x) }, t);
        state.series.forEach((s) => {
          const p = s.points.find((q) => q[0] === x) || nearestPoint(s.points, x);
          if (p) tipRow(t, s.color, (state.y.tipFmt || state.y.fmt || String)(p[1]), s.name);
        });
        (state.refs || []).filter((r) => r.x === x).forEach((r) => tipRow(t, r.color, (state.y.tipFmt || state.y.fmt || String)(r.y), r.label));
      });
    };
    geom.showAt = (x) => showAt(x);
    geom.hide = () => hover.setAttribute("visibility", "hidden");
    hit.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * width;
      hoverX = nearestX(px);
      showAt(hoverX, e.clientX, e.clientY);
      state.onHover && state.onHover(hoverX);
    });
    hit.addEventListener("pointerleave", () => {
      hoverX = null;
      hover.setAttribute("visibility", "hidden");
      hideTip();
      state.onHover && state.onHover(null);
    });
    if (state.onClick) hit.addEventListener("click", () => hoverX != null && state.onClick(hoverX));
  }

  function nearestPoint(points, x) {
    let best = null;
    for (const p of points) if (!best || Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p;
    return best && Math.abs(best[0] - x) <= (state.snap ?? Infinity) ? best : null;
  }

  const ro = new ResizeObserver(() => render());
  ro.observe(container);
  render();
  return {
    update(next) { state = { ...state, ...next }; render(); },
    setCursor(x) { state.cursor = x; render(); },
    showHover(x) { if (x == null) geom.hide(); else geom.showAt(x); },
  };
}

// ---------------------------------------------------------------- diverging weight bars (Bradley-Terry)

export function weightBars(container, names, labels, weights, { max = 2, stated = [] } = {}) {
  container.replaceChildren();
  names.forEach((n) => {
    const w = weights ? weights[n] || 0 : 0;
    const row = el("div", { class: `weight${stated.includes(n) ? " stated" : ""}` }, container);
    el("span", { class: "wn", text: labels[n] }, row);
    const bar = el("span", { class: "wbar" }, row);
    const frac = clamp(Math.abs(w) / max, 0, 1) * 50;
    el("i", { class: w >= 0 ? "pos" : "neg", style: w >= 0 ? `left:50%;width:${frac}%` : `left:${50 - frac}%;width:${frac}%` }, bar);
    el("span", { class: "wv", text: weights ? (w >= 0 ? "+" : "−") + Math.abs(w).toFixed(2) : "–" }, row);
  });
}

// ---------------------------------------------------------------- scrubber helper

export function scrubber({ range, play, ticks, count, onChange, interval = 900, moments = [] }) {
  range.max = String(count - 1);
  let timer = null;
  const tickEls = [];
  if (ticks) {
    ticks.replaceChildren();
    for (let i = 0; i < count; i++) {
      const t = el("i", { class: moments.includes(i) ? "mk" : null, style: `left:${(i / Math.max(count - 1, 1)) * 100}%` }, ticks);
      tickEls.push(t);
    }
  }
  const set = (i, fromUser) => {
    range.value = String(i);
    tickEls.forEach((t, j) => t.classList.toggle("passed", j <= i));
    onChange(Number(i), fromUser);
  };
  const stop = () => {
    clearInterval(timer);
    timer = null;
    play.classList.remove("playing");
    play.setAttribute("aria-label", "Play");
  };
  range.addEventListener("input", () => { stop(); set(Number(range.value), true); });
  play.addEventListener("click", () => {
    if (timer) return stop();
    if (Number(range.value) >= count - 1) set(0, true);
    play.classList.add("playing");
    play.setAttribute("aria-label", "Pause");
    timer = setInterval(() => {
      const next = Number(range.value) + 1;
      if (next >= count) return stop();
      set(next, true);
    }, interval);
  });
  return { set, stop, get value() { return Number(range.value); } };
}

export function chips(container, items, { selected = 0, onSelect, role = "tab" } = {}) {
  container.replaceChildren();
  const btns = items.map((label, i) => {
    const b = el("button", { type: "button", role, "aria-selected": String(i === selected), text: label }, container);
    b.addEventListener("click", () => select(i));
    return b;
  });
  function select(i) {
    btns.forEach((b, j) => b.setAttribute("aria-selected", String(i === j)));
    onSelect && onSelect(i);
  }
  return { select };
}
