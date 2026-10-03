// charts.js — hand-rolled SVG chart helpers (no external library). Globals used
// by dashboard.js and farmer.js. Charts scale to their container via viewBox;
// callers drop the returned <svg> (width:100%) into a sized wrapper. The monsoon
// timeline, calendar outlook, map legend and sowing-window bar are rendered as
// CSS/DOM in the renderers (see styles.css), not here.
const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, a) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in (a || {})) e.setAttribute(k, a[k]);
  return e;
}
function mkSVG(w, h) {
  const s = svgEl("svg", { viewBox: `0 0 ${w} ${h}` });
  s.setAttribute("style", "width:100%;height:auto;display:block;overflow:visible");
  return s;
}
function svgText(x, y, str, style) {
  const e = svgEl("text", { x, y });
  if (style) e.setAttribute("style", style);
  e.textContent = str;
  return e;
}
const AXIS_TXT = "font:600 11px system-ui,sans-serif;fill:#6b7280";
const VAL_TXT = "font:700 10px system-ui,sans-serif;fill:#0f172a";

// Vertical bar chart. items:[{label,value,color?,strong?}] opts:{max,unit,height}
function barChart(host, items, opts = {}) {
  host.innerHTML = "";
  const W = 340, H = opts.height || 170, L = 30, R = 10, T = 14, B = 26;
  const max = opts.max || Math.max(1, ...items.map((d) => d.value));
  const svg = mkSVG(W, H);
  const iw = W - L - R, ih = H - T - B, n = items.length, bw = iw / n;
  for (let g = 0; g <= 2; g++) {
    const y = T + ih * (g / 2);
    svg.appendChild(svgEl("line", { x1: L, y1: y, x2: W - R, y2: y, stroke: "#eef1f0", "stroke-width": 1 }));
    svg.appendChild(svgText(L - 5, y + 3, Math.round(max * (1 - g / 2)), AXIS_TXT + ";text-anchor:end"));
  }
  items.forEach((d, i) => {
    const h = ih * (d.value / max);
    const x = L + i * bw + bw * 0.2, w = bw * 0.6, y = T + ih - h;
    svg.appendChild(svgEl("rect", { x, y, width: w, height: Math.max(1, h), rx: 4, fill: d.color || "#3b82f6" }));
    if (d.value > 0 || d.strong)
      svg.appendChild(svgText(x + w / 2, y - 4, d.value + (opts.unit || ""), VAL_TXT + ";text-anchor:middle"));
    svg.appendChild(svgText(x + w / 2, H - 9, d.label, AXIS_TXT + ";text-anchor:middle;font-size:9px"));
  });
  host.appendChild(svg);
}
// Dual-axis chart: blue bars (left axis, mm) + a probability line (right axis, 0..1→%).
// opts:{bars:[{label,value}], line:[0..1], markIdx, barColor, lineColor, height}
function dualAxis(host, opts) {
  host.innerHTML = "";
  const bars = opts.bars || [], line = opts.line || [];
  const W = 360, H = opts.height || 190, L = 32, R = 34, T = 14, B = 28;
  const iw = W - L - R, ih = H - T - B, n = bars.length, bw = iw / Math.max(1, n);
  const maxB = Math.max(1, ...bars.map((d) => d.value));
  const svg = mkSVG(W, H);
  for (let g = 0; g <= 2; g++) {
    const y = T + ih * (g / 2);
    svg.appendChild(svgEl("line", { x1: L, y1: y, x2: W - R, y2: y, stroke: "#eef1f0", "stroke-width": 1 }));
    svg.appendChild(svgText(L - 5, y + 3, Math.round(maxB * (1 - g / 2)), AXIS_TXT + ";text-anchor:end"));
    svg.appendChild(svgText(W - R + 5, y + 3, Math.round(100 * (1 - g / 2)) + "%", AXIS_TXT + ";text-anchor:start;fill:#15803d"));
  }
  bars.forEach((d, i) => {
    const h = ih * (d.value / maxB), x = L + i * bw + bw * 0.18, w = bw * 0.64, y = T + ih - h;
    const sel = i === opts.markIdx;
    svg.appendChild(svgEl("rect", { x, y, width: w, height: Math.max(1, h), rx: 2,
      fill: sel ? "#1d4ed8" : (opts.barColor || "#60a5fa"), opacity: sel ? 1 : 0.9 }));
    if (i % 5 === 0 || sel)
      svg.appendChild(svgText(x + w / 2, H - 9, d.label, AXIS_TXT + ";text-anchor:middle;font-size:9px"));
  });
  const pts = line.map((v, i) => `${(L + i * bw + bw / 2).toFixed(1)},${(T + ih * (1 - Math.max(0, Math.min(1, v)))).toFixed(1)}`);
  svg.appendChild(svgEl("polyline", { points: pts.join(" "), fill: "none",
    stroke: opts.lineColor || "#15803d", "stroke-width": 2.4, "stroke-linejoin": "round" }));
  if (opts.markIdx != null && line[opts.markIdx] != null) {
    const cx = L + opts.markIdx * bw + bw / 2, cy = T + ih * (1 - Math.max(0, Math.min(1, line[opts.markIdx])));
    svg.appendChild(svgEl("circle", { cx, cy, r: 3.5, fill: "#fff", stroke: opts.lineColor || "#15803d", "stroke-width": 2 }));
  }
  host.appendChild(svg);
}

// Minimal filled sparkline. values:number[]; opts:{color,fill,height}
function sparkline(host, values, opts = {}) {
  host.innerHTML = "";
  const W = 300, H = opts.height || 54, P = 4;
  const vals = values && values.length ? values : [0];
  const mx = Math.max(...vals), mn = Math.min(...vals), span = (mx - mn) || 1;
  const svg = mkSVG(W, H);
  const xy = (v, i) => [P + (W - 2 * P) * (i / Math.max(1, vals.length - 1)), H - P - (H - 2 * P) * ((v - mn) / span)];
  const pts = vals.map((v, i) => xy(v, i).map((n) => n.toFixed(1)).join(","));
  const [x0] = xy(vals[0], 0), xN = xy(vals[vals.length - 1], vals.length - 1)[0];
  svg.appendChild(svgEl("polygon", { points: `${x0},${H - P} ${pts.join(" ")} ${xN},${H - P}`,
    fill: opts.fill || "#15803d", opacity: 0.14 }));
  svg.appendChild(svgEl("polyline", { points: pts.join(" "), fill: "none",
    stroke: opts.color || "#15803d", "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round" }));
  host.appendChild(svg);
}

