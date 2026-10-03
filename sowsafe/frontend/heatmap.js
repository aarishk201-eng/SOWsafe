// heatmap.js — offline-safe smooth heat field over the 8 Palghar block points.
// Low-res IDW rendered to a small offscreen canvas, then upscaled with a blur →
// smooth choropleth-like field with no polygons and no dependencies. Shared by the
// dashboard big map (as a Leaflet overlay on an Esri satellite base) and the mobile
// Block Risk Map (standalone canvas). Globals used by dashboard.js / farmer.js.

// Per-field colour ramps (low→high). Rainfall is a mm ramp; the rest are probability.
const FIELD_RAMPS = {
  rainfall: ["#e5e7eb", "#60a5fa", "#22c55e", "#facc15", "#ef4444"],
  onset:    ["#fee2e2", "#fde68a", "#86efac", "#22c55e", "#15803d"],
  dryspell: ["#dcfce7", "#fef9c3", "#fed7aa", "#fb923c", "#b91c1c"],
  heavy:    ["#dbeafe", "#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"],
};
const RAIN_CAP = 80; // mm/day mapped to the top of the rainfall ramp
const rampCss = (ramp) => `linear-gradient(90deg, ${ramp.join(", ")})`;

function _hex2rgb(h) {
  h = h.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
// Interpolate a colour across a multi-stop ramp. t in 0..1 → [r,g,b].
function rampColor(ramp, t) {
  t = Math.max(0, Math.min(1, t));
  const seg = (ramp.length - 1) * t, i = Math.min(ramp.length - 2, Math.floor(seg)), f = seg - i;
  const a = _hex2rgb(ramp[i]), b = _hex2rgb(ramp[i + 1]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f].map(Math.round);
}
// Normalise a risk-map row to 0..1 for the active field.
function norm01(field, row) {
  if (field === "rainfall") return Math.max(0, Math.min(1, (row.rain || 0) / RAIN_CAP));
  if (field === "onset") return Math.max(0, Math.min(1, row.p_onset || 0));
  if (field === "dryspell") return Math.max(0, Math.min(1, row.p_dryspell || 0));
  return Math.max(0, Math.min(1, row.p_heavy || 0));
}
// Draw the IDW heat field. pts:[{x,y,v}] (v already 0..1), in canvas px.
// opts:{w,h,alpha,blur,power,reach,fade}. Low-res grid → blurred upscale.
function drawField(canvas, pts, ramp, opts = {}) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = canvas.clientWidth || opts.w || 300, H = canvas.clientHeight || opts.h || 300;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!pts.length) return;
  const step = Math.max(6, Math.round(W / 42));
  const gw = Math.ceil(W / step), gh = Math.ceil(H / step);
  const off = document.createElement("canvas");
  off.width = gw; off.height = gh;
  const octx = off.getContext("2d"), img = octx.createImageData(gw, gh);
  const power = opts.power || 2.2, reach = opts.reach || 320, fade = opts.fade !== false;
  const baseA = opts.alpha != null ? opts.alpha : 180;
  for (let gy = 0; gy < gh; gy++) {
    for (let gx = 0; gx < gw; gx++) {
      const px = gx * step + step / 2, py = gy * step + step / 2;
      let wsum = 0, vsum = 0, nearest = 1e9;
      for (const p of pts) {
        const dx = px - p.x, dy = py - p.y, d2 = dx * dx + dy * dy;
        if (d2 < nearest) nearest = d2;
        const w = 1 / Math.pow(d2 + 60, power / 2);
        wsum += w; vsum += w * p.v;
      }
      const v = wsum ? vsum / wsum : 0, rgb = rampColor(ramp, v);
      let a = baseA;
      if (fade) {
        const dpx = Math.sqrt(nearest);
        a = Math.round(baseA * (dpx > reach ? 0 : dpx > reach * 0.6 ? (reach - dpx) / (reach * 0.4) : 1));
      }
      const o = (gy * gw + gx) * 4;
      img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = Math.max(0, a);
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.filter = `blur(${opts.blur != null ? opts.blur : 10}px)`;
  ctx.drawImage(off, 0, 0, gw, gh, 0, 0, W, H);
  ctx.filter = "none";
}
// Equirectangular projection of block lat/lon into a WxH canvas (with padding).
function projBounds(rows, W, H, pad = 0.12) {
  const lats = rows.map((r) => r.lat), lons = rows.map((r) => r.lon);
  let m1 = Math.min(...lats), m2 = Math.max(...lats), n1 = Math.min(...lons), n2 = Math.max(...lons);
  const dLat = (m2 - m1) || 0.1, dLon = (n2 - n1) || 0.1;
  m1 -= dLat * pad; m2 += dLat * pad; n1 -= dLon * pad; n2 += dLon * pad;
  return rows.map((r) => ({
    id: r.block, name: r.name, sss: r.sss, c: band(r.sss).c, v: norm01(window._fieldForProj || "rainfall", r),
    x: W * (r.lon - n1) / (n2 - n1), y: H * (1 - (r.lat - m1) / (m2 - m1)),
  }));
}
// Standalone canvas field + absolutely-positioned DOM markers (fallback + mobile).
function renderStandalone(stage, canvas, rows, field, sel, onPick) {
  const W = stage.clientWidth || 320, H = stage.clientHeight || 300;
  canvas.style.width = W + "px"; canvas.style.height = H + "px";
  window._fieldForProj = field;
  const pts = projBounds(rows, W, H);
  drawField(canvas, pts, FIELD_RAMPS[field], { w: W, h: H, alpha: 190, blur: 12, reach: Math.max(W, H) * 0.9 });
  stage.querySelectorAll(".risk-marker.dom").forEach((e) => e.remove());
  pts.forEach((p) => {
    const b = document.createElement("button");
    b.className = "risk-marker dom" + (p.id === sel ? " sel" : "");
    b.style.cssText = `position:absolute;left:${p.x}px;top:${p.y}px;transform:translate(-50%,-50%);background:none;border:0;cursor:pointer`;
    b.innerHTML = `<span class="risk-pin" style="background:${p.c}">${p.sss}</span><span class="pin-label">${p.name}</span>`;
    b.onclick = () => onPick && onPick(p.id);
    stage.appendChild(b);
  });
}
// Select a block from the map (keeps the officer selector dropdown in sync).
function pickBlock(id) {
  state.block = id; state._detailBlock = id;
  const sel = document.querySelector("#d-selectors select");
  if (sel) sel.value = id;
  onSelectionChange();
}
// ---- Dashboard: Leaflet satellite base + canvas heat overlay ----
function initDashLeaflet(rows) {
  if (state._leaflet) return state._leaflet;
  const map = L.map("d-leaflet", { zoomControl: true, attributionControl: false, zoomAnimation: false, fadeAnimation: false });
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 16 }).addTo(map);
  state._leaflet = map;
  state._markers = L.layerGroup().addTo(map);
  map.on("move zoom zoomend moveend resize", drawDashField);
  setTimeout(() => {
    map.invalidateSize();
    try {
      if (map.getSize().x > 0) {
        map.fitBounds(L.latLngBounds(rows.map((r) => [r.lat, r.lon])).pad(0.2), { maxZoom: 11 });
      }
    } catch (e) { }
    drawDashField();
  }, 60);
  return map;
}
function drawDashField() {
  const map = state._leaflet, rows = state._mapRows, canvas = document.getElementById("d-heat");
  if (!map || !map._loaded || !rows || !canvas) return;
  const size = map.getSize();
  let pts;
  try {
    pts = rows.map((r) => {
      const pt = map.latLngToContainerPoint([r.lat, r.lon]);
      return { x: pt.x, y: pt.y, v: norm01(state.dField, r) };
    });
  } catch (err) {
    // Leaflet throws 'Set map center and zoom first' if called before fitBounds completes
    return;
  }
  drawField(canvas, pts, FIELD_RAMPS[state.dField], { w: size.x, h: size.y, alpha: 165, blur: 14, reach: Math.max(size.x, size.y) * 0.7 });
}
function placeDashMarkers(rows, sel) {
  if (!state._markers) return;
  state._markers.clearLayers();
  rows.forEach((r) => {
    const c = band(r.sss).c;
    const icon = L.divIcon({ className: "", iconSize: [34, 34], iconAnchor: [17, 17],
      html: `<div class="risk-marker${r.block === sel ? " sel" : ""}"><span class="risk-pin" style="background:${c}">${r.sss}</span><span class="pin-label">${r.name}</span></div>` });
    L.marker([r.lat, r.lon], { icon }).addTo(state._markers).on("click", () => pickBlock(r.block));
  });
}
// Mount the dashboard map: Leaflet when available, else standalone canvas fallback.
function mountDashMap(rows, field, sel) {
  state._mapRows = rows; state.dField = field;
  const canvas = document.getElementById("d-heat");
  if (!canvas) return;
  if (window.L && document.getElementById("d-leaflet")) {
    initDashLeaflet(rows);
    placeDashMarkers(rows, sel);
    requestAnimationFrame(drawDashField);
    state._redraw = drawDashField;
  } else {
    const stage = canvas.parentElement;
    renderStandalone(stage, canvas, rows, field, sel, pickBlock);
    state._redraw = () => renderStandalone(stage, canvas, rows, field, sel, pickBlock);
  }
}
// Mount the mobile Block Risk Map (always standalone canvas).
function mountMobileMap(stage, canvas, rows, field, sel, onPick) {
  renderStandalone(stage, canvas, rows, field, sel, onPick);
  state._mredraw = () => renderStandalone(stage, canvas, rows, field, sel, onPick);
}
window.addEventListener("resize", () => {
  if (state.surface === "officer" && state._leaflet) { state._leaflet.invalidateSize(); drawDashField(); }
  if (state.surface === "farmer" && state._mredraw) state._mredraw();
});




