// dashboard.js — officer desktop dashboard. Defines renderDashboard() (called by
// app.js) and its sub-renderers. One /api/block call drives KPIs/rec/soil/timeline/
// daily; one /api/risk-map call drives the map + crop-wise summary. Both are cached
// by selection key (getDet/getRm, also used by farmer.js). Colours follow the two-
// system rule: action colour for action chips, band colour for scores/map.

async function getDet() {
  const key = `${state.block}|${state.crop}|${state.irr}|${state.day}`;
  if (state._detKey === key && state._lastDet) return state._lastDet;
  state._lastDet = await API(`/api/block/${state.block}?crop=${state.crop}&irrigation=${state.irr}&day=${state.day}`);
  state._detKey = key;
  return state._lastDet;
}
async function getRm() {
  const key = `${state.crop}|${state.irr}|${state.day}`;
  if (state._rmKey === key && state._lastRm) return state._lastRm;
  state._lastRm = await API(`/api/risk-map?crop=${state.crop}&irrigation=${state.irr}&day=${state.day}`);
  state._rmKey = key;
  return state._lastRm;
}
const curField = () => FIELDS.find((f) => f.id === state.dField) || FIELDS[0];
const ACCESS = [["access_whatsapp", "💬", "#25d366"], ["access_sms", "✉️", "#2563eb"],
  ["access_ivr", "📞", "#7c3aed"], ["access_web", "🌐", "#0f766e"]];

async function renderDashboard() {
  if (!state.meta) return;
  const [det, rm] = await Promise.all([getDet(), getRm()]);
  state._lastRm = rm;
  renderSubtitle(det); renderKPIs(det); renderMapPanel(rm); renderScrubber();
  renderTimeline(det); renderDaily(det); renderRec(det); renderCropwise(det);
  renderSoil(det); renderClimate(); renderAlerts(det); renderAccess();
}
// ---- Day scrubber + Play ----
function scheduleRender() { clearTimeout(state._rt); state._rt = setTimeout(renderDashboard, 110); }
function renderScrubber() {
  const r = $("#d-day"), lab = $("#d-day-label");
  if (r) {
    r.max = state.meta.n_days - 1; r.value = state.day;
    r.oninput = (e) => { state.day = +e.target.value; if (lab) lab.textContent = fmtDate(_dates()[state.day]); scheduleRender(); };
  }
  if (lab) lab.textContent = fmtDate(_dates()[state.day]);
  const p = $("#d-play"); if (p) p.onclick = togglePlay;
}
function togglePlay() {
  const p = $("#d-play");
  if (state._timer) return stopPlay();
  if (p) { p.classList.add("playing"); p.textContent = "⏸"; }
  state._timer = setInterval(() => {
    state.day = state.day + 1 >= state.meta.n_days ? 0 : state.day + 1;
    const r = $("#d-day"); if (r) r.value = state.day;
    renderDashboard();
  }, 850);
}
function stopPlay() {
  clearInterval(state._timer); state._timer = null;
  const p = $("#d-play"); if (p) { p.classList.remove("playing"); p.textContent = "▶"; }
}

function renderSubtitle(det) {
  const el = $("#d-subtitle"); if (!el) return;
  const irr = t(state.irr === "none" ? "irr_rainfed" : "irr_" + state.irr);
  el.textContent = `${det.name}, ${state.meta.district} · ${cropName(state.crop)} · ${irr} · ${fmtDate(det.date)}`;
}
// Representative "likely window" text from the real series (first threshold crossing).
function peakText(arr, from, thr, prefixKey) {
  const i = firstIdxFrom(arr, Math.max(0, from), (v) => v >= thr);
  if (i < 0) return "";
  const j = Math.min(i + 3, state.meta.n_days - 1);
  const startStr = fmtDay(_dates()[i]);
  const endStr = fmtDay(_dates()[j]);
  // If same month, show "14–17 Jun"; otherwise show "28 Jun–2 Jul"
  const startMonth = new Date(_dates()[i]).getMonth();
  const endMonth = new Date(_dates()[j]).getMonth();
  const endDisplay = startMonth === endMonth ? new Date(_dates()[j]).getDate() + " " + endStr.replace(/^\d+\s*/, "") : endStr;
  return `${t(prefixKey)}: ${startStr}–${endDisplay}`;
}
function renderKPIs(det) {
  const host = $("#d-kpis"); if (!host) return;
  const s = det.signals, ser = det.series, day = state.day;
  let delta = 0;
  if (day > 0 && ser) delta = Math.round(100 * (ser.p_onset_7[day] - ser.p_onset_7[day - 1]));
  const dCls = delta >= 0 ? "up" : "down", dTxt = (delta >= 0 ? "↑ " : "↓ ") + Math.abs(delta) + "%";
  const dry = riskLabel(s.p_dryspell), hv = riskLabel(s.p_heavy), b = band(det.sss);
  host.innerHTML = [
    `<div class="kpi t-blue"><div class="kpi-top"><div class="kpi-ic" style="background:#1d4ed8">🌧️</div>`
      + `${day > 0 ? `<span class="kpi-delta ${dCls}">${dTxt}</span>` : ""}</div>`
      + `<div class="kpi-title">${t("kpi_onset")}</div><div class="kpi-val">${onsetPct(s)}%</div>`
      + `<div class="kpi-note">${expectedOnsetText(s, day)}</div></div>`,
    `<div class="kpi t-amber"><div class="kpi-top"><div class="kpi-ic" style="background:${dry.c}">☀️</div></div>`
      + `<div class="kpi-title">${t("kpi_dry")}</div><div class="kpi-val">${pct(s.p_dryspell)}%</div>`
      + `<div class="kpi-note">${t(dry.key)}${ser ? " · " + (peakText(ser.p_dryspell, day + 1, 0.5, "kpi_likely") || t("risk_low")) : ""}</div></div>`,
    `<div class="kpi t-purple"><div class="kpi-top"><div class="kpi-ic" style="background:${hv.c}">🌊</div></div>`
      + `<div class="kpi-title">${t("kpi_heavy")}</div><div class="kpi-val">${pct(s.p_heavy)}%</div>`
      + `<div class="kpi-note">${t(hv.key)}${ser ? " · " + (peakText(ser.p_heavy, day + 1, 0.5, "kpi_likely") || t("risk_low")) : ""}</div></div>`,
    `<div class="kpi t-green"><div class="kpi-top"><div class="kpi-ic" style="background:${b.c}">🛡️</div>`
      + `<span class="kpi-badge" style="background:${b.c}">${bandLabel(det.sss)}</span></div>`
      + `<div class="kpi-title">${t("kpi_sss")}</div><div class="kpi-val">${det.sss}<small>/100</small></div>`
      + `<div class="kpi-note">${det.action_labels[state.lang]}</div></div>`,
  ].join("");
}
// ---- Map panel (field tabs + heat field + legend + badge) ----
function renderFieldTabs() {
  const host = $("#d-field-tabs"); if (!host) return;
  host.innerHTML = FIELDS.map((f) => `<button data-field="${f.id}" class="${f.id === state.dField ? "active" : ""}">${t(f.key)}</button>`).join("");
  $$("#d-field-tabs button").forEach((b) => b.onclick = () => { state.dField = b.dataset.field; refreshMapField(); });
}
function renderLegend() {
  const el = $("#d-legend"); if (!el) return;
  el.innerHTML = `<span>${t("legend_low")}</span><div class="legend-bar" style="background:${rampCss(FIELD_RAMPS[state.dField])}"></div><span>${t("legend_high")}</span>`;
}
function renderMapBadge(rm) {
  const el = $("#d-map-badge"); if (!el) return;
  const row = rm.blocks.find((b) => b.block === state.block) || rm.blocks[0];
  const f = curField();
  const val = f.id === "rainfall" ? Math.round(row.rain) + " mm"
    : Math.round((f.id === "onset" ? row.p_onset : f.id === "dryspell" ? row.p_dryspell : row.p_heavy) * 100) + "%";
  el.innerHTML = `${row.name}<small>${t(f.key)}: ${val}</small>`;
}
function renderMapPanel(rm) {
  renderFieldTabs();
  const sub = $("#d-map-sub"); if (sub) sub.textContent = `${t(curField().key)} • ${fmtDate(rm.date)}`;
  mountDashMap(rm.blocks, state.dField, state.block);
  renderLegend(); renderMapBadge(rm);
  const resetBtn = $("#d-map-reset");
  if (resetBtn) {
    resetBtn.onclick = () => {
      if (state._leaflet && state._mapRows) {
        state._leaflet.fitBounds(L.latLngBounds(state._mapRows.map((r) => [r.lat, r.lon])).pad(0.2), { maxZoom: 11 });
      }
    };
  }
}
function refreshMapField() {
  const rm = state._lastRm; if (!rm) return;
  const sub = $("#d-map-sub"); if (sub) sub.textContent = `${t(curField().key)} • ${fmtDate(rm.date)}`;
  renderFieldTabs();
  if (state._leaflet) requestAnimationFrame(drawDashField);   // repaint heat with new state.dField
  else mountDashMap(rm.blocks, state.dField, state.block);     // standalone: re-render field
  renderLegend(); renderMapBadge(rm);
}
// ---- Monsoon Timeline (HTML/CSS, placed from the real series) ----
function renderTimeline(det) {
  const host = $("#d-timeline"); if (!host) return;
  const ser = det.series, n = state.meta.n_days, day = state.day;
  const winS = Math.max(0, day - 2), winE = Math.min(n - 1, winS + 30), span = (winE - winS) || 1;
  const frac = (i) => Math.max(0, Math.min(1, (i - winS) / span));
  const ms = [];
  const oi = det.signals.onset_done ? winS : firstIdxFrom(ser.p_onset_7, winS, (v) => v >= 0.6);
  if (oi >= 0 && oi <= winE) ms.push({ f: frac(oi), name: t("ms_onset"), date: fmtDay(_dates()[oi]), c: "#15803d" });
  const gi = firstIdxFrom(ser.p_onset_7, Math.max(winS, oi < 0 ? winS : oi), (v, ix) => v >= 0.7 && ser.p_dryspell[ix] < 0.4);
  if (gi >= 0 && gi <= winE) ms.push({ f: frac(gi), name: t("ms_good"), date: fmtDay(_dates()[gi]), c: "#16a34a" });
  const di = firstIdxFrom(ser.p_dryspell, winS, (v) => v >= 0.5);
  if (di >= 0 && di <= winE) ms.push({ f: frac(di), name: t("ms_dry"), date: fmtDay(_dates()[di]), c: "#b45309" });
  const hi = firstIdxFrom(ser.p_heavy, winS, (v) => v >= 0.5);
  if (hi >= 0 && hi <= winE) ms.push({ f: frac(hi), name: t("ms_heavy"), date: fmtDay(_dates()[hi]), c: "#b91c1c" });
  
  // Stagger markers to avoid text overlap
  ms.sort((a, b) => a.f - b.f);
  for (let i = 1; i < ms.length; i++) {
    if (ms[i].f - ms[i-1].f < 0.15) ms[i].f = Math.min(1, ms[i-1].f + 0.15);
  }
  ms.forEach((m, i) => m.below = (i % 2 !== 0));
  const nowF = (frac(day) * 100).toFixed(1);
  host.innerHTML = `<div class="tl-track"><div class="tl-fill" style="width:${nowF}%"></div>`
    + `<div class="tl-now" style="left:${nowF}%"></div>`
    + ms.map((m) => `<div class="tl-dot${m.below ? " below" : ""}" style="left:${(m.f * 100).toFixed(1)}%;background:${m.c}">`
      + `<div class="tl-cap"><div class="tl-name">${m.name}</div><div class="tl-date">${m.date}</div></div></div>`).join("")
    + `</div>`;
}
// ---- Daily Rainfall & Probability (dual-axis, ~30-day window) ----
function renderDaily(det) {
  const host = $("#d-daily"); if (!host) return;
  const n = state.meta.n_days, day = state.day;
  let winE = Math.min(n - 1, Math.max(0, day - 15) + 29), winS = Math.max(0, winE - 29);
  const obs = det.observed_full, ser = det.series, bars = [], line = [];
  for (let i = winS; i <= winE; i++) {
    bars.push({ label: fmtDay(_dates()[i]).replace(/\s.*/, ""), value: Math.round(obs[i]) });
    line.push(ser.p_onset_7[i]);
  }
  dualAxis(host, { bars, line, markIdx: day - winS, barColor: "#60a5fa", lineColor: "#15803d" });
  host.insertAdjacentHTML("beforeend", `<div class="chart-legend"><span><i style="background:#60a5fa"></i>${t("legend_rain")}</span>`
    + `<span><i class="line" style="background:#15803d"></i>${t("legend_onsetprob")}</span></div>`);
}
// ---- Recommended Action rail ----
function checksFor(s) {
  return [
    { ok: (s.onset_done || s.p_onset_7 >= 0.6), key: "check_onset" },
    { ok: s.p_dryspell < 0.4, key: "check_dry" },
    { ok: s.p_heavy < 0.4, key: "check_heavy" },
    { ok: s.sm_adeq >= 0.6, key: "check_soil" },
  ];
}
function renderRec(det) {
  const host = $("#d-rec"); if (!host) return;
  const a = det.action_labels, s = det.signals;
  host.innerHTML = `<div class="panel-title">${t("panel_recommended")}</div>`
    + `<div class="rec-chip" style="background:${a.color}"><span class="ic">${a.icon}</span><span>${a[state.lang]}</span>`
    + `<span class="rec-score">${det.sss}<small>/100</small></span></div>`
    + `<div class="checklist">` + checksFor(s).map((c) => `<div class="check ${c.ok ? "ok" : "no"}"><span class="cx">${c.ok ? "✓" : "!"}</span><span>${t(c.key)}</span></div>`).join("") + `</div>`
    + `<button class="link-btn" id="d-view-detail">${t("view_detailed")} →</button>`;
  const btn = $("#d-view-detail"); if (btn) btn.onclick = () => $(".map-panel") && $(".map-panel").scrollIntoView({ behavior: "smooth" });
}
// ---- Crop-wise Advisory rail (selected crop real; others representative) ----
function renderCropwise(det) {
  const host = $("#d-cropwise"); if (!host) return;
  const rows = cropStatuses(det.signals, det.crop, det.sss);
  host.innerHTML = `<div class="panel-title">${t("panel_cropwise")}</div>`
    + rows.map((r) => `<div class="crop-row" style="${r.id === det.crop ? 'background:rgba(22,163,74,0.1); border:1px solid #16a34a;' : ''}"><span class="crop-ic">${cropIcon(r.id)}</span>`
      + `<div><div class="crop-name">${cropName(r.id)}${r.rep ? `<span class="rep-tag">${t("representative")}</span>` : ""}</div>`
      + `<div class="crop-sub">${t("sss_label")}: ${r.sss}</div></div>`
      + `<span class="status-pill" style="background:${r.c}22;color:${r.c}">${t(r.statusKey)}</span></div>`).join("");
}
// ---- Soil moisture (value + bar + real trajectory sparkline) ----
function renderSoil(det) {
  const host = $("#d-soil"); if (!host) return;
  const sm = det.signals.sm_adeq, p = pct(sm);
  const key = sm < 0.5 ? "soil_low" : sm < 0.9 ? "soil_adequate" : "soil_saturated";
  const c = sm < 0.5 ? "#b45309" : sm < 0.9 ? "#15803d" : "#0f766e";
  host.innerHTML = `<div class="panel-title">${t("panel_soil")}</div>`
    + `<div class="soil-bar"><span style="width:${p}%;background:${c}"></span></div>`
    + `<div class="big-stat"><b style="color:${c}">${p}%</b><span class="crop-sub">${t(key)} · ${t("soil_note")}</span></div>`
    + `<div class="chart-wrap" id="d-soil-spark"></div>`;
  const n = state.meta.n_days, s = Math.max(0, state.day - 20);
  sparkline($("#d-soil-spark"), det.series.sm_adeq.slice(s, Math.min(n, state.day + 1)), { color: c, fill: c, height: 48 });
}
// ---- Climate indicators (from meta.climate) ----
function renderClimate() {
  const host = $("#d-climate"); if (!host) return;
  host.innerHTML = `<div class="panel-title">${t("panel_climate")} <span class="panel-sub-inline">${t("clim_season")}</span></div>`
    + climateBadges(state.meta.climate).map((r) => `<div class="clim-row"><span class="clim-name">${t(r.nameKey)}</span>`
      + `<span class="clim-badge" style="background:${r.c}22;color:${r.c}">${t(r.vk)} · ${r.extra}</span></div>`).join("");
}
// ---- Recent alerts (data-driven synth) ----
function renderAlerts(det) {
  const host = $("#d-alerts"); if (!host) return;
  const al = synthAlerts(det.signals, state.day);
  host.innerHTML = `<div class="panel-title">${t("panel_alerts")}</div>`
    + (al.length ? al.map((a) => `<div class="alert-row"><span class="alert-ic" style="background:${a.c}22">${a.icon}</span>`
      + `<div class="alert-txt"><b>${t(a.titleKey)}${a.val != null ? ` · ${a.val}%` : ""}</b><span class="alert-time">${a.when}</span></div></div>`).join("")
      : `<div class="crop-sub">—</div>`);
}
// ---- Access channels (decorative) ----
function renderAccess() {
  const host = $("#d-access"); if (!host) return;
  host.innerHTML = `<div class="panel-title">${t("panel_access")}</div><div class="access-grid">`
    + ACCESS.map(([k, ic, c]) => `<button class="access-btn"><span class="ab-ic" style="background:${c}">${ic}</span>${t(k)}</button>`).join("")
    + `</div>`;
}






