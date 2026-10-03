// farmer.js — mobile farmer app. Defines gotoScreen() + renderFarmer() (both called
// by app.js) and the five screens (Home / Forecast / Advisory / Block Risk Map /
// Alerts) plus a More menu. Reuses getDet()/getRm() + helpers from app.js/dashboard.js
// and the shared heat field (mountMobileMap). Temps on the forecast strip are the one
// representative-only field (no real source); everything else is engine-derived.

function gotoScreen(id) {
  state.screen = id;
  $$(".surface-farmer .screen").forEach((s) => s.classList.toggle("active", s.id === "screen-" + id));
  $$("#m-nav button").forEach((b) => b.classList.toggle("active", b.dataset.screen === id));
  renderFarmerScreen(id);
  const body = $("#m-body"); if (body) body.scrollTop = 0;
}
function renderFarmer() { renderNavs(); renderFarmerScreen(state.screen); }
function renderFarmerScreen(id) {
  if (id === "home") return renderHome();
  if (id === "forecast") return renderMForecast();
  if (id === "advisory") return renderMAdvisory();
  if (id === "map") return renderMMap();
  if (id === "alerts") return renderMAlerts();
  if (id === "more") return renderMMore();
}
function wireGo(root) { $$("[data-go]", root).forEach((b) => b.onclick = () => gotoScreen(b.dataset.go)); }
function rainIcon(mm) { return mm >= 25 ? "⛈️" : mm >= 8 ? "🌧️" : mm >= 1 ? "🌦️" : "☀️"; }
function synthTemp(mm) { return Math.round(29 - Math.min(6, mm / 6)); } // representative monsoon temp
// ---- Home ----
function heroHtml(det) {
  return `<div class="hero"><span class="sun">⛅</span>`
    + `<div class="greet">${t("greeting_morning")}</div><div class="greet-sub">${t("greeting_sub")}</div>`
    + `<span class="loc">📍 ${det.name} · ${state.meta.district}</span></div>`;
}
function miniKpis(det) {
  const s = det.signals;
  const items = [
    { ic: "🌧️", c: "#1d4ed8", title: t("kpi_onset"), val: onsetPct(s) + "%" },
    { ic: "☀️", c: "#b45309", title: t("kpi_dry"), val: pct(s.p_dryspell) + "%" },
    { ic: "🌊", c: "#7c3aed", title: t("kpi_heavy"), val: pct(s.p_heavy) + "%" },
    { ic: "💧", c: "#0f766e", title: t("sig_sm"), val: pct(s.sm_adeq) + "%" },
  ];
  return items.map((i) => `<div class="kpi-mini"><div class="km-top"><span class="km-ic" style="background:${i.c}">${i.ic}</span>${i.title}</div><div class="km-val">${i.val}</div></div>`).join("");
}
function next7Cards(det) {
  const n = state.meta.n_days, out = [];
  for (let k = 1; k <= 7; k++) {
    const i = Math.min(n - 1, state.day + k), mm = Math.round(det.observed_full[i]);
    out.push(`<div class="day-card"><div class="dc-day">${fmtDay(_dates()[i]).replace(/\s/, " ")}</div>`
      + `<div class="dc-ic">${rainIcon(mm)}</div><div class="dc-mm ${mm > 0 ? "rainy" : ""}">${mm} mm</div>`
      + `<div class="dc-t">${synthTemp(mm)}°C</div></div>`);
  }
  return out.join("");
}
async function renderHome() {
  const el = $("#screen-home"); if (!el) return;
  const det = await getDet(), a = det.action_labels;
  el.innerHTML = heroHtml(det)
    + `<div class="m-card"><h3>${t("home_recommendation")}</h3>`
      + `<div class="reco-chip" style="background:${a.color}"><span class="ic">${a.icon}</span><span>${a[state.lang]}</span>`
      + `<span class="b">${bandLabel(det.sss)} · ${det.sss}</span></div>`
      + `<div class="reco-reason">${det.reason_i18n[state.lang]}</div>`
      + `<button class="link-btn" style="margin-top:.7rem" data-go="advisory">${t("adv_why")} →</button></div>`
    + `<div class="m-card"><h3>${t("kpi_sss")}</h3><div class="kpi-mini-grid">${miniKpis(det)}</div></div>`
    + `<div class="m-card"><h3>${t("home_next7")}</h3><div class="day-strip">${next7Cards(det)}</div></div>`;
  wireGo(el);
}
// ---- Monsoon Forecast ----
// 30-day outlook: colour each day vs the window mean (above / normal / below).
function renderCal(det) {
  const n = state.meta.n_days, start = state.day, end = Math.min(n - 1, start + 29);
  const obs = det.observed_full, win = [];
  for (let i = start; i <= end; i++) win.push(obs[i]);
  const mean = win.reduce((a, b) => a + b, 0) / (win.length || 1);
  const dow = ["S", "M", "T", "W", "T", "F", "S"];
  let html = dow.map((d) => `<div class="cal-dow">${d}</div>`).join("");
  const first = new Date(_dates()[start]).getDay();
  for (let k = 0; k < first; k++) html += `<div class="cal-cell" style="background:transparent"></div>`;
  for (let i = start; i <= end; i++) {
    const mm = obs[i], hi = mean * 1.25, lo = mean * 0.75;
    const c = mm > Math.max(hi, 1) ? "#38bdf8" : mm < lo ? "#fca5a5" : "#86efac";
    const d = new Date(_dates()[i]).getDate();
    html += `<div class="cal-cell" style="background:${c}" title="${fmtDay(_dates()[i])}: ${Math.round(mm)} mm">${d}</div>`;
  }
  return html;
}
function calLegendHtml() {
  return `<span><i style="background:#38bdf8"></i>${t("outlook_above")}</span>`
    + `<span><i style="background:#86efac"></i>${t("outlook_normal")}</span>`
    + `<span><i style="background:#fca5a5"></i>${t("outlook_below")}</span>`;
}
function insightsHtml(det) {
  const s = det.signals, out = [];
  out.push({ c: s.onset_done ? "#15803d" : s.p_onset_7 >= 0.6 ? "#1d4ed8" : "#b45309",
    txt: s.onset_done ? t("kpi_onset_observed") : `${t("kpi_onset")}: ${onsetPct(s)}%` });
  out.push({ c: s.p_dryspell >= 0.5 ? "#b91c1c" : s.p_dryspell >= 0.3 ? "#b45309" : "#15803d",
    txt: `${t("kpi_dry")}: ${pct(s.p_dryspell)}%` });
  out.push({ c: s.p_heavy >= 0.5 ? "#b91c1c" : s.p_heavy >= 0.3 ? "#7c3aed" : "#15803d",
    txt: `${t("kpi_heavy")}: ${pct(s.p_heavy)}%` });
  out.push({ c: s.sm_adeq >= 0.6 ? "#0f766e" : "#b45309",
    txt: `${t("sig_sm")}: ${pct(s.sm_adeq)}%` });
  return out.map((i) => `<div class="insight"><span class="dot" style="background:${i.c}"></span>${i.txt}</div>`).join("");
}
async function renderMForecast() {
  const el = $("#screen-forecast"); if (!el) return;
  const det = await getDet();
  el.innerHTML = `<div class="screen-title">${t("nav_forecast")}</div>`
    + `<div class="m-card"><h3>${t("forecast_7day")}</h3><div class="chart-wrap" id="mf-bars"></div></div>`
    + `<div class="m-card"><h3>${t("forecast_30day")}</h3><div class="cal-grid">${renderCal(det)}</div>`
      + `<div class="cal-legend">${calLegendHtml()}</div></div>`
    + `<div class="m-card"><h3>${t("forecast_insights")}</h3>${insightsHtml(det)}</div>`;
  const n = state.meta.n_days, bars = [];
  for (let k = 1; k <= 7; k++) {
    const i = Math.min(n - 1, state.day + k), mm = Math.round(det.observed_full[i]);
    bars.push({ label: fmtDay(_dates()[i]).replace(/\s.*/, ""), value: mm, color: "#60a5fa", strong: true });
  }
  barChart($("#mf-bars"), bars, { unit: "", height: 170 });
}
// ---- Crop Advisory ----
// Real engine crops (meta.crops) are selectable; the rest are representative chips.
function renderCropCircles() {
  const real = (state.meta.crops || []).map((c) => c.id);
  return `<div class="crop-circles">` + CROP_ORDER.map((id) => {
    const isReal = real.includes(id), active = id === state.crop;
    return `<button class="crop-circle${active ? " active" : ""}" ${isReal ? `data-crop="${id}"` : "disabled"} `
      + `title="${isReal ? cropName(id) : cropName(id) + " · " + t("representative")}">`
      + `<span class="cc-ring">${cropIcon(id)}</span><span class="cc-name">${cropName(id)}</span></button>`;
  }).join("") + `</div>`;
}
function whyChecklist(s) {
  return `<div class="checklist">` + checksFor(s).map((c) =>
    `<div class="check ${c.ok ? "ok" : "no"}"><span class="cx">${c.ok ? "✓" : "!"}</span><span>${t(c.key)}</span></div>`).join("") + `</div>`;
}
function windowHtml(det) {
  const wp = Math.round(Math.max(0, Math.min(1, det.signals.window_progress || 0)) * 100);
  return `<div class="window-bar"><div class="window-now" style="left:${wp}%" data-label="${wp}%"></div></div>`
    + `<div class="window-ends"><span>${t("window_open")}</span><span>${t("window_close")}</span></div>`;
}
async function renderMAdvisory() {
  const el = $("#screen-advisory"); if (!el) return;
  const det = await getDet(), a = det.action_labels;
  el.innerHTML = `<div class="screen-title">${t("nav_crop")}</div>`
    + `<div class="m-card">${renderCropCircles()}</div>`
    + `<div class="m-card"><h3>${t("adv_recommendation")}</h3>`
      + `<div class="reco-chip" style="background:${a.color}"><span class="ic">${a.icon}</span><span>${a[state.lang]}</span>`
      + `<span class="b">${bandLabel(det.sss)} · ${det.sss}</span></div>`
      + `<div class="reco-reason">${det.reason_i18n[state.lang]}</div></div>`
    + `<div class="m-card"><h3>${t("adv_why")}</h3>${whyChecklist(det.signals)}</div>`
    + `<div class="m-card"><h3>${t("adv_window")}</h3>${windowHtml(det)}</div>`
    + `<div class="m-card"><h3>${t("adv_details")}</h3>${insightsHtml(det)}</div>`;
  $$("[data-crop]", el).forEach((b) => b.onclick = () => { state.crop = b.dataset.crop; renderMAdvisory(); });
}
// ---- Block Risk Map (standalone canvas heat field) ----
function mFieldTabsHtml(active) {
  return `<div class="m-fieldtabs">` + FIELDS.map((f) =>
    `<button data-mfield="${f.id}" class="${f.id === active ? "active" : ""}">${t(f.key)}</button>`).join("") + `</div>`;
}
function blkVal(fid, row) {
  return fid === "rainfall" ? Math.round(row.rain) + " mm"
    : Math.round((fid === "onset" ? row.p_onset : fid === "dryspell" ? row.p_dryspell : row.p_heavy) * 100) + "%";
}
function renderBlockList(rm, filter) {
  const q = (filter || "").toLowerCase();
  return rm.blocks.filter((b) => b.name.toLowerCase().includes(q)).map((b) => {
    const c = band(b.sss).c;
    return `<button class="blk-row${b.block === state.block ? " sel" : ""}" data-blk="${b.block}">`
      + `<span class="blk-dot" style="background:${c}"></span>`
      + `<span class="blk-name">${b.name}</span><span class="blk-sub">${t("sss_label")}: ${b.sss}</span>`
      + `<span class="blk-val">${blkVal(state.mField, b)}</span></button>`;
  }).join("");
}
function wireBlockList(el, rm) {
  $$("[data-blk]", el).forEach((b) => b.onclick = () => mPickBlock(b.dataset.blk));
}
function mPickBlock(id) { state.block = id; state._detailBlock = id; renderMMap(); }
function renderMMapField(rm) {
  $$("#screen-map [data-mfield]").forEach((b) => b.classList.toggle("active", b.dataset.mfield === state.mField));
  const bar = $("#m-legend-bar"); if (bar) bar.style.background = rampCss(FIELD_RAMPS[state.mField]);
  const stage = $(".m-map-stage"), canvas = $("#m-heat");
  if (stage && canvas) mountMobileMap(stage, canvas, rm.blocks, state.mField, state.block, mPickBlock);
  const list = $("#m-blk-list"), search = $("#m-search");
  if (list) { list.innerHTML = renderBlockList(rm, search ? search.value : ""); wireBlockList($("#screen-map"), rm); }
}
async function renderMMap() {
  const el = $("#screen-map"); if (!el) return;
  const rm = await getRm();
  el.innerHTML = `<div class="screen-title">${t("risk_map")}</div>`
    + mFieldTabsHtml(state.mField)
    + `<div class="m-map-stage"><canvas id="m-heat"></canvas></div>`
    + `<div class="map-legend"><span>${t("legend_low")}</span><div class="legend-bar" id="m-legend-bar"></div><span>${t("legend_high")}</span></div>`
    + `<input class="search-box" id="m-search" placeholder="${t("search_block")}">`
    + `<div class="blk-list" id="m-blk-list">${renderBlockList(rm, "")}</div>`;
  const bar = $("#m-legend-bar"); if (bar) bar.style.background = rampCss(FIELD_RAMPS[state.mField]);
  const stage = $(".m-map-stage"), canvas = $("#m-heat");
  mountMobileMap(stage, canvas, rm.blocks, state.mField, state.block, mPickBlock);
  $$("[data-mfield]", el).forEach((b) => b.onclick = () => { state.mField = b.dataset.mfield; renderMMapField(rm); });
  wireBlockList(el, rm);
  const search = $("#m-search");
  if (search) search.oninput = () => { $("#m-blk-list").innerHTML = renderBlockList(rm, search.value); wireBlockList(el, rm); };
}
// ---- Alerts & Notifications ----
let _alertFilter = "all";
// Per-day alerts straight from the real series (same thresholds as synthAlerts).
function alertsAt(ser, i) {
  const out = [], when = fmtDay(_dates()[i]);
  if (ser.p_heavy[i] >= 0.45) out.push({ level: "weather", icon: "🌧️", c: "#b91c1c", titleKey: "field_heavy", val: pct(ser.p_heavy[i]), when });
  if (ser.p_dryspell[i] >= 0.4) out.push({ level: "weather", icon: "☀️", c: "#b45309", titleKey: "kpi_dry", val: pct(ser.p_dryspell[i]), when });
  if (ser.onset_done[i]) out.push({ level: "advisory", icon: "✅", c: "#15803d", titleKey: "ms_onset", val: null, when });
  else if (ser.p_onset_7[i] >= 0.5) out.push({ level: "advisory", icon: "🌱", c: "#1d4ed8", titleKey: "kpi_onset", val: pct(ser.p_onset_7[i]), when });
  if (ser.sm_adeq[i] >= 0.85) out.push({ level: "system", icon: "💧", c: "#0f766e", titleKey: "sig_sm", val: pct(ser.sm_adeq[i]), when });
  return out;
}
function alertCard(a) {
  return `<div class="alert-card" style="border-left-color:${a.c}">`
    + `<span class="ac-ic" style="background:${a.c}22">${a.icon}</span>`
    + `<div class="ac-body"><div class="ac-title">${t(a.titleKey)}${a.val != null ? " · " + a.val + "%" : ""}</div>`
    + `<div class="ac-msg">${t("alerts_" + a.level)}</div><div class="ac-time">${a.when}</div></div></div>`;
}
function alertPills() {
  const fs = [["all", "alerts_all"], ["advisory", "alerts_advisory"], ["weather", "alerts_weather"], ["system", "alerts_system"]];
  return `<div class="m-fieldtabs">` + fs.map(([k, lbl]) =>
    `<button data-afilter="${k}" class="${k === _alertFilter ? "active" : ""}">${t(lbl)}</button>`).join("") + `</div>`;
}
async function renderMAlerts() {
  const el = $("#screen-alerts"); if (!el) return;
  const det = await getDet(), ser = det.series, n = state.meta.n_days, day = state.day;
  const today = alertsAt(ser, day), week = [];
  for (let i = day + 1; i <= Math.min(n - 1, day + 6); i++) week.push(...alertsAt(ser, i));
  const f = (arr) => _alertFilter === "all" ? arr : arr.filter((a) => a.level === _alertFilter);
  const sec = (titleKey, arr) => `<div class="sec-label">${t(titleKey)}</div>`
    + (arr.length ? arr.map(alertCard).join("") : `<div class="crop-sub">${t("alerts_none")}</div>`);
  el.innerHTML = `<div class="screen-title">${t("nav_alerts")}</div>`
    + alertPills() + sec("alerts_today", f(today)) + sec("alerts_week", f(week));
  $$("[data-afilter]", el).forEach((b) => b.onclick = () => { _alertFilter = b.dataset.afilter; renderMAlerts(); });
}
// ---- More menu ----
function renderMMore() {
  const el = $("#screen-more"); if (!el) return;
  const items = [["map", "🗺️", t("block_risk_map")], ["__officer", "📊", t("view_officer")],
    ["__reports", "📄", t("nav_reports")], ["__settings", "⚙️", t("nav_settings")]];
  el.innerHTML = `<div class="screen-title">${t("more_title")}</div><div class="more-list">`
    + items.map(([k, ic, lbl]) => `<button data-more="${k}"><span>${ic}</span><span>${lbl}</span></button>`).join("")
    + `</div><div class="disclaimer">${t("disclaimer")}</div>`;
  $$("[data-more]", el).forEach((b) => b.onclick = () => {
    const k = b.dataset.more;
    if (k === "__officer") return switchSurface("officer");
    if (k === "map") return gotoScreen("map");
  });
}







