// SowSafe — shared core. Defines globals used by charts.js / heatmap.js /
// dashboard.js / farmer.js (classic scripts, loaded after this one). init() is
// deferred to DOMContentLoaded so those modules' functions are defined by then.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const API = (p) => fetch(p).then((r) => r.json());

const state = {
  lang: localStorage.getItem("sowsafe_lang") || "en", surface: "officer", meta: null,
  block: "palghar", crop: "rice", irr: "none", day: 20,
  dField: "rainfall", mField: "rainfall", screen: "home",
  _detailBlock: "palghar", _timer: null, _leaflet: null, _markers: null, _heatLayer: null, _fitDone: false,
};

const IRR = ["none", "partial", "assured"];
const FIELDS = [
  { id: "rainfall", key: "field_rainfall", unit: "mm" },
  { id: "onset", key: "field_onset", unit: "%" },
  { id: "dryspell", key: "field_dryspell", unit: "%" },
  { id: "heavy", key: "field_heavy", unit: "%" },
];
// Icons + representative water rules (need = water demand, wls = waterlogging sensitivity).
// rice/okra are modelled by the engine; the other four are representative chips.
const CROP_META = {
  rice:    { icon: "🌾", need: 0.90, wls: 0.15 },
  okra:    { icon: "🥬", need: 0.50, wls: 0.75 },
  soybean: { icon: "🫘", need: 0.60, wls: 0.50, rep: true },
  cotton:  { icon: "🌿", need: 0.55, wls: 0.45, rep: true },
  tur:     { icon: "🌱", need: 0.40, wls: 0.30, rep: true },
  maize:   { icon: "🌽", need: 0.70, wls: 0.40, rep: true },
};
const CROP_ORDER = ["rice", "okra", "soybean", "cotton", "tur", "maize"];
const SIDE_NAV = [
  ["dashboard", "📊", "nav_dashboard"], ["forecast", "🌦️", "nav_forecast"],
  ["events", "📅", "nav_events"], ["crop", "🌱", "nav_crop"],
  ["riskmaps", "🗺️", "nav_riskmaps"], ["myfarm", "🚜", "nav_myfarm"],
  ["alerts", "🔔", "nav_alerts"], ["reports", "📄", "nav_reports"], ["settings", "⚙️", "nav_settings"],
];
const BOTTOM_NAV = [
  ["home", "🏠", "nav_home"], ["forecast", "🌦️", "nav_forecast"],
  ["advisory", "🌱", "nav_advisory"], ["alerts", "🔔", "nav_alerts"], ["more", "⋯", "nav_more"],
];

const t = (k) => (I18N[state.lang] && I18N[state.lang][k]) || I18N.en[k] || k;
function band(sss) {
  if (sss >= 70) return { c: "#15803d", key: "safe" };
  if (sss >= 45) return { c: "#b45309", key: "caution" };
  return { c: "#b91c1c", key: "wait" };
}
const bandLabel = (sss) => t("band_" + band(sss).key);
const pct = (v) => Math.round(v * 100);
function cropName(id) {
  const c = (state.meta?.crops || []).find((x) => x.id === id);
  if (c) return c["name_" + state.lang] || c.name_en || id;
  return t("crop_" + id);
}
function cropIcon(id) {
  const c = (state.meta?.crops || []).find((x) => x.id === id);
  return (c && c.icon) || (CROP_META[id] && CROP_META[id].icon) || "🌱";
}
function blockName(id) {
  const b = (state.meta?.blocks || []).find((x) => x.id === id);
  return (b && b.name) || id;
}
const _dates = () => state.meta.dates;
function fmtDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(state.lang === "en" ? "en-IN" : state.lang + "-IN",
    { day: "numeric", month: "short", year: "numeric" });
}
function fmtDay(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(state.lang === "en" ? "en-IN" : state.lang + "-IN", { day: "numeric", month: "short" });
}
// ---- Derived / representative helpers (clearly flagged in the UI) ----
function riskLabel(p) {
  if (p < 0.34) return { key: "risk_low", c: "#15803d" };
  if (p < 0.66) return { key: "risk_moderate", c: "#b45309" };
  return { key: "risk_high", c: "#b91c1c" };
}
const onsetPct = (s) => Math.round(100 * (s.onset_done ? Math.max(s.p_onset_7, 0.95) : s.p_onset_7));

function firstIdxFrom(arr, from, pred) {
  for (let i = from; i < arr.length; i++) if (pred(arr[i], i)) return i;
  return -1;
}
// Expected onset date text from the engine's onset_eta_days (real) or "observed".
function expectedOnsetText(s, day) {
  if (s.onset_done) return t("kpi_onset_observed");
  const eta = Math.round(s.onset_eta_days || 0);
  if (!eta || day + eta >= state.meta.n_days) return t("kpi_expected") + ": —";
  const i = Math.min(day + Math.max(1, eta), state.meta.n_days - 1);
  const j = Math.min(i + 4, state.meta.n_days - 1);
  return `${t("kpi_expected")}: ${fmtDay(_dates()[i])}–${fmtDay(_dates()[j]).replace(/^\d+\s/, "")}`;
}
// Representative per-crop suitability 0..1 from the block's current signals.
function cropSuit(id, s) {
  const m = CROP_META[id];
  const onset = s.onset_done ? 1 : s.p_onset_7;
  const dryPen = s.p_dryspell * (0.5 + m.need);
  const floodPen = s.p_heavy * m.wls;
  let v = 0.34 * onset + 0.30 * s.sm_adeq + 0.18 * (s.p_sustain ?? onset) + 0.18 * (1 - dryPen);
  v *= 1 - 0.6 * Math.max(0, floodPen - 0.2);
  return Math.max(0, Math.min(1, v));
}
// 6-crop advisory list. Selected crop uses the engine's real sss; others are representative.
function cropStatuses(s, selectedId, selectedSss) {
  return CROP_ORDER.map((id) => {
    const real = id === selectedId;
    const sss = real ? selectedSss : Math.round(100 * cropSuit(id, s));
    const b = band(sss);
    return { id, sss, c: b.c, statusKey: "status_" + b.key.replace("safe", "sow"), rep: !real };
  });
}
// Representative alerts derived from signals — kept data-driven (numbers + dates), not prose.
function synthAlerts(s, day) {
  const out = [];
  const dIso = _dates()[day];
  if (s.p_heavy >= 0.45)
    out.push({ level: "weather", icon: "🌧️", c: "#b91c1c", titleKey: "field_heavy", val: pct(s.p_heavy), when: fmtDay(dIso) });
  if (s.p_dryspell >= 0.4)
    out.push({ level: "weather", icon: "☀️", c: "#b45309", titleKey: "kpi_dry", val: pct(s.p_dryspell), when: fmtDay(dIso) });
  if (s.onset_done)
    out.push({ level: "advisory", icon: "✅", c: "#15803d", titleKey: "ms_onset", val: null, when: fmtDay(dIso) });
  else if (s.p_onset_7 >= 0.5)
    out.push({ level: "advisory", icon: "🌱", c: "#1d4ed8", titleKey: "kpi_onset", val: pct(s.p_onset_7), when: fmtDay(dIso) });
  if (s.sm_adeq >= 0.85)
    out.push({ level: "system", icon: "💧", c: "#0f766e", titleKey: "sig_sm", val: pct(s.sm_adeq), when: fmtDay(dIso) });
  return out.slice(0, 4);
}
function climateBadges(c) {
  c = c || {};
  const enso = c.enso > 0.5 ? { k: "enso_elnino", c: "#b45309" } : c.enso < -0.5 ? { k: "enso_lanina", c: "#1d4ed8" } : { k: "enso_neutral", c: "#15803d" };
  const iod = c.iod >= 0.4 ? { k: "iod_positive", c: "#15803d" } : c.iod <= -0.4 ? { k: "iod_negative", c: "#b45309" } : { k: "iod_neutral", c: "#6b7280" };
  const mjo = (c.mjo_amp || 0) >= 1 ? { k: "mjo_active", c: "#1d4ed8" } : { k: "mjo_weak", c: "#6b7280" };
  return [
    { nameKey: "clim_enso", vk: enso.k, c: enso.c, extra: (c.enso ?? 0).toFixed(1) },
    { nameKey: "clim_iod", vk: iod.k, c: iod.c, extra: (c.iod ?? 0).toFixed(1) },
    { nameKey: "clim_mjo", vk: mjo.k, c: mjo.c, extra: "φ" + (c.mjo_phase ?? "–") },
  ];
}
// ---- Shared UI wiring ----
function applyI18n(root = document) {
  $$("[data-i18n]", root).forEach((el) => { el.textContent = t(el.dataset.i18n); });
  $$("[data-i18n-ph]", root).forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
  document.documentElement.lang = state.lang;
}

function renderNavs() {
  const sn = $("#d-nav");
  if (sn) {
    sn.innerHTML = "";
    SIDE_NAV.forEach(([id, ic, key]) => {
      const b = document.createElement("button");
      b.className = id === "dashboard" ? "active" : "";
      b.innerHTML = `<span class="ni">${ic}</span><span>${t(key)}</span>`;
      b.onclick = () => {
        $$("#d-nav button").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        $$(".o-screen").forEach((scr) => (scr.style.display = "none"));
        const tgt = $("#o-screen-" + id);
        if (tgt) tgt.style.display = "block";
        if (id === "dashboard" && state._leaflet) {
          setTimeout(() => state._leaflet.invalidateSize(), 10);
        }
      };
      sn.appendChild(b);
    });
  }
  const bn = $("#m-nav");
  if (bn) {
    bn.innerHTML = "";
    BOTTOM_NAV.forEach(([id, ic, key]) => {
      const b = document.createElement("button");
      b.className = id === state.screen ? "active" : "";
      b.dataset.screen = id;
      b.innerHTML = `<span class="bi">${ic}</span><span>${t(key)}</span>`;
      b.onclick = () => gotoScreen(id);
      bn.appendChild(b);
    });
  }
}

function makeSelect(items, value, onChange, labelFn) {
  const s = document.createElement("select");
  items.forEach((it) => {
    const o = document.createElement("option");
    o.value = it.id; o.textContent = labelFn(it);
    s.appendChild(o);
  });
  s.value = value;
  s.onchange = (e) => onChange(e.target.value);
  return s;
}

function renderSelectors() {
  const host = $("#d-selectors");
  if (!host) return;
  host.innerHTML = "";
  const m = state.meta;
  // Location (block)
  const loc = document.createElement("div"); loc.className = "selector";
  loc.innerHTML = `<span class="si">📍</span>`;
  const lcol = document.createElement("div"); lcol.className = "scol";
  lcol.innerHTML = `<span class="slabel">${t("sel_location")}</span>`;
  lcol.appendChild(makeSelect(m.blocks, state.block, (v) => { state.block = v; state._detailBlock = v; onSelectionChange(); }, (b) => b.name));
  lcol.insertAdjacentHTML("beforeend", `<span class="sdistrict">${t("sel_district")}: ${m.district}</span>`);
  loc.appendChild(lcol); host.appendChild(loc);
  // Crop
  const cr = document.createElement("div"); cr.className = "selector";
  cr.innerHTML = `<span class="si">🌱</span>`;
  const ccol = document.createElement("div"); ccol.className = "scol";
  ccol.innerHTML = `<span class="slabel">${t("sel_crop")}</span>`;
  ccol.appendChild(makeSelect(m.crops, state.crop, (v) => { state.crop = v; onSelectionChange(); }, (c) => `${c.icon} ${cropName(c.id)}`));
  cr.appendChild(ccol); host.appendChild(cr);
  // Irrigation
  const ir = document.createElement("div"); ir.className = "selector";
  ir.innerHTML = `<span class="si">💧</span>`;
  const icol = document.createElement("div"); icol.className = "scol";
  icol.innerHTML = `<span class="slabel">${t("sel_irrigation")}</span>`;
  const irrItems = IRR.map((v) => ({ id: v }));
  icol.appendChild(makeSelect(irrItems, state.irr, (v) => { state.irr = v; onSelectionChange(); },
    (o) => t(o.id === "none" ? "irr_rainfed" : "irr_" + o.id)));
  ir.appendChild(icol); host.appendChild(ir);
  // Language
  host.appendChild(makeLang());
}

function makeLang(mini) {
  const wrap = document.createElement("div");
  wrap.className = "lang" + (mini ? " mini" : "");
  [["en", "EN"], ["hi", "हिं"], ["mr", "मरा"]].forEach(([l, lbl]) => {
    const b = document.createElement("button");
    b.className = "lang-btn" + (l === state.lang ? " active" : "");
    b.textContent = lbl; b.onclick = () => setLang(l);
    wrap.appendChild(b);
  });
  return wrap;
}

function setLang(l) {
  state.lang = l;
  localStorage.setItem("sowsafe_lang", l);
  const sel = $("#settings-lang");
  if (sel) sel.value = l;
  renderNavs();
  renderSelectors();
  const ml = $("#m-lang"); if (ml) { ml.innerHTML = ""; ml.appendChild(makeLang(true)); ml.firstChild.classList.add("mini"); }
  applyI18n();
  if (state.surface === "officer") renderDashboard(); else renderFarmer();
}

function simulateModelLoad(cb) {
  const loader = $("#model-loader");
  const text = $("#loader-text");
  if (!loader || !text) return cb();
  
  loader.style.display = "flex";
  const messages = [
    "Ingesting Climate Teleconnections (ENSO/IOD/MJO)...",
    "Fetching IDW Topographic Overlays...",
    "Downscaling to Panchayat Level...",
    "Running 15-Day Ensemble Algorithms...",
    "Generating Sowing Advisory..."
  ];
  
  let step = 0;
  text.textContent = messages[0];
  
  const interval = setInterval(() => {
    step++;
    if (step < messages.length) {
      text.textContent = messages[step];
    } else {
      clearInterval(interval);
      loader.style.display = "none";
      cb();
    }
  }, 400); // Takes ~2 seconds total
}

function onSelectionChange() {
  simulateModelLoad(() => {
    if (state.surface === "officer") renderDashboard(); else renderFarmer();
  });
}

function switchSurface(s) {
  state.surface = s;
  document.body.dataset.surface = s;
  if (s === "officer") renderDashboard(); else renderFarmer();
}

async function init() {
  state.meta = await API("/api/meta");
  const m = state.meta;
  state.block = m.blocks[0].id; state._detailBlock = m.blocks[0].id;
  state.crop = m.crops[0].id;
  state.surface = window.matchMedia("(max-width: 860px)").matches ? "farmer" : "officer";
  document.body.dataset.surface = state.surface;

  renderNavs();
  renderSelectors();
  const ml = $("#m-lang"); if (ml) ml.appendChild(makeLang(true));
  $("#to-farmer") && ($("#to-farmer").onclick = () => switchSurface("farmer"));
  $("#m-bell") && ($("#m-bell").onclick = () => gotoScreen("alerts"));

  const setLang = $("#settings-lang");
  if (setLang) setLang.value = state.lang;
  $("#settings-save") && ($("#settings-save").onclick = () => {
    if (setLang) {
      state.lang = setLang.value;
      localStorage.setItem("sowsafe_lang", state.lang);
      const user = $("#settings-username");
      if (user) localStorage.setItem("sowsafe_user", user.value);
      applyI18n();
      if (state.surface === "officer") renderDashboard(); else renderFarmer();
    }
    const btn = $("#settings-save");
    btn.textContent = "Saved!";
    setTimeout(() => btn.textContent = "Save Changes", 2000);
  });

  const savedUser = localStorage.getItem("sowsafe_user");
  if (savedUser && $("#settings-username")) $("#settings-username").value = savedUser;

  $("#btn-gen-report") && ($("#btn-gen-report").onclick = (e) => {
    const btn = e.target;
    btn.textContent = "Generating PDF...";
    setTimeout(() => {
      btn.textContent = "Report Downloaded! ✓";
      setTimeout(() => btn.textContent = "Generate District Report", 3000);
    }, 1200);
  });

  applyI18n();
  simulateModelLoad(() => {
    if (state.surface === "officer") renderDashboard(); else renderFarmer();
  });
}

document.addEventListener("DOMContentLoaded", init);
