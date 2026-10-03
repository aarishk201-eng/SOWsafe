# SowSafe — Product Requirements Document (PRD)

> **Hyperlocal Monsoon Onset & Break Prediction System (Block / Village Scale)**
> A decision-first sowing advisory for rainfed farmers.

| | |
|---|---|
| **Product name** | SowSafe (स्वसेफ) |
| **Problem Statement** | SIH26086 — Smart India Hackathon 2026 |
| **Theme / Category** | Agriculture, Food Tech & Rural Development / Software |
| **Team** | Game Changer (ID 185376) |
| **Document owner** | Nafisa Shaikh |
| **Status** | Draft v1.0 — for prototype build |
| **Last updated** | 2026-10-01 |
| **Scope of this PRD** | Full product vision documented; **requirements scoped to a hackathon-grade working prototype** (1 pilot district, 2 crops, 1 Kharif window). See §4. |

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Prototype Scope (MVP)](#4-prototype-scope-mvp)
5. [Users & Personas](#5-users--personas)
6. [User Stories](#6-user-stories)
7. [Product Walkthrough (the core flow)](#7-product-walkthrough-the-core-flow)
8. [The Sowing Safety Score (SSS)](#8-the-sowing-safety-score-sss)
9. [Functional Requirements](#9-functional-requirements)
10. [System Architecture](#10-system-architecture)
11. [Tech Stack](#11-tech-stack)
12. [Data Sources](#12-data-sources)
13. [ML & Modeling Approach](#13-ml--modeling-approach)
14. [API Design](#14-api-design)
15. [Data Model](#15-data-model)
16. [Key Screens / UX](#16-key-screens--ux)
17. [Non-Functional Requirements](#17-non-functional-requirements)
18. [Security, Privacy & Trust](#18-security-privacy--trust)
19. [Success Metrics / KPIs](#19-success-metrics--kpis)
20. [Milestones & Build Plan](#20-milestones--build-plan)
21. [Risks & Mitigations](#21-risks--mitigations)
22. [Assumptions & Open Questions](#22-assumptions--open-questions)
23. [Future Scope](#23-future-scope)
24. [References](#24-references)
25. [Glossary](#25-glossary)

---

## 1. Executive Summary

~51% of India's net sown area is rainfed and ~40% of food production depends on it (ICAR-CRIDA). For a rainfed farmer, **the single most important decision of the season is *when* to sow.** Sow too early and a "false onset" — initial rains followed by a prolonged dry spell — kills germinating seed, forcing costly re-sowing. Sow too late and the crop loses the season.

Today's public services do not answer this decision at the right scale: IMD block forecasts run ~5 days, longer outlooks and the 2025 AI onset SMS are regional, and none of them give a **block/panchayat-level, crop-specific, 7–30 day "what should I do"** call.

**SowSafe is decision-first, not just a forecast.** A farmer provides only three things — **location, crop, and whether they can irrigate** — and SowSafe returns a **Sowing Safety Score (0–100)** with a confidence level and a plain-language action: **Sow Now / Wait / Sow with Irrigation / Change Crop**, delivered in the local language via web (PWA), SMS, WhatsApp, and IVR. Behind the score, an engine fuses global climate drivers (ENSO/IOD/MJO), NWP forecasts, satellite rainfall & soil moisture, and ground gauges, downscales them to the block, and runs a **false-onset check** against the crop's dry-spell tolerance.

This PRD documents the full vision and specifies a **working prototype** that demonstrates the complete farmer journey end-to-end for one pilot district and two Kharif crops.

---

## 2. Problem Statement

**SIH26086 — Hyperlocal Monsoon Onset & Break Prediction System (Block/Village Scale).**

- Rainfed agriculture is ~51% of net sown area (ICAR-CRIDA); **sowing date decides the season.**
- Early rain → sowing → prolonged dry spell (**false onset**) = poor germination, seed loss, re-sowing cost.
- Existing services are coarse: IMD block forecasts ~5 days; longer-range and the 2025 AI onset SMS are **regional**, not block-level.
- There is **no block-level, crop-specific 7–30 day call** on the real question: *sow now, wait, irrigate, or switch crop?*

**Core gap:** farmers get weather numbers, not a decision tied to their crop, their field, and their ability to irrigate.

---

## 3. Goals & Non-Goals

### 3.1 Product Goals
- **G1 — Decision, not data.** Convert a 7–30 day probabilistic rainfall outlook into one of four clear sowing actions with a 0–100 safety score.
- **G2 — Catch false onsets.** Explicitly estimate the probability that initial rains will *not* be sustained beyond the crop's germination/establishment window.
- **G3 — Crop- & irrigation-specific.** Same weather → different advice for different crops and for irrigated vs. purely rainfed fields.
- **G4 — Hyperlocal.** Block / panchayat granularity, not district/region.
- **G5 — Trustworthy.** Every advisory shows its **reason** and a **confidence** level; never a bare number.
- **G6 — Reach every farmer.** Local language, works on a basic phone (SMS/IVR) and low-connectivity (PWA/WhatsApp).
- **G7 — Plug into what exists.** IMD open data, KVK/extension workflows, government SMS/IVR channels.

### 3.2 Prototype Goals (what the demo must prove)
- End-to-end farmer journey works: input → score → action → explanation, in ≥2 languages.
- A reproducible pipeline turns **real open historical data** for the pilot district into onset / dry-spell / heavy-rain probabilities.
- A KVK/officer dashboard shows block-level risk maps and lets an officer review/validate advisories.
- Honest confidence + graceful fallback when data is stale/missing.

### 3.3 Non-Goals (explicitly out of scope for the prototype)
- **N1** — Not a replacement for IMD; we consume IMD/NWP products, we don't produce our own global forecast.
- **N2** — No claim of operational real-time nowcasting during the hackathon; the prototype runs on recent/historical data snapshots and a scheduled refresh, not a 24×7 ops pipeline.
- **N3** — No live national rollout; prototype = **1 district, 2 crops, 1 Kharif window.**
- **N4** — Not providing legally-binding or insurance-grade guarantees; advisories are probabilistic guidance.
- **N5** — No full production SMS/IVR telecom integration in the demo (simulated/sandbox); WhatsApp/SMS shown via sandbox/mock gateway.

---

## 4. Prototype Scope (MVP)

> **Recommended scope (assumption — confirm or adjust).** A hackathon prototype cannot operationally validate a brand-new 7–30 day monsoon model. So we scope for a **convincing, honest, end-to-end working demo** rather than a production forecasting service.

**In scope for the prototype:**
1. **Pilot area:** 1 district (recommended: a rainfed Kharif district with good IMD AWS coverage, e.g. in Maharashtra/Vidarbha or Telangana — *to confirm*), subdivided into its blocks/talukas.
2. **Crops:** 2 Kharif crops with contrasting water needs (recommended: **Soybean** and **Cotton**, or **Rice (nursery/direct)** vs a short-duration pulse) — chosen so the "same weather → different advice" story is visible.
3. **Data:** real **open historical** rainfall + climate indices + reanalysis for the district (IMD gridded rainfall, GPM, ERA5, NOAA ENSO/IOD/MJO indices), pre-downloaded as snapshots; a **scheduled refresh** job demonstrates the pipeline.
4. **Engine:** downscaling/forecast model producing **onset probability, dry-spell risk, heavy-rain risk** per block for a selectable "today" within a historical Kharif season, with calibrated uncertainty.
5. **Advisory:** Sowing Safety Score + 4-way action + reason + confidence, crop- and irrigation-aware.
6. **Delivery:** responsive **Web PWA** (farmer view + officer dashboard) in **English + 2 Indian languages** (e.g. Hindi + Marathi/Telugu), with **text-to-speech**; **SMS/WhatsApp/IVR shown via sandbox/mock**.
7. **Feedback loop:** farmers/officers can submit outcome feedback; stored and surfaced in the dashboard.

**Demo device:** a "time machine" control lets the presenter set the simulated current date inside a past Kharif season, so judges can watch the advice change as a real false-onset event unfolds — and compare it to what actually happened (hindcast).

**Deferred to post-prototype:** live 24×7 ingestion, nationwide blocks, trained deep models at scale, real telecom SMS/IVR contracts, insurer/FPO paid products.

---

## 5. Users & Personas

| Persona | Who | Primary need | Channel |
|---|---|---|---|
| **Rainfed farmer** (primary) | Small/marginal farmer, often low-literacy, basic or smartphone, patchy connectivity | "Is it safe to sow my crop now, or should I wait/irrigate/switch?" in my language | SMS, IVR, WhatsApp, PWA |
| **KVK scientist / Agromet officer** (validator) | District agromet unit, extension officer | Block-level risk overview; review & endorse advisories; push alerts | Web dashboard |
| **State Agriculture Dept / admin** | Program owner | Coverage, adoption, outcomes; configure crops/thresholds | Web dashboard (admin) |
| **Partner (future)** | Insurer / FPO / input dealer | Block risk maps & outlooks | API / paid risk maps |

**Design implication:** the farmer path must work with **zero typing beyond 3 inputs**, offline-tolerant, and voice-first; the officer path is data-rich and map-centric.

---

## 6. User Stories

**Farmer**
- As a farmer, I select my **village/block, crop, and irrigation** so I get advice specific to me.
- As a farmer, I see a **Sowing Safety Score and one clear action** so I know what to do today.
- As a farmer, I hear the advice **in my language by voice** so literacy isn't a barrier.
- As a farmer, I get an **alert before a false onset / dry spell** so I don't waste seed.
- As a farmer with no internet, I get the same advice **by SMS/IVR**.

**KVK / Officer**
- As an officer, I see a **block-level risk map** (onset / dry-spell / heavy-rain) so I can brief my area.
- As an officer, I can **review the reasoning and confidence** and **endorse or override** an advisory before it goes out.
- As an officer, I can **broadcast an alert** to farmers in a block.

**Admin**
- As an admin, I can **add a crop** with its agronomic parameters and **tune decision thresholds** without code.
- As an admin, I can see **adoption and outcome feedback** to measure impact.

---

## 7. Product Walkthrough (the core flow)

```
Farmer opens app / SMS / IVR
      ↓
Selects: Location (block/village) + Crop + Irrigation (yes / no / partial)
      ↓
Engine fuses: ENSO/IOD/MJO  +  IMD/NWP forecasts  +  rainfall  +  satellite  +  soil moisture
      ↓
Downscales to block/panchayat (7–30 days): onset, dry-spell and heavy-rain probability
      ↓
False-onset check: will the rain last beyond THIS crop's dry-spell tolerance?
      ↓
Sowing Safety Score (0–100)  +  Confidence
      ↓
Advice:  Sow Now  /  Wait  /  Sow with Irrigation  /  Change Crop
      ↓
Delivered in local language via Web / SMS / WhatsApp / Voice   →   Feedback captured
```

The flow is identical across channels; only the rendering differs (rich map on PWA, a 2-line message on SMS, a spoken menu on IVR).

---

## 8. The Sowing Safety Score (SSS)

The SSS is SowSafe's core IP: a single, explainable 0–100 number that encodes *"how safe is it to sow **this crop** at **this block** in the current outlook window?"*

### 8.1 Definitions
- **Monsoon onset (local):** first occurrence of sustained sowing-grade rainfall — e.g. ≥ *R_onset* mm accumulated over a short window with continuation (parameters configurable per agro-climatic zone, aligned with IMD/ICAR practice).
- **Dry spell:** a run of consecutive days below a crop-critical rainfall threshold during the **germination + establishment** window.
- **False onset:** onset-grade rain triggers sowing, then a dry spell **longer than the crop's dry-spell tolerance (`T_tol` days)** occurs before establishment → germination/stand failure.
- **Dry-spell tolerance `T_tol`:** crop-specific agronomic parameter (from ICAR/KVK) — days the germinating crop can survive without adequate moisture.

### 8.2 Signal probabilities (per block, per forecast window)
- `P_onset` — probability sowing-grade onset rain occurs within the sowing window.
- `P_sustain = 1 − P(dry spell > T_tol | onset)` — probability rain is sustained through establishment (the false-onset guard).
- `P_heavy` — probability of heavy-rain/waterlogging exceeding the crop's tolerance in the window.
- `SM_adeq` — current + near-term **soil-moisture adequacy** (0–1) from SMAP/reanalysis.

### 8.3 Scoring model (weighted, calibrated)
```
raw = w1·P_onset
    + w2·P_sustain          (false-onset guard — highest weight)
    + w3·(1 − P_heavy)
    + w4·SM_adeq

SSS = 100 × raw × IrrigationAdjustment
```
- Weights `w1..w4` sum to 1, agronomist-tunable per crop (defaults, e.g., w2 largest because sustained rain is what prevents the costly failure). **Not hardcoded magic numbers** — stored in a config/DB table and overridable by admins.
- **IrrigationAdjustment:** irrigation buffers dry-spell risk. If irrigation = *yes*, the penalty from low `P_sustain` is damped (farmer can bridge a dry spell); if *no*, full penalty; *partial* in between.
- Probabilities come from a **calibrated** model (reliability/Brier-checked) so a "70" means roughly 70%.

### 8.4 Score → Action mapping (illustrative, configurable)
| Condition | Action | Why |
|---|---|---|
| `SSS ≥ 70` and `P_sustain` high | **Sow Now** | High chance of sustained rain |
| `40 ≤ SSS < 70`, onset likely *later* | **Wait** | Rain not yet reliable; a better window is coming |
| Moderate SSS, `P_sustain` low, **irrigation available** | **Sow with Irrigation** | Can bridge the forecast dry spell |
| `SSS < 40` with high false-onset/break risk and crop ill-suited | **Change Crop** | This crop is too risky; suggest a shorter-duration/hardier option |

### 8.5 Confidence (shown separately from the score)
Confidence is a function of **model ensemble spread / calibration**, **forecast lead time** (7-day > 30-day), and **data freshness**. Displayed as High / Medium / Low with a one-line reason. **A low-confidence Sow Now is never shown as a certainty.** When inputs are stale/missing, SowSafe falls back to the last good data and **lowers the displayed confidence** rather than hiding the gap.

### 8.6 Explainability
Every result carries a human-readable reason string, e.g.:
> *"Onset likely in 5–8 days (72%), but a 6-day dry spell is likely right after (soybean tolerates ~3). Without irrigation, risk of re-sowing is high → **Wait**. Confidence: Medium (10-day lead)."*

---

## 9. Functional Requirements

Priority: **P0** = must-have for prototype demo, **P1** = should-have, **P2** = nice-to-have/stretch.

### 9.1 Location & input
- **FR-1 (P0)** Farmer selects location by **block/village** via searchable dropdown, map pin, or GPS; SMS/IVR accept a village code.
- **FR-2 (P0)** Farmer selects **crop** from the supported list (2 in prototype).
- **FR-3 (P0)** Farmer selects **irrigation**: none / partial / assured.
- **FR-4 (P1)** Remember last selection on the device for one-tap re-check.

### 9.2 Prediction engine
- **FR-5 (P0)** For a given block + date, produce **onset probability, dry-spell risk, heavy-rain risk** for the 7–30 day window.
- **FR-6 (P0)** Produce a **calibrated probability** (not just a point value) with uncertainty.
- **FR-7 (P0)** Run the **false-onset check** against the selected crop's `T_tol`.
- **FR-8 (P1)** Expose a **"time machine"** date selector (demo) to replay historical seasons.
- **FR-9 (P1)** Scheduled refresh job re-runs predictions when new data snapshots arrive.

### 9.3 Advisory engine
- **FR-10 (P0)** Compute **SSS (0–100)** per the §8 model, crop- & irrigation-aware.
- **FR-11 (P0)** Map SSS + risk flags to one of **Sow Now / Wait / Sow with Irrigation / Change Crop**.
- **FR-12 (P0)** Return a **reason string** and a **confidence** level with every advisory.
- **FR-13 (P1)** "Change Crop" suggests a concrete safer alternative from the crop library.
- **FR-14 (P1)** Decision thresholds & weights are **config-driven and admin-editable** (no redeploy).

### 9.4 Delivery
- **FR-15 (P0)** Responsive **Web PWA**, installable, offline-tolerant (cache last advisory).
- **FR-16 (P0)** **i18n**: UI + advisory strings in English + 2 Indian languages; language switch.
- **FR-17 (P0)** **Text-to-speech** readout of the advisory in the selected language.
- **FR-18 (P1)** **WhatsApp** advisory via sandbox (Twilio/Meta sandbox).
- **FR-19 (P1)** **SMS** advisory via mock/sandbox gateway.
- **FR-20 (P2)** **IVR** voice menu (simulated flow for demo).
- **FR-21 (P1)** Push/alert when a block's risk crosses a threshold (e.g. false-onset warning).

### 9.5 KVK / Officer dashboard
- **FR-22 (P0)** Block-level **choropleth risk maps** for onset / dry-spell / heavy-rain across the district.
- **FR-23 (P0)** Drill into a block: probabilities, SSS by crop, confidence, reason, underlying signals.
- **FR-24 (P1)** **Review & endorse / override** an advisory before broadcast (with an audit trail).
- **FR-25 (P1)** **Broadcast** an alert to a block/crop segment.
- **FR-26 (P2)** Hindcast view: predicted vs. actual for past seasons (validation evidence).

### 9.6 Feedback & monitoring
- **FR-27 (P1)** Farmers/officers submit **outcome feedback** (sowed? germinated? re-sowed?).
- **FR-28 (P1)** Admin sees **model performance** (ROC-AUC, Brier, CRPS on hindcasts) and adoption.
- **FR-29 (P2)** **Drift detection** flag when incoming data distribution shifts.

### 9.7 Admin
- **FR-30 (P1)** CRUD **crop library** (water need, `T_tol`, sowing window, agro-climatic rules).
- **FR-31 (P1)** Edit **SSS weights/thresholds** per crop/zone.
- **FR-32 (P2)** Manage languages and advisory message templates.

---

## 10. System Architecture

A 6-stage pipeline (**Data → Prediction → Risk → Decision → Farmer**, with monitoring closing the loop):

**1. Data Ingestion** — multi-source climate & agricultural data
  - Satellite: GPM, INSAT-3D/3DR, SMAP, NDVI, OLR
  - Weather forecasts / NWP: IMD, GFS, NCUM, ERA5
  - Global climate indices: ENSO (Niño 3.4), IOD (DMI), MJO (RMM)
  - Ground obs: IMD AWS/ARG block-level rain gauges
  - Agri & land: crop calendar, soil type, land cover, sowing dates (ICAR/KVK)

**2. Data Processing & Feature Layer** — clean + build block-level features
  - Cleaning & QC (missing-data handling, outlier removal, gap filling)
  - Spatial/temporal processing (regrid, resample, block/panchayat aggregation, time alignment)
  - Feature engineering (lagged rainfall, ENSO/IOD/MJO features, soil moisture, NDVI, elevation, land cover)
  - Feature store (PostgreSQL + PostGIS, TimescaleDB)

**3. Hybrid Prediction Engine** — global signals → block-level forecasts
  - Inputs: climate drivers + NWP/reanalysis
  - Downscaling ML (CNN / XGBoost / LSTM)
  - Probabilistic outputs (7–30 days): onset probability, dry-spell risk, heavy-rain risk
  - Uncertainty & calibration (ensembles, quantile regression, Brier/ROC-AUC reliability)

**4. Expert System / Advisory Engine** — predictions → farm advice
  - Knowledge base (crop requirements, germination moisture need, dry-spell tolerance, irrigation availability, agro-climatic rules — ICAR/KVK)
  - Decision logic (forecast probabilities, soil-moisture confidence, agro-climatic rules, risk thresholds)
  - Sowing Safety Score (weighted model) → Sow Now / Wait / Sow with Irrigation / Change Crop

**5. Delivery Layer** — reach farmers & officers
  - Web PWA (block risk maps, 7–30 day outlook, crop recommendations, dashboard)
  - Mobile / SMS / WhatsApp (local-language advisories, critical alerts, low-connectivity)
  - IVR / voice (automated calls, regional languages, low-literacy)
  - Localization (Hindi, Marathi, Telugu, Tamil, …; text-to-speech)

**6. Monitoring & Feedback** — keep it accurate & improve
  - Model performance (ROC-AUC, Brier, CRPS, hindcast verification)
  - Scheduled retraining (pre-monsoon & in-season; automated pipeline)
  - Model versioning (experiment tracking, model registry, MLflow)
  - Drift detection (data drift, real-time monitoring)
  - Farmer feedback loop (app/SMS → refine rules & models → DB + analytics)

> Diagrams: see the source deck (`Sawsafe Ppt.pptx`, slide 3) for the architecture flow and the tech-stack pyramid.

---

## 11. Tech Stack

| Layer | Technology |
|---|---|
| **Data ingestion sources** | NOAA, IMD, ISRO, ESA, NASA, Copernicus |
| **Data pipelines** | Apache Airflow, Python, xarray, GDAL, netCDF4 |
| **Storage & feature layer** | PostgreSQL, TimescaleDB, PostGIS, Zarr, MinIO (object store) |
| **Prediction engine** | PyTorch, XGBoost, scikit-learn, statsmodels |
| **Expert / advisory engine** | Python, rule engine, ICAR/KVK knowledge base |
| **Backend API** | FastAPI (Python) |
| **Delivery** | React (PWA), WhatsApp, Twilio (SMS/IVR), Web Speech / TTS |
| **MLOps** | MLflow, scheduled retraining, drift monitoring |

**Prototype-pragmatic subset** (to stay demo-focused): React + Vite PWA · FastAPI · PostgreSQL + PostGIS · XGBoost/scikit-learn (+ optional PyTorch LSTM) · xarray/GDAL/netCDF4 for data prep · Docker Compose for one-command spin-up · Twilio/Meta **sandbox** for WhatsApp/SMS. Airflow/MinIO/TimescaleDB/MLflow are in the architecture but can be represented by lighter equivalents (a cron/APScheduler job, local object storage) in the demo and called out as "productionizes to X."

---

## 12. Data Sources

All inputs are **open**. For the prototype these are pre-downloaded as historical snapshots for the pilot district; the pipeline shows how a scheduled job would refresh them.

| Signal | Source | Role | Access |
|---|---|---|---|
| Gridded daily rainfall | **IMD** (0.25°) | Ground truth, onset/dry-spell labels | Open |
| Satellite precipitation | **GPM / IMERG** (NASA) | Rainfall where gauges are sparse | Open |
| Automatic weather stations | **IMD AWS/ARG** | Block-level ground obs | Open |
| Soil moisture | **SMAP** (NASA) | Soil-moisture adequacy | Open |
| Reanalysis | **ERA5** (Copernicus/ECMWF) | Features, baselines | Open |
| Vegetation / NDVI | **MODIS/Sentinel** (ISRO/ESA/NASA) | Land condition features | Open |
| NWP forecasts | **IMD / NCUM / GFS (NOAA)** | Short–medium range forecast input | Open |
| Climate indices | **NOAA** — ENSO (Niño 3.4), IOD (DMI), MJO (RMM) | Global drivers | Open |
| Crop agronomy | **ICAR / KVK** | `T_tol`, water need, sowing windows, rules | Published |
| Admin boundaries | **Survey of India / district GIS** | Block/panchayat polygons | Open |

> **Fallback policy:** if a feed is late or missing, use the **last good snapshot** and **lower the displayed confidence** — never silently serve stale data as fresh.

---

## 13. ML & Modeling Approach

**Honesty principle:** a hackathon cannot prove a novel operational 7–30 day monsoon model. We therefore ship a **defensible, calibrated, hindcast-validated** pipeline and are explicit about baseline vs. learned components.

- **Targets:** per block & lead window — (a) onset occurrence, (b) dry-spell-exceeds-`T_tol` occurrence, (c) heavy-rain exceedance. All framed as **probabilistic classification**.
- **Features:** lagged/accumulated rainfall, ENSO/IOD/MJO indices & phases, reanalysis fields (ERA5), soil moisture (SMAP), NDVI, elevation/land cover, day-of-season, climatological normals.
- **Models:**
  - **Baseline (must-have):** climatology + logistic/quantile regression — the honest floor every learned model must beat.
  - **Primary:** **XGBoost** probabilistic classifiers (fast, strong on tabular, interpretable via SHAP).
  - **Optional:** **LSTM/CNN** (PyTorch) for sequence/spatial downscaling if time permits.
- **Calibration:** isotonic/Platt scaling; verify with **reliability diagrams, Brier score, ROC-AUC, CRPS**.
- **Validation:** **leave-one-season-out hindcasting** on historical Kharif seasons; report skill vs. baseline and vs. persistence. Showcase at least one real **false-onset season** the model would have flagged.
- **Explainability:** SHAP feature attributions feed the human-readable reason strings.
- **MLOps:** experiment tracking + model registry (MLflow); scheduled pre-monsoon/in-season retraining; drift detection on inputs.

---

## 14. API Design

REST/JSON over HTTPS (FastAPI). Illustrative core endpoints:

```
GET  /api/health
GET  /api/meta/crops                      → supported crops + agronomic params
GET  /api/meta/blocks?district=<id>       → block list + geometry (GeoJSON)
GET  /api/meta/languages                  → supported languages

POST /api/advisory
     body: { block_id, crop_id, irrigation: none|partial|assured, date?, lang }
     → { sss, action, confidence, reason, reason_i18n,
         signals: { p_onset, p_sustain, p_dryspell, p_heavy, sm_adeq },
         window: { from, to }, generated_at, data_freshness }

GET  /api/risk-map?district=<id>&layer=onset|dryspell|heavy&date=<d>
     → GeoJSON FeatureCollection with per-block probabilities (for choropleth)

GET  /api/block/{block_id}/detail?crop=<id>&date=<d>
     → full breakdown for the officer drill-down

POST /api/feedback      body: { block_id, crop_id, advisory_id, outcome, note }
GET  /api/admin/metrics → model perf (roc_auc, brier, crps), adoption, drift
POST /api/admin/thresholds (auth)   → update SSS weights/thresholds
POST /api/notify/broadcast (auth)   → queue SMS/WhatsApp/IVR to a segment (sandbox)
```

- **Auth:** officer/admin endpoints behind authentication + role-based access (see §18). Farmer advisory endpoint is public/read but rate-limited.
- **i18n:** advisory text returned both as a stable `reason` (English, for logs) and `reason_i18n` (localized, for display/TTS).

---

## 15. Data Model

Key entities (PostgreSQL + PostGIS; time series in TimescaleDB):

- **district / block** — `id, name, parent, geom(Polygon), agro_climatic_zone`
- **crop** — `id, name, water_need, dry_spell_tolerance_days (T_tol), sowing_window_start/end, heavy_rain_tolerance, rules_json`
- **observation** (timeseries) — `block_id, date, source, rainfall_mm, soil_moisture, ndvi, …`
- **climate_index** (timeseries) — `date, enso_nino34, iod_dmi, mjo_phase, mjo_amp`
- **forecast** — `block_id, run_date, window_from, window_to, p_onset, p_dryspell, p_heavy, sm_adeq, model_version, confidence`
- **advisory** — `id, block_id, crop_id, irrigation, date, sss, action, reason, confidence, forecast_id, created_at`
- **advisory_review** — `advisory_id, officer_id, status(endorsed/overridden), note, ts` (audit trail)
- **feedback** — `id, advisory_id, block_id, crop_id, outcome, note, source, ts`
- **config_weights** — `crop_id, zone, w1..w4, thresholds_json, updated_by, ts`
- **user** — `id, role(farmer/officer/admin), name, phone(hashed), lang, block_id`
- **model_run** — `model_version, metrics_json(roc_auc,brier,crps), trained_at, data_window`

---

## 16. Key Screens / UX

**Farmer — PWA (mobile-first, voice-first, ≤3 taps to answer):**
1. **Home / Input** — big language toggle; location (GPS/search/map); crop cards (with icons); irrigation toggle; one primary "Get my advice" button.
2. **Result** — a **gauge** showing SSS (0–100) with a colour band; the **action** in large type + icon (Sow Now 🌱 / Wait ⏳ / Irrigate 💧 / Change Crop 🔁); confidence chip; a one-line reason; a **🔊 Listen** button (TTS); "What does this mean?" expander; share to WhatsApp.
3. **Outlook** — simple 7–30 day rain-chance strip for the block; next check reminder.
4. **Feedback** — "Did you sow? What happened?" quick buttons.

**Officer / KVK — Web dashboard:**
1. **District map** — choropleth by layer (onset / dry-spell / heavy-rain), date slider / time-machine, block hover tooltips.
2. **Block detail** — probabilities, SSS per crop, confidence, signal breakdown (SHAP), reason.
3. **Advisory review** — queue of generated advisories → endorse / override + note; broadcast.
4. **Analytics** — model performance (ROC-AUC/Brier/CRPS), adoption, feedback outcomes, drift flags.

**Design tenets:** colour-blind-safe risk palette, large touch targets, works at 2G/offline (cache last advisory), minimal text + icon-led, every number paired with its confidence. Follow the dataviz guidance for the maps, gauge, and outlook charts.

---

## 17. Non-Functional Requirements

- **NFR-1 Performance:** advisory response < 1.5 s (served from precomputed block forecasts); map tiles/choropleth < 2 s.
- **NFR-2 Low-connectivity:** PWA installable, offline cache of last advisory; SMS/IVR paths need no internet on the farmer side; payloads small.
- **NFR-3 Accessibility:** WCAG 2.1 AA; screen-reader labels; TTS; large fonts; colour-blind-safe; low-literacy iconography.
- **NFR-4 Internationalization:** all farmer-facing strings externalized; RTL-safe; add a language without code changes; TTS per language.
- **NFR-5 Reliability / graceful degradation:** stale-data fallback with lowered confidence; never a hard failure to the farmer.
- **NFR-6 Scalability:** block-partitioned precompute; design scales from 1 district to all-India by adding blocks (stateless API, batch forecast jobs).
- **NFR-7 Observability:** structured logs, model metrics, data-freshness and drift monitors.
- **NFR-8 Reproducibility:** `docker compose up` brings up the full prototype with seeded pilot data; deterministic model training with pinned deps and versioned data.
- **NFR-9 Maintainability:** crop params & thresholds are data, not code.

---

## 18. Security, Privacy & Trust

> **Flagged proactively:** the system handles **farmer PII (phone numbers)** and an **officer/admin dashboard that can broadcast messages and change decision thresholds**. These need protection even in a prototype.

- **SEC-1 Access control:** officer/admin features require **authentication + role-based authorization**. The farmer advisory read path can be public but **rate-limited**; broadcast, review, threshold-edit, and metrics endpoints must **not** be reachable without an authenticated privileged role.
- **SEC-2 PII minimization:** collect only what a channel needs (phone for SMS/IVR). **Store phone numbers hashed/encrypted at rest**; never log raw PII; reference by key, not value.
- **SEC-3 Consent:** explicit opt-in before sending SMS/WhatsApp/IVR; easy opt-out; comply with DPDP Act 2023 principles.
- **SEC-4 Transport:** HTTPS everywhere; secrets in env/secret store, never in the repo; pin dependency versions.
- **SEC-5 Input validation:** validate block/crop/date/lang server-side; parameterized DB queries; no string-built SQL.
- **SEC-6 Abuse / cost control:** sandbox messaging in the prototype; broadcasts gated behind officer endorsement to prevent spam and runaway cost.
- **SEC-7 Audit:** advisory reviews, overrides, threshold changes, and broadcasts are logged with actor + timestamp.
- **SEC-8 Trust/UX:** every advisory shows reason + confidence; no advisory is presented as a guarantee (see §8.5–8.6).

---

## 19. Success Metrics / KPIs

**Prototype / demo success**
- End-to-end flow works in ≥2 languages with voice, from input to spoken advice.
- On a held-out historical false-onset season, SowSafe's advice **demonstrably beats** "sow on first rain" and the climatology baseline (fewer failed sowings in replay).
- Model calibration shown (reliability diagram), with ROC-AUC/Brier/CRPS vs. baseline.
- Officer dashboard renders live block risk maps and a working review→broadcast path.

**Product / impact (pilot & beyond)**
- % reduction in false-onset sowings / re-sowing incidents in the pilot block vs. control.
- Advisory adoption rate; % farmers acting on advice; feedback-confirmed good outcomes.
- Forecast skill sustained across seasons (hindcast + live verification).
- Coverage: blocks live; farmers reached by channel; officer endorsements.
- Cost per advisory delivered (reusing government SMS/IVR channels).

---

## 20. Milestones & Build Plan

A suggested build order for the prototype (adjust to your hackathon timeline). Each milestone is independently demoable.

| # | Milestone | Deliverable | Priority |
|---|---|---|---|
| **M0** | Repo & scaffold | Monorepo (`frontend/` React-PWA, `backend/` FastAPI, `ml/`, `data/`, `infra/` docker-compose), seeded Postgres+PostGIS, health check | P0 |
| **M1** | Data layer | Download & stage IMD rainfall + ENSO/IOD/MJO + ERA5/SMAP for pilot district; cleaning/QC + feature build; block polygons loaded | P0 |
| **M2** | Prediction engine | Baseline + XGBoost onset/dry-spell/heavy-rain probabilities per block; calibration; hindcast metrics | P0 |
| **M3** | Advisory engine | Crop library + SSS model + action mapping + reason/confidence strings | P0 |
| **M4** | API | `/advisory`, `/risk-map`, `/block/detail`, `/meta/*` wired to the engine | P0 |
| **M5** | Farmer PWA | Input → result gauge → action → reason → TTS; i18n (EN + 2 langs); offline cache | P0 |
| **M6** | Officer dashboard | Choropleth risk maps + time-machine + block drill-down | P0 |
| **M7** | Channels | WhatsApp/SMS via sandbox; alert on threshold; IVR flow (sim) | P1 |
| **M8** | Review & feedback | Endorse/override + audit; feedback capture; admin metrics & thresholds | P1 |
| **M9** | Polish & demo | False-onset replay script, screenshots, pitch, seed-data reset, README | P0 |

**Definition of done (prototype):** `docker compose up` → open the PWA → pick block+crop+irrigation → get SSS + action + spoken reason in a local language; open the dashboard → see block risk maps and replay a historical false onset; metrics page shows calibration vs. baseline.

---

## 21. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Late/missing data feeds | Wrong/stale advice | Last-good-data fallback + **lower confidence**; freshness indicator |
| 7–30 day skill is inherently limited | Over-promising | Always show **probability + confidence**; validate vs. baseline; frame as guidance not guarantee |
| Model overfits few seasons | Poor generalization | Leave-one-season-out hindcast; prefer calibrated baseline+XGBoost over fragile deep nets |
| Low literacy / language barrier | Low adoption | Icon-led UI, TTS, IVR, local languages |
| Low/no connectivity | Can't reach farmer | PWA offline cache + SMS/IVR paths |
| PII / messaging misuse | Privacy & cost | Hash PII, consent/opt-out, officer-gated broadcasts, sandbox in demo (§18) |
| Scope creep in hackathon | Nothing fully works | Strict P0 set; each milestone independently demoable |
| Crop params wrong | Bad advice | Params sourced from ICAR/KVK, officer-validated, admin-editable |

---

## 22. Assumptions & Open Questions

**Assumptions (confirm/adjust):**
- A1 — Prototype scope = **1 district, 2 crops, 1 Kharif window**, demoed on historical data with a time-machine (per §4).
- A2 — Pilot district & the 2 crops are **TBD by the team**; recommendation in §4.
- A3 — Prototype tech = React PWA + FastAPI + PostgreSQL/PostGIS + XGBoost, Docker Compose; messaging via sandbox.
- A4 — "Confidence" is derived from calibration + lead time + data freshness (§8.5).

**Open questions for you:**
1. Which **district** and which **2 crops** should the prototype target? (drives data download)
2. Which **2 languages** besides English? (e.g. Hindi + Marathi, or + Telugu)
3. For the hackathon demo, is **historical replay** acceptable, or do you need a **near-real-time** feed on current dates?
4. Do you want the **full tech stack** (Airflow/MinIO/TimescaleDB/MLflow) stood up, or the **pragmatic subset** with a clear "productionizes to X" story?
5. Any **branding** constraints (name shown as "SowSafe" / "SowSafe" / logo from the deck)?

---

## 23. Future Scope

- Live 24×7 ingestion & nationwide block coverage; full Airflow/MinIO/TimescaleDB/MLflow stack.
- Spatial deep-learning downscaling (CNN/ConvLSTM) once multi-year data is staged.
- Real telecom SMS/IVR integration via government channels (the Ministry's SMS platform reached ~38M farmers in 2025).
- Rabi season & more crops; pest/disease and irrigation-scheduling advisories.
- Paid **risk maps** for insurers, FPOs, input dealers (sustainability model; free for farmers, funded by State ag depts/KVKs).
- Integration with crop insurance (PMFBY) and agro-advisory ecosystems.

---

## 24. References

- Monsoon Onset Prediction — Echo State Networks: https://iopscience.iop.org/article/10.1088/1748-9326/ac0acb
- NCMRWF S2S Break Monsoon Study: https://nwp.ncmrwf.gov.in/publication/NCUM_S2S_TR_May2019.pdf
- Deep Learning Precipitation Downscaling — IMD: https://internal.imd.gov.in/press_release/20250114_pr_3552.pdf
- IOD & Indian Monsoon Rainfall Response: https://www.nature.com/articles/s41598-017-18396-6
- Extended-Range Monsoon Forecasting — IMD/NCMRWF: https://doi.org/10.1007/s12040-026-02736-0
- Source deck: `Sawsafe Ppt.pptx` (SIH26086, Team Game Changer).

---

## 25. Glossary

- **Onset (monsoon):** first sustained sowing-grade rainfall at a location.
- **Break / dry spell:** a run of days with inadequate rainfall during a crop-critical window.
- **False onset:** sowing-triggering rain followed by a dry spell longer than the crop can survive → stand failure, re-sowing.
- **`T_tol` (dry-spell tolerance):** crop-specific days of moisture stress the germinating crop can survive.
- **SSS (Sowing Safety Score):** SowSafe's 0–100 crop- & irrigation-aware safety index for sowing now.
- **ENSO / IOD / MJO:** large-scale climate drivers (El Niño–Southern Oscillation / Indian Ocean Dipole / Madden–Julian Oscillation) that modulate the Indian monsoon.
- **NWP:** Numerical Weather Prediction (e.g., GFS, NCUM).
- **Reanalysis:** blended historical atmospheric state (e.g., ERA5).
- **PWA:** Progressive Web App — installable, offline-capable web app.
- **Brier / ROC-AUC / CRPS:** scores for probabilistic-forecast quality & calibration.
- **Hindcast:** running the model on past seasons to measure skill against what actually happened.
- **KVK:** Krishi Vigyan Kendra — district farm-science/extension centre.

---

*End of PRD v1.0. This is a living document — update §22 as decisions are made, then we proceed to the prototype scaffold (M0).*

