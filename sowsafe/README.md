# 🌾 SowSafe — Hyperlocal Monsoon Sowing Advisory

**SIH 2026 · PS SIH26086** — *Hyperlocal Monsoon Onset & Break Prediction System (Block/Village Scale)*

A decision-first sowing advisory for rainfed farmers. A farmer picks **location + crop + irrigation**
and gets back one clear answer: a **Sowing Safety Score (0–100)**, a **confidence level**, and a single
**action** — *Sow Now · Wait · Sow with Irrigation · Change Crop* — with a plain-language reason, in
**English / Hindi / Marathi**, with text-to-speech.

Pilot: **Palghar district** (Konkan, Maharashtra) · Crops: **Rice** and **Okra (Bhindi)** · Season replayed: **Kharif 2023**.

---

## Why it matters
Rainfed farmers lose a whole sowing when they trust a **false onset** — tempting early rain followed by a
dry break that kills germinating seed. SowSafe fuses onset probability, dry-spell (break) risk beyond the
*crop's own tolerance*, heavy-rain/waterlogging risk, and soil-moisture adequacy into one score, and turns
it into an action a farmer can act on today.

## Run it (one command)
```bash
cd sowsafe/backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8077
```
Then open **http://127.0.0.1:8077**. One process serves the JSON API (`/api/*`) and the no-build frontend.

First run builds the forecast store (~50 s, ensemble precompute) and caches it to `backend/data/store.json`;
every run after that starts instantly. To force a rebuild: `python -m app.store`.

Requirements: Python 3.10+, `fastapi`, `uvicorn`, `numpy`.
```bash
pip install fastapi uvicorn numpy
```

## 🎬 Demo script (3 minutes)
Use the **Advisory date** slider as a *time machine* over Kharif 2023. Block **Palghar**, crop toggled Rice/Okra.

| # | Set | What you see | The point |
|---|-----|--------------|-----------|
| 1 | **25 May** (day 0), Rice/Okra, irr *None* | Both **Wait**, low score | Too early — onset rain isn't dependable yet |
| 2 | **early June** (days 6–16) | Still **Wait** despite a tempting shower | **False-onset guard** — a dry break is coming that the crop can't survive |
| 3 | **12 June** (day 18), irr *None* | **Rice → Sow Now (71)** but **Okra → Wait (56)** | *Same weather, different crop → different advice.* Okra's 4-day dry tolerance < Rice's 7 |
| 4 | **12 June** (day 18), Okra, irr *Assured* | Okra flips **Wait → Sow Now (70)** | **Irrigation buffers dry-spell risk** — the score and action respond |
| 5 | **14–18 June** (days 20–24) | Both **Sow Now**, high score, High confidence | **True onset** — reliable rain that will sustain |
| 6 | **27 June** (day 33), irr *None* | **Rice → Sow Now (98)** but **Okra → Change Crop (33)** | Heavy-rain spell likely; Okra waterlogs. *Switch to a hardier crop (Cowpea)* |
| 7 | Switch to **Officer** tab, crop **Okra**, slide to late June/July | Coastal blocks (Vasai, Talasari, Palghar) glow **red/amber**; hilly east (Jawhar, Mokhada) stays **green** | **Hyperlocal** — risk varies block to block within one district |

Press **🔊 Listen** on any advisory to hear it in the selected language. Switch **EN / हिं / मरा** anytime.

<!-- MORE -->

## How the score is built
```
SSS = 100 · [ 0.30·Onset + 0.38·Sustain + 0.17·(1 − HeavyPenalty) + 0.15·SoilMoisture ] · FloodFactor
```
- **Onset** — probability of an onset-grade sowing rain in the coming window (or 1 if onset already observed).
- **Sustain** = 1 − P(dry spell longer than *this crop's* tolerance). Irrigation lifts it toward 1 by the
  crop's irrigation-benefit (an *assured* source buffers a break; *none* doesn't).
- **HeavyPenalty** = P(heavy rain) × crop's waterlogging sensitivity.
- **FloodFactor** — for a flood-sensitive crop, a strong heavy-rain signal pulls the score down past the
  linear term, so a *Change Crop* day reads red, not green. (Abundant rain is a hazard, not an asset, for okra.)

The **action** is a small, auditable expert-system decision tree over these signals (see
[`app/scoring.py`](backend/app/scoring.py)) — not a black box. Weights and thresholds live as data so an
agronomist can tune them.

## Architecture
```
Climate indices (ENSO/IOD, real 2023)                 ┌─────────────── Frontend (no build) ───────────────┐
        │                                              │ Farmer: SSS gauge · action · reason · 🔊 TTS       │
        ▼                                              │ Officer: SVG risk map · time machine · drill-down  │
 rainfall.py  → stochastic daily rainfall (per block)  │ EN / HI / MR · PWA (installable, offline shell)    │
        │     (false-onset trap baked in, calibrated)  └───────────────────────┬────────────────────────────┘
        ▼                                                                       │ fetch /api/*
 engine.py   → ensemble → calibrated event probabilities      ┌────────────────┴───────────────┐
        │     (onset / dry-spell / heavy / soil moisture)      │ FastAPI (app/main.py)           │
        ▼                                                      │ /api/meta /advisory /risk-map   │
 store.py    → precompute signals per block×crop×day  ───────▶ │ /block/{id} /feedback /health   │
        │     (cached to data/store.json)                      └─────────────────────────────────┘
        ▼
 scoring.py  → SSS + action + reason (EN/HI/MR), computed on demand
```

| Module | Role |
|--------|------|
| [`config.py`](backend/app/config.py) | District, blocks (8 talukas), season, climatology, all tunables |
| [`crops.py`](backend/app/crops.py) | Crop agronomy: dry-spell tolerance, waterlogging sensitivity, sow window, fallback |
| [`climate.py`](backend/app/climate.py) | Real 2023 ENSO/IOD indices; `fetch_live()` plug-in point for NOAA feeds |
| [`rainfall.py`](backend/app/rainfall.py) | Physically-grounded probabilistic rainfall + conditioned ensemble |
| [`engine.py`](backend/app/engine.py) | Ensemble → onset / dry-spell / heavy-rain / soil-moisture probabilities + confidence |
| [`scoring.py`](backend/app/scoring.py) | Sowing Safety Score + action decision tree + trilingual reasons |
| [`store.py`](backend/app/store.py) | One-time precompute + disk cache for instant serving |
| [`main.py`](backend/app/main.py) | FastAPI endpoints + static frontend mount |

## A note on data honesty
The rainfall fields are **representative** daily series for the Palghar blocks, calibrated to published
Konkan SW-monsoon climatology (onset timing, seasonal totals, break spells) and modulated by **real**
2023 ENSO/IOD indices. They stand in for IMD-gridded / GPM-IMERG observations so the prototype runs
**offline and reproducibly**. The generator is a pluggable interface — swap in real feeds via
`climate.fetch_live()` and a rainfall data adapter **without touching the engine, scoring, or UI**. The
false-onset trap in the season is deliberate, so the advisory's core value is demonstrable end to end.

## What's a prototype vs. the full product
**Built here:** engine, scoring, API, trilingual PWA UI, officer dashboard, 1 district / 2 crops / 1 season.
**Full product (deck):** Airflow + xarray/GDAL ingestion, PostGIS/TimescaleDB/Zarr, NWP + satellite +
soil-moisture fusion, ML calibration (Brier/CRPS), WhatsApp/Twilio/IVR delivery, MLflow — see [`PRD.md`](PRD.md).

