"""SowSafe API — FastAPI app serving precomputed advisories + the static UI.

Endpoints are intentionally thin: the heavy ensemble work is precomputed into
the store (see store.py); here we only look up a day's *signal* and run the cheap
scoring/advice step for the chosen crop + irrigation. One uvicorn process serves
both the JSON API (/api/*) and the no-build frontend (/).
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import scoring
from .store import get_store

app = FastAPI(title="SowSafe API", version="0.1.0")

_FRONTEND = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
_FEEDBACK = os.path.join(os.path.dirname(__file__), "..", "data", "feedback.jsonl")

STORE = get_store()
_N = len(STORE["dates"])
_BLOCK_IDS = {b["id"]: b for b in STORE["meta"]["blocks"]}
_CROP_IDS = {c["id"] for c in STORE["meta"]["crops"]}


def _clamp_day(day: int) -> int:
    return max(0, min(_N - 1, day))


def _sig(block: str, crop: str, day: int) -> dict:
    if block not in _BLOCK_IDS:
        raise HTTPException(404, f"unknown block '{block}'")
    if crop not in _CROP_IDS:
        raise HTTPException(404, f"unknown crop '{crop}'")
    return STORE["sigs"][block][crop][_clamp_day(day)]


@app.get("/api/meta")
def meta() -> dict:
    """Everything the UI needs to render once: blocks, crops, dates, languages."""
    return STORE["meta"] | {"dates": STORE["dates"], "n_days": _N}


@app.get("/api/advisory")
def advisory(
    block: str = Query(...),
    crop: str = Query(...),
    irrigation: str = Query("none"),
    day: int = Query(...),
) -> dict:
    """The farmer's answer: Sowing Safety Score + action + reason (3 languages)."""
    d = _clamp_day(day)
    sig = _sig(block, crop, d)
    adv = scoring.advise(sig, crop, irrigation)
    obs = STORE["observed"][block]
    return {
        "block": block,
        "crop": crop,
        "irrigation": irrigation,
        "day": d,
        "date": STORE["dates"][d],
        "signals": sig,
        **adv,
        "observed_to_date": obs[: d + 1],
    }


@app.get("/api/risk-map")
def risk_map(
    crop: str = Query(...),
    irrigation: str = Query("none"),
    day: int = Query(...),
) -> dict:
    """Per-block advisory for one crop/day — powers the officer choropleth."""
    if crop not in _CROP_IDS:
        raise HTTPException(404, f"unknown crop '{crop}'")
    d = _clamp_day(day)
    rows = []
    for b in STORE["meta"]["blocks"]:
        sig = STORE["sigs"][b["id"]][crop][d]
        adv = scoring.advise(sig, crop, irrigation)
        # Effective onset probability: a block whose onset is already observed
        # reads as (near-)certain for the map's "Onset" heat field.
        p_onset = 1.0 if sig["onset_done"] else sig["p_onset_7"]
        rows.append({
            "block": b["id"], "name": b["name"], "lat": b["lat"], "lon": b["lon"],
            "zone": b["zone"], "sss": adv["sss"], "action": adv["action"],
            "action_labels": adv["action_labels"],
            "p_dryspell": sig["p_dryspell"], "p_heavy": sig["p_heavy"],
            "p_onset": round(p_onset, 3), "rain": STORE["observed"][b["id"]][d],
            "confidence_label": sig["confidence_label"],
        })
    return {"crop": crop, "irrigation": irrigation, "day": d, "date": STORE["dates"][d], "blocks": rows}


@app.get("/api/block/{block}")
def block_detail(
    block: str,
    crop: str = Query(...),
    irrigation: str = Query("none"),
    day: int = Query(...),
) -> dict:
    """Drill-down: advisory + the rainfall the farmer has actually seen so far."""
    d = _clamp_day(day)
    sig = _sig(block, crop, d)
    adv = scoring.advise(sig, crop, irrigation)
    obs = STORE["observed"][block]
    # Full 129-day signal trajectories — powers the dual-axis chart, the
    # monsoon timeline milestones, and the soil-moisture curve with real data.
    cells = STORE["sigs"][block][crop]
    series = {
        "p_onset_7": [round(c["p_onset_7"], 4) for c in cells],
        "p_dryspell": [round(c["p_dryspell"], 4) for c in cells],
        "p_heavy": [round(c["p_heavy"], 4) for c in cells],
        "sm_adeq": [round(c["sm_adeq"], 4) for c in cells],
        "onset_done": [bool(c["onset_done"]) for c in cells],
    }
    return {
        "block": block, "name": _BLOCK_IDS[block]["name"], "crop": crop,
        "irrigation": irrigation, "day": d, "date": STORE["dates"][d],
        "signals": sig, **adv,
        "observed_to_date": obs[: d + 1],
        "observed_full": obs,
        "series": series,
    }


@app.post("/api/feedback")
async def feedback(payload: dict) -> dict:
    """Lightweight 'was this useful?' capture (append-only JSONL)."""
    rec = {"ts": datetime.utcnow().isoformat(), **(payload or {})}
    os.makedirs(os.path.dirname(_FEEDBACK), exist_ok=True)
    with open(_FEEDBACK, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "days": _N, "blocks": len(_BLOCK_IDS), "crops": sorted(_CROP_IDS)}


# Static frontend last so /api/* wins. html=True serves index.html at /.
if os.path.isdir(_FRONTEND):
    app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="frontend")
