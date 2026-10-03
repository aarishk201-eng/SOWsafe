"""Precompute + cache the demo data store.

For every block, crop and day-of-season we precompute the engine *signals*
(the expensive ensemble step). The advisory (SSS + action + reason) is cheap and
computed on demand from a signal + the chosen irrigation, so the store stays
small and the API is instant. Results are cached to data/store.json.
"""
from __future__ import annotations

import json
import os
import time

from . import climate, config, engine, rainfall
from .crops import CROPS

_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "store.json")
_STORE: dict | None = None


def build_store(verbose: bool = True) -> dict:
    t0 = time.time()
    clim = climate.get_climate()
    dates, n = rainfall.season_dates()
    pilot_crops = list(CROPS.keys())

    observed: dict[str, list[float]] = {}
    sigs: dict[str, dict[str, list[dict]]] = {}

    for blk in config.BLOCKS:
        bid, zone = blk["id"], blk["zone"]
        obs = rainfall.observed_season(bid, zone, clim)
        observed[bid] = [round(float(x), 1) for x in obs]
        sigs[bid] = {c: [None] * n for c in pilot_crops}
        for i in range(n):
            ens = rainfall.ensemble_future(bid, zone, clim, obs, i)
            for crop in pilot_crops:
                sigs[bid][crop][i] = engine.assess(bid, crop, i, obs, ens, clim)
        if verbose:
            print(f"  built {bid} ({time.time() - t0:.1f}s)")

    store = {
        "meta": {
            "district": config.DISTRICT,
            "state": config.STATE,
            "season_year": config.SEASON_YEAR,
            "blocks": config.BLOCKS,
            "crops": [{"id": c["id"], **{k: c[k] for k in ("name_en", "name_hi", "name_mr", "icon")}}
                      for c in CROPS.values()],
            "languages": config.LANGUAGES,
            "climate": clim,
            "forecast_days": [config.FORECAST_MIN_DAYS, config.FORECAST_MAX_DAYS],
        },
        "dates": [d.isoformat() for d in dates],
        "observed": observed,
        "sigs": sigs,
    }
    if verbose:
        print(f"Store built in {time.time() - t0:.1f}s ({n} days x {len(config.BLOCKS)} blocks)")
    return store


def get_store(rebuild: bool = False) -> dict:
    global _STORE
    if _STORE is not None and not rebuild:
        return _STORE
    if os.path.exists(_CACHE) and not rebuild:
        with open(_CACHE, "r", encoding="utf-8") as f:
            _STORE = json.load(f)
        return _STORE
    _STORE = build_store()
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    with open(_CACHE, "w", encoding="utf-8") as f:
        json.dump(_STORE, f)
    return _STORE


if __name__ == "__main__":
    get_store(rebuild=True)
