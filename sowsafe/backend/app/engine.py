"""Hybrid prediction engine: ensemble -> calibrated event probabilities.

Turns the rainfall ensemble (conditioned on observations up to 'today') into the
three probabilistic outputs the product promises, per block and per crop:
  * onset probability (sufficient sowing rain in the coming window)
  * dry-spell risk beyond the crop's tolerance  (the false-onset guard)
  * heavy-rain / waterlogging risk
plus soil-moisture adequacy and a forecast-confidence estimate.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from . import config
from .crops import get_crop


def _window_progress(crop: dict, as_of_idx: int) -> float:
    """How far into this crop's sowing window we are (0 = not yet, 1 = window over)."""
    start = date.fromisoformat(config.SEASON_START)
    w0, w1 = crop["sow_window"]
    ws = (date.fromisoformat(f"{config.SEASON_YEAR}-{w0}") - start).days
    we = (date.fromisoformat(f"{config.SEASON_YEAR}-{w1}") - start).days
    if as_of_idx < ws:
        return 0.0
    if as_of_idx > we:
        return 1.0
    return round((as_of_idx - ws) / max(1, we - ws), 3)


def _longest_dry_run(series: np.ndarray, thr: float) -> int:
    best = cur = 0
    for v in series:
        if v < thr:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _onset_start_in(series: np.ndarray, thr_mm: float, win: int) -> bool:
    """True if an onset-grade spell (>= thr over `win` days) starts in series."""
    if len(series) < 1:
        return False
    acc = np.convolve(series, np.ones(win), mode="valid")
    return bool(np.any(acc >= thr_mm))


def soil_moisture_adequacy(observed: np.ndarray, as_of_idx: int, span: int = 10) -> float:
    """Antecedent Precipitation Index over the last `span` days, normalized 0-1."""
    lo = max(0, as_of_idx - span + 1)
    recent = observed[lo : as_of_idx + 1]
    if recent.size == 0:
        return 0.0
    weights = np.linspace(0.5, 1.0, recent.size)         # recent days weigh more
    api = float(np.sum(recent * weights))
    return float(np.clip(api / 120.0, 0.0, 1.0))         # ~120mm -> fully adequate


def assess(block_id: str, crop_id: str, as_of_idx: int,
           observed: np.ndarray, ensemble: np.ndarray, climate: dict) -> dict:
    crop = get_crop(crop_id)
    clim = config.CLIMATOLOGY
    n_days = observed.shape[0]

    # Establishment window after sowing today (germination + early growth).
    t_tol = crop["dry_spell_tolerance_days"]
    estab = min(n_days - as_of_idx - 1, max(t_tol + 5, 14))
    f_lo, f_hi = as_of_idx + 1, as_of_idx + 1 + estab
    future = ensemble[:, f_lo:f_hi]                       # [members, estab]

    # --- onset probability (next 7 / 14 days) ---
    def p_onset(days: int) -> float:
        hi = min(n_days, as_of_idx + 1 + days)
        seg = ensemble[:, as_of_idx + 1 : hi]
        if seg.shape[1] < clim["onset_window_days"]:
            return 0.0
        hits = [
            _onset_start_in(m, clim["onset_rain_threshold_mm"], clim["onset_window_days"])
            for m in seg
        ]
        return float(np.mean(hits))

    p_onset_7 = p_onset(7)
    p_onset_14 = p_onset(14)

    # Has onset already occurred by 'today' (from observations)?
    past = observed[: as_of_idx + 1]
    onset_done = _onset_start_in(past, clim["onset_rain_threshold_mm"], clim["onset_window_days"]) \
        and soil_moisture_adequacy(observed, as_of_idx) > 0.35

    # --- dry-spell beyond tolerance in establishment window (false-onset guard) ---
    dry_thr = clim["dry_spell_rain_threshold_mm"]
    dry_runs = np.array([_longest_dry_run(m, dry_thr) for m in future])
    p_dryspell = float(np.mean(dry_runs > t_tol))
    p_sustain = 1.0 - p_dryspell

    # --- heavy-rain / waterlogging risk in window ---
    p_heavy = float(np.mean(np.any(future >= clim["heavy_rain_threshold_mm"], axis=1)))

    # --- soil moisture adequacy (now) ---
    sm_adeq = soil_moisture_adequacy(observed, as_of_idx)

    # --- confidence: tighter ensemble + more soil info + shorter lead => higher ---
    window_tot = ensemble[:, f_lo:f_hi].sum(axis=1)
    spread = float(np.std(window_tot) / (np.mean(window_tot) + 1e-6))   # coeff of variation
    conf = float(np.clip(1.0 - spread / 1.4, 0.1, 0.95))
    conf = 0.65 * conf + 0.2 * sm_adeq + 0.15 * (1.0 if onset_done else 0.4)
    conf = float(np.clip(conf, 0.1, 0.95))
    conf_label = "High" if conf >= 0.66 else "Medium" if conf >= 0.4 else "Low"

    # Median onset ETA (days from today) across members, if not yet onset.
    onset_eta = None
    if not onset_done:
        etas = []
        for m in ensemble:
            acc = np.convolve(m[as_of_idx + 1 :], np.ones(clim["onset_window_days"]), mode="valid")
            idx = np.argmax(acc >= clim["onset_rain_threshold_mm"]) if np.any(
                acc >= clim["onset_rain_threshold_mm"]) else None
            if idx is not None:
                etas.append(int(idx))
        if etas:
            onset_eta = int(np.median(etas))

    return {
        "p_onset_7": round(p_onset_7, 3),
        "p_onset_14": round(p_onset_14, 3),
        "p_sustain": round(p_sustain, 3),
        "p_dryspell": round(p_dryspell, 3),
        "p_heavy": round(p_heavy, 3),
        "sm_adeq": round(sm_adeq, 3),
        "onset_done": bool(onset_done),
        "onset_eta_days": onset_eta,
        "confidence": round(conf, 3),
        "confidence_label": conf_label,
        "estab_days": int(estab),
        "window_progress": _window_progress(crop, as_of_idx),
    }
