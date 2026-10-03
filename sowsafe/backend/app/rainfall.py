"""Palghar monsoon rainfall engine (physically-grounded, probabilistic).

NOTE ON DATA HONESTY: these are *representative* daily rainfall fields for the
Palghar blocks, calibrated to published Konkan SW-monsoon climatology (onset
timing, seasonal totals, intra-seasonal break spells) and modulated by *real*
ENSO/IOD indices (see climate.py). They are a stand-in for IMD gridded /
GPM-IMERG observations so the prototype runs offline and reproducibly; the
generator is a pluggable interface — swap in real feeds without touching the
engine or UI. The structure (false-onset trap, dry spells, heavy-rain episodes)
is deliberately realistic so the advisory logic can be demonstrated end to end.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib

import numpy as np

from . import config


def _stable_int(*parts) -> int:
    """Deterministic 31-bit seed from arbitrary parts (not Python's salted hash)."""
    s = "|".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def season_dates() -> tuple[list[date], int]:
    start = date.fromisoformat(config.SEASON_START)
    end = date.fromisoformat(config.SEASON_END)
    n = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n)], n


def _zone_params(zone: str) -> dict:
    return {
        "coastal": dict(peak_mm=28.0, wet_p_peak=0.85, heavy_scale=1.35),
        "inland": dict(peak_mm=24.0, wet_p_peak=0.80, heavy_scale=1.15),
        "hilly": dict(peak_mm=22.0, wet_p_peak=0.78, heavy_scale=1.05),
    }.get(zone, dict(peak_mm=24.0, wet_p_peak=0.80, heavy_scale=1.15))


def onset_doy(climate: dict) -> int:
    """Climatological onset shifted by climate state (El Nino delays onset)."""
    base = config.CLIMATOLOGY["normal_onset_doy"]
    shift = 8.0 * climate.get("enso", 0.0) - 4.0 * climate.get("iod", 0.0)
    return int(round(base + shift))


def _seasonal_profile(doy: int, onset: int, zp: dict) -> tuple[float, float]:
    """(p_wet, mean_mm) climatological cycle for a day-of-year."""
    if doy < onset - 12:
        return 0.08, 3.0            # sporadic pre-monsoon showers
    if doy < onset:
        return 0.15, 5.0            # approaching onset
    days_since = doy - onset
    ramp = min(1.0, 0.40 + days_since / 25.0)
    decline = 1.0 - max(0, days_since - 70) / 120.0
    factor = max(0.30, ramp * decline)
    return min(0.95, zp["wet_p_peak"] * factor + 0.10), zp["peak_mm"] * factor


def simulate_season(
    rng: np.random.Generator,
    zone: str,
    climate: dict,
    onset_override: int | None = None,
    inject_false_onset: bool = True,
) -> np.ndarray:
    """One stochastic daily-rainfall realization (mm) for the whole season."""
    dates, n = season_dates()
    onset = onset_override if onset_override is not None else onset_doy(climate)
    zp = _zone_params(zone)
    intensity = max(0.6, 1.0 - 0.12 * climate.get("enso", 0.0) + 0.06 * climate.get("iod", 0.0))
    rain = np.zeros(n)
    wet_prev = False
    for i, d in enumerate(dates):
        doy = d.timetuple().tm_yday
        p_wet, mean_mm = _seasonal_profile(doy, onset, zp)
        if inject_false_onset:
            if onset - 14 <= doy <= onset - 12:
                p_wet, mean_mm = 0.90, 22.0     # tempting early rain (the bait)
            elif onset - 11 <= doy <= onset - 2:
                p_wet, mean_mm = 0.05, 2.0       # the trap: prolonged dry spell
        mean_mm *= intensity
        p = min(0.97, max(0.02, p_wet * (1.25 if wet_prev else 0.85)))
        wet = rng.random() < p
        if wet:
            amt = rng.gamma(shape=1.1, scale=max(1.0, mean_mm))
            if rng.random() < 0.04 * zp["heavy_scale"]:
                amt += rng.gamma(shape=2.0, scale=55.0 * zp["heavy_scale"])  # heavy episode
            rain[i] = float(amt)
        wet_prev = wet
    return rain


def observed_season(block_id: str, zone: str, climate: dict) -> np.ndarray:
    """The single 'truth' realization we replay (fixed per block + year)."""
    seed = _stable_int(config.RANDOM_SEED, block_id)
    rng = np.random.default_rng(seed)
    return simulate_season(rng, zone, climate)


def ensemble_future(
    block_id: str, zone: str, climate: dict, observed: np.ndarray, as_of_idx: int,
    n_members: int = config.ENSEMBLE_SIZE,
) -> np.ndarray:
    """Ensemble of full seasons that MATCH observed up to `as_of_idx` and are
    stochastic afterwards, with onset-timing uncertainty. Shape [n, days]."""
    base_onset = onset_doy(climate)
    members = np.empty((n_members, observed.shape[0]))
    for k in range(n_members):
        seed = _stable_int(config.RANDOM_SEED, "ens", block_id, as_of_idx, k)
        rng = np.random.default_rng(seed)
        sampled_onset = int(round(rng.normal(base_onset, 4.0)))
        real = simulate_season(rng, zone, climate, onset_override=sampled_onset)
        real[: as_of_idx + 1] = observed[: as_of_idx + 1]   # condition on the past
        members[k] = real
    return members
