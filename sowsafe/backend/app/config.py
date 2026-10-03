"""Core configuration for the SowSafe prototype.

Pilot: Palghar district (Konkan, Maharashtra). Kharif season replay demo.
All tunables live here so nothing downstream hardcodes magic numbers.
"""
from __future__ import annotations

DISTRICT = "Palghar"
STATE = "Maharashtra"

# Demo season we replay with the "time machine". Konkan SW-monsoon window.
SEASON_YEAR = 2023
SEASON_START = f"{SEASON_YEAR}-05-25"   # pre-onset
SEASON_END = f"{SEASON_YEAR}-09-30"     # established monsoon

# Forecast horizon shown to the farmer (days).
FORECAST_MIN_DAYS = 7
FORECAST_MAX_DAYS = 30

# The 8 talukas (blocks) of Palghar. Rough centroids (lat, lon) + a coastal flag.
# Coastal/western blocks get heavier rain; eastern tribal blocks (Jawhar, Mokhada)
# are hillier with more variable spells.
BLOCKS = [
    {"id": "palghar",   "name": "Palghar",   "lat": 19.69, "lon": 72.77, "zone": "coastal"},
    {"id": "vasai",     "name": "Vasai",     "lat": 19.39, "lon": 72.83, "zone": "coastal"},
    {"id": "dahanu",    "name": "Dahanu",    "lat": 19.98, "lon": 72.74, "zone": "coastal"},
    {"id": "talasari",  "name": "Talasari",  "lat": 20.17, "lon": 72.83, "zone": "coastal"},
    {"id": "wada",      "name": "Wada",      "lat": 19.65, "lon": 73.13, "zone": "inland"},
    {"id": "vikramgad", "name": "Vikramgad", "lat": 19.92, "lon": 73.08, "zone": "inland"},
    {"id": "jawhar",    "name": "Jawhar",    "lat": 19.91, "lon": 73.23, "zone": "hilly"},
    {"id": "mokhada",   "name": "Mokhada",   "lat": 19.93, "lon": 73.34, "zone": "hilly"},
]

# Monsoon climatology for Konkan/Palghar (calibration targets for the generator).
CLIMATOLOGY = {
    "normal_onset_doy": 160,      # ~June 9 (day-of-year) typical SW monsoon onset
    "season_total_mm": 2400,      # Palghar seasonal rainfall is high (~2000-2600 mm)
    "onset_rain_threshold_mm": 20,    # a "sowing rain" day
    "onset_window_days": 3,           # sustained over N days = onset
    "dry_spell_rain_threshold_mm": 2.5,  # below this = a dry day for spell counting
    "heavy_rain_threshold_mm": 115,      # IMD "heavy rainfall" ~ 64.5-115.5; waterlogging risk here
}

# Ensemble size for the probabilistic engine (stochastic realizations).
ENSEMBLE_SIZE = 50

# Reproducibility.
RANDOM_SEED = 20260601

# Supported UI languages.
LANGUAGES = ["en", "hi", "mr"]
