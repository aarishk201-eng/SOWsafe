"""Crop agronomy knowledge base (the ICAR/KVK layer).

Parameters drive the Sowing Safety Score. Values are representative, sourced
from standard Kharif agronomy for the Konkan region, and are meant to be
agronomist-validated and editable (not model-internal constants).

The two pilot crops are deliberately contrasting:
  - Rice (paddy): WANTS sustained monsoon rain; nursery/transplant timing is
    driven by onset. Tolerant of standing water (low waterlogging penalty).
  - Okra (bhindi): shorter window, moderate water need, and FEARS waterlogging
    / heavy rain. Same weather -> different advice.
"""
from __future__ import annotations

CROPS = {
    "rice": {
        "id": "rice",
        "name_en": "Rice (paddy)",
        "name_hi": "धान (चावल)",
        "name_mr": "भात",
        "icon": "🌾",
        # Days the just-sown/germinating crop survives moisture stress.
        "dry_spell_tolerance_days": 7,
        # Rain (mm over onset window) needed to safely sow/raise nursery.
        "germination_rain_mm": 25,
        # Relative water demand through establishment (0-1).
        "water_need": 0.9,
        # Penalty weight for heavy-rain/waterlogging (0 = loves water .. 1 = ruined).
        "waterlogging_sensitivity": 0.15,
        # Typical sowing/nursery window (month-day).
        "sow_window": ["06-01", "07-15"],
        # How much irrigation can rescue a dry spell (0-1).
        "irrigation_benefit": 0.8,
        # Safer alternative to suggest if this crop is too risky.
        "fallback_crop": "finger_millet",
        "note_en": "Onset-driven: raise nursery when sustained rain is likely.",
    },
    "okra": {
        "id": "okra",
        "name_en": "Okra (bhindi)",
        "name_hi": "भिंडी",
        "name_mr": "भेंडी",
        "icon": "🫑",
        "dry_spell_tolerance_days": 4,
        "germination_rain_mm": 15,
        "water_need": 0.5,
        # Okra hates waterlogging -> high heavy-rain penalty.
        "waterlogging_sensitivity": 0.75,
        "sow_window": ["06-10", "08-15"],
        "irrigation_benefit": 0.9,
        "fallback_crop": "cowpea",
        "note_en": "Needs drained soil: avoid sowing into a heavy-rain / waterlogging spell.",
    },
}

# Minimal data for suggested fallback crops (names only; shown in 'Change Crop').
FALLBACK_CROPS = {
    "finger_millet": {"name_en": "Finger millet (nagli)", "name_hi": "रागी", "name_mr": "नाचणी"},
    "cowpea": {"name_en": "Cowpea (chawli)", "name_hi": "लोबिया", "name_mr": "चवळी"},
}


def get_crop(crop_id: str) -> dict:
    if crop_id not in CROPS:
        raise KeyError(f"Unknown crop: {crop_id}")
    return CROPS[crop_id]


def crop_display(crop_id: str, lang: str = "en") -> str:
    c = CROPS.get(crop_id) or FALLBACK_CROPS.get(crop_id, {})
    return c.get(f"name_{lang}") or c.get("name_en", crop_id)
