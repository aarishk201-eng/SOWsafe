"""Sowing Safety Score (SSS) + advisory decision engine (the expert system).

SSS = 100 * [ w_onset*onset + w_sustain*sustain + w_heavy*(1-heavy) + w_sm*sm ]
with an irrigation adjustment that lets irrigation buffer dry-spell risk.
Weights/thresholds are declared here as data (agronomist-tunable), not scattered.
"""
from __future__ import annotations

from .crops import get_crop, crop_display, FALLBACK_CROPS

WEIGHTS = {"onset": 0.30, "sustain": 0.38, "heavy": 0.17, "sm": 0.15}  # sum = 1.0
IRRIGATION_FACTOR = {"none": 0.0, "partial": 0.5, "assured": 1.0}

ACTIONS = {
    "sow_now":        {"en": "Sow Now",             "hi": "अभी बुवाई करें",     "mr": "आता पेरणी करा",     "icon": "🌱", "color": "#15803d"},
    "wait":           {"en": "Wait",                "hi": "प्रतीक्षा करें",      "mr": "थांबा",            "icon": "⏳", "color": "#b45309"},
    "sow_irrigation": {"en": "Sow with Irrigation", "hi": "सिंचाई के साथ बुवाई", "mr": "सिंचनासह पेरणी करा", "icon": "💧", "color": "#1d4ed8"},
    "change_crop":    {"en": "Change Crop",         "hi": "फसल बदलें",          "mr": "पीक बदला",         "icon": "🔁", "color": "#b91c1c"},
}


def compute_sss(sig: dict, crop_id: str, irrigation: str) -> dict:
    crop = get_crop(crop_id)
    irr = IRRIGATION_FACTOR.get(irrigation, 0.0)

    onset = 1.0 if sig["onset_done"] else sig["p_onset_7"]
    # Irrigation buffers a dry spell up to the crop's responsiveness to irrigation.
    base_sustain = sig["p_sustain"]
    eff_sustain = base_sustain + (1.0 - base_sustain) * irr * crop["irrigation_benefit"]
    heavy_pen = sig["p_heavy"] * crop["waterlogging_sensitivity"]
    heavy_term = max(0.0, 1.0 - heavy_pen)
    sm = sig["sm_adeq"]

    raw = (WEIGHTS["onset"] * onset + WEIGHTS["sustain"] * eff_sustain
           + WEIGHTS["heavy"] * heavy_term + WEIGHTS["sm"] * sm)
    # Waterlogging hazard gate: for a flood-sensitive crop, a likely heavy-rain
    # spell turns abundant rain into a liability. The linear heavy term alone
    # can't overturn an otherwise high score, so we pull the final score down
    # past a hazard floor — a "change crop" day then reads red, not green.
    excess = max(0.0, heavy_pen - 0.25)
    flood_factor = max(0.2, 1.0 - 2.6 * excess)
    sss = int(round(100 * raw * flood_factor))
    return {
        "sss": max(0, min(100, sss)),
        "onset_term": round(onset, 3),
        "eff_sustain": round(eff_sustain, 3),
        "heavy_term": round(heavy_term, 3),
        "sm_term": round(sm, 3),
        "flood_factor": round(flood_factor, 3),
    }


def decide_action(sig: dict, sss: int, crop_id: str, irrigation: str) -> str:
    crop = get_crop(crop_id)
    irr = irrigation in ("partial", "assured")
    assured = irrigation == "assured"
    waterlog = sig["p_heavy"] * crop["waterlogging_sensitivity"]
    late = sig.get("window_progress", 0.0) >= 0.5   # running out of time to sow this crop

    # 1) Waterlogging-sensitive crop heading into a likely heavy-rain spell.
    #    Heavy rain inflates the rain-driven SSS, so for a flood-prone crop we
    #    judge on the hazard itself: a strong heavy-rain signal means switch to a
    #    hardier crop; a moderate one means hold off. (Irrigation can't fix a flood.)
    if crop["waterlogging_sensitivity"] >= 0.6 and waterlog >= 0.45:
        return "change_crop" if waterlog >= 0.48 else "wait"
    # 2) Deeply unsafe AND the sowing window is slipping -> switch crop.
    if sss < 40 and sig["p_sustain"] < 0.45 and not assured and late:
        return "change_crop"
    # 3) Strong green light.
    if sss >= 70 and (sig["p_sustain"] >= 0.6 or assured):
        return "sow_now"
    # 4) Dry-spell risk but irrigation on hand -> sow with irrigation.
    if sig["p_sustain"] < 0.55 and irr and sss >= 45:
        return "sow_irrigation"
    # 5) Rain/better window coming soon, or just too early -> wait.
    if not sig["onset_done"] and sig.get("onset_eta_days") is not None \
            and sig["onset_eta_days"] <= 15 and sig["p_onset_14"] >= 0.45:
        return "wait"
    # 6) Reasonable score and rain will hold -> sow; else hold off.
    if sss >= 58 and (sig["p_sustain"] >= 0.5 or assured):
        return "sow_now"
    return "wait"


def _pct(x: float) -> int:
    return int(round(100 * x))


def build_reason(action: str, sig: dict, crop_id: str, irrigation: str, lang: str) -> str:
    crop = get_crop(crop_id)
    cname = crop_display(crop_id, lang)
    t_tol = crop["dry_spell_tolerance_days"]
    on = _pct(sig["p_onset_7"]) if not sig["onset_done"] else 100
    dry = _pct(sig["p_dryspell"])
    heavy = _pct(sig["p_heavy"])
    eta = sig.get("onset_eta_days")
    fb_id = crop["fallback_crop"]
    fb = (FALLBACK_CROPS.get(fb_id) or {}).get(f"name_{lang}") or (FALLBACK_CROPS.get(fb_id) or {}).get("name_en", fb_id)
    conf = {"en": sig["confidence_label"],
            "hi": {"High": "उच्च", "Medium": "मध्यम", "Low": "कम"}[sig["confidence_label"]],
            "mr": {"High": "उच्च", "Medium": "मध्यम", "Low": "कमी"}[sig["confidence_label"]]}[lang]

    T = REASONS[lang][action]
    return T.format(crop=cname, on=on, dry=dry, heavy=heavy, ttol=t_tol,
                    eta=eta if eta is not None else "?", fb=fb, conf=conf)


REASONS = {
    "en": {
        "sow_now": "Onset rain is reliable and likely to continue past {crop}'s {ttol}-day tolerance (dry-spell risk {dry}%). Conditions are good to sow. Confidence: {conf}.",
        "wait": "Rain is not yet dependable (onset chance {on}% this week; a {dry}% chance of a dry spell longer than {crop} can survive). Better to wait ~{eta} days. Confidence: {conf}.",
        "sow_irrigation": "A dry spell is likely ({dry}% risk, above {crop}'s {ttol}-day limit), but with irrigation you can bridge it. Sow with irrigation ready. Confidence: {conf}.",
        "change_crop": "Heavy, waterlogging rain is likely ({heavy}% chance) and {crop} is easily damaged by it. A hardier crop like {fb} is the safer choice now. Confidence: {conf}.",
    },
    "hi": {
        "sow_now": "बुवाई की बारिश भरोसेमंद है और {crop} की {ttol}-दिन सहन-सीमा के बाद भी जारी रहने की संभावना है (सूखे की आशंका {dry}%)। बुवाई के लिए अच्छा समय। विश्वास: {conf}।",
        "wait": "बारिश अभी भरोसेमंद नहीं है (इस हफ़्ते बारिश की संभावना {on}%; {dry}% संभावना कि सूखा {crop} की सहन-सीमा से लंबा हो)। लगभग {eta} दिन प्रतीक्षा करें। विश्वास: {conf}।",
        "sow_irrigation": "सूखे की आशंका है ({dry}%, {crop} की {ttol}-दिन सीमा से अधिक), पर सिंचाई से इसे संभाला जा सकता है। सिंचाई के साथ बुवाई करें। विश्वास: {conf}।",
        "change_crop": "भारी जलभराव वाली बारिश की प्रबल संभावना है ({heavy}%), जिसे {crop} सह नहीं सकती। अभी {fb} जैसी मज़बूत फसल बेहतर विकल्प है। विश्वास: {conf}।",
    },
    "mr": {
        "sow_now": "पेरणीचा पाऊस भरवशाचा असून {crop}च्या {ttol}-दिवस सहनशीलतेनंतरही सुरू राहण्याची शक्यता आहे (कोरड्या खंडाचा धोका {dry}%). पेरणीसाठी चांगली वेळ. विश्वास: {conf}.",
        "wait": "पाऊस अजून भरवशाचा नाही (या आठवड्यात पावसाची शक्यता {on}%; {dry}% शक्यता की कोरडा खंड {crop} सहन करू शकेल त्यापेक्षा मोठा असेल). सुमारे {eta} दिवस थांबा. विश्वास: {conf}.",
        "sow_irrigation": "कोरड्या खंडाचा धोका आहे ({dry}%, {crop}च्या {ttol}-दिवस मर्यादेपेक्षा जास्त), पण सिंचनाने तो भरून काढता येईल. सिंचनासह पेरणी करा. विश्वास: {conf}.",
        "change_crop": "जोरदार, पाणी साचवणारा पाऊस येण्याची दाट शक्यता आहे ({heavy}%), जो {crop} सहन करू शकत नाही. आता {fb} सारखे टणक पीक अधिक सुरक्षित. विश्वास: {conf}.",
    },
}


def advise(sig: dict, crop_id: str, irrigation: str) -> dict:
    score = compute_sss(sig, crop_id, irrigation)
    action = decide_action(sig, score["sss"], crop_id, irrigation)
    reasons = {lang: build_reason(action, sig, crop_id, irrigation, lang) for lang in ("en", "hi", "mr")}
    return {
        "sss": score["sss"],
        "action": action,
        "action_labels": ACTIONS[action],
        "reason": reasons["en"],
        "reason_i18n": reasons,
        "terms": score,
    }
