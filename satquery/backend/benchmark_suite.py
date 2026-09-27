"""Bounded, fully-offline benchmark over local validation fixtures.

This computes the same metric families as the full VRSBench / RSVQA / CDVQA
evaluation (``scripts/evaluate_benchmarks.py``) — VQA accuracy, captioning
coverage, change-VQA accuracy, grounding detection and latency percentiles —
but over a small curated set of georeferenced fixtures so it finishes in a few
seconds inside a live HTTP request.

Every number is derived from the model's REAL output on REAL pixels. There are
no hardcoded answers: the ground-truth labels below describe what each fixture's
pixels physically represent, and the metrics measure how often the specialist
models recover that truth. A genuine miss (e.g. a built-up tile the optical
analyzer reads as bare soil) is reported as a miss, not hidden.
"""

import math
import os
import time
from typing import Any, Dict, List

_FIX = os.path.join(os.path.dirname(__file__), "fixtures")

# Land-cover VQA/captioning fixtures: filename -> ground-truth land-cover class.
_LANDCOVER_GT = [
    ("optical_water.tif", "water"),
    ("sample_water.tif", "water"),
    ("optical_vegetation.tif", "vegetation"),
    ("sample_landcover.tif", "vegetation"),
    ("sample_urban.tif", "built_up"),
    ("optical_built_up.tif", "built_up"),
    ("desert_scene.tif", "bare_soil"),
]

# Bi-temporal change fixtures: (t1, t2, change_expected).
_CHANGE_GT = [
    ("t1_base.tif", "t2_changed.tif", True),
    ("t1_base.tif", "t2_identical.tif", False),
]

# Grounding fixtures: (filename, phrase) — detection is scored, not IoU (no GT box).
_GROUNDING = [
    ("sample_urban.tif", "building"),
    ("optical_built_up.tif", "built-up area"),
]

# Keyword sets a correct caption should mention for each land-cover class.
_CAPTION_KW = {
    "water": ["water"],
    "vegetation": ["vegetation", "canopy"],
    "built_up": ["built-up", "urban", "impervious"],
    "bare_soil": ["soil", "arid", "bare"],
}


def _p(name: str) -> str:
    return os.path.join(_FIX, name)


def _pct(n: int, d: int):
    return round(100.0 * n / d, 1) if d else None


def _percentile(sorted_ms: List[float], p: float):
    if not sorted_ms:
        return None
    idx = int(math.ceil(p / 100.0 * len(sorted_ms))) - 1
    idx = min(len(sorted_ms) - 1, max(0, idx))
    return round(sorted_ms[idx], 1)


def run_quick_benchmark() -> Dict[str, Any]:
    """Runs the bounded offline benchmark and returns a metrics + breakdown dict."""
    from models.vqa import vqa_analyzer, caption
    from models.change_analysis import change_vqa
    from models.grounding import locate

    latencies: List[float] = []

    # --- VQA land-cover accuracy -------------------------------------------- #
    vqa_correct = vqa_total = 0
    vqa_rows: List[Dict[str, Any]] = []
    for fn, gt in _LANDCOVER_GT:
        path = _p(fn)
        if not os.path.exists(path):
            continue
        t0 = time.time()
        out = vqa_analyzer(path, "What is the predominant land cover in this scene?")
        latencies.append(time.time() - t0)
        pred = str(out.get("prediction"))
        ok = pred == gt
        vqa_correct += int(ok)
        vqa_total += 1
        vqa_rows.append({
            "fixture": fn, "expected": gt, "predicted": pred,
            "correct": ok, "confidence": out.get("confidence"),
        })

    # --- Captioning keyword-coverage ---------------------------------------- #
    cap_hits = cap_total = 0
    cap_rows: List[Dict[str, Any]] = []
    for fn, gt in _LANDCOVER_GT:
        path = _p(fn)
        if not os.path.exists(path):
            continue
        t0 = time.time()
        text = str(caption(path))
        latencies.append(time.time() - t0)
        hit = any(k in text.lower() for k in _CAPTION_KW.get(gt, [gt]))
        cap_hits += int(hit)
        cap_total += 1
        cap_rows.append({
            "fixture": fn, "expected_cover": gt,
            "mentions_expected": hit, "caption": text[:180],
        })

    # --- Change-VQA accuracy ------------------------------------------------ #
    cd_correct = cd_total = 0
    cd_rows: List[Dict[str, Any]] = []
    for t1, t2, expect in _CHANGE_GT:
        p1, p2 = _p(t1), _p(t2)
        if not (os.path.exists(p1) and os.path.exists(p2)):
            continue
        t0 = time.time()
        ans = change_vqa(p1, p2, "Has the built-up area changed between these scenes?")
        latencies.append(time.time() - t0)
        pred_change = str(ans).strip().lower().startswith("yes")
        ok = pred_change == expect
        cd_correct += int(ok)
        cd_total += 1
        cd_rows.append({
            "t1": t1, "t2": t2, "expected_change": expect,
            "predicted_change": pred_change, "correct": ok, "answer": str(ans)[:160],
        })

    # --- Grounding detection rate ------------------------------------------- #
    gr_hits = gr_total = 0
    gr_conf: List[float] = []
    gr_rows: List[Dict[str, Any]] = []
    for fn, phrase in _GROUNDING:
        path = _p(fn)
        if not os.path.exists(path):
            continue
        t0 = time.time()
        res = locate(path, phrase)
        latencies.append(time.time() - t0)
        bbox = res.get("bbox") if isinstance(res, dict) else None
        conf = float(res.get("confidence", 0.0)) if isinstance(res, dict) else 0.0
        detected = bbox is not None
        gr_hits += int(detected)
        gr_total += 1
        if detected:
            gr_conf.append(conf)
        gr_rows.append({"fixture": fn, "phrase": phrase, "detected": detected,
                        "bbox": bbox, "confidence": round(conf, 4)})

    latencies_ms = sorted(x * 1000.0 for x in latencies)

    vqa_acc = _pct(vqa_correct, vqa_total)
    cap_score = _pct(cap_hits, cap_total)
    cd_acc = _pct(cd_correct, cd_total)
    gr_rate = _pct(gr_hits, gr_total)

    scored = [x for x in (vqa_acc, cap_score, cd_acc, gr_rate) if x is not None]
    composite = round(sum(scored) / len(scored), 1) if scored else None

    return {
        "mode": "quick-offline-fixtures",
        "description": (
            "Bounded offline evaluation over local georeferenced validation "
            "fixtures. Every metric is computed from real specialist-model "
            "output on real pixels. Run scripts/evaluate_benchmarks.py for the "
            "full VRSBench / RSVQA / CDVQA splits."
        ),
        "metrics": {
            "vqa_accuracy": vqa_acc,
            "captioning_score": cap_score,
            "change_vqa_accuracy": cd_acc,
            "grounding_detection_rate": gr_rate,
            "composite_score": composite,
            "latency_p50_ms": _percentile(latencies_ms, 50),
            "latency_p95_ms": _percentile(latencies_ms, 95),
            "mean_grounding_confidence": round(sum(gr_conf) / len(gr_conf), 4) if gr_conf else None,
            "samples_evaluated": vqa_total + cap_total + cd_total + gr_total,
        },
        "breakdown": {
            "vqa": vqa_rows,
            "captioning": cap_rows,
            "change_vqa": cd_rows,
            "grounding": gr_rows,
        },
    }
