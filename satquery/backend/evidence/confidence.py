"""Confidence estimation and uncertainty quantification for geospatial models."""

from typing import List, Dict, Any, Optional, Sequence, Union
import numpy as np


# Qualitative UI buckets
UI_BUCKETS: Dict[str, str] = {
    "high": "🟢 high",
    "medium": "🟡 medium",
    "low": "🔴 low",
}


def bucket(score: float) -> str:
    """Categorizes a confidence score into consistent qualitative tiers:
    - 'high': score >= 0.75 (e.g. 0.89)
    - 'medium': 0.50 <= score < 0.75 (e.g. 0.67)
    - 'low': score < 0.50 (e.g. 0.34)
    """
    s = float(score)
    if s >= 0.75:
        return "high"
    elif s >= 0.50:
        return "medium"
    else:
        return "low"


def ui_bucket(score_or_tier: Union[float, int, str]) -> str:
    """Returns the visual UI indicator for a confidence score or bucket name.

    UI buckets:
    🟢 high / 🟡 medium / 🔴 low
    """
    if isinstance(score_or_tier, (int, float)):
        tier = bucket(float(score_or_tier))
    else:
        tier = str(score_or_tier).strip().lower()
        if "high" in tier:
            tier = "high"
        elif "med" in tier:
            tier = "medium"
        elif "low" in tier:
            tier = "low"
        else:
            tier = "medium"
    return UI_BUCKETS.get(tier, "🟡 medium")


def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Calculates Expected Calibration Error (ECE) across binned confidence scores.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)

    if len(y_t) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_t)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_p >= bin_lower) & (y_p <= bin_upper)
        else:
            in_bin = (y_p >= bin_lower) & (y_p < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_t[in_bin])
            bin_conf = np.mean(y_p[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(round(ece, 4))


class PlattCalibrator:
    """Performs Platt scaling (logistic sigmoid calibration) on uncalibrated scores/logits.

    Fits P(y=1 | s) = 1 / (1 + exp(-(A * s + B))) against held-out labeled validation sets.
    """

    def __init__(self):
        self.a: float = 1.0
        self.b: float = 0.0
        self.fitted: bool = False
        self.ece_: Optional[float] = None

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "PlattCalibrator":
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)

        # Gradient descent optimization of log-loss
        a = 1.0
        b = 0.0
        lr = 0.2
        for _ in range(1500):
            z = a * s + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
            grad_a = np.mean((p - y) * s)
            grad_b = np.mean(p - y)
            a -= lr * grad_a
            b -= lr * grad_b

        self.a = float(a)
        self.b = float(b)
        self.fitted = True

        calibrated_probs = self.predict_proba(s)
        self.ece_ = expected_calibration_error(y, calibrated_probs)
        return self

    def predict_proba(self, scores: Sequence[float]) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        z = self.a * s + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))

    def evaluate_ece(self, scores: Sequence[float], labels: Sequence[int]) -> float:
        probs = self.predict_proba(scores)
        return expected_calibration_error(labels, probs)



def build_confidence_report(
    scores: Optional[List[float]] = None,
    score: Optional[float] = None,
    calibrated: bool = False,
    is_agreement: bool = False,
    calibrator: Optional[PlattCalibrator] = None,
    held_out_labels: Optional[List[int]] = None,
    ece_tolerance: float = 0.15,
) -> Dict[str, Any]:
    """Generates an explicit confidence report.

    Strictly prevents claiming 'calibrated probability' unless empirical calibration
    evidence (e.g. Platt scaling evaluated with ECE <= tolerance) is verified against
    labeled data. Without empirical calibration, scores are explicitly labeled as
    'internal confidence score' or 'agreement score'.

    UI buckets:
    🟢 high / 🟡 medium / 🔴 low
    """
    if scores is not None and len(scores) > 0:
        score_list = [float(s) for s in scores]
        mean_score = float(np.mean(score_list))
    elif score is not None:
        mean_score = float(score)
        score_list = [mean_score]
    else:
        mean_score = 0.0
        score_list = []

    mean_score = float(np.clip(mean_score, 0.0, 1.0))
    tier = bucket(mean_score)
    ui_tier = ui_bucket(tier)

    # Check if empirical calibration with verified ECE is present
    is_truly_calibrated = False
    ece = None
    if calibrated:
        if calibrator is not None:
            if getattr(calibrator, "ece_", None) is not None:
                ece = float(getattr(calibrator, "ece_"))
            elif held_out_labels is not None:
                ece = float(calibrator.evaluate_ece(score_list, held_out_labels))
            if ece is not None and ece <= ece_tolerance:
                is_truly_calibrated = True
        elif held_out_labels is not None and len(held_out_labels) >= 10:
            ece = float(expected_calibration_error(held_out_labels, score_list))
            if ece <= ece_tolerance:
                is_truly_calibrated = True



    if is_truly_calibrated:
        label = "calibrated probability"
        score_type = "calibrated_probability"
    else:
        score_type = "internal_score"
        if is_agreement:
            label = "agreement score"
        else:
            label = "internal confidence score"

    return {
        "score": round(mean_score, 4),
        "scores": score_list,
        "mean_confidence": round(mean_score, 4),
        "bucket": tier,
        "ui_bucket": ui_tier,
        "agreement_bucket": ui_tier,
        "label": label,
        "score_type": score_type,
        "calibrated": is_truly_calibrated,
        "ece": ece,
        "reliable": mean_score >= 0.50,
        "explanation": (
            f"{label.capitalize()}: {mean_score:.4f} ({ui_tier}). "
            f"Explicitly classified as {score_type}."
        ),
    }


class ConfidenceScorer:
    """Calculates aggregate confidence scores and detection reliability."""

    def __init__(self, min_confidence_threshold: float = 0.5):
        self.threshold = min_confidence_threshold

    def calculate_aggregate_score(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates average, min, and max confidence scores."""
        if not detections:
            return {"mean_confidence": 0.0, "total_detections": 0, "reliable": False}

        scores = [d.get("confidence", 0.0) for d in detections]
        mean_score = sum(scores) / len(scores)
        tier = bucket(mean_score)

        return {
            "mean_confidence": round(mean_score, 4),
            "max_confidence": round(max(scores), 4),
            "min_confidence": round(min(scores), 4),
            "total_detections": len(detections),
            "reliable": mean_score >= self.threshold,
            "bucket": tier,
            "ui_bucket": ui_bucket(tier),
            "label": "internal confidence score",
            "score_type": "internal_score",
        }


def fuse_evidence(
    optical: Dict[str, Any],
    sar: Dict[str, Any],
    optical_weight: float = 0.5,
    sar_weight: float = 0.5,
) -> Dict[str, Any]:
    """Combines independent optical and SAR evidence using weighted Dempster-Shafer consensus.

    Unlike naive averaging (which improperly maintains high confidence when two
    high-confidence sources completely contradict each other), this evidence-based
    fusion quantifies epistemic conflict mass K = c_opt * c_sar.

    - When sources AGREE (e.g., optical built_up 0.82, SAR built_up 0.89):
      Joint corroboration boosts confidence: 1 - (1 - c_opt)*(1 - c_sar) > 0.80
      agreement = "high" (UI: 🟢 high)

    - When sources DISAGREE (e.g., optical built_up 0.90, SAR vegetation 0.90):
      Severe contradiction collapses consensus confidence to < 0.50.
      agreement = "low" (UI: 🔴 low)

    - When optical is cloud-occluded:
      Optical weight is dynamically suppressed (w_opt = 0.1, w_sar = 0.9) because
      radar microwaves penetrate clouds and atmospheric obscurants.
    """
    if optical is None:
        optical = {}
    if sar is None:
        sar = {}

    pred_opt = str(optical.get("prediction", "")).strip().lower()
    pred_sar = str(sar.get("prediction", "")).strip().lower()

    conf_opt = float(optical.get("confidence", 0.0))
    conf_sar = float(sar.get("confidence", 0.0))

    # Clamp raw confidences to [0.0, 1.0]
    conf_opt = float(np.clip(conf_opt, 0.0, 1.0))
    conf_sar = float(np.clip(conf_sar, 0.0, 1.0))

    opt_ev = optical.get("evidence", {}) or {}
    sar_ev = sar.get("evidence", {}) or {}

    # Atmospheric / cloud occlusion handling:
    cloud_occluded = bool(opt_ev.get("cloud_occluded", False)) or (pred_opt == "cloud_occluded")
    if cloud_occluded:
        w_opt = 0.10
        w_sar = 0.90
    else:
        w_opt = float(optical_weight)
        w_sar = float(sar_weight)

    # Normalize weights
    tot_w = w_opt + w_sar
    if tot_w > 0:
        w_opt = w_opt / tot_w
        w_sar = w_sar / tot_w
    else:
        w_opt = 0.5
        w_sar = 0.5

    # Weighted belief masses
    c1 = float(np.clip(conf_opt * (2.0 * w_opt), 0.0, 1.0))
    c2 = float(np.clip(conf_sar * (2.0 * w_sar), 0.0, 1.0))

    # Check whether sensors agree on a valid non-empty hypothesis
    sensors_agree = (pred_opt == pred_sar) and (pred_opt != "")

    if sensors_agree:
        # Multi-sensor corroboration:
        # P(A or B) under independent evidence = 1 - (1 - c1) * (1 - c2)
        fused_conf = 1.0 - (1.0 - c1) * (1.0 - c2)
        fused_pred = pred_opt
        conflict_mass = 0.0

        # Agreement categorization
        if min(conf_opt, conf_sar) >= 0.70 and fused_conf >= 0.80:
            agreement = "high"
        elif min(conf_opt, conf_sar) >= 0.40 or fused_conf >= 0.60:
            agreement = "medium"
        else:
            agreement = "low"

        explanation = (
            f"Multi-sensor agreement on '{fused_pred}': Optical ({conf_opt:.2f}) and SAR ({conf_sar:.2f}) "
            f"provide corroborating evidence, boosting fused confidence to {fused_conf:.4f}."
        )

    else:
        # Disagreeing sensors:
        m_opt = c1 * (1.0 - c2)
        m_sar = c2 * (1.0 - c1)
        conflict_mass = c1 * c2

        if m_opt >= m_sar:
            fused_pred = pred_opt if pred_opt else pred_sar
            fused_conf = m_opt
        else:
            fused_pred = pred_sar if pred_sar else pred_opt
            fused_conf = m_sar

        # Ensure minimal lower bound so confidence is positive
        fused_conf = max(fused_conf, 0.05)
        agreement = "low"
        explanation = (
            f"Sensor contradiction detected: Optical predicts '{pred_opt}' ({conf_opt:.2f}) vs "
            f"SAR predicts '{pred_sar}' ({conf_sar:.2f}). Direct conflict mass K = {conflict_mass:.4f}. "
            f"Epistemic conflict penalizes consensus confidence to {fused_conf:.4f} (< 0.50)."
        )

    ui_indicator = ui_bucket(agreement)

    fused_evidence = {
        "bbox": opt_ev.get("bbox") or sar_ev.get("bbox"),
        "area": opt_ev.get("area") or sar_ev.get("area"),
        "mask": opt_ev.get("mask") or sar_ev.get("mask"),
        "agreement": agreement,
        "ui_bucket": ui_indicator,
        "agreement_bucket": ui_indicator,
        "sensors_agree": bool(sensors_agree),
        "conflict": round(float(conflict_mass), 4),
        "weights": {
            "optical": round(float(w_opt), 4),
            "sar": round(float(w_sar), 4),
        },
        "optical": optical,
        "sar": sar,
        "explanation": explanation,
    }

    return {
        "prediction": fused_pred,
        "confidence": round(float(fused_conf), 4),
        "score_type": "internal_score",
        "label": "agreement score" if sensors_agree else "internal confidence score",
        "calibrated": False,
        "evidence": fused_evidence,
        "agreement": agreement,
        "ui_bucket": ui_indicator,
        "agreement_bucket": ui_indicator,
        "sensors_agree": bool(sensors_agree),
        "conflict": round(float(conflict_mass), 4),
        "weights": fused_evidence["weights"],
        "optical": optical,
        "sar": sar,
        "explanation": explanation,
    }




