"""Tests for confidence scoring, bucket thresholds, and empirical calibration verification (Root Mirror).

Pass gate:
- Confidence reports never claim 'calibrated probability' without empirical evidence.
- Bucket thresholds are consistent across score ranges.
- If calibrated (Platt scaling against held-out labels), calibration error (ECE) is tested and within tolerance.
"""

import os
import sys
import pytest
import numpy as np

_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "satquery", "backend"))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

from evidence.confidence import (
    bucket,
    ui_bucket,
    build_confidence_report,
    expected_calibration_error,
    PlattCalibrator,
    fuse_evidence,
)


def test_confidence_label_never_claims_calibration_without_evidence():
    output = build_confidence_report(scores=[0.86, 0.91, 0.88])
    assert "calibrated probability" not in output["label"].lower()
    assert output["label"] in ["internal confidence score", "agreement score"]


def test_bucket_thresholds_are_consistent():
    assert bucket(0.89) == "high"
    assert bucket(0.67) == "medium"
    assert bucket(0.34) == "low"


def test_ui_bucket_formatting():
    assert ui_bucket(0.89) == "🟢 high"
    assert ui_bucket(0.67) == "🟡 medium"
    assert ui_bucket(0.34) == "🔴 low"
    assert ui_bucket("high") == "🟢 high"
    assert ui_bucket("medium") == "🟡 medium"
    assert ui_bucket("low") == "🔴 low"


def test_fused_evidence_exposes_internal_score_and_buckets():
    fused = fuse_evidence(
        optical={"prediction": "built_up", "confidence": 0.82},
        sar={"prediction": "built_up", "confidence": 0.89},
    )
    assert fused["confidence"] > 0.8
    assert fused["agreement"] == "high"
    assert fused["ui_bucket"] == "🟢 high"
    assert fused["score_type"] == "internal_score"
    assert fused["calibrated"] is False


def test_calibration_error_within_tolerance():
    """Pass gate: When calibrating via Platt scaling against labeled data,

    Expected Calibration Error (ECE) is empirically verified to be within tolerance (ECE < 0.15).
    """
    np.random.seed(42)

    n_samples = 200
    true_probs = np.random.uniform(0.1, 0.9, n_samples)
    labels = np.random.binomial(1, true_probs)
    raw_scores = 1.0 / (1.0 + np.exp(-(true_probs * 4.0 - 2.0)))

    calibrator = PlattCalibrator()
    calibrator.fit(raw_scores, labels)

    assert calibrator.fitted is True
    assert calibrator.ece_ is not None
    assert calibrator.ece_ < 0.15

    # Without empirical calibrator, even requesting calibrated=True rejects the claim
    unverified_report = build_confidence_report(scores=[0.86, 0.91, 0.88], calibrated=True)
    assert "calibrated probability" not in unverified_report["label"].lower()
    assert unverified_report["calibrated"] is False
    assert unverified_report["score_type"] == "internal_score"

    report = build_confidence_report(
        scores=list(calibrator.predict_proba(raw_scores[:10])),
        calibrated=True,
        calibrator=calibrator,
        ece_tolerance=0.15,
    )
    assert report["calibrated"] is True
    assert report["label"] == "calibrated probability"
    assert report["score_type"] == "calibrated_probability"
    assert report["ece"] < 0.15

