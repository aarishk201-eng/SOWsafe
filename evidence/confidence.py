"""Root-level export forwarding to satquery.backend.evidence.confidence."""

import os
import sys

_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "satquery", "backend"))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

from satquery.backend.evidence.confidence import (
    ConfidenceScorer,
    fuse_evidence,
    bucket,
    ui_bucket,
    build_confidence_report,
    expected_calibration_error,
    PlattCalibrator,
    UI_BUCKETS,
)

__all__ = [
    "ConfidenceScorer",
    "fuse_evidence",
    "bucket",
    "ui_bucket",
    "build_confidence_report",
    "expected_calibration_error",
    "PlattCalibrator",
    "UI_BUCKETS",
]

