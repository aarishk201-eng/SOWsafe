"""Root-level export forwarding to satquery.backend.evidence."""

import os
import sys

_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "satquery", "backend"))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

from satquery.backend.evidence import (
    ConfidenceScorer,
    EvidenceVerifier,
    EvidenceBundle,
    evidence_verifier,
    fuse_evidence,
    bundle_evidence,
    combine_evidence,
    verify_and_bundle,
    bucket,
    ui_bucket,
    build_confidence_report,
    expected_calibration_error,
    PlattCalibrator,
    UI_BUCKETS,
)

__all__ = [
    "ConfidenceScorer",
    "EvidenceVerifier",
    "EvidenceBundle",
    "evidence_verifier",
    "fuse_evidence",
    "bundle_evidence",
    "combine_evidence",
    "verify_and_bundle",
    "bucket",
    "ui_bucket",
    "build_confidence_report",
    "expected_calibration_error",
    "PlattCalibrator",
    "UI_BUCKETS",
]


