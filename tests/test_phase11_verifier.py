"""Phase 11 Pass Gate Tests (Root): Specialist Tool Structured Schema & Evidence Bundle Verification.

Verifies:
1. Specialist tools conform to {prediction, confidence, evidence: {bbox?, area?, mask?}}.
2. evidence/verifier.py combines outputs into one EvidenceBundle per query.
3. Contradicting evidence collapses confidence while agreeing evidence boosts it.
"""

import os
import sys
import pytest

_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "satquery", "backend"))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

from evidence.verifier import (
    EvidenceVerifier,
    EvidenceBundle,
    bundle_evidence,
    combine_evidence,
)
from models.fusion import optical_analyzer, sar_analyzer

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "satquery", "backend", "fixtures")
IMAGE_A = os.path.join(FIXTURES_DIR, "image_a.tif")


def test_specialist_tools_return_structured_schema():
    opt = optical_analyzer(IMAGE_A)
    sar = sar_analyzer(IMAGE_A)

    for tool_name, out in [("optical", opt), ("sar", sar)]:
        assert "prediction" in out, f"{tool_name} missing 'prediction'"
        assert "confidence" in out, f"{tool_name} missing 'confidence'"
        assert "evidence" in out, f"{tool_name} missing 'evidence'"
        assert "bbox" in out["evidence"], f"{tool_name} evidence missing 'bbox'"
        assert "area" in out["evidence"], f"{tool_name} evidence missing 'area'"
        assert "mask" in out["evidence"], f"{tool_name} evidence missing 'mask'"


def test_combine_evidence_into_one_bundle_per_query():
    query = "Detect built-up areas with optical and SAR."
    opt = optical_analyzer(IMAGE_A)
    sar = sar_analyzer(IMAGE_A)

    bundle = bundle_evidence(query=query, tool_outputs={"optical": opt, "sar": sar})

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.query == query
    assert "bbox" in bundle.evidence
    assert "area" in bundle.evidence
    assert "mask" in bundle.evidence
    assert bundle["prediction"] is not None
    assert 0.0 <= bundle["confidence"] <= 1.0


def test_disagreeing_sources_collapse_confidence_in_bundle():
    query = "Classify ambiguous region."
    contradicting_tools = {
        "optical": {
            "prediction": "built_up",
            "confidence": 0.90,
            "evidence": {"bbox": None, "area": None, "mask": None},
        },
        "sar": {
            "prediction": "vegetation",
            "confidence": 0.90,
            "evidence": {"bbox": None, "area": None, "mask": None},
        },
    }

    bundle = bundle_evidence(query=query, tool_outputs=contradicting_tools)
    assert bundle.confidence < 0.50
    assert len(bundle.conflicts) > 0
    assert bundle.verified is False
