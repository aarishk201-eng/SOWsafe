"""Phase 11 Pass Gate Tests (Root Mirror): Specialist Tool Structured Schema & Evidence Bundle Verification.

Requirements:
1. test_every_specialist_output_matches_schema:
   Verifies that every specialist tool (VQA, Grounding, Change Detection, SAR Analyzer, Optical Analyzer)
   returns outputs conforming to {prediction, confidence, evidence}.
2. test_evidence_bundle_preserves_per_source_attribution:
   Verifies that combine_evidence preserves per-source attribution in bundle["sources"],
   where each source entry has 'source_tool'.
3. Pass Gate:
   Verifies that schema validation passes for every tool's raw output before it enters fusion,
   rejecting silently-malformed tool outputs (raising ValueError) rather than passing them through.
"""

import os
import sys
import pytest

_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "satquery", "backend"))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

from evidence.verifier import (
    EvidenceBundle,
    EvidenceVerifier,
    bundle_evidence,
    combine_evidence,
)
from models.vqa import predict, vqa_analyzer
from models.grounding import locate
from models.change.detector import detect_change
from models.fusion import optical_analyzer, sar_analyzer


def _find_fixture(filename: str) -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "satquery", "backend", "fixtures", filename),
        os.path.join(os.path.dirname(__file__), "..", "fixtures", filename),
        os.path.join("fixtures", filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return filename


IMAGE_A = _find_fixture("image_a.tif")
IMAGE_B = _find_fixture("image_b_with_synthetic_block.tif")
SAR_IMAGE = _find_fixture("sar_built_up.tif")


@pytest.fixture
def specialist_outputs():
    """Generates real specialist outputs across all tools."""
    vqa_out = predict(IMAGE_A, "What type of land cover is visible?")
    grounding_out = locate(IMAGE_A, "vegetation")
    change_out = detect_change(IMAGE_A, IMAGE_B)
    sar_out = sar_analyzer(SAR_IMAGE)
    opt_out = optical_analyzer(IMAGE_A)
    return {
        "vqa_output": vqa_out,
        "grounding_output": grounding_out,
        "change_output": change_out,
        "sar_output": sar_out,
        "optical_output": opt_out,
    }


def test_every_specialist_output_matches_schema(specialist_outputs):
    vqa_output = specialist_outputs["vqa_output"]
    grounding_output = specialist_outputs["grounding_output"]
    change_output = specialist_outputs["change_output"]
    sar_output = specialist_outputs["sar_output"]

    for tool_output in [vqa_output, grounding_output, change_output, sar_output]:
        assert set(["prediction", "confidence", "evidence"]).issubset(tool_output.keys())


def test_evidence_bundle_preserves_per_source_attribution(specialist_outputs):
    vqa_output = specialist_outputs["vqa_output"]
    grounding_output = specialist_outputs["grounding_output"]
    change_output = specialist_outputs["change_output"]

    bundle = combine_evidence([vqa_output, grounding_output, change_output])
    assert len(bundle["sources"]) == 3
    assert all("source_tool" in s for s in bundle["sources"])

    # Verify specific tool sources are tracked
    source_tools = [s["source_tool"] for s in bundle["sources"]]
    assert "vqa" in source_tools
    assert "grounding" in source_tools
    assert "change_detection" in source_tools


class TestSchemaValidationPassGate:
    """Pass gate: schema validation passes for every tool's raw output before it enters fusion

    — reject silently-malformed tool outputs rather than passing them through.
    """

    def test_missing_prediction_rejected_before_fusion(self, specialist_outputs):
        vqa_out = specialist_outputs["vqa_output"]
        malformed = {
            "confidence": 0.85,
            "evidence": {"bbox": None, "area": None, "mask": None},
        }
        with pytest.raises(ValueError, match="missing required schema keys.*prediction"):
            combine_evidence([vqa_out, malformed])

    def test_missing_confidence_rejected_before_fusion(self, specialist_outputs):
        vqa_out = specialist_outputs["vqa_output"]
        malformed = {
            "prediction": "built_up",
            "evidence": {"bbox": None, "area": None, "mask": None},
        }
        with pytest.raises(ValueError, match="missing required schema keys.*confidence"):
            combine_evidence([vqa_out, malformed])

    def test_missing_evidence_rejected_before_fusion(self, specialist_outputs):
        vqa_out = specialist_outputs["vqa_output"]
        malformed = {
            "prediction": "built_up",
            "confidence": 0.85,
        }
        with pytest.raises(ValueError, match="missing required schema keys.*evidence"):
            combine_evidence([vqa_out, malformed])

    def test_out_of_bounds_confidence_rejected_before_fusion(self, specialist_outputs):
        vqa_out = specialist_outputs["vqa_output"]
        malformed_high = {
            "prediction": "built_up",
            "confidence": 1.45,
            "evidence": {"bbox": None, "area": None, "mask": None},
        }
        with pytest.raises(ValueError, match="out of bounds"):
            combine_evidence([vqa_out, malformed_high])

        malformed_neg = {
            "prediction": "built_up",
            "confidence": -0.20,
            "evidence": {"bbox": None, "area": None, "mask": None},
        }
        with pytest.raises(ValueError, match="out of bounds"):
            combine_evidence([vqa_out, malformed_neg])

    def test_non_numeric_confidence_rejected_before_fusion(self, specialist_outputs):
        vqa_out = specialist_outputs["vqa_output"]
        malformed = {
            "prediction": "built_up",
            "confidence": "high_confidence",
            "evidence": {"bbox": None, "area": None, "mask": None},
        }
        with pytest.raises(ValueError, match="not a valid number"):
            combine_evidence([vqa_out, malformed])

    def test_evidence_not_a_mapping_rejected_before_fusion(self, specialist_outputs):
        vqa_out = specialist_outputs["vqa_output"]
        malformed = {
            "prediction": "built_up",
            "confidence": 0.85,
            "evidence": "not_a_dict",
        }
        with pytest.raises(ValueError, match="must be a dictionary"):
            combine_evidence([vqa_out, malformed])

    def test_none_tool_output_rejected_before_fusion(self, specialist_outputs):
        vqa_out = specialist_outputs["vqa_output"]
        with pytest.raises(ValueError, match="output.*is None"):
            combine_evidence([vqa_out, None])
