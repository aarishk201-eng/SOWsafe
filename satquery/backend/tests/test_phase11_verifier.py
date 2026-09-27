"""Phase 11 Tests: Specialist Tool Structured Schema & Evidence Bundle Verification.

Requirements:
1. Every specialist tool must return the structured schema:
   {prediction, confidence, evidence: {bbox?, area?, mask?}}
2. evidence/verifier.py combines these into one evidence bundle per query.
3. Cross-modal conflict detection collapses confidence on disagreement and boosts confidence on agreement.
4. Spatial evidence (bbox, area, mask) is verified and consolidated.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evidence.verifier import (
    EvidenceVerifier,
    EvidenceBundle,
    bundle_evidence,
    combine_evidence,
    verify_and_bundle,
    evidence_verifier,
)
from evidence.confidence import fuse_evidence
from models.cross_modal import optical_analyzer, sar_analyzer
from models.grounding import locate, ObjectGroundingModel
from models.change_analysis import detect_change, ChangeMask, ChangeVQAEngine
from models.vqa import SatelliteVQAModel, vqa_analyzer

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
IMAGE_A = os.path.join(FIXTURES_DIR, "image_a.tif")
IMAGE_B_BLOCK = os.path.join(FIXTURES_DIR, "image_b_with_synthetic_block.tif")
SAMPLE_LANDCOVER = os.path.join(FIXTURES_DIR, "subset/sample_0.tif")
SAMPLE_URBAN = os.path.join(FIXTURES_DIR, "subset/sample_1.tif")


class TestSpecialistToolStructuredSchema:
    """Verifies that every specialist tool returns the structured schema:

    {prediction, confidence, evidence: {bbox?, area?, mask?}}
    """

    def test_optical_analyzer_returns_structured_schema(self):
        result = optical_analyzer(IMAGE_A)
        assert "prediction" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert "evidence" in result
        assert isinstance(result["evidence"], dict)
        assert "bbox" in result["evidence"]
        assert "area" in result["evidence"]
        assert "mask" in result["evidence"]

    def test_sar_analyzer_returns_structured_schema(self):
        result = sar_analyzer(IMAGE_A)
        assert "prediction" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert "evidence" in result
        assert isinstance(result["evidence"], dict)
        assert "bbox" in result["evidence"]
        assert "area" in result["evidence"]
        assert "mask" in result["evidence"]

    def test_grounding_locate_returns_structured_schema(self):
        result = locate(SAMPLE_URBAN, "urban structure footprint")
        assert "prediction" in result
        assert "confidence" in result
        assert "evidence" in result
        assert isinstance(result["evidence"], dict)
        assert "bbox" in result["evidence"]
        assert "area" in result["evidence"]
        assert "mask" in result["evidence"]

        # Backwards compatibility check: result["bbox"] is also accessible directly
        assert "bbox" in result
        if result["evidence"]["bbox"]:
            assert result["bbox"] == result["evidence"]["bbox"]
            assert result["evidence"]["area"] is not None
            assert result["evidence"]["area"] > 0

    def test_change_detector_returns_structured_schema(self):
        mask = detect_change(IMAGE_A, IMAGE_B_BLOCK)
        # ChangeMask behaves as a 2D numpy array
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (256, 256)
        # AND exposes specialist tool schema
        assert hasattr(mask, "prediction")
        assert hasattr(mask, "confidence")
        assert hasattr(mask, "evidence")
        assert mask["prediction"] in ("change", "no_change")
        assert 0.0 <= mask["confidence"] <= 1.0
        assert isinstance(mask["evidence"], dict)
        assert "bbox" in mask["evidence"]
        assert "area" in mask["evidence"]
        assert "mask" in mask["evidence"]
        assert mask["evidence"]["area"] > 0
        assert mask["evidence"]["bbox"] is not None

    def test_change_vqa_engine_returns_structured_schema(self):
        engine = ChangeVQAEngine()
        result = engine.answer_query(IMAGE_A, IMAGE_B_BLOCK, "Did building construction increase?")
        assert "prediction" in result
        assert "confidence" in result
        assert "evidence" in result
        assert "bbox" in result["evidence"]
        assert "area" in result["evidence"]
        assert "mask" in result["evidence"]
        assert result["evidence"]["area"] > 0

    def test_vqa_model_returns_structured_schema(self):
        model = SatelliteVQAModel()
        result = model.predict(IMAGE_A, "What land cover type is visible?")
        assert "prediction" in result
        assert "confidence" in result
        assert "evidence" in result
        assert "bbox" in result["evidence"]
        assert "area" in result["evidence"]
        assert "mask" in result["evidence"]

    def test_fuse_evidence_returns_structured_schema(self):
        opt = {"prediction": "built_up", "confidence": 0.85, "evidence": {"bbox": None, "area": None, "mask": None}}
        sar = {"prediction": "built_up", "confidence": 0.88, "evidence": {"bbox": None, "area": None, "mask": None}}
        fused = fuse_evidence(opt, sar)

        assert "prediction" in fused
        assert "confidence" in fused
        assert "evidence" in fused
        assert "bbox" in fused["evidence"]
        assert "area" in fused["evidence"]
        assert "mask" in fused["evidence"]
        assert fused["evidence"]["agreement"] == "high"


class TestEvidenceVerifierAndBundle:
    """Verifies EvidenceVerifier and EvidenceBundle aggregation per query."""

    def test_tool_output_schema_validation(self):
        verifier = EvidenceVerifier()

        # Valid tool output
        valid_tool = {
            "prediction": "built_up",
            "confidence": 0.88,
            "evidence": {"bbox": [50, 50, 200, 200], "area": 22500.0, "mask": None},
        }
        res = verifier.verify_tool_output(valid_tool, image_width=512, image_height=512)
        assert res["valid"] is True
        assert len(res["errors"]) == 0

        # Invalid tool output (missing confidence, out of bounds bbox)
        invalid_tool = {
            "prediction": "water",
            "evidence": {"bbox": [-10, 0, 600, 200]},
        }
        res_bad = verifier.verify_tool_output(invalid_tool, image_width=512, image_height=512)
        assert res_bad["valid"] is False
        assert any("confidence" in e.lower() for e in res_bad["errors"])
        assert any("bbox" in e.lower() for e in res_bad["errors"])

    def test_combine_evidence_into_unified_bundle(self):
        query = "Identify newly built-up areas using optical and SAR."

        opt_out = optical_analyzer(IMAGE_A)
        sar_out = sar_analyzer(IMAGE_A)
        ground_out = locate(SAMPLE_URBAN, "urban structure footprint")

        bundle = bundle_evidence(
            query=query,
            tool_outputs={"optical": opt_out, "sar": sar_out, "grounding": ground_out},
            image_width=512,
            image_height=512,
        )

        assert isinstance(bundle, EvidenceBundle)
        assert bundle.query == query
        assert bundle.prediction is not None
        assert 0.0 <= bundle.confidence <= 1.0

        # Bundle conforms to the structured schema: {prediction, confidence, evidence: {bbox, area, mask}}
        assert "bbox" in bundle.evidence
        assert "area" in bundle.evidence
        assert "mask" in bundle.evidence

        # Dictionary subscripting
        assert bundle["prediction"] == bundle.prediction
        assert bundle["confidence"] == bundle.confidence
        assert bundle["evidence"] == bundle.evidence

        # Summary provides factual explanation
        assert "Consolidated evidence" in bundle.summary
        assert "built_up" in bundle.summary or bundle.prediction in bundle.summary

    def test_conflict_detection_collapses_confidence_in_bundle(self):
        """When specialist tools contradict each other, confidence collapses and conflict is flagged."""
        query = "Verify whether terrain is built-up or forest."
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

        assert bundle.confidence < 0.50  # Must collapse under epistemic conflict
        assert len(bundle.conflicts) > 0
        assert bundle.verified is False
        assert "contradiction" in bundle.conflicts[0]["type"] or "conflict" in bundle.conflicts[0]["type"]

    def test_multi_sensor_agreement_boosts_confidence(self):
        """When specialist tools corroborate each other, confidence is boosted."""
        query = "Confirm urban expansion."
        agreeing_tools = {
            "optical": {
                "prediction": "built_up",
                "confidence": 0.82,
                "evidence": {"bbox": [100, 100, 200, 200], "area": 10000.0, "mask": None},
            },
            "sar": {
                "prediction": "built_up",
                "confidence": 0.89,
                "evidence": {"bbox": [105, 95, 205, 195], "area": 10000.0, "mask": None},
            },
        }

        bundle = bundle_evidence(query=query, tool_outputs=agreeing_tools)

        assert bundle.confidence > 0.85
        assert bundle.verified is True
        assert len(bundle.conflicts) == 0
        assert bundle.evidence["bbox"] is not None
        assert bundle.evidence["area"] > 0

    def test_task_graph_engine_produces_evidence_bundle(self):
        """Verifies that planner.execute produces an EvidenceBundle at the ANSWER stage."""
        from agents.planner import build_task_graph, execute

        graph = build_task_graph(
            "Use optical and SAR to identify built-up areas.",
            images=[(IMAGE_A, IMAGE_B_BLOCK)],
        )
        result = execute(graph)

        assert result.status == "success"
        assert "evidence_bundle" in result.outputs
        bundle = result.outputs["evidence_bundle"]
        assert isinstance(bundle, EvidenceBundle)
        assert bundle.query == "Use optical and SAR to identify built-up areas."
        assert "bbox" in bundle.evidence
        assert "area" in bundle.evidence
        assert "mask" in bundle.evidence
