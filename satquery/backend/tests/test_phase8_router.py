"""Tests for Phase 8 SatQuery Intent Classifier and Query Router (agents/router.py).

Validates:
1. Intent classifier maps natural language queries to {task, tools} from the fixed set:
   {vqa, grounding, change, change_vqa, cross_modal_analysis}.
2. Standalone tools from Phases 2-7 are accurately bound to their respective tasks.
3. Disambiguation between Change Detection commands and Change VQA questions.
4. Disambiguation between Single-Scene VQA, Grounding, and Multi-Sensor Fusion.
5. FastAPI POST /route endpoint integration.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app
from agents.router import (
    route,
    classify_intent,
    SatQueryRouter,
    TaskType,
    KNOWN_TASKS,
    TASK_TOOLS,
)

client = TestClient(app)


class TestRouterTaskMapping:
    """Verifies that queries map to {task, tools} strictly from the fixed set."""

    def test_fixed_task_set_contains_required_five(self):
        expected_set = {"vqa", "grounding", "change", "change_vqa", "cross_modal_analysis"}
        assert expected_set.issubset(KNOWN_TASKS)
        for task in expected_set:
            assert task in TASK_TOOLS

    def test_route_returns_valid_schema(self):
        res = route("What is visible in this satellite scene?")
        assert "task" in res
        assert "tools" in res
        assert "confidence" in res
        assert "reasoning" in res
        assert res["task"] in KNOWN_TASKS
        assert isinstance(res["tools"], list)
        assert len(res["tools"]) > 0

    def test_route_vqa_queries(self):
        vqa_queries = [
            "What is the land cover type in this scene?",
            "How many airplanes are parked on the runway?",
            "Describe the terrain in this satellite scene",
            "Is this scene predominantly urban or agricultural?",
            "Count the number of circular irrigation fields",
            "Identify the water reservoir in this region",
        ]
        for q in vqa_queries:
            res = route(q)
            assert res["task"] == "vqa", f"Query '{q}' mapped to {res['task']} instead of vqa"
            assert "models.vqa.predict" in res["tools"]

    def test_route_grounding_queries(self):
        grounding_queries = [
            "Locate the water body in the image",
            "Find the runway bounding box",
            "Where is the cargo port located?",
            "Highlight storage tanks in the industrial area",
            "Pinpoint the vessel in the harbor",
            "Give me the bounding box coordinates of the solar farm",
        ]
        for q in grounding_queries:
            res = route(q)
            assert res["task"] == "grounding", f"Query '{q}' mapped to {res['task']} instead of grounding"
            assert "models.grounding.locate" in res["tools"]

    def test_route_change_detection_commands(self):
        change_commands = [
            "Detect change between image_a and image_b",
            "Compute change mask for t1 and t2",
            "Generate bi-temporal difference map",
            "Run change detection on the scene pair",
            "Find changed pixels between before and after images",
            "Calculate altered areas between t1 and t2",
        ]
        for q in change_commands:
            res = route(q)
            assert res["task"] == "change", f"Query '{q}' mapped to {res['task']} instead of change"
            assert "models.change_analysis.detect" in res["tools"]

    def test_route_change_vqa_questions(self):
        change_vqa_queries = [
            "Has the built-up area increased between t1 and t2?",
            "Did deforestation occur over the past 3 years?",
            "What changed between image A and image B?",
            "Was there any new construction in this area?",
            "How much did the urban footprint expand?",
            "Have flood waters receded since yesterday?",
        ]
        for q in change_vqa_queries:
            res = route(q)
            assert res["task"] == "change_vqa", f"Query '{q}' mapped to {res['task']} instead of change_vqa"
            assert "models.change_analysis.change_vqa" in res["tools"]
            assert "models.change_analysis.detect" in res["tools"]

    def test_route_optical_sar_fusion_queries(self):
        fusion_queries = [
            "Fuse optical and SAR imagery",
            "Combine Sentinel-1 radar and Sentinel-2 optical data",
            "Use radar backscatter to verify optical cloud occlusion",
            "Multimodal all-weather terrain classification",
            "Sensor fusion of optical reflectance and SAR polarization",
            "Cross-sensor evidence combination for flood monitoring",
        ]
        for q in fusion_queries:
            res = route(q)
            assert res["task"] == "cross_modal_analysis", f"Query '{q}' mapped to {res['task']} instead of cross_modal_analysis"
            assert "models.cross_modal.optical_analyzer" in res["tools"]
            assert "models.cross_modal.sar_analyzer" in res["tools"]
            assert "evidence.confidence.fuse_evidence" in res["tools"]


class TestRouterEdgeCases:
    """Verifies robustness and boundary handling."""

    def test_empty_or_whitespace_query_defaults_to_vqa(self):
        res1 = route("")
        assert res1["task"] == "vqa"
        res2 = route("   ")
        assert res2["task"] == "vqa"

    def test_disambiguation_between_change_command_and_change_question(self):
        cmd = route("Detect change between image_a and image_b")
        question = route("Has the built-up area increased between image_a and image_b?")
        assert cmd["task"] == "change"
        assert question["task"] == "change_vqa"

    def test_classify_intent_function_and_router_instance_parity(self):
        router = SatQueryRouter()
        q = "Locate all aircraft"
        res_fn = classify_intent(q)
        res_method = router.classify_intent(q)
        assert res_fn["task"] == res_method["task"] == "grounding"
        assert res_fn["tools"] == res_method["tools"]


class TestRouterAPIEndpoint:
    """Verifies the FastAPI POST /route endpoint."""

    def test_api_route_post_json(self):
        resp = client.post("/route", json={"query": "Fuse optical and SAR imagery"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "cross_modal_analysis"
        assert "evidence.confidence.fuse_evidence" in data["tools"]
        assert data["confidence"] > 0.8

    def test_api_route_post_change_vqa(self):
        resp = client.post("/route", json={"query": "Has the built-up area increased between t1 and t2?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "change_vqa"
        assert "models.change_analysis.change_vqa" in data["tools"]

    def test_api_route_missing_query_returns_400(self):
        resp = client.post("/route", json={})
        assert resp.status_code == 400
