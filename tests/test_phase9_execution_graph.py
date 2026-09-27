"""Tests for Phase 9 Ordered Execution Graph Planning (Root Mirror).

Validates:
1. Router emits an ordered execution graph (DAG) of subtasks rather than a single tool call.
2. Complex multi-step queries emit the full canonical pipeline:
   [INPUT_VALIDATION → OPTICAL_ANALYSIS → SAR_ANALYSIS → CHANGE_DETECTION → FUSION → GROUNDING → ANSWER].
3. Single-step queries emit focused graphs with pre-flight validation and answer synthesis.
4. Execution graph steps contain concrete tools, dependencies, and metadata.
5. FastAPI /route endpoint exposes execution graphs for frontend/client visualization.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "satquery", "backend"))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

from main import app
from agents.router import route
from agents.planner import ExecutionPlanner, ExecutionGraph

client = TestClient(app)


class TestOrderedExecutionGraphEmission:
    """Verifies that the router emits ordered execution graphs for multi-step queries."""

    def test_full_multimodal_change_grounding_pipeline_graph(self):
        query = (
            "Use optical and SAR to detect change between 2022 and 2025, "
            "fuse the evidence, highlight the new structures, and answer what changed."
        )
        res = route(query)

        graph = res["execution_graph"]
        assert isinstance(graph, list)
        assert res["is_multi_step"] is True

        expected_order = [
            "INPUT_VALIDATION",
            "OPTICAL_ANALYSIS",
            "SAR_ANALYSIS",
            "CHANGE_DETECTION",
            "FUSION",
            "GROUNDING",
            "ANSWER",
        ]
        assert list(graph) == expected_order

        # Arrow string representation verification
        expected_arrow = "INPUT_VALIDATION → OPTICAL_ANALYSIS → SAR_ANALYSIS → CHANGE_DETECTION → FUSION → GROUNDING → ANSWER"
        assert str(graph) == expected_arrow
        assert repr(graph) == f"[{expected_arrow}]"
        assert res["graph_repr"] == expected_arrow

    def test_change_vqa_ordered_execution_graph(self):
        query = "What changed between 2022 and 2025?"
        res = route(query)
        graph = res["execution_graph"]

        assert list(graph) == ["INPUT_VALIDATION", "CHANGE_DETECTION", "CHANGE_VQA", "ANSWER"]
        assert "models.change.detect" in res["tools"]
        assert "models.change.change_vqa" in res["tools"]

    def test_grounding_ordered_execution_graph(self):
        query = "Highlight the water body."
        res = route(query)
        graph = res["execution_graph"]

        assert list(graph) == ["INPUT_VALIDATION", "GROUNDING", "ANSWER"]
        assert "models.grounding.locate" in res["tools"]

    def test_vqa_ordered_execution_graph(self):
        query = "What type of land cover is visible?"
        res = route(query)
        graph = res["execution_graph"]

        assert list(graph) == ["INPUT_VALIDATION", "VQA", "ANSWER"]
        assert "models.vqa.predict" in res["tools"]

    def test_cross_modal_analysis_ordered_execution_graph(self):
        query = "Use optical and SAR together to identify built-up areas."
        res = route(query)
        graph = res["execution_graph"]

        assert list(graph) == [
            "INPUT_VALIDATION",
            "OPTICAL_ANALYSIS",
            "SAR_ANALYSIS",
            "FUSION",
            "ANSWER",
        ]
        assert "models.fusion.optical_analyzer" in res["tools"]
        assert "models.fusion.sar_analyzer" in res["tools"]
        assert "evidence.confidence.fuse_evidence" in res["tools"]

    def test_change_detection_command_ordered_execution_graph(self):
        query = "Detect change between image_a and image_b."
        res = route(query)
        graph = res["execution_graph"]

        assert list(graph) == ["INPUT_VALIDATION", "CHANGE_DETECTION", "ANSWER"]
        assert "models.change.detect" in res["tools"]

    def test_execution_steps_metadata_and_dependencies(self):
        query = "Highlight the cargo port and locate shipping vessels."
        res = route(query)
        steps = res["execution_steps"]

        assert len(steps) >= 3
        # First step is input validation with no prerequisites
        assert steps[0]["name"] == "INPUT_VALIDATION"
        assert steps[0]["depends_on"] == []
        assert "check_compatibility" in steps[0]["tool"]

        # Intermediate step depends on preceding step
        assert steps[1]["name"] == "GROUNDING"
        assert steps[1]["depends_on"] == ["INPUT_VALIDATION"]

        # Final step is answer synthesis
        assert steps[-1]["name"] == "ANSWER"
        assert len(steps[-1]["depends_on"]) > 0


class TestExecutionGraphAPIEndpoint:
    """Verifies that the /route API endpoint returns execution graphs."""

    def test_api_route_returns_execution_graph_and_plan(self):
        resp = client.post("/route", json={"query": "Use optical and SAR to detect change between 2022 and 2025, highlight new buildings, and answer what changed."})
        assert resp.status_code == 200
        data = resp.json()

        assert "execution_graph" in data
        assert "plan" in data
        assert "graph_repr" in data
        assert data["is_multi_step"] is True

        graph = data["execution_graph"]
        assert "INPUT_VALIDATION" in graph
        assert "CHANGE_DETECTION" in graph
        assert "ANSWER" in graph
        assert data["graph_repr"].startswith("INPUT_VALIDATION →")
