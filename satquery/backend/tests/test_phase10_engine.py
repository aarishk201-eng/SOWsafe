"""Comprehensive Tests for Phase 10 Task Graph Execution Engine.

Validates:
1. Pre-flight validation gate halts immediately on CRS mismatch or invalid rasters.
2. Downstream model steps NEVER execute upon validation failure.
3. Compatible image pairs pass validation and execute downstream steps in order.
4. Single-image task graphs validate raster integrity and execute downstream inference.
5. ExecutionResult exposes both attribute access and dict-style subscripting.
6. Exports are accessible across router and planner modules.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.planner import (
    build_task_graph,
    execute,
    TaskGraph,
    TaskStep,
    ExecutionResult,
)
from agents.router import (
    build_task_graph as router_build_graph,
    execute as router_execute,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
IMAGE_A = os.path.join(FIXTURES_DIR, "image_a.tif")
IMAGE_B_BLOCK = os.path.join(FIXTURES_DIR, "image_b_with_synthetic_block.tif")
IMAGE_B_WRONG_CRS = os.path.join(FIXTURES_DIR, "image_b_wrong_crs.tif")


class TestTaskGraphExecutionEngine:
    """Tests execution engine behaviors, validation gating, and result formatting."""

    def test_incompatible_pair_halts_and_omits_downstream_steps(self):
        incompatible_pair = (IMAGE_A, IMAGE_B_WRONG_CRS)
        graph = build_task_graph("Detect change and locate new buildings.", images=[incompatible_pair])

        assert graph.steps[0].name == "INPUT_VALIDATION"
        result = execute(graph)

        assert result.status == "failed"
        assert result.failed_at == "INPUT_VALIDATION"
        # Strict pass gate: downstream model steps MUST NOT run
        assert "OPTICAL_ANALYSIS" not in result.completed_steps
        assert "SAR_ANALYSIS" not in result.completed_steps
        assert "CHANGE_DETECTION" not in result.completed_steps
        assert "FUSION" not in result.completed_steps
        assert "GROUNDING" not in result.completed_steps
        assert len(result.completed_steps) == 0

    def test_nonexistent_file_halts_at_input_validation(self):
        missing_pair = ("fixtures/nonexistent_scene_1.tif", "fixtures/nonexistent_scene_2.tif")
        graph = build_task_graph("...", images=[missing_pair])
        result = execute(graph)

        assert result.status == "failed"
        assert result.failed_at == "INPUT_VALIDATION"
        assert "OPTICAL_ANALYSIS" not in result.completed_steps
        assert len(result.completed_steps) == 0

    def test_compatible_pair_executes_downstream_steps(self):
        compatible_pair = (IMAGE_A, IMAGE_B_BLOCK)
        graph = build_task_graph(
            "Use optical and SAR to detect change between 2022 and 2025, fuse the evidence, and highlight buildings.",
            images=[compatible_pair],
        )

        assert graph.steps[0].name == "INPUT_VALIDATION"
        result = execute(graph)

        assert result.status == "success"
        assert result.failed_at is None
        assert "INPUT_VALIDATION" in result.completed_steps
        assert "OPTICAL_ANALYSIS" in result.completed_steps
        assert "SAR_ANALYSIS" in result.completed_steps
        assert "CHANGE_DETECTION" in result.completed_steps
        assert "FUSION" in result.completed_steps
        assert "ANSWER" in result.completed_steps

    def test_single_raster_vqa_graph_execution(self):
        graph = build_task_graph("What type of land cover is present?", images=[IMAGE_A])
        assert graph.steps[0].name == "INPUT_VALIDATION"

        result = execute(graph)
        assert result.status == "success"
        assert "INPUT_VALIDATION" in result.completed_steps
        assert "VQA" in result.completed_steps
        assert "ANSWER" in result.completed_steps

    def test_execution_result_access_patterns(self):
        graph = build_task_graph("...", images=[(IMAGE_A, IMAGE_B_WRONG_CRS)])
        result = execute(graph)

        # Attribute access
        assert result.status == "failed"
        assert result.failed_at == "INPUT_VALIDATION"
        # Dict subscripting
        assert result["status"] == "failed"
        assert result["failed_at"] == "INPUT_VALIDATION"
        # Dict get
        assert result.get("status") == "failed"
        assert result.get("failed_at") == "INPUT_VALIDATION"

    def test_router_exports_work_identically(self):
        incompatible_pair = (IMAGE_A, IMAGE_B_WRONG_CRS)
        graph = router_build_graph("...", images=[incompatible_pair])
        result = router_execute(graph)

        assert result.status == "failed"
        assert result.failed_at == "INPUT_VALIDATION"
        assert "OPTICAL_ANALYSIS" not in result.completed_steps
