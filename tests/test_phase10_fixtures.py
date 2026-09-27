"""Phase 10 Pass Gate Tests (Root): Task Graph Execution Engine with Pre-Flight Validation Gating.

Requirements:
1. test_task_graph_runs_input_validation_first
2. test_task_graph_execution_halts_on_validation_failure
"""

import os
import sys
import pytest

_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "satquery", "backend"))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

from agents.planner import build_task_graph, execute, TaskGraph, TaskStep, ExecutionResult


def _find_fixture(filename: str) -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "satquery", "backend", "fixtures", filename),
        os.path.join("fixtures", filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return filename


image_a = _find_fixture("image_a.tif")
image_b_wrong_crs = _find_fixture("image_b_wrong_crs.tif")
incompatible_pair = (image_a, image_b_wrong_crs)


def test_task_graph_runs_input_validation_first():
    graph = build_task_graph("Identify newly built-up areas using optical and SAR.")
    assert graph.steps[0].name == "INPUT_VALIDATION"


def test_task_graph_execution_halts_on_validation_failure():
    graph = build_task_graph("...", images=[incompatible_pair])
    result = execute(graph)
    assert result.status == "failed"
    assert result.failed_at == "INPUT_VALIDATION"
    assert "OPTICAL_ANALYSIS" not in result.completed_steps  # must not run downstream after a validation failure
