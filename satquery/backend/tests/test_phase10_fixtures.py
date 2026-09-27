"""Phase 10 Pass Gate Tests: Task Graph Execution Engine with Pre-Flight Validation Gating.

Requirements:
1. test_task_graph_runs_input_validation_first:
   Verifies that build_task_graph places INPUT_VALIDATION as the very first step.
2. test_task_graph_execution_halts_on_validation_failure:
   Verifies that a deliberately incompatible input pair halts execution at INPUT_VALIDATION,
   returning status="failed", failed_at="INPUT_VALIDATION", and downstream model steps
   (such as OPTICAL_ANALYSIS) NEVER run and are not present in result.completed_steps.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.planner import build_task_graph, execute, TaskGraph, TaskStep, ExecutionResult


def _find_fixture(filename: str) -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "fixtures", filename),
        os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", filename),
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


def test_task_graph_structure_and_dependencies():
    """Validates step ordering and dependencies in the built task graph."""
    graph = build_task_graph("Identify newly built-up areas using optical and SAR.")
    assert isinstance(graph, TaskGraph)
    assert len(graph.steps) >= 3
    assert graph.steps[0].name == "INPUT_VALIDATION"
    assert graph.steps[0].action == "geospatial.validate_inputs"
    assert "check_compatibility" in graph.steps[0].tool


def test_task_graph_step_string_compatibility():
    """TaskStep should evaluate as both a str and an object with .name attribute."""
    graph = build_task_graph("Identify newly built-up areas using optical and SAR.")
    step0 = graph.steps[0]
    assert step0 == "INPUT_VALIDATION"
    assert step0.name == "INPUT_VALIDATION"
    assert graph[0] == "INPUT_VALIDATION"
    assert graph[0].name == "INPUT_VALIDATION"
