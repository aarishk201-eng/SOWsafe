import pytest
from agents.router import query_router
from agents.planner import ExecutionPlanner
from agents.registry import ToolRegistry
from geospatial.validator import GeospatialValidator
from geospatial.metadata import MetadataExtractor
from geospatial.preprocessing import TilePreprocessor
from evidence.confidence import ConfidenceScorer
from evidence.verifier import EvidenceVerifier


def test_agent_query_routing():
    res = query_router.route_query("Locate aircraft in airport")
    assert res["target_pipeline"] == "grounding"

    res_change = query_router.route_query("Find change in deforestation before and after")
    assert res_change["target_pipeline"] == "change"


def test_execution_planner():
    planner = ExecutionPlanner()
    plan = planner.create_plan("Identify port vessels", "vqa")
    assert len(plan) == 4
    assert plan[0]["action"] == "geospatial.validate_raster"


def test_tool_registry():
    registry = ToolRegistry()
    tools = registry.list_tools()
    models = registry.list_models()
    assert len(tools) >= 5
    assert len(models) >= 4


def test_geospatial_bounds_validation():
    validator = GeospatialValidator()
    assert validator.validate_bounds(10.0, 20.0, 15.0, 25.0) is True
    assert validator.validate_bounds(15.0, 20.0, 10.0, 25.0) is False  # Invalid min_x >= max_x


def test_geospatial_preprocessing_tiles():
    preprocessor = TilePreprocessor(tile_size=256, overlap=32)
    grid = preprocessor.calculate_tile_grid(500, 500)
    assert len(grid) > 0
    assert grid[0]["width"] <= 256
    assert grid[0]["height"] <= 256


def test_confidence_scorer():
    scorer = ConfidenceScorer(min_confidence_threshold=0.7)
    detections = [{"confidence": 0.85}, {"confidence": 0.95}]
    result = scorer.calculate_aggregate_score(detections)
    assert result["reliable"] is True
    assert result["mean_confidence"] == 0.90


def test_evidence_verifier():
    verifier = EvidenceVerifier()
    boxes = [[10.0, 10.0, 100.0, 100.0], [-5.0, 0.0, 50.0, 50.0]]
    result = verifier.verify_bounding_boxes(boxes, 500, 500)
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 1
