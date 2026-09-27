"""Tests for Auditable Trace Object across agents and API endpoints."""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app
from agents.trace import AuditTrace, TraceBuilder
from agents.router import query_router, classify_intent

client = TestClient(app)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
IMAGE_A = os.path.join(FIXTURES_DIR, "image_a.tif")
IMAGE_B = os.path.join(FIXTURES_DIR, "image_b_with_synthetic_block.tif")


class TestTraceBuilder:
    """Direct tests for AuditTrace and TraceBuilder."""

    def test_trace_builder_schema_completeness(self):
        tb = TraceBuilder(endpoint="/test", query="Detect ships in the harbor")
        tb.log_validation("GeoTIFF integrity", passed=True, detail="CRS=EPSG:32643")
        tb.log_model("TestModel", version="1.0", reason="Testing purposes")
        tb.log_evidence("test.tool", prediction="positive", confidence=0.92, metadata={"area": 120.5})
        tb.log_execution_steps([{"step": "VALIDATION", "tool": "validator"}])

        trace = tb.build()
        assert isinstance(trace, dict)
        assert "trace_id" in trace
        assert "timestamp" in trace
        assert trace["endpoint"] == "/test"
        assert trace["query"] == "Detect ships in the harbor"
        assert trace["status"] == "success"

        # Check model selections
        assert len(trace["model_selections"]) == 1
        assert trace["model_selections"][0]["name"] == "TestModel"
        assert trace["model_selections"][0]["version"] == "1.0"

        # Check validation checks
        assert len(trace["validation_checks"]) == 1
        assert trace["validation_checks"][0]["check"] == "GeoTIFF integrity"
        assert trace["validation_checks"][0]["passed"] is True

        # Check evidence sources
        assert len(trace["evidence_sources"]) == 1
        assert trace["evidence_sources"][0]["source"] == "test.tool"
        assert trace["evidence_sources"][0]["confidence"] == 0.92

        # Check execution steps
        assert len(trace["execution_steps"]) == 1

        # Check confidence summary
        assert "confidence_summary" in trace
        assert trace["confidence_summary"]["overall"] == 0.92
        assert trace["confidence_summary"]["bucket"] == "high"

    def test_trace_builder_failed_status(self):
        tb = TraceBuilder(endpoint="/fail")
        tb.log_validation("File readable", passed=False, detail="File not found")
        trace = tb.build(status="failed")
        assert trace["status"] == "failed"
        assert trace["validation_checks"][0]["passed"] is False


class TestRouterTraceIntegration:
    """Test router interaction with TraceBuilder."""

    def test_router_populates_trace(self):
        tb = TraceBuilder(endpoint="/route", query="Has urban sprawl expanded into forest between 2020 and 2023?")
        res = query_router.classify_intent("Has urban sprawl expanded into forest between 2020 and 2023?", trace=tb)

        trace = tb.build()
        assert len(trace["model_selections"]) >= 1
        assert trace["model_selections"][0]["name"] == "SatQueryRouter"
        assert len(trace["evidence_sources"]) >= 1
        assert trace["evidence_sources"][0]["source"] == "agents.router.classify_intent"
        assert trace["evidence_sources"][0]["prediction"] == str(res["task"])

    def test_router_empty_query_populates_trace(self):
        tb = TraceBuilder(endpoint="/route", query="")
        res = classify_intent("", trace=tb)

        trace = tb.build()
        assert len(trace["model_selections"]) >= 1
        assert trace["model_selections"][0]["name"] == "SatQueryRouter"
        assert len(trace["evidence_sources"]) >= 1
        assert trace["evidence_sources"][0]["prediction"] == "vqa"


class TestAPIEndpointsReturnTrace:
    """Verify that endpoints return the 'trace' key with full audit details."""

    def test_route_endpoint_trace(self):
        resp = client.post("/route", json={"query": "Find all aircraft on the runway"})
        assert resp.status_code == 200
        data = resp.json()
        assert "trace" in data
        trace = data["trace"]
        assert trace["endpoint"] == "/route"
        assert trace["query"] == "Find all aircraft on the runway"
        assert len(trace["model_selections"]) > 0
        assert len(trace["evidence_sources"]) > 0

    def test_validate_endpoint_trace(self):
        if os.path.exists(IMAGE_A):
            resp = client.post("/validate", json={"path": IMAGE_A})
            assert resp.status_code == 200
            data = resp.json()
            assert "trace" in data
            trace = data["trace"]
            assert trace["endpoint"] == "/validate"
            assert any(chk["check"] == "GeoTIFF integrity" and chk["passed"] for chk in trace["validation_checks"])

    def test_validate_pair_endpoint_trace(self):
        if os.path.exists(IMAGE_A) and os.path.exists(IMAGE_B):
            resp = client.post("/validate-pair", json={"path_a": IMAGE_A, "path_b": IMAGE_B})
            assert resp.status_code == 200
            data = resp.json()
            assert "trace" in data
            trace = data["trace"]
            assert trace["endpoint"] == "/validate-pair"
            checks = [chk["check"] for chk in trace["validation_checks"]]
            assert "GeoTIFF integrity (scene A)" in checks
            assert "CRS compatibility" in checks

    def test_fuse_endpoint_trace(self):
        if os.path.exists(IMAGE_A) and os.path.exists(IMAGE_B):
            resp = client.post("/fuse", json={"optical_path": IMAGE_A, "sar_path": IMAGE_B})
            assert resp.status_code == 200
            data = resp.json()
            assert "trace" in data
            trace = data["trace"]
            assert trace["endpoint"] == "/fuse"
            models = [m["name"] for m in trace["model_selections"]]
            assert "Optical Analyzer" in models
            assert "SAR Analyzer" in models


class TestPassGateExecutionTrace:
    """Pass gate: Verify execution_trace is derived dynamically from actual model execution."""

    def test_execution_trace_present_on_every_response(self):
        resp = client.post("/query", json={"query": "Has the built-up area increased?", "images": [IMAGE_A, IMAGE_B]})
        assert resp.status_code == 200
        trace = resp.json()["execution_trace"]
        assert "task" in trace and "models_used" in trace and "validation" in trace and "evidence" in trace

    def test_trace_models_used_matches_actual_calls_made(self, mocker):
        import models.change_analysis
        spy = mocker.spy(models.change_analysis, "detect")
        resp = client.post("/query", json={"query": "What changed?", "images": [IMAGE_A, IMAGE_B]})
        assert resp.status_code == 200
        assert "change_model" in resp.json()["execution_trace"]["models_used"]
        assert spy.call_count == 1  # trace must reflect reality, not a hardcoded list

    def test_trace_models_not_invoked_on_validation_failure(self, mocker):
        import models.change_analysis
        IMAGE_WRONG_CRS = os.path.join(FIXTURES_DIR, "image_b_wrong_crs.tif")
        spy = mocker.spy(models.change_analysis, "detect")
        resp = client.post("/query", json={"query": "What changed?", "images": [IMAGE_A, IMAGE_WRONG_CRS]})
        assert resp.status_code == 200
        trace = resp.json()["execution_trace"]
        assert "change_model" not in trace["models_used"]
        assert spy.call_count == 0  # pre-flight validation stopped execution, no lying in trace!

