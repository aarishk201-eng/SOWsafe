"""Tests for the Feature 1/2/3 endpoints: /caption, /benchmark, /registry.

These lock in the dedicated Scene Captioning task, the bounded offline benchmark
dashboard, and the agent registry / parameters surface so they cannot regress.
All computation is real and pixel-grounded; nothing is hardcoded.
"""

import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app  # noqa: E402

client = TestClient(app)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
VEG = os.path.join(FIXTURES_DIR, "optical_vegetation.tif")
WATER = os.path.join(FIXTURES_DIR, "optical_water.tif")


class TestCaptionEndpoint:
    def test_caption_multipart_returns_answer_and_trace(self):
        with open(VEG, "rb") as fh:
            res = client.post("/caption", files={"image": ("optical_vegetation.tif", fh, "image/tiff")})
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data["answer"], str) and len(data["answer"]) > 20
        trace = data["trace"]
        assert trace["endpoint"] == "/caption"
        assert trace["model_selections"], "captioning model must be logged"
        assert trace["evidence_sources"], "caption evidence must be logged"

    def test_caption_is_pixel_grounded_not_hardcoded(self):
        """Different scenes must yield different captions."""
        with open(VEG, "rb") as fh:
            a = client.post("/caption", files={"image": ("veg.tif", fh, "image/tiff")}).json()["answer"]
        with open(WATER, "rb") as fh:
            b = client.post("/caption", files={"image": ("water.tif", fh, "image/tiff")}).json()["answer"]
        assert a != b, "captions for vegetation vs water scenes must differ"
        assert "vegetation" in a.lower()
        assert "water" in b.lower()

    def test_caption_missing_image_is_rejected(self):
        res = client.post("/caption", data={})
        assert res.status_code in (400, 422)


class TestRegistryEndpoint:
    def test_registry_shape(self):
        res = client.get("/registry")
        assert res.status_code == 200
        data = res.json()
        for key in ("router", "pipeline_stages", "models", "tools", "task_types", "parameters"):
            assert key in data, f"registry missing '{key}'"
        model_ids = {m["id"] for m in data["models"]}
        assert {"vqa", "captioning", "grounding", "change", "fusion"}.issubset(model_ids)
        task_ids = {t["id"] for t in data["task_types"]}
        assert {"vqa", "captioning", "grounding", "change", "change_vqa", "cross_modal_analysis"}.issubset(task_ids)
        assert len(data["parameters"]) >= 5
        for p in data["parameters"]:
            assert {"name", "value", "stage", "description"}.issubset(p.keys())


class TestBenchmarkEndpoint:
    def test_benchmark_metrics_present_and_bounded(self):
        res = client.get("/benchmark")
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "quick-offline-fixtures"
        m = data["metrics"]
        for key in ("vqa_accuracy", "captioning_score", "change_vqa_accuracy",
                    "grounding_detection_rate", "composite_score",
                    "latency_p50_ms", "latency_p95_ms", "samples_evaluated"):
            assert key in m, f"benchmark missing metric '{key}'"
        # Percentage metrics must be within [0, 100] when computed.
        for key in ("vqa_accuracy", "captioning_score", "change_vqa_accuracy",
                    "grounding_detection_rate", "composite_score"):
            if m[key] is not None:
                assert 0.0 <= m[key] <= 100.0
        assert m["samples_evaluated"] >= 1

    def test_benchmark_breakdown_is_evidence_backed(self):
        data = client.get("/benchmark").json()
        bd = data["breakdown"]
        assert bd["vqa"] and all("predicted" in r and "expected" in r for r in bd["vqa"])
        assert bd["change_vqa"] and all("correct" in r for r in bd["change_vqa"])
