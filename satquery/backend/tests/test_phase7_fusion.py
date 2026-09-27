"""Comprehensive tests for Phase 7 Multi-Modal Evidence Fusion (Optical + SAR).

Validates:
1. Asymmetric SAR preprocessing (Lee filtering, dB calibration, dual-polarization decomposition).
2. Independent optical and SAR analyzers.
3. Weighted consensus evidence combination and conflict mass calculation.
4. Atmospheric cloud occlusion dynamic weight suppression.
5. SensorFusionModel end-to-end processing.
6. FastAPI /fuse and /fusion API endpoints.
"""

import os
import sys
import pytest
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app
from evidence.confidence import fuse_evidence
from models.cross_modal import optical_analyzer, sar_analyzer, SensorFusionModel
from geospatial.sar_preprocessing import SARPreprocessor

client = TestClient(app)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _fixture(name: str) -> str:
    return os.path.abspath(os.path.join(FIXTURES_DIR, name))


class TestSARPreprocessing:
    """Verifies dedicated SAR preprocessing physics and speckle reduction."""

    def test_sar_preprocessor_calibration_to_decibels(self):
        pre = SARPreprocessor()
        # Linear power 0.01 -> -20 dB
        arr = np.array([[0.01, 0.1], [0.001, 1.0]], dtype=np.float32)
        db = pre.calibrate_to_decibels(arr)
        assert -20.5 < db[0, 0] < -19.5
        assert -10.5 < db[0, 1] < -9.5
        assert -30.0 <= db[1, 0] <= -29.0
        assert -0.5 < db[1, 1] <= 0.0

    def test_lee_filter_reduces_speckle_variance(self):
        pre = SARPreprocessor()
        np.random.seed(42)
        # Create homogenous surface with multiplicative Rayleigh speckle noise
        clean_surface = np.full((32, 32), 0.20, dtype=np.float32)
        multiplicative_speckle = np.random.exponential(1.0, (32, 32)).astype(np.float32)
        noisy = clean_surface * multiplicative_speckle

        filtered = pre.lee_speckle_filter(noisy, window_size=5, num_looks=1)

        assert np.var(filtered) < np.var(noisy)
        assert np.isclose(np.mean(filtered), np.mean(clean_surface), atol=0.05)

    def test_polarimetric_features_decomposition(self):
        pre = SARPreprocessor()
        vv = np.full((16, 16), 0.25, dtype=np.float32)
        vh = np.full((16, 16), 0.035, dtype=np.float32)
        feats = pre.compute_polarimetric_features(vv, vh)

        assert "mean_vv_db" in feats
        assert "mean_vh_db" in feats
        assert "mean_vh_vv_ratio_db" in feats
        assert "mean_rvi" in feats
        assert feats["mean_vv_db"] > feats["mean_vh_db"]  # Co-pol typically exceeds cross-pol


class TestAnalyzersOnFixtures:
    """Verifies that optical_analyzer and sar_analyzer extract accurate features from GeoTIFFs."""

    def test_optical_analyzer_built_up(self):
        path = _fixture("optical_built_up.tif")
        res = optical_analyzer(path)
        assert res["prediction"] == "built_up"
        assert res["confidence"] > 0.80
        assert res["evidence"]["sensor"] == "optical"
        assert res["evidence"]["cloud_occluded"] is False

    def test_sar_analyzer_built_up(self):
        path = _fixture("sar_built_up.tif")
        res = sar_analyzer(path)
        assert res["prediction"] == "built_up"
        assert res["confidence"] > 0.80
        assert res["evidence"]["sensor"] == "sar"
        assert res["evidence"]["speckle_filtered"] is True

    def test_optical_analyzer_cloud_detection(self):
        path = _fixture("optical_cloudy.tif")
        res = optical_analyzer(path)
        assert res["prediction"] == "cloud_occluded"
        assert res["confidence"] < 0.35
        assert res["evidence"]["cloud_occluded"] is True
        assert res["evidence"]["cloud_score"] > 0.50

    def test_optical_analyzer_vegetation(self):
        path = _fixture("optical_vegetation.tif")
        res = optical_analyzer(path)
        assert res["prediction"] == "vegetation"
        assert res["confidence"] > 0.85
        assert res["evidence"]["ndvi"] > 0.50

    def test_sar_analyzer_vegetation(self):
        path = _fixture("sar_vegetation.tif")
        res = sar_analyzer(path)
        assert res["prediction"] == "vegetation"
        assert res["confidence"] > 0.80

    def test_optical_water_and_sar_water(self):
        opt_path = _fixture("optical_water.tif")
        sar_path = _fixture("sar_water.tif")
        opt_res = optical_analyzer(opt_path)
        sar_res = sar_analyzer(sar_path)

        assert opt_res["prediction"] == "water"
        assert opt_res["evidence"]["ndwi"] > 0.50
        assert sar_res["prediction"] == "water"
        assert sar_res["evidence"]["mean_vv_db"] < -20.0


class TestEvidenceFusionEngine:
    """Verifies evidence combination math and dynamic reweighting."""

    def test_cloud_occlusion_dynamic_reweighting(self):
        """When optical is occluded by clouds, SAR takes over with 90% weight."""
        opt_cloudy = optical_analyzer(_fixture("optical_cloudy.tif"))
        sar_bu = sar_analyzer(_fixture("sar_built_up.tif"))

        fused = fuse_evidence(opt_cloudy, sar_bu)
        assert fused["prediction"] == "built_up"
        assert fused["confidence"] > 0.85
        assert fused["weights"]["sar"] == 0.90
        assert fused["weights"]["optical"] == 0.10

    def test_sensor_fusion_model_integration(self):
        model = SensorFusionModel()
        res = model.fuse(_fixture("optical_built_up.tif"), _fixture("sar_built_up.tif"))

        assert res["status"] == "fused"
        assert res["prediction"] == "built_up"
        assert res["confidence"] > 0.80
        assert res["agreement"] == "high"
        assert res["sensors_agree"] is True


class TestFusionAPIEndpoints:
    """Verifies HTTP POST /fuse and /fusion endpoints."""

    def test_api_fuse_with_json_evidence_agree(self):
        payload = {
            "optical": {"prediction": "built_up", "confidence": 0.82},
            "sar": {"prediction": "built_up", "confidence": 0.89},
        }
        resp = client.post("/fuse", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] > 0.80
        assert data["agreement"] == "high"
        assert data["prediction"] == "built_up"

    def test_api_fuse_with_json_evidence_disagree(self):
        payload = {
            "optical": {"prediction": "built_up", "confidence": 0.90},
            "sar": {"prediction": "vegetation", "confidence": 0.90},
        }
        resp = client.post("/fusion", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] < 0.50
        assert data["agreement"] == "low"
        assert data["conflict"] > 0.70

    def test_api_fuse_with_file_paths(self):
        payload = {
            "optical_path": _fixture("optical_built_up.tif"),
            "sar_path": _fixture("sar_built_up.tif"),
        }
        resp = client.post("/fuse", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] == "built_up"
        assert data["confidence"] > 0.80

    def test_api_fuse_with_multipart_files(self):
        with open(_fixture("optical_water.tif"), "rb") as f_opt, open(_fixture("sar_water.tif"), "rb") as f_sar:
            resp = client.post(
                "/fuse",
                files={
                    "optical": ("optical_water.tif", f_opt, "image/tiff"),
                    "sar": ("sar_water.tif", f_sar, "image/tiff"),
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] == "water"
        assert data["confidence"] > 0.90
        assert data["agreement"] == "high"
