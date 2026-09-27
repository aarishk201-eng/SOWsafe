"""Phase 5 Tests: Bi-temporal Change Detection with Geospatial Compatibility Pre-flight Gating.

Verifies:
1. detect(image_t1, image_t2) returns 2D binary change mask and change statistics.
2. Identical scenes return zero change.
3. Pre-flight compatibility gate strictly verifies CRS, resolution, and spatial overlap.
4. Mismatched inputs immediately raise IncompatibleScenesError to prevent misregistration noise.
5. POST /change API endpoint functions with JSON and multipart inputs, rejecting incompatible pairs.
"""

import os
import sys
import pytest
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app
from models.change_analysis import (
    detect,
    IncompatibleScenesError,
    ChangeMask,
    ChangeDetectionModel,
)
from geospatial.validator import GeoTiffValidationError

client = TestClient(app)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
T1_BASE = os.path.join(FIXTURES_DIR, "t1_base.tif")
T2_CHANGED = os.path.join(FIXTURES_DIR, "t2_changed.tif")
T2_IDENTICAL = os.path.join(FIXTURES_DIR, "t2_identical.tif")
T2_DIFF_CRS = os.path.join(FIXTURES_DIR, "t2_diff_crs.tif")
T2_DIFF_RES = os.path.join(FIXTURES_DIR, "t2_diff_res.tif")
T2_DISJOINT = os.path.join(FIXTURES_DIR, "t2_disjoint.tif")
CORRUPT = os.path.join(FIXTURES_DIR, "corrupt.tif")


def test_detect_returns_change_mask_for_compatible_scenes():
    """Verifies that detect returns a 2D ChangeMask ndarray with change statistics."""
    mask = detect(T1_BASE, T2_CHANGED)

    assert isinstance(mask, np.ndarray)
    assert isinstance(mask, ChangeMask)
    assert mask.ndim == 2
    assert mask.shape == (256, 256)
    assert mask.change_detected is True
    assert mask.change_percentage > 0.5
    assert mask.changed_pixels > 100
    assert mask.total_pixels == 256 * 256
    # Ensure binary values only (0 or 1)
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 1})
    # Central patch where change was introduced should contain changed pixels
    patch = mask[98:158, 98:158]
    assert np.sum(patch) > 50


def test_detect_zero_change_for_identical_scenes():
    """Verifies that comparing an image against itself or an identical scene produces zero change."""
    mask = detect(T1_BASE, T2_IDENTICAL)

    assert isinstance(mask, ChangeMask)
    assert mask.change_detected is False
    assert mask.change_percentage == 0.0
    assert mask.changed_pixels == 0
    assert np.sum(mask) == 0


def test_detect_rejects_mismatched_crs():
    """Hard Requirement: Mismatched CRS must raise IncompatibleScenesError to avoid misregistration noise."""
    with pytest.raises(IncompatibleScenesError) as exc_info:
        detect(T1_BASE, T2_DIFF_CRS)

    err_msg = str(exc_info.value)
    assert "CRS mismatch" in err_msg
    assert "misregistration noise" in err_msg
    assert exc_info.value.compatibility["crs_compatible"] is False


def test_detect_rejects_mismatched_resolution():
    """Hard Requirement: Mismatched resolution must raise IncompatibleScenesError to avoid misregistration noise."""
    with pytest.raises(IncompatibleScenesError) as exc_info:
        detect(T1_BASE, T2_DIFF_RES)

    err_msg = str(exc_info.value).lower()
    assert "resolution mismatch" in err_msg
    assert "misregistration noise" in err_msg
    assert exc_info.value.compatibility["resolution_compatible"] is False


def test_detect_rejects_disjoint_spatial_overlap():
    """Hard Requirement: Disjoint scenes (no spatial overlap) must raise IncompatibleScenesError."""
    with pytest.raises(IncompatibleScenesError) as exc_info:
        detect(T1_BASE, T2_DISJOINT)

    err_msg = str(exc_info.value).lower()
    assert "spatial overlap" in err_msg
    assert "misregistration noise" in err_msg
    assert exc_info.value.compatibility["spatial_overlap"] is False


def test_detect_rejects_corrupt_or_non_georeferenced_raster():
    """Invalid or unreferenced rasters must raise GeoTiffValidationError."""
    with pytest.raises(GeoTiffValidationError):
        detect(T1_BASE, CORRUPT)


def test_change_detection_model_class_wrapper():
    """Tests the ChangeDetectionModel object wrapper."""
    model = ChangeDetectionModel()
    result = model.detect_change(T1_BASE, T2_CHANGED)

    assert result["model"] == "sat-change-v1"
    assert result["change_detected"] is True
    assert result["change_percentage"] > 0
    assert "compatibility" in result


def test_api_change_endpoint_json_payload():
    """Tests POST /change with JSON payload."""
    resp = client.post("/change", json={"image_t1": T1_BASE, "image_t2": T2_CHANGED})
    assert resp.status_code == 200
    data = resp.json()

    assert data["change_detected"] is True
    assert data["change_percentage"] > 0.5
    assert data["changed_pixels"] > 0
    assert data["shape"] == [256, 256]
    assert data["compatibility"]["ready_for_pairwise_analysis"] is True


def test_api_change_endpoint_multipart_upload():
    """Tests POST /change with multipart file upload."""
    with open(T1_BASE, "rb") as f1, open(T2_CHANGED, "rb") as f2:
        resp = client.post(
            "/change",
            files={
                "image_t1": ("t1.tif", f1, "image/tiff"),
                "image_t2": ("t2.tif", f2, "image/tiff"),
            },
        )
    assert resp.status_code == 200
    assert resp.json()["change_detected"] is True


def test_api_change_endpoint_rejects_incompatible_crs():
    """Tests that POST /change returns HTTP 400 when CRS is mismatched."""
    resp = client.post("/change", json={"image_t1": T1_BASE, "image_t2": T2_DIFF_CRS})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "IncompatibleScenesError" in detail["error"]
    assert "CRS mismatch" in detail["message"]
    assert detail["compatibility"]["crs_compatible"] is False


def test_api_change_endpoint_rejects_disjoint_pair():
    """Tests that POST /change returns HTTP 400 when scenes have no spatial overlap."""
    resp = client.post("/change", json={"image_t1": T1_BASE, "image_t2": T2_DISJOINT})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "IncompatibleScenesError" in detail["error"]
    assert "spatial overlap" in detail["message"]
