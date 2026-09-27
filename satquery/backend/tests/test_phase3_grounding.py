import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import app
from models.grounding import locate, ObjectGroundingModel

client = TestClient(app)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _get_fixture(filename: str) -> str:
    local = os.path.join("fixtures", filename)
    if os.path.exists(local):
        return local
    backend_local = os.path.join(FIXTURES_DIR, filename)
    return backend_local


def test_locate_present_target_returns_valid_bbox_and_confidence():
    """Verify locate returns a valid bounding box and high confidence when target is present."""
    scene = _get_fixture("subset/sample_0.tif")
    result = locate(scene, "cultivated agricultural parcel")

    assert result["bbox"] is not None
    assert isinstance(result["bbox"], list)
    assert len(result["bbox"]) == 4
    x1, y1, x2, y2 = result["bbox"]
    assert 0 <= x1 < x2 <= 512
    assert 0 <= y1 < y2 <= 512
    assert result["confidence"] >= 0.70


def test_locate_absent_target_returns_null_and_low_confidence():
    """Conservative filtering: Return null/low-confidence rather than a guessed box when phrase isn't found."""
    scene = _get_fixture("subset/sample_0.tif")
    result = locate(scene, "commercial airplane on runway")

    # Bounding box must be None (not a guessed box)
    assert result["bbox"] is None
    assert result["confidence"] < 0.25


def test_locate_absent_target_in_port():
    """Verify absent targets in port scene return null bbox."""
    scene = _get_fixture("subset/sample_3.tif")
    result = locate(scene, "commercial airliner")

    assert result["bbox"] is None
    assert result["confidence"] < 0.25


def test_locate_urban_structures():
    """Verify structure localization in urban scene."""
    scene = _get_fixture("subset/sample_1.tif")
    result = locate(scene, "urban structure footprint")

    assert result["bbox"] is not None
    assert len(result["bbox"]) == 4
    assert result["confidence"] >= 0.70


def test_locate_raises_file_not_found():
    """Verify locate raises FileNotFoundError on missing raster path."""
    with pytest.raises(FileNotFoundError):
        locate("non_existent_scene.tif", "agriculture")


def test_locate_empty_phrase():
    """Verify empty phrase returns null box and 0 confidence."""
    scene = _get_fixture("subset/sample_0.tif")
    result = locate(scene, "   ")
    assert result["bbox"] is None
    assert result["confidence"] == 0.0


def test_object_grounding_model_wrapper():
    """Verify ObjectGroundingModel wrapper compatibility."""
    scene = _get_fixture("subset/sample_0.tif")
    model = ObjectGroundingModel()
    res = model.locate(scene, "vegetation canopy")
    assert "detections" in res
    assert len(res["detections"]) == 1
    assert res["detections"][0]["bbox"] is not None


def test_api_locate_json_endpoint():
    """Verify POST /locate endpoint returns {bbox, confidence} via JSON."""
    scene = _get_fixture("subset/sample_0.tif")
    resp = client.post("/locate", json={"image": scene, "phrase": "agricultural field"})
    assert resp.status_code == 200
    data = resp.json()
    assert "bbox" in data
    assert "confidence" in data
    assert data["bbox"] is not None
    assert data["confidence"] >= 0.70


def test_api_locate_absent_target_json():
    """Verify POST /locate returns bbox null for absent phrase."""
    scene = _get_fixture("subset/sample_0.tif")
    resp = client.post("/locate", json={"image": scene, "phrase": "swimming pool"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["bbox"] is None
    assert data["confidence"] < 0.25


def test_api_locate_multipart_upload():
    """Verify POST /locate supports multipart form upload."""
    scene = _get_fixture("subset/sample_0.tif")
    if not os.path.exists(scene):
        scene = _get_fixture("sample_scene.tif")
    with open(scene, "rb") as f:
        resp = client.post(
            "/locate",
            files={"image": ("sample.tif", f, "image/tiff")},
            data={"phrase": "agricultural field"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["bbox"] is not None
    assert data["confidence"] >= 0.70




def test_api_locate_rejects_corrupt_file():
    """Verify POST /locate rejects corrupt or non-georeferenced raster."""
    corrupt = _get_fixture("corrupt.tif")
    with open(corrupt, "rb") as f:
        resp = client.post(
            "/locate",
            files={"image": ("bad.tif", f, "image/tiff")},
            data={"phrase": "water"},
        )
    assert resp.status_code == 400


def test_grounding_source_explicit_tag():
    """Explicitly verify which grounding_source was used for a given locate run."""
    scene = _get_fixture("sample_scene.tif")
    if not os.path.exists(scene):
        scene = "sample_scene.tif"
    if os.path.exists(scene):
        result = locate(scene, "built-up areas")
        assert "grounding_source" in result
        assert result["grounding_source"] in ["owlvit_model", "change_mask_fallback"]
        # Assert that real OwlViT model is executing in this environment
        assert result["grounding_source"] == "owlvit_model", f"Expected owlvit_model but got {result['grounding_source']}"

