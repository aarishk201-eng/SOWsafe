import os
import tempfile
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from main import app
from models.vqa import predict, build_rs_vqa_prompt, RS_VQA_SYSTEM_PROMPT

client = TestClient(app)


@pytest.fixture
def sample_raster(tmp_path):
    """Creates a sample 4-band synthetic satellite GeoTIFF."""
    p = str(tmp_path / "scene.tif")
    transform = from_origin(500000.0, 2200000.0, 10.0, 10.0)
    data = (np.random.rand(4, 128, 128) * 3000).astype("uint16")
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        width=128,
        height=128,
        count=4,
        dtype="uint16",
        crs="EPSG:32643",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data)
    return p


def test_prompt_engineering_guidance():
    """Verify RS-specific prompt emphasizes nadir overhead perspectives."""
    prompt = build_rs_vqa_prompt("Count the ships in the harbor", {"resolution": [10.0, 10.0]})
    assert "nadir" in prompt.lower() or "overhead" in prompt.lower()
    assert "shadows" in prompt.lower()
    assert "Question: Count the ships in the harbor" in prompt


def test_predict_returns_str(sample_raster):
    """Verify predict(image_path, question) returns a string answer."""
    answer = predict(sample_raster, "What is the predominant land cover in this scene?")
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_predict_counting_query(sample_raster):
    """Verify predict handles overhead counting queries."""
    answer = predict(sample_raster, "How many aircraft are visible?")
    assert isinstance(answer, str)
    assert any(char.isdigit() for char in answer)


def test_predict_presence_query(sample_raster):
    """Verify predict handles detection/presence queries."""
    answer = predict(sample_raster, "Is there an open water body present?")
    assert isinstance(answer, str)
    assert "water" in answer.lower() or "yes" in answer.lower() or "present" in answer.lower()


def test_predict_raises_file_not_found():
    """Verify predict raises FileNotFoundError on invalid paths."""
    with pytest.raises(FileNotFoundError):
        predict("non_existent_scene.tif", "What is visible?")


def test_predict_raises_value_error_on_empty_question(sample_raster):
    """Verify predict raises ValueError on empty questions."""
    with pytest.raises(ValueError):
        predict(sample_raster, "   ")


def test_api_vqa_json_payload(sample_raster):
    """Verify POST /vqa with JSON payload returns {answer: str}."""
    response = client.post("/vqa", json={"image": sample_raster, "question": "Describe the terrain"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_api_vqa_multipart_upload(sample_raster):
    """Verify POST /vqa with uploaded file returns {answer: str}."""
    with open(sample_raster, "rb") as f:
        response = client.post(
            "/vqa",
            files={"image": ("scene.tif", f, "image/tiff")},
            data={"question": "Count the structures in this area"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)


def test_api_vqa_missing_params(sample_raster):
    """Verify POST /vqa validates required parameters."""
    # Missing question
    res1 = client.post("/vqa", json={"image": sample_raster})
    assert res1.status_code == 400

    # Missing image
    res2 = client.post("/vqa", json={"question": "What is here?"})
    assert res2.status_code == 400

    # Non-existent image file
    res3 = client.post("/vqa", json={"image": "invalid_missing.tif", "question": "What is here?"})
    assert res3.status_code == 400
