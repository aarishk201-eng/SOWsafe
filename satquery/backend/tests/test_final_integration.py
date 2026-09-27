import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import app

client = TestClient(app)

def test_full_pipeline_single_image_vqa():
    resp = client.post("/query", files={"image": open("fixtures/subset/sample_0.tif","rb")},
                        data={"query": "What type of land cover is visible?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert "execution_trace" in body
    assert body["execution_trace"]["task"] == "vqa"

def test_full_pipeline_change_vqa_two_images():
    resp = client.post("/query", files={"t1": open("fixtures/t1_base.tif","rb"),
                                         "t2": open("fixtures/t2_changed.tif","rb")},
                        data={"query": "Has the built-up area increased?"})
    assert resp.status_code == 200
    assert resp.json()["execution_trace"]["task"] == "change_vqa"

def test_full_pipeline_optical_sar_fusion():
    resp = client.post("/query", files={"optical": open("fixtures/optical_built_up.tif","rb"),
                                         "sar": open("fixtures/subset/sample_5.tif","rb")},
                        data={"query": "Use optical and SAR together to identify built-up areas."})
    assert resp.status_code == 200
    trace = resp.json()["execution_trace"]
    assert "optical_model" in trace["models_used"]
    assert "sar_model" in trace["models_used"]

def test_full_pipeline_incompatible_pair_fails_at_validation_not_downstream(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    
    def _write_tif(path, crs):
        transform = from_origin(300000.0, 4000000.0, 10.0, 10.0)
        data = np.zeros((1, 10, 10), dtype=np.uint8)
        with rasterio.open(path, 'w', driver='GTiff', height=10, width=10, count=1, dtype=data.dtype, crs=crs, transform=transform) as dst:
            dst.write(data)
        return path

    img1 = _write_tif(str(tmp_path / "img1.tif"), "EPSG:4326")
    img2 = _write_tif(str(tmp_path / "img2.tif"), "EPSG:3857")
    
    resp = client.post("/query", files={"t1": open(img1,"rb"),
                                         "t2": open(img2,"rb")},
                        data={"query": "What changed?"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"
    assert "compat" in str(resp.json()).lower() or "validation" in str(resp.json()).lower() or "crs" in str(resp.json()).lower()

def test_full_pipeline_report_download_matches_trace():
    resp = client.post("/query", files={"image": open("fixtures/subset/sample_0.tif","rb")},
                        data={"query": "Describe this scene."})
    assert resp.status_code == 200
    query_id = resp.json().get("query_id")
    assert query_id is not None
    report = client.get(f"/report/{query_id}")
    assert report.status_code == 200
    assert report.json()["execution_trace"] == resp.json()["execution_trace"]  # must match exactly, not diverge

def test_full_pipeline_is_reproducible():
    r1 = client.post("/query", files={"image": open("fixtures/subset/sample_0.tif","rb")}, data={"query": "Describe this scene."})
    r2 = client.post("/query", files={"image": open("fixtures/subset/sample_0.tif","rb")}, data={"query": "Describe this scene."})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["answer"] == r2.json()["answer"]
