import os
import sys
import tempfile
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import app
from models.vqa.eval_utils import load_ids

client = TestClient(app)

def _write_tif(path, width=256, height=256, bands=4, crs="EPSG:32643", res=10.0, nodata=0, corrupt=False):
    if corrupt:
        # Write invalid garbage to simulate corrupt file
        with open(path, "wb") as f:
            f.write(b"not a valid tiff file")
        return path

    transform = from_origin(300000.0, 4000000.0, res, res)
    data = np.random.randint(1, 255, size=(bands, height, width)).astype(np.uint8)
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return path


def test_no_hardcoded_dimensions(tmp_path):
    # Pass an odd, non-square dimension (e.g. 512x128)
    img_path = str(tmp_path / "odd_shape.tif")
    _write_tif(img_path, width=512, height=128)
    
    payload = {
        "query": "Is there any water in this scene?",
        "images": [img_path]
    }
    
    resp = client.post("/query", json=payload)
    # The pipeline should not crash internally due to 256x256 assumptions
    assert resp.status_code == 200, "Pipeline crashed when given a 512x128 image, likely due to hardcoded dimensions."
    data = resp.json()
    assert "answer" in data


def test_cartosat_risat_compatibility(tmp_path):
    # Cartosat-2S typical simulation (high res optical, e.g. 0.6m, EPSG:32643)
    cartosat_path = str(tmp_path / "cartosat.tif")
    _write_tif(cartosat_path, width=500, height=500, bands=4, crs="EPSG:32643", res=0.6)
    
    # RISAT typical simulation (C-band SAR, here we use same CRS but maybe different bands)
    risat_path = str(tmp_path / "risat.tif")
    _write_tif(risat_path, width=500, height=500, bands=2, crs="EPSG:32643", res=0.6)
    
    # Passing both to the fusion endpoint
    payload = {
        "query": "What is the primary land use across this scene?",
        "images": [cartosat_path, risat_path]
    }
    
    resp = client.post("/query", json=payload)
    # Ensure it works on novel specifications and doesn't hardcode Sentinel's 10m/20m res
    assert resp.status_code == 200
    assert "trace" in resp.json()


def test_graceful_failure_corrupt_input(tmp_path):
    # Generate a truly corrupt TIFF file
    corrupt_path = str(tmp_path / "corrupt.tif")
    _write_tif(corrupt_path, corrupt=True)
    
    payload = {
        "query": "What does this image show?",
        "images": [corrupt_path]
    }
    
    resp = client.post("/query", json=payload)
    
    # The unified /query endpoint returns 200 OK but sets status='failed' 
    # instead of crashing the server with 500. This is the definition of graceful.
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert "error" in str(data["answer"]).lower() or "invalid" in str(data["answer"]).lower() or "not a valid" in str(data["answer"]).lower()


def test_no_train_eval_leakage_sanity_check():
    import hashlib
    train_ids = load_ids("train_split.json")
    val_ids = load_ids("val_split.json")
    
    train_hashes = {hashlib.sha256(str(sid).encode("utf-8")).hexdigest() for sid in train_ids}
    val_hashes = {hashlib.sha256(str(sid).encode("utf-8")).hexdigest() for sid in val_ids}
    
    assert train_hashes.isdisjoint(val_hashes), "Training data is leaking into the evaluation splits!"


def test_blind_test_unseen_image_pair(tmp_path):
    # A completely random, unseen dimension and CRS configuration
    img1 = str(tmp_path / "blind1.tif")
    img2 = str(tmp_path / "blind2.tif")
    
    _write_tif(img1, width=300, height=300, crs="EPSG:3857", res=2.0)
    _write_tif(img2, width=300, height=300, crs="EPSG:3857", res=2.0)
    
    payload = {
        "query": "Is there structural change?",
        "images": [img1, img2]
    }
    
    resp = client.post("/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    
    # Check that execution trace successfully built a graph and executed it
    trace = data.get("trace", {})
    assert trace, "Pipeline failed to produce an execution trace on blind input"
