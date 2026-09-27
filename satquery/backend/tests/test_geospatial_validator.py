import os
import tempfile
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from main import app
from geospatial.validator import (
    extract_metadata,
    check_compatibility,
    GeoTiffMetadata,
    GeoTiffValidationError,
)

client = TestClient(app)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def create_synthetic_raster(
    file_path: str,
    width: int = 100,
    height: int = 100,
    count: int = 3,
    crs: str | None = "EPSG:4326",
    origin_x: float = 10.0,
    origin_y: float = 50.0,
    res_x: float = 0.01,
    res_y: float = 0.01,
    dtype: str = "uint8",
):
    """Utility to generate synthetic GeoTIFF files for testing."""
    transform = from_origin(origin_x, origin_y, res_x, res_y)
    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        data = np.ones((count, height, width), dtype=dtype)
        dst.write(data)


def test_extract_metadata_valid_geotiff(temp_dir):
    tif_path = os.path.join(temp_dir, "valid_scene.tif")
    create_synthetic_raster(tif_path, width=200, height=150, count=4, crs="EPSG:4326")

    meta = extract_metadata(tif_path)
    assert isinstance(meta, GeoTiffMetadata)
    assert meta.driver == "GTiff"
    assert meta.crs == "EPSG:4326"
    assert meta.width == 200
    assert meta.height == 150
    assert meta.band_count == 4
    assert meta.dtype == "uint8"
    assert abs(meta.resolution[0] - 0.01) < 1e-6
    assert abs(meta.resolution[1] - 0.01) < 1e-6
    assert "left" in meta.bounds
    assert meta.bounds["left"] == 10.0


def test_extract_metadata_missing_crs(temp_dir):
    tif_path = os.path.join(temp_dir, "no_crs.tif")
    create_synthetic_raster(tif_path, crs=None)

    with pytest.raises(GeoTiffValidationError) as excinfo:
        extract_metadata(tif_path)
    assert "missing Coordinate Reference System (CRS)" in str(excinfo.value)


def test_extract_metadata_invalid_file(temp_dir):
    corrupt_path = os.path.join(temp_dir, "not_a_raster.txt")
    with open(corrupt_path, "w") as f:
        f.write("Just some text content")

    with pytest.raises(GeoTiffValidationError):
        extract_metadata(corrupt_path)


def test_extract_metadata_nonexistent_file():
    with pytest.raises(GeoTiffValidationError) as excinfo:
        extract_metadata("non_existent_file.tif")
    assert "File not found" in str(excinfo.value)


def test_check_compatibility_identical_scenes(temp_dir):
    path_a = os.path.join(temp_dir, "scene_a.tif")
    path_b = os.path.join(temp_dir, "scene_b.tif")
    create_synthetic_raster(path_a, origin_x=10.0, origin_y=50.0)
    create_synthetic_raster(path_b, origin_x=10.0, origin_y=50.0)

    meta_a = extract_metadata(path_a)
    meta_b = extract_metadata(path_b)

    compat = check_compatibility(meta_a, meta_b)
    assert compat["crs_compatible"] is True
    assert compat["resolution_compatible"] is True
    assert compat["dimensions_compatible"] is True
    assert compat["spatial_overlap"] is True
    assert compat["overlap_bounds"] is not None
    assert compat["ready_for_pairwise_analysis"] is True


def test_check_compatibility_disjoint_scenes(temp_dir):
    path_a = os.path.join(temp_dir, "scene_a.tif")
    path_b = os.path.join(temp_dir, "scene_b.tif")
    # Scene A in Europe (10, 50), Scene B in Australia (140, -30)
    create_synthetic_raster(path_a, origin_x=10.0, origin_y=50.0)
    create_synthetic_raster(path_b, origin_x=140.0, origin_y=-30.0)

    meta_a = extract_metadata(path_a)
    meta_b = extract_metadata(path_b)

    compat = check_compatibility(meta_a, meta_b)
    assert compat["spatial_overlap"] is False
    assert compat["overlap_bounds"] is None
    assert compat["ready_for_pairwise_analysis"] is False


def test_check_compatibility_reprojected_overlap_different_crs(temp_dir):
    # Scene A in WGS84 (EPSG:4326), Scene B in Web Mercator (EPSG:3857)
    # Covering roughly the same location near London (0.0 longitude, 51.5 latitude)
    path_a = os.path.join(temp_dir, "scene_wgs84.tif")
    path_b = os.path.join(temp_dir, "scene_mercator.tif")

    # WGS84: -0.1 to +0.9 deg lon, 51.0 to 52.0 deg lat
    create_synthetic_raster(path_a, crs="EPSG:4326", origin_x=-0.1, origin_y=52.0, res_x=0.01, res_y=0.01)
    # Web Mercator around lon=0, lat=51.5 in meters (~0m X, ~6700000m Y)
    create_synthetic_raster(path_b, crs="EPSG:3857", origin_x=-5000, origin_y=6720000, res_x=100, res_y=100)

    meta_a = extract_metadata(path_a)
    meta_b = extract_metadata(path_b)

    compat = check_compatibility(meta_a, meta_b)
    assert compat["crs_compatible"] is False
    # Despite different CRS, spatial overlap is detected because b's bounds are reprojected into a's CRS
    assert compat["spatial_overlap"] is True
    assert compat["overlap_bounds"] is not None


def test_api_validate_single_endpoint(temp_dir):
    tif_path = os.path.join(temp_dir, "api_test.tif")
    create_synthetic_raster(tif_path, width=80, height=80, count=2)

    # Test via JSON body with path
    response = client.post("/validate", json={"path": tif_path})
    assert response.status_code == 200
    data = response.json()
    assert data["driver"] == "GTiff"
    assert data["width"] == 80
    assert data["height"] == 80
    assert data["band_count"] == 2


def test_api_validate_upload_file(temp_dir):
    tif_path = os.path.join(temp_dir, "upload_test.tif")
    create_synthetic_raster(tif_path, width=64, height=64, count=1)

    with open(tif_path, "rb") as f:
        response = client.post("/validate", files={"file": ("upload_test.tif", f, "image/tiff")})

    assert response.status_code == 200
    data = response.json()
    assert data["driver"] == "GTiff"
    assert data["width"] == 64


def test_api_validate_invalid_file(temp_dir):
    corrupt_path = os.path.join(temp_dir, "bad.tif")
    with open(corrupt_path, "wb") as f:
        f.write(b"not a valid tiff")

    response = client.post("/validate", json={"path": corrupt_path})
    assert response.status_code == 400
    assert "Invalid raster file" in response.json()["detail"]


def test_api_validate_pair_endpoint(temp_dir):
    path_a = os.path.join(temp_dir, "pair_a.tif")
    path_b = os.path.join(temp_dir, "pair_b.tif")
    create_synthetic_raster(path_a, origin_x=10.0, origin_y=50.0)
    create_synthetic_raster(path_b, origin_x=10.0, origin_y=50.0)

    response = client.post("/validate-pair", json={"path_a": path_a, "path_b": path_b})
    assert response.status_code == 200
    data = response.json()
    assert "meta_a" in data
    assert "meta_b" in data
    assert "compatibility" in data
    assert data["compatibility"]["spatial_overlap"] is True
    assert data["compatibility"]["ready_for_pairwise_analysis"] is True
