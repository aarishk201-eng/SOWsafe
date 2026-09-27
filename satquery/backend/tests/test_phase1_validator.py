# backend/tests/test_phase1_validator.py
import numpy as np, rasterio, pytest, sys, os
from rasterio.transform import from_origin
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geospatial.validator import extract_metadata, check_compatibility, GeoTiffValidationError


def _write_tif(path, width=512, height=512, crs="EPSG:32643", res=10.0,
               bands=4, origin=(500000, 2200000), dtype="uint16", nodata=0):
    transform = from_origin(origin[0], origin[1], res, res)
    data = (np.random.rand(bands, height, width) * 3000).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                        count=bands, dtype=dtype, crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(data)


@pytest.fixture
def tif_a(tmp_path):
    p = str(tmp_path / "a.tif"); _write_tif(p); return p


@pytest.fixture
def tif_b_matching(tmp_path):
    p = str(tmp_path / "b_match.tif"); _write_tif(p, origin=(500000, 2200000)); return p


@pytest.fixture
def tif_b_diff_crs(tmp_path):
    p = str(tmp_path / "b_diffcrs.tif")
    _write_tif(p, crs="EPSG:4326", res=0.0001, origin=(75.0, 19.896))  # = a's UTM origin in lon/lat
    return p


@pytest.fixture
def tif_b_no_overlap(tmp_path):
    p = str(tmp_path / "b_far.tif"); _write_tif(p, origin=(900000, 3000000)); return p


def test_metadata_extraction_reads_correct_fields(tif_a):
    meta = extract_metadata(tif_a)
    assert meta.width == 512 and meta.height == 512
    assert meta.band_count == 4
    assert meta.crs == "EPSG:32643"
    assert abs(meta.resolution[0] - 10.0) < 1e-6


def test_rejects_non_georeferenced_or_missing_file(tmp_path):
    bad = tmp_path / "not_a_raster.txt"; bad.write_text("hello")
    with pytest.raises(GeoTiffValidationError):
        extract_metadata(str(bad))


def test_matching_pair_is_compatible(tif_a, tif_b_matching):
    result = check_compatibility(extract_metadata(tif_a), extract_metadata(tif_b_matching))
    assert result["ready_for_pairwise_analysis"] is True


def test_different_crs_pair_detects_overlap_after_reprojection(tif_a, tif_b_diff_crs):
    result = check_compatibility(extract_metadata(tif_a), extract_metadata(tif_b_diff_crs))
    assert result["crs_compatible"] is False
    assert result["spatial_overlap"] is True          # only correct because we reproject before comparing
    assert result["ready_for_pairwise_analysis"] is False


def test_non_overlapping_pair_is_flagged_incompatible(tif_a, tif_b_no_overlap):
    result = check_compatibility(extract_metadata(tif_a), extract_metadata(tif_b_no_overlap))
    assert result["spatial_overlap"] is False
