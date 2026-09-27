"""Geospatial validation and compatibility engine.

Validates GeoTIFF metadata, coordinate reference systems (CRS), dimensions,
and determines pairwise raster compatibility using reprojection of bounds.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import os
from pydantic import BaseModel

import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.errors import RasterioIOError, CRSError


class GeoTiffValidationError(Exception):
    """Raised when a file is not a valid raster or has no CRS."""
    pass


class GeoTiffMetadata(BaseModel):
    """Metadata container for validated GeoTIFF datasets."""
    driver: str
    crs: str
    width: int
    height: int
    band_count: int
    dtype: str
    resolution: Tuple[float, float]
    bounds: Dict[str, float]
    nodata: Optional[float] = None


def extract_metadata(path: Union[str, Path]) -> GeoTiffMetadata:
    """Reads driver, CRS, width, height, band count, dtype, resolution, bounds, nodata via rasterio.

    Raises GeoTiffValidationError if the file isn't a valid raster or has no CRS.
    """
    path_str = str(path)
    if not os.path.exists(path_str):
        raise GeoTiffValidationError(f"File not found: {path_str}")

    try:
        with rasterio.open(path_str) as src:
            if src.crs is None:
                raise GeoTiffValidationError(
                    f"File '{path_str}' is not a valid GeoTIFF: missing Coordinate Reference System (CRS)."
                )

            crs_str = src.crs.to_string() if hasattr(src.crs, "to_string") else str(src.crs)
            if not crs_str:
                raise GeoTiffValidationError(f"File '{path_str}' has an empty or unresolvable CRS.")

            # Resolution (x_res, y_res)
            res_tuple = (float(src.res[0]), float(src.res[1]))

            # Extents
            b = src.bounds
            bounds_dict = {
                "left": float(b.left),
                "bottom": float(b.bottom),
                "right": float(b.right),
                "top": float(b.top),
            }

            dtype_str = str(src.dtypes[0]) if src.dtypes else "unknown"

            return GeoTiffMetadata(
                driver=str(src.driver),
                crs=crs_str,
                width=int(src.width),
                height=int(src.height),
                band_count=int(src.count),
                dtype=dtype_str,
                resolution=res_tuple,
                bounds=bounds_dict,
                nodata=float(src.nodata) if src.nodata is not None else None,
            )
    except GeoTiffValidationError:
        raise
    except (RasterioIOError, CRSError, Exception) as e:
        raise GeoTiffValidationError(f"Invalid raster file '{path_str}': {str(e)}") from e


def check_compatibility(meta_a: GeoTiffMetadata, meta_b: GeoTiffMetadata) -> dict:
    """The Compatibility Engine.

    Compares CRS, resolution, and dimensions directly; for spatial overlap,
    reprojects b's bounds into a's CRS with rasterio.warp.transform_bounds before
    intersecting (does not compare raw bounds across different CRSs).

    Returns:
        crs_compatible: bool
        resolution_compatible: bool
        dimensions_compatible: bool
        spatial_overlap: bool
        overlap_bounds: Optional[Dict[str, float]]
        ready_for_pairwise_analysis: bool
    """
    # 1. Compare CRS directly
    try:
        crs_a = CRS.from_user_input(meta_a.crs)
        crs_b = CRS.from_user_input(meta_b.crs)
        crs_compatible = bool(crs_a == crs_b)
    except Exception:
        crs_compatible = bool(meta_a.crs.strip().upper() == meta_b.crs.strip().upper())

    # 2. Compare resolution directly (with floating point tolerance)
    res_x_match = abs(meta_a.resolution[0] - meta_b.resolution[0]) <= 1e-5
    res_y_match = abs(meta_a.resolution[1] - meta_b.resolution[1]) <= 1e-5
    resolution_compatible = bool(res_x_match and res_y_match)

    # 3. Compare dimensions directly
    dimensions_compatible = bool(
        meta_a.width == meta_b.width and meta_a.height == meta_b.height
    )

    # 4. Compute spatial overlap
    # Reproject b's bounds into a's CRS with rasterio.warp.transform_bounds before intersecting
    try:
        b_bounds = meta_b.bounds
        if crs_compatible:
            b_trans_left = b_bounds["left"]
            b_trans_bottom = b_bounds["bottom"]
            b_trans_right = b_bounds["right"]
            b_trans_top = b_bounds["top"]
        else:
            trans_left, trans_bottom, trans_right, trans_top = transform_bounds(
                meta_b.crs,
                meta_a.crs,
                b_bounds["left"],
                b_bounds["bottom"],
                b_bounds["right"],
                b_bounds["top"],
            )
            b_trans_left = trans_left
            b_trans_bottom = trans_bottom
            b_trans_right = trans_right
            b_trans_top = trans_top

        a_bounds = meta_a.bounds
        inter_left = max(a_bounds["left"], b_trans_left)
        inter_bottom = max(a_bounds["bottom"], b_trans_bottom)
        inter_right = min(a_bounds["right"], b_trans_right)
        inter_top = min(a_bounds["top"], b_trans_top)

        if inter_left < inter_right and inter_bottom < inter_top:
            spatial_overlap = True
            overlap_bounds = {
                "left": round(inter_left, 6),
                "bottom": round(inter_bottom, 6),
                "right": round(inter_right, 6),
                "top": round(inter_top, 6),
            }
        else:
            spatial_overlap = False
            overlap_bounds = None
    except Exception:
        spatial_overlap = False
        overlap_bounds = None

    # 5. Readiness for pairwise analysis
    ready_for_pairwise_analysis = bool(
        crs_compatible and resolution_compatible and dimensions_compatible and spatial_overlap
    )

    return {
        "crs_compatible": crs_compatible,
        "resolution_compatible": resolution_compatible,
        "dimensions_compatible": dimensions_compatible,
        "spatial_overlap": spatial_overlap,
        "overlap_bounds": overlap_bounds,
        "ready_for_pairwise_analysis": ready_for_pairwise_analysis,
    }


class GeospatialValidator:
    """Legacy helper class for raster datasets and coordinate integrity."""

    def __init__(self, allowed_crs: Optional[List[str]] = None):
        self.allowed_crs = allowed_crs or ["EPSG:4326", "EPSG:3857", "EPSG:32632", "EPSG:32633"]

    def validate_bounds(self, min_x: float, min_y: float, max_x: float, max_y: float) -> bool:
        """Verifies if the bounding box coordinates are well-formed."""
        if min_x >= max_x or min_y >= max_y:
            return False
        return True

    def validate_raster_file(self, file_path: str) -> Dict[str, Any]:
        """Validates raster file presence, band integrity, and CRS."""
        try:
            metadata = extract_metadata(file_path)
            valid_bounds = self.validate_bounds(
                metadata.bounds["left"],
                metadata.bounds["bottom"],
                metadata.bounds["right"],
                metadata.bounds["top"],
            )
            return {
                "valid": valid_bounds,
                "file_path": file_path,
                "crs": metadata.crs,
                "band_count": metadata.band_count,
                "dimensions": [metadata.width, metadata.height],
                "bounds": metadata.bounds,
            }
        except GeoTiffValidationError as e:
            return {
                "valid": False,
                "file_path": file_path,
                "error": str(e),
            }
