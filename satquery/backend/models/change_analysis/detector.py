"""Bi-Temporal Satellite Change Detection Subsystem.

Implements Change Vector Analysis (CVA) and multi-spectral differencing with
mandatory pre-flight geospatial compatibility verification.
"""

from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import os
import math
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from geospatial.validator import (
    extract_metadata,
    check_compatibility,
    GeoTiffMetadata,
    GeoTiffValidationError,
)


class IncompatibleScenesError(ValueError):
    """Raised when two raster scenes fail geospatial compatibility checks.

    Detecting change across scenes with mismatched CRS, mismatched resolution,
    or non-overlapping bounds yields misregistration noise rather than real ground changes.
    """

    def __init__(self, message: str, compatibility: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.compatibility = compatibility or {}


class ChangeMask(np.ndarray):
    """2D binary change mask with geospatial and statistical change attributes."""

    def __new__(
        cls,
        input_array: np.ndarray,
        change_detected: bool = False,
        change_percentage: float = 0.0,
        changed_pixels: int = 0,
        total_pixels: int = 0,
        overlap_bounds: Optional[Dict[str, float]] = None,
        meta_t1: Optional[GeoTiffMetadata] = None,
        meta_t2: Optional[GeoTiffMetadata] = None,
        compatibility: Optional[Dict[str, Any]] = None,
        confidence: float = 0.95,
    ):
        obj = np.asarray(input_array, dtype=np.uint8).view(cls)
        obj.change_detected = bool(change_detected)
        obj.change_percentage = float(change_percentage)
        obj.changed_pixels = int(changed_pixels)
        obj.total_pixels = int(total_pixels)
        obj.overlap_bounds = overlap_bounds or {}
        obj.meta_t1 = meta_t1
        obj.meta_t2 = meta_t2
        obj.compatibility = compatibility or {}

        # Compute bounding box of changed pixels if any
        bbox = None
        if change_detected and changed_pixels > 0:
            ys, xs = np.where(obj == 1)
            if len(ys) > 0:
                bbox = [int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))]
        obj.bbox = bbox

        obj.prediction = "change" if change_detected else "no_change"
        obj.confidence = float(confidence)
        obj.source_tool = "change_detection"
        obj.evidence = {
            "bbox": bbox,
            "area": float(changed_pixels),
            "mask": obj,
            "change_percentage": float(change_percentage),
            "changed_pixels": int(changed_pixels),
            "total_pixels": int(total_pixels),
        }
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.change_detected = getattr(obj, "change_detected", False)
        self.change_percentage = getattr(obj, "change_percentage", 0.0)
        self.changed_pixels = getattr(obj, "changed_pixels", 0)
        self.total_pixels = getattr(obj, "total_pixels", 0)
        self.overlap_bounds = getattr(obj, "overlap_bounds", {})
        self.meta_t1 = getattr(obj, "meta_t1", None)
        self.meta_t2 = getattr(obj, "meta_t2", None)
        self.compatibility = getattr(obj, "compatibility", {})
        self.bbox = getattr(obj, "bbox", None)
        self.source_tool = getattr(obj, "source_tool", "change_detection")
        self.prediction = getattr(obj, "prediction", "change" if self.change_detected else "no_change")
        self.confidence = getattr(obj, "confidence", 0.90)
        self.evidence = getattr(obj, "evidence", {
            "bbox": self.bbox,
            "area": float(self.changed_pixels),
            "mask": self,
            "change_percentage": float(self.change_percentage),
        })

    def keys(self):
        return {
            "prediction",
            "confidence",
            "evidence",
            "source_tool",
            "change_detected",
            "change_percentage",
            "changed_pixels",
            "total_pixels",
            "bbox",
        }

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            return item in self.keys() or hasattr(self, item)
        return super().__contains__(item)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item == "source_tool":
                return self.source_tool
            if item == "prediction":
                return self.prediction
            if item == "confidence":
                return self.confidence
            if item == "evidence":
                return self.evidence
            if item == "bbox":
                return self.bbox
            if hasattr(self, item):
                return getattr(self, item)
            raise KeyError(item)
        return super().__getitem__(item)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes change mask statistics to a dictionary adhering to the specialist tool schema."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "evidence": {
                "bbox": self.bbox,
                "area": float(self.changed_pixels),
                "mask": np.asarray(self).tolist(),
                "change_percentage": float(self.change_percentage),
                "changed_pixels": int(self.changed_pixels),
            },
            "source_tool": self.source_tool,
            "change_detected": bool(self.change_detected),
            "change_percentage": float(self.change_percentage),
            "changed_pixels": int(self.changed_pixels),
            "total_pixels": int(self.total_pixels),
            "shape": list(self.shape),
            "overlap_bounds": self.overlap_bounds,
            "bbox": self.bbox,
        }


def detect(
    image_t1: Union[str, Path],
    image_t2: Union[str, Path],
    threshold_sigma: float = 2.0,
    min_change_pixels: int = 15,
) -> ChangeMask:
    """Detects spatial and spectral ground changes between two temporal acquisitions.

    Hard Requirement:
        Before executing change detection, verifies via Phase 1's check_compatibility
        that both images share CRS and resolution and have spatial overlap. Mismatched
        inputs immediately raise IncompatibleScenesError to avoid misregistration noise.

    Args:
        image_t1: Path to Time 1 (reference/pre-event) GeoTIFF.
        image_t2: Path to Time 2 (post-event) GeoTIFF.
        threshold_sigma: Multiplier for adaptive standard deviation thresholding.
        min_change_pixels: Minimum number of changed pixels to flag positive change.

    Returns:
        ChangeMask: 2D binary numpy array (0 = unchanged, 1 = changed) with statistics.

    Raises:
        GeoTiffValidationError: If either file is invalid, missing, or lacks CRS.
        IncompatibleScenesError: If images mismatch in CRS, resolution, or spatial overlap.
    """
    path_t1 = str(image_t1)
    path_t2 = str(image_t2)

    # 1. Extract metadata for both rasters
    meta_t1 = extract_metadata(path_t1)
    meta_t2 = extract_metadata(path_t2)

    # 2. Compatibility Engine pre-flight verification
    compat = check_compatibility(meta_t1, meta_t2)

    if not compat["crs_compatible"]:
        raise IncompatibleScenesError(
            f"Cannot detect change: CRS mismatch between '{meta_t1.crs}' and '{meta_t2.crs}'. "
            f"Mismatched inputs will produce a change mask that is really just misregistration noise.",
            compatibility=compat,
        )

    if not compat["resolution_compatible"]:
        raise IncompatibleScenesError(
            f"Cannot detect change: Spatial resolution mismatch between {meta_t1.resolution} and {meta_t2.resolution}. "
            f"Mismatched inputs will produce a change mask that is really just misregistration noise.",
            compatibility=compat,
        )

    if not compat["spatial_overlap"]:
        raise IncompatibleScenesError(
            f"Cannot detect change: No spatial overlap between scene A {meta_t1.bounds} and scene B {meta_t2.bounds}. "
            f"Mismatched inputs will produce a change mask that is really just misregistration noise.",
            compatibility=compat,
        )

    # 3. Read raster data over overlapping window
    with rasterio.open(path_t1) as src1, rasterio.open(path_t2) as src2:
        overlap_bounds = compat.get("overlap_bounds")
        
        if compat["dimensions_compatible"] and (
            abs(meta_t1.bounds["left"] - meta_t2.bounds["left"]) < 1e-4
            and abs(meta_t1.bounds["top"] - meta_t2.bounds["top"]) < 1e-4
        ):
            # Identical footprints
            arr1 = src1.read().astype(np.float32)
            arr2 = src2.read().astype(np.float32)
        else:
            # Windowed read over overlapping spatial bounds
            b = overlap_bounds
            win1 = from_bounds(b["left"], b["bottom"], b["right"], b["top"], src1.transform)
            win2 = from_bounds(b["left"], b["bottom"], b["right"], b["top"], src2.transform)
            arr1 = src1.read(window=win1).astype(np.float32)
            arr2 = src2.read(window=win2).astype(np.float32)

            # Match array dimensions if fractional pixel rounding differs
            min_h = min(arr1.shape[1], arr2.shape[1])
            min_w = min(arr1.shape[2], arr2.shape[2])
            arr1 = arr1[:, :min_h, :min_w]
            arr2 = arr2[:, :min_h, :min_w]

        nodata1 = src1.nodata
        nodata2 = src2.nodata

    # 4. Handle nodata masking
    valid_mask = np.ones((arr1.shape[1], arr1.shape[2]), dtype=bool)
    if nodata1 is not None:
        valid_mask &= ~(np.any(arr1 == nodata1, axis=0))
    if nodata2 is not None:
        valid_mask &= ~(np.any(arr2 == nodata2, axis=0))

    # Match band counts for differencing
    common_bands = min(arr1.shape[0], arr2.shape[0])
    diff = np.abs(arr2[:common_bands] - arr1[:common_bands])

    # 5. Change Vector Analysis (CVA) Magnitude
    # Euclidean distance across shared spectral bands
    cva_magnitude = np.sqrt(np.sum(diff ** 2, axis=0))

    # Incorporate NDVI difference if 4+ bands available (NIR - Red)
    if common_bands >= 4:
        # Band 3 = Red, Band 4 = NIR
        red1, nir1 = arr1[2], arr1[3]
        red2, nir2 = arr2[2], arr2[3]

        denom1 = np.maximum(nir1 + red1, 1e-5)
        denom2 = np.maximum(nir2 + red2, 1e-5)
        ndvi1 = (nir1 - red1) / denom1
        ndvi2 = (nir2 - red2) / denom2
        delta_ndvi = np.abs(ndvi2 - ndvi1)
        # Normalize and weight NDVI difference
        cva_magnitude += delta_ndvi * (np.max(cva_magnitude) * 0.5 + 1.0)

    # 6. Statistical Thresholding
    valid_diffs = cva_magnitude[valid_mask]
    total_valid_pixels = int(np.sum(valid_mask))

    if total_valid_pixels == 0 or np.all(valid_diffs == 0):
        # Identical scenes or no valid pixels
        mask_arr = np.zeros((arr1.shape[1], arr1.shape[2]), dtype=np.uint8)
        return ChangeMask(
            mask_arr,
            change_detected=False,
            change_percentage=0.0,
            changed_pixels=0,
            total_pixels=max(1, total_valid_pixels),
            overlap_bounds=overlap_bounds,
            meta_t1=meta_t1,
            meta_t2=meta_t2,
            compatibility=compat,
        )

    mean_diff = float(np.mean(valid_diffs))
    std_diff = float(np.std(valid_diffs))

    # If scenes are nearly identical with only floating-point noise
    if std_diff < 1e-3 and mean_diff < 1e-2:
        mask_arr = np.zeros((arr1.shape[1], arr1.shape[2]), dtype=np.uint8)
        return ChangeMask(
            mask_arr,
            change_detected=False,
            change_percentage=0.0,
            changed_pixels=0,
            total_pixels=total_valid_pixels,
            overlap_bounds=overlap_bounds,
            meta_t1=meta_t1,
            meta_t2=meta_t2,
            compatibility=compat,
        )

    # Dynamic threshold: mean + threshold_sigma * std
    thresh = mean_diff + threshold_sigma * std_diff
    raw_change = (cva_magnitude > thresh) & valid_mask

    # 7. Morphological Cleanup (filter single-pixel sensor jitter)
    cleaned_mask = np.zeros_like(raw_change, dtype=np.uint8)
    h, w = raw_change.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if raw_change[y, x]:
                # Require at least 2 connected neighbors to eliminate speckle noise
                neighbors = np.sum(raw_change[y - 1 : y + 2, x - 1 : x + 2]) - 1
                if neighbors >= 2:
                    cleaned_mask[y, x] = 1

    changed_pixels = int(np.sum(cleaned_mask))
    change_detected = changed_pixels >= min_change_pixels
    change_pct = round((changed_pixels / total_valid_pixels) * 100, 2)

    confidence = 0.95
    if change_detected and std_diff > 1e-6:
        # Calculate the mean Z-score of the changed pixels
        changed_magnitudes = cva_magnitude[cleaned_mask == 1]
        mean_changed = float(np.mean(changed_magnitudes))
        z_score = (mean_changed - mean_diff) / std_diff
        # Use the cumulative distribution function (CDF) of the standard normal distribution
        # to represent the statistical certainty that these pixels are true outliers.
        confidence = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
        # Bound confidence between [0.0, 1.0] and cap at 0.99 for numerical stability
        confidence = max(0.0, min(0.99, float(confidence)))

    return ChangeMask(
        cleaned_mask,
        change_detected=change_detected,
        change_percentage=change_pct,
        changed_pixels=changed_pixels,
        total_pixels=total_valid_pixels,
        overlap_bounds=overlap_bounds,
        meta_t1=meta_t1,
        meta_t2=meta_t2,
        compatibility=compat,
        confidence=confidence,
    )


# Alias matching functional test conventions
detect_change = detect
