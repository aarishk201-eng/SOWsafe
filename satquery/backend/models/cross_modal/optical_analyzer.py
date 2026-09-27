"""Optical Multi-Spectral Scene Analyzer.

Analyzes visible, near-infrared, and shortwave bands, extracting vegetation indices,
water indices, and atmospheric/cloud occlusion metrics.
"""

from typing import Dict, Any, Union
from pathlib import Path
import os
import rasterio
import numpy as np


def optical_analyzer(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Analyzes optical multi-spectral satellite imagery.

    Returns:
        Dict[str, Any]: {
            "prediction": "built_up" | "vegetation" | "water" | "bare_soil",
            "confidence": float,
            "evidence": {
                "ndvi": float,
                "ndwi": float,
                "mean_reflectance": float,
                "cloud_score": float,
                "cloud_occluded": bool,
                "sensor": "optical",
            }
        }
    """
    path_str = str(image_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Optical image not found at: {path_str}")

    with rasterio.open(path_str) as src:
        bands = src.read().astype(np.float32)
        count = src.count

    # Normalize reflectance from typical 12-bit [0, 10000] scale to [0, 1]
    max_val = np.nanmax(bands)
    if max_val > 10.0:
        norm_bands = bands / 10000.0
    else:
        norm_bands = bands

    # Standard band layout assumption:
    # 4-band: 0=Blue, 1=Green, 2=Red, 3=NIR
    # 3-band: 0=Red, 1=Green, 2=Blue
    if count >= 4:
        blue, green, red, nir = norm_bands[0], norm_bands[1], norm_bands[2], norm_bands[3]
    elif count == 3:
        red, green, blue = norm_bands[0], norm_bands[1], norm_bands[2]
        nir = red * 1.1
    else:
        gray = norm_bands[0]
        red = green = blue = nir = gray

    # Spectral Indices
    # NDVI: (NIR - Red) / (NIR + Red)
    ndvi_denom = np.maximum(nir + red, 1e-5)
    ndvi = (nir - red) / ndvi_denom
    mean_ndvi = float(np.nanmean(ndvi))

    # NDWI: (Green - NIR) / (Green + NIR)
    ndwi_denom = np.maximum(green + nir, 1e-5)
    ndwi = (green - nir) / ndwi_denom
    mean_ndwi = float(np.nanmean(ndwi))

    mean_brightness = float(np.nanmean((red + green + blue) / 3.0))

    # Cloud Occlusion Assessment
    # Clouds exhibit high reflectance across all visible bands and high spatial variance
    cloud_score = 0.0
    mean_blue = float(np.nanmean(blue))
    if mean_blue > 0.35 and mean_brightness > 0.40 and mean_ndvi < 0.1:
        cloud_score = float(np.clip((mean_brightness - 0.35) * 3.0, 0.0, 1.0))
    cloud_occluded = cloud_score > 0.45

    evidence = {
        "sensor": "optical",
        "band_count": count,
        "ndvi": round(mean_ndvi, 4),
        "ndwi": round(mean_ndwi, 4),
        "mean_brightness": round(mean_brightness, 4),
        "cloud_score": round(cloud_score, 4),
        "cloud_occluded": cloud_occluded,
        "bbox": None,
        "area": None,
        "mask": None,
    }

    # If heavily cloud occluded, optical confidence collapses
    if cloud_occluded:
        return {
            "prediction": "cloud_occluded",
            "confidence": 0.25,
            "evidence": evidence,
            "source_tool": "optical_analyzer",
        }

    # Land cover classification from optical physics
    if mean_ndwi > 0.15:
        prediction = "water"
        confidence = float(np.clip(0.75 + mean_ndwi * 0.3, 0.70, 0.96))
    elif mean_ndvi > 0.25:
        prediction = "vegetation"
        confidence = float(np.clip(0.70 + mean_ndvi * 0.35, 0.70, 0.95))
    elif mean_brightness > 0.22 and mean_ndvi < 0.20:
        prediction = "built_up"
        confidence = float(np.clip(0.75 + mean_brightness * 0.4, 0.70, 0.94))
    else:
        prediction = "bare_soil"
        confidence = 0.78

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "evidence": evidence,
        "source_tool": "optical_analyzer",
    }
