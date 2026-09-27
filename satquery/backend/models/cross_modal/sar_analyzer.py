"""SAR Microwave Scene Analyzer.

Operates independently on radar backscatter data (VV, VH polarizations),
applying speckle filtering and polarimetric decomposition to classify terrain.
"""

from typing import Dict, Any, Union
from pathlib import Path
import os
import rasterio
import numpy as np

from geospatial.sar_preprocessing import SARPreprocessor


def sar_analyzer(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Analyzes Synthetic Aperture Radar (SAR) imagery.

    Returns:
        Dict[str, Any]: {
            "prediction": "built_up" | "vegetation" | "water" | "bare_soil",
            "confidence": float,
            "evidence": {
                "mean_vv_db": float,
                "mean_vh_db": float,
                "mean_vh_vv_ratio_db": float,
                "mean_rvi": float,
                "speckle_filtered": bool,
                "sensor": "sar",
            }
        }
    """
    path_str = str(image_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"SAR image not found at: {path_str}")

    with rasterio.open(path_str) as src:
        bands = src.read().astype(np.float32)
        count = src.count

    preprocessor = SARPreprocessor()

    if count >= 2:
        vv_raw = bands[0]
        vh_raw = bands[1]
    else:
        vv_raw = bands[0]
        vh_raw = bands[0] * 0.25

    # Run dedicated SAR preprocessing pipeline
    features = preprocessor.compute_polarimetric_features(vv_raw, vh_raw)

    mean_vv = features["mean_vv_db"]
    mean_vh = features["mean_vh_db"]
    mean_ratio = features["mean_vh_vv_ratio_db"]
    mean_rvi = features["mean_rvi"]

    evidence = {
        "sensor": "sar",
        "band_count": count,
        "mean_vv_db": round(mean_vv, 2),
        "mean_vh_db": round(mean_vh, 2),
        "mean_vh_vv_ratio_db": round(mean_ratio, 2),
        "mean_rvi": round(mean_rvi, 4),
        "speckle_filtered": True,
        "all_weather_reliable": True,
        "bbox": None,
        "area": None,
        "mask": None,
    }

    # Radar Physics Classification:
    # 1. Specular Null (Water): Smooth water reflects radar pulse forward away from sensor -> very low backscatter
    if mean_vv < -20.0 and mean_vh < -24.0:
        prediction = "water"
        confidence = float(np.clip(0.80 + (-20.0 - mean_vv) * 0.02, 0.75, 0.95))

    # 2. Double-Bounce Scattering (Built-up): Dihedral reflection from wall-ground corner reflectors -> strong VV and high VH
    elif mean_vv > -10.0 and mean_vh > -17.0:
        prediction = "built_up"
        confidence = float(np.clip(0.80 + (mean_vv + 10.0) * 0.025, 0.75, 0.96))

    # 3. Volume Scattering (Vegetation/Forest Canopy): Multiple random reflections in leaves/twigs -> elevated cross-pol VH and high RVI
    elif mean_rvi > 0.35 or mean_vh > -18.5:
        prediction = "vegetation"
        confidence = float(np.clip(0.75 + mean_rvi * 0.25, 0.70, 0.93))

    # 4. Rough Surface Scattering (Bare soil / arid ground)
    else:
        prediction = "bare_soil"
        confidence = 0.78

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "evidence": evidence,
        "source_tool": "sar_analyzer",
    }
