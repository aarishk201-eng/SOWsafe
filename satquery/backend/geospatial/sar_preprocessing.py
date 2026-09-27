"""SAR (Synthetic Aperture Radar) Preprocessing and Radar Physics Module.

Specifically handles microwave backscatter calibration, speckle noise reduction
via edge-preserving Lee filtering, and dual-polarization decomposition (VV, VH, VH/VV).
Explicitly separated from optical preprocessing to respect radar physics.
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np


class SARPreprocessor:
    """Specialized preprocessor for Sentinel-1 / RISAT microwave SAR data."""

    def __init__(self, calibration_constant: float = -83.0, default_looks: int = 1):
        self.calibration_constant = calibration_constant
        self.default_looks = default_looks

    def calibrate_to_decibels(
        self,
        dn_array: np.ndarray,
        min_db: float = -30.0,
        max_db: float = 0.0,
    ) -> np.ndarray:
        """Converts raw digital numbers (DN) or linear amplitude to sigma-nought (sigma0) in dB.

        Formula: sigma0_db = 10 * log10(DN^2 + eps)
        Clipped to the standard Sentinel-1/BigEarthNet backscatter range [-30 dB, 0 dB].
        """
        arr = np.asarray(dn_array, dtype=np.float32)
        # Handle zero or negative intensity
        intensity = np.maximum(arr ** 2 if np.nanmax(arr) > 50 else arr, 1e-7)
        sigma0_db = 10.0 * np.log10(intensity)

        # Clip to physical radar backscatter bounds
        clipped_db = np.clip(sigma0_db, min_db, max_db)
        return clipped_db

    def normalize_db(self, db_array: np.ndarray, min_db: float = -30.0, max_db: float = 0.0) -> np.ndarray:
        """Normalizes decibel backscatter into the range [0, 1] for model input."""
        clipped = np.clip(db_array, min_db, max_db)
        return (clipped - min_db) / (max_db - min_db)

    def lee_speckle_filter(
        self,
        img: np.ndarray,
        window_size: int = 5,
        num_looks: int = 1,
    ) -> np.ndarray:
        """Edge-preserving Lee filter for multiplicative SAR speckle noise reduction.

        R_hat = I_bar + W * (I - I_bar)
        where W = max(0, (var_I - var_noise) / var_I)
        var_noise = I_bar^2 / num_looks
        """
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 3:
            # Multi-channel (e.g. VV and VH)
            return np.stack([self.lee_speckle_filter(arr[b], window_size, num_looks) for b in range(arr.shape[0])])

        h, w = arr.shape
        pad = window_size // 2
        padded = np.pad(arr, pad, mode="reflect")
        output = np.zeros_like(arr)

        # Precompute square for fast local variance
        padded_sq = padded ** 2
        k_area = window_size * window_size

        for y in range(h):
            for x in range(w):
                patch = padded[y : y + window_size, x : x + window_size]
                local_mean = float(np.mean(patch))
                local_var = float(np.var(patch))

                # Multiplicative noise variance model
                noise_var = (local_mean ** 2) / float(num_looks)

                if local_var > noise_var and local_var > 1e-8:
                    weight = (local_var - noise_var) / local_var
                else:
                    weight = 0.0

                center_val = arr[y, x]
                output[y, x] = local_mean + weight * (center_val - local_mean)

        return output

    def compute_polarimetric_features(
        self,
        vv_band: np.ndarray,
        vh_band: np.ndarray,
    ) -> Dict[str, Any]:
        """Computes dual-polarization SAR decomposition metrics (VV, VH, VH/VV ratio, and RVI).

        Physical Interpretation:
        - Calm Water: Specular forward scatter away from antenna -> very low VV and extremely low VH.
        - Built-up / Urban: Dihedral double-bounce reflection -> high VV, moderate to high VH.
        - Dense Forest / Canopy: Depolarizing volume scattering -> high VH and elevated VH/VV cross-ratio.
        """
        vv_filtered = self.lee_speckle_filter(vv_band)
        vh_filtered = self.lee_speckle_filter(vh_band)

        vv_db = self.calibrate_to_decibels(vv_filtered)
        vh_db = self.calibrate_to_decibels(vh_filtered)

        # Cross-polarization ratio: VH_db - VV_db (equivalent to log10(VH/VV))
        vh_vv_ratio_db = vh_db - vv_db

        # Radar Vegetation Index (RVI) approximation: 4 * VH / (VV + VH)
        vv_lin = 10.0 ** (vv_db / 10.0)
        vh_lin = 10.0 ** (vh_db / 10.0)
        rvi = (4.0 * vh_lin) / np.maximum(vv_lin + vh_lin, 1e-7)
        rvi = np.clip(rvi, 0.0, 1.0)

        return {
            "vv_db": vv_db,
            "vh_db": vh_db,
            "vh_vv_ratio_db": vh_vv_ratio_db,
            "rvi": rvi,
            "mean_vv_db": float(np.mean(vv_db)),
            "mean_vh_db": float(np.mean(vh_db)),
            "mean_vh_vv_ratio_db": float(np.mean(vh_vv_ratio_db)),
            "mean_rvi": float(np.mean(rvi)),
        }
