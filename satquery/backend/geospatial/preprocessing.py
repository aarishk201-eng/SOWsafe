"""Geospatial preprocessing utility.

Provides tiling, spectral normalization, and image chip extraction for satellite scenes.
"""

from typing import List, Dict, Any


class TilePreprocessor:
    """Chunks large satellite scenes into model-ingestible chips."""

    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap

    def calculate_tile_grid(self, width: int, height: int) -> List[Dict[str, int]]:
        """Calculates window offsets for chip slicing."""
        tiles = []
        stride = self.tile_size - self.overlap
        y = 0
        tile_idx = 0
        while y < height:
            x = 0
            while x < width:
                w = min(self.tile_size, width - x)
                h = min(self.tile_size, height - y)
                tiles.append({
                    "tile_id": tile_idx,
                    "x_offset": x,
                    "y_offset": y,
                    "width": w,
                    "height": h,
                })
                tile_idx += 1
                if x + w >= width:
                    break
                x += stride
            if y + h >= height:
                break
            y += stride
        return tiles

    def normalize_bands(self, band_data: List[float], min_val: float = 0.0, max_val: float = 10000.0) -> List[float]:
        """Normalizes Sentinel-2/Landsat reflectance values to [0, 1]."""
        if max_val <= min_val:
            return band_data
        return [max(0.0, min(1.0, (val - min_val) / (max_val - min_val))) for val in band_data]
