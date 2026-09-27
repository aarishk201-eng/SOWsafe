"""Geospatial metadata extraction utility.

Extracts resolution, bounding boxes, coordinate projections, and acquisition details.
"""

from typing import Dict, Any, Tuple


class MetadataExtractor:
    """Extracts geospatial properties and projection details."""

    def __init__(self, default_crs: str = "EPSG:4326"):
        self.default_crs = default_crs

    def transform_coordinates(
        self, x: float, y: float, source_crs: str, target_crs: str
    ) -> Tuple[float, float]:
        """Transforms coordinates between CRS using pyproj."""
        try:
            from pyproj import Transformer

            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            new_x, new_y = transformer.transform(x, y)
            return float(new_x), float(new_y)
        except ImportError:
            # Fallback if pyproj is not installed in local environment
            return x, y

    def extract_summary(self, bounds: Dict[str, float], crs: str = "EPSG:4326") -> Dict[str, Any]:
        """Summarizes geospatial metadata."""
        width = abs(bounds.get("right", 0.0) - bounds.get("left", 0.0))
        height = abs(bounds.get("top", 0.0) - bounds.get("bottom", 0.0))

        return {
            "crs": crs,
            "bounds": bounds,
            "spatial_extent": {"width_units": width, "height_units": height},
            "is_geographic": crs.lower() in ["epsg:4326", "wgs84"],
        }
