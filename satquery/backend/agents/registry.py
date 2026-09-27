"""Registry for SatQuery agents, models, and geospatial tools."""

from typing import Dict, Any, List


class ToolRegistry:
    """Maintains a catalog of available satellite models and geospatial analysis tools."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {
            "geospatial.validator": {
                "name": "Raster & Vector Validator",
                "description": "Verifies CRS alignment, spatial bounds, and raster integrity.",
            },
            "geospatial.metadata": {
                "name": "Metadata Extractor",
                "description": "Extracts resolution, bounding box coordinates, and projections.",
            },
            "geospatial.preprocessing": {
                "name": "Imagery Preprocessor",
                "description": "Performs band normalization, cloud masking, and chip tiling.",
            },
            "evidence.confidence": {
                "name": "Confidence Scorer",
                "description": "Calculates probabilistic confidence intervals for detections.",
            },
            "evidence.verifier": {
                "name": "Evidence Verifier",
                "description": "Performs temporal and spatial constraint verification.",
            },
        }

        self._models: Dict[str, Dict[str, Any]] = {
            "vqa": {
                "name": "Satellite VQA Engine",
                "version": "1.2.0",
                "capabilities": ["semantic classification", "visual counting", "scene interrogation"],
            },
            "captioning": {
                "name": "Scene Captioning / Description",
                "version": "1.0.0",
                "capabilities": ["natural-language scene description", "land-cover & major-object summary"],
            },
            "grounding": {
                "name": "Earth Observation Grounding",
                "version": "2.0.0",
                "capabilities": ["natural language bbox detection", "infrastructure localization"],
            },
            "change": {
                "name": "Bi-temporal Change Detector",
                "version": "1.5.0",
                "capabilities": ["urban expansion", "deforestation monitoring", "disaster assessment"],
            },
            "fusion": {
                "name": "Multi-Sensor Fusion (SAR/Optical)",
                "version": "1.0.0",
                "capabilities": ["all-weather imagery synthesis", "cross-modality alignment"],
            },
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        return [{"id": k, **v} for k, v in self._tools.items()]

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": k, **v} for k, v in self._models.items()]
