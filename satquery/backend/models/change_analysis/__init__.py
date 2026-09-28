"""Bi-temporal Satellite Change Detection Subsystem."""

from typing import Dict, Any, Union
from pathlib import Path

from .detector import (
    detect,
    detect_change,
    IncompatibleScenesError,
    ChangeMask,
)
from .change_vqa import (
    change_vqa,
    ChangeVQAEngine,
    analyze_change_evidence,
)


class ChangeDetectionModel:
    """Detects spatial and spectral alterations between multi-temporal acquisitions."""

    def __init__(self, model_name: str = "sat-change-v1"):
        self.model_name = model_name

    def detect_change(self, pre_image: Union[str, Path], post_image: Union[str, Path]) -> Dict[str, Any]:
        """Runs change detection pipeline and returns structured summary."""
        mask = detect(pre_image, post_image)
        return {
            "model": self.model_name,
            "pre_image": str(pre_image),
            "post_image": str(post_image),
            "change_detected": mask.change_detected,
            "change_percentage": mask.change_percentage,
            "changed_pixels": mask.changed_pixels,
            "total_pixels": mask.total_pixels,
            "confidence": float(mask.confidence),
            "compatibility": mask.compatibility,
        }


__all__ = [
    "detect",
    "detect_change",
    "IncompatibleScenesError",
    "ChangeMask",
    "ChangeDetectionModel",
    "change_vqa",
    "ChangeVQAEngine",
    "analyze_change_evidence",
]
