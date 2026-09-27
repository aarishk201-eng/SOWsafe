"""Phase 7 Pass Gate Tests: Baseline Multi-Modal Evidence Fusion (Optical + SAR).

Pass gate requirements:
1. test_high_agreement_produces_high_fused_confidence:
   fused = fuse_evidence(optical={"prediction":"built_up","confidence":0.82},
                          sar={"prediction":"built_up","confidence":0.89})
   assert fused["confidence"] > 0.8
   assert fused["agreement"] == "high"

2. test_disagreement_produces_low_confidence_not_averaged_high:
   fused = fuse_evidence(optical={"prediction":"built_up","confidence":0.9},
                          sar={"prediction":"vegetation","confidence":0.9})
   assert fused["confidence"] < 0.5  # disagreeing high-confidence sources should NOT average to "high confidence"

Pass gate note: the disagreement test above is the one most fusion implementations get wrong
by naively averaging — make sure yours doesn't.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evidence.confidence import fuse_evidence
from models.cross_modal import optical_analyzer, sar_analyzer, SensorFusionModel


def test_high_agreement_produces_high_fused_confidence():
    fused = fuse_evidence(
        optical={"prediction": "built_up", "confidence": 0.82},
        sar={"prediction": "built_up", "confidence": 0.89},
    )
    assert fused["confidence"] > 0.8
    assert fused["agreement"] == "high"


def test_disagreement_produces_low_confidence_not_averaged_high():
    fused = fuse_evidence(
        optical={"prediction": "built_up", "confidence": 0.9},
        sar={"prediction": "vegetation", "confidence": 0.9},
    )
    assert fused["confidence"] < 0.5  # disagreeing high-confidence sources should NOT average to "high confidence"


def test_high_agreement_across_other_classes():
    """Verifies that high agreement on vegetation and water also produces boosted confidence."""
    fused_veg = fuse_evidence(
        optical={"prediction": "vegetation", "confidence": 0.85},
        sar={"prediction": "vegetation", "confidence": 0.88},
    )
    assert fused_veg["confidence"] > 0.85
    assert fused_veg["agreement"] == "high"
    assert fused_veg["sensors_agree"] is True

    fused_water = fuse_evidence(
        optical={"prediction": "water", "confidence": 0.92},
        sar={"prediction": "water", "confidence": 0.94},
    )
    assert fused_water["confidence"] > 0.90
    assert fused_water["agreement"] == "high"
    assert fused_water["sensors_agree"] is True


def test_moderate_disagreement_suppresses_confidence():
    """Verifies that moderate contradictory predictions also collapse confidence below 0.5."""
    fused = fuse_evidence(
        optical={"prediction": "water", "confidence": 0.75},
        sar={"prediction": "bare_soil", "confidence": 0.70},
    )
    assert fused["confidence"] < 0.5
    assert fused["agreement"] == "low"
    assert fused["sensors_agree"] is False
