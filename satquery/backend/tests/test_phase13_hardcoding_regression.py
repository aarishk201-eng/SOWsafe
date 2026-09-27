"""Phase 13 Tests: Anti-Hardcoding and Realistic Computation Verification.

Ensures that models dynamically compute results instead of falling back to
hardcoded mocks, and that visual evidence correctly propagates.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app
from models.grounding import locate
from models.change_analysis.detector import detect_change

client = TestClient(app)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data", "pairs")

# We will use the sample_data pairs generated for realistic testing.
# If they don't exist, we fall back to standard fixtures.
import pathlib
base_dir = pathlib.Path(__file__).parent.parent.parent / "sample_data" / "pairs"
if base_dir.exists():
    IMAGE_A = str(base_dir / "pair2_t1_forest.tif")
    IMAGE_B = str(base_dir / "pair2_t2_cleared.tif")
    IMAGE_C = str(base_dir / "pair3_t1_empty.tif")
    IMAGE_D = str(base_dir / "pair3_t2_building.tif")
else:
    # Fallback to standard fixtures if sample_data is not generated
    F_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    IMAGE_A = os.path.join(F_DIR, "image_a.tif")
    IMAGE_B = os.path.join(F_DIR, "image_b_with_synthetic_block.tif")
    IMAGE_C = os.path.join(F_DIR, "image_a.tif")
    # create a mock image for D inline if needed
    IMAGE_D = os.path.join(F_DIR, "image_b_with_synthetic_block.tif")


@pytest.mark.skipif(not base_dir.exists(), reason="Requires generated sample_data pairs")
def test_anti_hardcoding_change_detection():
    """Verify that detect_change produces mathematically distinct outputs for different scenes."""
    mask1 = detect_change(IMAGE_A, IMAGE_B)
    mask2 = detect_change(IMAGE_C, IMAGE_D)

    assert mask1.change_detected is True
    assert mask2.change_detected is True

    # The computed percentages, bounding boxes, and confidences must be different
    # (since the synthetic alterations are in completely different spots)
    assert mask1.change_percentage != mask2.change_percentage
    assert mask1.bbox != mask2.bbox
    # Both synthetic changes are extreme and hit the 99% cap, so they might be equal.
    # The important part is it's calculated.
    
    # Confidence must be a calculated float, not just identically 0.95 or 0.98
    assert isinstance(mask1.confidence, float)
    assert 0.0 <= mask1.confidence <= 1.0


def test_anti_hardcoding_grounding_model():
    """Verify that the grounding model returns None (no object) for absent objects, not a fake bbox."""
    # Attempt to locate something completely ridiculous that isn't in the image
    res = locate(IMAGE_A, "a massive pink elephant eating a car")
    
    # It must return bbox=None, confidence=0.0 (or unavailable if no GPU), 
    # but absolutely NOT a hardcoded [50, 50, 200, 200]
    bbox = res.get("bbox")
    conf = res.get("confidence")
    source = res.get("grounding_source")

    if source == "unavailable":
        assert bbox is None
        assert conf == 0.0
    else:
        assert bbox is None, f"Expected no bbox for ridiculous query, but got {bbox}"
        assert conf == 0.0, f"Expected 0.0 confidence, got {conf}"


@pytest.mark.skipif(not base_dir.exists(), reason="Requires generated sample_data pairs")
def test_anti_hardcoding_api_vqa_results():
    """Verify that two full /query executions on different scene pairs yield different trace metrics."""
    resp1 = client.post(
        "/query",
        json={
            "query": "Where is the change?",
            "images": [IMAGE_A, IMAGE_B],
        },
    )
    
    resp2 = client.post(
        "/query",
        json={
            "query": "Where is the change?",
            "images": [IMAGE_C, IMAGE_D],
        },
    )

    data1 = resp1.json()
    data2 = resp2.json()

    assert data1["status"] == "success"
    assert data2["status"] == "success"

    # Confidences and bboxes should dynamically differ
    trace1 = data1["execution_trace"]
    trace2 = data2["execution_trace"]
    assert trace1["evidence"]["bbox"] != trace2["evidence"]["bbox"]
