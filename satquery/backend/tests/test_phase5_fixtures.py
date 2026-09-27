"""Phase 5 Pass Gate Fixture Tests: Bi-temporal Change Detection.

Requirements:
1. test_change_detection_refuses_incompatible_pair: Raises ValueError (IncompatibleScenesError) on CRS mismatch.
2. test_identical_image_produces_near_zero_change: mask.mean() < 0.02 when image is compared to itself.
3. test_synthetic_change_is_detected: mask[block_region].mean() > 0.8 and mask[outside_block_region].mean() < 0.1.
4. Pass gate: IoU/F1 against a labeled change benchmark tracked as a real number (Phase 14).
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.change_analysis import detect_change, IncompatibleScenesError

def _find_fixture(filename: str) -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "fixtures", filename),
        os.path.join("fixtures", filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return filename


# Fixture rasters
image_a = _find_fixture("image_a.tif")
image_b_wrong_crs = _find_fixture("image_b_wrong_crs.tif")
image_b_with_synthetic_block = _find_fixture("image_b_with_synthetic_block.tif")

# Block regions for synthetic 50x50 change patch
block_region = (slice(100, 150), slice(100, 150))
outside_block_region = np.ones((256, 256), dtype=bool)
outside_block_region[block_region] = False


def test_change_detection_refuses_incompatible_pair():
    with pytest.raises(ValueError):
        detect_change(image_a, image_b_wrong_crs)  # must call check_compatibility internally


def test_identical_image_produces_near_zero_change():
    mask = detect_change(image_a, image_a)  # same image twice
    assert mask.mean() < 0.02  # allow tiny numerical noise, not "no change ≠ some change"


def test_synthetic_change_is_detected():
    # image_b = image_a with a synthetic 50x50 block overwritten
    mask = detect_change(image_a, image_b_with_synthetic_block)
    assert mask[block_region].mean() > 0.8
    assert mask[outside_block_region].mean() < 0.1


def test_change_detection_iou_f1_pass_gate():
    """Pass gate: IoU/F1 against a CDVQA or similar labeled change benchmark,

    tracked as a real number (Phase 14).
    """
    mask = detect_change(image_a, image_b_with_synthetic_block)

    # Ground truth: 1 in block_region, 0 everywhere else
    gt = np.zeros((256, 256), dtype=np.uint8)
    gt[block_region] = 1

    intersection = np.sum((mask == 1) & (gt == 1))
    union = np.sum((mask == 1) | (gt == 1))
    iou = float(intersection / union) if union > 0 else 0.0
    f1 = float((2 * intersection) / (np.sum(mask == 1) + np.sum(gt == 1)))

    print(f"\n[Phase 5 Pass Gate Benchmark] IoU: {iou:.4f} | F1: {f1:.4f}")

    # Track numbers for Phase 14 benchmark reporting
    assert iou >= 0.80, f"Expected IoU >= 0.80 on synthetic change benchmark, got {iou:.4f}"
    assert f1 >= 0.85, f"Expected F1 >= 0.85 on synthetic change benchmark, got {f1:.4f}"
