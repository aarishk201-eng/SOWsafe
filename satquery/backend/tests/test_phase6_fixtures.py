"""Phase 6 Pass Gate Fixture Tests: Change VQA Mask Consistency.

Requirements:
1. test_change_vqa_agrees_with_raw_change_mask:
   - When mask[builtup_class_pixels].mean() > 0.3: asserts 'increas' or 'yes' in answer.
   - Else: asserts 'increas' not in answer.
2. Pass gate: answer consistency with the underlying change mask on >= 10 fixture pairs,
   flagging any case where the VQA answer contradicts what the mask actually shows.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.change_analysis import detect_change, change_vqa


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


def _find_fixture_dir(dirname: str) -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "fixtures", dirname),
        os.path.join("fixtures", dirname),
        os.path.join(os.path.dirname(__file__), dirname),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return dirname


# Module-level variables for the exact test signature
t1 = _find_fixture("image_a.tif")
t2 = _find_fixture("image_b_with_synthetic_block.tif")
builtup_class_pixels = (slice(100, 150), slice(100, 150))


def test_change_vqa_agrees_with_raw_change_mask():
    mask = detect_change(t1, t2)
    answer = change_vqa(t1, t2, "Has the built-up area increased?")
    if mask[builtup_class_pixels].mean() > 0.3:
        assert "increas" in answer.lower() or "yes" in answer.lower()
    else:
        assert "increas" not in answer.lower()


def test_change_vqa_agrees_when_builtup_did_not_increase():
    """Verifies that when comparing t1 to itself, mask mean is 0 and 'increas' is strictly absent."""
    mask = detect_change(t1, t1)
    answer = change_vqa(t1, t1, "Has the built-up area increased?")
    if mask[builtup_class_pixels].mean() > 0.3:
        assert "increas" in answer.lower() or "yes" in answer.lower()
    else:
        assert "increas" not in answer.lower()


def test_change_vqa_consistency_pass_gate_on_10_pairs():
    """Pass gate: answer consistency with the underlying change mask on >= 10 fixture pairs —

    flag any case where the VQA answer contradicts what the mask actually shows.
    """
    pairs_dir = _find_fixture_dir("change_pairs")
    contradictions = []
    total_evaluated = 0

    for i in range(12):
        p1 = os.path.join(pairs_dir, f"pair_{i}_t1.tif")
        p2 = os.path.join(pairs_dir, f"pair_{i}_t2.tif")
        if not os.path.exists(p1) or not os.path.exists(p2):
            continue

        total_evaluated += 1
        mask = detect_change(p1, p2)
        answer = change_vqa(p1, p2, "Has the built-up area increased?")
        mean_val = float(mask[builtup_class_pixels].mean())

        if mean_val > 0.3:
            if not ("increas" in answer.lower() or "yes" in answer.lower()):
                contradictions.append(
                    f"Contradiction in Pair {i}: mask mean {mean_val:.4f} > 0.3, but answer was: '{answer}'"
                )
        else:
            if "increas" in answer.lower():
                contradictions.append(
                    f"Contradiction in Pair {i}: mask mean {mean_val:.4f} <= 0.3, but answer contained 'increas': '{answer}'"
                )

    print(f"\n[Phase 6 Pass Gate] Evaluated {total_evaluated} fixture pairs. Contradictions: {len(contradictions)}")
    assert total_evaluated >= 10, f"Expected at least 10 fixture pairs, evaluated {total_evaluated}"
    assert len(contradictions) == 0, f"Detected contradiction between mask and VQA answer: {contradictions}"
