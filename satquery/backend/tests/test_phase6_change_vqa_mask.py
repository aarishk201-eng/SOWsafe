import pytest
from unittest.mock import patch
from models.change_analysis import ChangeMask

def monkeypatch_change_detect_to_return_all_zeros():
    pass # We will mock it inline instead.

@patch('models.change_analysis.detect')
def test_change_vqa_actually_uses_the_change_mask(mock_detect_change):
    from models.change_analysis import change_vqa
    import numpy as np
    
    # Deliberately break the change detector's output to return all zeros
    empty_mask = ChangeMask(
        input_array=np.zeros((10, 10), dtype=np.uint8),
        change_detected=False,
        change_percentage=0.0,
        changed_pixels=0,
        total_pixels=100
    )
    mock_detect_change.return_value = empty_mask
    
    import os
    FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    img_a = os.path.join(FIXTURES_DIR, "t1_base.tif")
    img_b = os.path.join(FIXTURES_DIR, "t2_changed.tif")
    
    # We can pass real files so validator succeeds, but the mock will intercept the mask output
    answer = change_vqa(img_a, img_b, "Has the built-up area increased?")
    
    assert "increas" not in answer.lower(), \
        "Change-VQA says 'increased' even when the change mask shows zero change — it's not actually reading the mask."
