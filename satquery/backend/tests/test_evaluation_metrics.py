import pytest
import numpy as np

# These will be imported once they are created in the next steps
from models.vqa.matcher import new_matcher
from scripts.evaluate_benchmarks import compute_iou

def test_new_matcher_rejects_clearly_wrong_answers():
    assert new_matcher("water body", "built-up area") is False
    assert new_matcher("dense forest", "arid desert") is False
    assert new_matcher("commercial aircraft", "airport tarmac") is False
    assert new_matcher("solar panels", "forest") is False
    assert new_matcher("parking lot", "water") is False

def test_new_matcher_accepts_clear_paraphrases():
    assert new_matcher("agricultural field", "cultivated agriculture") is True
    assert new_matcher("vegetation", "dense forest cover") is True
    assert new_matcher("urban infrastructure", "built-up area") is True
    assert new_matcher("coastal water", "water body") is True

def test_iou_is_not_computed_from_boolean_flag():
    # guards against regressing back to the change_detected-as-IoU bug
    import inspect
    source = inspect.getsource(compute_iou)
    assert "change_detected" not in source
