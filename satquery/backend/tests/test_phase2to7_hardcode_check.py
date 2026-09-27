"""
Phase 2-7 Combined "Real vs Hardcoded" Test Suite
===================================================
Run this AFTER wiring your actual model functions in the ADAPTERS section below.
Goal: catch stub/hardcoded outputs before you waste time on Phase 14 benchmarking.

Core idea used everywhere in this file:
  Feed the SAME function genuinely DIFFERENT inputs.
  If the output doesn't change at all -> it's hardcoded/stub. FLAG IT.
  If the output changes -> it's at least reading the input. Doesn't prove accuracy,
  only proves it's not a stub. Accuracy is Phase 14's job, not this file's.

HOW TO USE:
1. Fill in the 6 adapter functions in the ADAPTERS section (import your real
   functions there — do NOT change the test functions below them).
2. Run: pytest tests/test_phase2to7_hardcode_check.py -v
3. Any FAIL here means: stop, fix that model's wiring, THEN move to Phase 13/14.
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# FIXTURES — synthetic GeoTIFFs with genuinely different content
# ---------------------------------------------------------------------------
def _write_tif(path, pattern="random", width=256, height=256, bands=4,
               crs="EPSG:32643", res=10.0, origin=(500000, 2200000), seed=0):
    rng = np.random.default_rng(seed)
    transform = from_origin(origin[0], origin[1], res, res)
    if pattern == "random":
        data = (rng.random((bands, height, width)) * 3000).astype("uint16")
    elif pattern == "water":  # low reflectance dominant = water-like signature
        data = np.full((bands, height, width), 200, dtype="uint16")
        data += (rng.random((bands, height, width)) * 100).astype("uint16")  # small texture noise only
    elif pattern == "builtup":  # high reflectance dominant = built-up-like
        data = np.full((bands, height, width), 5500, dtype="uint16")
        data += (rng.random((bands, height, width)) * 200).astype("uint16")
    elif pattern == "empty":  # near-uniform, nothing to find
        data = np.full((bands, height, width), 1000, dtype="uint16")
    else:
        raise ValueError(pattern)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                        count=bands, dtype="uint16", crs=crs, transform=transform) as dst:
        dst.write(data)
    return path


@pytest.fixture
def img_water(tmp_path):
    return _write_tif(str(tmp_path / "water.tif"), pattern="water", seed=1)


@pytest.fixture
def img_builtup(tmp_path):
    return _write_tif(str(tmp_path / "builtup.tif"), pattern="builtup", seed=2)


@pytest.fixture
def img_empty(tmp_path):
    return _write_tif(str(tmp_path / "empty.tif"), pattern="empty", seed=3)


@pytest.fixture
def img_random_a(tmp_path):
    return _write_tif(str(tmp_path / "rand_a.tif"), pattern="random", seed=10)


@pytest.fixture
def img_random_b(tmp_path):
    return _write_tif(str(tmp_path / "rand_b.tif"), pattern="random", seed=99)


@pytest.fixture
def img_synthetic_change_pair(tmp_path):
    """Same base scene twice, but t2 has a 60x60 block overwritten -> real localized change."""
    t1 = _write_tif(str(tmp_path / "t1.tif"), pattern="random", seed=5)
    rng = np.random.default_rng(5)
    with rasterio.open(t1) as src:
        data = src.read()
        profile = src.profile
    data2 = data.copy()
    data2[:, 80:140, 80:140] = 6000  # inject a hard, unmistakable change block
    t2 = str(tmp_path / "t2.tif")
    with rasterio.open(t2, "w", **profile) as dst:
        dst.write(data2)
    return t1, t2, (80, 140, 80, 140)  # (row0,row1,col0,col1) of the injected change


# ---------------------------------------------------------------------------
# ADAPTERS — plug your real functions in here. Nothing below this needs editing.
# ---------------------------------------------------------------------------
def call_vqa(image_path: str, question: str) -> str:
    """Phase 2. Must return a string answer."""
    from models.vqa import predict as vqa_predict          # <-- adjust import path
    return vqa_predict(image_path, question)


def call_grounding(image_path: str, phrase: str) -> dict:
    """Phase 3. Must return {'bbox': [...], 'confidence': float} (bbox can be None)."""
    from models.grounding import locate                    # <-- adjust import path
    return locate(image_path, phrase)


def call_change_detect(t1_path: str, t2_path: str) -> np.ndarray:
    """Phase 5. Must return a 2D array (H, W) of change scores/probabilities."""
    from models.change_analysis import detect                       # <-- adjust import path
    return detect(t1_path, t2_path)


def call_change_vqa(t1_path: str, t2_path: str, question: str) -> str:
    """Phase 6. Must return a string answer grounded in the change."""
    from models.change_analysis import change_vqa                   # <-- adjust import path
    return change_vqa(t1_path, t2_path, question)


def call_optical_sar_fusion(optical_evidence: dict, sar_evidence: dict) -> dict:
    """Phase 7. Must return {'confidence': float, 'agreement': str, ...}."""
    from evidence.confidence import fuse_evidence           # <-- adjust import path
    return fuse_evidence(optical=optical_evidence, sar=sar_evidence)


def is_rs_adapted() -> bool:
    """
    Phase 4 requirement check (separate from hardcode check).
    Return True only if you actually ran LoRA/QLoRA/fine-tuning on RS data
    and are serving that checkpoint. Return False if Phase 2 is still calling
    a stock general-purpose VLM with no RS adaptation.
    """
    return True   # <-- set this truthfully; don't guess


# ---------------------------------------------------------------------------
# PHASE 2 — VQA: real vs hardcoded
# ---------------------------------------------------------------------------
class TestPhase2VQA:
    def test_answer_changes_across_genuinely_different_images(self, img_water, img_builtup):
        a1 = call_vqa(img_water, "What type of land cover is visible?")
        a2 = call_vqa(img_builtup, "What type of land cover is visible?")
        assert isinstance(a1, str) and isinstance(a2, str)
        assert len(a1) > 0 and len(a2) > 0
        assert a1 != a2, "HARDCODED SUSPECT: identical answer for a water-pattern and a built-up-pattern image."

    def test_answer_changes_when_question_changes(self, img_water):
        a1 = call_vqa(img_water, "What type of land cover is visible?")
        a2 = call_vqa(img_water, "Is there any water present?")
        assert a1 != a2, "HARDCODED SUSPECT: same answer regardless of the question asked."

    def test_two_random_but_different_images_dont_collapse_to_same_string(self, img_random_a, img_random_b):
        a1 = call_vqa(img_random_a, "Describe this scene.")
        a2 = call_vqa(img_random_b, "Describe this scene.")
        # Not a strict requirement they differ (could legitimately be similar-looking random noise),
        # but flag it loudly so a human checks manually rather than assuming it's fine.
        if a1 == a2:
            pytest.xfail("Same answer on two different random images — verify manually this isn't a stub.")


# ---------------------------------------------------------------------------
# PHASE 3 — Grounding: real vs hardcoded
# ---------------------------------------------------------------------------
class TestPhase3Grounding:
    def test_bbox_or_confidence_differs_between_present_and_absent_object(self, img_water, img_empty):
        present = call_grounding(img_water, "water body")
        absent = call_grounding(img_empty, "water body")
        assert present["confidence"] != absent["confidence"], \
            "HARDCODED SUSPECT: identical confidence whether the object is plausibly present or not."

    def test_bbox_stays_within_image_dimensions(self, img_water):
        result = call_grounding(img_water, "water body")
        if result["bbox"] is not None:
            x0, y0, x1, y1 = result["bbox"]
            assert 0 <= x0 < x1 <= 256
            assert 0 <= y0 < y1 <= 256


# ---------------------------------------------------------------------------
# PHASE 5 — Change detection: real vs hardcoded
# ---------------------------------------------------------------------------
class TestPhase5Change:
    def test_identical_image_pair_produces_near_zero_change(self, img_random_a):
        mask = call_change_detect(img_random_a, img_random_a)
        mask = np.asarray(mask)
        assert mask.mean() < 0.05, \
            f"Same image compared to itself shows {mask.mean():.2f} average 'change' — detector isn't reading pixels correctly."

    def test_injected_change_block_is_localized_correctly(self, img_synthetic_change_pair):
        t1, t2, (r0, r1, c0, c1) = img_synthetic_change_pair
        mask = np.asarray(call_change_detect(t1, t2))
        inside = mask[r0:r1, c0:c1].mean()
        # sample a same-size region well outside the injected block
        outside = mask[0:60, 0:60].mean()
        assert inside > outside + 0.2, \
            f"HARDCODED/BROKEN SUSPECT: change score inside injected block ({inside:.2f}) " \
            f"not meaningfully higher than outside ({outside:.2f})."


# ---------------------------------------------------------------------------
# PHASE 6 — Change-VQA: must agree with the actual change mask
# ---------------------------------------------------------------------------
class TestPhase6ChangeVQA:
    def test_change_vqa_says_no_change_for_identical_pair(self, img_random_a):
        answer = call_change_vqa(img_random_a, img_random_a, "Has the built-up area increased?")
        low = answer.lower()
        assert "increas" not in low, \
            f"Change-VQA claims an increase on an identical image pair: '{answer}'"

    def test_change_vqa_detects_injected_change(self, img_synthetic_change_pair):
        t1, t2, _ = img_synthetic_change_pair
        answer = call_change_vqa(t1, t2, "Did anything change between these two images?")
        low = answer.lower()
        assert any(w in low for w in ["chang", "yes", "differ", "increas", "decreas"]), \
            f"Change-VQA missed an obvious injected change: '{answer}'"


# ---------------------------------------------------------------------------
# PHASE 7 — Optical+SAR fusion: agreement vs naive averaging
# ---------------------------------------------------------------------------
class TestPhase7Fusion:
    def test_high_agreement_gives_high_confidence(self):
        fused = call_optical_sar_fusion(
            {"prediction": "built_up", "confidence": 0.85},
            {"prediction": "built_up", "confidence": 0.88},
        )
        assert fused["confidence"] > 0.75

    def test_disagreement_does_not_average_into_false_high_confidence(self):
        fused = call_optical_sar_fusion(
            {"prediction": "built_up", "confidence": 0.9},
            {"prediction": "vegetation", "confidence": 0.9},
        )
        assert fused["confidence"] < 0.5, \
            f"NAIVE-AVERAGING BUG SUSPECT: two disagreeing high-confidence sources fused to {fused['confidence']:.2f}."


# ---------------------------------------------------------------------------
# PHASE 4 — RS-adaptation requirement (separate axis from hardcoding)
# ---------------------------------------------------------------------------
class TestPhase4Requirement:
    def test_rs_adaptation_flag_is_explicit(self):
        adapted = is_rs_adapted()
        if not adapted:
            pytest.xfail(
                "Phase 2 is currently serving a non-RS-adapted general VLM. "
                "This passes the 'not hardcoded' check but FAILS the problem "
                "statement's explicit requirement: adaptation via BigEarthNet "
                "or other RS data is mandatory before final submission."
            )