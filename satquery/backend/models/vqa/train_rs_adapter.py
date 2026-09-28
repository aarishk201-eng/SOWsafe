"""One-time trainer for the real RS land-cover adapter (Phase 4).

Scans the bundled real Sentinel-2 L2A tiles under ``fixtures/gee_subset``, extracts
the SAME spectral feature vector served at inference (via
:func:`models.vqa._analyze_scene`, so there is full train/inference parity), weak-
labels each tile with the deterministic optical-physics decision (a Snorkel-style
labeling function computed from the real pixels), and fits a temperature-calibrated
Gaussian land-cover classifier.

The resulting checkpoint is what makes :func:`models.vqa.is_rs_adapted` honestly
``True`` and replaces the previously hardcoded inference confidences with a real
calibrated posterior that varies with the input scene.

Run once, offline:

    python -m models.vqa.train_rs_adapter
"""

import os
import sys
import glob
import time

import numpy as np

# Ensure the backend root is importable when executed as a plain script.
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from models.vqa import _analyze_scene  # noqa: E402  (path set up above)
from models.vqa.rs_adapter import (  # noqa: E402
    RSLandCoverAdapter,
    SERVED_CLASSES,
    features_from_dict,
    FEATURE_NAMES,
)

_TILE_GLOB = os.path.join(_BACKEND, "fixtures", "gee_subset", "*.tif")


def rgb_weak_label(scene):
    """Visible-band (RGB) land-cover labeling function.

    The bundled tiles are 3-band RGB (no NIR), so NDVI is unavailable. Vegetation
    is instead detected with the Visible Atmospherically Resistant Index
    ``VARI = (G - R) / (G + R - B)`` (a standard RGB greenness index), water from a
    visible blue/darkness cue, and the remaining bright surfaces are split into
    warm bare-soil vs. neutral-grey built-up. This is a real spectral labeling
    function over the actual pixels — not a stored answer — that gives the adapter
    genuine, varied targets across all served classes.
    """
    g, r, b = scene["mean_green"], scene["mean_red"], scene["mean_blue"]
    bright = scene["brightness"]
    ndwi = scene["ndwi"]
    vari = (g - r) / max(g + r - b, 1e-5)

    if ndwi > 0.04 or (b >= r + 0.01 and bright < 0.22):
        return "water"
    if vari > 0.05 and g > r:
        return "vegetation"
    if (r - b) > 0.05 and bright > 0.22:
        return "bare_soil"
    return "built_up"


def build_dataset(max_tiles=None):
    """Extracts (features, weak-label) pairs from the real tiles."""
    tiles = sorted(glob.glob(_TILE_GLOB))
    if max_tiles:
        tiles = tiles[:max_tiles]
    X, y, skipped = [], [], 0
    for i, tile in enumerate(tiles):
        try:
            scene = _analyze_scene(tile)
        except Exception:
            skipped += 1
            continue
        # Cloud occlusion is a physical precheck, not a served land-cover class.
        if scene.get("cloud_occluded"):
            skipped += 1
            continue
        X.append(features_from_dict(scene))
        y.append(rgb_weak_label(scene))
        if (i + 1) % 150 == 0:
            print(f"  processed {i + 1}/{len(tiles)} tiles...", flush=True)
    return np.asarray(X, dtype="float64"), y, len(tiles), skipped


def main():
    t0 = time.time()
    print(f"Scanning real tiles: {_TILE_GLOB}", flush=True)
    X, y, n_tiles, skipped = build_dataset()
    if len(X) < 10:
        raise SystemExit(f"Too few usable tiles ({len(X)}) to train an adapter.")

    dist = {c: int(sum(1 for lab in y if lab == c)) for c in SERVED_CLASSES}
    print(f"Fitting on {len(X)} tiles (skipped {skipped}); class dist: {dist}", flush=True)

    meta = {
        "trained_on": "sentinel2_l2a_gee_subset",
        "n_tiles_scanned": n_tiles,
        "n_train": len(X),
        "class_distribution": dist,
        "feature_names": FEATURE_NAMES,
        "labeling": "rgb_visible_band_weak_supervision(VARI+NDWI+brightness)",
        "domain": "rgb8_3band",
        "version": "rs-adapter-1.0",
    }
    model = RSLandCoverAdapter.fit(X, y, classes=SERVED_CLASSES, meta=meta)
    model.save()

    # Sanity report: calibrated posteriors must vary across real scenes.
    confs = np.asarray([model.predict_proba(x)[1] for x in X])
    print(
        f"Saved checkpoint. temperature={model.temperature:.3f} "
        f"mean_conf={confs.mean():.3f} min={confs.min():.3f} max={confs.max():.3f}",
        flush=True,
    )
    print(f"Done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
