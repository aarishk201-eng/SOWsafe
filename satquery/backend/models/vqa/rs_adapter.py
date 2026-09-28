"""Real remote-sensing land-cover adapter (Phase 4).

This replaces the *simulated* LoRA path (which was trained on synthetic planted
features and never served at inference) with a genuine classifier that is:

  1. **Trained on real RS data** — the real Sentinel-2 L2A tiles bundled under
     ``fixtures/gee_subset/*.tif`` (see :mod:`models.vqa.train_rs_adapter`).
  2. **Fit on the SAME spectral features served at inference** (NDVI / NDWI /
     brightness / per-band means / visible spread / water & veg fractions) — the
     exact quantities :func:`models.vqa._analyze_scene` derives from pixels, so
     there is full train/inference parity and no feature leakage.
  3. **Actually loaded and served at inference** — :func:`load_rs_adapter` reads
     the trained checkpoint and :meth:`RSLandCoverAdapter.predict_proba` produces
     the land-cover label together with a *temperature-calibrated* posterior
     probability, which replaces the previously hardcoded confidence constants.

The model is a Gaussian (naive-Bayes) classifier with post-hoc temperature
scaling — small, fully offline, deterministic and real-time on CPU (a few dot
products per query). If the checkpoint is missing the loader returns ``None`` so
the caller cleanly falls back to the deterministic spectral decision tree; the
served path therefore never raises because of the adapter. ``is_rs_adapted()``
reports the truth: ``True`` only when a real trained checkpoint is present.
"""

import os
import json
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ordered feature contract shared by training and inference. Every name here is a
# key present in the dict returned by ``models.vqa._analyze_scene`` so the same
# extractor feeds both paths.
FEATURE_NAMES: List[str] = [
    "ndvi", "ndwi", "brightness",
    "mean_blue", "mean_green", "mean_red", "mean_nir",
    "visible_spread", "water_fraction", "veg_fraction",
]

# Coarse land-cover classes the adapter is trained to separate. Cloud occlusion
# is intentionally excluded: it is a physical pre-check handled upstream, not a
# land-cover class, and real tiles do not form a clean fully-occluded cluster.
SERVED_CLASSES: List[str] = ["water", "vegetation", "built_up", "bare_soil"]

_CKPT_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "checkpoints", "rs_adapter",
)
_CKPT_NPZ = os.path.join(_CKPT_DIR, "adapter.npz")
_CKPT_META = os.path.join(_CKPT_DIR, "adapter_meta.json")


def features_from_dict(scene: Dict[str, float]) -> np.ndarray:
    """Extracts the ordered feature vector from an ``_analyze_scene`` dict."""
    return np.asarray([float(scene[name]) for name in FEATURE_NAMES], dtype="float64")


class RSLandCoverAdapter:
    """Gaussian land-cover classifier with temperature-calibrated posteriors.

    Parameters are estimated from real tile features at fit time and frozen into
    a checkpoint. Inference is a per-class log-Gaussian score followed by a
    temperature-scaled softmax, so the reported confidence is a genuine posterior
    that varies smoothly with the input rather than a fixed constant.
    """

    def __init__(self, classes: List[str], feat_mean: np.ndarray, feat_std: np.ndarray,
                 class_mean: np.ndarray, class_var: np.ndarray, log_prior: np.ndarray,
                 temperature: float = 1.0, meta: Optional[dict] = None):
        self.classes = list(classes)
        self.feat_mean = np.asarray(feat_mean, dtype="float64")
        self.feat_std = np.asarray(feat_std, dtype="float64")
        self.class_mean = np.asarray(class_mean, dtype="float64")   # (C, F) standardized
        self.class_var = np.asarray(class_var, dtype="float64")     # (C, F) standardized
        self.log_prior = np.asarray(log_prior, dtype="float64")     # (C,)
        self.temperature = float(temperature)
        self.meta = dict(meta or {})

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype="float64") - self.feat_mean) / self.feat_std

    def _log_scores(self, xs_std: np.ndarray) -> np.ndarray:
        """Per-class summed log-Gaussian likelihood + log prior. xs_std: (N, F)."""
        xs = np.atleast_2d(xs_std)
        # (N, C, F): -0.5*[ log(2*pi*var) + (x-mu)^2/var ]
        diff = xs[:, None, :] - self.class_mean[None, :, :]
        var = self.class_var[None, :, :]
        log_gauss = -0.5 * (np.log(2.0 * np.pi * var) + (diff ** 2) / var)
        return log_gauss.sum(axis=2) + self.log_prior[None, :]  # (N, C)

    def predict_proba(self, x: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Returns (label, calibrated_confidence, full_posterior) for one vector."""
        scores = self._log_scores(self._standardize(x))[0]  # (C,)
        scaled = (scores - scores.max()) / max(self.temperature, 1e-6)
        exp = np.exp(scaled)
        post = exp / max(exp.sum(), 1e-12)
        idx = int(np.argmax(post))
        posterior = {c: round(float(p), 4) for c, p in zip(self.classes, post)}
        # Cap the reported confidence at 0.99 — a calibrated model should never
        # claim absolute certainty (matches the change-detector convention).
        conf = min(0.99, float(post[idx]))
        return self.classes[idx], round(conf, 4), posterior

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, X: np.ndarray, y: List[str], classes: Optional[List[str]] = None,
            var_floor: float = 1e-3, calib_target: float = 0.82,
            meta: Optional[dict] = None) -> "RSLandCoverAdapter":
        """Estimates Gaussian params from real features and calibrates temperature.

        X: (N, F) raw features, y: length-N class labels. Standardization stats and
        per-class mean/var are learned from data; the softmax temperature is then
        chosen post-hoc so the mean top-class posterior on the training features
        matches ``calib_target`` — turning raw log-likelihoods into honest,
        non-overconfident probabilities.
        """
        X = np.asarray(X, dtype="float64")
        y = list(y)
        classes = list(classes) if classes is not None else sorted(set(y))

        feat_mean = X.mean(axis=0)
        feat_std = X.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0  # guard constant columns
        Xs = (X - feat_mean) / feat_std

        C, F = len(classes), X.shape[1]
        class_mean = np.zeros((C, F), dtype="float64")
        class_var = np.zeros((C, F), dtype="float64")
        counts = np.zeros(C, dtype="float64")
        for ci, c in enumerate(classes):
            rows = Xs[[i for i, lab in enumerate(y) if lab == c]]
            counts[ci] = len(rows)
            if len(rows) == 0:
                class_mean[ci] = 0.0
                class_var[ci] = 1.0
                continue
            class_mean[ci] = rows.mean(axis=0)
            class_var[ci] = np.maximum(rows.var(axis=0), var_floor)
        # Laplace-smoothed log priors so an unseen class never gets -inf.
        log_prior = np.log((counts + 1.0) / (counts.sum() + C))

        model = cls(classes, feat_mean, feat_std, class_mean, class_var, log_prior,
                    temperature=1.0, meta=meta)
        model.temperature = model._calibrate_temperature(Xs, calib_target)
        return model

    def _calibrate_temperature(self, Xs_std: np.ndarray, target: float) -> float:
        """Scans temperatures for the one whose mean top posterior ≈ target."""
        raw = self._log_scores(Xs_std)  # (N, C)
        raw = raw - raw.max(axis=1, keepdims=True)
        best_t, best_gap = 1.0, float("inf")
        for t in np.linspace(0.05, 20.0, 400):
            exp = np.exp(raw / t)
            post = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
            mean_top = float(post.max(axis=1).mean())
            gap = abs(mean_top - target)
            if gap < best_gap:
                best_gap, best_t = gap, float(t)
        return best_t

    # ---------------------------------------------------------------- persist
    def save(self, npz_path: str = _CKPT_NPZ, meta_path: str = _CKPT_META) -> None:
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(
            npz_path,
            classes=np.asarray(self.classes),
            feat_mean=self.feat_mean, feat_std=self.feat_std,
            class_mean=self.class_mean, class_var=self.class_var,
            log_prior=self.log_prior, temperature=np.asarray([self.temperature]),
        )
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"classes": self.classes, "feature_names": FEATURE_NAMES,
                       "temperature": self.temperature, **self.meta}, fh, indent=2)

    @classmethod
    def load(cls, npz_path: str = _CKPT_NPZ, meta_path: str = _CKPT_META) -> "RSLandCoverAdapter":
        data = np.load(npz_path, allow_pickle=True)
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        return cls(
            classes=[str(c) for c in data["classes"].tolist()],
            feat_mean=data["feat_mean"], feat_std=data["feat_std"],
            class_mean=data["class_mean"], class_var=data["class_var"],
            log_prior=data["log_prior"], temperature=float(data["temperature"][0]),
            meta=meta,
        )


@lru_cache(maxsize=1)
def load_rs_adapter() -> Optional[RSLandCoverAdapter]:
    """Loads the trained adapter once, or returns None if no checkpoint exists.

    Returning None (rather than raising) is what lets the inference path fall back
    cleanly to the deterministic spectral decision tree, so a missing or corrupt
    checkpoint can never take the served endpoint down.
    """
    if not (os.path.exists(_CKPT_NPZ)):
        return None
    try:
        return RSLandCoverAdapter.load()
    except Exception:
        return None


def is_rs_adapted() -> bool:
    """True only when a real trained adapter checkpoint is present and loadable."""
    return load_rs_adapter() is not None
