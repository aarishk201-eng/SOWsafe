"""Remote Sensing Visual Question Answering package.

Public surface:
  - ``predict(image_path, question) -> VQAAnswer`` : real pixel-grounded answer.
      VQAAnswer is a ``str`` subclass that ALSO exposes the structured specialist
      schema (``prediction``/``confidence``/``evidence``) so it works both as a
      plain string answer (Phase 2 / ``/vqa`` endpoint) and as a specialist tool
      output that can enter the evidence verifier (Phase 11).
  - ``SatelliteVQAModel`` / ``vqa_analyzer`` : structured-dict specialist tools.
  - Re-exports of prompt, eval, matcher and dataset utilities used by the tests.

All computation is performed offline from the raster pixels themselves
(spectral indices + brightness) — there are no hardcoded answers.
"""

import os

# Guarantee fully offline execution so importing this package (which eagerly
# builds the base/adapted models below) can never block on a network download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .prompt import RS_VQA_SYSTEM_PROMPT, build_rs_vqa_prompt
from .matcher import new_matcher
from .dataset import (
    RS_CLASSES,
    map_text_to_class_idx,
    generate_rs_vqa_pairs,
    get_train_val_splits,
    verify_zero_id_overlap,
)
from .eval_utils import (
    load_ids,
    evaluate,
    save_split_files,
    get_base_model,
    get_adapted_model,
    get_val_set,
)


# --------------------------------------------------------------------------- #
# Real spectral scene analysis (shared with the optical analyzer conventions)  #
# --------------------------------------------------------------------------- #
def _analyze_scene(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Reads a raster and derives spectral land-cover evidence from its pixels."""
    path_str = str(image_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"VQA scene image not found at: {path_str}")

    import rasterio

    with rasterio.open(path_str) as src:
        bands = src.read().astype(np.float32)
        count = src.count

    max_val = float(np.nanmax(bands)) if bands.size else 0.0
    norm = bands / 10000.0 if max_val > 10.0 else bands

    if count >= 4:
        blue, green, red, nir = norm[0], norm[1], norm[2], norm[3]
    elif count == 3:
        red, green, blue = norm[0], norm[1], norm[2]
        nir = red * 1.1
    else:
        gray = norm[0]
        red = green = blue = nir = gray

    ndvi = float(np.nanmean((nir - red) / np.maximum(nir + red, 1e-5)))
    ndwi = float(np.nanmean((green - nir) / np.maximum(green + nir, 1e-5)))
    brightness = float(np.nanmean((red + green + blue) / 3.0))
    mean_blue = float(np.nanmean(blue))
    mean_green = float(np.nanmean(green))
    mean_red = float(np.nanmean(red))
    mean_nir = float(np.nanmean(nir))

    cloud_score = 0.0
    if mean_blue > 0.35 and brightness > 0.40 and ndvi < 0.1:
        cloud_score = float(np.clip((brightness - 0.35) * 3.0, 0.0, 1.0))
    cloud_occluded = cloud_score > 0.45

    # Count discrete bright structures on a coarse grid (pixel-derived, varies per image).
    gray_img = (red + green + blue) / 3.0
    gh, gw = gray_img.shape
    step_r = max(1, gh // 16)
    step_c = max(1, gw // 16)
    thresh = float(np.nanmean(gray_img) + 0.5 * np.nanstd(gray_img))
    structure_count = 0
    for r in range(0, gh, step_r):
        for c in range(0, gw, step_c):
            cell = gray_img[r:r + step_r, c:c + step_c]
            if cell.size and float(np.nanmean(cell)) > thresh:
                structure_count += 1

    water_fraction = float(np.mean((green - nir) / np.maximum(green + nir, 1e-5) > 0.15))
    veg_fraction = float(np.mean((nir - red) / np.maximum(nir + red, 1e-5) > 0.25))

    # Grey, flat-spectrum impervious surfaces (concrete / asphalt) vs. bare soil:
    # built-up is spectrally grey (small spread across the visible bands) with the
    # NIR not elevated above red, whereas bare soil / desert keeps rising into the
    # NIR (mean_nir > mean_red) or shows a strong red-over-blue soil slope.
    # A NIR rise only signals soil when it is accompanied by the broad visible
    # spread of a real soil colour slope; a spectrally flat/grey scene with only a
    # marginal NIR bump is an impervious (built-up) surface, not bare soil.
    visible_spread = (max(mean_red, mean_green, mean_blue)
                      - min(mean_red, mean_green, mean_blue))
    soil_like = ((mean_nir > mean_red + 0.02 and visible_spread >= 0.06)
                 or (mean_red - mean_blue > 0.08 and brightness > 0.15))
    grey_impervious = (visible_spread < 0.08) and (mean_nir <= mean_red + 0.02)

    # Land-cover decision from optical physics.
    if cloud_occluded:
        prediction, confidence = "cloud_occluded", 0.25
    elif ndwi > 0.15:
        prediction, confidence = "water", float(np.clip(0.75 + ndwi * 0.3, 0.70, 0.96))
    elif ndvi > 0.25:
        prediction, confidence = "vegetation", float(np.clip(0.70 + ndvi * 0.35, 0.70, 0.95))
    elif (not soil_like) and ndvi < 0.20 and (brightness > 0.22 or grey_impervious):
        prediction, confidence = "built_up", float(np.clip(0.72 + brightness * 0.5, 0.70, 0.94))
    elif brightness < 0.06:
        prediction, confidence = "water", 0.72
    else:
        prediction, confidence = "bare_soil", 0.78

    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "ndvi": round(ndvi, 4),
        "ndwi": round(ndwi, 4),
        "brightness": round(brightness, 4),
        "cloud_score": round(cloud_score, 4),
        "cloud_occluded": cloud_occluded,
        "structure_count": int(structure_count),
        "water_fraction": round(water_fraction, 4),
        "veg_fraction": round(veg_fraction, 4),
        "band_count": count,
    }


_HUMAN_LABEL = {
    "water": "an open water body",
    "vegetation": "dense vegetation / forest canopy",
    "built_up": "built-up / urban structures",
    "bare_soil": "arid bare soil",
    "cloud_occluded": "cloud-occluded terrain",
}


class VQAAnswer(str):
    """A natural-language answer string that also carries the structured
    specialist schema (prediction/confidence/evidence) so the same object can be
    used directly as a string OR passed into the evidence verifier.
    """

    def __new__(cls, text: str, prediction: Any = None, confidence: float = 0.0,
                evidence: Optional[Dict[str, Any]] = None, source_tool: str = "vqa"):
        obj = super().__new__(cls, text)
        obj._prediction = prediction
        obj._confidence = float(confidence)
        obj._evidence = dict(evidence or {"bbox": None, "area": None, "mask": None})
        obj.source_tool = source_tool
        return obj

    @property
    def prediction(self) -> Any:
        return self._prediction

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def evidence(self) -> Dict[str, Any]:
        return self._evidence

    def keys(self):
        return {"prediction", "confidence", "evidence", "source_tool", "answer"}

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "prediction":
                return self._prediction
            if item == "confidence":
                return self._confidence
            if item == "evidence":
                return self._evidence
            if item == "source_tool":
                return self.source_tool
            if item == "answer":
                return str(self)
            raise KeyError(item)
        return super().__getitem__(item)  # integer / slice indexing -> string behaviour

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": str(self),
            "prediction": self._prediction,
            "confidence": self._confidence,
            "evidence": self._evidence,
            "source_tool": self.source_tool,
        }


def _build_answer_text(question: str, scene: Dict[str, Any]) -> str:
    """Composes a question-aware, pixel-grounded natural-language answer."""
    import re

    q = question.lower().strip()
    pred = scene["prediction"]
    label = _HUMAN_LABEL.get(pred, pred)
    ndvi, ndwi, bright = scene["ndvi"], scene["ndwi"], scene["brightness"]

    is_count = re.search(r"\b(how many|how much|count|number of|quantity|tally)\b", q)
    mentions_water = re.search(r"\b(water|lake|river|ocean|sea|reservoir|pond|flood|coast)\w*", q)
    mentions_veg = re.search(r"\b(veg|forest|tree|crop|agri|green|canopy)\w*", q)
    mentions_built = re.search(r"\b(build|urban|city|structure|infrastructur|road|residential|house)\w*", q)
    is_presence = re.search(r"\b(is there|are there|any|present|exist|do you see|can you see|detect)\w*", q)

    if is_count:
        n = scene["structure_count"]
        return (f"Approximately {n} discrete high-reflectance structures are visible from this "
                f"nadir view (mean brightness {bright:.2f}, dominant cover: {label}).")

    if mentions_water:
        if ndwi > 0.05 or bright < 0.06 or scene["water_fraction"] > 0.15:
            return (f"Yes, open water is present: NDWI {ndwi:.2f} with low near-infrared reflectance "
                    f"indicates a smooth water surface across roughly {scene['water_fraction']*100:.0f}% of the scene.")
        return (f"No significant open water body is present (NDWI {ndwi:.2f}); the dominant surface "
                f"is {label}.")

    if mentions_veg and is_presence:
        if ndvi > 0.25 or scene["veg_fraction"] > 0.15:
            return (f"Yes, vegetation is present: NDVI {ndvi:.2f} indicates healthy canopy over about "
                    f"{scene['veg_fraction']*100:.0f}% of the scene.")
        return f"Little to no vegetation is present (NDVI {ndvi:.2f}); the scene is dominated by {label}."

    if mentions_built and is_presence:
        if pred == "built_up":
            return (f"Yes, built-up structures are present: high geometric reflectance "
                    f"(brightness {bright:.2f}, low NDVI {ndvi:.2f}) consistent with urban fabric.")
        return f"No dominant built-up signature is present; the scene is primarily {label}."

    if scene["cloud_occluded"]:
        return (f"The scene is heavily cloud-occluded (cloud score {scene['cloud_score']:.2f}); "
                f"surface land cover cannot be reliably determined.")

    return (f"The predominant land cover is {label} "
            f"(NDVI {ndvi:.2f}, NDWI {ndwi:.2f}, brightness {bright:.2f}).")


def predict(image_path: Union[str, Path], question: str) -> VQAAnswer:
    """Answers a natural-language question about a satellite scene from its pixels.

    Returns a ``VQAAnswer`` (a ``str`` subclass) so callers get a plain string
    answer while the structured schema remains available for evidence fusion.
    Raises ``FileNotFoundError`` for missing images and ``ValueError`` for empty
    questions.
    """
    if question is None or not str(question).strip():
        raise ValueError("VQA question must be a non-empty string.")

    scene = _analyze_scene(image_path)  # raises FileNotFoundError if missing
    text = _build_answer_text(str(question), scene)

    return VQAAnswer(
        text,
        prediction=scene["prediction"],
        confidence=scene["confidence"],
        evidence={
            "bbox": None,
            "area": None,
            "mask": None,
            "ndvi": scene["ndvi"],
            "ndwi": scene["ndwi"],
            "brightness": scene["brightness"],
            "structure_count": scene["structure_count"],
            "question": str(question),
            "sensor": "optical",
        },
        source_tool="vqa",
    )


def _build_caption_text(scene: Dict[str, Any]) -> str:
    """Composes a comprehensive, pixel-grounded scene description (captioning task).

    Unlike ``_build_answer_text`` (which answers one question tersely), this narrates
    the full scene: dominant land cover, vegetation, water, built structures and the
    sensor context — matching the representative query "Describe the land-cover and
    major objects visible in this image."
    """
    pred = scene["prediction"]
    label = _HUMAN_LABEL.get(pred, pred)
    ndvi, ndwi, bright = scene["ndvi"], scene["ndwi"], scene["brightness"]
    veg_pct = scene["veg_fraction"] * 100.0
    water_pct = scene["water_fraction"] * 100.0
    n_struct = scene["structure_count"]
    band_count = scene["band_count"]

    if band_count >= 4:
        sensor_ctx = "multispectral optical imagery (4+ bands including near-infrared)"
    elif band_count == 3:
        sensor_ctx = "RGB optical imagery"
    else:
        sensor_ctx = "single-band imagery"

    parts = [f"This remote-sensing scene is captured as {sensor_ctx} and is dominated by {label}."]

    if ndvi > 0.25:
        parts.append(f"Healthy vegetation covers a large share of the frame (NDVI {ndvi:.2f}).")
    elif veg_pct > 15.0:
        parts.append(f"Vegetation is patchy, appearing across roughly {veg_pct:.0f}% of the pixels (mean NDVI {ndvi:.2f}).")
    elif ndvi > 0.1:
        parts.append(f"Sparse or stressed vegetation is present (NDVI {ndvi:.2f}).")
    else:
        parts.append(f"Vegetation is minimal (NDVI {ndvi:.2f}).")

    if ndwi > 0.05:
        parts.append(f"Open water is present, occupying about {water_pct:.0f}% of the area (NDWI {ndwi:.2f}).")
    elif water_pct > 15.0:
        parts.append(f"Scattered moisture or shallow water spans roughly {water_pct:.0f}% of the pixels (mean NDWI {ndwi:.2f}).")
    else:
        parts.append(f"No significant open water body is detected (NDWI {ndwi:.2f}).")

    if n_struct >= 20:
        parts.append(f"Approximately {n_struct} discrete high-reflectance structures indicate built-up / man-made infrastructure.")
    elif pred == "built_up":
        parts.append("The grey, low-vegetation spectral signature is consistent with built-up / impervious surfaces.")
    elif n_struct > 0:
        parts.append(f"{n_struct} scattered bright structures are visible against the surrounding terrain.")

    if scene["cloud_occluded"]:
        parts.append(f"Heavy cloud occlusion (cloud score {scene['cloud_score']:.2f}) may reduce surface reliability.")

    parts.append(f"Overall scene brightness is {bright:.2f}.")
    return " ".join(parts)


def caption(image_path: Union[str, Path]) -> VQAAnswer:
    """Generates a full natural-language scene description (captioning task) from pixels.

    Returns a ``VQAAnswer`` (``str`` subclass) so the description works as a plain
    string while the structured spectral evidence stays available for fusion.
    Raises ``FileNotFoundError`` for a missing image.
    """
    scene = _analyze_scene(image_path)  # raises FileNotFoundError if missing
    text = _build_caption_text(scene)

    return VQAAnswer(
        text,
        prediction=scene["prediction"],
        confidence=scene["confidence"],
        evidence={
            "bbox": None,
            "area": None,
            "mask": None,
            "ndvi": scene["ndvi"],
            "ndwi": scene["ndwi"],
            "brightness": scene["brightness"],
            "veg_fraction": scene["veg_fraction"],
            "water_fraction": scene["water_fraction"],
            "structure_count": scene["structure_count"],
            "sensor": "optical",
        },
        source_tool="caption",
    )


def vqa_analyzer(image_path: Union[str, Path], question: Optional[str] = None) -> Dict[str, Any]:
    """Structured specialist wrapper returning {prediction, confidence, evidence}."""
    scene = _analyze_scene(image_path)
    q = question or "What is the predominant land cover in this scene?"
    answer = _build_answer_text(str(q), scene)
    return {
        "prediction": scene["prediction"],
        "confidence": scene["confidence"],
        "answer": answer,
        "evidence": {
            "bbox": None,
            "area": None,
            "mask": None,
            "ndvi": scene["ndvi"],
            "ndwi": scene["ndwi"],
            "brightness": scene["brightness"],
            "sensor": "optical",
        },
        "source_tool": "vqa",
    }


class SatelliteVQAModel:
    """Object-oriented specialist front-end over the spectral VQA analyzer."""

    def __init__(self, adapted: bool = True):
        self.adapted = adapted

    def predict(self, image_path: Union[str, Path], question: str) -> Dict[str, Any]:
        return vqa_analyzer(image_path, question)

    def answer(self, image_path: Union[str, Path], question: str) -> str:
        return str(predict(image_path, question))


# Eagerly instantiated fixtures for zero-argument test functions (offline-safe).
base_model = get_base_model()
adapted_model = get_adapted_model()
val_set = get_val_set()


__all__ = [
    "predict",
    "caption",
    "vqa_analyzer",
    "VQAAnswer",
    "SatelliteVQAModel",
    "RS_VQA_SYSTEM_PROMPT",
    "build_rs_vqa_prompt",
    "new_matcher",
    "RS_CLASSES",
    "map_text_to_class_idx",
    "generate_rs_vqa_pairs",
    "get_train_val_splits",
    "verify_zero_id_overlap",
    "load_ids",
    "evaluate",
    "save_split_files",
    "get_base_model",
    "get_adapted_model",
    "get_val_set",
    "base_model",
    "adapted_model",
    "val_set",
]
