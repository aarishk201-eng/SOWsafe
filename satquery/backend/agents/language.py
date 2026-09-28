"""Multilingual language detection + embedding intent scoring for SatQuery.

Requirement PS 26167 #5: the assistant must understand queries in **English,
Hindi (Devanagari), and Hinglish (romanized Hindi)** — in real time, offline, and
without any hardcoded answer table.

This module provides two real, computed capabilities:

1. :func:`detect_language` — script/lexicon based detection returning
   ``"en" | "hi" | "hinglish"``. Devanagari code points => Hindi; a lexicon of
   *distinctively* romanized-Hindi tokens (never ordinary English words) =>
   Hinglish; otherwise English.

2. :func:`score_intent` — embeds the query with the already-cached
   ``paraphrase-multilingual-MiniLM-L12-v2`` sentence-transformer and scores it,
   by real cosine similarity, against per-intent English **anchor phrases**.
   Because the model maps English, Hindi and Hinglish into one shared vector
   space, a Hindi or Hinglish query is matched cross-lingually to the right
   task with no translation step and no network access.

The model and its anchor embeddings are computed once and memoized, so per-query
scoring is a single forward pass (real time on CPU). Everything runs offline via
the same ``local_files_only`` / ``*_OFFLINE`` guards used elsewhere in the app.
"""

import os
import re
from functools import lru_cache
from typing import List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Multilingual embedder already present in the local HF cache alongside
# all-MiniLM-L6-v2. Embeds en/hi/hinglish into one space -> offline, no download.
_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"

# Devanagari Unicode block -> unambiguous Hindi.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Distinctively romanized-Hindi tokens. Deliberately excludes words that are also
# ordinary English ("is", "me", "par", "se", "the", "in") so an English query is
# never misdetected as Hinglish (which would divert it off the locked English
# router fast-path). Matched on whole tokens, not substrings.
_HINGLISH_MARKERS = {
    "hai", "hain", "kya", "kaha", "kahan", "kaun", "kaunsa", "kitne", "kitni",
    "kitna", "mein", "dikhao", "dikha", "dikhaao", "batao", "bataye", "bataiye",
    "karo", "kariye", "raha", "rahi", "rahe", "nahi", "nahin", "kaise", "kaisa",
    "kaisi", "dhundo", "dhoondo", "khojo", "yahan", "wahan", "paani", "pani",
    "jal", "nadi", "jheel", "samundar", "jungle", "jangal", "ped", "khet",
    "fasal", "hariyali", "shahar", "imarat", "sadak", "makan", "badlaav",
    "badlav", "badla", "parivartan", "antar", "kshetra", "chhavi", "tasveer",
}

# Per-intent English anchor phrases. Cross-lingual similarity against these
# decides the task for non-English queries. The five tasks are the router's
# fixed set: vqa / grounding / change / change_vqa / cross_modal_analysis.
INTENT_ANCHORS = {
    "vqa": [
        "what is in this satellite image",
        "describe the land cover in this scene",
        "what type of terrain is shown here",
        "identify the dominant surface in the image",
        "how many structures are visible in this scene",
        "is this area urban or agricultural",
        "what does this satellite picture show",
    ],
    "grounding": [
        "where is the water body located in the image",
        "locate the building in this scene",
        "find the bounding box of the object",
        "highlight the region of interest",
        "pinpoint the exact location in the image",
        "show the coordinates of the field",
    ],
    "change": [
        "detect the change between the two images",
        "compute the change mask for the image pair",
        "generate a difference map between the two dates",
        "run change detection on the before and after scenes",
    ],
    "change_vqa": [
        "how has the area changed over time",
        "did deforestation happen between the two dates",
        "has the built up area increased since before",
        "what changed between the earlier and later image",
        "how much did the urban region expand",
    ],
    "cross_modal_analysis": [
        "fuse the optical and radar imagery together",
        "combine SAR and optical satellite data",
        "all weather multimodal terrain analysis",
        "cross sensor fusion of radar and optical evidence",
    ],
}


def detect_language(query: Optional[str]) -> str:
    """Detects the query language as ``"en"``, ``"hi"`` or ``"hinglish"``.

    Devanagari characters => Hindi. Otherwise, whole-token overlap with the
    romanized-Hindi marker lexicon => Hinglish. Otherwise English.
    """
    if not query or not str(query).strip():
        return "en"
    text = str(query)
    if _DEVANAGARI.search(text):
        return "hi"
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    if tokens & _HINGLISH_MARKERS:
        return "hinglish"
    return "en"


@lru_cache(maxsize=1)
def _get_model():
    """Loads the multilingual sentence-transformer from the local cache once."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_ID)


@lru_cache(maxsize=1)
def _anchor_matrix():
    """Encodes every intent anchor once. Returns (intent_labels, matrix[N,D])."""
    import numpy as np

    model = _get_model()
    labels: List[str] = []
    phrases: List[str] = []
    for intent, anchors in INTENT_ANCHORS.items():
        for phrase in anchors:
            labels.append(intent)
            phrases.append(phrase)
    emb = model.encode(phrases, normalize_embeddings=True)
    return labels, np.asarray(emb, dtype="float32")


def score_intent(query: Optional[str]) -> Optional[Tuple[str, float]]:
    """Returns ``(task, cosine_similarity)`` for the nearest intent, or ``None``.

    Similarity is the max cosine over an intent's anchor phrases (normalized
    embeddings => dot product is cosine). ``None`` is returned only when the
    query is empty or the embedder is unavailable, so callers can fall back to
    the deterministic router.
    """
    q = (query or "").strip()
    if not q:
        return None
    try:
        import numpy as np

        model = _get_model()
        labels, matrix = _anchor_matrix()
        q_emb = np.asarray(model.encode([q], normalize_embeddings=True)[0], dtype="float32")
        sims = matrix @ q_emb  # cosine similarity per anchor
        best: dict = {}
        for label, sim in zip(labels, sims):
            s = float(sim)
            if label not in best or s > best[label]:
                best[label] = s
        task = max(best, key=best.get)
        return task, round(best[task], 4)
    except Exception:
        return None


__all__ = ["detect_language", "score_intent", "INTENT_ANCHORS"]
