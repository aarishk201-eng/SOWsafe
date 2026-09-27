# Implementation Plan: Audit and Remove Hardcoded/Mocked Values in Change VQA Pipeline

## Goal
Audit the `change_vqa` task graph (including `grounding_model`, `change_model`, and `change_vqa_model`) for any hardcoded bounding boxes, artificially inflated confidence scores, and fallback stubs. Replace these with actual computed outputs or honest failure statuses, ensuring the pipeline's behavior reflects genuine analysis of the provided scene pairs.

## Audit Findings

### 1. `models/grounding/__init__.py`
*   **Lines 65-70**: Hardcoded exclusion for phrases like "airplane", "airliner", "swimming pool" which return `bbox: None, confidence: 0.10`.
*   **Lines 75-80**: If the OwlViT weights/processor cannot be loaded (e.g., no GPU/VRAM or offline), it falls back to a mocked bounding box `[100, 100, 150, 150]` and `confidence: 0.85` instead of failing honestly.
*   **Line 103**: Artificial inflation of the real OwlViT confidence: `max(0.75, round(float(confidence), 2))`. This guarantees the confidence never drops below 75% even if the model is extremely uncertain.
*   **Lines 107-112**: **(The Smoking Gun)** If OwlViT runs but finds absolutely no matching objects (`len(results["scores"]) == 0`), instead of returning `None`, it hallucinates a bounding box `[50, 50, 200, 200]` with a fixed `confidence: 0.75`.
*   **Lines 117-121**: Global exception handler swallows inference errors and returns a stubbed `[100, 100, 150, 150]` with `0.85` confidence.

### 2. `models/change_analysis/change_vqa.py`
*   **Lines 293**: `ChangeVQAEngine.answer_query` hardcodes the overall "Audited Answer" confidence to a fixed band chosen by an if/else branch: `0.94 if evidence["change_detected"] else 0.98`. It completely ignores the actual confidence of the underlying evidence.

### 3. `models/change_analysis/detector.py`
*   **Line 68**: The mathematical Change Vector Analysis (CVA) assigns a synthetic confidence heuristic `0.95 if not change_detected else float(min(0.98, max(0.70, 0.70 + change_percentage / 100.0)))`. While CVA is non-probabilistic, this heuristic clamps the confidence to an artificial minimum of 70%.

## Proposed Changes

### [MODIFY] `models/grounding/__init__.py`
*   Remove the "airplane/swimming pool" mock blocks.
*   If `processor` or `model` fails to load, return `bbox: None`, `confidence: 0.0`, and `grounding_source: "unavailable"`.
*   If inference finds zero matching objects, return `bbox: None`, `confidence: 0.0`, instead of `[50, 50, 200, 200]`.
*   Remove the `max(0.75, ...)` inflation; return the exact `float(confidence)` output by OwlViT.
*   Remove the generic try/except mock fallback that hides real crashes.

### [MODIFY] `models/change_analysis/change_vqa.py`
*   Modify `ChangeVQAEngine.answer_query` so it derives its confidence dynamically from the `mask.confidence` (the underlying detector's confidence) rather than a hardcoded `0.94`/`0.98`.

### [MODIFY] `models/change_analysis/detector.py`
*   Update `ChangeMask`'s confidence heuristic so it isn't artificially clamped to a 70% minimum floor, making it more representative of the statistical certainty of the change (e.g. based on standard deviations above the mean rather than a fixed minimum).

### [NEW] `tests/test_phase13_hardcoding_regression.py`
*   Add a test that asserts the output of two different query runs on completely different scene pairs produce definitively distinct `bbox`, `confidence`, and `change_percentage` values.
*   Assert that an empty result from OwlViT returns `bbox: None` instead of `[50, 50, 200, 200]`.

> [!WARNING]
> **User Review Required**: OwlViT currently takes >2GB of RAM/VRAM to load. If it fails to load on your machine due to memory constraints, the new honest fallback will mean Grounding will explicitly return "unavailable" instead of the fake bbox. This is the correct behavior you requested, but I want to ensure you're prepared for the UI to show missing bboxes if your local environment cannot run OwlViT natively.

## Verification Plan
1. Run `pytest tests/test_phase13_hardcoding_regression.py` to prove the outputs dynamically reflect the input data.
2. Run the full existing test suite (`pytest tests/`) to ensure no downstream dependencies break because they relied on the mocks.
3. Manually run the pipeline twice in the frontend UI to demonstrate varying results.
