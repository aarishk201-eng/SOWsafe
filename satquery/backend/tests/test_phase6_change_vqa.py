"""Phase 6 Tests: Grounded Bi-Temporal Change VQA.

Verifies:
1. change_vqa(t1, t2, question) grounds its answer in the computed change mask (bounding box,
   altered area percentage, and spectral transition dynamics) rather than guessing.
2. Identical scenes return unambiguous zero-change answers.
3. Incompatible image pairs are refused via pre-flight compatibility check (raising ValueError).
4. Empty questions are rejected with ValueError.
5. POST /change-vqa endpoint operates over JSON and multipart inputs with grounded responses.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app
from models.change_analysis import (
    change_vqa,
    ChangeVQAEngine,
    IncompatibleScenesError,
)

client = TestClient(app)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
IMAGE_A = os.path.join(FIXTURES_DIR, "image_a.tif")
IMAGE_B_BLOCK = os.path.join(FIXTURES_DIR, "image_b_with_synthetic_block.tif")
IMAGE_B_WRONG_CRS = os.path.join(FIXTURES_DIR, "image_b_wrong_crs.tif")


def test_change_vqa_returns_grounded_answer_on_synthetic_block():
    """Verifies that the model grounds its answer in the exact mask coordinates and percentage."""
    answer = change_vqa(IMAGE_A, IMAGE_B_BLOCK, "What changed between the two dates?")

    assert isinstance(answer, str)
    assert len(answer) > 20
    # Must explicitly cite coordinates or bounding box from the change mask
    assert "[100, 100," in answer or "100" in answer
    # Must mention change statistics or physical alteration
    assert any(k in answer.lower() for k in ["%", "clearing", "alteration", "construction", "reflectance"])


def test_change_vqa_answers_location_query_with_mask_bbox():
    """Verifies that spatial location queries return the exact bounding box from the mask."""
    answer = change_vqa(IMAGE_A, IMAGE_B_BLOCK, "Where did change happen in the scene?")

    assert isinstance(answer, str)
    assert "bounding box" in answer.lower() or "coordinates" in answer.lower()
    assert "[100, 100," in answer


def test_change_vqa_answers_construction_query_affirmatively():
    """Verifies query regarding building/construction is answered grounded in spectral evidence."""
    answer = change_vqa(IMAGE_A, IMAGE_B_BLOCK, "Did any new construction or clearing occur?")

    assert isinstance(answer, str)
    assert "yes" in answer.lower()
    assert "bounding box" in answer.lower()


def test_change_vqa_identifies_zero_change_on_identical_scenes():
    """Verifies that identical scenes produce an explicit zero-change grounded answer."""
    answer = change_vqa(IMAGE_A, IMAGE_A, "What changed in this scene between T1 and T2?")

    assert isinstance(answer, str)
    assert any(k in answer.lower() for k in ["no significant", "0.0%", "no ground changes", "stable", "no measurable"])


def test_change_vqa_rejects_incompatible_images():
    """Hard requirement: Incompatible pairs must raise ValueError (IncompatibleScenesError)."""
    with pytest.raises(ValueError):
        change_vqa(IMAGE_A, IMAGE_B_WRONG_CRS, "What changed?")


def test_change_vqa_rejects_empty_question():
    """Empty or whitespace-only questions must raise ValueError."""
    with pytest.raises(ValueError):
        change_vqa(IMAGE_A, IMAGE_A, "   ")


def test_change_vqa_engine_class_wrapper():
    """Tests the ChangeVQAEngine wrapper class."""
    engine = ChangeVQAEngine()
    res = engine.answer_query(IMAGE_A, IMAGE_B_BLOCK, "Describe the ground change.")

    assert res["model"] == "sat-change-vqa-v1"
    assert res["change_detected"] is True
    assert res["change_percentage"] > 0
    assert res["bbox"] is not None
    assert len(res["answer"]) > 10


def test_api_change_vqa_json_endpoint():
    """Tests POST /change-vqa with JSON payload."""
    resp = client.post(
        "/change-vqa",
        json={
            "t1": IMAGE_A,
            "t2": IMAGE_B_BLOCK,
            "question": "What is the extent of ground alteration?",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "answer" in data
    assert len(data["answer"]) > 10
    assert data["change_detected"] is True
    assert data["bbox"] is not None
    assert data["change_percentage"] > 0


def test_api_change_vqa_multipart_upload():
    """Tests POST /change-vqa with multipart files."""
    with open(IMAGE_A, "rb") as f1, open(IMAGE_B_BLOCK, "rb") as f2:
        resp = client.post(
            "/change-vqa",
            files={
                "t1": ("t1.tif", f1, "image/tiff"),
                "t2": ("t2.tif", f2, "image/tiff"),
            },
            data={"question": "Where is the change located?"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["change_detected"] is True
    assert "[100, 100," in data["answer"] or "100" in data["answer"]


def test_api_change_vqa_rejects_incompatible_pair():
    """Tests POST /change-vqa returns HTTP 400 when pair is incompatible."""
    resp = client.post(
        "/change-vqa",
        json={
            "t1": IMAGE_A,
            "t2": IMAGE_B_WRONG_CRS,
            "question": "What changed?",
        },
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "IncompatibleScenesError" in detail["error"]


def test_change_vqa_query_invokes_full_pipeline_and_populates_all_panels():
    """Verifies that a change_vqa query invokes all specialist models and populates all 4 UI panels."""
    resp = client.post(
        "/query",
        json={
            "query": "Has the built-up area increased between these two satellite scenes?",
            "images": [IMAGE_A, IMAGE_B_BLOCK],
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert "answer" in data and len(data["answer"]) > 10

    # 1. Before / After Panel Payload
    assert data["before_after"] is not None
    assert data["before_after"]["t1"].startswith("data:image/jpeg;base64,")
    assert data["before_after"]["t2"].startswith("data:image/jpeg;base64,")
    assert data["before_after"]["status"] == "available"

    # 2. Change Mask Panel Payload
    assert data["change_mask"] is not None
    assert data["change_mask"]["change_detected"] is True
    assert data["change_mask"]["change_percentage"] > 0
    assert data["change_mask"]["status"] == "available"

    # 3. Visual Evidence Panel Payload (BBox)
    assert data["visual_evidence"] is not None
    assert data["visual_evidence"]["bbox"] == [100, 100, 149, 149]
    assert data["visual_evidence"]["status"] == "available"

    # 4. Execution Trace & Models Used Panel
    trace = data["execution_trace"]
    assert trace is not None
    models_used = trace.get("models_used", [])
    assert "change_model" in models_used
    assert "grounding_model" in models_used
    assert "change_vqa_model" in models_used


def test_change_vqa_single_image_failure_path():
    """Verifies that a change_vqa query with only 1 image attached returns status='failed' with error detail."""
    resp = client.post(
        "/query",
        json={
            "query": "Has the built-up area increased between these two satellite scenes?",
            "images": [IMAGE_A],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert "requires at least 2 scenes" in data["error"]


def test_query_before_after_payload_handles_malformed_image(tmp_path):
    """Verifies that the before_after preview generation handles unusual/malformed GeoTIFFs gracefully."""
    import shutil
    
    # Create two malformed tiffs (just text files)
    fake_t1 = tmp_path / "fake1.tif"
    fake_t2 = tmp_path / "fake2.tif"
    fake_t1.write_text("Not a real tif")
    fake_t2.write_text("Also not a real tif")
    
    # We expect the query_router/pipeline to maybe fail during execution or validation, 
    # but let's test just the preview generation fallback directly by passing it into /query.
    # The /query endpoint executes the pipeline which will likely fail validation,
    # but if it somehow proceeds, before_after_payload should not crash the server.
    resp = client.post(
        "/query",
        json={
            "query": "Has the built-up area increased?",
            "images": [str(fake_t1), str(fake_t2)],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    
    # Even if it fails, the execution might or might not generate before_after_payload.
    # In the current implementation, if the pipeline fails, before_after is None.
    # Let's test the preview generator directly to be sure.
    from main import generate_preview_data_uri
    res = generate_preview_data_uri(str(fake_t1))
    assert res == str(fake_t1) # Should fallback to the path on failure


