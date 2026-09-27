import os
import sys
import json
import time
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import app

client = TestClient(app)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _get_fixture_path(filename: str) -> str:
    """Helper ensuring fixture paths resolve from any working directory."""
    local = os.path.join("fixtures", filename)
    if os.path.exists(local):
        return local
    backend_local = os.path.join(FIXTURES_DIR, filename)
    return backend_local


def test_vqa_returns_nonempty_answer_for_valid_image():
    fixture_path = _get_fixture_path("subset/sample_0.tif")
    with open(fixture_path, "rb") as f:
        resp = client.post(
            "/vqa",
            files={"image": f},
            data={"question": "What type of land cover is visible?"},
        )
    assert resp.status_code == 200
    assert len(resp.json()["answer"]) > 0


def test_vqa_rejects_non_georeferenced_or_corrupt_image():
    fixture_path = _get_fixture_path("corrupt.tif")
    with open(fixture_path, "rb") as f:
        resp = client.post(
            "/vqa",
            files={"image": f},
            data={"question": "What is visible?"},
        )
    assert resp.status_code == 400


def test_vqa_latency_under_budget():
    fixture_path = _get_fixture_path("subset/sample_0.tif")
    t0 = time.time()
    with open(fixture_path, "rb") as f:
        client.post(
            "/vqa",
            files={"image": f},
            data={"question": "Describe this scene."},
        )
    assert time.time() - t0 < 10  # Budget: 10s


def test_vqa_hand_labeled_ground_truth_eval_set():
    """Pass gate: on >=3 hand-labeled fixture images, the model's answer is evaluated against ground truth."""
    labels_file = _get_fixture_path("labels.json")
    with open(labels_file, "r") as f:
        eval_cases = json.load(f)

    assert len(eval_cases) >= 3, "Eval set must contain at least 3 labeled images"

    eval_results = []
    for case in eval_cases:
        fixture_path = _get_fixture_path(os.path.basename(case["file"]))
        with open(fixture_path, "rb") as f:
            resp = client.post(
                "/vqa",
                files={"image": f},
                data={"question": case["question"]},
            )

        assert resp.status_code == 200
        answer = resp.json()["answer"]

        # Check that answer contains expected domain keywords
        matched = any(kw.lower() in answer.lower() for kw in case["expected_keywords"])
        eval_results.append({
            "file": case["file"],
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "answer": answer,
            "matched": matched,
        })
        assert matched, f"Answer '{answer}' did not match ground truth expectations for {case['file']}"

    assert all(r["matched"] for r in eval_results)
