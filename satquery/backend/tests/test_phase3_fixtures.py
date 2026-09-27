import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import app
from geospatial.validator import extract_metadata

client = TestClient(app)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _get_fixture(filename: str) -> str:
    local = os.path.join("fixtures", filename)
    if os.path.exists(local):
        return local
    backend_local = os.path.join(FIXTURES_DIR, filename)
    return backend_local


def compute_iou(box_a, box_b):
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = float(box_a_area + box_b_area - inter_area)

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def test_grounding_returns_bbox_within_image_bounds():
    fixture_path = _get_fixture("subset/sample_2.tif")
    with open(fixture_path, "rb") as f:
        resp = client.post(
            "/ground",
            files={"image": f},
            data={"phrase": "water body"},
        )
    assert resp.status_code == 200
    bbox = resp.json()["bbox"]
    assert bbox is not None
    meta = extract_metadata(fixture_path)
    assert 0 <= bbox[0] < bbox[2] <= meta.width
    assert 0 <= bbox[1] < bbox[3] <= meta.height


def test_grounding_low_confidence_for_absent_object():
    fixture_path = _get_fixture("desert_scene.tif")
    with open(fixture_path, "rb") as f:
        resp = client.post(
            "/ground",
            files={"image": f},
            data={"phrase": "swimming pool"},
        )
    assert resp.status_code == 200
    assert resp.json()["confidence"] < 0.3
    assert resp.json()["bbox"] is None


def test_grounding_iou_pass_gate_on_5_fixtures():
    """Pass gate: IoU >= 0.5 against hand-drawn ground-truth boxes on >= 5 fixture images."""
    labels_file = _get_fixture("grounding_labels.json")
    with open(labels_file, "r") as f:
        eval_cases = json.load(f)

    assert len(eval_cases) >= 5, "Eval set must contain at least 5 labeled images"

    iou_results = []
    for case in eval_cases:
        fixture_path = _get_fixture(os.path.basename(case["file"]))
        gt_box = case["ground_truth_bbox"]

        with open(fixture_path, "rb") as f:
            resp = client.post(
                "/ground",
                files={"image": f},
                data={"phrase": case["phrase"]},
            )

        assert resp.status_code == 200
        pred_box = resp.json()["bbox"]
        assert pred_box is not None, f"Expected box for phrase '{case['phrase']}' in {case['file']}"

        iou = compute_iou(pred_box, gt_box)
        iou_results.append({
            "file": case["file"],
            "phrase": case["phrase"],
            "pred_box": pred_box,
            "gt_box": gt_box,
            "iou": round(iou, 4),
        })

        assert iou >= 0.5, f"IoU for {case['file']} is {iou:.4f}, expected >= 0.5"

    mean_iou = sum(r["iou"] for r in iou_results) / len(iou_results)
    assert mean_iou >= 0.5
    print(f"\nMean Grounding IoU on {len(iou_results)} scenes: {mean_iou:.4f}")
