"""Phase 8 Pass Gate Tests (Root Mirror): Intent Classifier & Query Router (agents/router.py).

Requirements:
1. test_router_maps_query_to_correct_task (parameterized across 4 core examples):
   - ("Highlight the water body.", "grounding")
   - ("What changed between 2022 and 2025?", "change_vqa")
   - ("Use optical and SAR together to identify built-up areas.", "cross_modal_analysis")
   - ("What type of land cover is visible?", "vqa")
2. test_router_handles_ambiguous_query_gracefully:
   - result = route("tell me something interesting")
   - assert result["task"] in KNOWN_TASKS or result.get("clarification_needed") is True
3. Pass gate: >=90% correct routing on a hand-written set of >=20 varied queries
   covering all 5 intents plus edge cases.
"""

import os
import sys
import pytest

_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "satquery", "backend"))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

from agents.router import route, KNOWN_TASKS


@pytest.mark.parametrize("query,expected_task", [
    ("Highlight the water body.", "grounding"),
    ("What changed between 2022 and 2025?", "change_vqa"),
    ("Use optical and SAR together to identify built-up areas.", "cross_modal_analysis"),
    ("What type of land cover is visible?", "vqa"),
])
def test_router_maps_query_to_correct_task(query, expected_task):
    result = route(query)
    assert result["task"] == expected_task


def test_router_handles_ambiguous_query_gracefully():
    result = route("tell me something interesting")
    assert result["task"] in KNOWN_TASKS or result.get("clarification_needed") is True


# Hand-written benchmark of 26 varied queries covering all 5 intents plus edge cases
BENCHMARK_QUERIES = [
    # 1. GROUNDING
    ("Highlight the water body.", "grounding"),
    ("Locate the cargo port in the satellite image.", "grounding"),
    ("Find the runway bounding box on the airfield.", "grounding"),
    ("Where is the oil storage tank located?", "grounding"),
    ("Pinpoint shipping vessels docked in the harbor.", "grounding"),

    # 2. CHANGE_VQA
    ("What changed between 2022 and 2025?", "change_vqa"),
    ("Has the built-up area increased between t1 and t2?", "change_vqa"),
    ("Did deforestation occur over the past 3 years?", "change_vqa"),
    ("How much did the urban footprint expand from 2020 to 2024?", "change_vqa"),
    ("Have flood waters receded since yesterday?", "change_vqa"),

    # 3. CROSS_MODAL_ANALYSIS / OPTICAL_SAR
    ("Use optical and SAR together to identify built-up areas.", "cross_modal_analysis"),
    ("Fuse optical and SAR imagery.", "cross_modal_analysis"),
    ("Combine Sentinel-1 radar and Sentinel-2 optical data for land classification.", "cross_modal_analysis"),
    ("All-weather flood extent mapping fusing optical reflectance and radar backscatter.", "cross_modal_analysis"),
    ("Multimodal cross-sensor evidence fusion for cloud-occluded regions.", "cross_modal_analysis"),

    # 4. VQA
    ("What type of land cover is visible?", "vqa"),
    ("How many airplanes are parked on the tarmac?", "vqa"),
    ("Describe the terrain shown in this satellite image.", "vqa"),
    ("Is this area predominantly agricultural or residential?", "vqa"),
    ("Identify the primary geological features in the image.", "vqa"),

    # 5. CHANGE (imperative detection commands)
    ("Detect change between image_a and image_b.", "change"),
    ("Compute change mask for t1 and t2.", "change"),
    ("Generate bi-temporal difference map.", "change"),
    ("Calculate altered areas between before and after scenes.", "change"),

    # 6. EDGE CASES
    ("tell me something interesting", "ambiguous"),
    ("   ", "ambiguous"),
]


def test_pass_gate_90_percent_accuracy_on_20_plus_queries():
    """Pass gate: >=90% correct routing on a hand-written set of >=20 varied queries

    covering all 5 intents plus edge cases.
    """
    total = len(BENCHMARK_QUERIES)
    correct = 0
    failures = []

    for query, expected in BENCHMARK_QUERIES:
        res = route(query)
        if expected == "ambiguous":
            if res.get("clarification_needed") is True or res["task"] in KNOWN_TASKS:
                correct += 1
            else:
                failures.append(f"Ambiguous query '{query}' failed graceful handling: {res}")
        else:
            if res["task"] == expected:
                correct += 1
            else:
                failures.append(f"Query '{query}' expected '{expected}', got '{res['task']}'")

    accuracy = (correct / total) * 100.0
    print(f"\n[Phase 8 Pass Gate] Evaluated {total} queries. Correct: {correct}/{total} ({accuracy:.1f}%)")
    if failures:
        print("Failures:")
        for f in failures:
            print("  -", f)

    assert total >= 20, f"Expected at least 20 queries, got {total}"
    assert accuracy >= 90.0, f"Expected >= 90% routing accuracy, achieved {accuracy:.1f}%"
