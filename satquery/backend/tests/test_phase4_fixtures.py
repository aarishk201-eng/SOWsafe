"""Phase 4 Pass Gate Fixture Tests: Leakage Prevention and LoRA Validation Outperformance.

Pass gate requirements:
1. test_no_train_test_leakage: Asserts train and held-out validation splits are strictly disjoint.
2. test_adapted_model_outperforms_base_on_val_set: Asserts adapted model outperforms base model.
3. Pass gate: adapted model shows a measurable, reproducible accuracy improvement over base on
   a held-out split — not just anecdotally better on 2-3 cherry-picked examples.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.vqa import (
    load_ids,
    evaluate,
    base_model,
    adapted_model,
    val_set,
    get_base_model,
    get_adapted_model,
    get_val_set,
)


def test_no_train_test_leakage():
    train_ids = set(load_ids("train_split.json"))
    val_ids = set(load_ids("val_split.json"))
    assert train_ids.isdisjoint(val_ids)


def test_adapted_model_outperforms_base_on_val_set():
    base_acc = evaluate(base_model, val_set)
    adapted_acc = evaluate(adapted_model, val_set)
    assert adapted_acc >= base_acc  # if this fails, don't ship the adapter — ship the base model


def test_adapted_model_shows_measurable_reproducible_accuracy_improvement_pass_gate():
    """Pass gate: adapted model shows a measurable, reproducible accuracy improvement over base

    on a held-out split — not just anecdotally better on 2-3 cherry-picked examples.
    """
    base_acc = evaluate(base_model, val_set)
    adapted_acc = evaluate(adapted_model, val_set)

    # Must be strictly superior, not merely equal
    accuracy_delta = adapted_acc - base_acc
    print(f"\n[Phase 4 Pass Gate] Base Acc: {base_acc:.4f} | Adapted Acc: {adapted_acc:.4f} | Delta: {accuracy_delta:+.4f}")

    assert adapted_acc > base_acc, f"Adapted model ({adapted_acc}) did not beat base model ({base_acc}) on held-out split"
    assert accuracy_delta >= 0.50, f"Expected at least +50% gain on RS domain, got {accuracy_delta * 100:.1f}%"
    assert len(val_set) >= 10, f"Held-out validation split must have >= 10 samples for statistical validity, got {len(val_set)}"
