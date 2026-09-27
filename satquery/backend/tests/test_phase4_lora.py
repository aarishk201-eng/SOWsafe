import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.vqa.lora import LoRALinear, RSVQALoRAModel
from models.vqa.dataset import get_train_val_splits, generate_rs_vqa_pairs
from models.vqa.train_lora import evaluate_split, train_lora_model


def test_held_out_validation_split_never_touches_training():
    """Verify held-out validation split is strictly disjoint from the training set."""
    train_split, val_split = get_train_val_splits(val_ratio=0.25)
    train_ids = {s["id"] for s in train_split}
    val_ids = {s["id"] for s in val_split}

    assert len(train_ids) > 0
    assert len(val_ids) > 0
    # Overlap must be empty
    overlap = train_ids.intersection(val_ids)
    assert len(overlap) == 0, f"Held-out validation split contaminated with training sample: {overlap}"


def test_base_weights_strictly_frozen_during_lora():
    """Verify that during LoRA fine-tuning, base model weights W_0 receive 0 updates."""
    layer = LoRALinear(in_features=32, out_features=16, r=4, lora_alpha=8.0)
    w_initial = layer.weight.copy()

    # Run dummy inputs through forward and backward passes
    rng = np.random.RandomState(42)
    x = rng.randn(4, 32).astype(np.float32)
    grad_out = rng.randn(4, 16).astype(np.float32)

    for _ in range(5):
        layer.backward(x, grad_out, lr=0.01)

    # Base weight must be bit-for-bit identical to initial weight
    assert np.array_equal(layer.weight, w_initial), "Base weights W_0 were modified! They must remain frozen."
    # Adapter weights must have changed
    assert not np.array_equal(layer.lora_A, np.zeros_like(layer.lora_A))
    assert not np.array_equal(layer.lora_B, np.zeros_like(layer.lora_B))


def test_lora_adapter_initialization_starts_at_zero():
    """Verify Delta W starts at exactly 0 because B is initialized to zeros."""
    layer = LoRALinear(in_features=32, out_features=16, r=4)
    x = np.random.randn(2, 32).astype(np.float32)

    base_out = layer.forward(x, use_adapter=False)
    adapter_out = layer.forward(x, use_adapter=True)

    # Before backward, adapter output should match base output exactly
    np.testing.assert_allclose(base_out, adapter_out, rtol=1e-5)


def test_both_base_and_lora_checkpoints_exist_and_loadable(tmp_path):
    """Verify both base and adapted checkpoints are saved and loadable."""
    chk_dir = str(tmp_path / "checkpoints_test")
    summary = train_lora_model(epochs=3, checkpoints_dir=chk_dir)

    base_path = os.path.join(chk_dir, "base")
    lora_path = os.path.join(chk_dir, "lora_adapted")

    # Base checkpoint files
    assert os.path.exists(os.path.join(base_path, "base_config.json"))
    assert os.path.exists(os.path.join(base_path, "base_weights.npz"))

    # Adapted checkpoint files
    assert os.path.exists(os.path.join(lora_path, "adapter_config.json"))
    assert os.path.exists(os.path.join(lora_path, "adapter_model.npz"))

    # Load adapter back into fresh model
    new_model = RSVQALoRAModel(feature_dim=128, hidden_dim=128, num_classes=8, r=16, alpha=32.0)
    new_model.load_lora_checkpoint(lora_path)
    assert not np.all(new_model.vlm_projection.lora_B == 0)


def test_training_loss_history_logged(tmp_path):
    """Verify training loss is logged at steps and epochs, showing progressive reduction."""
    chk_dir = str(tmp_path / "checkpoints_log")
    summary = train_lora_model(epochs=5, checkpoints_dir=chk_dir)

    log_path = os.path.join(chk_dir, "training_loss.json")
    assert os.path.exists(log_path)

    with open(log_path, "r") as f:
        log_data = json.load(f)

    assert "epoch_history" in log_data
    assert len(log_data["epoch_history"]) == 5
    # Loss should decrease from initial epoch
    initial_loss = log_data["epoch_history"][0]["train_loss"]
    final_loss = log_data["epoch_history"][-1]["train_loss"]
    assert final_loss < initial_loss, f"Final loss ({final_loss}) was not lower than initial ({initial_loss})"


def test_before_after_eval_metrics_improved(tmp_path):
    """Verify before vs after numbers on the held-out validation split."""
    chk_dir = str(tmp_path / "checkpoints_eval")
    summary = train_lora_model(epochs=5, checkpoints_dir=chk_dir)

    before_val_loss = summary["before_finetuning"]["val_loss"]
    after_val_loss = summary["after_finetuning"]["val_loss"]

    assert after_val_loss < before_val_loss
    assert summary["metrics_delta"]["val_loss_reduction"] > 0
    assert summary["metrics_delta"]["val_accuracy_gain"] >= 0
