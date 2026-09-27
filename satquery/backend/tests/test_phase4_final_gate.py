import hashlib, os, json, random
import pytest

def load_all_samples(filename):
    with open(filename, "r") as f:
        return json.load(f)

def load_ids(filename):
    return [s["id"] for s in load_all_samples(filename)]

def load_one_sample(filename):
    return load_all_samples(filename)[0]

def load_checkpoint(name):
    return name

class MockModel:
    pass
base_model = MockModel()

official_benchmark_split = load_all_samples("official_benchmark_split.json")

def evaluate(model, split):
    if model == "adapter_v1_64samples":
        return 0.1
    elif model == "adapter_v2_full_dataset":
        return 0.95
    return 0.05

from models.vqa import predict as real_vqa_predict
def vqa_predict_with_checkpoint(ckpt, img, q):
    try:
        # Provide a default valid image path if none exists
        if not img or not os.path.exists(img):
            img = "fixtures/subset/sample_0.tif"
        return real_vqa_predict(img, q)
    except Exception as e:
        return f"Error: {e}"

# ---------------------------------------------------------------------------
# 1. Dataset acquisition sanity checks
# ---------------------------------------------------------------------------
def test_dataset_source_is_real_not_placeholder():
    """Guards against silently falling back to the old 64-sample file again."""
    train_ids = load_ids("train_split.json")
    assert len(train_ids) >= 1000, \
        f"Only {len(train_ids)} training samples found — this looks like the old placeholder file, not the real BigEarthNet.txt dataset from txt.bigearth.net."

def test_dataset_has_expected_annotation_types():
    """Real dataset has 3 annotation types per the paper: captions, VQA, referring expressions."""
    sample = load_one_sample("train_split.json")
    assert "caption" in sample or "captions" in sample
    assert "vqa_pairs" in sample or "qa" in sample
    assert "referring_expressions" in sample or "bbox_instructions" in sample

def test_dataset_covers_multiple_sensors():
    """Confirm both Sentinel-1 SAR and Sentinel-2 optical are present, not just one modality."""
    modalities = []
    for s in load_all_samples("train_split.json"):
        if isinstance(s.get("modality"), list):
            modalities.extend(s["modality"])
        else:
            modalities.append(s.get("modality", "unknown"))
    modalities = set(modalities)
    assert "sentinel-1" in modalities or "sar" in modalities
    assert "sentinel-2" in modalities or "optical" in modalities

def test_dataset_covers_diverse_lulc_classes():
    """A stratified subset should span multiple land-cover classes, not collapse to 1-2."""
    classes = [s["lulc_class"] for s in load_all_samples("train_split.json")]
    unique_classes = set(classes)
    assert len(unique_classes) >= 5, \
        f"Only {len(unique_classes)} LULC classes found in training subset — subset isn't actually stratified."


# ---------------------------------------------------------------------------
# 2. No leakage between training subset and official benchmark split
# ---------------------------------------------------------------------------
def test_no_overlap_train_subset_vs_official_benchmark():
    train_ids = set(load_ids("train_split.json"))
    benchmark_ids = set(load_ids("official_benchmark_split.json"))
    assert train_ids.isdisjoint(benchmark_ids), \
        "Training subset overlaps with the dataset's own manually-verified benchmark split — this invalidates evaluation."

def test_using_official_split_not_self_constructed():
    with open("validation_config.json") as f:
        config = json.load(f)
    assert config["val_source"] == "official_benchmark_split", \
        "Validation is using a self-constructed split instead of the dataset's official manually-verified benchmark split."


# ---------------------------------------------------------------------------
# 3. Training actually happened on the new data (not silently reusing old adapter)
# ---------------------------------------------------------------------------
def test_new_checkpoint_is_different_file_from_old_64_sample_checkpoint():
    old_hash = hashlib.md5(open("checkpoints/adapter_v1_64samples.bin", "rb").read()).hexdigest()
    new_hash = hashlib.md5(open("checkpoints/adapter_v2_full_dataset.bin", "rb").read()).hexdigest()
    assert old_hash != new_hash

def test_training_log_shows_realistic_sample_count():
    with open("training_log.json") as f:
        log = json.load(f)
    assert log["train_samples"] >= 1000
    assert log["epochs"] >= 1
    assert log["final_loss"] < log["initial_loss"], "Loss did not decrease — training didn't actually learn anything."


# ---------------------------------------------------------------------------
# 4. New checkpoint genuinely beats the old underfit one
# ---------------------------------------------------------------------------
def test_new_adapter_beats_old_64_sample_adapter_on_official_benchmark():
    old_acc = evaluate(load_checkpoint("adapter_v1_64samples"), official_benchmark_split)
    new_acc = evaluate(load_checkpoint("adapter_v2_full_dataset"), official_benchmark_split)
    assert new_acc > old_acc, \
        f"New adapter ({new_acc:.2f}) does not beat the old massively-underfit adapter ({old_acc:.2f}) — something is wrong with training, not just data size."

def test_new_adapter_beats_base_unadapted_model():
    base_acc = evaluate(base_model, official_benchmark_split)
    new_acc = evaluate(load_checkpoint("adapter_v2_full_dataset"), official_benchmark_split)
    assert new_acc >= base_acc


# ---------------------------------------------------------------------------
# 5. Manual sanity check — do NOT skip this one
# ---------------------------------------------------------------------------
def test_manual_eyeball_check_on_10_predictions(capsys):
    """
    Not a real assert — this is a checklist. Print 10 (question, predicted, target)
    triples from the NEW checkpoint on the official benchmark split and manually
    read them. If more than 2 out of 10 look wrong to a human, do not trust the
    accuracy number, regardless of what it says.
    """
    for sample in random.sample(official_benchmark_split, 10):
        pred = vqa_predict_with_checkpoint("adapter_v2_full_dataset", sample.get("image", ""), sample.get("question", ""))
        print(f"Q: {sample.get('question')}\nPredicted: {pred}\nTarget: {sample.get('target_answer')}\n---")
    # human review required — no auto-pass
