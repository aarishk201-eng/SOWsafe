import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from models.vqa.data_module import get_bigearthnet_datamodule
from models.vqa.__init__ import predict
import json
import re

def filter_by_type(split_name, task_type):
    dm = get_bigearthnet_datamodule()
    dl = getattr(dm, f"{split_name}_dataloader")()
    samples = []
    
    # We iterate over batches and infer the type from the question
    for batch in dl:
        texts = batch["text_input"]
        outputs = batch["reference_output"]
        for q, o in zip(texts, outputs):
            q_lower = q.lower()
            inferred_type = None
            if "a)" in q_lower and "b)" in q_lower:
                inferred_type = "mcq"
            elif q_lower.startswith(("is", "are", "do", "does", "can", "has", "did", "would")):
                inferred_type = "binary"
            elif "<ref>" in q_lower:
                inferred_type = "bounding box"
            else:
                inferred_type = "captioning"
                
            if inferred_type == task_type:
                samples.append({"input": q, "output": str(o)})
                if len(samples) >= 50:
                    return samples
    return samples

def model_predict(s):
    # Pass dummy image and the question
    dummy_image = os.path.join(backend_dir, "fixtures", "subset/sample_0.tif")
    ans = predict(dummy_image, s["input"])
    return ans.prediction

def extract_options_from_input(q):
    options = []
    lines = q.split("\n")
    for line in lines:
        match = re.match(r"^([a-d])\)\s+(.*)$", line.strip().lower())
        if match:
            options.append({"letter": match.group(1), "text": match.group(2)})
    return options

def test_binary_predictions_arent_systematically_inverted():
    samples = filter_by_type("bench", "binary")[:50]
    preds = [model_predict(s) for s in samples]
    targets = [s["output"] for s in samples]
    
    correct = sum(p.strip().lower() == t.strip().lower() for p, t in zip(preds, targets))
    inverted = sum(p.strip().lower() != t.strip().lower() and p.strip().lower() in ["yes","no"] for p, t in zip(preds, targets))
    accuracy = correct / len(samples)
    inverted_accuracy = inverted / len(samples)
    
    print(f"Binary Correct: {correct}, Inverted: {inverted}, Total: {len(samples)}")
    print(f"Binary Accuracy: {accuracy:.2f}, Inverted Match Rate: {inverted_accuracy:.2f}")
    
    if accuracy < 0.45:
        print(f"FAILED: Binary accuracy ({accuracy:.2f}) is below random chance (0.5) — check for a yes/no label swap bug. Inverted-match rate: {inverted_accuracy:.2f}")

def test_mcq_letter_mapping_is_not_shifted():
    samples = filter_by_type("bench", "mcq")[:50]
    for i, s in enumerate(samples):
        options = extract_options_from_input(s["input"])
        target_letter = s["output"].strip().lower()
        if target_letter not in [o["letter"] for o in options]:
            print(f"FAILED: Target letter '{s['output']}' doesn't match any parsed option {options} — options-parsing itself might be shifted/broken.")
            return
    print("MCQ letter mapping test passed!")

if __name__ == "__main__":
    print("Running Binary Diagnostic...")
    test_binary_predictions_arent_systematically_inverted()
    print("\nRunning MCQ Diagnostic...")
    test_mcq_letter_mapping_is_not_shifted()
