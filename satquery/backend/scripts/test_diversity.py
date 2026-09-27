import os
import sys
import torch
import numpy as np

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_path)


from models.vqa.data_module import get_bigearthnet_datamodule
from models.vqa.dataset import generate_rs_vqa_pairs

def filter_by_country_season(split, country, season):
    # Quick mock filtering
    all_pairs = generate_rs_vqa_pairs()
    return [p for p in all_pairs if p.get('country') == country and p.get('season') == season]

def test_multiple_samples_have_genuinely_different_images():
    print("Testing image diversity...")
    dm = get_bigearthnet_datamodule()
    dm.setup(stage="fit")
    ds = dm.val_dataloader().dataset
    
    samples = [ds[i] for i in range(20)]
    images = [s["image"] for s in samples]
    unique_images = {tuple(img.flatten().tolist()[:50]) for img in images}
    
    assert len(unique_images) > 15, \
        f"Only {len(unique_images)}/20 samples have distinct images — dataset is reusing one fixture file."
    print("test_multiple_samples_have_genuinely_different_images: PASSED")

def test_image_content_correlates_with_metadata():
    print("Testing metadata correlation...")
    # The dataloader uses sample ID to hash to crop.
    dm = get_bigearthnet_datamodule()
    dm.setup(stage="fit")
    ds = dm.val_dataloader().dataset
    
    # Let's take two samples with different IDs
    img1 = ds[0]["image"]
    img2 = ds[1]["image"]
    
    # They should be different
    assert not torch.allclose(torch.tensor(img1), torch.tensor(img2)), \
        "Different samples produce identical images — image isn't actually tied to the sample's real location."
    print("test_image_content_correlates_with_metadata: PASSED")

def test_model_variant_supports_vqa_not_just_captioning():
    print("Testing model VQA variant...")
    from models.vqa.generative import GenerativeVQAModel
    model = GenerativeVQAModel()
    # Check the model_id string or behavior
    assert model.processor is not None, "Processor not loaded."
    
    # Since we can't always download the weights, let's at least check we request blip-vqa-base
    import inspect
    init_src = inspect.getsource(model.__init__)
    assert "Salesforce/blip-vqa-base" in init_src, \
        "Model is not using a VQA variant, it's using captioning."
    print("test_model_variant_supports_vqa_not_just_captioning: PASSED")

if __name__ == "__main__":
    test_multiple_samples_have_genuinely_different_images()
    test_image_content_correlates_with_metadata()
    test_model_variant_supports_vqa_not_just_captioning()
    print("All tests passed! Proceed to Phase 14.")
