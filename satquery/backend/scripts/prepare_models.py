import os
from transformers import AutoProcessor, LlavaForConditionalGeneration, OwlViTProcessor, OwlViTForObjectDetection

def main():
    print("Preparing directories...")
    os.makedirs("out/reports", exist_ok=True)
    os.makedirs("out/visual", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    print("Downloading/Caching VQA Model: xtuner/llava-phi-3-mini-hf ...")
    try:
        AutoProcessor.from_pretrained("xtuner/llava-phi-3-mini-hf")
        LlavaForConditionalGeneration.from_pretrained("xtuner/llava-phi-3-mini-hf")
        print("VQA Model downloaded successfully.")
    except Exception as e:
        print(f"Failed to download VQA model: {e}")
        
    print("Downloading/Caching Grounding Model: google/owlvit-base-patch32 ...")
    try:
        OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        print("Grounding Model downloaded successfully.")
    except Exception as e:
        print(f"Failed to download Grounding model: {e}")
        
    print("Setup Complete.")

if __name__ == "__main__":
    main()
