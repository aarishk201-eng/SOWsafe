import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaConfig

class GenerativeVQAModel(nn.Module):
    """
    A true generative Vision-Language Model that structurally supports 
    autoregressive text generation (captions, bounding boxes, labels),
    replacing the old classification-head approach.
    """
    def __init__(self, model_id="xtuner/llava-phi-3-mini-hf", use_lora=True):
        super().__init__()
        
        # In a real environment, we would load the pretrained weights.
        # For this hackathon environment where downloading 1GB+ might take long,
        # we will attempt to load it, and if it fails (network/timeout), we initialize an empty architecture.
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.vlm = LlavaForConditionalGeneration.from_pretrained(model_id)
        except Exception as e:
            print(f"Warning: Falling back to structural mock due to {e}")
            # Fallback to structural instantiation without pretrained weights
            config = LlavaConfig()
            self.vlm = LlavaForConditionalGeneration(config)
            self.processor = None
            
        self.is_adapter_enabled = use_lora
        
        if use_lora:
            # Inject LoRA into the language model's attention projections
            self._apply_lora_to_decoder()

    def _apply_lora_to_decoder(self):
        """Minimal conceptual LoRA injection for the generative decoder."""
        # For hackathon purposes, we mock PEFT by just enabling gradients on query/value projections
        for name, param in self.vlm.named_parameters():
            if "q_proj" in name.lower() or "v_proj" in name.lower() or "lora" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    def forward(self, pixel_values, input_ids, attention_mask=None):
        """Standard causal language modeling forward pass for training."""
        return self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids # Causal LM loss
        )
        
    def generate(self, pixel_values, text=None, max_new_tokens=50):
        """Autoregressive generation."""
        if hasattr(self.vlm, "generate"):
            if self.processor is not None and text is not None:
                # Format prompt for LLaVA
                # Format: USER: <image>\n{text}\nASSISTANT: 
                prompt = f"USER: <image>\n{text}\nASSISTANT: "
                inputs = self.processor(text=prompt, images=pixel_values, return_tensors="pt")
                # Need to extract just the generated part
                # pyrefly: ignore [bad-argument-type]
                outputs = self.vlm.generate(
                    inputs,
                    max_new_tokens=max_new_tokens
                )
                return outputs
            # Fallback if structural mock
            return self.vlm.generate(pixel_values=pixel_values, max_new_tokens=max_new_tokens)  # type: ignore
        return None
