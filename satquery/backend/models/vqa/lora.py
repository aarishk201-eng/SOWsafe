"""Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation).

Implements low-rank decomposition matrices A and B injected into linear projection
layers of the Vision-Language Model.
W = W_0 + (alpha / r) * (B @ A)
where W_0 is frozen (gradient updates = 0) and only A and B are updated.
"""

from typing import Dict, Any, Tuple, Optional
import os
import json
import numpy as np


class LoRALinear:
    """Low-Rank Adaptation Linear Layer.

    Args:
        in_features: Input dimensionality (d_in).
        out_features: Output dimensionality (d_out).
        r: Rank of the low-rank adaptation decomposition matrices.
        lora_alpha: Scaling factor (alpha).
        lora_dropout: Optional dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: float = 32.0,
        base_weight: Optional[np.ndarray] = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Base pretrained weight W_0 (strictly frozen)
        if base_weight is not None:
            self.weight = base_weight.copy()
        else:
            # Simulated pretrained projection weights
            rng = np.random.RandomState(42)
            self.weight = rng.randn(out_features, in_features).astype(np.float32) * 0.02

        self.initial_base_weight = self.weight.copy()  # Snapshot to verify freezing

        # LoRA decomposition matrices:
        # A: Gaussian initialization
        rng_lora = np.random.RandomState(1337)
        self.lora_A = rng_lora.randn(r, in_features).astype(np.float32) / np.sqrt(r)
        # B: Initialized to zeros so Delta W starts at 0 (exact initial equivalence)
        self.lora_B = np.zeros((out_features, r), dtype=np.float32)

        # Gradients for optimization
        self.grad_A = np.zeros_like(self.lora_A)
        self.grad_B = np.zeros_like(self.lora_B)

    def forward(self, x: np.ndarray, use_adapter: bool = True) -> np.ndarray:
        """Forward pass: y = x @ W_0.T + (alpha/r) * (x @ A.T @ B.T)"""
        # Base projection (using frozen W_0)
        base_out = np.dot(x, self.weight.T)

        if not use_adapter:
            return base_out

        # LoRA delta projection: x @ A.T @ B.T * scaling
        lora_out = np.dot(np.dot(x, self.lora_A.T), self.lora_B.T) * self.scaling
        return base_out + lora_out

    def backward(self, x: np.ndarray, grad_output: np.ndarray, lr: float = 1e-3, weight_decay: float = 0.01):
        """Backward pass: updates ONLY lora_A and lora_B. W_0 receives ZERO gradient."""
        # Ensure base weight was never altered
        assert np.array_equal(self.weight, self.initial_base_weight), "Base weights must remain strictly frozen!"

        batch_size = x.shape[0] if len(x.shape) > 1 else 1

        # Intermediate activation: x @ A.T
        h_A = np.dot(x, self.lora_A.T)  # (batch, r)

        # Gradients with respect to B: grad_output.T @ h_A * scaling
        grad_B = np.dot(grad_output.T, h_A) * self.scaling / batch_size
        # Gradients with respect to A: (grad_output @ B) .T @ x * scaling
        grad_h_A = np.dot(grad_output, self.lora_B) * self.scaling
        grad_A = np.dot(grad_h_A.T, x) / batch_size

        # Parameter update (AdamW / SGD with weight decay)
        self.lora_B -= lr * (grad_B + weight_decay * self.lora_B)
        self.lora_A -= lr * (grad_A + weight_decay * self.lora_A)


class RSVQALoRAModel:
    """Vision-Language Model projection head adapted with LoRA for Remote Sensing VQA."""

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 128, num_classes: int = 8, r: int = 16, alpha: float = 32.0):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.r = r
        self.alpha = alpha
        self.is_adapter_enabled = True

        # Vision-Language multi-modal projection adapter
        self.vlm_projection = LoRALinear(feature_dim, hidden_dim, r=r, lora_alpha=alpha)
        # Classification / vocabulary projection adapter
        self.classifier = LoRALinear(hidden_dim, num_classes, r=r, lora_alpha=alpha)

    def forward(self, x: np.ndarray, use_adapter: Optional[bool] = None) -> np.ndarray:
        if use_adapter is None:
            use_adapter = self.is_adapter_enabled
        hidden = np.maximum(0, self.vlm_projection.forward(x, use_adapter=use_adapter))  # ReLU
        logits = self.classifier.forward(hidden, use_adapter=use_adapter)
        return logits

    def compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
        """Softmax cross-entropy loss."""
        # Numerical stability shift
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        batch_size = logits.shape[0]
        # Cross entropy loss
        loss = -np.mean(np.log(np.maximum(probs[np.arange(batch_size), targets], 1e-12)))

        # Gradient of loss w.r.t logits
        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), targets] -= 1.0
        return float(loss), grad_logits

    def train_step(self, x: np.ndarray, targets: np.ndarray, lr: float = 5e-3) -> float:
        """Executes a forward and backward step updating only LoRA parameters."""
        hidden_raw = self.vlm_projection.forward(x, use_adapter=True)
        hidden = np.maximum(0, hidden_raw)
        logits = self.classifier.forward(hidden, use_adapter=True)

        loss, grad_logits = self.compute_loss(logits, targets)

        # Backward through classifier adapter
        grad_hidden = np.dot(grad_logits, (self.classifier.weight + self.classifier.scaling * np.dot(self.classifier.lora_B, self.classifier.lora_A)))
        grad_hidden[hidden_raw <= 0] = 0.0  # ReLU gradient

        self.classifier.backward(hidden, grad_logits, lr=lr)
        # Backward through VLM projection adapter
        self.vlm_projection.backward(x, grad_hidden, lr=lr)

        return loss

    def save_base_checkpoint(self, dir_path: str):
        """Saves frozen base VLM weights and configuration."""
        os.makedirs(dir_path, exist_ok=True)
        config = {
            "model_type": "base_vlm_projection",
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "status": "frozen_base",
        }
        with open(os.path.join(dir_path, "base_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        np.savez_compressed(
            os.path.join(dir_path, "base_weights.npz"),
            proj_weight=self.vlm_projection.weight,
            clf_weight=self.classifier.weight,
        )

    def save_lora_checkpoint(self, dir_path: str):
        """Saves only the trained LoRA adapter matrices and metadata."""
        os.makedirs(dir_path, exist_ok=True)
        config = {
            "adapter_type": "LoRA",
            "r": self.r,
            "lora_alpha": self.alpha,
            "scaling": self.alpha / self.r,
            "target_modules": ["vlm_projection", "classifier"],
            "base_model": "base_vlm_projection",
        }
        with open(os.path.join(dir_path, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        np.savez_compressed(
            os.path.join(dir_path, "adapter_model.npz"),
            proj_lora_A=self.vlm_projection.lora_A,
            proj_lora_B=self.vlm_projection.lora_B,
            clf_lora_A=self.classifier.lora_A,
            clf_lora_B=self.classifier.lora_B,
        )

    def evaluate(self, dataset_split: Any) -> float:
        """Computes top-1 classification accuracy on a dataset split."""
        use_adapter = getattr(self, "is_adapter_enabled", True)
        x = np.stack([s["feature"] for s in dataset_split], axis=0)
        targets = np.array([s["label_idx"] for s in dataset_split], dtype=int)
        logits = self.forward(x, use_adapter=use_adapter)
        preds = np.argmax(logits, axis=-1)
        return float(np.mean(preds == targets))

    def load_base_checkpoint(self, dir_path: str):
        """Loads frozen base VLM weights and disables adapter by default."""
        weights_file = os.path.join(dir_path, "base_weights.npz")
        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"Base checkpoint not found at {weights_file}")
        data = np.load(weights_file)
        self.vlm_projection.weight = data["proj_weight"].copy()
        self.vlm_projection.initial_base_weight = data["proj_weight"].copy()
        self.classifier.weight = data["clf_weight"].copy()
        self.classifier.initial_base_weight = data["clf_weight"].copy()
        self.is_adapter_enabled = False

    def load_lora_checkpoint(self, dir_path: str):
        """Loads LoRA adapter matrices into the model."""
        adapter_file = os.path.join(dir_path, "adapter_model.npz")
        if not os.path.exists(adapter_file):
            raise FileNotFoundError(f"Adapter checkpoint not found at {adapter_file}")

        data = np.load(adapter_file)
        self.vlm_projection.lora_A = data["proj_lora_A"].copy()
        self.vlm_projection.lora_B = data["proj_lora_B"].copy()
        self.classifier.lora_A = data["clf_lora_A"].copy()
        self.classifier.lora_B = data["clf_lora_B"].copy()
        self.is_adapter_enabled = True
