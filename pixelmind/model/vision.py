"""
Pluggable Visual Encoder System for PixelMind
==============================================
Abstract base class + concrete implementations for different visual encoders.

Supported encoders:
  - SigLIP2 (default) — balanced vision features, 95M params frozen
  - DINOv2 — strong for fine-grained recognition / OCR
  - InternViT — SOTA for charts, OCR, dense text (via InternVL)
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn, Tensor
from PIL import Image


# ─────────────────────────────────────────────────────────────────────
# Abstract Interface
# ─────────────────────────────────────────────────────────────────────

class BaseVisionEncoder(nn.Module, ABC):
    """
    Abstract interface for pluggable visual encoders.

    To add a new encoder, subclass this and implement:
      - forward(pixel_values) → Tensor [B, N, hidden_size]
      - hidden_size property
      - from_pretrained(path) classmethod
      - preprocess(image) → dict (returns a dict of tensors)
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, pixel_values) -> Tensor:
        """Encode images to visual features.
        Args:
          pixel_values: image tensors of shape [B, C, H, W]
        Returns: features of shape [B, num_patches, hidden_size]
        """

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Output dimension per patch."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: str) -> "BaseVisionEncoder":
        """Load encoder from a local directory."""

    @abstractmethod
    def preprocess(self, image: Image.Image) -> dict:
        """Convert a PIL Image to model inputs dict.
        Returns: dict of torch tensors (e.g. {"pixel_values": tensor})
        """


# ─────────────────────────────────────────────────────────────────────
# SigLIP2 Encoder (Default)
# ─────────────────────────────────────────────────────────────────────

class SigLIP2Encoder(BaseVisionEncoder):
    """
    SigLIP2-base Vision Transformer encoder (~95M params, frozen).
    Input: 256×256 images, patch_size=32 → 64 visual tokens.
    Output: 64 tokens × 768 hidden_dim.
    """

    def __init__(self, model_path: str):
        super().__init__()
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SigLIP2 model not found at {model_path}. "
                f"Download it or set a different vision_encoder_path."
            )

        from transformers import SiglipVisionModel, SiglipImageProcessor

        self._model = SiglipVisionModel.from_pretrained(model_path)
        self._processor = SiglipImageProcessor.from_pretrained(model_path)

        # Freeze all parameters
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.eval()

    @property
    def hidden_size(self) -> int:
        return self._model.config.hidden_size

    def forward(self, pixel_values) -> Tensor:
        with torch.no_grad():
            # Handle dict-format pixel_values from preprocess
            if hasattr(pixel_values, "keys"):
                inputs = {
                    k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v
                    for k, v in pixel_values.items()
                }
                outputs = self._model(**inputs)
            else:
                outputs = self._model(pixel_values)
        return outputs.last_hidden_state

    @classmethod
    def from_pretrained(cls, path: str) -> "SigLIP2Encoder":
        return cls(model_path=path)

    def preprocess(self, image: Image.Image) -> dict:
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")
        return self._processor(images=image, return_tensors="pt")

    @property
    def processor(self):
        """Expose processor for external use (e.g., image2tensor)."""
        return self._processor


# ─────────────────────────────────────────────────────────────────────
# DINOv2 Encoder (Alternative)
# ─────────────────────────────────────────────────────────────────────

class DINOv2Encoder(BaseVisionEncoder):
    """
    DINOv2 ViT encoder — strong self-supervised features.
    Good for fine-grained object recognition and OCR.

    Uses a CLS token → tile-replication to produce patch tokens
    (since DINOv2 outputs CLS + patches, we treat all non-CLS tokens as patches).
    """

    def __init__(self, model_path: str):
        super().__init__()
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"DINOv2 model not found at {model_path}."
            )

        from transformers import AutoImageProcessor, AutoModel

        self._model = AutoModel.from_pretrained(model_path)
        self._processor = AutoImageProcessor.from_pretrained(model_path)

        # Freeze
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.eval()

    @property
    def hidden_size(self) -> int:
        return self._model.config.hidden_size

    def forward(self, pixel_values) -> Tensor:
        with torch.no_grad():
            if hasattr(pixel_values, "keys"):
                outputs = self._model(**pixel_values)
            else:
                outputs = self._model(pixel_values)
        # DINOv2 returns [CLS, patch_1, ..., patch_N]
        # Exclude CLS token, keep patches
        return outputs.last_hidden_state[:, 1:, :]

    @classmethod
    def from_pretrained(cls, path: str) -> "DINOv2Encoder":
        return cls(model_path=path)

    def preprocess(self, image: Image.Image) -> dict:
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")
        return self._processor(images=image, return_tensors="pt")


# ─────────────────────────────────────────────────────────────────────
# InternViT Encoder (Alternative)
# ─────────────────────────────────────────────────────────────────────

class InternViTEncoder(BaseVisionEncoder):
    """
    InternViT encoder from InternVL family.
    Strong for OCR, charts, and dense-text understanding.
    """

    def __init__(self, model_path: str):
        super().__init__()
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"InternViT model not found at {model_path}."
            )

        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        # Use default image processor; adapt mean/std if needed
        from transformers import AutoImageProcessor
        self._processor = AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        for param in self._model.parameters():
            param.requires_grad = False
        self._model.eval()

    @property
    def hidden_size(self) -> int:
        return self._model.config.hidden_size

    def forward(self, pixel_values) -> Tensor:
        with torch.no_grad():
            if hasattr(pixel_values, "keys"):
                outputs = self._model(**pixel_values)
            else:
                outputs = self._model(pixel_values)
        return outputs.last_hidden_state

    @classmethod
    def from_pretrained(cls, path: str) -> "InternViTEncoder":
        return cls(model_path=path)

    def preprocess(self, image: Image.Image) -> dict:
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")
        return self._processor(images=image, return_tensors="pt")


# ─────────────────────────────────────────────────────────────────────
# Encoder Registry
# ─────────────────────────────────────────────────────────────────────

ENCODER_REGISTRY = {
    "siglip2": SigLIP2Encoder,
    "dinov2": DINOv2Encoder,
    "internvit": InternViTEncoder,
}


def build_vision_encoder(name: str, path: str) -> BaseVisionEncoder:
    """
    Factory function to build a vision encoder by name.

    Args:
        name: one of "siglip2", "dinov2", "internvit"
        path: local directory with the encoder weights

    Returns:
        BaseVisionEncoder instance

    Raises:
        ValueError: if encoder name is not registered
    """
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder: '{name}'. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )
    return ENCODER_REGISTRY[name].from_pretrained(path)


# ─────────────────────────────────────────────────────────────────────
# MMVisionProjector
# ─────────────────────────────────────────────────────────────────────

class MMVisionProjector(nn.Module):
    """
    Multi-modal vision projector (2-layer MLP).

    Architecture: LayerNorm → Linear → GELU → Linear

    Projects visual features from encoder hidden_size to LLM hidden_size.
    Typically: 768 → 768 for PixelMind default config.
    Handles dtype conversion between encoder output (fp16) and LLM (fp32/fp16).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        source_tokens: int = 64,
        target_tokens: int = 64,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Cast to match projector weights dtype (handles fp16 encoder → fp32 projector)
        return self.mlp(x.to(self.mlp[0].weight.dtype))
