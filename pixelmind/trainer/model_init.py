"""
Model initialization helpers for PixelMind training.
"""

import os

import torch
from transformers import AutoTokenizer

from ..config import PixelMindConfig
from ..model import PixelMindForCausalLM, PixelMind
from .utils import Logger, get_model_params


def init_llm_model(
    config: PixelMindConfig,
    from_weight="pretrain",
    tokenizer_path="./model/tokenizer",
    save_dir="./out",
    device="cuda",
):
    """
    Initialize a PixelMind LLM model + tokenizer.

    Args:
        config: PixelMindConfig
        from_weight: checkpoint name prefix ("none" = from scratch)
        tokenizer_path: path to BPE tokenizer
        save_dir: directory containing trained weights
        device: torch device

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = PixelMindForCausalLM(config)

    if from_weight != "none":
        weight_path = f"{save_dir}/{from_weight}_{config.hidden_size}.pth"
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            model.load_state_dict(weights, strict=False)
            Logger(f"Loaded weights from {weight_path}")
        else:
            Logger(f"Warning: weight file not found at {weight_path}, starting fresh")

    get_model_params(model, config)
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ) / 1e6
    Logger(f"Trainable Params: {trainable:.3f}M")
    return model.to(device), tokenizer


def init_vlm_model(
    config: PixelMindConfig,
    from_weight="pretrain_vlm",
    tokenizer_path="./model/tokenizer",
    vision_encoder_path="./model/siglip2-base-p32-256-ve",
    save_dir="./out",
    device="cuda",
    freeze_llm=1,
):
    """
    Initialize a PixelMind VLM model + tokenizer + image preprocessor.

    Args:
        config: PixelMindConfig
        from_weight: checkpoint name prefix ("none" = from scratch projector)
        tokenizer_path: path to BPE tokenizer
        vision_encoder_path: path to vision encoder weights
        save_dir: directory containing trained weights
        device: torch device
        freeze_llm: freeze strategy
            0 = unfreeze all LLM+projector (full training)
            1 = unfreeze projector + LLM first & last layers only (default SFT)
            2 = unfreeze only projector (pretrain alignment)

    Returns:
        (model, tokenizer, preprocess_fn)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = PixelMind(
        config,
        vision_encoder_path=vision_encoder_path,
    )

    # Load LLM + projector weights
    if from_weight != "none":
        weight_path = f"{save_dir}/{from_weight}_{config.hidden_size}.pth"
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            model.load_state_dict(weights, strict=False)
            Logger(f"Loaded weights from {weight_path}")
        else:
            Logger(f"Warning: weight file not found at {weight_path}, starting fresh")

    # ── Apply freeze strategy ──
    # Step 1: Freeze vision encoder always
    for name, param in model.named_parameters():
        if "vision_encoder" in name:
            param.requires_grad = False

    if freeze_llm == 0:
        # Unfreeze everything except vision_encoder
        for name, param in model.named_parameters():
            if "vision_encoder" not in name:
                param.requires_grad = True
    elif freeze_llm == 1:
        # Freeze all LLM layers first (keep projector trainable)
        for name, param in model.named_parameters():
            if "vision_proj" not in name:
                param.requires_grad = False
        # Then unfreeze first and last LLM layers
        last_idx = config.num_hidden_layers - 1
        for name, param in model.model.named_parameters():
            if f"layers.0." in name or f"layers.{last_idx}." in name:
                param.requires_grad = True
    elif freeze_llm == 2:
        # Only projector is trainable (everything else frozen)
        for name, param in model.named_parameters():
            if "vision_proj" in name:
                param.requires_grad = True
            elif "vision_encoder" not in name:
                param.requires_grad = False

    get_model_params(model, config, ignore_patterns={"vision_encoder"})
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ) / 1e6
    Logger(f"Trainable Params: {trainable:.3f}M")

    # Get image preprocessor from encoder
    preprocess = model.vision_encoder.preprocess
    return model.to(device), tokenizer, preprocess
