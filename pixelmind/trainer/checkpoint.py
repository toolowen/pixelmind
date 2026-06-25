"""
Checkpoint utilities for PixelMind training.
Provides llm_checkpoint and vlm_checkpoint (save & load).
"""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .utils import Logger


def _save_checkpoint(
    config,
    weight,
    model,
    optimizer,
    epoch,
    step,
    wandb,
    save_dir,
    clean_fn=None,
    **kwargs,
):
    """Shared save logic for LLM and VLM checkpoints."""
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = f"{save_dir}/{weight}_{config.hidden_size}.pth"
    resume_path = f"{save_dir}/{weight}_{config.hidden_size}_resume.pth"

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    state_dict = raw_model.state_dict()

    if clean_fn:
        state_dict = clean_fn(state_dict)

    state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

    # Atomic save
    ckp_tmp = ckp_path + ".tmp"
    torch.save(state_dict, ckp_tmp)
    os.replace(ckp_tmp, ckp_path)

    # Wandb ID for resume
    wandb_id = None
    if wandb:
        if hasattr(wandb, "get_run"):
            run = wandb.get_run()
            wandb_id = getattr(run, "id", None) if run else None
        else:
            wandb_id = getattr(wandb, "id", None)

    resume_data = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "wandb_id": wandb_id,
    }
    for key, value in kwargs.items():
        if value is not None:
            if hasattr(value, "state_dict"):
                raw_value = (
                    value.module
                    if isinstance(value, DistributedDataParallel)
                    else value
                )
                raw_value = getattr(raw_value, "_orig_mod", raw_value)
                resume_data[key] = raw_value.state_dict()
            else:
                resume_data[key] = value

    resume_tmp = resume_path + ".tmp"
    torch.save(resume_data, resume_tmp)
    os.replace(resume_tmp, resume_path)
    torch.cuda.empty_cache()


def _load_checkpoint(config, weight, save_dir):
    """Shared load logic for LLM and VLM checkpoints."""
    resume_path = f"{save_dir}/{weight}_{config.hidden_size}_resume.pth"
    if os.path.exists(resume_path):
        ckp_data = torch.load(resume_path, map_location="cpu")
        saved_ws = ckp_data.get("world_size", 1)
        current_ws = dist.get_world_size() if dist.is_initialized() else 1
        if saved_ws != current_ws:
            ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
            Logger(
                f"GPU count changed ({saved_ws}→{current_ws}), "
                f"step adjusted to {ckp_data['step']}"
            )
        return ckp_data
    return None


def llm_checkpoint(config, weight="full_sft", model=None, optimizer=None,
                   epoch=0, step=0, wandb=None, save_dir="../checkpoints", **kwargs):
    """Save or load LLM training checkpoint.
    - With model: save mode
    - Without model: load mode
    """
    if model is not None:
        _save_checkpoint(config, weight, model, optimizer, epoch, step,
                         wandb, save_dir, clean_fn=None, **kwargs)
    else:
        return _load_checkpoint(config, weight, save_dir)


def vlm_checkpoint(config, weight="sft_vlm", model=None, optimizer=None,
                   epoch=0, step=0, wandb=None, save_dir="../checkpoints", **kwargs):
    """Save or load VLM training checkpoint.
    On save: strips vision_encoder weights (they're frozen and shared).
    """
    def clean_vlm_state(state_dict):
        return {
            k: v for k, v in state_dict.items()
            if not k.startswith("vision_encoder.")
        }

    if model is not None:
        _save_checkpoint(config, weight, model, optimizer, epoch, step,
                         wandb, save_dir, clean_fn=clean_vlm_state, **kwargs)
    else:
        return _load_checkpoint(config, weight, save_dir)
