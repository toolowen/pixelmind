"""
Collate functions for DataLoader batching.

Provides:
  - text_collate_fn: stacks (input_ids, labels)
  - vlm_collate_fn: stacks (input_ids, labels, pixel_values)
  - vlm_rl_collate_fn: collects dicts for GRPO
"""

import torch


def text_collate_fn(batch):
    """Collate text-only batch: (input_ids, labels) pairs."""
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return input_ids, labels


def vlm_collate_fn(batch):
    """
    Collate VLM batch: (input_ids, labels, pixel_values) tuples.

    Handles both dict-format pixel_values (SigLIP processor output)
    and raw tensor-format pixel_values.
    """
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]

    if hasattr(pixel_data[0], "keys"):
        # Dict format: stack each key independently
        pixel_values = {
            k: torch.stack([d[k] for d in pixel_data])
            for k in pixel_data[0].keys()
        }
    else:
        # Raw tensor format
        pixel_values = torch.stack(pixel_data)

    return input_ids, labels, pixel_values


def vlm_rl_collate_fn(batch):
    """
    Collate VLM RL batch for GRPO.

    Returns dict with:
      - prompts: list[str]
      - prompt_ids: padded tensor [B, max_len]
      - pixel_values: batched pixel data
      - raw_images_list: list[bytes] — raw JPEG bytes for VLM reward judge
    """
    prompts = [b["prompt"] for b in batch]
    prompt_ids_list = [b["prompt_ids"] for b in batch]
    pixel_data = [b["pixel_values"] for b in batch]
    raw_images_list = [b["raw_images"] for b in batch]

    # Pad prompt_ids
    prompt_ids = torch.nn.utils.rnn.pad_sequence(
        prompt_ids_list, batch_first=True, padding_value=0
    ).to(prompt_ids_list[0].device)

    # Stack pixel values
    if hasattr(pixel_data[0], "keys"):
        pixel_values = {
            k: torch.stack([d[k] for d in pixel_data])
            for k in pixel_data[0].keys()
        }
    else:
        pixel_values = torch.stack(pixel_data)

    return {
        "prompts": prompts,
        "prompt_ids": prompt_ids,
        "pixel_values": pixel_values,
        "raw_images_list": raw_images_list,  # list of bytes, one per sample
    }
