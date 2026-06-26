"""
Multimodal datasets for VLM training (Parquet format).

Provides:
  - VLMDataset: image+text pairs with row-group-aware shuffling
  - VLMRLDataset: variant for GRPO training (prompt-only, no labels)

Based on the VLM dataset pipeline with efficient row-group caching.
"""

import io
import json
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from .text_dataset import pre_processing_chat, post_processing_chat


# ── VLMDataset ──

class VLMDataset(Dataset):
    """
    Vision-Language dataset stored as Parquet files.

    Each row has:
      - conversations: JSON string of multi-turn chat
      - image_bytes: binary JPEG image data

    Features:
      - Row-group-aware shuffling: shuffle row groups, then rows within
        each group. This achieves randomness while keeping cache hit rates
        near 100% for sequential access.
      - Image preprocessing via the encoder's preprocess function.
    """

    def __init__(
        self,
        parquet_path,
        tokenizer,
        preprocess=None,
        max_length=512,
        image_special_token="<|image_pad|>",
        image_token_len=64,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids

        # Load the entire parquet table into memory for O(1) per-sample access.
        # This matches MiniMind-V's approach — row-group disk streaming kills
        # throughput when access is shuffled. Cost: ~8-10 GB RAM for 2.9M rows.
        self.table = pq.read_table(parquet_path)
        self.num_rows = self.table.num_rows

    def __len__(self):
        return self.num_rows

    def create_chat_prompt(self, conversations):
        """Format conversations with image placeholder token."""
        messages = []
        for turn in conversations:
            content = turn["content"]
            if turn.get("role") != "system":
                content = content.replace("<image>", self.image_special_token)
            messages.append({"role": turn["role"], "content": content})
        tools = (
            conversations[0]["functions"]
            if (
                conversations
                and conversations[0]["role"] == "system"
                and conversations[0].get("functions")
            )
            else None
        )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )

    def generate_labels(self, input_ids):
        """Labels: mask everything except assistant responses."""
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(
                    start, min(end + len(self.eos_id), self.max_length)
                ):
                    labels[j] = input_ids[j]
                i = (
                    end + len(self.eos_id)
                    if end < len(input_ids)
                    else len(input_ids)
                )
            else:
                i += 1
        return labels

    def __getitem__(self, index: int):
        # Direct column access — same as MiniMind-V (O(1), no copy)
        conv_json = self.table["conversations"][index].as_py()
        image_bytes = self.table["image_bytes"][index].as_py()

        conversations = json.loads(conv_json)
        image_bytes = (
            image_bytes
            if isinstance(image_bytes, list)
            else [image_bytes]
        )

        conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (
            self.max_length - len(input_ids)
        )
        labels = self.generate_labels(input_ids)

        # Process images
        image_inputs_list = []
        for img_bytes in image_bytes:
            img = Image.open(io.BytesIO(img_bytes))
            image_inputs_list.append(self.preprocess(img))

        # Stack image data
        if hasattr(image_inputs_list[0], "keys"):
            image_data = {
                k: torch.cat([inp[k] for inp in image_inputs_list], dim=0)
                for k in image_inputs_list[0].keys()
            }
        else:
            image_data = torch.stack(image_inputs_list)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            image_data,
        )


# ── VLMRLDataset (for GRPO) ──

class VLMRLDataset(Dataset):
    """
    VLM dataset for GRPO reinforcement learning.

    Returns prompt-only data (no labels) — the model generates responses
    during rollout. Visual inputs are preprocessed and carried through.

    Returns dict: {"prompt": str, "pixel_values": tensor_dict}
    """

    def __init__(
        self,
        parquet_path,
        tokenizer,
        preprocess=None,
        max_length=512,
        image_special_token="<|image_pad|>",
        image_token_len=64,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len

        # Load entire table into memory — same approach as VLMDataset
        self.table = pq.read_table(parquet_path)
        self.num_rows = self.table.num_rows

    def __len__(self):
        return self.num_rows

    def __getitem__(self, index: int):
        # Direct column access — same as MiniMind-V (O(1), no copy)
        conv_json = self.table["conversations"][index].as_py()
        image_bytes = self.table["image_bytes"][index].as_py()

        conversations = json.loads(conv_json)
        image_bytes = (
            image_bytes
            if isinstance(image_bytes, list)
            else [image_bytes]
        )

        # Build prompt-only format (stop at last user message)
        conversations = pre_processing_chat(conversations, add_system_ratio=0.5)

        # Tokenize without assistant response (add_generation_prompt=True)
        messages = []
        for turn in conversations:
            content = turn["content"]
            if turn.get("role") != "system":
                content = content.replace("<image>", self.image_special_token)
            messages.append({"role": turn["role"], "content": content})

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_text = post_processing_chat(prompt_text)

        # Process images
        image_inputs_list = [
            self.preprocess(Image.open(io.BytesIO(img_bytes)))
            for img_bytes in image_bytes
        ]
        if hasattr(image_inputs_list[0], "keys"):
            pixel_values = {
                k: torch.cat([inp[k] for inp in image_inputs_list], dim=0)
                for k in image_inputs_list[0].keys()
            }
        else:
            pixel_values = torch.stack(image_inputs_list)

        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).input_ids[0]

        return {
            "prompt": prompt_text,
            "prompt_ids": prompt_ids,
            "pixel_values": pixel_values,
        }
