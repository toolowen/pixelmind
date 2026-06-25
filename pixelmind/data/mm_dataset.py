"""
Multimodal datasets for VLM training (Parquet format).

Provides:
  - VLMDataset: image+text pairs with row-group-aware shuffling
  - VLMRLDataset: variant for GRPO training (prompt-only, no labels)

Based on the VLM dataset pipeline with efficient row-group caching.
"""

import bisect
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
        self.parquet_file = pq.ParquetFile(parquet_path)
        self.num_rows = self.parquet_file.metadata.num_rows
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

        # Precompute row-group offsets
        row_offsets = []
        cum = 0
        for i in range(self.parquet_file.metadata.num_row_groups):
            row_offsets.append(cum)
            cum += self.parquet_file.metadata.row_group(i).num_rows
        self._rg_offsets = row_offsets
        self._num_rg = len(row_offsets)

        # Row-group shuffle: shuffle groups, then rows within each group
        rg_order = list(range(self._num_rg))
        random.shuffle(rg_order)
        perm = []
        for rg in rg_order:
            start = self._rg_offsets[rg]
            nr = self.parquet_file.metadata.row_group(rg).num_rows
            rows = list(range(start, start + nr))
            random.shuffle(rows)
            perm.extend(rows)
        self._perm_map = perm

        # Row-group cache (single-worker scenarios hit nearly always)
        self._cached_rg_idx = -1
        self._cached_rg_table = None

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

    # ── Row-group lookup ──

    def _global_row(self, sampled_index: int):
        """Map shuffled index back to global parquet row number."""
        return self._perm_map[sampled_index]

    def _row_group_of(self, global_row: int):
        """Find which row group a global row belongs to."""
        rg_idx = bisect.bisect_right(self._rg_offsets, global_row) - 1
        return rg_idx, self._rg_offsets[rg_idx]

    def _read_row(self, global_row: int):
        """Read a single row from parquet with row-group caching."""
        rg_idx, rg_start = self._row_group_of(global_row)
        if rg_idx != self._cached_rg_idx:
            self._cached_rg_table = self.parquet_file.read_row_group(
                rg_idx, columns=["conversations", "image_bytes"]
            ).to_pydict()
            self._cached_rg_idx = rg_idx
        offset = global_row - rg_start
        return (
            self._cached_rg_table["conversations"][offset],
            self._cached_rg_table["image_bytes"][offset],
        )

    def __getitem__(self, index: int):
        global_row = self._global_row(index)
        conv_json, image_bytes = self._read_row(global_row)
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
        self.parquet_file = pq.ParquetFile(parquet_path)
        self.num_rows = self.parquet_file.metadata.num_rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len

        # Row-group offsets
        row_offsets = []
        cum = 0
        for i in range(self.parquet_file.metadata.num_row_groups):
            row_offsets.append(cum)
            cum += self.parquet_file.metadata.row_group(i).num_rows
        self._rg_offsets = row_offsets
        self._num_rg = len(row_offsets)

        # Shuffle groups + rows
        rg_order = list(range(self._num_rg))
        random.shuffle(rg_order)
        perm = []
        for rg in rg_order:
            start = self._rg_offsets[rg]
            nr = self.parquet_file.metadata.row_group(rg).num_rows
            rows = list(range(start, start + nr))
            random.shuffle(rows)
            perm.extend(rows)
        self._perm_map = perm

        self._cached_rg_idx = -1
        self._cached_rg_table = None

    def __len__(self):
        return self.num_rows

    def _global_row(self, sampled_index: int):
        return self._perm_map[sampled_index]

    def _row_group_of(self, global_row: int):
        rg_idx = bisect.bisect_right(self._rg_offsets, global_row) - 1
        return rg_idx, self._rg_offsets[rg_idx]

    def _read_row(self, global_row: int):
        rg_idx, rg_start = self._row_group_of(global_row)
        if rg_idx != self._cached_rg_idx:
            self._cached_rg_table = self.parquet_file.read_row_group(
                rg_idx, columns=["conversations", "image_bytes"]
            ).to_pydict()
            self._cached_rg_idx = rg_idx
        offset = global_row - rg_start
        return (
            self._cached_rg_table["conversations"][offset],
            self._cached_rg_table["image_bytes"][offset],
        )

    def __getitem__(self, index: int):
        global_row = self._global_row(index)
        conv_json, image_bytes = self._read_row(global_row)
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
