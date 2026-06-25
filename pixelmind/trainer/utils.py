"""
Training utilities for PixelMind: logger, learning rate schedule,
distributed mode detection, random seed, batch sampler.
"""

import os
import random
import math

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def Logger(content):
    """Print only on rank 0 (main process) in distributed mode."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(content)


def is_main_process():
    """Check if current process is the main one."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_lr(current_step, total_steps, lr):
    """
    Cosine learning rate schedule with warmup floor.
    Returns lr * (0.1 + 0.45 * (1 + cos(...))) — min at 0.1×, max at 1.0×.
    """
    return lr * (
        0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps))
    )


def init_distributed_mode():
    """
    Detect DDP mode from environment variables.
    Returns local_rank (0 for non-DDP, else local GPU index).
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # single-GPU or CPU mode

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_params(model, config, ignore_patterns=None):
    """
    Log model parameter count in millions.
    Excludes parameters matching ignore_patterns from the count.
    """
    if ignore_patterns is None:
        ignore_patterns = set()

    def should_count(name):
        return not any(p in name for p in ignore_patterns)

    total = sum(
        p.numel()
        for n, p in model.named_parameters()
        if should_count(n)
    ) / 1e6
    Logger(f"Model Params: {total:.2f}M")


class SkipBatchSampler(Sampler):
    """
    Wraps a sampler (or index list) to skip batches, enabling
    training resumption at an exact step boundary.
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (
            len(self.sampler) + self.batch_size - 1
        ) // self.batch_size
        return max(0, total_batches - self.skip_batches)
