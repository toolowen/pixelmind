"""
SwiGLU FeedForward layer for PixelMind.

Standard SwiGLU MLP as used in Llama / Qwen style architectures.
"""

from torch import nn
from transformers.activations import ACT2FN


class FeedForward(nn.Module):
    """
    SwiGLU FeedForward Network.

    Architecture: gate_proj(x) * up_proj(x) → down_proj
    (SiLU activation on gate)
    """
    def __init__(self, config, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = nn.Linear(
            config.hidden_size, intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, intermediate_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
