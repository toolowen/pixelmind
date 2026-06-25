"""
PixelMindBlock: a single Transformer decoder layer.

Pre-Norm architecture with GQA attention + SwiGLU FFN.
"""

from torch import nn
from .attention import Attention, RMSNorm
from .feedforward import FeedForward


class PixelMindBlock(nn.Module):
    """
    Single Transformer Decoder Block.

    Architecture:
        input → RMSNorm → Attention (GQA + RoPE) → residual
               → RMSNorm → FeedForward (SwiGLU) → residual
    """
    def __init__(self, layer_id: int, config):
        super().__init__()
        self.layer_id = layer_id

        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        # Self-attention with residual
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + residual

        # FeedForward with residual
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value
