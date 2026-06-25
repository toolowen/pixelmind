"""
Attention components for PixelMind: RMSNorm, RoPE, GQA Attention.

Based on native PyTorch implementation — no flash-attn dependency,
falls back to PyTorch's scaled_dot_product_attention when available.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn


# ─────────────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self._norm(x.float())).type_as(x)


# ─────────────────────────────────────────────────────────────────────
# Rotary Position Embedding (RoPE) with YaRN support
# ─────────────────────────────────────────────────────────────────────

def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: dict = None,
):
    """
    Precompute cosine and sine frequency tables for RoPE.

    Args:
        dim: head dimension
        end: maximum sequence length
        rope_base: RoPE theta base (default 1e6 for extended context)
        rope_scaling: optional YaRN scaling config dict

    Returns:
        (freqs_cos, freqs_sin): precomputed cos/sin tables of shape [end, dim]
    """
    freqs = 1.0 / (
        rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    attn_factor = 1.0

    if rope_scaling is not None:
        # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        if end / orig_max > 1.0:
            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                    2 * math.log(rope_base)
                )

            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0, 1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embeddings to query and key tensors."""

    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (
        (q * cos.unsqueeze(unsqueeze_dim))
        + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    ).to(q.dtype)
    k_embed = (
        (k * cos.unsqueeze(unsqueeze_dim))
        + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    ).to(k.dtype)
    return q_embed, k_embed


# ─────────────────────────────────────────────────────────────────────
# KV-Head Replication (for Grouped-Query Attention)
# ─────────────────────────────────────────────────────────────────────

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads to match the number of query heads for GQA.

    Args:
        x: [batch, seq_len, num_kv_heads, head_dim]
        n_rep: number of query heads per KV head

    Returns:
        [batch, seq_len, num_q_heads, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


# ─────────────────────────────────────────────────────────────────────
# Grouped-Query Attention
# ─────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """
    Grouped-Query Attention (GQA) with QK-Norm, RoPE, and optional flash attention.

    Default: 8 query heads, 4 KV heads (2:1 ratio).
    """
    def __init__(self, config):
        super().__init__()
        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.dropout = config.dropout

        # Projections
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

        # QK Normalization (PixelMind style)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Flash attention detection
        self.flash = (
            hasattr(F, "scaled_dot_product_attention") and config.flash_attn
        )

    def forward(
        self,
        x,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        bsz, seq_len, _ = x.shape

        # Project
        xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # QK Normalization
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # RoPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # KV cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # Transpose to [B, H, S, D] for attention
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # Attention computation
        if (
            self.flash
            and seq_len > 1
            and (not self.is_causal or past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            # Flash attention (PyTorch SDPA)
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            # Manual attention
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                scores[:, :, :, -seq_len:] += (
                    torch.full((seq_len, seq_len), float("-inf"), device=scores.device)
                    .triu(1)
                )
            if attention_mask is not None:
                scores += (
                    (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
                )
            attn = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq))
            output = attn @ xv

        # Merge heads
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
