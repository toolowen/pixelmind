"""
PixelMind LLM Backbone
======================
Decoder-only Transformer with GQA, RoPE, SwiGLU. No MoE — dense only.

Provides:
  - PixelMindModel: raw Transformer (embeddings → blocks → norm)
  - PixelMindForCausalLM: model + lm_head with generation support
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin

from ..config import PixelMindConfig
from .attention import RMSNorm, precompute_freqs_cis
from .block import PixelMindBlock


# ─────────────────────────────────────────────────────────────────────
# PixelMindModel — Transformer backbone
# ─────────────────────────────────────────────────────────────────────

class PixelMindModel(nn.Module):
    """Decoder-only Transformer: Embed → Dropout → N×Blocks → RMSNorm."""

    def __init__(self, config: PixelMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            PixelMindBlock(idx, config) for idx in range(self.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Precompute RoPE frequency tables (registered as non-persistent buffer)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape

        # Reset past_key_values if it comes wrapped (transformers >= 5.x)
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        start_pos = (
            past_key_values[0][0].shape[1]
            if past_key_values[0] is not None
            else 0
        )

        # Token embeddings
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # Recompute RoPE buffers if lost during meta-device init (transformers >= 5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.freqs_cos = freqs_cos.to(hidden_states.device)
            self.freqs_sin = freqs_sin.to(hidden_states.device)

        position_embeddings = (
            self.freqs_cos[start_pos: start_pos + seq_length],
            self.freqs_sin[start_pos: start_pos + seq_length],
        )

        # Pass through all transformer blocks
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


# ─────────────────────────────────────────────────────────────────────
# PixelMindForCausalLM — Model with LM head
# ─────────────────────────────────────────────────────────────────────

class PixelMindForCausalLM(PreTrainedModel, GenerationMixin):
    """PixelMind LLM with language modeling head and native generate()."""
    config_class = PixelMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: PixelMindConfig = None):
        self.config = config or PixelMindConfig()
        super().__init__(self.config)

        self.model = PixelMindModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        **kwargs,
    ):
        hidden_states, past_key_values = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )

        # Keep only last N logits
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            slice_indices = logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": past_key_values,
            "hidden_states": hidden_states,
        }

    # ── Native generate() — key educational feature ──

    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.0,
        **kwargs,
    ):
        input_ids = kwargs.pop("input_ids", inputs)
        input_ids = input_ids.repeat(num_return_sequences, 1)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(num_return_sequences, 1)

        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )

        if streamer:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            past_len = (
                past_key_values[0][0].shape[1] if past_key_values else 0
            )
            outputs = self.forward(
                input_ids[:, past_len:],
                attention_mask,
                past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_ones(attention_mask.shape[0], 1),
                ], dim=-1)

            # Temperature scaling + sampling
            logits = outputs["logits"][:, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(
                        score > 0,
                        score / repetition_penalty,
                        score * repetition_penalty,
                    )

            # Top-K filtering
            if top_k > 0:
                threshold = torch.topk(logits, top_k)[0][..., -1, None]
                logits[logits < threshold] = float("-inf")

            # Top-P (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                mask = cum_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                logits[mask.scatter(1, sorted_indices, mask)] = float("-inf")

            # Sample or argmax
            if do_sample:
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), num_samples=1
                )
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Handle finished sequences
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs["past_key_values"] if use_cache else None

            if streamer:
                streamer.put(next_token.cpu())

            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        if streamer:
            streamer.end()

        if kwargs.get("return_kv"):
            return {"generated_ids": input_ids, "past_kv": past_key_values}
        return input_ids
