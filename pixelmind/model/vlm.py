"""
PixelMind VLM Model
====================
Vision-Language Model extending PixelMindForCausalLM with a visual encoder
and projector. The key mechanism: treat images as a "foreign language" by
replacing <|image_pad|> placeholder tokens with projected visual embeddings.

Core algo change vs. LLM: ~50 lines (count_vision_proj + vision injection).
"""

from typing import Optional, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..config import PixelMindConfig
from .llm import PixelMindForCausalLM
from .attention import precompute_freqs_cis
from .vision import (
    BaseVisionEncoder,
    MMVisionProjector,
    build_vision_encoder,
)


class PixelMind(PixelMindForCausalLM):
    """
    PixelMind VLM — PixelMindForCausalLM + Vision.

    Architecture:
        Image → [Vision Encoder] → features [B, N, hidden]
                → [Projector] → projected tokens [B, N, d_model]
                → Replace <|image_pad|> in LLM hidden states
                → LLM Transformer → lm_head → text output

    The vision encoder is frozen during training; only the projector
    (and optionally LLM layers) are trained.
    """
    config_class = PixelMindConfig

    def __init__(
        self,
        config: PixelMindConfig = None,
        vision_encoder: Optional[BaseVisionEncoder] = None,
        vision_encoder_path: Optional[str] = None,
    ):
        self.config = config or PixelMindConfig()
        super().__init__(self.config)

        # Vision encoder (pluggable)
        if vision_encoder is not None:
            self.vision_encoder = vision_encoder
        elif vision_encoder_path is not None:
            self.vision_encoder = build_vision_encoder(
                self.config.vision_encoder_name, vision_encoder_path
            )
        else:
            # Default: use SigLIP2 from ./model/siglip2-base-p32-256-ve/
            default_path = "./model/siglip2-base-p32-256-ve"
            self.vision_encoder = build_vision_encoder(
                self.config.vision_encoder_name, default_path
            )

        # Projection: encoder hidden_size → LLM hidden_size
        self.vision_proj = MMVisionProjector(
            in_dim=self.vision_encoder.hidden_size,
            out_dim=self.config.hidden_size,
            target_tokens=self.config.image_token_len,
        )

    # ── Static helpers (compatibility wrappers) ──

    @staticmethod
    def image2tensor(image, encoder: BaseVisionEncoder):
        """Convert PIL Image to tensor dict using encoder's preprocessor."""
        return encoder.preprocess(image)

    @staticmethod
    def get_image_embeddings(image_inputs, vision_encoder: BaseVisionEncoder):
        """Encode raw image inputs through the vision encoder."""
        # Squeeze batch dim if needed
        if hasattr(image_inputs, "keys"):
            image_inputs = {
                k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v
                for k, v in image_inputs.items()
            }
        return vision_encoder(image_inputs)

    # ── Vision token injection ──

    @torch.compiler.disable
    def count_vision_proj(
        self,
        tokens,
        hidden_states,
        vision_tensors=None,
        seqlen=512,
    ):
        """
        Replace <|image_pad|> marker tokens with projected visual features.

        This is the core VLM mechanism: for each batch item, find runs of
        the image_pad token in the token sequence, replace the corresponding
        positions in the hidden states with projected vision embeddings.
        """
        if vision_tensors is None or not self.config.image_ids:
            return hidden_states

        marker = self.config.image_ids[0]
        vision_features = vision_tensors

        # Ensure [B, num_images, token_len, hidden] shape
        if vision_features.dim() == 3:
            vision_features = vision_features.unsqueeze(1)

        results = []
        for b in range(hidden_states.size(0)):
            hb = hidden_states[b]  # [seq, hidden]
            seq = tokens[b].tolist()
            k = 0  # image index counter
            i = 0  # token index
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vision_features.size(1):
                        # Replace the marker span with vision tokens
                        hb = torch.cat(
                            (
                                hb[:start],
                                vision_features[b][k][: i - start],
                                hb[i:],
                            ),
                            dim=0,
                        )[:seqlen]
                        k += 1
                else:
                    i += 1
            results.append(hb)
        return torch.stack(results)

    # ── Forward Pass ──

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape

        # Reset wrapped past_key_values (transformers >= 5.x)
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)

        start_pos = (
            past_key_values[0][0].shape[1]
            if past_key_values[0] is not None
            else 0
        )

        # Token embeddings
        hidden_states = self.model.dropout(
            self.model.embed_tokens(input_ids)
        )

        # ── Vision injection (only at start_pos=0, i.e. first forward) ──
        if pixel_values is not None and start_pos == 0:
            # Handle different pixel_value formats
            if hasattr(pixel_values, "keys"):
                # Dict format (from SigLIP processor)
                sample_val = next(iter(pixel_values.values()))
                if sample_val.ndim == 5:
                    bs, num = sample_val.shape[:2]
                    # Flatten batch+image dims for encoder
                    flat_inputs = {
                        k: v.flatten(0, 1)
                        for k, v in pixel_values.items()
                    }
                    features = self.get_image_embeddings(
                        flat_inputs, self.vision_encoder
                    )
                    vision_tensors = self.vision_proj(features).view(
                        bs, num, self.config.image_token_len, -1
                    )
                else:
                    features = self.get_image_embeddings(
                        pixel_values, self.vision_encoder
                    )
                    vision_tensors = self.vision_proj(features)
            else:
                # Raw tensor format [bs, num_images, C, H, W]
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                bs, num, c, im_h, im_w = pixel_values.shape
                vision_tensors = torch.stack(
                    [
                        self.vision_proj(
                            self.get_image_embeddings(
                                pixel_values[:, i, :, :, :],
                                self.vision_encoder,
                            )
                        )
                        for i in range(num)
                    ],
                    dim=1,
                )

            # Inject vision tokens into hidden states
            hidden_states = self.count_vision_proj(
                tokens=input_ids,
                hidden_states=hidden_states,
                vision_tensors=vision_tensors,
                seqlen=input_ids.shape[1],
            )

        # ── RoPE (recompute if lost) ──
        if self.model.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.model.freqs_cos = freqs_cos.to(hidden_states.device)
            self.model.freqs_sin = freqs_sin.to(hidden_states.device)
        position_embeddings = (
            self.model.freqs_cos[start_pos: start_pos + seq_length],
            self.model.freqs_sin[start_pos: start_pos + seq_length],
        )

        # ── Transformer blocks ──
        presents = []
        for layer, past_key_value in zip(self.model.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        # ── Final norm + lm_head ──
        hidden_states = self.model.norm(hidden_states)

        # Dummy gradient through projector for DDP (ensures grad sync)
        _dummy = sum(p.sum() for p in self.vision_proj.parameters()) * 0

        # Slice for logits
        if isinstance(logits_to_keep, int):
            slice_indices = (
                slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
            )
        else:
            slice_indices = logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # ── Loss ──
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
            "past_key_values": presents,
            "hidden_states": hidden_states,
        }

    # ── Generate (override to handle pixel_values) ──

    def generate(self, *args, num_return_sequences=1, **kwargs):
        """
        Generate with pixel_values support.
        Repeats pixel_values for num_return_sequences when >1.
        """
        if num_return_sequences > 1 and "pixel_values" in kwargs:
            pv = kwargs["pixel_values"]
            if hasattr(pv, "keys"):
                kwargs["pixel_values"] = {
                    k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1)))
                    for k, v in pv.items()
                }
            else:
                kwargs["pixel_values"] = pv.repeat(
                    num_return_sequences, *([1] * (pv.ndim - 1))
                )
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)
