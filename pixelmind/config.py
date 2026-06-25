import math
from transformers import PretrainedConfig


class PixelMindConfig(PretrainedConfig):
    """
    Unified config for PixelMind LLM and VLM. No MoE — dense only.

    Inherits from PretrainedConfig for HuggingFace Transformers compatibility.
    """
    model_type = "pixelmind"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        dropout: float = 0.0,
        vocab_size: int = 6400,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        flash_attn: bool = True,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = None,
        hidden_act: str = "silu",
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1e6,
        tie_word_embeddings: bool = True,
        inference_rope_scaling: bool = False,
        # ── VLM parameters ──
        image_special_token: str = "<|image_pad|>",
        image_ids: list = None,
        image_hidden_size: int = 768,
        image_token_len: int = 64,
        vision_encoder_name: str = "siglip2",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ── LLM params ──
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.flash_attn = flash_attn
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size or (
            math.ceil(hidden_size * math.pi / 64) * 64
        )
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.inference_rope_scaling = inference_rope_scaling

        # YaRN rope scaling (applied when inference_rope_scaling=True)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn",
        } if self.inference_rope_scaling else None

        # ── VLM params ──
        self.image_special_token = image_special_token
        self.image_ids = image_ids or [12]
        self.image_hidden_size = image_hidden_size
        self.image_token_len = image_token_len
        self.vision_encoder_name = vision_encoder_name
