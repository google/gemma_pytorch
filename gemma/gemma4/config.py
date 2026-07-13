"""Configuration classes for Gemma 4 PyTorch implementation."""

from __future__ import annotations

from typing import Any


class Gemma4VisionConfig:
    """Configuration for Gemma 4 Vision sub-model."""

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        patch_size: int = 14,
        image_size: int = 224,
        **kwargs: Any,
    ):
        """Initialize the Gemma4VisionConfig."""
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        for key, value in kwargs.items():
            setattr(self, key, value)


class Gemma4AudioConfig:
    """Configuration for Gemma 4 Audio sub-model."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        **kwargs: Any,
    ):
        """Initialize the Gemma4AudioConfig."""
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        for key, value in kwargs.items():
            setattr(self, key, value)


class Gemma4Config:
    """Configuration for native PyTorch Gemma 4."""

    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_size: int = 2048,
        num_hidden_layers: int = 18,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 1,
        intermediate_size: int = 16384,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 256,
        pad_token_id: int = 0,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        router_jitter_noise: float = 0.0,
        sliding_window: int = 4096,
        global_attn_layers: list[int] | None = None,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 1.0,
        vision_config: dict[str, Any] | None = None,
        audio_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize the Gemma4Config."""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.head_dim = head_dim
        self.pad_token_id = pad_token_id

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_jitter_noise = router_jitter_noise

        self.sliding_window = sliding_window
        self.global_attn_layers = global_attn_layers or []

        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor

        if vision_config is None:
            self.vision_config = Gemma4VisionConfig()
        else:
            self.vision_config = Gemma4VisionConfig(**vision_config)

        if audio_config is None:
            self.audio_config = Gemma4AudioConfig()
        else:
            self.audio_config = Gemma4AudioConfig(**audio_config)

        for key, value in kwargs.items():
            setattr(self, key, value)
