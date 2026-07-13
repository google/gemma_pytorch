"""Vision modules for Gemma 4."""

from __future__ import annotations

import torch
from torch import nn

from .config import Gemma4VisionConfig
from .layers import Gemma4RMSNorm


class Gemma4VisionEmbeddings(nn.Module):
    """Vision embeddings for Gemma 4."""

    def __init__(self, config: Gemma4VisionConfig):
        """Initialize Gemma4VisionEmbeddings."""
        super().__init__()
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for vision embeddings."""
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        positions = (
            torch.arange(self.num_patches, device=pixel_values.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        position_embeds = self.position_embedding(positions)

        return patch_embeds + position_embeds


class Gemma4VisionAttention(nn.Module):
    """Vision self-attention for Gemma 4."""

    def __init__(self, config: Gemma4VisionConfig):
        """Initialize Gemma4VisionAttention."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for vision attention."""
        bsz, seq_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_output = nn.functional.scaled_dot_product_attention(q, k, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.hidden_size)
        )
        return self.o_proj(attn_output)


class Gemma4VisionEncoderLayer(nn.Module):
    """Vision encoder layer for Gemma 4."""

    def __init__(self, config: Gemma4VisionConfig):
        """Initialize Gemma4VisionEncoderLayer."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma4VisionAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.intermediate_size, self.hidden_size),
        )
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for vision encoder layer."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma4VisionModel(nn.Module):
    """Vision model for Gemma 4."""

    def __init__(self, config: Gemma4VisionConfig):
        """Initialize Gemma4VisionModel."""
        super().__init__()
        self.config = config
        self.embeddings = Gemma4VisionEmbeddings(config)
        self.layers = nn.ModuleList(
            [Gemma4VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_layernorm = Gemma4RMSNorm(config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for vision model."""
        hidden_states = self.embeddings(pixel_values)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.post_layernorm(hidden_states)
