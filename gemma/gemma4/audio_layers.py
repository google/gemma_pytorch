"""Audio layers for Gemma 4."""

from __future__ import annotations

import torch
from torch import nn

from .config import Gemma4AudioConfig


class Gemma4AudioFeatureExtractor(nn.Module):
    """Audio feature extractor using 1D temporal convolutions."""

    def __init__(self, config: Gemma4AudioConfig):
        """Initialize Gemma4AudioFeatureExtractor."""
        super().__init__()
        self.conv1 = nn.Conv1d(
            1, config.hidden_size, kernel_size=10, stride=5, bias=False
        )
        self.conv2 = nn.Conv1d(
            config.hidden_size, config.hidden_size, kernel_size=3, stride=2, bias=False
        )
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for audio feature extractor."""
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)

        hidden_states = self.conv1(input_values)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.activation(hidden_states)

        return hidden_states.transpose(1, 2)


class Gemma4AudioEncoderBlock(nn.Module):
    """Audio-specific transformer encoder block."""

    def __init__(self, config: Gemma4AudioConfig):
        """Initialize Gemma4AudioEncoderBlock."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for audio encoder block."""
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        attn_output, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states
