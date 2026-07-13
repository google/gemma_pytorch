"""Audio modules for Gemma 4."""

from __future__ import annotations

import torch
from torch import nn

from .audio_layers import Gemma4AudioEncoderBlock, Gemma4AudioFeatureExtractor
from .config import Gemma4Config


class Gemma4AudioModel(nn.Module):
    """Audio model for Gemma 4."""

    def __init__(self, config: Gemma4Config):
        """Initialize Gemma4AudioModel."""
        super().__init__()
        self.config = config
        self.feature_extractor = Gemma4AudioFeatureExtractor(config.audio_config)
        self.layers = nn.ModuleList(
            [
                Gemma4AudioEncoderBlock(config.audio_config)
                for _ in range(config.audio_config.num_hidden_layers)
            ]
        )
        self.audio_projector = nn.Linear(
            config.audio_config.hidden_size, config.hidden_size, bias=False
        )

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for audio model."""
        hidden_states = self.feature_extractor(audio_values)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.audio_projector(hidden_states)
