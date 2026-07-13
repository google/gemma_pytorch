"""Audio attention for Gemma 4."""

from __future__ import annotations

import torch
from torch import nn

from .config import Gemma4Config


class Gemma4AudioCrossAttention(nn.Module):
    """Audio cross-attention mechanism for Gemma 4."""

    def __init__(self, config: Gemma4Config):
        """Initialize Gemma4AudioCrossAttention."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.audio_hidden_size = config.audio_config.hidden_size
        self.num_heads = config.num_attention_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.audio_hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.audio_hidden_size, self.hidden_size, bias=False)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            kdim=self.hidden_size,
            vdim=self.hidden_size,
            batch_first=True,
        )

    def forward(
        self, hidden_states: torch.Tensor, audio_states: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for audio cross-attention."""
        q = self.q_proj(hidden_states)
        k = self.k_proj(audio_states)
        v = self.v_proj(audio_states)

        attn_output, _ = self.self_attn(
            query=q,
            key=k,
            value=v,
        )
        return attn_output
