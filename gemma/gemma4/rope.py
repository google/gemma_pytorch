"""Rotary Positional Embeddings for Gemma 4."""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Gemma4RotaryEmbedding(nn.Module):
    """Gemma 4 Rotary Embedding."""

    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: torch.device | None = None,
    ):
        """Initialize Gemma4RotaryEmbedding."""
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Set cos and sin cache."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class Gemma4RotaryEmbedding2D(nn.Module):
    """Gemma 4 2D Rotary Embedding for Vision patches."""

    inv_freq: torch.Tensor

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: torch.device | None = None,
    ):
        """Initialize Gemma4RotaryEmbedding2D."""
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 2D RoPE splits the dim in half for height and width
        self.half_dim = dim // 2

        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.half_dim, 2, dtype=torch.int64).float().to(device)
                / self.half_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for 2D RoPE."""
        device = x.device
        dtype = x.dtype

        t_h = torch.arange(height, device=device, dtype=torch.int64).type_as(
            self.inv_freq
        )
        t_w = torch.arange(width, device=device, dtype=torch.int64).type_as(
            self.inv_freq
        )

        freqs_h = torch.outer(t_h, self.inv_freq)
        freqs_w = torch.outer(t_w, self.inv_freq)

        freqs_h = (
            freqs_h.unsqueeze(1).expand(-1, width, -1).reshape(-1, self.half_dim // 2)
        )
        freqs_w = (
            freqs_w.unsqueeze(0).expand(height, -1, -1).reshape(-1, self.half_dim // 2)
        )

        freqs = torch.cat((freqs_h, freqs_w), dim=-1)
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos().to(dtype), emb.sin().to(dtype)
