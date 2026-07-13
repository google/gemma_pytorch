"""Layers for Gemma 4."""

from __future__ import annotations

import torch
from torch import nn

from .config import Gemma4Config


class Gemma4RMSNorm(nn.Module):
    """RMSNorm for Gemma 4."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize Gemma4RMSNorm."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Gemma4MLP(nn.Module):
    """MLP for Gemma 4."""

    def __init__(self, config: Gemma4Config):
        """Initialize Gemma4MLP."""
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
