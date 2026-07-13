"""Decoder layer for Gemma 4."""

from __future__ import annotations

import torch
from torch import nn

from .attention import Gemma4Attention
from .cache import Cache
from .config import Gemma4Config
from .layers import Gemma4MLP, Gemma4RMSNorm
from .moe import Gemma4MoE


class Gemma4DecoderLayer(nn.Module):
    """Decoder layer for Gemma 4."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        """Initialize Gemma4DecoderLayer."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma4Attention(config=config, layer_idx=layer_idx)

        self.mlp: nn.Module
        if config.num_experts > 1:
            self.mlp = Gemma4MoE(config)
        else:
            self.mlp = Gemma4MLP(config)

        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | Cache | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | Cache | None,
        torch.Tensor | None,
    ]:
        """Forward pass for decoder layer."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if isinstance(self.mlp, Gemma4MoE):
            hidden_states, router_logits = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            router_logits = None

        hidden_states = residual + hidden_states

        return hidden_states, present_key_value, router_logits
