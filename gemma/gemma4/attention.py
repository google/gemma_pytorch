"""Attention for Gemma 4."""

from __future__ import annotations

import math

import torch
from torch import nn

from .cache import Cache
from .config import Gemma4Config
from .rope import Gemma4RotaryEmbedding, apply_rotary_pos_emb


class Gemma4Attention(nn.Module):
    """Attention mechanism for Gemma 4."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        """Initialize Gemma4Attention."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window
        self.is_global = layer_idx in config.global_attn_layers

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = Gemma4RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.sliding_window * 2
            if config.sliding_window
            else 8192,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | Cache | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | Cache | None]:
        """Forward pass for attention."""
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if isinstance(past_key_value, tuple):
                kv_seq_len += past_key_value[0].shape[-2]
            else:
                kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        if position_ids is None:
            position_ids = (
                torch.arange(
                    kv_seq_len - q_len,
                    kv_seq_len,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                .unsqueeze(0)
                .expand(bsz, -1)
            )

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            if isinstance(past_key_value, tuple):
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                past_key_value = (key_states, value_states)
            else:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx
                )
        else:
            past_key_value = (key_states, value_states)

        num_key_value_groups = self.num_heads // self.num_key_value_heads
        key_states = key_states[:, :, None, :, :].expand(
            bsz,
            self.num_key_value_heads,
            num_key_value_groups,
            key_states.shape[2],
            self.head_dim,
        )
        key_states = key_states.reshape(
            bsz, self.num_heads, key_states.shape[3], self.head_dim
        )

        value_states = value_states[:, :, None, :, :].expand(
            bsz,
            self.num_key_value_heads,
            num_key_value_groups,
            value_states.shape[2],
            self.head_dim,
        )
        value_states = value_states.reshape(
            bsz, self.num_heads, value_states.shape[3], self.head_dim
        )

        # SDPA fallback
        is_causal = attention_mask is None and q_len > 1

        if attention_mask is None and self.sliding_window is None:
            attn_output = nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=is_causal,
            )
        else:
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            if not self.is_global and self.sliding_window is not None:
                min_val = torch.finfo(attn_weights.dtype).min
                window_mask = torch.ones_like(attn_weights, dtype=torch.bool).tril(
                    diagonal=0
                )
                window_mask = torch.logical_and(
                    window_mask,
                    torch.ones_like(attn_weights, dtype=torch.bool).triu(
                        diagonal=-self.sliding_window + 1
                    ),
                )
                attn_weights = torch.where(
                    window_mask,
                    attn_weights,
                    torch.tensor(
                        min_val, dtype=attn_weights.dtype, device=attn_weights.device
                    ),
                )

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value
