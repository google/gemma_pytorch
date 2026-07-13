"""Gemma 4 Audio Model implementation."""

from typing import Optional

import torch
from torch import nn

from gemma import config as gemma_config
from gemma.gemma4_model import Gemma4RMSNorm

from gemma.gemma4_audio.audio_attention import (
    Gemma4AudioAttention,
    Gemma4AudioRelPositionalEncoding,
)
from gemma.gemma4_audio.audio_layers import (
    Gemma4AudioFeedForward,
    Gemma4AudioLightConv1d,
    Gemma4AudioSubSampleConvProjection,
)


class Gemma4AudioLayer(nn.Module):
    """A single layer of the audio transformer model."""

    def __init__(self, config: gemma_config.AudioConfig):
        super().__init__()
        self.feed_forward1 = Gemma4AudioFeedForward(config)
        self.feed_forward2 = Gemma4AudioFeedForward(config)
        self.self_attn = Gemma4AudioAttention(config)
        self.lconv1d = Gemma4AudioLightConv1d(config)
        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_out = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_clipping = config.gradient_clipping

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply a single audio transformer layer.

        Args:
            x: Audio features (B, seq_len, hidden_size).
            pos_emb: Relative positional embeddings.
            mask: Optional attention mask (5D).

        Returns:
            Processed audio features.
        """
        x = self.feed_forward1(x)
        residual = x
        x = torch.clamp(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.norm_pre_attn(x)

        x = self.self_attn(x, pos_emb, mask)

        x = torch.clamp(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.norm_post_attn(x)
        x = x + residual

        x = self.lconv1d(x)
        x = self.feed_forward2(x)
        x = torch.clamp(x, -self.gradient_clipping, self.gradient_clipping)
        return self.norm_out(x)


class Gemma4AudioModel(nn.Module):
    """An audio encoder based on the Universal Speech Model architecture."""

    def __init__(self, config: gemma_config.AudioConfig):
        super().__init__()
        self.config = config
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(config)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [Gemma4AudioLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.output_proj = nn.Linear(config.hidden_size, config.output_proj_dims)

    def _convert_4d_mask_to_blocked_5d(self, mask_4d: torch.Tensor) -> torch.Tensor:
        """Convert a 4D attention mask to a 5D blocked format."""
        batch_size, _, seq_len, _ = mask_4d.shape
        chunk_size = self.config.attention_chunk_size
        max_past_horizon = self.config.attention_context_left - 1
        max_future_horizon = self.config.attention_context_right

        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        padded_seq_len = num_blocks * chunk_size
        pad_amount = padded_seq_len - seq_len

        if pad_amount > 0:
            mask_4d = torch.nn.functional.pad(mask_4d, (0, pad_amount, 0, pad_amount))

        mask_5d = mask_4d.view(batch_size, 1, num_blocks, chunk_size, padded_seq_len)
        mask_5d = torch.nn.functional.pad(
            mask_5d, (max_past_horizon, max_future_horizon)
        )

        block_starts = torch.arange(num_blocks, device=mask_4d.device) * chunk_size
        offsets = torch.arange(
            chunk_size + max_past_horizon + max_future_horizon, device=mask_4d.device
        )

        kv_indices = block_starts.unsqueeze(1) + offsets.unsqueeze(0)
        kv_indices = kv_indices.view(1, 1, num_blocks, 1, -1).expand(
            batch_size, 1, num_blocks, chunk_size, -1
        )

        return torch.gather(mask_5d, dim=-1, index=kv_indices)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for the Gemma 4 Audio model.

        Args:
            input_features: Raw input features (B, T, F).
            attention_mask: Optional mask (B, T).

        Returns:
            Encoded audio representation projected to text embedding space.
        """
        x, mask = self.subsample_conv_projection(input_features, attention_mask)
        pos_emb = self.rel_pos_enc(x)

        if mask is not None:
            mask_4d = mask.unsqueeze(1).unsqueeze(-1) * mask.unsqueeze(1).unsqueeze(1)
            mask_5d = self._convert_4d_mask_to_blocked_5d(mask_4d)
        else:
            mask_5d = None

        for layer in self.layers:
            x = layer(x, pos_emb, mask_5d)

        return self.output_proj(x)
