"""Audio layers and submodules for Gemma 4."""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from gemma import config as gemma_config
from gemma.gemma4_model import Gemma4RMSNorm
from gemma.gemma4_audio.audio_attention import Gemma4ClippableLinear


class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):
    """A single convolutional projection layer for audio subsampling."""

    def __init__(self, in_channels: int, channels: int, norm_eps: float):
        super().__init__()
        # JAX params: kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
        self.conv = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        self.norm = nn.LayerNorm(channels, eps=norm_eps, elementwise_affine=False)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the subsample convolution projection layer.

        Args:
            x: Input tensor. Expected shape in JAX was (B, 1, T, F), which PyTorch handles differently
               PyTorch Conv2d expects (B, C, H, W). We assume x comes as (B, C, T, F).
            mask: Optional mask tensor (B, T)
        """
        if mask is not None:
            # mask: (B, T) -> x: (B, C, T, F). Broadcsting: mask.unsqueeze(1).unsqueeze(-1)
            x = x * mask.unsqueeze(1).unsqueeze(-1)

        x = self.conv(x)

        # apply LayerNorm over the last dimension (channels in JAX, F in PyTorch originally, but after conv it's the channels)
        # In PyTorch after conv, shape is (B, channels, T', F').
        # JAX applied layer norm over `channels`. So we permute, apply norm, permute back.
        x = x.permute(0, 2, 3, 1)  # (B, T', F', C)
        x = self.norm(x)
        x = F.relu(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, T', F')

        if mask is not None:
            mask = mask[:, ::2]  # subsample mask

        return x, mask


class Gemma4AudioSubSampleConvProjection(nn.Module):
    """Full convolutional projection module for audio subsampling."""

    def __init__(self, config: gemma_config.AudioConfig):
        super().__init__()
        c0, c1 = config.subsampling_conv_channels
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(
            1, c0, config.rms_norm_eps
        )
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(
            c0, c1, config.rms_norm_eps
        )

        proj_input_dim = (c0 // 4) * c1  # JAX used c0 // 4 * c1
        self.input_proj_linear = nn.Linear(
            proj_input_dim, config.hidden_size, bias=False
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the full subsample convolution projection.

        Args:
            x: Audio input features (B, T, F).
            mask: Optional attention mask (B, T).
        """
        x = x.unsqueeze(1)  # (B, 1, T, F)

        x, mask = self.layer0(x, mask)
        x, mask = self.layer1(x, mask)

        # x is now (B, c1, T', F')
        batch_size, channels, seq_len, feat_dim = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, seq_len, -1)  # (B, T', F'*c1)

        return self.input_proj_linear(x), mask


class Gemma4AudioFeedForward(nn.Module):
    """Feed forward network used in the audio tower."""

    def __init__(self, config: gemma_config.AudioConfig):
        super().__init__()
        self.ffw_layer_1 = Gemma4ClippableLinear(
            config.hidden_size,
            config.hidden_size * 4,
            use_clipped_linears=config.use_clipped_linears,
        )
        self.ffw_layer_2 = Gemma4ClippableLinear(
            config.hidden_size * 4,
            config.hidden_size,
            use_clipped_linears=config.use_clipped_linears,
        )
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_layer_norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed forward network."""
        residual = x
        x = torch.clamp(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.pre_layer_norm(x)
        x = self.ffw_layer_1(x)
        x = F.silu(x)
        x = self.ffw_layer_2(x)
        x = torch.clamp(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.post_layer_norm(x)
        x = x * self.post_layer_scale
        return residual + x


class Gemma4AudioCausalConv1d(nn.Module):
    """Causal 1D convolution layer for audio processing."""

    def __init__(self, config: gemma_config.AudioConfig):
        super().__init__()
        self.kernel_size = config.conv_kernel_size
        self.left_pad = self.kernel_size - 1

        # depthwise 1D conv
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=self.kernel_size,
            groups=config.hidden_size,
            bias=False,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal 1D convolution.

        Args:
            x: Input tensor (B, seq_len, hidden_size).
        """
        # (B, seq_len, C) -> (B, C, seq_len)
        x = x.transpose(1, 2)

        # Pad on the left for causality
        x = F.pad(x, (self.left_pad, 0))

        x = self.conv(x)

        # (B, C, seq_len) -> (B, seq_len, C)
        return x.transpose(1, 2)


class Gemma4AudioLightConv1d(nn.Module):
    """Lightweight 1D convolution module for audio."""

    def __init__(self, config: gemma_config.AudioConfig):
        super().__init__()
        self.linear_start = Gemma4ClippableLinear(
            config.hidden_size,
            config.hidden_size * 2,
            use_clipped_linears=config.use_clipped_linears,
        )
        self.linear_end = Gemma4ClippableLinear(
            config.hidden_size,
            config.hidden_size,
            use_clipped_linears=config.use_clipped_linears,
        )
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(config)
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conv_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_clipping = config.gradient_clipping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lightweight 1D convolution."""
        residual = x
        x = self.pre_layer_norm(x)
        x = self.linear_start(x)

        x, gate = torch.split(x, x.size(-1) // 2, dim=-1)
        x = x * torch.sigmoid(gate)

        x = self.depthwise_conv1d(x)
        x = torch.clamp(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.conv_norm(x)
        x = F.silu(x)

        x = self.linear_end(x)
        return residual + x
