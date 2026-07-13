"""Audio attention and relative positional encoding for Gemma 4."""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from gemma import config as gemma_config
from gemma.gemma4_model import Gemma4RMSNorm


class ConstVar(nn.Module):
    """A wrapper for a constant tensor in PyTorch."""

    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.register_buffer("value", tensor)

    def __call__(self) -> torch.Tensor:
        return self.value


class Gemma4ClippableLinear(nn.Linear):
    """Linear layer with optional gradient clipping."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_clipped_linears: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.use_clipped_linears = use_clipped_linears

    # Gradient clipping in PyTorch is usually done globally, but we
    # keep the class for compatibility.


class Gemma4AudioRelPositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding for the audio encoder."""

    def __init__(self, config: gemma_config.AudioConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.context_size = (
            config.attention_chunk_size
            + config.attention_context_left
            - 1
            + config.attention_context_right
        )
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(
            num_timescales - 1, 1
        )

        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        self.inv_timescales = ConstVar(inv_timescales.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply relative positional encoding.

        Args:
            x: The input tensor to determine dtype and device.

        Returns:
            The positional embeddings.
        """
        position_ids = torch.arange(
            self.context_size - 1, -1, -1, dtype=torch.float32, device=x.device
        )
        position_ids = position_ids.unsqueeze(-1)
        scaled_time = position_ids * self.inv_timescales()
        pos_embed = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return pos_embed.to(x.dtype)


def _convert_to_block(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Reshapes the input into chunks/blocks for block-wise attention."""
    batch_size, seq_len, num_heads, head_dim = x.shape
    num_blocks = (seq_len + chunk_size - 1) // chunk_size
    pad_len = num_blocks * chunk_size - seq_len

    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))

    return x.view(batch_size, num_blocks, chunk_size, num_heads, head_dim)


def _extract_block_context(
    x: torch.Tensor, attn: "Gemma4AudioAttention"
) -> torch.Tensor:
    """Extract the left context block for block-wise attention."""
    batch_size, seq_len, num_heads, head_dim = x.shape

    # Pad seq_len dimension
    x = F.pad(
        x,
        (
            0,
            0,
            0,
            0,
            attn.max_past_horizon,
            attn.max_future_horizon + attn.chunk_size - 1,
        ),
    )
    num_blocks = (seq_len + attn.chunk_size - 1) // attn.chunk_size

    blocks = []
    for i in range(num_blocks):
        start = i * attn.chunk_size
        block = x[:, start : start + attn.context_size, :, :]
        blocks.append(block)

    return torch.stack(blocks, dim=1)


def _rel_shift(x: torch.Tensor, context_size: int) -> torch.Tensor:
    """Perform relative shift on attention scores."""
    batch_size, num_heads, num_blocks, block_size, position_length = x.shape

    pad_len = context_size + 1 - position_length
    if pad_len > 0:
        x = F.pad(x, (0, pad_len))

    x = x.view(batch_size, num_heads, num_blocks, block_size * (context_size + 1))
    x = x[..., : block_size * context_size]
    return x.view(batch_size, num_heads, num_blocks, block_size, context_size)


def _compute_audio_attention_outputs(
    attn: "Gemma4AudioAttention",
    qkv: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    pos_emb: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute the multi-head attention outputs for audio."""
    q, k, v = qkv
    batch_size, seq_len, _ = q.shape[:3]

    q = q * attn.q_scale * F.softplus(attn.per_dim_scale)
    k = k * attn.k_scale

    q_block = _convert_to_block(q, attn.chunk_size)
    k_context = _extract_block_context(k, attn)
    v_context = _extract_block_context(v, attn)

    num_blocks = q_block.shape[1]

    # rel_k: (context_size, num_heads, head_dim)
    rel_k = (
        attn.relative_k_proj(pos_emb)
        .view(-1, attn.num_heads, attn.head_dim)
        .to(q.dtype)
    )

    queries = q_block.permute(
        0, 3, 1, 2, 4
    )  # (B, num_heads, num_blocks, chunk_size, head_dim)
    keys = k_context.permute(
        0, 3, 1, 4, 2
    )  # (B, num_heads, num_blocks, head_dim, context_size)

    matrix_ac = torch.matmul(queries, keys)

    queries_flat = queries.reshape(batch_size, attn.num_heads, -1, attn.head_dim)
    rel_k_t = rel_k.permute(1, 2, 0)  # (num_heads, head_dim, context_size)

    matrix_bd = torch.matmul(queries_flat, rel_k_t)
    matrix_bd = matrix_bd.view(
        batch_size, attn.num_heads, num_blocks, attn.chunk_size, -1
    )
    matrix_bd = _rel_shift(matrix_bd, attn.context_size)

    attn_weights = matrix_ac + matrix_bd
    attn_weights = attn_weights / attn.softcap
    attn_weights = torch.tanh(attn_weights) * attn.softcap

    if mask is not None:
        attn_weights = torch.where(
            mask.bool(),
            attn_weights,
            torch.tensor(
                attn.invalid_logits_value,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            ),
        )

    attn_weights = F.softmax(attn_weights, dim=-1).to(v_context.dtype)
    values = v_context.permute(
        0, 3, 1, 2, 4
    )  # (B, num_heads, num_blocks, context_size, head_dim)

    out = torch.matmul(attn_weights, values)
    out = out.permute(0, 2, 3, 1, 4)  # (B, num_blocks, chunk_size, num_heads, head_dim)
    out = out.reshape(batch_size, num_blocks * attn.chunk_size, -1)
    out = out[:, :seq_len, :]

    return attn.post(out)


class Gemma4AudioAttention(nn.Module):
    """Chunked local attention with relative position bias for audio."""

    def __init__(self, config: gemma_config.AudioConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )
        self.softcap = config.attention_logit_cap
        self.invalid_logits_value = config.attention_invalid_logits_value

        hs = config.hidden_size
        self.q_proj = Gemma4ClippableLinear(
            hs,
            self.num_heads * self.head_dim,
            use_clipped_linears=config.use_clipped_linears,
        )
        self.k_proj = Gemma4ClippableLinear(
            hs,
            self.num_heads * self.head_dim,
            use_clipped_linears=config.use_clipped_linears,
        )
        self.v_proj = Gemma4ClippableLinear(
            hs,
            self.num_heads * self.head_dim,
            use_clipped_linears=config.use_clipped_linears,
        )
        self.post = Gemma4ClippableLinear(
            hs, hs, use_clipped_linears=config.use_clipped_linears
        )

        self.relative_k_proj = nn.Linear(hs, self.num_heads * self.head_dim, bias=False)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the multi-head attention for audio inputs.

        Args:
            x: Input tensor (B, seq_len, hidden_size)
            pos_emb: Positional embeddings.
            mask: Attention mask.

        Returns:
            Attention outputs.
        """
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        return _compute_audio_attention_outputs(self, (q, k, v), pos_emb, mask)
