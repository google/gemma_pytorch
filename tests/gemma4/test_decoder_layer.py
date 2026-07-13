"""Tests for Gemma 4 Decoder Layer."""

import torch
from gemma.gemma4.decoder_layer import Gemma4DecoderLayer
from gemma.gemma4.config import Gemma4Config


def test_decoder_layer_with_moe():
    """Test Gemma4DecoderLayer using MoE."""
    config = Gemma4Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
    )

    layer = Gemma4DecoderLayer(config, layer_idx=0)
    hidden_states = torch.randn(2, 5, 64)

    out, present_kv, router_logits = layer(hidden_states)

    assert out.shape == (2, 5, 64)
    assert isinstance(present_kv, tuple)
    assert router_logits is not None
    assert router_logits.shape == (10, 4)


def test_decoder_layer_with_mlp():
    """Test Gemma4DecoderLayer using standard MLP."""
    config = Gemma4Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=1,  # Forces standard MLP
    )

    layer = Gemma4DecoderLayer(config, layer_idx=0)
    hidden_states = torch.randn(2, 5, 64)

    out, present_kv, router_logits = layer(hidden_states)

    assert out.shape == (2, 5, 64)
    assert router_logits is None
