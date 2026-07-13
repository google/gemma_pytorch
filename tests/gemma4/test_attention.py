"""Tests for Gemma 4 Attention."""

import torch
from gemma.gemma4.attention import Gemma4Attention
from gemma.gemma4.config import Gemma4Config
from gemma.gemma4.cache import DynamicCache


def test_attention_forward():
    """Test standard forward pass of attention."""
    config = Gemma4Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=4096,
        global_attn_layers=[0],
    )

    attention = Gemma4Attention(config, layer_idx=0)

    hidden_states = torch.randn(2, 5, 64)
    out, past_kv = attention(hidden_states)

    assert out.shape == (2, 5, 64)
    assert isinstance(past_kv, tuple)
    assert past_kv[0].shape == (2, 2, 5, 16)
    assert past_kv[1].shape == (2, 2, 5, 16)


def test_attention_with_dynamic_cache():
    """Test attention with dynamic cache."""
    config = Gemma4Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    attention = Gemma4Attention(config, layer_idx=0)

    cache = DynamicCache()
    hidden_states_1 = torch.randn(1, 3, 64)
    out1, cache1 = attention(hidden_states_1, past_key_value=cache)

    assert cache1.get_seq_length(0) == 3

    hidden_states_2 = torch.randn(1, 1, 64)
    pos_ids = torch.tensor([[3]], dtype=torch.long)
    out2, cache2 = attention(
        hidden_states_2, position_ids=pos_ids, past_key_value=cache1
    )

    assert out2.shape == (1, 1, 64)
    assert cache2.get_seq_length(0) == 4


def test_attention_sliding_window():
    """Test attention with sliding window logic."""
    config = Gemma4Config(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=2,
        global_attn_layers=[],
    )

    attention = Gemma4Attention(config, layer_idx=0)
    hidden_states = torch.randn(1, 4, 32)
    mask = torch.zeros(1, 1, 4, 4)

    out, past_kv = attention(hidden_states, attention_mask=mask)
    assert out.shape == (1, 4, 32)

    dummy_past = (torch.randn(1, 2, 2, 16), torch.randn(1, 2, 2, 16))
    # Provide correctly sized attention mask for sequence length of 6 (2 past + 4 new)
    mask2 = torch.zeros(1, 1, 4, 6)
    out2, past_kv2 = attention(
        hidden_states, attention_mask=mask2, past_key_value=dummy_past
    )
    assert past_kv2[0].shape == (1, 2, 6, 16)
