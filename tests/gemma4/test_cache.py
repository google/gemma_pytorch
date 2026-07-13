"""Tests for Gemma 4 Cache abstractions."""

import pytest
import torch
from gemma.gemma4.cache import Cache, DynamicCache, StaticCache
from gemma.gemma4.config import Gemma4Config


def test_base_cache_not_implemented():
    """Test that base Cache methods raise NotImplementedError."""
    cache = Cache()
    with pytest.raises(NotImplementedError):
        cache.update(torch.randn(1), torch.randn(1), 0)
    with pytest.raises(NotImplementedError):
        cache.get_seq_length(0)
    with pytest.raises(NotImplementedError):
        cache.get_max_length()
    with pytest.raises(NotImplementedError):
        cache.reorder_cache(torch.tensor([0]))


def test_dynamic_cache():
    """Test DynamicCache operations."""
    cache = DynamicCache()
    assert cache.get_max_length() is None
    assert cache.get_seq_length(0) == 0

    batch_size = 2
    num_heads = 4
    head_dim = 16
    seq_len = 3

    k1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v1 = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # First update
    k_out, v_out = cache.update(k1, v1, 0)
    assert k_out.shape == (batch_size, num_heads, seq_len, head_dim)
    assert cache.get_seq_length(0) == seq_len

    # Second update
    k2 = torch.randn(batch_size, num_heads, 2, head_dim)
    v2 = torch.randn(batch_size, num_heads, 2, head_dim)
    k_out2, v_out2 = cache.update(k2, v2, 0)

    assert k_out2.shape == (batch_size, num_heads, seq_len + 2, head_dim)
    assert cache.get_seq_length(0) == seq_len + 2

    # Reorder test
    beam_idx = torch.tensor([1, 0])
    cache.reorder_cache(beam_idx)
    assert torch.allclose(cache.key_cache[0][0], k_out2[1])
    assert torch.allclose(cache.key_cache[0][1], k_out2[0])


def test_static_cache():
    """Test StaticCache operations."""
    config = Gemma4Config(num_hidden_layers=2, num_key_value_heads=4, head_dim=16)
    max_batch_size = 2
    max_cache_len = 10
    device = torch.device("cpu")

    cache = StaticCache(config, max_batch_size, max_cache_len, device)

    assert cache.get_max_length() == max_cache_len
    assert cache.get_seq_length(0) == 0

    batch_size = 1
    seq_len = 3
    k1 = torch.randn(batch_size, 4, seq_len, 16)
    v1 = torch.randn(batch_size, 4, seq_len, 16)

    k_out, v_out = cache.update(k1, v1, 0)

    assert k_out.shape == (batch_size, 4, seq_len, 16)

    # Update seen tokens which is usually done by the model
    cache.seen_tokens += seq_len
    assert cache.get_seq_length(0) == seq_len

    # Reorder test
    beam_idx = torch.tensor([0, 0])
    cache.reorder_cache(beam_idx)
    # Check that shapes are unchanged
    assert cache.key_cache[0].shape == (max_batch_size, 4, max_cache_len, 16)
