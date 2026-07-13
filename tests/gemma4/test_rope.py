"""Tests for Gemma 4 RoPE."""

import torch
from gemma.gemma4.rope import (
    rotate_half,
    apply_rotary_pos_emb,
    Gemma4RotaryEmbedding,
    Gemma4RotaryEmbedding2D,
)


def test_rotate_half():
    """Test rotate_half."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    out = rotate_half(x)
    assert torch.allclose(out, expected)


def test_apply_rotary_pos_emb():
    """Test apply_rotary_pos_emb."""
    seq_len = 2
    dim = 4
    q = torch.randn(1, 1, seq_len, dim)
    k = torch.randn(1, 1, seq_len, dim)

    # Dummy cos/sin
    cos = torch.ones(seq_len, dim)
    sin = torch.zeros(seq_len, dim)

    pos_ids = torch.arange(seq_len).unsqueeze(0)  # [1, 2]

    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, pos_ids, unsqueeze_dim=1)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

    # Since sin=0 and cos=1, it should just be identity
    assert torch.allclose(q, q_out)
    assert torch.allclose(k, k_out)


def test_rotary_embedding():
    """Test Gemma4RotaryEmbedding."""
    dim = 16
    rope = Gemma4RotaryEmbedding(dim, max_position_embeddings=10)

    x = torch.randn(1, 1, 5, dim)
    cos, sin = rope(x, seq_len=5)

    assert cos.shape == (5, dim)
    assert sin.shape == (5, dim)

    # Test caching growth
    cos2, sin2 = rope(x, seq_len=15)
    assert cos2.shape == (15, dim)
    assert sin2.shape == (15, dim)


def test_rotary_embedding_2d():
    """Test Gemma4RotaryEmbedding2D."""
    dim = 16
    rope = Gemma4RotaryEmbedding2D(dim)

    x = torch.randn(1, 1, 4, dim)  # not really used for shape
    h, w = 2, 2
    cos, sin = rope(x, height=h, width=w)

    assert cos.shape == (h * w, dim)
    assert sin.shape == (h * w, dim)
