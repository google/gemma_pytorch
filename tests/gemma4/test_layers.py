"""Tests for Gemma 4 layers."""

import torch
from gemma.gemma4.layers import Gemma4RMSNorm, Gemma4MLP
from gemma.gemma4.config import Gemma4Config


def test_rms_norm():
    """Test Gemma4RMSNorm output and shape."""
    dim = 16
    norm = Gemma4RMSNorm(dim)

    x = torch.randn(2, 4, dim)
    out = norm(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype

    # Check properties of RMS norm approximately
    # The variance should be close to 1
    var = (out.float() ** 2).mean(-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2)


def test_mlp():
    """Test Gemma4MLP output and shape."""
    config = Gemma4Config(hidden_size=16, intermediate_size=32)
    mlp = Gemma4MLP(config)

    x = torch.randn(2, 4, 16)
    out = mlp(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
