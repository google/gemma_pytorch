"""Tests for Gemma 4 param utils."""

import torch
import numpy as np
from gemma.gemma4.utils_params import translate_jax_to_pytorch


def test_translate_jax_to_pytorch():
    """Test translate_jax_to_pytorch."""

    # Mock JAX-like structure using numpy arrays
    class MockJaxArray:
        def __init__(self, data):
            self.data = np.array(data)

        def __array__(self, dtype=None):
            return self.data

    jax_params = {
        "attention.query.kernel": MockJaxArray([[1.0, 2.0], [3.0, 4.0]]),
        "norm.scale": MockJaxArray([1.0, 1.0]),
        "bias": [0.1, 0.2],
    }

    pt_params = translate_jax_to_pytorch(jax_params)

    assert "attention.query.weight" in pt_params
    assert pt_params["attention.query.weight"].shape == (2, 2)
    # Check transpose
    expected_weight = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float64)
    assert torch.allclose(pt_params["attention.query.weight"], expected_weight)

    assert "norm.weight" in pt_params
    assert pt_params["norm.weight"].shape == (2,)

    assert "bias" in pt_params
    assert pt_params["bias"].shape == (2,)
