"""Tests for Gemma 4 Mixture of Experts."""

import torch
from gemma.gemma4.moe import Gemma4MoE, Gemma4MoERouter, calculate_load_balancing_loss
from gemma.gemma4.config import Gemma4Config


def test_moe_router():
    """Test Gemma4MoERouter."""
    config = Gemma4Config(
        hidden_size=32, num_experts=4, num_experts_per_tok=2, router_jitter_noise=0.1
    )

    router = Gemma4MoERouter(config)
    router.train()  # Enable jitter

    hidden_states = torch.randn(2, 3, 32)
    weights, selected, logits = router(hidden_states)

    assert weights.shape == (2, 3, 2)
    assert selected.shape == (2, 3, 2)
    assert logits.shape == (2, 3, 4)

    # Sum of weights per token should be 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3))


def test_moe_layer():
    """Test Gemma4MoE forward pass."""
    config = Gemma4Config(
        hidden_size=32,
        intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
    )

    moe = Gemma4MoE(config)
    hidden_states = torch.randn(2, 5, 32)

    out, logits = moe(hidden_states)

    assert out.shape == (2, 5, 32)
    assert logits.shape == (10, 4)  # (batch * seq_len, num_experts)


def test_calculate_load_balancing_loss():
    """Test calculate_load_balancing_loss."""
    logits = torch.randn(10, 4)  # 10 tokens, 4 experts
    loss = calculate_load_balancing_loss(logits, num_experts=4, top_k=2)

    assert loss.dim() == 0  # scalar
    assert loss.item() >= 0
