"""Tests for Gemma 4 root models."""

import torch
from gemma.gemma4.modeling import Gemma4MultiModalProjector, Gemma4ForCausalLM
from gemma.gemma4.config import Gemma4Config


def test_multimodal_projector():
    """Test Gemma4MultiModalProjector."""
    config = Gemma4Config(hidden_size=64, vision_config={"hidden_size": 128})

    projector = Gemma4MultiModalProjector(config)
    image_features = torch.randn(2, 4, 128)

    out = projector(image_features)
    assert out.shape == (2, 4, 64)


def test_gemma4_for_causal_lm_text_only():
    """Test Gemma4ForCausalLM forward pass with text only."""
    config = Gemma4Config(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        num_experts=1,
    )

    model = Gemma4ForCausalLM(config)
    input_ids = torch.randint(0, 1000, (2, 5))

    logits, _ = model(input_ids)

    assert logits.shape == (2, 5, 1000)


def test_gemma4_for_causal_lm_multimodal():
    """Test Gemma4ForCausalLM with multimodal inputs."""
    config = Gemma4Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=128,
        num_experts=1,
        vision_config={
            "hidden_size": 32,
            "patch_size": 14,
            "image_size": 28,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        },
        audio_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        },
    )

    model = Gemma4ForCausalLM(config)

    input_ids = torch.randint(0, 100, (2, 5))
    pixel_values = torch.randn(2, 3, 28, 28)  # 4 patches
    audio_values = torch.randn(2, 1, 1000)  # Output len ~4 depending on conv stride

    logits, _ = model(input_ids, pixel_values=pixel_values, audio_values=audio_values)

    # 5 text + 4 vision + audio tokens
    assert logits.shape[0] == 2
    assert logits.shape[-1] == 100
    assert logits.shape[1] > 5


def test_gemma4_generate():
    """Test standard generation loop wrapper."""
    config = Gemma4Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=1,
    )

    model = Gemma4ForCausalLM(config)
    input_ids = torch.randint(0, 100, (1, 3))

    out = model.generate(input_ids, max_new_tokens=2)
    assert out.shape == (1, 5)
