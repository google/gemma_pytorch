"""Tests for Gemma 4 Vision modules."""

import torch
from gemma.gemma4.vision import (
    Gemma4VisionEmbeddings,
    Gemma4VisionAttention,
    Gemma4VisionEncoderLayer,
    Gemma4VisionModel,
)
from gemma.gemma4.config import Gemma4VisionConfig


def test_vision_embeddings():
    """Test Gemma4VisionEmbeddings."""
    config = Gemma4VisionConfig(
        hidden_size=64,
        patch_size=14,
        image_size=28,  # 2x2 patches
    )

    emb = Gemma4VisionEmbeddings(config)

    # 2 images, 3 channels, 28x28
    pixel_values = torch.randn(2, 3, 28, 28)
    out = emb(pixel_values)

    # Num patches = (28/14)**2 = 4
    assert out.shape == (2, 4, 64)


def test_vision_attention():
    """Test Gemma4VisionAttention."""
    config = Gemma4VisionConfig(hidden_size=64, num_attention_heads=4)

    attn = Gemma4VisionAttention(config)
    hidden_states = torch.randn(2, 4, 64)

    out = attn(hidden_states)
    assert out.shape == (2, 4, 64)


def test_vision_encoder_layer():
    """Test Gemma4VisionEncoderLayer."""
    config = Gemma4VisionConfig(
        hidden_size=64, num_attention_heads=4, intermediate_size=128
    )

    layer = Gemma4VisionEncoderLayer(config)
    hidden_states = torch.randn(2, 4, 64)

    out = layer(hidden_states)
    assert out.shape == (2, 4, 64)


def test_vision_model():
    """Test full Gemma4VisionModel."""
    config = Gemma4VisionConfig(
        hidden_size=64,
        patch_size=14,
        image_size=28,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )

    model = Gemma4VisionModel(config)
    pixel_values = torch.randn(2, 3, 28, 28)

    out = model(pixel_values)
    # Output should have shape (batch, num_patches, hidden_size)
    assert out.shape == (2, 4, 64)
