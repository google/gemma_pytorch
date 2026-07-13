"""Tests for Gemma 4 Audio modules."""

import torch
from gemma.gemma4.audio_layers import (
    Gemma4AudioFeatureExtractor,
    Gemma4AudioEncoderBlock,
)
from gemma.gemma4.audio_attention import Gemma4AudioCrossAttention
from gemma.gemma4.audio import Gemma4AudioModel
from gemma.gemma4.config import Gemma4Config, Gemma4AudioConfig


def test_audio_feature_extractor():
    """Test Gemma4AudioFeatureExtractor."""
    config = Gemma4AudioConfig(hidden_size=64)
    extractor = Gemma4AudioFeatureExtractor(config)

    # Input format: (batch, seq_len) -> expanded to (batch, 1, seq_len)
    input_values_2d = torch.randn(2, 1000)
    out_2d = extractor(input_values_2d)
    assert out_2d.shape[0] == 2
    assert out_2d.shape[2] == 64

    # Input format: (batch, channel, seq_len)
    input_values_3d = torch.randn(2, 1, 1000)
    out_3d = extractor(input_values_3d)
    assert out_3d.shape[0] == 2
    assert out_3d.shape[2] == 64


def test_audio_encoder_block():
    """Test Gemma4AudioEncoderBlock."""
    config = Gemma4AudioConfig(hidden_size=64, num_attention_heads=4)
    block = Gemma4AudioEncoderBlock(config)

    hidden_states = torch.randn(2, 50, 64)
    out = block(hidden_states)

    assert out.shape == (2, 50, 64)


def test_audio_cross_attention():
    """Test Gemma4AudioCrossAttention."""
    config = Gemma4Config(
        hidden_size=128, num_attention_heads=4, audio_config={"hidden_size": 64}
    )

    attn = Gemma4AudioCrossAttention(config)
    text_states = torch.randn(2, 10, 128)
    audio_states = torch.randn(2, 50, 64)

    out = attn(text_states, audio_states)
    assert out.shape == (2, 10, 128)


def test_audio_model():
    """Test full Gemma4AudioModel."""
    config = Gemma4Config(
        hidden_size=128,
        audio_config={
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
        },
    )

    model = Gemma4AudioModel(config)
    audio_values = torch.randn(2, 1, 1000)

    out = model(audio_values)
    # The output is projected to text hidden size (128)
    assert out.shape[0] == 2
    assert out.shape[2] == 128
