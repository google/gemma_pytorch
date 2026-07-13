"""Tests for Gemma 4 Configuration."""

from gemma.gemma4.config import Gemma4Config, Gemma4VisionConfig, Gemma4AudioConfig


def test_vision_config_defaults():
    """Test default values for Gemma4VisionConfig."""
    config = Gemma4VisionConfig()
    assert config.hidden_size == 1152
    assert config.intermediate_size == 4304
    assert config.num_hidden_layers == 27
    assert config.num_attention_heads == 16
    assert config.patch_size == 14
    assert config.image_size == 224


def test_audio_config_defaults():
    """Test default values for Gemma4AudioConfig."""
    config = Gemma4AudioConfig()
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 12


def test_gemma4_config_defaults():
    """Test default values for Gemma4Config."""
    config = Gemma4Config()
    assert config.vocab_size == 256000
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 18
    assert config.num_attention_heads == 8
    assert config.num_key_value_heads == 1
    assert config.intermediate_size == 16384
    assert config.rms_norm_eps == 1e-6
    assert config.head_dim == 256
    assert config.pad_token_id == 0
    assert config.num_experts == 8
    assert config.num_experts_per_tok == 2
    assert config.router_jitter_noise == 0.0
    assert config.sliding_window == 4096
    assert config.global_attn_layers == []
    assert config.rope_theta == 10000.0
    assert config.partial_rotary_factor == 1.0
    assert isinstance(config.vision_config, Gemma4VisionConfig)
    assert isinstance(config.audio_config, Gemma4AudioConfig)


def test_gemma4_config_kwargs():
    """Test kwargs pass through for config classes."""
    v_config = Gemma4VisionConfig(custom_val="v_test")
    assert getattr(v_config, "custom_val", None) == "v_test"

    a_config = Gemma4AudioConfig(custom_val="a_test")
    assert getattr(a_config, "custom_val", None) == "a_test"

    config = Gemma4Config(custom_val="test")
    assert getattr(config, "custom_val", None) == "test"


def test_gemma4_config_with_dicts():
    """Test Gemma4Config init with dict configs."""
    v_dict = {"hidden_size": 128}
    a_dict = {"hidden_size": 256}
    config = Gemma4Config(vision_config=v_dict, audio_config=a_dict)

    assert config.vision_config.hidden_size == 128
    assert config.audio_config.hidden_size == 256
