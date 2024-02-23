from pathlib import Path
import dataclasses
import immutabledict
import torch
from typing import Optional, Union


_STR_DTYPE_TO_TORCH_DTYPE = immutabledict.immutabledict({
    'float16': torch.float16,
    'float': torch.float32,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
})


@dataclasses.dataclass
class GemmaConfig:
    vocab_size: int = 256000
    max_position_embeddings: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    hidden_size: int = 3072
    intermediate_size: int = 24576
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    dtype: str = 'bfloat16'
    quant: bool = False
    tokenizer: Optional[Union[str, Path]] = Path('tokenizer/tokenizer.model')

    def get_dtype(self) -> Optional[torch.dtype]:
        return _STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)


def get_config_for_7b() -> GemmaConfig:
    return GemmaConfig()


def get_config_for_2b() -> GemmaConfig:
    return GemmaConfig(
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        hidden_size=2048,
        intermediate_size=16384
    )


class InvalidVariantError(ValueError):
    pass


def get_model_config(variant: str) -> GemmaConfig:
    if variant == '7b':
        return get_config_for_7b()
    elif variant == '2b':
        return get_config_for_2b()
    else:
        raise InvalidVariantError(f'Invalid variant {variant}. '
                                  f'Supported variants are "2b" and "7b"')


# Unit tests
def test_gemma_config():
    config = GemmaConfig()
    assert isinstance(config.tokenizer, (str, Path))

    config_7b = get_config_for_7b()
    assert config_7b.num_hidden_layers == 28

    config_2b = get_config_for_2b()
    assert config_2b.num_hidden_layers == 18

    try:
        get_model_config('3b')
    except InvalidVariantError as e:
        assert str(e) == 'Invalid variant 3b. Supported variants are "2b" and "7b"'


if __name__ == "__main__":
    test_gemma_config()
