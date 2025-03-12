# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma model config."""

import dataclasses
from . import preprocessor


# https://developers.googleblog.com/en/gemma-explained-paligemma-architecture/
@dataclasses.dataclass
class SiglipVisionModelConfig:
  """Returns the model config for the vision model of Gemma 3 andPaliGemma."""
  # The number of transformer encoder blocks in the siglip encoder model.
  num_hidden_layers: int = 27
  # The dimension of the embedding.
  embedding_dim: int = 1152
  # Whether to use bias in the 2D conv embedding layer.
  embedding_use_bias: bool = True
  # The number of channels in the input images.
  input_channels: int = 3
  # The input image size.
  image_size: int = preprocessor.DEFAULT_IMAGE_SIZE
  # Kernel size of 2D convolution layer.
  conv2d_patch_size = 14
  # The number of attention heads used in the attention layers of the model.
  num_attention_heads: int = 16
  # The number of head dimensions.
  head_dim: int = 72
  # Clarify: is num_key_value same as num_query_groups?
  num_key_value_heads: int = 16
  # The number of query groups for implementing attention.
  num_query_groups: int = 16
  # Clarify: usecase of this field is not clear.
  qkv_use_bias: bool = True
  # Clarify: usecase of this field is not clear.
  output_proj_use_bias: bool = True
  # The dimension of the MLP representations.
  intermediate_size: int = 4304
  # The epsilon used by the layer normalization layers.
  layer_norm_eps: float = 1e-6
  # Clarify: identify if the dtype varies for the siglip vision model.
  dtype: str = 'bfloat16'
  # Whether a quantized version of the model is used.
  quant: bool = False
  # The sequence length of the encoding.
  encoding_sequence_length: int = 256


def get_siglip_vision_model_config() -> SiglipVisionModelConfig:
  """Returns the default model config for the vision model of Gemma 3 and PaliGemma."""
  return SiglipVisionModelConfig()

