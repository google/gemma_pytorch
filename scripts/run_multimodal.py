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


import contextlib
import random

from absl import app
from absl import flags
import numpy as np
from PIL import Image
import torch

from gemma import config
from gemma import gemma3_model

# Define flags
FLAGS = flags.FLAGS

_CKPT = flags.DEFINE_string(
    'ckpt', None, 'Path to the checkpoint file.', required=True
)
_VARIANT = flags.DEFINE_string('variant', '4b', 'Model variant.')
_DEVICE = flags.DEFINE_string('device', 'cpu', 'Device to run the model on.')
_OUTPUT_LEN = flags.DEFINE_integer(
    'output_len', 10, 'Length of the output sequence.'
)
_SEED = flags.DEFINE_integer('seed', 12345, 'Random seed.')
_QUANT = flags.DEFINE_boolean('quant', False, 'Whether to use quantization.')

# Define valid multimodal model variants
_VALID_MODEL_VARIANTS = ['4b', '12b', '27b_v3']

# Define valid devices
_VALID_DEVICES = ['cpu', 'cuda']


# Validator function for the 'variant' flag
def validate_variant(variant):
  if variant not in _VALID_MODEL_VARIANTS:
    raise ValueError(
        f'Invalid variant: {variant}. Valid variants are:'
        f' {_VALID_MODEL_VARIANTS}'
    )
  return True


# Validator function for the 'device' flag
def validate_device(device):
  if device not in _VALID_DEVICES:
    raise ValueError(
        f'Invalid device: {device}. Valid devices are: {_VALID_DEVICES}'
    )
  return True


# Register the validator for the 'variant' flag
flags.register_validator(
    'variant', validate_variant, message='Invalid model variant.'
)

# Register the validator for the 'device' flag
flags.register_validator('device', validate_device, message='Invalid device.')


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)


def main(_):
  # Construct the model config.
  model_config = config.get_model_config(_VARIANT.value)
  model_config.dtype = 'float32'
  model_config.quant = _QUANT.value
  image_paths = {"cow_in_beach": "scripts/images/cow_in_beach.jpg",
                   "lilly": "scripts/images/lilly.jpg",
                   "sunflower": "scripts/images/sunflower.JPG",
                   'golden_test_image': (
                           'scripts/images/test_image.jpg'
                    ),
    }

  image = {}
  for key in image_paths:
    try:
      image[key] = Image.open(image_paths[key]) # Open local file
      image[key].show()
    except IOError as e:
      print(f"Error loading image: {e}")
      exit()

  # Seed random.
  random.seed(_SEED.value)
  np.random.seed(_SEED.value)
  torch.manual_seed(_SEED.value)

  # Create the model and load the weights.
  device = torch.device(_DEVICE.value)
  with _set_default_tensor_type(model_config.get_dtype()):
    model = gemma3_model.Gemma3ForMultimodalLM(model_config)
    model.load_state_dict(torch.load(_CKPT.value)['model_state_dict'])
    # model.load_weights(_CKPT.value)
    model = model.to(device).eval()
  print('Model loading done')

  # Generate text only.
  result = model.generate(
      [
          [
              '<start_of_turn>user The capital of Italy'
              ' is?<end_of_turn>\n<start_of_turn>model'
          ],
          [
              '<start_of_turn>user What is your'
              ' purpose?<end_of_turn>\n<start_of_turn>model'
          ],
      ],
      device,
      output_len=_OUTPUT_LEN.value,
  )

  # Print the results.
  print('======================================')
  print(f'Text only RESULT: {result}')
  print('======================================')

  # Generate golden Gemax test image.
  result = model.generate(
      [[
          '<start_of_turn>user\n',
          image['golden_test_image'],
          'Caption this image. <end_of_turn>\n<start_of_turn>model',
      ]],
      device,
      output_len=_OUTPUT_LEN.value,
  )

  # Print the result.
  print('======================================')
  print(f'Golden test image RESULT: {result}')
  print('======================================')

  # Generate text and image.
  result = model.generate(
      [[
          '<start_of_turn>user\n',
          image['cow_in_beach'],
          (
              'The name of the animal in the image is'
              ' <end_of_turn>\n<start_of_turn>model'
          ),
      ]],
      device,
      output_len=_OUTPUT_LEN.value,
  )

  # Print the result.
  print('======================================')
  print(f'Single image RESULT: {result}')
  print('======================================')

  # Generate interleave text and multiple images.
  result = model.generate(
      [[
          '<start_of_turn>user\nThis image',
          image['lilly'],
          'and this image',
          image['sunflower'],
          'are similar because? <end_of_turn>\n<start_of_turn>model',
      ]],
      device,
      output_len=_OUTPUT_LEN.value,
  )

  # Print the result.
  print('======================================')
  print(f'Interleave images RESULT: {result}')
  print('======================================')


if __name__ == '__main__':
  app.run(main)
