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

import numpy as np
import torch
from absl import app, flags

from gemma import config
from gemma import model as gemma_model

# Define flags
FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt', None, 'Path to the checkpoint file.', required=True)
flags.DEFINE_string('variant', '4b', 'Model variant.')
flags.DEFINE_string('device', 'cpu', 'Device to run the model on.')
flags.DEFINE_integer('output_len', 10, 'Length of the output sequence.')
flags.DEFINE_integer('seed', 12345, 'Random seed.')
flags.DEFINE_boolean('quant', False, 'Whether to use quantization.')
flags.DEFINE_string('prompt', 'What are large language models?', 'Input prompt for the model.')

# Define valid text only model variants
_VALID_MODEL_VARIANTS = ['2b', '2b-v2', '7b', '9b', '27b', '1b']

# Define valid devices
_VALID_DEVICES = ['cpu', 'cuda']

# Validator function for the 'variant' flag
def validate_variant(variant):
    if variant not in _VALID_MODEL_VARIANTS:
        raise ValueError(f'Invalid variant: {variant}. Valid variants are: {_VALID_MODEL_VARIANTS}')
    return True

# Validator function for the 'device' flag
def validate_device(device):
    if device not in _VALID_DEVICES:
        raise ValueError(f'Invalid device: {device}. Valid devices are: {_VALID_DEVICES}')
    return True

# Register the validator for the 'variant' flag
flags.register_validator('variant', validate_variant, message='Invalid model variant.')

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
    model_config = config.get_model_config(FLAGS.variant)
    model_config.dtype = "float32"
    model_config.quant = FLAGS.quant

    # Seed random.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # Create the model and load the weights.
    device = torch.device(FLAGS.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        model.load_weights(FLAGS.ckpt)
        model = model.to(device).eval()
    print("Model loading done")

    # Generate the response.
    result = model.generate(FLAGS.prompt, device, output_len=FLAGS.output_len)

    # Print the prompts and results.
    print('======================================')
    print(f'PROMPT: {FLAGS.prompt}')
    print(f'RESULT: {result}')
    print('======================================')

if __name__ == "__main__":
    app.run(main)


# How to run this script:

# Example command (replace with your actual paths and values):
# python scripts/run.py --device=cpu --ckpt=/path/to/your/pytorch_checkpoint/model.ckpt --output_len=2 --prompt="The name of the capital of Italy is"
# Important:
# - Replace '/path/to/your/pytorch_checkpoint/model.ckpt' with the actual path to your checkpoint file.
# - Choose the correct --variant (model size).
# - Use --device=cuda if you have a GPU; otherwise, use --device=cpu.