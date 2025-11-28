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

"""
Master Interactive Agent (iAgent) for Gemma Models

This script provides an interactive conversational interface for Gemma models,
supporting multi-turn conversations, system prompts, and advanced sampling parameters.
"""

import contextlib
import random
import sys
from typing import List, Optional

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
flags.DEFINE_integer('output_len', 256, 'Maximum length of the output sequence.')
flags.DEFINE_integer('seed', 12345, 'Random seed.')
flags.DEFINE_boolean('quant', False, 'Whether to use quantization.')
flags.DEFINE_float('temperature', 0.7, 'Sampling temperature (0.0 for greedy, higher for more random).')
flags.DEFINE_float('top_p', 0.95, 'Top-p (nucleus) sampling parameter.')
flags.DEFINE_integer('top_k', 64, 'Top-k sampling parameter.')
flags.DEFINE_string('system_prompt', None, 'Optional system prompt to guide the conversation.')
flags.DEFINE_boolean('interactive', True, 'Run in interactive mode.')
flags.DEFINE_string('prompt', None, 'Single prompt for non-interactive mode.')
flags.DEFINE_integer('max_history', 10, 'Maximum number of conversation turns to keep in history.')

# Define valid model variants
_VALID_MODEL_VARIANTS = ['2b', '2b-v2', '7b', '9b', '27b', '1b', '4b', '12b', '27b_v3']

# Define valid devices
_VALID_DEVICES = ['cpu', 'cuda']

# Validator functions
def validate_variant(variant):
    if variant not in _VALID_MODEL_VARIANTS:
        raise ValueError(f'Invalid variant: {variant}. Valid variants are: {_VALID_MODEL_VARIANTS}')
    return True

def validate_device(device):
    if device not in _VALID_DEVICES:
        raise ValueError(f'Invalid device: {device}. Valid devices are: {_VALID_DEVICES}')
    return True

def validate_temperature(temperature):
    if temperature < 0.0:
        raise ValueError('Temperature must be non-negative.')
    return True

# Register validators
flags.register_validator('variant', validate_variant, message='Invalid model variant.')
flags.register_validator('device', validate_device, message='Invalid device.')
flags.register_validator('temperature', validate_temperature, message='Invalid temperature.')


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


class MasterIAgent:
    """Master Interactive Agent for conversing with Gemma models."""

    def __init__(self, model, device, config_params):
        self.model = model
        self.device = device
        self.config = config_params
        self.conversation_history = []
        self.system_prompt = config_params.get('system_prompt')

    def format_prompt(self, user_input: str, include_history: bool = True) -> str:
        """Format the prompt with conversation history and system prompt."""
        prompt_parts = []

        # Add system prompt if provided
        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}\n")

        # Add conversation history
        if include_history and self.conversation_history:
            for turn in self.conversation_history[-self.config['max_history']:]:
                prompt_parts.append(f"User: {turn['user']}")
                prompt_parts.append(f"Assistant: {turn['assistant']}\n")

        # Add current user input
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def generate_response(self, user_input: str) -> str:
        """Generate a response to the user's input."""
        prompt = self.format_prompt(user_input)

        # Set temperature to None for greedy decoding if temperature is 0
        temperature = None if self.config['temperature'] == 0.0 else self.config['temperature']

        response = self.model.generate(
            prompt,
            self.device,
            output_len=self.config['output_len'],
            temperature=temperature,
            top_p=self.config['top_p'],
            top_k=self.config['top_k']
        )

        # Clean up the response
        response = response.strip()

        return response

    def add_to_history(self, user_input: str, assistant_response: str):
        """Add a conversation turn to the history."""
        self.conversation_history.append({
            'user': user_input,
            'assistant': assistant_response
        })

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")

    def run_interactive(self):
        """Run the interactive conversation loop."""
        print("=" * 70)
        print("Master Interactive Agent (iAgent) - Gemma Model")
        print("=" * 70)
        print(f"Model: {self.config['variant']}")
        print(f"Device: {self.device}")
        print(f"Temperature: {self.config['temperature']}")
        print(f"Top-p: {self.config['top_p']}, Top-k: {self.config['top_k']}")
        if self.system_prompt:
            print(f"System Prompt: {self.system_prompt}")
        print("=" * 70)
        print("\nCommands:")
        print("  /clear  - Clear conversation history")
        print("  /exit   - Exit the interactive agent")
        print("  /help   - Show this help message")
        print("\nType your message and press Enter to chat!")
        print("=" * 70)

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() == '/exit':
                    print("\nExiting Master iAgent. Goodbye!")
                    break
                elif user_input.lower() == '/clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == '/help':
                    print("\nCommands:")
                    print("  /clear  - Clear conversation history")
                    print("  /exit   - Exit the interactive agent")
                    print("  /help   - Show this help message")
                    continue

                # Generate response
                print("\nAssistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)

                # Add to history
                self.add_to_history(user_input, response)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /exit to quit or continue chatting.")
                continue
            except Exception as e:
                print(f"\nError generating response: {e}")
                continue

    def run_single(self, prompt: str):
        """Run a single prompt without interactive mode."""
        print("=" * 70)
        print(f"PROMPT: {prompt}")
        print("=" * 70)

        response = self.generate_response(prompt)

        print(f"\nRESPONSE: {response}")
        print("=" * 70)


def main(_):
    # Construct the model config
    model_config = config.get_model_config(FLAGS.variant)
    model_config.dtype = "float32" if FLAGS.device == "cpu" else "float16"
    model_config.quant = FLAGS.quant

    # Seed random
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # Create the model and load the weights
    device = torch.device(FLAGS.device)
    print("Loading model...")
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        model.load_weights(FLAGS.ckpt)
        model = model.to(device).eval()
    print("Model loaded successfully!")

    # Create configuration for the agent
    agent_config = {
        'variant': FLAGS.variant,
        'output_len': FLAGS.output_len,
        'temperature': FLAGS.temperature,
        'top_p': FLAGS.top_p,
        'top_k': FLAGS.top_k,
        'system_prompt': FLAGS.system_prompt,
        'max_history': FLAGS.max_history,
    }

    # Create the master iAgent
    agent = MasterIAgent(model, device, agent_config)

    # Run in interactive or single-prompt mode
    if FLAGS.interactive:
        agent.run_interactive()
    else:
        if FLAGS.prompt is None:
            print("Error: --prompt must be provided in non-interactive mode.")
            sys.exit(1)
        agent.run_single(FLAGS.prompt)


if __name__ == "__main__":
    app.run(main)


# Usage Examples:
#
# Interactive mode (default):
# python scripts/master_iagent.py --ckpt=/path/to/checkpoint --variant=4b --device=cuda
#
# With system prompt:
# python scripts/master_iagent.py --ckpt=/path/to/checkpoint --variant=4b --system_prompt="You are a helpful coding assistant."
#
# Non-interactive mode (single prompt):
# python scripts/master_iagent.py --ckpt=/path/to/checkpoint --variant=4b --interactive=false --prompt="Explain quantum computing"
#
# With custom sampling parameters:
# python scripts/master_iagent.py --ckpt=/path/to/checkpoint --variant=4b --temperature=0.9 --top_p=0.95 --top_k=50
#
# CPU mode with quantization:
# python scripts/master_iagent.py --ckpt=/path/to/checkpoint --variant=4b --device=cpu --quant=true
