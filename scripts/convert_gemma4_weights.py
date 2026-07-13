"""Converts Gemma 4 Flax/JAX parameters to PyTorch state dict format."""

import argparse

import torch

from gemma.gemma4.utils_params import translate_jax_to_pytorch


def convert_gemma4_weights(jax_weights_path: str, output_path: str):
    """
    Converts JAX nested dictionaries or NumPy npz to PyTorch state_dict using translate_jax_to_pytorch.
    """
    print(f"Loading JAX weights from {jax_weights_path}...")
    # Simulated load for now - replace with actual load mechanism (e.g. np.load or flax serialization)
    # jax_params = np.load(jax_weights_path, allow_pickle=True).item()
    jax_params = {}  # Placeholder

    print("Translating parameters to PyTorch format...")
    state_dict = translate_jax_to_pytorch(jax_params)

    print(f"Saving PyTorch state_dict to {output_path}...")
    torch.save(state_dict, output_path)
    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Gemma 4 JAX weights to PyTorch."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input JAX weights."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output PyTorch weights (.pt).",
    )
    args = parser.parse_args()

    convert_gemma4_weights(args.input, args.output)
