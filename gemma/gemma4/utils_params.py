"""Parameter translation utilities for JAX to PyTorch."""

from typing import Any

import torch


def translate_jax_to_pytorch(jax_params: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Translate JAX parameters to PyTorch state dict.

    This function handles the necessary transposition for Dense/Linear layers.
    """
    pytorch_state_dict: dict[str, torch.Tensor] = {}

    for key, value in jax_params.items():
        if hasattr(value, "__array__"):
            import numpy as np

            tensor = torch.from_numpy(np.array(value))
        else:
            tensor = torch.tensor(value)

        if "kernel" in key:
            tensor = tensor.t()
            key = key.replace("kernel", "weight")

        if "scale" in key:
            key = key.replace("scale", "weight")

        pytorch_state_dict[key] = tensor

    return pytorch_state_dict
