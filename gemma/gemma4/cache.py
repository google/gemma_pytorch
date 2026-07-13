"""Cache mechanisms for Gemma 4."""

from __future__ import annotations

from typing import Any

import torch


class Cache:
    """Base class for all caches."""

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the cache and return the new key/value states."""
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the current sequence length of the cache."""
        raise NotImplementedError(
            "Make sure to implement `get_seq_length` in a subclass."
        )

    def get_max_length(self) -> int | None:
        """Get the maximum length the cache can hold."""
        raise NotImplementedError(
            "Make sure to implement `get_max_length` in a subclass."
        )

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """Reorder the cache for beam search."""
        raise NotImplementedError(
            "Make sure to implement `reorder_cache` in a subclass."
        )


class DynamicCache(Cache):
    """Dynamic cache for autoregressive generation."""

    def __init__(self) -> None:
        """Initialize DynamicCache."""
        super().__init__()
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new key and value states."""
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the sequence length of the specified layer."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def get_max_length(self) -> int | None:
        """Return the maximum length (None for dynamic cache)."""
        return None

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """Reorder the cache for beam search."""
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx
            )


class StaticCache(Cache):
    """Static cache pre-allocated for torch.compile and CUDA graphs."""

    def __init__(
        self,
        config: Any,
        max_batch_size: int,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize StaticCache."""
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        for _ in range(config.num_hidden_layers):
            self.key_cache.append(
                torch.zeros(
                    (
                        max_batch_size,
                        self.num_key_value_heads,
                        max_cache_len,
                        self.head_dim,
                    ),
                    dtype=dtype,
                    device=device,
                )
            )
            self.value_cache.append(
                torch.zeros(
                    (
                        max_batch_size,
                        self.num_key_value_heads,
                        max_cache_len,
                        self.head_dim,
                    ),
                    dtype=dtype,
                    device=device,
                )
            )

        self.seen_tokens = 0

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new key and value states."""
        batch_size, _, seq_len, _ = key_states.shape

        self.key_cache[layer_idx][
            :batch_size, :, self.seen_tokens : self.seen_tokens + seq_len, :
        ] = key_states
        self.value_cache[layer_idx][
            :batch_size, :, self.seen_tokens : self.seen_tokens + seq_len, :
        ] = value_states

        return self.key_cache[layer_idx][
            :batch_size, :, : self.seen_tokens + seq_len, :
        ], self.value_cache[layer_idx][:batch_size, :, : self.seen_tokens + seq_len, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the current sequence length (seen tokens)."""
        return self.seen_tokens

    def get_max_length(self) -> int | None:
        """Get the maximum sequence length the static cache can hold."""
        return self.max_cache_len

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """Reorder the cache for beam search."""
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx
            )
