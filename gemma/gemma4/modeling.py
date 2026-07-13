"""PyTorch native Gemma 4 modeling."""

from __future__ import annotations

import torch
from torch import nn

from .audio import Gemma4AudioModel
from .cache import Cache
from .config import Gemma4Config
from .decoder_layer import Gemma4DecoderLayer
from .layers import Gemma4RMSNorm
from .vision import Gemma4VisionModel


class Gemma4MultiModalProjector(nn.Module):
    """Multimodal projector for Gemma 4."""

    def __init__(self, config: Gemma4Config):
        """Initialize Gemma4MultiModalProjector."""
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.hidden_size, bias=True
        )
        self.act = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Forward pass for multimodal projector."""
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Gemma4ForCausalLM(nn.Module):
    """Gemma 4 model for causal language modeling."""

    def __init__(self, config: Gemma4Config):
        """Initialize Gemma4ForCausalLM."""
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Text embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        # Multimodal sub-models
        self.vision_model = Gemma4VisionModel(config.vision_config)
        self.multi_modal_projector = Gemma4MultiModalProjector(config)
        self.audio_model = Gemma4AudioModel(config)

        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed_tokens.weight = self.lm_head.weight  # Tie weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]
        | Cache
        | None = None,
        pixel_values: torch.Tensor | None = None,
        audio_values: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...] | Cache | None
    ]:
        """Forward pass of the model."""
        hidden_states = self.embed_tokens(input_ids)

        if pixel_values is not None:
            vision_outputs = self.vision_model(pixel_values)
            image_features = self.multi_modal_projector(vision_outputs)

            # Very simplified interleaving: assume image tokens are placed at the end of the sequence for now
            # A real implementation would find the `<image>` token in `input_ids` and splice `image_features` there.
            hidden_states = torch.cat([image_features, hidden_states], dim=1)

        if audio_values is not None:
            audio_features = self.audio_model(audio_values)
            # Very simplified interleaving
            hidden_states = torch.cat([audio_features, hidden_states], dim=1)

        next_decoder_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value: tuple[torch.Tensor, torch.Tensor] | Cache | None
            if past_key_values is None:
                past_key_value = None
            elif isinstance(past_key_values, Cache):
                past_key_value = past_key_values
            else:
                past_key_value = past_key_values[idx]

            hidden_states, present_key_value, _router_logits = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            if present_key_value is not None and not isinstance(
                present_key_value, Cache
            ):
                next_decoder_cache += (present_key_value,)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if isinstance(past_key_values, Cache):
            return logits, past_key_values

        return logits, next_decoder_cache if len(next_decoder_cache) > 0 else None

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        min_new_tokens: int = 0,
    ) -> torch.Tensor:
        """Generate text."""
        past_key_values = None
        for _ in range(max_new_tokens):
            logits, past_key_values = self(
                input_ids[:, -1:] if past_key_values is not None else input_ids,
                past_key_values=past_key_values,
            )
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        return input_ids
