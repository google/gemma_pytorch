"""Gemma 4 Multimodal model implementation."""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from gemma import config as gemma_config
from gemma import tokenizer
from gemma import model as gemma_model

from gemma.siglip_vision import siglip_vision_model
from gemma.gemma4_model import Gemma4ForCausalLM, Gemma4MultiModalProjector
from gemma.gemma4_audio import Gemma4AudioModel


def batched_merge_modalities(
    modality_emb: torch.Tensor,
    text_emb: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Merge image/audio and text embeddings based on a token mask.

    Args:
        modality_emb: The modality embeddings (B, Num_Modality_Tokens, H).
        text_emb: The text embeddings (B, Seq_Len, H).
        token_mask: The boolean/int mask indicating where the modality tokens are in the sequence (B, Seq_Len).

    Returns:
        The merged embeddings (B, Seq_Len, H).
    """
    b, seq_len = token_mask.shape

    # Cumulative sum to find indices of the modality tokens
    modality_indices = torch.cumsum(token_mask, dim=1) - 1

    # Clip to avoid out of bounds on the padding/unmasked tokens
    safe_indices = torch.clamp(modality_indices, min=0, max=modality_emb.size(1) - 1)

    # Gather the modality embeddings
    safe_indices_expanded = (
        safe_indices.unsqueeze(-1).expand(-1, -1, modality_emb.size(-1)).long()
    )
    aligned_modality = torch.gather(modality_emb, 1, safe_indices_expanded)

    # Mask it out
    mask_expanded = token_mask.unsqueeze(-1).bool()
    return torch.where(mask_expanded, aligned_modality, text_emb)


class Gemma4ForMultimodalLM(nn.Module):
    """Gemma 4 model for multimodal (Text, Vision, Audio) causal LM."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.dtype = config.get_dtype()
        assert config.architecture == gemma_config.Architecture.GEMMA_4
        self.config = config

        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)

        # Text language model (contains embeddings and decoder stack)
        self.lm = Gemma4ForCausalLM(config)
        self.sampler = gemma_model.Sampler(config.vocab_size, config)

        # Vision Model
        if config.vision_config is not None:
            self.vision_model = siglip_vision_model.SiglipVisionModel(
                config.vision_config
            )
            self.vision_projector = Gemma4MultiModalProjector(config)

        # Audio Model
        if config.audio_config is not None:
            self.audio_model = Gemma4AudioModel(config.audio_config)

        # We need RoPE wave lengths configured
        if config.rope_wave_length is None:
            raise ValueError("rope_wave_length must be provided for Gemma4.")

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        rope_lengths = config.rope_wave_length
        defaults = {
            gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
            gemma_config.AttentionType.GLOBAL: 10_000,
        }

        self._register_freqs_cis(
            "local_freqs_cis",
            head_dim,
            max_seq_len,
            theta=rope_lengths.get(
                gemma_config.AttentionType.LOCAL_SLIDING,
                defaults[gemma_config.AttentionType.LOCAL_SLIDING],
            ),
        )
        self._register_freqs_cis(
            "global_freqs_cis",
            head_dim,
            max_seq_len,
            theta=rope_lengths.get(
                gemma_config.AttentionType.GLOBAL,
                defaults[gemma_config.AttentionType.GLOBAL],
            ),
            rope_scaling_factor=config.rope_scaling_factor,
        )

    def _register_freqs_cis(
        self,
        name: str,
        head_dim: int,
        max_seq_len: int,
        theta: int = 10_000,
        rope_scaling_factor: Optional[int] = None,
    ):
        factor = rope_scaling_factor if rope_scaling_factor is not None else 1
        self.register_buffer(
            name,
            gemma_model.precompute_freqs_cis(
                head_dim, max_seq_len * 2, theta=theta, rope_scaling_factor=factor
            ),
        )

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        image_patches: Optional[torch.Tensor] = None,
        image_token_mask: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        audio_token_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass handling multimodal inputs and causal generation.

        Args:
            input_token_ids: Text token IDs (B, L).
            input_positions: Positions for RoPE.
            kv_caches: Key/Value caches for generation.
            mask: Attention mask (B, 1, L, L) or (B, L).
            output_positions: Generation indices.
            temperatures: Sampler temperature.
            top_ps: Sampler top-p.
            top_ks: Sampler top-k.
            image_patches: Raw image patches (B, N, C, H, W).
            image_token_mask: Mask of image tokens in text sequence (B, L).
            audio_features: Audio features (B, T, F).
            audio_token_mask: Mask of audio tokens in text sequence (B, L).
            audio_attention_mask: Mask for valid audio features (B, T).
            local_mask: Sliding window mask.

        Returns:
            Tuple of selected next tokens and updated kv_caches.
        """
        # 1. Base Text Embeddings
        hidden_states = self.lm.embed_tokens(input_token_ids)
        normalizer = torch.tensor(
            self.config.hidden_size**0.5,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_states = hidden_states * normalizer

        # 2. Vision Injection
        if (
            image_patches is not None
            and self.config.vision_config is not None
            and image_token_mask is not None
        ):
            B, N, C, H, W = image_patches.shape
            flattened_input = image_patches.reshape(B * N, C, H, W)

            # (B*N, num_patches, vision_dim)
            vision_outputs = self.vision_model.patch_embedding(flattened_input)
            vision_outputs = vision_outputs.flatten(2).transpose(1, 2)
            pos_ids = self.vision_model.position_ids.to(vision_outputs.device)
            vision_outputs = vision_outputs + self.vision_model.position_embedding(
                pos_ids
            )

            for block in self.vision_model.encoder_blocks:
                vision_outputs = block(vision_outputs)
            vision_outputs = self.vision_model.final_norm(vision_outputs)

            # Pool and project
            image_embeddings = self.vision_projector(
                vision_outputs
            )  # (B*N, mm_tokens_per_image, hidden_size)
            _, U, D = image_embeddings.shape
            image_embeddings = image_embeddings.reshape(B, N * U, D)  # (B, N*U, D)

            hidden_states = batched_merge_modalities(
                image_embeddings, hidden_states, image_token_mask
            )

        # 3. Audio Injection
        if (
            audio_features is not None
            and self.config.audio_config is not None
            and audio_token_mask is not None
        ):
            audio_embeddings = self.audio_model(
                audio_features, attention_mask=audio_attention_mask
            )
            hidden_states = batched_merge_modalities(
                audio_embeddings, hidden_states, audio_token_mask
            )

        # 4. Forward through LM Decoder stack
        # We pass hidden_states as inputs_embeds to bypass CausalLM's internal embedding layer
        logits, new_kv_caches = self.lm.forward(
            input_ids=None,
            positions=input_positions,
            attention_mask=mask,
            kv_caches=kv_caches,
            inputs_embeds=hidden_states,
        )

        # 5. Sampler and Tokens
        next_tokens = self.sampler(
            embedding=self.lm.embed_tokens.weight,
            hidden_states=logits,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )

        return next_tokens, new_kv_caches
