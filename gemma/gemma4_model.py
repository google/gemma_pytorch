"""Gemma 4 native PyTorch modeling."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

from . import config as gemma_config
from . import model as gemma_model

# We use the same base primitives from gemma_model (RMSNorm, Sampler, Linear, etc.)
# but re-implement the necessary layers to match the new MoE, Audio, Vision interactions.


class Gemma4RMSNorm(nn.Module):
    """Gemma 4 RMSNorm implementation ensuring numerical stability."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.with_scale:
            return output * self.weight
        return output


class Gemma4MLP(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = gemma_model.Linear(
            self.hidden_size, self.intermediate_size, config.quant
        )
        self.up_proj = gemma_model.Linear(
            self.hidden_size, self.intermediate_size, config.quant
        )
        self.down_proj = gemma_model.Linear(
            self.intermediate_size, self.hidden_size, config.quant
        )
        # Using exact same activation as Gemma2
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4RoutedExperts(nn.Module):
    """Monolithic MoE expert module vectorizing all routed experts for Gemma 4."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.moe_intermediate_size
            if config.moe_intermediate_size is not None
            else config.intermediate_size
        )

        # We store expert weights as single concatenated tensors for efficient batched matmul
        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_size)
        )

        nn.init.normal_(self.gate_proj_weight, std=self.hidden_size**-0.5)
        nn.init.normal_(self.up_proj_weight, std=self.hidden_size**-0.5)
        nn.init.normal_(self.down_proj_weight, std=self.hidden_size**-0.5)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(
        self, x: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (B, T, H)
        topk_indices: (B, T, K)
        topk_weights: (B, T, K)
        """
        b, t, h = x.shape
        k = topk_indices.shape[-1]

        x_flat = x.view(b * t, h)
        idx_flat = topk_indices.view(b * t, k)
        w_flat = topk_weights.view(b * t, k)

        # We process one expert channel at a time, or gather weights.
        # Gathering weights is memory intensive but fast.

        # idx_flat: (B*T, K)
        gate_w = self.gate_proj_weight[idx_flat]  # (B*T, K, H, I)
        up_w = self.up_proj_weight[idx_flat]  # (B*T, K, H, I)
        down_w = self.down_proj_weight[idx_flat]  # (B*T, K, I, H)

        x_expanded = x_flat.unsqueeze(1).unsqueeze(2)  # (B*T, 1, 1, H)

        # x: (B*T, 1, 1, H), gate_w: (B*T, K, H, I) -> (B*T, K, 1, I)
        gate_out = torch.matmul(x_expanded, gate_w)
        up_out = torch.matmul(x_expanded, up_w)

        act = self.act_fn(gate_out) * up_out

        # act: (B*T, K, 1, I), down_w: (B*T, K, I, H) -> (B*T, K, 1, H)
        out = torch.matmul(act, down_w).squeeze(2)  # (B*T, K, H)

        out = out * w_flat.unsqueeze(2)  # (B*T, K, H)
        out = out.sum(dim=1)  # (B*T, H)

        return out.view(b, t, h).to(x.dtype)


class Gemma4MoE(nn.Module):
    """Gemma 4 Mixture of Experts combining routed and shared experts."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config

        # Shared expert
        shared_config = config
        if config.num_shared_experts is not None and config.num_shared_experts > 0:
            shared_config.intermediate_size = (
                config.intermediate_size * config.num_shared_experts
            )
        self.shared_experts = Gemma4MLP(shared_config)

        self.gate_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Weight scales
        self.pre_forward_scale_2 = nn.Parameter(torch.ones(config.hidden_size))

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

        self.routed_experts = Gemma4RoutedExperts(config)

        self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, x: torch.Tensor, original_x: torch.Tensor) -> torch.Tensor:
        shared_out = self.shared_experts(x)
        shared_out = self.post_feedforward_layernorm_1(shared_out)

        routed_inputs = self.pre_feedforward_layernorm_2(original_x)
        unscaled_norm = self.gate_norm(original_x)

        root_size = self.config.hidden_size**-0.5
        router_scale = self.pre_forward_scale_2.to(unscaled_norm.dtype)
        gate_inputs = unscaled_norm * root_size * router_scale

        router_logits = self.gate(gate_inputs.float())

        if self.training and self.config.router_jitter_noise > 0.0:
            noise = torch.rand_like(router_logits) * self.config.router_jitter_noise
            router_logits = router_logits + noise

        routing_weights = torch.softmax(router_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(
            routing_weights, self.config.num_experts_per_tok, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        per_expert = self.per_expert_scale.to(topk_weights.dtype)
        topk_weights = topk_weights * per_expert[topk_indices]
        topk_weights = topk_weights.to(x.dtype)

        routed_out = self.routed_experts(routed_inputs, topk_indices, topk_weights)
        routed_out = self.post_feedforward_layernorm_2(routed_out)

        return shared_out + routed_out


class Gemma4RotaryEmbedding(nn.Module):
    """Gemma 4 RoPE implementation."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.dim = int(config.head_dim * config.partial_rotary_factor)
        self.base = config.rope_theta

        # Precompute frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = position_ids.max().item() + 1
        self._update_cos_sin_cache(seq_len, device=q.device, dtype=q.dtype)

        # cos: (B, seq_len, head_dim/2) -> (B, 1, seq_len, head_dim/2)
        cos = self._cos_cached[position_ids].unsqueeze(2)
        sin = self._sin_cached[position_ids].unsqueeze(2)

        # We rotate half, so q shape becomes (B, num_heads, seq_len, head_dim)
        # But wait, self._rotate_half does it on the last dimension.
        # But cos/sin only have head_dim/2. We need to duplicate them or apply it as complex.
        # Wait, the previous implementation:
        # q_embed = (q * cos) + (self._rotate_half(q) * sin)
        # That means cos/sin must have head_dim size!
        # Ah, in _update_cos_sin_cache:
        # emb = torch.cat((freqs, freqs), dim=-1)
        # So emb has size (seq_len, head_dim) (if dim was head_dim/2).
        # Yes, self.dim is head_dim * partial_rotary_factor.
        # So cos/sin is (B, 1, seq_len, head_dim)

        q_rot = q[..., : self.dim]
        q_pass = q[..., self.dim :]
        k_rot = k[..., : self.dim]
        k_pass = k[..., self.dim :]

        q_embed = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

        q_out = torch.cat((q_embed, q_pass), dim=-1)
        k_out = torch.cat((k_embed, k_pass), dim=-1)

        return q_out.type_as(q), k_out.type_as(k)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class Gemma4Attention(nn.Module):
    """Multi-Head / Grouped-Query Attention for Gemma 4 (supporting Global and Local)."""

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attention_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.config = config
        self.attention_type = attention_type

        self.num_heads = config.num_attention_heads
        if attention_type == gemma_config.AttentionType.GLOBAL:
            self.num_kv_heads = (
                config.num_key_value_heads
            )  # Or global_num_kv_heads if available
            self.head_dim = (
                config.global_head_dim
                if config.global_head_dim is not None
                else config.head_dim
            )
        else:
            self.num_kv_heads = config.num_key_value_heads
            self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.q_proj = gemma_model.Linear(
            self.hidden_size, self.num_heads * self.head_dim, config.quant
        )
        self.k_proj = gemma_model.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, config.quant
        )
        self.v_proj = gemma_model.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, config.quant
        )
        self.o_proj = gemma_model.Linear(
            self.num_heads * self.head_dim, self.hidden_size, config.quant
        )

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, with_scale=False
        )

        self.rope = Gemma4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        q, k = self.rope(q, k, position_ids)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
            kv_cache = (k, v)

        # Grouped query logic
        num_rep = self.num_heads // self.num_kv_heads
        if num_rep > 1:
            k = k.repeat_interleave(num_rep, dim=2)
            v = v.repeat_interleave(num_rep, dim=2)

        q = q.transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(2, 3)) * scale

        if self.config.attn_logit_softcapping is not None:
            scores = scores / self.config.attn_logit_softcapping
            scores = torch.tanh(scores) * self.config.attn_logit_softcapping

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(weights, v)  # (bsz, num_heads, q_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        return self.o_proj(out), kv_cache


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config

        # Determine attention type based on GEMMA4_ATTENTION_PATTERN or fallback
        # Typical pattern: local, global, local, ... or all global
        self.attention_type = gemma_config.AttentionType.GLOBAL
        if config.attn_types:
            self.attention_type = config.attn_types[layer_idx % len(config.attn_types)]

        self.pre_self_attention_norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attention = Gemma4Attention(config, self.attention_type)
        self.post_self_attention_norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.pre_ffw_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.num_experts is not None and config.num_experts > 1:
            self.mlp = Gemma4MoE(config)
        else:
            self.mlp = Gemma4MLP(config)

        self.post_ffw_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        lnx = self.pre_self_attention_norm(hidden_states)
        attn_out, kv_cache = self.self_attention(
            lnx, position_ids, attention_mask=attention_mask, kv_cache=kv_cache
        )
        attn_out = self.post_self_attention_norm(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        lnx2 = self.pre_ffw_norm(hidden_states)

        if isinstance(self.mlp, Gemma4MoE):
            mlp_out = self.mlp(lnx2, original_x=hidden_states)
        else:
            mlp_out = self.mlp(lnx2)

        mlp_out = self.post_ffw_norm(mlp_out)
        hidden_states = residual + mlp_out

        return hidden_states, kv_cache


class Gemma4ForCausalLM(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = gemma_model.Embedding(
            config.vocab_size, config.hidden_size, config.quant
        )

        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = gemma_model.Linear(
            config.hidden_size, config.vocab_size, config.quant
        )
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list[Tuple[torch.Tensor, torch.Tensor]]]:

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
            normalizer = torch.tensor(
                self.config.hidden_size**0.5,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            hidden_states = hidden_states * normalizer

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None

            hidden_states, new_cache = layer(
                hidden_states,
                positions,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            if new_cache is not None:
                new_kv_caches.append(new_cache)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits) * self.config.final_logit_softcapping

        return logits, new_kv_caches


class Gemma4MultiModalProjector(nn.Module):
    """Projects vision features into the language model's hidden dimension.

    Pools patch tokens using position-based weighted averaging, then projects
    into the text model's hidden space.
    """

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.text_config = config
        self.vision_config = config.vision_config

        vhs = self.vision_config.embedding_dim
        ths = config.hidden_size

        self.patches_per_img = (
            self.vision_config.image_size // self.vision_config.conv2d_patch_size
        )
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_img // self.tokens_per_side
        self.num_output_tokens = self.tokens_per_side * self.tokens_per_side

        self.mm_input_projection_weight = nn.Parameter(torch.zeros(ths, vhs))
        self.mm_soft_emb_norm = Gemma4RMSNorm(
            vhs, eps=self.vision_config.layer_norm_eps
        )

    def _avg_pool_vision_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Pools patch tokens into a fixed grid using position-based averaging."""
        b, num_patches, hidden = x.shape
        k_sq = self.kernel_size * self.kernel_size

        positions = torch.arange(num_patches, device=x.device)
        row = positions // self.patches_per_img
        col = positions % self.patches_per_img

        kernel_idxs = (row // self.kernel_size) * self.tokens_per_side + (
            col // self.kernel_size
        )

        # Create one-hot weights (num_patches, num_output_tokens)
        weights = (
            torch.nn.functional.one_hot(
                kernel_idxs, num_classes=self.num_output_tokens
            ).to(x.dtype)
            / k_sq
        )

        # x: (B, num_patches, hidden), weights: (num_patches, num_output_tokens)
        # We want: (B, num_output_tokens, hidden)
        return torch.matmul(weights.t().unsqueeze(0), x.float()).to(x.dtype)

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        """Projects and pools the vision outputs.

        Args:
            vision_outputs: Patch embeddings from the vision encoder (B, num_patches, hidden_size).

        Returns:
            Projected image tokens (B, num_output_tokens, text_hidden_size).
        """
        pooled = self._avg_pool_vision_outputs(vision_outputs)
        pooled = pooled * math.sqrt(self.vision_config.embedding_dim)

        # The normalization is applied before the projection in Gemma 4
        pooled = self.mm_soft_emb_norm(pooled)
        return torch.nn.functional.linear(pooled, self.mm_input_projection_weight)
