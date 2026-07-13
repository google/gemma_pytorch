"""Mixture of Experts for Gemma 4."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import Gemma4Config
from .layers import Gemma4MLP


class Gemma4MoERouter(nn.Module):
    """Router for Mixture of Experts."""

    def __init__(self, config: Gemma4Config):
        """Initialize Gemma4MoERouter."""
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.router_jitter_noise = config.router_jitter_noise

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the router."""
        if self.training and self.router_jitter_noise > 0:
            jitter = torch.empty_like(hidden_states).uniform_(
                -self.router_jitter_noise, self.router_jitter_noise
            )
            hidden_states = hidden_states * (1.0 + jitter)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)

        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        return routing_weights, selected_experts, router_logits


def calculate_load_balancing_loss(
    router_logits: torch.Tensor, num_experts: int, top_k: int
) -> torch.Tensor:
    """Calculate the auxiliary load balancing loss."""
    router_probs = F.softmax(router_logits, dim=-1)
    router_probs_mean = router_probs.mean(dim=0)

    # Calculate fraction of tokens routed to each expert
    _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
    expert_mask = expert_mask.sum(dim=1).float()  # (batch_size * seq_len, num_experts)
    expert_mask_mean = expert_mask.mean(dim=0)

    loss = (router_probs_mean * expert_mask_mean).sum() * num_experts
    return loss


class Gemma4MoE(nn.Module):
    """Gemma 4 Mixture of Experts layer."""

    def __init__(self, config: Gemma4Config):
        """Initialize Gemma4MoE."""
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router = Gemma4MoERouter(config)
        self.experts = nn.ModuleList(
            [Gemma4MLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for MoE layer."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        routing_weights, selected_experts, router_logits = self.router(hidden_states)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(
            2, 0, 1
        )  # (num_experts, batch * seq_len, num_experts_per_tok)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            idx, top_x = torch.where(expert_mask[expert_idx])

            if idx.shape[0] == 0:
                continue

            expert_tokens = hidden_states[idx]
            expert_weights = routing_weights[idx, top_x].unsqueeze(-1)

            expert_out = expert_layer(expert_tokens)
            expert_out = expert_out * expert_weights
            final_hidden_states.index_add_(0, idx, expert_out.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        return final_hidden_states, router_logits
