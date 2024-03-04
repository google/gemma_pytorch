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

"""Inference-only Gemma model implementation."""

import re
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Sequence, Tuple, Union

from gemma import config as gemma_config
from gemma.xla_model_parallel import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
)


class Sampler(nn.Module):

    def __init__(self, vocab_size: int, world_size: int, rank: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.rank = rank

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        hidden_states = hidden_states.index_select(
            1, output_positions).squeeze(dim=1)

        hidden_states_parallel = scatter_to_model_parallel_region(
            hidden_states,
            groups=None,
            world_size=self.world_size,
            rank=self.rank)
        hidden_states_parallel = torch.matmul(hidden_states_parallel,
                                              embedding.t())
        logits = reduce_from_model_parallel_region(
            hidden_states_parallel,
            groups=None,
            world_size=self.world_size,
            rank=self.rank,
        )
        if embedding_bias is not None:
            logits += embedding_bias

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1)

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1],
                                   device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs,
                                           num_samples=1,
                                           replacement=True).squeeze(dim=-1)
        return next_token_ids


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
                    dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output


class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        world_size: int,
        rank: int,
        quant: bool,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        def init_method(x):
            return x

        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            quant=quant,
        )

        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            quant=quant,
        )

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            quant=quant,
        )

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        world_size: int,
        rank: int,
        quant: bool,
    ):
        super().__init__()
        self.rank = rank

        def init_method(x):
            return x

        self.total_num_heads = num_heads
        assert self.total_num_heads % world_size == 0
        self.num_heads = self.total_num_heads // world_size  # head per shard

        if num_kv_heads < world_size:
            assert world_size % num_kv_heads == 0
            self.total_num_kv_heads = world_size
        else:
            assert num_kv_heads % world_size == 0
            self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads // world_size  # kv head per shard

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5

        self.qkv_proj = ColumnParallelLinear(
            self.hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) *
            self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            quant=quant,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            quant=quant,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],
                               dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.o_proj(output)
        return output


class GemmaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        world_size: int,
        rank: int,
    ):
        super().__init__()
        self.rank = rank
        self.self_attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            world_size=world_size,
            rank=rank,
            quant=config.quant,
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            world_size=world_size,
            rank=rank,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        world_size: int,
        rank: int
    ):
        super().__init__()
        self.config = config
        self.rank = rank
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(GemmaDecoderLayer(config, world_size, rank))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        world_size: int,
        rank: int,
        device: torch.device,
    ):
        super().__init__()
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.device = device

        assert config.num_attention_heads % world_size == 0
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        def init_method(x):
            return x

        self.embedder = ParallelEmbedding(
            vocab_size,
            config.hidden_size,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            quant=config.quant,
        )
        self.model = GemmaModel(config, world_size, rank)
        self.sampler = Sampler(vocab_size, world_size, rank)

        rope_theta = getattr(config, 'rope_theta', 10000)
        # [head_dim * 2, ] -> complex -> two dim (real, imaginary) implicitly
        freqs_cis = precompute_freqs_cis(head_dim,
                                         max_seq_len * 2,
                                         theta=rope_theta)
        self.register_buffer('freqs_cis', freqs_cis)

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        freqs_cis = self.freqs_cis.index_select(0, input_positions)
        kv_write_indices = input_positions

        hidden_states = self.embedder(input_token_ids)
        # Gemma normalizes the embedding by sqrt(hidden_size).
        hidden_states = hidden_states * (self.config.hidden_size**0.5)
        # hidden_states should be [batch_size, input_len, hidden_size]

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
        )
        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = (
                embedder_weight * self.embedder.weight_scaler.unsqueeze(-1))
        next_tokens = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens

    def load_weights(self, model_path: str):
        checkpoint = torch.load(model_path, weights_only=True)
        model_state_dict = checkpoint['model_state_dict']

        num_attn_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        hidden_size = self.config.hidden_size

        def split(tensor: torch.Tensor, axis: int) -> torch.Tensor:
            axis_len = tensor.shape[axis]
            split_len = axis_len // self.world_size
            split_start = split_len * self.rank
            split_end = split_start + split_len
            tensor = torch.moveaxis(tensor, axis, 0)
            tensor = tensor[split_start:split_end, ...]
            tensor = torch.moveaxis(tensor, 0, axis)
            return tensor

        for k, v in model_state_dict.items():
            if k == 'freqs_cis':
                continue
            if (k == 'model.norm.weight' or re.fullmatch(
                    r'model.layers.\d+.input_layernorm.weight', k)
                    or re.fullmatch(
                        r'model.layers.\d+.post_attention_layernorm.weight',
                        k) or k.endswith('weight_scaler')):
                pass
            elif (k == 'embedder.weight' or re.fullmatch(
                    r'model.layers.\d+.mlp.down_proj.weight', k)):
                v = split(v, 1)
            elif (re.fullmatch(r'model.layers.\d+.mlp.gate_proj.weight', k)
                  or re.fullmatch(r'model.layers.\d+.mlp.up_proj.weight', k)):
                v = split(v, 0)
            elif re.fullmatch(r'model.layers.\d+.self_attn.qkv_proj.weight',
                              k):
                if num_kv_heads <= self.world_size:
                    num_replicas = self.world_size // num_kv_heads
                    v = v.reshape(num_attn_heads + num_kv_heads * 2, head_dim,
                                  hidden_size)
                    query = v[:num_attn_heads, ...]
                    key = v[num_attn_heads:num_attn_heads + num_kv_heads,
                            ...].repeat(num_replicas, 1, 1)
                    value = v[-num_kv_heads:, ...].repeat(num_replicas, 1, 1)
                    v = torch.cat(
                        (split(query, 0), split(key, 0), split(value, 0)),
                        dim=0)
                else:
                    v = v.reshape(3, num_attn_heads, head_dim, hidden_size)
                    v = split(v, 1)
                v = v.reshape(-1, hidden_size)
            elif re.fullmatch(r'model.layers.\d+.self_attn.o_proj.weight', k):
                v = v.reshape(hidden_size, num_attn_heads, head_dim)
                v = split(v, 1)
                v = v.reshape(hidden_size, -1)
            else:
                raise ValueError(f'Unrecognized key: {k}')
            self.state_dict()[k].copy_(v)
