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
import argparse
import contextlib
import os
import random
import socket
import sys
from typing import List, Union

import numpy as np
import torch
import torch.multiprocessing

from gemma.config import GemmaConfig, get_model_config
from gemma.model_xla import GemmaForCausalLM
from gemma.tokenizer import Tokenizer
import gemma.xla_model_parallel as xla_model_parallel

USE_CUDA = os.environ.get('USE_CUDA', False)
if not USE_CUDA:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
else:
    # Choose an available port.
    with contextlib.closing(socket.socket(socket.AF_INET,
                                          socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        MASTER_PORT = str(s.getsockname()[1])


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def generate(
    i: int,
    model_config: GemmaConfig,
    ckpt_path: str,
    prompts: List[str],
    output_lens: List[int],
    temperatures: Union[List[float], None],
    top_ps: List[float],
    top_ks: List[int],
    seed: int
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_CUDA:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = MASTER_PORT
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "nccl",
                rank=int(os.environ.get("RANK", 0)),
                world_size=int(os.environ.get("WORLD_SIZE", 1)))
        xla_model_parallel.set_g_group()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)
    else:
        device = xm.xla_device()
        xm.set_rng_state(seed, device)

    rank = xla_model_parallel.get_model_parallel_rank()
    world_size = xla_model_parallel.get_model_parallel_world_size()
    if rank > 0:
        sys.stdout = open(os.devnull, 'w')

    # build, load and compile model.
    with _set_default_tensor_type(model_config.get_dtype()):
        model = GemmaForCausalLM(model_config, world_size, rank, device)
        model.load_weights(ckpt_path)
        model = model.to(device).eval()

    # create tokenizer.
    tokenizer = Tokenizer(model_config.tokenizer)

    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
    min_prompt_len = min(len(p) for p in prompt_tokens)

    batch_size = len(prompts)
    if temperatures is not None:
        assert batch_size == len(temperatures)
    assert batch_size == len(top_ps)
    assert batch_size == len(top_ks)
    max_seq_len = max([len(p) + o for p, o in zip(prompt_tokens, output_lens)])
    assert max_seq_len <= model_config.max_position_embeddings
    if model_config.num_key_value_heads < world_size:
        assert world_size % model_config.num_key_value_heads == 0
        n_local_heads = 1
    else:
        assert model_config.num_key_value_heads % world_size == 0
        n_local_heads = model_config.num_key_value_heads // world_size

    # build KV caches
    kv_caches = []
    for _ in range(model_config.num_hidden_layers):
        k_cache = torch.zeros(
            size=(batch_size, max_seq_len, n_local_heads,
                  model_config.head_dim),
            dtype=model_config.get_dtype(),
            device=device,
        )
        v_cache = torch.zeros(
            size=(batch_size, max_seq_len, n_local_heads,
                  model_config.head_dim),
            dtype=model_config.get_dtype(),
            device=device,
        )
        kv_caches.append((k_cache, v_cache))

    # prepare inputs
    token_ids_tensor = torch.full((batch_size, max_seq_len),
                                  tokenizer.pad_id,
                                  dtype=torch.int64)
    input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                        tokenizer.pad_id,
                                        dtype=torch.int64)
    prompt_length = [len(p) for p in prompt_tokens]
    for i, p in enumerate(prompt_tokens):
        token_ids_tensor[i, :len(p)] = torch.tensor(p)
        input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
            p[:min_prompt_len])
    token_ids_tensor = token_ids_tensor.to(device)
    prompt_mask_tensor = token_ids_tensor != tokenizer.pad_id
    input_token_ids_tensor = input_token_ids_tensor.to(device)
    input_positions_tensor = torch.arange(0, min_prompt_len,
                                          dtype=torch.int64).to(device)
    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                             -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
    temperatures_tensor = None if not temperatures else torch.FloatTensor(temperatures).to(device)
    top_ps_tensor = torch.FloatTensor(top_ps).to(device)
    top_ks_tensor = torch.LongTensor(top_ks).to(device)
    output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(device)
    eos_flags_tensor = torch.tensor([False] * batch_size).to(device)

    if not USE_CUDA:
        xm.mark_step()
    # Prefill up to min_prompt_len tokens, then treat other prefill as decode and ignore output.
    for i in range(max_seq_len - min_prompt_len):
        next_token_ids, _ = model(
            input_token_ids=input_token_ids_tensor,
            input_positions=input_positions_tensor,
            kv_write_indices=None,
            kv_caches=kv_caches,
            mask=curr_mask_tensor,
            output_positions=output_positions_tensor,
            temperatures=temperatures_tensor,
            top_ps=top_ps_tensor,
            top_ks=top_ks_tensor,
        )
        curr_prompt_mask = prompt_mask_tensor.index_select(
            1, output_index).squeeze(dim=1)
        curr_token_ids = token_ids_tensor.index_select(
            1, output_index).squeeze(dim=1)
        output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                       next_token_ids).unsqueeze(dim=1)
        token_ids_tensor.index_copy_(1, output_index, output_token_ids)

        input_token_ids_tensor = output_token_ids
        input_positions_tensor = output_index.unsqueeze(dim=-1)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(device)
        output_index = output_index + 1
        if not USE_CUDA:
            xm.mark_step()

        # Check if all sequences have reached EOS.
        batch_eos_idx = (next_token_ids == tokenizer.eos_id).nonzero(
            as_tuple=True)[0]
        for eos_idx in batch_eos_idx:
            if output_index >= prompt_length[eos_idx]:
                eos_flags_tensor[eos_idx] = True

        if eos_flags_tensor.all():
            break

    # Detokenization.
    token_ids = token_ids_tensor.tolist()
    results = []
    for i, tokens in enumerate(token_ids):
        trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i]) +
                                output_lens[i]]
        if tokenizer.eos_id in trimmed_output:
            eos_index = trimmed_output.index(tokenizer.eos_id)
            trimmed_output = trimmed_output[:eos_index]
        results.append(tokenizer.decode(trimmed_output))

    for prompt, result in zip(prompts, results):
        print('======================================')
        print(f'PROMPT: {prompt}')
        print(f'RESULT: {result}')
        print('======================================')


def main(args):
    model_config = get_model_config(args.variant)
    model_config.quant = args.quant

    prompts = [args.prompt]
    n = len(prompts)
    output_lengths = [args.output_len] * n
    temperatures = [0.95] * n
    top_ps = [1.0] * n
    top_ks = [100] * n

    if USE_CUDA:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = MASTER_PORT
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "nccl",
                rank=int(os.environ.get("RANK", 0)),
                world_size=int(os.environ.get("WORLD_SIZE", 1)))
        xla_model_parallel.set_g_group()
        torch.multiprocessing.spawn(
            generate,
            args=(
                model_config,
                args.ckpt,
                prompts,
                output_lengths,
                temperatures,
                top_ps,
                top_ks,
                args.seed,
            ),
        )
    else:
        xmp.spawn(
            generate,
            args=(
                model_config,
                args.ckpt,
                prompts,
                output_lengths,
                temperatures,
                top_ps,
                top_ks,
                args.seed,
            ),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--variant",
                        type=str,
                        default="2b",
                        choices=["2b", "2b-v2", "7b", "9b", "27b"])
    parser.add_argument("--output_len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    args = parser.parse_args()

    main(args)
