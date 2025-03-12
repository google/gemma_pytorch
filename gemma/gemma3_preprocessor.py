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
"""Preprocessor for Gemma3 input."""
import token

from typing import Union, Any, Sequence

import torch
from absl import app
from PIL import Image
from .siglip_vision import preprocessor as siglip_vision_preprocessor
from .siglip_vision import pan_and_scan
from . import tokenizer
from . import config as gemma_config

CROPPED_IMAGE_PREFIX = "here is the original image"
CROPPED_IMAGE_FILLER = "and here are some crops to help you see better"


def gemma3_input_preprocessor(
    raw_user_prompt: Sequence[Union[Image.Image, str]],
) -> Sequence[Union[torch.Tensor, str]]:
  """Preprocessor for Gemma3 input.

  Args:
    raw_user_prompt: A list of images or strings, as provided by the user.

  Returns:
    A list of preprocessed images or strings.
  """
  preprocessed_input: list[Union[torch.Tensor, str]] = []
  for element in raw_user_prompt:
    if isinstance(element, Image.Image):
      cropped_images = pan_and_scan.pan_and_scan(element)
      preprocessed_images_cropped = siglip_vision_preprocessor.preprocess_images_for_siglip_vision(cropped_images)
      preprocessed_images_uncropped = siglip_vision_preprocessor.preprocess_images_for_siglip_vision([element])
      if len(preprocessed_images_cropped) == 1:
        preprocessed_input.append(preprocessed_images_uncropped[0])
      elif len(preprocessed_images_cropped) > 1:
        preprocessed_input.append(CROPPED_IMAGE_PREFIX)
        preprocessed_input.append(preprocessed_images_uncropped[0])
        preprocessed_input.append(CROPPED_IMAGE_FILLER)
        preprocessed_input.extend(preprocessed_images_cropped)
      else:
        raise ValueError("No images found in the input.")
    else:
      preprocessed_input.append(element)

  return preprocessed_input


def gemma3_batch_input_preprocessor(raw_input: Sequence[Sequence[Union[Image.Image, str]]]):
    """Preprocessor for Gemma3 batch input.
    """
    preprocessed_input: list[Sequence[Union[torch.Tensor, str]]] = []
    for element in raw_input:
      preprocessed_input.append(gemma3_input_preprocessor(element))
    return preprocessed_input


def tokenize_raw_input(
        tokenizer_obj: tokenizer.Tokenizer,
        raw_input: Sequence[Sequence[Union[str, Image.Image]]],
        config: gemma_config.GemmaConfig,
        output_len: int,
        device: Any,
    ) -> dict[str, Any]:
    """
        Converts a preprocessed batch of interleaved text and image inputs into
        token IDs and an image batch suitable for gemma3 model.

        Args:
            preprocessed_batch: List of lists containing strings and torch.Tensor images.
            image_token_id: Token ID to represent image placeholders.
            max_image_tokens: Number of tokens reserved for each image.
            image_size: Expected size of images (C, H, W).

        Returns:
            user_input_token_ids: Batch of token IDs with shape (B, L), where L is the max sequence length.
            image_batch: Batch of images with shape (B, N, C, H, W), where N is the max number of images.
    """
    if config.vision_config is None:
        raise ValueError('vision_config must be provided for Gemma3.')

    preprocessed_batch = gemma3_batch_input_preprocessor(raw_input)

    # Initialize lists to store token IDs and image tensors
    all_token_ids = []
    all_images = []
    prompt_lengths = []

    max_prompt_len = 0
    min_prompt_len = float("inf")
    max_num_images = 0
    # Iterate over each user prompt in the batch
    for prompt in preprocessed_batch:
        token_ids = []
        images = []
        token_ids.append(tokenizer_obj.bos_id)
        # Process each element in the prompt
        for element in prompt:
            if isinstance(element, str):
                # Tokenize text and add to token_ids
                tokens = tokenizer_obj.encode(element, bos=False, eos=False)
                token_ids.extend(tokens)
            elif isinstance(element, torch.Tensor):
                # Prepend (dual endline + tokenizer_obj.boi)
                token_ids.extend(tokenizer_obj.encode("\n\n", bos=False, eos=False))
                token_ids.append(tokenizer_obj.boi_id)
                # Add image placeholder tokens
                token_ids.extend(
                            [tokenizer_obj.image_token_placeholder_id]
                            * config.vision_config.encoding_sequence_length
                        )
                # Append (tokenizer_obj.eoi + dual endline)
                token_ids.append(tokenizer_obj.eoi_id)
                token_ids.extend(tokenizer_obj.encode("\n\n", bos=False, eos=False))
                # Store the image tensor
                images.append(element)
            else:
                raise ValueError(
                            "Unsupported type in prompt. Expected str or torch.Tensor."
                        )
        curr_prompt_len = len(token_ids)
        prompt_lengths.append(curr_prompt_len)

        max_prompt_len = max(max_prompt_len, curr_prompt_len)
        min_prompt_len = min(min_prompt_len, curr_prompt_len)
        max_num_images = max(max_num_images, len(images))

        all_token_ids.append(token_ids)
        all_images.append(images)

    max_seq_len = max_prompt_len + output_len

    # Pad token IDs to the maximum sequence length
    user_input_token_ids = []
    for token_ids in all_token_ids:
        pad_length = max_seq_len - len(token_ids)
        padded_token_ids = token_ids + [tokenizer_obj.pad_id] * pad_length
        user_input_token_ids.append(padded_token_ids)

    # Pad images to the maximum number of images in the batch
    image_batch = []
    image_presence_mask = []
    for images in all_images:
        # Check if all images within the current sublist have the same shape
        if images:  # Check if the sublist is not empty
            first_shape = images[0].shape
            for img in images:
                assert img.shape == first_shape, "Images within a sublist must have the same shape."
        pad_length = max_num_images - len(images)
        padded_images = images.copy() #create a copy so the original data is not altered.
        presence_mask = [True] * len(images)

        if pad_length > 0:
            # Create a list of zero tensors for padding
            padding = [
                torch.zeros(
                    (
                        config.vision_config.input_channels,
                        config.vision_config.image_size,
                        config.vision_config.image_size,
                    ), device=device
                )
                for _ in range(pad_length)
            ]
            padded_images.extend(padding)
            presence_mask.extend([False] * pad_length)
        image_batch.append(padded_images)
        image_presence_mask.append(presence_mask)

    # Convert lists to tensors
    user_input_token_ids = torch.tensor(user_input_token_ids, dtype=torch.long, device=device)
    if max_num_images > 0:
        image_batch = torch.stack([torch.stack(images) for images in image_batch]).to(
            device
        )
        image_presence_mask = torch.tensor(image_presence_mask, dtype=torch.bool, device=device)
    else:
        image_batch = None
        image_presence_mask = None

    # Prepare the output dictionary
    output_dict = {
            "user_input_token_ids": user_input_token_ids,
            "image_batch": image_batch,
            "batch_size": len(preprocessed_batch),
            "min_prompt_len": min_prompt_len,
            "max_prompt_len": max_prompt_len,
            "max_seq_len": max_seq_len,
            "image_presence_mask": image_presence_mask,
        }

    return output_dict
