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
import os
from typing import List, Optional

import sentencepiece

def _assert_file_exists(model_path: str):
    assert os.path.isfile(model_path), model_path

_BEGIN_IMAGE_TOKEN = 255999
_END_IMAGE_TOKEN = 256000

class Tokenizer:

    def __init__(self, model_path: Optional[str]):
        _assert_file_exists(model_path)
        self.sp_model = sentencepiece.SentencePieceProcessor()
        self.sp_model.Load(model_path)

        # BOS / EOS token IDs.
        self.n_words: int = self.sp_model.GetPieceSize()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        self.boi_id: int = _BEGIN_IMAGE_TOKEN
        self.eoi_id: int = _END_IMAGE_TOKEN
        self.image_token_placeholder_id: int = self.sp_model.pad_id()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
        """Converts a string into a list of tokens."""
        assert isinstance(s, str)
        t = self.sp_model.EncodeAsIds(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """Converts a list of tokens into a string."""
        return self.sp_model.DecodeIds(t)
