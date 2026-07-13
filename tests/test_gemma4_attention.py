import torch
import unittest
from typing import Tuple

from gemma import config as gemma_config
from gemma.gemma4_model import Gemma4Attention


class TestGemma4Attention(unittest.TestCase):
    def setUp(self):
        self.config = gemma_config.GemmaConfig(
            architecture=gemma_config.Architecture.GEMMA_4,
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            rms_norm_eps=1e-6,
            attn_logit_softcapping=50.0,
            quant=False,
        )

    def test_attention_forward(self):
        # Global Attention Test
        attn = Gemma4Attention(
            self.config, attention_type=gemma_config.AttentionType.GLOBAL
        )

        batch_size = 2
        seq_len = 10
        hidden_size = self.config.hidden_size
        head_dim = self.config.head_dim

        x = torch.randn(batch_size, seq_len, hidden_size)

        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        out, cache = attn(x, position_ids)

        self.assertEqual(out.shape, (batch_size, seq_len, hidden_size))
        self.assertIsNone(
            cache
        )  # No KV cache passed in, so none returned unless generated

    def test_torch_compile(self):
        attn = Gemma4Attention(
            self.config, attention_type=gemma_config.AttentionType.GLOBAL
        )
        compiled_attn = torch.compile(attn, backend="aot_eager")

        batch_size = 2
        seq_len = 10
        hidden_size = self.config.hidden_size
        head_dim = self.config.head_dim

        x = torch.randn(batch_size, seq_len, hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        out, _ = compiled_attn(x, position_ids)
        self.assertEqual(out.shape, (batch_size, seq_len, hidden_size))


if __name__ == "__main__":
    unittest.main()
