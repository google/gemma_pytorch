import torch
import unittest
from torch.nn import functional as F

from gemma import config as gemma_config
from gemma.gemma4_model import Gemma4MoE, Gemma4RoutedExperts


class TestGemma4MoE(unittest.TestCase):
    def setUp(self):
        self.config = gemma_config.GemmaConfig(
            architecture=gemma_config.Architecture.GEMMA_4,
            hidden_size=256,
            intermediate_size=512,
            moe_intermediate_size=512,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            rms_norm_eps=1e-6,
            quant=False,
        )

    def test_routed_experts(self):
        # Test routing explicitly
        routed_experts = Gemma4RoutedExperts(self.config)
        batch_size = 2
        seq_len = 10
        hidden_size = self.config.hidden_size
        k = self.config.num_experts_per_tok

        x = torch.randn(batch_size, seq_len, hidden_size)
        topk_indices = torch.randint(
            0, self.config.num_experts, (batch_size, seq_len, k)
        )
        topk_weights = torch.softmax(torch.randn(batch_size, seq_len, k), dim=-1)

        out = routed_experts(x, topk_indices, topk_weights)
        self.assertEqual(out.shape, (batch_size, seq_len, hidden_size))
        self.assertFalse(torch.isnan(out).any())

    def test_moe_layer(self):
        # Test full MoE forward pass
        moe = Gemma4MoE(self.config)
        batch_size = 2
        seq_len = 5
        hidden_size = self.config.hidden_size

        x = torch.randn(batch_size, seq_len, hidden_size)
        original_x = torch.randn(batch_size, seq_len, hidden_size)

        out = moe(x, original_x)
        self.assertEqual(out.shape, (batch_size, seq_len, hidden_size))

    def test_torch_compile(self):
        # Test that torch.compile works with Gemma4MoE (critical for performance)
        moe = Gemma4MoE(self.config)
        compiled_moe = torch.compile(
            moe, backend="aot_eager"
        )  # Using eager backend just for syntax testing

        batch_size = 2
        seq_len = 5
        hidden_size = self.config.hidden_size

        x = torch.randn(batch_size, seq_len, hidden_size)
        original_x = x.clone()

        out = compiled_moe(x, original_x)
        self.assertEqual(out.shape, (batch_size, seq_len, hidden_size))


if __name__ == "__main__":
    unittest.main()
