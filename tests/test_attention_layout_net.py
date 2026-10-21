"""The tiny residual-attention pair used to compare BHND vs BNHD."""

from __future__ import annotations

import unittest

from tests.attention_layout_net import (
    PRESETS,
    compare_pair,
    pack_traffic,
)


class TestAttentionLayoutNet(unittest.TestCase):
    def test_tiny_bhnd_and_bnhd_match_sdpa(self):
        result = compare_pair(PRESETS["tiny"], seed=0)
        self.assertGreater(result["bhnd_hist"]["Transpose"], 0)
        self.assertEqual(result["bnhd_hist"].get("Transpose", 0), 0)
        self.assertEqual(result["bhnd_hist"]["Attention"], 1)
        self.assertEqual(result["bnhd_hist"]["Attention"], 1)
        self.assertEqual(result["bnhd_layouts"], ["BNHD"])
        self.assertEqual(result["bhnd_layouts"], ["BHND"])
        self.assertLess(result["maxdiff_bhnd_golden"], 2e-5)
        self.assertLess(result["maxdiff_bnhd_golden"], 2e-5)
        self.assertLess(result["maxdiff_bhnd_bnhd"], 2e-5)

    def test_two_blocks_drop_all_qkv_transposes(self):
        spec = PRESETS["tiny"]
        from tests.attention_layout_net import Spec
        from tests.attention_layout_net import compare_pair as _cmp

        two = Spec(batch=1, seq=16, heads=4, head_dim=32, blocks=2)
        result = _cmp(two, seed=1)
        self.assertEqual(result["bhnd_hist"]["Transpose"], 8)  # 4 per block
        self.assertEqual(result["bnhd_hist"].get("Transpose", 0), 0)
        self.assertEqual(result["bnhd_layouts"], ["BNHD", "BNHD"])
        self.assertLess(result["maxdiff_bhnd_bnhd"], 2e-5)

    def test_encoder_pack_beats_stride_model(self):
        traffic = pack_traffic(PRESETS["encoder"])
        self.assertLess(traffic["pack_bursts"] * 8, traffic["stride_txns"])


if __name__ == "__main__":
    unittest.main()
