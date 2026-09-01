import os
import sys
import unittest

import torch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(TEST_DIR)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from birealnet_hmv import HardBinaryConv, birealnet18


class HMVMathTest(unittest.TestCase):
    def make_conv(self, in_channels=64, kernel_size=1, padding=0):
        conv = HardBinaryConv(
            in_channels,
            1,
            kernel_size=kernel_size,
            padding=padding,
            hmv_group_size=64,
            hmv_wiring="spatial",
            hmv_enabled=True,
        )
        with torch.no_grad():
            conv.weight.fill_(1.0)
            conv.majority_voter.alpha.fill_(1.0)
        return conv

    def test_default_32_to_32_tie_is_negative(self):
        conv = self.make_conv()
        x = torch.cat((torch.ones(1, 32, 1, 1), -torch.ones(1, 32, 1, 1)), dim=1)
        self.assertEqual(conv.integer_thresholds(), (33, 1))
        self.assertLess(conv(x).item(), 0)

    def test_learned_l1_threshold_can_accept_tie(self):
        conv = self.make_conv()
        x = torch.cat((torch.ones(1, 32, 1, 1), -torch.ones(1, 32, 1, 1)), dim=1)
        with torch.no_grad():
            conv.threshold_l1.fill_(32.0)
        self.assertGreater(conv(x).item(), 0)

    def test_spatial_padding_uses_only_valid_bits(self):
        conv = self.make_conv(in_channels=64, kernel_size=3, padding=1)
        x = torch.ones(1, 64, 2, 2)
        output = conv(x)
        self.assertEqual(tuple(output.shape), (1, 1, 2, 2))
        self.assertTrue(torch.all(output > 0))

    def test_threshold_forward_is_integer_and_nonnegative(self):
        conv = self.make_conv()
        with torch.no_grad():
            conv.threshold_l1.fill_(32.49)
            conv.threshold_l2.fill_(-3.0)
        conv.clamp_thresholds_()
        self.assertEqual(conv.integer_thresholds(), (32, 0))

    def test_both_thresholds_receive_surrogate_gradients(self):
        conv = self.make_conv()
        x = torch.cat((torch.ones(1, 32, 1, 1), -torch.ones(1, 32, 1, 1)), dim=1)
        conv(x).sum().backward()
        self.assertIsNotNone(conv.threshold_l1.grad)
        self.assertIsNotNone(conv.threshold_l2.grad)
        self.assertTrue(torch.isfinite(conv.threshold_l1.grad))
        self.assertTrue(torch.isfinite(conv.threshold_l2.grad))

    def test_birealnet18_has_expected_16_layers(self):
        model = birealnet18(
            hmv_config={"hmv_group_size": 64, "hmv_wiring": "spatial", "hmv_enabled": True}
        )
        rows = model.threshold_rows()
        self.assertEqual(len(rows), 16)
        self.assertEqual([row["reduction_length"] for row in rows], [576] * 5 + [1152] * 4 + [2304] * 4 + [4608] * 3)
        self.assertEqual([row["group_count"] for row in rows], [9] * 5 + [18] * 4 + [36] * 4 + [72] * 3)


if __name__ == "__main__":
    unittest.main()
