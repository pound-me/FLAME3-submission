from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.baseline_runtime import build_model, load_pretrained_if_available


class FourChannelStemInitializationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(91)
        self.source = build_model({"MODEL": "pidnet_s", "MODE": "rgb", "NUM_CLASSES": 3}, augment=True)
        self.temporary = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.temporary.name) / "imagenet_like.pth"
        torch.save({"state_dict": self.source.state_dict()}, self.checkpoint)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_random_strategy_preserves_complete_kaiming_stem(self) -> None:
        torch.manual_seed(200)
        model = build_model({"MODEL": "pidnet_s", "MODE": "fusion", "NUM_CLASSES": 3}, augment=True)
        before = model.conv1[0].weight.detach().clone()
        config = {"PRETRAINED": str(self.checkpoint), "PRETRAIN_SKIP_KEYS": ["conv1.0.weight"], "PRETRAIN_STEM_STRATEGY": "random"}
        load_pretrained_if_available(model, config)
        self.assertTrue(torch.equal(before, model.conv1[0].weight))
        self.assertIn("conv1.0.weight", config["PRETRAIN_LOAD_AUDIT"]["explicit_skip_keys"])
        self.assertEqual(config["PRETRAIN_LOAD_AUDIT"]["stem_strategy"], "random")

    def test_rgb_copy_ir_mean_is_exact(self) -> None:
        torch.manual_seed(200)
        model = build_model({"MODEL": "pidnet_s", "MODE": "fusion", "NUM_CLASSES": 3}, augment=True)
        config = {"PRETRAINED": str(self.checkpoint), "PRETRAIN_SKIP_KEYS": [], "PRETRAIN_STEM_STRATEGY": "rgb_copy_ir_mean"}
        matched = load_pretrained_if_available(model, config)
        source = self.source.conv1[0].weight.detach()
        destination = model.conv1[0].weight.detach()
        self.assertTrue(torch.equal(destination[:, :3], source))
        self.assertTrue(torch.equal(destination[:, 3:4], source.mean(dim=1, keepdim=True)))
        self.assertTrue(torch.equal(model.conv1[1].weight, self.source.conv1[1].weight))
        self.assertEqual(config["PRETRAIN_LOAD_AUDIT"]["stem_strategy"], "rgb_copy_ir_mean")
        self.assertIn("conv1.0.weight", config["PRETRAIN_LOAD_AUDIT"]["loaded_tensor_keys"])
        self.assertGreater(matched, 0)

    def test_strategy_and_skip_list_are_mutually_consistent(self) -> None:
        model = build_model({"MODEL": "pidnet_s", "MODE": "fusion", "NUM_CLASSES": 3}, augment=True)
        with self.assertRaisesRegex(ValueError, "requires conv1.0.weight"):
            load_pretrained_if_available(model, {"PRETRAINED": str(self.checkpoint), "PRETRAIN_SKIP_KEYS": [], "PRETRAIN_STEM_STRATEGY": "random"})


if __name__ == "__main__":
    unittest.main()
