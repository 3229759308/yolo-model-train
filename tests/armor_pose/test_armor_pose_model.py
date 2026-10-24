"""Protocol and trainability checks for the armor-specific pose model."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from armor_pose.common.head import ArmorPoseHead
from armor_pose.models.yolov8.model import ArmorPoseModel


class ArmorPoseModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_inference_protocol_and_training_dictionary(self):
        model = ArmorPoseModel(pretrained=False).eval()
        images = torch.rand(1, 3, 480, 640)
        with torch.no_grad():
            output = model(images)
            raw = model(images, return_raw=True)
        self.assertEqual(output.shape, (1, 21, 6300))
        for name, shape in (
            ("color_logits", (1, 6300, 4)),
            ("class_logits", (1, 6300, 9)),
            ("keypoints", (1, 6300, 4, 2)),
            ("raw_keypoints", (1, 6300, 4, 2)),
            ("anchor_points", (6300, 2)),
            ("strides", (6300, 1)),
        ):
            self.assertEqual(raw[name].shape, shape, name)
        torch.testing.assert_close(output[:, :4].transpose(1, 2), raw["color_logits"].sigmoid())
        torch.testing.assert_close(output[:, 4:13].transpose(1, 2), raw["class_logits"].sigmoid())
        torch.testing.assert_close(output[:, 13:].transpose(1, 2), raw["keypoints"].flatten(2))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((output[:, :13] > 0) & (output[:, :13] < 1)).all())

    def test_row_major_grids_and_non_degenerate_corner_initialization(self):
        head = ArmorPoseHead(channels=(8, 16, 32)).eval()
        features = [torch.zeros(1, channels, h, w) for channels, h, w in ((8, 4, 4), (16, 2, 2), (32, 1, 1))]
        with torch.no_grad():
            raw = head(features, return_raw=True)
        torch.testing.assert_close(raw["anchor_points"][:5], torch.tensor([[4., 4.], [12., 4.], [20., 4.], [28., 4.], [4., 12.]]))
        torch.testing.assert_close(raw["anchor_points"][[16, 20]], torch.tensor([[8., 8.], [16., 16.]]))
        torch.testing.assert_close(raw["strides"][:, 0], torch.tensor([8.] * 16 + [16.] * 4 + [32.]))
        expected_first_points = torch.tensor([[-4., 0.], [-4., 8.], [12., 8.], [12., 0.]])
        torch.testing.assert_close(raw["keypoints"][0, 0], expected_first_points)
        # Out-of-image predictions are allowed and are never silently clamped.
        self.assertLess(raw["keypoints"][0, 0, 0, 0].item(), 0)
        spans = raw["keypoints"].amax(dim=2) - raw["keypoints"].amin(dim=2)
        self.assertTrue((spans > 0).all())
        centers = raw["keypoints"].mean(dim=2)
        torch.testing.assert_close(centers[0], raw["anchor_points"])

    def test_backbone_and_all_head_branches_are_trainable(self):
        model = ArmorPoseModel(pretrained=False).train()
        raw = model(torch.rand(2, 3, 64, 96), return_raw=True)
        loss = raw["color_logits"].square().mean() + raw["class_logits"].square().mean()
        loss = loss + (raw["keypoints"] / 96).square().mean()
        loss.backward()
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        for parameter in (
            model.backbone[0].conv.weight,
            model.head.color_branches[0][-1].weight,
            model.head.class_branches[0][-1].weight,
            model.head.keypoint_branches[0][-1].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_state_dictionary_reload_does_not_load_pretrained_weights(self):
        model = ArmorPoseModel(pretrained=False).eval()
        with patch(f"{ArmorPoseModel.__module__}.load_checkpoint", side_effect=AssertionError("must remain local")):
            restored = ArmorPoseModel(pretrained=False).eval()
        restored.load_state_dict(model.state_dict())
        image = torch.rand(2, 3, 64, 96)
        with torch.no_grad():
            torch.testing.assert_close(restored(image), model(image))
        self.assertEqual(restored.architecture_config()["point_names"], ["LT", "LB", "RB", "RT"])
        self.assertEqual(restored.architecture_config()["output_channels"], 21)

    def test_invalid_inputs_and_missing_local_checkpoint_are_reported(self):
        with self.assertRaises(FileNotFoundError):
            ArmorPoseModel(backbone_weights="/tmp/no-such-armor-checkpoint.pt", pretrained=True)
        model = ArmorPoseModel(pretrained=False).eval()
        for shape in ((1, 1, 64, 64), (1, 3, 63, 64), (1, 3, 0, 64), (3, 64, 64)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape))


if __name__ == "__main__":
    unittest.main()
