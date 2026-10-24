"""Original YOLOv5 backbone semantics and shared-head training contracts."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from armor_pose.common.loss import ArmorPoseLoss
from armor_pose.models.yolov5.model import YoloV5ArmorPoseModel


class YoloV5Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_original_stem_c3_depth_and_sppf_activation(self):
        model = YoloV5ArmorPoseModel(pretrained=False)
        self.assertEqual(model.backbone[0].conv.kernel_size, (6, 6))
        self.assertEqual(model.backbone[0].conv.padding, (2, 2))
        self.assertEqual([len(model.backbone[i].m) for i in (2, 4, 6, 8)], [1, 2, 3, 1])
        self.assertIsInstance(model.backbone[9].cv1.act, nn.SiLU)
        self.assertEqual(model.backbone[0].bn.eps, .001)
        self.assertEqual(model.backbone[0].bn.momentum, .03)

    def test_fixed_protocol_and_training_gradient(self):
        model = YoloV5ArmorPoseModel(pretrained=False)
        images = torch.rand(1, 3, 480, 640)
        model.eval()
        with torch.no_grad():
            features = model.forward_features(images)
            self.assertEqual([list(f.shape) for f in features],
                             [[1, 64, 60, 80], [1, 128, 30, 40], [1, 256, 15, 20]])
            self.assertEqual(list(model(images).shape), [1, 21, 6300])
        model.train()
        target = {'points': torch.tensor([[[100., 100.], [100., 120.], [140., 120.], [140., 100.]]]),
                  'color_ids': torch.tensor([0]), 'class_ids': torch.tensor([3])}
        loss, _ = ArmorPoseLoss()(model(images, return_raw=True), [target], (480, 640))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.backbone[0].conv.weight.grad.abs().sum().item(), 0)
        for branches in (model.head.color_branches, model.head.class_branches, model.head.keypoint_branches):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in branches.parameters() if p.grad is not None), 0)

    def test_pretrained_transfer_requires_every_backbone_tensor(self):
        reference = YoloV5ArmorPoseModel(pretrained=False)
        original = nn.Module()
        original.model = nn.Sequential(*copy.deepcopy(reference.backbone))
        with patch('armor_pose.models.yolov5.model.load_official_checkpoint', return_value=original):
            loaded = YoloV5ArmorPoseModel(pretrained=True)
        for key, value in reference.backbone.state_dict().items():
            torch.testing.assert_close(loaded.backbone.state_dict()[key], value, rtol=0, atol=0)
        original.model[0].conv = nn.Conv2d(3, 32, 6, 2, 2)
        with patch('armor_pose.models.yolov5.model.load_official_checkpoint', return_value=original):
            with self.assertRaisesRegex(ValueError, 'pretrained backbone mismatch'):
                YoloV5ArmorPoseModel(pretrained=True)

    def test_missing_weights_fail_and_offline_build_never_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                YoloV5ArmorPoseModel(Path(directory) / 'missing.pt', pretrained=True)
        with patch('armor_pose.models.yolov5.model.load_official_checkpoint', side_effect=AssertionError('offline build loaded weights')):
            YoloV5ArmorPoseModel(pretrained=False)


if __name__ == '__main__':
    unittest.main()
