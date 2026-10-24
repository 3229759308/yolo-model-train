from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from armor_pose.data.dataset import ArmorPoseDataset, armor_collate
from armor_pose.data.prepare import prepare_dataset
from .test_prepare_pose_dataset import PrepareFixture


class ArmorDatasetTests(PrepareFixture):
    def setUp(self):
        super().setUp()
        prepare_dataset(self.export, self.output, val_groups=['camera/video_b'])
        self.data = self.output / 'data.yaml'

    def test_letterbox_uses_actual_rounded_scales_and_rgb(self):
        dataset = ArmorPoseDataset(self.data, split='val')
        image, target, metadata = dataset[0]
        self.assertEqual(image.shape, (3, 480, 640))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(metadata['original_size'], (53, 101))
        self.assertEqual(metadata['resized_size'], (336, 640))
        self.assertEqual(metadata['padding'], (0, 72))
        self.assertEqual(metadata['scale'], (640 / 101, 336 / 53))
        expected = torch.tensor([[[-10 * 640 / 101, -4 * 336 / 53 + 72],
                                  [10 * 640 / 101, 40 * 336 / 53 + 72],
                                  [90 * 640 / 101, 40 * 336 / 53 + 72],
                                  [90 * 640 / 101, 4 * 336 / 53 + 72]]])
        torch.testing.assert_close(target['points'], expected)
        self.assertLess(target['points'][0, 0, 0].item(), 0)
        self.assertEqual(target['color_ids'].tolist(), [1])
        self.assertEqual(target['class_ids'].tolist(), [3])
        torch.testing.assert_close(image[:, 0, 0], torch.full((3,), 114 / 255))
        torch.testing.assert_close(image[:, 100, 100], torch.tensor([10, 20, 30]) / 255)

    def test_empty_target_shapes_and_collation(self):
        dataset = ArmorPoseDataset(self.data, split='val')
        _, target, _ = dataset[1]
        self.assertEqual(target['points'].shape, (0, 4, 2))
        self.assertEqual(target['color_ids'].shape, (0,))
        self.assertEqual(target['class_ids'].shape, (0,))
        self.assertEqual(target['class_ids'].dtype, torch.long)
        images, targets, metadata = armor_collate([dataset[0], dataset[1]])
        self.assertEqual(images.shape, (2, 3, 480, 640))
        self.assertEqual(len(targets), 2)
        self.assertEqual(len(metadata), 2)

    def test_source_and_cached_targets_unchanged_by_read_and_mutation(self):
        source = self.images / self.metadata['images'][2]['image']
        original = source.read_bytes()
        dataset = ArmorPoseDataset(self.data, split='val')
        _, first, _ = dataset[0]
        points = first['points'].clone()
        first['points'][:] = -999
        _, second, _ = dataset[0]
        torch.testing.assert_close(second['points'], points)
        self.assertEqual(source.read_bytes(), original)

    def test_brightness_only_changes_pixels(self):
        plain = ArmorPoseDataset(self.data, split='val')
        augmented = ArmorPoseDataset(self.data, split='train', augment=True, brightness=.15)
        with patch('armor_pose.data.dataset.random.uniform', return_value=1.1):
            image, target, metadata = augmented[0]
        base, expected_target, expected_metadata = plain[0]
        torch.testing.assert_close(image, (base * 1.1).clamp(0, 1))
        torch.testing.assert_close(target['points'], expected_target['points'])
        self.assertEqual(metadata['scale'], expected_metadata['scale'])

    def test_invalid_input_size_and_brightness_are_rejected(self):
        for kwargs in [{'input_size': (0, 640)}, {'input_size': (480.0, 640)},
                       {'brightness': float('nan')}, {'brightness': 2}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ArmorPoseDataset(self.data, **kwargs)

    def test_validation_never_augments(self):
        dataset = ArmorPoseDataset(self.data, split='val', augment=True)
        with patch('armor_pose.data.dataset.random.uniform', side_effect=AssertionError('validation must not augment')):
            dataset[0]


if __name__ == '__main__':
    unittest.main()
