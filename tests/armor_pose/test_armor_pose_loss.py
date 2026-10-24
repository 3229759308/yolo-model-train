"""Focused gradient and assignment checks for the custom 21-channel head."""
from __future__ import annotations

import unittest

import torch

from armor_pose.common.loss import ArmorPoseLoss


def prediction(count: int = 4, batch: int = 1) -> dict:
    anchors = torch.stack((torch.arange(count).float() * 8 + 4, torch.full((count,), 4.0)), dim=-1)
    return {
        'color_logits': torch.zeros(batch, count, 4, requires_grad=True),
        'class_logits': torch.zeros(batch, count, 9, requires_grad=True),
        'keypoints': anchors.view(1, count, 1, 2).expand(batch, count, 4, 2).clone().requires_grad_(),
        'anchor_points': anchors,
        'strides': torch.full((count, 1), 8.0),
    }


def target(points=None, colors=None, classes=None) -> dict:
    if points is None:
        points = []
    count = len(points)
    return {
        'points': torch.tensor(points, dtype=torch.float32).reshape(count, 4, 2),
        'color_ids': torch.tensor(colors if colors is not None else [0] * count, dtype=torch.long),
        'class_ids': torch.tensor(classes if classes is not None else [1] * count, dtype=torch.long),
    }


BOX = [[2, 1], [2, 7], [10, 7], [10, 1]]


class ArmorPoseLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_empty_batch_trains_both_background_branches(self):
        pred = prediction(batch=2)
        loss, stats = ArmorPoseLoss()(pred, [target(), target()], (32, 32))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['num_positive'].item(), 0)
        self.assertEqual(stats['num_targets'], 0)
        loss.backward()
        self.assertTrue((pred['color_logits'].grad > 0).all())
        self.assertTrue((pred['class_logits'].grad > 0).all())
        self.assertIsNotNone(pred['keypoints'].grad)
        self.assertEqual(pred['keypoints'].grad.abs().sum().item(), 0)

    def test_background_loss_does_not_grow_with_anchor_count(self):
        small, _ = ArmorPoseLoss()(prediction(4), [target()], (480, 640))
        large, _ = ArmorPoseLoss()(prediction(6300), [target()], (480, 640))
        torch.testing.assert_close(small, large)

    def test_degenerate_initial_predictions_still_train_points(self):
        pred = prediction()
        loss, stats = ArmorPoseLoss()(pred, [target([BOX])], (32, 32))
        self.assertGreaterEqual(stats['num_positive'].item(), 1)
        self.assertGreater(stats['loss_keypoint'].item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(pred['keypoints'].grad).all())
        self.assertGreater(pred['keypoints'].grad.abs().sum().item(), 0)

    def test_overlapping_objects_get_distinct_reserved_anchors(self):
        pred = prediction()
        loss, stats = ArmorPoseLoss(topk=1)(pred, [target([BOX, BOX], [0, 3], [1, 8])], (32, 32))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(stats['num_positive'].item(), 2)
        self.assertEqual(stats['num_targets'], 2)
        loss.backward()
        self.assertTrue(torch.isfinite(pred['color_logits'].grad).all())
        self.assertTrue(torch.isfinite(pred['class_logits'].grad).all())

    def test_exact_ordered_points_have_zero_regression_loss(self):
        pred = prediction()
        pred['keypoints'] = torch.tensor(BOX, dtype=torch.float32).view(1, 1, 4, 2).repeat(1, 4, 1, 1).requires_grad_()
        _, stats = ArmorPoseLoss()(pred, [target([BOX])], (32, 32))
        self.assertAlmostEqual(stats['loss_keypoint'].item(), 0.0)
        self.assertAlmostEqual(stats['loss_giou'].item(), 0.0)
        swapped = dict(pred)
        swapped['keypoints'] = pred['keypoints'][:, :, [1, 0, 3, 2], :]
        _, swapped_stats = ArmorPoseLoss()(swapped, [target([BOX])], (32, 32))
        self.assertGreater(swapped_stats['loss_keypoint'].item(), 0)
        self.assertAlmostEqual(swapped_stats['loss_giou'].item(), 0.0)

    def test_outside_points_are_supervised_without_clipping(self):
        pred = prediction()
        outside = [[-20, 1], [-20, 7], [-10, 7], [-10, 1]]
        loss, stats = ArmorPoseLoss(topk=1, box_weight=0)(pred, [target([outside])], (32, 32))
        self.assertGreater(stats['num_positive'].item(), 0)
        loss.backward()
        # A positive x gradient moves the predicted corners toward negative x.
        self.assertGreater(pred['keypoints'].grad[..., 0].sum().item(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_stride_normalization_is_invariant_to_uniform_resize(self):
        pred = prediction()
        gt = target([BOX])
        first, _ = ArmorPoseLoss()(pred, [gt], (32, 32))
        scaled = {key: value * 2 if key in {'keypoints', 'anchor_points', 'strides'} else value for key, value in pred.items()}
        scaled_gt = {**gt, 'points': gt['points'] * 2}
        second, _ = ArmorPoseLoss()(scaled, [scaled_gt], (64, 64))
        torch.testing.assert_close(first, second)

    def test_batched_anchor_layout_and_mixed_empty_images(self):
        pred = prediction(batch=2)
        pred['anchor_points'] = pred['anchor_points'].unsqueeze(0).repeat(2, 1, 1)
        pred['strides'] = pred['strides'].unsqueeze(0).repeat(2, 1, 1)
        loss, stats = ArmorPoseLoss()(pred, [target(), target([BOX], [3], [5])], (32, 32))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['num_targets'], 1)
        self.assertGreater(stats['num_positive'].item(), 0)
        loss.backward()
        self.assertEqual(pred['keypoints'].grad[0].abs().sum().item(), 0)
        self.assertGreater(pred['keypoints'].grad[1].abs().sum().item(), 0)

    def test_positive_color_and_class_gradients_raise_the_correct_scores(self):
        pred = prediction(1)
        loss, _ = ArmorPoseLoss()(pred, [target([BOX], [2], [7])], (32, 32))
        loss.backward()
        self.assertLess(pred['color_logits'].grad[0, 0, 2].item(), 0)
        self.assertLess(pred['class_logits'].grad[0, 0, 7].item(), 0)
        self.assertGreater(pred['color_logits'].grad[0, 0, 0].item(), 0)
        self.assertGreater(pred['class_logits'].grad[0, 0, 0].item(), 0)

    def test_target_id_range_errors_are_explicit(self):
        for colors, classes in [([4], [0]), ([0], [9]), ([-1], [0])]:
            with self.subTest(colors=colors, classes=classes), self.assertRaises(ValueError):
                ArmorPoseLoss()(prediction(), [target([BOX], colors, classes)], (32, 32))

    def test_invalid_stride_is_rejected(self):
        pred = prediction()
        pred['strides'][0] = 0
        with self.assertRaises(ValueError):
            ArmorPoseLoss()(pred, [target([BOX])], (32, 32))

    def test_stats_do_not_retain_the_backward_graph(self):
        _, stats = ArmorPoseLoss()(prediction(), [target([BOX])], (32, 32))
        for value in stats.values():
            if isinstance(value, torch.Tensor):
                self.assertFalse(value.requires_grad)


if __name__ == '__main__':
    unittest.main()
