"""Detection metric and postprocessing checks for the custom armor protocol."""
from __future__ import annotations

import io
import json
import unittest

import numpy as np
import torch

from armor_pose.common.runtime import DetectionMetrics, decode_predictions


def box_points(left=0., top=0., right=20., bottom=10.):
    return np.asarray([[left, top], [left, bottom], [right, bottom], [right, top]], dtype=np.float32)


def target(points=(), colors=None, classes=None):
    count = len(points)
    return {
        'points': np.asarray(points, dtype=np.float32).reshape(count, 4, 2),
        'color_ids': np.asarray(colors if colors is not None else [0] * count, dtype=np.int64),
        'class_ids': np.asarray(classes if classes is not None else [1] * count, dtype=np.int64),
    }


def prediction(points=(), scores=None, colors=None, classes=None):
    result = target(points, colors, classes)
    result['scores'] = np.asarray(scores if scores is not None else [.9] * len(points), dtype=np.float32)
    return result


def packed(points=(), scores=None, colors=None, classes=None):
    detections = prediction(points, scores, colors, classes)
    count = len(detections['points'])
    output = np.zeros((21, count), dtype=np.float32)
    for index in range(count):
        confidence = np.sqrt(detections['scores'][index])
        output[detections['color_ids'][index], index] = confidence
        output[4 + detections['class_ids'][index], index] = confidence
        output[13:, index] = detections['points'][index].reshape(-1)
    return output


class ArmorPoseRuntimeTests(unittest.TestCase):
    def test_matched_metrics_can_be_json_and_safe_checkpoint_serialized(self):
        metrics = DetectionMetrics()
        metrics.update(prediction([box_points(1, 0, 21, 10)]), target([box_points()]))
        result = metrics.compute()
        self.assertIs(type(result['corner_error_input_px']), float)
        self.assertEqual(json.loads(json.dumps(result)), result)
        stream = io.BytesIO()
        torch.save({'metrics': result}, stream)
        stream.seek(0)
        self.assertEqual(torch.load(stream, weights_only=True)['metrics'], result)

    def test_perfect_prediction_has_unit_ap_and_zero_corner_error(self):
        points = [box_points(), box_points(30, 5, 60, 20)]
        metrics = DetectionMetrics()
        metrics.update(prediction(points, [.9, .8], [0, 1], [1, 8]), target(points, [0, 1], [1, 8]))
        result = metrics.compute()
        self.assertEqual(result['mAP50'], 1.)
        self.assertEqual(result['mAP50_95'], 1.)
        self.assertEqual(result['corner_error_input_px'], 0.)
        self.assertEqual(result['corner_matches'], 2)
        self.assertEqual(len(result['per_color_class']), 2)

    def test_missed_target_contributes_zero_unreached_recall(self):
        points = [box_points(), box_points(30, 5, 60, 20)]
        metrics = DetectionMetrics()
        metrics.update(prediction(points[:1]), target(points))
        self.assertAlmostEqual(metrics.compute()['mAP50'], 51 / 101)

    def test_high_confidence_false_positive_reduces_precision(self):
        actual, false = box_points(), box_points(30, 5, 60, 20)
        metrics = DetectionMetrics()
        metrics.update(prediction([false, actual], [.95, .9]), target([actual]))
        result = metrics.compute()
        self.assertEqual(result['mAP50'], .5)
        self.assertEqual(result['predictions'], 2)
        self.assertEqual(result['corner_matches'], 1)

    def test_duplicate_predictions_cannot_match_a_target_twice(self):
        actual, missed = box_points(), box_points(30, 5, 60, 20)
        metrics = DetectionMetrics()
        metrics.update(prediction([actual, actual], [.95, .9]), target([actual, missed]))
        result = metrics.compute()
        self.assertAlmostEqual(result['mAP50'], 51 / 101)
        self.assertEqual(result['corner_matches'], 1)

    def test_matching_consumes_the_best_remaining_target(self):
        points = [box_points(0, 0, 20, 10), box_points(5, 0, 25, 10)]
        metrics = DetectionMetrics()
        metrics.update(prediction(points, [.95, .9]), target(points))
        self.assertEqual(metrics.compute()['mAP50_95'], 1.)

    def test_color_and_class_mismatch_cannot_match(self):
        point = box_points()
        for colors, classes in [([1], [1]), ([0], [2])]:
            with self.subTest(colors=colors, classes=classes):
                metrics = DetectionMetrics()
                metrics.update(prediction([point], colors=colors, classes=classes), target([point]))
                result = metrics.compute()
                self.assertEqual(result['mAP50_95'], 0.)
                self.assertIsNone(result['corner_error_input_px'])

    def test_matching_uses_each_iou_threshold_including_exact_boundary(self):
        # Intersection 3 / union 4 gives exactly .75, which must qualify at .75.
        metrics = DetectionMetrics()
        metrics.update(prediction([box_points(0, 0, 3, 1)]), target([box_points(0, 0, 4, 1)]))
        result = metrics.compute()
        self.assertEqual(result['mAP50'], 1.)
        self.assertAlmostEqual(result['mAP50_95'], .6)

    def test_empty_predictions_and_empty_targets_are_stable(self):
        for predictions, targets in [
            (prediction(), target()),
            (prediction(), target([box_points()])),
            (prediction([box_points()]), target()),
        ]:
            with self.subTest(predictions=len(predictions['points']), targets=len(targets['points'])):
                metrics = DetectionMetrics()
                metrics.update(predictions, targets)
                result = metrics.compute()
                self.assertEqual(result['mAP50_95'], 0.)
                self.assertIsNone(result['corner_error_input_px'])
                self.assertEqual(result['corner_matches'], 0)
        result = DetectionMetrics().compute()
        self.assertEqual(result['images'], 0)
        self.assertEqual(result['mAP50'], 0.)

    def test_ap_accumulates_missed_objects_across_images(self):
        metrics = DetectionMetrics()
        metrics.update(prediction([box_points()]), target([box_points()]))
        metrics.update(prediction(), target([box_points()]))
        result = metrics.compute()
        self.assertEqual(result['images'], 2)
        self.assertEqual(result['targets'], 2)
        self.assertAlmostEqual(result['mAP50'], 51 / 101)

    def test_ordered_corner_error_remains_independent_of_box_ap(self):
        point = box_points()
        metrics = DetectionMetrics()
        metrics.update(prediction([point[[1, 0, 3, 2]]]), target([point]))
        result = metrics.compute()
        self.assertEqual(result['mAP50_95'], 1.)
        self.assertEqual(result['corner_error_input_px'], 10.)

    def test_nms_suppresses_only_the_same_color_and_class_pair(self):
        point = box_points()
        output = packed([point] * 4, [.95, .9, .85, .8], [0, 0, 1, 0], [1, 1, 1, 2])
        result = decode_predictions(output, confidence=.1, nms_iou=.5)
        self.assertEqual(len(result['points']), 3)
        self.assertEqual(list(zip(result['color_ids'], result['class_ids'])), [(0, 1), (1, 1), (0, 2)])
        np.testing.assert_allclose(result['scores'], [.95, .85, .8], rtol=1e-6)

    def test_decode_uses_branch_product_confidence_and_max_det(self):
        output = packed([box_points(), box_points(30, 0, 50, 10)], [.3, .8])
        result = decode_predictions(output, confidence=.5, max_det=1)
        self.assertEqual(len(result['points']), 1)
        np.testing.assert_allclose(result['scores'], [.8], rtol=1e-6)

    def test_decode_preserves_outside_corners(self):
        point = box_points(-5, -3, 20, 10)
        result = decode_predictions(packed([point]))
        np.testing.assert_equal(result['points'][0], point)

    def test_decode_filters_nonfinite_and_degenerate_points(self):
        points = [box_points(), box_points(0, 0, 0, 10), box_points()]
        output = packed(points)
        output[13, 2] = np.nan
        result = decode_predictions(output)
        self.assertEqual(len(result['points']), 1)

    def test_decode_handles_empty_output_and_rejects_invalid_shape(self):
        self.assertEqual(len(decode_predictions(np.empty((21, 0)))['points']), 0)
        for invalid in [np.array(1.), np.zeros(21), np.zeros((1, 21, 10)), np.zeros((20, 10))]:
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                decode_predictions(invalid)


if __name__ == '__main__':
    unittest.main()
