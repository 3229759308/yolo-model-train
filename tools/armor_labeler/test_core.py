"""Meaningful contract tests for the custom armor head and annotation storage."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

import core


def prediction(candidates, count=6300):
    """Build the precise color(4), class(9), points(8) model output."""
    output = np.zeros((1, 21, count), dtype=np.float32)
    for index, color_id, class_id, color_score, class_score, points in candidates:
        output[0, color_id, index] = color_score
        output[0, 4 + class_id, index] = class_score
        output[0, 13:21, index] = np.asarray(points, dtype=np.float32).reshape(8)
    return output


def annotation(points=None, color_id=1, class_id=3):
    if points is None:
        points = [[100, 80], [100, 160], [200, 160], [200, 80]]
    return core.ArmorAnnotation(color_id=color_id, class_id=class_id, points=points)


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.config = core.default_config()
        self.points = [[1000, 700], [1000, 900], [1200, 900], [1200, 700]]

    def test_21_channel_head_and_original_pixel_coordinates(self):
        output = prediction([(10, 2, 7, 0.9, 0.8, self.points)])
        result = core.decode_output(output, self.config)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].color_id, 2)
        self.assertEqual(result[0].class_id, 7)
        np.testing.assert_allclose(result[0].points, self.points)
        self.assertAlmostEqual(result[0].confidence, 0.72, places=5)

    def test_decoder_keeps_raw_points_for_both_coordinate_modes(self):
        output = prediction([(10, 2, 7, 0.9, 0.8, self.points)])
        self.config['output_coordinates'] = 'input'
        result = core.decode_output(output, self.config)
        np.testing.assert_allclose(result[0].points, self.points)

    def test_nonfinite_and_degenerate_candidates_are_discarded(self):
        bad_points = [[float('nan'), 700], [1000, 900], [1200, 900], [1200, 700]]
        output = prediction([
            (0, 0, 0, 0.9, 0.9, bad_points),
            (1, 0, 0, 0.9, 0.9, [[100, 100]] * 4),
            (2, 2, 7, 0.9, 0.8, self.points),
        ])
        result = core.decode_output(output, self.config)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].class_id, 7)

    def test_candidate_major_layout_is_equivalent(self):
        output = prediction([(4, 3, 8, 0.9, 0.8, self.points)])
        a = core.decode_output(output, self.config)
        b = core.decode_output(output.transpose(0, 2, 1), self.config)
        self.assertEqual(len(a), len(b))
        self.assertEqual(a[0].color_id, b[0].color_id)
        self.assertEqual(a[0].class_id, b[0].class_id)
        np.testing.assert_allclose(a[0].points, b[0].points)

    def test_confidence_filters_product_of_branches(self):
        output = prediction([(0, 0, 0, 0.5, 0.5, self.points)])
        self.assertEqual(len(core.decode_output(output, self.config, confidence=0.3)), 0)
        self.assertEqual(len(core.decode_output(output, self.config, confidence=0.2)), 1)

    def test_nms_suppresses_duplicate_but_keeps_other_class(self):
        points = [[100, 80], [100, 160], [200, 160], [200, 80]]
        output = prediction([
            (0, 1, 3, 0.95, 0.95, points),
            (1, 1, 3, 0.9, 0.9, points),
            (2, 1, 4, 0.9, 0.9, points),
        ])
        result = core.decode_output(output, self.config)
        self.assertEqual(sorted(item.class_id for item in result), [3, 4])
        self.config['class_aware'] = False
        self.assertEqual(len(core.decode_output(output, self.config)), 1)

    def test_empty_output_and_invalid_shape(self):
        self.assertEqual(core.decode_output(np.zeros((1, 21, 6300), np.float32), self.config), [])
        with self.assertRaises(ValueError):
            core.decode_output(np.zeros((1, 20, 6300), np.float32), self.config)


class PreprocessTests(unittest.TestCase):
    def test_nchw_rgb_normalized_stretch(self):
        config = core.default_config()
        image = np.zeros((108, 144, 3), dtype=np.uint8)
        image[:] = [10, 20, 255]  # BGR
        tensor = core.preprocess_image(image, config)
        self.assertEqual(tensor.shape, (1, 3, 480, 640))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_allclose(tensor[0, :, 200, 200], [1.0, 20 / 255, 10 / 255], atol=1e-6)

    def test_bgr_and_no_normalization_are_configurable(self):
        config = core.default_config()
        config.update(rgb=False, normalize=False)
        image = np.full((2, 4, 3), [10, 20, 30], dtype=np.uint8)
        tensor = core.preprocess_image(image, config)
        np.testing.assert_allclose(tensor[0, :, 200, 200], [10, 20, 30])

    def test_letterbox_preserves_aspect_ratio(self):
        config = core.default_config()
        config['preprocess'] = 'letterbox'
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        tensor = core.preprocess_image(image, config)
        self.assertEqual(tensor.shape, (1, 3, 480, 640))
        self.assertAlmostEqual(float(tensor[0, 0, 240, 320]), 1.0)
        self.assertLess(float(tensor[0, 0, 240, 0]), 1.0)

    def test_bad_image_is_rejected(self):
        with self.assertRaises(ValueError):
            core.preprocess_image(np.zeros((0, 0, 3), np.uint8), core.default_config())


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images = self.root / 'images'
        self.labels = self.root / 'labels'
        (self.images / 'sub').mkdir(parents=True)
        self.image = self.images / 'sub' / 'a.jpg'
        self.image.write_bytes(b'not decoded by storage')
        self.store = core.AnnotationStore(self.images, self.labels)

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_stem_images_have_distinct_annotation_files(self):
        other = self.image.with_suffix('.png')
        other.touch()
        first = self.store.label_path(self.image)
        second = self.store.label_path(other)
        self.assertNotEqual(first, second)
        self.assertEqual(first, self.labels / 'sub' / 'a.jpg.json')
        self.assertEqual(second, self.labels / 'sub' / 'a.png.json')

    def test_empty_annotations_are_valid_and_saved(self):
        self.assertIsNone(self.store.load(self.image))
        path = self.store.save(self.image, 640, 480, [], reviewed=True)
        self.assertTrue(path.is_file())
        data = self.store.load(self.image)
        self.assertEqual(data['annotations'], [])
        self.assertTrue(data['reviewed'])

    def test_pixel_coordinates_and_attributes_round_trip(self):
        item = annotation([[1000.25, 700], [1000, 900], [1200, 900], [1200, 700]])
        self.store.save(self.image, 1440, 1080, [item])
        data = self.store.load(self.image)
        restored = data['annotations'][0]
        if isinstance(restored, dict):
            restored = core.ArmorAnnotation.from_dict(restored)
        self.assertEqual(restored.color_id, 1)
        self.assertEqual(restored.class_id, 3)
        np.testing.assert_allclose(restored.points, item.points)

    def test_corrupt_existing_json_is_rejected(self):
        path = self.store.label_path(self.image)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{bad json', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.store.load(self.image)
        self.assertEqual(path.read_text(encoding='utf-8'), '{bad json')
        with self.assertRaises(ValueError):
            self.store.save(self.image, 640, 480, [])
        self.assertEqual(path.read_text(encoding='utf-8'), '{bad json')

    def test_second_save_keeps_a_backup_of_previous_labels(self):
        path = self.store.save(self.image, 640, 480, [annotation()])
        original = path.read_text(encoding='utf-8')
        self.store.save(self.image, 640, 480, [], reviewed=True)
        self.assertEqual(path.with_name(path.name + '.bak').read_text(encoding='utf-8'), original)
        self.assertEqual(self.store.load(self.image)['annotations'], [])

    def test_invalid_annotation_cannot_be_saved(self):
        item = annotation()
        item.points = [[1, 2]]  # GUI objects are mutable; save must revalidate.
        with self.assertRaises(ValueError):
            self.store.save(self.image, 640, 480, [item])
        self.assertFalse(self.store.label_path(self.image).exists())

    def test_nan_coordinate_cannot_be_saved(self):
        with self.assertRaises(ValueError):
            self.store.save(self.image, 640, 480, [annotation([[float('nan'), 0], [0, 1], [1, 1], [1, 0]])])


class ConfigTests(unittest.TestCase):
    def test_validation_does_not_mutate_caller_configuration(self):
        supplied = {'point_names': ['LT', 'LB', 'RB', 'RT']}
        validated = core.validate_config(supplied)
        validated['point_names'][0] = 'changed'
        self.assertEqual(supplied['point_names'][0], 'LT')
        self.assertEqual(validated['input_width'], 640)

    def test_color_and_class_counts_are_fixed_by_head(self):
        for key, names in [('color_names', ['red']), ('class_names', ['armor'])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                core.validate_config({key: names})

    def test_output_coordinate_system_is_explicit_and_validated(self):
        self.assertEqual(core.default_config()['output_coordinates'], 'original')
        for coordinate_system in ('original', 'input'):
            self.assertEqual(core.validate_config({'output_coordinates': coordinate_system})['output_coordinates'],
                             coordinate_system)
        with self.assertRaises(ValueError):
            core.validate_config({'output_coordinates': 'network'})

    def test_model_path_accepts_empty_or_string_and_rejects_other_types(self):
        self.assertEqual(core.validate_config({})['model_path'], '')
        self.assertEqual(core.validate_config({'model_path': '../model.onnx'})['model_path'], '../model.onnx')
        with self.assertRaises(ValueError):
            core.validate_config({'model_path': 17})


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'labels.txt'

    def tearDown(self):
        self.tmp.cleanup()

    def test_custom_export_preserves_branch_ids_and_pixel_coordinates(self):
        item = annotation()
        core.export_custom(self.path, [item])
        values = self.path.read_text(encoding='utf-8').split()
        self.assertEqual(len(values), 10)
        self.assertEqual(list(map(int, values[:2])), [1, 3])
        np.testing.assert_allclose(list(map(float, values[2:])), np.asarray(item.points).reshape(8))

    def test_yolo_export_uses_combined_class_and_normalized_coordinates(self):
        core.export_yolo(self.path, [annotation()], 400, 200)
        values = list(map(float, self.path.read_text(encoding='utf-8').split()))
        self.assertEqual(len(values), 17)
        self.assertEqual(values[0], 12)
        np.testing.assert_allclose(values[1:5], [0.375, 0.6, 0.25, 0.4], atol=1e-6)
        np.testing.assert_allclose(values[5:8], [0.25, 0.4, 2], atol=1e-6)

    def test_yolo_rejects_outside_image_points(self):
        with self.assertRaises(ValueError):
            core.export_yolo(self.path, [annotation()], 100, 100)
        self.assertFalse(self.path.exists())

    def test_empty_custom_export_is_a_valid_negative_label(self):
        core.export_custom(self.path, [])
        self.assertEqual(self.path.read_text(encoding='utf-8'), '')


if __name__ == '__main__':
    unittest.main()
