"""Real ONNX Runtime integration tests; generated models live only in TemporaryDirectory.

Run using a Python with onnx + onnxruntime installed. A synthetic graph verifies
our adapter's declared coordinate contract; it cannot validate a third-party model.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from core import OnnxDetector, default_config as _default_config

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import onnxruntime  # noqa: F401 -- explicitly test runtime availability
except ImportError:
    onnx = None


def default_config():
    """Keep numerical adapter checks deterministic across CPU and CUDA hosts."""
    config = _default_config()
    config['providers'] = ['CPUExecutionProvider']
    return config


@unittest.skipUnless(onnx is not None, 'optional onnx and onnxruntime are required')
class OnnxIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='armor-onnx-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_model(self, *, dtype='float32', input_shape=(1, 3, 480, 640),
                   channels=21, transposed=False, inputs=1, red_dependent=False, points=None):
        """Build the requested graph without writing any fixture into the repository."""
        tensor_type = {'float32': TensorProto.FLOAT, 'float16': TensorProto.FLOAT16,
                       'uint8': TensorProto.UINT8}[dtype]
        output_type = TensorProto.FLOAT16 if dtype == 'float16' else TensorProto.FLOAT
        output_dtype = np.float16 if dtype == 'float16' else np.float32
        values = np.zeros((1, channels, 6300), dtype=output_dtype)
        if channels == 21:
            values[0, 0, 0] = 0.5 if red_dependent else 0.9
            values[0, 4 + 6, 0] = 0.8
            values[0, 13:21, 0] = ([1100, 900, 1100, 1000, 1200, 1000, 1200, 900]
                                         if points is None else np.asarray(points).reshape(8))
        if transposed:
            values = values.transpose(0, 2, 1)
        nodes, initializers = [], []
        if red_dependent:
            # Use the first input channel: this makes the result sensitive to
            # BGR -> RGB, normalization, input shape, and float16 conversion.
            initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), 'channel_zero'))
            initializers.append(numpy_helper.from_array(values, 'base'))
            mask = np.zeros_like(values)
            mask[0, 0, 0] = 1
            initializers.append(numpy_helper.from_array(mask, 'mask'))
            nodes.extend([
                helper.make_node('Gather', ['images', 'channel_zero'], ['red'], axis=1),
                helper.make_node('ReduceMean', ['red'], ['mean_red'], keepdims=0),
                helper.make_node('Mul', ['mean_red', 'mask'], ['adjustment']),
                helper.make_node('Add', ['base', 'adjustment'], ['output']),
            ])
        else:
            nodes.append(helper.make_node('Constant', [], ['output'],
                                          value=numpy_helper.from_array(values)))
        graph_inputs = [helper.make_tensor_value_info('images', tensor_type, list(input_shape))]
        for index in range(1, inputs):
            graph_inputs.append(helper.make_tensor_value_info(f'additional_{index}', tensor_type, [1]))
        graph = helper.make_graph(nodes, 'synthetic-armor-head', graph_inputs,
                                  [helper.make_tensor_value_info('output', output_type, list(values.shape))],
                                  initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        # Current onnx may produce a newer IR than some deployed runtime versions.
        model.ir_version = 8
        onnx.checker.check_model(model)
        path = self.root / f'model-{len(list(self.root.glob("*.onnx")))}.onnx'
        onnx.save(model, str(path))
        return path

    def test_float32_end_to_end_rgb_normalization_and_original_pixels(self):
        detector = OnnxDetector(self.make_model(red_dependent=True), default_config())
        image = np.empty((1200, 1600, 3), dtype=np.uint8)
        image[:] = [10, 20, 30]
        result = detector.predict(image)
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0].color_id, result[0].class_id), (0, 6))
        self.assertAlmostEqual(result[0].confidence, (0.5 + 30 / 255) * 0.8, places=4)
        self.assertEqual(result[0].points, [[1100, 900], [1100, 1000], [1200, 1000], [1200, 900]])
        self.assertEqual(result[0].source, 'model')
        self.assertEqual(detector.providers, ['CPUExecutionProvider'])

    def test_float16_model_receives_float16_blob(self):
        detector = OnnxDetector(self.make_model(dtype='float16', red_dependent=True), default_config())
        self.assertEqual(detector.input_dtype, np.float16)
        image = np.full((1200, 1600, 3), 51, dtype=np.uint8)
        result = detector.predict(image)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.7 * 0.8, places=3)
        self.assertEqual(result[0].points[0], [1100, 900])

    def test_candidate_major_output(self):
        detector = OnnxDetector(self.make_model(transposed=True), default_config())
        result = detector.predict(np.zeros((1200, 1600, 3), dtype=np.uint8))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].points[-1], [1200, 900])

    def test_letterbox_does_not_undo_original_pixel_output(self):
        config = default_config()
        config['preprocess'] = 'letterbox'
        detector = OnnxDetector(self.make_model(), config)
        for height, width in [(1200, 1600), (1080, 1920)]:
            with self.subTest(height=height, width=width):
                result = detector.predict(np.zeros((height, width, 3), dtype=np.uint8))
                self.assertEqual(result[0].points[0], [1100, 900])
                self.assertEqual(result[0].points[-1], [1200, 900])

    def test_input_coordinate_stretch_uses_original_width_and_height(self):
        config = default_config()
        config['output_coordinates'] = 'input'
        raw = np.array([[100, 100], [100, 200], [200, 200], [200, 100]])
        detector = OnnxDetector(self.make_model(points=raw), config)
        result = detector.predict(np.zeros((1000, 1440, 3), dtype=np.uint8))
        expected = raw * np.array([1440 / 640, 1000 / 480])
        np.testing.assert_allclose(result[0].points, expected)

    def test_input_coordinate_letterbox_uses_integer_padding_and_actual_rounded_scales(self):
        config = default_config()
        config.update(output_coordinates='input', preprocess='letterbox')
        raw = np.array([[10, 40], [10, 200], [400, 200], [400, 40]])
        detector = OnnxDetector(self.make_model(points=raw), config)
        # First resize gives width 617 with left=11/right=12; second gives
        # height 213 with top=133/bottom=134. Separate scales retain rounding.
        for height, width, resized_w, resized_h, left, top in [
                (777, 999, 617, 480, 11, 0),
                (333, 1001, 640, 213, 0, 133)]:
            with self.subTest(height=height, width=width):
                result = detector.predict(np.zeros((height, width, 3), dtype=np.uint8))
                expected = (raw - np.array([left, top])) * np.array([width / resized_w, height / resized_h])
                np.testing.assert_allclose(result[0].points, expected)
                self.assertLess(np.asarray(result[0].points).min(), 0)  # No clipping.

    def test_standard_yolo_pose_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '21'):
            OnnxDetector(self.make_model(channels=56), default_config())

    def test_wrong_static_input_resolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '输入'):
            OnnxDetector(self.make_model(input_shape=(1, 3, 640, 640)), default_config())

    def test_additional_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '一个图像输入'):
            OnnxDetector(self.make_model(inputs=2), default_config())

    def test_integer_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'float32/float16'):
            OnnxDetector(self.make_model(dtype='uint8'), default_config())

    def test_dynamic_input_dimensions_are_supported(self):
        detector = OnnxDetector(self.make_model(input_shape=(1, 3, 'height', 'width')), default_config())
        result = detector.predict(np.zeros((1200, 1600, 3), dtype=np.uint8))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].points[2], [1200, 1000])


if __name__ == '__main__':
    unittest.main()
