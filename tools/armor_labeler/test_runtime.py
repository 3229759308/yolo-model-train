"""Provider selection regressions without a GPU or an installed ONNX Runtime.

The fake runtime exposes the providers a session actually registers, including
silent CUDA fallback, so configured preferences cannot be mistaken for GPU use.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from core import OnnxDetector, default_config, validate_config


CPU = 'CPUExecutionProvider'
CUDA = 'CUDAExecutionProvider'


class FakeSessionOptions:
    def __init__(self):
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None
        self.execution_mode = None
        self.entries = {}

    def add_session_config_entry(self, key, value):
        self.entries[key] = value


class FakeRuntime:
    def __init__(self, *, available=(CUDA, CPU), actual=None, cuda_fails=False,
                 preload_fails=False):
        self.available = list(available)
        self.actual = actual
        self.cuda_fails = cuda_fails
        self.preload_fails = preload_fails
        self.events = []
        self.calls = []
        self.fallback_disabled = False
        self.SessionOptions = FakeSessionOptions
        self.ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL=object())

    def get_available_providers(self):
        return list(self.available)

    def preload_dlls(self):
        self.events.append('preload')
        if self.preload_fails:
            raise OSError('CUDA dependencies are missing')

    def InferenceSession(self, model_path, *, sess_options=None, providers=None):
        self.events.append('session')
        requested = list(providers)
        self.calls.append((model_path, sess_options, requested))
        if self.cuda_fails and CUDA in requested:
            raise RuntimeError('CUDA init failed')
        actual = list(self.actual if self.actual is not None else requested)
        return SimpleNamespace(
            get_providers=lambda: actual,
            disable_fallback=lambda: setattr(self, 'fallback_disabled', True),
            get_inputs=lambda: [SimpleNamespace(
                name='images', shape=[1, 3, 480, 640], type='tensor(float16)')],
            get_outputs=lambda: [SimpleNamespace(name='output', shape=[1, 21, 6300])],
        )


class RuntimeConfigurationTests(unittest.TestCase):
    def test_defaults_prefer_cuda_with_cpu_fallback_and_limit_cpu_work(self):
        config = default_config()
        self.assertEqual(config['providers'], [CUDA, CPU])
        self.assertEqual(config['cpu_threads'], 2)
        self.assertIs(config['thread_spinning'], False)

    def test_cpu_threads_must_be_positive_integer(self):
        for value in (0, -1, True, False, 2.5, '2', None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'cpu_threads'):
                    validate_config({'cpu_threads': value})
        self.assertEqual(validate_config({'cpu_threads': 4})['cpu_threads'], 4)

    def test_thread_spinning_must_be_boolean(self):
        for value in (0, 1, 'false', None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'thread_spinning'):
                    validate_config({'thread_spinning': value})
        self.assertIs(validate_config({'thread_spinning': True})['thread_spinning'], True)


class RuntimeProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='armor-runtime-test-')
        self.addCleanup(self.temp.cleanup)
        self.model_path = Path(self.temp.name) / 'model.onnx'
        self.model_path.write_bytes(b'fake runtime does not parse this file')

    def detector(self, runtime, **settings):
        config = default_config()
        config.update(settings)
        with patch.dict('sys.modules', {'onnxruntime': runtime}):
            return OnnxDetector(self.model_path, config)

    def assert_cpu_fallback_visible(self, detector):
        self.assertEqual(detector.providers, [CPU])
        self.assertIn('CPU', detector.backend_summary)
        self.assertTrue(detector.warnings, 'GPU fallback must remain visible to the user')
        self.assertTrue(all(isinstance(warning, str) and warning for warning in detector.warnings))

    def test_cuda_preloads_libraries_before_session_and_reports_actual_gpu(self):
        runtime = FakeRuntime()
        detector = self.detector(runtime)
        self.assertEqual(runtime.events, ['preload', 'session'])
        self.assertEqual(runtime.calls[0][2], [CUDA, CPU])
        self.assertEqual(detector.providers, [CUDA, CPU])
        self.assertIn('GPU', detector.backend_summary)
        self.assertIn('CUDA', detector.backend_summary)
        self.assertEqual(detector.warnings, [])
        self.assertTrue(runtime.fallback_disabled, 'runtime fallback must not silently change the displayed backend')

    def test_cpu_only_does_not_load_cuda_libraries(self):
        runtime = FakeRuntime()
        detector = self.detector(runtime, providers=[CPU])
        self.assertEqual(runtime.events, ['session'])
        self.assertEqual(runtime.calls[0][2], [CPU])
        self.assertEqual(detector.providers, [CPU])
        self.assertIn('CPU', detector.backend_summary)
        self.assertEqual(detector.warnings, [])

    def test_session_options_limit_threads_and_disable_spinning(self):
        runtime = FakeRuntime()
        detector = self.detector(runtime, providers=[CPU])
        options = runtime.calls[0][1]
        self.assertIs(options, detector.session_options)
        self.assertEqual(options.intra_op_num_threads, 2)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertIs(options.execution_mode, runtime.ExecutionMode.ORT_SEQUENTIAL)
        self.assertEqual(options.entries['session.intra_op.allow_spinning'], '0')
        self.assertEqual(options.entries['session.inter_op.allow_spinning'], '0')

    def test_configured_threads_and_spinning_are_applied(self):
        runtime = FakeRuntime()
        detector = self.detector(runtime, providers=[CPU], cpu_threads=4, thread_spinning=True)
        self.assertEqual(detector.session_options.intra_op_num_threads, 4)
        self.assertEqual(detector.session_options.entries['session.intra_op.allow_spinning'], '1')
        self.assertEqual(detector.session_options.entries['session.inter_op.allow_spinning'], '1')

    def test_cuda_not_installed_falls_back_to_cpu_with_explanation(self):
        runtime = FakeRuntime(available=[CPU])
        detector = self.detector(runtime)
        self.assertEqual(runtime.calls[0][2], [CPU])
        self.assert_cpu_fallback_visible(detector)

    def test_silent_runtime_fallback_reports_cpu_instead_of_requested_cuda(self):
        runtime = FakeRuntime(actual=[CPU])
        detector = self.detector(runtime)
        self.assertEqual(runtime.calls[0][2], [CUDA, CPU])
        self.assert_cpu_fallback_visible(detector)

    def test_cuda_session_creation_failure_retries_cpu_and_reports_it(self):
        runtime = FakeRuntime(cuda_fails=True)
        detector = self.detector(runtime)
        self.assertEqual([call[2] for call in runtime.calls], [[CUDA, CPU], [CPU]])
        self.assert_cpu_fallback_visible(detector)

    def test_preload_failure_still_obeys_actual_session_provider(self):
        runtime = FakeRuntime(preload_fails=True, actual=[CPU])
        detector = self.detector(runtime)
        self.assert_cpu_fallback_visible(detector)

    def test_cuda_only_unavailable_does_not_silently_add_cpu(self):
        runtime = FakeRuntime(available=[CPU])
        with self.assertRaises((ValueError, RuntimeError)):
            self.detector(runtime, providers=[CUDA])
        self.assertEqual(runtime.calls, [])

    def test_cuda_only_silent_fallback_is_an_error(self):
        runtime = FakeRuntime(actual=[CPU])
        with self.assertRaises((ValueError, RuntimeError)):
            self.detector(runtime, providers=[CUDA])

    def test_cuda_only_creation_failure_is_an_error(self):
        runtime = FakeRuntime(cuda_fails=True)
        with self.assertRaises((ValueError, RuntimeError)):
            self.detector(runtime, providers=[CUDA])
        self.assertEqual(len(runtime.calls), 1)

    def test_unknown_provider_is_an_error_even_with_cpu_fallback(self):
        runtime = FakeRuntime()
        with self.assertRaisesRegex(ValueError, 'provider'):
            self.detector(runtime, providers=['NotARealExecutionProvider', CPU])
        self.assertEqual(runtime.calls, [])


if __name__ == '__main__':
    unittest.main()
