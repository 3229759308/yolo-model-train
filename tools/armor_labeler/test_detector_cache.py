"""Regression tests for reuse, invalidation, and failed detector construction."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
import tempfile
import threading
import unittest
import weakref

from detector_cache import DetectorCache


class FakeDetector:
    backend_summary = 'GPU · CUDA / CPU'


class DetectorCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='armor-cache-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / 'model.onnx'
        self.model.write_bytes(b'fixture')
        self.cache = DetectorCache()
        self.calls = []
        self.config = {'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                       'nested': {'threshold': 0.6}}

        def factory(path, config):
            self.calls.append((path, config))
            return FakeDetector()

        self.factory = factory

    def test_same_model_and_full_config_reuse_one_instance(self):
        first = self.cache.get(self.model, self.config, factory=self.factory)
        reordered = {'nested': {'threshold': 0.6}, 'providers': self.config['providers'].copy()}
        second = self.cache.get(self.root / '.' / 'model.onnx', reordered, factory=self.factory)
        self.assertIs(first, second)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(second.backend_summary, 'GPU · CUDA / CPU')
        self.assertEqual(self.calls[0][0], self.model.resolve())

    def test_nested_config_mutation_invalidates_independent_snapshot(self):
        first = self.cache.get(self.model, self.config, factory=self.factory)
        self.config['nested']['threshold'] = 0.8
        self.assertEqual(self.calls[0][1]['nested']['threshold'], 0.6)
        second = self.cache.get(self.model, self.config, factory=self.factory)
        self.assertIsNot(first, second)
        self.assertEqual(len(self.calls), 2)

    def test_factory_cannot_mutate_callers_config(self):
        def mutating_factory(path, config):
            config['nested']['threshold'] = 0
            return FakeDetector()

        self.cache.get(self.model, self.config, factory=mutating_factory)
        self.assertEqual(self.config['nested']['threshold'], 0.6)

    def test_new_model_path_and_factory_identity_invalidate(self):
        first = self.cache.get(self.model, self.config, factory=self.factory)
        second_path = self.root / 'other.onnx'
        second_path.write_bytes(self.model.read_bytes())
        second = self.cache.get(second_path, self.config, factory=self.factory)
        self.assertIsNot(first, second)
        third = self.cache.get(second_path, self.config, factory=lambda path, config: FakeDetector())
        self.assertIsNot(second, third)

    def test_mtime_and_size_independently_invalidate(self):
        first = self.cache.get(self.model, self.config, factory=self.factory)
        status = self.model.stat()
        os.utime(self.model, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000))
        second = self.cache.get(self.model, self.config, factory=self.factory)
        self.assertIsNot(first, second)
        status = self.model.stat()
        self.model.write_bytes(b'a longer fixture')
        os.utime(self.model, ns=(status.st_atime_ns, status.st_mtime_ns))
        third = self.cache.get(self.model, self.config, factory=self.factory)
        self.assertIsNot(second, third)
        self.assertEqual(len(self.calls), 3)

    def test_missing_model_is_valid_key_until_factory_validates_or_file_appears(self):
        missing = self.root / 'not-yet-present.onnx'
        first = self.cache.get(missing, self.config, factory=self.factory)
        self.assertIs(first, self.cache.get(missing, self.config, factory=self.factory))
        missing.write_bytes(b'new model')
        self.assertIsNot(first, self.cache.get(missing, self.config, factory=self.factory))

    def test_factory_failure_is_not_cached_and_next_request_retries(self):
        attempts = []

        def unreliable(path, config):
            attempts.append(path)
            if len(attempts) == 1:
                raise RuntimeError('model load failed')
            return FakeDetector()

        with self.assertRaisesRegex(RuntimeError, 'model load failed'):
            self.cache.get(self.model, self.config, factory=unreliable)
        detector = self.cache.get(self.model, self.config, factory=unreliable)
        self.assertIs(detector, self.cache.get(self.model, self.config, factory=unreliable))
        self.assertEqual(len(attempts), 2)

    def test_invalidation_releases_old_instance_even_when_new_factory_fails(self):
        detector = self.cache.get(self.model, self.config, factory=self.factory)
        reference = weakref.ref(detector)
        del detector

        def failing(path, config):
            self.assertIsNone(reference())
            raise RuntimeError('failed replacement')

        with self.assertRaisesRegex(RuntimeError, 'failed replacement'):
            self.cache.get(self.model, self.config, factory=failing)
        self.assertIsNone(reference())

    def test_clear_releases_cache_and_next_request_rebuilds(self):
        detector = self.cache.get(self.model, self.config, factory=self.factory)
        reference = weakref.ref(detector)
        del detector
        self.cache.clear()
        gc.collect()
        self.assertIsNone(reference())
        self.cache.get(self.model, self.config, factory=self.factory)
        self.assertEqual(len(self.calls), 2)

    def test_concurrent_get_constructs_once(self):
        started = threading.Event()
        release = threading.Event()
        attempts = []

        def blocked_factory(path, config):
            attempts.append(path)
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return FakeDetector()

        with ThreadPoolExecutor(max_workers=4) as pool:
            first = pool.submit(self.cache.get, self.model, self.config, factory=blocked_factory)
            self.assertTrue(started.wait(timeout=5))
            others = [pool.submit(self.cache.get, self.model, self.config, factory=blocked_factory)
                      for _ in range(3)]
            release.set()
            detector = first.result(timeout=5)
            self.assertTrue(all(result.result(timeout=5) is detector for result in others))
        self.assertEqual(len(attempts), 1)


if __name__ == '__main__':
    unittest.main()
