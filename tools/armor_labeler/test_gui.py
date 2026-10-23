"""Offscreen GUI regression tests; no real model or image dataset is required.

Run with the existing desktop environment::

    QT_QPA_PLATFORM=offscreen ../bit-auto-label/.venv/bin/python -m unittest test_gui -v
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from PySide6.QtCore import QLibraryInfo, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

# Importing the OpenCV wheel can point Qt at its incompatible plugin directory.
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = (
    QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath) + '/platforms'
)
if 'cv2' in os.environ.get('QT_QPA_FONTDIR', ''):
    os.environ.pop('QT_QPA_FONTDIR', None)

import app as labeler
from core import ArmorAnnotation, default_config, load_config, save_config


POINTS = [[80.0, 70.0], [220.0, 75.0], [218.0, 160.0], [82.0, 158.0]]


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])
        cls.qt.setStyle('Fusion')
        cls.qt.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='armor-gui-')
        self.base = Path(self.temp.name)
        self.images = self.base / 'images'
        self.labels = self.base / 'labels'
        self.images.mkdir()
        for name in ('01.png', '02.png'):
            self.assertTrue(cv2.imwrite(str(self.images / name),
                                       np.full((480, 640, 3), 72, np.uint8)))
        self.args = argparse.Namespace(config=str(self.base / 'config.json'),
                                       model=None, images=None, labels=None)
        self.windows = []
        self.errors = []
        self.window = self.make_window()
        self.window.open_folder(self.images, self.labels)
        self.qt.processEvents()
        self.assertEqual(self.window.current_path, self.images / '01.png')

    def make_window(self, args=None):
        window = labeler.MainWindow(args or self.args)
        # Modal errors must be observable without blocking the offscreen runner.
        window.error = lambda title, error: self.errors.append((title, str(error)))
        window.show()
        self.qt.processEvents()
        self.windows.append(window)
        return window

    def tearDown(self):
        for window in reversed(self.windows):
            if window.worker and window.worker.isRunning():
                window.worker.requestInterruption()
                self.assertTrue(window.worker.wait(3000), 'test worker did not stop')
            window.autosave.stop()
            window.dirty = False
            window.close()
            window.deleteLater()
        self.qt.processEvents()
        self.temp.cleanup()

    def create_with_clicks(self):
        canvas = self.window.canvas
        canvas.fit()
        canvas.start_create()
        for point in POINTS:
            QTest.mouseClick(canvas, Qt.MouseButton.LeftButton,
                             pos=canvas.to_view(point).toPoint())
        self.qt.processEvents()
        self.assertEqual(len(self.window.annotations), 1)
        self.assertFalse(canvas.creating)
        self.assertEqual(canvas.draft, [])
        return self.window.annotations[0]

    @staticmethod
    def prediction():
        return ArmorAnnotation(color_id=1, class_id=4,
                               points=[[300, 200], [400, 200], [400, 280], [300, 280]],
                               confidence=.91, source='model')

    def test_four_click_drag_attributes_autosave_and_reopen(self):
        ann = self.create_with_clicks()
        for actual, expected in zip(ann.points, POINTS):
            np.testing.assert_allclose(actual, expected, atol=1 / self.window.canvas.scale)
        canvas = self.window.canvas
        target = [110.0, 95.0]
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton,
                         pos=canvas.to_view(ann.points[0]).toPoint())
        QTest.mouseMove(canvas, canvas.to_view(target).toPoint(), delay=10)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton,
                           pos=canvas.to_view(target).toPoint())
        np.testing.assert_allclose(ann.points[0], target, atol=1 / canvas.scale)
        self.window.color.setCurrentIndex(3)
        self.window.category.setCurrentIndex(8)
        expected = copy.deepcopy(ann.to_dict())
        self.assertTrue(self.window.dirty)
        QTest.qWait(self.window.autosave.interval() + 150)
        self.assertFalse(self.window.dirty)
        data = self.window.store.load(self.window.current_path)
        self.assertEqual(data['annotations'], [expected])
        self.window.navigate(1)
        self.assertEqual(self.window.current_path.name, '02.png')
        self.assertEqual(self.window.annotations, [])
        self.window.navigate(-1)
        self.assertEqual(self.window.annotations[0].to_dict(), expected)
        reopened = self.make_window()
        reopened.open_folder(self.images, self.labels)
        self.assertEqual(reopened.annotations[0].to_dict(), expected)
        self.assertEqual(self.errors, [])

    def test_undo_redo_creation_coordinates_category_and_delete(self):
        original = self.create_with_clicks().to_dict()
        self.window.undo()
        self.assertEqual(self.window.annotations, [])
        self.window.redo()
        self.assertEqual(self.window.annotations[0].to_dict(), original)
        self.window.spins[2][1].setValue(177.25)
        self.window.category.setCurrentIndex(7)
        changed = self.window.annotations[0].to_dict()
        self.window.undo()
        self.assertEqual(self.window.annotations[0].class_id, original['class_id'])
        self.assertEqual(self.window.annotations[0].points[2][1], 177.25)
        self.window.undo()
        self.assertEqual(self.window.annotations[0].to_dict(), original)
        self.window.redo()
        self.window.redo()
        self.assertEqual(self.window.annotations[0].to_dict(), changed)
        self.window.delete_selected()
        self.assertEqual(self.window.annotations, [])
        self.window.undo()
        self.assertEqual(self.window.annotations[0].to_dict(), changed)

    def test_reviewed_empty_image_is_saved_and_restored(self):
        self.assertIsNone(self.window.store.load(self.window.current_path))
        QTest.mouseClick(self.window.review, Qt.MouseButton.LeftButton)
        self.assertTrue(self.window.reviewed)
        self.window.navigate(1)
        data = self.window.store.load(self.images / '01.png')
        self.assertTrue(data['reviewed'])
        self.assertEqual(data['annotations'], [])
        self.window.navigate(-1)
        self.assertTrue(self.window.review.isChecked())
        self.assertEqual(self.window.file_states[str(self.window.current_path)], 'reviewed')
        self.assertEqual(self.errors, [])

    def test_corrupt_sidecar_cannot_replace_current_image_or_be_overwritten(self):
        path = self.window.store.label_path(self.images / '02.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        corrupt = b'{"annotations": [broken JSON'
        path.write_bytes(corrupt)
        self.create_with_clicks()
        original_path = self.window.current_path
        original_annotation = self.window.annotations[0].to_dict()
        self.window.navigate(1)
        self.assertEqual(self.window.current_path, original_path)
        self.assertEqual(self.window.files.currentRow(), 0)
        self.assertEqual(self.window.annotations[0].to_dict(), original_annotation)
        self.assertEqual(path.read_bytes(), corrupt)
        self.assertEqual(len(self.errors), 1)
        self.assertIn('读取失败', self.errors[0][0])

    def test_save_failure_keeps_manual_edits_and_prevents_navigation(self):
        self.create_with_clicks()
        self.assertTrue(self.window.save_current())
        path = self.window.store.label_path(self.window.current_path)
        corrupt = b'externally damaged JSON'
        path.write_bytes(corrupt)
        self.window.spins[0][0].setValue(101.5)
        original_path = self.window.current_path
        self.window.navigate(1)
        self.assertEqual(self.window.current_path, original_path)
        self.assertEqual(self.window.annotations[0].points[0][0], 101.5)
        self.assertTrue(self.window.dirty)
        self.assertEqual(path.read_bytes(), corrupt)
        self.assertIn('保存失败', self.errors[0][0])

    def test_batch_prediction_does_not_overwrite_new_manual_edits(self):
        self.window.batch_mode = True
        self.create_with_clicks()
        expected = self.window.annotations[0].to_dict()
        self.window.accept_prediction(str(self.window.current_path), [self.prediction()])
        self.assertEqual([a.to_dict() for a in self.window.annotations], [expected])
        self.assertIsNone(self.window.store.load(self.window.current_path))
        self.assertTrue(self.window.dirty)
        self.assertEqual(self.errors, [])

    def test_batch_prediction_does_not_overwrite_saved_reviewed_empty_image(self):
        self.window.batch_mode = True
        self.window.review.setChecked(True)
        self.assertTrue(self.window.save_current())
        path = self.window.store.label_path(self.window.current_path)
        before = path.read_bytes()
        self.window.accept_prediction(str(self.window.current_path), [self.prediction()])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.window.annotations, [])
        self.assertTrue(self.window.reviewed)
        self.assertEqual(self.errors, [])

    def test_current_prediction_appends_and_preserves_manual_annotation(self):
        self.window.batch_mode = False
        manual = self.create_with_clicks().to_dict()
        prediction = self.prediction()
        self.window.accept_prediction(str(self.window.current_path), [prediction])
        self.assertEqual(self.window.annotations[0].to_dict(), manual)
        self.assertEqual(self.window.annotations[1].to_dict(), prediction.to_dict())
        self.assertTrue(self.window.save_current())
        self.assertEqual(len(self.window.store.load(self.window.current_path)['annotations']), 2)
        self.window.undo()
        self.assertEqual([a.to_dict() for a in self.window.annotations], [manual])
        self.assertEqual(self.errors, [])

    def test_prediction_after_navigation_preserves_saved_manual_target(self):
        self.window.batch_mode = False
        source_path = self.window.current_path
        manual = self.create_with_clicks().to_dict()
        self.window.navigate(1)
        prediction = self.prediction()
        self.window.accept_prediction(str(source_path), [prediction])
        self.assertEqual(self.window.current_path.name, '02.png')
        self.assertEqual(self.window.annotations, [])
        self.window.navigate(-1)
        self.assertEqual([a.to_dict() for a in self.window.annotations],
                         [manual, prediction.to_dict()])
        self.assertEqual(self.errors, [])

    def test_prediction_does_not_erase_in_progress_coordinate_input(self):
        self.window.batch_mode = False
        self.create_with_clicks()
        self.assertTrue(self.window.save_current())
        spin = self.window.spins[0][0]
        spin.setFocus()
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin.lineEdit(), '123.5')
        self.assertEqual(spin.lineEdit().text(), '123.5')
        self.window.accept_prediction(str(self.window.current_path), [self.prediction()])
        # The worker result may append a target, but must preserve user input.
        QTest.keyClick(spin.lineEdit(), Qt.Key.Key_Return)
        self.assertEqual(self.window.annotations[0].points[0][0], 123.5)
        self.assertEqual(self.errors, [])

    def test_prediction_during_drag_waits_and_preserves_undo_boundary(self):
        self.window.batch_mode = False
        original = self.create_with_clicks().to_dict()
        canvas = self.window.canvas
        target = [125.0, 108.0]
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton,
                         pos=canvas.to_view(original['points'][0]).toPoint())
        self.window.accept_prediction(str(self.window.current_path), [self.prediction()])
        self.assertEqual(len(self.window.annotations), 1, 'result must wait for drag to end')
        QTest.mouseMove(canvas, canvas.to_view(target).toPoint(), delay=10)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton,
                           pos=canvas.to_view(target).toPoint())
        changed = self.window.annotations[0].to_dict()
        np.testing.assert_allclose(changed['points'][0], target, atol=1 / canvas.scale)
        QTest.qWait(150)
        self.assertEqual(len(self.window.annotations), 2)
        self.assertEqual(self.window.annotations[0].to_dict(), changed)
        self.window.undo()
        self.assertEqual([a.to_dict() for a in self.window.annotations], [changed])
        self.window.undo()
        self.assertEqual([a.to_dict() for a in self.window.annotations], [original])

    def test_prediction_during_creation_preserves_draft_and_default_attributes(self):
        self.window.batch_mode = False
        self.window.color.setCurrentIndex(3)
        self.window.category.setCurrentIndex(7)
        canvas = self.window.canvas
        canvas.start_create()
        for point in POINTS[:2]:
            QTest.mouseClick(canvas, Qt.MouseButton.LeftButton,
                             pos=canvas.to_view(point).toPoint())
        draft = copy.deepcopy(canvas.draft)
        self.window.accept_prediction(str(self.window.current_path), [self.prediction()])
        self.assertTrue(canvas.creating)
        self.assertEqual(canvas.draft, draft)
        for point in POINTS[2:]:
            QTest.mouseClick(canvas, Qt.MouseButton.LeftButton,
                             pos=canvas.to_view(point).toPoint())
        QTest.qWait(150)
        manual = [ann for ann in self.window.annotations if ann.source == 'manual']
        self.assertEqual(len(manual), 1)
        self.assertEqual((manual[0].color_id, manual[0].class_id), (3, 7))
        self.assertEqual(len(self.window.annotations), 2)

    def test_deferred_prediction_cannot_cross_project_switch(self):
        self.window.batch_mode = False
        ann = self.create_with_clicks()
        canvas = self.window.canvas
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton,
                         pos=canvas.to_view(ann.points[0]).toPoint())
        self.window.accept_prediction(str(self.window.current_path), [self.prediction()])
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton,
                           pos=canvas.to_view(ann.points[0]).toPoint())
        new_images = self.base / 'second_project'
        new_images.mkdir()
        self.assertTrue(cv2.imwrite(str(new_images / 'new.png'), np.zeros((320, 400, 3), np.uint8)))
        self.window.open_folder(new_images, self.base / 'second_labels')
        QTest.qWait(150)
        self.assertEqual(self.window.current_path, new_images / 'new.png')
        self.assertEqual(self.window.annotations, [])
        self.assertIsNone(self.window.store.load(self.window.current_path))
        self.assertEqual(self.errors, [])

    def test_opening_project_with_bad_first_label_cannot_leave_stale_canvas(self):
        self.create_with_clicks()
        new_images = self.base / 'second_project'
        new_labels = self.base / 'second_labels'
        new_images.mkdir()
        new_labels.mkdir()
        self.assertTrue(cv2.imwrite(str(new_images / 'bad.png'), np.zeros((320, 400, 3), np.uint8)))
        bad = new_labels / 'bad.png.json'
        bad.write_bytes(b'bad JSON')
        self.window.open_folder(new_images, new_labels)
        # Keeping the complete previous project or clearing the canvas is safe;
        # an active old canvas without a current image path is not.
        if self.window.current_path is None:
            self.assertIsNone(self.window.canvas.image)
            self.assertEqual(self.window.canvas.annotations, [])
        else:
            self.assertEqual(self.window.current_path, self.images / '01.png')
        self.assertEqual(bad.read_bytes(), b'bad JSON')

    def test_closing_window_flushes_pending_autosave(self):
        expected = self.create_with_clicks().to_dict()
        source = self.window.current_path
        self.assertTrue(self.window.dirty)
        self.assertTrue(self.window.close())
        self.assertFalse(self.window.dirty)
        self.assertEqual(self.window.store.load(source)['annotations'], [expected])

    def test_config_relative_model_path_and_cli_model_override(self):
        config_path = self.base / 'configs' / 'model.json'
        config = default_config()
        config['model_path'] = '../weights/armor.onnx'
        save_config(config, config_path)
        configured_args = argparse.Namespace(config=str(config_path), model=None,
                                             images=None, labels=None)
        configured = self.make_window(configured_args)
        expected = str((config_path.parent / '../weights/armor.onnx').resolve())
        self.assertEqual(configured.model_path, expected)
        self.assertEqual(configured.model_label.text(), expected)
        override_args = copy.copy(configured_args)
        override_args.model = str(self.base / 'override.onnx')
        overridden = self.make_window(override_args)
        self.assertEqual(overridden.model_path, str((self.base / 'override.onnx').resolve()))
        self.assertEqual(overridden.model_label.text(), overridden.model_path)
        self.assertEqual(load_config(config_path)['model_path'], '../weights/armor.onnx')
        self.assertEqual(self.errors, [])

    def test_inference_settings_autosave_and_reopen_restore(self):
        for preprocess, coordinates, confidence, iou in (
                ('letterbox', 'input', .6, .2),
                ('stretch', 'original', .35, .55)):
            self.window.preprocessing.setCurrentIndex(
                self.window.preprocessing.findData(preprocess))
            self.window.output_space.setCurrentIndex(
                self.window.output_space.findData(coordinates))
            self.window.confidence.setValue(confidence)
            self.window.iou.setValue(iou)
            self.assertTrue(self.window.settings_timer.isActive())
            QTest.qWait(self.window.settings_timer.interval() + 150)
            saved = load_config(self.args.config)
            self.assertEqual(saved['preprocess'], preprocess)
            self.assertEqual(saved['output_coordinates'], coordinates)
            self.assertAlmostEqual(saved['confidence'], confidence)
            self.assertAlmostEqual(saved['iou'], iou)
            reopened = self.make_window()
            self.assertEqual(reopened.preprocessing.currentData(), preprocess)
            self.assertEqual(reopened.output_space.currentData(), coordinates)
            self.assertAlmostEqual(reopened.confidence.value(), confidence)
            self.assertAlmostEqual(reopened.iou.value(), iou)
        self.assertEqual(self.errors, [])

    def test_custom_export_contains_only_reviewed_images_with_manifest_mapping(self):
        self.create_with_clicks()
        self.window.color.setCurrentIndex(2)
        self.window.category.setCurrentIndex(6)
        self.window.review.setChecked(True)
        self.assertTrue(self.window.save_current())
        self.window.navigate(1)
        self.create_with_clicks()
        self.assertTrue(self.window.save_current())
        self.assertFalse(self.window.reviewed)
        export_parent = self.base / 'exports'
        export_parent.mkdir()
        with (patch.object(labeler.QDialog, 'exec',
                           return_value=labeler.QDialog.DialogCode.Accepted),
              patch.object(labeler.QFileDialog, 'getExistingDirectory',
                           return_value=str(export_parent)),
              patch.object(labeler.QMessageBox, 'information')):
            self.window.export_labels()
        published = list(export_parent.glob('armor_export_*'))
        self.assertEqual(len(published), 1)
        export_root = published[0]
        labels = list((export_root / 'labels').rglob('*.txt'))
        self.assertEqual([p.relative_to(export_root).as_posix() for p in labels],
                         ['labels/01.png.txt'])
        fields = labels[0].read_text(encoding='utf-8').split()
        self.assertEqual(len(fields), 10)
        self.assertEqual(fields[:2], ['2', '6'])
        self.assertTrue(all(np.isfinite(float(value)) for value in fields[2:]))
        manifest = json.loads((export_root / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['format'], 'custom-head-pixels')
        self.assertEqual(manifest['image_root'], str(self.images))
        self.assertEqual(manifest['images'], [
            {'image': '01.png', 'label': 'labels/01.png.txt', 'reviewed': True}])
        self.assertEqual(self.errors, [])

    def test_inference_worker_uses_fake_detector_and_finishes(self):
        prediction = self.prediction()

        class FakeDetector:
            def __init__(self, model_path, config):
                self.model_path = model_path

            def predict(self, image):
                if image.shape != (480, 640, 3):
                    raise ValueError('unexpected test image')
                return [copy.deepcopy(prediction)]

        self.window.model_path = 'fake.onnx'
        with patch.object(labeler, 'OnnxDetector', FakeDetector):
            self.window.infer_current()
            for _ in range(100):
                QTest.qWait(10)
                if not self.window.worker.isRunning() and self.window.infer_button.isEnabled():
                    break
            self.assertFalse(self.window.worker.isRunning())
            self.assertTrue(self.window.infer_button.isEnabled())
        self.assertEqual(len(self.window.annotations), 1)
        self.assertEqual(self.window.annotations[0].source, 'model')
        self.assertEqual(self.errors, [])

    def test_stop_batch_finishes_current_image_without_processing_next(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []
        prediction = self.prediction()

        class FakeDetector:
            def __init__(self, model_path, config):
                pass

            def predict(self, image):
                calls.append(image.shape)
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test did not release detector')
                return [copy.deepcopy(prediction)]

        self.window.model_path = 'fake.onnx'
        with patch.object(labeler, 'OnnxDetector', FakeDetector):
            try:
                self.window.infer_batch()
                for _ in range(100):
                    QTest.qWait(10)
                    if entered.is_set():
                        break
                self.assertTrue(entered.is_set())
                self.assertFalse(self.window.preprocessing.isEnabled())
                self.assertFalse(self.window.output_space.isEnabled())
                self.window.stop_inference()
            finally:
                release.set()
            for _ in range(100):
                QTest.qWait(10)
                if not self.window.worker.isRunning() and self.window.infer_button.isEnabled():
                    break
            self.assertFalse(self.window.worker.isRunning())
            self.assertTrue(self.window.infer_button.isEnabled())
            self.assertTrue(self.window.preprocessing.isEnabled())
            self.assertTrue(self.window.output_space.isEnabled())
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(self.window.store.load(self.images / '01.png')['annotations']), 1)
        self.assertIsNone(self.window.store.load(self.images / '02.png'))
        self.assertEqual(self.errors, [])


if __name__ == '__main__':
    unittest.main()
