"""Model dispatch and lazy CLI contracts introduced by the package layout."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import Mock

import torch
from torch import nn

from armor_pose.common.runtime import load_model
from armor_pose.models import (
    DEFAULT_MODEL,
    MODEL_BUILDERS,
    build_model,
    register_model,
)


class DummyAdapter(nn.Module):
    """A small state dictionary makes dispatch tests independent of weights."""

    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(1))


class PackageContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.original_builders = MODEL_BUILDERS.copy()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint_path = Path(self.temporary.name) / 'model.pt'

    def tearDown(self):
        MODEL_BUILDERS.clear()
        MODEL_BUILDERS.update(self.original_builders)

    def save_checkpoint(self, **extra):
        source = DummyAdapter()
        with torch.no_grad():
            source.offset.fill_(7.)
        checkpoint = {'model_state_dict': source.state_dict(), **extra}
        torch.save(checkpoint, self.checkpoint_path)
        return checkpoint

    def test_unimplemented_families_and_unknown_names_never_fall_back(self):
        fallback = Mock(side_effect=AssertionError('an unsupported model must not use YOLOv8'))
        MODEL_BUILDERS[DEFAULT_MODEL] = fallback
        for name in ('yolo26n', 'unknown-armor-model'):
            with self.subTest(model=name), self.assertRaisesRegex(ValueError, 'Unsupported armor model'):
                build_model(name, pretrained=False)
        fallback.assert_not_called()

    def test_registered_builder_is_used_with_the_requested_arguments(self):
        adapter = DummyAdapter()
        builder = Mock(return_value=adapter)
        register_model('test-contract-adapter', builder)
        model = build_model('test-contract-adapter', backbone_weights='/local/unused.pt', pretrained=False)
        self.assertIs(model, adapter)
        builder.assert_called_once_with(pretrained=False, backbone_weights='/local/unused.pt')

    def test_legacy_checkpoints_reload_the_default_adapter_without_pretraining(self):
        for metadata in ({}, {'architecture': None}, {'architecture': {'output_channels': 21}}):
            with self.subTest(metadata=metadata):
                expected = self.save_checkpoint(**metadata)
                builder = Mock(side_effect=lambda **_kwargs: DummyAdapter())
                MODEL_BUILDERS[DEFAULT_MODEL] = builder
                model, checkpoint = load_model(self.checkpoint_path, device='cpu')
                builder.assert_called_once_with(pretrained=False)
                self.assertIsInstance(model, DummyAdapter)
                torch.testing.assert_close(model.offset, expected['model_state_dict']['offset'])
                self.assertFalse(model.training)
                self.assertEqual(model.offset.device.type, 'cpu')
                self.assertEqual(checkpoint.keys(), expected.keys())

    def test_checkpoint_protocol_mismatches_are_rejected_before_building(self):
        builder = Mock(side_effect=AssertionError('invalid protocol must not construct an adapter'))
        MODEL_BUILDERS[DEFAULT_MODEL] = builder
        for architecture in (
            {'output_channels': 52},
            {'num_colors': 3},
            {'num_classes': 36},
            {'num_keypoints': 17},
            [],
            'invalid architecture',
        ):
            with self.subTest(architecture=architecture):
                self.save_checkpoint(architecture=architecture)
                with self.assertRaises(ValueError):
                    load_model(self.checkpoint_path)
        builder.assert_not_called()

    def test_runtime_load_uses_the_checkpoint_architecture_registry_entry(self):
        selected = DummyAdapter()
        selected_builder = Mock(return_value=selected)
        default_builder = Mock(side_effect=AssertionError('runtime must dispatch by checkpoint architecture'))
        MODEL_BUILDERS[DEFAULT_MODEL] = default_builder
        register_model('test-runtime-adapter', selected_builder)
        expected = self.save_checkpoint(architecture={
            'name': 'test-runtime-adapter', 'output_channels': 21,
            'num_colors': 4, 'num_classes': 9, 'num_keypoints': 4,
        })
        model, _checkpoint = load_model(self.checkpoint_path)
        self.assertIs(model, selected)
        selected_builder.assert_called_once_with(pretrained=False)
        default_builder.assert_not_called()
        torch.testing.assert_close(model.offset, expected['model_state_dict']['offset'])
        self.assertFalse(model.training)

    def test_unsupported_checkpoint_architecture_does_not_use_the_default(self):
        fallback = Mock(side_effect=AssertionError('runtime must not replace an unknown architecture'))
        MODEL_BUILDERS[DEFAULT_MODEL] = fallback
        self.save_checkpoint(architecture={'name': 'yolo26n', 'output_channels': 21})
        with self.assertRaisesRegex(ValueError, 'Unsupported armor model'):
            load_model(self.checkpoint_path)
        fallback.assert_not_called()

    def test_registered_yolov5_and_yolov8_checkpoint_round_trips_remain_distinct(self):
        models = {
            name: build_model(name, pretrained=False).eval()
            for name in ('yolov5n-armor-pose', 'yolov8n-armor-pose')
        }
        self.assertIsNot(type(models['yolov5n-armor-pose']), type(models['yolov8n-armor-pose']))
        images = torch.rand(1, 3, 64, 96)
        for name, source in models.items():
            for checkpoint_name in (name, name.removesuffix('-armor-pose')):
                with self.subTest(model=name, checkpoint_name=checkpoint_name):
                    architecture = {**source.architecture_config(), 'name': checkpoint_name}
                    torch.save({'architecture': architecture, 'model_state_dict': source.state_dict()},
                               self.checkpoint_path)
                    restored, _ = load_model(self.checkpoint_path)
                    self.assertIs(type(restored), type(source))
                    self.assertEqual(restored.architecture_config()['name'], name)
                    self.assertFalse(restored.training)
                    with torch.inference_mode():
                        torch.testing.assert_close(restored(images), source(images))

    def test_checkpoint_with_the_other_backbone_name_cannot_load_silently(self):
        models = {
            name: build_model(name, pretrained=False).eval()
            for name in ('yolov5n-armor-pose', 'yolov8n-armor-pose')
        }
        for source_name, metadata_name in (
            ('yolov5n-armor-pose', 'yolov8n-armor-pose'),
            ('yolov8n-armor-pose', 'yolov5n-armor-pose'),
        ):
            with self.subTest(weights=source_name, architecture=metadata_name):
                torch.save({
                    'architecture': models[metadata_name].architecture_config(),
                    'model_state_dict': models[source_name].state_dict(),
                }, self.checkpoint_path)
                with self.assertRaises(RuntimeError):
                    load_model(self.checkpoint_path)

    def test_prepare_help_does_not_load_torch_in_a_fresh_process(self):
        repository = Path(__file__).resolve().parents[2]
        # tests/armor_pose/ lives two directories beneath the repository root.
        environment = os.environ.copy()
        environment['PYTHONPATH'] = os.pathsep.join(filter(None, (
            str(repository / 'src'), environment.get('PYTHONPATH', ''),
        )))
        program = textwrap.dedent('''
            import sys
            assert 'torch' not in sys.modules
            from armor_pose.__main__ import main
            try:
                main(['prepare', '--help'])
            except SystemExit as exit_status:
                assert exit_status.code == 0, exit_status.code
            else:
                raise AssertionError('prepare --help must exit after displaying options')
            assert 'torch' not in sys.modules, 'dataset help imported the training dependency'
            assert 'ultralytics' not in sys.modules
        ''')
        result = subprocess.run(
            [sys.executable, '-c', program], cwd=repository, env=environment,
            capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--export', result.stdout)
        self.assertIn('--val-groups', result.stdout)


if __name__ == '__main__':
    unittest.main()
