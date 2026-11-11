"""Experiment layout: process files under runs/NAME, results under results/NAME."""
from contextlib import redirect_stderr, redirect_stdout
import io
import os
import unittest
from unittest.mock import patch

import torch
import yaml

from armor_pose.data.prepare import prepare_dataset
from armor_pose.settings import PROJECT_ROOT, project_path
from armor_pose.training import engine
from .test_prepare_pose_dataset import PrepareFixture


class TrainingLayoutTests(PrepareFixture):
    def setUp(self):
        super().setUp()
        prepare_dataset(self.export, self.output, val_groups=['camera/video_b'])
        self.data = self.output / 'data.yaml'
        self.runs, self.results = self.root / 'runs', self.root / 'results'
        for name, directory in (('RUNS_DIR', self.runs), ('RESULTS_DIR', self.results)):
            patcher = patch.object(engine, name, directory)
            patcher.start()
            self.addCleanup(patcher.stop)

    def parse(self, *argv):
        with redirect_stderr(io.StringIO()):
            return engine.parse_config(['--device', 'cpu', '--workers', '0', '--batch', '2', *argv])

    def train(self, config, resume=None):
        with redirect_stdout(io.StringIO()):
            return engine.run_training(config, resume)

    def test_process_files_and_results_are_split_and_resume_keeps_the_pair(self):
        _, config = self.parse('--data', str(self.data), '--name', 'unit', '--epochs', '1', '--from-scratch')
        self.assertEqual(self.train(config), self.results / 'unit')
        run, result = self.runs / 'unit', self.results / 'unit'
        self.assertEqual(sorted(p.name for p in run.iterdir()), ['args.yaml', 'last.pt', 'metrics.json', 'results.csv'])
        self.assertEqual(sorted(p.name for p in result.iterdir()), ['args.yaml', 'best.pt', 'metrics.json'])
        self.assertNotIn('optimizer_state_dict', torch.load(result / 'best.pt', weights_only=True))
        arguments, config = self.parse('--resume', str(run / 'last.pt'), '--epochs', '2')
        self.assertEqual(config['name'], 'unit')
        self.train(config, arguments.resume)
        self.assertEqual(len((run / 'results.csv').read_text().splitlines()), 3)
        self.assertEqual(yaml.safe_load((result / 'args.yaml').read_text())['epochs'], 2)
        self.assertEqual([p.name for p in self.runs.iterdir()], ['unit'])

    def test_nonempty_experiment_is_refused_before_creating_directories(self):
        (self.results / 'unit').mkdir(parents=True)
        (self.results / 'unit' / 'best.pt').write_bytes(b'')
        _, config = self.parse('--data', str(self.data), '--name', 'unit')
        with self.assertRaises(FileExistsError):
            self.train(config)
        self.assertFalse(self.runs.exists())

    def test_invalid_names_missing_data_and_name_with_resume_are_rejected(self):
        for argv in (['--name', 'nested/run'], ['--name', '..'], ['--name', 'unit', '--resume', 'last.pt']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                self.parse('--data', str(self.data), *argv)
        with self.assertRaises(SystemExit):
            self.parse('--data', str(self.root / 'missing.yaml'))

    def test_command_line_paths_follow_the_working_directory_and_yaml_paths_the_repository(self):
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.root)
        _, config = self.parse('--data', 'dataset/data.yaml')
        self.assertEqual(config['data'], str(self.data.resolve()))
        override = self.root / 'override.yaml'
        override.write_text(yaml.safe_dump({'data': os.path.relpath(self.data, PROJECT_ROOT)}))
        _, config = self.parse('--config', str(override))
        self.assertEqual(project_path(config['data']).resolve(), self.data.resolve())


if __name__ == '__main__':
    unittest.main()
