import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from armor_pose.data.prepare import (CLASS_NAMES_EN, POINT_NAMES, choose_val_groups,
                                     convert_custom_row, load_export, prepare_dataset)


class PrepareFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'source'
        self.export = self.root / 'export'
        self.output = self.root / 'dataset'
        self.images.mkdir()
        (self.export / 'labels').mkdir(parents=True)
        self.metadata = {'format': 'custom-head-pixels', 'image_root': str(self.images),
                         'point_names': POINT_NAMES, 'images': []}
        for group in ['video_a', 'video_b', 'video_c']:
            for number in range(2):
                relative = Path('camera') / f'{group}_f{number:04d}.png'
                image = self.images / relative
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (101, 53), (10, 20, 30)).save(image)
                label = Path('labels') / relative.with_name(relative.name + '.txt')
                (self.export / label).parent.mkdir(parents=True, exist_ok=True)
                (self.export / label).write_text('1 3 -10 -4 10 40 90 40 90 4\n' if number == 0 else '')
                self.metadata['images'].append({'image': relative.as_posix(),
                                                'label': label.as_posix(), 'reviewed': False})
        self.save_manifest()

    def save_manifest(self):
        (self.export / 'manifest.json').write_text(json.dumps(self.metadata))


class ConversionTests(unittest.TestCase):
    def test_preserves_ids_outside_coordinates_and_precision(self):
        line = '2 8 -1.123456789012345 3 4 10 20 10 20 3'
        converted, stats = convert_custom_row(line, 15, 8)
        self.assertEqual(converted, line)
        self.assertEqual(stats['outside_points_preserved'], 4)
        self.assertEqual(stats['source_objects'], 1)

    def test_invalid_labels_are_rejected(self):
        for line in ['1 3 0 0 1 0 1 1', '4 0 0 0 1 0 1 1 0 1',
                     '1 9 0 0 1 0 1 1 0 1', '1.5 0 0 0 1 0 1 1 0 1',
                     '1 3 nan 0 1 0 1 1 0 1', '1 3 0 inf 1 0 1 1 0 1',
                     '1 3 1 0 1 1 1 2 1 3']:
            with self.subTest(line=line), self.assertRaises(ValueError):
                convert_custom_row(line, 100, 100)


class PrepareTests(PrepareFixture):
    def test_all_images_empty_labels_and_originals_are_preserved(self):
        before = {p: p.read_bytes() for root in [self.images, self.export] for p in root.rglob('*') if p.is_file()}
        report = prepare_dataset(self.export, self.output, val_groups=['camera/video_b'])
        self.assertEqual(report['totals']['images'], 6)
        self.assertEqual(report['totals']['objects'], 3)
        self.assertEqual(report['totals']['empty_images'], 3)
        self.assertEqual(report['totals']['unreviewed_images'], 6)
        self.assertEqual(report['totals']['outside_points_preserved'], 3)
        self.assertEqual(report['splits']['train']['images'], 4)
        self.assertEqual(report['splits']['val']['images'], 2)
        label = self.output / 'labels/val/camera/video_b_f0000.txt'
        self.assertEqual(label.read_text(), '1 3 -10 -4 10 40 90 40 90 4\n')
        self.assertEqual((self.output / 'labels/val/camera/video_b_f0001.txt').read_bytes(), b'')
        for source, data in before.items():
            self.assertEqual(source.read_bytes(), data)
        config = yaml.safe_load((self.output / 'data.yaml').read_text())
        self.assertEqual(config['names'], dict(enumerate(CLASS_NAMES_EN)))
        self.assertEqual(config['nc'], 9)
        self.assertEqual(config['ncolor'], 4)
        self.assertEqual(config['kpt_shape'], [4, 2])
        self.assertEqual(config['flip_idx'], [3, 2, 1, 0])
        self.assertEqual(config['input_size'], [480, 640])
        mapping = json.loads((self.output / 'class_mapping.json').read_text())
        self.assertEqual(mapping['armor_classes'][0]['name_zh'], '哨兵')
        self.assertFalse(mapping['armor_classes'][5]['teacher_recognizes'])
        self.assertFalse(mapping['colors'][3]['teacher_recognizes'])

    def test_video_groups_do_not_overlap_and_split_is_deterministic(self):
        first = prepare_dataset(self.export, self.output, seed=7)
        second = prepare_dataset(self.export, self.root / 'other_dataset', seed=7)
        self.assertEqual(first['val_groups'], second['val_groups'])
        self.assertTrue(first['val_groups'])
        self.assertFalse(set(first['val_groups']) & set(first['train_groups']))
        manifest = json.loads((self.output / 'split_manifest.json').read_text())
        for video in first['groups']:
            assignments = {e['split'] for e in manifest['images'] if e['video_group'] == video}
            self.assertEqual(len(assignments), 1)

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_text('original')
        with self.assertRaises(FileExistsError):
            prepare_dataset(self.export, self.output)
        self.assertEqual(marker.read_text(), 'original')

    def test_reject_image_and_label_path_traversal(self):
        for key in ['image', 'label']:
            original = self.metadata['images'][0][key]
            for unsafe in ['../escape.png', '/etc/passwd']:
                with self.subTest(key=key, unsafe=unsafe):
                    self.metadata['images'][0][key] = unsafe
                    self.save_manifest()
                    with self.assertRaises(ValueError):
                        prepare_dataset(self.export, self.output)
                    self.assertFalse(self.output.exists())
            self.metadata['images'][0][key] = original
        self.save_manifest()

    def test_same_stem_different_image_suffix_is_rejected(self):
        original = self.metadata['images'][0]
        relative = Path(original['image']).with_suffix('.jpg')
        Image.new('RGB', (101, 53)).save(self.images / relative)
        label = Path('labels') / relative.with_name(relative.name + '.txt')
        (self.export / label).write_text('')
        self.metadata['images'].append({'image': relative.as_posix(), 'label': label.as_posix()})
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'collid'):
            prepare_dataset(self.export, self.output)
        self.assertFalse(self.output.exists())

    def test_symlink_escape_is_rejected(self):
        image = self.images / self.metadata['images'][0]['image']
        image.unlink()
        other = self.root / 'outside.png'
        Image.new('RGB', (10, 10)).save(other)
        image.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            load_export(self.export)

    def test_point_order_and_standard_pose_format_are_rejected(self):
        self.metadata['point_names'] = ['LT', 'RT', 'RB', 'LB']
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'point_names'):
            load_export(self.export)
        self.metadata['point_names'] = POINT_NAMES
        self.metadata['format'] = 'yolo-pose-36'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'format'):
            load_export(self.export)

    def test_invalid_source_leaves_no_partial_output(self):
        label = self.export / self.metadata['images'][-1]['label']
        label.write_text('0 1 2 2 2 2 2 2 2 2')
        with self.assertRaisesRegex(ValueError, 'zero-area'):
            prepare_dataset(self.export, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('.dataset-prepare-*')), [])

    def test_explicit_groups_cannot_cross_or_empty_a_split(self):
        _, entries = load_export(self.export)
        stats = {e.video_group: {} for e in entries}
        for selected in [[], list(stats), ['unknown_video']]:
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                choose_val_groups(stats, explicit=selected)


if __name__ == '__main__':
    unittest.main()
