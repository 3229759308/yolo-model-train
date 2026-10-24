import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from armor_pose.data.subsample import largest_remainder, subsample_dataset


class EmptySubsetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.output = self.root / 'empty10'
        self.source.mkdir()
        self.config = {'path': str(self.source), 'train': 'images/train', 'val': 'images/val',
                       'label_format': 'custom-head-pixels', 'point_names': ['LT', 'LB', 'RB', 'RT'],
                       'nc': 9, 'ncolor': 4, 'nkpt': 4, 'input_size': [480, 640]}
        (self.source / 'data.yaml').write_text(yaml.safe_dump(self.config))
        (self.source / 'class_mapping.json').write_text('{"meaning": "preserve this mapping"}\n')
        (self.source / 'preparation_report.json').write_text('{"old_statistics": true}')
        entries = []
        for split, group, positives, empties in [('train', 'video_a', 2, 13), ('train', 'video_b', 1, 4),
                                                  ('val', 'video_c', 1, 6), ('val', 'video_d', 1, 4)]:
            for frame in range(positives + empties):
                image = Path('images') / split / 'camera' / f'{group}_f{frame:04d}.png'
                label = Path('labels') / split / 'camera' / f'{group}_f{frame:04d}.txt'
                for path in [image, label]:
                    (self.source / path).parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (10, 8), (frame, 30, 60)).save(self.source / image)
                objects = 2 if frame == 0 and group == 'video_a' else int(frame < positives)
                text = '1 3 -1 0 0 6 8 6 8 0\n' * objects if objects else ' \n\t\n'
                (self.source / label).write_text(text)
                entries.append({'image': image.as_posix(), 'label': label.as_posix(), 'split': split,
                                'video_group': f'camera/{group}', 'objects': objects, 'width': 10, 'height': 8,
                                'source_image': f'/original/{group}_f{frame:04d}.png',
                                'source_label': f'/original_labels/{group}_f{frame:04d}.txt', 'reviewed': False})
        self.manifest = {'output_format': 'custom-head-pixels', 'point_names': self.config['point_names'],
                         'source_export': '/original_export', 'images': entries}
        self.write_manifest()

    def write_manifest(self):
        (self.source / 'split_manifest.json').write_text(json.dumps(self.manifest))

    def test_largest_remainder_exact_real_dataset_allocation(self):
        self.assertEqual(largest_remainder({'train': 7165, 'val': 1824}, 899), {'train': 717, 'val': 182})
        self.assertEqual(largest_remainder({'a': 2, 'b': 2, 'c': 2}, 2), {'a': 1, 'b': 1, 'c': 0})
        self.assertEqual(largest_remainder({'a': 0, 'b': 0}, 0), {'a': 0, 'b': 0})

    def test_counts_all_positives_and_source_bytes_are_preserved(self):
        originals = {path: path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
        report = subsample_dataset(self.source, self.output)
        self.assertEqual(report['before']['totals']['images'], 32)
        self.assertEqual(report['before']['totals']['empty_images'], 27)
        self.assertEqual(report['after']['totals']['images'], 8)
        self.assertEqual(report['after']['totals']['positive_images'], 5)
        self.assertEqual(report['after']['totals']['empty_images'], 3)
        self.assertEqual(report['after']['totals']['objects'], 6)
        self.assertEqual(report['split_empty_allocation'], {'train': 2, 'val': 1})
        for path, data in originals.items():
            self.assertEqual(path.read_bytes(), data)
        new = json.loads((self.output / 'split_manifest.json').read_text())
        source_positive = {entry['image'] for entry in self.manifest['images'] if entry['objects']}
        new_positive = {entry['image'] for entry in new['images'] if entry['objects']}
        self.assertEqual(new_positive, source_positive)
        for entry in new['images']:
            self.assertEqual((self.output / entry['image']).read_bytes(), originals[self.source / entry['image']])
            self.assertEqual((self.output / entry['label']).read_bytes(), originals[self.source / entry['label']])
            self.assertEqual(entry['parent_image'], str(self.source / entry['image']))
            self.assertEqual(entry['parent_label'], str(self.source / entry['label']))
            self.assertTrue(entry['source_image'].startswith('/original/'))
        self.assertFalse((self.output / 'preparation_report.json').exists())
        self.assertEqual((self.output / 'class_mapping.json').read_bytes(), originals[self.source / 'class_mapping.json'])
        self.assertEqual(yaml.safe_load((self.output / 'data.yaml').read_text())['path'], str(self.output))

    def test_fixed_seed_is_independent_of_manifest_order_and_preserves_split_groups(self):
        first = subsample_dataset(self.source, self.output, seed=42)
        self.manifest['images'].reverse()
        self.write_manifest()
        second_output = self.root / 'second'
        second = subsample_dataset(self.source, second_output, seed=42)
        first_manifest = json.loads((self.output / 'split_manifest.json').read_text())
        second_manifest = json.loads((second_output / 'split_manifest.json').read_text())
        self.assertEqual({e['image'] for e in first_manifest['images']}, {e['image'] for e in second_manifest['images']})
        self.assertEqual(first['group_empty_allocation'], second['group_empty_allocation'])
        assigned = {}
        for entry in first_manifest['images']:
            assigned.setdefault(entry['video_group'], set()).add(entry['split'])
        self.assertTrue(all(len(splits) == 1 for splits in assigned.values()))
        original_split = {entry['image']: entry['split'] for entry in self.manifest['images']}
        self.assertTrue(all(entry['split'] == original_split[entry['image']] for entry in first_manifest['images']))

    def test_zero_and_full_empty_retention(self):
        zero = subsample_dataset(self.source, self.output, keep_empty=0)
        self.assertEqual(zero['after']['totals']['images'], 5)
        self.assertEqual(zero['after']['totals']['empty_images'], 0)
        full = subsample_dataset(self.source, self.root / 'full', keep_empty=1)
        self.assertEqual(full['after']['totals']['images'], 32)

    def test_existing_output_and_nested_paths_are_refused(self):
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_text('unchanged')
        with self.assertRaises(FileExistsError):
            subsample_dataset(self.source, self.output)
        self.assertEqual(marker.read_text(), 'unchanged')
        for output in [self.source, self.source / 'nested', self.source.parent]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                subsample_dataset(self.source, output)
        self.assertFalse((self.source / 'nested').exists())

    def test_unsafe_manifest_paths_are_refused(self):
        for key in ['image', 'label']:
            original = self.manifest['images'][0][key]
            for value in ['../escape.png', '/etc/passwd', 'images\\train\\file.png']:
                self.manifest['images'][0][key] = value
                self.write_manifest()
                with self.subTest(key=key, path=value), self.assertRaises(ValueError):
                    subsample_dataset(self.source, self.output)
                self.assertFalse(self.output.exists())
            self.manifest['images'][0][key] = original

    def test_duplicate_image_and_manifest_label_count_mismatch_are_refused(self):
        self.manifest['images'].append(self.manifest['images'][0].copy())
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            subsample_dataset(self.source, self.output)
        self.manifest['images'].pop()
        self.manifest['images'][0]['objects'] = 0
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'objects count'):
            subsample_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_cross_split_video_group_and_label_misassociation_are_refused(self):
        index = next(i for i, entry in enumerate(self.manifest['images']) if entry['split'] == 'val')
        self.manifest['images'][index]['video_group'] = 'camera/video_a'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'video_group'):
            subsample_dataset(self.source, self.output)
        self.manifest['images'][index]['video_group'] = 'camera/video_c'
        self.manifest['images'][0]['label'] = self.manifest['images'][1]['label']
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'label path'):
            subsample_dataset(self.source, self.output)

    def test_symlink_escapes_and_failure_cleanup(self):
        image = self.source / self.manifest['images'][0]['image']
        saved = image.read_bytes()
        image.unlink()
        external = self.root / 'external.png'
        external.write_bytes(saved)
        image.symlink_to(external)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            subsample_dataset(self.source, self.output)
        image.unlink()
        image.write_bytes(saved)
        patch_target = f'{subsample_dataset.__module__}.shutil.copy2'
        with patch(patch_target, side_effect=OSError('simulated disk failure')):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                subsample_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('.empty10-subsample-*')), [])

    def test_invalid_fraction_is_refused(self):
        for fraction in [-.1, 1.1, float('nan'), float('inf')]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                subsample_dataset(self.source, self.output, keep_empty=fraction)


if __name__ == '__main__':
    unittest.main()
