#!/usr/bin/env python3
"""Keep every positive image and a stratified fraction of empty armor images.

The existing video-disjoint train/val assignment is preserved. Original image
and label bytes remain unchanged. A new dataset is published only after all
manifest entries have been validated and the selected files have been copied.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from typing import Sequence

from PIL import Image
import yaml

from ..common.definitions import POINT_NAMES as _POINT_NAMES
from ..settings import DATASETS_DIR
from .prepare import convert_custom_row, video_group


POINT_NAMES = list(_POINT_NAMES)
DEFAULT_SOURCE = DATASETS_DIR / 'armor_pose_custom'
DEFAULT_OUTPUT = DATASETS_DIR / 'armor_pose_custom_empty10'


@dataclass
class Sample:
    entry: dict
    image: Path
    label: Path
    split: str
    group: str
    objects: int
    joint_counts: Counter


def _safe_file(root: Path, relative: object, context: str) -> tuple[Path, Path]:
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError(f'{context}: expected a POSIX relative file path')
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or path == Path('.'):
        raise ValueError(f'{context}: absolute paths and path traversal are forbidden')
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f'{context}: file escapes the source dataset or is not a regular file')
    return resolved, path


def _nonnegative_integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f'{context}: expected a nonnegative integer')
    return value


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate exactly total slots in proportion to counts, with stable ties."""
    capacity = sum(counts.values())
    if any(not isinstance(v, int) or v < 0 for v in counts.values()) or not 0 <= total <= capacity:
        raise ValueError('invalid proportional allocation counts or total')
    if not capacity:
        return {key: 0 for key in sorted(counts)}
    # Integer division keeps allocation exact, including large manifests.
    result = {key: total * counts[key] // capacity for key in sorted(counts)}
    priority = sorted(counts, key=lambda key: (-(total * counts[key] % capacity), key))
    for key in priority[:total - sum(result.values())]:
        result[key] += 1
    return result


def _read_source(source: Path) -> tuple[dict, dict, list[Sample]]:
    config = yaml.safe_load((source / 'data.yaml').read_text(encoding='utf-8'))
    if not isinstance(config, dict) or config.get('label_format') != 'custom-head-pixels':
        raise ValueError('source data.yaml must declare label_format: custom-head-pixels')
    configured_root = config.get('path')
    if not isinstance(configured_root, str) or not configured_root:
        raise ValueError('source data.yaml must declare its dataset path')
    root = Path(configured_root)
    if not root.is_absolute():
        root = source / root
    if root.resolve(strict=True) != source:
        raise ValueError('source data.yaml path does not match --source')
    for split in ('train', 'val'):
        if config.get(split) != f'images/{split}':
            raise ValueError(f'source data.yaml {split} must be images/{split}')
    manifest = json.loads((source / 'split_manifest.json').read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or manifest.get('output_format') != 'custom-head-pixels':
        raise ValueError('split_manifest.json must describe custom-head-pixels')
    if manifest.get('point_names') != POINT_NAMES or config.get('point_names') != POINT_NAMES:
        raise ValueError(f'source point_names must be {POINT_NAMES}')
    entries = manifest.get('images')
    if not isinstance(entries, list) or not entries:
        raise ValueError('split_manifest.json images must be a nonempty list')
    samples, seen_images, seen_labels, seen_stems = [], set(), set(), set()
    groups = {}
    for index, entry in enumerate(entries):
        context = f'manifest image {index}'
        if not isinstance(entry, dict) or entry.get('split') not in ('train', 'val'):
            raise ValueError(f'{context}: invalid split or entry')
        split = entry['split']
        image, relative_image = _safe_file(source, entry.get('image'), context)
        label, relative_label = _safe_file(source, entry.get('label'), context)
        if relative_image.parts[:2] != ('images', split) or len(relative_image.parts) < 3:
            raise ValueError(f'{context}: image path does not match its split')
        image_under_split = Path(*relative_image.parts[2:])
        expected_label = Path('labels') / split / image_under_split.with_suffix('.txt')
        if relative_label != expected_label:
            raise ValueError(f'{context}: label path must match its image stem and split')
        stem = relative_image.with_suffix('').as_posix().casefold()
        if image in seen_images or label in seen_labels or stem in seen_stems:
            raise ValueError(f'{context}: duplicate image/label or colliding image stem')
        seen_images.add(image)
        seen_labels.add(label)
        seen_stems.add(stem)
        group = entry.get('video_group')
        expected_group = video_group(image_under_split)
        if not isinstance(group, str) or group != expected_group:
            raise ValueError(f'{context}: video_group must match image filename: {expected_group}')
        if group in groups and groups[group] != split:
            raise ValueError(f'{context}: video group crosses train/val: {group}')
        groups[group] = split
        width = _nonnegative_integer(entry.get('width'), context)
        height = _nonnegative_integer(entry.get('height'), context)
        if not width or not height:
            raise ValueError(f'{context}: image dimensions must be positive')
        try:
            with Image.open(image) as opened:
                if opened.size != (width, height):
                    raise ValueError('manifest dimensions differ from actual image')
                opened.verify()
        except Exception as exc:
            raise ValueError(f'{context}: invalid source image: {exc}') from exc
        objects, counts = 0, Counter()
        for line_number, line in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            validated, _ = convert_custom_row(line, width, height, f'{label}:{line_number}')
            fields = validated.split()
            objects += 1
            counts[int(float(fields[0])) * 9 + int(float(fields[1]))] += 1
        if _nonnegative_integer(entry.get('objects'), context) != objects:
            raise ValueError(f'{context}: manifest objects count differs from actual label ({objects})')
        samples.append(Sample(entry, image, label, split, group, objects, counts))
    if {sample.split for sample in samples} != {'train', 'val'}:
        raise ValueError('source must contain both train and val images')
    return config, manifest, samples


def _stats(samples: Sequence[Sample]) -> dict:
    counts = Counter()
    for sample in samples:
        counts.update(sample.joint_counts)
    return {'images': len(samples), 'positive_images': sum(s.objects > 0 for s in samples),
            'empty_images': sum(s.objects == 0 for s in samples),
            'objects': sum(s.objects for s in samples),
            'combined_class_counts': {str(i): counts[i] for i in range(36)}}


def _summary(samples: Sequence[Sample]) -> dict:
    return {'totals': _stats(samples),
            'splits': {split: _stats([s for s in samples if s.split == split]) for split in ('train', 'val')},
            'groups': {group: {'split': next(s.split for s in samples if s.group == group),
                               **_stats([s for s in samples if s.group == group])}
                       for group in sorted({s.group for s in samples})}}


def subsample_dataset(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT, *,
                      keep_empty: float = .1, seed: int = 42) -> dict:
    """Copy positives plus a deterministic, split/video-stratified empty subset."""
    if not isinstance(keep_empty, (int, float)) or not math.isfinite(keep_empty) or not 0 <= keep_empty <= 1:
        raise ValueError('keep_empty must be a finite fraction in [0,1]')
    source = Path(source).resolve(strict=True)
    output = Path(output).absolute()
    target = output.resolve()
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('source and output must be distinct non-nested directories')
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'output already exists; choose a new --output: {output}')
    config, manifest, samples = _read_source(source)
    before = _summary(samples)
    empty_total = before['totals']['empty_images']
    desired_total = round(empty_total * keep_empty)
    split_empty = {split: before['splits'][split]['empty_images'] for split in ('train', 'val')}
    split_allocation = largest_remainder(split_empty, desired_total)
    group_allocation, selected_images = {}, {sample.image for sample in samples if sample.objects}
    rng = random.Random(seed)
    for split in ('train', 'val'):
        groups = sorted({sample.group for sample in samples if sample.split == split})
        candidates = {group: sorted([s for s in samples if s.split == split and s.group == group and not s.objects],
                                    key=lambda sample: sample.entry['image']) for group in groups}
        allocation = largest_remainder({g: len(candidates[g]) for g in groups}, split_allocation[split])
        for group in groups:
            group_allocation[group] = {'split': split, 'available_empty_images': len(candidates[group]),
                                       'selected_empty_images': allocation[group]}
            selected_images.update(sample.image for sample in rng.sample(candidates[group], allocation[group]))
    selected = [sample for sample in samples if sample.image in selected_images]
    after = _summary(selected)
    for group, stats in before['groups'].items():
        after['groups'].setdefault(group, {'split': stats['split'], **_stats([])})
    after['groups'] = dict(sorted(after['groups'].items()))
    if after['totals']['positive_images'] != before['totals']['positive_images'] or after['totals']['objects'] != before['totals']['objects']:
        raise RuntimeError('internal error: positive samples were lost')
    if after['totals']['empty_images'] != desired_total:
        raise RuntimeError('internal error: selected empty count does not match allocation')
    if any(after['splits'][split]['images'] == 0 for split in ('train', 'val')):
        raise ValueError('empty-image reduction would empty a split; keep more empty images')
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_dataset': str(source),
              'source_manifest': str(source / 'split_manifest.json'), 'output_dataset': str(output),
              'keep_empty': keep_empty, 'seed': seed, 'target_empty_images': desired_total,
              'selection_policy': 'Keep every positive image; round(total_empty * keep_empty); largest remainder by existing split, then video group; seeded sampling within groups sorted by path.',
              'source_policy': 'Source images and labels read only; selected image and label bytes copied unchanged; original train/val membership retained.',
              'before': before, 'after': after, 'split_empty_allocation': split_allocation,
              'group_empty_allocation': group_allocation}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-subsample-', dir=output.parent))
    try:
        new_entries = []
        for split in ('train', 'val'):
            (staging / 'images' / split).mkdir(parents=True, exist_ok=True)
            (staging / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for sample in selected:
            for original, relative in ((sample.image, sample.entry['image']), (sample.label, sample.entry['label'])):
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
            new_entries.append({**sample.entry, 'parent_image': str(sample.image), 'parent_label': str(sample.label)})
        new_manifest = {**manifest, 'parent_dataset': str(source),
                        'parent_manifest': str(source / 'split_manifest.json'),
                        'subsampling': {'keep_empty': keep_empty, 'seed': seed,
                                        'report': 'subsampling_report.json'}, 'images': new_entries}
        (staging / 'data.yaml').write_text(yaml.safe_dump({**config, 'path': str(output)},
                                           sort_keys=False, allow_unicode=True), encoding='utf-8')
        for name, value in (('split_manifest.json', new_manifest), ('subsampling_report.json', report)):
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        mapping = source / 'class_mapping.json'
        if mapping.exists():
            if not mapping.resolve().is_relative_to(source) or not mapping.is_file():
                raise ValueError('class_mapping.json escapes source dataset or is not a file')
            shutil.copy2(mapping, staging / 'class_mapping.json')
        if output.exists() or output.is_symlink():
            raise FileExistsError(f'output appeared during subsampling: {output}')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--keep-empty', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    try:
        report = subsample_dataset(args.source, args.output, keep_empty=args.keep_empty, seed=args.seed)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f'Subsampling failed: {exc}\n')
    before, after = report['before']['totals'], report['after']['totals']
    print(f'Dataset ready: {args.output / "data.yaml"}')
    print(f"Images: {before['images']} -> {after['images']}; empty: {before['empty_images']} -> {after['empty_images']}; objects retained: {after['objects']}")
    for split in ('train', 'val'):
        stats = report['after']['splits'][split]
        print(f"{split}: images={stats['images']}, positive={stats['positive_images']}, empty={stats['empty_images']}, objects={stats['objects']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
