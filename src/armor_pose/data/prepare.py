#!/usr/bin/env python3
"""Prepare an armor_labeler export for the 21-channel custom pose head.

The source images and labels are read only. All exported images, including
unreviewed and empty images, are retained. The output is published atomically
only after every source file has passed validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Sequence

from PIL import Image
import yaml


from ..common.definitions import (
    CLASS_NAMES_EN, CLASS_NAMES_ZH, COLOR_NAMES_EN, COLOR_NAMES_ZH,
    FLIP_IDX, POINT_NAMES as _POINT_NAMES,
)
from ..settings import DATA_DIR, DATASETS_DIR


DEFAULT_EXPORT = DATA_DIR / 'raw_image_label_end/armor_export_20261023_071657_908993'
DEFAULT_OUTPUT = DATASETS_DIR / 'armor_pose_custom'
POINT_NAMES = list(_POINT_NAMES)
COMBINED_NAMES = [f'{color}_{cls}' for color in COLOR_NAMES_EN for cls in CLASS_NAMES_EN]
SUPPORTED_FORMATS = {'custom-head-pixels'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


@dataclass
class ExportEntry:
    image: Path
    label: Path
    relative_image: Path
    video_group: str
    width: int
    height: int
    rows: list[str]
    class_counts: Counter
    stats: Counter
    reviewed: bool


def _integer(text: str, low: int, high: int, context: str) -> int:
    try:
        value = float(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{context}: invalid integer {text!r}') from exc
    if not math.isfinite(value) or not value.is_integer() or not low <= value <= high:
        raise ValueError(f'{context}: expected integer in [{low}, {high}], got {text!r}')
    return int(value)


def _numbers(tokens: Sequence[str], context: str) -> list[float]:
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f'{context}: nonnumeric coordinate') from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f'{context}: non-finite coordinate')
    return values


def _check_dimensions(width: int, height: int) -> None:
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError('image dimensions must be positive integers')


def convert_custom_row(line: str, width: int, height: int, context: str = '') -> tuple[str, dict]:
    """Validate and preserve color/class + four original pixel-coordinate points.

    Outside points are legitimate regression targets and remain untouched.
    Reject zero-area boxes rather than silently deleting an object.
    """
    _check_dimensions(width, height)
    tokens = line.split()
    if len(tokens) != 10:
        raise ValueError(f'{context}: custom label requires 10 columns, got {len(tokens)}')
    _integer(tokens[0], 0, 3, context)
    _integer(tokens[1], 0, 8, context)
    values = _numbers(tokens[2:], context)
    points = list(zip(values[::2], values[1::2]))
    if max(x for x, _ in points) <= min(x for x, _ in points) or max(y for _, y in points) <= min(y for _, y in points):
        raise ValueError(f'{context}: zero-area bounding box')
    outside = sum(not (0 <= x <= width and 0 <= y <= height) for x, y in points)
    return ' '.join(tokens), {'outside_points_preserved': outside,
                             'objects_with_outside_points': int(outside > 0),
                             'source_objects': 1}


def _safe_source(root: Path, relative: object, context: str) -> tuple[Path, Path]:
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError(f'{context}: expected a POSIX relative file path')
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or path == Path('.'):
        raise ValueError(f'{context}: absolute paths and path traversal are forbidden')
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f'{context}: file escapes source root or is not a regular file')
    return resolved, path


def video_group(relative_image: Path) -> str:
    """Preserve video grouping across subfolders; reject filenames without frame IDs."""
    prefix, sep, frame = relative_image.stem.rpartition('_f')
    if not sep or not prefix or not re.fullmatch(r'[0-9]+', frame):
        raise ValueError(f'{relative_image}: expected <video>_f<digits> filename for video-disjoint split')
    return (relative_image.parent / prefix).as_posix()


def load_export(export_dir: Path) -> tuple[dict, list[ExportEntry]]:
    export_dir = Path(export_dir).resolve(strict=True)
    metadata = json.loads((export_dir / 'manifest.json').read_text(encoding='utf-8'))
    if not isinstance(metadata, dict) or metadata.get('format') not in SUPPORTED_FORMATS:
        raise ValueError(f'manifest format must be one of {sorted(SUPPORTED_FORMATS)}')
    if metadata.get('point_names') != POINT_NAMES:
        raise ValueError(f'manifest point_names must be {POINT_NAMES}; no implicit point reordering is performed')
    root_text = metadata.get('image_root')
    if not isinstance(root_text, str) or not root_text:
        raise ValueError('manifest image_root must be a directory path')
    image_root = Path(root_text)
    if not image_root.is_absolute():
        image_root = export_dir / image_root
    image_root = image_root.resolve(strict=True)
    if not image_root.is_dir():
        raise ValueError('manifest image_root is not a directory')
    images = metadata.get('images')
    if not isinstance(images, list) or not images:
        raise ValueError('manifest images must be a nonempty list')
    result = []
    seen_images, seen_labels, seen_stems = set(), set(), set()
    converter = convert_custom_row
    for index, item in enumerate(images):
        context = f'manifest image {index}'
        if not isinstance(item, dict):
            raise ValueError(f'{context}: entry must be an object')
        image, relative_image = _safe_source(image_root, item.get('image'), context)
        label, _ = _safe_source(export_dir, item.get('label'), context)
        if relative_image.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f'{context}: unsupported training image suffix: {relative_image}')
        # Ultralytics maps different image suffixes to the same stem.txt label.
        stem = relative_image.with_suffix('').as_posix().casefold()
        if image in seen_images or label in seen_labels or stem in seen_stems:
            raise ValueError(f'{context}: duplicate image/label or colliding output stem: {relative_image}')
        seen_images.add(image)
        seen_labels.add(label)
        seen_stems.add(stem)
        group = video_group(relative_image)
        try:
            with Image.open(image) as source:
                width, height = source.size
                source.verify()
            _check_dimensions(width, height)
        except Exception as exc:
            raise ValueError(f'{relative_image}: invalid image: {exc}') from exc
        rows, counts, stats = [], Counter(), Counter()
        for number, line in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            row, row_stats = converter(line, width, height, f'{label}:{number}')
            stats.update(row_stats)
            rows.append(row)
            fields = row.split()
            counts[int(float(fields[0])) * 9 + int(float(fields[1]))] += 1
        result.append(ExportEntry(image, label, relative_image, group, width, height,
                                  rows, counts, stats, bool(item.get('reviewed', False))))
    return {**metadata, 'image_root': str(image_root)}, result


def _aggregate(entries: Sequence[ExportEntry]) -> dict:
    counts, stats = Counter(), Counter()
    for entry in entries:
        counts.update(entry.class_counts)
        stats.update(entry.stats)
    armor_counts, color_counts = Counter(), Counter()
    for cls, count in counts.items():
        color_counts[cls // 9] += count
        armor_counts[cls % 9] += count
    return {'images': len(entries), 'positive_images': sum(bool(e.rows) for e in entries),
            'empty_images': sum(not e.rows for e in entries), 'objects': sum(counts.values()),
            'combined_class_counts': {str(i): counts[i] for i in range(36)},
            'armor_class_counts': {str(i): armor_counts[i] for i in range(9)},
            'color_counts': {str(i): color_counts[i] for i in range(4)},
            'source_objects': stats['source_objects'],
            'outside_points_preserved': stats['outside_points_preserved'],
            'objects_with_outside_points': stats['objects_with_outside_points'],
            'reviewed_images': sum(e.reviewed for e in entries),
            'unreviewed_images': sum(not e.reviewed for e in entries)}


def choose_val_groups(group_stats: dict[str, dict], val_ratio: float = .2, seed: int = 42,
                      explicit: Sequence[str] | None = None) -> set[str]:
    """Choose entire videos, balancing image ratio and common-class coverage.

    Exhaustive search is practical for this export's 13 videos. For larger
    exports seeded greedy candidates cap preparation time. No frames are split
    randomly. A class is considered common at >=20 instances in >=2 videos.
    """
    if not 0 < val_ratio < 1 or not math.isfinite(val_ratio):
        raise ValueError('val_ratio must be between 0 and 1')
    groups = sorted(group_stats)
    if len(groups) < 2:
        raise ValueError('at least two video groups are required for train/val isolation')
    if explicit is not None:
        selected = set(explicit)
        unknown = selected.difference(groups)
        if unknown:
            raise ValueError(f'unknown validation video groups: {sorted(unknown)}')
        if not selected or len(selected) == len(groups):
            raise ValueError('validation groups must leave at least one video in each split')
        return selected
    rng = random.Random(seed)
    rng.shuffle(groups)
    total = sum(s['images'] for s in group_stats.values())
    total_positive = sum(s['positive_images'] for s in group_stats.values())
    totals = [sum(s['combined_class_counts'][str(i)] for s in group_stats.values()) for i in range(36)]
    common = [i for i in range(36) if totals[i] >= 20 and
              sum(s['combined_class_counts'][str(i)] > 0 for s in group_stats.values()) >= 2]

    def score(candidate: tuple[str, ...]) -> tuple:
        n = sum(group_stats[g]['images'] for g in candidate)
        positive = sum(group_stats[g]['positive_images'] for g in candidate)
        class_counts = [sum(group_stats[g]['combined_class_counts'][str(i)] for g in candidate) for i in common]
        missing = sum(count == 0 or count == totals[i] for i, count in zip(common, class_counts))
        missing_positive = total_positive > 1 and (positive == 0 or positive == total_positive)
        ratio_error = abs(n / total - val_ratio)
        return (ratio_error + .025 * missing + .25 * missing_positive, ratio_error, tuple(sorted(candidate)))

    if len(groups) <= 16:
        candidates = (combo for size in range(1, len(groups)) for combo in itertools.combinations(groups, size))
    else:
        pool = {(g,) for g in groups}
        for _ in range(256):
            rng.shuffle(groups)
            chosen, size = [], 0
            for group in groups[:-1]:
                if abs((size + group_stats[group]['images']) / total - val_ratio) <= abs(size / total - val_ratio) or not chosen:
                    chosen.append(group)
                    size += group_stats[group]['images']
            pool.add(tuple(sorted(chosen)))
        candidates = iter(sorted(pool))
    return set(min(candidates, key=score))


def dataset_yaml(output_dir: Path) -> dict:
    return {'path': str(output_dir), 'train': 'images/train', 'val': 'images/val',
            'label_format': 'custom-head-pixels', 'nc': 9, 'ncolor': 4, 'nkpt': 4,
            'names': dict(enumerate(CLASS_NAMES_EN)), 'color_names': dict(enumerate(COLOR_NAMES_EN)),
            'kpt_shape': [4, 2], 'point_names': POINT_NAMES, 'flip_idx': FLIP_IDX,
            'input_size': [480, 640]}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare_dataset(export_dir: Path = DEFAULT_EXPORT, output_dir: Path = DEFAULT_OUTPUT, *,
                    val_ratio: float = .2, seed: int = 42,
                    val_groups: Sequence[str] | None = None) -> dict:
    """Validate, convert, copy, and atomically publish a new training dataset."""
    export_dir = Path(export_dir).resolve(strict=True)
    output_dir = Path(output_dir).absolute()
    # Use lexists behavior so a dangling symlink cannot be replaced either.
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f'output already exists; choose a new --output: {output_dir}')
    metadata, entries = load_export(export_dir)
    image_root = Path(metadata['image_root'])
    target = output_dir.resolve()
    if target.is_relative_to(export_dir) or target.is_relative_to(image_root):
        raise ValueError('output must be outside the read-only export and source image directories')
    grouped: dict[str, list[ExportEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.video_group, []).append(entry)
    group_stats = {name: _aggregate(group) for name, group in sorted(grouped.items())}
    selected = choose_val_groups(group_stats, val_ratio, seed, val_groups)
    splits = {name: [e for e in entries if (e.video_group in selected) == (name == 'val')]
              for name in ('train', 'val')}
    total = _aggregate(entries)
    split_stats = {name: _aggregate(group) for name, group in splits.items()}
    missing = {name: [i for i in range(36) if total['combined_class_counts'][str(i)] and
                      not stats['combined_class_counts'][str(i)]] for name, stats in split_stats.items()}
    rare = [{'combined_class_id': i, 'name': COMBINED_NAMES[i],
             'objects': total['combined_class_counts'][str(i)],
             'video_groups': [g for g, s in group_stats.items() if s['combined_class_counts'][str(i)]],
             'missing_in_splits': [name for name in splits if i in missing[name]]}
            for i in range(36) if total['combined_class_counts'][str(i)] and
            (total['combined_class_counts'][str(i)] < 20 or any(i in v for v in missing.values()))]
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_export': str(export_dir),
              'source_format': metadata['format'], 'source_image_root': str(image_root),
              'output': str(output_dir), 'review_policy': 'All manifest images are used regardless of reviewed; no manual review gate.',
              'outside_point_policy': 'All four original pixel coordinates are preserved, including points outside the image; zero-area objects are rejected.',
              'split_policy': 'Whole video groups only; no adjacent-frame leakage between train and val.',
              'seed': seed, 'requested_val_ratio': val_ratio,
              'actual_val_ratio': len(splits['val']) / len(entries),
              'val_groups': sorted(selected), 'train_groups': sorted(set(grouped) - selected),
              'point_names': POINT_NAMES, 'flip_idx': FLIP_IDX,
              'totals': total, 'splits': split_stats, 'groups': group_stats,
              'missing_present_classes_by_split': missing, 'rare_classes': rare,
              'absent_combined_classes': [i for i in range(36) if not total['combined_class_counts'][str(i)]]}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}-prepare-', dir=output_dir.parent))
    try:
        manifest_entries = []
        for name, group in splits.items():
            for entry in group:
                rel_image = Path('images') / name / entry.relative_image
                rel_label = Path('labels') / name / entry.relative_image.with_suffix('.txt')
                image_target, label_target = staging / rel_image, staging / rel_label
                image_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entry.image, image_target)
                label_target.write_text('\n'.join(entry.rows) + ('\n' if entry.rows else ''), encoding='utf-8')
                manifest_entries.append({'image': rel_image.as_posix(), 'label': rel_label.as_posix(),
                                         'split': name, 'video_group': entry.video_group,
                                         'source_image': str(entry.image), 'source_label': str(entry.label),
                                         'width': entry.width, 'height': entry.height,
                                         'objects': len(entry.rows), 'reviewed': entry.reviewed})
        (staging / 'data.yaml').write_text(yaml.safe_dump(dataset_yaml(output_dir), sort_keys=False, allow_unicode=True), encoding='utf-8')
        _write_json(staging / 'preparation_report.json', report)
        _write_json(staging / 'split_manifest.json', {'source_export': str(export_dir),
                     'source_image_root': str(image_root), 'source_format': metadata['format'],
                     'output_format': 'custom-head-pixels', 'label_columns': ['color_id', 'class_id', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                     'point_names': POINT_NAMES, 'flip_idx': FLIP_IDX, 'images': manifest_entries})
        _write_json(staging / 'class_mapping.json', {
            'label_format': 'custom-head-pixels', 'head_channels': {'color_scores': [0, 3], 'class_scores': [4, 12], 'keypoint_xy': [13, 20]},
            'joint_statistics_formula': 'color_id * 9 + class_id (statistics only; training has independent color/class branches)',
            'point_names': POINT_NAMES, 'flip_idx': FLIP_IDX,
            'colors': [{'id': i, 'name': name, 'name_zh': COLOR_NAMES_ZH[i], 'teacher_recognizes': i != 3}
                       for i, name in enumerate(COLOR_NAMES_EN)],
            'armor_classes': [{'id': i, 'name': name, 'name_zh': CLASS_NAMES_ZH[i], 'teacher_recognizes': i != 5}
                              for i, name in enumerate(CLASS_NAMES_EN)],
            'teacher_note': 'v8n-0726 no longer recognizes infantry_5 (class 5) or purple (color 3); IDs are preserved.'})
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f'output appeared during preparation: {output_dir}')
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, default=DEFAULT_EXPORT, help='armor_labeler export directory')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='new dataset directory; existing paths are refused')
    parser.add_argument('--val-ratio', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-groups', nargs='+', help='explicit whole-video validation groups')
    args = parser.parse_args(argv)
    try:
        report = prepare_dataset(args.export, args.output, val_ratio=args.val_ratio,
                                 seed=args.seed, val_groups=args.val_groups)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f'Preparation failed: {exc}\n')
    print(f"Dataset ready: {args.output / 'data.yaml'}")
    print(f"Images: train={report['splits']['train']['images']}, val={report['splits']['val']['images']}; "
          f"objects={report['totals']['objects']}; empty images retained={report['totals']['empty_images']}")
    print(f"Validation videos: {', '.join(report['val_groups'])}")
    print(f"Outside points preserved={report['totals']['outside_points_preserved']}")
    for split, missing in report['missing_present_classes_by_split'].items():
        if missing:
            print(f"Classes absent from {split}: {', '.join(COMBINED_NAMES[i] for i in missing)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
