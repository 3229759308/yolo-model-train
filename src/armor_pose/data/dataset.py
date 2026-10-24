"""Torch data input for independent armor color/class and four-point heads.

Labels are ten columns: color_id class_id x0 y0 ... x3 y3, in original
image pixels. No clipping, visibility mask, or combined-class conversion occurs.
"""
from __future__ import annotations

from pathlib import Path
import random
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from ..common.definitions import POINT_NAMES as _POINT_NAMES
from .prepare import convert_custom_row


POINT_NAMES = list(_POINT_NAMES)

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _input_size(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in value):
        raise ValueError('input_size must be two positive integers (height, width)')
    return int(value[0]), int(value[1])


def read_custom_label(path: Path) -> dict[str, torch.Tensor]:
    """Cache finite, nondegenerate labels without changing original coordinates."""
    points, colors, classes = [], [], []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        # Conversion validates IDs, finiteness, column count and box area;
        # dimensions do not affect the values returned for the custom format.
        validated, _ = convert_custom_row(line, 1, 1, f'{path}:{number}')
        values = [float(v) for v in validated.split()]
        if any(abs(v) > np.finfo(np.float32).max for v in values[2:]):
            raise ValueError(f'{path}:{number}: coordinate overflows float32 training tensors')
        colors.append(int(values[0]))
        classes.append(int(values[1]))
        points.append(np.asarray(values[2:], dtype=np.float32).reshape(4, 2))
    return {'points': torch.tensor(np.asarray(points, dtype=np.float32).reshape(-1, 4, 2)),
            'color_ids': torch.tensor(colors, dtype=torch.long),
            'class_ids': torch.tensor(classes, dtype=torch.long)}


class ArmorPoseDataset(Dataset):
    """Read custom armor labels and letterbox RGB images to a fixed input size.

    Brightness-only training augmentation preserves color/number semantics.
    Labels are cached; image pixels are decoded on demand. In metadata,
    resized_size is the actual resized content before padding; input_size is
    the final network image size. All sizes use (height, width).
    """

    def __init__(self, data_yaml: str | Path, split: str = 'train',
                 input_size: tuple[int, int] = (480, 640), augment: bool = False,
                 brightness: float = .15):
        self.data_yaml = Path(data_yaml).resolve(strict=True)
        config = yaml.safe_load(self.data_yaml.read_text(encoding='utf-8'))
        if not isinstance(config, dict) or config.get('label_format') != 'custom-head-pixels':
            raise ValueError('data_yaml must declare label_format: custom-head-pixels')
        expected = {'nc': 9, 'ncolor': 4, 'nkpt': 4, 'kpt_shape': [4, 2], 'point_names': POINT_NAMES}
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f'data_yaml requires {key}: {value}')
        if split not in ('train', 'val') or not isinstance(config.get(split), str):
            raise ValueError('split must be train or val with a directory in data_yaml')
        root = Path(config.get('path', '.'))
        if not root.is_absolute():
            root = self.data_yaml.parent / root
        self.root = root.resolve(strict=True)
        relative = Path(config[split])
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'images':
            raise ValueError('split directories must be relative paths under images/')
        image_dir = (self.root / relative).resolve(strict=True)
        if not image_dir.is_relative_to(self.root) or not image_dir.is_dir():
            raise ValueError('split image directory escapes dataset or is not a directory')
        self.input_size = _input_size(input_size)
        if not np.isfinite(brightness) or not 0 <= brightness <= 1:
            raise ValueError('brightness must be in [0, 1]')
        self.augment = bool(augment) and split == 'train'
        self.brightness = float(brightness)
        self.split = split
        self.images, self.targets = [], []
        seen = set()
        for path in sorted(image_dir.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(self.root):
                raise ValueError(f'image escapes dataset: {path}')
            rel = path.relative_to(self.root)
            stem = rel.with_suffix('').as_posix().casefold()
            if stem in seen:
                raise ValueError(f'image label stem collision: {path}')
            seen.add(stem)
            label = (self.root / 'labels' / Path(*rel.parts[1:])).with_suffix('.txt')
            if not label.resolve(strict=True).is_relative_to(self.root):
                raise ValueError(f'label escapes dataset: {label}')
            self.images.append(resolved)
            self.targets.append(read_custom_label(label))
        if not self.images:
            raise ValueError(f'no images in {image_dir}')

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        path = self.images[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'cannot decode image: {path}')
        height, width = image.shape[:2]
        input_h, input_w = self.input_size
        gain = min(input_w / width, input_h / height)
        resized_w = min(input_w, max(1, round(width * gain)))
        resized_h = min(input_h, max(1, round(height * gain)))
        scale_x, scale_y = resized_w / width, resized_h / height
        left = (input_w - resized_w) // 2
        top = (input_h - resized_h) // 2
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        # Allocate a new array; neither source files nor cached targets are mutated.
        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_h, left:left + resized_w] = resized
        pixels = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.augment and self.brightness:
            pixels = np.clip(pixels * random.uniform(1 - self.brightness, 1 + self.brightness), 0, 1)
        image_tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
        target = {key: value.clone() for key, value in self.targets[index].items()}
        target['points'][..., 0] = target['points'][..., 0] * scale_x + left
        target['points'][..., 1] = target['points'][..., 1] * scale_y + top
        metadata = {'path': str(path), 'original_size': (height, width),
                    'resized_size': (resized_h, resized_w), 'input_size': (input_h, input_w),
                    'padding': (left, top), 'scale': (scale_x, scale_y)}
        return image_tensor, target, metadata


def armor_collate(batch):
    images, targets, metadata = zip(*batch)
    return torch.stack(images), list(targets), list(metadata)
