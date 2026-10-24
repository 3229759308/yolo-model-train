"""Shared offline runtime, decoding and validation for the 21-channel armor head."""
from __future__ import annotations

from .. import settings  # Configure offline paths before importing adapters.
from .definitions import CLASS_NAMES_EN as CLASS_NAMES, COLOR_NAMES_EN as COLOR_NAMES

import numpy as np
import torch


def load_model(weights, device='cpu'):
    from ..models import model_from_architecture
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError('Expected a custom armor checkpoint containing model_state_dict')
    architecture = checkpoint.get('architecture')
    model = model_from_architecture(architecture)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.to(device).eval(), checkpoint


def point_boxes(points):
    return np.concatenate((points.min(axis=-2), points.max(axis=-2)), axis=-1)


def box_iou(box, boxes):
    left = np.maximum(box[:2], boxes[:, :2])
    right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(right - left, 0).prod(axis=1)
    area_a = np.maximum(box[2:] - box[:2], 0).prod()
    area_b = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def decode_predictions(output, confidence=.01, nms_iou=.5, max_det=300):
    """Return points, separate IDs, and product confidence; NMS per ID pair."""
    if isinstance(output, torch.Tensor):
        output = output.detach().float().cpu().numpy()
    if output.ndim != 2 or output.shape[0] != 21:
        raise ValueError(f'Expected (21,N), got {output.shape}')
    colors, classes = output[:4].argmax(axis=0), output[4:13].argmax(axis=0)
    scores = output[:4].max(axis=0) * output[4:13].max(axis=0)
    points = output[13:].T.reshape(-1, 4, 2)
    boxes = point_boxes(points)
    valid = (np.isfinite(output).all(axis=0) & (scores >= confidence)
             & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(-scores[indices], kind='stable')]
    kept = []
    # Limit the NMS work on an uncalibrated head, as in normal detector inference.
    indices = indices[:3000]
    while len(indices) and len(kept) < max_det:
        index = indices[0]
        kept.append(index)
        rest = indices[1:]
        same = (colors[rest] == colors[index]) & (classes[rest] == classes[index])
        indices = rest[~(same & (box_iou(boxes[index], boxes[rest]) > nms_iou))]
    kept = np.asarray(kept, dtype=np.int64)
    return {'points': points[kept], 'color_ids': colors[kept],
            'class_ids': classes[kept], 'scores': scores[kept]}


class DetectionMetrics:
    """Class/color-pair box AP; no human OKS sigmas are used for armor corners."""
    thresholds = np.linspace(.5, .95, 10)

    def __init__(self):
        self.records = []
        self.target_counts = np.zeros(36, dtype=np.int64)
        self.corner_error_sum = 0.
        self.corner_matches = 0
        self.images = 0

    def update(self, prediction, target):
        self.images += 1
        def array(value):
            return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        true_points = array(target['points'])
        true_ids = array(target['color_ids']) * 9 + array(target['class_ids'])
        pred_ids = prediction['color_ids'] * 9 + prediction['class_ids']
        self.target_counts += np.bincount(true_ids.astype(int), minlength=36)
        correct = np.zeros((len(pred_ids), len(self.thresholds)), dtype=bool)
        true_boxes = point_boxes(true_points) if len(true_points) else np.empty((0, 4))
        pred_boxes = point_boxes(prediction['points'])
        for k, threshold in enumerate(self.thresholds):
            used = set()
            for p in np.argsort(-prediction['scores'], kind='stable'):
                candidates = np.flatnonzero(true_ids == pred_ids[p])
                candidates = np.asarray([t for t in candidates if t not in used], dtype=int)
                if not len(candidates):
                    continue
                overlaps = box_iou(pred_boxes[p], true_boxes[candidates])
                best = int(overlaps.argmax())
                if overlaps[best] >= threshold:
                    matched = int(candidates[best])
                    used.add(matched)
                    correct[p, k] = True
                    if k == 0:
                        self.corner_error_sum += float(np.linalg.norm(prediction['points'][p] - true_points[matched], axis=1).sum())
                        self.corner_matches += 4
        for p in range(len(pred_ids)):
            self.records.append((float(prediction['scores'][p]), int(pred_ids[p]), correct[p]))

    def compute(self):
        rows = sorted(self.records, key=lambda row: -row[0])
        per_class, aps = [], []
        for identity, count in enumerate(self.target_counts):
            if not count:
                continue
            correct = np.asarray([row[2] for row in rows if row[1] == identity], dtype=bool).reshape(-1, 10)
            if len(correct):
                tp = correct.cumsum(axis=0)
                fp = (~correct).cumsum(axis=0)
                recall, precision = tp / count, tp / np.maximum(tp + fp, 1)
                # Precision envelope on a 101-point recall grid. Missed recall
                # contributes zero, including models that predict nothing.
                ap = np.array([np.mean([precision[recall[:, k] >= r, k].max(initial=0.)
                                       for r in np.linspace(0, 1, 101)]) for k in range(10)])
            else:
                ap = np.zeros(10)
            aps.append(ap)
            per_class.append({'color_id': identity // 9, 'class_id': identity % 9,
                              'targets': int(count), 'AP50': float(ap[0]), 'mAP50_95': float(ap.mean())})
        mean_ap = np.mean(aps, axis=0) if aps else np.zeros(10)
        return {'images': self.images, 'targets': int(self.target_counts.sum()),
                'predictions': len(rows), 'mAP50': float(mean_ap[0]),
                'mAP50_95': float(mean_ap.mean()),
                'corner_error_input_px': (float(self.corner_error_sum / self.corner_matches) if self.corner_matches else None),
                'corner_matches': self.corner_matches // 4, 'per_color_class': per_class}
