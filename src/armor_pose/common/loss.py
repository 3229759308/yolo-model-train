"""Loss for the 4-color, 9-class, four-corner armor head.

There is no objectness, bounding-box, or keypoint-visibility output. Both
classification branches use all-zero targets for background anchors. Boxes
are derived from the four ordered points only for assignment and GIoU loss.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _boxes(points: Tensor) -> Tensor:
    return torch.cat((points.amin(dim=-2), points.amax(dim=-2)), dim=-1)


def _pairwise_iou(first: Tensor, second: Tensor) -> Tensor:
    intersection = (
        torch.minimum(first[:, None, 2:], second[None, :, 2:])
        - torch.maximum(first[:, None, :2], second[None, :, :2])
    ).clamp_min(0).prod(dim=-1)
    first_area = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=-1)
    return intersection / (first_area[:, None] + second_area[None, :] - intersection).clamp_min(1e-7)


def _giou(first: Tensor, second: Tensor) -> Tensor:
    intersection = (
        torch.minimum(first[:, 2:], second[:, 2:])
        - torch.maximum(first[:, :2], second[:, :2])
    ).clamp_min(0).prod(dim=-1)
    first_area = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=-1)
    union = first_area + second_area - intersection
    enclosure = (
        torch.maximum(first[:, 2:], second[:, 2:])
        - torch.minimum(first[:, :2], second[:, :2])
    ).clamp_min(0).prod(dim=-1)
    return intersection / union.clamp_min(1e-7) - (enclosure - union) / enclosure.clamp_min(1e-7)


class ArmorPoseLoss(nn.Module):
    """Train independent color/class logits and four input-pixel coordinates.

    ``predictions`` contains ``color_logits`` (B,N,4), ``class_logits``
    (B,N,9), ``keypoints`` (B,N,4,2), ``anchor_points`` (N,2 or B,N,2),
    and ``strides`` (N,1 or B,N,1). Each target dictionary contains
    ``points`` (M,4,2), ``color_ids`` (M,), and ``class_ids`` (M,).
    Points outside the image remain supervised, without clipping or masking.

    Assignment uses detached joint branch scores, point-derived IoU, and a
    geometric center prior. A unique nearest anchor is reserved per object,
    so degenerate initial predicted boxes and overlapping objects cannot
    remove all of an object's positive anchors. Other matches use top-k
    task alignment. Positive and background focal terms are normalized
    separately by anchor count, keeping empty images and dense grids stable.
    """

    def __init__(
        self,
        topk: int = 9,
        color_weight: float = 1.0,
        class_weight: float = 1.0,
        keypoint_weight: float = 5.0,
        box_weight: float = 1.0,
        background_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if topk < 1:
            raise ValueError('topk must be positive')
        if min(color_weight, class_weight, keypoint_weight, box_weight, background_weight) < 0:
            raise ValueError('loss weights must be nonnegative')
        self.topk = int(topk)
        self.color_weight = color_weight
        self.class_weight = class_weight
        self.keypoint_weight = keypoint_weight
        self.box_weight = box_weight
        self.background_weight = background_weight

    @torch.no_grad()
    def _assign(
        self,
        colors: Tensor,
        classes: Tensor,
        points: Tensor,
        anchors: Tensor,
        strides: Tensor,
        target: dict[str, Tensor],
    ) -> Tensor:
        count = anchors.shape[0]
        gt_points = target['points']
        gt_count = gt_points.shape[0]
        assigned = torch.full((count,), -1, dtype=torch.long, device=anchors.device)
        if gt_count == 0:
            return assigned
        if gt_count > count:
            raise ValueError('more ground-truth objects than available anchors')

        gt_boxes = _boxes(gt_points)
        centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
        half_size = (gt_boxes[:, 2:] - gt_boxes[:, :2]) * 0.5
        delta = anchors[None] - centers[:, None]
        stride = strides[:, 0]
        distance = (delta / stride[None, :, None]).square().sum(dim=-1)
        # A stride should represent objects roughly four cells across. This
        # keeps a distant coarse-scale anchor from winning by stride alone.
        extent = (half_size.amax(dim=-1) * 2).clamp_min(1)
        scale_cost = torch.log2(extent[:, None] / (4 * stride[None])).square()
        geometry_cost = distance + scale_cost
        eligible = (delta.abs() <= torch.maximum(half_size[:, None], 1.5 * strides[None])).all(dim=-1)

        joint = colors.sigmoid()[:, target['color_ids']].T * classes.sigmoid()[:, target['class_ids']].T
        overlap = _pairwise_iou(gt_boxes, _boxes(points))
        # A small nonzero center prior supplies ranking before the model has
        # any useful localization or confidence predictions.
        prior = 0.05 * torch.exp(-0.5 * geometry_cost)
        alignment = joint.sqrt() * (overlap + prior).square()
        scores = alignment.masked_fill(~eligible, -1)
        indices = scores.topk(min(self.topk, count), dim=1).indices
        selected = torch.zeros_like(eligible).scatter_(1, indices, True) & eligible
        quality, owner = scores.masked_fill(~selected, -1).max(dim=0)
        assigned[quality >= 0] = owner[quality >= 0]

        # Give small/crowded targets first choice, reserving distinct anchors.
        taken = torch.zeros(count, dtype=torch.bool, device=anchors.device)
        area = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp_min(0).prod(dim=-1)
        for gt_index in area.argsort():
            anchor_index = geometry_cost[gt_index].masked_fill(taken, torch.inf).argmin()
            assigned[anchor_index] = gt_index
            taken[anchor_index] = True
        return assigned

    def _classification(self, logits: Tensor, labels: Tensor, foreground: Tensor) -> Tensor:
        probabilities = logits.sigmoid()
        cross_entropy = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = probabilities * labels + (1 - probabilities) * (1 - labels)
        alpha = 0.25 * labels + 0.75 * (1 - labels)
        focal = (alpha * (1 - pt).square() * cross_entropy).sum(dim=-1)
        positive = (focal * foreground).sum() / foreground.sum().clamp_min(1)
        background = ~foreground
        negative = (focal * background).sum() / background.sum().clamp_min(1)
        return positive + self.background_weight * negative

    def forward(
        self,
        predictions: dict[str, Tensor],
        targets: Sequence[dict[str, Tensor]],
        image_size: tuple[int, int],
    ) -> tuple[Tensor, dict[str, Tensor | int]]:
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError('image_size must be positive (height, width)')
        colors = predictions['color_logits'].float()
        classes = predictions['class_logits'].float()
        points = predictions['keypoints'].float()
        if colors.ndim != 3 or colors.shape[-1] != 4:
            raise ValueError('color_logits must have shape (batch, anchors, 4)')
        batch, count = colors.shape[:2]
        if batch < 1 or count < 1 or len(targets) != batch:
            raise ValueError('nonempty predictions and one target dictionary per image are required')
        if classes.shape != (batch, count, 9) or points.shape != (batch, count, 4, 2):
            raise ValueError('class_logits/keypoints must have shapes (B,N,9)/(B,N,4,2)')
        anchors = predictions['anchor_points'].float()
        strides = predictions['strides'].float()
        if anchors.shape == (count, 2):
            anchors = anchors.unsqueeze(0).expand(batch, -1, -1)
        if strides.shape == (count, 1):
            strides = strides.unsqueeze(0).expand(batch, -1, -1)
        if anchors.shape != (batch, count, 2) or strides.shape != (batch, count, 1):
            raise ValueError('anchor_points/strides must have shapes (N,2)/(N,1) or batched equivalents')
        if not torch.all(strides > 0):
            raise ValueError('strides must be positive')

        color_labels = torch.zeros_like(colors)
        class_labels = torch.zeros_like(classes)
        foreground = torch.zeros((batch, count), dtype=torch.bool, device=colors.device)
        matched_points = torch.zeros_like(points)
        num_targets = 0
        for image_index, target in enumerate(targets):
            gt = {
                'points': target['points'].to(device=points.device, dtype=torch.float32),
                'color_ids': target['color_ids'].to(device=colors.device, dtype=torch.long),
                'class_ids': target['class_ids'].to(device=colors.device, dtype=torch.long),
            }
            gt_count = gt['points'].shape[0]
            if (
                gt['points'].shape != (gt_count, 4, 2)
                or gt['color_ids'].shape != (gt_count,)
                or gt['class_ids'].shape != (gt_count,)
            ):
                raise ValueError('target points/IDs must have shapes (M,4,2)/(M,)/(M,)')
            if not torch.isfinite(gt['points']).all():
                raise ValueError('target points must be finite')
            if (
                torch.any((gt['color_ids'] < 0) | (gt['color_ids'] > 3))
                or torch.any((gt['class_ids'] < 0) | (gt['class_ids'] > 8))
            ):
                raise ValueError('color IDs must be in [0,3] and class IDs in [0,8]')
            num_targets += gt_count
            assigned = self._assign(
                colors[image_index], classes[image_index], points[image_index],
                anchors[image_index], strides[image_index], gt,
            )
            fg = assigned >= 0
            foreground[image_index] = fg
            gt_index = assigned[fg]
            color_labels[image_index, fg] = F.one_hot(gt['color_ids'][gt_index], 4).float()
            class_labels[image_index, fg] = F.one_hot(gt['class_ids'][gt_index], 9).float()
            matched_points[image_index, fg] = gt['points'][gt_index]

        color_loss = self._classification(colors, color_labels, foreground)
        class_loss = self._classification(classes, class_labels, foreground)
        positive_count = foreground.sum()
        # The zero-valued graph term also gives the point branch zero gradients
        # on entirely empty batches, including under distributed training.
        zero = points.sum() * 0.0
        if num_targets:
            prediction_points = points[foreground]
            target_points = matched_points[foreground]
            positive_strides = strides[foreground, :, None]
            keypoint_loss = F.smooth_l1_loss(
                prediction_points / positive_strides, target_points / positive_strides, beta=1.0,
            )
            box_loss = (1 - _giou(_boxes(prediction_points), _boxes(target_points))).mean()
        else:
            keypoint_loss = box_loss = zero
        total = (
            self.color_weight * color_loss + self.class_weight * class_loss
            + self.keypoint_weight * keypoint_loss + self.box_weight * box_loss
        )
        stats = {
            'loss_color': color_loss.detach(),
            'loss_class': class_loss.detach(),
            'loss_keypoint': keypoint_loss.detach(),
            'loss_giou': box_loss.detach(),
            'num_positive': positive_count.detach(),
            'num_targets': num_targets,
        }
        return total, stats
