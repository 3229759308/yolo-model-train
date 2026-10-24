"""Shared three-scale armor head: independent color/class/four-point outputs."""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from .. import settings  # Configure the offline environment before Ultralytics.
from .definitions import POINT_NAMES
from ultralytics.nn.modules import Conv


def _branch(in_channels: int, hidden_channels: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        Conv(in_channels, hidden_channels, 3),
        Conv(hidden_channels, hidden_channels, 3),
        nn.Conv2d(hidden_channels, outputs, kernel_size=1),
    )


class ArmorPoseHead(nn.Module):
    """Three-scale, independent color/class/four-corner prediction branches.

    Flattening is P3, P4, P5, with each feature map in row-major order. Integer
    grid coordinates decode points as ``(raw * 2 + grid) * stride``. Anchor
    centers used for training assignment are ``(grid + 0.5) * stride``.
    """

    num_colors = 4
    num_classes = 9
    num_keypoints = 4
    point_names = POINT_NAMES

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 256),
        strides: Sequence[int] = (8, 16, 32),
        prior_probability: float = 0.01,
    ) -> None:
        super().__init__()
        if len(channels) != 3 or len(strides) != 3:
            raise ValueError("The armor head requires three P3/P4/P5 feature maps")
        if any(value <= 0 for value in (*channels, *strides)):
            raise ValueError("Feature channels and strides must be positive")
        if not 0 < prior_probability < 1:
            raise ValueError("prior_probability must be between 0 and 1")
        self.channels = tuple(channels)
        self.register_buffer("stride", torch.tensor(strides, dtype=torch.float32))
        cls_hidden = max(channels[0], self.num_classes)
        # The supplied v8n-20260726 teacher uses 32 hidden keypoint channels,
        # and 64 for each classification branch on the nano feature pyramid.
        kpt_hidden = max(channels[0] // 2, self.num_keypoints * 2)
        self.color_branches = nn.ModuleList(_branch(c, cls_hidden, 4) for c in channels)
        self.class_branches = nn.ModuleList(_branch(c, cls_hidden, 9) for c in channels)
        self.keypoint_branches = nn.ModuleList(_branch(c, kpt_hidden, 8) for c in channels)
        self._initialize_predictions(prior_probability)

    def _initialize_predictions(self, probability: float) -> None:
        prior_logit = math.log(probability / (1 - probability))
        for branches in (self.color_branches, self.class_branches):
            for branch in branches:
                nn.init.constant_(branch[-1].bias, prior_logit)
        # A cold-start candidate spans 2 strides horizontally and 1 vertically,
        # centered on its anchor. Distinct corners make enclosing-box losses and
        # assignment meaningful before the head has learned its first offsets.
        corner_bias = torch.tensor([-0.25, 0.0, -0.25, 0.5, 0.75, 0.5, 0.75, 0.0])
        for branch in self.keypoint_branches:
            nn.init.zeros_(branch[-1].weight)
            with torch.no_grad():
                branch[-1].bias.copy_(corner_bias)

    def forward(
        self, features: Sequence[torch.Tensor], return_raw: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        color_logits, class_logits, raw_keypoints, keypoints = [], [], [], []
        anchor_points, anchor_strides = [], []
        for level, feature in enumerate(features):
            batch, _, height, width = feature.shape
            colors = self.color_branches[level](feature).flatten(2).transpose(1, 2)
            classes = self.class_branches[level](feature).flatten(2).transpose(1, 2)
            raw = self.keypoint_branches[level](feature).flatten(2).transpose(1, 2)
            raw = raw.reshape(batch, -1, self.num_keypoints, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=feature.device, dtype=feature.dtype),
                torch.arange(width, device=feature.device, dtype=feature.dtype),
                indexing="ij",
            )
            grid = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
            stride = self.stride[level].to(dtype=feature.dtype)
            color_logits.append(colors)
            class_logits.append(classes)
            raw_keypoints.append(raw)
            keypoints.append((raw * 2 + grid[None, :, None, :]) * stride)
            anchor_points.append((grid + 0.5) * stride)
            anchor_strides.append(torch.ones_like(grid[:, :1]) * stride)

        colors = torch.cat(color_logits, dim=1)
        classes = torch.cat(class_logits, dim=1)
        points = torch.cat(keypoints, dim=1)
        if return_raw:
            return {
                "color_logits": colors,
                "class_logits": classes,
                "keypoints": points,
                "raw_keypoints": torch.cat(raw_keypoints, dim=1),
                "anchor_points": torch.cat(anchor_points, dim=0),
                "strides": torch.cat(anchor_strides, dim=0),
            }
        return torch.cat((colors.sigmoid(), classes.sigmoid(), points.flatten(2)), dim=2).transpose(1, 2)
