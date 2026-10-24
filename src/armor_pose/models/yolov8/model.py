"""YOLOv8n-Pose backbone and neck with the shared 21-channel armor head.

This adapter owns the YOLOv8-specific feature routing. The shared head keeps
color, class and keypoint supervision independent of the backbone family.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ... import settings  # Configure the offline environment before Ultralytics.
from ...common.definitions import POINT_NAMES
from ...common.head import ArmorPoseHead
import ultralytics
from ultralytics.nn.tasks import PoseModel, load_checkpoint


DEFAULT_BACKBONE_WEIGHTS = settings.WEIGHTS_DIR / "yolov8n-pose.pt"


class ArmorPoseModel(nn.Module):
    """Trainable YOLOv8n-Pose backbone/neck and a newly initialized armor head.

    ``pretrained=True`` reads the local COCO checkpoint, transferring only layers
    0..21. Its human detection/pose head is discarded. ``pretrained=False`` builds
    the same architecture from the installed YAML and never downloads weights;
    use that mode before loading one of this project's state dictionaries.
    """

    feature_indices = (15, 18, 21)
    num_colors = 4
    num_classes = 9
    num_keypoints = 4
    point_names = POINT_NAMES

    def __init__(
        self,
        backbone_weights: str | Path = DEFAULT_BACKBONE_WEIGHTS,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_weights = str(backbone_weights)
        if pretrained:
            weights_path = Path(backbone_weights).expanduser()
            if not weights_path.is_file():
                raise FileNotFoundError(f"Local pretrained backbone not found: {weights_path}")
            source, _ = load_checkpoint(str(weights_path), device="cpu", fuse=False)
        else:
            # The n-scaled path resolves to the installed yolov8-pose.yaml, with
            # scale=n. There is no remote model identifier in this code path.
            config_path = Path(ultralytics.__file__).resolve().parent / "cfg/models/v8/yolov8n-pose.yaml"
            source = PoseModel(str(config_path), verbose=False)
        if len(source.model) != 23 or tuple(source.model[-1].f) != self.feature_indices:
            raise ValueError("Expected a YOLOv8n-Pose backbone with feature layers 15, 18, 21")
        input_channels = tuple(branch[0].conv.in_channels for branch in source.model[-1].cv4)
        if input_channels != (64, 128, 256):
            raise ValueError(f"Expected YOLOv8n feature channels (64, 128, 256), got {input_channels}")
        self.backbone = nn.ModuleList(list(source.model.children())[:-1])
        self.saved_indices = frozenset(source.save)
        self.head = ArmorPoseHead(channels=input_channels)
        self.requires_grad_(True)
        self.train()

    @property
    def stride(self) -> torch.Tensor:
        return self.head.stride

    @staticmethod
    def architecture_config() -> dict:
        """Serializable architecture/protocol metadata for checkpoints/ONNX."""
        return {
            "name": "yolov8n-armor-pose",
            "backbone": "yolov8n-pose",
            "feature_layers": [15, 18, 21],
            "feature_channels": [64, 128, 256],
            "classification_hidden_channels": 64,
            "keypoint_hidden_channels": 32,
            "strides": [8, 16, 32],
            "num_colors": 4,
            "num_classes": 9,
            "num_keypoints": 4,
            "point_names": list(POINT_NAMES),
            "output_channels": 21,
            "coordinate_space": "network_input_pixels",
            "keypoint_decode": "(raw * 2 + integer_grid) * stride",
        }

    def forward(
        self, images: torch.Tensor, return_raw: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export() or torch.compiler.is_compiling()):
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError("Input must have shape B x 3 x H x W")
            if images.shape[2] % 32 or images.shape[3] % 32 or min(images.shape[2:]) < 32:
                raise ValueError("Input height and width must be positive multiples of 32")
        saved = []
        x = images
        for layer in self.backbone:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [
                    x if index == -1 else saved[index] for index in layer.f
                ]
            x = layer(x)
            saved.append(x if layer.i in self.saved_indices else None)
        return self.head([saved[index] for index in self.feature_indices], return_raw=return_raw)


# Explicit family name; the original class name remains compatible.
YoloV8ArmorPoseModel = ArmorPoseModel
