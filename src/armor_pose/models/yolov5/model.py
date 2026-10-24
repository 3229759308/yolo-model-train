"""Original YOLOv5n v7.0 backbone/PAN neck with the shared armor head.

Structure source: ultralytics/yolov5 v7.0 models/yolov5n.yaml. The COCO
anchor-based Detect layer is replaced with this project's 4/9/8 head.
"""
from __future__ import annotations

from pathlib import Path
import pickle
import types

import torch
from torch import nn

from ... import settings
from ...common.definitions import POINT_NAMES
from ...common.head import ArmorPoseHead
from ultralytics.nn.modules import Bottleneck, C3, Concat, Conv, SPPF


DEFAULT_BACKBONE_WEIGHTS = settings.WEIGHTS_DIR / 'yolov5n.pt'


class _V5Unpickler(pickle.Unpickler):
    """Resolve original v7 class names without importing its training scripts.

    Used only for the trusted local official COCO checkpoint. Our own training
    checkpoints contain state dictionaries and use weights_only=True instead.
    """
    def find_class(self, module, name):
        if module == 'models.common':
            classes = {'Conv': Conv, 'Bottleneck': Bottleneck, 'C3': C3,
                       'SPPF': SPPF, 'Concat': Concat}
            if name in classes:
                return classes[name]
        if module == 'models.yolo' and name in ('Model', 'DetectionModel', 'Detect'):
            return nn.Module
        return super().find_class(module, name)


def load_official_checkpoint(path: str | Path) -> nn.Module:
    """Read trusted original YOLOv5 weights; never fetch or auto-install code."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Local original YOLOv5n pretrained weights not found: {path}')
    compatibility = types.ModuleType('armor_pose_yolov5_pickle')
    compatibility.Unpickler = _V5Unpickler
    compatibility.load = pickle.load
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, pickle_module=compatibility)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected an original YOLOv5 COCO checkpoint dictionary')
    model = checkpoint.get('ema') or checkpoint.get('model')
    if not isinstance(model, nn.Module) or not hasattr(model, 'model'):
        raise ValueError('Expected the original YOLOv5 model in the checkpoint')
    if len(model.model) != 25 or model.model[0].conv.kernel_size != (6, 6):
        raise ValueError('Weights are not an original YOLOv5 P5 model with a 6x6 stem')
    return model.float().eval()


class YoloV5ArmorPoseModel(nn.Module):
    feature_indices = (17, 20, 23)
    num_colors = 4
    num_classes = 9
    num_keypoints = 4
    point_names = POINT_NAMES

    def __init__(self, backbone_weights: str | Path = DEFAULT_BACKBONE_WEIGHTS,
                 pretrained: bool = True):
        super().__init__()
        self.backbone_weights = str(backbone_weights)
        # Channels and repeated C3 blocks are the original n scale (.25/.33).
        specifications = [
            (-1, Conv(3, 16, 6, 2, 2)),
            (-1, Conv(16, 32, 3, 2)),
            (-1, C3(32, 32, 1)),
            (-1, Conv(32, 64, 3, 2)),
            (-1, C3(64, 64, 2)),
            (-1, Conv(64, 128, 3, 2)),
            (-1, C3(128, 128, 3)),
            (-1, Conv(128, 256, 3, 2)),
            (-1, C3(256, 256, 1)),
            (-1, SPPF(256, 256, 5)),
            (-1, Conv(256, 128, 1, 1)),
            (-1, nn.Upsample(scale_factor=2, mode='nearest')),
            ([-1, 6], Concat(1)),
            (-1, C3(256, 128, 1, False)),
            (-1, Conv(128, 64, 1, 1)),
            (-1, nn.Upsample(scale_factor=2, mode='nearest')),
            ([-1, 4], Concat(1)),
            (-1, C3(128, 64, 1, False)),
            (-1, Conv(64, 64, 3, 2)),
            ([-1, 14], Concat(1)),
            (-1, C3(128, 128, 1, False)),
            (-1, Conv(128, 128, 3, 2)),
            ([-1, 10], Concat(1)),
            (-1, C3(256, 256, 1, False)),
        ]
        self.backbone = nn.ModuleList()
        for index, (route, layer) in enumerate(specifications):
            layer.i, layer.f = index, route
            self.backbone.append(layer)
        # Original v7 SPPF uses SiLU after cv1. Recent Ultralytics versions use
        # Identity there; explicitly restore original semantics for every Conv.
        for module in self.backbone.modules():
            if isinstance(module, Conv):
                module.act = nn.SiLU(inplace=True)
            if isinstance(module, nn.BatchNorm2d):
                module.eps, module.momentum = 1e-3, 0.03
        self.saved_indices = frozenset((4, 6, 10, 14, 17, 20, 23))
        self.pretrained_info = None
        if pretrained:
            original = load_official_checkpoint(backbone_weights)
            source = original.state_dict()
            expected = self.backbone.state_dict()
            transfer = {}
            for key, tensor in expected.items():
                source_key = 'model.' + key
                if source_key not in source or source[source_key].shape != tensor.shape:
                    raise ValueError(f'YOLOv5n pretrained backbone mismatch at {source_key}')
                transfer[key] = source[source_key]
            self.backbone.load_state_dict(transfer, strict=True)
            self.pretrained_info = {'source': str(backbone_weights), 'tensors': len(transfer),
                                    'elements': sum(t.numel() for t in transfer.values())}
        # Keep exactly the same head and initialization used by the YOLOv8 run.
        self.head = ArmorPoseHead(channels=(64, 128, 256))
        self.requires_grad_(True)
        self.train()

    @property
    def stride(self):
        return self.head.stride

    @staticmethod
    def architecture_config():
        return {'name': 'yolov5n-armor-pose', 'backbone': 'yolov5n-v7.0',
                'feature_layers': [17, 20, 23], 'feature_channels': [64, 128, 256],
                'classification_hidden_channels': 64, 'keypoint_hidden_channels': 32,
                'strides': [8, 16, 32], 'num_colors': 4, 'num_classes': 9,
                'num_keypoints': 4, 'point_names': list(POINT_NAMES),
                'output_channels': 21, 'coordinate_space': 'network_input_pixels',
                'keypoint_decode': '(raw * 2 + integer_grid) * stride'}

    def forward_features(self, images):
        saved = []
        x = images
        for layer in self.backbone:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [
                    x if index == -1 else saved[index] for index in layer.f]
            x = layer(x)
            saved.append(x if layer.i in self.saved_indices else None)
        return [saved[index] for index in self.feature_indices]

    def forward(self, images, return_raw=False):
        if not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export() or torch.compiler.is_compiling()):
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError('Input must have shape B x 3 x H x W')
            if images.shape[2] % 32 or images.shape[3] % 32 or min(images.shape[2:]) < 32:
                raise ValueError('Input dimensions must be positive multiples of 32')
        return self.head(self.forward_features(images), return_raw=return_raw)


ArmorPoseModel = YoloV5ArmorPoseModel
