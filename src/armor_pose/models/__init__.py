"""Explicit model selection; unimplemented backbone families never fall back.

Only YOLOv8n has a working adapter. Register a separately implemented adapter
before selecting a new backbone family; changing a YAML name is insufficient.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch import nn

from .yolov8 import YoloV8ArmorPoseModel


DEFAULT_MODEL = "yolov8n-armor-pose"
MODEL_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    DEFAULT_MODEL: YoloV8ArmorPoseModel,
}
MODEL_ALIASES = {"yolov8n": DEFAULT_MODEL}


def available_models() -> tuple[str, ...]:
    """Return implemented canonical model names, excluding aliases/placeholders."""
    return tuple(sorted(MODEL_BUILDERS))


def register_model(name: str, builder: Callable[..., nn.Module]) -> None:
    """Register a real adapter without replacing an existing model or alias."""
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError("Model name must be a nonempty string without surrounding whitespace")
    if not callable(builder):
        raise TypeError("Model builder must be callable")
    if name in MODEL_BUILDERS or name in MODEL_ALIASES:
        raise ValueError(f"Armor model {name!r} is already registered")
    MODEL_BUILDERS[name] = builder


def build_model(
    name: str = DEFAULT_MODEL,
    backbone_weights: str | Path | None = None,
    pretrained: bool = True,
) -> nn.Module:
    """Build only the requested registered adapter, optionally with local weights."""
    if not isinstance(name, str):
        raise ValueError("Model name must be a string")
    canonical = MODEL_ALIASES.get(name, name)
    if canonical not in MODEL_BUILDERS:
        available = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(
            f"Unsupported armor model {name!r}. Implemented models: {available}. "
            "YOLOv5 and YOLO26 adapters are not implemented; no fallback is performed."
        )
    kwargs: dict[str, Any] = {"pretrained": pretrained}
    if backbone_weights is not None:
        kwargs["backbone_weights"] = backbone_weights
    return MODEL_BUILDERS[canonical](**kwargs)


def model_from_architecture(architecture: dict | None) -> nn.Module:
    """Build offline for checkpoint loading, validating the shared protocol.

    Historical checkpoints omitted architecture metadata or its name; these
    explicitly retain the original YOLOv8n default. Any supplied model name is
    resolved through the registry and must identify an implemented adapter.
    """
    if architecture is None:
        architecture = {}
    if not isinstance(architecture, dict):
        raise ValueError("Checkpoint architecture must be a dictionary or None")
    for field, expected in (
        ("output_channels", 21), ("num_colors", 4),
        ("num_classes", 9), ("num_keypoints", 4),
    ):
        if field in architecture and architecture[field] != expected:
            raise ValueError(f"Checkpoint architecture requires {field}={expected}, got {architecture[field]!r}")
    return build_model(architecture.get("name", DEFAULT_MODEL), pretrained=False)


__all__ = [
    "MODEL_BUILDERS", "MODEL_ALIASES", "DEFAULT_MODEL",
    "available_models", "register_model", "build_model", "model_from_architecture",
]
