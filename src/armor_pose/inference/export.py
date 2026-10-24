#!/usr/bin/env python3
"""Export an armor training state dictionary to the fixed 21-channel ONNX API.

Example:
  /workspace/tools/bit-auto-label/.venv/bin/python src/armor.py export \\
      --weights runs/armor_pose/best.pt --verify

The existing annotation-tool environment includes ONNX and ONNX Runtime. Export
does not download pretrained weights or modify the input checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch

from .. import settings
from ..common.definitions import POINT_NAMES, CLASS_NAMES_EN, COLOR_NAMES_EN
from ..models import model_from_architecture


EXISTING_EXPORT_PYTHON = str(settings.PROJECT_ROOT / 'tools/bit-auto-label/.venv/bin/python')
DEFAULT_NAMES = CLASS_NAMES_EN
DEFAULT_COLORS = COLOR_NAMES_EN


def _dependencies(verify: bool):
    try:
        import onnx
        import numpy as np
        if verify:
            import onnxruntime as ort
        else:
            ort = None
    except ImportError as exc:
        raise RuntimeError(
            f"Missing export dependency {exc.name!r}. Use the existing environment: "
            f"{EXISTING_EXPORT_PYTHON} src/armor.py export --weights CHECKPOINT --verify"
        ) from exc
    return onnx, np, ort


def _name_mapping(value, default: list[str], field: str) -> dict[int, str]:
    if value is None:
        return dict(enumerate(default))
    if isinstance(value, (list, tuple)):
        result = dict(enumerate(value))
    elif isinstance(value, dict):
        try:
            result = {int(key): item for key, item in value.items()}
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{field} must use contiguous numeric IDs") from exc
    else:
        raise ValueError(f"{field} must be a list or dictionary")
    if set(result) != set(range(len(default))) or any(not isinstance(name, str) or not name for name in result.values()):
        raise ValueError(f"{field} must contain {len(default)} nonempty names with IDs 0..{len(default) - 1}")
    return result


def _model_metadata(checkpoint: dict, architecture: dict) -> dict[str, str]:
    data = checkpoint.get("data_config", {})
    if not isinstance(data, dict):
        data = {}
    names = _name_mapping(checkpoint.get("names", data.get("names")), DEFAULT_NAMES, "names")
    colors = _name_mapping(
        checkpoint.get("color_names", checkpoint.get("colors", data.get("color_names"))),
        DEFAULT_COLORS, "color_names",
    )
    preprocess = checkpoint.get("preprocess", "letterbox")
    if not isinstance(preprocess, str):
        preprocess = json.dumps(preprocess, ensure_ascii=False)
    return {
        "task": "armor-pose",
        "architecture": json.dumps(architecture, ensure_ascii=False),
        "names": json.dumps(names, ensure_ascii=False),
        "colors": json.dumps(colors, ensure_ascii=False),
        "color_names": json.dumps(colors, ensure_ascii=False),
        "point_names": json.dumps(list(POINT_NAMES)),
        "coordinate_space": "network_input_pixels",
        "preprocess": preprocess,
        "color_order": "RGB",
        "normalization": "divide_by_255",
        "letterbox_padding_value": "114",
        "input_size": "[480, 640]",
        "output_layout": "B,21,N: color[0:4],class[4:13],xy[13:21]",
        "candidate_order": "P3,P4,P5; row-major within each feature map",
        "keypoint_decode": "(raw * 2 + integer_grid) * stride",
    }


def export_checkpoint(
    weights: str | Path,
    output: str | Path | None = None,
    device: str = "cpu",
    verify: bool = False,
) -> dict:
    """Write a checked ONNX model; optionally compare one ORT and Torch result."""
    onnx, np, ort = _dependencies(verify)
    weights = Path(weights).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve() if output else weights.with_suffix(".onnx")
    if output == weights or output.suffix.lower() != ".onnx":
        raise ValueError("output must be an .onnx file different from the checkpoint")
    if not weights.is_file():
        raise ValueError(f"Checkpoint is not a file: {weights}")
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected an armor checkpoint containing model_state_dict")
    if tuple(checkpoint.get("input_size", (480, 640))) != (480, 640):
        raise ValueError("The deployment protocol requires checkpoint input_size [480, 640]")
    architecture = checkpoint.get("architecture", {})
    device = "cuda:" + device if device.isdigit() else device
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment; export with --device cpu")
    model = model_from_architecture(architecture)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.float().to(selected_device).eval()
    generator = torch.Generator().manual_seed(0)
    images_cpu = torch.rand(1, 3, 480, 640, generator=generator)
    images = images_cpu.to(selected_device)
    with torch.inference_mode():
        expected = model(images).cpu().numpy()
    if expected.shape != (1, 21, 6300) or not np.isfinite(expected).all():
        raise ValueError(f"Invalid model output: shape={expected.shape}, all_finite={np.isfinite(expected).all()}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.stem}.", suffix=".onnx", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                model, images, str(temporary), dynamo=False, opset_version=17,
                input_names=["images"], output_names=["output0"],
                do_constant_folding=True, external_data=False,
            )
        graph = onnx.load(str(temporary))
        onnx.helper.set_model_props(graph, _model_metadata(checkpoint, model.architecture_config()))
        onnx.checker.check_model(graph)
        onnx.save(graph, str(temporary))
        result = {"weights": str(weights), "output": str(output), "input_shape": [1, 3, 480, 640],
                  "output_shape": [1, 21, 6300], "verified": bool(verify)}
        if verify:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            session = ort.InferenceSession(str(temporary), sess_options=options, providers=["CPUExecutionProvider"])
            actual = session.run(["output0"], {"images": images_cpu.numpy()})[0]
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError(f"Invalid ONNX Runtime output: {actual.shape}")
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
            result["max_abs_error"] = float(np.abs(actual - expected).max())
        temporary.replace(output)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="Armor trainer checkpoint (.pt)")
    parser.add_argument("--output", type=Path, help="ONNX destination (default: checkpoint stem + .onnx)")
    parser.add_argument("--device", default="cpu", help="Torch export device: cpu, cuda:0, or 0")
    parser.add_argument("--verify", action="store_true", help="Compare fixed random input in ONNX Runtime and Torch")
    args = parser.parse_args(argv)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    try:
        result = export_checkpoint(args.weights, args.output, args.device, args.verify)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
