#!/usr/bin/env python3
"""Predict original-image four-corner armor labels from a custom checkpoint.

Example:
  python src/armor.py predict --weights runs/armor_pose/best.pt \\
      --source /workspace/data/raw_image --device 0 --save-images

Each labels/*.jpg.txt row is color_id class_id followed by LT/LB/RB/RT x,y
coordinates in source-image pixels. Image files and checkpoints are read only;
an existing nonempty output directory is rejected.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from ..common.runtime import CLASS_NAMES, COLOR_NAMES, decode_predictions, load_model

import cv2
import numpy as np
import torch

from ..common.definitions import POINT_NAMES as _POINT_NAMES
from ..settings import OUTPUTS_DIR


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_OUTPUT = OUTPUTS_DIR / 'armor_pose_predictions'
POINT_NAMES = list(_POINT_NAMES)
PREVIEW_COLORS = [(255, 100, 30), (70, 70, 255), (240, 240, 240), (220, 80, 220)]


def letterbox(image: np.ndarray, input_size: tuple[int, int] = (480, 640)):
    """Match ArmorPoseDataset's rounded resize, integer padding and RGB scale."""
    height, width = image.shape[:2]
    input_h, input_w = input_size
    gain = min(input_w / width, input_h / height)
    resized_w = min(input_w, max(1, round(width * gain)))
    resized_h = min(input_h, max(1, round(height * gain)))
    left, top = (input_w - resized_w) // 2, (input_h - resized_h) // 2
    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = cv2.resize(
        image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR,
    )
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
    metadata = {
        "original_size": [height, width], "resized_size": [resized_h, resized_w],
        "input_size": list(input_size), "padding": [left, top],
        "scale": [resized_w / width, resized_h / height],
        "preprocess": "letterbox", "padding_value": 114,
        "color_order": "RGB", "normalization": "divide_by_255",
    }
    return tensor, metadata


def restore_points(points: np.ndarray, metadata: dict) -> np.ndarray:
    """Undo the exact per-axis rounded scale without clipping image-edge points."""
    return (points.astype(np.float64) - np.asarray(metadata["padding"])) / np.asarray(metadata["scale"])


def _images(source: Path) -> tuple[Path, list[Path]]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {source}")
        return source.parent, [source]
    if not source.is_dir():
        raise ValueError(f"Source is not an image or directory: {source}")
    images = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No supported images found in {source}")
    return source, images


def _preview(image: np.ndarray, prediction: dict) -> np.ndarray:
    image = image.copy()
    height, width = image.shape[:2]
    for points, color_id, class_id, score in zip(
        prediction["points"], prediction["color_ids"], prediction["class_ids"], prediction["scores"],
    ):
        color_id, class_id = int(color_id), int(class_id)
        color = PREVIEW_COLORS[color_id]
        # Preview-only limits keep OpenCV integer conversion safe. The original
        # floating-point predictions, TXT and JSON coordinates are untouched.
        vertices = np.rint(np.clip(points, -1e7, 1e7)).astype(np.int32)
        cv2.polylines(image, [vertices], True, color, 2, cv2.LINE_AA)
        for index, (x, y) in enumerate(vertices):
            cv2.circle(image, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
            cv2.putText(image, f"P{index}", (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, color, 1, cv2.LINE_AA)
        text_x = int(np.clip(vertices[:, 0].min(), 0, max(width - 1, 0)))
        text_y = int(np.clip(vertices[:, 1].min() - 7, min(16, height - 1), max(height - 1, 0)))
        text = f"{COLOR_NAMES[color_id]}/{CLASS_NAMES[class_id]} {float(score):.3f}"
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    return image


def predict(
    weights: str | Path,
    source: str | Path,
    output: str | Path = DEFAULT_OUTPUT,
    device: str = "0",
    confidence: float = .25,
    nms_iou: float = .5,
    save_images: bool = False,
) -> dict:
    """Infer a file/tree and write custom pixel labels plus confidence metadata."""
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("conf must be in [0, 1]")
    if not math.isfinite(nms_iou) or not 0 <= nms_iou <= 1:
        raise ValueError("iou must be in [0, 1]")
    weights = Path(weights).expanduser().resolve(strict=True)
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be new or an empty directory: {output}")
    image_root, images = _images(source)
    if source.is_dir() and output.is_relative_to(source):
        raise ValueError("Output must be outside the source image directory")
    device = "cuda:" + device if device.isdigit() else device
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    model, checkpoint = load_model(str(weights), selected_device)
    if tuple(checkpoint.get("input_size", (480, 640))) != (480, 640):
        raise ValueError("This deployment requires checkpoint input_size [480, 640]")
    if checkpoint.get("preprocess", "letterbox") != "letterbox":
        raise ValueError("Checkpoint preprocessing must be letterbox")
    if checkpoint.get("point_names", POINT_NAMES) != POINT_NAMES:
        raise ValueError("Checkpoint point order must be LT, LB, RB, RT")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "custom-head-pixels", "weights": str(weights), "source": str(source),
        "image_root": str(image_root), "coordinate_space": "original_image_pixels",
        "input_size": [480, 640], "point_names": POINT_NAMES,
        "class_names": checkpoint.get("names", CLASS_NAMES),
        "color_names": checkpoint.get("color_names", COLOR_NAMES),
        "confidence_threshold": confidence, "nms_iou": nms_iou,
        "confidence_definition": "max(color_scores) * max(class_scores)",
        "nms_grouping": "per color_id and class_id", "max_detections_per_image": 300,
        "images": [],
    }
    total_objects = 0
    started = time.monotonic()
    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        tensor, metadata = letterbox(image)
        with torch.inference_mode():
            prediction = decode_predictions(model(tensor.to(selected_device))[0], confidence, nms_iou)
        prediction["points"] = restore_points(prediction["points"], metadata)
        relative_image = image_path.relative_to(image_root)
        relative_label = Path("labels") / relative_image.parent / (relative_image.name + ".txt")
        label_path = output / relative_label
        label_path.parent.mkdir(parents=True, exist_ok=True)
        rows, detections = [], []
        for points, color_id, class_id, score in zip(
            prediction["points"], prediction["color_ids"], prediction["class_ids"], prediction["scores"],
        ):
            values = [int(color_id), int(class_id), *points.reshape(-1).tolist()]
            rows.append(" ".join(format(value, ".12g") for value in values))
            detections.append({"color_id": int(color_id), "class_id": int(class_id),
                               "confidence": float(score), "points": points.tolist()})
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        record = {"image": relative_image.as_posix(), "source": str(image_path),
                  "label": relative_label.as_posix(), "original_size": metadata["original_size"],
                  "metadata": metadata, "detections": detections}
        if save_images:
            relative_preview = Path("images") / relative_image.parent / (relative_image.name + ".jpg")
            preview_path = output / relative_preview
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(preview_path), _preview(image, prediction)):
                raise OSError(f"Cannot save preview: {preview_path}")
            record["preview"] = relative_preview.as_posix()
        manifest["images"].append(record)
        total_objects += len(rows)
        if (index + 1) % 100 == 0:
            print(f"Predicted {index + 1}/{len(images)} images, {total_objects} objects", flush=True)
    manifest["summary"] = {"images": len(images), "objects": total_objects,
                           "seconds": time.monotonic() - started}
    (output / "predictions.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output": str(output), **manifest["summary"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="One image or a recursively scanned image directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0", help="0, cuda:0, or cpu")
    parser.add_argument("--conf", type=float, default=.25, help="Color probability times class probability")
    parser.add_argument("--iou", type=float, default=.5, help="NMS IoU threshold per color/class pair")
    parser.add_argument("--save-images", action="store_true", help="Save annotated preview copies")
    args = parser.parse_args(argv)
    try:
        result = predict(args.weights, args.source, args.output, args.device, args.conf, args.iou, args.save_images)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
