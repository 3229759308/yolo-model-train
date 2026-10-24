"""Data, storage, and ONNX inference for the v8n-20260726 armor head.

Output channels are color[0:4], class[4:13], and four xy pairs[13:21].
Output xy units are declared by output_coordinates: 'original' preserves them;
'input' reverses preprocessing once in OnnxDetector.predict. Stored annotations
and exported labels always use original-image pixels.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
from uuid import uuid4

import cv2
import numpy as np

SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _integer(value: Any, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} 必须是整数')
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f'{name} 超出允许范围')
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f'{name} 必须是有限数值')
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} 必须是有限数值') from exc
    if not math.isfinite(number):
        raise ValueError(f'{name} 必须是有限数值')
    return number


def _threshold(value: Any, name: str) -> float:
    number = _finite(value, name)
    if not 0 <= number <= 1:
        raise ValueError(f'{name} 必须在 0 到 1 之间')
    return number


@dataclass
class ArmorAnnotation:
    """Four ordered points in original-image pixels; outside points are allowed."""

    id: str = field(default_factory=lambda: str(uuid4()))
    color_id: int = 0
    class_id: int = 0
    points: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0] for _ in range(4)])
    confidence: float | None = None
    source: str = 'manual'

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError('标注 id 必须是非空字符串')
        self.color_id = _integer(self.color_id, 'color_id', 0, 3)
        self.class_id = _integer(self.class_id, 'class_id', 0, 8)
        if not isinstance(self.points, (list, tuple, np.ndarray)) or len(self.points) != 4:
            raise ValueError('标注必须包含按固定顺序排列的 4 个关键点')
        converted = []
        for point in self.points:
            if not isinstance(point, (list, tuple, np.ndarray)) or len(point) != 2:
                raise ValueError('每个关键点必须包含 x、y 两个坐标')
            converted.append([_finite(point[0], 'x'), _finite(point[1], 'y')])
        self.points = converted
        if self.confidence is not None:
            self.confidence = _threshold(self.confidence, 'confidence')
        if self.source not in ('manual', 'model'):
            raise ValueError('source 必须为 manual 或 model')

    def to_dict(self) -> dict[str, Any]:
        # Revalidate because the GUI edits points and class selections in place.
        validated = ArmorAnnotation(self.id, self.color_id, self.class_id,
                                    self.points, self.confidence, self.source)
        return {'id': validated.id, 'color_id': validated.color_id,
                'class_id': validated.class_id, 'points': validated.points,
                'confidence': validated.confidence, 'source': validated.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ArmorAnnotation':
        if not isinstance(data, dict):
            raise ValueError('标注必须是 JSON 对象')
        required = ('color_id', 'class_id', 'points')
        if any(key not in data for key in required):
            raise ValueError('标注缺少 color_id、class_id 或 points')
        return cls(id=data.get('id', str(uuid4())), color_id=data['color_id'],
                   class_id=data['class_id'], points=data['points'],
                   confidence=data.get('confidence'), source=data.get('source', 'manual'))


def default_config() -> dict[str, Any]:
    return {
        'schema_version': SCHEMA_VERSION,
        'color_names': ['蓝', '红', '白', '紫'],
        'class_names': [f'class_{index}' for index in range(9)],
        'point_names': ['P0', 'P1', 'P2', 'P3'],
        'input_width': 640, 'input_height': 480,
        'preprocess': 'stretch', 'rgb': True, 'normalize': True,
        'scores_are_logits': False, 'score_mode': 'product',
        'confidence': 0.25, 'iou': 0.45, 'class_aware': True,
        'output_coordinates': 'original',
        'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        'cpu_threads': 2, 'thread_spinning': False, 'model_path': '',
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge supplied settings with defaults and return a validated independent dict."""
    if not isinstance(config, dict):
        raise ValueError('配置必须是 JSON 对象')
    result = default_config()
    result.update(copy.deepcopy(config))
    if _integer(result['schema_version'], 'schema_version', 1) != SCHEMA_VERSION:
        raise ValueError('不支持此配置 schema_version')
    for key, count in [('color_names', 4), ('class_names', 9), ('point_names', 4)]:
        names = result[key]
        if (not isinstance(names, list) or len(names) != count
                or any(not isinstance(name, str) or not name.strip() for name in names)):
            raise ValueError(f'{key} 必须包含 {count} 个非空名称')
    for key in ('input_width', 'input_height', 'cpu_threads'):
        result[key] = _integer(result[key], key, 1)
    if result['preprocess'] not in ('stretch', 'letterbox'):
        raise ValueError('preprocess 必须为 stretch 或 letterbox')
    if result['score_mode'] not in ('product', 'min', 'class', 'color'):
        raise ValueError('score_mode 必须为 product、min、class 或 color')
    if result['output_coordinates'] not in ('original', 'input'):
        raise ValueError('output_coordinates 必须为 original 或 input')
    if not isinstance(result['model_path'], str):
        raise ValueError('model_path 必须是字符串（可留空手动选择模型）')
    for key in ('rgb', 'normalize', 'scores_are_logits', 'class_aware', 'thread_spinning'):
        if not isinstance(result[key], bool):
            raise ValueError(f'{key} 必须是布尔值')
    for key in ('confidence', 'iou'):
        result[key] = _threshold(result[key], key)
    providers = result['providers']
    if (not isinstance(providers, list) or not providers
            or any(not isinstance(item, str) or not item for item in providers)):
        raise ValueError('providers 必须是非空的 ONNX Runtime provider 名称列表')
    return result


def _atomic_write(path: Path, text: str, *, backup: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    backup_name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if backup and path.exists():
            with tempfile.NamedTemporaryFile('wb', dir=path.parent,
                                             prefix=f'.{path.name}.', suffix='.bak.tmp', delete=False) as handle:
                backup_name = handle.name
                handle.write(path.read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(backup_name, path.with_name(path.name + '.bak'))
            backup_name = None
        os.replace(temp_name, path)
        temp_name = None
    finally:
        for temporary in (temp_name, backup_name):
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).with_name('model_config.json')
    with Path(path).open(encoding='utf-8') as handle:
        return validate_config(json.load(handle))


def save_config(config: dict[str, Any], path: str | Path) -> None:
    _atomic_write(Path(path), json.dumps(validate_config(config), ensure_ascii=False,
                                         indent=2, allow_nan=False) + '\n', backup=True)


def scan_images(root: str | Path) -> list[Path]:
    """Find images recursively, skipping symlinks whose targets leave the root."""
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError('图片根目录不是文件夹')
    images = []
    seen = set()
    for item in root.rglob('*'):
        if item.suffix.lower() not in IMAGE_EXTENSIONS or not item.is_file():
            continue
        resolved = item.resolve()
        if not resolved.is_relative_to(root) or resolved in seen:
            continue
        seen.add(resolved)
        images.append(item)
    return sorted(images, key=lambda item: item.relative_to(root).as_posix().casefold())


class AnnotationStore:
    """Versioned JSON sidecars, preserving image suffixes and previous saves."""

    def __init__(self, image_root: str | Path, label_root: str | Path):
        self.image_root = Path(image_root).expanduser().resolve(strict=True)
        if not self.image_root.is_dir():
            raise ValueError('图片根目录不是文件夹')
        self.label_root = Path(label_root).expanduser().resolve()

    def _relative_image(self, image_path: str | Path) -> Path:
        path = Path(image_path).expanduser()
        if not path.is_absolute():
            path = self.image_root / path
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self.image_root) or not resolved.is_file():
            raise ValueError('图片路径必须指向图片根目录内的文件')
        return resolved.relative_to(self.image_root)

    def label_path(self, image_path: str | Path) -> Path:
        relative = self._relative_image(image_path)
        target = self.label_root / relative.parent / (relative.name + '.json')
        if not target.resolve().is_relative_to(self.label_root):
            raise ValueError('标签路径超出标签根目录')
        return target

    def load(self, image_path: str | Path) -> dict[str, Any] | None:
        relative = self._relative_image(image_path)
        path = self.label_path(image_path)
        if not path.exists():
            return None
        try:
            with path.open(encoding='utf-8') as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError('标签必须是 JSON 对象')
            if _integer(data.get('schema_version'), 'schema_version', 1) != SCHEMA_VERSION:
                raise ValueError('不支持标签 schema_version')
            image = data.get('image')
            if not isinstance(image, dict) or image.get('relative_path') != relative.as_posix():
                raise ValueError('标签中的图片路径与当前图片不一致')
            _integer(image.get('width'), 'image.width', 1)
            _integer(image.get('height'), 'image.height', 1)
            if not isinstance(data.get('reviewed'), bool):
                raise ValueError('标签 reviewed 必须是布尔值')
            if not isinstance(data.get('annotations'), list):
                raise ValueError('标签 annotations 必须是数组')
            annotations = [ArmorAnnotation.from_dict(item).to_dict() for item in data['annotations']]
            if len({item['id'] for item in annotations}) != len(annotations):
                raise ValueError('标签中存在重复标注 id')
            data['annotations'] = annotations
            return data
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'标签损坏，未将其视为空标签：{path}\n{exc}') from exc

    def save(self, image_path: str | Path, width: int, height: int,
             annotations: Iterable[ArmorAnnotation], reviewed: bool = False) -> Path:
        width = _integer(width, 'width', 1)
        height = _integer(height, 'height', 1)
        if not isinstance(reviewed, bool):
            raise ValueError('reviewed 必须是布尔值')
        validated = [_annotation(item).to_dict() for item in annotations]
        if len({item['id'] for item in validated}) != len(validated):
            raise ValueError('不能保存重复的标注 id')
        relative = self._relative_image(image_path)
        path = self.label_path(image_path)
        previous = self.load(image_path)  # Never overwrite a corrupt sidecar as an empty image.
        now = datetime.now(timezone.utc).isoformat()
        data = dict(previous or {})
        data.update(schema_version=SCHEMA_VERSION,
                    image={'relative_path': relative.as_posix(), 'width': width, 'height': height},
                    reviewed=reviewed, annotations=validated,
                    created_at=data.get('created_at', now), updated_at=now)
        _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                      backup=True)
        return path


def _annotation(value: ArmorAnnotation | dict[str, Any]) -> ArmorAnnotation:
    return ArmorAnnotation.from_dict(value.to_dict() if isinstance(value, ArmorAnnotation) else value)


def export_custom(path: str | Path, annotations: Iterable[ArmorAnnotation]) -> None:
    """Write color class x0 y0 ... x3 y3, preserving original pixel coordinates."""
    lines = []
    for item in annotations:
        ann = _annotation(item)
        coordinates = ' '.join(format(value, '.12g') for point in ann.points for value in point)
        lines.append(f'{ann.color_id} {ann.class_id} {coordinates}')
    _atomic_write(Path(path), '\n'.join(lines) + ('\n' if lines else ''), backup=True)


def export_yolo(path: str | Path, annotations: Iterable[ArmorAnnotation],
                width: int, height: int) -> None:
    """Write 36-class YOLO pose rows; reject outside or degenerate boxes.

    Class ID is color_id * 9 + class_id. Points remain in their stored order.
    Each point uses visibility 2. This is an interchange format, not the custom head layout.
    """
    width, height = _integer(width, 'width', 1), _integer(height, 'height', 1)
    lines = []
    for item in annotations:
        ann = _annotation(item)
        points = np.asarray(ann.points, dtype=np.float64)
        if np.any(points < 0) or np.any(points[:, 0] > width) or np.any(points[:, 1] > height):
            raise ValueError(f'标注 {ann.id} 的关键点超出图片，无法导出 YOLO；请先手动调整')
        low, high = points.min(axis=0), points.max(axis=0)
        if np.any(high <= low):
            raise ValueError(f'标注 {ann.id} 的边界框退化，无法导出 YOLO')
        values = [ann.color_id * 9 + ann.class_id,
                  (low[0] + high[0]) / (2 * width), (low[1] + high[1]) / (2 * height),
                  (high[0] - low[0]) / width, (high[1] - low[1]) / height]
        for x, y in points:
            values.extend((x / width, y / height, 2))
        lines.append(' '.join(format(value, '.12g') for value in values))
    _atomic_write(Path(path), '\n'.join(lines) + ('\n' if lines else ''), backup=True)


def _resize_geometry(width: int, height: int, config: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return actual resized width/height and integer left/top padding."""
    target_width, target_height = config['input_width'], config['input_height']
    if config['preprocess'] == 'stretch':
        return target_width, target_height, 0, 0
    scale = min(target_width / width, target_height / height)
    resized_width = min(target_width, max(1, round(width * scale)))
    resized_height = min(target_height, max(1, round(height * scale)))
    return (resized_width, resized_height,
            (target_width - resized_width) // 2, (target_height - resized_height) // 2)


def preprocess_image(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """BGR HWC -> contiguous float32 NCHW using the configured resize and padding."""
    config = validate_config(config)
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or image.shape[0] < 1 or image.shape[1] < 1 or image.dtype != np.uint8):
        raise ValueError('输入图片必须是非空 uint8 BGR 三通道图像')
    width, height = config['input_width'], config['input_height']
    new_width, new_height, left, top = _resize_geometry(image.shape[1], image.shape[0], config)
    small = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    if config['preprocess'] == 'stretch':
        resized = small
    else:
        resized = np.full((height, width, 3), 114, dtype=np.uint8)
        resized[top:top + new_height, left:left + new_width] = small
    if config['rgb']:
        resized = resized[:, :, ::-1]
    blob = resized.astype(np.float32)
    if config['normalize']:
        blob /= 255.0
    return np.ascontiguousarray(blob.transpose(2, 0, 1)[None])


def decode_output(output: np.ndarray, config: dict[str, Any],
                  confidence: float | None = None, iou: float | None = None) -> list[ArmorAnnotation]:
    """Decode 1x21xN or 1xNx21, preserving raw xy units without coordinate conversion.

    Only OnnxDetector.predict can map input coordinates because it knows image size.
    """
    config = validate_config(config)
    confidence = config['confidence'] if confidence is None else _threshold(confidence, 'confidence')
    iou = config['iou'] if iou is None else _threshold(iou, 'iou')
    array = np.asarray(output)
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f'输出形状必须是 1×21×N 或 1×N×21，实际为 {array.shape}')
    if array.shape[1] == 21:
        rows = array[0].T
    elif array.shape[2] == 21:
        rows = array[0]
    else:
        raise ValueError(f'输出通道数必须是 21，实际为 {array.shape}')
    if not np.issubdtype(rows.dtype, np.number) or np.iscomplexobj(rows):
        raise ValueError('输出张量必须包含实数')
    rows = rows.astype(np.float64, copy=False)
    rows = rows[np.isfinite(rows).all(axis=1)]
    if not len(rows):
        return []
    branches = rows[:, :13].copy()
    if config['scores_are_logits']:
        branches = 1.0 / (1.0 + np.exp(-np.clip(branches, -700, 700)))
    valid_scores = (branches >= 0).all(axis=1) & (branches <= 1).all(axis=1)
    rows, branches = rows[valid_scores], branches[valid_scores]
    if not len(rows):
        return []
    color_ids = branches[:, :4].argmax(axis=1)
    class_ids = branches[:, 4:13].argmax(axis=1)
    indexes = np.arange(len(rows))
    color_scores = branches[indexes, color_ids]
    class_scores = branches[indexes, 4 + class_ids]
    mode = config['score_mode']
    scores = (color_scores * class_scores if mode == 'product' else
              np.minimum(color_scores, class_scores) if mode == 'min' else
              class_scores if mode == 'class' else color_scores)
    points = rows[:, 13:21].reshape(-1, 4, 2)
    lows, highs = points.min(axis=1), points.max(axis=1)
    valid = (scores >= confidence) & (highs > lows).all(axis=1)
    points, lows, highs, scores = points[valid], lows[valid], highs[valid], scores[valid]
    color_ids, class_ids = color_ids[valid], class_ids[valid]
    areas = np.prod(highs - lows, axis=1)
    order = np.argsort(-scores, kind='stable')
    selected = []
    while len(order):
        current = int(order[0])
        selected.append(current)
        remaining = order[1:]
        if not len(remaining):
            break
        intersection = np.prod(np.maximum(0, np.minimum(highs[current], highs[remaining])
                                             - np.maximum(lows[current], lows[remaining])), axis=1)
        overlap = intersection / (areas[current] + areas[remaining] - intersection)
        same_class = ((color_ids[remaining] == color_ids[current])
                      & (class_ids[remaining] == class_ids[current])) if config['class_aware'] else True
        order = remaining[~((overlap > iou) & same_class)]
    return [ArmorAnnotation(color_id=int(color_ids[index]), class_id=int(class_ids[index]),
                            points=points[index].tolist(), confidence=float(scores[index]), source='model')
            for index in selected]


class OnnxDetector:
    """Lazy optional ONNX Runtime loader; invalid models raise instead of producing labels."""

    def __init__(self, model_path: str | Path, config: dict[str, Any]):
        self.config = validate_config(config)
        self.model_path = Path(model_path).expanduser().resolve(strict=True)
        if not self.model_path.is_file():
            raise ValueError('模型路径不是文件')
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('自动标注需要 onnxruntime，请在当前 Python 环境运行：python -m pip install onnxruntime') from exc
        requested = list(dict.fromkeys(self.config['providers']))
        available = ort.get_available_providers()
        cuda = 'CUDAExecutionProvider'
        cpu = 'CPUExecutionProvider'
        cuda_requested = cuda in requested
        cpu_fallback = cpu in requested
        missing = [provider for provider in requested if provider != cuda and provider not in available]
        if missing:
            raise ValueError(f'当前 ONNX Runtime 不支持配置的 provider：{missing}；可用：{available}')
        self.warnings: list[str] = []
        self.session_options = ort.SessionOptions()
        self.session_options.intra_op_num_threads = self.config['cpu_threads']
        self.session_options.inter_op_num_threads = 1
        self.session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        spinning = '1' if self.config['thread_spinning'] else '0'
        self.session_options.add_session_config_entry('session.intra_op.allow_spinning', spinning)
        self.session_options.add_session_config_entry('session.inter_op.allow_spinning', spinning)

        providers = requested.copy()
        if cuda_requested and cuda not in available:
            if not cpu_fallback:
                raise RuntimeError('当前 ONNX Runtime 不提供 CUDA，且未配置 CPU 兜底；请启用 CPU 或安装支持 CUDA 的 Runtime')
            providers.remove(cuda)
            self.warnings.append('当前 ONNX Runtime 不提供 CUDA，已回退到 CPU')
        elif cuda_requested:
            # CUDA libraries can already be installed with PyTorch/NVIDIA wheels,
            # yet remain invisible to ORT's provider loader until preloaded.
            preload = getattr(ort, 'preload_dlls', None)
            if callable(preload):
                try:
                    preload()
                except Exception as exc:
                    self.warnings.append(f'CUDA 依赖预加载失败，继续检查实际执行后端：{exc}')

        try:
            self.session = ort.InferenceSession(str(self.model_path),
                                               sess_options=self.session_options, providers=providers)
        except Exception as exc:
            if not cuda_requested:
                raise
            if not cpu_fallback:
                raise RuntimeError(f'CUDA 模型加载失败，且未配置 CPU 兜底：{exc}') from exc
            self.warnings.append(f'CUDA 模型加载失败，已回退到 CPU：{exc}')
            self.session = ort.InferenceSession(str(self.model_path),
                                               sess_options=self.session_options, providers=[cpu])

        self.providers = list(self.session.get_providers())
        if cuda_requested and cuda not in self.providers:
            if not cpu_fallback:
                raise RuntimeError(f'CUDA 未成功启用，实际执行后端为 {self.providers}；未配置 CPU 兜底，已停止加载')
            if cpu not in self.providers:
                raise RuntimeError(f'CUDA 未成功启用且 CPU 回退不可用，实际执行后端：{self.providers}')
            if not any('回退' in warning for warning in self.warnings):
                self.warnings.append('CUDA 未成功启用，已回退到 CPU；请检查 CUDA 依赖和 Runtime 日志')
        missing_other = [provider for provider in requested if provider not in (cuda, cpu)
                         and provider not in self.providers]
        if missing_other:
            raise RuntimeError(f'配置的执行后端未成功启用：{missing_other}；实际执行后端：{self.providers}')
        # Avoid ORT silently rebuilding a failed CUDA session on CPU during run().
        # Initialization fallback above is explicit and reflected in the UI.
        disable_fallback = getattr(self.session, 'disable_fallback', None)
        if callable(disable_fallback):
            disable_fallback()
        labels = {cuda: 'CUDA', cpu: 'CPU', 'TensorrtExecutionProvider': 'TensorRT'}
        active = [labels.get(provider, provider.removesuffix('ExecutionProvider'))
                  for provider in self.providers]
        gpu_active = any(provider != cpu for provider in self.providers)
        self.backend_summary = ('GPU · ' + ' / '.join(active)) if gpu_active else 'CPU'
        fallback_warnings = [warning for warning in self.warnings if '回退' in warning]
        if fallback_warnings:
            self.backend_summary += ' · ' + '；'.join(fallback_warnings)
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError('模型必须只有一个图像输入')
        image_input = inputs[0]
        expected = (1, 3, self.config['input_height'], self.config['input_width'])
        if len(image_input.shape) != 4:
            raise ValueError(f'模型输入应为 NCHW {expected}，实际为 {image_input.shape}')
        for actual, wanted in zip(image_input.shape, expected):
            if isinstance(actual, (int, np.integer)) and actual > 0 and actual != wanted:
                raise ValueError(f'模型输入应为 {expected}，实际为 {image_input.shape}；请修改模型配置')
        if image_input.type not in ('tensor(float)', 'tensor(float16)'):
            raise ValueError(f'模型输入只支持 float32/float16，实际为 {image_input.type}')
        self.input_name = image_input.name
        self.input_dtype = np.float16 if image_input.type == 'tensor(float16)' else np.float32
        candidates = []
        for item in self.session.get_outputs():
            shape = item.shape
            if len(shape) != 3:
                continue
            if isinstance(shape[0], (int, np.integer)) and shape[0] != 1:
                continue
            if any(isinstance(value, (int, np.integer)) and value == 21 for value in shape[1:]):
                candidates.append(item)
            elif all(not isinstance(value, (int, np.integer)) for value in shape[1:]):
                candidates.append(item)
        if len(candidates) != 1:
            shapes = [item.shape for item in self.session.get_outputs()]
            raise ValueError(f'模型必须有一个可识别的 21 通道输出，实际输出：{shapes}')
        self.output_name = candidates[0].name

    def predict(self, image: np.ndarray, confidence: float | None = None,
                iou: float | None = None) -> list[ArmorAnnotation]:
        blob = preprocess_image(image, self.config).astype(self.input_dtype, copy=False)
        output = self.session.run([self.output_name], {self.input_name: blob})[0]
        annotations = decode_output(output, self.config, confidence=confidence, iou=iou)
        if self.config['output_coordinates'] == 'input':
            height, width = image.shape[:2]
            resized_width, resized_height, left, top = _resize_geometry(width, height, self.config)
            # Use each actual rounded resize scale. Do not clip partially outside targets.
            scale_x, scale_y = resized_width / width, resized_height / height
            for annotation in annotations:
                annotation.points = [[(x - left) / scale_x, (y - top) / scale_y]
                                     for x, y in annotation.points]
        return annotations
