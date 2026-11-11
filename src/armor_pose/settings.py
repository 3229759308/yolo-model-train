"""Repository paths and offline defaults shared by every model adapter."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
DATASETS_DIR = DATA_DIR / 'datasets'
PRETRAINED_DIR = PROJECT_ROOT / 'models/pretrained'
CONFIGS_DIR = PROJECT_ROOT / 'configs'
RUNS_DIR = PROJECT_ROOT / 'runs'
RESULTS_DIR = PROJECT_ROOT / 'results'
DEFAULT_TRAIN_CONFIG = CONFIGS_DIR / 'armor_pose/yolov8n/default.yaml'


def project_path(value) -> Path:
    """Resolve a configured path; relative paths start at the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


os.environ.setdefault('YOLO_CONFIG_DIR', str(PROJECT_ROOT / '.cache/ultralytics_armor_training'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_OFFLINE', 'true')
