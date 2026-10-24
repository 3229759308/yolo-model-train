"""Repository paths and offline defaults shared by every model adapter."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
DATASETS_DIR = PROJECT_ROOT / 'datasets'
WEIGHTS_DIR = PROJECT_ROOT / 'weights'
CONFIGS_DIR = PROJECT_ROOT / 'configs'
RUNS_DIR = PROJECT_ROOT / 'runs'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
DEFAULT_TRAIN_CONFIG = CONFIGS_DIR / 'armor_pose/yolov8n/default.yaml'

os.environ.setdefault('YOLO_CONFIG_DIR', str(PROJECT_ROOT / '.cache/ultralytics_armor_training'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_OFFLINE', 'true')
