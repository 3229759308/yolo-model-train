"""在独立环境中启动 ArmorLabel，不修改子模块源码。"""
import argparse
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manual', action='store_true', help='仅使用人工标注，不加载 SAM3 权重')
parser.add_argument('--smoke-test', action='store_true', help='打开窗口后自动关闭，检查启动与退出')
args, rest = parser.parse_known_args()
project = Path(__file__).resolve().parents[1] / 'bit-auto-label'
os.chdir(project)
sys.path.insert(0, str(project))
sys.argv = [str(project / 'main.py'), *rest]

# OpenCV 的 Qt 插件与 PySide6 不同；先导入 OpenCV，再清除它注入的插件路径。
import cv2  # noqa: E402
from PySide6.QtCore import QLibraryInfo, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath) + '/platforms'
if 'cv2' in os.environ.get('QT_QPA_FONTDIR', ''):
    os.environ.pop('QT_QPA_FONTDIR', None)

import main as armorlabel  # noqa: E402

if args.manual or not (project / 'models/sam3.pt').is_file():
    def manual_mode(self, checkpoint_path):
        self.model_status_changed.emit(False, '人工标注模式：未加载 SAM3 权重')
    armorlabel.SAMClient.load_model_async = manual_mode
    print('人工标注模式。启用 SAM3 自动辅助请先准备 models/sam3.pt。', flush=True)

if args.smoke_test:
    original_window = armorlabel.MainWindow

    class SmokeWindow(original_window):
        def __init__(self):
            super().__init__()
            QTimer.singleShot(2000, self.finish_check)

        def finish_check(self):
            print('GUI_SMOKE_TEST_OK：窗口构造、事件循环与关闭检查通过。', flush=True)
            self.close()
            QApplication.instance().quit()

    armorlabel.MainWindow = SmokeWindow

armorlabel.main()
