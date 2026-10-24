"""四点装甲板标注桌面工具；独立于 bit-auto-label 的自定义头适配器。"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal, QLibraryInfo
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QTableWidget,
    QTableWidgetItem, QToolBar, QVBoxLayout, QWidget,
)

from core import (
    ArmorAnnotation, AnnotationStore, OnnxDetector, default_config, load_config,
    save_config, scan_images, validate_config, export_custom, export_yolo,
)
from detector_cache import DetectorCache


COLORS = [QColor('#58a6ff'), QColor('#ff6478'), QColor('#f4f4f5'), QColor('#bd8bfa')]
RELATIVE_NAME_ROLE = int(Qt.ItemDataRole.UserRole) + 1


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法解码图片：{path}')
    return image


@dataclass(frozen=True)
class InferenceResult:
    annotations: list[ArmorAnnotation]
    width: int
    height: int


class Canvas(QWidget):
    selectionChanged = Signal(int)
    editStarted = Signal()
    edited = Signal()
    createRequested = Signal(list)
    nudgeRequested = Signal(float, float)
    statusChanged = Signal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 350)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.image = None
        self.annotations = []
        self.selected = -1
        self.selected_point = -1
        self.point_names = ['P0', 'P1', 'P2', 'P3']
        self.color_names = ['蓝', '红', '白', '紫']
        self.class_names = [f'class_{i}' for i in range(9)]
        self.scale = 1.0
        self.offset = QPointF()
        self.draft = []
        self.creating = False
        self.drag = None
        self.space = False
        self.allow_outside = True
        self.show_names = True
        self.auto_fit = True

    def set_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                            QImage.Format.Format_RGB888).copy()
        self.selected = self.selected_point = -1
        self.draft = []
        self.creating = False
        self.fit()

    def fit(self):
        self.auto_fit = True
        if self.image:
            self.scale = min((self.width()-48) / self.image.width(),
                             (self.height()-48) / self.image.height())
            self.offset = QPointF((self.width()-self.image.width()*self.scale)/2,
                                  (self.height()-self.image.height()*self.scale)/2)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit()

    def to_image(self, p):
        return (p-self.offset) / self.scale

    def to_view(self, p):
        return QPointF(p[0]*self.scale, p[1]*self.scale) + self.offset

    def bounded(self, p):
        if self.allow_outside or not self.image:
            return [p.x(), p.y()]
        return [max(0, min(self.image.width(), p.x())),
                max(0, min(self.image.height(), p.y()))]

    def start_create(self):
        if not self.image:
            return
        self.creating = True
        self.draft = []
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        self.statusChanged.emit('依次点击 4 个点；按模型点序放置，Esc 取消。')
        self.update()

    def cancel_create(self):
        self.creating = False
        self.draft = []
        self.unsetCursor()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor('#11151c'))
        if not self.image:
            p.setPen(QColor('#9aa8b8'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       '打开图片文件夹，开始四点标注\n\nN 新建 · 拖动点位 · 滚轮缩放 · 中键 / 空格拖动画布')
            return
        p.drawImage(QRectF(self.offset.x(), self.offset.y(), self.image.width()*self.scale,
                           self.image.height()*self.scale), self.image)
        for i, ann in enumerate(self.annotations):
            pts = [self.to_view(pt) for pt in ann.points]
            color = COLORS[ann.color_id]
            fill = QColor(color)
            fill.setAlpha(35 if i == self.selected else 12)
            p.setBrush(fill)
            p.setPen(QPen(color, 2.5 if i == self.selected else 1.5))
            p.drawPolygon(QPolygonF(pts))
            for k, pt in enumerate(pts):
                p.setBrush(QColor('#ffd166') if i == self.selected and k == self.selected_point else color)
                p.drawEllipse(pt, 5 if i == self.selected else 3, 5 if i == self.selected else 3)
                if i == self.selected:
                    p.setPen(QColor('#fff7df'))
                    p.drawText(pt + QPointF(8, -8), f'{k}: {self.point_names[k]}')
                    p.setPen(QPen(color, 2))
            if self.show_names:
                label = f'{i+1} · {self.color_names[ann.color_id]} / {self.class_names[ann.class_id]}'
                if ann.confidence is not None:
                    label += f' {ann.confidence:.2f}'
                top = min(pts, key=lambda pt: pt.y()) + QPointF(0, -9)
                rect = p.fontMetrics().boundingRect(label).adjusted(-4, -2, 4, 2)
                rect.moveBottomLeft(top.toPoint())
                p.fillRect(rect, QColor(15, 20, 28, 210))
                p.setPen(color)
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
        if self.draft:
            p.setPen(QPen(QColor('#ffd166'), 2, Qt.PenStyle.DashLine))
            pts = [self.to_view(pt) for pt in self.draft]
            p.drawPolyline(QPolygonF(pts))
            for i, pt in enumerate(pts):
                p.drawEllipse(pt, 5, 5)
                p.drawText(pt+QPointF(9, -9), self.point_names[i])
        p.setPen(QColor('#c0ccd9'))
        p.drawText(14, self.height()-14,
                   f'{self.image.width()} × {self.image.height()}  ·  {self.scale*100:.0f}%  ·  原图像素坐标')

    def mousePressEvent(self, event):
        if not self.image:
            return
        self.setFocus()
        pos = event.position()
        if event.button() == Qt.MouseButton.RightButton:
            if self.creating and self.draft:
                self.draft.pop()
                self.update()
            else:
                self.cancel_create()
            return
        if event.button() == Qt.MouseButton.MiddleButton or (self.space and event.button() == Qt.MouseButton.LeftButton):
            self.auto_fit = False
            self.drag = ('pan', pos, QPointF(self.offset))
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.creating:
            self.draft.append(self.bounded(self.to_image(pos)))
            if len(self.draft) == 4:
                points = copy.deepcopy(self.draft)
                self.cancel_create()
                self.createRequested.emit(points)
            else:
                self.statusChanged.emit(f'已放置 {len(self.draft)}/4 点，下一个：{self.point_names[len(self.draft)]}')
            self.update()
            return
        # 优先命中点柄，其次选中四边形内部；点序由模型定义，不自动重排。
        indices = list(reversed(range(len(self.annotations))))
        if self.selected in indices:
            indices.remove(self.selected)
            indices.insert(0, self.selected)
        for i in indices:
            ann = self.annotations[i]
            for k, pt in enumerate(ann.points):
                delta = self.to_view(pt)-pos
                if delta.x()**2 + delta.y()**2 < 100:
                    self.selected, self.selected_point = i, k
                    self.selectionChanged.emit(i)
                    self.editStarted.emit()
                    self.drag = ('point', i, k)
                    self.update()
                    return
        for i in indices:
            poly = QPolygonF([self.to_view(pt) for pt in self.annotations[i].points])
            if poly.containsPoint(pos, Qt.FillRule.OddEvenFill):
                self.selected, self.selected_point = i, -1
                self.selectionChanged.emit(i)
                self.editStarted.emit()
                self.drag = ('shape', i, self.to_image(pos), copy.deepcopy(self.annotations[i].points))
                self.update()
                return
        self.selected = self.selected_point = -1
        self.selectionChanged.emit(-1)
        self.update()

    def mouseMoveEvent(self, event):
        if not self.drag:
            return
        pos = event.position()
        if self.drag[0] == 'pan':
            self.offset = self.drag[2] + pos-self.drag[1]
        elif self.drag[0] == 'point':
            self.annotations[self.drag[1]].points[self.drag[2]] = self.bounded(self.to_image(pos))
            self.edited.emit()
        elif self.drag[0] == 'shape':
            delta = self.to_image(pos)-self.drag[2]
            if not self.allow_outside:
                original = self.drag[3]
                delta.setX(max(-min(x[0] for x in original),
                               min(self.image.width()-max(x[0] for x in original), delta.x())))
                delta.setY(max(-min(x[1] for x in original),
                               min(self.image.height()-max(x[1] for x in original), delta.y())))
            self.annotations[self.drag[1]].points = [[p[0]+delta.x(), p[1]+delta.y()] for p in self.drag[3]]
            self.edited.emit()
        self.update()

    def mouseReleaseEvent(self, event):
        self.drag = None
        if not self.creating:
            self.unsetCursor()

    def wheelEvent(self, event):
        if not self.image:
            return
        anchor = self.to_image(event.position())
        self.auto_fit = False
        self.scale = max(.015, min(40, self.scale * (1.18 if event.angleDelta().y() > 0 else 1/1.18)))
        self.offset = event.position()-anchor*self.scale
        self.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.space = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif key == Qt.Key.Key_Escape:
            self.cancel_create()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            dx = step if key == Qt.Key.Key_Right else -step if key == Qt.Key.Key_Left else 0
            dy = step if key == Qt.Key.Key_Down else -step if key == Qt.Key.Key_Up else 0
            self.nudgeRequested.emit(dx, dy)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space = False
            self.unsetCursor()
        else:
            super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self.space = False
        self.drag = None
        super().focusOutEvent(event)


class ConfigDialog(QDialog):
    def __init__(self, config, parent):
        super().__init__(parent)
        self.setWindowTitle('检测头配置 / 类别与点序')
        self.resize(710, 700)
        layout = QVBoxLayout(self)
        label = QLabel('颜色 ID 固定 0～3，类别 ID 固定 0～8；名称只改变显示。\n'
                       'point_names 按模型输出的 P0～P3 顺序填写，不会自动重排已有坐标。\n'
                       '预处理请与模型导出时保持一致；输出坐标固定为原图像素。')
        label.setWordWrap(True)
        layout.addWidget(label)
        self.editor = QPlainTextEdit(json.dumps(config, ensure_ascii=False, indent=2))
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.editor)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.check)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def check(self):
        try:
            self.config = validate_config(json.loads(self.editor.toPlainText()))
        except Exception as exc:
            QMessageBox.warning(self, '配置无效', str(exc))
            return
        self.accept()


class InferenceWorker(QThread):
    ready = Signal(str)
    predicted = Signal(str, object)
    failed = Signal(str, str)
    progress = Signal(int, int)

    def __init__(self, model_path, config, paths, detector_cache=None):
        super().__init__()
        self.model_path = model_path
        self.config = copy.deepcopy(config)
        self.paths = list(paths)
        self.detector_cache = detector_cache

    def run(self):
        try:
            detector = (self.detector_cache.get(self.model_path, self.config, factory=OnnxDetector)
                        if self.detector_cache is not None else OnnxDetector(self.model_path, self.config))
            self.ready.emit(getattr(detector, 'backend_summary', '未报告'))
        except Exception as exc:
            self.failed.emit('', str(exc))
            return
        for i, path in enumerate(self.paths):
            if self.isInterruptionRequested():
                break
            try:
                image = read_image(path)
                annotations = detector.predict(image)
                self.predicted.emit(str(path), InferenceResult(annotations, image.shape[1], image.shape[0]))
            except Exception as exc:
                self.failed.emit(str(path), str(exc))
                break
            self.progress.emit(i+1, len(self.paths))


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle('Armor Labeler · 四点装甲板标注')
        self.resize(1520, 930)
        self.args = args
        self.config_path = Path(args.config).expanduser().resolve()
        self.config = load_config(self.config_path) if self.config_path.exists() else default_config()
        self.store = None
        self.root = None
        self.paths = []
        self.current_path = None
        self.image = None
        self.annotations = []
        self.reviewed = False
        self.dirty = False
        self.history = []
        self.redo_history = []
        self.loading = False
        self.worker = None
        self.detector_cache = DetectorCache()
        self.inference_error = False
        configured_model = self.config.get('model_path', '')
        self.model_path = (str(Path(args.model).expanduser().resolve()) if args.model else
                           self.resolve_model_path(configured_model))
        self.file_states = {}
        self.file_items = {}
        self.file_counts = Counter()
        self.visible_count = 0
        self.autosave = QTimer(self)
        self.autosave.setSingleShot(True)
        self.autosave.setInterval(900)
        self.autosave.timeout.connect(self.save_current)
        self.settings_timer = QTimer(self)
        self.settings_timer.setSingleShot(True)
        self.settings_timer.setInterval(500)
        self.settings_timer.timeout.connect(self.persist_settings)
        self.build_ui()
        self.update_config_ui()
        self.update_status('打开图片文件夹；或使用 --images 指定输入目录。')
        if args.images:
            self.open_folder(Path(args.images), Path(args.labels) if args.labels else None)

    def button(self, text, slot):
        button = QPushButton(text)
        button.clicked.connect(slot)
        return button

    def resolve_model_path(self, value):
        if not value:
            return ''
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else self.config_path.parent/path).resolve())

    def action(self, toolbar, text, shortcut, slot):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        toolbar.addAction(action)
        return action

    def build_ui(self):
        toolbar = QToolBar('标注', self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.action(toolbar, '打开文件夹', 'Ctrl+O', self.choose_folder)
        self.action(toolbar, '保存', 'Ctrl+S', lambda: self.save_current(force=True))
        toolbar.addSeparator()
        self.action(toolbar, '上一张 [A]', 'A', lambda: self.navigate(-1))
        self.action(toolbar, '下一张 [D]', 'D', lambda: self.navigate(1))
        self.action(toolbar, '新建四点 [N]', 'N', lambda: self.canvas.start_create())
        self.action(toolbar, '适应画布 [F]', 'F', lambda: self.canvas.fit())
        self.action(toolbar, '撤销', 'Ctrl+Z', self.undo)
        self.action(toolbar, '重做', 'Ctrl+Shift+Z', self.redo)
        self.action(toolbar, '删除', 'Delete', self.delete_selected)
        toolbar.addSeparator()
        self.action(toolbar, '当前图推理 [I]', 'I', self.infer_current)
        self.action(toolbar, '配置', None, self.edit_config)
        self.action(toolbar, '导出', None, self.export_labels)
        self.action(toolbar, '帮助', 'F1', self.help)
        splitter = QSplitter()
        self.setCentralWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.folder_label = QLabel('未打开图片文件夹')
        self.folder_label.setWordWrap(True)
        left_layout.addWidget(self.folder_label)
        self.search = QLineEdit()
        self.search.setPlaceholderText('筛选文件名…')
        self.search.textChanged.connect(self.filter_files)
        left_layout.addWidget(self.search)
        self.filter = QComboBox()
        self.filter.addItems(['全部图片', '未保存标注', '待复核', '已复核'])
        self.filter.currentIndexChanged.connect(self.filter_files)
        left_layout.addWidget(self.filter)
        self.files = QListWidget()
        # 文件条目均为单行；固定高度避免每次状态文字变化都重算整个目录的布局。
        self.files.setUniformItemSizes(True)
        self.files.currentItemChanged.connect(self.file_selected)
        left_layout.addWidget(self.files)
        self.stats = QLabel('0 张图片')
        left_layout.addWidget(self.stats)
        self.label_dir = QLabel('标签目录：打开图片后选择')
        self.label_dir.setWordWrap(True)
        left_layout.addWidget(self.label_dir)
        left_layout.addWidget(self.button('更换标签目录…', self.choose_labels))
        splitter.addWidget(left)

        self.canvas = Canvas()
        self.canvas.selectionChanged.connect(self.select_annotation)
        self.canvas.editStarted.connect(self.push_history)
        self.canvas.edited.connect(self.canvas_edited)
        self.canvas.createRequested.connect(self.create_annotation)
        self.canvas.nudgeRequested.connect(self.nudge)
        self.canvas.statusChanged.connect(self.update_status)
        splitter.addWidget(self.canvas)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(320)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel('目标 / 独立颜色与类别'))
        self.targets = QListWidget()
        self.targets.setMinimumHeight(120)
        self.targets.setMaximumHeight(240)
        self.targets.currentRowChanged.connect(self.select_annotation)
        right_layout.addWidget(self.targets)
        form = QFormLayout()
        self.color = QComboBox()
        self.color.currentIndexChanged.connect(self.attributes_changed)
        form.addRow('颜色分支', self.color)
        self.category = QComboBox()
        self.category.currentIndexChanged.connect(self.attributes_changed)
        form.addRow('类别分支', self.category)
        right_layout.addLayout(form)
        self.details = QLabel('无选中目标；下方类别作为新建默认值。')
        self.details.setWordWrap(True)
        right_layout.addWidget(self.details)
        self.coordinates = QTableWidget(4, 2)
        self.coordinates.setHorizontalHeaderLabels(['X / px', 'Y / px'])
        self.coordinates.setMaximumHeight(174)
        self.coordinates.horizontalHeader().setStretchLastSection(True)
        self.spins = []
        for i in range(4):
            row = []
            for j in range(2):
                spin = QDoubleSpinBox()
                spin.setRange(-1e7, 1e7)
                spin.setDecimals(3)
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(lambda value, k=i, axis=j: self.coordinate_changed(k, axis, value))
                self.coordinates.setCellWidget(i, j, spin)
                row.append(spin)
            self.spins.append(row)
        right_layout.addWidget(self.coordinates)
        row = QHBoxLayout()
        row.addWidget(self.button('起点轮换', self.rotate_points))
        row.addWidget(self.button('反转点序', self.reverse_points))
        right_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.button('复制目标', self.duplicate_selected))
        row.addWidget(self.button('删除目标', self.delete_selected))
        right_layout.addLayout(row)
        self.outside = QCheckBox('允许点位超出图片（截断目标）')
        self.outside.setChecked(True)
        self.outside.toggled.connect(lambda value: setattr(self.canvas, 'allow_outside', value))
        right_layout.addWidget(self.outside)
        self.names = QCheckBox('显示目标名称')
        self.names.setChecked(True)
        self.names.toggled.connect(self.toggle_names)
        right_layout.addWidget(self.names)
        self.review = QCheckBox('当前图片已复核（含确认无目标）')
        self.review.toggled.connect(self.review_changed)
        right_layout.addWidget(self.review)
        right_layout.addSpacing(14)
        right_layout.addWidget(QLabel('自定义 ONNX 检测头'))
        self.backend_label = QLabel('推理后端：尚未加载模型')
        self.backend_label.setWordWrap(True)
        right_layout.addWidget(self.backend_label)
        self.model_label = QLabel(self.model_path or '未选择模型；人工标注可直接使用')
        self.model_label.setWordWrap(True)
        right_layout.addWidget(self.model_label)
        self.model_button = self.button('选择 .onnx 模型…', self.choose_model)
        right_layout.addWidget(self.model_button)
        infer_form = QFormLayout()
        self.preprocessing = QComboBox()
        self.preprocessing.addItem('直接缩放至 640×480', 'stretch')
        self.preprocessing.addItem('Letterbox 等比例填充', 'letterbox')
        infer_form.addRow('输入预处理', self.preprocessing)
        self.output_space = QComboBox()
        self.output_space.addItem('原图像素（直接使用）', 'original')
        self.output_space.addItem('网络输入像素（还原原图）', 'input')
        infer_form.addRow('模型输出坐标', self.output_space)
        self.preprocessing.currentIndexChanged.connect(self.preprocessing_changed)
        self.output_space.currentIndexChanged.connect(self.preprocessing_changed)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0, 1)
        self.confidence.setSingleStep(.05)
        infer_form.addRow('置信度阈值', self.confidence)
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0, 1)
        self.iou.setSingleStep(.05)
        infer_form.addRow('NMS IoU', self.iou)
        right_layout.addLayout(infer_form)
        self.confidence.valueChanged.connect(self.threshold_changed)
        self.iou.valueChanged.connect(self.threshold_changed)
        self.infer_button = self.button('当前图推理（追加）', self.infer_current)
        right_layout.addWidget(self.infer_button)
        self.batch_button = self.button('批量推理（仅未保存图片）', self.infer_batch)
        right_layout.addWidget(self.batch_button)
        self.stop_button = self.button('停止批量', self.stop_inference)
        self.stop_button.setEnabled(False)
        right_layout.addWidget(self.stop_button)
        right_layout.addStretch()
        self.contract_hint = QLabel()
        self.contract_hint.setWordWrap(True)
        self.contract_hint.setStyleSheet('color: #a0aebe; padding: 8px; background: #18212e; border-radius: 5px;')
        right_layout.addWidget(self.contract_hint)
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setSizes([250, 920, 350])
        splitter.setStretchFactor(1, 1)

    def update_status(self, text):
        self.statusBar().showMessage(text)

    def error(self, title, exc):
        self.update_status(f'{title}：{exc}')
        QMessageBox.warning(self, title, str(exc))

    def update_config_ui(self):
        self.loading = True
        self.color.clear()
        self.color.addItems([f'{i} · {name}' for i, name in enumerate(self.config['color_names'])])
        self.category.clear()
        self.category.addItems([f'{i} · {name}' for i, name in enumerate(self.config['class_names'])])
        self.coordinates.setVerticalHeaderLabels([f'{i} {s}' for i, s in enumerate(self.config['point_names'])])
        self.canvas.color_names = self.config['color_names']
        self.canvas.class_names = self.config['class_names']
        self.canvas.point_names = self.config['point_names']
        self.confidence.setValue(self.config['confidence'])
        self.iou.setValue(self.config['iou'])
        self.preprocessing.setCurrentIndex(self.preprocessing.findData(self.config['preprocess']))
        self.output_space.setCurrentIndex(self.output_space.findData(self.config['output_coordinates']))
        self.update_contract_hint()
        self.loading = False
        self.refresh_annotations()

    def choose_folder(self):
        if self.worker and self.worker.isRunning():
            self.update_status('请先停止批量推理，再切换文件夹。')
            return
        directory = QFileDialog.getExistingDirectory(self, '选择图片目录', str(self.root or Path('/workspace')))
        if directory:
            self.open_folder(Path(directory))

    def open_folder(self, root, labels=None):
        if not self.save_current():
            return
        try:
            root = root.expanduser().resolve()
            paths = scan_images(root)
            if not paths:
                raise ValueError('目录内没有可读取的图片文件。')
            label_root = labels.expanduser().resolve() if labels else root.parent / (root.name+'_labels')
            store = AnnotationStore(root, label_root)
        except Exception as exc:
            self.error('打开目录失败', exc)
            return
        self.root, self.paths, self.store = root, paths, store
        self.current_path = None
        self.image = None
        self.annotations = []
        self.dirty = False
        self.reviewed = False
        self.history, self.redo_history = [], []
        self.setWindowTitle(f'{root.name} · Armor Labeler')
        self.canvas.image = None
        self.canvas.annotations = []
        self.canvas.selected = self.canvas.selected_point = -1
        self.canvas.cancel_create()
        self.canvas.update()
        self.refresh_annotations(-1)
        self.folder_label.setText(str(root))
        self.label_dir.setText(f'标签目录：{label_root}')
        self.files.blockSignals(True)
        self.files.clear()
        self.file_states = {}
        self.file_items = {}
        self.file_counts = Counter()
        self.visible_count = 0
        for path in paths:
            state = self.state_from_disk(path)
            self.file_states[str(path)] = state
            self.file_counts[state] += 1
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setData(RELATIVE_NAME_ROLE, str(path.relative_to(root)))
            self.files.addItem(item)
            self.file_items[str(path)] = item
            self.update_file_item(item)
        self.files.blockSignals(False)
        self.search.clear()
        self.filter.setCurrentIndex(0)
        self.filter_files()
        self.files.setCurrentRow(0)

    def state_from_disk(self, path):
        try:
            data = self.store.load(path)
            return 'reviewed' if data and data.get('reviewed') else 'saved' if data is not None else 'new'
        except Exception:
            return 'error'

    def update_file_item(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        state = self.file_states.get(path, 'new')
        badge = {'new': '○', 'saved': '●', 'reviewed': '✓', 'error': '!'}[state]
        relative = item.data(RELATIVE_NAME_ROLE)
        if relative is None:
            relative = str(Path(path).relative_to(self.root))
            item.setData(RELATIVE_NAME_ROLE, relative)
        item.setText(f'{badge} {relative}')
        item.setToolTip(path)
        item.setForeground(QColor('#62cfac' if state == 'reviewed' else '#ffad66' if state == 'error' else '#dce3ed'))

    def _file_matches(self, item, text=None, mode=None):
        text = self.search.text().casefold() if text is None else text
        mode = self.filter.currentIndex() if mode is None else mode
        path = item.data(Qt.ItemDataRole.UserRole)
        state = self.file_states.get(path, 'new')
        name = item.data(RELATIVE_NAME_ROLE)
        if name is None:
            name = str(Path(path).relative_to(self.root)) if self.root else path
        return (text in name.casefold() and
                (mode == 0 or (mode == 1 and state == 'new') or
                 (mode == 2 and state == 'saved') or (mode == 3 and state == 'reviewed')))

    def _refresh_file_stats(self):
        self.stats.setText(f'{self.visible_count} / {len(self.paths)} 张 · 已复核 {self.file_counts["reviewed"]}')

    def _set_file_state(self, path_string, state):
        previous = self.file_states.get(path_string)
        if previous != state:
            if previous is not None:
                self.file_counts[previous] -= 1
            self.file_counts[state] += 1
            self.file_states[path_string] = state
        item = self.file_items.get(path_string)
        if item is not None:
            self.update_file_item(item)
            was_visible = not item.isHidden()
            visible = self._file_matches(item)
            if visible != was_visible:
                item.setHidden(not visible)
                self.visible_count += int(visible) - int(was_visible)
        self._refresh_file_stats()

    def filter_files(self):
        text = self.search.text().casefold()
        mode = self.filter.currentIndex()
        self.visible_count = 0
        for i in range(self.files.count()):
            item = self.files.item(i)
            visible = self._file_matches(item, text, mode)
            if item.isHidden() == visible:
                item.setHidden(not visible)
            self.visible_count += visible
        self._refresh_file_stats()

    def file_selected(self, item, previous):
        if self.loading or item is None:
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        if path == self.current_path:
            return
        if not self.load_image(path):
            self.files.blockSignals(True)
            self.files.setCurrentItem(previous)
            self.files.blockSignals(False)

    def load_image(self, path):
        if not self.save_current():
            return False
        try:
            image = read_image(path)
            data = self.store.load(path)
            if data and (data['image']['width'] != image.shape[1] or data['image']['height'] != image.shape[0]):
                raise ValueError('标注记录的图片尺寸与当前文件不同；请检查图片或选择新的标签目录。')
            annotations = [ArmorAnnotation.from_dict(a) for a in data['annotations']] if data else []
        except Exception as exc:
            self.error('读取失败，未覆盖标签', exc)
            return False
        self.current_path, self.image, self.annotations = path, image, annotations
        self.reviewed = bool(data and data.get('reviewed'))
        self.dirty = False
        self.history, self.redo_history = [], []
        self.canvas.set_image(image)
        self.refresh_annotations()
        self.setWindowTitle(f'{path.name} · Armor Labeler')
        self.update_status(f'{path.relative_to(self.root)} · {len(annotations)} 个目标 · 编辑后自动保存')
        return True

    def navigate(self, direction):
        if not self.paths:
            return
        current = self.files.currentRow()
        i = current+direction
        while 0 <= i < self.files.count():
            if not self.files.item(i).isHidden():
                self.files.setCurrentRow(i)
                return
            i += direction

    def choose_labels(self):
        if not self.root:
            return
        if self.worker and self.worker.isRunning():
            self.update_status('请先停止批量推理，再更换标签目录。')
            return
        if not self.save_current():
            return
        directory = QFileDialog.getExistingDirectory(self, '选择标签目录', str(self.root.parent))
        if directory:
            self.open_folder(self.root, Path(directory))

    def refresh_annotations(self, selected=None):
        if selected is None:
            selected = self.canvas.selected
        self.loading = True
        self.canvas.annotations = self.annotations
        self.canvas.selected = selected if 0 <= selected < len(self.annotations) else -1
        self.targets.clear()
        for i, ann in enumerate(self.annotations):
            item = QListWidgetItem(f'{i+1} · {self.config["color_names"][ann.color_id]} / {self.config["class_names"][ann.class_id]}')
            item.setForeground(COLORS[ann.color_id])
            self.targets.addItem(item)
        self.targets.setCurrentRow(self.canvas.selected)
        self.review.setChecked(self.reviewed)
        self.loading = False
        self.select_annotation(self.canvas.selected)
        self.canvas.update()

    def selected_annotation(self):
        i = self.canvas.selected
        return self.annotations[i] if 0 <= i < len(self.annotations) else None

    def select_annotation(self, index):
        if self.loading:
            return
        self.loading = True
        if index != self.canvas.selected:
            self.canvas.selected_point = -1
        self.canvas.selected = index
        self.targets.setCurrentRow(index)
        ann = self.selected_annotation()
        if ann:
            self.color.setCurrentIndex(ann.color_id)
            self.category.setCurrentIndex(ann.class_id)
            score = f' · score {ann.confidence:.4f}' if ann.confidence is not None else ''
            self.details.setText(f'{ann.source}{score} · {ann.id[:8]}')
        else:
            self.details.setText('无选中目标；类别作为新建默认值。')
        for k in range(4):
            for j in range(2):
                self.spins[k][j].setEnabled(ann is not None)
                self.spins[k][j].setValue(ann.points[k][j] if ann else 0)
        self.loading = False
        self.canvas.update()

    def snapshot(self):
        return ([a.to_dict() for a in self.annotations], self.reviewed, self.canvas.selected)

    def push_history(self):
        if self.current_path is None:
            return
        snapshot = self.snapshot()
        if not self.history or self.history[-1] != snapshot:
            self.history.append(copy.deepcopy(snapshot))
            self.history = self.history[-100:]
        self.redo_history.clear()

    def restore(self, snapshot):
        annotations, reviewed, selected = snapshot
        self.annotations = [ArmorAnnotation.from_dict(a) for a in annotations]
        self.reviewed = reviewed
        self.mark_dirty(reset_review=False)
        self.refresh_annotations(selected)

    def undo(self):
        if self.history:
            self.redo_history.append(copy.deepcopy(self.snapshot()))
            self.restore(self.history.pop())

    def redo(self):
        if self.redo_history:
            self.history.append(copy.deepcopy(self.snapshot()))
            self.restore(self.redo_history.pop())

    def mark_dirty(self, reset_review=True):
        self.dirty = True
        if reset_review:
            self.reviewed = False
            self.review.blockSignals(True)
            self.review.setChecked(False)
            self.review.blockSignals(False)
        self.autosave.start()
        self.setWindowTitle(f'* {self.current_path.name} · Armor Labeler')

    def canvas_edited(self):
        ann = self.selected_annotation()
        if ann:
            ann.source = 'manual'
            ann.confidence = None
        self.mark_dirty()
        self.select_annotation(self.canvas.selected)

    def create_annotation(self, points):
        if self.current_path is None:
            return
        self.push_history()
        ann = ArmorAnnotation(color_id=self.color.currentIndex(), class_id=self.category.currentIndex(), points=points)
        self.annotations.append(ann)
        self.mark_dirty()
        self.refresh_annotations(len(self.annotations)-1)
        self.update_status('已创建目标；可拖动四点或在右侧输入坐标。N 继续新建。')

    def attributes_changed(self):
        ann = self.selected_annotation()
        if self.loading or ann is None:
            return
        if (ann.color_id, ann.class_id) == (self.color.currentIndex(), self.category.currentIndex()):
            return
        self.push_history()
        ann.color_id, ann.class_id = self.color.currentIndex(), self.category.currentIndex()
        ann.source = 'manual'
        ann.confidence = None
        self.mark_dirty()
        self.refresh_annotations()

    def coordinate_changed(self, k, axis, value):
        ann = self.selected_annotation()
        if self.loading or ann is None or ann.points[k][axis] == value:
            return
        if not self.canvas.allow_outside:
            value = max(0, min(self.image.shape[1 if axis == 0 else 0], value))
        self.push_history()
        ann.points[k][axis] = value
        ann.source = 'manual'
        ann.confidence = None
        self.mark_dirty()
        self.select_annotation(self.canvas.selected)

    def mutate_selected(self, fn):
        ann = self.selected_annotation()
        if ann:
            self.push_history()
            fn(ann)
            ann.source = 'manual'
            ann.confidence = None
            self.mark_dirty()
            self.refresh_annotations()

    def rotate_points(self):
        self.mutate_selected(lambda ann: setattr(ann, 'points', ann.points[1:] + ann.points[:1]))

    def reverse_points(self):
        self.mutate_selected(lambda ann: setattr(ann, 'points', [ann.points[0], *reversed(ann.points[1:])]))

    def nudge(self, dx, dy):
        ann = self.selected_annotation()
        if not ann:
            return
        self.push_history()
        indices = [self.canvas.selected_point] if self.canvas.selected_point >= 0 else range(4)
        for k in indices:
            ann.points[k] = self.canvas.bounded(QPointF(ann.points[k][0]+dx, ann.points[k][1]+dy))
        ann.source = 'manual'
        ann.confidence = None
        self.mark_dirty()
        self.refresh_annotations()

    def duplicate_selected(self):
        ann = self.selected_annotation()
        if ann:
            self.push_history()
            self.annotations.append(ArmorAnnotation(color_id=ann.color_id, class_id=ann.class_id,
                points=[self.canvas.bounded(QPointF(p[0]+8, p[1]+8)) for p in ann.points]))
            self.mark_dirty()
            self.refresh_annotations(len(self.annotations)-1)

    def delete_selected(self):
        index = self.canvas.selected
        if self.selected_annotation():
            self.push_history()
            self.annotations.pop(index)
            self.mark_dirty()
            self.refresh_annotations(min(index, len(self.annotations)-1))

    def review_changed(self, value):
        if self.loading or self.current_path is None:
            return
        self.push_history()
        self.reviewed = value
        self.mark_dirty(reset_review=False)

    def save_current(self, force=False):
        self.autosave.stop()
        if self.current_path is None or (not self.dirty and not force):
            return True
        try:
            path = self.store.save(self.current_path, self.image.shape[1], self.image.shape[0],
                                   self.annotations, reviewed=self.reviewed)
        except Exception as exc:
            self.error('保存失败，当前编辑仍保留在内存', exc)
            return False
        self.dirty = False
        self._set_file_state(str(self.current_path), 'reviewed' if self.reviewed else 'saved')
        self.setWindowTitle(f'{self.current_path.name} · Armor Labeler')
        self.update_status(f'已保存 {len(self.annotations)} 个目标 → {path}')
        return True

    def toggle_names(self, value):
        self.canvas.show_names = value
        self.canvas.update()

    def threshold_changed(self):
        if not self.loading:
            self.config['confidence'] = self.confidence.value()
            self.config['iou'] = self.iou.value()
            self.settings_timer.start()

    def preprocessing_changed(self):
        if not self.loading:
            self.config['preprocess'] = self.preprocessing.currentData()
            self.config['output_coordinates'] = self.output_space.currentData()
            self.update_contract_hint()
            self.settings_timer.start()

    def update_contract_hint(self):
        coord_hint = ('模型输出为网络坐标，按预处理还原原图' if self.config['output_coordinates'] == 'input'
                      else '模型输出为原图坐标，直接使用')
        self.contract_hint.setText(f'输入 1×3×{self.config["input_height"]}×{self.config["input_width"]}\n'
                                   '输出 1×21×6300\n颜色 0:4 / 类别 4:13 / 点位 13:21\n'
                                   f'{coord_hint}\n所有保存标注均为原图像素')

    def persist_settings(self):
        try:
            save_config(self.config, self.config_path)
        except Exception as exc:
            self.error('推理配置保存失败', exc)

    def edit_config(self):
        if self.worker and self.worker.isRunning():
            self.update_status('请先停止推理，再更改配置。')
            return
        dialog = ConfigDialog(self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                save_config(dialog.config, self.config_path)
            except Exception as exc:
                self.error('配置保存失败', exc)
                return
            self.config = dialog.config
            self.backend_label.setText('推理后端：下次推理使用新配置')
            self.model_path = self.resolve_model_path(self.config.get('model_path', '')) or self.model_path
            self.model_label.setText(self.model_path or '未选择模型；人工标注可直接使用')
            self.update_config_ui()
            self.update_status(f'配置已保存到 {self.config_path}')

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择自定义检测头 ONNX', '/workspace', 'ONNX (*.onnx)')
        if path:
            self.model_path = path
            self.backend_label.setText('推理后端：尚未加载新模型')
            self.model_label.setText(path)
            self.config['model_path'] = os.path.relpath(path, self.config_path.parent)
            self.settings_timer.start()

    def infer_current(self):
        if self.current_path:
            self.start_inference([self.current_path], batch=False)

    def infer_batch(self):
        if not self.store or not self.save_current():
            return
        # 任何已有标签（包括空图、坏标签）均跳过；不覆盖人工复核结果。
        paths = [p for p in self.paths if self.state_from_disk(p) == 'new']
        if not paths:
            self.update_status('所有图片已有标签；批量推理无需处理。')
            return
        self.start_inference(paths, batch=True)

    def start_inference(self, paths, batch):
        if self.worker and self.worker.isRunning():
            return
        if not self.model_path:
            self.choose_model()
            if not self.model_path:
                return
        if not self.save_current():
            return
        self.batch_mode = batch
        self.prediction_store = self.store
        self.inference_error = False
        self.worker = InferenceWorker(self.model_path, self.config, paths, detector_cache=self.detector_cache)
        self.worker.ready.connect(self.inference_ready)
        self.worker.predicted.connect(self.accept_prediction)
        self.worker.failed.connect(self.inference_failed)
        self.worker.progress.connect(lambda i, total: self.update_status(f'推理进度 {i}/{total}；自动结果待复核'))
        self.worker.finished.connect(self.inference_finished)
        self.infer_button.setEnabled(False)
        self.batch_button.setEnabled(False)
        self.model_button.setEnabled(False)
        self.preprocessing.setEnabled(False)
        self.output_space.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.backend_label.setText('推理后端：正在加载…')
        self.update_status('正在加载 ONNX 模型…')
        self.worker.start()

    def inference_ready(self, summary):
        self.backend_label.setText(f'推理后端：{summary}')
        self.backend_label.setToolTip(summary)
        self.update_status(f'ONNX 模型已加载 · {summary} · 正在推理…')

    def accept_prediction(self, path_string, predictions, *, store=None, batch_mode=None):
        store = store if store is not None else getattr(self, 'prediction_store', self.store)
        batch_mode = self.batch_mode if batch_mode is None else batch_mode
        if self.store is not store:
            return
        path = Path(path_string)
        try:
            # 提交正在输入的数值，避免异步刷新抹去尚未失焦的坐标编辑。
            focused = QApplication.focusWidget()
            if isinstance(focused, QDoubleSpinBox):
                focused.interpretText()
            if path == self.current_path and self.canvas.drag and self.canvas.drag[0] != 'pan':
                # 四点正在拖动时推迟合入，保留本次鼠标手势及它的撤销边界。
                QTimer.singleShot(100, lambda: self.accept_prediction(
                    path_string, predictions, store=store, batch_mode=batch_mode))
                return
            image_size = None
            if isinstance(predictions, InferenceResult):
                image_size = (predictions.width, predictions.height)
                predictions = predictions.annotations
            if batch_mode:
                # 推理期间用户可编辑；提交前再次检查，避免覆盖新保存的人工标签。
                if self.store.load(path) is not None:
                    return
                if path == self.current_path and self.dirty:
                    return
                if image_size is None:
                    image = self.image if path == self.current_path else read_image(path)
                    image_size = (image.shape[1], image.shape[0])
                self.store.save(path, *image_size, predictions, reviewed=False)
                self._set_file_state(path_string, 'saved')
                if path == self.current_path:
                    self.push_history()
                    self.annotations = predictions
                    self.reviewed = False
                    self.refresh_annotations(-1)
            elif path == self.current_path:
                self.push_history()
                self.annotations.extend(predictions)
                self.mark_dirty()
                # 新目标绘制中的颜色/类别作为默认值，保持到四点完成。
                self.refresh_annotations(-1 if self.canvas.creating else len(self.annotations)-1)
            else:
                # 切走后仍以追加语义保存，保留目标图上的既有内容。
                data = self.store.load(path)
                old = [ArmorAnnotation.from_dict(a) for a in data['annotations']] if data else []
                if image_size is None:
                    image = read_image(path)
                    image_size = (image.shape[1], image.shape[0])
                self.store.save(path, *image_size, old+predictions, reviewed=False)
                self._set_file_state(path_string, 'saved')
        except Exception as exc:
            self.inference_failed(path_string, str(exc))
            self.stop_inference()

    def inference_failed(self, path, reason):
        self.inference_error = True
        self.backend_label.setText('推理后端：加载或推理失败，请查看错误信息')
        self.error('推理失败', f'{path}\n{reason}' if path else reason)

    def stop_inference(self):
        if self.worker:
            self.worker.requestInterruption()
            self.update_status('将在当前图片推理结束后停止，已保存结果保留。')

    def inference_finished(self):
        self.infer_button.setEnabled(True)
        self.batch_button.setEnabled(True)
        self.model_button.setEnabled(True)
        self.preprocessing.setEnabled(True)
        self.output_space.setEnabled(True)
        self.stop_button.setEnabled(False)
        if not self.inference_error:
            self.update_status('推理已结束；结果可自由编辑，请逐图复核。')

    def export_labels(self):
        if not self.store or not self.save_current():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('导出标注')
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('导出所有已保存图片，保持子目录和图片扩展名。\n自动建立新目录，避免覆盖历史导出。'))
        mode = QComboBox()
        mode.addItems(['自定义头 TXT：color class x0 y0 … x3 y3（原图像素）',
                       '标准 YOLO Pose：36 类（color×9+class），四点 v=2'])
        layout.addWidget(mode)
        reviewed_only = QCheckBox('仅导出已复核图片')
        reviewed_only.setChecked(True)
        layout.addWidget(reviewed_only)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        output = QFileDialog.getExistingDirectory(self, '选择导出目标的父目录', str(self.root.parent))
        if not output:
            return
        export_dir = None
        try:
            # 全部验证、生成后再发布目录；导出失败不会留下看似完整的数据集。
            from datetime import datetime
            import tempfile
            import shutil
            output = Path(output)
            export_dir = Path(tempfile.mkdtemp(prefix='.armor-export-', dir=output))
            count = 0
            manifest = []
            for path in self.paths:
                data = self.store.load(path)
                if data is None or (reviewed_only.isChecked() and not data.get('reviewed')):
                    continue
                annotations = [ArmorAnnotation.from_dict(a) for a in data['annotations']]
                rel = path.relative_to(self.root)
                target = export_dir / 'labels' / rel.with_name(rel.name+'.txt')
                target.parent.mkdir(parents=True, exist_ok=True)
                if mode.currentIndex() == 0:
                    export_custom(target, annotations)
                else:
                    export_yolo(target, annotations, data['image']['width'], data['image']['height'])
                manifest.append({'image': str(rel), 'label': str(target.relative_to(export_dir)),
                                 'reviewed': data.get('reviewed', False)})
                count += 1
            if not count:
                shutil.rmtree(export_dir)
                self.update_status('没有符合筛选条件的已保存图片。')
                return
            metadata = {'format': 'custom-head-pixels' if mode.currentIndex() == 0 else 'yolo-pose-36',
                        'image_root': str(self.root), 'point_names': self.config['point_names'],
                        'color_names': self.config['color_names'], 'class_names': self.config['class_names'],
                        'images': manifest}
            (export_dir/'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            if mode.currentIndex() == 1:
                names = [f'{c}_{k}' for c in self.config['color_names'] for k in self.config['class_names']]
                (export_dir/'classes.txt').write_text('\n'.join(names)+'\n', encoding='utf-8')
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            published = output / f'armor_export_{stamp}'
            export_dir.rename(published)
            self.update_status(f'已导出 {count} 张图片的标签 → {published}')
            QMessageBox.information(self, '导出完成', f'{count} 张图片\n{published}\n\n图片不复制，manifest.json 记录源图路径与标签对应关系。')
        except Exception as exc:
            if export_dir and export_dir.exists() and export_dir.name.startswith('.armor-export-'):
                import shutil
                shutil.rmtree(export_dir)
            self.error('导出失败', exc)

    def help(self):
        QMessageBox.information(self, '使用方法',
            'N：依点序点击四点；右键撤回一点，Esc 取消。\n'
            '左键拖动点柄；拖动四边形内部移动整个目标。\n'
            '滚轮缩放；中键或空格+左键平移；F 适应窗口。\n'
            '方向键微调选中点 / 整体，Shift 加速 10 px。\n'
            'A / D 切换图片；Ctrl+S 保存；Ctrl+Z 撤销；Ctrl+Shift+Z 重做。\n'
            'Delete 删除目标；I 当前图推理并追加目标。\n\n'
            '编辑后自动保存，切图与关闭前也会保存。空图片请勾选“已复核”。\n'
            '标签为独立颜色、类别和四个原图像素点，无须先画 bbox。\n'
            '配置中可改类别名称、点序名称、预处理和置信度组合方式。\n'
            '批量只处理没有标签的图片，自动结果始终待复核。')

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.stop_inference()
            self.update_status('正在停止推理，完成后可关闭窗口。')
            event.ignore()
            return
        if self.save_current():
            self.detector_cache.clear()
            if self.settings_timer.isActive():
                self.settings_timer.stop()
                self.persist_settings()
            event.accept()
        else:
            event.ignore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', help='图片目录（递归）')
    parser.add_argument('--labels', help='标注 JSON 目录，默认 <图片目录>_labels')
    parser.add_argument('--model', help='自定义检测头 ONNX 模型')
    parser.add_argument('--config', default=str(Path(__file__).with_name('model_config.json')))
    parser.add_argument('--smoke-test', action='store_true', help='构造窗口并检查事件循环后退出')
    args = parser.parse_args()
    # 图片解码/缩放仍由 CPU 完成，避免 OpenCV 再创建一组高并发线程。
    cv2.setNumThreads(1)
    # OpenCV wheel 会注入自己的 Qt 插件路径，必须恢复 PySide6 的插件目录。
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)+'/platforms'
    if 'cv2' in os.environ.get('QT_QPA_FONTDIR', ''):
        os.environ.pop('QT_QPA_FONTDIR', None)
    app = QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    app.setStyleSheet('''
        QMainWindow, QWidget { background: #202833; color: #e0e6ee; }
        QToolBar { spacing: 7px; padding: 7px; border-bottom: 1px solid #3a4656; }
        QPushButton, QComboBox, QLineEdit, QDoubleSpinBox { padding: 6px; background: #2b3747; border: 1px solid #435166; border-radius: 4px; }
        QPushButton:hover { background: #3a4b62; } QPushButton:disabled { color: #637285; }
        QListWidget, QPlainTextEdit, QTableWidget { background: #18212c; border: 1px solid #394759; }
        QListWidget::item { padding: 7px; } QListWidget::item:selected { background: #344d6c; }
        QHeaderView::section { background: #2b3747; padding: 5px; border: none; }
        QToolTip { background: #29384c; color: white; } QStatusBar { color: #b6c8da; }
    ''')
    try:
        window = MainWindow(args)
    except Exception as exc:
        traceback.print_exc()
        QMessageBox.critical(None, '启动失败', str(exc))
        return 1
    window.show()
    if args.smoke_test:
        def finish():
            print('ARMOR_LABELER_SMOKE_OK', flush=True)
            window.close()
            app.quit()
        QTimer.singleShot(600, finish)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
