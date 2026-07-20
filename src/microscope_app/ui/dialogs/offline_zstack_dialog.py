"""Offline Z-stack reconstruction dialog for direct desktop use."""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...paths import ENTRY_POINT
from ...reconstruction.io_utils import load_zstack_from_path
from ...reconstruction.pointcloud import create_point_cloud_from_depth
from ...reconstruction.reconstruction import ReconstructionConfig, reconstruct_from_stack
from ...reconstruction.visualization import save_reconstruction_outputs


def _rgb_to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = np.clip(np.asarray(image), 0, 255).astype(np.uint8)
    rgb_c = np.ascontiguousarray(rgb)
    height, width = rgb_c.shape[:2]
    qimg = QImage(rgb_c.data, width, height, width * 3, QImage.Format_RGB888)
    qimg._keep_alive = rgb_c
    return QPixmap.fromImage(qimg)


class OfflineZStackDialog(QDialog):
    _sig_log = pyqtSignal(str)
    _sig_done = pyqtSignal(bool, object)

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self._running = False
        self._last_manifest = ""
        self._last_output_dir = ""
        self._full_focus_pixmap = None
        self._heatmap_pixmap = None

        self.setWindowTitle("离线 Z-stack 三维重建")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.resize(1180, 760)

        self._setup_ui()
        self._sig_log.connect(self._append_log)
        self._sig_done.connect(self._on_done)
        self._load_defaults()

    def _setup_ui(self):
        root = QVBoxLayout(self)

        controls = QGroupBox("参数")
        controls_form = QFormLayout(controls)

        self.edtInput = QLineEdit()
        self.btnBrowseInput = QPushButton("选择图像目录...")
        self.btnBrowseInput.clicked.connect(self._browse_input_dir)
        input_row = QHBoxLayout()
        input_row.addWidget(self.edtInput, 1)
        input_row.addWidget(self.btnBrowseInput)
        controls_form.addRow("图像序列目录", self._wrap_layout(input_row))

        self.edtOutput = QLineEdit()
        self.btnBrowseOutput = QPushButton("选择输出目录...")
        self.btnBrowseOutput.clicked.connect(self._browse_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.edtOutput, 1)
        output_row.addWidget(self.btnBrowseOutput)
        controls_form.addRow("输出目录", self._wrap_layout(output_row))

        self.cmbZUnit = QComboBox()
        self.cmbZUnit.addItems(["auto", "index", "um", "mm"])
        self.cmbZUnit.setCurrentText("auto")
        controls_form.addRow("文件名 Z 单位", self.cmbZUnit)

        self.edtZStepUm = QLineEdit("5")
        self.edtZStartMm = QLineEdit("0.0")
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("步距(um)"))
        step_row.addWidget(self.edtZStepUm)
        step_row.addSpacing(12)
        step_row.addWidget(QLabel("起始 Z(mm)"))
        step_row.addWidget(self.edtZStartMm)
        controls_form.addRow("Z 参数", self._wrap_layout(step_row))

        self.edtPixelsPerMm = QLineEdit()
        controls_form.addRow("像素密度 pixels/mm", self.edtPixelsPerMm)

        algo_row = QHBoxLayout()
        self.cmbFocusMethod = QComboBox()
        self.cmbFocusMethod.addItems(["combined", "laplacian", "sobel", "tenengrad"])
        self.edtWindowSize = QLineEdit("9")
        self.edtFocusThreshold = QLineEdit("8")
        algo_row.addWidget(QLabel("清晰度"))
        algo_row.addWidget(self.cmbFocusMethod)
        algo_row.addSpacing(8)
        algo_row.addWidget(QLabel("窗口"))
        algo_row.addWidget(self.edtWindowSize)
        algo_row.addSpacing(8)
        algo_row.addWidget(QLabel("阈值百分位"))
        algo_row.addWidget(self.edtFocusThreshold)
        controls_form.addRow("重建算法", self._wrap_layout(algo_row))

        filter_row = QHBoxLayout()
        self.edtMedianSize = QLineEdit("5")
        self.edtGaussianSigma = QLineEdit("0.8")
        self.cmbSmoothing = QComboBox()
        self.cmbSmoothing.addItems(["light", "medium", "off", "strong"])
        self.cmbSmoothing.setCurrentText("light")
        filter_row.addWidget(QLabel("中值核"))
        filter_row.addWidget(self.edtMedianSize)
        filter_row.addSpacing(8)
        filter_row.addWidget(QLabel("高斯 sigma"))
        filter_row.addWidget(self.edtGaussianSigma)
        filter_row.addSpacing(8)
        filter_row.addWidget(QLabel("平滑强度"))
        filter_row.addWidget(self.cmbSmoothing)
        controls_form.addRow("深度平滑", self._wrap_layout(filter_row))

        options_row = QHBoxLayout()
        self.chkAlign = QCheckBox("采集漂移时自动对齐")
        self.chkAlign.setChecked(True)
        self.chkExportPointCloud = QCheckBox("导出 PLY/CSV 点云")
        self.chkExportPointCloud.setChecked(True)
        self.edtZExaggeration = QLineEdit("1.0")
        options_row.addWidget(self.chkAlign)
        options_row.addSpacing(12)
        options_row.addWidget(self.chkExportPointCloud)
        options_row.addSpacing(12)
        options_row.addWidget(QLabel("Z 缩放"))
        options_row.addWidget(self.edtZExaggeration)
        options_row.addStretch(1)
        controls_form.addRow("选项", self._wrap_layout(options_row))

        root.addWidget(controls)

        actions = QHBoxLayout()
        self.btnRun = QPushButton("开始重建")
        self.btnRun.clicked.connect(self._start_reconstruction)
        self.btnMeasure = QPushButton("打开测量窗口")
        self.btnMeasure.clicked.connect(self._open_measurement)
        self.btnMeasure.setEnabled(False)
        self.btnOpenOutput = QPushButton("打开输出目录")
        self.btnOpenOutput.clicked.connect(self._open_output_dir)
        self.btnOpenOutput.setEnabled(False)
        actions.addWidget(self.btnRun)
        actions.addWidget(self.btnMeasure)
        actions.addWidget(self.btnOpenOutput)
        actions.addStretch(1)
        root.addLayout(actions)

        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        root.addWidget(self.progressBar)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("全焦图"))
        self.lblFullFocus = QLabel("暂无结果")
        self.lblFullFocus.setAlignment(Qt.AlignCenter)
        self.lblFullFocus.setMinimumSize(420, 300)
        self.lblFullFocus.setStyleSheet("border: 1px solid #909090; background: #101010; color: #d0d0d0;")
        left_layout.addWidget(self.lblFullFocus, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("高度热力图"))
        self.lblHeatmap = QLabel("暂无结果")
        self.lblHeatmap.setAlignment(Qt.AlignCenter)
        self.lblHeatmap.setMinimumSize(420, 300)
        self.lblHeatmap.setStyleSheet("border: 1px solid #909090; background: #101010; color: #d0d0d0;")
        right_layout.addWidget(self.lblHeatmap, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        self.txtLog = QPlainTextEdit()
        self.txtLog.setReadOnly(True)
        self.txtLog.setMinimumHeight(180)
        root.addWidget(self.txtLog)

    def _load_defaults(self):
        self.edtOutput.setText(self.config_manager.effective_save_path())
        self.edtPixelsPerMm.setText("{:.4f}".format(float(self.config_manager.pixels_per_mm)))

    def _browse_input_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择 Z-stack 图像目录", self.config_manager.effective_save_path())
        if path:
            self.edtInput.setText(path)

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.config_manager.effective_save_path())
        if path:
            self.edtOutput.setText(path)

    def _start_reconstruction(self):
        if self._running:
            return
        try:
            params = self._collect_params()
        except Exception as exc:
            QMessageBox.warning(self, "参数错误", str(exc), QMessageBox.Ok)
            return

        self._running = True
        self.progressBar.setRange(0, 0)
        self.progressBar.setValue(0)
        self.btnRun.setEnabled(False)
        self.btnMeasure.setEnabled(False)
        self.btnOpenOutput.setEnabled(False)
        self._append_log("开始离线重建...")
        threading.Thread(target=self._worker_reconstruct, args=(params,), daemon=True).start()

    def _worker_reconstruct(self, params):
        try:
            self._sig_log.emit("读取图像序列: {}".format(params["input_dir"]))
            stack = load_zstack_from_path(
                input_path=params["input_dir"],
                filename_z_unit=params["filename_z_unit"],
                z_step_mm=params["z_step_mm"],
                z_start_mm=params["z_start_mm"],
                align=params["align"],
            )
            self._sig_log.emit("已加载 {} 张图像".format(len(stack.file_paths)))
            self._sig_log.emit("Z 来源: {}".format(stack.z_source))
            if stack.skipped_files:
                self._sig_log.emit("已自动忽略 {} 个非堆栈图像/尺寸不匹配图像".format(len(stack.skipped_files)))
            self._sig_log.emit(
                "XY 尺度: 由 pixels_per_mm = {:.6f} 计算，距离公式为 pixel_distance / pixels_per_mm".format(
                    params["pixels_per_mm"]
                )
            )

            config = ReconstructionConfig(
                focus_method=params["focus_method"],
                focus_window_size=params["window_size"],
                focus_threshold_percentile=params["focus_threshold_percentile"],
                median_filter_size=params["median_filter_size"],
                gaussian_sigma=params["gaussian_sigma"],
                smoothing_strength=params["smoothing_strength"],
            )
            result = reconstruct_from_stack(stack, config=config)

            point_cloud = None
            if params["export_point_cloud"]:
                point_cloud = create_point_cloud_from_depth(
                    depth_map_mm=result.depth_map_mm,
                    texture_rgb=result.full_focus_rgb,
                    pixels_per_mm=params["pixels_per_mm"],
                    valid_mask=result.valid_mask,
                    z_exaggeration=params["z_exaggeration"],
                )

            output_dir = os.path.join(params["output_dir"], "offline_zstack_result")
            os.makedirs(output_dir, exist_ok=True)
            saved_paths = save_reconstruction_outputs(
                result=result,
                output_dir=output_dir,
                pixels_per_mm=params["pixels_per_mm"],
                point_cloud=point_cloud,
                save_point_cloud_file=params["export_point_cloud"],
            )
            self._sig_done.emit(
                True,
                {
                    "result": result,
                    "saved_paths": saved_paths,
                    "point_count": 0 if point_cloud is None else len(point_cloud.points_mm),
                },
            )
        except Exception as exc:
            self._sig_done.emit(False, str(exc))

    def _on_done(self, success, payload):
        self._running = False
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(100 if success else 0)
        self.btnRun.setEnabled(True)

        if not success:
            self._append_log("失败: {}".format(payload))
            QMessageBox.warning(self, "重建失败", str(payload), QMessageBox.Ok)
            return

        result = payload["result"]
        saved_paths = payload["saved_paths"]
        self._last_manifest = saved_paths.get("manifest_json", "")
        self._last_output_dir = os.path.dirname(self._last_manifest) if self._last_manifest else ""
        self.btnMeasure.setEnabled(bool(self._last_manifest))
        self.btnOpenOutput.setEnabled(bool(self._last_output_dir))

        self._full_focus_pixmap = _rgb_to_pixmap(result.full_focus_rgb)
        self.lblFullFocus.setPixmap(
            self._full_focus_pixmap.scaled(self.lblFullFocus.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        heatmap_path = saved_paths.get("depth_heatmap_png", "")
        if heatmap_path and os.path.exists(heatmap_path):
            self._heatmap_pixmap = QPixmap(heatmap_path)
            self.lblHeatmap.setPixmap(
                self._heatmap_pixmap.scaled(self.lblHeatmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        self._append_log("重建完成")
        self._append_log("输出目录: {}".format(self._last_output_dir))
        self._append_log("测量文件: {}".format(self._last_manifest))
        self._append_log("点云点数: {}".format(payload["point_count"]))

    def _collect_params(self):
        input_dir = self.edtInput.text().strip()
        output_dir = self.edtOutput.text().strip()
        if not input_dir:
            raise ValueError("请选择图像序列目录")
        if not os.path.exists(input_dir):
            raise ValueError("图像目录不存在: {}".format(input_dir))
        if not output_dir:
            raise ValueError("请选择输出目录")

        filename_z_unit = self.cmbZUnit.currentText().strip()
        z_step_mm = None
        if filename_z_unit == "index":
            z_step_mm = float(self.edtZStepUm.text().strip()) / 1000.0

        pixels_per_mm = float(self.edtPixelsPerMm.text().strip())
        if pixels_per_mm <= 0:
            raise ValueError("pixels_per_mm 必须大于 0")

        return {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "filename_z_unit": filename_z_unit,
            "z_step_mm": z_step_mm,
            "z_start_mm": float(self.edtZStartMm.text().strip()),
            "pixels_per_mm": pixels_per_mm,
            "focus_method": self.cmbFocusMethod.currentText().strip(),
            "window_size": int(self.edtWindowSize.text().strip()),
            "focus_threshold_percentile": float(self.edtFocusThreshold.text().strip()),
            "median_filter_size": int(self.edtMedianSize.text().strip()),
            "gaussian_sigma": float(self.edtGaussianSigma.text().strip()),
            "smoothing_strength": self.cmbSmoothing.currentText().strip(),
            "align": self.chkAlign.isChecked(),
            "export_point_cloud": self.chkExportPointCloud.isChecked(),
            "z_exaggeration": float(self.edtZExaggeration.text().strip()),
        }

    def _open_measurement(self):
        if not self._last_manifest or not os.path.exists(self._last_manifest):
            QMessageBox.information(self, "提示", "请先完成一次重建", QMessageBox.Ok)
            return
        try:
            subprocess.Popen([sys.executable, str(ENTRY_POINT), "measure", "--manifest", self._last_manifest])
            self._append_log("已打开测量窗口")
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", str(exc), QMessageBox.Ok)

    def _open_output_dir(self):
        if self._last_output_dir and os.path.exists(self._last_output_dir):
            os.startfile(self._last_output_dir)

    def _append_log(self, text):
        self.txtLog.appendPlainText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._full_focus_pixmap is not None:
            self.lblFullFocus.setPixmap(
                self._full_focus_pixmap.scaled(self.lblFullFocus.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        if self._heatmap_pixmap is not None:
            self.lblHeatmap.setPixmap(
                self._heatmap_pixmap.scaled(self.lblHeatmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    @staticmethod
    def _wrap_layout(layout):
        widget = QWidget()
        widget.setLayout(layout)
        return widget
