"""一键扫描、融合并保存出图结果。"""
import os
import threading
import time
from datetime import datetime

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
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
    QSizePolicy,
    QVBoxLayout,
)

from algorithms import (
    build_best_focus_maps,
    build_best_focus_color_maps,
    export_point_cloud,
    get_mpl_font,
    merge_focus_maps,
    point_cloud_from_depth,
    save_composite_image,
    save_output_bundle,
    select_worst_single_frame,
    select_focus_window,
)


# ─────────────────────────────────────────────────────────────
# 辅助：numpy uint8 数组 → QPixmap
# ─────────────────────────────────────────────────────────────
def _arr_to_pixmap(arr: np.ndarray) -> QPixmap:
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    arr_c = np.ascontiguousarray(arr_u8)
    h, w = arr_c.shape
    qimg = QImage(arr_c.data, w, h, w, QImage.Format_Grayscale8)
    # 必须持有 arr_c 引用防止 GC 释放内存
    qimg._keep_alive = arr_c
    return QPixmap.fromImage(qimg)


# ─────────────────────────────────────────────────────────────
class OneClickDialog(QDialog):
    """一键出图对话框"""

    _sig_status = pyqtSignal(str, str)   # (text, color)
    _sig_progress = pyqtSignal(int)
    _sig_log = pyqtSignal(str)
    _sig_done = pyqtSignal(bool, object)  # (success, intensity_map | None)

    def __init__(self, device_controller, config_manager, parent=None):
        super().__init__(parent)
        self.device_controller = device_controller
        self.config_manager = config_manager
        self.setWindowTitle("一键出图")
        self.setMinimumWidth(520)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)

        self._running = False
        self._intensity_map = None   # 最终合成图（float32 H×W）
        self._depth_map = None       # 深度图（float32 H×W）
        self._sharp_map = None       # 锐度图（float32 H×W）
        self._point_cloud = None     # 点云数据（Nx4 数组）
        self._img_size = (0, 0)      # 图像尺寸 (width, height)
        self._last_save_path = ""
        self._last_output_paths = {}
        # 持有 QImage 的底层数组引用，防止 GC
        self._preview_arr_ref = None

        self._setup_ui()
        self._sig_status.connect(self._on_status)
        self._sig_progress.connect(self.progressBar.setValue)
        self._sig_log.connect(self._append_log)
        self._sig_done.connect(self._on_done)
        self._z_timer = QTimer(self)
        self._z_timer.timeout.connect(self._refresh_z_label)
        self._z_timer.start(200)

    def _refresh_z_label(self):
        z = self.device_controller._z_position
        min_limit = getattr(self.device_controller, "_z_min_limit", 0.0)
        max_limit = getattr(self.device_controller, "_z_soft_limit", 68.0)
        self.lblZPos.setText(
            "Z 当前位置: {:.3f} mm    最低提醒: {:.1f} mm    最高提醒: {:.1f} mm".format(
                z, min_limit, max_limit
            )
        )
        color = (
            "#ff3333" if z <= min_limit or z >= max_limit
            else "#ffb000" if z <= min_limit + 1.0 or z >= max_limit - 5.0
            else "#00aaff"
        )
        self.lblZPos.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: {};"
            "background: #ffffff; border: 1px solid {}; border-radius: 4px; padding: 4px;".format(color, color)
        )

    def sync_z_inputs_to_current(self):
        z = float(self.device_controller._z_position)
        try:
            span = abs(float(self.edtZHigh.text().strip()) - float(self.edtZLow.text().strip()))
        except ValueError:
            span = 4.0
        span = max(0.1, span)
        self.edtZHigh.setText("{:.3f}".format(z + span))
        self.edtZLow.setText("{:.3f}".format(z))
        self._refresh_z_label()

    # ── UI 构建 ────────────────────────────────────────────
    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        self.lblZPos = QLabel("Z 当前位置: -- mm")
        self.lblZPos.setAlignment(Qt.AlignCenter)
        self.lblZPos.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #0078d7;"
            "background: #ffffff; border: 1px solid #0078d7; border-radius: 4px; padding: 4px;"
        )
        root.addWidget(self.lblZPos)

        # ── 参数组 ──
        grp_params = QGroupBox("扫描参数（Z 轴：从上向下）")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.edtZHigh = QLineEdit("2.0")
        self.edtZLow = QLineEdit("-2.0")
        self.edtZStep = QLineEdit("0.2")
        self.edtDelay = QLineEdit("0.20")

        form.addRow("Z 高位-起点 (mm):", self.edtZHigh)
        form.addRow("Z 低位-终点 (mm):", self.edtZLow)
        form.addRow("Z 步长 (mm):",      self.edtZStep)
        form.addRow("每步延时 (s):",      self.edtDelay)
        grp_params.setLayout(form)
        root.addWidget(grp_params)

        # ── 方向说明标签 ──
        lbl_dir = QLabel("拍摄方向：Z 高位 → 低位")
        lbl_dir.setStyleSheet("color: #1a6a0a; font-size: 11px; padding: 2px 4px;")
        lbl_dir.setWordWrap(True)
        root.addWidget(lbl_dir)

        # ── 融合方式说明 ──
        lbl_fuse = QLabel("融合方式：DFF 焦点融合")
        lbl_fuse.setStyleSheet("color: #555; font-size: 11px; padding: 2px 4px;")
        lbl_fuse.setWordWrap(True)
        root.addWidget(lbl_fuse)

        # ── 高级选项：点云生成（可选） ──
        grp_advanced = QGroupBox("高级选项")
        adv_layout = QVBoxLayout()
        adv_layout.setSpacing(4)

        # 复选框：是否生成点云
        self.chkPointCloud = QCheckBox("自动输出三维点云（PLY/OBJ）")
        self.chkPointCloud.setChecked(True)
        self.chkPointCloud.stateChanged.connect(self._on_pointcloud_toggle)
        adv_layout.addWidget(self.chkPointCloud)

        # 点云参数（默认隐藏）
        self.grpPcParams = QGroupBox()
        self.grpPcParams.setStyleSheet("QGroupBox { border: none; margin: 0; padding: 0; }")
        form_pc = QFormLayout()
        form_pc.setLabelAlignment(Qt.AlignRight)
        form_pc.setContentsMargins(20, 0, 0, 0)
        self.edtZScale = QLineEdit("1.0")
        self.edtMinSharpness = QLineEdit("5.0")
        self.edtZScale.setMaximumWidth(80)
        self.edtMinSharpness.setMaximumWidth(80)
        form_pc.addRow("Z 轴缩放系数:", self.edtZScale)
        form_pc.addRow("最小锐度(%, 0-100):", self.edtMinSharpness)
        self.grpPcParams.setLayout(form_pc)
        self.grpPcParams.setVisible(False)
        adv_layout.addWidget(self.grpPcParams)

        # 复选框：是否启用粗扫+细扫
        self.chkCoarseFine = QCheckBox("启用粗扫+细扫")
        self.chkCoarseFine.setChecked(False)
        self.chkCoarseFine.stateChanged.connect(self._on_coarsefine_toggle)
        adv_layout.addWidget(self.chkCoarseFine)

        self.grpCfParams = QGroupBox()
        self.grpCfParams.setStyleSheet("QGroupBox { border: none; margin: 0; padding: 0; }")
        form_cf = QFormLayout()
        form_cf.setLabelAlignment(Qt.AlignRight)
        form_cf.setContentsMargins(20, 0, 0, 0)
        self.edtCoarseFactor = QLineEdit("3")
        self.edtFinePct = QLineEdit("30")
        self.edtCoarseFactor.setMaximumWidth(80)
        self.edtFinePct.setMaximumWidth(80)
        form_cf.addRow("粗扫步长倍数:", self.edtCoarseFactor)
        form_cf.addRow("精扫区间比例 %:", self.edtFinePct)
        self.grpCfParams.setLayout(form_cf)
        self.grpCfParams.setVisible(False)
        adv_layout.addWidget(self.grpCfParams)

        grp_advanced.setLayout(adv_layout)
        root.addWidget(grp_advanced)

        # ── 进度条 ──
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        root.addWidget(self.progressBar)

        # ── 状态标签 ──
        self.lblStatus = QLabel("就绪 - 请确认相机已开启采集且串口已连接")
        self.lblStatus.setWordWrap(True)
        self.lblStatus.setStyleSheet("color: gray; padding: 2px;")
        root.addWidget(self.lblStatus)

        # ── 日志框 ──
        self.txtLog = QPlainTextEdit()
        self.txtLog.setReadOnly(True)
        self.txtLog.setMaximumHeight(110)
        root.addWidget(self.txtLog)

        # ── 结果预览 ──
        grp_preview = QGroupBox("合成图预览")
        preview_layout = QVBoxLayout()
        self.lblPreview = QLabel("（扫描完成后显示合成图）")
        self.lblPreview.setAlignment(Qt.AlignCenter)
        self.lblPreview.setMinimumHeight(160)
        self.lblPreview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lblPreview.setStyleSheet("background: #111; color: #888;")
        preview_layout.addWidget(self.lblPreview)
        grp_preview.setLayout(preview_layout)
        root.addWidget(grp_preview, stretch=1)
        # ── 保存路径 ──
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("保存路径:"))
        self._edt_save_path = QLineEdit(self.config_manager.effective_save_path())
        self._edt_save_path.setReadOnly(True)
        self._edt_save_path.setStyleSheet("font-size: 10px; color: #555;")
        self._btn_browse_save = QPushButton("浏览...")
        self._btn_browse_save.setMaximumWidth(60)
        self._btn_browse_save.clicked.connect(self._browse_save_path)
        save_row.addWidget(self._edt_save_path, stretch=1)
        save_row.addWidget(self._btn_browse_save)
        root.addLayout(save_row)
        # ── 按钮行 1：开始/停止 ──
        row1 = QHBoxLayout()
        self.bnStart = QPushButton("一键出图")
        self.bnStart.setStyleSheet(
            "QPushButton{background:#1a7a20;color:white;font-weight:bold;padding:6px;} "
            "QPushButton:disabled{background:#aaa;}"
        )
        self.bnStop = QPushButton("停止")
        self.bnStop.setEnabled(False)
        row1.addWidget(self.bnStart, stretch=2)
        row1.addWidget(self.bnStop, stretch=1)
        root.addLayout(row1)

        # ── 按钮行 2：导出 ──
        row2 = QHBoxLayout()
        self.bnExport = QPushButton("另存合成图…")
        self.bnExport.setEnabled(False)
        row2.addWidget(self.bnExport)
        root.addLayout(row2)

        # ── 按钮行 3：点云操作（勾选生成点云后才启用） ──
        row3 = QHBoxLayout()
        self.bnVisualize = QPushButton("可视化点云")
        self.bnExportPly = QPushButton("导出点云…")
        self.bnVisualize.setEnabled(False)
        self.bnExportPly.setEnabled(False)
        self.bnVisualize.setVisible(False)
        self.bnExportPly.setVisible(False)
        row3.addWidget(self.bnVisualize)
        row3.addWidget(self.bnExportPly)
        root.addLayout(row3)

        self.bnStart.clicked.connect(self._start)
        self.bnStop.clicked.connect(self._stop)
        self.bnExport.clicked.connect(self._export)
        self.bnVisualize.clicked.connect(self._visualize_point_cloud)
        self.bnExportPly.clicked.connect(self._export_point_cloud)
        self._on_pointcloud_toggle(Qt.Checked if self.chkPointCloud.isChecked() else Qt.Unchecked)

    # ── 信号槽 ─────────────────────────────────────────────────
    def _browse_save_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存路径", self.config_manager.effective_save_path())
        if path:
            self.config_manager.save_path = path
            self._edt_save_path.setText(path)

    def _on_pointcloud_toggle(self, state):
        """复选框切换：显示/隐藏点云参数和按钮"""
        checked = state == Qt.Checked
        self.grpPcParams.setVisible(checked)
        self.bnVisualize.setVisible(checked)
        self.bnExportPly.setVisible(checked)
        # 如果取消勾选，禁用按鈕
        if not checked:
            self.bnVisualize.setEnabled(False)
            self.bnExportPly.setEnabled(False)

    def _on_coarsefine_toggle(self, state):
        self.grpCfParams.setVisible(state == Qt.Checked)

    def _on_status(self, message, color):
        self.lblStatus.setText(message)
        self.lblStatus.setStyleSheet("color: {}; padding: 2px;".format(color))

    def _append_log(self, message):
        self.txtLog.appendPlainText(message)
        self.txtLog.verticalScrollBar().setValue(
            self.txtLog.verticalScrollBar().maximum()
        )

    def _on_done(self, success, intensity_map):
        self.bnStart.setEnabled(True)
        self.bnStop.setEnabled(False)
        if success and intensity_map is not None:
            self._intensity_map = intensity_map
            self.bnExport.setEnabled(True)
            self._show_preview(intensity_map)
            # 如果生成了点云，启用点云按钮
            if self._point_cloud is not None and len(self._point_cloud) > 0:
                self.bnVisualize.setEnabled(True)
                self.bnExportPly.setEnabled(True)
    # ── 参数解析 ───────────────────────────────────────────────
    def _parse_params(self):
        z_high = float(self.edtZHigh.text().strip())
        z_low  = float(self.edtZLow.text().strip())
        z_step = float(self.edtZStep.text().strip())
        delay  = float(self.edtDelay.text().strip())

        if z_high <= z_low:
            raise ValueError(
                "Z 高位({:.3f}) 必须大于 Z 低位({:.3f})！\n"
                "拍摄方向：高位 → 低位（从上向下）".format(z_high, z_low)
            )
        if z_step <= 0:
            raise ValueError("Z 步长必须为正数（实际移动方向由程序保证为负）")
        if delay < 0:
            raise ValueError("每步延时不能为负数")

        n_steps = int(round((z_high - z_low) / z_step)) + 1
        if n_steps < 2:
            raise ValueError("步数过少（{}步），请减小步长或增大扫描范围".format(n_steps))

        return {
            "z_high": z_high,
            "z_low":  z_low,
            "z_step": z_step,
            "delay":  delay,
            "n_steps": n_steps,
        }

    # ── 开始 ───────────────────────────────────────────────────
    def _start(self):
        try:
            params = self._parse_params()
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return

        if not self.device_controller.grabbing:
            QMessageBox.warning(self, "错误", "请先开启相机采集！")
            return
        if not self.device_controller.serial_connected:
            QMessageBox.warning(self, "错误",
                                "请先连接串口（串口用于控制 Z 轴运动）！")
            return

        # 读取点云参数
        gen_pointcloud = self.chkPointCloud.isChecked()
        if gen_pointcloud:
            try:
                z_scale = float(self.edtZScale.text().strip())
                min_sharp = float(self.edtMinSharpness.text().strip())
            except ValueError:
                z_scale = 1.0
                min_sharp = 20.0
            params["gen_pointcloud"] = True
            params["z_scale"] = z_scale
            params["min_sharp"] = min_sharp
        else:
            params["gen_pointcloud"] = False

        # 读取粗扫+细扫参数
        coarse_fine = self.chkCoarseFine.isChecked()
        params["coarse_fine"] = coarse_fine
        if coarse_fine:
            try:
                params["coarse_factor"] = max(2, int(self.edtCoarseFactor.text().strip()))
            except ValueError:
                params["coarse_factor"] = 3
            try:
                params["fine_pct"] = max(10, min(80, float(self.edtFinePct.text().strip())))
            except ValueError:
                params["fine_pct"] = 30

        confirm_msg = (
            "拍摄方向：从上向下\n"
            "Z 轴：{:.3f} mm（高位）→ {:.3f} mm（低位）\n"
            "步长：{:.3f} mm，共 {} 步，每步延时 {:.2f}s\n"
            "融合方式：DFF 焦点融合（亚步长插值）\n"
        ).format(
            params["z_high"], params["z_low"],
            params["z_step"], params["n_steps"], params["delay"]
        )
        if coarse_fine:
            confirm_msg += "粗扫+细扫：是（粗步×{}，精扫区间={}%）\n".format(
                params["coarse_factor"], int(params["fine_pct"]))
        if gen_pointcloud:
            confirm_msg += "点云生成：是（Z缩放={:.2f}，锐度阈值={:.1f}）\n".format(z_scale, min_sharp)
        confirm_msg += "\n确认开始？"

        reply = QMessageBox.question(
            self, "确认出图", confirm_msg,
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._running = True
        self._intensity_map = None
        self._depth_map = None
        self._sharp_map = None
        self._point_cloud = None
        self._img_size = (0, 0)
        self._last_save_path = ""
        self._last_output_paths = {}
        self.bnStart.setEnabled(False)
        self.bnStop.setEnabled(True)
        self.bnExport.setEnabled(False)
        self.bnVisualize.setEnabled(False)
        self.bnExportPly.setEnabled(False)
        self.progressBar.setValue(0)
        self.txtLog.clear()
        self.lblPreview.setText("（扫描中…）")
        self.lblPreview.setPixmap(QPixmap())
        self._sig_status.emit("准备中...", "blue")

        threading.Thread(
            target=self._worker, args=(params,), daemon=True
        ).start()

    def _stop(self):
        self._running = False
        self._sig_status.emit("停止请求已发送，等待当前步完成...", "orange")

    # ── 后台工作线程 ───────────────────────────────────────────
    def _worker(self, params):
        z_high  = params["z_high"]
        z_low   = params["z_low"]
        z_step  = params["z_step"]
        delay   = params["delay"]
        n_steps = params["n_steps"]
        gen_pointcloud = params.get("gen_pointcloud", False)
        z_scale = params.get("z_scale", 1.0)
        min_sharp = params.get("min_sharp", 20.0)

        def emit_status(msg, color="blue"):
            self._sig_status.emit(msg, color)

        def emit_log(msg):
            self._sig_log.emit(msg)

        def emit_progress(v):
            self._sig_progress.emit(v)

        try:
            emit_log("开始：Z {:.3f}->{:.3f}mm，步长 {:.3f}mm，{} 步".format(
                z_high, z_low, z_step, n_steps))

            # ── 步骤 1：移动到高位起点 ──
            emit_status("步骤 1/4：移动到高位起点 Z={:.3f}mm...".format(z_high))
            try:
                self.device_controller.move_z_absolute_wait(z_high, feed=300)
            except Exception as exc:
                emit_status("串口发送失败: " + str(exc), "red")
                self._sig_done.emit(False, None)
                return

            # 等待运动台到达起点
            time.sleep(0.1)

            # ── 步骤 2：检查相机图像尺寸 ──
            emit_status("步骤 2/4：获取图像尺寸...")
            gray0, width, height = self.device_controller.get_gray_frame()
            if gray0 is None or width == 0 or height == 0:
                emit_status("错误：无法获取相机帧，请检查相机是否正在输出图像", "red")
                self._sig_done.emit(False, None)
                return
            emit_log("尺寸：{}×{}".format(width, height))

            # ── 内部扫描函数 ──
            def do_scan(z_start, z_end, step, label, prog_base, prog_span):
                n = max(1, int(round((z_start - z_end) / step)) + 1)
                frames = []
                color_frames = []
                zpos = []
                for i in range(n):
                    if not self._running:
                        return None, None, None
                    z_cur = round(z_start - i * step, 6)
                    zpos.append(z_cur)
                    if i > 0:
                        try:
                            self.device_controller.move_z_relative_wait(-step, feed=300)
                        except Exception as exc:
                            emit_status("串口错误: " + str(exc), "red")
                            return None, None, None
                        time.sleep(delay)
                    emit_status("{} Z={:.3f}mm ({}/{})".format(label, z_cur, i + 1, n))
                    gray, color, fw, fh = self.device_controller.get_gray_color_frame()
                    if gray is not None and fw == width and fh == height:
                        frames.append(gray.astype(np.float32))
                        color_frames.append(color)
                    else:
                        frames.append(np.zeros((height, width), dtype=np.float32))
                        color_frames.append(None)
                        emit_log("  [警告] Z={:.3f}mm 帧尺寸异常".format(z_cur))
                    emit_progress(int(prog_base + (i + 1) / n * prog_span))
                return frames, zpos, color_frames

            coarse_fine = params.get("coarse_fine", False)

            if coarse_fine:
                coarse_factor = params.get("coarse_factor", 3)
                fine_pct = params.get("fine_pct", 30)
                coarse_step = z_step * coarse_factor

                # ── 粗扫 ──
                emit_log("粗扫：步长 {:.3f}mm".format(coarse_step))
                coarse_frames, coarse_z, coarse_colors = do_scan(
                    z_high, z_low, coarse_step, "粗扫", 0, 30)
                if coarse_frames is None:
                    emit_status("已停止", "orange")
                    self._sig_done.emit(False, None)
                    return
                emit_log("粗扫完成：{} 帧".format(len(coarse_frames)))

                emit_status("粗扫焦点融合 + 定位焦点区间…")
                c_depth, c_sharp, c_intensity = build_best_focus_maps(
                    coarse_frames, coarse_z)
                if c_intensity is None:
                    emit_status("错误：粗扫融合失败", "red")
                    self._sig_done.emit(False, None)
                    return

                fine_z0, fine_z1 = select_focus_window(
                    coarse_z, coarse_frames, fine_pct)
                emit_log("精扫区间：{:.3f}~{:.3f}mm".format(fine_z0, fine_z1))
                emit_progress(35)

                # 移动到精扫起点（高位）
                try:
                    self.device_controller.move_z_absolute_wait(fine_z1, feed=300)
                except Exception as exc:
                    emit_status("串口错误: " + str(exc), "red")
                    self._sig_done.emit(False, None)
                    return
                time.sleep(0.1)

                # ── 精扫 ──
                emit_log("精扫：步长 {:.3f}mm".format(z_step))
                fine_frames, fine_z, fine_colors = do_scan(
                    fine_z1, fine_z0, z_step, "精扫", 35, 35)
                if fine_frames is None:
                    emit_status("已停止", "orange")
                    self._sig_done.emit(False, None)
                    return
                emit_log("精扫完成：{} 帧".format(len(fine_frames)))

                emit_status("焦点融合中…")
                f_depth, f_sharp, f_intensity = build_best_focus_maps(
                    fine_frames, fine_z)
                if f_intensity is not None:
                    updated = merge_focus_maps(
                        c_depth, c_sharp, c_intensity,
                        f_depth, f_sharp, f_intensity)
                    emit_log("精扫更新：{:,} 像素".format(updated))

                depth_map = c_depth
                sharp_map = c_sharp
                intensity_map = c_intensity
                frames_gray = coarse_frames + (fine_frames or [])
                z_positions = coarse_z + (fine_z or [])
                frames_color = coarse_colors + (fine_colors or [])
                _, _, _, color_map = build_best_focus_color_maps(frames_gray, z_positions, frames_color)

            else:
                # ── 单次扫描（原始模式） ──
                emit_status("步骤 3/4：从上向下扫描采帧…")
                frames_gray, z_positions, frames_color = do_scan(
                    z_high, z_low, z_step, "扫描", 0, 70)
                if frames_gray is None:
                    emit_status("已停止", "orange")
                    self._sig_done.emit(False, None)
                    return
                if not self._running:
                    emit_status("已停止", "orange")
                    self._sig_done.emit(False, None)
                    return

                emit_log("采帧完成：{} 帧".format(len(frames_gray)))

                emit_status("步骤 4/4：焦点融合中…")
                emit_progress(75)
                depth_map, sharp_map, intensity_map, color_map = build_best_focus_color_maps(
                    frames_gray, z_positions, frames_color)

            emit_progress(88)

            if intensity_map is None:
                emit_status("错误：融合失败，没有有效图像数据", "red")
                self._sig_done.emit(False, None)
                return
            reference_map, reference_z, reference_score = select_worst_single_frame(frames_gray, z_positions)
            reference_color_map = None
            if reference_z is not None and frames_color:
                for idx, z_pos in enumerate(z_positions):
                    if float(z_pos) == float(reference_z) and idx < len(frames_color):
                        reference_color_map = frames_color[idx]
                        break
            if reference_map is not None:
                emit_log("最差单帧（对比参考）：Z={:.3f}mm".format(float(reference_z)))

            # 保存深度图和图像尺寸到成员变量
            self._depth_map = depth_map
            self._sharp_map = sharp_map
            self._img_size = (width, height)

            # ── 可选：生成点云 ──
            point_count = 0
            coverage = 0.0
            if gen_pointcloud:
                emit_status("生成点云数据...")
                emit_progress(90)
                point_cloud, coverage = point_cloud_from_depth(
                    depth_map,
                    sharp_map,
                    intensity_map,
                    self.config_manager.pixels_per_mm,
                    min_sharp,
                    z_scale,
                )
                self._point_cloud = point_cloud
                point_count = len(point_cloud)
                emit_log("点云：{:,} 点，覆盖率 {:.1f}%".format(point_count, coverage))

            # ── 自动保存完整输出包 ──
            save_dir = self.config_manager.effective_save_path()
            emit_status("保存文件中… 请稍候", "blue")
            emit_progress(92)
            try:
                output_params = {
                    "zh": z_high,
                    "zl": z_low,
                    "step": z_step,
                    "mode": "cxf" if params.get("coarse_fine") else "scan",
                    "zscale": z_scale if gen_pointcloud else None,
                    "sharp": min_sharp if gen_pointcloud else None,
                }
                output_paths = save_output_bundle(
                    save_dir,
                    "oneclick",
                    intensity_map,
                    depth_map,
                    self._point_cloud,
                    self.config_manager.pixels_per_mm,
                    params=output_params,
                    z_scale=z_scale,
                    comment="Generated by OneClick Dialog",
                    reference_map=reference_map,
                    reference_label="worst Z={:.3f} mm".format(float(reference_z)) if reference_z is not None else "worst single frame",
                    color_map=color_map,
                    reference_color_map=reference_color_map,
                    frames_gray=frames_gray,
                    z_positions=z_positions,
                    frames_color=frames_color,
                )
                self._last_output_paths = output_paths
                save_path = output_paths.get("full_focus", "")
                self._last_save_path = save_path
                emit_log("保存：{}".format(save_dir))
            except Exception as exc:
                emit_log("  [警告] 自动保存失败：{}".format(exc))
                save_path = "（保存失败）"

            emit_progress(100)

            # 构建状态消息
            status_msg = "完成：{}×{}，目录：{}".format(width, height, save_dir)
            if gen_pointcloud:
                status_msg += "\n点云：{:,} 点".format(point_count)

            emit_status(status_msg, "green")
            self._sig_done.emit(True, intensity_map)

        except Exception as exc:
            emit_status("发生异常: " + str(exc), "red")
            emit_log("[ERROR] " + str(exc))
            self._sig_done.emit(False, None)
        finally:
            self._running = False

    # ── 预览图显示 ─────────────────────────────────────────────
    def _show_preview(self, intensity_map: np.ndarray):
        pixmap = _arr_to_pixmap(intensity_map)
        # 按标签当前尺寸缩放，保持宽高比
        label_size = self.lblPreview.size()
        if label_size.width() > 50 and label_size.height() > 50:
            pixmap = pixmap.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self.lblPreview.setPixmap(pixmap)
        self.lblPreview.setText("")

    def resizeEvent(self, event):
        """窗口缩放时重新适配预览图"""
        super().resizeEvent(event)
        if self._intensity_map is not None:
            self._show_preview(self._intensity_map)

    # ── 另存为 ────────────────────────────────────────────────
    def _export(self):
        if self._intensity_map is None:
            QMessageBox.warning(self, "提示", "没有可保存的合成图，请先执行出图。")
            return

        default_dir = (
            os.path.dirname(self._last_save_path)
            if self._last_save_path
            else self.config_manager.effective_save_path()
        )
        default_name = "composite_{}.png".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "另存合成图",
            os.path.join(default_dir, default_name),
            "PNG 图像 (*.png);;TIFF 图像 (*.tif *.tiff);;BMP 图像 (*.bmp);;全部文件 (*.*)",
        )
        if not file_path:
            return
        try:
            save_composite_image(self._intensity_map, file_path)
            QMessageBox.information(
                self, "已保存",
                "合成图已保存至：\n{}".format(file_path)
            )
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))

    # ── 可视化点云 ─────────────────────────────────────────────
    def _visualize_point_cloud(self):
        if self._depth_map is None or self._point_cloud is None or len(self._point_cloud) == 0:
            QMessageBox.warning(self, "提示", "暂无点云/深度图数据，请先勾选「自动输出三维点云」并执行出图。")
            return
        try:
            import matplotlib
            matplotlib.use("Qt5Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fp = get_mpl_font()
            title_kw = {"fontproperties": fp} if fp else {}
            fig = plt.figure(figsize=(14, 6))
            fig.suptitle("一键出图结果 - 点云数据", fontsize=14, **title_kw)

            ppmm = self.config_manager.pixels_per_mm if self.config_manager.pixels_per_mm > 0 else 1.0
            width, height = self._img_size
            depth = np.asarray(self._depth_map, dtype=np.float32)
            finite = depth[np.isfinite(depth)]
            depth_show = depth - float(np.nanpercentile(finite, 2.0)) if finite.size else depth
            d_valid = depth_show[np.isfinite(depth_show)]
            d_min = float(np.nanpercentile(d_valid, 2.0))  if d_valid.size else 0.0
            d_max = float(np.nanpercentile(d_valid, 98.0)) if d_valid.size else 1.0
            ax1 = fig.add_subplot(1, 2, 1)
            image = ax1.imshow(
                depth_show,
                cmap="plasma",
                origin="upper",
                vmin=d_min, vmax=d_max,
                extent=[-width / (2 * ppmm), width / (2 * ppmm), height / (2 * ppmm), -height / (2 * ppmm)],
            )
            cbar = plt.colorbar(image, ax=ax1)
            cbar.set_label("相对深度 (mm)", fontproperties=fp if fp else None)
            cbar.ax.text(0.5, 1.02, "黑紫=高(近)", transform=cbar.ax.transAxes,
                         ha="center", va="bottom", fontsize=8, color="#333")
            cbar.ax.text(0.5, -0.04, "黄=低(远)", transform=cbar.ax.transAxes,
                         ha="center", va="top", fontsize=8, color="#333")
            step = max(1, depth_show.shape[0] // 300)
            ds = depth_show[::step, ::step]
            xc = np.linspace(-width / (2 * ppmm), width / (2 * ppmm), ds.shape[1])
            yc = np.linspace(-height / (2 * ppmm), height / (2 * ppmm), ds.shape[0])
            levels = np.linspace(d_min, d_max, 8)[1:-1]
            try:
                cs = ax1.contour(xc, yc, ds, levels=levels, colors="white", linewidths=0.6, alpha=0.75)
                ax1.clabel(cs, fmt="%.2f mm", fontsize=7, inline=True, inline_spacing=4)
            except Exception:
                pass
            ax1.set_title("深度图  (黑紫=高面 / 黄=低面)", **title_kw)
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")

            pc = self._point_cloud
            x, y, z, intensity = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
            max_pts = 80000
            if len(x) > max_pts:
                index = np.random.default_rng(42).choice(len(x), max_pts, replace=False)
                x, y, z, intensity = x[index], y[index], z[index], intensity[index]
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            scatter = ax2.scatter(x, y, z, c=z, cmap="turbo", s=0.35, alpha=0.70, linewidths=0)
            plt.colorbar(scatter, ax=ax2, label="Relative height (mm)")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            ax2.set_zlabel("Z (mm)")
            ax2.set_title("Point Cloud ({:,} pts)".format(len(self._point_cloud)), **title_kw)
            try:
                ax2.set_box_aspect((max(float(np.ptp(x)), 1e-6), max(float(np.ptp(y)), 1e-6), max(float(np.ptp(z)), 1e-6)))
            except Exception:
                pass
            plt.tight_layout()
            plt.show()
        except ImportError:
            QMessageBox.warning(self, "依赖缺失", "可视化需要 matplotlib:\npip install matplotlib")
        except Exception as exc:
            QMessageBox.warning(self, "可视化错误", str(exc))

    # ── 导出点云 ───────────────────────────────────────────────
    def _export_point_cloud(self):
        if self._point_cloud is None or len(self._point_cloud) == 0:
            QMessageBox.warning(self, "提示", "暂无点云数据，请先勾选「生成点云数据」并执行出图。")
            return

        default_name = "pointcloud_{}.ply".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出点云",
            os.path.join(self.config_manager.effective_save_path(), default_name),
            "PLY 文件 (*.ply);;OBJ 文件 (*.obj);;CSV 文件 (*.csv);;全部文件 (*.*)",
        )
        if not file_path:
            return
        try:
            file_type = export_point_cloud(
                file_path,
                self._point_cloud,
                self.config_manager.pixels_per_mm,
                "Generated by OneClick Dialog",
            )
            QMessageBox.information(
                self,
                "导出完成",
                "已导出 {:,} 个点（{}）\n{}".format(len(self._point_cloud), file_type.upper(), file_path),
            )
        except Exception as exc:
            QMessageBox.warning(self, "导出错误", str(exc))

