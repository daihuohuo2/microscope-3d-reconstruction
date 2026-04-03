"""一键出图对话框

拍摄方向（固定）：Z 轴从高位 → 低位，即从上向下。

流程
----
1. 检查相机、串口、参数合法性
2. 移动 Z 轴到高位（扫描起点）
3. 从高位逐步向低位运动，每步拍一帧灰度图
4. 用 DFF（逐像素焦点融合）生成合成图
5. 自动保存合成图到指定目录（BMP）
6. 在对话框内预览结果图
7. 输出完整日志
"""
import os
import threading
import time
from datetime import datetime

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
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

from algorithms import compute_dff_volume, ensure_dir, save_composite_image, point_cloud_from_depth, export_point_cloud, get_mpl_font


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
        # 持有 QImage 的底层数组引用，防止 GC
        self._preview_arr_ref = None

        self._setup_ui()
        self._sig_status.connect(self._on_status)
        self._sig_progress.connect(self.progressBar.setValue)
        self._sig_log.connect(self._append_log)
        self._sig_done.connect(self._on_done)

    # ── UI 构建 ────────────────────────────────────────────────
    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

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
        lbl_dir = QLabel("拍摄方向：Z 轴 高位 → 低位（从上向下），步长始终为负方向")
        lbl_dir.setStyleSheet("color: #1a6a0a; font-size: 11px; padding: 2px 4px;")
        lbl_dir.setWordWrap(True)
        root.addWidget(lbl_dir)

        # ── 融合方式说明 ──
        lbl_fuse = QLabel("融合方式：DFF 焦点融合——逐像素保留最清晰帧的灰度值，输出全焦合成图")
        lbl_fuse.setStyleSheet("color: #555; font-size: 11px; padding: 2px 4px;")
        lbl_fuse.setWordWrap(True)
        root.addWidget(lbl_fuse)

        # ── 高级选项：点云生成（可选） ──
        grp_advanced = QGroupBox("高级选项")
        adv_layout = QVBoxLayout()
        adv_layout.setSpacing(4)

        # 复选框：是否生成点云
        self.chkPointCloud = QCheckBox("同时生成点云数据（用于 3D 测量和导出）")
        self.chkPointCloud.setChecked(False)
        self.chkPointCloud.stateChanged.connect(self._on_pointcloud_toggle)
        adv_layout.addWidget(self.chkPointCloud)

        # 点云参数（默认隐藏）
        self.grpPcParams = QGroupBox()
        self.grpPcParams.setStyleSheet("QGroupBox { border: none; margin: 0; padding: 0; }")
        form_pc = QFormLayout()
        form_pc.setLabelAlignment(Qt.AlignRight)
        form_pc.setContentsMargins(20, 0, 0, 0)
        self.edtZScale = QLineEdit("1.0")
        self.edtMinSharpness = QLineEdit("20.0")
        self.edtZScale.setMaximumWidth(80)
        self.edtMinSharpness.setMaximumWidth(80)
        form_pc.addRow("Z 轴缩放系数:", self.edtZScale)
        form_pc.addRow("最小锐度阈值:", self.edtMinSharpness)
        self.grpPcParams.setLayout(form_pc)
        self.grpPcParams.setVisible(False)
        adv_layout.addWidget(self.grpPcParams)

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

    # ── 信号槽 ─────────────────────────────────────────────────
    def _on_pointcloud_toggle(self, state):
        """复选框切换：显示/隐藏点云参数和按钮"""
        checked = state == Qt.Checked
        self.grpPcParams.setVisible(checked)
        self.bnVisualize.setVisible(checked)
        self.bnExportPly.setVisible(checked)
        # 如果取消勾选，禁用按钮
        if not checked:
            self.bnVisualize.setEnabled(False)
            self.bnExportPly.setEnabled(False)

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

        confirm_msg = (
            "拍摄方向：从上向下\n"
            "Z 轴：{:.3f} mm（高位）→ {:.3f} mm（低位）\n"
            "步长：{:.3f} mm，共 {} 步，每步延时 {:.2f}s\n"
            "融合方式：DFF 焦点融合\n"
        ).format(
            params["z_high"], params["z_low"],
            params["z_step"], params["n_steps"], params["delay"]
        )
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
            emit_log("═══ 一键出图 开始 ═══")
            emit_log("拍摄方向：从上向下（Z 高位→低位）")
            emit_log("Z 高位={:.3f}mm  Z 低位={:.3f}mm  步长={:.3f}mm  步数={}".format(
                z_high, z_low, z_step, n_steps))

            # ── 步骤 1：移动到高位起点 ──
            emit_status("步骤 1/4：移动到高位起点 Z={:.3f}mm...".format(z_high))
            try:
                self.device_controller.move_z_absolute(z_high, feed=300)
                emit_log("  move_z_absolute({:.4f}, F300) 已发送".format(z_high))
            except Exception as exc:
                emit_status("串口发送失败: " + str(exc), "red")
                self._sig_done.emit(False, None)
                return

            # 等待运动台到达起点
            wait_s = max(0.8, abs(z_high) / 5.0 + 0.5)
            t0 = time.time()
            while time.time() - t0 < wait_s:
                if not self._running:
                    emit_status("已停止", "orange")
                    self._sig_done.emit(False, None)
                    return
                time.sleep(0.05)

            # ── 步骤 2：检查相机图像尺寸 ──
            emit_status("步骤 2/4：获取图像尺寸...")
            gray0, width, height = self.device_controller.get_gray_frame()
            if gray0 is None or width == 0 or height == 0:
                emit_status("错误：无法获取相机帧，请检查相机是否正在输出图像", "red")
                self._sig_done.emit(False, None)
                return
            emit_log("  图像尺寸 {}×{}".format(width, height))

            # ── 步骤 3：逐步从上向下扫描 ──
            emit_status("步骤 3/4：从上向下扫描采帧…")
            frames_gray = []
            z_positions = []

            for idx in range(n_steps):
                if not self._running:
                    emit_status("已停止", "orange")
                    self._sig_done.emit(False, None)
                    return

                # 当前 Z 位置（从高到低，step 取负）
                z_cur = round(z_high - idx * z_step, 6)
                z_positions.append(z_cur)

                if idx > 0:
                    # 往低位移动一步
                    try:
                        self.device_controller.move_z_relative(-z_step, feed=300)
                    except Exception as exc:
                        emit_status("串口错误: " + str(exc), "red")
                        self._sig_done.emit(False, None)
                        return
                    time.sleep(delay)

                emit_status("扫描 Z={:.3f}mm  ({}/{})".format(z_cur, idx + 1, n_steps))
                gray, fw, fh = self.device_controller.get_gray_frame()
                if gray is not None and fw == width and fh == height:
                    frames_gray.append(gray.astype(np.float32))
                else:
                    # 尺寸不一致时补零帧（不影响后续融合）
                    frames_gray.append(np.zeros((height, width), dtype=np.float32))
                    emit_log("  [警告] Z={:.3f}mm 帧尺寸异常，用零帧替代".format(z_cur))

                emit_progress(int((idx + 1) / n_steps * 70))

            if not self._running:
                emit_status("已停止", "orange")
                self._sig_done.emit(False, None)
                return

            emit_log("  采帧完成：共 {} 帧，Z 范围 {:.3f}~{:.3f}mm".format(
                len(frames_gray), z_positions[-1], z_positions[0]))

            # ── 步骤 4：DFF 焦点融合 ──
            emit_status("步骤 4/4：DFF 焦点融合中…")
            emit_progress(75)
            depth_map, sharp_map, intensity_map = compute_dff_volume(
                frames_gray, z_positions
            )
            emit_progress(88)

            if intensity_map is None:
                emit_status("错误：融合失败，没有有效图像数据", "red")
                self._sig_done.emit(False, None)
                return

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
                emit_log("  点云生成完成：{:,} 点，覆盖率 {:.1f}%".format(point_count, coverage))

            # ── 自动保存 ──
            save_dir = self.config_manager.effective_save_path()
            ensure_dir(save_dir)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, "composite_{}.bmp".format(ts))
            try:
                save_composite_image(intensity_map, save_path)
                self._last_save_path = save_path
                emit_log("  已保存合成图：{}".format(save_path))
            except Exception as exc:
                emit_log("  [警告] 自动保存失败：{}".format(exc))
                save_path = "（保存失败）"

            emit_progress(100)

            # ── 输出日志摘要 ──
            emit_log("─" * 48)
            emit_log("出图摘要")
            emit_log("  拍摄方向  ：从上向下（Z {:.3f} → {:.3f} mm）".format(
                z_positions[0], z_positions[-1]))
            emit_log("  步数      ：{}".format(len(frames_gray)))
            emit_log("  步长      ：{:.3f} mm（负方向）".format(z_step))
            emit_log("  图像尺寸  ：{}×{}".format(width, height))
            emit_log("  融合方式  ：DFF 焦点融合")
            emit_log("  输出路径  ：{}".format(save_path))
            if gen_pointcloud:
                emit_log("  点云数据  ：{:,} 点（覆盖率 {:.1f}%）".format(point_count, coverage))
            emit_log("═" * 48)

            # 构建状态消息
            status_msg = "出图完成！已保存 {}×{} 合成图\n路径：{}".format(width, height, save_path)
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
        default_name = "composite_{}.bmp".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "另存合成图",
            os.path.join(default_dir, default_name),
            "BMP 图像 (*.bmp);;PNG 图像 (*.png);;全部文件 (*.*)",
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
        if self._point_cloud is None or len(self._point_cloud) == 0:
            QMessageBox.warning(self, "提示", "暂无点云数据，请先勾选「生成点云数据」并执行出图。")
            return
        try:
            import matplotlib
            matplotlib.use("Qt5Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fp = get_mpl_font()
            title_kw = {"fontproperties": fp} if fp else {}
            fig = plt.figure(figsize=(14, 6))
            fig.suptitle("一键出图 - 点云数据", fontsize=14, **title_kw)

            # 左侧：深度图
            ax1 = fig.add_subplot(1, 2, 1)
            if self._depth_map is not None:
                width, height = self._img_size
                ppmm = self.config_manager.pixels_per_mm
                image = ax1.imshow(
                    self._depth_map,
                    cmap="plasma",
                    origin="upper",
                    extent=[
                        -width / (2 * ppmm),
                        width / (2 * ppmm),
                        height / (2 * ppmm),
                        -height / (2 * ppmm),
                    ],
                )
                plt.colorbar(image, ax=ax1).set_label("Depth (mm)")
                ax1.set_title("Depth Map", **title_kw)
                ax1.set_xlabel("X (mm)")
                ax1.set_ylabel("Y (mm)")

            # 右侧：3D 点云
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            points = self._point_cloud
            x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
            # 限制显示点数以提高性能
            max_plot = 60000
            if len(x) > max_plot:
                index = np.random.choice(len(x), max_plot, replace=False)
                x, y, z, intensity = x[index], y[index], z[index], intensity[index]
            scatter = ax2.scatter(x, y, z, c=intensity, cmap="gray", s=0.5, alpha=0.6)
            plt.colorbar(scatter, ax=ax2, label="Intensity")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            ax2.set_zlabel("Z (mm)")
            ax2.set_title("Point Cloud ({:,} pts)".format(len(self._point_cloud)), **title_kw)
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
            "PLY 文件 (*.ply);;CSV 文件 (*.csv);;全部文件 (*.*)",
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
