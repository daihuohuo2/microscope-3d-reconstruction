import os
import threading
import time
from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal
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
    QVBoxLayout,
)

from algorithms import (
    build_best_focus_maps,
    export_point_cloud,
    get_mpl_font,
    merge_focus_maps,
    point_cloud_from_depth,
    select_focus_window,
)


class TemporalDepthDialog(QDialog):
    _sig_status = pyqtSignal(str, str)
    _sig_progress = pyqtSignal(int)
    _sig_log = pyqtSignal(str)
    _sig_done = pyqtSignal(bool, str)

    def __init__(self, device_controller, config_manager, parent=None):
        super().__init__(parent)
        self.device_controller = device_controller
        self.config_manager = config_manager
        self.setWindowTitle("连续扫描重建")
        self.setMinimumWidth(500)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self._running = False
        self._worker = None
        self._depth_map = None
        self._sharp_map = None
        self._pc = None
        self._img_size = (0, 0)
        self._setup_ui()
        self._sig_status.connect(self._on_status)
        self._sig_progress.connect(self.progressBar.setValue)
        self._sig_log.connect(self._append_log)
        self._sig_done.connect(self._on_done)

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        grp1 = QGroupBox("第一轮：粗扫参数（连续匀速扫描）")
        form1 = QFormLayout()
        form1.setLabelAlignment(Qt.AlignRight)
        self.edtZ0 = QLineEdit("-2.0")
        self.edtZ1 = QLineEdit("2.0")
        self.edtSpeed = QLineEdit("1.0")
        self.edtItvMs = QLineEdit("80")
        form1.addRow("Z 起始 (mm):", self.edtZ0)
        form1.addRow("Z 结束 (mm):", self.edtZ1)
        form1.addRow("扫描速度 (mm/s):", self.edtSpeed)
        form1.addRow("采帧间隔 (ms):", self.edtItvMs)
        grp1.setLayout(form1)
        root.addWidget(grp1)

        grp2 = QGroupBox("第二轮：嵌套精扫")
        form2 = QFormLayout()
        form2.setLabelAlignment(Qt.AlignRight)
        self.chkNested = QCheckBox("启用嵌套精扫（粗扫结束后自动执行）")
        self.chkNested.setChecked(True)
        self.edtFineSpeed = QLineEdit("0.3")
        self.edtFineItvMs = QLineEdit("50")
        self.edtFinePct = QLineEdit("30")
        form2.addRow(self.chkNested)
        form2.addRow("精扫速度 (mm/s):", self.edtFineSpeed)
        form2.addRow("精扫采帧间隔 (ms):", self.edtFineItvMs)
        form2.addRow("聚焦区间比例 (%):", self.edtFinePct)
        grp2.setLayout(form2)
        root.addWidget(grp2)

        grp3 = QGroupBox("点云参数")
        form3 = QFormLayout()
        form3.setLabelAlignment(Qt.AlignRight)
        self.edtMinSharp = QLineEdit("10.0")
        self.edtZScale = QLineEdit("1.0")
        form3.addRow("最小锐度阈值:", self.edtMinSharp)
        form3.addRow("Z 轴缩放系数:", self.edtZScale)
        grp3.setLayout(form3)
        root.addWidget(grp3)

        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        root.addWidget(self.progressBar)

        self.lblStatus = QLabel("就绪 - 请确认相机已开启采集（串口连接可选）")
        self.lblStatus.setWordWrap(True)
        self.lblStatus.setStyleSheet("color: gray; padding: 2px;")
        root.addWidget(self.lblStatus)

        self.txtLog = QPlainTextEdit()
        self.txtLog.setReadOnly(True)
        self.txtLog.setMaximumHeight(120)
        root.addWidget(self.txtLog)

        row1 = QHBoxLayout()
        self.bnStart = QPushButton("开始扫描")
        self.bnStop = QPushButton("停止")
        self.bnStop.setEnabled(False)
        row1.addWidget(self.bnStart)
        row1.addWidget(self.bnStop)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        self.bnViz = QPushButton("可视化点云")
        self.bnExport = QPushButton("导出点云")
        self.bnViz.setEnabled(False)
        self.bnExport.setEnabled(False)
        row2.addWidget(self.bnViz)
        row2.addWidget(self.bnExport)
        root.addLayout(row2)

        self.bnStart.clicked.connect(self._start)
        self.bnStop.clicked.connect(self._stop)
        self.bnViz.clicked.connect(self._visualize)
        self.bnExport.clicked.connect(self._export)

    def _on_status(self, message, color):
        self.lblStatus.setText(message)
        self.lblStatus.setStyleSheet("color: {}; padding: 2px;".format(color))

    def _append_log(self, message):
        self.txtLog.appendPlainText(message)
        scrollbar = self.txtLog.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_done(self, success, summary):
        self.bnStart.setEnabled(True)
        self.bnStop.setEnabled(False)
        if success:
            self.bnViz.setEnabled(True)
            self.bnExport.setEnabled(True)

    def _parse_params(self):
        z0 = float(self.edtZ0.text().strip())
        z1 = float(self.edtZ1.text().strip())
        speed = float(self.edtSpeed.text().strip())
        itv = float(self.edtItvMs.text().strip()) / 1000.0
        nested = self.chkNested.isChecked()
        fine_speed = float(self.edtFineSpeed.text().strip())
        fine_itv = float(self.edtFineItvMs.text().strip()) / 1000.0
        fine_pct = float(self.edtFinePct.text().strip())
        min_sharp = float(self.edtMinSharp.text().strip())
        z_scale = float(self.edtZScale.text().strip())

        if z0 >= z1:
            raise ValueError("Z 起始必须小于 Z 结束")
        if speed <= 0 or itv <= 0 or fine_speed <= 0 or fine_itv <= 0:
            raise ValueError("速度和采帧间隔必须为正数")
        if not (1 <= fine_pct <= 100):
            raise ValueError("聚焦区间比例须在 1~100 之间")

        sweep_s = (z1 - z0) / speed
        n_est = int(sweep_s / itv) + 1
        if n_est < 3:
            raise ValueError("预估采帧数过少（{}帧），请降低速度或缩小采帧间隔".format(n_est))

        return {
            "z0": z0,
            "z1": z1,
            "speed": speed,
            "itv": itv,
            "nested": nested,
            "fine_speed": fine_speed,
            "fine_itv": fine_itv,
            "fine_pct": fine_pct,
            "min_sharp": min_sharp,
            "z_scale": z_scale,
            "sweep_s": sweep_s,
            "n_est": n_est,
        }

    def _start(self):
        try:
            params = self._parse_params()
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return

        if not self.device_controller.grabbing:
            QMessageBox.warning(self, "错误", "请先开启相机采集！")
            return

        nested_note = (
            "  + 嵌套精扫：速度 {:.2f} mm/s，采帧间隔 {:.0f} ms，聚焦区间前 {:.0f}%\n".format(
                params["fine_speed"], params["fine_itv"] * 1000, params["fine_pct"]
            )
            if params["nested"]
            else "  （不执行嵌套精扫）\n"
        )
        message = (
            "Z 轴将匀速从 {:.2f} mm 扫描到 {:.2f} mm\n"
            "速度 {:.2f} mm/s，预计用时 {:.1f}s，约 {} 帧\n"
            "采帧间隔 {:.0f} ms\n"
            "{}确认开始？".format(
                params["z0"],
                params["z1"],
                params["speed"],
                params["sweep_s"],
                params["n_est"],
                params["itv"] * 1000,
                nested_note,
            )
        )
        if QMessageBox.question(self, "确认", message, QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return

        self._running = True
        self._depth_map = None
        self._sharp_map = None
        self._pc = None
        self.bnStart.setEnabled(False)
        self.bnStop.setEnabled(True)
        self.bnViz.setEnabled(False)
        self.bnExport.setEnabled(False)
        self.progressBar.setValue(0)
        self.txtLog.clear()
        self._sig_status.emit("准备中...", "blue")

        self._worker = threading.Thread(target=self._worker_fn, args=(params,), daemon=True)
        self._worker.start()

    def _stop(self):
        self._running = False
        self._sig_status.emit("停止请求已发送...", "orange")

    def _gcode(self, cmd):
        try:
            self.device_controller.send_gcode(cmd)
            self._sig_log.emit("  >> " + cmd.strip())
            return True
        except Exception as exc:
            self._sig_log.emit("  [串口跳过/失败: {}] {}".format(cmd.strip(), exc))
            return False

    def _do_sweep(self, z0, z1, speed, itv, label="粗扫"):
        sweep_s = abs(z1 - z0) / speed
        self._sig_status.emit("{} - 移动到起始 Z={:.3f}mm...".format(label, z0), "blue")
        self._gcode("G90\n")
        self._gcode("G1 Z{:.4f} F300\n".format(z0))

        wait = max(0.8, abs(z0) / 5.0 + 0.5)
        start_wait = time.time()
        while time.time() - start_wait < wait:
            if not self._running:
                return None, None
            time.sleep(0.05)

        gray0, width, height = self.device_controller.get_gray_frame()
        if gray0 is None or width == 0 or height == 0:
            self._sig_status.emit("错误：无法获取相机帧", "red")
            return None, None
        self._sig_log.emit("  图像尺寸 {}x{}".format(width, height))

        self._sig_status.emit("{} - Z 轴开始匀速扫描，速度 {:.2f} mm/s...".format(label, speed), "blue")
        self._gcode("G1 Z{:.4f} F{:.1f}\n".format(z1, speed * 60.0))

        t_start = time.time()
        next_cap = t_start
        frames_gray = []
        z_list = []
        z_est = z0

        while self._running:
            now = time.time()
            elapsed = now - t_start
            z_est = z0 + (z1 - z0) * min(elapsed / sweep_s, 1.0)
            if now >= next_cap:
                gray, frame_w, frame_h = self.device_controller.get_gray_frame()
                if gray is not None and frame_w == width and frame_h == height:
                    frames_gray.append(gray)
                    z_list.append(z_est)
                next_cap += itv

            frac = min(elapsed / sweep_s, 1.0)
            if label == "粗扫":
                self._sig_progress.emit(int(frac * 45))
            else:
                self._sig_progress.emit(50 + int(frac * 30))

            if elapsed >= sweep_s + 0.3:
                break
            time.sleep(max(0.0, next_cap - time.time()))

        self._gcode("G1 Z{:.4f} F300\n".format(z_est if z_list else z1))
        self._sig_log.emit(
            "  {} 结束：采集 {} 帧，Z 估算范围 {:.3f}~{:.3f} mm".format(
                label,
                len(frames_gray),
                min(z_list) if z_list else z0,
                max(z_list) if z_list else z1,
            )
        )
        return frames_gray, z_list

    def _worker_fn(self, params):
        try:
            self._sig_log.emit("═══ 开始 连续扫描重建 ═══")
            self._sig_log.emit(
                "粗扫: Z {:.2f}->{:.2f}mm  速度 {:.2f}mm/s  采帧 {:.0f}ms".format(
                    params["z0"], params["z1"], params["speed"], params["itv"] * 1000
                )
            )

            self._sig_status.emit("第一轮 粗扫...", "blue")
            frames1, zlist1 = self._do_sweep(params["z0"], params["z1"], params["speed"], params["itv"], "粗扫")
            if not self._running or frames1 is None or len(frames1) < 3:
                self._sig_status.emit(
                    "粗扫{}".format("已停止" if not self._running else "帧不足，请检查相机"), "orange"
                )
                self._sig_done.emit(False, "")
                return

            depth_map, sharp_map, gray_map = build_best_focus_maps(frames1, zlist1)
            if depth_map is None:
                self._sig_status.emit("锐度栈生成失败", "red")
                self._sig_done.emit(False, "")
                return

            height, width = depth_map.shape
            self._img_size = (width, height)
            self._sig_progress.emit(52)

            if params["nested"] and self._running:
                self._sig_log.emit("── 嵌套精扫开始 ──────────────────────────────")
                fine_z0, fine_z1 = select_focus_window(zlist1, frames1, params["fine_pct"])
                self._sig_log.emit(
                    "  精扫区间: {:.3f} ~ {:.3f} mm  (Delta {:.3f} mm)".format(
                        fine_z0, fine_z1, fine_z1 - fine_z0
                    )
                )
                frames2, zlist2 = self._do_sweep(
                    fine_z0, fine_z1, params["fine_speed"], params["fine_itv"], "精扫"
                )
                if frames2 and len(frames2) >= 2 and self._running:
                    depth_f, sharp_f, gray_f = build_best_focus_maps(frames2, zlist2)
                    updated = merge_focus_maps(depth_map, sharp_map, gray_map, depth_f, sharp_f, gray_f)
                    self._sig_log.emit("个像素由精扫更新".format(updated))
                else:
                    self._sig_log.emit("  精扫帧不足或已停止，跳过融合")
            elif not params["nested"]:
                self._sig_log.emit("（未启用嵌套精扫）")

            if not self._running:
                self._sig_status.emit("已停止", "orange")
                self._sig_done.emit(False, "")
                return

            self._sig_progress.emit(85)
            self._sig_status.emit("生成点云...", "blue")
            point_cloud, coverage = point_cloud_from_depth(
                depth_map,
                sharp_map,
                gray_map,
                self.config_manager.pixels_per_mm,
                params["min_sharp"],
                params["z_scale"],
            )
            if len(point_cloud) == 0:
                self._sig_status.emit(
                    "没有找到有效点！请降低最小锐度阈值（当前 {:.1f}）".format(params["min_sharp"]),
                    "red",
                )
                self._sig_done.emit(False, "")
                return

            self._pc = point_cloud
            self._depth_map = depth_map
            self._sharp_map = sharp_map
            self._sig_progress.emit(100)
            summary = (
                "扫描完成\n点数: {:,}   覆盖率: {:.1f}%\n"
                "pixels/mm={:.2f}  图像 {}x{}".format(
                    len(point_cloud), coverage, self.config_manager.pixels_per_mm, width, height
                )
            )
            self._sig_status.emit(summary, "green")
            self._sig_log.emit("═══ 完成：{:,} 点，覆盖率 {:.1f}% ═══".format(len(point_cloud), coverage))
            self._sig_done.emit(True, summary)
        except Exception as exc:
            self._sig_status.emit("发生异常: " + str(exc), "red")
            self._sig_log.emit("[ERROR] " + str(exc))
            self._sig_done.emit(False, "")
        finally:
            self._running = False

    def _visualize(self):
        if self._pc is None or len(self._pc) == 0:
            QMessageBox.warning(self, "提示", "暂无点云数据，请先执行扫描重建。")
            return
        try:
            import matplotlib

            matplotlib.use("Qt5Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fp = get_mpl_font()
            title_kw = {"fontproperties": fp} if fp else {}
            fig = plt.figure(figsize=(16, 6))
            fig.suptitle("连续扫描重建", fontsize=14, **title_kw)

            ax1 = fig.add_subplot(1, 3, 1)
            ppmm = self.config_manager.pixels_per_mm
            width, height = self._img_size
            image = ax1.imshow(
                self._depth_map,
                cmap="plasma",
                origin="upper",
                extent=[-width / (2 * ppmm), width / (2 * ppmm), height / (2 * ppmm), -height / (2 * ppmm)],
            )
            plt.colorbar(image, ax=ax1).set_label("Depth (mm)")
            ax1.set_title("Depth Map", **title_kw)
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")

            ax2 = fig.add_subplot(1, 3, 2)
            image2 = ax2.imshow(np.log1p(self._sharp_map), cmap="hot", origin="upper")
            plt.colorbar(image2, ax=ax2).set_label("log(1+sharpness)")
            ax2.set_title("Sharpness Map (log)", **title_kw)

            ax3 = fig.add_subplot(1, 3, 3, projection="3d")
            x, y, z, intensity = self._pc[:, 0], self._pc[:, 1], self._pc[:, 2], self._pc[:, 3]
            max_pts = 60000
            if len(x) > max_pts:
                index = np.random.choice(len(x), max_pts, replace=False)
                x, y, z, intensity = x[index], y[index], z[index], intensity[index]
            scatter = ax3.scatter(x, y, z, c=intensity, cmap="gray", s=0.5, alpha=0.6)
            plt.colorbar(scatter, ax=ax3, label="Intensity")
            ax3.set_xlabel("X (mm)")
            ax3.set_ylabel("Y (mm)")
            ax3.set_zlabel("Z (mm)")
            ax3.set_title("Point Cloud ({:,} pts)".format(len(self._pc)), **title_kw)
            plt.tight_layout()
            plt.show()
        except ImportError:
            QMessageBox.warning(self, "依赖缺失", "可视化需要 matplotlib:\npip install matplotlib")
        except Exception as exc:
            QMessageBox.warning(self, "可视化错误", str(exc))

    def _export(self):
        if self._pc is None or len(self._pc) == 0:
            QMessageBox.warning(self, "提示", "暂无点云数据，请先执行扫描重建。")
            return
        default_name = "temporal_depth_{}.ply".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
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
                self._pc,
                self.config_manager.pixels_per_mm,
                "Generated by BasicDemo TemporalDepth",
            )
            QMessageBox.information(
                self, "导出完成", "已导出 {:,} 个点（{}）\n{}".format(len(self._pc), file_type.upper(), file_path)
            )
        except Exception as exc:
            QMessageBox.warning(self, "导出错误", str(exc))
