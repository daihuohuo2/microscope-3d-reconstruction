import os
import threading
import time
from datetime import datetime

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from algorithms import (
    export_point_cloud,
    get_mpl_font,
    point_cloud_from_depth,
    save_output_bundle,
)


class PointCloudReconDialog(QDialog):
    _sig_status     = pyqtSignal(str, str)
    _sig_progress   = pyqtSignal(int)
    _sig_recon_done = pyqtSignal(bool)

    def __init__(self, device_controller, config_manager, parent=None):
        super().__init__(parent)
        self.device_controller = device_controller
        self.config_manager = config_manager
        self.setWindowTitle("三维重建 - 点云数据重建")
        self.setMinimumWidth(460)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self._running = False
        self._worker_thread = None
        self._depth_map = None
        self._sharp_map = None
        self._intensity_map = None
        self._point_cloud = None
        self._img_size = (0, 0)
        self._last_output_paths = {}
        self._setup_ui()
        self._sig_status.connect(self._set_status)
        self._sig_progress.connect(self.progressBar.setValue)
        self._sig_recon_done.connect(self._on_worker_finished)
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
            span = abs(float(self.edtZEnd.text().strip()) - float(self.edtZStart.text().strip()))
        except ValueError:
            span = 4.0
        span = max(0.1, span)
        self.edtZStart.setText("{:.3f}".format(z))
        self.edtZEnd.setText("{:.3f}".format(z + span))
        self._refresh_z_label()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.lblZPos = QLabel("Z 当前位置: -- mm")
        self.lblZPos.setAlignment(Qt.AlignCenter)
        self.lblZPos.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #0078d7;"
            "background: #ffffff; border: 1px solid #0078d7; border-radius: 4px; padding: 4px;"
        )
        layout.addWidget(self.lblZPos)

        grp_scan = QGroupBox("扫描参数（Z 轴）")
        form_scan = QFormLayout()
        form_scan.setLabelAlignment(Qt.AlignRight)
        self.edtZStart = QLineEdit("-2.0")
        self.edtZEnd = QLineEdit("2.0")
        self.edtZStep = QLineEdit("0.1")
        self.edtDelay = QLineEdit("0.15")
        form_scan.addRow("Z 起始位置 (mm):", self.edtZStart)
        form_scan.addRow("Z 结束位置 (mm):", self.edtZEnd)
        form_scan.addRow("Z 步长 (mm):", self.edtZStep)
        form_scan.addRow("每步延时 (s):", self.edtDelay)
        grp_scan.setLayout(form_scan)
        layout.addWidget(grp_scan)

        grp_pc = QGroupBox("点云参数")
        form_pc = QFormLayout()
        form_pc.setLabelAlignment(Qt.AlignRight)
        self.edtZScale = QLineEdit("1.0")
        self.edtMinSharpness = QLineEdit("5.0")
        form_pc.addRow("Z 轴缩放系数:", self.edtZScale)
        form_pc.addRow("最小锐度(%, 0-100):", self.edtMinSharpness)
        grp_pc.setLayout(form_pc)
        layout.addWidget(grp_pc)

        self.progressBar = QProgressBar()
        layout.addWidget(self.progressBar)

        self.lblStatus = QLabel("就绪 - 请确认相机已开启采集且串口已连接")
        self.lblStatus.setWordWrap(True)
        self.lblStatus.setStyleSheet("color: gray; padding: 2px;")
        layout.addWidget(self.lblStatus)

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
        layout.addLayout(save_row)

        row1 = QHBoxLayout()
        self.bnStart = QPushButton("开始重建")
        self.bnStop = QPushButton("停止")
        self.bnStop.setEnabled(False)
        row1.addWidget(self.bnStart)
        row1.addWidget(self.bnStop)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.bnVisualize = QPushButton("可视化点云")
        self.bnExport = QPushButton("导出点云")
        self.bnVisualize.setEnabled(False)
        self.bnExport.setEnabled(False)
        row2.addWidget(self.bnVisualize)
        row2.addWidget(self.bnExport)
        layout.addLayout(row2)

        self.bnStart.clicked.connect(self._start_reconstruction)
        self.bnStop.clicked.connect(self._stop_reconstruction)
        self.bnVisualize.clicked.connect(self._visualize_point_cloud)
        self.bnExport.clicked.connect(self._export_point_cloud)

    def _browse_save_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存路径", self.config_manager.effective_save_path())
        if path:
            self.config_manager.save_path = path
            self._edt_save_path.setText(path)

    def _set_status(self, message, color="gray"):
        self.lblStatus.setText(message)
        self.lblStatus.setStyleSheet("color: {}; padding: 2px;".format(color))

    def _start_reconstruction(self):
        try:
            z_start = float(self.edtZStart.text().strip())
            z_end = float(self.edtZEnd.text().strip())
            z_step = float(self.edtZStep.text().strip())
            delay = float(self.edtDelay.text().strip())
            if z_step <= 0:
                raise ValueError("步长必须为正数")
            if z_start >= z_end:
                raise ValueError("起始位置必须小于结束位置")
            if delay < 0:
                raise ValueError("延时不能为负数")
            n_steps = int(round((z_end - z_start) / z_step)) + 1
            if n_steps < 2:
                raise ValueError("步数过少（{}步），请减小步长或增大扫描范围".format(n_steps))
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return

        if not self.device_controller.grabbing:
            QMessageBox.warning(self, "错误", "请先开启相机采集！")
            return
        if not self.device_controller.serial_connected:
            QMessageBox.warning(self, "错误", "请先连接串口（用于控制 Z 轴运动）！")
            return

        reply = QMessageBox.question(
            self,
            "确认",
            "将扫描 Z 轴 {:.2f} -> {:.2f} mm，步长 {:.3f} mm，共 {} 步。\n确认开始？".format(
                z_start, z_end, z_step, n_steps
            ),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._running = True
        self._depth_map = None
        self._sharp_map = None
        self._intensity_map = None
        self._point_cloud = None
        self._last_output_paths = {}
        self.bnStart.setEnabled(False)
        self.bnStop.setEnabled(True)
        self.bnVisualize.setEnabled(False)
        self.bnExport.setEnabled(False)
        self.progressBar.setValue(0)
        self._set_status("准备中...", "blue")

        self._worker_thread = threading.Thread(
            target=self._recon_worker, args=(z_start, z_end, z_step, delay), daemon=True
        )
        self._worker_thread.start()

    def _stop_reconstruction(self):
        self._running = False
        self._set_status("停止请求已发送，等待当前步完成...", "orange")

    def _recon_worker(self, z_start, z_end, z_step, delay):
        try:
            import numpy as np
        except ImportError:
            self._sig_status.emit("依赖缺失：三维重建需要 numpy: pip install numpy", "red")
            self._running = False
            self._sig_recon_done.emit(False)
            return

        def set_status(message, color="blue"):
            self._sig_status.emit(message, color)

        def set_progress(value):
            self._sig_progress.emit(value)

        _success = False
        try:
            n_steps = int(round((z_end - z_start) / z_step)) + 1
            z_positions = [z_start + index * z_step for index in range(n_steps)]

            set_status("移动到起始位置 Z={:.3f} mm...".format(z_start))
            self.device_controller.move_z_absolute_wait(z_start, feed=300)
            time.sleep(0.1)

            gray0, color0, width, height = self.device_controller.get_gray_color_frame()
            if gray0 is None or width == 0 or height == 0:
                set_status("错误：无法获取相机帧，请检查相机是否正在输出图像", "red")
                return

            # 增量 DFF：逐帧处理，不存储全部帧，内存占用从 N×帧大小 降到 3×帧大小
            from algorithms import compute_laplacian_sharpness_map as _lap
            best_sharp = None
            best_z_map = None
            best_gray  = None
            best_color = None
            improve_margin = 0.08
            best_single_frame = None
            best_single_color = None
            best_single_z = None
            best_single_score = None
            all_frames_gray = []
            all_frames_color = []

            for idx, z_pos in enumerate(z_positions):
                if not self._running:
                    set_status("已停止", "orange")
                    return
                if idx > 0:
                    self.device_controller.move_z_relative_wait(z_step, feed=300)
                    time.sleep(delay)

                set_status("扫描 Z={:.3f} mm ({}/{})".format(z_pos, idx + 1, n_steps))
                gray, color, frame_w, frame_h = self.device_controller.get_gray_color_frame()
                if gray is not None and frame_w == width and frame_h == height:
                    frame_f = gray.astype(np.float32)
                    color_f = color
                else:
                    frame_f = np.zeros((height, width), dtype=np.float32)
                    color_f = None
                all_frames_gray.append(frame_f.copy())
                all_frames_color.append(color_f.copy() if color_f is not None else None)
                sharp = _lap(frame_f)
                single_score = float(np.mean(sharp))
                if best_single_score is None or single_score < best_single_score:
                    best_single_score = single_score
                    best_single_z = z_pos
                    best_single_frame = frame_f.copy()
                    best_single_color = color_f.copy() if color_f is not None else None
                if best_sharp is None:
                    best_sharp = sharp.copy()
                    best_z_map = np.full((height, width), z_pos, dtype=np.float32)
                    best_gray  = frame_f.copy()
                    best_color = color_f.copy() if color_f is not None else None
                else:
                    mask = sharp > (best_sharp * (1.0 + improve_margin))
                    best_sharp[mask] = sharp[mask]
                    best_z_map[mask] = z_pos
                    best_gray[mask]  = frame_f[mask]
                    if best_color is not None and color_f is not None:
                        best_color[mask] = color_f[mask]
                set_progress(int((idx + 1) / n_steps * 80))

            if not self._running:
                set_status("已停止", "orange")
                return

            if best_sharp is None:
                set_status("错误：没有可用图像数据", "red")
                return

            depth_map     = best_z_map
            sharp_map     = best_sharp
            intensity_map = best_gray
            set_progress(82)

            try:
                min_sharp = float(self.edtMinSharpness.text().strip())
            except ValueError:
                min_sharp = 20.0
            try:
                z_scale = float(self.edtZScale.text().strip())
            except ValueError:
                z_scale = 1.0

            point_cloud, coverage = point_cloud_from_depth(
                depth_map,
                sharp_map,
                intensity_map,
                self.config_manager.pixels_per_mm,
                min_sharp,
                z_scale,
            )
            self._depth_map = depth_map
            self._sharp_map = sharp_map
            self._intensity_map = intensity_map
            self._point_cloud = point_cloud
            self._img_size = (width, height)

            set_status("保存文件中…")
            set_progress(90)
            save_dir = self.config_manager.effective_save_path()
            output_paths = save_output_bundle(
                save_dir,
                "recon",
                intensity_map,
                depth_map,
                point_cloud,
                self.config_manager.pixels_per_mm,
                params={
                    "z0": z_start,
                    "z1": z_end,
                    "step": z_step,
                    "zscale": z_scale,
                    "sharp": min_sharp,
                },
                z_scale=z_scale,
                comment="Generated by BasicDemo 3D Reconstruction",
                reference_map=best_single_frame,
                reference_label="worst Z={:.3f} mm".format(best_single_z) if best_single_z is not None else "worst single frame",
                color_map=best_color,
                reference_color_map=best_single_color,
                frames_gray=all_frames_gray,
                z_positions=z_positions,
                frames_color=all_frames_color,
            )
            self._last_output_paths = output_paths

            set_progress(100)
            message = "完成：{}x{}，点云 {:,} 点，覆盖率 {:.1f}%\n目录：{}".format(
                width, height, len(point_cloud), coverage, save_dir
            )
            set_status(message, "green")
            _success = True
        except Exception as exc:
            set_status("重建失败: " + str(exc), "red")
        finally:
            self._running = False
            self._sig_recon_done.emit(_success)

    def _on_worker_finished(self, success):
        self.bnStart.setEnabled(True)
        self.bnStop.setEnabled(False)
        if success:
            self.bnVisualize.setEnabled(True)
            self.bnExport.setEnabled(True)

    def _visualize_point_cloud(self):
        if self._depth_map is None or self._point_cloud is None or len(self._point_cloud) == 0:
            QMessageBox.warning(self, "提示", "暂无点云/深度图数据，请先执行重建。")
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
            fig.suptitle("三维重建结果 - 点云数据", fontsize=14, **title_kw)

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
            # 等深度轮廓线
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

    def _export_point_cloud(self):
        if self._point_cloud is None or len(self._point_cloud) == 0:
            QMessageBox.warning(self, "提示", "暂无点云数据，请先执行重建。")
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
                "Generated by BasicDemo 3D Reconstruction",
            )
            QMessageBox.information(
                self,
                "导出完成",
                "已导出 {:,} 个点（{}）\n{}".format(len(self._point_cloud), file_type.upper(), file_path),
            )
        except Exception as exc:
            QMessageBox.warning(self, "导出错误", str(exc))

