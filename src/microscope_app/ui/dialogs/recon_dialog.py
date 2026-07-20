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

from ...core.algorithms import (
    StreamingDFFAccumulator,
    estimate_translation_shift,
    export_point_cloud,
    get_mpl_font,
    point_cloud_from_surface,
    prepare_depth_surface,
    save_output_bundle,
    warp_frame_translation,
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
        self._surface_depth = None
        self._surface_mask = None
        self._confidence_map = None
        self._recon_quality = {}
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
        self.edtZStep = QLineEdit("0.05")
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

        quality_tip = QLabel(
            "高质量建议：先用 0.05 mm 粗扫确定范围，再用 0.01–0.02 mm 精扫；"
            "反光样品请降低曝光或使用漫反射/偏振照明。"
        )
        quality_tip.setWordWrap(True)
        quality_tip.setStyleSheet("color: #4f6173; font-size: 11px; padding: 2px 4px;")
        layout.addWidget(quality_tip)

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
        self._surface_depth = None
        self._surface_mask = None
        self._confidence_map = None
        self._recon_quality = {}
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

            # 流式 DFF：保留最佳/次佳峰值与峰值相邻层，同时支持亚步距拟合。
            accumulator = StreamingDFFAccumulator(
                image_shape=(height, width),
                color_enabled=color0 is not None,
                focus_window_size=11,
            )
            best_single_frame = None
            best_single_color = None
            best_single_z = None
            best_single_score = None
            previous_raw_gray = None
            cumulative_dx = 0.0
            cumulative_dy = 0.0
            alignment_offsets = []

            # 原始帧边采集边落盘，避免将整个 Z-stack 长时间占在内存中。
            save_dir = self.config_manager.effective_save_path()
            folder_name = "{}_recon".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
            out_dir = os.path.join(save_dir, folder_name)
            suffix = 2
            while os.path.exists(out_dir):
                out_dir = os.path.join(save_dir, "{}_{}".format(folder_name, suffix))
                suffix += 1
            frames_dir = os.path.join(out_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            from ...core.algorithms import save_color_image, save_composite_image

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
                    color_f = color.copy() if color is not None else None
                else:
                    frame_f = np.zeros((height, width), dtype=np.float32)
                    color_f = None

                frame_name = "frame_{:03d}_z{:+.3f}mm.tif".format(idx + 1, float(z_pos))
                frame_path = os.path.join(frames_dir, frame_name)
                if color_f is not None:
                    save_color_image(color_f, frame_path)
                else:
                    save_composite_image(frame_f, frame_path)

                # 相邻层位移通常很小，递推配准比用首层（可能严重失焦）
                # 直接作为固定参考更稳定。
                aligned_gray = frame_f
                aligned_color = color_f
                if previous_raw_gray is not None:
                    delta_dx, delta_dy = estimate_translation_shift(
                        previous_raw_gray,
                        frame_f,
                        max_width=768,
                        max_shift_px=24.0,
                    )
                    cumulative_dx += delta_dx
                    cumulative_dy += delta_dy
                    aligned_gray, aligned_color = warp_frame_translation(
                        frame_f,
                        color_f,
                        cumulative_dx,
                        cumulative_dy,
                    )
                previous_raw_gray = frame_f
                alignment_offsets.append((float(cumulative_dx), float(cumulative_dy)))

                sharp = accumulator.update(aligned_gray, z_pos, aligned_color)
                single_score = float(np.mean(sharp))
                if best_single_score is None or single_score < best_single_score:
                    best_single_score = single_score
                    best_single_z = z_pos
                    best_single_frame = aligned_gray.copy()
                    best_single_color = aligned_color.copy() if aligned_color is not None else None
                set_progress(int((idx + 1) / n_steps * 80))

            if not self._running:
                set_status("已停止", "orange")
                return

            if accumulator.best_score is None:
                set_status("错误：没有可用图像数据", "red")
                return

            set_status("亚步距深度拟合与置信度估计...")
            dff_result = accumulator.finalize()
            depth_map = dff_result["depth_map"]
            sharp_map = dff_result["sharp_map"]
            confidence_map = dff_result["confidence_map"]
            intensity_map = dff_result["intensity_map"]
            best_color = dff_result["color_map"]
            set_progress(82)

            try:
                min_sharp = float(self.edtMinSharpness.text().strip())
            except ValueError:
                min_sharp = 20.0
            try:
                z_scale = float(self.edtZScale.text().strip())
            except ValueError:
                z_scale = 1.0

            set_status("置信度筛选与边缘保持表面优化...")
            surface_depth, surface_mask, quality = prepare_depth_surface(
                depth_map=depth_map,
                sharp_map=sharp_map,
                intensity_map=intensity_map,
                min_sharp=min_sharp,
                confidence_map=confidence_map,
                z_step=z_step,
            )
            focus_indices = dff_result["focus_index_map"]
            quality["boundary_percent"] = 100.0 * float(
                np.mean((focus_indices == 0) | (focus_indices == n_steps - 1))
            )
            quality["median_confidence"] = float(np.median(confidence_map))
            quality["saturated_percent"] = 100.0 * float(
                np.mean(np.max(best_color, axis=2) >= 250)
            )
            quality["max_alignment_px"] = float(
                np.max(np.abs(np.asarray(alignment_offsets, dtype=np.float32)))
            ) if alignment_offsets else 0.0
            point_cloud, coverage = point_cloud_from_surface(
                surface_depth=surface_depth,
                surface_mask=surface_mask,
                intensity_map=intensity_map,
                pixels_per_mm=self.config_manager.pixels_per_mm,
                z_scale=z_scale,
            )
            self._depth_map = depth_map
            self._sharp_map = sharp_map
            self._intensity_map = intensity_map
            self._point_cloud = point_cloud
            self._surface_depth = surface_depth
            self._surface_mask = surface_mask
            self._confidence_map = confidence_map
            self._recon_quality = quality
            self._img_size = (width, height)

            set_status("保存文件中…")
            set_progress(90)
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
                z_positions=z_positions,
                output_dir=out_dir,
                preexisting_frames_dir=frames_dir,
                confidence_map=confidence_map,
                surface_depth=surface_depth,
                surface_mask=surface_mask,
                alignment_offsets=alignment_offsets,
                quality_metrics=quality,
            )
            self._last_output_paths = output_paths

            set_progress(100)
            out_dir = output_paths.get("output_dir", save_dir)
            message = (
                "完成：{}x{}，点云 {:,} 点，表面覆盖率 {:.1f}%，"
                "可靠锚点 {:.1f}%，边界焦层 {:.1f}%\n目录：{}"
            ).format(
                width,
                height,
                len(point_cloud),
                coverage,
                quality.get("anchor_percent", 0.0),
                quality.get("boundary_percent", 0.0),
                out_dir,
            )
            warnings = []
            if quality["boundary_percent"] > 5.0:
                warnings.append("较多像素落在扫描边界，建议向两端扩大 Z 范围")
            if quality["saturated_percent"] > 1.0:
                warnings.append("存在明显过曝，建议降低曝光或改用漫反射/偏振照明")
            if warnings:
                message += "\n质量提示：" + "；".join(warnings)
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
        if self._surface_depth is None or self._point_cloud is None or len(self._point_cloud) == 0:
            QMessageBox.warning(self, "提示", "暂无点云/深度图数据，请先执行重建。")
            return
        try:
            import matplotlib

            matplotlib.use("Qt5Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fp = get_mpl_font()
            ppmm = float(self.config_manager.pixels_per_mm)
            ppmm = ppmm if ppmm > 0 else 1.0
            width, height = self._img_size
            depth = np.asarray(self._surface_depth, dtype=np.float32)
            mask = np.asarray(self._surface_mask, dtype=bool) & np.isfinite(depth)
            if not np.any(mask):
                raise ValueError("表面掩码中没有可视点")

            # 规则网格降采样保留轮廓和局部起伏，不使用随机抽点。
            target_points = 600000
            stride = max(1, int(np.ceil(np.sqrt(float(np.count_nonzero(mask)) / target_points))))
            sampled_depth = depth[::stride, ::stride]
            sampled_mask = mask[::stride, ::stride] & np.isfinite(sampled_depth)
            # Preview-only smoothing operates after decimation.  It removes
            # isolated visual needles without changing the exported metric cloud.
            try:
                import cv2

                preview_fill = float(np.nanmedian(sampled_depth[sampled_mask]))
                preview_depth = np.where(sampled_mask, sampled_depth, preview_fill).astype(np.float32)
                preview_depth = cv2.medianBlur(preview_depth, 5)
                preview_span = float(
                    np.nanpercentile(preview_depth[sampled_mask], 95.0)
                    - np.nanpercentile(preview_depth[sampled_mask], 5.0)
                )
                preview_depth = cv2.bilateralFilter(
                    preview_depth,
                    d=7,
                    sigmaColor=max(preview_span * 0.035, 0.01),
                    sigmaSpace=4.0,
                    borderType=cv2.BORDER_REFLECT101,
                )
                sampled_depth = np.where(sampled_mask, preview_depth, np.nan)
            except Exception:
                pass
            sy, sx = np.where(sampled_mask)
            x = (sx.astype(np.float32) * stride - (width - 1) * 0.5) / ppmm
            y = ((height - 1) * 0.5 - sy.astype(np.float32) * stride) / ppmm
            z_raw = sampled_depth[sy, sx].astype(np.float32)
            z_base = float(np.percentile(z_raw, 1.0))
            try:
                z_scale = max(0.01, float(self.edtZScale.text().strip()))
            except ValueError:
                z_scale = 1.0
            z = (z_raw - z_base) * z_scale

            color_low = float(np.percentile(z, 1.0))
            color_high = float(np.percentile(z, 99.0))
            if color_high <= color_low + 1e-9:
                color_high = color_low + 1.0

            fig = plt.figure(figsize=(12.8, 8.0), facecolor="#000000")
            ax = fig.add_subplot(111, projection="3d", facecolor="#000000")
            ax.scatter(
                x,
                y,
                z,
                c=z,
                cmap="turbo",
                vmin=color_low,
                vmax=color_high,
                s=2.0,
                alpha=0.96,
                linewidths=0,
                edgecolors="none",
                depthshade=False,
            )
            ax.set_axis_off()
            ax.grid(False)
            try:
                ax.set_proj_type("persp", focal_length=0.85)
            except Exception:
                pass
            ax.view_init(elev=28, azim=-62)

            x_span = max(float(np.ptp(x)), 1e-6)
            y_span = max(float(np.ptp(y)), 1e-6)
            z_span = max(float(np.ptp(z)), 1e-6)
            # 非常平的样品仅在预览纵横比上做最小增强，导出坐标不变。
            visual_z_span = max(z_span, min(x_span, y_span) * 0.14)
            ax.set_box_aspect((x_span, y_span, visual_z_span), zoom=1.25)
            ax.set_xlim(float(np.min(x)), float(np.max(x)))
            ax.set_ylim(float(np.min(y)), float(np.max(y)))
            ax.set_zlim(color_low, color_high)

            title = "高度伪彩三维点云  ·  {:,} 点".format(len(self._point_cloud))
            fig.text(
                0.5,
                0.955,
                title,
                ha="center",
                va="top",
                color="white",
                fontsize=15,
                fontproperties=fp if fp else None,
            )
            fig.text(
                0.5,
                0.025,
                "鼠标左键旋转  ·  右键缩放  ·  颜色由蓝到红表示由低到高",
                ha="center",
                color="#b8c1cc",
                fontsize=10,
                fontproperties=fp if fp else None,
            )
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.045, top=0.94)
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

