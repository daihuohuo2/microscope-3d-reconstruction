import os
import threading
import time
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
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

from algorithms import compute_dff_volume, export_point_cloud, get_mpl_font, point_cloud_from_depth


class PointCloudReconDialog(QDialog):
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
        self._point_cloud = None
        self._img_size = (0, 0)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

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
        self.edtMinSharpness = QLineEdit("20.0")
        form_pc.addRow("Z 轴缩放系数:", self.edtZScale)
        form_pc.addRow("最小锐度阈值:", self.edtMinSharpness)
        grp_pc.setLayout(form_pc)
        layout.addWidget(grp_pc)

        self.progressBar = QProgressBar()
        layout.addWidget(self.progressBar)

        self.lblStatus = QLabel("就绪 - 请确认相机已开启采集且串口已连接")
        self.lblStatus.setWordWrap(True)
        self.lblStatus.setStyleSheet("color: gray; padding: 2px;")
        layout.addWidget(self.lblStatus)

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
        self._point_cloud = None
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
            QTimer.singleShot(
                0, lambda: QMessageBox.critical(self, "依赖缺失", "三维重建需要 numpy:\npip install numpy")
            )
            self._running = False
            QTimer.singleShot(0, self._on_worker_done)
            return

        def set_status(message, color="blue"):
            QTimer.singleShot(0, lambda m=message, c=color: self._set_status(m, c))

        def set_progress(value):
            QTimer.singleShot(0, lambda v=value: self.progressBar.setValue(v))

        try:
            z_positions = []
            frames_gray = []
            n_steps = int(round((z_end - z_start) / z_step)) + 1
            z_positions = [z_start + index * z_step for index in range(n_steps)]

            set_status("移动到起始位置 Z={:.3f} mm...".format(z_start))
            self.device_controller.move_z_absolute(z_start, feed=300)
            time.sleep(max(0.8, abs(z_start) / 5.0 + 0.4))

            gray0, width, height = self.device_controller.get_gray_frame()
            if gray0 is None or width == 0 or height == 0:
                set_status("错误：无法获取相机帧，请检查相机是否正在输出图像", "red")
                return

            for idx, z_pos in enumerate(z_positions):
                if not self._running:
                    set_status("已停止", "orange")
                    return
                if idx > 0:
                    self.device_controller.move_z_relative(z_step, feed=300)
                    time.sleep(delay)

                set_status("扫描 Z={:.3f} mm ({}/{})".format(z_pos, idx + 1, n_steps))
                gray, frame_w, frame_h = self.device_controller.get_gray_frame()
                if gray is not None and frame_w == width and frame_h == height:
                    frames_gray.append(gray.astype(np.float32))
                else:
                    frames_gray.append(np.zeros((height, width), dtype=np.float32))
                set_progress(int((idx + 1) / n_steps * 80))

            if not self._running:
                set_status("已停止", "orange")
                return

            set_status("生成深度图...")
            set_progress(82)
            depth_map, sharp_map, intensity_map = compute_dff_volume(frames_gray, z_positions)
            set_progress(88)
            if depth_map is None:
                set_status("错误：没有可用图像数据", "red")
                return

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
            self._point_cloud = point_cloud
            self._img_size = (width, height)
            set_progress(100)
            message = (
                "重建完成\n点数: {:,}   像素覆盖率: {:.1f}%\n"
                "pixels/mm={:.2f}  分辨率: {}x{}".format(
                    len(point_cloud), coverage, self.config_manager.pixels_per_mm, width, height
                )
            )
            set_status(message, "green")
            QTimer.singleShot(0, lambda: self.bnVisualize.setEnabled(True))
            QTimer.singleShot(0, lambda: self.bnExport.setEnabled(True))
        except Exception as exc:
            QTimer.singleShot(0, lambda err=str(exc): self._set_status("重建失败: " + err, "red"))
        finally:
            self._running = False
            QTimer.singleShot(0, self._on_worker_done)

    def _on_worker_done(self):
        self.bnStart.setEnabled(True)
        self.bnStop.setEnabled(False)

    def _visualize_point_cloud(self):
        if self._point_cloud is None or len(self._point_cloud) == 0:
            QMessageBox.warning(self, "提示", "暂无点云数据，请先执行重建。")
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

            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            points = self._point_cloud
            x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
            max_plot = 60000
            if len(x) > max_plot:
                index = np.random.choice(len(x), max_plot, replace=False)
                x, y, z, intensity = x[index], y[index], z[index], intensity[index]
            scatter = ax2.scatter(x, y, z, c=intensity, cmap="gray", s=0.5, alpha=0.6)
            plt.colorbar(scatter, ax=ax2, label="Intensity")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            ax2.set_zlabel("Z (mm)")
            ax2.set_title("Point Cloud ({:,} pts)".format(len(points)), **title_kw)
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
            "PLY 文件 (*.ply);;CSV 文件 (*.csv);;全部文件 (*.*)",
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
