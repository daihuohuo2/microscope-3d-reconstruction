import os
import threading
import time
from datetime import datetime

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMainWindow, QMessageBox

from sdk.MvErrorDefine_const import MV_E_PARAMETER, MV_OK
from algorithms import compute_sharpness_score, ensure_dir, phase_correlation_shift
from config_manager import ConfigManager
from device_controller import DeviceController, SERIAL_AVAILABLE, to_hex_str
from dialogs import PointCloudReconDialog, TemporalDepthDialog, OneClickDialog
from overlays import ResizeFilter, ScaleBarOverlay
from ui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.settings_file = os.path.join(self.script_dir, "setting.ini")
        self.config_manager = ConfigManager(self.settings_file, self.script_dir)
        self.device_controller = DeviceController()
        self.device_controller.initialize_sdk()

        self.auto_capture_running = False
        self.autofocus_running = False
        self.auto_calib_running = False
        self.dark_frame_captured = False
        self._cam_img_width = 0
        self._recon3d_dialog = None
        self._temporal_depth_dialog = None
        self._one_click_dialog = None
        self._cleaned_up = False

        self.scale_overlay = ScaleBarOverlay(self.ui.widgetDisplay)
        self._resize_filter = ResizeFilter(self.scale_overlay)
        self.ui.widgetDisplay.installEventFilter(self._resize_filter)

        self._bind_signals()
        self._create_menu()
        self.load_settings()
        self.scale_overlay.set_visible(self.ui.chkShowScaleBar.isChecked())
        self.enable_controls()

    def _bind_signals(self):
        self.ui.bnEnum.clicked.connect(self.enum_devices)
        self.ui.bnOpen.clicked.connect(self.open_device)
        self.ui.bnClose.clicked.connect(self.close_device)
        self.ui.bnStart.clicked.connect(self.start_grabbing)
        self.ui.bnStop.clicked.connect(self.stop_grabbing)

        self.ui.bnSoftwareTrigger.clicked.connect(self.trigger_once)
        self.ui.radioTriggerMode.clicked.connect(self.set_software_trigger_mode)
        self.ui.radioContinueMode.clicked.connect(self.set_continue_mode)

        self.ui.bnGetParam.clicked.connect(self.get_param)
        self.ui.bnSetParam.clicked.connect(self.set_param)

        self.ui.bnSaveImage.clicked.connect(self.save_bmp)
        self.ui.bnAutoCapture.clicked.connect(self.start_auto_capture)
        self.ui.bnSetSavePath.clicked.connect(self.set_save_path)
        self.ui.bnAutoFocus.clicked.connect(self.start_autofocus)
        self.ui.bnStopAutoFocus.clicked.connect(self.stop_autofocus)
        self.ui.bnRefreshPort.clicked.connect(self.refresh_serial_ports)
        self.ui.bnConnectSerial.clicked.connect(self.connect_serial)
        self.ui.bnHomeZ.clicked.connect(self.action_home_z)
        self.ui.bnMoveStep.clicked.connect(self.action_move_z_step)
        self.ui.bnSetLight.clicked.connect(self.action_set_light)
        self.ui.bnSetScaleCalib.clicked.connect(self.apply_scale_calib)
        self.ui.bnAutoCalib.clicked.connect(self.start_auto_calib)
        self.ui.bnCaptureDark.clicked.connect(self.capture_dark_frame)
        self.ui.chkDarkSub.stateChanged.connect(self.toggle_dark_sub)
        self.ui.bnClearDark.clicked.connect(self.clear_dark_frame)
        self.ui.chkShowScaleBar.stateChanged.connect(self.toggle_scale_bar)

    def _create_menu(self):
        menubar = self.menuBar()
        menu_recon = menubar.addMenu("三维重建(&3D)")
        action_recon = menu_recon.addAction("① 点云重建...")
        action_recon.triggered.connect(self.open_recon3d_dialog)
        action_temporal = menu_recon.addAction("② 连续扫描重建...")
        action_temporal.triggered.connect(self.open_temporal_depth_dialog)
        menu_recon.addSeparator()
        action_help = menu_recon.addAction("使用说明")
        action_help.triggered.connect(self.show_recon_help)

        menu_imaging = menubar.addMenu("出图(&I)")
        action_one_click = menu_imaging.addAction("一键出图...")
        action_one_click.triggered.connect(self.open_one_click_dialog)

    def show_recon_help(self):
        QMessageBox.information(
            self,
            "三维重建使用说明",
            "① 点云重建\n"
            "原理：Z 轴逐步停顿，每步拍一帧，逐像素取锐度最大 Z 值。\n\n"
            "② 连续扫描重建\n"
            "原理：Z 轴匀速连续扫描，相机按时间间隔采帧，按时间映射 Z 位置，"
            "可选嵌套精扫融合。\n\n"
            "两种模式都支持可视化和导出 .ply / .csv。\n\n"
            "【出图菜单】一键出图\n"
            "原理：Z 轴从高位（上）向低位（下）逐步停拍，\n"
            "DFF 焦点融合生成一张全焦合成图，自动保存为 BMP，并在对话框内预览。",
        )

    def load_settings(self):
        config = self.config_manager.load()
        self._update_path_label()

        baud_rates = ["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"]
        self.ui.cmbBaudRate.clear()
        self.ui.cmbBaudRate.addItems(baud_rates)
        baud_index = self.ui.cmbBaudRate.findText(config.baud_rate)
        self.ui.cmbBaudRate.setCurrentIndex(baud_index if baud_index >= 0 else baud_rates.index("115200"))
        self.ui.edtSerialTimeout.setText(config.serial_timeout)

        self.refresh_serial_ports()
        if config.serial_port:
            port_index = self.ui.cmbSerialPort.findText(config.serial_port)
            if port_index >= 0:
                self.ui.cmbSerialPort.setCurrentIndex(port_index)

        self._update_serial_status()
        self.ui.edtPixelsPerMm.setText("{:.4f}".format(config.pixels_per_mm))
        self.scale_overlay.set_pixels_per_mm(config.pixels_per_mm)
        self._update_scale_info_label()

    def save_settings(self):
        self.config_manager.serial_port = (
            self.ui.cmbSerialPort.currentText() if self.ui.cmbSerialPort.count() > 0 else ""
        )
        self.config_manager.baud_rate = self.ui.cmbBaudRate.currentText()
        self.config_manager.serial_timeout = self.ui.edtSerialTimeout.text().strip()
        self.config_manager.save()

    def _update_path_label(self):
        self.ui.lblSavePathInfo.setText("保存至: " + self.config_manager.effective_save_path())

    def enum_devices(self):
        try:
            devices = self.device_controller.enum_devices()
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc), QMessageBox.Ok)
            return

        self.ui.ComboDevices.clear()
        if not devices:
            QMessageBox.warning(self, "Info", "Find no device", QMessageBox.Ok)
            return
        self.ui.ComboDevices.addItems(devices)
        self.ui.ComboDevices.setCurrentIndex(0)

    def open_device(self):
        try:
            if self.ui.ComboDevices.currentIndex() < 0:
                raise RuntimeError("Please select a camera!")
            params = self.device_controller.open_camera(self.ui.ComboDevices.currentIndex())
            self.ui.edtExposureTime.setText("{0:.2f}".format(params["exposure_time"]))
            self.ui.edtGain.setText("{0:.2f}".format(params["gain"]))
            self.ui.edtFrameRate.setText("{0:.2f}".format(params["frame_rate"]))
            self.enable_controls()
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc), QMessageBox.Ok)

    def close_device(self):
        self.device_controller.close_camera()
        self.enable_controls()

    def start_grabbing(self):
        try:
            self.device_controller.start_grabbing(self.ui.widgetDisplay.winId())
            self.enable_controls()
            QTimer.singleShot(800, self.poll_cam_img_width)
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc), QMessageBox.Ok)

    def stop_grabbing(self):
        try:
            self.device_controller.stop_grabbing()
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc), QMessageBox.Ok)
        self.enable_controls()

    def set_continue_mode(self):
        try:
            self.device_controller.set_continue_mode()
            self.ui.radioContinueMode.setChecked(True)
            self.ui.radioTriggerMode.setChecked(False)
            self.ui.bnSoftwareTrigger.setEnabled(False)
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc), QMessageBox.Ok)

    def set_software_trigger_mode(self):
        try:
            self.device_controller.set_software_trigger_mode()
            self.ui.radioContinueMode.setChecked(False)
            self.ui.radioTriggerMode.setChecked(True)
            self.ui.bnSoftwareTrigger.setEnabled(self.device_controller.grabbing)
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc), QMessageBox.Ok)

    def trigger_once(self):
        try:
            self.device_controller.trigger_once()
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc), QMessageBox.Ok)

    def save_bmp(self):
        try:
            self.device_controller.save_bmp()
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc), QMessageBox.Ok)

    def set_save_path(self):
        new_path = QFileDialog.getExistingDirectory(self, "选择图片保存路径", self.config_manager.effective_save_path())
        if new_path:
            self.config_manager.save_path = new_path
            self.save_settings()
            self._update_path_label()

    def _auto_capture_worker(self, count):
        save_dir = self.config_manager.effective_save_path()
        ensure_dir(save_dir)
        success_count = 0
        timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")
        for index in range(count):
            if not self.auto_capture_running:
                break
            file_name = "{}_{:03d}.bmp".format(timestamp_base, index + 1)
            full_path = os.path.join(save_dir, file_name)
            try:
                ret = self.device_controller.save_bmp_with_path(full_path)
                if ret == MV_OK:
                    success_count += 1
                else:
                    print("Auto capture [{}/{}] failed, ret: {}".format(index + 1, count, to_hex_str(ret)))
            except Exception as exc:
                print("Auto capture [{}/{}] error: {}".format(index + 1, count, exc))
            time.sleep(0.2)

        self.auto_capture_running = False
        message = "自动拍摄完成！成功保存 {}/{} 张图片\n保存路径: {}".format(success_count, count, save_dir)
        QTimer.singleShot(0, lambda: QMessageBox.information(self, "完成", message))

    def start_auto_capture(self):
        if self.auto_capture_running:
            QMessageBox.warning(self, "提示", "自动拍摄正在进行中！", QMessageBox.Ok)
            return
        count_str = self.ui.edtCaptureCount.text().strip()
        if not count_str.isdigit() or int(count_str) <= 0:
            QMessageBox.warning(self, "错误", "请输入正整数！", QMessageBox.Ok)
            return
        self.auto_capture_running = True
        threading.Thread(target=self._auto_capture_worker, args=(int(count_str),), daemon=True).start()

    def get_param(self):
        try:
            params = self.device_controller.get_parameters()
            self.ui.edtExposureTime.setText("{0:.2f}".format(params["exposure_time"]))
            self.ui.edtGain.setText("{0:.2f}".format(params["gain"]))
            self.ui.edtFrameRate.setText("{0:.2f}".format(params["frame_rate"]))
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc), QMessageBox.Ok)

    def set_param(self):
        frame_rate = self.ui.edtFrameRate.text()
        exposure = self.ui.edtExposureTime.text()
        gain = self.ui.edtGain.text()
        if not self.is_float(frame_rate) or not self.is_float(exposure) or not self.is_float(gain):
            QMessageBox.warning(self, "Error", "Set param failed ret:" + to_hex_str(MV_E_PARAMETER), QMessageBox.Ok)
            return MV_E_PARAMETER
        try:
            self.device_controller.set_parameters(frame_rate, exposure, gain)
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc), QMessageBox.Ok)
        return MV_OK

    def refresh_serial_ports(self):
        if not SERIAL_AVAILABLE:
            self.ui.cmbSerialPort.clear()
            self.ui.cmbSerialPort.addItem("pyserial 未安装")
            return
        current = self.ui.cmbSerialPort.currentText()
        ports = self.device_controller.list_serial_ports()
        self.ui.cmbSerialPort.clear()
        if ports:
            self.ui.cmbSerialPort.addItems(ports)
            index = self.ui.cmbSerialPort.findText(current)
            if index >= 0:
                self.ui.cmbSerialPort.setCurrentIndex(index)
        else:
            self.ui.cmbSerialPort.addItem("无可用串口")

    def _update_serial_status(self):
        if self.device_controller.serial_connected:
            self.ui.lblSerialStatus.setText("● 已连接")
            self.ui.lblSerialStatus.setStyleSheet("color: green; font-weight: bold;")
            self.ui.bnConnectSerial.setText("断开串口")
        else:
            self.ui.lblSerialStatus.setText("● 未连接")
            self.ui.lblSerialStatus.setStyleSheet("color: red; font-weight: bold;")
            self.ui.bnConnectSerial.setText("连接串口")

    def connect_serial(self):
        if not SERIAL_AVAILABLE:
            QMessageBox.warning(self, "错误", "pyserial 未安装，请执行:\npip install pyserial", QMessageBox.Ok)
            return
        if self.device_controller.serial_connected:
            self.device_controller.disconnect_serial()
            self._update_serial_status()
            self.save_settings()
            return

        port = self.ui.cmbSerialPort.currentText()
        if not port or port in ("无可用串口", "pyserial 未安装"):
            QMessageBox.warning(self, "错误", "请先选择有效的串口！", QMessageBox.Ok)
            return
        try:
            baud = int(self.ui.cmbBaudRate.currentText())
            timeout = float(self.ui.edtSerialTimeout.text().strip())
            self.device_controller.connect_serial(port, baud, timeout)
            self.config_manager.serial_port = port
            self._update_serial_status()
            self.save_settings()
        except Exception as exc:
            QMessageBox.warning(self, "串口错误", str(exc), QMessageBox.Ok)

    def send_gcode(self, cmd):
        try:
            self.device_controller.send_gcode(cmd)
            print("已发送 G-code: {}".format(cmd.strip()))
            return True
        except Exception as exc:
            QMessageBox.warning(self, "串口错误", "发送失败:\n" + str(exc), QMessageBox.Ok)
            return False

    def action_home_z(self):
        self.send_gcode("G28 Z\n")

    def action_move_z_step(self):
        if self.send_gcode("G91\n"):
            self.send_gcode("G1 Z0.1 F300\n")
            self.send_gcode("G90\n")

    def action_set_light(self):
        value_str = self.ui.edtLightValue.text().strip()
        if not value_str.isdigit():
            QMessageBox.warning(self, "错误", "请输入 0-255 之间的整数！", QMessageBox.Ok)
            return
        value = int(value_str)
        if not (0 <= value <= 255):
            QMessageBox.warning(self, "错误", "亮度值必须在 0-255 范围内！", QMessageBox.Ok)
            return
        self.send_gcode("M106 S{}\n".format(value))

    def poll_cam_img_width(self):
        if self.device_controller.grabbing:
            try:
                _, width, _ = self.device_controller.get_frame_numpy()
                if width > 0:
                    self._cam_img_width = width
                    self.scale_overlay.set_img_width(width)
            except Exception:
                pass

    def apply_scale_calib(self):
        try:
            value = float(self.ui.edtPixelsPerMm.text().strip())
            if value <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的正数（像素/mm）。", QMessageBox.Ok)
            return
        self.config_manager.pixels_per_mm = value
        if self.device_controller.grabbing:
            self.poll_cam_img_width()
        self.scale_overlay.set_pixels_per_mm(value)
        self._update_scale_info_label()
        self.save_settings()

    def toggle_scale_bar(self, state):
        self.scale_overlay.set_visible(state == Qt.Checked)

    def _update_scale_info_label(self):
        ppmm = self.config_manager.pixels_per_mm
        if ppmm <= 0:
            self.ui.lblScaleBarInfo.setText("未标定")
            return
        self.ui.lblScaleBarInfo.setText("1mm={:.1f}px | {:.3f}µm/px".format(ppmm, 1000.0 / ppmm))

    def _compute_sharpness(self):
        gray, width, height = self.device_controller.get_gray_frame()
        if gray is None or width == 0 or height == 0:
            return 0.0
        return compute_sharpness_score(gray)

    def _af_move_z(self, step_mm):
        try:
            self.device_controller.move_z_relative(step_mm, feed=300)
            time.sleep(0.25)
            return True
        except Exception:
            return False

    def _autofocus_worker(self):
        def set_status(message):
            QTimer.singleShot(0, lambda m=message: self.ui.lblAutoFocusStatus.setText(m))

        try:
            set_status("对焦中… 粗搜索")
            coarse_step = 1.0
            coarse_half = 3
            current_pos = 0.0

            if not self._af_move_z(-coarse_step * coarse_half):
                set_status("对焦失败：串口错误")
                return
            current_pos = -coarse_step * coarse_half

            scores_c = []
            pos_c = []
            total_c = coarse_half * 2 + 1
            for index in range(total_c):
                if not self.autofocus_running:
                    set_status("已停止")
                    return
                time.sleep(0.15)
                score = self._compute_sharpness()
                scores_c.append(score)
                pos_c.append(current_pos)
                set_status("对焦中… 粗搜索 {}/{}".format(index + 1, total_c))
                if index < total_c - 1:
                    self._af_move_z(coarse_step)
                    current_pos += coarse_step

            best_c = int(max(range(len(scores_c)), key=lambda k: scores_c[k]))
            best_pos = pos_c[best_c]
            self._af_move_z(best_pos - current_pos)
            current_pos = best_pos

            set_status("对焦中… 中等搜索")
            medium_step = 0.2
            medium_half = 4
            self._af_move_z(-medium_step * medium_half)
            current_pos -= medium_step * medium_half

            scores_m = []
            pos_m = []
            total_m = medium_half * 2 + 1
            for index in range(total_m):
                if not self.autofocus_running:
                    set_status("已停止")
                    return
                time.sleep(0.12)
                score = self._compute_sharpness()
                scores_m.append(score)
                pos_m.append(current_pos)
                set_status("对焦中… 中等搜索 {}/{}".format(index + 1, total_m))
                if index < total_m - 1:
                    self._af_move_z(medium_step)
                    current_pos += medium_step

            best_m = int(max(range(len(scores_m)), key=lambda k: scores_m[k]))
            best_pos = pos_m[best_m]
            self._af_move_z(best_pos - current_pos)
            current_pos = best_pos

            set_status("对焦中… PID 精细")
            kp, ki, kd = 0.06, 0.004, 0.018
            integral = 0.0
            prev_error = 0.0
            probe = 0.05
            max_iter = 20
            conv_thr = 0.004

            for _ in range(max_iter):
                if not self.autofocus_running:
                    break
                self._af_move_z(probe)
                current_pos += probe
                time.sleep(0.1)
                s_plus = self._compute_sharpness()
                self._af_move_z(-2 * probe)
                current_pos -= 2 * probe
                time.sleep(0.1)
                s_minus = self._compute_sharpness()
                self._af_move_z(probe)
                current_pos += probe
                gradient = (s_plus - s_minus) / (2 * probe)
                error = gradient
                integral = max(-0.5, min(0.5, integral + error * 0.01))
                derivative = error - prev_error
                control = kp * error + ki * integral + kd * derivative
                step = max(-0.15, min(0.15, control))
                prev_error = error
                if abs(step) < conv_thr:
                    break
                self._af_move_z(step)
                current_pos += step

            final_score = self._compute_sharpness()
            set_status("对焦完成 ✓  锐度:{:.0f}".format(final_score))
        except Exception as exc:
            set_status("对焦失败: " + str(exc))
        finally:
            self.autofocus_running = False
            QTimer.singleShot(0, lambda: self.ui.bnAutoFocus.setEnabled(True))
            QTimer.singleShot(0, lambda: self.ui.bnStopAutoFocus.setEnabled(False))

    def start_autofocus(self):
        if not self.device_controller.serial_connected:
            QMessageBox.warning(self, "错误", "请先连接串口！", QMessageBox.Ok)
            return
        if self.autofocus_running:
            QMessageBox.warning(self, "提示", "对焦正在进行！", QMessageBox.Ok)
            return
        self.autofocus_running = True
        self.ui.bnAutoFocus.setEnabled(False)
        self.ui.bnStopAutoFocus.setEnabled(True)
        threading.Thread(target=self._autofocus_worker, daemon=True).start()

    def stop_autofocus(self):
        self.autofocus_running = False
        self.ui.lblAutoFocusStatus.setText("停止中…")

    def _auto_calib_worker(self, move_mm, axis):
        def set_status(message):
            QTimer.singleShot(0, lambda m=message: self.ui.lblAutoCalibStatus.setText(m))

        try:
            set_status("标定中… 采集初始帧")
            frame1, width, height = self.device_controller.get_gray_frame()
            if frame1 is None or width == 0 or height == 0:
                set_status("标定失败：无法获取初始帧")
                return

            set_status("标定中… 移动 {:.3f}mm ({})轴".format(move_mm, axis))
            self.send_gcode("G91\n")
            self.send_gcode("G1 {}{:.4f} F300\n".format(axis, move_mm))
            self.send_gcode("G90\n")
            time.sleep(max(0.5, abs(move_mm) / 10.0 + 0.4))

            set_status("标定中… 采集移动后帧")
            frame2, width2, height2 = self.device_controller.get_gray_frame()
            if frame2 is None or width2 == 0 or height2 == 0:
                set_status("标定失败：无法获取移动后帧")
                return

            dx, dy = phase_correlation_shift(frame1, frame2)
            shift_px = abs(dx) if axis.upper() == "X" else abs(dy)
            if shift_px < 1.0:
                set_status("标定失败：位移过小({:.2f}px)".format(shift_px))
                return

            ppmm = shift_px / abs(move_mm)

            def update_ui():
                self.config_manager.pixels_per_mm = ppmm
                self.ui.edtPixelsPerMm.setText("{:.4f}".format(ppmm))
                self.apply_scale_calib()
                self.ui.lblAutoCalibStatus.setText(
                    "标定完成 ✓\n{:.2f} px/mm | {:.3f} µm/px".format(ppmm, 1000.0 / ppmm)
                )

            QTimer.singleShot(0, update_ui)
        except Exception as exc:
            set_status("标定失败: " + str(exc))
        finally:
            self.auto_calib_running = False
            QTimer.singleShot(0, lambda: self.ui.bnAutoCalib.setEnabled(True))

    def start_auto_calib(self):
        if not self.device_controller.grabbing:
            QMessageBox.warning(self, "错误", "请先开始采集！", QMessageBox.Ok)
            return
        if not self.device_controller.serial_connected:
            QMessageBox.warning(self, "错误", "请先连接串口！", QMessageBox.Ok)
            return
        if self.auto_calib_running:
            QMessageBox.warning(self, "提示", "标定正在进行中！", QMessageBox.Ok)
            return
        try:
            move_mm = float(self.ui.edtCalibMoveMm.text().strip())
            if abs(move_mm) < 0.01:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的移动距离（mm），最小 0.01mm。", QMessageBox.Ok)
            return

        axis, ok = QInputDialog.getItem(self, "选择标定轴", "请选择移动轴：", ["X", "Y"], 0, False)
        if not ok:
            return
        self.auto_calib_running = True
        self.ui.bnAutoCalib.setEnabled(False)
        self.ui.lblAutoCalibStatus.setText("标定中…")
        threading.Thread(target=self._auto_calib_worker, args=(move_mm, axis), daemon=True).start()

    def capture_dark_frame(self):
        if not self.device_controller.grabbing:
            QMessageBox.warning(self, "错误", "请先开始采集！", QMessageBox.Ok)
            return
        try:
            info = self.device_controller.capture_dark_frame()
            self.dark_frame_captured = True
            self.ui.chkDarkSub.setEnabled(True)
            self.ui.bnClearDark.setEnabled(True)
            self.ui.lblDarkSubStatus.setText(
                "帧大小: {}x{}\n均值: {:.1f}\n底噪帧已就绪".format(info["width"], info["height"], info["mean"])
            )
        except Exception as exc:
            QMessageBox.warning(self, "错误", "采集底噪帧失败:\n" + str(exc), QMessageBox.Ok)

    def toggle_dark_sub(self, state):
        enabled = state == Qt.Checked
        self.device_controller.set_dark_sub_enabled(enabled)
        self._update_dark_sub_status_label(self.dark_frame_captured, enabled)

    def _update_dark_sub_status_label(self, captured, enabled):
        if not captured:
            self.ui.lblDarkSubStatus.setText("未采集")
        elif enabled:
            self.ui.lblDarkSubStatus.setText("底噪帧已就绪\n已开启 ✔")
        else:
            self.ui.lblDarkSubStatus.setText("底噪帧已就绪\n未开启")

    def clear_dark_frame(self):
        self.device_controller.clear_dark_frame()
        self.dark_frame_captured = False
        self.ui.chkDarkSub.setChecked(False)
        self.ui.chkDarkSub.setEnabled(False)
        self.ui.bnClearDark.setEnabled(False)
        self.ui.lblDarkSubStatus.setText("未采集")

    def open_recon3d_dialog(self):
        if self._recon3d_dialog is None:
            self._recon3d_dialog = PointCloudReconDialog(self.device_controller, self.config_manager, self)
        self._recon3d_dialog.show()
        self._recon3d_dialog.raise_()
        self._recon3d_dialog.activateWindow()

    def open_temporal_depth_dialog(self):
        if self._temporal_depth_dialog is None:
            self._temporal_depth_dialog = TemporalDepthDialog(self.device_controller, self.config_manager, self)
        self._temporal_depth_dialog.show()
        self._temporal_depth_dialog.raise_()
        self._temporal_depth_dialog.activateWindow()

    def open_one_click_dialog(self):
        if self._one_click_dialog is None:
            self._one_click_dialog = OneClickDialog(self.device_controller, self.config_manager, self)
        self._one_click_dialog.show()
        self._one_click_dialog.raise_()
        self._one_click_dialog.activateWindow()

    def enable_controls(self):
        is_open = self.device_controller.opened
        is_grabbing = self.device_controller.grabbing
        self.ui.groupGrab.setEnabled(is_open)
        self.ui.groupParam.setEnabled(is_open)

        self.ui.bnOpen.setEnabled(not is_open)
        self.ui.bnClose.setEnabled(is_open)
        self.ui.bnStart.setEnabled(is_open and not is_grabbing)
        self.ui.bnStop.setEnabled(is_open and is_grabbing)
        self.ui.bnSoftwareTrigger.setEnabled(is_grabbing and self.ui.radioTriggerMode.isChecked())
        self.ui.bnSaveImage.setEnabled(is_open and is_grabbing)
        self.ui.bnAutoCapture.setEnabled(is_open and is_grabbing)
        self.ui.bnAutoFocus.setEnabled(is_open and is_grabbing and not self.autofocus_running)
        self.ui.bnStopAutoFocus.setEnabled(self.autofocus_running)
        self.ui.bnAutoCalib.setEnabled(is_open and is_grabbing and not self.auto_calib_running)
        self.ui.bnCaptureDark.setEnabled(is_open and is_grabbing)
        if not (is_open and is_grabbing):
            self.device_controller.set_dark_sub_enabled(False)
            self.ui.chkDarkSub.setChecked(False)

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        try:
            self.device_controller.cleanup()
        except Exception:
            pass

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)
