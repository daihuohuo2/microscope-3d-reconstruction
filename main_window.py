import os
import threading
import time
from datetime import datetime

from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from sdk.MvErrorDefine_const import MV_E_PARAMETER, MV_OK
from algorithms import (
    CALIB_DOT_SPACING_UM,
    compute_blob_scale_calibration,
    ensure_dir,
    phase_correlation_shift,
    save_autofocus_curve,
    save_autofocus_curve_csv,
)
from config_manager import ConfigManager
from device_controller import DeviceController, SERIAL_AVAILABLE, to_hex_str
from dialogs import PointCloudReconDialog, TemporalDepthDialog, OneClickDialog, ProgrammableShootingDialog, OfflineZStackDialog
from overlays import DoubleClickFilter, ResizeFilter, ScaleBarOverlay
from ui import Ui_MainWindow


class MainWindow(QMainWindow):
    _quick_scale_done = pyqtSignal(dict)   # 线程安全：blob结果 → 主线程更新UI
    _quick_scale_fail = pyqtSignal(str)    # 线程安全：错误信息 → 主线程弹窗
    DEFAULT_EXPOSURE_US = 80000.0
    DEFAULT_GAIN_DB = 5.0
    MAGNIFICATION_SCALE_TABLE = [
        (0.7, 439.0185), (0.8, 481.4586), (0.9, 546.7106), (1.0, 625.1945),
        (1.1, 691.8781), (1.2, 752.3955), (1.3, 777.7159), (1.4, 869.0678),
        (1.5, 959.5854), (1.6, 910.9615), (1.7, 1068.2162), (1.8, 1137.9918),
        (1.9, 1159.1079), (2.0, 1194.4625), (2.1, 1273.0709), (2.2, 1299.7029),
        (2.3, 1239.4471), (2.4, 1482.0338), (2.5, 1532.8936), (2.6, 1622.9466),
        (2.7, 1687.4260), (2.8, 1715.2986), (2.9, 1775.1524), (3.0, 1849.5680),
        (3.1, 1951.2858), (3.2, 2018.0778), (3.3, 2046.1368), (3.4, 2042.3262),
        (3.5, 2147.2584), (3.6, 2122.9608), (3.7, 2321.6342), (3.8, 2302.5984),
        (3.9, 2354.7970), (4.0, 2461.2426), (4.1, 2557.7970), (4.2, 2613.0566),
        (4.3, 2663.3777), (4.4, 2739.3315), (4.5, 2773.7936), (4.6, 2760.1828),
        (4.7, 2755.1015), (4.8, 2870.0385), (4.9, 3051.7449), (5.0, 3126.2564),
        (5.1, 3164.7405), (5.2, 3228.6153), (5.3, 3297.4938), (5.4, 3310.0723),
        (5.5, 3370.7108), (5.6, 3465.4857),
    ]

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
        self.quick_scale_running = False
        self.dark_frame_captured = False
        self._z_at_home = False
        self._cam_img_width = 0
        self._cam_img_height = 0
        self._af_roi_center = None   # (img_cx, img_cy) set by double-click; None = global focus
        self._af_curve_samples = []
        self._last_af_curve_paths = {}
        self._recon3d_dialog = None
        self._temporal_depth_dialog = None
        self._one_click_dialog = None
        self._programmable_dialog = None
        self._offline_zstack_dialog = None
        self._cleaned_up = False
        self._z_poll_busy = False

        # 强制为预览 widget 分配独立原生 HWND，并阻止 Qt 双缓冲覆盖，
        # 确保 SDK 的 MV_CC_DisplayOneFrame (GDI) 能正常渲染，不会卡住。
        self.ui.widgetDisplay.setAttribute(Qt.WA_NativeWindow, True)
        self.ui.widgetDisplay.setAttribute(Qt.WA_PaintOnScreen, True)

        self.scale_overlay = ScaleBarOverlay(self.ui.widgetDisplay)
        self._resize_filter = ResizeFilter(self.scale_overlay)
        self.ui.widgetDisplay.installEventFilter(self._resize_filter)
        self.ui.centralWidget.installEventFilter(self._resize_filter)
        self.installEventFilter(self._resize_filter)   # 主窗口移动时更新叠加层位置

        self._dblclick_filter = DoubleClickFilter(self._on_display_dblclick)
        self.ui.widgetDisplay.installEventFilter(self._dblclick_filter)

        self._bind_signals()
        self._create_menu()
        self.load_settings()
        self.scale_overlay.set_visible(self.ui.chkShowScaleBar.isChecked())
        self.enable_controls()
        self._z_timer = QTimer(self)
        self._z_timer.timeout.connect(self.refresh_z_position)
        self._z_timer.start(1000)
        self._update_z_display()

    def _bind_signals(self):
        self.ui.bnEnum.clicked.connect(self.enum_devices)
        self.ui.bnOpen.clicked.connect(self.open_device)
        self.ui.bnClose.clicked.connect(self.close_device)
        self.ui.bnStart.clicked.connect(self.start_grabbing)
        self.ui.bnStop.clicked.connect(self.stop_grabbing)

        self.ui.bnGetParam.clicked.connect(self.get_param)
        self.ui.bnSetParam.clicked.connect(self.set_param)

        self.ui.bnAutoFocus.clicked.connect(self.start_autofocus)
        self.ui.bnStopAutoFocus.clicked.connect(self.stop_autofocus)
        self.ui.bnRefreshPort.clicked.connect(self.refresh_serial_ports)
        self.ui.bnConnectSerial.clicked.connect(self.connect_serial)
        self.ui.bnHomeZ.clicked.connect(self.action_home_z)
        self.ui.bnCoarseUp.clicked.connect(self.action_coarse_up)
        self.ui.bnCoarseDown.clicked.connect(self.action_coarse_down)
        self.ui.bnMediumUp.clicked.connect(self.action_medium_up)
        self.ui.bnMediumDown.clicked.connect(self.action_medium_down)
        self.ui.bnFineUp.clicked.connect(self.action_fine_up)
        self.ui.bnFineDown.clicked.connect(self.action_fine_down)
        self.ui.bnMoveStep.clicked.connect(self.action_move_z_step)
        self.ui.bnMoveStepDown.clicked.connect(self.action_move_z_step_down)
        self.ui.sliderLight.valueChanged.connect(self.action_slider_light)
        self.ui.edtLightValue.editingFinished.connect(self.action_light_input)
        self.ui.bnQuickScale.clicked.connect(self.start_quick_scale)
        self._quick_scale_done.connect(self._on_quick_scale_done)
        self._quick_scale_fail.connect(self._on_quick_scale_fail)
        self.ui.edtMagnification.editingFinished.connect(self.apply_magnification_scale)
        self.ui.edtPixelsPerMm.editingFinished.connect(self.apply_manual_pixels_per_mm)
        self.ui.bnCaptureDark.clicked.connect(self.capture_dark_frame)
        self.ui.chkDarkSub.stateChanged.connect(self.toggle_dark_sub)
        self.ui.bnClearDark.clicked.connect(self.clear_dark_frame)
        self.ui.chkShowScaleBar.stateChanged.connect(self.toggle_scale_bar)
        self.ui.chkHdr.stateChanged.connect(self.toggle_hdr)

    def _create_menu(self):
        menubar = self.menuBar()
        action_recon = menubar.addAction("点云重建(&3)...")
        action_recon.triggered.connect(self.open_recon3d_dialog)
        action_temporal = menubar.addAction("连续扫描重建(&S)...")
        action_temporal.triggered.connect(self.open_temporal_depth_dialog)

        action_one_click = menubar.addAction("一键出图(&I)")
        action_one_click.triggered.connect(self.open_one_click_dialog)

        action_prog = menubar.addAction("可编程拍摄(&P)")
        action_prog.triggered.connect(self.open_programmable_shooting_dialog)
        action_offline = menubar.addAction("Offline Z-stack (&Z)...")
        action_offline.triggered.connect(self.open_offline_zstack_dialog)
        action_af_curve = menubar.addAction("导出自动对焦锐度曲线(&F)...")
        action_af_curve.triggered.connect(self.export_last_autofocus_curve)

    def load_settings(self):
        config = self.config_manager.load()

        baud_rates = ["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"]
        self.ui.cmbBaudRate.clear()
        self.ui.cmbBaudRate.addItems(baud_rates)
        baud_index = self.ui.cmbBaudRate.findText(config.baud_rate)
        self.ui.cmbBaudRate.setCurrentIndex(baud_index if baud_index >= 0 else baud_rates.index("19200"))
        self.ui.edtSerialTimeout.setText(config.serial_timeout)

        self.refresh_serial_ports()
        if config.serial_port:
            port_index = self.ui.cmbSerialPort.findText(config.serial_port)
            if port_index >= 0:
                self.ui.cmbSerialPort.setCurrentIndex(port_index)

        self._update_serial_status()
        self.ui.edtMagnification.setText("{:.2f}".format(config.magnification).rstrip("0").rstrip("."))
        self.ui.edtPixelsPerMm.setText("{:.4f}".format(config.pixels_per_mm))
        self.scale_overlay.set_pixels_per_mm(config.pixels_per_mm)

    def save_settings(self):
        self.config_manager.serial_port = (
            self.ui.cmbSerialPort.currentText() if self.ui.cmbSerialPort.count() > 0 else ""
        )
        self.config_manager.baud_rate = self.ui.cmbBaudRate.currentText()
        self.config_manager.serial_timeout = self.ui.edtSerialTimeout.text().strip()
        self.config_manager.save()

    @classmethod
    def _lookup_curve_pixels_per_mm(cls, magnification):
        mag = float(magnification)
        table = cls.MAGNIFICATION_SCALE_TABLE
        if mag <= table[0][0]:
            x0, y0 = table[0]
            x1, y1 = table[1]
        elif mag >= table[-1][0]:
            x0, y0 = table[-2]
            x1, y1 = table[-1]
        else:
            for idx in range(len(table) - 1):
                x0, y0 = table[idx]
                x1, y1 = table[idx + 1]
                if x0 <= mag <= x1:
                    break
        ratio = (mag - x0) / (x1 - x0)
        return y0 + ratio * (y1 - y0)

    def _set_pixels_per_mm(self, ppmm):
        ppmm = max(float(ppmm), 0.001)
        self.config_manager.pixels_per_mm = ppmm
        self.ui.edtPixelsPerMm.setText("{:.4f}".format(ppmm))
        self.scale_overlay.set_pixels_per_mm(ppmm)
        self.scale_overlay.update()

    def _current_magnification(self):
        text = self.ui.edtMagnification.text().strip()
        if not text:
            return None
        return float(text)

    def apply_magnification_scale(self):
        try:
            mag = self._current_magnification()
            if mag is None or mag <= 0:
                return
            curve_ppmm = self._lookup_curve_pixels_per_mm(mag)
            ppmm = curve_ppmm * self.config_manager.scale_curve_factor
            self.config_manager.magnification = mag
            self._set_pixels_per_mm(ppmm)
            self.save_settings()
            self.ui.lblQuickScaleStatus.setText(
                "倍率 {:.2f}x → {:.2f} px/mm".format(mag, ppmm)
            )
        except Exception as exc:
            self.ui.lblQuickScaleStatus.setText("倍率换算失败: " + str(exc))

    def apply_manual_pixels_per_mm(self):
        try:
            ppmm = float(self.ui.edtPixelsPerMm.text().strip())
            self._set_pixels_per_mm(ppmm)
            self.save_settings()
        except Exception as exc:
            self.ui.lblQuickScaleStatus.setText("像素/mm无效: " + str(exc))

    def enum_devices(self):
        try:
            devices = self.device_controller.enum_devices()
        except Exception as exc:
            QMessageBox.warning(self, "查找设备失败", str(exc), QMessageBox.Ok)
            return

        self.ui.ComboDevices.clear()
        if not devices:
            QMessageBox.warning(self, "未找到设备", "未发现任何相机设备，请检查连接。", QMessageBox.Ok)
            return
        self.ui.ComboDevices.addItems(devices)
        self.ui.ComboDevices.setCurrentIndex(0)

    def open_device(self):
        try:
            if self.ui.ComboDevices.currentIndex() < 0:
                raise RuntimeError("Please select a camera!")
            params = self.device_controller.open_camera(self.ui.ComboDevices.currentIndex())
            self.device_controller.set_exposure(self.DEFAULT_EXPOSURE_US)
            self.device_controller.set_gain(self.DEFAULT_GAIN_DB)
            params = self.device_controller.get_parameters()
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
            self.device_controller.set_continue_mode()
            # 调用 winId() 前确保原生 HWND 已分配，防止 GDI 渲染时拿到无效句柄
            win_id = int(self.ui.widgetDisplay.winId())
            self.device_controller.start_grabbing(win_id)
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
        self._update_z_motion_buttons()
        self._update_z_display()

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
            self._sync_z_from_device(show_error=False)
            self.save_settings()
            # Send default brightness
            self.send_gcode("M106 S{}\n".format(self.ui.sliderLight.value()))
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
        try:
            self.device_controller.home_z_wait()
            self._z_at_home = True
            self._update_z_motion_buttons()
            self._update_z_display()
        except Exception as exc:
            QMessageBox.warning(self, "Z 轴归零失败", str(exc), QMessageBox.Ok)

    def _sync_z_from_device(self, show_error=False):
        try:
            self.device_controller.refresh_z_position(timeout=0.8)
            self._z_at_home = self.device_controller._z_position <= 0.001
            self._update_z_motion_buttons()
            self._update_z_display()
            return True
        except Exception as exc:
            if show_error:
                QMessageBox.warning(self, "Z 轴位置读取失败", str(exc), QMessageBox.Ok)
            return False

    def refresh_z_position(self):
        if self._z_poll_busy or not self.device_controller.serial_connected:
            return
        self._z_poll_busy = True

        def worker():
            try:
                self.device_controller.refresh_z_position(timeout=0.6)
                QTimer.singleShot(0, self._update_z_display)
            finally:
                self._z_poll_busy = False

        threading.Thread(target=worker, daemon=True).start()

    def _update_z_display(self):
        """刷新主界面和所有已打开 dialog 的 Z 位置显示。"""
        z = self.device_controller._z_position
        min_limit = getattr(self.device_controller, "_z_min_limit", 0.0)
        max_limit = getattr(self.device_controller, "_z_soft_limit", 68.0)
        self.ui.lblZPos.setText("实时 Z 位置: {:.3f} mm".format(z))
        if hasattr(self.ui, "lblZMinLimit"):
            self.ui.lblZMinLimit.setText("最低提醒: {:.1f} mm".format(min_limit))
        if hasattr(self.ui, "lblZMaxLimit"):
            self.ui.lblZMaxLimit.setText("最高提醒: {:.1f} mm".format(max_limit))
        if z <= min_limit or z >= max_limit:
            z_color = "#d80000"
            border = "#d80000"
        elif z <= min_limit + 1.0 or z >= max_limit - 5.0:
            z_color = "#b56a00"
            border = "#ffb000"
        else:
            z_color = "#0078d7"
            border = "#0078d7"
        self.ui.lblZPos.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: {};"
            "background: #ffffff; border: 1px solid {}; border-radius: 4px; padding: 3px;".format(z_color, border)
        )
        if hasattr(self.ui, "lblZMinLimit"):
            min_color = "#d80000" if z <= min_limit else "#b56a00" if z <= min_limit + 1.0 else "#444444"
            self.ui.lblZMinLimit.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: {};"
                "background: #ffffff; border: 1px solid #cccccc; border-radius: 4px; padding: 3px;".format(min_color)
            )
        if hasattr(self.ui, "lblZMaxLimit"):
            max_color = "#d80000" if z >= max_limit else "#b56a00" if z >= max_limit - 5.0 else "#444444"
            self.ui.lblZMaxLimit.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: {};"
                "background: #ffffff; border: 1px solid #cccccc; border-radius: 4px; padding: 3px;".format(max_color)
            )
        self.ui.lblZPos.setToolTip(
            "Z 轴最低提醒点 {:.1f} mm，最高提醒点 {:.1f} mm；接近端点请谨慎调整。".format(
                min_limit, max_limit
            )
        )

    def _update_z_motion_buttons(self):
        """Keep manual Z adjustment available; soft limit remains enforced in DeviceController."""
        serial_ready = self.device_controller.serial_connected
        for button in (
            self.ui.bnCoarseUp,
            self.ui.bnCoarseDown,
            self.ui.bnMediumUp,
            self.ui.bnMediumDown,
            self.ui.bnFineUp,
            self.ui.bnFineDown,
            self.ui.bnMoveStep,
            self.ui.bnMoveStepDown,
        ):
            button.setEnabled(serial_ready)

    def action_coarse_up(self):
        self._move_z_relative_from_ui(1.00)

    def action_coarse_down(self):
        self._move_z_relative_from_ui(-1.00)

    def action_medium_up(self):
        self._move_z_relative_from_ui(0.10)

    def action_medium_down(self):
        self._move_z_relative_from_ui(-0.10)

    def action_fine_up(self):
        self._move_z_relative_from_ui(0.05)

    def action_fine_down(self):
        self._move_z_relative_from_ui(-0.05)

    def action_move_z_step(self):
        self._move_z_relative_from_ui(0.005)

    def action_move_z_step_down(self):
        self._move_z_relative_from_ui(-0.005)

    def _move_z_relative_from_ui(self, step_mm):
        try:
            self.device_controller.move_z_relative_wait(step_mm, feed=2000)
            self._z_at_home = self.device_controller._z_position <= 0.001
            self._update_z_motion_buttons()
            self._update_z_display()
        except Exception as exc:
            QMessageBox.warning(self, "Z 轴移动失败", str(exc), QMessageBox.Ok)

    def action_slider_light(self, value):
        self.ui.edtLightValue.setText(str(value))
        self.send_gcode("M106 S{}\n".format(value))

    def action_light_input(self):
        try:
            value = max(0, min(255, int(self.ui.edtLightValue.text())))
        except ValueError:
            value = self.ui.sliderLight.value()
        self.ui.edtLightValue.setText(str(value))
        self.ui.sliderLight.blockSignals(True)
        self.ui.sliderLight.setValue(value)
        self.ui.sliderLight.blockSignals(False)
        self.send_gcode("M106 S{}\n".format(value))

    def poll_cam_img_width(self):
        if not self.device_controller.grabbing:
            return
        try:
            _, width, height = self.device_controller.get_frame_numpy()
            if width > 0:
                self._cam_img_width = width
                self._cam_img_height = height
                self.scale_overlay.set_img_width(width)
        except Exception:
            pass

    def _on_display_dblclick(self, wx, wy):
        """Handle double-click on preview widget: map to image coords and start ROI autofocus."""
        if not self.device_controller.grabbing:
            return
        if not self.device_controller.serial_connected:
            return
        widget_w = self.ui.widgetDisplay.width()
        widget_h = self.ui.widgetDisplay.height()
        if widget_w == 0 or widget_h == 0:
            return
        # Prefer cached dimensions; fall back to a live query
        img_w = self._cam_img_width
        img_h = self._cam_img_height
        if img_w == 0 or img_h == 0:
            try:
                _, img_w, img_h = self.device_controller.get_frame_numpy()
            except Exception:
                return
        if img_w == 0 or img_h == 0:
            return
        # Map widget pixel → image pixel (SDK stretches to fill widget)
        img_cx = int(wx * img_w / widget_w)
        img_cy = int(wy * img_h / widget_h)
        half = 25
        img_cx = max(half, min(img_w - half, img_cx))
        img_cy = max(half, min(img_h - half, img_cy))
        self._af_roi_center = (img_cx, img_cy)
        self.start_autofocus()

    @staticmethod
    def _af_smooth3(gray):
        import numpy as np

        arr = np.asarray(gray, dtype=np.float32)
        if arr.shape[0] < 3 or arr.shape[1] < 3:
            return arr.copy()
        padded = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
        return (
            padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2] + 4.0 * padded[1:-1, 1:-1] + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
        ) / 16.0

    @staticmethod
    def _af_focus_components(gray):
        import numpy as np

        arr = np.asarray(gray, dtype=np.float32)
        gx = np.zeros_like(arr, dtype=np.float32)
        gy = np.zeros_like(arr, dtype=np.float32)
        gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
        gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
        grad2 = gx * gx + gy * gy

        lap = np.zeros_like(arr, dtype=np.float32)
        lap[1:-1, 1:-1] = (
            arr[:-2, 1:-1]
            + arr[2:, 1:-1]
            + arr[1:-1, :-2]
            + arr[1:-1, 2:]
            - 4.0 * arr[1:-1, 1:-1]
        )

        brenner = np.zeros_like(arr, dtype=np.float32)
        brenner[:, :-2] += (arr[:, 2:] - arr[:, :-2]) ** 2
        brenner[:-2, :] += (arr[2:, :] - arr[:-2, :]) ** 2
        return grad2, lap * lap, brenner

    @staticmethod
    def _af_texture_mask(roi_f, body_mask, block=16):
        import numpy as np

        h, w = roi_f.shape
        if h < block * 2 or w < block * 2:
            return body_mask

        h2 = h // block * block
        w2 = w // block * block
        cropped = roi_f[:h2, :w2]
        body = body_mask[:h2, :w2]
        blocks = cropped.reshape(h2 // block, block, w2 // block, block)
        body_blocks = body.reshape(h2 // block, block, w2 // block, block)
        block_var = np.var(blocks, axis=(1, 3))
        body_ratio = np.mean(body_blocks, axis=(1, 3))
        usable = body_ratio >= 0.18
        if not np.any(usable):
            return body_mask

        var_cut = float(np.percentile(block_var[usable], 55.0))
        keep_blocks = usable & (block_var >= max(6.0, var_cut))
        keep = np.zeros((h2 // block, w2 // block, block, block), dtype=bool)
        keep[keep_blocks] = True
        texture = np.zeros_like(body_mask, dtype=bool)
        texture[:h2, :w2] = keep.reshape(h2, w2)
        texture &= body_mask
        if float(np.mean(texture)) < 0.04:
            return body_mask
        return texture

    def _af_atlas_score(self, roi_raw):
        """ATLAS Focus score: texture-guided multi-scale Tenengrad/Laplacian/Brenner fusion."""
        import numpy as np

        if roi_raw is None or roi_raw.size < 16:
            return 0.0

        roi_f = self._normalize_gray_for_analysis(roi_raw)
        if roi_f.shape[0] < 3 or roi_f.shape[1] < 3:
            return float(np.var(roi_f))

        p02 = float(np.percentile(roi_f, 2.0))
        p10 = float(np.percentile(roi_f, 10.0))
        p50 = float(np.percentile(roi_f, 50.0))
        p90 = float(np.percentile(roi_f, 90.0))
        p98 = float(np.percentile(roi_f, 98.0))
        contrast = p90 - p10
        if contrast < 8.0:
            return 0.0

        body_low = max(p02 + 6.0, p50 - 55.0)
        body_high = min(p98 - 10.0, p50 + 85.0, 238.0)
        body_mask = (roi_f >= body_low) & (roi_f <= body_high)
        if float(np.mean(body_mask)) < 0.18:
            body_mask = (roi_f >= p10) & (roi_f <= min(p98, 242.0))
        body_mask &= roi_f < 248.0

        texture_mask = self._af_texture_mask(roi_f, body_mask)
        if int(np.count_nonzero(texture_mask)) < 64:
            return 0.0

        grad2, lap2, brenner = self._af_focus_components(roi_f)
        smooth = self._af_smooth3(roi_f)
        grad2_smooth, _, _ = self._af_focus_components(smooth)

        tenengrad_hi = float(np.percentile(grad2[texture_mask], 92.0))
        tenengrad_smooth_hi = float(np.percentile(grad2_smooth[texture_mask], 92.0))
        lap_hi = float(np.percentile(lap2[texture_mask], 92.0))
        brenner_hi = float(np.percentile(brenner[texture_mask], 92.0))

        local_values = roi_f[texture_mask]
        local_contrast = float(np.percentile(local_values, 90.0) - np.percentile(local_values, 10.0))
        highlight_penalty = 1.0 / (1.0 + max(0.0, float(np.mean(roi_f >= 248.0)) - 0.003) * 80.0)
        contrast_weight = max(0.15, min(1.0, local_contrast / 80.0))
        coverage_weight = max(0.45, min(1.0, float(np.mean(texture_mask)) / 0.18))

        fused = (
            0.45 * tenengrad_hi
            + 0.25 * tenengrad_smooth_hi
            + 0.20 * lap_hi
            + 0.10 * brenner_hi
        )
        return float(fused * contrast_weight * coverage_weight * highlight_penalty)

    def _compute_roi_contrast(self, cx, cy, half=25, sample_count=1):
        """ATLAS ROI focus score around a double-clicked image point."""
        import numpy as np

        scores = []
        for _ in range(max(1, int(sample_count))):
            gray, width, height = self.device_controller.get_gray_frame()
            if gray is None or width == 0 or height == 0:
                continue
            x1 = max(0, cx - half)
            x2 = min(width, cx + half)
            y1 = max(0, cy - half)
            y2 = min(height, cy + half)
            roi = gray[y1:y2, x1:x2]
            if roi.size < 16:
                continue
            scores.append(self._af_atlas_score(roi))
            if sample_count > 1:
                time.sleep(0.04)
        if not scores:
            return 0.0
        return float(np.median(scores))

    def toggle_scale_bar(self, state):
        self.scale_overlay.set_visible(state == Qt.Checked)
        self.scale_overlay.update_size()

    def _set_quick_scale_status(self, message):
        self.ui.lblQuickScaleStatus.setText(message)

    def _quick_scale_worker(self):
        def set_status(message):
            QTimer.singleShot(0, lambda m=message: self._set_quick_scale_status(m))

        try:
            import time as _time
            set_status("识别圆点中… 取帧")
            gray, width, height = self.device_controller.get_gray_frame()
            if gray is None or width == 0 or height == 0:
                raise ValueError("无法获取当前图像，请确认相机正在采集")

            set_status("识别圆点中… blob检测")
            try:
                spacing_um = float(self.ui.edtDotSpacing.text())
            except ValueError:
                spacing_um = CALIB_DOT_SPACING_UM
            result = compute_blob_scale_calibration(gray, spacing_um=spacing_um)
            self._quick_scale_done.emit(result)
        except Exception as exc:
            self._quick_scale_fail.emit(str(exc))
        finally:
            self.quick_scale_running = False
            QTimer.singleShot(0, self.enable_controls)

    def _on_quick_scale_done(self, result):
        try:
            ppmm = result["pixels_per_mm"]
            mag = self._current_magnification()
            curve_factor = None
            if mag is not None and mag > 0:
                curve_ppmm = self._lookup_curve_pixels_per_mm(mag)
                if curve_ppmm > 0:
                    curve_factor = ppmm / curve_ppmm
                    self.config_manager.magnification = mag
                    self.config_manager.scale_curve_factor = curve_factor
            self._set_pixels_per_mm(ppmm)
            self.save_settings()
            self.ui.chkShowScaleBar.setChecked(True)
            self.scale_overlay.set_visible(True)
            if self.device_controller.grabbing:
                self.poll_cam_img_width()
            try:
                spacing_um = float(self.ui.edtDotSpacing.text())
            except ValueError:
                spacing_um = CALIB_DOT_SPACING_UM
            status = "完成 ✓ {}点 | {:.2f}px≈{}µm".format(
                result["blob_count"], result["spacing_px"], int(spacing_um)
            )
            if curve_factor is not None:
                status += " | 倍率曲线修正 {:.4f}x".format(curve_factor)
            self.ui.lblQuickScaleStatus.setText(status)
        except Exception as e:
            self.ui.lblQuickScaleStatus.setText("UI更新失败: " + str(e))

    def _on_quick_scale_fail(self, message):
        self.ui.lblQuickScaleStatus.setText("失败: " + message)
        QMessageBox.warning(self, "快速比例尺", message, QMessageBox.Ok)

    def start_quick_scale(self):
        if not self.device_controller.grabbing:
            QMessageBox.warning(self, "错误", "请先开始采集！", QMessageBox.Ok)
            return
        if self.quick_scale_running:
            QMessageBox.warning(self, "提示", "快速比例尺正在进行中！", QMessageBox.Ok)
            return
        self.quick_scale_running = True
        self.ui.bnQuickScale.setEnabled(False)
        self.ui.lblQuickScaleStatus.setText("识别圆点中…")
        threading.Thread(target=self._quick_scale_worker, daemon=True).start()

    @staticmethod
    def _normalize_gray_for_analysis(gray):
        import numpy as np

        arr = np.asarray(gray, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros(arr.shape, dtype=np.float32)
        p999 = float(np.percentile(finite, 99.9))
        if p999 <= 255.0:
            return np.clip(arr, 0.0, 255.0).astype(np.float32)
        sensor_max = 4095.0 if p999 <= 4095.0 else 65535.0
        return np.clip(arr / sensor_max * 255.0, 0.0, 255.0).astype(np.float32)

    def _get_center_roi(self, roi_fraction=0.55):
        gray, width, height = self.device_controller.get_gray_frame()
        if gray is None or width == 0 or height == 0:
            return None
        cy, cx = height // 2, width // 2
        rh = max(8, int(height * roi_fraction * 0.5))
        rw = max(8, int(width * roi_fraction * 0.5))
        roi = gray[max(0, cy - rh):min(height, cy + rh), max(0, cx - rw):min(width, cx + rw)]
        if roi.size == 0:
            return None
        if roi.size > 450000:
            roi = roi[::2, ::2]
        return roi

    def _compute_sharpness(self, sample_count=1, roi_fraction=0.65):
        import numpy as np

        # When a ROI center has been set by double-click, score only that local target.
        if self._af_roi_center is not None:
            cx, cy = self._af_roi_center
            return self._compute_roi_contrast(cx, cy, half=25, sample_count=sample_count)

        scores = []
        for _ in range(max(1, int(sample_count))):
            roi_raw = self._get_center_roi(roi_fraction=roi_fraction)
            if roi_raw is None:
                continue
            scores.append(self._af_atlas_score(roi_raw))
            if sample_count > 1:
                time.sleep(0.04)
        if not scores:
            return 0.0
        return float(np.median(scores))

    def _get_exposure_stats(self):
        import numpy as np

        roi = self._get_center_roi(roi_fraction=0.75)
        if roi is None:
            return {
                "mean": 128.0, "p50": 128.0, "p95": 180.0, "p99": 220.0,
                "p70": 150.0, "bright_pct": 0.0, "dark_pct": 0.0,
                "subject_pct": 0.0, "meter": "fallback"
            }
        roi_u8 = self._normalize_gray_for_analysis(roi)
        flat = roi_u8.reshape(-1)

        # 背景很亮、主体偏暗时，全局均值/高分位会被背景带偏。
        # 用 Otsu 把中央 ROI 分成亮/暗两类，优先对暗类主体测光。
        hist, _ = np.histogram(flat, bins=256, range=(0.0, 255.0))
        total = float(flat.size)
        indices = np.arange(256, dtype=np.float64)
        weight_bg = np.cumsum(hist).astype(np.float64)
        weight_fg = total - weight_bg
        sum_bg = np.cumsum(hist * indices)
        sum_total = sum_bg[-1]
        valid = (weight_bg > 0.0) & (weight_fg > 0.0)
        mean_bg = np.zeros_like(indices)
        mean_fg = np.zeros_like(indices)
        mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
        mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        threshold = float(np.argmax(between))

        dark_mask = roi_u8 <= min(205.0, max(45.0, threshold + 8.0))
        dark_mask &= roi_u8 >= 4.0
        dark_fraction = float(np.mean(dark_mask))
        if 0.04 <= dark_fraction <= 0.88:
            metered = roi_u8[dark_mask]
            meter = "subject"
        else:
            metered = flat
            meter = "center"

        return {
            "mean": float(np.mean(metered)),
            "p50": float(np.percentile(metered, 50.0)),
            "p70": float(np.percentile(metered, 70.0)),
            "p95": float(np.percentile(metered, 95.0)),
            "p99": float(np.percentile(metered, 99.0)),
            "bright_pct": float(np.mean(metered >= 248.0)),
            "dark_pct": float(np.mean(metered <= 8.0)),
            "subject_pct": dark_fraction,
            "meter": meter,
        }

    def _get_image_brightness(self):
        """Get center ROI median brightness normalized to 0-255."""
        return self._get_exposure_stats()["p50"]

    def _af_auto_expose(self, set_status):
        """Auto-adjust exposure and gain before focusing.

        核心策略：先从中央 ROI 中分离偏暗主体，再对主体亮度测光。
        白背景可以过亮一些，主体纹理必须先被拉起来。

        关键修复：settle 时间按当前曝光值动态计算，避免读到旧帧。
        曝光上限 AF_EXP_CAP 防止曝光值失控性增长。
        """
        TARGET_P70 = 125.0    # 主体中高亮度目标，保留纹理不过曝
        OVEREXP_P99 = 245.0   # 主体 p99 超过此值视为过曝
        AF_EXP_CAP = 80000.0  # 自动对焦期间曝光上限 80ms（≥12fps）
        MAX_ITERATIONS = 8
        EXP_MIN = 10.0
        EXP_MAX = 1000000.0
        GAIN_MIN = 0.0
        GAIN_MAX = 10.0

        try:
            self.device_controller.set_exposure(self.DEFAULT_EXPOSURE_US)
            self.device_controller.set_gain(self.DEFAULT_GAIN_DB)
            current_exp = self.DEFAULT_EXPOSURE_US
            current_gain = self.DEFAULT_GAIN_DB
        except Exception:
            current_exp = self.DEFAULT_EXPOSURE_US
            current_gain = self.DEFAULT_GAIN_DB

        current_exp = max(EXP_MIN, min(EXP_MAX, current_exp))
        current_gain = max(GAIN_MIN, min(GAIN_MAX, current_gain))

        for iteration in range(MAX_ITERATIONS):
            if not self.autofocus_running:
                return
            # settle 时间 = max(0.12s, 当前曝光时间×2.2 + 50ms)
            # 保证相机至少输出 2 帧新曝光画面后再读统计
            settle_s = max(0.12, min(0.70, current_exp / 1e6 * 2.2 + 0.05))
            time.sleep(settle_s)

            stats = self._get_exposure_stats()
            if set_status:
                meter_label = "主体" if stats["meter"] == "subject" else "中心"
                set_status("调参中… {} p70:{:.0f} p99:{:.0f} 曝光:{:.0f}µs 增益:{:.1f}dB".format(
                    meter_label, stats["p70"], stats["p99"], current_exp, current_gain))

            overexposed = stats["bright_pct"] > 0.015 or stats["p99"] >= OVEREXP_P99
            good = (
                not overexposed
                and stats["p70"] >= TARGET_P70 - 12.0
                and stats["p70"] <= TARGET_P70 + 35.0
            )
            if good:
                break

            if overexposed:
                ratio = TARGET_P70 / max(stats["p70"], 10.0) * 0.90
                ratio = max(0.45, min(0.90, ratio))
            else:
                ratio = TARGET_P70 / max(stats["p70"], 12.0)
                ratio = max(1.0, min(1.55, ratio))

            new_exp = max(EXP_MIN, min(AF_EXP_CAP, current_exp * ratio))
            new_gain = current_gain
            if overexposed and current_gain > GAIN_MIN:
                new_gain = max(GAIN_MIN, current_gain - 1.0)
                new_exp = current_exp
            elif new_exp >= AF_EXP_CAP * 0.99 and stats["p70"] < TARGET_P70 - 12.0:
                new_gain = min(GAIN_MAX, current_gain + 1.0)

            try:
                self.device_controller.set_exposure(new_exp)
                self.device_controller.set_gain(new_gain)
                current_exp = new_exp
                current_gain = new_gain
            except Exception:
                break

        self._af_update_param_ui()

    def _af_quick_expose(self):
        """精扫阶段的单步曝光微调，逻辑与 _af_auto_expose 一致。"""
        TARGET_P70 = 125.0
        OVEREXP_P99 = 245.0
        AF_EXP_CAP = 80000.0
        EXP_MIN, EXP_MAX = 10.0, 1000000.0
        GAIN_MIN, GAIN_MAX = 0.0, 10.0

        stats = self._get_exposure_stats()
        # 已在合理范围则跳过
        if (
            stats["bright_pct"] <= 0.015
            and stats["p99"] < OVEREXP_P99
            and TARGET_P70 - 12.0 <= stats["p70"] <= TARGET_P70 + 35.0
        ):
            return

        try:
            params = self.device_controller.get_parameters()
            cur_exp = params["exposure_time"]
            cur_gain = params["gain"]
        except Exception:
            return

        overexposed = stats["bright_pct"] > 0.015 or stats["p99"] >= OVEREXP_P99
        if overexposed:
            ratio = TARGET_P70 / max(stats["p70"], 10.0) * 0.90
            ratio = max(0.50, min(0.90, ratio))
        else:
            ratio = TARGET_P70 / max(stats["p70"], 12.0)
            ratio = max(1.0, min(1.5, ratio))

        new_exp = max(EXP_MIN, min(AF_EXP_CAP, cur_exp * ratio))
        new_gain = cur_gain
        if overexposed and cur_gain > GAIN_MIN:
            new_gain = max(GAIN_MIN, cur_gain - 0.8)
            new_exp = cur_exp
        elif new_exp >= AF_EXP_CAP * 0.99 and stats["p70"] < TARGET_P70 - 12.0:
            new_gain = min(GAIN_MAX, cur_gain + 0.8)

        try:
            self.device_controller.set_exposure(new_exp)
            self.device_controller.set_gain(new_gain)
        except Exception:
            pass
        self._af_update_param_ui()

    def _af_update_param_ui(self):
        """Read current camera params and update the UI panel."""
        try:
            params = self.device_controller.get_parameters()
            exp = params["exposure_time"]
            gain = params["gain"]
            fps = params["frame_rate"]
        except Exception:
            return
        def _update():
            self.ui.edtExposureTime.setText("{:.2f}".format(exp))
            self.ui.edtGain.setText("{:.2f}".format(gain))
            self.ui.edtFrameRate.setText("{:.2f}".format(fps))
        QTimer.singleShot(0, _update)

    @staticmethod
    def _brightness_at(new_exp, current_gain, current_brightness, current_exp):
        """Estimate brightness after changing exposure."""
        return current_brightness * (new_exp / max(current_exp, 1.0))

    def _af_move_z(self, step_mm):
        """Move Z relative, wait for completion, then settle."""
        if abs(step_mm) < 0.0001:
            return True
        try:
            self.device_controller.move_z_relative_wait(step_mm, feed=2000)
            time.sleep(max(0.035, min(0.10, abs(step_mm) * 0.14)))
            return True
        except Exception:
            return False

    @staticmethod
    def _af_quadratic_peak(positions, scores, best_idx):
        if best_idx <= 0 or best_idx >= len(scores) - 1:
            return positions[best_idx]
        y0 = float(scores[best_idx - 1])
        y1 = float(scores[best_idx])
        y2 = float(scores[best_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) < 1e-9:
            return positions[best_idx]
        step = float(positions[best_idx + 1] - positions[best_idx])
        offset = 0.5 * (y0 - y2) / denom * step
        offset = max(-abs(step), min(abs(step), offset))
        return float(positions[best_idx] + offset)

    @staticmethod
    def _af_pick_best_index(scores):
        import numpy as np

        arr = np.asarray(scores, dtype=np.float32)
        if arr.size == 0:
            return 0
        best_idx = int(arr.argmax())
        if arr.size < 3 or best_idx not in (0, arr.size - 1):
            return best_idx

        local_maxima = [
            idx for idx in range(1, arr.size - 1)
            if arr[idx] >= arr[idx - 1] and arr[idx] >= arr[idx + 1]
        ]
        if not local_maxima:
            return best_idx

        best_local_idx = max(local_maxima, key=lambda idx: float(arr[idx]))
        if float(arr[best_local_idx]) >= float(arr[best_idx]) * 0.96:
            return int(best_local_idx)
        return best_idx

    @staticmethod
    def _af_peak_confidence(scores, best_idx):
        import numpy as np

        arr = np.asarray(scores, dtype=np.float32)
        if arr.size < 3:
            return 1.0
        best = max(float(arr[best_idx]), 1e-6)
        distances = np.abs(np.arange(arr.size) - int(best_idx))
        side = arr[distances >= 2]
        if side.size == 0:
            side = np.delete(arr, int(best_idx))
        baseline = max(float(np.median(side)), 1e-6)
        return best / baseline

    def _af_scan_window(self, start_pos, end_pos, step, label, accumulated, set_status, sample_count=1):
        import numpy as np

        if not self._af_move_z(start_pos - accumulated):
            return None, None, accumulated
        accumulated = start_pos

        count = int(round((end_pos - start_pos) / step)) + 1
        positions = []
        scores = []
        for i in range(count):
            if not self.autofocus_running:
                set_status("已停止")
                return None, None, accumulated
            score = self._compute_sharpness(sample_count=sample_count)
            positions.append(accumulated)
            scores.append(score)
            self._record_af_curve_sample(label, accumulated, score)
            set_status("{} {}/{}  Z{:+.3f}  锐度:{:.0f}".format(label, i + 1, count, accumulated, score))
            if i < count - 1:
                if not self._af_move_z(step):
                    return None, None, accumulated
                accumulated += step

        return np.asarray(positions, dtype=np.float32), np.asarray(scores, dtype=np.float32), accumulated

    def _record_af_curve_sample(self, phase, z_mm, score):
        phase_name = str(phase).split()[0] if phase else "AF"
        self._af_curve_samples.append({
            "phase": phase_name,
            "z_mm": float(z_mm),
            "score": float(score),
        })

    def _save_autofocus_curve_outputs(self):
        samples = list(self._af_curve_samples)
        if not samples:
            return {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.config_manager.effective_save_path()
        ensure_dir(output_dir)
        base_name = "autofocus_sharpness_curve_{}".format(timestamp)
        png_path = os.path.join(output_dir, base_name + ".png")
        csv_path = os.path.join(output_dir, base_name + ".csv")
        saved = {}
        try:
            if save_autofocus_curve(png_path, samples):
                saved["png"] = png_path
        except Exception:
            pass
        try:
            if save_autofocus_curve_csv(csv_path, samples):
                saved["csv"] = csv_path
        except Exception:
            pass
        self._last_af_curve_paths = saved
        return saved

    def export_last_autofocus_curve(self):
        if not self._af_curve_samples:
            QMessageBox.information(self, "提示", "还没有自动对焦锐度数据，请先运行一次自动对焦。", QMessageBox.Ok)
            return
        default_name = "autofocus_sharpness_curve_{}.png".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        default_path = os.path.join(self.config_manager.effective_save_path(), default_name)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出自动对焦锐度曲线",
            default_path,
            "PNG 图片 (*.png);;所有文件 (*.*)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".png"):
            file_path += ".png"
        try:
            save_autofocus_curve(file_path, self._af_curve_samples)
            QMessageBox.information(self, "完成", "自动对焦锐度曲线已导出：\n{}".format(file_path), QMessageBox.Ok)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc), QMessageBox.Ok)

    def _af_adaptive_refine(self, center_pos, accumulated, set_status):
        import numpy as np

        current_center = float(center_pos)
        all_positions = []
        all_scores = []
        step_plan = [
            (0.180, 0.030, 1),
            (0.075, 0.015, 1),
            (0.032, 0.008, 2),
            (0.016, 0.004, 2),
        ]

        for round_idx, (span, step, sample_count) in enumerate(step_plan, start=1):
            if not self.autofocus_running:
                set_status("已停止")
                return None, None, accumulated

            positions, scores, accumulated = self._af_scan_window(
                current_center - span,
                current_center + span,
                step,
                "ATLAS精扫{}  步长{:.3f}mm".format(round_idx, step),
                accumulated,
                set_status,
                sample_count=sample_count,
            )
            if scores is None:
                return None, None, accumulated

            all_positions.extend([float(pos) for pos in positions])
            all_scores.extend([float(score) for score in scores])
            best_idx = self._af_pick_best_index(scores)
            current_center = self._af_quadratic_peak(positions, scores, best_idx)

            if float(np.max(scores)) <= 0.0:
                break

        if not all_scores:
            return None, None, accumulated

        all_positions = np.asarray(all_positions, dtype=np.float32)
        all_scores = np.asarray(all_scores, dtype=np.float32)
        order = np.argsort(all_positions)
        return all_positions[order], all_scores[order], accumulated

    def _autofocus_worker(self):
        import numpy as np

        def set_status(message):
            QTimer.singleShot(0, lambda m=message: self.ui.lblAutoFocusStatus.setText(m))

        try:
            self._af_curve_samples = []
            self._last_af_curve_paths = {}
            # Phase 1a: 窄范围快速粗扫 ±0.8mm / 0.2mm 步 → 9 个位置
            # 样品通常已接近焦点，大多数情况这一阶段即可定位
            accumulated = 0.0
            coarse_step = 0.20
            coarse_positions, coarse_scores, accumulated = self._af_scan_window(
                -1.0, 1.0, coarse_step, "ATLAS粗扫", accumulated, set_status, sample_count=1
            )
            if coarse_scores is None:
                return
            best_idx = self._af_pick_best_index(coarse_scores)
            coarse_peak = self._af_quadratic_peak(coarse_positions, coarse_scores, best_idx)

            # Phase 1b: 峰值在边界 → 向外以 0.4mm 步扩展到 ±2mm
            if best_idx in (0, len(coarse_scores) - 1) and self.autofocus_running:
                direction = -1.0 if best_idx == 0 else 1.0
                ext_start = float(coarse_positions[best_idx] + direction * 0.4)
                ext_end = float(coarse_positions[best_idx] + direction * 2.2)
                ext_positions, ext_scores, accumulated = self._af_scan_window(
                    ext_start, ext_end, 0.4 * direction, "ATLAS扩展粗扫", accumulated, set_status, sample_count=1
                )
                if ext_scores is not None and len(ext_scores) > 0:
                    all_positions = np.concatenate([coarse_positions, ext_positions])
                    all_scores = np.concatenate([coarse_scores, ext_scores])
                    order = np.argsort(all_positions)
                    all_positions = all_positions[order]
                    all_scores = all_scores[order]
                    best_idx = self._af_pick_best_index(all_scores)
                    coarse_peak = self._af_quadratic_peak(all_positions, all_scores, best_idx)

            if not self._af_move_z(coarse_peak - accumulated):
                set_status("对焦失败：串口错误")
                return
            accumulated = coarse_peak

            fine_positions, fine_scores, accumulated = self._af_adaptive_refine(coarse_peak, accumulated, set_status)
            if fine_scores is None:
                return
            best_idx_f = self._af_pick_best_index(fine_scores)
            best_fine_pos = self._af_quadratic_peak(fine_positions, fine_scores, best_idx_f)
            best_measured_pos = float(fine_positions[best_idx_f])
            best_measured_score = float(fine_scores[best_idx_f])
            confidence = self._af_peak_confidence(fine_scores, best_idx_f)

            if not self._af_move_z(best_fine_pos - accumulated):
                set_status("对焦失败：串口错误")
                return

            if not self.autofocus_running:
                set_status("已停止")
                return
            # 最终确认：实测分数明显低于精扫最优时回退到实测峰位
            final_score = self._compute_sharpness(sample_count=2, roi_fraction=0.72)
            if final_score < best_measured_score * 0.88:
                if self._af_move_z(best_measured_pos - best_fine_pos):
                    best_fine_pos = best_measured_pos
                    final_score = self._compute_sharpness(sample_count=2, roi_fraction=0.72)
            self._record_af_curve_sample("最终确认", best_fine_pos, final_score)
            if best_measured_score <= 0.0:
                status_msg = "对焦失败：有效纹理不足，请调整光照或样品位置"
                saved = self._save_autofocus_curve_outputs()
                if saved.get("png"):
                    status_msg += "  曲线: {}".format(saved["png"])
                set_status(status_msg)
            elif confidence < 1.06:
                status_msg = "ATLAS完成 △ 峰值不明显  位置偏移:{:+.3f}mm  锐度:{:.0f}  置信:{:.2f}".format(
                    best_fine_pos, final_score, confidence)
                saved = self._save_autofocus_curve_outputs()
                if saved.get("png"):
                    status_msg += "  曲线: {}".format(saved["png"])
                set_status(status_msg)
            else:
                status_msg = "ATLAS完成 ✓  位置偏移:{:+.3f}mm  锐度:{:.0f}  置信:{:.2f}".format(
                    best_fine_pos, final_score, confidence)
                saved = self._save_autofocus_curve_outputs()
                if saved.get("png"):
                    status_msg += "  曲线: {}".format(saved["png"])
                set_status(status_msg)
        except Exception as exc:
            set_status("对焦失败: " + str(exc))
        finally:
            stopped_by_user = not self.autofocus_running
            self.autofocus_running = False
            def _af_cleanup():
                self.ui.bnAutoFocus.setEnabled(True)
                self.ui.bnStopAutoFocus.setEnabled(False)
                self._update_z_display()
                if stopped_by_user:
                    self.ui.lblAutoFocusStatus.setText("已停止")
            QTimer.singleShot(0, _af_cleanup)

    def start_autofocus(self):
        if not self.device_controller.serial_connected:
            QMessageBox.warning(self, "错误", "请先连接串口！", QMessageBox.Ok)
            return
        if self.autofocus_running:
            QMessageBox.warning(self, "提示", "对焦正在进行！", QMessageBox.Ok)
            return
        # Button click → global focus; double-click path sets _af_roi_center before calling here
        if self.sender() is self.ui.bnAutoFocus:
            self._af_roi_center = None
        self.autofocus_running = True
        self.ui.bnAutoFocus.setEnabled(False)
        self.ui.bnStopAutoFocus.setEnabled(True)
        threading.Thread(target=self._autofocus_worker, daemon=True).start()

    def start_autofocus_worker_only(self):
        """供可编程拍摄调用：直接启动对焦线程，不做 UI 弹窗检查。"""
        threading.Thread(target=self._autofocus_worker, daemon=True).start()

    def stop_autofocus(self):
        self.autofocus_running = False
        self.ui.bnAutoFocus.setEnabled(True)
        self.ui.bnStopAutoFocus.setEnabled(False)
        self.ui.lblAutoFocusStatus.setText("已停止")

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
                "帧大小: {}x{}\n采样: {}帧\n均值: {:.1f}\n底噪帧已就绪".format(
                    info["width"], info["height"], info.get("frames", 1), info["mean"]
                )
            )
        except Exception as exc:
            QMessageBox.warning(self, "错误", "采集底噪帧失败:\n" + str(exc), QMessageBox.Ok)

    def toggle_dark_sub(self, state):
        enabled = state == Qt.Checked
        self.device_controller.set_dark_sub_enabled(enabled)
        self._update_dark_sub_status_label(self.dark_frame_captured, enabled)

    def toggle_hdr(self, state):
        enabled = state == Qt.Checked
        self.device_controller.set_hdr_enabled(enabled)
        self.ui.lblHdrStatus.setText("实时增强中" if enabled else "未开启")

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
        self._sync_z_from_device(show_error=False)
        if self._recon3d_dialog is None:
            self._recon3d_dialog = PointCloudReconDialog(self.device_controller, self.config_manager, self)
        self._recon3d_dialog.sync_z_inputs_to_current()
        self._recon3d_dialog.show()
        self._recon3d_dialog.raise_()
        self._recon3d_dialog.activateWindow()

    def open_temporal_depth_dialog(self):
        self._sync_z_from_device(show_error=False)
        if self._temporal_depth_dialog is None:
            self._temporal_depth_dialog = TemporalDepthDialog(self.device_controller, self.config_manager, self)
        self._temporal_depth_dialog.sync_z_inputs_to_current()
        self._temporal_depth_dialog.show()
        self._temporal_depth_dialog.raise_()
        self._temporal_depth_dialog.activateWindow()

    def open_one_click_dialog(self):
        self._sync_z_from_device(show_error=False)
        if self._one_click_dialog is None:
            self._one_click_dialog = OneClickDialog(self.device_controller, self.config_manager, self)
        self._one_click_dialog.sync_z_inputs_to_current()
        self._one_click_dialog.show()
        self._one_click_dialog.raise_()
        self._one_click_dialog.activateWindow()

    def open_programmable_shooting_dialog(self):
        if self._programmable_dialog is None:
            self._programmable_dialog = ProgrammableShootingDialog(
                self.device_controller, self.config_manager, self, self)
        self._programmable_dialog.show()
        self._programmable_dialog.raise_()
        self._programmable_dialog.activateWindow()

    def open_offline_zstack_dialog(self):
        if self._offline_zstack_dialog is None:
            self._offline_zstack_dialog = OfflineZStackDialog(self.config_manager, self)
        self._offline_zstack_dialog.show()
        self._offline_zstack_dialog.raise_()
        self._offline_zstack_dialog.activateWindow()

    def enable_controls(self):
        is_open = self.device_controller.opened
        is_grabbing = self.device_controller.grabbing
        self.ui.groupGrab.setEnabled(is_open)
        self.ui.groupParam.setEnabled(is_open)
        self.ui.groupHdr.setEnabled(is_open)

        self.ui.bnOpen.setEnabled(not is_open)
        self.ui.bnClose.setEnabled(is_open)
        self.ui.bnStart.setEnabled(is_open and not is_grabbing)
        self.ui.bnStop.setEnabled(is_open and is_grabbing)
        self.ui.bnAutoFocus.setEnabled(is_open and is_grabbing and not self.autofocus_running)
        self.ui.bnStopAutoFocus.setEnabled(self.autofocus_running)
        self.ui.bnQuickScale.setEnabled(is_open and is_grabbing and not self.quick_scale_running)
        self.ui.bnCaptureDark.setEnabled(is_open and is_grabbing)
        self._update_z_motion_buttons()
        if not (is_open and is_grabbing):
            self.device_controller.set_dark_sub_enabled(False)
            self.device_controller.set_hdr_enabled(False)
            self.ui.chkDarkSub.setChecked(False)
            self.ui.chkHdr.setChecked(False)
            self.ui.lblHdrStatus.setText("未开启")

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
            self.scale_overlay.hide()
            self.scale_overlay.close()
        except Exception:
            pass
        try:
            self.device_controller.cleanup()
        except Exception:
            pass

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)
