import time
import re
import threading
from ctypes import POINTER, cast

try:
    import serial
    import serial.tools.list_ports

    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False

try:
    from sdk.CamOperation_class import CameraOperation
    from sdk.MvCameraControl_class import MvCamera, _SDK_LOAD_ERROR as _MV_SDK_ERR
    from sdk.MvErrorDefine_const import MV_OK
    from sdk.CameraParams_header import (
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_GENTL_CAMERALINK_DEVICE,
        MV_GENTL_CXP_DEVICE,
        MV_GENTL_GIGE_DEVICE,
        MV_GENTL_XOF_DEVICE,
        MV_GIGE_DEVICE,
        MV_USB_DEVICE,
    )
    MV_SDK_AVAILABLE = (_MV_SDK_ERR is None)
    MV_SDK_ERROR_MSG = _MV_SDK_ERR or ""
except Exception as _e:
    MV_SDK_AVAILABLE = False
    MV_SDK_ERROR_MSG = str(_e)
    # Provide lightweight stubs so the rest of the module can still be imported
    MvCamera = None
    CameraOperation = None
    MV_OK = 0
    MV_CC_DEVICE_INFO = None
    MV_CC_DEVICE_INFO_LIST = None
    MV_GENTL_CAMERALINK_DEVICE = MV_GENTL_CXP_DEVICE = MV_GENTL_GIGE_DEVICE = 0
    MV_GENTL_XOF_DEVICE = MV_GIGE_DEVICE = MV_USB_DEVICE = 0


def to_hex_str(num):
    chars = {10: "a", 11: "b", 12: "c", 13: "d", 14: "e", 15: "f"}
    hex_str = ""
    if num < 0:
        num += 2 ** 32
    while num >= 16:
        digit = num % 16
        hex_str = chars.get(digit, str(digit)) + hex_str
        num //= 16
    return chars.get(num, str(num)) + hex_str


def decode_ctypes_string(value):
    byte_str = memoryview(value).tobytes()
    null_index = byte_str.find(b"\x00")
    if null_index >= 0:
        byte_str = byte_str[:null_index]
    for encoding in ["gbk", "utf-8", "latin-1"]:
        try:
            return byte_str.decode(encoding)
        except UnicodeDecodeError:
            continue
    return byte_str.decode("latin-1", errors="replace")


class DeviceController:
    def __init__(self):
        self.cam = MvCamera() if MV_SDK_AVAILABLE else None
        self.device_list = MV_CC_DEVICE_INFO_LIST() if MV_SDK_AVAILABLE else None
        self.obj_cam_operation = None
        self.opened = False
        self.grabbing = False
        self.serial_conn = None
        self.serial_connected = False
        self._sdk_initialized = False
        self._z_soft_limit = 68.0
        self._z_min_limit = -3.0
        self._z_position = 0.0
        self._z_origin_offset = 0.0
        self._serial_lock = threading.RLock()

    def initialize_sdk(self):
        if not MV_SDK_AVAILABLE:
            return
        if not self._sdk_initialized:
            MvCamera.MV_CC_Initialize()
            self._sdk_initialized = True

    def finalize_sdk(self):
        if not MV_SDK_AVAILABLE:
            return
        if self._sdk_initialized:
            MvCamera.MV_CC_Finalize()
            self._sdk_initialized = False

    def cleanup(self):
        self.disconnect_serial()
        self.close_camera()
        self.finalize_sdk()

    def enum_devices(self):
        if not MV_SDK_AVAILABLE:
            raise RuntimeError(
                "相机 SDK 未安装，无法查找设备。\n"
                "请安装海康威视 MVS SDK 后重新运行。\n\n"
                + ("详细信息：" + MV_SDK_ERROR_MSG if MV_SDK_ERROR_MSG else "")
            )
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        layer_type = (
            MV_GIGE_DEVICE
            | MV_USB_DEVICE
            | MV_GENTL_CAMERALINK_DEVICE
            | MV_GENTL_CXP_DEVICE
            | MV_GENTL_XOF_DEVICE
        )
        ret = MvCamera.MV_CC_EnumDevices(layer_type, self.device_list)
        if ret != 0:
            raise RuntimeError("Enum devices fail! ret = {}".format(to_hex_str(ret)))

        devices = []
        for index in range(self.device_list.nDeviceNum):
            info = cast(self.device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
            devices.append(self._format_device_info(index, info))
        return devices

    def _format_device_info(self, index, info):
        if info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            user_defined_name = decode_ctypes_string(info.SpecialInfo.stGigEInfo.chUserDefinedName)
            model_name = decode_ctypes_string(info.SpecialInfo.stGigEInfo.chModelName)
            nip1 = (info.SpecialInfo.stGigEInfo.nCurrentIp & 0xFF000000) >> 24
            nip2 = (info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00FF0000) >> 16
            nip3 = (info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000FF00) >> 8
            nip4 = info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000FF
            return "[{}]GigE: {} {} ({}.{}.{}.{})".format(
                index, user_defined_name, model_name, nip1, nip2, nip3, nip4
            )
        if info.nTLayerType == MV_USB_DEVICE:
            user_defined_name = decode_ctypes_string(info.SpecialInfo.stUsb3VInfo.chUserDefinedName)
            model_name = decode_ctypes_string(info.SpecialInfo.stUsb3VInfo.chModelName)
            serial_number = "".join(chr(item) for item in info.SpecialInfo.stUsb3VInfo.chSerialNumber if item != 0)
            return "[{}]USB: {} {} ({})".format(index, user_defined_name, model_name, serial_number)
        if info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
            user_defined_name = decode_ctypes_string(info.SpecialInfo.stCMLInfo.chUserDefinedName)
            model_name = decode_ctypes_string(info.SpecialInfo.stCMLInfo.chModelName)
            serial_number = "".join(chr(item) for item in info.SpecialInfo.stCMLInfo.chSerialNumber if item != 0)
            return "[{}]CML: {} {} ({})".format(index, user_defined_name, model_name, serial_number)
        if info.nTLayerType == MV_GENTL_CXP_DEVICE:
            user_defined_name = decode_ctypes_string(info.SpecialInfo.stCXPInfo.chUserDefinedName)
            model_name = decode_ctypes_string(info.SpecialInfo.stCXPInfo.chModelName)
            serial_number = "".join(chr(item) for item in info.SpecialInfo.stCXPInfo.chSerialNumber if item != 0)
            return "[{}]CXP: {} {} ({})".format(index, user_defined_name, model_name, serial_number)
        user_defined_name = decode_ctypes_string(info.SpecialInfo.stXoFInfo.chUserDefinedName)
        model_name = decode_ctypes_string(info.SpecialInfo.stXoFInfo.chModelName)
        serial_number = "".join(chr(item) for item in info.SpecialInfo.stXoFInfo.chSerialNumber if item != 0)
        return "[{}]XoF: {} {} ({})".format(index, user_defined_name, model_name, serial_number)

    def open_camera(self, index):
        if self.opened:
            raise RuntimeError("Camera is Running!")
        if index < 0:
            raise RuntimeError("Please select a camera!")
        self.obj_cam_operation = CameraOperation(self.cam, self.device_list, index)
        ret = self.obj_cam_operation.Open_device()
        if ret != 0:
            self.obj_cam_operation = None
            raise RuntimeError("Open device failed ret:{}".format(to_hex_str(ret)))
        self.opened = True
        self.set_continue_mode()
        return self.get_parameters()

    def close_camera(self):
        if self.obj_cam_operation and self.opened:
            self.obj_cam_operation.Close_device()
        self.obj_cam_operation = None
        self.opened = False
        self.grabbing = False

    def start_grabbing(self, win_id):
        self._ensure_camera()
        ret = self.obj_cam_operation.Start_grabbing(win_id)
        if ret != 0:
            raise RuntimeError("Start grabbing failed ret:{}".format(to_hex_str(ret)))
        self.grabbing = True

    def stop_grabbing(self):
        self._ensure_camera()
        ret = self.obj_cam_operation.Stop_grabbing()
        if ret != 0:
            raise RuntimeError("Stop grabbing failed ret:{}".format(to_hex_str(ret)))
        self.grabbing = False

    def set_continue_mode(self):
        self._ensure_camera()
        ret = self.obj_cam_operation.Set_trigger_mode(False)
        if ret != 0:
            raise RuntimeError("Set continue mode failed ret:{}".format(to_hex_str(ret)))

    def save_bmp_with_path(self, path):
        self._ensure_camera()
        return self.obj_cam_operation.Save_Bmp_with_path(path)

    def get_parameters(self):
        self._ensure_camera()
        ret = self.obj_cam_operation.Get_parameter()
        if ret != MV_OK:
            raise RuntimeError("Get param failed ret:{}".format(to_hex_str(ret)))
        return {
            "exposure_time": self.obj_cam_operation.exposure_time,
            "gain": self.obj_cam_operation.gain,
            "frame_rate": self.obj_cam_operation.frame_rate,
        }

    def set_parameters(self, frame_rate, exposure, gain):
        self._ensure_camera()
        ret = self.obj_cam_operation.Set_parameter(frame_rate, exposure, gain)
        if ret != MV_OK:
            raise RuntimeError("Set param failed ret:{}".format(to_hex_str(ret)))
        return ret

    def set_exposure(self, exposure_us):
        """Set exposure time in microseconds directly."""
        self._ensure_camera()
        self.obj_cam_operation.obj_cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        ret = self.obj_cam_operation.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))
        if ret != 0:
            raise RuntimeError("Set exposure failed ret:{}".format(to_hex_str(ret)))

    def set_gain(self, gain_db):
        """Set gain in dB directly."""
        self._ensure_camera()
        try:
            self.obj_cam_operation.obj_cam.MV_CC_SetEnumValue("GainAuto", 0)
        except Exception:
            pass
        ret = self.obj_cam_operation.obj_cam.MV_CC_SetFloatValue("Gain", float(gain_db))
        if ret != 0:
            raise RuntimeError("Set gain failed ret:{}".format(to_hex_str(ret)))

    def get_frame_num(self):
        """返回相机当前帧的序号，可用于判断是否有新帧到来。"""
        if self.obj_cam_operation is None:
            return -1
        return int(self.obj_cam_operation.st_frame_info.nFrameNum)

    def get_frame_numpy(self):
        self._ensure_camera()
        return self.obj_cam_operation.Get_frame_numpy()

    def get_color_frame(self):
        """Return RGB uint8 frame when the camera provides color data."""
        self._ensure_camera()
        rgb, w, h = self.obj_cam_operation.Get_frame_rgb_numpy()
        if rgb is None or w == 0 or h == 0:
            return None, 0, 0
        return rgb, w, h

    def get_gray_color_frame(self):
        """Return analysis gray frame plus an RGB frame for color output."""
        import numpy as np

        rgb, w, h = self.get_color_frame()
        if rgb is not None:
            rgb_f = rgb.astype(np.float32)
            gray = (
                0.299 * rgb_f[:, :, 0]
                + 0.587 * rgb_f[:, :, 1]
                + 0.114 * rgb_f[:, :, 2]
            ).astype(np.float32)
            return gray, rgb.copy(), w, h

        gray, w, h = self.get_gray_frame()
        if gray is None:
            return None, None, 0, 0
        display = np.clip(gray, 0, 255).astype(np.uint8)
        color = np.repeat(display[:, :, None], 3, axis=2)
        return gray, color, w, h

    def get_gray_frame(self):
        import numpy as np

        data, w, h = self.get_frame_numpy()
        if data is None or w == 0 or h == 0 or len(data) < w * h:
            return None, 0, 0
        # Preserve high-bit-depth Mono10/Mono12 frames as float values for DFF and TIFF export.
        gray = data[: w * h].reshape(h, w).astype(np.float32)
        return gray, w, h

    def list_serial_ports(self):
        if not SERIAL_AVAILABLE:
            return []
        return [port.device for port in serial.tools.list_ports.comports()]

    def connect_serial(self, port, baudrate, timeout):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial 未安装，请执行: pip install pyserial")
        self.serial_conn = serial.Serial(
            port=port,
            baudrate=int(baudrate),
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=float(timeout),
        )
        self.serial_connected = True
        with self._serial_lock:
            self.send_gcode("G90\n")
        self.refresh_z_position(timeout=2.0)

    def disconnect_serial(self):
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.serial_conn = None
        self.serial_connected = False

    def send_gcode(self, cmd):
        if not self.serial_connected or self.serial_conn is None:
            raise RuntimeError("串口未连接，请先连接串口！")
        with self._serial_lock:
            self.serial_conn.write(cmd.encode("utf-8"))
        return True

    def flush_serial_input(self):
        """Clear any pending data in serial input buffer."""
        if self.serial_conn and self.serial_conn.in_waiting:
            self.serial_conn.read(self.serial_conn.in_waiting)

    def send_gcode_wait(self, cmd, timeout=10.0):
        """Send G-code and wait for 'ok' response from firmware."""
        if not self.serial_connected or self.serial_conn is None:
            raise RuntimeError("串口未连接，请先连接串口！")
        with self._serial_lock:
            self.flush_serial_input()
            self.serial_conn.write(cmd.encode("utf-8"))
            import time as _time
            deadline = _time.time() + timeout
            while _time.time() < deadline:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode("utf-8", errors="ignore").strip().lower()
                    if line.startswith("ok"):
                        return True
                else:
                    _time.sleep(0.01)
        return False

    @staticmethod
    def _parse_z_from_position_line(line):
        match = re.search(r"(?:^|\s)Z:\s*(-?\d+(?:\.\d+)?)", line, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def refresh_z_position(self, timeout=1.0):
        """Read the firmware-reported Z position with M114 and update the shared cache."""
        if not self.serial_connected or self.serial_conn is None:
            return self._z_position
        found_z = self._read_raw_z_position(timeout)
        if found_z is not None:
            self._z_position = found_z - self._z_origin_offset
        return self._z_position

    def _read_raw_z_position(self, timeout=1.0):
        if not self.serial_connected or self.serial_conn is None:
            return None
        with self._serial_lock:
            self.flush_serial_input()
            self.serial_conn.write(b"M114\n")
            deadline = time.time() + float(timeout)
            found_z = None
            while time.time() < deadline:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
                    parsed = self._parse_z_from_position_line(line)
                    if parsed is not None:
                        found_z = parsed
                    if line.lower().startswith("ok") and found_z is not None:
                        break
                else:
                    time.sleep(0.01)
            return found_z

    def _check_z_soft_limit(self, target_z):
        if target_z <= self._z_min_limit:
            self.send_gcode("M211 S1\n")
            print("[安全] Z={:.2f}mm <= {:.1f}mm，已启用软限位 (M211 S1)".format(
                target_z, self._z_min_limit))
            raise RuntimeError(
                "Z 轴已达 {:.1f}mm 最低安全限位，已自动启用软限位防止电机断电。".format(
                    self._z_min_limit))
        if target_z >= self._z_soft_limit:
            self.send_gcode("M211 S1\n")
            print("[安全] Z={:.2f}mm >= {:.0f}mm，已启用软限位 (M211 S1)".format(
                target_z, self._z_soft_limit))
            raise RuntimeError(
                "Z 轴已达 {:.0f}mm 安全限位，已自动启用软限位防止撞击样品。".format(
                    self._z_soft_limit))

    def move_z_relative(self, step_mm, feed=300):
        self._check_z_soft_limit(self._z_position + step_mm)
        with self._serial_lock:
            self.send_gcode("G91\nG1 Z{:.4f} F{}\nG90\n".format(step_mm, feed))
            self._z_position += step_mm

    def move_z_relative_wait(self, step_mm, feed=2000):
        """Move Z relative and wait for physical completion using M400."""
        self._check_z_soft_limit(self._z_position + step_mm)
        with self._serial_lock:
            self.send_gcode("G91\nG1 Z{:.4f} F{}\nG90\n".format(step_mm, feed))
            self._z_position += step_mm
            self.send_gcode_wait("M400\n", timeout=10.0)
        self.refresh_z_position(timeout=0.8)

    def move_z_absolute(self, position_mm, feed=300):
        self._check_z_soft_limit(position_mm)
        command_z = float(position_mm) + self._z_origin_offset
        with self._serial_lock:
            self.send_gcode("G90\nG1 Z{:.4f} F{}\n".format(command_z, feed))
            self._z_position = position_mm

    def move_z_absolute_wait(self, position_mm, feed=300, timeout=20.0):
        self.move_z_absolute(position_mm, feed=feed)
        self.send_gcode_wait("M400\n", timeout=timeout)
        self.refresh_z_position(timeout=0.8)

    def home_z_wait(self, timeout=30.0):
        """Home Z and synchronize cached position from firmware."""
        with self._serial_lock:
            self.send_gcode("M999\n")
            self.send_gcode("M211 S0\n")
            self.send_gcode("G91\nG1 Z-3 F100\nG90\n")
            self.send_gcode_wait("M400\n", timeout=10.0)
            self.send_gcode("G28 Z\n")
            self.send_gcode_wait("M400\n", timeout=timeout)
            # Some controllers keep a homing offset and report Z=5 after G28.
            # The microscope UI treats the homed focus reference as 0.000 mm.
            self.send_gcode("G92 Z0\n")
            self.send_gcode("M211 S1\n")
        raw_z = self._read_raw_z_position(timeout=1.0)
        self._z_origin_offset = raw_z if raw_z is not None else 0.0
        self._z_position = 0.0
        return 0.0

    def capture_dark_frame(self, frame_count=50):
        import numpy as np

        self._ensure_camera()
        previous_enabled = bool(self.obj_cam_operation.apply_dark_sub)
        self.obj_cam_operation.apply_dark_sub = False
        frames = []
        width = 0
        height = 0
        try:
            for _ in range(max(1, int(frame_count))):
                data, w, h = self.get_frame_numpy()
                if data is None or w == 0 or h == 0 or len(data) < w * h:
                    time.sleep(0.03)
                    continue
                frame = data[: w * h].astype(np.float32)
                if frames and (w != width or h != height):
                    continue
                width, height = w, h
                frames.append(frame)
                time.sleep(0.03)
        finally:
            self.obj_cam_operation.apply_dark_sub = previous_enabled

        if not frames:
            raise RuntimeError("无法获取当前帧，请确认相机正在输出图像。")

        dark = np.mean(np.stack(frames, axis=0), axis=0).astype(np.float32)
        self.obj_cam_operation.dark_frame = dark
        return {
            "width": width,
            "height": height,
            "mean": float(np.mean(dark)),
            "frames": len(frames),
        }

    def clear_dark_frame(self):
        if self.obj_cam_operation:
            self.obj_cam_operation.dark_frame = None
            self.obj_cam_operation.apply_dark_sub = False

    def set_dark_sub_enabled(self, enabled):
        if self.obj_cam_operation:
            self.obj_cam_operation.apply_dark_sub = bool(enabled)

    def set_hdr_enabled(self, enabled):
        if self.obj_cam_operation:
            self.obj_cam_operation.apply_hdr = bool(enabled)

    def _ensure_camera(self):
        if not self.obj_cam_operation or not self.opened:
            raise RuntimeError("Camera is not open")
