import time
from ctypes import POINTER, cast

try:
    import serial
    import serial.tools.list_ports

    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False

from sdk.CamOperation_class import CameraOperation
from sdk.MvCameraControl_class import MvCamera
from sdk.MvErrorDefine_const import MV_E_CALLORDER, MV_OK
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
        self.cam = MvCamera()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.obj_cam_operation = None
        self.opened = False
        self.grabbing = False
        self.serial_conn = None
        self.serial_connected = False
        self._sdk_initialized = False
        self._z_soft_limit = 68.0
        self._z_position = 0.0

    def initialize_sdk(self):
        if not self._sdk_initialized:
            MvCamera.MV_CC_Initialize()
            self._sdk_initialized = True

    def finalize_sdk(self):
        if self._sdk_initialized:
            MvCamera.MV_CC_Finalize()
            self._sdk_initialized = False

    def cleanup(self):
        self.disconnect_serial()
        self.close_camera()
        self.finalize_sdk()

    def enum_devices(self):
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

    def set_software_trigger_mode(self):
        self._ensure_camera()
        ret = self.obj_cam_operation.Set_trigger_mode(True)
        if ret != 0:
            raise RuntimeError("Set trigger mode failed ret:{}".format(to_hex_str(ret)))

    def trigger_once(self):
        self._ensure_camera()
        ret = self.obj_cam_operation.Trigger_once()
        if ret != 0:
            raise RuntimeError("TriggerSoftware failed ret:{}".format(to_hex_str(ret)))

    def save_bmp(self):
        self._ensure_camera()
        ret = self.obj_cam_operation.Save_Bmp()
        if ret != MV_OK:
            raise RuntimeError("Save BMP failed ret:{}".format(to_hex_str(ret)))

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
        ret = self.obj_cam_operation.obj_cam.MV_CC_SetFloatValue("Gain", float(gain_db))
        if ret != 0:
            raise RuntimeError("Set gain failed ret:{}".format(to_hex_str(ret)))

    def get_frame_numpy(self):
        self._ensure_camera()
        return self.obj_cam_operation.Get_frame_numpy()

    def get_gray_frame(self):
        import numpy as np

        data, w, h = self.get_frame_numpy()
        if data is None or w == 0 or h == 0 or len(data) < w * h:
            return None, 0, 0
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
        self.send_gcode("G91\n")

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

    def try_send_gcode(self, cmd):
        try:
            self.send_gcode(cmd)
            return True
        except Exception:
            return False

    def home_z(self):
        self.send_gcode("G28 Z\n")
        self._z_position = 0.0
        self.send_gcode("G91\n")

    def _check_z_soft_limit(self, target_z):
        if target_z >= self._z_soft_limit:
            self.send_gcode("M211 S1\n")
            print("[安全] Z={:.2f}mm >= {:.0f}mm，已启用软限位 (M211 S1)".format(
                target_z, self._z_soft_limit))
            raise RuntimeError(
                "Z 轴已达 {:.0f}mm 安全限位，已自动启用软限位防止撞击样品。".format(
                    self._z_soft_limit))

    def move_z_relative(self, step_mm, feed=300):
        self._check_z_soft_limit(self._z_position + step_mm)
        self.send_gcode("G91\nG1 Z{:.4f} F{}\n".format(step_mm, feed))
        self._z_position += step_mm

    def move_z_relative_wait(self, step_mm, feed=2000):
        """Move Z relative and wait for physical completion using M400."""
        self._check_z_soft_limit(self._z_position + step_mm)
        self.send_gcode("G91\nG1 Z{:.4f} F{}\n".format(step_mm, feed))
        self._z_position += step_mm
        self.send_gcode_wait("M400\n", timeout=10.0)

    def move_z_absolute(self, position_mm, feed=300):
        self._check_z_soft_limit(position_mm)
        self.send_gcode("G90\n")
        self.send_gcode("G1 Z{:.4f} F{}\n".format(position_mm, feed))
        self.send_gcode("G91\n")
        self._z_position = position_mm

    def set_light(self, value):
        self.send_gcode("M106 S{}\n".format(int(value)))

    def capture_dark_frame(self):
        import numpy as np

        data, w, h = self.get_frame_numpy()
        if data is None or w == 0 or h == 0 or len(data) < w * h:
            raise RuntimeError("无法获取当前帧，请确认相机正在输出图像。")
        dark = data.astype(np.int16)
        self.obj_cam_operation.dark_frame = dark
        return {"width": w, "height": h, "mean": float(np.mean(data[: w * h]))}

    def clear_dark_frame(self):
        if self.obj_cam_operation:
            self.obj_cam_operation.dark_frame = None
            self.obj_cam_operation.apply_dark_sub = False

    def set_dark_sub_enabled(self, enabled):
        if self.obj_cam_operation:
            self.obj_cam_operation.apply_dark_sub = bool(enabled)

    def _ensure_camera(self):
        if not self.obj_cam_operation or not self.opened:
            raise RuntimeError("Camera is not open")
