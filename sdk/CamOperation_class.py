# -- coding: utf-8 --
import threading
import time
import sys
import inspect
import ctypes
import random
import os
import platform
from ctypes import *

currentsystem = platform.system()
if currentsystem == 'Windows':
    _mvcam_env = os.getenv('MVCAM_COMMON_RUNENV')
    if _mvcam_env:
        sys.path.append(os.path.join(_mvcam_env, "Samples", "Python", "MvImport"))
else:
    sys.path.append(os.path.join("..", "..", "MvImport"))

from CameraParams_header import *
from MvCameraControl_class import *

# 强制关闭线程
def Async_raise(tid, exctype):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


# 停止线程
def Stop_thread(thread):
    Async_raise(thread.ident, SystemExit)


# 转为16进制字符串
def To_hex_str(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr


# 是否是Mono图像
def Is_mono_data(enGvspPixelType):
    if PixelType_Gvsp_Mono8 == enGvspPixelType or PixelType_Gvsp_Mono10 == enGvspPixelType \
            or PixelType_Gvsp_Mono10_Packed == enGvspPixelType or PixelType_Gvsp_Mono12 == enGvspPixelType \
            or PixelType_Gvsp_Mono12_Packed == enGvspPixelType:
        return True
    else:
        return False


# 是否是彩色图像
def Is_color_data(enGvspPixelType):
    if PixelType_Gvsp_BayerGR8 == enGvspPixelType or PixelType_Gvsp_BayerRG8 == enGvspPixelType \
            or PixelType_Gvsp_BayerGB8 == enGvspPixelType or PixelType_Gvsp_BayerBG8 == enGvspPixelType \
            or PixelType_Gvsp_BayerGR10 == enGvspPixelType or PixelType_Gvsp_BayerRG10 == enGvspPixelType \
            or PixelType_Gvsp_BayerGB10 == enGvspPixelType or PixelType_Gvsp_BayerBG10 == enGvspPixelType \
            or PixelType_Gvsp_BayerGR12 == enGvspPixelType or PixelType_Gvsp_BayerRG12 == enGvspPixelType \
            or PixelType_Gvsp_BayerGB12 == enGvspPixelType or PixelType_Gvsp_BayerBG12 == enGvspPixelType \
            or PixelType_Gvsp_BayerGR10_Packed == enGvspPixelType or PixelType_Gvsp_BayerRG10_Packed == enGvspPixelType \
            or PixelType_Gvsp_BayerGB10_Packed == enGvspPixelType or PixelType_Gvsp_BayerBG10_Packed == enGvspPixelType \
            or PixelType_Gvsp_BayerGR12_Packed == enGvspPixelType or PixelType_Gvsp_BayerRG12_Packed == enGvspPixelType \
            or PixelType_Gvsp_BayerGB12_Packed == enGvspPixelType or PixelType_Gvsp_BayerBG12_Packed == enGvspPixelType \
            or PixelType_Gvsp_BayerRBGG8 == enGvspPixelType \
            or PixelType_Gvsp_BayerGR16 == enGvspPixelType  or PixelType_Gvsp_BayerRG16 == enGvspPixelType or PixelType_Gvsp_BayerGB16 == enGvspPixelType or  PixelType_Gvsp_BayerBG16 == enGvspPixelType \
            or PixelType_Gvsp_YUV422_Packed == enGvspPixelType or PixelType_Gvsp_YUV422_YUYV_Packed == enGvspPixelType:
        return True
    else:
        return False


# 相机操作类
class CameraOperation:

    def __init__(self, obj_cam, st_device_list, n_connect_num=0, b_open_device=False, b_start_grabbing=False,
                 h_thread_handle=None,
                 b_thread_closed=False, st_frame_info=None, b_exit=False, b_save_bmp=False, b_save_jpg=False,
                 buf_save_image=None,
                 n_save_image_size=0, n_win_gui_id=0, frame_rate=0, exposure_time=0, gain=0):

        self.obj_cam = obj_cam
        self.st_device_list = st_device_list
        self.n_connect_num = n_connect_num
        self.b_open_device = b_open_device
        self.b_start_grabbing = b_start_grabbing
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = MV_FRAME_OUT_INFO_EX()
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.buf_save_image_len = 0
        self.n_save_image_size = n_save_image_size
        self.h_thread_handle = h_thread_handle
        self.b_thread_closed
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain
        self.buf_lock = threading.Lock()  # 取图和存图的buffer锁
        self.dark_frame = None        # 底噪模板（解码后的逐像素 numpy 数组）
        self.apply_dark_sub = False   # 是否启用底噪扣除
        self.apply_hdr = False        # 是否启用实时 HDR 局部对比度增强
        self._last_log_time = 0.0

    # 打开相机
    def Open_device(self):
        if not self.b_open_device:
            if self.n_connect_num < 0:
                return MV_E_CALLORDER

            # ch:选择设备并创建句柄 | en:Select device and create handle
            nConnectionNum = int(self.n_connect_num)
            stDeviceList = cast(self.st_device_list.pDeviceInfo[int(nConnectionNum)],
                                POINTER(MV_CC_DEVICE_INFO)).contents
            self.obj_cam = MvCamera()
            ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()
                return ret

            ret = self.obj_cam.MV_CC_OpenDevice()
            if ret != 0:
                return ret
            print("open device successfully!")
            self.b_open_device = True
            self.b_thread_closed = False

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE or stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        print("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print("warning: set packet size fail! ret[0x%x]" % nPacketSize)

            stBool = c_bool(False)
            ret = self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print("get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                print("set trigger mode fail! ret[0x%x]" % ret)
            return MV_OK

    # 开始取图
    def Start_grabbing(self, winHandle):
        if not self.b_start_grabbing and self.b_open_device:
            self.b_exit = False
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                return ret
            self.b_start_grabbing = True
            print("start grabbing successfully!")
            try:
                self.h_thread_handle = threading.Thread(target=CameraOperation.Work_thread, args=(self, winHandle))
                self.h_thread_handle.daemon = True
                self.h_thread_handle.start()
                self.b_thread_closed = True
            finally:
                pass
            return MV_OK

        return MV_E_CALLORDER

    # 停止取图
    def Stop_grabbing(self):
        if self.b_start_grabbing and self.b_open_device:
            self.b_exit = True
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                return ret
            if self.b_thread_closed and self.h_thread_handle is not None:
                self.h_thread_handle.join(timeout=1.5)
            print("stop grabbing successfully!")
            self.b_start_grabbing = False
            self.b_thread_closed = False
            return MV_OK
        else:
            return MV_E_CALLORDER

    # 关闭相机
    def Close_device(self):
        if self.b_open_device:
            self.b_exit = True
            if self.b_start_grabbing:
                ret = self.obj_cam.MV_CC_StopGrabbing()
                if ret != 0:
                    return ret
                self.b_start_grabbing = False
            if self.b_thread_closed and self.h_thread_handle is not None:
                self.h_thread_handle.join(timeout=1.5)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_CloseDevice()
            if ret != 0:
                return ret

        # ch:销毁句柄 | Destroy handle
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_exit = True
        print("close device successfully!")

        return MV_OK

    # 设置触发模式
    def Set_trigger_mode(self, is_trigger_mode):
        if not self.b_open_device:
            return MV_E_CALLORDER

        if not is_trigger_mode:
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", 0)
            if ret != 0:
                return ret
        else:
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", 1)
            if ret != 0:
                return ret
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerSource", 7)
            if ret != 0:
                return ret

        return MV_OK

    # 软触发一次
    def Trigger_once(self):
        if self.b_open_device:
            return self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")

    # 获取参数
    def Get_parameter(self):
        if self.b_open_device:
            stFloatParam_FrameRate = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_FrameRate), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_exposureTime = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_gain = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.obj_cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_FrameRate)
            if ret != 0:
                return ret
            self.frame_rate = stFloatParam_FrameRate.fCurValue

            ret = self.obj_cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
            if ret != 0:
                return ret
            self.exposure_time = stFloatParam_exposureTime.fCurValue

            ret = self.obj_cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
            if ret != 0:
                return ret
            self.gain = stFloatParam_gain.fCurValue

            return MV_OK

    # 设置参数
    def Set_parameter(self, frameRate, exposureTime, gain):
        if '' == frameRate or '' == exposureTime or '' == gain:
            print('show info', 'please type in the text box !')
            return MV_E_PARAMETER
        if self.b_open_device:
            ret = self.obj_cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            time.sleep(0.2)
            ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(exposureTime))
            if ret != 0:
                print('show error', 'set exposure time fail! ret = ' + To_hex_str(ret))
                return ret

            try:
                self.obj_cam.MV_CC_SetEnumValue("GainAuto", 0)
            except Exception:
                pass
            ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(gain))
            if ret != 0:
                print('show error', 'set gain fail! ret = ' + To_hex_str(ret))
                return ret

            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frameRate))
            if ret != 0:
                print('show error', 'set acquistion frame rate fail! ret = ' + To_hex_str(ret))
                return ret

            print('show info', 'set parameter success!')

            return MV_OK

    # 取图线程函数
    def Work_thread(self, winHandle):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        while not self.b_exit:
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if 0 == ret:

                # 拷贝图像和图像信息
                # 获取缓存锁
                self.buf_lock.acquire()
                try:
                    if self.buf_save_image_len < stOutFrame.stFrameInfo.nFrameLen:
                        if self.buf_save_image is not None:
                            del self.buf_save_image
                            self.buf_save_image = None
                        self.buf_save_image = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                        self.buf_save_image_len = stOutFrame.stFrameInfo.nFrameLen

                    memmove(byref(self.st_frame_info), byref(stOutFrame.stFrameInfo), sizeof(MV_FRAME_OUT_INFO_EX))
                    memmove(byref(self.buf_save_image), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)

                    if self.apply_dark_sub and self.dark_frame is not None:
                        try:
                            self._apply_dark_sub_locked()
                        except Exception as _e:
                            print("[DarkSub] error:", _e)
                    if self.apply_hdr:
                        try:
                            self._apply_hdr_locked()
                        except Exception as _e:
                            print("[HDR] error:", _e)
                finally:
                    self.buf_lock.release()

                now = time.time()
                if now - self._last_log_time >= 1.0:
                    self._last_log_time = now
                    print("preview frame: Width[%d], Height[%d], nFrameNum[%d]"
                          % (self.st_frame_info.nWidth, self.st_frame_info.nHeight, self.st_frame_info.nFrameNum))

                # 释放缓存
                self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                if not self.b_exit:
                    print("no data, ret = " + To_hex_str(ret))
                continue

            # 使用Display接口显示图像
            stDisplayParam = MV_DISPLAY_FRAME_INFO()
            memset(byref(stDisplayParam), 0, sizeof(stDisplayParam))
            self.buf_lock.acquire()
            try:
                stDisplayParam.hWnd = int(winHandle)
                stDisplayParam.nWidth = self.st_frame_info.nWidth
                stDisplayParam.nHeight = self.st_frame_info.nHeight
                stDisplayParam.enPixelType = self.st_frame_info.enPixelType
                stDisplayParam.pData = self.buf_save_image
                stDisplayParam.nDataLen = self.st_frame_info.nFrameLen
                self.obj_cam.MV_CC_DisplayOneFrame(stDisplayParam)
            finally:
                self.buf_lock.release()

    # 存jpg图像
    def Save_jpg(self):

        if self.buf_save_image is None:
            return

        # 获取缓存锁
        self.buf_lock.acquire()

        file_path = str(self.st_frame_info.nFrameNum) + ".jpg"
        c_file_path = file_path.encode('ascii')
        stSaveParam = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
        stSaveParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stSaveParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stSaveParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stSaveParam.nDataLen = self.st_frame_info.nFrameLen
        stSaveParam.pData = cast(self.buf_save_image, POINTER(c_ubyte))
        stSaveParam.enImageType = MV_Image_Jpeg  # ch:需要保存的图像类型 | en:Image format to save
        stSaveParam.nQuality = 80
        stSaveParam.pcImagePath = ctypes.create_string_buffer(c_file_path)
        stSaveParam.iMethodValue = 1
        ret = self.obj_cam.MV_CC_SaveImageToFileEx(stSaveParam)

        self.buf_lock.release()
        return ret

    # 存BMP图像
    def Save_Bmp(self):

        if 0 == self.buf_save_image:
            return

        # 获取缓存锁
        self.buf_lock.acquire()

        file_path = str(self.st_frame_info.nFrameNum) + ".bmp"
        c_file_path = file_path.encode('ascii')

        stSaveParam = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
        stSaveParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stSaveParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stSaveParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stSaveParam.nDataLen = self.st_frame_info.nFrameLen
        stSaveParam.pData = cast(self.buf_save_image, POINTER(c_ubyte))
        stSaveParam.enImageType = MV_Image_Bmp  # ch:需要保存的图像类型 | en:Image format to save
        stSaveParam.pcImagePath = ctypes.create_string_buffer(c_file_path)
        stSaveParam.iMethodValue = 1
        ret = self.obj_cam.MV_CC_SaveImageToFileEx(stSaveParam)

        self.buf_lock.release()

        return ret

    # 获取当前帧的 numpy 数组副本（用于锐度计算）
    @staticmethod
    def _hdr_enhance_u8(image_u8):
        import numpy as np

        src = np.asarray(image_u8, dtype=np.uint8)
        try:
            import cv2

            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            enhanced = clahe.apply(src)
        except Exception:
            p1 = float(np.percentile(src, 1.0))
            p99 = float(np.percentile(src, 99.0))
            if p99 <= p1 + 1.0:
                return src.copy()
            enhanced = np.clip((src.astype(np.float32) - p1) * 255.0 / (p99 - p1), 0, 255).astype(np.uint8)

        base = src.astype(np.float32)
        detail = enhanced.astype(np.float32)
        return np.clip(base * 0.25 + detail * 0.75, 0, 255).astype(np.uint8)

    def _apply_hdr_to_linear_u8(self, raw, width, height):
        import numpy as np

        image = raw[: width * height].reshape(height, width)
        raw[: width * height] = self._hdr_enhance_u8(image).reshape(-1).astype(np.uint8)

    def _apply_hdr_to_linear_u16(self, raw, width, height, max_value):
        import numpy as np

        pixel_count = width * height
        image16 = raw[:pixel_count].reshape(height, width)
        image8 = np.clip(image16.astype(np.float32) / float(max_value) * 255.0, 0, 255).astype(np.uint8)
        enhanced8 = self._hdr_enhance_u8(image8)
        raw[:pixel_count] = np.clip(
            enhanced8.astype(np.float32) / 255.0 * float(max_value), 0, max_value
        ).astype(np.uint16).reshape(-1)

    def _apply_hdr_to_packed12(self, raw, pixel_count, width, height):
        import numpy as np

        packed = raw[: (pixel_count * 3 + 1) // 2]
        groups = packed[: (len(packed) // 3) * 3].reshape(-1, 3).astype(np.uint16)
        out = np.empty(groups.shape[0] * 2, dtype=np.uint16)
        out[0::2] = groups[:, 0] | ((groups[:, 1] & 0x0F) << 8)
        out[1::2] = (groups[:, 1] >> 4) | (groups[:, 2] << 4)
        if len(out) < pixel_count and len(packed) % 3 == 2:
            tail = np.array([packed[-2] | ((packed[-1] & 0x0F) << 8)], dtype=np.uint16)
            out = np.concatenate([out, tail])
        corrected = out[:pixel_count].copy()
        self._apply_hdr_to_linear_u16(corrected, width, height, 4095)

        full_pairs = pixel_count // 2
        if full_pairs:
            even = corrected[: full_pairs * 2:2]
            odd = corrected[1: full_pairs * 2:2]
            packed_groups = packed[: full_pairs * 3].reshape(-1, 3)
            packed_groups[:, 0] = (even & 0xFF).astype(np.uint8)
            packed_groups[:, 1] = (((even >> 8) & 0x0F) | ((odd & 0x0F) << 4)).astype(np.uint8)
            packed_groups[:, 2] = ((odd >> 4) & 0xFF).astype(np.uint8)
        if pixel_count % 2 and len(packed) >= full_pairs * 3 + 2:
            last = corrected[-1]
            packed[full_pairs * 3] = int(last & 0xFF)
            packed[full_pairs * 3 + 1] = int((last >> 8) & 0x0F)

    def _apply_hdr_locked(self):
        import numpy as np

        if self.buf_save_image is None:
            return
        w = int(self.st_frame_info.nWidth)
        h = int(self.st_frame_info.nHeight)
        pixel_count = w * h
        if pixel_count <= 0:
            return

        pixel_type = self.st_frame_info.enPixelType
        frame_len = int(self.st_frame_info.nFrameLen)
        rgb8 = globals().get("PixelType_Gvsp_RGB8_Packed")
        bgr8 = globals().get("PixelType_Gvsp_BGR8_Packed")

        if pixel_type == PixelType_Gvsp_Mono8 or pixel_type in (
            PixelType_Gvsp_BayerGR8,
            PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerGB8,
            PixelType_Gvsp_BayerBG8,
            PixelType_Gvsp_BayerRBGG8,
        ):
            raw = np.frombuffer(self.buf_save_image, dtype=np.uint8, count=min(frame_len, pixel_count))
            if raw.size >= pixel_count:
                self._apply_hdr_to_linear_u8(raw, w, h)
            return

        if (rgb8 is not None and pixel_type == rgb8) or (bgr8 is not None and pixel_type == bgr8):
            if frame_len < pixel_count * 3:
                return
            raw = np.frombuffer(self.buf_save_image, dtype=np.uint8, count=pixel_count * 3)
            image = raw.reshape(h, w, 3)
            try:
                import cv2

                code_to_ycc = cv2.COLOR_RGB2YCrCb if pixel_type == rgb8 else cv2.COLOR_BGR2YCrCb
                code_from_ycc = cv2.COLOR_YCrCb2RGB if pixel_type == rgb8 else cv2.COLOR_YCrCb2BGR
                ycc = cv2.cvtColor(image, code_to_ycc)
                ycc[:, :, 0] = self._hdr_enhance_u8(ycc[:, :, 0])
                raw[:] = cv2.cvtColor(ycc, code_from_ycc).reshape(-1)
            except Exception:
                for channel in range(3):
                    image[:, :, channel] = self._hdr_enhance_u8(image[:, :, channel])
            return

        if pixel_type in (
            PixelType_Gvsp_Mono10,
            PixelType_Gvsp_BayerGR10,
            PixelType_Gvsp_BayerRG10,
            PixelType_Gvsp_BayerGB10,
            PixelType_Gvsp_BayerBG10,
        ):
            raw = np.frombuffer(self.buf_save_image, dtype=np.uint16, count=min(pixel_count, frame_len // 2))
            if raw.size >= pixel_count:
                self._apply_hdr_to_linear_u16(raw, w, h, 1023)
            return

        if pixel_type in (
            PixelType_Gvsp_Mono12,
            PixelType_Gvsp_BayerGR12,
            PixelType_Gvsp_BayerRG12,
            PixelType_Gvsp_BayerGB12,
            PixelType_Gvsp_BayerBG12,
        ):
            raw = np.frombuffer(self.buf_save_image, dtype=np.uint16, count=min(pixel_count, frame_len // 2))
            if raw.size >= pixel_count:
                self._apply_hdr_to_linear_u16(raw, w, h, 4095)
            return

        if pixel_type in (
            PixelType_Gvsp_BayerGR16,
            PixelType_Gvsp_BayerRG16,
            PixelType_Gvsp_BayerGB16,
            PixelType_Gvsp_BayerBG16,
        ):
            raw = np.frombuffer(self.buf_save_image, dtype=np.uint16, count=min(pixel_count, frame_len // 2))
            if raw.size >= pixel_count:
                self._apply_hdr_to_linear_u16(raw, w, h, 65535)
            return

        if pixel_type in (
            PixelType_Gvsp_Mono12_Packed,
            PixelType_Gvsp_BayerGR12_Packed,
            PixelType_Gvsp_BayerRG12_Packed,
            PixelType_Gvsp_BayerGB12_Packed,
            PixelType_Gvsp_BayerBG12_Packed,
        ):
            raw = np.frombuffer(self.buf_save_image, dtype=np.uint8, count=frame_len)
            if raw.size >= (pixel_count * 3 + 1) // 2:
                self._apply_hdr_to_packed12(raw, pixel_count, w, h)

    def _apply_dark_sub_locked(self):
        import numpy as np

        if self.dark_frame is None or self.buf_save_image is None:
            return
        w = int(self.st_frame_info.nWidth)
        h = int(self.st_frame_info.nHeight)
        pixel_count = w * h
        if pixel_count <= 0 or len(self.dark_frame) < pixel_count:
            return

        pixel_type = self.st_frame_info.enPixelType
        dark = np.asarray(self.dark_frame[:pixel_count], dtype=np.int32)

        if pixel_type == PixelType_Gvsp_Mono8:
            raw = np.frombuffer(self.buf_save_image, dtype=np.uint8, count=pixel_count)
            raw[:] = np.clip(raw.astype(np.int32) - dark, 0, 255).astype(np.uint8)
            return

        if pixel_type in (PixelType_Gvsp_Mono10, PixelType_Gvsp_Mono12):
            raw = np.frombuffer(self.buf_save_image, dtype=np.uint16, count=pixel_count)
            max_value = 1023 if pixel_type == PixelType_Gvsp_Mono10 else 4095
            raw[:] = np.clip(raw.astype(np.int32) - dark, 0, max_value).astype(np.uint16)
            return

        if pixel_type == PixelType_Gvsp_Mono12_Packed:
            frame_len = int(self.st_frame_info.nFrameLen)
            raw = np.frombuffer(self.buf_save_image, dtype=np.uint8, count=frame_len)
            packed = raw[: (pixel_count * 3 + 1) // 2]
            groups = packed[: (len(packed) // 3) * 3].reshape(-1, 3).astype(np.uint16)
            out = np.empty(groups.shape[0] * 2, dtype=np.uint16)
            out[0::2] = groups[:, 0] | ((groups[:, 1] & 0x0F) << 8)
            out[1::2] = (groups[:, 1] >> 4) | (groups[:, 2] << 4)
            if len(out) < pixel_count and len(packed) % 3 == 2:
                tail = np.array([packed[-2] | ((packed[-1] & 0x0F) << 8)], dtype=np.uint16)
                out = np.concatenate([out, tail])
            corrected = np.clip(out[:pixel_count].astype(np.int32) - dark, 0, 4095).astype(np.uint16)

            full_pairs = pixel_count // 2
            if full_pairs:
                even = corrected[: full_pairs * 2:2]
                odd = corrected[1: full_pairs * 2:2]
                packed_groups = packed[: full_pairs * 3].reshape(-1, 3)
                packed_groups[:, 0] = (even & 0xFF).astype(np.uint8)
                packed_groups[:, 1] = (((even >> 8) & 0x0F) | ((odd & 0x0F) << 4)).astype(np.uint8)
                packed_groups[:, 2] = ((odd >> 4) & 0xFF).astype(np.uint8)
            if pixel_count % 2 and len(packed) >= full_pairs * 3 + 2:
                last = corrected[-1]
                packed[full_pairs * 3] = int(last & 0xFF)
                packed[full_pairs * 3 + 1] = int((last >> 8) & 0x0F)

    def Get_frame_numpy(self):
        """
        返回当前帧的 numpy 一维数组副本及宽高。
        返回值: (data: ndarray | None, width: int, height: int)
        Mono8 返回 uint8，Mono10/Mono12 返回 uint16，data 长度 = width * height。
        """
        if self.buf_save_image is None or self.buf_save_image_len == 0:
            return None, 0, 0
        self.buf_lock.acquire()
        try:
            import numpy as np
            w = self.st_frame_info.nWidth
            h = self.st_frame_info.nHeight
            frame_len = self.st_frame_info.nFrameLen
            pixel_type = self.st_frame_info.enPixelType
            # 将 ctypes 缓冲复制到 numpy 数组
            raw = (c_ubyte * frame_len).from_buffer_copy(self.buf_save_image)
            byte_data = np.frombuffer(raw, dtype=np.uint8).copy()
            pixel_count = int(w * h)

            unpacked_10_12 = (
                PixelType_Gvsp_Mono10,
                PixelType_Gvsp_Mono12,
                PixelType_Gvsp_BayerGR10,
                PixelType_Gvsp_BayerRG10,
                PixelType_Gvsp_BayerGB10,
                PixelType_Gvsp_BayerBG10,
                PixelType_Gvsp_BayerGR12,
                PixelType_Gvsp_BayerRG12,
                PixelType_Gvsp_BayerGB12,
                PixelType_Gvsp_BayerBG12,
            )
            if pixel_type in unpacked_10_12:
                count = min(pixel_count, len(byte_data) // 2)
                data = np.frombuffer(byte_data.tobytes(), dtype="<u2", count=count).copy()
                if pixel_type in (
                    PixelType_Gvsp_Mono10,
                    PixelType_Gvsp_BayerGR10,
                    PixelType_Gvsp_BayerRG10,
                    PixelType_Gvsp_BayerGB10,
                    PixelType_Gvsp_BayerBG10,
                ):
                    data = np.bitwise_and(data, 0x03FF).astype(np.uint16, copy=False)
                else:
                    data = np.bitwise_and(data, 0x0FFF).astype(np.uint16, copy=False)
                return data, w, h

            if pixel_type == PixelType_Gvsp_BayerGR16 or pixel_type == PixelType_Gvsp_BayerRG16 \
                    or pixel_type == PixelType_Gvsp_BayerGB16 or pixel_type == PixelType_Gvsp_BayerBG16:
                count = min(pixel_count, len(byte_data) // 2)
                data = np.frombuffer(byte_data.tobytes(), dtype="<u2", count=count).copy()
                return data, w, h

            packed_12 = (
                PixelType_Gvsp_Mono12_Packed,
                PixelType_Gvsp_BayerGR12_Packed,
                PixelType_Gvsp_BayerRG12_Packed,
                PixelType_Gvsp_BayerGB12_Packed,
                PixelType_Gvsp_BayerBG12_Packed,
            )
            if pixel_type in packed_12 and len(byte_data) >= (pixel_count * 3 + 1) // 2:
                packed = byte_data[: (pixel_count * 3 + 1) // 2]
                groups = packed[: (len(packed) // 3) * 3].reshape(-1, 3).astype(np.uint16)
                out = np.empty(groups.shape[0] * 2, dtype=np.uint16)
                out[0::2] = groups[:, 0] | ((groups[:, 1] & 0x0F) << 8)
                out[1::2] = (groups[:, 1] >> 4) | (groups[:, 2] << 4)
                if len(out) < pixel_count and len(packed) % 3 == 2:
                    tail = np.array([packed[-2] | ((packed[-1] & 0x0F) << 8)], dtype=np.uint16)
                    out = np.concatenate([out, tail])
                data = out[:pixel_count].copy()
                return data, w, h

            data = byte_data[:pixel_count].copy()
            return data, w, h
        except Exception as e:
            print("[Get_frame_numpy] error:", e)
            return None, 0, 0
        finally:
            self.buf_lock.release()

    def Get_frame_rgb_numpy(self):
        """
        返回当前帧的 RGB 图像副本及宽高。
        支持 RGB/BGR8、Bayer8/10/12/16；Mono 帧返回 None，由上层退回灰度。
        """
        if self.buf_save_image is None or self.buf_save_image_len == 0:
            return None, 0, 0
        self.buf_lock.acquire()
        try:
            import numpy as np
            import cv2

            w = int(self.st_frame_info.nWidth)
            h = int(self.st_frame_info.nHeight)
            frame_len = int(self.st_frame_info.nFrameLen)
            pixel_type = self.st_frame_info.enPixelType
            pixel_count = w * h
            raw = (c_ubyte * frame_len).from_buffer_copy(self.buf_save_image)
            byte_data = np.frombuffer(raw, dtype=np.uint8).copy()

            rgb8 = globals().get("PixelType_Gvsp_RGB8_Packed")
            bgr8 = globals().get("PixelType_Gvsp_BGR8_Packed")
            if rgb8 is not None and pixel_type == rgb8 and len(byte_data) >= pixel_count * 3:
                return byte_data[: pixel_count * 3].reshape(h, w, 3).copy(), w, h
            if bgr8 is not None and pixel_type == bgr8 and len(byte_data) >= pixel_count * 3:
                bgr = byte_data[: pixel_count * 3].reshape(h, w, 3)
                return bgr[:, :, ::-1].copy(), w, h

            bayer8_codes = {
                globals().get("PixelType_Gvsp_BayerGR8"): cv2.COLOR_BayerGR2RGB,
                globals().get("PixelType_Gvsp_BayerRG8"): cv2.COLOR_BayerRG2RGB,
                globals().get("PixelType_Gvsp_BayerGB8"): cv2.COLOR_BayerGB2RGB,
                globals().get("PixelType_Gvsp_BayerBG8"): cv2.COLOR_BayerBG2RGB,
            }
            if pixel_type in bayer8_codes and len(byte_data) >= pixel_count:
                mosaic = byte_data[:pixel_count].reshape(h, w)
                rgb = cv2.cvtColor(mosaic, bayer8_codes[pixel_type])
                return rgb[:, :, ::-1].copy(), w, h

            bayer16_codes = {
                globals().get("PixelType_Gvsp_BayerGR10"): cv2.COLOR_BayerGR2RGB,
                globals().get("PixelType_Gvsp_BayerRG10"): cv2.COLOR_BayerRG2RGB,
                globals().get("PixelType_Gvsp_BayerGB10"): cv2.COLOR_BayerGB2RGB,
                globals().get("PixelType_Gvsp_BayerBG10"): cv2.COLOR_BayerBG2RGB,
                globals().get("PixelType_Gvsp_BayerGR12"): cv2.COLOR_BayerGR2RGB,
                globals().get("PixelType_Gvsp_BayerRG12"): cv2.COLOR_BayerRG2RGB,
                globals().get("PixelType_Gvsp_BayerGB12"): cv2.COLOR_BayerGB2RGB,
                globals().get("PixelType_Gvsp_BayerBG12"): cv2.COLOR_BayerBG2RGB,
                globals().get("PixelType_Gvsp_BayerGR16"): cv2.COLOR_BayerGR2RGB,
                globals().get("PixelType_Gvsp_BayerRG16"): cv2.COLOR_BayerRG2RGB,
                globals().get("PixelType_Gvsp_BayerGB16"): cv2.COLOR_BayerGB2RGB,
                globals().get("PixelType_Gvsp_BayerBG16"): cv2.COLOR_BayerBG2RGB,
            }
            if pixel_type in bayer16_codes and len(byte_data) >= pixel_count * 2:
                mosaic16 = np.frombuffer(byte_data.tobytes(), dtype="<u2", count=pixel_count).reshape(h, w)
                rgb16 = cv2.cvtColor(mosaic16, bayer16_codes[pixel_type])
                max_value = 4095.0 if "12" in str(pixel_type) else 65535.0
                rgb = np.clip(rgb16.astype(np.float32) / max_value * 255.0, 0, 255).astype(np.uint8)
                return rgb[:, :, ::-1].copy(), w, h

            bayer12_packed_codes = {
                globals().get("PixelType_Gvsp_BayerGR12_Packed"): cv2.COLOR_BayerGR2RGB,
                globals().get("PixelType_Gvsp_BayerRG12_Packed"): cv2.COLOR_BayerRG2RGB,
                globals().get("PixelType_Gvsp_BayerGB12_Packed"): cv2.COLOR_BayerGB2RGB,
                globals().get("PixelType_Gvsp_BayerBG12_Packed"): cv2.COLOR_BayerBG2RGB,
            }
            if pixel_type in bayer12_packed_codes and len(byte_data) >= (pixel_count * 3 + 1) // 2:
                packed = byte_data[: (pixel_count * 3 + 1) // 2]
                groups = packed[: (len(packed) // 3) * 3].reshape(-1, 3).astype(np.uint16)
                out = np.empty(groups.shape[0] * 2, dtype=np.uint16)
                out[0::2] = groups[:, 0] | ((groups[:, 1] & 0x0F) << 8)
                out[1::2] = (groups[:, 1] >> 4) | (groups[:, 2] << 4)
                if len(out) < pixel_count and len(packed) % 3 == 2:
                    tail = np.array([packed[-2] | ((packed[-1] & 0x0F) << 8)], dtype=np.uint16)
                    out = np.concatenate([out, tail])
                mosaic16 = out[:pixel_count].reshape(h, w)
                rgb16 = cv2.cvtColor(mosaic16, bayer12_packed_codes[pixel_type])
                rgb = np.clip(rgb16.astype(np.float32) / 4095.0 * 255.0, 0, 255).astype(np.uint8)
                return rgb[:, :, ::-1].copy(), w, h

            return None, 0, 0
        except Exception as e:
            print("[Get_frame_rgb_numpy] error:", e)
            return None, 0, 0
        finally:
            self.buf_lock.release()

    # 存BMP图像到指定完整路径
    def Save_Bmp_with_path(self, full_path):
        if self.buf_save_image is None or self.buf_save_image_len == 0:
            return MV_E_RESOURCE

        # 获取缓存锁
        self.buf_lock.acquire()
        try:
            # 使用 gbk 编码以支持中文路径（Windows 环境）
            c_file_path = full_path.encode('gbk', errors='replace')
            stSaveParam = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
            stSaveParam.enPixelType = self.st_frame_info.enPixelType
            stSaveParam.nWidth = self.st_frame_info.nWidth
            stSaveParam.nHeight = self.st_frame_info.nHeight
            stSaveParam.nDataLen = self.st_frame_info.nFrameLen
            stSaveParam.pData = cast(self.buf_save_image, POINTER(c_ubyte))
            stSaveParam.enImageType = MV_Image_Bmp
            stSaveParam.pcImagePath = ctypes.create_string_buffer(c_file_path)
            stSaveParam.iMethodValue = 1
            ret = self.obj_cam.MV_CC_SaveImageToFileEx(stSaveParam)
        finally:
            self.buf_lock.release()
        return ret
