"""可编程拍摄对话框

基于 CSV 文件内容，对比系统时间触发自动化拍摄流程。
CSV 列定义：
  1. 拍照时间 (YYYYMMDDHHmmss)
  2. 是否自动对焦 (1=自动 / 0=直接拍摄)
  3. 快门时间 10-1000000 µs
  4. 增益 0-10
  5. 灯光亮度 0-255

等待过程中灯光关闭，拍照时开启灯光 20 秒。
拍摄完成后自动将图片合成 24fps MP4 视频。
"""
import csv
import os
import threading
import time
from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
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

from algorithms import ensure_dir
from sdk.MvErrorDefine_const import MV_OK


# ── 常量 ──
SHUTTER_MIN = 10
SHUTTER_MAX = 1000000
GAIN_MIN = 0
GAIN_MAX = 20
LIGHT_MIN = 0
LIGHT_MAX = 255
LIGHT_ON_DURATION = 20       # 秒
LIGHT_STABILIZE_SECS = 3.0   # 开灯后等待灯光和白平衡稳定的最短时间（秒）
FRESH_FRAME_MIN = 3          # 拍照前要求至少到来的新帧数
FRESH_FRAME_TIMEOUT = 12.0   # 等待新帧的超时时间（秒）


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _parse_csv(path):
    """解析 CSV 文件，返回任务列表。"""
    tasks = []
    with open(path, "r", encoding="gbk", errors="replace") as f:
        reader = csv.reader(f)
        for row_num, row in enumerate(reader, start=1):
            if not row or all(c.strip() == "" for c in row):
                continue
            # 跳过表头行
            first = row[0].strip()
            if not first.isdigit():
                continue
            if len(row) < 5:
                raise ValueError("第 {} 行列数不足 5".format(row_num))
            shoot_time = None
            for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M"):
                try:
                    shoot_time = datetime.strptime(first, fmt)
                    break
                except ValueError:
                    pass
            if shoot_time is None:
                raise ValueError("第 {} 行时间格式错误: '{}'，需要 YYYYMMDDHHmmss 或 YYYYMMDDHHmm".format(row_num, first))
            try:
                auto_focus = int(row[1].strip()) == 1
                shutter = _clamp(int(row[2].strip()), SHUTTER_MIN, SHUTTER_MAX)
                gain = _clamp(float(row[3].strip()), GAIN_MIN, GAIN_MAX)
                light = _clamp(int(row[4].strip()), LIGHT_MIN, LIGHT_MAX)
            except ValueError:
                raise ValueError("第 {} 行参数格式错误，自动对焦/快门/增益/灯光必须为数字".format(row_num))
            tasks.append({
                "time": shoot_time,
                "auto_focus": auto_focus,
                "shutter": shutter,
                "gain": gain,
                "light": light,
            })
    if not tasks:
        raise ValueError("CSV 文件中未找到有效拍摄任务")
    tasks.sort(key=lambda t: t["time"])
    return tasks


def _validate_task_times(tasks, now=None):
    """Validate task list is non-empty; past tasks will execute immediately."""
    if not tasks:
        raise ValueError("CSV 文件中未找到有效拍摄任务")


def _images_to_mp4(image_dir, output_path, fps=24, image_names=None):
    """将图片序列合成 MP4 视频。"""
    import cv2

    if image_names is None:
        images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".bmp", ".png", ".jpg"))
        ])
    else:
        images = [
            os.path.basename(f) for f in image_names
            if os.path.basename(f).lower().endswith((".bmp", ".png", ".jpg"))
        ]
    if not images:
        raise RuntimeError("文件夹中无图片可合成视频")

    first = cv2.imread(os.path.join(image_dir, images[0]))
    if first is None:
        raise RuntimeError("无法读取图片: {}".format(images[0]))
    h, w = first.shape[:2]

    # 优先使用 H.264 (avc1)，Windows 媒体播放器原生支持；
    # 若 OpenCV 不支持则回退到 mp4v（改用 .avi 扩展名以确保可播放）。
    h264_fourcc = cv2.VideoWriter_fourcc(*"avc1")
    test_writer = cv2.VideoWriter(output_path, h264_fourcc, fps, (w, h))
    if test_writer.isOpened():
        fourcc = h264_fourcc
        writer = test_writer
    else:
        test_writer.release()
        # H.264 不可用，改用 mp4v 写入 .avi
        base, _ = os.path.splitext(output_path)
        output_path = base + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    try:
        for img_name in images:
            frame = cv2.imread(os.path.join(image_dir, img_name))
            if frame is not None:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
    finally:
        writer.release()
    return output_path


class ProgrammableShootingDialog(QDialog):
    """可编程拍摄对话框"""

    _sig_status = pyqtSignal(str, str)
    _sig_progress = pyqtSignal(int, int)
    _sig_log = pyqtSignal(str)
    _sig_done = pyqtSignal(bool, str)

    def __init__(self, device_controller, config_manager, main_window, parent=None):
        super().__init__(parent)
        self.device_controller = device_controller
        self.config_manager = config_manager
        self.main_window = main_window
        self.setWindowTitle("可编程拍摄")
        self.setMinimumWidth(560)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)

        self._running = False
        self._stop_requested = False
        self._csv_path = ""
        self._save_dir = ""
        self._tasks = []

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── CSV 文件选择 ──
        grp_csv = QGroupBox("CSV 任务文件")
        csv_layout = QHBoxLayout(grp_csv)
        self._edt_csv = QLineEdit()
        self._edt_csv.setReadOnly(True)
        self._edt_csv.setPlaceholderText("选择 CSV 文件...")
        self._btn_csv = QPushButton("浏览...")
        csv_layout.addWidget(self._edt_csv, 1)
        csv_layout.addWidget(self._btn_csv)
        layout.addWidget(grp_csv)

        # ── 保存文件夹 ──
        grp_save = QGroupBox("图片保存文件夹")
        save_layout = QHBoxLayout(grp_save)
        self._edt_save = QLineEdit()
        self._edt_save.setReadOnly(True)
        self._edt_save.setPlaceholderText("选择保存文件夹...")
        self._btn_save = QPushButton("浏览...")
        save_layout.addWidget(self._edt_save, 1)
        save_layout.addWidget(self._btn_save)
        layout.addWidget(grp_save)

        # ── 任务预览 ──
        grp_preview = QGroupBox("任务预览")
        preview_layout = QVBoxLayout(grp_preview)
        self._lbl_task_count = QLabel("尚未加载任务")
        preview_layout.addWidget(self._lbl_task_count)
        self._txt_preview = QPlainTextEdit()
        self._txt_preview.setReadOnly(True)
        self._txt_preview.setMaximumBlockCount(2000)
        self._txt_preview.setFixedHeight(160)
        self._txt_preview.setVisible(False)
        preview_layout.addWidget(self._txt_preview)
        layout.addWidget(grp_preview)

        # ── 控制按钮 ──
        btn_layout = QHBoxLayout()
        self._btn_start = QPushButton("开始执行")
        self._btn_stop = QPushButton("停止")
        self._btn_stop.setEnabled(False)
        btn_layout.addWidget(self._btn_start)
        btn_layout.addWidget(self._btn_stop)
        layout.addLayout(btn_layout)

        # ── 状态 ──
        self._lbl_status = QLabel("就绪")
        self._lbl_status.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._lbl_status)

        # ── 进度条 ──
        self._progress = QProgressBar()
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # ── 日志 ──
        grp_log = QGroupBox("日志")
        log_layout = QVBoxLayout(grp_log)
        self._txt_log = QPlainTextEdit()
        self._txt_log.setReadOnly(True)
        self._txt_log.setMaximumBlockCount(500)
        self._txt_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self._txt_log)
        layout.addWidget(grp_log, 1)

    def _connect_signals(self):
        self._btn_csv.clicked.connect(self._select_csv)
        self._btn_save.clicked.connect(self._select_save_dir)
        self._btn_start.clicked.connect(self._start)
        self._btn_stop.clicked.connect(self._stop)

        self._sig_status.connect(self._on_status)
        self._sig_progress.connect(self._on_progress)
        self._sig_log.connect(self._on_log)
        self._sig_done.connect(self._on_done)

    # ── UI 回调 ──

    def _select_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 CSV 任务文件", "",
            "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if path:
            self._csv_path = path
            self._edt_csv.setText(path)
            self._load_csv_preview()

    def _select_save_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "选择图片保存文件夹",
            self.config_manager.effective_save_path()
        )
        if d:
            self._save_dir = d
            self._edt_save.setText(d)

    def _load_csv_preview(self):
        try:
            self._tasks = _parse_csv(self._csv_path)
            _validate_task_times(self._tasks)
            lines = []
            for i, t in enumerate(self._tasks):
                lines.append("#{}: {} | AF={} | 快门={}µs | 增益={} | 灯光={}".format(
                    i + 1,
                    t["time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "是" if t["auto_focus"] else "否",
                    t["shutter"], t["gain"], t["light"],
                ))
            self._lbl_task_count.setText("共 {} 条任务".format(len(self._tasks)))
            self._txt_preview.setPlainText("\n".join(lines))
            self._txt_preview.setVisible(True)
        except Exception as exc:
            self._tasks = []
            self._lbl_task_count.setText("加载失败: {}".format(exc))
            QMessageBox.warning(self, "CSV 解析错误", str(exc))

    def _start(self):
        if self._running:
            return
        if not self._tasks:
            QMessageBox.warning(self, "提示", "请先加载有效的 CSV 任务文件！")
            return
        try:
            _validate_task_times(self._tasks)
        except ValueError as exc:
            QMessageBox.warning(self, "时间设置错误", str(exc))
            return
        if not self._save_dir:
            QMessageBox.warning(self, "提示", "请先选择图片保存文件夹！")
            return
        if not self.device_controller.opened or not self.device_controller.grabbing:
            QMessageBox.warning(self, "提示", "请先打开相机并开始取流！")
            return
        if not self.device_controller.serial_connected:
            QMessageBox.warning(self, "提示", "请先连接串口；可编程拍摄需要串口控制灯光和自动对焦。")
            return

        self._running = True
        self._stop_requested = False
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_csv.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._txt_log.clear()
        self._progress.setValue(0)
        self._progress.setMaximum(len(self._tasks))

        threading.Thread(target=self._worker, daemon=True).start()

    def _stop(self):
        self._stop_requested = True
        self._sig_log.emit("用户请求停止，将在当前等待结束后停止...")

    def _on_status(self, text, color):
        self._lbl_status.setText(text)
        if color:
            self._lbl_status.setStyleSheet("font-weight: bold; color: {};".format(color))
        else:
            self._lbl_status.setStyleSheet("font-weight: bold;")

    def _on_progress(self, current, total):
        self._progress.setMaximum(total)
        self._progress.setValue(current)

    def _on_log(self, text):
        self._txt_log.appendPlainText(text)

    def _on_done(self, success, message):
        self._running = False
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_csv.setEnabled(True)
        self._btn_save.setEnabled(True)
        if success:
            self._sig_status.emit("完成", "green")
            QMessageBox.information(self, "完成", message)
        else:
            self._sig_status.emit("已停止" if self._stop_requested else "失败", "red")
            if message:
                QMessageBox.warning(self, "提示", message)

    # ── 工作线程 ──

    def _worker(self):
        try:
            self._run_tasks()
        except Exception as exc:
            self._set_light(0)
            self._sig_log.emit("异常: {}".format(exc))
            self._sig_done.emit(False, "执行异常: {}".format(exc))

    def _set_light(self, value):
        """通过主窗口的 send_gcode 设置灯光亮度。"""
        try:
            self.device_controller.send_gcode("M106 S{}\n".format(value))
        except Exception:
            self._sig_log.emit("灯光控制失败")

    def _sleep_with_stop(self, seconds):
        """Sleep in short chunks so the stop button still responds promptly."""
        deadline = time.time() + max(0.0, seconds)
        while time.time() < deadline:
            if self._stop_requested:
                return False
            time.sleep(min(0.1, deadline - time.time()))
        return True

    def _wait_for_new_frame(self, min_frames=FRESH_FRAME_MIN, timeout=FRESH_FRAME_TIMEOUT):
        """阻塞直到相机采集了至少 min_frames 帧新图像，或超时。
        通过对比 nFrameNum 的变化来判断是否有新帧到来，确保保存的是
        当前曝光参数和灯光条件下真正采集的帧，而非缓存的旧帧。
        """
        try:
            base = self.device_controller.get_frame_num()
        except Exception:
            time.sleep(1.0)
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop_requested:
                return
            try:
                current = self.device_controller.get_frame_num()
            except Exception:
                time.sleep(0.1)
                continue
            if current - base >= min_frames:
                return
            time.sleep(0.05)
        self._sig_log.emit("警告: 等待新帧超时（{}s），继续拍摄".format(timeout))

    def _wait_for_light_ready(self):
        """Wait until the light is stable, then wait for fresh camera frames.

        The camera save path writes the latest preview buffer.  If a frame was
        exposed while the LED was still ramping, it can look yellow even though
        later preview frames are normal, so the frame counter baseline is taken
        only after the stabilization delay has passed.
        """
        self._sig_log.emit(
            "等待灯光稳定 {:.1f}s，并丢弃前 {} 帧过渡画面...".format(
                LIGHT_STABILIZE_SECS, FRESH_FRAME_MIN))
        if not self._sleep_with_stop(LIGHT_STABILIZE_SECS):
            return
        self._wait_for_new_frame(min_frames=FRESH_FRAME_MIN, timeout=FRESH_FRAME_TIMEOUT)
        self._sig_log.emit("灯光稳定，已获取稳定后的新帧")

    def _do_autofocus(self):
        """调用主窗口的自动对焦功能并等待完成。"""
        self._sig_log.emit("开始自动对焦...")
        mw = self.main_window
        if mw.autofocus_running:
            self._sig_log.emit("自动对焦正在进行中，跳过")
            return

        mw.autofocus_running = True
        mw.start_autofocus_worker_only()

        # 等待自动对焦完成
        while mw.autofocus_running:
            if self._stop_requested:
                mw.autofocus_running = False
                return
            time.sleep(0.5)
        self._sig_log.emit("自动对焦完成")

    def _run_tasks(self):
        total = len(self._tasks)
        ensure_dir(self._save_dir)

        # 关闭灯光
        self._set_light(0)
        self._sig_log.emit("灯光已关闭，等待任务开始...")

        completed = 0
        captured_images = []
        for idx, task in enumerate(self._tasks):
            if self._stop_requested:
                self._sig_log.emit("用户停止，已完成 {}/{} 个任务".format(completed, total))
                self._sig_done.emit(False, "已停止，完成 {}/{} 个任务".format(completed, total))
                # 确保灯光关闭
                self._set_light(0)
                return

            target_time = task["time"]
            self._sig_status.emit(
                "等待任务 #{} @ {}".format(idx + 1, target_time.strftime("%H:%M:%S")),
                "blue"
            )
            self._sig_log.emit("任务 #{}: 等待到 {}".format(
                idx + 1, target_time.strftime("%Y-%m-%d %H:%M:%S")))

            # ── 等待到指定时间 ──
            while True:
                if self._stop_requested:
                    break
                now = datetime.now()
                diff = (target_time - now).total_seconds()
                if diff <= 0:
                    break
                self._sig_status.emit(
                    "等待任务 #{} | 剩余 {:.0f}s".format(idx + 1, diff),
                    "blue"
                )
                time.sleep(min(diff, 1.0))

            if self._stop_requested:
                continue

            # ── 执行拍摄 ──
            self._sig_status.emit("执行任务 #{}".format(idx + 1), "orange")
            self._sig_log.emit("任务 #{} 开始执行: 快门={}µs 增益={} 灯光={}".format(
                idx + 1, task["shutter"], task["gain"], task["light"]))

            # 1. 设置相机参数
            try:
                self.device_controller.set_exposure(task["shutter"])
                self.device_controller.set_gain(task["gain"])
                self._sig_log.emit("相机参数已设置")
            except Exception as exc:
                self._sig_log.emit("设置相机参数失败: {}".format(exc))
                continue

            # 2. 到达拍摄时间后开灯。自动对焦也包含在这次 20s 亮灯窗口内。
            self._set_light(task["light"])
            self._sig_log.emit("灯光已开启: 亮度={}".format(task["light"]))
            light_on_started = time.time()

            # 3. 等待灯光稳定后再取新帧，避免保存 LED 刚亮时的过渡色帧。
            self._wait_for_light_ready()
            if self._stop_requested:
                self._set_light(0)
                self._sig_log.emit("灯光已关闭")
                continue

            # 4. 自动对焦（如果需要）
            if task["auto_focus"]:
                self._do_autofocus()
                if self._stop_requested:
                    self._set_light(0)
                    self._sig_log.emit("灯光已关闭")
                    continue
                # 对焦完成后再等一帧，确保对焦后的帧已进入缓冲区
                self._wait_for_new_frame(min_frames=1, timeout=FRESH_FRAME_TIMEOUT)

            # 5. 拍照保存
            file_name = "prog_{:04d}_{}.bmp".format(
                idx + 1, target_time.strftime("%Y%m%d_%H%M%S"))
            file_path = os.path.join(self._save_dir, file_name)
            try:
                ret = self.device_controller.save_bmp_with_path(file_path)
                if ret == MV_OK:
                    self._sig_log.emit("图片已保存: {}".format(file_name))
                    completed += 1
                    captured_images.append(file_name)
                else:
                    self._sig_log.emit("保存失败，错误码: {}".format(ret))
            except Exception as exc:
                self._sig_log.emit("拍照异常: {}".format(exc))

            self._sig_progress.emit(idx + 1, total)

            # 6. 灯光从开启时刻起累计保持 20 秒后关闭
            remaining_light = max(0.0, LIGHT_ON_DURATION - (time.time() - light_on_started))
            self._sig_log.emit("灯光从拍摄时刻起保持 {}s，剩余 {:.1f}s...".format(
                LIGHT_ON_DURATION, remaining_light))
            light_off_time = light_on_started + LIGHT_ON_DURATION

            while time.time() < light_off_time:
                if self._stop_requested:
                    break
                time.sleep(0.5)

            self._set_light(0)
            self._sig_log.emit("灯光已关闭")

        # ── 所有任务完成，合成视频 ──
        self._sig_status.emit("正在合成视频...", "purple")
        self._sig_log.emit("所有拍摄任务完成 ({}/{}), 开始合成视频...".format(completed, total))

        video_path = os.path.join(self._save_dir, "output_24fps.mp4")
        try:
            actual_path = _images_to_mp4(self._save_dir, video_path, fps=24, image_names=captured_images)
            self._sig_log.emit("视频已保存: {}".format(actual_path))
            self._sig_done.emit(True,
                "全部完成！\n成功拍摄 {}/{} 张\n视频: {}".format(completed, total, actual_path))
        except ImportError:
            self._sig_log.emit("未安装 opencv-python，跳过视频合成。请执行: pip install opencv-python")
            self._sig_done.emit(True,
                "拍摄完成 ({}/{})，但视频合成失败（未安装 opencv-python）".format(completed, total))
        except Exception as exc:
            self._sig_log.emit("视频合成失败: {}".format(exc))
            self._sig_done.emit(True,
                "拍摄完成 ({}/{})，但视频合成失败: {}".format(completed, total, exc))
