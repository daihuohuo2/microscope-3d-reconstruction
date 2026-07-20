# 代码索引文档

> 本文档记录项目所有功能、算法、UI 元素与对应代码的映射关系。
> 格式：**功能 / 按钮** — `文件名:行号` `函数名`

---

## 目录

1. [程序入口](#1-程序入口)
2. [主窗口逻辑](#2-主窗口逻辑-srcmicroscope_appuimain_windowpy)
3. [主界面 UI 布局](#3-主界面-ui-布局-srcmicroscope_appuigeneratedpy)
4. [一键出图](#4-一键出图-srcmicroscope_appuidialogsone_click_dialogpy)
5. [点云重建](#5-点云重建-srcmicroscope_appuidialogsrecon_dialogpy)
6. [连续扫描重建](#6-连续扫描重建-srcmicroscope_appuidialogstemporal_depth_dialogpy)
7. [可编程拍摄](#7-可编程拍摄-srcmicroscope_appuidialogsprogrammable_shooting_dialogpy)
8. [核心算法](#8-核心算法-srcmicroscope_appcorealgorithmspy)
9. [设备控制](#9-设备控制-srcmicroscope_apphardwarecontrollerpy)
10. [配置管理](#10-配置管理-srcmicroscope_appcoreconfigpy)
11. [比例尺叠加层](#11-比例尺叠加层-srcmicroscope_appcoreoverlayspy)
12. [SDK 封装](#12-sdk-封装-srcmicroscope_apphardwaresdk)

---

## 1. 程序入口

| 功能 | 文件 | 说明 |
|------|------|------|
| 程序启动入口 | `main.py:1` | `QApplication` 创建、`MainWindow` 实例化与 `show()` |

---

## 2. 主窗口逻辑 `src/microscope_app/ui/main_window.py`

### 2.1 类定义与初始化

| 功能 | 行号 | 函数/代码 |
|------|------|-----------|
| 类定义 | `L22` | `class MainWindow(QMainWindow)` |
| 构造函数 | `L29` | `__init__` — 创建 UI、ConfigManager、DeviceController，绑定信号，创建菜单 |
| 默认曝光 | `L27` | `DEFAULT_EXPOSURE_US = 80000.0` |
| 默认增益 | `L28` | `DEFAULT_GAIN_DB = 5.0` |
| 预览 widget GDI 配置 | `L53~55` | `WA_NativeWindow` + `WA_PaintOnScreen`，确保 SDK GDI 渲染正常 |
| 比例尺叠加层初始化 | `L57~60` | `ScaleBarOverlay` + `ResizeFilter` |
| Z 轴轮询定时器 | `L67~69` | `QTimer` 每秒调用 `refresh_z_position` |

### 2.2 信号绑定 `_bind_signals`（行 `L73`）

| UI 元素（按钮/控件） | 绑定信号 | 处理函数 | 行号 |
|---------------------|---------|---------|------|
| **查找设备** `bnEnum` | `clicked` | `enum_devices` | `L75` |
| **打开设备** `bnOpen` | `clicked` | `open_device` | `L76` |
| **关闭设备** `bnClose` | `clicked` | `close_device` | `L77` |
| **开始采集** `bnStart` | `clicked` | `start_grabbing` | `L78` |
| **停止采集** `bnStop` | `clicked` | `stop_grabbing` | `L79` |
| **获取参数** `bnGetParam` | `clicked` | `get_param` | `L81` |
| **设置参数** `bnSetParam` | `clicked` | `set_param` | `L82` |
| **开始自动对焦** `bnAutoFocus` | `clicked` | `start_autofocus` | `L84` |
| **停止对焦** `bnStopAutoFocus` | `clicked` | `stop_autofocus` | `L85` |
| **刷新串口** `bnRefreshPort` | `clicked` | `refresh_serial_ports` | `L86` |
| **连接串口/断开串口** `bnConnectSerial` | `clicked` | `connect_serial` | `L87` |
| **Z 轴归零** `bnHomeZ` | `clicked` | `action_home_z` | `L88` |
| **Z 粗调上 +1.00mm** `bnCoarseUp` | `clicked` | `action_coarse_up` | `L89` |
| **Z 粗调下 -1.00mm** `bnCoarseDown` | `clicked` | `action_coarse_down` | `L90` |
| **Z 中调上 +0.10mm** `bnMediumUp` | `clicked` | `action_medium_up` | `L91` |
| **Z 中调下 -0.10mm** `bnMediumDown` | `clicked` | `action_medium_down` | `L92` |
| **Z 细调上 +0.05mm** `bnFineUp` | `clicked` | `action_fine_up` | `L93` |
| **Z 细调下 -0.05mm** `bnFineDown` | `clicked` | `action_fine_down` | `L94` |
| **Z 极细调上 +0.005mm** `bnMoveStep` | `clicked` | `action_move_z_step` | `L95` |
| **Z 极细调下 -0.005mm** `bnMoveStepDown` | `clicked` | `action_move_z_step_down` | `L96` |
| **亮度滑条** `sliderLight` | `valueChanged` | `action_slider_light` | `L97` |
| **快速比例尺** `bnQuickScale` | `clicked` | `start_quick_scale` | `L98` |
| **采集底噪帧** `bnCaptureDark` | `clicked` | `capture_dark_frame` | `L101` |
| **启用底噪扣除** `chkDarkSub` | `stateChanged` | `toggle_dark_sub` | `L102` |
| **清除底噪帧** `bnClearDark` | `clicked` | `clear_dark_frame` | `L103` |
| **显示比例尺** `chkShowScaleBar` | `stateChanged` | `toggle_scale_bar` | `L104` |
| **启用 HDR** `chkHdr` | `stateChanged` | `toggle_hdr` | 主窗口绑定区 |

### 2.3 菜单栏 `_create_menu`（行 `L106`）

| 菜单项 | 快捷键 | 触发函数 | 行号 |
|--------|--------|---------|------|
| **点云重建** | `&3` | `open_recon3d_dialog` | `L108` |
| **连续扫描重建** | `&S` | `open_temporal_depth_dialog` | `L110` |
| **一键出图** | `&I` | `open_one_click_dialog` | `L112` |
| **可编程拍摄** | `&P` | `open_programmable_shooting_dialog` | `L114` |

### 2.4 设置管理

| 功能 | 行号 | 函数 |
|------|------|------|
| 加载配置文件 | `L119` | `load_settings` — 读取串口、波特率、像素/mm |
| 保存配置文件 | `L141` | `save_settings` — 写入 `runtime/settings.ini` |

### 2.5 相机控制

| 功能 | 行号 | 函数 |
|------|------|------|
| 枚举设备 | `L151` | `enum_devices` — 调用 `device_controller.enum_devices()` |
| 打开设备 | `L163` | `open_device` — 打开相机，设置默认曝光/增益，更新参数 UI |
| 关闭设备 | `L178` | `close_device` |
| 开始取流 | `L181` | `start_grabbing` — 设置连续模式，获取 HWND，启动 GDI 渲染 |
| 停止取流 | `L191` | `stop_grabbing` |
| 获取相机参数 | `L197` | `get_param` — 读取曝光/增益/帧率并填入输入框 |
| 设置相机参数 | `L207` | `set_param` — 从 UI 读取帧率/曝光/增益写入相机 |
| 轮询图像宽度 | `L413` | `poll_cam_img_width` — 取帧获取宽度，更新比例尺叠加层 |

### 2.6 串口控制

| 功能 | 行号 | 函数 |
|------|------|------|
| 刷新串口列表 | `L220` | `refresh_serial_ports` |
| 更新串口状态标签 | `L234` | `_update_serial_status` — 更新 `lblSerialStatus` 颜色和文字 |
| 连接/断开串口 | `L247` | `connect_serial` — 连接后发送 `G90` 并同步 Z 位置 |
| 发送 G-code | `L276` | `send_gcode` |

### 2.7 Z 轴运动控制

| 功能 | 行号 | 函数 |
|------|------|------|
| Z 轴归零 | `L284` | `action_home_z` |
| 同步 Z 坐标到 UI | `L292` | `_sync_z_from_device` |
| 定时轮询 Z 位置 | `L305` | `refresh_z_position` — 后台线程 + QTimer.singleShot 更新 UI |
| 更新 Z 位置显示 | `L321` | `_update_z_display` — 根据位置变色（正常蓝 / 接近端点橙 / 超限红） |
| 更新运动按钮使能 | `L360` | `_update_z_motion_buttons` — 串口连接后才使能 |
| 粗调上 +1.00mm | `L375` | `action_coarse_up` → `_move_z_relative_from_ui(1.00)` |
| 粗调下 -1.00mm | `L378` | `action_coarse_down` → `_move_z_relative_from_ui(-1.00)` |
| 中调上 +0.10mm | `L381` | `action_medium_up` → `_move_z_relative_from_ui(0.10)` |
| 中调下 -0.10mm | `L384` | `action_medium_down` → `_move_z_relative_from_ui(-0.10)` |
| 细调上 +0.05mm | `L387` | `action_fine_up` → `_move_z_relative_from_ui(0.05)` |
| 细调下 -0.05mm | `L390` | `action_fine_down` → `_move_z_relative_from_ui(-0.05)` |
| 极细调上 +0.005mm | `L393` | `action_move_z_step` → `_move_z_relative_from_ui(0.005)` |
| 极细调下 -0.005mm | `L396` | `action_move_z_step_down` → `_move_z_relative_from_ui(-0.005)` |
| 相对移动公共实现 | `L399` | `_move_z_relative_from_ui(step_mm)` — 调用 `device_controller.move_z_relative_wait` |
| 亮度滑条改变 | `L409` | `action_slider_light` — 发送 `M106 S{value}` G-code |

### 2.8 比例尺功能

| 功能 | 行号 | 函数 |
|------|------|------|
| 切换比例尺显示 | `L421` | `toggle_scale_bar` — 控制 `ScaleBarOverlay.set_visible()` |
| 快速比例尺入口 | `L493` | `start_quick_scale` — 检查相机，启动线程 |
| 快速比例尺后台线程 | `L427` | `_quick_scale_worker` — 取灰度帧 → 调用 `compute_blob_scale_calibration` |
| 快速比例尺成功回调 | `L461` | `_on_quick_scale_done` — 写入 pixels_per_mm，保存配置，勾选显示比例尺 |
| 快速比例尺失败回调 | `L487` | `_on_quick_scale_fail` — 弹出错误弹窗 |
| 设置点距输入框 | `UI` | `edtDotSpacing` — 默认 200 µm，在 `_quick_scale_worker` 中读取 |

### 2.9 底噪扣除

| 功能 | 行号 | 函数 |
|------|------|------|
| 采集底噪帧 | `L101`（绑定） | `capture_dark_frame` — 采集 50 帧计算均值模板 |
| 启用/禁用底噪扣除 | `L102`（绑定） | `toggle_dark_sub` — 控制 `DeviceController` 实时扣除 |
| 清除底噪帧 | `L103`（绑定） | `clear_dark_frame` |

### 2.10 HDR 增强

| 功能 | 行号 | 函数 |
|------|------|------|
| 启用/禁用 HDR | 绑定区 | `toggle_hdr` — 控制 `DeviceController.set_hdr_enabled()` |
| 实时增强帧缓冲 | `src/microscope_app/hardware/sdk/CamOperation_class.py` | `_apply_hdr_locked` — 在显示前增强 Mono/Bayer/RGB 常见像素格式 |
| 局部对比度算法 | `src/microscope_app/hardware/sdk/CamOperation_class.py` | `_hdr_enhance_u8` — 优先 CLAHE，缺少 OpenCV 时退回百分位拉伸 |

### 2.11 自动对焦

| 功能 | 行号 | 函数 |
|------|------|------|
| 启动自动对焦 | （绑定 `L84`） | `start_autofocus` — 启动后台线程 |
| 停止自动对焦 | （绑定 `L85`） | `stop_autofocus` — 设置 `autofocus_running = False` |
| 自动调整曝光/增益 | `L555` | `_af_auto_expose` — Otsu 主体测光 + 迭代调整，最多 8 次 |
| 精扫阶段曝光微调 | `L635` | `_af_quick_expose` — 单步调整，跳过已合理值 |
| 更新参数 UI | `L668` | `_af_update_param_ui` — 读取相机参数更新 3 个输入框 |
| Z 轴移动（对焦内） | `L680` | `_af_move_z(step_mm)` — 相对移动并等待稳定 |
| 三点二次拟合焦点 | `L686` | `_af_quadratic_peak` — 计算亚步长精确焦面位置 |
| 中央/点击 ROI 锐度评分 | `src/microscope_app/ui/main_window.py` | `_compute_sharpness` / `_af_atlas_score` — ATLAS Focus：有效纹理块筛选 + 多尺度 Tenengrad/Laplacian/Brenner 融合，屏蔽过曝区域 |
| 曝光统计（Otsu 分割） | `L577` | `_get_exposure_stats` — 区分主体/背景后测光 |
| 图像亮度读取 | `L623` | `_get_image_brightness` — 中央 ROI p50 亮度 |
| 中央 ROI 截取 | `L520` | `_get_center_roi(roi_fraction=0.55)` |
| 灰度归一化为分析用 8-bit | `L505` | `_normalize_gray_for_analysis` — 自适应 8/12/16bit |

### 2.11 对话框开启

| 菜单动作 | 行号 | 函数 | 说明 |
|---------|------|------|------|
| 打开点云重建 | （菜单绑定） | `open_recon3d_dialog` | 懒加载 `PointCloudReconDialog` |
| 打开连续扫描重建 | （菜单绑定） | `open_temporal_depth_dialog` | 懒加载 `TemporalDepthDialog` |
| 打开一键出图 | （菜单绑定） | `open_one_click_dialog` | 懒加载 `OneClickDialog` |
| 打开可编程拍摄 | （菜单绑定） | `open_programmable_shooting_dialog` | 懒加载 `ProgrammableShootingDialog` |

---

## 3. 主界面 UI 布局 `src/microscope_app/ui/generated.py`

### 3.1 总体布局

| 内容 | 行号 | 说明 |
|------|------|------|
| 类定义 | `L18` | `class Ui_MainWindow` |
| `setupUi` | `L24` | 主布局：左侧（相机列表+预览），右侧双列（col1+col2） |
| 主窗口尺寸 | `L26` | 默认 `1400×760`，最小 `1060×620` |
| 相机下拉框 | `L38` | `ComboDevices` |
| 相机预览区域 | `L42` | `widgetDisplay` — `WA_NativeWindow` + 黑色背景，SDK GDI 渲染 |
| 右侧双列容器 | `L64~86` | col1W（宽 260-320px）+ col2W（宽 260-320px） |

### 3.2 各 Group 构建方法

| Group 名称 | 行号 | 方法 | 包含控件 |
|-----------|------|------|---------|
| **初始化** | `L96` | `_make_init_group` | `bnEnum`、`bnOpen`、`bnClose` |
| **采集** | `L120` | `_make_grab_group` | `bnStart`、`bnStop`、`bnAutoFocus`、`bnStopAutoFocus`、`lblAutoFocusStatus` |
| **参数** | `L145` | `_make_param_group` | 曝光`edtExposureTime`、增益`edtGain`、帧率`edtFrameRate`、`bnGetParam`、`bnSetParam` |
| **串口设置** | `L163` | `_make_serial_group` | `cmbSerialPort`、`bnRefreshPort`、`cmbBaudRate`、`edtSerialTimeout`、`bnConnectSerial`、`lblSerialStatus` |
| **运动控制** | `L196` | `_make_motion_group` | `bnHomeZ`、粗/中/细/极细调按钮、`lblZPos`、`lblZMinLimit`、`lblZMaxLimit`、`sliderLight`、`lblLightValue` |
| **比例尺** | `L259` | `_make_scale_group` | `chkShowScaleBar`、`edtPixelsPerMm`、`label_dot_spacing`、`edtDotSpacing`(默认200)、`bnQuickScale`、`lblQuickScaleStatus` |
| **底噪扣除** | `L297` | `_make_dark_group` | `bnCaptureDark`、`chkDarkSub`、`bnClearDark`、`lblDarkSubStatus` |

### 3.3 文字设置 `retranslateUi`（行 `L327`）

所有按钮/标签中文文本均在此处设置，逐行对应 `groupInit.setTitle`、`bnEnum.setText`... 等约 40 个控件。

---

## 4. 一键出图 `src/microscope_app/ui/dialogs/one_click_dialog.py`

### 4.1 类与初始化

| 内容 | 行号 | 说明 |
|------|------|------|
| 类定义 | `L57` | `class OneClickDialog(QDialog)` |
| 构造函数 | `L70` | 初始化信号、Z 定时器（200ms）、调用 `_setup_ui` |
| 信号定义 | `L62~65` | `_sig_status`、`_sig_progress`、`_sig_log`、`_sig_done` |

### 4.2 UI 控件（`_setup_ui` 约行 `L120`）

| 控件 | 变量名 | 说明 |
|------|--------|------|
| Z 位置实时显示 | `lblZPos` | 200ms 定时更新，超限变红 |
| Z 高位输入 | `edtZHigh` | 默认 `2.0` mm |
| Z 低位输入 | `edtZLow` | 默认 `-2.0` mm |
| Z 步长输入 | `edtZStep` | 默认 `0.2` mm |
| 每步延时输入 | `edtDelay` | 默认 `0.20` s |
| **自动输出三维点云** 复选框 | `chkPointCloud` | 默认勾选，控制点云区域显隐 |
| Z 轴缩放系数 | `edtZScale` | 默认 `1.0` |
| 最小锐度阈值 | `edtMinSharpness` | 默认 `5.0`（0-100，相对峰值%) |
| **启用粗扫+细扫** 复选框 | `chkCoarseFine` | 默认不勾选 |
| 粗扫步长倍数 | `edtCoarseFactor` | 默认 `3` |
| 精扫区间比例 | `edtFinePct` | 默认 `30` % |
| 进度条 | `progressBar` | 0-100 |
| 状态标签 | `lblStatus` | 显示当前阶段文字 |
| 日志框 | `txtLog` | 只读，最大高度 110px |
| 合成图预览 | `lblPreview` | 扫描完成后显示全焦图 |
| **保存路径** 显示框 | `_edt_save_path` | 只读，显示当前保存目录 |
| **浏览...** | `_btn_browse_save` | 选择保存目录 |
| **一键出图** | `bnStart` | 绿色粗体按钮，触发扫描 |
| **停止** | `bnStop` | 初始禁用 |
| **另存合成图…** | `bnExport` | 扫描完成后启用 |
| **可视化点云** | `bnVisualize` | 勾选点云且完成后启用 |
| **导出点云…** | `bnExportPly` | 勾选点云且完成后启用 |

### 4.3 信号槽

| 按钮 | 行号 | 处理函数 | 说明 |
|------|------|---------|------|
| 一键出图 `bnStart` | `L285` | `_start` | 参数校验→后台线程扫描 |
| 停止 `bnStop` | `L286` | `_stop` | 设置 `_running=False` |
| 另存合成图 `bnExport` | `L287` | `_export` | QFileDialog 另存 PNG/TIFF |
| 可视化点云 `bnVisualize` | `L288` | `_visualize_point_cloud` | matplotlib 3D 散点图 |
| 导出点云 `bnExportPly` | `L289` | `_export_point_cloud` | PLY/OBJ/CSV |
| 浏览保存路径 `_btn_browse_save` | `L270` | `_browse_save_path` | 更新 config_manager.save_path |

### 4.4 扫描流程（后台线程）

| 阶段 | 函数 | 算法调用 |
|------|------|---------|
| 参数解析与检查 | `_start` | 验证 z_high/z_low/step/delay |
| 扫描工作线程 | `_worker` | Z 轴步进 + 每步取帧 |
| 普通 DFF 融合 | `_worker` | `build_best_focus_maps` |
| 粗扫+细扫融合 | `_worker` | `select_focus_window` + 二次 `build_best_focus_maps` + `merge_focus_maps` |
| 点云生成 | `_worker` | `point_cloud_from_depth` |
| 结果保存 | `_worker` | `save_output_bundle` (PNG/TIFF/16bit深度TIFF/JSON清单) |
| 深度图可视化 | `_on_done` | matplotlib 带等深线 + 中文颜色注释 |

---

## 5. 点云重建 `src/microscope_app/ui/dialogs/recon_dialog.py`

### 5.1 类与初始化

| 内容 | 行号 | 说明 |
|------|------|------|
| 类定义 | `L28` | `class PointCloudReconDialog(QDialog)` |
| 构造函数 | `L41` | 初始化信号，Z 定时器 200ms |

### 5.2 UI 控件（`_setup_ui` 约行 `L80`）

| 控件 | 变量名 | 说明 |
|------|--------|------|
| Z 位置实时显示 | `lblZPos` | 超限变红 |
| Z 起始位置 | `edtZStart` | 默认 `-2.0` mm |
| Z 结束位置 | `edtZEnd` | 默认 `2.0` mm |
| Z 步长 | `edtZStep` | 默认 `0.1` mm |
| 每步延时 | `edtDelay` | 默认 `0.15` s |
| Z 轴缩放系数 | `edtZScale` | 默认 `1.0` |
| 最小锐度 | `edtMinSharpness` | 默认 `5.0` % |
| 进度条 | `progressBar` | |
| 状态标签 | `lblStatus` | |
| **保存路径** | `_edt_save_path` | 只读 |
| **浏览...** | `_btn_browse_save` | |
| **开始重建** | `bnStart` | |
| **停止** | `bnStop` | |
| **可视化点云** | `bnVisualize` | 重建完成后启用 |
| **导出点云** | `bnExport` | 重建完成后启用 |

### 5.3 信号槽

| 按钮 | 处理函数 | 说明 |
|------|---------|------|
| 开始重建 `bnStart` | `_start_reconstruction` | 参数校验→后台线程 |
| 停止 `bnStop` | `_stop_reconstruction` | |
| 可视化点云 `bnVisualize` | `_visualize_point_cloud` | matplotlib 3D 散点图 + 深度图 |
| 导出点云 `bnExport` | `_export_point_cloud` | PLY/OBJ/CSV |
| 浏览保存路径 | `_browse_save_path` | |

---

## 6. 连续扫描重建 `src/microscope_app/ui/dialogs/temporal_depth_dialog.py`

### 6.1 类与初始化

| 内容 | 行号 | 说明 |
|------|------|------|
| 类定义 | `L36` | `class TemporalDepthDialog(QDialog)` |
| 构造函数 | `L49` | 初始化信号，Z 定时器 200ms |

### 6.2 UI 控件（`_setup_ui` 约行 `L100`）

| 控件 | 变量名 | 说明 |
|------|--------|------|
| Z 位置实时显示 | `lblZPos` | |
| Z 起始 | `edtZ0` | |
| Z 终点 | `edtZ1` | |
| 扫描速度 | `edtSpeed` | mm/s |
| 采帧间隔 | `edtInterval` | s |
| **嵌套精扫** 复选框 | `chkNested` | 勾选后显示精扫参数 |
| 精扫区间比例 | `edtFinePct` | % |
| 点云相关参数 | `edtZScale`、`edtMinSharpness` | |
| **保存路径** | `_edt_save_path` | |
| **浏览...** | `_btn_browse_save` | |
| **开始扫描** | `bnStart` | |
| **停止** | `bnStop` | |
| **可视化点云**、**导出点云** | `bnVisualize`、`bnExport` | |

### 6.3 扫描流程

| 阶段 | 说明 |
|------|------|
| 匀速扫描取帧 | Z 轴发送连续运动指令，按时间间隔取帧，根据时间戳推算 Z 位置 |
| 嵌套精扫 | `select_focus_window` 找最清晰的 Z 范围，再精细步进扫描，用 `merge_focus_maps` 融合 |
| 点云生成 | `point_cloud_from_depth` |

---

## 7. 可编程拍摄 `src/microscope_app/ui/dialogs/programmable_shooting_dialog.py`

### 7.1 模块级常量（行 `L44`）

| 常量 | 值 | 说明 |
|------|----|------|
| `SHUTTER_MIN` | `10` | 快门最小值 µs |
| `SHUTTER_MAX` | `1000000` | 快门最大值 µs |
| `GAIN_MIN/MAX` | `0/10` | 增益范围 |
| `LIGHT_MIN/MAX` | `0/255` | 亮度范围 |
| `LIGHT_ON_DURATION` | `20` | 灯光保持秒数 |

### 7.2 核心函数

| 功能 | 行号 | 函数 | 说明 |
|------|------|------|------|
| CSV 解析 | `L57` | `_parse_csv(path)` | 编码 GBK，支持 14 位（`YYYYMMDDHHmmss`）和 12 位（`YYYYMMDDHHmm`）时间格式，超范围自动 clamp |
| 任务校验 | `L100` | `_validate_task_times(tasks)` | 仅检查非空；过期时间立即执行，无报错 |
| 图片合成视频 | `L105` | `_images_to_mp4` | OpenCV 将 BMP 序列合成 24fps MP4 |
| 参数 clamp | `L52` | `_clamp(value, lo, hi)` | 超范围归到边界 |

### 7.3 对话框类

| 内容 | 行号 | 说明 |
|------|------|------|
| 类定义 | `L120` | `class ProgrammableShootingDialog` |
| UI 控件 | `_setup_ui` | CSV 路径选择、图片保存文件夹选择、进度条、日志框、**开始执行**/**停止** 按钮 |

### 7.4 执行流程

| 阶段 | 说明 |
|------|------|
| 等待阶段 | 系统时间对比任务时间，期间关闭灯光（`M106 S0`） |
| 执行阶段 | 到达时间 → 设置快门/增益 → 开灯 → 可选自动对焦 → 拍摄保存 BMP |
| 灯光控制 | 开灯后计时 20 秒自动关 |
| 视频合成 | 全部任务完成后调用 `_images_to_mp4` → `output_24fps.mp4` |
| 文件命名 | `prog_序号_YYYYMMDD_HHMMSS.bmp` |

---

## 8. 核心算法 `src/microscope_app/core/algorithms.py`

### 8.1 锐度计算

| 函数 | 行号 | 说明 |
|------|------|------|
| `compute_sharpness_score(gray, lap_weight=0.6)` | `L25` | Tenengrad + Laplacian 方差混合得分，用于选最佳单帧 |
| `compute_laplacian_sharpness_map(gray, window_size=9)` | `L65` | 逐像素 Laplacian 平方并做盒式均值模糊，输出与输入同尺寸的锐度图 |
| `_box_mean(image, size=9)` | `L40` | 盒式均值模糊（优先用 OpenCV，退而用 scipy/纯 numpy） |

### 8.2 DFF 焦点深度恢复

| 函数 | 行号 | 说明 |
|------|------|------|
| `build_best_focus_maps(frames_gray, z_list, improve_margin=0.08)` | `L320` | 逐帧比较锐度图，取最大值构建深度图 + 全焦灰度图 |
| `build_best_focus_color_maps(frames_gray, z_list, frames_color, ...)` | `L345` | 在灰度 DFF 基础上合并彩色帧，进行颜色统计匹配和亮度注入 |
| `compute_dff_volume(frames_gray, z_positions)` | `L447` | 生成完整锐度体，argmax 取最佳帧索引，三点二次插值提升深度精度 |
| `merge_focus_maps(base_..., extra_...)` | `L497` | 将两次扫描的 DFF 结果融合（精扫覆盖粗扫） |
| `select_focus_window(z_list, frames_gray, fine_pct)` | `L504` | 粗扫后分析各 Z 区间锐度，返回精扫的 [z0, z1] 范围 |
| `select_best_single_frame(frames_gray, z_list)` | `L430` | 选锐度最高的单帧（用于设为参考帧） |

### 8.3 点云生成与导出

| 函数 | 行号 | 说明 |
|------|------|------|
| `point_cloud_from_depth(depth_map, sharp_map, intensity_map, pixels_per_mm, min_sharp, z_scale)` | `L538` | 深度图 → XYZ 点云，含连通性过滤、中值滤波、Z 离群点过滤 |
| `export_point_cloud(file_path, point_cloud, pixels_per_mm, comment)` | `L622` | 导出为 `.ply`（二进制）、`.obj` 或 `.csv`（带 XYZ/强度/RGB）|
| `_jet_rgb_from_values(values)` | `L670` | 将 Z 值映射为 Jet 彩色（用于点云颜色） |
| `_intensity_to_rgb(intensity)` | `L660` | 强度值归一化为 RGB 灰度 |

### 8.4 比例尺标定

| 函数 | 行号 | 说明 |
|------|------|------|
| `compute_blob_scale_calibration(gray, spacing_um=200.0, sample_count=5)` | `L130` | 白底黑点标定板 blob 检测，随机取 5 组圆点计算最近邻像素间距，换算 pixels/mm |
| `_blob_threshold_white_bg(gray_u8)` | `L230` | Otsu 二值化（OpenCV），fallback 纯 numpy |
| `_detect_blob_centers(binary)` | `L245` | `SimpleBlobDetector` 检测，fallback 连通域分析 |
| `_detect_blob_centers_cc(binary)` | `L271` | 纯 numpy BFS 连通域分析，筛选圆形度 ≥ 0.55 的 blob |
| `CALIB_DOT_SPACING_UM` | `L10` | 默认点距常量 `200.0` µm |

### 8.5 相位相关位移

| 函数 | 行号 | 说明 |
|------|------|------|
| `phase_correlation_shift(frame1, frame2)` | `L82` | FFT 互相关计算亚像素位移（用于自动标定） |
| `_parabolic_peak_offset(left, center, right)` | `L124` | 三点抛物线插值，精确定位峰位 |

### 8.6 文件保存与输出

| 函数 | 行号 | 说明 |
|------|------|------|
| `save_composite_image(path, gray_map, pixels_per_mm, ...)` | （下半部分） | 保存 PNG（含比例尺条）和 16bit/原始 TIFF |
| `save_output_bundle(save_dir, prefix, ...)` | （下半部分） | 统一保存全焦图/深度TIFF/点云/JSON清单 |
| `ensure_dir(path)` | （靠后） | `os.makedirs(path, exist_ok=True)` 工具函数 |

### 8.7 颜色处理（彩色相机支持）

| 函数 | 行号 | 说明 |
|------|------|------|
| `_match_color_statistics(image, reference)` | `L400` | 按通道 p10/p90 匹配颜色统计，防止颜色漂移 |
| `_inject_luminance_from_gray(color_image, gray_map)` | `L420` | 将灰度 DFF 亮度注入彩色图，再做轻微 unsharp mask |
| `_unsharp_color(image, amount=0.35, radius=0.8)` | `L445` | Unsharp mask 锐化 |
| `_foreground_mask_from_intensity(intensity_map)` | `L690` | Otsu 分割前景/背景，用于深度图后处理遮罩 |
| `_regularize_depth_for_surface(depth, mask)` | `L750` | 中值滤波 + 高斯滤波平滑深度图 |

### 8.8 工具函数

| 函数 | 行号 | 说明 |
|------|------|------|
| `get_mpl_font()` | `L12` | 获取 matplotlib 中文字体（优先 SimHei/微软雅黑） |
| `_normalize_to_uint8(img)` | `L215` | 自适应 p01/p99 拉伸并转 uint8 |
| `_smooth_weight_map(weight)` | `L390` | 高斯模糊权重图（cv2 / scipy / fallback） |

---

## 9. 设备控制 `src/microscope_app/hardware/controller.py`

### 9.1 类定义与初始化

| 内容 | 行号 | 说明 |
|------|------|------|
| 类定义 | `L41` | `class DeviceController` |
| 构造函数 | `L42` | 初始化相机/串口状态，Z 位置/限位等成员变量 |
| `_z_soft_limit` | `L51` | 默认 `68.0` mm（软件上限） |
| `_z_min_limit` | `L52` | 默认 `-3.0` mm（软件下限） |

### 9.2 相机操作

| 功能 | 行号 | 函数 |
|------|------|------|
| SDK 初始化 | `L58` | `initialize_sdk` → `MV_CC_Initialize` |
| SDK 析构 | `L62` | `finalize_sdk` → `MV_CC_Finalize` |
| 枚举设备 | `L70` | `enum_devices` → `MV_CC_EnumDevices`，返回设备名列表 |
| 格式化设备信息 | `L102` | `_format_device_info` — 支持 GigE/USB/CameraLink/CXP/XoF |
| 打开相机 | `L140` | `open_camera(index)` → `CameraOperation.Open_device` |
| 关闭相机 | `L154` | `close_camera` |
| 开始取流 | `L160` | `start_grabbing(win_id)` → `Start_grabbing`（GDI 渲染） |
| 停止取流 | `L167` | `stop_grabbing` |
| 设置连续模式 | `L174` | `set_continue_mode` → `Set_trigger_mode(False)` |
| 获取参数 | `L182` | `get_parameters` → 返回 `{exposure_time, gain, frame_rate}` |
| 设置参数 | `L191` | `set_parameters(frame_rate, exposure, gain)` |
| 设置曝光 | `L198` | `set_exposure(exposure_us)` → `MV_CC_SetFloatValue("ExposureTime", ...)` |
| 设置增益 | `L207` | `set_gain(gain_db)` → `MV_CC_SetFloatValue("Gain", ...)` |
| 取 numpy 帧 | `L218` | `get_frame_numpy` → 返回 raw 字节+宽高 |
| 取 RGB 帧 | `L222` | `get_color_frame` → `Get_frame_rgb_numpy` |
| 取灰度+彩色帧 | `L228` | `get_gray_color_frame` — 彩色相机返回 RGB，单色相机复制为灰度 |
| 取灰度帧 | `L242` | `get_gray_frame` — 保留高位深（Mono10/12）为 float32 |
| 保存 BMP | `L178` | `save_bmp_with_path(path)` |

### 9.3 串口操作

| 功能 | 行号 | 函数 |
|------|------|------|
| 列出串口 | `L252` | `list_serial_ports` → `serial.tools.list_ports.comports` |
| 连接串口 | `L256` | `connect_serial(port, baudrate, timeout)` — 8N1，连接后发 `G90`，同步 Z 位置 |
| 断开串口 | `L267` | `disconnect_serial` |
| 发送 G-code | `L274` | `send_gcode(cmd)` — 线程安全（`_serial_lock`） |
| 清空串口缓冲 | `L281` | `flush_serial_input` |
| Z 轴归零等待 | （下半部分） | `home_z_wait` — 发送 `G28 Z` + `M400` |
| Z 轴相对移动等待 | （下半部分） | `move_z_relative_wait(step_mm, feed)` — 软限位检查 + `G91 G1 Zxx + G90 + M400` |
| 刷新 Z 位置 | （下半部分） | `refresh_z_position(timeout)` — 发送 `M114`，解析 `Z:xx.xxx` |
| 控制灯光 | （通过 main_window） | 发送 `M106 S{value}` G-code |

---

## 10. 配置管理 `src/microscope_app/core/config.py`

| 内容 | 行号 | 说明 |
|------|------|------|
| `AppConfig` 数据类 | `L7` | 字段：`save_path`、`serial_port`、`baud_rate`（默认19200）、`serial_timeout`（默认1.0）、`pixels_per_mm`（默认100.0） |
| 类定义 | `L14` | `class ConfigManager` |
| 加载配置 | `L22` | `load()` — 读取 `runtime/settings.ini`，[Settings]/[Serial]/[Scale] 三个 section |
| 保存配置 | `L37` | `save()` — 写入 `runtime/settings.ini` |
| 有效保存路径 | `L90` | `effective_save_path()` — 返回 save_path 或 default_dir |
| 存储文件 | — | `runtime/settings.ini`（运行时自动生成） |

---

## 11. 比例尺叠加层 `src/microscope_app/core/overlays.py`

| 内容 | 行号 | 说明 |
|------|------|------|
| 类定义 | `L6` | `class ScaleBarOverlay(QWidget)` |
| 构造函数 | `L13` | 独立顶级窗口（`Qt.Tool | FramelessWindowHint | WindowTransparentForInput`），WA_TranslucentBackground |
| 刻度列表 | `L8` | `NICE_LENGTHS_MM` — 从 0.01mm 到 100mm 的 nice 长度 |
| 设置像素/mm | `L28` | `set_pixels_per_mm(value)` |
| 设置图像宽度 | `L32` | `set_img_width(width)` — 用于计算显示缩放比 |
| 控制显隐 | `L36` | `set_visible(visible)` — 显示时调用 `update_size()` |
| 更新位置大小 | `L43` | `update_size()` — 映射到 `widgetDisplay` 全局坐标 |
| 绘制比例尺 | `L49` | `paintEvent` — 目标宽度 20%，右下角，带黑色背景、端点刻度线、白色文字 |
| 事件过滤器 | `L135` | `class ResizeFilter(QObject)` — 监听 Resize/Move 事件，触发 `update_size` |

---

## 12. SDK 封装 `src/microscope_app/hardware/sdk/`

| 文件 | 类/内容 | 说明 |
|------|---------|------|
| `src/microscope_app/hardware/sdk/MvCameraControl_class.py` | `class MvCamera` | 封装 MVS DLL 所有 C API（`MV_CC_Initialize`、`MV_CC_EnumDevices` 等） |
| `src/microscope_app/hardware/sdk/CamOperation_class.py` | `class CameraOperation` | 高层相机操作（`Open_device`、`Start_grabbing`、`Get_frame_numpy`、`Get_frame_rgb_numpy`、`Save_Bmp_with_path` 等） |
| `src/microscope_app/hardware/sdk/MvErrorDefine_const.py` | 常量 | `MV_OK = 0`、`MV_E_PARAMETER` 等错误码 |
| `src/microscope_app/hardware/sdk/CameraParams_header.py` | ctypes 结构体 | `MV_CC_DEVICE_INFO`、`MV_CC_DEVICE_INFO_LIST`、`MV_GIGE_DEVICE`、`MV_USB_DEVICE` 等 |

---

## 附录 A：关键数据流

```
相机取流
  └─ DeviceController.get_gray_frame()
       └─ CameraOperation.Get_frame_numpy()
            └─ MvCamera.MV_CC_GetOneFrameTimeout()

扫描 → DFF → 点云
  ├─ DeviceController.move_z_relative_wait()  # Z轴步进
  ├─ DeviceController.get_gray_frame()         # 取帧
  ├─ algorithms.compute_laplacian_sharpness_map()  # 锐度图
  ├─ algorithms.build_best_focus_maps()        # DFF 融合
  ├─ algorithms.point_cloud_from_depth()       # 生成点云
  └─ algorithms.save_output_bundle()           # 保存文件

比例尺标定
  ├─ DeviceController.get_gray_frame()         # 取当前帧
  ├─ algorithms.compute_blob_scale_calibration()  # blob检测+间距计算
  └─ ConfigManager.save()                      # 写入 runtime/settings.ini
```

## 附录 B：文件行数速查

| 文件 | 估计总行数 | 核心内容 |
|------|-----------|---------|
| `main.py` | ~10 | 程序入口 |
| `src/microscope_app/ui/main_window.py` | ~850 | 主窗口全部逻辑 |
| `src/microscope_app/ui/generated.py` | ~380 | 主界面 UI 定义 |
| `src/microscope_app/core/algorithms.py` | ~1100 | 所有图像处理和算法 |
| `src/microscope_app/hardware/controller.py` | ~400 | 相机+串口控制 |
| `src/microscope_app/core/config.py` | ~95 | INI 读写 |
| `src/microscope_app/core/overlays.py` | ~145 | 比例尺叠加层 |
| `src/microscope_app/ui/dialogs/one_click_dialog.py` | ~600 | 一键出图对话框 |
| `src/microscope_app/ui/dialogs/recon_dialog.py` | ~500 | 点云重建对话框 |
| `src/microscope_app/ui/dialogs/temporal_depth_dialog.py` | ~550 | 连续扫描重建对话框 |
| `src/microscope_app/ui/dialogs/programmable_shooting_dialog.py` | ~450 | 可编程拍摄对话框 |
