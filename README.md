# 显微成像三维重建系统

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)

一个基于焦点堆叠（Focus Stacking）和深度重建技术的显微成像三维重建系统。通过控制 Z 轴扫描和相机自动拍摄，生成深度图、点云数据和全焦合成图像。

---

## 📋 目录

- [功能特性](#-功能特性)
- [技术原理](#-技术原理)
- [输出类型](#-输出类型)
- [系统要求](#-系统要求)
- [安装说明](#-安装说明)
- [使用指南](#-使用指南)
- [项目结构](#-项目结构)
- [依赖库](#-依赖库)
- [常见问题](#-常见问题)
- [许可证](#-许可证)

---

## ✨ 功能特性

### 1. 点云重建
- **原理**：Z 轴逐步停顿，每步拍摄一帧灰度图，通过逐像素锐度分析确定每个像素的最佳焦平面深度值
- **算法**：DFF（Depth From Focus）焦点融合算法
- **输出**：深度图 + 点云数据（.ply / .csv 格式）+ 全焦合成图

### 2. 连续扫描重建
- **原理**：Z 轴匀速连续扫描，相机按固定时间间隔采集帧，根据时间戳映射 Z 位置
- **高级功能**：支持嵌套精扫融合，提高深度精度
- **输出**：高精度深度图 + 点云数据 + 锐度图

### 3. 一键出图（快速全焦成像）
- **原理**：Z 轴从高位向低位逐步停拍，自动执行焦点融合
- **特点**：全自动流程，无需手动干预
- **输出**：全焦合成图（BMP 格式），自动保存并预览
- **高级选项**（可选）：同时生成点云数据，支持可视化和导出 PLY/CSV

### 4. 可视化与导出
- 实时预览拍摄画面
- 支持比例尺叠加显示
- 点云 3D 可视化（matplotlib 3D 渲染）
- 多格式导出：PLY（点云标准格式）、CSV（原始数据）、BMP（图像）

---

## 🔬 技术原理

### DFF（Depth From Focus）算法
通过分析同一场景在不同焦平面的图像锐度分布，逐像素选择最清晰的焦平面，从而反推深度信息。

**核心步骤**：
1. **多层拍摄**：Z 轴移动到不同深度位置，采集多张图像
2. **锐度计算**：使用拉普拉斯算子计算每个像素的锐度值
3. **深度映射**：每个像素选择锐度最大的帧对应的 Z 值作为深度
4. **焦点融合**：合成全焦图像（所有像素都处于最佳焦点状态）

### 点云生成
- 根据深度图和相机参数，将 2D 深度图转换为 3D 点云
- 支持 Z 轴缩放调整，适配不同显微系统
- 可导出为 PLY 格式，兼容 MeshLab、CloudCompare 等点云处理软件

---

## 🖼️ 输出类型

| 输出类型 | 格式 | 说明 |
|---------|------|------|
| **深度图** | 灰度图像 / NumPy 数组 | 每个像素值代表该点的深度（Z 坐标） |
| **全焦合成图** | BMP / PNG | 所有像素都处于最清晰焦点的合成图像 |
| **点云数据** | PLY / CSV | 三维坐标点集，包含 X, Y, Z 和灰度信息 |
| **锐度图** | 灰度图像 | 每个像素的锐度分布，用于质量评估 |

### 示例输出

```
output/
├── depth_map_20260321_143022.png       # 深度图（伪彩色）
├── composite_20260321_143022.bmp       # 全焦合成图
├── pointcloud_20260321_143022.ply      # 点云文件（3D）
└── pointcloud_20260321_143022.csv      # 点云原始数据
```

---

## 💻 系统要求

### 硬件要求
- **相机**：支持 MvCamera SDK 的工业相机（海康威视等）
- **运动平台**：Z 轴电动平台，串口通信（可选）
- **操作系统**：Windows 7/10/11
- **内存**：建议 8GB 及以上

### 软件要求
- Python 3.7 或更高版本
- PyQt5（图形界面）
- NumPy（数值计算）
- Matplotlib（3D 可视化）

---

## 📦 安装说明

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/microscope-3d-reconstruction.git
cd microscope-3d-reconstruction
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 安装相机 SDK
根据你的相机型号，安装对应的 SDK 驱动：
- 海康威视工业相机：安装 MVS SDK
- 其他品牌：将 SDK 的 Python 接口放置到 `sdk/` 目录

### 4. 配置串口（可选）
如果需要控制 Z 轴运动平台，安装 pySerial：
```bash
pip install pyserial
```

---

## 🚀 使用指南

### 启动程序
```bash
python main.py
```

### 基本操作流程

#### 方法一：点云重建
1. **连接设备**
   - 点击"枚举设备"查找相机
   - 选择相机并点击"打开设备"
   - 如有 Z 轴平台，连接串口

2. **设置参数**
   - 打开"三维重建" → "点云重建"
   - 设置扫描范围（如 -2.0 ~ 2.0 mm）
   - 设置步长（如 0.1 mm，步长越小精度越高）
   - 设置每步延时（如 0.15 秒）

3. **开始扫描**
   - 点击"开始扫描"按钮
   - 系统自动控制 Z 轴移动并采集图像
   - 实时显示进度和状态

4. **查看结果**
   - 扫描完成后自动生成深度图和点云
   - 点击"可视化"查看 3D 点云
   - 点击"导出 PLY" 或 "导出 CSV" 保存数据

#### 方法二：连续扫描重建
1. 打开"三维重建" → "连续扫描重建"
2. 设置扫描速度和采集间隔
3. 可选开启"嵌套精扫"提高精度
4. 开始扫描并自动生成结果

#### 方法三：一键出图（最快）
1. 打开"出图菜单" → "一键出图"
2. 设置 Z 轴扫描范围（从上到下）
3. （可选）勾选"同时生成点云数据"，设置 Z 轴缩放和锐度阈值
4. 点击"开始"，全自动完成拍摄和融合
5. 预览并保存全焦合成图
6. 如果生成了点云，可点击"可视化点云"或"导出点云"

---

## 📁 项目结构

```
.
├── main.py                    # 程序入口
├── main_window.py             # 主窗口逻辑
├── device_controller.py       # 设备控制（相机 + 串口）
├── algorithms.py              # 核心算法（DFF、点云生成等）
├── config_manager.py          # 配置文件管理
├── overlays.py                # 界面叠加层（比例尺等）
├── setting.ini                # 配置文件
│
├── dialogs/                   # 功能对话框
│   ├── __init__.py
│   ├── recon_dialog.py        # 点云重建对话框
│   ├── temporal_depth_dialog.py   # 时间换位深度对话框
│   └── one_click_dialog.py    # 一键出图对话框
│
├── sdk/                       # 相机 SDK 接口
│   ├── CamOperation_class.py
│   ├── MvCameraControl_class.py
│   ├── MvErrorDefine_const.py
│   └── CameraParams_header.py
│
└── ui/                        # UI 界面文件
    └── PyUICBasicDemo.py
```

---

## 📚 依赖库

创建 `requirements.txt` 文件：

```
PyQt5>=5.15.0
numpy>=1.19.0
matplotlib>=3.3.0
pyserial>=3.5
opencv-python>=4.5.0  # 可选，用于图像处理
```

安装命令：
```bash
pip install -r requirements.txt
```

---

## ❓ 常见问题

### Q1: 相机连接失败怎么办？
**A**:
- 检查相机是否正确连接并供电
- 确认已安装相机厂商提供的 SDK
- 尝试重新枚举设备

### Q2: Z 轴不移动怎么办？
**A**:
- 检查串口连接是否正常
- 确认波特率设置正确（默认 115200）
- 检查运动平台供电和使能状态

### Q3: 生成的点云质量差怎么办？
**A**:
- 减小 Z 轴步长，提高采样密度
- 调整相机曝光时间，确保图像清晰
- 增加每步延时，确保平台稳定后再拍摄
- 使用"连续扫描重建 + 嵌套精扫"模式

### Q4: 支持哪些相机品牌？
**A**:
当前主要支持海康威视工业相机（MVS SDK）。其他品牌相机需要修改 `sdk/` 目录下的接口代码。

### Q5: 输出的 .ply 文件如何查看？
**A**:
推荐使用以下软件：
- **MeshLab**（免费开源）
- **CloudCompare**（免费开源）
- **Blender**（免费开源，功能强大）

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 📧 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 Issue：[GitHub Issues](https://github.com/yourusername/microscope-3d-reconstruction/issues)
- 邮箱：your.email@example.com

---

## 🙏 致谢

感谢以下开源项目：
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - 图形界面框架
- [NumPy](https://numpy.org/) - 数值计算库
- [Matplotlib](https://matplotlib.org/) - 数据可视化库

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
