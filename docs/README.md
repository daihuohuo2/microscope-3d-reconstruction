# 项目文档

本目录只存放正式文档和共享预约网站源码。所有运行入口仍统一使用项目根目录的 `main.py`。

## 文档导航

### 使用指南

- [用户使用手册](guides/user-guide.md)：安装、设备连接、主要功能和故障排除。
- [Z-stack 三维重建指南](guides/zstack-3d.md)：离线重建、测量命令和自动化接口。
- [点云高质量采集指南](guides/point-cloud-reconstruction.md)：采集建议、参数选择和质量诊断。

### 开发文档

- [代码索引](development/code-index.md)：界面、算法、硬件控制与源码文件的对应关系。

### 部署文档

- [GitHub Pages 部署](deployment/github-pages.md)：共享预约网站的发布和配置方法。

## 目录约定

```text
docs/
├── README.md                   # 文档总入口
├── guides/                    # 面向用户和操作人员
├── development/               # 面向开发与维护人员
├── deployment/                # 部署和运维说明
└── website/                   # 可直接发布的静态网站源码
```

新增文档时请放入对应分类，并在本页补充链接；不要把导出图片、PDF 预览页或临时报告放入 `docs/`。
