"""Qt application lifecycle and GUI launchers."""

import sys

from .paths import PROJECT_ROOT, SETTINGS_FILE, ensure_runtime_dirs


def run_gui() -> int:
    """Start the main microscope control interface."""

    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QDesktopServices
    from PyQt5.QtWidgets import QApplication, QMessageBox

    from .hardware.controller import MV_SDK_AVAILABLE, MV_SDK_ERROR_MSG
    from .ui.main_window import MainWindow

    ensure_runtime_dirs()
    app = QApplication(sys.argv)

    if not MV_SDK_AVAILABLE:
        sdk_url = "https://www.hikrobotics.com/cn/machinevision/service/download?module=0"
        message = QMessageBox()
        message.setIcon(QMessageBox.Warning)
        message.setWindowTitle("相机 SDK 未找到")
        message.setText(
            "海康威视 MVS 相机 SDK 未安装或无法加载，\n"
            "相机相关功能将不可用，其他功能正常使用。\n\n"
            "请前往官网下载并安装 MVS SDK：\n" + sdk_url
        )
        if MV_SDK_ERROR_MSG:
            message.setDetailedText(MV_SDK_ERROR_MSG)
        open_button = message.addButton("打开下载页面", QMessageBox.ActionRole)
        message.addButton("知道了", QMessageBox.AcceptRole)
        message.exec_()
        if message.clickedButton() == open_button:
            QDesktopServices.openUrl(QUrl(sdk_url))

    window = MainWindow()
    window.show()
    exit_code = app.exec_()
    window.cleanup()
    return exit_code


def run_zstack_gui() -> int:
    """Start the standalone offline Z-stack reconstruction interface."""

    from PyQt5.QtWidgets import QApplication

    from .core.config import ConfigManager
    from .ui.dialogs.offline_zstack_dialog import OfflineZStackDialog

    ensure_runtime_dirs()
    app = QApplication(sys.argv)
    config = ConfigManager(str(SETTINGS_FILE), str(PROJECT_ROOT))
    config.load()
    window = OfflineZStackDialog(config)
    window.show()
    return app.exec_()
