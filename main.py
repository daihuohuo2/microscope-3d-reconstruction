"""Single entry point for the microscope application and its tools."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def ensure_project_python(*, require_qt: bool) -> None:
    """Relaunch with a Python installation that provides the project dependencies."""

    required_modules = ["numpy", "cv2", "scipy", "matplotlib"]
    if require_qt:
        required_modules.append("PyQt5")
    if all(importlib.util.find_spec(module) is not None for module in required_modules):
        return

    candidates = [
        os.environ.get("MICROSCOPE_PYTHON", ""),
        r"D:\Anaconda\python.exe",
        os.path.expandvars(r"%USERPROFILE%\anaconda3\python.exe"),
    ]
    current = os.path.normcase(os.path.abspath(sys.executable))
    for candidate in candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        if os.path.normcase(os.path.abspath(candidate)) == current:
            continue
        os.execv(candidate, [candidate, os.path.abspath(__file__), *sys.argv[1:]])

    raise RuntimeError(
        "当前 Python 缺少项目依赖，且未找到可用的项目 Conda 环境。"
        "请安装 requirements.txt 中的依赖，或设置 MICROSCOPE_PYTHON。"
    )


def print_help() -> None:
    print(
        "用法:\n"
        "  python main.py                  启动主界面\n"
        "  python main.py gui              启动主界面\n"
        "  python main.py zstack-gui       启动离线 Z-stack 界面\n"
        "  python main.py reconstruct ...  执行 Z-stack 重建\n"
        "  python main.py measure ...      执行三维测量\n"
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments[0] if arguments else "gui"

    if command in {"-h", "--help", "help"}:
        print_help()
        return 0
    if command in {"gui", "zstack-gui"}:
        ensure_project_python(require_qt=True)
        from microscope_app.application import run_gui, run_zstack_gui

        return run_gui() if command == "gui" else run_zstack_gui()
    if command in {"reconstruct", "measure"}:
        ensure_project_python(require_qt=False)
        from microscope_app.reconstruction.cli import main as reconstruction_main

        return reconstruction_main(arguments)

    print(f"未知命令: {command}", file=sys.stderr)
    print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
