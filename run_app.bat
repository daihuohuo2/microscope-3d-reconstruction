@echo off
setlocal
cd /d "%~dp0"
set "APP_ROOT=%~dp0"
set "RUNTIME_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "PYTHONPATH=%APP_ROOT%.venv\Lib\site-packages"
set "QT_PLUGIN_PATH=%APP_ROOT%.venv\Lib\site-packages\PyQt5\Qt5\plugins"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%APP_ROOT%.venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms"
"%RUNTIME_PYTHON%" "%APP_ROOT%main.py"
