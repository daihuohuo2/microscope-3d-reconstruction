"""Centralized filesystem locations used by the application."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINT = PROJECT_ROOT / "main.py"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
SETTINGS_FILE = RUNTIME_DIR / "settings.ini"


def ensure_runtime_dirs() -> None:
    """Create directories that hold user-specific runtime state."""

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
