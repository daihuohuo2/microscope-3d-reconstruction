"""Robust image I/O helpers, especially for Windows paths with non-ASCII characters."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def cv2_read_image_any_path(file_path: str, flags: int):
    """Read image data through cv2.imdecode so Windows Unicode paths are handled reliably."""

    if cv2 is None:
        raise ImportError("OpenCV is required for cv2_read_image_any_path")
    data = np.fromfile(os.fspath(file_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def cv2_write_image_any_path(file_path: str, image: np.ndarray, params: Optional[list] = None) -> bool:
    """Write image data through cv2.imencode + tofile so Windows Unicode paths are handled reliably."""

    if cv2 is None:
        raise ImportError("OpenCV is required for cv2_write_image_any_path")
    ext = os.path.splitext(os.fspath(file_path))[1]
    if not ext:
        raise ValueError("Output path must include an image extension: {}".format(file_path))
    ok, encoded = cv2.imencode(ext, image, params or [])
    if not ok:
        return False
    encoded.tofile(os.fspath(file_path))
    return True
