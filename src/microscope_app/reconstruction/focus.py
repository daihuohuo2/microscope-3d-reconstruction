"""Pixel-wise and frame-wise focus metric computation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def compute_focus_volume(
    gray_stack: np.ndarray,
    method: str = "combined",
    window_size: int = 9,
    sobel_ksize: int = 3,
    laplacian_weight: float = 0.5,
) -> np.ndarray:
    """Compute a focus score map for every image in the stack."""

    gray_stack = np.asarray(gray_stack, dtype=np.float32)
    if gray_stack.ndim != 3:
        raise ValueError("gray_stack must have shape [n, h, w]")

    method = method.lower()
    if method == "combined":
        # Use stack-global normalization instead of per-frame normalization.
        # Per-frame normalization makes every Z slice look equally "strong",
        # which weakens the true focus peak along the Z axis.
        lap_volume = np.stack([_laplacian_map(gray, window_size=window_size) for gray in gray_stack], axis=0)
        sob_volume = np.stack(
            [_sobel_map(gray, window_size=window_size, sobel_ksize=sobel_ksize) for gray in gray_stack],
            axis=0,
        )
        lap_norm = _normalize_focus_volume(lap_volume)
        sob_norm = _normalize_focus_volume(sob_volume)
        combined = (laplacian_weight * lap_norm) + ((1.0 - laplacian_weight) * sob_norm)
        return combined.astype(np.float32)

    volume = []
    for gray in gray_stack:
        focus_map = compute_focus_map(
            gray=gray,
            method=method,
            window_size=window_size,
            sobel_ksize=sobel_ksize,
            laplacian_weight=laplacian_weight,
        )
        volume.append(focus_map.astype(np.float32))
    return np.stack(volume, axis=0)


def compute_focus_map(
    gray: np.ndarray,
    method: str = "combined",
    window_size: int = 9,
    sobel_ksize: int = 3,
    laplacian_weight: float = 0.5,
) -> np.ndarray:
    """Compute a pixel-wise focus map."""

    gray = np.asarray(gray, dtype=np.float32)
    method = method.lower()

    if method == "laplacian":
        return _laplacian_map(gray, window_size=window_size)
    if method == "sobel":
        return _sobel_map(gray, window_size=window_size, sobel_ksize=sobel_ksize)
    if method == "tenengrad":
        return _sobel_map(gray, window_size=window_size, sobel_ksize=sobel_ksize)
    if method == "combined":
        lap = _laplacian_map(gray, window_size=window_size)
        sob = _sobel_map(gray, window_size=window_size, sobel_ksize=sobel_ksize)
        lap_norm = _normalize_focus_map(lap)
        sob_norm = _normalize_focus_map(sob)
        return ((laplacian_weight * lap_norm) + ((1.0 - laplacian_weight) * sob_norm)).astype(np.float32)

    raise ValueError("Unsupported focus metric: {}".format(method))


def compute_frame_focus_scores(
    gray_stack: np.ndarray,
    method: str = "combined",
    window_size: int = 9,
    sobel_ksize: int = 3,
    laplacian_weight: float = 0.5,
) -> np.ndarray:
    """Return a single global focus score for every frame."""

    volume = compute_focus_volume(
        gray_stack=gray_stack,
        method=method,
        window_size=window_size,
        sobel_ksize=sobel_ksize,
        laplacian_weight=laplacian_weight,
    )
    return np.mean(volume, axis=(1, 2)).astype(np.float32)


def _laplacian_map(gray: np.ndarray, window_size: int) -> np.ndarray:
    if cv2 is not None:
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        energy = lap * lap
    else:
        lap = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4.0 * gray[1:-1, 1:-1]
        )
        energy = np.zeros_like(gray, dtype=np.float32)
        energy[1:-1, 1:-1] = lap * lap
    return _box_blur(energy, window_size)


def _sobel_map(gray: np.ndarray, window_size: int, sobel_ksize: int) -> np.ndarray:
    if cv2 is not None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
        energy = gx * gx + gy * gy
    else:
        gx = np.zeros_like(gray, dtype=np.float32)
        gy = np.zeros_like(gray, dtype=np.float32)
        gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
        gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
        energy = gx * gx + gy * gy
    return _box_blur(energy, window_size)


def _box_blur(image: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.asarray(image, dtype=np.float32)
    if cv2 is not None:
        return cv2.blur(np.asarray(image, dtype=np.float32), (size, size), borderType=cv2.BORDER_REFLECT101)

    # Integral-image fallback keeps the implementation dependency-light.
    arr = np.asarray(image, dtype=np.float32)
    pad = size // 2
    padded = np.pad(arr, pad, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    out = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return out / float(size * size)


def _normalize_focus_map(focus_map: np.ndarray) -> np.ndarray:
    arr = np.asarray(focus_map, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    scale = float(np.percentile(finite, 99.5))
    if scale <= 1e-9:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip(arr / scale, 0.0, 1.0).astype(np.float32)


def _normalize_focus_volume(focus_volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(focus_volume, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)

    scale = float(np.percentile(finite, 99.8))
    if scale <= 1e-9:
        return np.zeros(arr.shape, dtype=np.float32)

    # Keep a little headroom instead of clipping everything to 1.0.
    return np.clip(arr / scale, 0.0, 4.0).astype(np.float32)
