"""Depth-from-focus reconstruction for microscope Z-stack data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .focus import compute_focus_volume


@dataclass
class ReconstructionConfig:
    """Configuration for depth reconstruction."""

    focus_method: str = "combined"
    focus_window_size: int = 9
    sobel_ksize: int = 3
    laplacian_weight: float = 0.5
    focus_threshold_percentile: float = 8.0
    median_filter_size: int = 5
    gaussian_sigma: float = 0.8
    enable_parabolic_refinement: bool = True
    enable_depth_smoothing: bool = True
    fill_invalid_for_smoothing: bool = True
    smoothing_strength: str = "light"


@dataclass
class ReconstructionResult:
    """Output bundle produced by the reconstruction pipeline."""

    file_paths: List[str]
    z_positions_mm: np.ndarray
    depth_map_mm: np.ndarray
    raw_depth_map_mm: np.ndarray
    display_depth_map_mm: np.ndarray
    focus_index_map: np.ndarray
    focus_score_map: np.ndarray
    full_focus_gray: np.ndarray
    full_focus_rgb: np.ndarray
    valid_mask: np.ndarray
    focus_volume_shape: tuple
    frame_focus_scores: np.ndarray
    alignment_offsets_px: List[tuple]
    reconstruction_config: Dict[str, object]


def reconstruct_from_stack(stack, config: Optional[ReconstructionConfig] = None) -> ReconstructionResult:
    """Reconstruct a height map from a loaded Z-stack."""

    config = config or ReconstructionConfig()
    gray_stack = np.asarray(stack.gray_stack, dtype=np.float32)
    color_stack = np.asarray(stack.color_stack, dtype=np.uint8)
    z_positions = np.asarray(stack.z_positions_mm, dtype=np.float32)

    if gray_stack.ndim != 3:
        raise ValueError("stack.gray_stack must have shape [n, h, w]")
    if len(gray_stack) < 2:
        raise ValueError("At least two frames are required for Z-stack reconstruction")

    focus_volume = compute_focus_volume(
        gray_stack=gray_stack,
        method=config.focus_method,
        window_size=config.focus_window_size,
        sobel_ksize=config.sobel_ksize,
        laplacian_weight=config.laplacian_weight,
    )
    focus_index = np.argmax(focus_volume, axis=0).astype(np.int32)
    row_index, col_index = np.indices(focus_index.shape)
    focus_score = focus_volume[focus_index, row_index, col_index].astype(np.float32)

    raw_depth = z_positions[focus_index].astype(np.float32)
    if config.enable_parabolic_refinement and len(z_positions) >= 3:
        raw_depth = _refine_depth_substep(focus_volume, focus_index, z_positions, raw_depth)

    full_focus_gray = gray_stack[focus_index, row_index, col_index].astype(np.float32)
    full_focus_rgb = color_stack[focus_index, row_index, col_index].astype(np.uint8)

    valid_mask = _estimate_valid_mask(full_focus_gray, focus_score, config.focus_threshold_percentile)
    depth_map = raw_depth.copy()
    if config.enable_depth_smoothing:
        depth_map = _smooth_depth_map(
            depth_map=depth_map,
            valid_mask=valid_mask,
            median_filter_size=config.median_filter_size,
            gaussian_sigma=config.gaussian_sigma,
            fill_invalid=config.fill_invalid_for_smoothing,
            smoothing_strength=config.smoothing_strength,
        )
    depth_map = depth_map.astype(np.float32)
    depth_map[~valid_mask] = np.nan
    raw_depth_visible = raw_depth.astype(np.float32).copy()
    raw_depth_visible[~valid_mask] = np.nan

    frame_scores = np.mean(focus_volume, axis=(1, 2)).astype(np.float32)
    return ReconstructionResult(
        file_paths=list(stack.file_paths),
        z_positions_mm=z_positions,
        depth_map_mm=depth_map,
        raw_depth_map_mm=raw_depth.astype(np.float32),
        display_depth_map_mm=raw_depth_visible,
        focus_index_map=focus_index,
        focus_score_map=focus_score,
        full_focus_gray=full_focus_gray,
        full_focus_rgb=full_focus_rgb,
        valid_mask=valid_mask.astype(bool),
        focus_volume_shape=tuple(focus_volume.shape),
        frame_focus_scores=frame_scores,
        alignment_offsets_px=list(stack.alignment_offsets_px),
        reconstruction_config=asdict(config),
    )


def _refine_depth_substep(
    focus_volume: np.ndarray,
    focus_index: np.ndarray,
    z_positions_mm: np.ndarray,
    depth_map_mm: np.ndarray,
) -> np.ndarray:
    depth_map = np.asarray(depth_map_mm, dtype=np.float32).copy()
    interior = (focus_index > 0) & (focus_index < len(z_positions_mm) - 1)
    rows, cols = np.where(interior)
    if len(rows) == 0:
        return depth_map

    center_index = focus_index[rows, cols]
    prev_score = focus_volume[center_index - 1, rows, cols]
    curr_score = focus_volume[center_index, rows, cols]
    next_score = focus_volume[center_index + 1, rows, cols]
    denom = prev_score + next_score - 2.0 * curr_score
    valid = denom < -1e-9
    if not np.any(valid):
        return depth_map

    numerator = prev_score - next_score
    z_center = z_positions_mm[center_index]
    z_half_span = (z_positions_mm[center_index + 1] - z_positions_mm[center_index - 1]) * 0.5
    offsets = np.zeros_like(z_center, dtype=np.float32)
    safe_denom = np.where(valid, denom, -1.0)
    offsets[valid] = numerator[valid] / (2.0 * safe_denom[valid]) * z_half_span[valid]
    max_offset = np.abs(z_positions_mm[center_index + 1] - z_center)
    offsets = np.clip(offsets, -max_offset, max_offset)
    depth_map[rows, cols] = z_center + offsets
    return depth_map


def _estimate_valid_mask(
    full_focus_gray: np.ndarray,
    focus_score: np.ndarray,
    focus_threshold_percentile: float,
) -> np.ndarray:
    gray = np.asarray(full_focus_gray, dtype=np.float32)
    focus = np.asarray(focus_score, dtype=np.float32)

    focus_finite = focus[np.isfinite(focus)]
    if focus_finite.size == 0:
        return np.ones(gray.shape, dtype=bool)
    focus_threshold = float(np.percentile(focus_finite, max(0.0, min(100.0, focus_threshold_percentile))))
    focus_mask = focus >= focus_threshold

    intensity_u8 = _normalize_uint8(gray)
    border = max(3, int(min(gray.shape) * 0.05))
    border_samples = np.concatenate(
        [
            intensity_u8[:border, :].ravel(),
            intensity_u8[-border:, :].ravel(),
            intensity_u8[:, :border].ravel(),
            intensity_u8[:, -border:].ravel(),
        ]
    ).astype(np.float32)
    background_level = float(np.median(border_samples))
    background_spread = float(np.percentile(border_samples, 90.0) - np.percentile(border_samples, 10.0))
    threshold = max(10.0, background_spread * 1.2)
    if background_level >= 150.0:
        object_mask = intensity_u8 <= background_level - threshold
    elif background_level <= 100.0:
        object_mask = intensity_u8 >= background_level + threshold
    else:
        object_mask = np.abs(intensity_u8.astype(np.float32) - background_level) >= threshold

    object_mask = _morphology_cleanup(object_mask)
    object_coverage = float(np.mean(object_mask))
    if object_coverage < 0.01 or object_coverage > 0.98:
        return focus_mask.astype(bool)
    return (focus_mask & object_mask).astype(bool)


def _smooth_depth_map(
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    median_filter_size: int,
    gaussian_sigma: float,
    fill_invalid: bool,
    smoothing_strength: str,
) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid_mask):
        return depth

    if fill_invalid:
        fill_value = float(np.nanmedian(depth[valid_mask]))
        depth[~valid_mask] = fill_value
    else:
        depth = np.where(valid_mask, depth, np.nan)

    median_filter_size = int(max(1, median_filter_size))
    if median_filter_size % 2 == 0:
        median_filter_size += 1

    if smoothing_strength == "off":
        return depth

    if smoothing_strength == "light":
        median_filter_size = max(1, min(median_filter_size, 3))
        if median_filter_size % 2 == 0:
            median_filter_size += 1
        gaussian_sigma = min(float(gaussian_sigma), 0.45)
    elif smoothing_strength == "medium":
        pass
    elif smoothing_strength == "strong":
        median_filter_size = max(median_filter_size, 7)
        if median_filter_size % 2 == 0:
            median_filter_size += 1
        gaussian_sigma = max(float(gaussian_sigma), 1.2)
    else:
        raise ValueError("Unsupported smoothing_strength: {}".format(smoothing_strength))

    if cv2 is not None:
        smoothed = cv2.medianBlur(depth.astype(np.float32), median_filter_size)
        if gaussian_sigma > 0:
            smoothed = cv2.GaussianBlur(smoothed, (0, 0), gaussian_sigma)
    else:
        smoothed = _median_filter_numpy(depth, median_filter_size)
        if gaussian_sigma > 0:
            smoothed = _box_blur_numpy(smoothed, max(3, int(round(gaussian_sigma * 4.0)) | 1))

    outlier = np.abs(smoothed - depth)
    residual = outlier[valid_mask]
    if residual.size:
        tolerance = max(0.002, float(np.percentile(residual, 90.0)) * 1.8)
        depth[valid_mask & (outlier > tolerance)] = smoothed[valid_mask & (outlier > tolerance)]
    depth[valid_mask] = smoothed[valid_mask]
    return depth


def _normalize_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.0))
    if high <= low + 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - low) / (high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _morphology_cleanup(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    if cv2 is not None:
        kernel = np.ones((5, 5), dtype=np.uint8)
        opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if num_labels <= 1:
            return closed.astype(bool)
        areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
        areas[0] = 0
        largest = int(np.argmax(areas))
        return labels == largest

    pad = np.pad(mask.astype(np.uint8), 1, mode="edge")
    neighbor_sum = (
        pad[:-2, :-2]
        + pad[:-2, 1:-1]
        + pad[:-2, 2:]
        + pad[1:-1, :-2]
        + pad[1:-1, 1:-1]
        + pad[1:-1, 2:]
        + pad[2:, :-2]
        + pad[2:, 1:-1]
        + pad[2:, 2:]
    )
    return neighbor_sum >= 4


def _median_filter_numpy(image: np.ndarray, size: int) -> np.ndarray:
    pad = size // 2
    padded = np.pad(image, pad, mode="edge")
    out = np.empty_like(image, dtype=np.float32)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            window = padded[row : row + size, col : col + size]
            out[row, col] = float(np.median(window))
    return out


def _box_blur_numpy(image: np.ndarray, size: int) -> np.ndarray:
    pad = size // 2
    padded = np.pad(image, pad, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    out = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return out / float(size * size)
