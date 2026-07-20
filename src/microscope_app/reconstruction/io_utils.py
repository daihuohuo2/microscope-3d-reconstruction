"""Image discovery, loading, and optional stack alignment utilities."""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - fallback is used when OpenCV is absent
    cv2 = None

try:
    import matplotlib.image as mpimg
except ImportError:  # pragma: no cover
    mpimg = None

from .image_codec import cv2_read_image_any_path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
Z_TOKEN_PATTERN = re.compile(r"[zZ]([+-]?\d+(?:\.\d+)?)")
Z_WITH_UNIT_PATTERN = re.compile(r"[zZ]([+-]?\d+(?:\.\d+)?)(mm|um)\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
IGNORED_IMAGE_KEYWORDS = (
    "full_focus",
    "focus_compare",
    "depth_heatmap",
    "frame_focus_scores",
    "surface_point_cloud",
    "depth_map",
    "point_cloud",
)


@dataclass
class LoadedStack:
    """Container for a loaded Z-stack."""

    file_paths: List[str]
    z_positions_mm: np.ndarray
    gray_stack: np.ndarray
    color_stack: np.ndarray
    alignment_offsets_px: List[Tuple[float, float]]
    source_dir: str
    parsed_z_values: List[Optional[float]]
    z_source: str
    skipped_files: List[str]


def discover_image_paths(input_path: str) -> List[str]:
    """Return image paths from a directory, wildcard, or single file."""

    input_path = os.path.abspath(input_path)
    input_path = _resolve_stack_directory(input_path)
    if os.path.isdir(input_path):
        paths = []
        for entry in os.listdir(input_path):
            full_path = os.path.join(input_path, entry)
            if (
                os.path.isfile(full_path)
                and entry.lower().endswith(IMAGE_EXTENSIONS)
                and not _is_derived_output_image(entry)
            ):
                paths.append(full_path)
    elif any(char in input_path for char in "*?[]"):
        paths = [os.path.abspath(path) for path in glob.glob(input_path)]
        paths = [path for path in paths if os.path.isfile(path)]
    elif os.path.isfile(input_path):
        paths = [input_path]
    else:
        raise FileNotFoundError("Input path does not exist: {}".format(input_path))

    if not paths:
        raise FileNotFoundError("No supported images found under: {}".format(input_path))

    z_tagged = [path for path in paths if _has_z_token(os.path.basename(path))]
    if z_tagged:
        paths = z_tagged

    paths = sorted(paths, key=_sort_key_for_stack)
    return paths


def parse_z_value_from_name(file_path: str) -> Optional[float]:
    """Extract the numeric token after a 'z' marker in the file name."""

    stem = os.path.splitext(os.path.basename(file_path))[0]
    match = Z_TOKEN_PATTERN.search(stem)
    if match:
        return float(match.group(1))

    numbers = NUMBER_PATTERN.findall(stem)
    if numbers:
        return float(numbers[-1])
    return None


def parse_z_value_with_unit_from_name(file_path: str) -> Optional[Tuple[float, str]]:
    """Return a tuple like (0.123, 'mm') when the name contains an explicit unit."""

    stem = os.path.splitext(os.path.basename(file_path))[0]
    match = Z_WITH_UNIT_PATTERN.search(stem)
    if not match:
        return None
    return float(match.group(1)), match.group(2).lower()


def infer_z_positions_mm(
    file_paths: Sequence[str],
    filename_z_unit: str = "auto",
    z_step_mm: Optional[float] = None,
    z_start_mm: float = 0.0,
    z_positions_override_mm: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[Optional[float]], str]:
    """Infer per-frame Z positions in millimeters.

    filename_z_unit:
        - "index": filename number is treated as a frame index and multiplied by z_step_mm
        - "um": filename number is treated as micrometers
        - "mm": filename number is treated as millimeters
    """

    if z_positions_override_mm is not None:
        override = np.asarray(list(z_positions_override_mm), dtype=np.float32)
        if len(override) != len(file_paths):
            raise ValueError(
                "Z metadata length mismatch: {} positions for {} files".format(len(override), len(file_paths))
            )
        parsed_values = [float(v) for v in override]
        return override.astype(np.float32), parsed_values, "metadata"

    parsed_values = [parse_z_value_from_name(path) for path in file_paths]
    explicit_units = [parse_z_value_with_unit_from_name(path) for path in file_paths]

    if filename_z_unit == "auto":
        units_found = {unit for item in explicit_units if item is not None for unit in [item[1]]}
        if len(units_found) == 1 and all(item is not None for item in explicit_units):
            filename_z_unit = next(iter(units_found))
        elif z_step_mm is not None:
            filename_z_unit = "index"
        else:
            raise ValueError(
                "Could not infer real Z positions automatically. "
                "Use filenames like '..._z+0.123mm.png', or provide a Z step, or use a folder that contains manifest.json."
            )

    z_values_mm = []
    for index, parsed in enumerate(parsed_values):
        if filename_z_unit == "index":
            frame_index = parsed if parsed is not None else float(index)
            if z_step_mm is None:
                z_mm = z_start_mm + frame_index
            else:
                z_mm = z_start_mm + frame_index * float(z_step_mm)
        elif filename_z_unit == "um":
            unit_item = explicit_units[index]
            value_um = unit_item[0] if unit_item is not None else parsed
            if value_um is None:
                raise ValueError(
                    "File '{}' does not contain a readable Z token; cannot use filename_z_unit='um'".format(
                        os.path.basename(file_paths[index])
                    )
                )
            z_mm = float(value_um) / 1000.0
        elif filename_z_unit == "mm":
            unit_item = explicit_units[index]
            value_mm = unit_item[0] if unit_item is not None else parsed
            if value_mm is None:
                raise ValueError(
                    "File '{}' does not contain a readable Z token; cannot use filename_z_unit='mm'".format(
                        os.path.basename(file_paths[index])
                    )
                )
            z_mm = float(value_mm)
        else:
            raise ValueError("Unsupported filename_z_unit: {}".format(filename_z_unit))
        z_values_mm.append(z_mm)

    z_positions = np.asarray(z_values_mm, dtype=np.float32)
    return z_positions, parsed_values, "filename:{}".format(filename_z_unit)


def load_zstack_from_path(
    input_path: str,
    filename_z_unit: str = "auto",
    z_step_mm: Optional[float] = None,
    z_start_mm: float = 0.0,
    align: bool = False,
    reference_index: Optional[int] = None,
) -> LoadedStack:
    """Load a batch of microscope images into grayscale and RGB stacks."""

    file_paths = discover_image_paths(input_path)
    stack_dir = os.path.dirname(file_paths[0]) if file_paths else os.path.abspath(input_path)
    z_override_mm = _find_z_positions_metadata(file_paths, stack_dir)
    z_positions_mm, parsed_z_values, z_source = infer_z_positions_mm(
        file_paths=file_paths,
        filename_z_unit=filename_z_unit,
        z_step_mm=z_step_mm,
        z_start_mm=z_start_mm,
        z_positions_override_mm=z_override_mm,
    )

    loaded_items = []
    shape_counts = {}
    skipped_files = []

    for path in file_paths:
        gray, color = _read_image(path)
        shape = tuple(int(v) for v in gray.shape)
        loaded_items.append((path, gray, color, shape))
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    if not loaded_items:
        raise ValueError("No readable images found in stack input")

    dominant_shape = max(shape_counts.items(), key=lambda item: item[1])[0]
    kept_paths = []
    gray_frames = []
    color_frames = []
    kept_parsed = []
    kept_z_positions = []
    for index, (path, gray, color, shape) in enumerate(loaded_items):
        if shape != dominant_shape:
            skipped_files.append(path)
            continue
        kept_paths.append(path)
        gray_frames.append(gray)
        color_frames.append(color)
        kept_parsed.append(parsed_z_values[index])
        kept_z_positions.append(float(z_positions_mm[index]))

    if len(gray_frames) < 2:
        raise ValueError(
            "Usable stack frames are fewer than 2 after filtering mismatched images. "
            "Dominant size: {}, skipped: {}".format(dominant_shape, len(skipped_files))
        )

    gray_stack = np.stack(gray_frames, axis=0).astype(np.float32)
    color_stack = np.stack(color_frames, axis=0).astype(np.uint8)

    if align and len(gray_stack) > 1:
        gray_stack, color_stack, offsets = align_stack(gray_stack, color_stack, reference_index=reference_index)
    else:
        offsets = [(0.0, 0.0) for _ in range(len(gray_stack))]

    source_dir = os.path.dirname(kept_paths[0]) if kept_paths else os.path.abspath(input_path)
    return LoadedStack(
        file_paths=list(kept_paths),
        z_positions_mm=np.asarray(kept_z_positions, dtype=np.float32),
        gray_stack=gray_stack,
        color_stack=color_stack,
        alignment_offsets_px=offsets,
        source_dir=source_dir,
        parsed_z_values=kept_parsed,
        z_source=z_source,
        skipped_files=skipped_files,
    )


def align_stack(
    gray_stack: np.ndarray,
    color_stack: np.ndarray,
    reference_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """Register each frame to a reference frame using translation-only alignment."""

    if gray_stack.ndim != 3:
        raise ValueError("gray_stack must have shape [n, h, w]")
    if color_stack.ndim != 4:
        raise ValueError("color_stack must have shape [n, h, w, 3]")

    frame_count = gray_stack.shape[0]
    reference_index = frame_count // 2 if reference_index is None else int(reference_index)
    reference_index = max(0, min(reference_index, frame_count - 1))
    reference = _normalize_for_alignment(gray_stack[reference_index])

    aligned_gray = np.empty_like(gray_stack)
    aligned_color = np.empty_like(color_stack)
    offsets = []

    for index in range(frame_count):
        moving = _normalize_for_alignment(gray_stack[index])
        dx, dy = _phase_correlation_shift(reference, moving)
        aligned_gray[index] = _warp_image(gray_stack[index], dx, dy)
        aligned_color[index] = _warp_image(color_stack[index], dx, dy)
        offsets.append((float(dx), float(dy)))
    return aligned_gray, aligned_color, offsets


def _sort_key_for_stack(file_path: str) -> Tuple[int, float, str]:
    parsed = parse_z_value_from_name(file_path)
    if parsed is None:
        return (1, 0.0, os.path.basename(file_path).lower())
    return (0, float(parsed), os.path.basename(file_path).lower())


def _has_z_token(file_name: str) -> bool:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    return bool(Z_TOKEN_PATTERN.search(stem))


def _is_derived_output_image(file_name: str) -> bool:
    name = os.path.basename(file_name).lower()
    return any(keyword in name for keyword in IGNORED_IMAGE_KEYWORDS)


def _resolve_stack_directory(input_path: str) -> str:
    if not os.path.isdir(input_path):
        return input_path
    frames_dir = os.path.join(input_path, "frames")
    if os.path.isdir(frames_dir):
        frame_files = [
            name for name in os.listdir(frames_dir)
            if os.path.isfile(os.path.join(frames_dir, name)) and name.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if frame_files:
            return frames_dir
    return input_path


def _find_z_positions_metadata(file_paths: Sequence[str], stack_dir: str) -> Optional[List[float]]:
    manifest_candidates = [
        os.path.join(stack_dir, "manifest.json"),
        os.path.join(os.path.dirname(stack_dir), "manifest.json"),
    ]
    for manifest_path in manifest_candidates:
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            continue
        z_positions = manifest.get("z_positions_mm")
        if z_positions is None:
            z_positions = manifest.get("capture_z_positions_mm")
        if not isinstance(z_positions, list) or len(z_positions) != len(file_paths):
            continue
        try:
            return [float(value) for value in z_positions]
        except Exception:
            continue
    return None


def _read_image(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a single file and return (gray_float32, rgb_uint8)."""

    if cv2 is not None:
        image = cv2_read_image_any_path(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Failed to read image: {}".format(file_path))
        if image.ndim == 2:
            gray = image.astype(np.float32)
            rgb = _gray_to_rgb8(gray)
            return gray, rgb
        if image.ndim == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]
            if image.dtype != np.uint8:
                image_rgb = _scale_to_uint8(image[:, :, ::-1])
            else:
                image_rgb = image[:, :, ::-1].copy()
            gray = _rgb_to_gray_float32(image_rgb)
            return gray, image_rgb

    if mpimg is None:
        raise ImportError("Neither OpenCV nor matplotlib.image is available for image loading")

    image = mpimg.imread(file_path)
    if image.ndim == 2:
        gray = _to_analysis_float32(image)
        rgb = _gray_to_rgb8(gray)
        return gray, rgb

    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype.kind == "f":
        image_rgb = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        image_rgb = np.clip(image, 0, 255).astype(np.uint8)
    gray = _rgb_to_gray_float32(image_rgb)
    return gray, image_rgb


def _to_analysis_float32(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype.kind == "f":
        finite_max = float(np.nanmax(arr)) if np.size(arr) else 1.0
        scale = 255.0 if finite_max <= 1.5 else 1.0
        arr = arr * scale
    return arr.astype(np.float32)


def _rgb_to_gray_float32(rgb: np.ndarray) -> np.ndarray:
    rgb_f = np.asarray(rgb, dtype=np.float32)
    return (
        0.299 * rgb_f[:, :, 0]
        + 0.587 * rgb_f[:, :, 1]
        + 0.114 * rgb_f[:, :, 2]
    ).astype(np.float32)


def _gray_to_rgb8(gray: np.ndarray) -> np.ndarray:
    gray_u8 = _scale_to_uint8(gray)
    return np.repeat(gray_u8[:, :, None], 3, axis=2)


def _scale_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo + 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _normalize_for_alignment(gray: np.ndarray) -> np.ndarray:
    gray_u8 = _scale_to_uint8(gray).astype(np.float32)
    gray_u8 -= float(np.mean(gray_u8))
    std = float(np.std(gray_u8))
    if std > 1e-6:
        gray_u8 /= std
    return gray_u8


def _phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> Tuple[float, float]:
    if cv2 is not None:
        try:
            shift, _response = cv2.phaseCorrelate(reference.astype(np.float32), moving.astype(np.float32))
            return float(shift[0]), float(shift[1])
        except Exception:
            pass

    reference_f = reference.astype(np.float64)
    moving_f = moving.astype(np.float64)
    cross_power = np.fft.fft2(reference_f) * np.conj(np.fft.fft2(moving_f))
    denom = np.abs(cross_power)
    denom[denom < 1e-12] = 1e-12
    corr = np.fft.ifft2(cross_power / denom).real
    y_peak, x_peak = np.unravel_index(np.argmax(corr), corr.shape)
    height, width = corr.shape
    if x_peak > width // 2:
        x_peak -= width
    if y_peak > height // 2:
        y_peak -= height
    return float(x_peak), float(y_peak)


def _warp_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if cv2 is not None:
        height, width = image.shape[:2]
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        interpolation = cv2.INTER_LINEAR if image.ndim == 2 else cv2.INTER_LINEAR
        border_value = 0 if image.ndim == 2 else (0, 0, 0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=interpolation,
            borderMode=cv2.BORDER_REFLECT101,
            borderValue=border_value,
        )

    # Fallback path uses integer shifts only.
    x_shift = int(round(dx))
    y_shift = int(round(dy))
    return np.roll(np.roll(image, y_shift, axis=0), x_shift, axis=1)
