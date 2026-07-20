"""Acquisition scaffolding for turning microscope capture + reconstruction into one workflow."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .image_codec import cv2_write_image_any_path
from .io_utils import load_zstack_from_path
from .pointcloud import create_point_cloud_from_depth
from .reconstruction import ReconstructionConfig, reconstruct_from_stack
from .visualization import save_reconstruction_outputs


@dataclass
class AcquisitionConfig:
    """Configuration for hardware-driven stack capture."""

    z_start_mm: float
    z_end_mm: float
    z_step_mm: float
    output_dir: str
    file_prefix: str = "img"
    settle_time_s: float = 0.2
    move_feed: float = 300.0
    save_raw_frames: bool = True


def build_z_positions(z_start_mm: float, z_end_mm: float, z_step_mm: float) -> np.ndarray:
    """Build an inclusive Z scan trajectory."""

    if z_step_mm == 0:
        raise ValueError("z_step_mm must be non-zero")
    direction = 1.0 if z_end_mm >= z_start_mm else -1.0
    step = abs(float(z_step_mm)) * direction
    values = []
    current = float(z_start_mm)
    limit = float(z_end_mm)
    epsilon = abs(step) * 0.25
    while (direction > 0 and current <= limit + epsilon) or (direction < 0 and current >= limit - epsilon):
        values.append(current)
        current += step
    return np.asarray(values, dtype=np.float32)


class DeviceControllerStackAcquirer:
    """Adapter that connects the existing DeviceController to the standalone toolkit."""

    def __init__(self, device_controller):
        self.device_controller = device_controller

    def move_to_z(self, z_mm: float, feed: float) -> None:
        self.device_controller.move_z_absolute_wait(float(z_mm), feed=float(feed))

    def capture_gray_color(self):
        gray, color, _width, _height = self.device_controller.get_gray_color_frame()
        if gray is None or color is None:
            raise RuntimeError("Failed to capture frame from microscope device")
        return gray.astype(np.float32), color.astype(np.uint8)

    def acquire_stack(self, config: AcquisitionConfig) -> List[str]:
        """Drive the Z axis, capture every plane, and save to disk."""

        z_positions = build_z_positions(config.z_start_mm, config.z_end_mm, config.z_step_mm)
        raw_dir = os.path.join(config.output_dir, "raw_stack")
        os.makedirs(raw_dir, exist_ok=True)
        saved_paths = []

        for index, z_mm in enumerate(z_positions):
            self.move_to_z(z_mm, feed=config.move_feed)
            time.sleep(float(config.settle_time_s))
            gray, color = self.capture_gray_color()
            file_stem = "{}_z{:+0.6f}mm".format(config.file_prefix, z_mm)
            if config.save_raw_frames:
                file_path = os.path.join(raw_dir, file_stem + ".png")
                _write_capture_frame(file_path, gray, color)
                saved_paths.append(file_path)
        return saved_paths


def run_acquisition_and_reconstruction(
    acquirer: DeviceControllerStackAcquirer,
    acquisition_config: AcquisitionConfig,
    reconstruction_config: Optional[ReconstructionConfig],
    pixels_per_mm: float,
    z_exaggeration: float = 1.0,
) -> dict:
    """Capture a stack from hardware, reconstruct it, and save all outputs."""

    raw_paths = acquirer.acquire_stack(acquisition_config)
    if not raw_paths:
        raise RuntimeError("No raw frames were captured")

    stack = load_zstack_from_path(
        input_path=os.path.join(acquisition_config.output_dir, "raw_stack"),
        filename_z_unit="mm",
        align=True,
    )
    result = reconstruct_from_stack(stack, config=reconstruction_config or ReconstructionConfig())
    point_cloud = create_point_cloud_from_depth(
        depth_map_mm=result.depth_map_mm,
        texture_rgb=result.full_focus_rgb,
        pixels_per_mm=pixels_per_mm,
        valid_mask=result.valid_mask,
        z_exaggeration=z_exaggeration,
    )
    return save_reconstruction_outputs(
        result=result,
        output_dir=acquisition_config.output_dir,
        pixels_per_mm=pixels_per_mm,
        point_cloud=point_cloud,
        save_point_cloud_file=True,
    )


def _write_capture_frame(output_path: str, gray: np.ndarray, color: np.ndarray) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if cv2 is not None:
        if color is not None and color.ndim == 3:
            if cv2_write_image_any_path(output_path, color[:, :, ::-1]):
                return
        if float(np.nanmax(gray)) > 255.0:
            if cv2_write_image_any_path(output_path, np.clip(gray, 0, 65535).astype(np.uint16)):
                return
        else:
            if cv2_write_image_any_path(output_path, np.clip(gray, 0, 255).astype(np.uint8)):
                return

    import matplotlib.pyplot as plt

    if color is not None and color.ndim == 3:
        plt.imsave(output_path, np.clip(color, 0, 255).astype(np.uint8))
    else:
        plt.imsave(output_path, gray, cmap="gray")
