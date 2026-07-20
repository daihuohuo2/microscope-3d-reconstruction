"""Depth-map to point-cloud conversion and export helpers."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None


@dataclass
class PointCloudData:
    """Flattened point cloud representation derived from a height map."""

    points_mm: np.ndarray
    colors_rgb: np.ndarray
    intensity: np.ndarray
    pixel_x: np.ndarray
    pixel_y: np.ndarray


def create_point_cloud_from_depth(
    depth_map_mm: np.ndarray,
    texture_rgb: np.ndarray,
    pixels_per_mm: float,
    valid_mask: Optional[np.ndarray] = None,
    z_exaggeration: float = 1.0,
    zero_reference: str = "min",
) -> PointCloudData:
    """Convert a 2.5D height map into a metric point cloud."""

    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be positive for metric point cloud generation")

    depth = np.asarray(depth_map_mm, dtype=np.float32)
    texture = np.asarray(texture_rgb, dtype=np.uint8)
    if depth.ndim != 2:
        raise ValueError("depth_map_mm must be 2D")
    if texture.ndim != 3 or texture.shape[:2] != depth.shape:
        raise ValueError("texture_rgb must have shape [h, w, 3] and match depth dimensions")

    if valid_mask is None:
        valid_mask = np.isfinite(depth)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(depth)

    ys, xs = np.where(valid_mask)
    if len(xs) == 0:
        return PointCloudData(
            points_mm=np.zeros((0, 3), dtype=np.float32),
            colors_rgb=np.zeros((0, 3), dtype=np.uint8),
            intensity=np.zeros((0,), dtype=np.float32),
            pixel_x=np.zeros((0,), dtype=np.int32),
            pixel_y=np.zeros((0,), dtype=np.int32),
        )

    height, width = depth.shape
    x_mm = (xs.astype(np.float32) - (width - 1) * 0.5) / float(pixels_per_mm)
    y_mm = (ys.astype(np.float32) - (height - 1) * 0.5) / float(pixels_per_mm)
    z_mm = depth[ys, xs].astype(np.float32)

    if zero_reference == "min":
        z_mm = z_mm - float(np.nanmin(z_mm))
    elif zero_reference == "mean":
        z_mm = z_mm - float(np.nanmean(z_mm))
    elif zero_reference != "absolute":
        raise ValueError("Unsupported zero_reference: {}".format(zero_reference))

    z_mm = z_mm * float(z_exaggeration)
    colors_rgb = texture[ys, xs].astype(np.uint8)
    intensity = (
        0.299 * colors_rgb[:, 0].astype(np.float32)
        + 0.587 * colors_rgb[:, 1].astype(np.float32)
        + 0.114 * colors_rgb[:, 2].astype(np.float32)
    ).astype(np.float32)
    points = np.column_stack([x_mm, y_mm, z_mm]).astype(np.float32)
    return PointCloudData(
        points_mm=points,
        colors_rgb=colors_rgb,
        intensity=intensity,
        pixel_x=xs.astype(np.int32),
        pixel_y=ys.astype(np.int32),
    )


def save_point_cloud(output_path: str, point_cloud: PointCloudData) -> str:
    """Save a point cloud as .ply or .csv based on the extension."""

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".csv":
        _save_point_cloud_csv(output_path, point_cloud)
        return output_path
    if ext != ".ply":
        raise ValueError("Only .ply and .csv exports are supported, got: {}".format(ext))
    _save_point_cloud_ply(output_path, point_cloud)
    return output_path


def to_open3d_point_cloud(point_cloud: PointCloudData):
    """Convert to an Open3D PointCloud if Open3D is available."""

    if o3d is None:
        raise ImportError("Open3D is not installed; cannot create an Open3D point cloud")
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud.points_mm.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(point_cloud.colors_rgb.astype(np.float64) / 255.0)
    return pc


def _save_point_cloud_csv(output_path: str, point_cloud: PointCloudData) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_mm", "y_mm", "z_mm", "intensity", "pixel_x", "pixel_y", "red", "green", "blue"])
        for index in range(len(point_cloud.points_mm)):
            x_mm, y_mm, z_mm = point_cloud.points_mm[index]
            r, g, b = point_cloud.colors_rgb[index]
            writer.writerow(
                [
                    float(x_mm),
                    float(y_mm),
                    float(z_mm),
                    float(point_cloud.intensity[index]),
                    int(point_cloud.pixel_x[index]),
                    int(point_cloud.pixel_y[index]),
                    int(r),
                    int(g),
                    int(b),
                ]
            )


def _save_point_cloud_ply(output_path: str, point_cloud: PointCloudData) -> None:
    n_points = len(point_cloud.points_mm)
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("intensity", "<f4"),
            ("pixel_x", "<i4"),
            ("pixel_y", "<i4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    payload = np.empty(n_points, dtype=dtype)
    if n_points:
        payload["x"] = point_cloud.points_mm[:, 0]
        payload["y"] = point_cloud.points_mm[:, 1]
        payload["z"] = point_cloud.points_mm[:, 2]
        payload["intensity"] = point_cloud.intensity
        payload["pixel_x"] = point_cloud.pixel_x
        payload["pixel_y"] = point_cloud.pixel_y
        payload["red"] = point_cloud.colors_rgb[:, 0]
        payload["green"] = point_cloud.colors_rgb[:, 1]
        payload["blue"] = point_cloud.colors_rgb[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment generated_by=zstack_3d\n"
        "comment coordinates_unit=mm\n"
        "element vertex {}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float intensity\n"
        "property int pixel_x\n"
        "property int pixel_y\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).format(n_points)

    with open(output_path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(payload.tobytes())
