"""Height measurement utilities on reconstructed microscope surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector, RectangleSelector
import matplotlib.pyplot as plt


@dataclass
class PointSample:
    pixel_x: float
    pixel_y: float
    x_mm: float
    y_mm: float
    z_mm: float


@dataclass
class PointToPointMeasurement:
    point_a: PointSample
    point_b: PointSample
    delta_height_mm: float
    planar_distance_mm: float
    three_d_distance_mm: float
    slope_deg: float


@dataclass
class RegionMeasurement:
    shape: str
    area_pixels: int
    area_mm2: float
    mean_height_mm: float
    min_height_mm: float
    max_height_mm: float
    max_delta_mm: float
    std_height_mm: float


@dataclass
class LineProfileMeasurement:
    start: PointSample
    end: PointSample
    distance_mm: np.ndarray
    height_mm: np.ndarray
    delta_height_mm: float
    max_delta_mm: float


class HeightAnalyzer:
    """Programmatic measurements on a 2.5D reconstructed surface."""

    def __init__(self, depth_map_mm: np.ndarray, pixels_per_mm: float, valid_mask: Optional[np.ndarray] = None):
        self.depth_map_mm = np.asarray(depth_map_mm, dtype=np.float32)
        self.pixels_per_mm = float(pixels_per_mm)
        if self.pixels_per_mm <= 0:
            raise ValueError("pixels_per_mm must be positive")
        if valid_mask is None:
            self.valid_mask = np.isfinite(self.depth_map_mm)
        else:
            self.valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(self.depth_map_mm)

    def sample_point(self, pixel_x: float, pixel_y: float) -> PointSample:
        """Bilinearly sample the surface height at image-space coordinates."""

        x = float(np.clip(pixel_x, 0, self.depth_map_mm.shape[1] - 1))
        y = float(np.clip(pixel_y, 0, self.depth_map_mm.shape[0] - 1))
        z_mm = float(self._bilinear_sample(self.depth_map_mm, x, y))
        x_mm = (x - (self.depth_map_mm.shape[1] - 1) * 0.5) / self.pixels_per_mm
        y_mm = (y - (self.depth_map_mm.shape[0] - 1) * 0.5) / self.pixels_per_mm
        return PointSample(pixel_x=x, pixel_y=y, x_mm=x_mm, y_mm=y_mm, z_mm=z_mm)

    def measure_point_pair(self, point_a: Tuple[float, float], point_b: Tuple[float, float]) -> PointToPointMeasurement:
        """Measure height difference and distances between two selected points."""

        sample_a = self.sample_point(point_a[0], point_a[1])
        sample_b = self.sample_point(point_b[0], point_b[1])
        delta_height = float(sample_b.z_mm - sample_a.z_mm)
        planar_distance = float(np.hypot(sample_b.x_mm - sample_a.x_mm, sample_b.y_mm - sample_a.y_mm))
        three_d_distance = float(np.sqrt(planar_distance ** 2 + delta_height ** 2))
        slope_deg = float(np.degrees(np.arctan2(delta_height, planar_distance + 1e-12)))
        return PointToPointMeasurement(
            point_a=sample_a,
            point_b=sample_b,
            delta_height_mm=delta_height,
            planar_distance_mm=planar_distance,
            three_d_distance_mm=three_d_distance,
            slope_deg=slope_deg,
        )

    def measure_line_profile(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_samples: int = 256,
    ) -> LineProfileMeasurement:
        """Sample a height profile along a line segment."""

        start_sample = self.sample_point(start[0], start[1])
        end_sample = self.sample_point(end[0], end[1])
        ts = np.linspace(0.0, 1.0, int(max(2, num_samples)), dtype=np.float32)
        xs = start_sample.pixel_x + (end_sample.pixel_x - start_sample.pixel_x) * ts
        ys = start_sample.pixel_y + (end_sample.pixel_y - start_sample.pixel_y) * ts
        heights = np.asarray([self._bilinear_sample(self.depth_map_mm, x, y) for x, y in zip(xs, ys)], dtype=np.float32)
        distances_px = np.hypot(xs - xs[0], ys - ys[0])
        distances_mm = distances_px / self.pixels_per_mm
        return LineProfileMeasurement(
            start=start_sample,
            end=end_sample,
            distance_mm=distances_mm,
            height_mm=heights,
            delta_height_mm=float(heights[-1] - heights[0]),
            max_delta_mm=float(np.nanmax(heights) - np.nanmin(heights)),
        )

    def measure_rectangle(self, point_a: Tuple[float, float], point_b: Tuple[float, float]) -> RegionMeasurement:
        """Measure region statistics inside an axis-aligned rectangle."""

        x0, y0 = point_a
        x1, y1 = point_b
        x_min, x_max = sorted([int(round(x0)), int(round(x1))])
        y_min, y_max = sorted([int(round(y0)), int(round(y1))])
        x_min = max(0, min(x_min, self.depth_map_mm.shape[1] - 1))
        x_max = max(0, min(x_max, self.depth_map_mm.shape[1] - 1))
        y_min = max(0, min(y_min, self.depth_map_mm.shape[0] - 1))
        y_max = max(0, min(y_max, self.depth_map_mm.shape[0] - 1))
        mask = np.zeros_like(self.valid_mask, dtype=bool)
        mask[y_min : y_max + 1, x_min : x_max + 1] = True
        return self._measure_mask(mask, shape="rectangle")

    def measure_polygon(self, vertices: Sequence[Tuple[float, float]]) -> RegionMeasurement:
        """Measure region statistics inside an arbitrary polygon."""

        yy, xx = np.mgrid[0 : self.depth_map_mm.shape[0], 0 : self.depth_map_mm.shape[1]]
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        path = Path(vertices)
        mask = path.contains_points(coords).reshape(self.depth_map_mm.shape)
        return self._measure_mask(mask, shape="polygon")

    def _measure_mask(self, mask: np.ndarray, shape: str) -> RegionMeasurement:
        mask = np.asarray(mask, dtype=bool) & self.valid_mask & np.isfinite(self.depth_map_mm)
        values = self.depth_map_mm[mask]
        if values.size == 0:
            raise ValueError("Selected region does not contain valid height samples")
        area_pixels = int(values.size)
        area_mm2 = float(area_pixels / (self.pixels_per_mm ** 2))
        return RegionMeasurement(
            shape=shape,
            area_pixels=area_pixels,
            area_mm2=area_mm2,
            mean_height_mm=float(np.nanmean(values)),
            min_height_mm=float(np.nanmin(values)),
            max_height_mm=float(np.nanmax(values)),
            max_delta_mm=float(np.nanmax(values) - np.nanmin(values)),
            std_height_mm=float(np.nanstd(values)),
        )

    @staticmethod
    def _bilinear_sample(image: np.ndarray, x: float, y: float) -> float:
        height, width = image.shape
        x = float(np.clip(x, 0, width - 1))
        y = float(np.clip(y, 0, height - 1))
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, width - 1)
        y1 = min(y0 + 1, height - 1)
        fx = x - x0
        fy = y - y0
        q00 = float(image[y0, x0])
        q10 = float(image[y0, x1])
        q01 = float(image[y1, x0])
        q11 = float(image[y1, x1])
        top = q00 * (1.0 - fx) + q10 * fx
        bottom = q01 * (1.0 - fx) + q11 * fx
        return top * (1.0 - fy) + bottom * fy


class InteractiveMeasurementTool:
    """Interactive measurement UI using matplotlib."""

    def __init__(
        self,
        analyzer: HeightAnalyzer,
        preview_image: Optional[np.ndarray] = None,
        window_title: str = "Z-stack Height Measurement",
    ):
        self.analyzer = analyzer
        self.preview_image = preview_image
        self.window_title = window_title
        self.mode = "line"
        self.pending_points = []
        self.measurements = []

        self.fig = None
        self.ax_image = None
        self.ax_profile = None
        self._rect_selector = None
        self._poly_selector = None
        self._image_artist = None
        self._mode_text = None
        self._result_text = None

    def show(self) -> None:
        """Open the interactive viewer."""

        self.fig, (self.ax_image, self.ax_profile) = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
        self.fig.canvas.manager.set_window_title(self.window_title)
        self._draw_background()
        self._mode_text = self.fig.text(0.02, 0.97, "", fontsize=11, va="top")
        self._result_text = self.fig.text(0.02, 0.02, "", fontsize=10, va="bottom", family="monospace")
        self._update_mode_text()

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._rect_selector = RectangleSelector(
            self.ax_image,
            self._on_rectangle_select,
            useblit=False,
            button=[1],
            interactive=True,
        )
        self._rect_selector.set_active(False)

        self._poly_selector = PolygonSelector(self.ax_image, self._on_polygon_select, useblit=False)
        self._poly_selector.set_active(False)

        plt.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
        plt.show()

    def _draw_background(self) -> None:
        self.ax_image.clear()
        if self.preview_image is not None:
            if self.preview_image.ndim == 2:
                self._image_artist = self.ax_image.imshow(self.preview_image, cmap="gray")
            else:
                self._image_artist = self.ax_image.imshow(self.preview_image)
        else:
            masked = np.ma.array(self.analyzer.depth_map_mm, mask=~self.analyzer.valid_mask)
            self._image_artist = self.ax_image.imshow(masked, cmap="turbo")
        self.ax_image.set_title("Surface View")
        self.ax_image.set_xlabel("X (pixel)")
        self.ax_image.set_ylabel("Y (pixel)")

        self.ax_profile.clear()
        self.ax_profile.set_title("Line Profile / Measurement Output")
        self.ax_profile.set_xlabel("Distance (mm)")
        self.ax_profile.set_ylabel("Height (mm)")
        self.ax_profile.grid(True, alpha=0.3)

    def _on_key(self, event) -> None:
        if event.key == "1":
            self.mode = "points"
        elif event.key == "2":
            self.mode = "line"
        elif event.key == "3":
            self.mode = "rectangle"
        elif event.key == "4":
            self.mode = "polygon"
        elif event.key == "c":
            self.pending_points = []
            self._draw_background()
        elif event.key == "s":
            self._save_measurements()
        elif event.key == "h":
            self._set_result_text(
                "Keys: 1=point pair, 2=line profile, 3=rectangle, 4=polygon, c=clear, s=save json"
            )
        else:
            return
        self._apply_mode()
        self._update_mode_text()
        self.fig.canvas.draw_idle()

    def _apply_mode(self) -> None:
        self.pending_points = []
        self._draw_background()
        if self._rect_selector is not None:
            self._rect_selector.set_active(self.mode == "rectangle")
        if self._poly_selector is not None:
            self._poly_selector.set_active(self.mode == "polygon")

    def _update_mode_text(self) -> None:
        if self._mode_text is not None:
            self._mode_text.set_text(
                "Mode: {} | 1=points 2=line 3=rect 4=poly c=clear s=save".format(self.mode)
            )

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax_image or event.xdata is None or event.ydata is None:
            return
        if self.mode not in ("points", "line"):
            return

        self.pending_points.append((float(event.xdata), float(event.ydata)))
        self.ax_image.plot(event.xdata, event.ydata, "wo", markersize=5, markeredgecolor="k")
        if len(self.pending_points) == 2:
            p0, p1 = self.pending_points
            self.ax_image.plot([p0[0], p1[0]], [p0[1], p1[1]], "w-", linewidth=1.6)
            if self.mode == "points":
                measurement = self.analyzer.measure_point_pair(p0, p1)
                self.measurements.append(asdict(measurement))
                self._set_result_text(_format_point_measurement(measurement))
            else:
                pair_measurement = self.analyzer.measure_point_pair(p0, p1)
                profile = self.analyzer.measure_line_profile(p0, p1)
                self.measurements.append(
                    {
                        "pair_measurement": asdict(pair_measurement),
                        "line_profile": {
                            "start": asdict(profile.start),
                            "end": asdict(profile.end),
                            "distance_mm": [float(v) for v in profile.distance_mm],
                            "height_mm": [float(v) for v in profile.height_mm],
                            "delta_height_mm": float(profile.delta_height_mm),
                            "max_delta_mm": float(profile.max_delta_mm),
                        },
                    }
                )
                self.ax_profile.clear()
                self.ax_profile.plot(profile.distance_mm, profile.height_mm, color="#1565c0", linewidth=1.8)
                self.ax_profile.set_title("Line Profile")
                self.ax_profile.set_xlabel("Distance (mm)")
                self.ax_profile.set_ylabel("Height (mm)")
                self.ax_profile.grid(True, alpha=0.3)
                self._set_result_text(
                    "{}\nline delta_h={:.6f} mm | line max_delta={:.6f} mm".format(
                        _format_point_measurement(pair_measurement),
                        profile.delta_height_mm,
                        profile.max_delta_mm,
                    )
                )
            self.pending_points = []
        self.fig.canvas.draw_idle()

    def _on_rectangle_select(self, eclick, erelease) -> None:
        if self.mode != "rectangle":
            return
        p0 = (float(eclick.xdata), float(eclick.ydata))
        p1 = (float(erelease.xdata), float(erelease.ydata))
        measurement = self.analyzer.measure_rectangle(p0, p1)
        self.measurements.append(asdict(measurement))
        self._set_result_text(_format_region_measurement(measurement))
        self.fig.canvas.draw_idle()

    def _on_polygon_select(self, vertices: Sequence[Tuple[float, float]]) -> None:
        if self.mode != "polygon":
            return
        measurement = self.analyzer.measure_polygon(vertices)
        self.measurements.append(asdict(measurement))
        self._set_result_text(_format_region_measurement(measurement))
        self.fig.canvas.draw_idle()

    def _set_result_text(self, text: str) -> None:
        if self._result_text is not None:
            self._result_text.set_text(text)

    def _save_measurements(self) -> None:
        if not self.measurements:
            self._set_result_text("No measurements collected yet")
            return
        import json
        import tempfile

        output_path = tempfile.mktemp(prefix="zstack_measurements_", suffix=".json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self.measurements, handle, ensure_ascii=False, indent=2)
        self._set_result_text("Saved measurement log:\n{}".format(output_path))


def _format_point_measurement(measurement: PointToPointMeasurement) -> str:
    return (
        "point A: ({:.2f}px, {:.2f}px) z={:.6f} mm\n"
        "point B: ({:.2f}px, {:.2f}px) z={:.6f} mm\n"
        "delta_h={:.6f} mm | planar={:.6f} mm | 3d={:.6f} mm | slope={:.3f} deg"
    ).format(
        measurement.point_a.pixel_x,
        measurement.point_a.pixel_y,
        measurement.point_a.z_mm,
        measurement.point_b.pixel_x,
        measurement.point_b.pixel_y,
        measurement.point_b.z_mm,
        measurement.delta_height_mm,
        measurement.planar_distance_mm,
        measurement.three_d_distance_mm,
        measurement.slope_deg,
    )


def _format_region_measurement(measurement: RegionMeasurement) -> str:
    return (
        "{} area={} px ({:.6f} mm^2)\n"
        "mean={:.6f} mm | min={:.6f} mm | max={:.6f} mm | max_delta={:.6f} mm | std={:.6f} mm"
    ).format(
        measurement.shape,
        measurement.area_pixels,
        measurement.area_mm2,
        measurement.mean_height_mm,
        measurement.min_height_mm,
        measurement.max_height_mm,
        measurement.max_delta_mm,
        measurement.std_height_mm,
    )
