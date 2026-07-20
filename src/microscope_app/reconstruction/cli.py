"""Command line entry points for standalone Z-stack reconstruction."""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np

try:
    import matplotlib.image as mpimg
except ImportError:  # pragma: no cover
    mpimg = None

from .io_utils import load_zstack_from_path
from .measurement import HeightAnalyzer, InteractiveMeasurementTool
from .pointcloud import create_point_cloud_from_depth
from .reconstruction import ReconstructionConfig, reconstruct_from_stack
from .visualization import save_reconstruction_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Microscope Z-stack depth reconstruction and measurement toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reconstruct_parser = subparsers.add_parser("reconstruct", help="Reconstruct depth map and point cloud from Z-stack")
    reconstruct_parser.add_argument("--input", required=True, help="Input directory, wildcard, or single image path")
    reconstruct_parser.add_argument("--output", required=True, help="Output directory")
    reconstruct_parser.add_argument(
        "--filename-z-unit",
        default="auto",
        choices=["auto", "index", "um", "mm"],
        help="How to interpret the number encoded in file names",
    )
    reconstruct_parser.add_argument("--z-step-mm", type=float, default=None, help="Physical Z step in mm when filename_z_unit=index")
    reconstruct_parser.add_argument("--z-step-um", type=float, default=None, help="Physical Z step in um when filename_z_unit=index")
    reconstruct_parser.add_argument("--z-start-mm", type=float, default=0.0, help="Starting Z position in mm")
    reconstruct_parser.add_argument("--pixels-per-mm", type=float, required=True, help="Lateral calibration in pixels/mm")
    reconstruct_parser.add_argument("--focus-method", default="combined", choices=["laplacian", "sobel", "tenengrad", "combined"])
    reconstruct_parser.add_argument("--window-size", type=int, default=9, help="Focus metric smoothing window size")
    reconstruct_parser.add_argument("--sobel-ksize", type=int, default=3, help="Sobel kernel size")
    reconstruct_parser.add_argument("--laplacian-weight", type=float, default=0.5, help="Weight for laplacian in combined metric")
    reconstruct_parser.add_argument("--focus-threshold-percentile", type=float, default=8.0, help="Discard the lowest focus-score percentile")
    reconstruct_parser.add_argument("--median-filter-size", type=int, default=5, help="Median filter kernel size for depth smoothing")
    reconstruct_parser.add_argument("--gaussian-sigma", type=float, default=0.8, help="Gaussian sigma for final depth smoothing")
    reconstruct_parser.add_argument(
        "--smoothing-strength",
        default="light",
        choices=["off", "light", "medium", "strong"],
        help="How aggressively the reconstructed depth map is smoothed",
    )
    reconstruct_parser.add_argument("--align", action="store_true", help="Align the stack before reconstruction")
    reconstruct_parser.add_argument("--reference-index", type=int, default=None, help="Reference frame for alignment")
    reconstruct_parser.add_argument("--no-point-cloud", action="store_true", help="Skip PLY/CSV point cloud export")
    reconstruct_parser.add_argument("--z-exaggeration", type=float, default=1.0, help="Vertical scaling factor for point cloud export")
    reconstruct_parser.add_argument("--heatmap-cmap", default="turbo", help="Matplotlib colormap used for the heatmap image")

    measure_parser = subparsers.add_parser("measure", help="Measure heights from a saved reconstruction output")
    measure_parser.add_argument("--manifest", default=None, help="Manifest JSON created by the reconstruct command")
    measure_parser.add_argument("--depth-npy", default=None, help="Path to depth_map_mm.npy")
    measure_parser.add_argument("--valid-mask", default=None, help="Optional valid_mask.npy path")
    measure_parser.add_argument("--preview", default=None, help="Optional preview image path for the measurement UI")
    measure_parser.add_argument("--pixels-per-mm", type=float, default=None, help="Required when manifest is not used")
    measure_parser.add_argument("--point-pair", nargs=4, type=float, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    measure_parser.add_argument("--line", nargs=4, type=float, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    measure_parser.add_argument("--rect", nargs=4, type=float, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    measure_parser.add_argument("--interactive", action="store_true", help="Open the interactive measurement UI")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "reconstruct":
        run_reconstruct_command(args)
        return 0
    if args.command == "measure":
        run_measure_command(args)
        return 0
    parser.error("Unsupported command")
    return 1


def run_reconstruct_command(args) -> None:
    z_step_mm = args.z_step_mm
    if z_step_mm is None and args.z_step_um is not None:
        z_step_mm = float(args.z_step_um) / 1000.0

    stack = load_zstack_from_path(
        input_path=args.input,
        filename_z_unit=args.filename_z_unit,
        z_step_mm=z_step_mm,
        z_start_mm=args.z_start_mm,
        align=args.align,
        reference_index=args.reference_index,
    )

    config = ReconstructionConfig(
        focus_method=args.focus_method,
        focus_window_size=args.window_size,
        sobel_ksize=args.sobel_ksize,
        laplacian_weight=args.laplacian_weight,
        focus_threshold_percentile=args.focus_threshold_percentile,
        median_filter_size=args.median_filter_size,
        gaussian_sigma=args.gaussian_sigma,
        smoothing_strength=args.smoothing_strength,
    )
    result = reconstruct_from_stack(stack, config=config)

    point_cloud = None
    if not args.no_point_cloud:
        point_cloud = create_point_cloud_from_depth(
            depth_map_mm=result.depth_map_mm,
            texture_rgb=result.full_focus_rgb,
            pixels_per_mm=args.pixels_per_mm,
            valid_mask=result.valid_mask,
            z_exaggeration=args.z_exaggeration,
        )

    saved_paths = save_reconstruction_outputs(
        result=result,
        output_dir=os.path.abspath(args.output),
        pixels_per_mm=args.pixels_per_mm,
        point_cloud=point_cloud,
        save_point_cloud_file=not args.no_point_cloud,
        heatmap_cmap=args.heatmap_cmap,
    )
    print(json.dumps(saved_paths, ensure_ascii=False, indent=2))
    _print_reconstruction_summary(result, args.pixels_per_mm, point_cloud)


def run_measure_command(args) -> None:
    depth_path, valid_mask_path, preview_path, pixels_per_mm = _resolve_measurement_inputs(args)
    depth_map_mm = np.load(depth_path).astype(np.float32)
    valid_mask = np.load(valid_mask_path).astype(bool) if valid_mask_path else np.isfinite(depth_map_mm)
    preview_image = None
    if preview_path:
        preview_image = _read_preview_image(preview_path)

    analyzer = HeightAnalyzer(depth_map_mm=depth_map_mm, pixels_per_mm=pixels_per_mm, valid_mask=valid_mask)

    if args.point_pair is not None:
        measurement = analyzer.measure_point_pair(
            (args.point_pair[0], args.point_pair[1]),
            (args.point_pair[2], args.point_pair[3]),
        )
        print(json.dumps(_point_measurement_to_dict(measurement), ensure_ascii=False, indent=2))

    if args.line is not None:
        profile = analyzer.measure_line_profile(
            (args.line[0], args.line[1]),
            (args.line[2], args.line[3]),
        )
        print(
            json.dumps(
                {
                    "start": vars(profile.start),
                    "end": vars(profile.end),
                    "delta_height_mm": float(profile.delta_height_mm),
                    "max_delta_mm": float(profile.max_delta_mm),
                    "distance_mm": [float(v) for v in profile.distance_mm],
                    "height_mm": [float(v) for v in profile.height_mm],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if args.rect is not None:
        region = analyzer.measure_rectangle(
            (args.rect[0], args.rect[1]),
            (args.rect[2], args.rect[3]),
        )
        print(json.dumps(vars(region), ensure_ascii=False, indent=2))

    if args.interactive or (args.point_pair is None and args.line is None and args.rect is None):
        tool = InteractiveMeasurementTool(analyzer=analyzer, preview_image=preview_image)
        tool.show()


def _resolve_measurement_inputs(args):
    if args.manifest:
        manifest_path = os.path.abspath(args.manifest)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        base_dir = os.path.dirname(manifest_path)
        artifacts = manifest.get("artifacts", {})
        depth_path = os.path.join(
            base_dir,
            artifacts.get("raw_depth_npy", artifacts.get("depth_npy", "depth_map_mm.npy")),
        )
        valid_mask_path = os.path.join(base_dir, artifacts.get("valid_mask_npy", "valid_mask.npy"))
        preview_path = args.preview or os.path.join(base_dir, artifacts.get("full_focus_png", "full_focus.png"))
        pixels_per_mm = float(manifest["pixels_per_mm"])
        return depth_path, valid_mask_path, preview_path, pixels_per_mm

    if not args.depth_npy or args.pixels_per_mm is None:
        raise ValueError("Without --manifest you must provide --depth-npy and --pixels-per-mm")
    return (
        os.path.abspath(args.depth_npy),
        os.path.abspath(args.valid_mask) if args.valid_mask else None,
        os.path.abspath(args.preview) if args.preview else None,
        float(args.pixels_per_mm),
    )


def _read_preview_image(preview_path: str):
    if mpimg is None:
        return None
    image = mpimg.imread(preview_path)
    if image.dtype.kind == "f" and np.nanmax(image) <= 1.5:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _point_measurement_to_dict(measurement):
    return {
        "point_a": vars(measurement.point_a),
        "point_b": vars(measurement.point_b),
        "delta_height_mm": float(measurement.delta_height_mm),
        "planar_distance_mm": float(measurement.planar_distance_mm),
        "three_d_distance_mm": float(measurement.three_d_distance_mm),
        "slope_deg": float(measurement.slope_deg),
    }


def _print_reconstruction_summary(result, pixels_per_mm: float, point_cloud) -> None:
    valid_ratio = float(np.mean(result.valid_mask)) * 100.0
    z_min = float(np.nanmin(result.depth_map_mm[result.valid_mask])) if np.any(result.valid_mask) else float("nan")
    z_max = float(np.nanmax(result.depth_map_mm[result.valid_mask])) if np.any(result.valid_mask) else float("nan")
    print("frames: {}".format(len(result.file_paths)))
    print("image size: {} x {}".format(result.depth_map_mm.shape[1], result.depth_map_mm.shape[0]))
    print("valid coverage: {:.2f}%".format(valid_ratio))
    print("z range: {:.6f} mm .. {:.6f} mm".format(z_min, z_max))
    print("pixels_per_mm: {:.6f}".format(float(pixels_per_mm)))
    if point_cloud is not None:
        print("point count: {}".format(len(point_cloud.points_mm)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
