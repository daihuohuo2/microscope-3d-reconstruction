# Z-stack 3D Reconstruction Guide

## 1. Overview

This toolkit adds a standalone offline reconstruction pipeline for microscope Z-stack data.

Main capabilities:

- Batch load `PNG/JPG/BMP/TIF/TIFF` image stacks
- Evaluate focus quality with Laplacian, Sobel, or combined metrics
- Reconstruct a per-pixel height map from the best-focused Z plane
- Export a 3D point cloud as `PLY` and `CSV`
- Measure point-to-point height difference, line profile, region average height, and region max height difference
- Save pseudo-color heatmaps and frame-wise focus curves
- Provide an automation adapter that can be connected to your current `DeviceController`

The code is organized as:

- `src/microscope_app/reconstruction/io_utils.py`: stack discovery, image loading, alignment
- `src/microscope_app/reconstruction/focus.py`: focus metrics
- `src/microscope_app/reconstruction/reconstruction.py`: depth reconstruction
- `src/microscope_app/reconstruction/pointcloud.py`: point cloud generation/export
- `src/microscope_app/reconstruction/measurement.py`: programmatic and interactive measurements
- `src/microscope_app/reconstruction/automation.py`: capture + reconstruct automation scaffold
- `src/microscope_app/reconstruction/cli.py`: command line implementation

## 2. Installation

Recommended packages:

```bash
pip install numpy opencv-python matplotlib open3d scipy
```

If you only need offline reconstruction and do not need the Open3D viewer, the core pipeline can still run without `open3d`.

## 3. Preparing the image sequence

### Recommended naming

If the filename stores only an index:

```text
img_z0.png
img_z1.png
img_z2.png
...
```

Use `--filename-z-unit index` together with `--z-step-um` or `--z-step-mm`.

If the filename stores the real Z height:

```text
img_z0.000mm.png
img_z0.010mm.png
img_z0.020mm.png
```

Use `--filename-z-unit mm`.

### Capture recommendations for measurement-grade results

- Keep the same field of view and exposure during the whole stack
- Use a stable and repeatable Z step
- Minimize vibration during scan
- Avoid large illumination changes between frames
- Use a calibration target to determine `pixels_per_mm`
- If your stage has backlash, always scan in one direction only
- If the sample is reflective, consider reducing glare before acquisition

## 4. Running the reconstruction

### Example A: file names contain only Z indices

```bash
python main.py reconstruct ^
  --input "D:\data\stack_01" ^
  --output "D:\data\stack_01_out" ^
  --filename-z-unit index ^
  --z-step-um 5 ^
  --pixels-per-mm 4200 ^
  --focus-method combined ^
  --window-size 9 ^
  --align
```

### Example B: file names contain real Z heights in millimeters

```bash
python main.py reconstruct ^
  --input "D:\data\stack_mm" ^
  --output "D:\data\stack_mm_out" ^
  --filename-z-unit mm ^
  --pixels-per-mm 4200 ^
  --focus-method laplacian ^
  --align
```

### Output files

The output directory will contain:

- `full_focus.png`: all-in-focus image
- `depth_map_mm.npy`: float32 depth map in mm
- `valid_mask.npy`: reliable measurement mask
- `depth_map_um16.tif`: 16-bit relative height TIFF in um
- `depth_heatmap.png`: pseudo-color height map
- `frame_focus_scores.csv`: per-frame focus score table
- `frame_focus_scores.png`: focus curve
- `surface_point_cloud.ply`: metric point cloud
- `surface_point_cloud.csv`: tabular point cloud
- `manifest.json`: metadata and paths

## 5. Parameter tuning

### Focus metric

- `laplacian`: good for fine local texture, often sharper but more noise-sensitive
- `sobel` or `tenengrad`: more stable on edge-rich samples
- `combined`: good default for mixed samples

### Window size

- Smaller `--window-size` keeps more local detail
- Larger `--window-size` suppresses noise and improves stability
- Typical range: `5` to `15`

### Alignment

- Use `--align` when the sample drifts laterally during the scan
- If the hardware is already stable and locked, you may disable alignment for speed

### Focus threshold percentile

- Lower values keep more pixels
- Higher values reject weak-focus areas more aggressively
- Typical range: `5` to `15`

### Depth smoothing

- `--median-filter-size`: removes isolated spikes
- `--gaussian-sigma`: reduces roughness
- Increase carefully; too much smoothing will remove real micro-topography

## 6. Height measurement

### Interactive mode

Open the saved reconstruction for measurement:

```bash
python main.py measure --manifest "D:\data\stack_01_out\manifest.json"
```

Shortcuts inside the measurement window:

- `1`: two-point height difference
- `2`: line profile
- `3`: rectangle region statistics
- `4`: polygon region statistics
- `c`: clear current drawing
- `s`: save collected measurements to a temporary JSON log

### Batch measurement examples

Two-point height difference:

```bash
python main.py measure ^
  --manifest "D:\data\stack_01_out\manifest.json" ^
  --point-pair 120 80 240 160
```

Line profile:

```bash
python main.py measure ^
  --manifest "D:\data\stack_01_out\manifest.json" ^
  --line 120 80 240 160
```

Rectangle region:

```bash
python main.py measure ^
  --manifest "D:\data\stack_01_out\manifest.json" ^
  --rect 100 60 260 220
```

## 7. Viewing the PLY in MeshLab / CloudCompare

### MeshLab

1. Open MeshLab
2. `File -> Import Mesh`
3. Select `surface_point_cloud.ply`
4. Use `Render -> Show Layer Dialog` to manage layers
5. Use `Filters -> Normals, Curvatures and Orientation` if you want surface normals
6. Use camera controls to inspect the 3D topography

### CloudCompare

1. Open CloudCompare
2. Drag `surface_point_cloud.ply` into the window
3. Confirm the import options
4. Use the left toolbar to switch to measurement tools
5. Use `Tools -> Distances` or the point picking tool for additional manual inspection

The exported point cloud coordinates are in millimeters, so the Z values can be used directly for metrology if your `pixels_per_mm` and Z step calibration are correct.

## 8. Full automation with your microscope API

The file `src/microscope_app/reconstruction/automation.py` includes `DeviceControllerStackAcquirer`, which can wrap the current `DeviceController`.

Typical workflow:

1. Home or reference the Z axis
2. Move to `z_start_mm`
3. Capture one frame per Z plane
4. Save the raw stack with Z encoded in file names
5. Load the raw stack into the offline reconstruction pipeline
6. Export depth map, heatmap, and point cloud

Example:

```python
from microscope_app.hardware.controller import DeviceController
from microscope_app.reconstruction.automation import (
    AcquisitionConfig,
    DeviceControllerStackAcquirer,
    run_acquisition_and_reconstruction,
)
from microscope_app.reconstruction.reconstruction import ReconstructionConfig

controller = DeviceController()
# controller.initialize_sdk()
# controller.open_camera(index)
# controller.connect_serial(port="COM3", baudrate=115200, timeout=0.5)

acquirer = DeviceControllerStackAcquirer(controller)
acq_cfg = AcquisitionConfig(
    z_start_mm=0.0,
    z_end_mm=0.3,
    z_step_mm=0.01,
    output_dir=r"D:\auto_stack_run",
    settle_time_s=0.15,
    move_feed=300,
)
recon_cfg = ReconstructionConfig(
    focus_method="combined",
    focus_window_size=9,
    focus_threshold_percentile=8.0,
)

artifacts = run_acquisition_and_reconstruction(
    acquirer=acquirer,
    acquisition_config=acq_cfg,
    reconstruction_config=recon_cfg,
    pixels_per_mm=4200.0,
)
print(artifacts)
```

To make this production-ready on your microscope, add:

- repeatable `pixels_per_mm` calibration
- repeatable Z step calibration
- backlash compensation
- flat-field or brightness correction
- vibration waiting time after each move
- rejected-frame retry logic
- hardware timestamp and scan log storage
- fixture-based referencing so the sample datum is stable

## 9. Important note about "industrial equipment effect"

This code is organized like an industrial pipeline, but true industrial metrology depends on hardware calibration and motion repeatability as much as software.

To reach repeatable measurement performance, you still need:

- calibrated stage motion
- known optical magnification
- controlled lighting
- temperature and vibration control
- periodic verification on a height standard

Without these, the software can reconstruct stable relative height maps, but absolute metrology accuracy will still be limited by the hardware chain.
