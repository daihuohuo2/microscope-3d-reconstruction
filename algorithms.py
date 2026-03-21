import csv
import os


def get_mpl_font():
    try:
        from matplotlib.font_manager import FontProperties

        for family in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Arial Unicode MS"]:
            try:
                return FontProperties(family=family)
            except Exception:
                continue
        return FontProperties()
    except Exception:
        return None


def compute_sharpness_score(gray):
    import numpy as np

    gray = gray.astype(np.float32)
    dx = gray[:, 1:] - gray[:, :-1]
    dy = gray[1:, :] - gray[:-1, :]
    tenengrad = float(np.mean(dx ** 2) + np.mean(dy ** 2))
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    lap_var = float(np.var(lap))
    return 0.5 * tenengrad + 0.5 * lap_var


def compute_laplacian_sharpness_map(gray):
    import numpy as np

    gray = gray.astype(np.float32)
    h, w = gray.shape
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    sharp = np.zeros((h, w), dtype=np.float32)
    sharp[1:-1, 1:-1] = lap * lap
    return sharp


def phase_correlation_shift(frame1, frame2):
    import numpy as np

    try:
        f1 = frame1.astype(np.float64) - float(np.mean(frame1))
        f2 = frame2.astype(np.float64) - float(np.mean(frame2))
        f1_fft = np.fft.fft2(f1)
        f2_fft = np.fft.fft2(f2)
        cross = f1_fft * np.conj(f2_fft)
        eps = np.abs(cross).max() * 1e-10 + 1e-30
        cross = cross / (np.abs(cross) + eps)
        corr = np.fft.ifft2(cross).real
        index = np.unravel_index(np.argmax(corr), corr.shape)
        dy, dx = int(index[0]), int(index[1])
        h, w = frame1.shape[:2]
        if dy > h // 2:
            dy -= h
        if dx > w // 2:
            dx -= w
        return float(dx), float(dy)
    except Exception:
        return 0.0, 0.0


def build_best_focus_maps(frames_gray, z_list):
    import numpy as np

    if not frames_gray:
        return None, None, None
    h, w = frames_gray[0].shape
    best_sharp = np.zeros((h, w), dtype=np.float32)
    best_z_map = np.zeros((h, w), dtype=np.float32)
    best_gray = np.zeros((h, w), dtype=np.float32)

    for gray, z_pos in zip(frames_gray, z_list):
        sharp = compute_laplacian_sharpness_map(gray)
        mask = sharp > best_sharp
        best_sharp[mask] = sharp[mask]
        best_z_map[mask] = float(z_pos)
        best_gray[mask] = gray[mask]

    return best_z_map, best_sharp, best_gray


def compute_dff_volume(frames_gray, z_positions):
    import numpy as np

    if not frames_gray:
        return None, None, None
    h, w = frames_gray[0].shape
    n_steps = len(frames_gray)
    sharpness_stack = np.zeros((n_steps, h, w), dtype=np.float32)
    intensity_stack = np.zeros((n_steps, h, w), dtype=np.float32)

    for index, gray in enumerate(frames_gray):
        sharpness_stack[index] = compute_laplacian_sharpness_map(gray)
        intensity_stack[index] = gray.astype(np.float32)

    best_z_idx = np.argmax(sharpness_stack, axis=0)
    best_sharp = np.max(sharpness_stack, axis=0)
    z_arr = np.array(z_positions, dtype=np.float32)
    depth_map = z_arr[best_z_idx]
    row_idx, col_idx = np.indices((h, w))
    intensity = intensity_stack[best_z_idx, row_idx, col_idx]
    return depth_map, best_sharp, intensity


def merge_focus_maps(base_depth, base_sharp, base_gray, extra_depth, extra_sharp, extra_gray):
    mask = extra_sharp > base_sharp
    base_sharp[mask] = extra_sharp[mask]
    base_depth[mask] = extra_depth[mask]
    base_gray[mask] = extra_gray[mask]
    return int(mask.sum())


def select_focus_window(z_list, frames_gray, fine_pct):
    import numpy as np

    n_bins = max(10, len(frames_gray))
    z_min = float(min(z_list))
    z_max = float(max(z_list))
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_sharp = np.zeros(n_bins, dtype=np.float64)
    bin_count = np.zeros(n_bins, dtype=np.int32)

    for frame, z_pos in zip(frames_gray, z_list):
        ratio = (z_pos - z_min) / (z_max - z_min + 1e-9)
        bucket = min(int(ratio * n_bins), n_bins - 1)
        sharp = compute_laplacian_sharpness_map(frame)
        bin_sharp[bucket] += float(np.mean(sharp))
        bin_count[bucket] += 1

    mean_sharp = np.where(bin_count > 0, bin_sharp / np.maximum(bin_count, 1), 0.0)
    n_select = max(1, int(n_bins * fine_pct / 100.0))
    top_bins = np.argsort(mean_sharp)[::-1][:n_select]
    z0 = float(bin_edges[min(top_bins)])
    z1 = float(bin_edges[min(max(top_bins) + 1, n_bins)])
    if z1 - z0 < 0.1:
        mid = (z0 + z1) / 2.0
        z0, z1 = mid - 0.05, mid + 0.05
    return z0, z1


def point_cloud_from_depth(depth_map, sharp_map, intensity_map, pixels_per_mm, min_sharp, z_scale):
    import numpy as np

    h, w = depth_map.shape
    ppmm = pixels_per_mm if pixels_per_mm > 0 else 1.0
    valid = sharp_map > float(min_sharp)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 4), dtype=np.float32), 0.0
    x_mm = (xs.astype(np.float32) - w / 2.0) / ppmm
    y_mm = (ys.astype(np.float32) - h / 2.0) / ppmm
    z_mm = depth_map[ys, xs].astype(np.float32) * float(z_scale)
    intensity = intensity_map[ys, xs].astype(np.float32)
    cloud = np.column_stack([x_mm, y_mm, z_mm, intensity]).astype(np.float32)
    coverage = 100.0 * len(cloud) / float(w * h) if w * h else 0.0
    return cloud, coverage


def export_point_cloud(file_path, point_cloud, pixels_per_mm, comment):
    if file_path.lower().endswith(".csv"):
        with open(file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["x_mm", "y_mm", "z_mm", "intensity"])
            writer.writerows(point_cloud.tolist())
        return "csv"

    with open(file_path, "w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("comment {}\n".format(comment))
        file.write("comment pixels_per_mm={:.4f}\n".format(pixels_per_mm))
        file.write("element vertex {}\n".format(len(point_cloud)))
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property float intensity\n")
        file.write("end_header\n")
        for row in point_cloud:
            file.write(
                "{:.6f} {:.6f} {:.6f} {:.2f}\n".format(
                    float(row[0]), float(row[1]), float(row[2]), float(row[3])
                )
            )
    return "ply"


def timestamped_pointcloud_name(prefix, extension):
    from datetime import datetime

    return "{}_{}.{}".format(prefix, datetime.now().strftime("%Y%m%d_%H%M%S"), extension.lstrip("."))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_composite_image(intensity_map, file_path):
    """将 float32 灰度强度图保存为图像文件（BMP / PNG）。
    优先用 PyQt5 QImage.save()，失败则退化为纯 Python BMP 写入。
    返回实际保存路径。
    """
    import numpy as np

    arr = np.clip(intensity_map, 0, 255).astype(np.uint8)
    try:
        from PyQt5.QtGui import QImage
        arr_c = np.ascontiguousarray(arr)
        h, w = arr_c.shape
        qimg = QImage(arr_c.data, w, h, w, QImage.Format_Grayscale8)
        if qimg.save(file_path):
            return file_path
    except Exception:
        pass
    # 退化：写原始 8‑bpp BMP
    _write_bmp_grayscale(arr, file_path)
    return file_path


def _write_bmp_grayscale(arr, file_path):
    """将 uint8 二维 numpy 数组写为 8 位灰度 BMP（无需第三方库）。"""
    import struct

    h, w = arr.shape
    row_padded = (w + 3) & ~3          # 每行 4 字节对齐
    palette_bytes = 256 * 4            # 256 色调色板
    pixel_offset = 54 + palette_bytes
    file_size = pixel_offset + row_padded * h

    with open(file_path, "wb") as f:
        # BITMAPFILEHEADER (14 bytes)
        f.write(b"BM")
        f.write(struct.pack("<I", file_size))
        f.write(struct.pack("<HH", 0, 0))
        f.write(struct.pack("<I", pixel_offset))
        # BITMAPINFOHEADER (40 bytes)，负高度 = 从上到下存储
        f.write(struct.pack("<IiiHHIIiiII",
                            40, w, -h, 1, 8, 0,
                            row_padded * h, 2835, 2835, 256, 0))
        # 灰度调色板
        for i in range(256):
            f.write(bytes([i, i, i, 0]))
        # 像素数据
        padding = bytes(row_padded - w)
        for row in arr:
            f.write(bytes(row.tobytes()))
            if padding:
                f.write(padding)
