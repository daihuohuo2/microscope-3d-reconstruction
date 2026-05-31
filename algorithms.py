import csv
import json
import os
import re
import struct
import zlib
from datetime import datetime


CALIB_DOT_SPACING_UM = 200.0


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


def compute_sharpness_score(gray, lap_weight=0.6):
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
    return (1.0 - lap_weight) * tenengrad + lap_weight * lap_var


def _box_mean(image, size=9):
    import numpy as np

    arr = np.asarray(image, dtype=np.float32)
    if size <= 1:
        return arr
    try:
        import cv2

        return cv2.blur(arr, (size, size), borderType=cv2.BORDER_REPLICATE).astype(np.float32)
    except Exception:
        pass
    try:
        from scipy.ndimage import uniform_filter

        return uniform_filter(arr, size=size, mode="nearest").astype(np.float32)
    except Exception:
        pass

    pad = size // 2
    padded = np.pad(arr, pad, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    out = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return (out / float(size * size)).astype(np.float32)


def compute_laplacian_sharpness_map(gray, window_size=9):
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
    return _box_mean(sharp, window_size)


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
        sub_dy = _parabolic_peak_offset(
            corr[(dy - 1) % h, dx], corr[dy, dx], corr[(dy + 1) % h, dx]
        )
        sub_dx = _parabolic_peak_offset(
            corr[dy, (dx - 1) % w], corr[dy, dx], corr[dy, (dx + 1) % w]
        )
        dy = float(dy) + sub_dy
        dx = float(dx) + sub_dx
        if dy > h // 2:
            dy -= h
        if dx > w // 2:
            dx -= w
        return float(dx), float(dy)
    except Exception:
        return 0.0, 0.0


def _parabolic_peak_offset(left, center, right):
    denom = float(left - 2.0 * center + right)
    if abs(denom) < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, 0.5 * float(left - right) / denom))


def compute_blob_scale_calibration(gray, spacing_um=CALIB_DOT_SPACING_UM, sample_count=5):
    import numpy as np

    if gray is None or gray.size == 0:
        raise ValueError("无法获取图像")

    img = np.asarray(gray, dtype=np.float32)
    img_u8 = _normalize_to_uint8(img)
    mean_val = float(np.mean(img_u8))
    p01 = float(np.percentile(img_u8, 1.0))
    p99 = float(np.percentile(img_u8, 99.0))
    contrast = p99 - p01
    if mean_val < 35.0 and p99 < 80.0:
        raise ValueError("图像过暗，请提高曝光或灯光后重试")
    if mean_val > 245.0 and (p01 > 220.0 or contrast < 15.0):
        raise ValueError("图像过亮，请降低曝光或灯光后重试")

    # 缩小到不超过 1200px 宽再做 blob 检测，避免高分辨率图卡死
    # 检测后把坐标乘以缩放比例还原到原始像素距离
    MAX_DETECT_WIDTH = 1200
    h_orig, w_orig = img_u8.shape[:2]
    if w_orig > MAX_DETECT_WIDTH:
        scale = MAX_DETECT_WIDTH / w_orig
        new_w = MAX_DETECT_WIDTH
        new_h = int(h_orig * scale)
        try:
            import cv2 as _cv2
            img_small = _cv2.resize(img_u8, (new_w, new_h), interpolation=_cv2.INTER_AREA)
        except Exception:
            img_small = img_u8[::int(1/scale), ::int(1/scale)]
            scale = 1.0 / int(1/scale)
    else:
        img_small = img_u8
        scale = 1.0

    _, binary = _blob_threshold_white_bg(img_small)
    raw_centers = _detect_blob_centers(binary)
    # 把缩小图上的坐标还原到原图坐标
    if scale != 1.0 and raw_centers:
        centers = [(x / scale, y / scale) for x, y in raw_centers]
    else:
        centers = raw_centers
    if len(centers) < max(6, sample_count + 1):
        raise ValueError("未找到足够的标定圆点，请确认标定板已进入视野")

    centers = np.asarray(centers, dtype=np.float32)
    nearest_distances = []
    sample_total = min(sample_count, len(centers))
    chosen_indices = np.random.default_rng().choice(len(centers), sample_total, replace=False)

    for idx in chosen_indices:
        deltas = centers - centers[idx]
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        distances[idx] = np.inf
        nearest = float(np.min(distances))
        if np.isfinite(nearest) and nearest > 1.0:
            nearest_distances.append(nearest)

    if len(nearest_distances) < max(3, min(5, sample_total)):
        raise ValueError("圆点间距计算失败，请检查图像清晰度和标定板")

    avg_spacing_px = float(np.mean(nearest_distances))
    if avg_spacing_px <= 0:
        raise ValueError("圆点间距无效")

    pixels_per_mm = avg_spacing_px / (float(spacing_um) / 1000.0)
    return {
        "pixels_per_mm": float(pixels_per_mm),
        "spacing_px": avg_spacing_px,
        "samples": len(nearest_distances),
        "blob_count": int(len(centers)),
        "mean_brightness": mean_val,
        "contrast": contrast,
    }


def _normalize_to_uint8(img):
    import numpy as np

    arr = np.asarray(img, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    p01 = float(np.percentile(finite, 1.0))
    p99 = float(np.percentile(finite, 99.0))
    if p99 <= 255.0 and p01 >= 0.0:
        return np.clip(arr, 0, 255).astype(np.uint8)
    if p99 <= p01 + 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    norm = (arr - p01) / (p99 - p01) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def _blob_threshold_white_bg(gray_u8):
    import numpy as np

    try:
        import cv2

        blur = cv2.GaussianBlur(gray_u8, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return True, binary
    except Exception:
        threshold = float(np.mean(gray_u8) - 0.7 * np.std(gray_u8))
        threshold = max(0.0, min(255.0, threshold))
        binary = (gray_u8 <= threshold).astype(np.uint8) * 255
        return False, binary.astype(np.uint8)


def _detect_blob_centers(binary):
    try:
        import cv2

        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        area = binary.shape[0] * binary.shape[1]
        params.minArea = max(9.0, area * 0.00001)
        params.maxArea = max(params.minArea * 2.0, area * 0.02)
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary)
        centers = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        if centers:
            return centers
    except Exception:
        pass

    return _detect_blob_centers_cc(binary)


def _detect_blob_centers_cc(binary):
    import numpy as np

    mask = binary > 0
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    centers = []
    min_area = max(9, (h * w) // 100000)
    max_area = max(min_area * 2, (h * w) // 50)

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            queue = [(y, x)]
            visited[y, x] = True
            pixels = []
            while queue:
                cy, cx = queue.pop()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            area = len(pixels)
            if area < min_area or area > max_area:
                continue
            ys = np.array([p[0] for p in pixels], dtype=np.float32)
            xs = np.array([p[1] for p in pixels], dtype=np.float32)
            width = float(xs.max() - xs.min() + 1)
            height = float(ys.max() - ys.min() + 1)
            aspect = min(width, height) / max(width, height, 1.0)
            fill_ratio = area / max(width * height, 1.0)
            if aspect < 0.55 or not (0.35 <= fill_ratio <= 0.95):
                continue
            centers.append((float(xs.mean()), float(ys.mean())))
    return centers


def build_best_focus_maps(frames_gray, z_list, improve_margin=0.08):
    import numpy as np

    if not frames_gray:
        return None, None, None
    h, w = frames_gray[0].shape
    reference_frame, reference_z, _ = select_best_single_frame(frames_gray, z_list)
    if reference_frame is None:
        reference_frame = frames_gray[0]
        reference_z = z_list[0] if z_list else 0.0

    best_sharp = compute_laplacian_sharpness_map(reference_frame)
    best_z_map = np.full((h, w), float(reference_z), dtype=np.float32)
    best_gray = reference_frame.astype(np.float32).copy()
    margin = max(0.0, float(improve_margin))

    for gray, z_pos in zip(frames_gray, z_list):
        sharp = compute_laplacian_sharpness_map(gray)
        mask = sharp > (best_sharp * (1.0 + margin))
        best_sharp[mask] = sharp[mask]
        best_z_map[mask] = float(z_pos)
        best_gray[mask] = gray[mask]

    return best_z_map, best_sharp, best_gray


def build_best_focus_color_maps(frames_gray, z_list, frames_color, improve_margin=0.08):
    import numpy as np

    depth_map, sharp_map, gray_map = build_best_focus_maps(frames_gray, z_list, improve_margin=improve_margin)
    if depth_map is None or not frames_color:
        return depth_map, sharp_map, gray_map, None

    h, w = frames_gray[0].shape
    if len(frames_color) != len(frames_gray):
        return depth_map, sharp_map, gray_map, None

    reference_frame, reference_z, _ = select_best_single_frame(frames_gray, z_list)
    reference_index = 0
    if reference_z is not None:
        for index, z_pos in enumerate(z_list):
            if float(z_pos) == float(reference_z):
                reference_index = index
                break
    if frames_color[reference_index] is None or frames_color[reference_index].shape[:2] != (h, w):
        reference_index = next(
            (
                index
                for index, color in enumerate(frames_color)
                if color is not None and color.shape[:2] == (h, w)
            ),
            None,
        )
        if reference_index is None:
            return depth_map, sharp_map, gray_map, None

    best_sharp = compute_laplacian_sharpness_map(reference_frame if reference_frame is not None else frames_gray[0])
    base_color = frames_color[reference_index].astype(np.float32)
    color_accum = base_color.copy()
    weight_accum = np.ones((h, w), dtype=np.float32)
    margin = max(0.0, float(improve_margin))

    for gray, color in zip(frames_gray, frames_color):
        if color is None or color.shape[:2] != (h, w):
            continue
        sharp = compute_laplacian_sharpness_map(gray)
        ratio = sharp / (best_sharp + 1e-6)
        weight = np.clip((ratio - (1.0 + margin * 0.35)) / 0.45, 0.0, 1.0).astype(np.float32)
        weight *= np.clip(sharp / (np.percentile(sharp, 92.0) + 1e-6), 0.0, 1.0).astype(np.float32)
        weight = _smooth_weight_map(weight)
        if float(np.max(weight)) <= 0.001:
            continue
        color_accum += color.astype(np.float32) * weight[:, :, None]
        weight_accum += weight
        best_sharp = np.maximum(best_sharp, sharp)

    best_color = color_accum / np.maximum(weight_accum[:, :, None], 1e-6)
    best_color = _match_color_statistics(best_color, base_color)
    best_color = _inject_luminance_from_gray(best_color, gray_map)
    return depth_map, sharp_map, gray_map, np.clip(best_color, 0, 255).astype(np.uint8)


def _smooth_weight_map(weight):
    try:
        import cv2

        return cv2.GaussianBlur(weight.astype("float32"), (0, 0), 1.2)
    except Exception:
        pass

    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(weight.astype("float32"), sigma=1.2)
    except Exception:
        return weight


def _match_color_statistics(image, reference):
    import numpy as np

    out = np.asarray(image, dtype=np.float32).copy()
    ref = np.asarray(reference, dtype=np.float32)
    for ch in range(3):
        src_ch = out[:, :, ch]
        ref_ch = ref[:, :, ch]
        src_p10, src_p90 = np.percentile(src_ch, (10.0, 90.0))
        ref_p10, ref_p90 = np.percentile(ref_ch, (10.0, 90.0))
        src_span = max(float(src_p90 - src_p10), 1.0)
        ref_span = max(float(ref_p90 - ref_p10), 1.0)
        src_mid = float((src_p10 + src_p90) * 0.5)
        ref_mid = float((ref_p10 + ref_p90) * 0.5)
        out[:, :, ch] = (src_ch - src_mid) * min(1.25, max(0.75, ref_span / src_span)) + ref_mid
    return np.clip(out, 0, 255)


def _inject_luminance_from_gray(color_image, gray_map):
    import numpy as np

    color = np.asarray(color_image, dtype=np.float32)
    if color.ndim != 3 or color.shape[2] < 3 or gray_map is None:
        return color

    gray = _to_uint8_image(gray_map).astype(np.float32)
    if gray.shape != color.shape[:2]:
        return color

    current_luma = (
        0.299 * color[:, :, 0]
        + 0.587 * color[:, :, 1]
        + 0.114 * color[:, :, 2]
    )
    ratio = gray / np.maximum(current_luma, 1.0)
    ratio = np.clip(ratio, 0.45, 2.20)
    restored = color * ratio[:, :, None]
    return _unsharp_color(restored, amount=0.35, radius=0.8)


def _unsharp_color(image, amount=0.35, radius=0.8):
    import numpy as np

    img = np.asarray(image, dtype=np.float32)
    try:
        import cv2

        blur = cv2.GaussianBlur(img, (0, 0), float(radius))
    except Exception:
        return np.clip(img, 0, 255)
    return np.clip(img + float(amount) * (img - blur), 0, 255)


def select_best_single_frame(frames_gray, z_list=None):
    """Pick the one raw frame with the highest whole-image sharpness score."""
    if not frames_gray:
        return None, None, 0.0
    best_frame = None
    best_z = None
    best_score = None
    for index, gray in enumerate(frames_gray):
        score = compute_sharpness_score(gray)
        if best_score is None or score > best_score:
            best_frame = gray
            best_z = z_list[index] if z_list is not None and index < len(z_list) else None
            best_score = score
    return best_frame, best_z, float(best_score or 0.0)


def select_worst_single_frame(frames_gray, z_list=None):
    """Pick the one raw frame with the lowest whole-image sharpness score (most blurred)."""
    if not frames_gray:
        return None, None, 0.0
    worst_frame = None
    worst_z = None
    worst_score = None
    for index, gray in enumerate(frames_gray):
        score = compute_sharpness_score(gray)
        if worst_score is None or score < worst_score:
            worst_frame = gray
            worst_z = z_list[index] if z_list is not None and index < len(z_list) else None
            worst_score = score
    return worst_frame, worst_z, float(worst_score or 0.0)


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

    # Sub-step parabolic interpolation for depth precision
    if n_steps >= 3:
        interior = (best_z_idx > 0) & (best_z_idx < n_steps - 1)
        ri, ci = np.where(interior)
        if len(ri) > 0:
            idx_c = best_z_idx[ri, ci]
            s_prev = sharpness_stack[idx_c - 1, ri, ci]
            s_curr = sharpness_stack[idx_c, ri, ci]
            s_next = sharpness_stack[idx_c + 1, ri, ci]
            denom = s_prev + s_next - 2.0 * s_curr
            can_fit = denom < -1e-6
            numer = s_prev - s_next
            z_c = z_arr[idx_c]
            z_half_span = (z_arr[idx_c + 1] - z_arr[idx_c - 1]) * 0.5
            offset = np.where(
                can_fit,
                numer / (2.0 * np.where(can_fit, denom, -1.0)) * z_half_span,
                0.0,
            )
            max_off = np.abs(z_arr[idx_c + 1] - z_c)
            offset = np.clip(offset, -max_off, max_off)
            depth_map[ri, ci] = z_c + offset

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

    MAX_W = 640  # 降采样后估算锐度，速度提升约 70 倍
    for frame, z_pos in zip(frames_gray, z_list):
        ratio = (z_pos - z_min) / (z_max - z_min + 1e-9)
        bucket = min(int(ratio * n_bins), n_bins - 1)
        h_f, w_f = frame.shape[:2]
        step = max(1, w_f // MAX_W)
        small = frame[::step, ::step]
        sharp = compute_laplacian_sharpness_map(small)
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

    # min_sharp 语义：
    #   0        → 不过滤（保留全部）
    #   (0, 100] → 相对阈值：保留锐度 ≥ 全图峰值 × (min_sharp/100) 的像素
    #   > 100    → 绝对值（兼容旧配置）
    ms = float(min_sharp)
    if ms <= 0.0:
        actual_thresh = 0.0
    elif ms <= 100.0:
        finite_sharp = sharp_map[np.isfinite(sharp_map)]
        peak = float(np.percentile(finite_sharp, 99.5)) if finite_sharp.size else 0.0
        actual_thresh = peak * (ms / 100.0) if peak > 0.0 else 0.0
    else:
        actual_thresh = ms

    valid = (sharp_map > actual_thresh) & np.isfinite(depth_map)

    # ── 连通性过滤：去除完全孤立的单像素噪点（至少 1 个有效邻居） ──
    valid_f = valid.astype(np.float32)
    pad = np.pad(valid_f, 1, mode='constant')
    neighbor_sum = (
        pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
        pad[1:-1, :-2]                  + pad[1:-1, 2:] +
        pad[2:,  :-2]  + pad[2:,  1:-1] + pad[2:,  2:]
    )
    valid = valid & (neighbor_sum >= 3)   # 至少 3 个有效邻居，避免竖刺噪点

    # ── 深度图中值滤波：消除单像素深度跳变 ──
    depth_use = depth_map
    try:
        from scipy.ndimage import median_filter as _mf
        depth_use = _mf(depth_map.astype(np.float32), size=7)
        local = _mf(depth_use, size=17)
        residual = np.abs(depth_use - local)
        valid_res = residual[valid & np.isfinite(residual)]
        if valid_res.size:
            tol = max(0.025, float(np.percentile(valid_res, 90.0)) * 1.8)
            valid = valid & (residual <= tol)
    except Exception:
        try:
            import cv2
            depth_use = cv2.medianBlur(depth_map.astype(np.float32), 7)
            local = cv2.medianBlur(depth_use, 17)
            residual = np.abs(depth_use - local)
            valid_res = residual[valid & np.isfinite(residual)]
            if valid_res.size:
                tol = max(0.025, float(np.percentile(valid_res, 90.0)) * 1.8)
                valid = valid & (residual <= tol)
        except Exception:
            pass

    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 4), dtype=np.float32), 0.0
    x_mm = (xs.astype(np.float32) - w / 2.0) / ppmm
    y_mm = (ys.astype(np.float32) - h / 2.0) / ppmm
    z_raw = depth_use[ys, xs].astype(np.float32)
    if len(z_raw) >= 50:
        z0 = float(np.percentile(z_raw, 2.0))
        z1 = float(np.percentile(z_raw, 98.0))
        keep = (z_raw >= z0) & (z_raw <= z1)
        xs, ys, z_raw = xs[keep], ys[keep], z_raw[keep]
        x_mm = x_mm[keep]
        y_mm = y_mm[keep]
    z_base = float(np.percentile(z_raw, 2.0)) if len(z_raw) else 0.0
    z_mm = (z_raw - z_base).astype(np.float32) * float(z_scale)
    intensity = intensity_map[ys, xs].astype(np.float32)

    # ── Z 离群点过滤：仅去除极端异常值（±3倍标准差外的竖柱噪点） ──
    if len(z_mm) >= 50:
        z_mean = float(np.mean(z_mm))
        z_std  = float(np.std(z_mm))
        if z_std > 0:
            keep = np.abs(z_mm - z_mean) <= 3.0 * z_std
            x_mm, y_mm, z_mm, intensity = x_mm[keep], y_mm[keep], z_mm[keep], intensity[keep]

    cloud = np.column_stack([x_mm, y_mm, z_mm, intensity]).astype(np.float32)
    coverage = 100.0 * len(cloud) / float(w * h) if w * h else 0.0
    return cloud, coverage


def _intensity_to_rgb(intensity):
    import numpy as np

    intensity = np.asarray(intensity, dtype=np.float32)
    scale = np.where(intensity > 4095.0, 65535.0, np.where(intensity > 255.0, 4095.0, 255.0))
    value = np.clip(intensity / scale * 255.0, 0, 255).astype(np.uint8)
    return value, value, value


def _jet_rgb_from_values(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax > vmin:
        t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        t = np.zeros(values.shape, dtype=np.float32)
    r = np.clip(1.5 - np.abs(t * 4.0 - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(t * 4.0 - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(t * 4.0 - 1.0), 0, 1)
    return (np.column_stack([r, g, b]) * 255).astype(np.uint8)


def _foreground_mask_from_intensity(intensity_map):
    import numpy as np

    if intensity_map is None:
        return None
    img = _normalize_to_uint8(intensity_map)
    h, w = img.shape
    if h < 20 or w < 20:
        return np.ones((h, w), dtype=bool)
    border = max(3, int(min(h, w) * 0.05))
    samples = np.concatenate([
        img[:border, :].ravel(),
        img[-border:, :].ravel(),
        img[:, :border].ravel(),
        img[:, -border:].ravel(),
    ]).astype(np.float32)
    img_f = img.astype(np.float32)
    bg = float(np.median(samples))
    spread = float(np.percentile(samples, 90.0) - np.percentile(samples, 10.0))
    threshold = max(12.0, spread * 1.1)

    if bg >= 175.0:
        # Common microscope case: tooth/object is darker than a bright white background.
        mask = img_f <= bg - threshold
    elif bg <= 70.0:
        # Dark background: object is usually brighter.
        mask = img_f >= bg + threshold
    else:
        mask = np.abs(img_f - bg) >= threshold

    # Saturated reflection spots are not object shape; they should be filled from neighbors later.
    mask = mask & (img_f < 248.0)

    try:
        from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, binary_erosion, label
        mask = binary_opening(mask, iterations=1)
        mask = binary_closing(mask, iterations=5)
        mask = binary_fill_holes(mask)
        labeled, count = label(mask)
        if count > 0:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            largest = int(np.argmax(sizes))
            mask = labeled == largest
        if float(np.mean(mask)) > 0.88:
            mask = binary_erosion(mask, iterations=2)
    except Exception:
        # Small numpy fallback: close pinholes by neighbor majority.
        for _ in range(2):
            pad = np.pad(mask.astype(np.uint8), 1, mode="edge")
            neighbor_sum = (
                pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
                pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
                pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
            )
            mask = neighbor_sum >= 4

    coverage = float(np.mean(mask))
    if coverage < 0.02:
        return np.ones((h, w), dtype=bool)
    if coverage > 0.95:
        # A mask that covers nearly everything is usually background leakage.
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h / 2.0, w / 2.0
        ry, rx = h * 0.46, w * 0.46
        return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0)
    return mask.astype(bool)


def _regularize_depth_for_surface(depth, mask):
    import numpy as np

    depth_use = np.asarray(depth, dtype=np.float32)
    try:
        from scipy.ndimage import gaussian_filter, median_filter
        finite = np.isfinite(depth_use) & mask
        if np.any(finite):
            fill_value = float(np.median(depth_use[finite]))
            depth_use = np.where(np.isfinite(depth_use), depth_use, fill_value)
            depth_use = np.where(mask, depth_use, fill_value)
        local = median_filter(depth_use, size=9)
        residual = depth_use - local
        valid_res = np.abs(residual[mask & np.isfinite(residual)])
        if valid_res.size:
            tol = max(0.025, float(np.percentile(valid_res, 80.0)) * 1.5)
            depth_use = np.where(np.abs(residual) > tol, local, depth_use)
        depth_use = median_filter(depth_use, size=11)
        depth_use = gaussian_filter(depth_use, sigma=2.0)
    except Exception:
        try:
            import cv2
            finite = np.isfinite(depth_use) & mask
            if np.any(finite):
                fill_value = float(np.median(depth_use[finite]))
                depth_use = np.where(np.isfinite(depth_use), depth_use, fill_value)
                depth_use = np.where(mask, depth_use, fill_value)
            local = cv2.medianBlur(depth_use.astype(np.float32), 9)
            residual = depth_use - local
            valid_res = np.abs(residual[mask & np.isfinite(residual)])
            if valid_res.size:
                tol = max(0.025, float(np.percentile(valid_res, 80.0)) * 1.5)
                depth_use = np.where(np.abs(residual) > tol, local, depth_use)
            depth_use = cv2.medianBlur(depth_use.astype(np.float32), 11)
            depth_use = cv2.GaussianBlur(depth_use.astype(np.float32), (0, 0), 2.0)
        except Exception:
            pass
    return depth_use.astype(np.float32)


def export_point_cloud(file_path, point_cloud, pixels_per_mm, comment):
    import numpy as np

    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    height_rgb = _jet_rgb_from_values(point_cloud[:, 2]) if len(point_cloud) else np.zeros((0, 3), dtype=np.uint8)
    if file_path.lower().endswith(".csv"):
        with open(file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["x_mm", "y_mm", "z_mm", "intensity", "red", "green", "blue"])
            for row, color in zip(point_cloud, height_rgb):
                writer.writerow(
                    [
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        int(color[0]),
                        int(color[1]),
                        int(color[2]),
                    ]
                )
        return "csv"

    if file_path.lower().endswith(".obj"):
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("# {}\n".format(comment))
            file.write("# pixels_per_mm={:.4f}\n".format(pixels_per_mm))
            file.write("# coordinates_unit=mm\n")
            file.write("# color_source=relative_height\n")
            file.write("# vertex format: v x_mm y_mm z_mm red green blue\n")
            for row, color in zip(point_cloud, height_rgb):
                file.write(
                    "v {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        color[0] / 255.0,
                        color[1] / 255.0,
                        color[2] / 255.0,
                    )
                )
        return "obj"

    # 二进制 PLY：numpy 结构化数组一次性写入，速度比逐行 ASCII 快约 100 倍
    import numpy as np
    pc = np.asarray(point_cloud, dtype=np.float32)
    n = len(pc)
    rgb_u8 = _jet_rgb_from_values(pc[:, 2])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment {}\n".format(comment)
        + "comment pixels_per_mm={:.4f}\n".format(pixels_per_mm)
        + "comment coordinates_unit=mm\n"
        "comment color_source=relative_height\n"
        "element vertex {}\n".format(n)
        + "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float intensity\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    data = np.empty(n, dtype=dt)
    data["x"] = pc[:, 0]
    data["y"] = pc[:, 1]
    data["z"] = pc[:, 2]
    data["intensity"] = pc[:, 3]
    data["red"] = rgb_u8[:, 0]
    data["green"] = rgb_u8[:, 1]
    data["blue"] = rgb_u8[:, 2]
    with open(file_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    return "ply"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_autofocus_curve(output_path, samples, title="自动对焦锐度曲线"):
    """Save autofocus Z/score samples as a PNG curve."""
    import numpy as np

    if not samples:
        return None

    z_values = np.asarray([float(item["z_mm"]) for item in samples], dtype=np.float32)
    scores = np.asarray([float(item["score"]) for item in samples], dtype=np.float32)
    phases = [str(item.get("phase", "")) for item in samples]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    font_prop = get_mpl_font()
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)

    order = np.argsort(z_values)
    ax.plot(
        z_values[order],
        scores[order],
        color="#1f77b4",
        linewidth=1.8,
        alpha=0.72,
        label="Z-sorted curve",
    )

    phase_order = []
    for phase in phases:
        if phase not in phase_order:
            phase_order.append(phase)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]
    for idx, phase in enumerate(phase_order):
        mask = np.asarray([p == phase for p in phases], dtype=bool)
        label = phase or "sample"
        ax.scatter(
            z_values[mask],
            scores[mask],
            s=24,
            color=colors[idx % len(colors)],
            edgecolors="white",
            linewidths=0.6,
            label=label,
            zorder=3,
        )

    if scores.size:
        best_idx = int(np.argmax(scores))
        ax.axvline(float(z_values[best_idx]), color="#d62728", linestyle="--", linewidth=1.2, alpha=0.65)
        ax.annotate(
            "Best Z {:+.3f} mm\nScore {:.0f}".format(float(z_values[best_idx]), float(scores[best_idx])),
            xy=(float(z_values[best_idx]), float(scores[best_idx])),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=9,
            fontproperties=font_prop,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d62728", alpha=0.88),
        )

    ax.set_xlabel("Z 偏移 (mm)", fontproperties=font_prop)
    ax.set_ylabel("锐度分数", fontproperties=font_prop)
    ax.set_title(title, fontproperties=font_prop)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", fontsize=8, prop=font_prop)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_autofocus_curve_csv(output_path, samples):
    if not samples:
        return None
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "phase", "z_offset_mm", "sharpness_score"])
        for index, item in enumerate(samples):
            writer.writerow([
                index,
                str(item.get("phase", "")),
                float(item["z_mm"]),
                float(item["score"]),
            ])
    return output_path


def save_composite_image(intensity_map, file_path):
    """将 float32 灰度强度图保存为图像文件（BMP / PNG / TIFF）。
    优先用 cv2.imwrite()，其次 PyQt5 QImage.save()，失败则退化为纯 Python 写入。
    返回实际保存路径。
    """
    import numpy as np

    source = np.asarray(intensity_map, dtype=np.float32)
    if file_path.lower().endswith((".tif", ".tiff")):
        arr16 = _to_at_least_12bit_image(source)
        _write_tiff_grayscale(arr16, file_path, bits_per_sample=16)
        return file_path

    arr = _to_uint8_image(source)

    # 首选 cv2：PNG 压缩 level=1（最快），速度约是 Qt 默认 level=6 的 10-20 倍
    try:
        import cv2 as _cv2
        if file_path.lower().endswith(".png"):
            _cv2.imwrite(file_path, arr, [_cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            _cv2.imwrite(file_path, arr)
        return file_path
    except Exception:
        pass

    # 次选 Qt
    try:
        from PyQt5.QtGui import QImage
        arr_c = np.ascontiguousarray(arr)
        h, w = arr_c.shape
        qimg = QImage(arr_c.data, w, h, w, QImage.Format_Grayscale8)
        qimg._keep = arr_c  # 防止 GC
        if qimg.save(file_path):
            return file_path
    except Exception:
        pass

    if file_path.lower().endswith(".png"):
        _write_png_grayscale(arr, file_path)
        return file_path
    # 退化：写原始 8-bpp BMP
    _write_bmp_grayscale(arr, file_path)
    return file_path


def save_color_image(rgb_map, file_path):
    import numpy as np

    if rgb_map is None:
        return None
    rgb = np.asarray(rgb_map)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    rgb8 = np.clip(rgb[:, :, :3], 0, 255).astype(np.uint8)
    try:
        import cv2 as _cv2

        bgr = rgb8[:, :, ::-1]
        if file_path.lower().endswith(".png"):
            _cv2.imwrite(file_path, bgr, [_cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            _cv2.imwrite(file_path, bgr)
        return file_path
    except Exception:
        pass
    try:
        from PyQt5.QtGui import QImage

        arr_c = np.ascontiguousarray(rgb8)
        h, w = arr_c.shape[:2]
        qimg = QImage(arr_c.data, w, h, w * 3, QImage.Format_RGB888)
        qimg._keep = arr_c
        if qimg.save(file_path):
            return file_path
    except Exception:
        pass
    return None


def save_focus_comparison_image(reference_map, full_focus_map, pixels_per_mm, file_path, reference_label="single best frame"):
    """Save no-fusion vs DFF full-focus as one side-by-side PNG, both panels with scale bar."""
    import numpy as np

    if reference_map is None or full_focus_map is None:
        return None
    # Apply scale bars to both panels before compositing
    left = add_scale_bar_to_image(reference_map, pixels_per_mm)
    right = add_scale_bar_to_image(full_focus_map, pixels_per_mm)
    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    left = left[:h, :w]
    right = right[:h, :w]
    divider = np.full((h, max(3, w // 300)), 255, dtype=np.uint8)
    canvas = np.concatenate([left, divider, right], axis=1)

    try:
        import cv2 as _cv2

        rgb = _cv2.cvtColor(canvas, _cv2.COLOR_GRAY2BGR)
        font = _cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.7, min(1.8, w / 1200.0))
        thickness = max(1, int(round(scale * 2)))
        pad = max(18, int(24 * scale))
        _cv2.putText(
            rgb,
            "No fusion: {}".format(reference_label),
            (pad, pad * 2),
            font,
            scale,
            (255, 255, 255),
            thickness,
            _cv2.LINE_AA,
        )
        _cv2.putText(
            rgb,
            "DFF full-focus fusion",
            (w + divider.shape[1] + pad, pad * 2),
            font,
            scale,
            (255, 255, 255),
            thickness,
            _cv2.LINE_AA,
        )
        if _cv2.imwrite(file_path, rgb, [_cv2.IMWRITE_PNG_COMPRESSION, 1]):
            return file_path
    except Exception:
        pass

    save_composite_image(canvas, file_path)
    return file_path


def save_focus_comparison_color_image(reference_rgb, full_focus_rgb, file_path, reference_label="single best frame", pixels_per_mm=0.0):
    import numpy as np

    if reference_rgb is None or full_focus_rgb is None:
        return None
    # Apply scale bars to both panels
    left = add_scale_bar_to_color_image(reference_rgb, pixels_per_mm)
    right = add_scale_bar_to_color_image(full_focus_rgb, pixels_per_mm)
    left = np.clip(np.asarray(left)[:, :, :3], 0, 255).astype(np.uint8)
    right = np.clip(np.asarray(right)[:, :, :3], 0, 255).astype(np.uint8)
    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    left = left[:h, :w]
    right = right[:h, :w]
    divider = np.full((h, max(3, w // 300), 3), 255, dtype=np.uint8)
    canvas = np.concatenate([left, divider, right], axis=1)

    try:
        import cv2 as _cv2

        bgr = canvas[:, :, ::-1].copy()
        font = _cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.7, min(1.8, w / 1200.0))
        thickness = max(1, int(round(scale * 2)))
        pad = max(18, int(24 * scale))
        _cv2.putText(bgr, "No fusion: {}".format(reference_label), (pad, pad * 2), font, scale, (255, 255, 255), thickness, _cv2.LINE_AA)
        _cv2.putText(bgr, "DFF full-focus fusion", (w + divider.shape[1] + pad, pad * 2), font, scale, (255, 255, 255), thickness, _cv2.LINE_AA)
        if _cv2.imwrite(file_path, bgr, [_cv2.IMWRITE_PNG_COMPRESSION, 1]):
            return file_path
    except Exception:
        pass

    return save_color_image(canvas, file_path)


def _estimate_gray_bit_depth(arr):
    import numpy as np

    finite = np.asarray(arr)[np.isfinite(arr)]
    if finite.size == 0:
        return 8
    max_value = float(np.max(finite))
    if max_value <= 255.0:
        return 8
    if max_value <= 4095.0:
        return 12
    return 16


def _to_uint8_image(arr):
    import numpy as np

    source = np.asarray(arr, dtype=np.float32)
    bit_depth = _estimate_gray_bit_depth(source)
    if bit_depth > 8:
        scale = 4095.0 if bit_depth == 12 else 65535.0
        source = source / scale * 255.0
    return np.clip(np.nan_to_num(source, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)


def _to_at_least_12bit_image(arr):
    import numpy as np

    source = np.asarray(arr, dtype=np.float32)
    source = np.nan_to_num(source, nan=0.0, posinf=65535.0, neginf=0.0)
    bit_depth = _estimate_gray_bit_depth(source)
    if bit_depth <= 8:
        # TIFF has no portable 12-bit grayscale baseline in this writer, so store
        # 12-bit values in a 16-bit container instead of silently saving 8-bit.
        source = source / 255.0 * 4095.0
    return np.clip(source, 0.0, 65535.0).astype(np.uint16)


def _draw_scale_bar_on_array(arr, pixels_per_mm):
    """In-place: draw a labeled scale bar (bar + text) on a uint8 ndarray (HxW or HxWx3)."""
    import numpy as np

    is_gray = arr.ndim == 2
    h, w = arr.shape[:2]
    if h < 32 or w < 32 or pixels_per_mm <= 0:
        return arr

    bar_len_px = max(4, int(round(w * 0.20)))
    bar_len_mm = bar_len_px / float(pixels_per_mm)
    # Label: use μm when < 1 mm, otherwise mm
    if bar_len_mm < 1.0:
        label = "{:.0f} um".format(bar_len_mm * 1000)
    else:
        label = "{:.2f} mm".format(bar_len_mm).rstrip("0").rstrip(".") + " mm"

    try:
        import cv2 as _cv2

        # Convert gray → BGR for drawing, convert back after
        if is_gray:
            canvas = _cv2.cvtColor(arr, _cv2.COLOR_GRAY2BGR)
        else:
            canvas = arr[:, :, :3].copy()

        font = _cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.55, min(1.6, w / 1400.0))
        thickness = max(1, int(round(font_scale * 2)))
        bar_h = max(6, h // 100)
        margin = max(14, min(w, h) // 30)

        (tw, th), baseline = _cv2.getTextSize(label, font, font_scale, thickness)
        text_gap = max(4, bar_h // 2)
        total_h = bar_h + text_gap + th + baseline

        x0 = max(margin, w - margin - bar_len_px)
        x1 = min(w - margin, x0 + bar_len_px)
        y_bar = max(margin, h - margin - total_h)
        y_text = y_bar + bar_h + text_gap + th

        # Dark background box
        bg_pad = max(4, bar_h // 2)
        bx0 = max(0, x0 - bg_pad)
        bx1 = min(w, x1 + bg_pad)
        by0 = max(0, y_bar - bg_pad)
        by1 = min(h, y_text + baseline + bg_pad)
        canvas[by0:by1, bx0:bx1] = (canvas[by0:by1, bx0:bx1].astype(np.float32) * 0.30).astype(np.uint8)

        # Scale bar line + end ticks
        tick_h = max(bar_h + 2, bar_h * 2)
        _cv2.rectangle(canvas, (x0, y_bar), (x1, y_bar + bar_h), (255, 255, 255), -1)
        _cv2.rectangle(canvas, (x0, y_bar), (x0 + max(2, bar_h // 2), y_bar + tick_h), (255, 255, 255), -1)
        _cv2.rectangle(canvas, (x1 - max(2, bar_h // 2), y_bar), (x1, y_bar + tick_h), (255, 255, 255), -1)

        # Text centered under bar
        tx = x0 + (bar_len_px - tw) // 2
        _cv2.putText(canvas, label, (tx, y_text), font, font_scale, (255, 255, 255), thickness, _cv2.LINE_AA)

        if is_gray:
            return _cv2.cvtColor(canvas, _cv2.COLOR_BGR2GRAY)
        return canvas
    except Exception:
        return arr


def add_scale_bar_to_image(intensity_map, pixels_per_mm):
    import numpy as np

    arr = _to_uint8_image(intensity_map).copy()
    if pixels_per_mm <= 0 or arr.ndim != 2:
        return arr
    return _draw_scale_bar_on_array(arr, pixels_per_mm)


def add_scale_bar_to_color_image(rgb_map, pixels_per_mm):
    import numpy as np

    rgb = np.asarray(rgb_map)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb_map
    arr = np.clip(rgb[:, :, :3], 0, 255).astype(np.uint8).copy()
    if pixels_per_mm <= 0:
        return arr
    return _draw_scale_bar_on_array(arr, pixels_per_mm)


def save_depth_tiff16(depth_map, file_path, z_scale=1.0):
    """Save a relative-height depth map as 16-bit grayscale TIFF.

    The stored pixel values are relative height in micrometers.
    """
    import numpy as np

    if depth_map is None:
        raise ValueError("depth_map is empty")
    depth = np.asarray(depth_map, dtype=np.float32) * float(z_scale)
    valid = np.isfinite(depth)
    if not np.any(valid):
        raise ValueError("depth_map has no finite values")
    relative_um = (depth - float(np.nanmin(depth[valid]))) * 1000.0
    arr = np.clip(np.nan_to_num(relative_um, nan=0.0, posinf=65535.0, neginf=0.0), 0, 65535)
    _write_tiff_grayscale(arr.astype(np.uint16), file_path, bits_per_sample=16)
    return file_path


def _write_tiff_grayscale(arr, file_path, bits_per_sample):
    """Write a simple little-endian uncompressed grayscale TIFF."""
    import numpy as np

    if bits_per_sample not in (8, 16):
        raise ValueError("bits_per_sample must be 8 or 16")
    if arr.ndim != 2:
        raise ValueError("Only 2D grayscale images are supported")

    h, w = arr.shape
    if bits_per_sample == 8:
        pixel_data = np.asarray(arr, dtype=np.uint8).tobytes(order="C")
    else:
        pixel_data = np.asarray(arr, dtype="<u2").tobytes(order="C")

    entries = [
        (256, 4, 1, w),                       # ImageWidth
        (257, 4, 1, h),                       # ImageLength
        (258, 3, 1, bits_per_sample),         # BitsPerSample
        (259, 3, 1, 1),                       # Compression: none
        (262, 3, 1, 1),                       # PhotometricInterpretation: black is zero
        (273, 4, 1, 0),                       # StripOffsets, patched below
        (277, 3, 1, 1),                       # SamplesPerPixel
        (278, 4, 1, h),                       # RowsPerStrip
        (279, 4, 1, len(pixel_data)),         # StripByteCounts
    ]
    ifd_offset = 8
    ifd_size = 2 + len(entries) * 12 + 4
    pixel_offset = ifd_offset + ifd_size
    entries[5] = (273, 4, 1, pixel_offset)

    with open(file_path, "wb") as file:
        file.write(b"II")
        file.write(struct.pack("<H", 42))
        file.write(struct.pack("<I", ifd_offset))
        file.write(struct.pack("<H", len(entries)))
        for tag, field_type, count, value in entries:
            file.write(struct.pack("<HHI", tag, field_type, count))
            if field_type == 3 and count == 1:
                file.write(struct.pack("<H", value))
                file.write(b"\x00\x00")
            else:
                file.write(struct.pack("<I", value))
        file.write(struct.pack("<I", 0))
        file.write(pixel_data)


def _write_png_grayscale(arr, file_path):
    """Write an 8-bit grayscale PNG without external image libraries."""
    import numpy as np

    data = np.asarray(arr, dtype=np.uint8)
    if data.ndim != 2:
        raise ValueError("Only 2D grayscale images are supported")
    h, w = data.shape
    raw = b"".join(b"\x00" + data[row].tobytes() for row in range(h))

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    with open(file_path, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")
        file.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)))
        file.write(chunk(b"IDAT", zlib.compress(raw, 1)))  # level=1 最快
        file.write(chunk(b"IEND", b""))


def _format_param_value(value):
    if isinstance(value, float):
        text = "{:.3f}".format(value).rstrip("0").rstrip(".")
    else:
        text = str(value)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff.-]+", "", text).replace(".", "p")


def build_output_basename(prefix, params=None, timestamp=None):
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [prefix, timestamp]
    for key, value in (params or {}).items():
        if value is None or value == "":
            continue
        safe_key = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(key))
        safe_value = _format_param_value(value)
        if safe_key and safe_value:
            parts.append("{}{}".format(safe_key, safe_value))
    return "_".join(parts)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def save_output_bundle(
    save_dir,
    prefix,
    intensity_map,
    depth_map,
    point_cloud,
    pixels_per_mm,
    params=None,
    z_scale=1.0,
    comment="Generated by microscope reconstruction",
    reference_map=None,
    reference_label="single best frame",
    color_map=None,
    reference_color_map=None,
    frames_gray=None,
    z_positions=None,
    frames_color=None,
):
    # 每次拍摄创建独立子文件夹，文件夹名 = 拍摄时间_类型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = "{}_{}".format(timestamp, prefix)
    out_dir = os.path.join(save_dir, folder_name)
    ensure_dir(out_dir)

    full_focus_bit_depth = 8 if color_map is not None else max(12, _estimate_gray_bit_depth(intensity_map))
    paths = {}
    paths["full_focus"] = os.path.join(out_dir, "full_focus.png")
    if color_map is not None:
        save_color_image(add_scale_bar_to_color_image(color_map, pixels_per_mm), paths["full_focus"])
    else:
        save_composite_image(add_scale_bar_to_image(intensity_map, pixels_per_mm), paths["full_focus"])
        paths["full_focus_tiff"] = os.path.join(out_dir, "full_focus.tif")
        save_composite_image(intensity_map, paths["full_focus_tiff"])

    if reference_map is not None:
        paths["focus_compare"] = os.path.join(out_dir, "focus_compare.png")
        if color_map is not None and reference_color_map is not None:
            save_focus_comparison_color_image(reference_color_map, color_map, paths["focus_compare"], reference_label, pixels_per_mm)
        else:
            save_focus_comparison_image(reference_map, intensity_map, pixels_per_mm, paths["focus_compare"], reference_label)

    paths["depth"] = os.path.join(out_dir, "depth_um16.tif")
    save_depth_tiff16(depth_map, paths["depth"], z_scale=z_scale)

    if point_cloud is not None and len(point_cloud) > 0:
        paths["point_cloud_ply"] = os.path.join(out_dir, "point_cloud.ply")
        export_point_cloud(paths["point_cloud_ply"], point_cloud, pixels_per_mm, comment)
    # ── 逐帧原始图 ──
    if frames_gray is not None and z_positions is not None and len(frames_gray) > 0:
        import numpy as _np
        frames_dir = os.path.join(out_dir, "frames")
        ensure_dir(frames_dir)
        for _i, (_gray, _z) in enumerate(zip(frames_gray, z_positions)):
            _fname = "frame_{:03d}_z{:+.3f}mm".format(_i + 1, float(_z))
            _color = (frames_color[_i]
                      if (frames_color and _i < len(frames_color))
                      else None)
            if _color is not None:
                save_color_image(_color,
                                 os.path.join(frames_dir, _fname + ".tif"))
            else:
                save_composite_image(
                    _np.asarray(_gray, dtype=_np.float32),
                    os.path.join(frames_dir, _fname + ".tif"))
        paths["frames_dir"] = frames_dir

    paths["manifest"] = os.path.join(out_dir, "manifest.json")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "folder": folder_name,
        "generator": comment,
        "parameters": _json_safe(params or {}),
        "z_positions_mm": _json_safe(z_positions if z_positions is not None else []),
        "units": {
            "depth_tiff_pixel_value": "relative_height_um",
            "point_cloud_coordinates": "mm",
            "pixels_per_mm": float(pixels_per_mm),
            "z_scale": float(z_scale),
        },
        "full_focus_bit_depth": full_focus_bit_depth,
        "files": {key: os.path.basename(value) for key, value in paths.items()},
        "notes": [
            "full_focus PNG is color when RGB camera data is available; grayscale fallback is only used when no color frames are captured",
            "full_focus TIFF is only written for grayscale fallback and is saved in a 16-bit container",
            "focus_compare PNG shows the worst single raw frame beside the DFF full-focus result when available",
            "depth TIFF is 16-bit grayscale; pixel values are relative height in micrometers",
            "PLY (binary) vertices include RGB values derived from the full-focus intensity texture",
        ],
    }
    with open(paths["manifest"], "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    # 返回实际保存目录，方便调用方展示
    paths["output_dir"] = out_dir
    return paths


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
