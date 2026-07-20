import unittest

import cv2
import numpy as np

from microscope_app.core.algorithms import (
    StreamingDFFAccumulator,
    estimate_translation_shift,
    point_cloud_from_surface,
    prepare_depth_surface,
)


class StreamingDFFTests(unittest.TestCase):
    def test_recovers_two_continuous_focus_planes(self):
        height, width = 120, 160
        yy, xx = np.indices((height, width))
        texture = ((((xx // 4) + (yy // 4)) % 2) * 180 + 40).astype(np.float32)
        accumulator = StreamingDFFAccumulator((height, width), color_enabled=False, focus_window_size=9)

        for z_position in np.linspace(0.0, 1.0, 11):
            left = cv2.GaussianBlur(texture, (0, 0), 0.45 + abs(z_position - 0.3) * 8.0)
            right = cv2.GaussianBlur(texture, (0, 0), 0.45 + abs(z_position - 0.7) * 8.0)
            frame = np.where(xx < width // 2, left, right).astype(np.float32)
            accumulator.update(frame, float(z_position))

        result = accumulator.finalize()
        left_depth = float(np.median(result["depth_map"][:, 8 : width // 2 - 8]))
        right_depth = float(np.median(result["depth_map"][:, width // 2 + 8 : -8]))
        self.assertAlmostEqual(left_depth, 0.3, delta=0.025)
        self.assertAlmostEqual(right_depth, 0.7, delta=0.025)
        self.assertGreater(float(np.median(result["confidence_map"])), 0.5)

    def test_translation_estimator_returns_alignment_shift(self):
        rng = np.random.default_rng(42)
        reference = rng.normal(size=(128, 160)).astype(np.float32)
        moving = np.roll(np.roll(reference, 3, axis=0), -5, axis=1)
        dx, dy = estimate_translation_shift(reference, moving, max_width=160, max_shift_px=20)
        self.assertAlmostEqual(dx, 5.0, delta=0.15)
        self.assertAlmostEqual(dy, -3.0, delta=0.15)

    def test_surface_pipeline_repairs_low_confidence_hole(self):
        height, width = 100, 140
        yy, xx = np.indices((height, width))
        depth = (0.2 + 0.002 * xx + 0.001 * yy).astype(np.float32)
        depth[35:65, 50:90] = 1.8
        sharpness = np.full((height, width), 100.0, dtype=np.float32)
        intensity = (80 + ((xx + yy) % 80)).astype(np.float32)
        confidence = np.ones((height, width), dtype=np.float32)
        confidence[35:65, 50:90] = 0.0

        surface, mask, quality = prepare_depth_surface(
            depth,
            sharpness,
            intensity,
            min_sharp=5.0,
            confidence_map=confidence,
            z_step=0.05,
        )
        expected_center = 0.2 + 0.002 * 70 + 0.001 * 50
        self.assertAlmostEqual(float(surface[50, 70]), expected_center, delta=0.15)
        self.assertTrue(bool(mask[50, 70]))
        cloud, coverage = point_cloud_from_surface(surface, mask, intensity, 100.0, 1.0)
        self.assertGreater(len(cloud), height * width * 0.8)
        self.assertGreater(coverage, 80.0)
        self.assertIn("confidence_threshold", quality)


if __name__ == "__main__":
    unittest.main()
