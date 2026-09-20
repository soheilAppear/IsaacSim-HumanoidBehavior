# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check distant pointing stability and intentional aim response without Kit."""

from __future__ import annotations

import importlib.util
import math
import statistics
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
    / "point_ray.py"
)
SPEC = importlib.util.spec_from_file_location("point_ray_under_test", MODULE_PATH)
RAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RAY)


def pointing_direction(pitch_deg: float, yaw_deg: float = 0.0) -> tuple[float, float, float]:
    """Create a measured unit ray, with positive pitch pointing down toward floor."""
    pitch, yaw = math.radians(pitch_deg), math.radians(yaw_deg)
    return math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)


def angular_error_degrees(first: tuple, second: tuple) -> float:
    """Compare normalized ray directions independently of their magnitudes."""
    dot = sum(a * b for a, b in zip(first, second)) / (math.hypot(*first) * math.hypot(*second))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


class TestPointRayControls(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = RAY.PointRayFilter()
        self.origin = (0.0, 0.0, 1.2)
        self.direction = pointing_direction(12.0)

    def update(self, now_s: float, **overrides: object) -> tuple | None:
        """Supply valid measured geometry unless a test varies the input."""
        values = {"origin": self.origin, "direction": self.direction, "now_s": now_s}
        values.update(overrides)
        return self.filter.update(**values)

    def test_first_observation_uses_current_pose_and_normalizes_direction(self) -> None:
        origin, direction = self.update(0.0, direction=(2.0, 0.0, -2.0))
        self.assertEqual(origin, self.origin)
        self.assertAlmostEqual(math.hypot(*direction), 1.0)
        self.assertLess(angular_error_degrees(direction, (1.0, 0.0, -1.0)), 1e-5)

    def test_distant_floor_jitter_is_reduced_without_smoothing_collision_points(self) -> None:
        raw_floor_hits, filtered_floor_hits = [], []
        # At 12 degrees below horizontal, a 1.2 m-high ray reaches floor 5.65 m
        # away. Subdegree optical jitter is therefore visible as large hit motion.
        for sample in range(361):
            now = sample / 90.0
            direction = pointing_direction(12.0 + 0.8 * math.sin(2.0 * math.pi * 8.0 * now))
            origin, filtered = self.update(now, direction=direction)
            if now > 1.0:
                raw_floor_hits.append(-self.origin[2] * direction[0] / direction[2])
                filtered_floor_hits.append(origin[0] - origin[2] * filtered[0] / filtered[2])
        self.assertLess(statistics.pstdev(filtered_floor_hits), 0.35 * statistics.pstdev(raw_floor_hits))
        self.assertAlmostEqual(statistics.mean(filtered_floor_hits), 1.2 / math.tan(math.radians(12.0)), delta=0.04)

    def test_deliberate_large_turn_follows_quickly_without_overshoot(self) -> None:
        self.update(0.0)
        target = pointing_direction(12.0, 30.0)
        errors = []
        for sample in range(1, 36):
            unused_origin, direction = self.update(sample / 100.0, direction=target)
            errors.append(angular_error_degrees(direction, target))
            self.assertAlmostEqual(math.hypot(*direction), 1.0)
        self.assertLess(errors[9], 5.0)
        self.assertLess(errors[-1], 0.7)
        self.assertTrue(all(current <= previous for previous, current in zip(errors, errors[1:])))

    def test_origin_tracks_wrist_translation_faster_than_direction_filter(self) -> None:
        self.update(0.0)
        origin, direction = self.update(0.05, origin=(0.2, 0.0, 1.2))
        self.assertGreater(origin[0], 0.18)
        self.assertLessEqual(origin[0], 0.2)
        self.assertLess(angular_error_degrees(direction, self.direction), 1e-5)

    def test_small_angular_deadband_suppresses_tremor(self) -> None:
        self.update(0.0)
        for sample in range(1, 21):
            unused_origin, direction = self.update(
                sample * 0.01, direction=pointing_direction(12.0 + (-1) ** sample * 0.15)
            )
            self.assertEqual(direction, self.direction)

    def test_wall_time_response_matches_across_callback_rates(self) -> None:
        directions = []
        for rate in (25, 50, 100, 200):
            self.filter.reset()
            self.update(0.0)
            for sample in range(1, round(0.12 * rate) + 1):
                unused_origin, direction = self.update(sample / rate, direction=pointing_direction(13.8))
            directions.append(direction)
        for direction in directions[1:]:
            self.assertLess(angular_error_degrees(direction, directions[0]), 1e-5)

    def test_tracking_loss_clears_history_and_never_emits_a_cached_ray(self) -> None:
        self.update(0.0)
        self.assertIsNone(self.update(0.01, tracked=False))
        target = pointing_direction(45.0, -25.0)
        unused_origin, direction = self.update(0.02, direction=target)
        self.assertLess(angular_error_degrees(direction, target), 1e-5)

    def test_reset_on_gesture_release_prevents_aim_from_previous_gesture(self) -> None:
        self.update(0.0)
        self.filter.reset()
        target = pointing_direction(60.0, 70.0)
        origin, direction = self.update(0.01, origin=(1.0, 2.0, 3.0), direction=target)
        self.assertEqual(origin, (1.0, 2.0, 3.0))
        self.assertLess(angular_error_degrees(direction, target), 1e-5)

    def test_long_gap_reseeds_current_geometry_without_predicting_missing_motion(self) -> None:
        self.update(0.0)
        target = pointing_direction(50.0)
        origin, direction = self.update(0.5, origin=(0.3, 0.0, 0.9), direction=target)
        self.assertEqual(origin, (0.3, 0.0, 0.9))
        self.assertLess(angular_error_degrees(direction, target), 1e-5)

    def test_invalid_and_duplicate_timestamps_clear_filter(self) -> None:
        for invalid in (None, True, "1.0", -1.0, math.nan, math.inf, 1.0, 0.5):
            with self.subTest(invalid=invalid):
                self.filter.reset()
                self.update(1.0)
                self.assertIsNone(self.update(invalid))
                target = pointing_direction(40.0)
                unused_origin, direction = self.update(1.1, direction=target)
                self.assertLess(angular_error_degrees(direction, target), 1e-5)

    def test_invalid_geometry_clears_filter(self) -> None:
        for key in ("origin", "direction"):
            for value in (None, (), (1, 2), (1, 2, 3, 4), (1, math.nan, 0), (1, math.inf, 0), (True, 0, 0), "100"):
                with self.subTest(key=key, value=value):
                    self.filter.reset()
                    self.update(0.0)
                    self.assertIsNone(self.update(0.01, **{key: value}))
                    self.assertIsNone(self.filter._last_time_s)
        self.assertIsNone(self.update(1.0, direction=(0, 0, 0)))

    def test_opposite_direction_never_produces_a_zero_length_ray(self) -> None:
        self.update(0.0, direction=(1, 0, 0))
        unused_origin, direction = self.update(0.01, direction=(-1, 0, 0))
        self.assertEqual(direction, (-1.0, 0.0, 0.0))

    def test_configuration_rejects_invalid_filter_constants(self) -> None:
        for name in ("origin_tau_s", "direction_tau_s", "fast_direction_tau_s", "deadband_rad", "max_gap_s"):
            for invalid in (0, -0.1, math.nan, math.inf, True, "0.1"):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    RAY.PointRayFilter(**{name: invalid})
        for settings in (
            {"fast_direction_tau_s": 0.2},
            {"deadband_rad": 0.5},
            {"adaptation_start_rad": 0.5},
            {"adaptation_end_rad": math.pi},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                RAY.PointRayFilter(**settings)


if __name__ == "__main__":
    unittest.main()
