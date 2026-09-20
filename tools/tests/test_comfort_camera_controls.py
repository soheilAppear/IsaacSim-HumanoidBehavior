# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check robot-only VR stabilization independently of Kit and HMD tracking."""

from __future__ import annotations

import importlib.util
import inspect
import math
import statistics
import unittest
from pathlib import Path

from pxr import Gf

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
    / "comfort_camera.py"
)
SPEC = importlib.util.spec_from_file_location("comfort_camera_under_test", MODULE_PATH)
CAMERA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMERA)


class TestComfortCameraControls(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = CAMERA.ComfortCameraFilter()

    def update(self, now_s: float, **overrides: object) -> tuple[Gf.Vec3d, float] | None:
        """Use a still robot unless a test supplies a motion input."""
        values = {"position": (0, 0, 0.8), "yaw": 0.0, "eye_height": 1.22, "now_s": now_s, "moving": True}
        values.update(overrides)
        return self.filter.update(**values)

    def test_initial_pose_uses_calibrated_height_without_a_startup_transition(self) -> None:
        eye, yaw = self.update(0.0, position=(2, 3, 0.8), yaw=math.pi / 2)
        self.assertEqual(eye, Gf.Vec3d(2, 3, 1.22))
        self.assertAlmostEqual(yaw, math.pi / 2)

    def test_body_bob_and_repeated_eye_height_input_do_not_change_calibration(self) -> None:
        self.update(0.0)
        for sample in range(1, 301):
            t = sample / 100.0
            eye, unused_yaw = self.update(t, position=(0, 0, 0.8 + 0.1 * math.sin(t * 12)), eye_height=2.0)
            self.assertEqual(eye[2], 1.22)

    def test_two_hz_gait_oscillation_is_attenuated_on_position_and_heading(self) -> None:
        raw, filtered_xy, filtered_yaw = [], [], []
        for sample in range(1001):
            t = sample / 100.0
            wave = math.sin(2.0 * math.pi * 2.0 * t)
            eye, yaw = self.update(t, position=(0.03 * wave, 0, 0.8), yaw=0.06 * wave)
            if t > 2.0:
                raw.append(wave)
                filtered_xy.append(eye[0] / 0.03)
                filtered_yaw.append(yaw / 0.06)
        self.assertLess(statistics.pstdev(filtered_xy), 0.43 * statistics.pstdev(raw))
        self.assertLess(statistics.pstdev(filtered_yaw), 0.32 * statistics.pstdev(raw))

    def test_command_start_removes_idle_hold_immediately(self) -> None:
        self.update(0.0, moving=False)
        self.update(1.1, moving=False)
        eye, yaw = self.update(1.11, position=(0.02, 0, 0.8), yaw=0.02, moving=True)
        self.assertGreater(eye[0], 0)
        self.assertLess(eye[0], 0.02)
        self.assertGreater(yaw, 0)
        self.assertLess(yaw, 0.02)

    def test_stop_coast_is_followed_before_idle_hold(self) -> None:
        self.update(0.0)
        first, first_yaw = self.update(0.02, position=(0.02, 0, 0.8), yaw=0.02, moving=False)
        second, second_yaw = self.update(0.5, position=(0.03, 0, 0.8), yaw=0.03, moving=False)
        self.assertGreater(first[0], 0)
        self.assertGreater(second[0], first[0])
        self.assertGreater(second_yaw, first_yaw)

    def test_idle_balance_tremor_is_held_after_coasting(self) -> None:
        self.update(0.0, moving=False)
        self.update(1.1, moving=False)
        for sample in range(1, 201):
            wave = math.sin(sample * 0.3)
            eye, yaw = self.update(1.1 + sample / 100.0, position=(0.03 * wave, 0, 0.8), yaw=0.09 * wave, moving=False)
            self.assertEqual(eye, Gf.Vec3d(0, 0, 1.22))
            self.assertEqual(yaw, 0)

    def test_idle_body_displacement_beyond_deadband_is_followed_without_teleporting(self) -> None:
        self.update(0.0, moving=False)
        self.update(1.1, moving=False)
        eye, yaw = self.update(1.12, position=(0.20, 0, 0.8), yaw=0.3, moving=False)
        self.assertGreater(eye[0], 0)
        self.assertLess(eye[0], 0.20)
        self.assertGreater(yaw, 0)
        self.assertLess(yaw, 0.3)

    def test_small_accumulated_idle_drift_eventually_exceeds_held_frame_deadband(self) -> None:
        self.update(0.0, moving=False)
        self.update(1.1, moving=False)
        for sample in range(1, 101):
            eye, unused_yaw = self.update(1.1 + sample / 100.0, position=(sample * 0.001, 0, 0.8), moving=False)
        self.assertGreater(eye[0], 0.05)
        self.assertLessEqual(eye[0], 0.1)

    def test_non_increasing_timestamps_hold_filtered_values(self) -> None:
        self.update(1.0)
        previous = self.update(1.1, position=(1, 0, 0.8), yaw=0.5)
        for timestamp in (1.1, 1.0, 0.0):
            self.assertEqual(self.update(timestamp, position=(10, 10, 1), yaw=1.0), previous)
        self.assertEqual(self.filter._last_time_s, 1.1)

    def test_invalid_samples_hold_previous_result_without_poisoning_time(self) -> None:
        bad = [
            {"position": None},
            {"position": (1, 2)},
            {"position": (1, 2, math.nan)},
            {"position": (True, 2, 3)},
            {"position": "123"},
            {"yaw": math.nan},
            {"yaw": True},
            {"eye_height": "1.2"},
            {"now_s": math.inf},
            {"now_s": -1},
            {"now_s": "1.0"},
            {"moving": 1},
        ]
        for invalid in bad:
            with self.subTest(invalid=invalid):
                self.filter.reset()
                values = {"now_s": 0.0, **invalid}
                self.assertIsNone(self.update(**values))
                previous = self.update(1.0)
                values = {"now_s": 2.0, **invalid}
                self.assertEqual(self.update(**values), previous)
                self.assertEqual(self.filter._last_time_s, 1.0)

    def test_wall_time_response_is_independent_of_callback_rate(self) -> None:
        outputs = []
        for hz in (20, 50, 100, 200):
            self.filter.reset()
            self.update(0.0)
            for sample in range(1, hz + 1):
                output = self.update(sample / hz, position=(1, -1, 0.8), yaw=0.4)
            outputs.append(output)
        for eye, yaw in outputs[1:]:
            self.assertLess((eye - outputs[0][0]).GetLength(), 1e-12)
            self.assertAlmostEqual(yaw, outputs[0][1], places=12)

    def test_long_wall_time_gap_is_capped_instead_of_jumping_to_raw_pose(self) -> None:
        self.update(0.0)
        eye, yaw = self.update(20.0, position=(1, 0, 0.8), yaw=1.0)
        self.assertAlmostEqual(eye[0], 1.0 - math.exp(-0.1 / 0.18))
        self.assertAlmostEqual(yaw, 1.0 - math.exp(-0.1 / 0.25))

    def test_heading_wrap_uses_shortest_direction(self) -> None:
        self.update(0.0, yaw=math.radians(179))
        unused_eye, yaw = self.update(0.02, yaw=math.radians(-179))
        self.assertGreater(yaw, math.radians(179))
        self.assertLess(yaw, math.pi)

    def test_explicit_height_recalibration_transitions_without_gait_following(self) -> None:
        self.update(0.0)
        self.assertTrue(self.filter.set_eye_height(1.32))
        eye, unused_yaw = self.update(0.1, position=(0, 0, 20), eye_height=5)
        self.assertGreater(eye[2], 1.22)
        self.assertLess(eye[2], 1.32)
        self.assertAlmostEqual(eye[2], 1.22 + 0.1 * (1 - math.exp(-0.1 / 0.3)))
        for sample in range(2, 31):
            eye, unused_yaw = self.update(sample / 10.0, eye_height=5)
        self.assertAlmostEqual(eye[2], 1.32, delta=0.00001)

    def test_invalid_eye_height_calibration_preserves_previous_target(self) -> None:
        self.update(0.0)
        for invalid in (None, True, "1.2", math.nan, math.inf):
            self.assertFalse(self.filter.set_eye_height(invalid))
        self.assertEqual(self.update(0.1)[0][2], 1.22)

    def test_reset_clears_idle_pose_time_and_eye_calibration(self) -> None:
        self.update(10.0, moving=False)
        self.filter.set_eye_height(4.0)
        self.filter.reset()
        eye, yaw = self.update(0.0, position=(2, 3, 0.8), yaw=1.0, eye_height=1.4)
        self.assertEqual(eye, Gf.Vec3d(2, 3, 1.4))
        self.assertEqual(yaw, 1.0)

    def test_returned_position_cannot_mutate_filter_history(self) -> None:
        eye, unused_yaw = self.update(0.0)
        eye[0] = 99
        self.assertEqual(self.update(0.0)[0], Gf.Vec3d(0, 0, 1.22))

    def test_interface_cannot_filter_hmd_rotation_or_body_roll_pitch(self) -> None:
        self.assertEqual(
            set(inspect.signature(self.filter.update).parameters), {"position", "yaw", "eye_height", "now_s", "moving"}
        )

    def test_invalid_tunings_are_rejected(self) -> None:
        for name in inspect.signature(CAMERA.ComfortCameraFilter).parameters:
            for invalid in (None, 0, -1, True, "0.1", math.nan, math.inf):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    CAMERA.ComfortCameraFilter(**{name: invalid})


if __name__ == "__main__":
    unittest.main()
