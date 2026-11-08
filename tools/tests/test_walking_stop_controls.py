# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check support-based physical-stop commands without simulator dependencies."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
    / "walking_stop.py"
)
SPEC = importlib.util.spec_from_file_location("walking_stop_under_test", MODULE_PATH)
STOP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STOP)


class TestWalkingStopControls(unittest.TestCase):
    def setUp(self) -> None:
        self.regulator = STOP.WalkingStopRegulator()

    def tick(self, **overrides):
        values = dict(
            dt=0.05,
            operator_command=(0.0, 0.0, 0.0),
            filtered_command=(0.0, 0.0, 0.0),
            support_pose=(0.0, 0.0, 0.0),
            yaw=0.0,
        )
        values.update(overrides)
        return self.regulator.update(**values)

    def prime(self, pose=(0.0, 0.0, 0.0)):
        for _ in range(21):
            self.tick(support_pose=pose)
        self.assertEqual(self.regulator.status["support_anchor"], pose)

    def hold(self, pose, **kwargs):
        for _ in range(30):
            result = self.tick(support_pose=pose, **kwargs)
        return result

    def test_unassisted_braking_captures_delayed_support_anchor(self):
        for index in range(1, 20):
            pose = (index * 0.005, 0.0, 0.0)
            self.assertEqual(self.tick(support_pose=pose), (0.0, 0.0, 0.0))
            self.assertIsNone(self.regulator.status["support_anchor"])
        self.tick(support_pose=(0.1, 0.0, 0.0))
        self.assertEqual(self.regulator.status["support_anchor"], (0.1, 0.0, 0.0))
        self.hold((0.12, 0.0, 0.0))
        self.assertEqual(self.regulator.status["support_anchor"], (0.1, 0.0, 0.0))

    def test_nonzero_intent_passes_operator_filter_and_resets_anchor(self):
        self.prime()
        self.assertLess(self.hold((0.1, 0.0, 0.0))[0], 0.0)
        for requested in ((0.4, 0, 0), (0, -0.2, 0), (0, 0, 0.4)):
            filtered = tuple(v * 0.5 for v in requested)
            self.assertEqual(
                self.tick(operator_command=requested, filtered_command=filtered, support_pose=None, yaw=math.nan),
                filtered,
            )
            self.assertIsNone(self.regulator.status["support_anchor"])
            self.assertFalse(self.regulator.status["active"])
            self.assertEqual(self.regulator.status["elapsed_s"], 0.0)
        self.prime((0.1, 0.0, 0.0))
        self.assertEqual(self.tick(support_pose=(0.1, 0.0, 0.0)), (0.0, 0.0, 0.0))

    def test_filtered_command_tail_does_not_capture_an_early_anchor(self):
        for tail in ((0.021, 0, 0), (0.015, 0.015, 0), (0, 0, -0.021)):
            self.assertEqual(self.tick(filtered_command=tail), tail)
            self.assertEqual(self.regulator.status["phase"], "command_tail")
            self.assertIsNone(self.regulator.status["support_anchor"])
        self.prime()
        self.assertLess(self.hold((0.1, 0, 0), filtered_command=(0.019, 0, 0))[0], 0.0)
        self.assertEqual(self.regulator.status["effective"], self.regulator.status["correction"])

    def test_stationary_feet_produce_zero_despite_changing_root_heading(self):
        self.prime()
        for index in range(100):
            self.assertEqual(self.tick(yaw=0.3 * math.sin(index * 0.4)), (0.0, 0.0, 0.0))
        self.assertEqual(self.regulator.status["filtered_world_velocity"], (0.0, 0.0))

    def test_support_position_error_remains_after_velocity_returns_to_zero(self):
        self.prime()
        result = self.hold((0.1, 0.0, 0.0))
        self.assertAlmostEqual(self.regulator.status["filtered_world_velocity"][0], 0.0)
        self.assertAlmostEqual(result[0], -1.5 * (0.1 - 0.02), places=4)
        self.assertEqual(self.regulator.status["support_anchor"], (0.0, 0.0, 0.0))

    def test_support_velocity_window_interpolates_irregular_simulation_steps(self):
        elapsed = 0.0
        for dt in (0.07, 0.09, 0.13, 0.06, 0.12, 0.08, 0.11, 0.14, 0.15, 0.13, 0.12):
            elapsed += dt
            self.tick(dt=dt, support_pose=(0.1 * elapsed, -0.05 * elapsed, 0.2 * elapsed))
        status = self.regulator.status
        self.assertAlmostEqual(status["filtered_world_velocity"][0], 0.1)
        self.assertAlmostEqual(status["filtered_world_velocity"][1], -0.05)
        self.assertAlmostEqual(status["filtered_yaw_rate"], 0.2)
        self.assertLess(len(self.regulator._samples), 8)

    def test_correction_rotates_world_support_error_into_body_axes(self):
        for yaw, expected in ((0, (-0.2, 0)), (math.pi / 2, (0, 0.2)), (-math.pi / 2, (0, -0.2)), (math.pi, (0.2, 0))):
            with self.subTest(yaw=yaw):
                self.setUp()
                self.prime()
                result = self.hold((1.0, 0.0, 0.0), yaw=yaw)
                for actual, wanted in zip(result[:2], expected):
                    self.assertAlmostEqual(actual, wanted, places=4)

    def test_vector_and_yaw_caps_bound_filtered_corrections(self):
        for sign in (-1, 1):
            self.setUp()
            self.prime()
            result = self.hold((sign, sign, sign))
            self.assertLessEqual(math.hypot(*result[:2]), 0.2 + 1e-12)
            self.assertGreater(math.hypot(*result[:2]), 0.19)
            self.assertAlmostEqual(result[0], result[1])
            self.assertLess(result[0] * sign, 0)
            self.assertLessEqual(abs(result[2]), 0.3)
            self.assertLess(result[2] * sign, -0.29)

    def test_yaw_wrap_preserves_short_error_and_signed_sole_velocity(self):
        self.prime((0.0, 0.0, math.radians(179)))
        result = self.hold((0.0, 0.0, math.radians(-171)))
        self.assertAlmostEqual(self.regulator.status["heading_error_rad"], -math.radians(10))
        self.assertAlmostEqual(self.regulator.status["filtered_yaw_rate"], 0.0)
        self.assertLess(result[2], 0.0)
        self.assertGreater(result[2], -0.3)

    def test_slow_small_support_changes_stay_inside_continuous_deadbands(self):
        self.prime()
        for index in range(1, 21):
            pose = (0.01 * index / 20, 0.0, 0.02 * index / 20)
            self.assertEqual(self.tick(support_pose=pose), (0.0, 0.0, 0.0))
        self.assertEqual(self.hold((0.01, 0.0, 0.02)), (0.0, 0.0, 0.0))
        result = self.hold((0.021, 0.0, 0.051))
        self.assertLess(result[0], 0.0)
        self.assertLess(result[2], 0.0)
        self.assertLess(max(abs(v) for v in result), 0.003)

    def test_lifecycle_and_invalid_input_clear_prior_feedback(self):
        invalid = (
            {"dt": 0},
            {"dt": math.nan},
            {"operator_command": (False, 0, 0)},
            {"filtered_command": (0, math.inf, 0)},
            {"support_pose": None},
            {"support_pose": (0, 0)},
            {"support_pose": (0, 0, math.inf)},
            {"yaw": True},
            {"ready": 1},
        )
        for values in invalid:
            self.setUp()
            self.prime()
            self.hold((0.1, 0.0, 0.0))
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.tick(**values)
            self.assertEqual(self.regulator.status["effective"], (0.0, 0.0, 0.0))
            self.assertIsNone(self.regulator.status["support_anchor"])
        self.prime()
        self.hold((0.1, 0.0, 0.0))
        self.assertEqual(self.tick(ready=False, support_pose=None), (0.0, 0.0, 0.0))
        self.assertIsNone(self.regulator.status["support_anchor"])

    def test_configuration_validation_and_defensive_status(self):
        for field in (
            "linear_gain",
            "yaw_gain",
            "linear_damping",
            "yaw_damping",
            "braking_s",
            "window_s",
            "filter_s",
            "max_linear_speed",
            "max_yaw_speed",
            "position_deadband",
            "heading_deadband",
            "linear_deadband",
            "yaw_deadband",
            "tail_threshold",
        ):
            for value in (-0.1, math.nan, math.inf, True, None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    STOP.WalkingStopRegulator(**{field: value})
        for field in ("filter_s", "window_s"):
            with self.assertRaises(ValueError):
                STOP.WalkingStopRegulator(**{field: 0})
        self.prime()
        result = self.hold((0.1, 0.0, 0.0))
        status = self.regulator.status
        self.assertEqual(status["requested"], (0.0, 0.0, 0.0))
        self.assertEqual(status["filtered"], (0.0, 0.0, 0.0))
        self.assertEqual(status["correction"], result)
        status["support_anchor"] = (9, 9, 9)
        self.assertEqual(self.regulator.status["support_anchor"], (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
