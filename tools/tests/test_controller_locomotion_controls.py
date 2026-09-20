# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check joystick-only intent without importing Kit, XR, or the robot actor."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
    / "controller_locomotion.py"
)
SPEC = importlib.util.spec_from_file_location("controller_locomotion_under_test", MODULE_PATH)
CONTROLLER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTROLLER
SPEC.loader.exec_module(CONTROLLER)


class TestControllerLocomotionControls(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = CONTROLLER.ControllerLocomotion()
        self.time = 0.0

    def update(self, **kwargs):
        self.time += 0.01
        values = dict(left_x=0.0, left_y=0.0, right_x=0.0, left_valid=True, right_valid=True, now_s=self.time)
        values.update(kwargs)
        return self.controller.update(**values)

    def test_nonzero_startup_requires_neutral_before_moving(self) -> None:
        for _ in range(5):
            command = self.update(left_y=1.0)
            self.assertEqual(command.state, "awaiting_neutral")
            self.assertEqual(command.velocity, (0, 0, 0))
        self.assertEqual(self.update().state, "idle")
        self.assertEqual(self.update(left_y=1).velocity, (1, 0, 0))

    def test_confirmed_layout_has_correct_forward_reverse_lateral_and_turn_signs(self) -> None:
        self.update()
        for axes, expected in (
            ({"left_y": 1}, (1, 0, 0)),
            ({"left_y": -1}, (-1, 0, 0)),
            ({"left_x": 1}, (0, -1, 0)),
            ({"left_x": -1}, (0, 1, 0)),
            ({"right_x": 1}, (0, 0, -1)),
            ({"right_x": -1}, (0, 0, 1)),
        ):
            with self.subTest(axes=axes):
                self.assertEqual(self.update(**axes).velocity, expected)

    def test_axis_deadzone_removes_noise_and_rescales_remaining_travel(self) -> None:
        command = self.update(left_x=-0.15, left_y=0.15, right_x=0.149)
        self.assertEqual(command.state, "idle")
        self.assertEqual(command.velocity, (0, 0, 0))
        self.assertAlmostEqual(self.update(left_y=0.575).velocity[0], 0.5)
        self.assertAlmostEqual(self.update(left_y=-0.575).velocity[0], -0.5)

    def test_diagonal_translation_is_bounded_without_suppressing_turn(self) -> None:
        self.update()
        forward, lateral, yaw = self.update(left_x=1, left_y=1, right_x=-1).velocity
        self.assertAlmostEqual(math.hypot(forward, lateral), 1)
        self.assertAlmostEqual(forward, -lateral)
        self.assertEqual(yaw, 1)

    def test_stick_release_returns_exact_zero_without_a_filter_tail(self) -> None:
        self.update()
        self.update(left_y=1, right_x=1)
        command = self.update()
        self.assertEqual(command.state, "idle")
        self.assertEqual(command.velocity, (0, 0, 0))

    def test_loss_of_either_device_stops_all_axes_and_requires_rearming(self) -> None:
        for invalid in ("left_valid", "right_valid"):
            self.update()
            self.update(left_y=1, right_x=-1)
            command = self.update(left_y=1, right_x=-1, **{invalid: False})
            self.assertEqual(command.state, "unavailable")
            self.assertEqual(command.velocity, (0, 0, 0))
            self.assertEqual(self.update(left_y=1).state, "awaiting_neutral")
            self.update()
            self.assertEqual(self.update(left_y=1).state, "active")

    def test_invalid_axes_stop_and_cannot_rearm_from_untrusted_zero(self) -> None:
        for value in (math.nan, math.inf, -math.inf, 1.01, -1.01, True, None, "0"):
            self.update()
            command = self.update(left_y=value)
            self.assertEqual(command.state, "unavailable")
            self.assertEqual(command.velocity, (0, 0, 0))
            self.assertEqual(self.update(left_y=1).state, "awaiting_neutral")

    def test_reset_and_long_pause_do_not_resume_a_held_stick(self) -> None:
        self.update()
        self.update(left_y=1)
        self.controller.reset()
        self.assertEqual(self.update(left_y=1).state, "awaiting_neutral")
        self.update()
        self.update(left_y=1)
        self.time += 0.5
        self.assertEqual(self.update(left_y=1).state, "awaiting_neutral")
        self.assertEqual(self.update(left_y=1).velocity, (0, 0, 0))
        self.update()
        self.assertEqual(self.update(left_y=1).state, "active")

    def test_repeated_frame_time_is_allowed_but_backwards_time_stops(self) -> None:
        self.update(now_s=1)
        self.assertEqual(self.update(left_y=1, now_s=1).state, "active")
        self.assertEqual(self.update(left_y=1, now_s=0.99).state, "awaiting_neutral")
        self.assertEqual(self.update(left_y=1, now_s=1).velocity, (0, 0, 0))

    def test_invalid_clock_and_validity_flags_fail_closed(self) -> None:
        for values in ({"now_s": math.nan}, {"now_s": -1}, {"left_valid": 1}, {"right_valid": None}):
            self.update()
            self.assertEqual(self.update(left_y=1, **values).state, "unavailable")

    def test_optional_carry_scale_only_reduces_commands(self) -> None:
        self.update()
        command = self.update(left_y=1, right_x=-1, speed_scale=0.3)
        self.assertEqual(command.velocity, (0.3, 0, 0.3))
        self.assertEqual(self.update(left_y=1, speed_scale=0).velocity, (0, 0, 0))
        self.assertEqual(self.update(left_y=1).state, "awaiting_neutral")
        for value in (1.1, -0.1, math.nan, True, "1"):
            self.assertEqual(self.update(speed_scale=value).state, "unavailable")

    def test_invalid_configuration_is_rejected(self) -> None:
        for value in (-0.01, 1, math.nan, True, "0.15"):
            with self.assertRaises(ValueError):
                CONTROLLER.ControllerLocomotion(deadzone=value)
        for value in (0, -1, math.inf, True, "0.25"):
            with self.assertRaises(ValueError):
                CONTROLLER.ControllerLocomotion(max_sample_gap_s=value)


if __name__ == "__main__":
    unittest.main()
