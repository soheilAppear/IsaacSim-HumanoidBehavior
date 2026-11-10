# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check real gaze/stick dispatch, handoffs, preview and lifecycle interlocks."""

from __future__ import annotations

import math
import types
import unittest
from unittest.mock import patch

import numpy as np
import test_gaze_integration_controls as gaze_tests
from humanoid_test_support import HUMANOID
from test_controller_ownership_controls import StickDevice


class TestHybridIntegrationControls(unittest.TestCase):
    def make_example(self, preview=False, **overrides):
        options = {
            "G1_GAZE_MODE": "hold",
            "G1_GAZE_DWELL_S": "0",
            "G1_GAZE_CONTROLLER_OVERRIDE": "1",
            "G1_GAZE_FORWARD_SPEED": "0.6",
            "G1_GAZE_BACKWARD_SPEED": "0.35",
            "G1_GAZE_YAW_SPEED": "0.4",
        }
        options.update(overrides)
        return gaze_tests.TestGazeIntegrationControls.make_example(self, preview, **options)

    def setUp(self) -> None:
        gaze_tests.TestGazeIntegrationControls.setUp(self)
        self.left, self.right = StickDevice(), StickDevice()
        self.present = True
        self.ex._read_tracked_controller_stick = types.MethodType(
            HUMANOID.HumanoidExample._read_tracked_controller_stick, self.ex
        )
        self.ex._get_xr_input_device = lambda path: (
            (self.left if path.endswith("left") else self.right) if self.present else None
        )

    def tick(self, target=None):
        self.target = target
        self.time += 0.02
        with patch.object(HUMANOID.time, "perf_counter", return_value=self.time):
            self.ex._update_controller_command(0.005)
        self.assertIsNone(self.ex._articulation_health_fault)
        return self.ex.gaze_locomotion_status()

    def start_gaze(self):
        self.tick()
        status = self.tick("forward")
        self.assertEqual(status["hybrid"]["source"], "gaze")
        self.assertGreater(status["requested_velocity"][0], 0)

    def test_default_hybrid_keeps_manipulation_enabled_and_optout_available(self) -> None:
        self.assertTrue(self.ex._gaze_controller_override_enabled)
        self.assertTrue(self.ex._controller_arm_control_enabled)
        self.assertTrue(self.ex._hand_tracking_arm_control_enabled)
        self.assertTrue(self.ex._finger_control_enabled)
        self.assertFalse(self.make_example(G1_GAZE_CONTROLLER_OVERRIDE="0")._gaze_controller_override_enabled)
        with self.assertRaisesRegex(ValueError, "G1_GAZE_CONTROLLER_OVERRIDE"):
            self.make_example(G1_GAZE_CONTROLLER_OVERRIDE="invalid")

    def test_missing_controllers_do_not_gate_gaze(self) -> None:
        self.present = False
        self.start_gaze()
        self.assertEqual(self.ex._controller_locomotion.command.state, "unavailable")

    def test_right_y_does_not_move_or_take_gaze_ownership(self) -> None:
        self.tick()
        for right_y in (-1.0, 1.0):
            self.right.y = right_y
            status = self.tick()
            self.assertEqual(status["hybrid"]["source"], "gaze")
            np.testing.assert_array_equal(status["requested_velocity"], (0, 0, 0))
        self.assertGreater(self.tick("forward")["requested_velocity"][0], 0)

    def test_lateral_stick_sign_and_partial_throw_use_the_physical_speed_cap(self) -> None:
        self.tick()
        self.assertEqual(self.ex._max_lateral_speed, 0.3)
        for left_x, lateral in ((1.0, -0.3), (-1.0, 0.3), (0.575, -0.15), (-0.575, 0.15)):
            with self.subTest(left_x=left_x):
                self.left.x = left_x
                status = self.tick()
                self.assertEqual(status["hybrid"]["source"], "controller")
                np.testing.assert_allclose(status["requested_velocity"], (0, lateral, 0))

    def test_diagonal_translation_stays_normalized_with_independent_yaw(self) -> None:
        self.tick()
        for forward_sign in (-1, 1):
            for left_x_sign in (-1, 1):
                for right_x_sign in (-1, 1):
                    with self.subTest(forward=forward_sign, left_x=left_x_sign, right_x=right_x_sign):
                        self.left.x, self.left.y, self.right.x = left_x_sign, forward_sign, right_x_sign
                        status = self.tick()
                        forward_cap = 0.6 if forward_sign > 0 else 0.35
                        expected = (
                            forward_sign * forward_cap / math.sqrt(2),
                            -left_x_sign * 0.3 / math.sqrt(2),
                            -right_x_sign * 0.4,
                        )
                        np.testing.assert_allclose(status["requested_velocity"], expected)
                        forward, lateral, _ = status["requested_velocity"]
                        self.assertLessEqual(math.hypot(forward / forward_cap, lateral / 0.3), 1.000001)

    def test_lateral_takeover_and_release_require_fresh_gaze_selection(self) -> None:
        self.start_gaze()
        self.ex._base_command[0] = 0.5
        self.left.x = -1
        status = self.tick("forward")
        self.assertEqual(status["hybrid"]["source"], "controller")
        np.testing.assert_allclose(status["requested_velocity"], (0, 0.3, 0))
        np.testing.assert_array_equal(self.ex._base_command.numpy(), (0, 0, 0))
        self.left.x = 0
        for _ in range(5):
            np.testing.assert_array_equal(self.tick("forward")["requested_velocity"], (0, 0, 0))
        self.tick()
        self.assertAlmostEqual(self.tick("forward")["requested_velocity"][0], 0.6)

    def test_deflected_lateral_axis_requires_neutral_and_blocks_partial_tracking(self) -> None:
        self.left.x = 1
        for _ in range(3):
            status = self.tick("forward")
            self.assertEqual(status["hybrid"]["source"], "none")
            np.testing.assert_array_equal(status["requested_velocity"], (0, 0, 0))
        self.left.x = 0
        self.tick()
        self.left.x = 1
        self.assertLess(self.tick()["requested_velocity"][1], 0)
        self.right.flags = 3
        for _ in range(3):
            status = self.tick("forward")
            self.assertEqual(status["hybrid"]["source"], "none")
            np.testing.assert_array_equal(status["requested_velocity"], (0, 0, 0))
        self.right.flags = 15
        np.testing.assert_array_equal(self.tick()["requested_velocity"], (0, 0, 0))
        self.left.x = 0
        self.tick()
        self.left.x = 1
        self.assertLess(self.tick()["requested_velocity"][1], 0)

    def test_sticks_replace_gaze_and_clear_previous_filter_only_at_handoff(self) -> None:
        self.start_gaze()
        self.ex._base_command[0] = 0.35
        self.left.y, self.right.x, self.left.x = -1.0, 1.0, 1.0
        status = self.tick("forward")
        self.assertEqual(status["hybrid"]["source"], "controller")
        np.testing.assert_allclose(status["requested_velocity"], (-0.35 / math.sqrt(2), -0.3 / math.sqrt(2), -0.4))
        np.testing.assert_array_equal(self.ex._base_command.numpy(), (0, 0, 0))
        self.assertFalse(status["heading"]["active"])
        self.ex._base_command[0] = -0.1
        status = self.tick("forward")
        self.assertFalse(status["hybrid"]["reset_gaze"])
        self.assertAlmostEqual(float(self.ex._base_command[0]), -0.1)

    def test_stick_release_requires_new_gaze_look_away_then_return(self) -> None:
        self.start_gaze()
        self.left.y = -1
        self.tick("forward")
        self.left.y = 0
        for action in ("forward", "turn_left", "back", "turn_right") * 2:
            status = self.tick(action)
            np.testing.assert_array_equal(status["requested_velocity"], (0, 0, 0))
        self.tick()
        self.assertGreater(self.tick("forward")["requested_velocity"][0], 0)

    def test_controller_loss_stops_then_reconnect_requires_neutral(self) -> None:
        self.start_gaze()
        self.left.y = -1
        self.tick("forward")
        self.left.flags = 3
        self.ex._base_command[0] = -0.2
        status = self.tick("forward")
        np.testing.assert_array_equal(status["requested_velocity"], (0, 0, 0))
        np.testing.assert_array_equal(self.ex._base_command.numpy(), (0, 0, 0))
        self.left.flags = 15
        self.assertEqual(self.tick("forward")["hybrid"]["source"], "none")
        self.left.y = 0
        self.tick("forward")
        self.left.y = -1
        self.assertLess(self.tick("forward")["requested_velocity"][0], 0)

    def test_eye_loss_does_not_disable_valid_controller_movement(self) -> None:
        self.tick()
        self.left.y = 1
        self.ex._eye_gaze_tracker.read_control_ray.return_value = None
        status = self.tick()
        self.assertFalse(status["eye_valid"])
        self.assertEqual(status["state"], "controller")
        self.assertGreater(status["requested_velocity"][0], 0)
        self.assertEqual(self.ex._gaze_controls.update.call_args.args[0]["state"], "controller")

    def test_session_loss_stops_sticks_and_clears_the_filter(self) -> None:
        self.tick()
        self.left.y = 1
        self.tick()
        self.ex._base_command[0] = 0.3
        self.ex._eye_gaze_tracker.control_session_ready.return_value = False
        status = self.tick()
        np.testing.assert_array_equal(status["requested_velocity"], (0, 0, 0))
        np.testing.assert_array_equal(self.ex._base_command.numpy(), (0, 0, 0))
        self.ex._eye_gaze_tracker.control_session_ready.return_value = True
        np.testing.assert_array_equal(self.tick()["requested_velocity"], (0, 0, 0))

    def test_preview_reports_sticks_but_all_physical_commands_stay_zero(self) -> None:
        self.ex._gaze_locomotion = self.make_example(preview=True)._gaze_locomotion
        self.tick()
        # Preview uses a stationary profile with a different incidental lateral
        # cap; displayed intent must retain the physical hybrid cap of 0.30 m/s.
        self.ex._max_lateral_speed = 0.5
        self.left.x, self.left.y, self.right.x = 1, -1, 1
        for _ in range(5):
            status = self.tick("forward")
            self.assertEqual(status["state"], "controller_preview")
            np.testing.assert_allclose(
                status["controller_requested_velocity"], (-0.35 / math.sqrt(2), -0.3 / math.sqrt(2), -0.4)
            )
            np.testing.assert_array_equal(status["requested_velocity"], (0, 0, 0))
            np.testing.assert_array_equal(self.ex._base_command.numpy(), (0, 0, 0))

    def test_stick_override_cancels_accepted_step_instead_of_resuming_it(self) -> None:
        self.ex._gaze_locomotion = self.make_example(G1_GAZE_MODE="step")._gaze_locomotion
        self.start_gaze()
        self.left.y = -1
        self.tick("forward")
        self.left.y = 0
        for _ in range(5):
            np.testing.assert_array_equal(self.tick("forward")["requested_velocity"], (0, 0, 0))
        self.assertIsNone(self.ex._gaze_locomotion._action)

    def test_reset_requires_stick_neutral_and_new_gaze_selection(self) -> None:
        self.start_gaze()
        self.left.y = -1
        self.tick("forward")
        self.ex._reset_gesture_locomotion()
        np.testing.assert_array_equal(self.tick("forward")["requested_velocity"], (0, 0, 0))
        self.left.y = 0
        self.tick("forward")
        np.testing.assert_array_equal(self.tick("forward")["requested_velocity"], (0, 0, 0))
        self.tick()
        self.assertGreater(self.tick("forward")["requested_velocity"][0], 0)

    def test_hud_is_published_once_per_stick_tick(self) -> None:
        self.start_gaze()
        self.left.y = -1
        self.ex._gaze_controls.update.reset_mock()
        self.tick("forward")
        self.ex._gaze_controls.update.assert_called_once()


if __name__ == "__main__":
    unittest.main()
