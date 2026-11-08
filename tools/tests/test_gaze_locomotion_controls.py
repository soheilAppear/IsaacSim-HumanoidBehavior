# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise gaze dwell, measured movement, and stopping without Kit or XR hardware."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
    / "gaze_locomotion.py"
)
SPEC = importlib.util.spec_from_file_location("gaze_locomotion_under_test", MODULE_PATH)
GAZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GAZE
SPEC.loader.exec_module(GAZE)


class TestGazeLocomotionControls(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GAZE.GazeLocomotionConfig(dwell_s=0.3, preview=False)
        self.controller = GAZE.GazeLocomotion(self.config)
        self.time = 0.0
        self.pose = (0.0, 0.0, 0.0)

    def tick(self, target: str | None = None, *, wall_dt: float = 0.1, **kwargs):
        """Supply deterministic wall/simulation clocks and a measured planar pose."""
        self.time += wall_dt
        values = dict(
            target=target,
            eye_valid=True,
            now_s=self.time,
            dt=0.05,
            pose=self.pose,
            linear_speed_m_s=0.0,
            yaw_speed_rad_s=0.0,
        )
        values.update(kwargs)
        return self.controller.update(**values)

    def fire(self, action: str = "forward"):
        """Look away once, then complete exactly one continuous dwell."""
        self.tick()
        result = None
        for _ in range(4):
            result = self.tick(action)
        self.assertEqual(result.activated_action, action)
        return result

    def settle(self, target: str | None = None):
        """Observe one complete low-speed interval after zero was requested."""
        result = None
        for _ in range(4):
            result = self.tick(target)
        return result

    def test_default_configuration_is_preview_with_requested_small_movements(self) -> None:
        config = GAZE.GazeLocomotionConfig()
        self.assertTrue(config.preview)
        self.assertEqual((config.dwell_s, config.distance_m, config.turn_degrees), (0.7, 0.2, 10))
        self.assertEqual(GAZE.GazeLocomotion().command.velocity, (0, 0, 0))

    def test_configuration_rejects_nonfinite_boolean_and_out_of_range_values(self) -> None:
        ranges = {
            "dwell_s": (0.09, 5.01),
            "distance_m": (0.009, 1.01),
            "turn_degrees": (0.99, 90.01),
            "forward_speed": (0, 1.01),
            "backward_speed": (-0.01, 1.01),
            "yaw_speed": (0, 1.51),
            "max_gaze_gap_s": (0.049, 1.01),
        }
        for field, invalid in ranges.items():
            for value in (*invalid, math.nan, math.inf, -math.inf, True, False, None, "1"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(self.config, **{field: value})
        for value in (0, 1, "false", None):
            with self.subTest(preview=value), self.assertRaises(ValueError):
                replace(self.config, preview=value)

    def test_startup_requires_valid_look_outside_all_buttons(self) -> None:
        for _ in range(8):
            result = self.tick("forward")
            self.assertEqual(result.state, "awaiting_release")
            self.assertEqual(result.activation_count, 0)
        self.tick(eye_valid=False)
        self.assertEqual(self.tick("forward").state, "awaiting_release")
        self.assertEqual(self.tick().state, "idle")
        self.assertEqual(self.tick("forward").progress, 0)

    def test_dwell_uses_wall_time_and_first_sample_has_zero_progress(self) -> None:
        self.tick()
        self.assertEqual(self.tick("forward", dt=0.2).progress, 0)
        result = self.tick("forward", wall_dt=0.15, dt=0.001)
        self.assertAlmostEqual(result.progress, 0.5)
        self.assertEqual(self.tick("forward", wall_dt=0, dt=0.2).progress, result.progress)
        result = self.tick("forward", wall_dt=0.15, dt=0.001)
        self.assertEqual(result.state, "active")
        self.assertEqual(result.activation_count, 1)

    def test_target_change_discards_partial_dwell(self) -> None:
        self.tick()
        self.tick("forward")
        self.tick("forward")
        result = self.tick("turn_left")
        self.assertEqual(result.progress, 0)
        self.assertEqual(result.target, "turn_left")
        self.tick("turn_left")
        self.assertIsNone(self.tick("turn_left").activated_action)
        self.assertEqual(self.tick("turn_left").activated_action, "turn_left")

    def test_look_away_before_threshold_cancels_without_activation(self) -> None:
        self.tick()
        self.tick("forward")
        self.tick("forward")
        result = self.tick()
        self.assertEqual((result.state, result.activation_count), ("idle", 0))
        self.assertEqual(self.tick("forward").progress, 0)

    def test_invalid_eye_sample_clears_partial_dwell(self) -> None:
        self.tick()
        self.tick("forward")
        self.tick("forward")
        self.assertEqual(self.tick("forward", eye_valid=False).state, "unavailable")
        self.assertEqual(self.tick("forward").progress, 0)

    def test_long_gap_and_clock_reversal_do_not_complete_dwell(self) -> None:
        for wall_dt in (0.3, -0.05):
            with self.subTest(wall_dt=wall_dt):
                self.setUp()
                self.tick()
                self.tick("forward")
                self.tick("forward")
                result = self.tick("forward", wall_dt=wall_dt)
                self.assertEqual(result.activation_count, 0)
                self.assertEqual(result.velocity, (0, 0, 0))
                self.assertEqual(self.tick("forward").progress, 0)

    def test_preview_fires_once_and_never_needs_robot_motion_measurements(self) -> None:
        self.controller = GAZE.GazeLocomotion(replace(self.config, preview=True))
        self.pose = None
        for action in GAZE.ACTIONS:
            result = self.fire(action)
            self.assertEqual(result.state, "preview")
            self.assertIn("Would ", result.detail)
            self.assertEqual(result.velocity, (0, 0, 0))
            count = result.activation_count
            for _ in range(10):
                result = self.tick(action, linear_speed_m_s=math.nan, yaw_speed_rad_s=math.nan)
                self.assertEqual(result.state, "preview")
                self.assertEqual((result.target, result.progress), (action, 1.0))
                self.assertEqual(result.velocity, (0, 0, 0))
                self.assertIsNone(result.activated_action)
                self.assertEqual(result.activation_count, count)

    def test_tracking_loss_and_unknown_button_do_not_rearm_completed_preview(self) -> None:
        self.controller = GAZE.GazeLocomotion(replace(self.config, preview=True))
        self.fire()
        self.tick(eye_valid=False)
        self.tick("unknown")
        for _ in range(6):
            result = self.tick("turn_left")
            self.assertEqual(result.activation_count, 1)
        self.assertEqual(self.fire("turn_left").activation_count, 2)

    def test_zero_backward_speed_disables_reverse_with_visible_reason(self) -> None:
        self.controller = GAZE.GazeLocomotion(replace(self.config, backward_speed=0))
        self.tick()
        for _ in range(10):
            result = self.tick("back")
            self.assertEqual(result.activation_count, 0)
            self.assertEqual(result.velocity, (0, 0, 0))
            self.assertIn("disabled", result.detail)
        self.assertEqual(self.fire("turn_right").velocity, (0, 0, -self.config.yaw_speed))

    def test_forward_goal_uses_initial_heading_projected_displacement(self) -> None:
        self.pose = (4.0, -2.0, math.pi / 2)
        self.assertEqual(self.fire().velocity, (0.2, 0, 0))
        self.pose = (4.5, -1.9, math.pi / 2)
        result = self.tick("forward")
        self.assertEqual(result.state, "active")
        self.assertAlmostEqual(result.progress, 0.5)
        self.pose = (4.5, -1.8, math.pi / 2)
        result = self.tick("forward")
        self.assertEqual(result.state, "stopping")
        self.assertEqual(result.velocity, (0, 0, 0))

    def test_backward_goal_has_negative_velocity_and_signed_progress(self) -> None:
        self.assertEqual(self.fire("back").velocity, (-0.15, 0, 0))
        self.pose = (0.1, 0.0, 0.0)
        self.assertEqual(self.tick().progress, 0)
        self.pose = (-0.1, 0.0, 0.0)
        self.assertAlmostEqual(self.tick().progress, 0.5)
        self.pose = (-0.2, 0.0, 0.0)
        self.assertEqual(self.tick().state, "stopping")

    def test_left_turn_accumulates_yaw_across_positive_pi_boundary(self) -> None:
        self.pose = (0.0, 0.0, math.radians(179))
        self.assertEqual(self.fire("turn_left").velocity, (0, 0, 0.25))
        self.pose = (0.0, 0.0, math.radians(-176))
        self.assertAlmostEqual(self.tick().progress, 0.5)
        self.pose = (0.0, 0.0, math.radians(-171))
        self.assertEqual(self.tick().state, "stopping")

    def test_right_turn_accumulates_negative_yaw_across_negative_pi_boundary(self) -> None:
        self.pose = (0.0, 0.0, math.radians(-179))
        self.assertEqual(self.fire("turn_right").velocity, (0, 0, -0.25))
        self.pose = (0.0, 0.0, math.radians(176))
        self.assertAlmostEqual(self.tick().progress, 0.5)
        self.pose = (0.0, 0.0, math.radians(171))
        self.assertEqual(self.tick().state, "stopping")

    def test_wrong_direction_turn_does_not_count_toward_goal(self) -> None:
        self.fire("turn_left")
        self.pose = (0.0, 0.0, -0.2)
        result = self.tick()
        self.assertEqual((result.state, result.progress), ("active", 0))
        self.pose = (0.0, 0.0, 0.1)
        self.assertAlmostEqual(self.tick().progress, 0.1 / math.radians(10))

    def test_active_motion_ignores_look_away_eye_loss_and_new_button_dwell(self) -> None:
        self.fire()
        for index in range(8):
            self.pose = (0.01 * index, 0.0, 0.0)
            result = self.tick("turn_left" if index % 2 else None, eye_valid=index % 3 != 0, wall_dt=0.4)
            self.assertEqual(result.state, "active")
            self.assertEqual(result.velocity, (0.2, 0, 0))
            self.assertEqual(result.activation_count, 1)
            self.assertIsNone(result.activated_action)

    def test_look_away_during_active_or_stopping_cannot_queue_next_action(self) -> None:
        self.fire()
        self.tick()
        self.pose = (0.2, 0.0, 0.0)
        self.assertEqual(self.tick().state, "stopping")
        self.assertEqual(self.settle().state, "awaiting_release")
        for _ in range(6):
            result = self.tick("turn_left")
            self.assertEqual(result.state, "awaiting_release")
            self.assertEqual(result.activation_count, 1)
        self.assertEqual(self.fire("turn_left").activation_count, 2)

    def test_stopping_requires_both_measured_speeds_to_remain_low(self) -> None:
        self.fire()
        self.pose = (0.2, 0.0, 0.0)
        self.tick()
        self.tick(linear_speed_m_s=0.04)
        self.tick()
        self.tick()
        self.assertEqual(self.tick(yaw_speed_rad_s=-0.07).state, "stopping")
        for _ in range(3):
            self.assertEqual(self.tick().state, "stopping")
        self.assertEqual(self.tick().state, "awaiting_release")

    def test_failure_to_settle_latches_fault_even_after_valid_look_away(self) -> None:
        self.fire()
        self.pose = (0.2, 0.0, 0.0)
        self.tick()
        for _ in range(math.ceil(self.controller.SETTLE_TIMEOUT_S / 0.05)):
            result = self.tick(linear_speed_m_s=0.1)
            self.assertEqual(result.velocity, (0, 0, 0))
        self.assertEqual(result.state, "fault")
        self.assertIn("settle", result.detail)
        self.assertEqual(self.tick().state, "fault")
        self.controller.reset()
        self.assertEqual(self.tick("forward").state, "awaiting_release")
        self.assertEqual(self.fire().state, "active")

    def test_stalled_movement_times_out_in_simulation_time_then_settles(self) -> None:
        self.fire()
        for _ in range(39):
            self.assertEqual(self.tick(wall_dt=0.001, dt=0.1).state, "active")
        result = self.tick(wall_dt=0.001, dt=0.1)
        self.assertEqual(result.state, "stopping")
        self.assertIn("timed out", result.detail)
        self.assertEqual(result.velocity, (0, 0, 0))
        self.assertEqual(self.settle("forward").state, "awaiting_release")

    def test_slow_rendering_does_not_spend_simulation_motion_timeout(self) -> None:
        self.fire()
        for _ in range(10):
            result = self.tick(wall_dt=2.0, dt=0.01, eye_valid=False)
            self.assertEqual(result.state, "active")
            self.assertEqual(result.velocity, (0.2, 0, 0))

    def test_readiness_loss_cancels_motion_and_requires_fresh_look_away(self) -> None:
        self.fire()
        result = self.tick("forward", ready=False)
        self.assertEqual((result.state, result.velocity), ("unavailable", (0, 0, 0)))
        for _ in range(6):
            self.assertEqual(self.tick("forward").state, "awaiting_release")
        self.assertEqual(self.fire("back").activation_count, 2)

    def test_reset_drops_motion_and_previous_selection_history(self) -> None:
        self.fire()
        self.controller.reset()
        result = self.controller.command
        self.assertEqual((result.velocity, result.activation_count, result.last_action), ((0, 0, 0), 0, None))
        self.assertEqual(self.tick("forward").state, "awaiting_release")

    def test_invalid_motion_measurements_fault_an_active_action(self) -> None:
        invalid_cases = (
            {"pose": None},
            {"pose": (0, math.nan, 0)},
            {"pose": (0, 0)},
            {"linear_speed_m_s": math.nan},
            {"yaw_speed_rad_s": True},
        )
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid):
                self.setUp()
                self.fire()
                result = self.tick(**invalid)
                self.assertEqual(result.state, "fault")
                self.assertEqual(result.velocity, (0, 0, 0))
                self.assertEqual(self.tick().state, "fault")

    def test_bad_motion_data_cannot_begin_live_dwell(self) -> None:
        for _ in range(6):
            result = self.tick("forward", pose=None)
            self.assertEqual((result.state, result.activation_count), ("unavailable", 0))
        self.assertEqual(self.tick("forward").state, "awaiting_release")

    def test_invalid_clock_or_timestep_cancels_selection_and_faults_motion(self) -> None:
        for invalid in ({"now_s": math.nan}, {"now_s": -1}, {"dt": 0}, {"dt": True}, {"dt": math.inf}):
            with self.subTest(invalid=invalid):
                self.setUp()
                self.tick()
                self.tick("forward")
                self.assertEqual(self.tick("forward", **invalid).state, "unavailable")
                self.assertEqual(self.tick("forward").state, "awaiting_release")
                self.fire()
                result = self.tick(**invalid)
                self.assertEqual((result.state, result.velocity), ("fault", (0, 0, 0)))

    def test_public_status_is_immutable_and_selection_requires_no_hand_inputs(self) -> None:
        result = self.fire()
        self.assertEqual(result.last_action, "forward")
        with self.assertRaises(AttributeError):
            result.velocity = (1, 0, 0)
        # The entire selection sequence supplied gaze and base measurements only.
        self.assertEqual(result.activation_count, 1)


if __name__ == "__main__":
    unittest.main()
