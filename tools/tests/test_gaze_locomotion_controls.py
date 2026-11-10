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
        self.assertEqual(config.control_mode, "step")
        self.assertEqual((config.dwell_s, config.distance_m, config.turn_degrees), (0.7, 0.2, 10))
        self.assertEqual((config.heading_hold, config.heading_gain, config.heading_max_yaw_speed), (True, 3.0, 0.15))
        self.assertEqual(GAZE.GazeLocomotion().command.velocity, (0, 0, 0))

    def test_configuration_rejects_nonfinite_boolean_and_out_of_range_values(self) -> None:
        ranges = {
            "dwell_s": (-0.01, 5.01),
            "distance_m": (0.009, 1.01),
            "turn_degrees": (0.99, 90.01),
            "forward_speed": (0, 1.01),
            "backward_speed": (-0.01, 1.01),
            "yaw_speed": (0, 1.51),
            "max_gaze_gap_s": (0.049, 1.01),
            "heading_gain": (-0.01, 5.01),
            "heading_max_yaw_speed": (-0.01, 0.401),
        }
        for field, invalid in ranges.items():
            for value in (*invalid, math.nan, math.inf, -math.inf, True, False, None, "1"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(self.config, **{field: value})
        for value in (0, 1, "false", None):
            with self.subTest(preview=value), self.assertRaises(ValueError):
                replace(self.config, preview=value)
            with self.subTest(heading_hold=value), self.assertRaises(ValueError):
                replace(self.config, heading_hold=value)
        for value in (None, True, 0, "continuous", "HOLD", ""):
            with self.subTest(control_mode=value), self.assertRaises(ValueError):
                replace(self.config, control_mode=value)

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
            # Keep pose and velocity consistent: continuing translation must not
            # be mistaken for bounded standing sway after the zero request.
            self.pose = (self.pose[0] + 0.005, 0.0, 0.0)
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

    def test_hold_keeps_each_direction_active_beyond_step_goal_and_timeout(self) -> None:
        for action in GAZE.ACTIONS:
            with self.subTest(action=action):
                self.setUp()
                self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode="hold"))
                result = self.fire(action)
                expected_velocity = result.velocity
                for index in range(150):
                    self.pose = (index * 0.02, 0.0, index * 0.03 if action.startswith("turn_") else 0.0)
                    result = self.tick(action, dt=0.1)
                    self.assertEqual((result.state, result.velocity), ("active", expected_velocity))
                    self.assertEqual((result.activation_count, result.progress), (1, 1.0))
                    self.assertIsNone(result.activated_action)
                self.assertIn("while looking", result.detail)

    def test_heading_feedback_corrects_both_directions_and_translation_modes_with_a_cap(self) -> None:
        for mode in ("step", "hold"):
            for action in ("forward", "back"):
                for drift in (-0.5, 0.5):
                    with self.subTest(mode=mode, action=action, drift=drift):
                        self.setUp()
                        self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode=mode))
                        initial = self.fire(action)
                        self.assertEqual(initial.velocity[2], 0.0)
                        self.pose = (0.0, 0.0, drift)
                        for _ in range(8):
                            result = self.tick(action)
                        self.assertEqual(result.velocity[:2], initial.velocity[:2])
                        self.assertAlmostEqual(result.velocity[2], math.copysign(0.15, -drift))
                        status = self.controller.heading_status
                        self.assertTrue(status["active"])
                        self.assertEqual(status["target_yaw_rad"], 0.0)
                        self.assertAlmostEqual(status["error_rad"], -drift)
                        self.assertEqual(status["correction_rad_s"], result.velocity[2])
                        status["correction_rad_s"] = 9.0
                        self.assertEqual(self.controller.heading_status["correction_rad_s"], result.velocity[2])

    def test_heading_feedback_wraps_across_pi_using_the_short_direction(self) -> None:
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                self.setUp()
                self.pose = (0.0, 0.0, sign * math.radians(179))
                self.fire()
                self.pose = (0.0, 0.0, -sign * math.radians(179))
                for _ in range(8):
                    result = self.tick("forward")
                self.assertAlmostEqual(self.controller.heading_status["error_rad"], -sign * math.radians(2))
                self.assertLess(result.velocity[2] * sign, 0.0)
                self.assertLess(abs(result.velocity[2]), self.config.heading_gain * math.radians(2))

    def test_heading_feedback_deadband_and_disabled_configuration_request_no_yaw(self) -> None:
        for config, drift in (
            (self.config, math.radians(0.25)),
            (replace(self.config, heading_hold=False), 0.5),
            (replace(self.config, heading_gain=0.0), 0.5),
            (replace(self.config, heading_max_yaw_speed=0.0), 0.5),
        ):
            with self.subTest(config=config, drift=drift):
                self.setUp()
                self.controller = GAZE.GazeLocomotion(config)
                self.fire()
                self.pose = (0.0, 0.0, drift)
                for _ in range(10):
                    self.assertEqual(self.tick("forward").velocity, (0.2, 0.0, 0.0))
                if not config.heading_hold:
                    self.assertFalse(self.controller.heading_status["active"])

    def test_heading_filter_uses_simulation_time_independently_of_wall_time(self) -> None:
        results = {}
        for wall_dt, physics_dt in ((0.01, 0.05), (0.20, 0.05), (0.01, 0.15)):
            self.setUp()
            self.fire()
            self.pose = (0.0, 0.0, -0.1)
            results[wall_dt, physics_dt] = self.tick("forward", wall_dt=wall_dt, dt=physics_dt).velocity[2]
        self.assertGreater(results[0.01, 0.05], 0.0)
        self.assertAlmostEqual(results[0.01, 0.05], results[0.20, 0.05])
        self.assertGreater(results[0.01, 0.15], results[0.01, 0.05])

    def test_heading_feedback_clears_on_hold_release_loss_fault_and_reset_then_reanchors(self) -> None:
        for release in ({"target": None}, {"eye_valid": False}, {"target": "occluded"}, {"pose": None}):
            with self.subTest(release=release):
                self.setUp()
                self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode="hold"))
                self.fire()
                self.pose = (0.0, 0.0, -0.2)
                self.assertGreater(self.tick("forward").velocity[2], 0.0)
                values = {"target": "forward", **release}
                self.assertEqual(self.tick(**values).velocity, (0.0, 0.0, 0.0))
                self.assertFalse(self.controller.heading_status["active"])
                self.assertEqual(self.controller.heading_status["correction_rad_s"], 0.0)
                self.controller.reset()
                self.assertIsNone(self.controller.heading_status["target_yaw_rad"])
                self.assertEqual(self.fire().velocity[2], 0.0)
                self.assertEqual(self.controller.heading_status["target_yaw_rad"], -0.2)

    def test_heading_feedback_preserves_turns_preview_and_step_completion(self) -> None:
        for action, expected in (("turn_left", 0.25), ("turn_right", -0.25)):
            self.setUp()
            self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode="hold"))
            self.fire(action)
            self.pose = (0.0, 0.0, 0.4)
            self.assertEqual(self.tick(action).velocity, (0.0, 0.0, expected))
            self.assertFalse(self.controller.heading_status["active"])
        self.setUp()
        self.controller = GAZE.GazeLocomotion(replace(self.config, preview=True))
        self.assertEqual(self.fire().velocity, (0.0, 0.0, 0.0))
        self.assertFalse(self.controller.heading_status["active"])
        self.setUp()
        self.fire()
        self.pose = (0.1, 0.0, -0.2)
        result = self.tick(eye_valid=False)  # Accepted steps still own heading after eye loss.
        self.assertAlmostEqual(result.progress, 0.5)
        self.assertGreater(result.velocity[2], 0.0)
        self.pose = (0.2, 0.0, -0.2)
        result = self.tick()
        self.assertEqual((result.state, result.velocity), ("stopping", (0.0, 0.0, 0.0)))
        self.assertFalse(self.controller.heading_status["active"])

    def test_hold_look_away_requests_normal_stop_without_a_settling_interlock(self) -> None:
        self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode="hold"))
        self.fire()
        result = self.tick(None, linear_speed_m_s=0.2)
        self.assertEqual((result.state, result.target, result.velocity), ("idle", None, (0, 0, 0)))
        self.assertFalse(result.stop_immediately)
        self.assertIsNone(self.controller._action)
        # Only selection dwell remains; no zero-speed/settling interval
        # is imposed by hold. Physical motion supervision is external.
        for _ in range(4):
            result = self.tick("turn_left", linear_speed_m_s=0.08)
        self.assertEqual((result.state, result.activation_count), ("active", 2))

    def test_zero_dwell_hold_switches_every_direction_pair_on_the_changed_sample(self) -> None:
        velocities = {
            "forward": (0.2, 0, 0),
            "back": (-0.15, 0, 0),
            "turn_left": (0, 0, 0.25),
            "turn_right": (0, 0, -0.25),
        }
        for first in GAZE.ACTIONS:
            for second in GAZE.ACTIONS:
                if first == second:
                    continue
                with self.subTest(first=first, second=second):
                    self.setUp()
                    self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode="hold", dwell_s=0))
                    self.tick()
                    self.assertEqual(self.tick(first).activated_action, first)
                    self.pose = (0.1, -0.1, 0.35)
                    switched = self.tick(second, linear_speed_m_s=0.2, yaw_speed_rad_s=0.1)
                    self.assertEqual(
                        (switched.state, switched.target, switched.activated_action), ("active", second, second)
                    )
                    self.assertEqual((switched.activation_count, switched.velocity), (2, velocities[second]))
                    self.assertFalse(switched.stop_immediately)
                    if second in ("forward", "back"):
                        self.assertEqual(self.controller.heading_status["target_yaw_rad"], 0.35)
                    for _ in range(3):
                        held = self.tick(second)
                        self.assertEqual((held.activation_count, held.velocity), (2, velocities[second]))
                        self.assertIsNone(held.activated_action)

    def test_hold_direction_change_starts_fresh_configured_dwell_and_discards_partial_switch(self) -> None:
        self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode="hold"))
        self.fire()
        switched = self.tick("turn_left", linear_speed_m_s=0.2)
        self.assertEqual((switched.state, switched.target, switched.progress), ("dwelling", "turn_left", 0))
        self.assertEqual((switched.velocity, switched.activation_count), ((0, 0, 0), 1))
        self.assertIsNone(switched.activated_action)
        self.assertAlmostEqual(self.tick("turn_left").progress, 1 / 3)
        changed_again = self.tick("back")
        self.assertEqual((changed_again.state, changed_again.target, changed_again.progress), ("dwelling", "back", 0))
        halfway = self.tick("back", wall_dt=0.15)
        self.assertAlmostEqual(halfway.progress, 0.5)
        self.assertEqual(halfway.velocity, (0, 0, 0))
        completed = self.tick("back", wall_dt=0.15)
        self.assertEqual(
            (completed.state, completed.activated_action, completed.activation_count), ("active", "back", 2)
        )
        self.assertEqual(completed.velocity, (-0.15, 0, 0))

    def test_hold_preview_switches_directly_with_zero_or_positive_dwell_and_never_moves(self) -> None:
        for dwell in (0.0, 0.3):
            with self.subTest(dwell=dwell):
                self.setUp()
                self.controller = GAZE.GazeLocomotion(
                    replace(self.config, control_mode="hold", dwell_s=dwell, preview=True)
                )
                self.pose = None
                self.tick()
                for count, action in enumerate(GAZE.ACTIONS, start=1):
                    result = self.tick(action, linear_speed_m_s=math.nan, yaw_speed_rad_s=math.nan)
                    if dwell:
                        self.assertEqual((result.state, result.target, result.progress), ("dwelling", action, 0))
                        for _ in range(3):
                            result = self.tick(action)
                    self.assertEqual(
                        (result.state, result.target, result.activated_action), ("preview", action, action)
                    )
                    self.assertEqual((result.velocity, result.activation_count), ((0, 0, 0), count))
                    for _ in range(3):
                        held = self.tick(action)
                        self.assertEqual(
                            (held.state, held.velocity, held.activation_count), ("preview", (0, 0, 0), count)
                        )
                        self.assertIsNone(held.activated_action)

    def test_hold_switch_onto_disabled_back_stops_without_activation_and_keeps_valid_selection_available(self) -> None:
        for preview in (False, True):
            with self.subTest(preview=preview):
                self.setUp()
                self.controller = GAZE.GazeLocomotion(
                    replace(self.config, control_mode="hold", dwell_s=0, backward_speed=0, preview=preview)
                )
                self.tick()
                self.assertEqual(self.tick("forward").activation_count, 1)
                for _ in range(3):
                    disabled = self.tick("back")
                    self.assertEqual(
                        (disabled.state, disabled.velocity, disabled.activation_count), ("unavailable", (0, 0, 0), 1)
                    )
                    self.assertIn("disabled", disabled.detail)
                    self.assertIsNone(disabled.activated_action)
                resumed = self.tick("turn_right")
                self.assertEqual((resumed.activated_action, resumed.activation_count), ("turn_right", 2))
                self.assertEqual(resumed.velocity, (0, 0, 0) if preview else (0, 0, -0.25))

    def test_hold_tracking_gaps_occlusion_and_readiness_loss_stop_immediately(self) -> None:
        cases = (
            {"eye_valid": False},
            {"target": "occluded"},
            {"wall_dt": 0.3},
            {"wall_dt": -0.05},
            {"ready": False},
        )
        for invalid in cases:
            with self.subTest(invalid=invalid):
                self.setUp()
                self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode="hold"))
                self.fire()
                # A newly selected direction on a failed sample must not bypass
                # the tracking/readiness interlock used for the previous hold.
                values = dict(target="turn_left", linear_speed_m_s=0.1)
                values.update(invalid)
                result = self.tick(**values)
                self.assertEqual((result.state, result.velocity), ("unavailable", (0, 0, 0)))
                self.assertTrue(result.stop_immediately)
                for action in (*GAZE.ACTIONS, "turn_left"):
                    result = self.tick(action, linear_speed_m_s=0.1)
                    self.assertEqual((result.state, result.activation_count), ("awaiting_release", 1))
                    self.assertEqual(result.velocity, (0, 0, 0))
                self.assertEqual(self.fire("back").activation_count, 2)

    def test_hold_tracking_loss_after_normal_release_requests_immediate_zero_and_disarms(self) -> None:
        self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode="hold"))
        self.fire()
        self.assertFalse(self.tick(linear_speed_m_s=0.1).stop_immediately)
        self.assertTrue(self.tick(eye_valid=False, linear_speed_m_s=0.1).stop_immediately)
        self.assertTrue(self.tick(eye_valid=False, linear_speed_m_s=0.1).stop_immediately)
        for _ in range(5):
            self.assertEqual(self.tick("forward").state, "awaiting_release")
        self.assertEqual(self.fire().activation_count, 2)

    def test_hold_release_does_not_fault_or_claim_settled_after_long_balance_motion(self) -> None:
        self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode="hold"))
        self.fire()
        self.assertEqual(self.tick(linear_speed_m_s=0.08).state, "idle")
        # This would exceed the former hold deadline and its strict pelvis-only
        # envelope. The caller's foot-support/runaway guards still own whether
        # physical residual motion is acceptable; this controller claims no rest.
        for index in range(160):
            self.pose = (0.03 * math.sin(index * 0.2), 0.03 * math.cos(index * 0.2), 0.03 * math.sin(index * 0.2))
            result = self.tick(linear_speed_m_s=0.08, yaw_speed_rad_s=0.12)
            self.assertEqual((result.state, result.velocity, result.activation_count), ("idle", (0, 0, 0), 1))
        self.assertFalse(self.controller.stop_diagnostics["settling_evaluated"])
        self.assertNotIn("settled", result.detail.lower())
        self.assertEqual(self.fire().activation_count, 2)

    def test_hold_stop_diagnostics_survive_reset_and_cannot_be_mutated_by_caller(self) -> None:
        self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode="hold"))
        self.assertEqual(self.controller.stop_diagnostics, {})
        self.fire()
        self.pose = (0.2, -0.1, 0.3)
        self.tick(eye_valid=False, linear_speed_m_s=0.08, yaw_speed_rad_s=-0.12)
        diagnostics = self.controller.stop_diagnostics
        self.assertEqual(diagnostics["action"], "forward")
        self.assertEqual(diagnostics["pose"], self.pose)
        self.assertEqual((diagnostics["linear_speed_m_s"], diagnostics["yaw_speed_rad_s"]), (0.08, 0.12))
        self.assertTrue(diagnostics["stop_immediately"])
        self.assertFalse(diagnostics["verified_outside"])
        self.assertFalse(diagnostics["settling_evaluated"])
        diagnostics["reason"] = "caller mutation"
        self.assertEqual(self.controller.stop_diagnostics["reason"], "Eye tracking lost")
        retained = self.controller.stop_diagnostics
        self.controller.reset()
        self.assertEqual(self.controller.stop_diagnostics, retained)
        self.fire("back")
        self.tick()
        self.assertEqual(self.controller.stop_diagnostics["action"], "back")
        self.assertTrue(self.controller.stop_diagnostics["verified_outside"])

    def test_hold_preview_is_zero_and_one_activation_per_released_fixation(self) -> None:
        for invalid in (
            {"eye_valid": False},
            {"target": "occluded"},
            {"wall_dt": 0.3},
            {"wall_dt": -0.05},
            {"ready": False},
        ):
            with self.subTest(invalid=invalid):
                self.setUp()
                self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode="hold", preview=True))
                self.pose = None
                result = self.fire()
                self.assertIn("Would move forward while looking", result.detail)
                for _ in range(5):
                    result = self.tick("forward")
                    self.assertEqual(
                        (result.state, result.velocity, result.activation_count), ("preview", (0, 0, 0), 1)
                    )
                    self.assertIsNone(result.activated_action)
                self.tick(**{"target": "turn_left", **invalid})
                for action in GAZE.ACTIONS:
                    result = self.tick(action)
                    self.assertEqual((result.velocity, result.activation_count), ((0, 0, 0), 1))
                    self.assertIsNone(result.activated_action)
                self.assertEqual(self.fire("back").activation_count, 2)

    def test_hold_reset_cancels_motion_and_requires_new_release_and_dwell(self) -> None:
        self.controller = GAZE.GazeLocomotion(replace(self.config, control_mode="hold"))
        self.fire()
        self.controller.reset()
        self.assertEqual(self.controller.command.velocity, (0, 0, 0))
        for _ in range(5):
            self.assertEqual(self.tick("forward").state, "awaiting_release")
        self.assertEqual(self.fire().activation_count, 1)

    def test_small_recording_scale_balance_sway_settles_without_changing_deadline(self) -> None:
        self.fire()
        self.pose = (0.2, 0.0, math.pi - 0.005)
        self.tick(linear_speed_m_s=0.05)
        result = None
        # Synthetic 8 mm planar sway and 0.012 rad yaw reproduce the measured
        # recording's scale, not private recorded samples. Planar speed remains
        # above the old instantaneous threshold throughout, including yaw wrap.
        for index in range(1, 51):
            elapsed = index * 0.05
            phase = 2 * math.pi * elapsed
            yaw = math.pi - 0.005 + 0.012 * math.sin(phase)
            self.pose = (
                0.2 + 0.008 * math.sin(phase),
                0.008 * (math.cos(phase) - 1),
                (yaw + math.pi) % (2 * math.pi) - math.pi,
            )
            result = self.tick(
                linear_speed_m_s=2 * math.pi * 0.008, yaw_speed_rad_s=2 * math.pi * 0.012 * math.cos(phase)
            )
            self.assertEqual(result.velocity, (0, 0, 0))
            if result.state == "awaiting_release":
                break
        self.assertEqual(result.state, "awaiting_release")
        self.assertGreaterEqual(elapsed, self.controller.SETTLE_WINDOW_S)
        self.assertLess(elapsed, self.controller.SETTLE_TIMEOUT_S)
        self.assertEqual(self.controller.SETTLE_TIMEOUT_S, 4.0)

    def test_ongoing_slow_translation_or_rotation_does_not_pass_sway_settling(self) -> None:
        for action in ("translation", "rotation"):
            with self.subTest(action=action):
                self.setUp()
                self.fire()
                self.pose = (0.2, 0.0, 0.0)
                self.tick()
                for index in range(1, 81):
                    elapsed = index * 0.05
                    self.pose = (
                        0.2 + (0.035 * elapsed if action == "translation" else 0),
                        0,
                        0.07 * elapsed if action == "rotation" else 0,
                    )
                    result = self.tick(
                        linear_speed_m_s=0.035 if action == "translation" else 0,
                        yaw_speed_rad_s=0.07 if action == "rotation" else 0,
                    )
                    self.assertEqual(result.velocity, (0, 0, 0))
                self.assertEqual(result.state, "fault")
                self.assertTrue(result.stop_immediately)

    def test_zero_net_motion_with_large_excursions_or_speed_peaks_does_not_settle(self) -> None:
        for case in ("position_envelope", "yaw_envelope", "linear_peak", "yaw_peak"):
            with self.subTest(case=case):
                self.setUp()
                self.fire()
                self.pose = (0.2, 0.0, 0.0)
                self.tick()
                for index in range(1, 81):
                    elapsed = index * 0.05
                    phase = 2 * math.pi * elapsed
                    radius = 0.03 if case == "position_envelope" else 0.005
                    yaw_amplitude = 0.05 if case == "yaw_envelope" else 0.005
                    self.pose = (
                        0.2 + radius * math.sin(phase),
                        radius * (math.cos(phase) - 1),
                        yaw_amplitude * math.sin(phase),
                    )
                    result = self.tick(
                        linear_speed_m_s=0.2 if case == "linear_peak" else 0.04,
                        yaw_speed_rad_s=0.4 if case == "yaw_peak" else 0.08,
                    )
                self.assertEqual(result.state, "fault")

    def test_sway_confirmation_must_be_continuous_and_cannot_override_new_drift(self) -> None:
        self.fire()
        self.pose = (0.2, 0.0, 0.0)
        self.tick()
        for _ in range(20):
            result = self.tick(linear_speed_m_s=0.04)
        self.assertEqual(result.state, "stopping")
        # The pose window just became eligible; a new sustained movement must
        # clear the alternative confirmation before it reaches 0.2 seconds.
        for index in range(1, 61):
            self.pose = (0.2 + 0.005 * index, 0, 0)
            result = self.tick(linear_speed_m_s=0.2)
        self.assertEqual(result.state, "fault")

    def test_step_timeout_retains_failed_window_metrics_after_fault_and_reset(self) -> None:
        self.fire()
        self.pose = (0.2, 0, 0)
        self.tick()
        for index in range(1, 81):
            self.pose = (0.2 + 0.005 * index, 0, 0)
            result = self.tick(linear_speed_m_s=0.1)
        self.assertEqual(result.state, "fault")
        self.assertFalse(self.controller._stop_window)
        diagnostics = self.controller.stop_diagnostics
        self.assertEqual(diagnostics["decision"], "step_settle_timeout_fault")
        self.assertIn("position_envelope_m", diagnostics["failed_checks"])
        self.assertIn("endpoint_drift_m_s", diagnostics["failed_checks"])
        self.assertIn("fitted_drift_m_s", diagnostics["failed_checks"])
        self.assertAlmostEqual(diagnostics["elapsed_s"], 4.0)
        self.assertAlmostEqual(diagnostics["endpoint_drift_m_s"], 0.1)
        self.assertAlmostEqual(diagnostics["fitted_drift_m_s"], 0.1)
        self.controller.reset()
        self.assertEqual(self.controller.stop_diagnostics, diagnostics)


if __name__ == "__main__":
    unittest.main()
