# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise face recognition and measured motion without a headset or Kit."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
)
DEPENDENCY_NAME = "isaacsim.robot.policy.examples.interactive.humanoid.gaze_locomotion"
PREVIOUS_DEPENDENCY = sys.modules.get(DEPENDENCY_NAME)
GAZE_SPEC = importlib.util.spec_from_file_location(DEPENDENCY_NAME, DIRECTORY / "gaze_locomotion.py")
GAZE = importlib.util.module_from_spec(GAZE_SPEC)
sys.modules[DEPENDENCY_NAME] = GAZE
GAZE_SPEC.loader.exec_module(GAZE)
SPEC = importlib.util.spec_from_file_location("facial_locomotion_under_test", DIRECTORY / "facial_locomotion.py")
FACE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FACE
SPEC.loader.exec_module(FACE)
if PREVIOUS_DEPENDENCY is not None:
    sys.modules[DEPENDENCY_NAME] = PREVIOUS_DEPENDENCY
else:
    del sys.modules[DEPENDENCY_NAME]

EXPRESSIONS = {
    "forward": {"JawDrop": 0.9},
    "back": {"LipPuckerL": 0.9, "LipPuckerR": 0.9},
    "strafe_left": {"MouthLeft": 0.9},
    "strafe_right": {"MouthRight": 0.9},
    "turn_left": {"EyesClosedL": 0.9},
    "turn_right": {"EyesClosedR": 0.9},
}


class TestFacialLocomotionControls(unittest.TestCase):
    def setUp(self) -> None:
        self.config = FACE.FacialLocomotionConfig(preview=False)
        self.controller = FACE.FacialLocomotion(self.config)
        self.time = 0.0
        self.sequence = 0
        self.pose = (0.0, 0.0, 0.0)

    def tick(self, expression=None, *, wall_dt=0.05, fresh=True, valid=True, sample_changes=None, **kwargs):
        self.time += wall_dt
        self.sequence += int(fresh)
        values = dict.fromkeys(FACE.REQUIRED_WEIGHTS, 0.0)
        if isinstance(expression, str):
            values.update(EXPRESSIONS[expression])
        elif expression is not None:
            values.update(expression)
        sample = SimpleNamespace(
            weights=values,
            valid=valid,
            reason="",
            received_at=self.time,
            age_s=0.0,
            sequence=self.sequence,
        )
        for name, value in (sample_changes or {}).items():
            setattr(sample, name, value)
        arguments = dict(
            sample=sample,
            now_s=self.time,
            dt=0.05,
            pose=self.pose,
            linear_speed_m_s=0.0,
            yaw_speed_rad_s=0.0,
        )
        arguments.update(kwargs)
        return self.controller.update(**arguments)

    def neutral(self):
        result = None
        for _ in range(6):
            result = self.tick()
        self.assertTrue(result.neutral_ready)
        return result

    def activate(self, action):
        self.neutral()
        result = None
        for _ in range(5):
            result = self.tick(action)
        if self.config.control_mode == "step" and action.startswith("turn"):
            self.assertEqual(result.state, "wink_release")
            result = self.tick()
        self.assertEqual(result.activated_action, action)
        return result

    def use_config(self, **kwargs):
        self.config = replace(self.config, **kwargs)
        self.controller = FACE.FacialLocomotion(self.config)

    def test_default_is_hold_preview_with_supported_speeds(self) -> None:
        config = FACE.FacialLocomotionConfig()
        self.assertTrue(config.preview)
        self.assertEqual(config.control_mode, "hold")
        self.assertEqual((config.distance_m, config.turn_degrees), (0.15, 10))
        self.assertEqual(FACE.FacialLocomotion().command.velocity, (0, 0, 0))

    def test_configuration_rejects_invalid_numbers_modes_and_thresholds(self) -> None:
        limits = {
            "distance_m": (0, 1.01),
            "turn_degrees": (0, 45.01),
            "forward_speed": (0, 0.651),
            "backward_speed": (0, 0.351),
            "lateral_speed": (0, 0.301),
            "yaw_speed": (0, 0.401),
            "activation_threshold": (0.39, 1.01),
            "release_threshold": (-0.01, 0.51),
            "confirm_s": (0.079, 1.01),
            "neutral_s": (0.099, 1.01),
            "max_sample_age_s": (0.049, 1.01),
            "max_gesture_s": (0.399, 5.01),
        }
        for name, invalid in limits.items():
            for value in (*invalid, math.nan, math.inf, True, "1", None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(self.config, **{name: value})
        for value in ("continuous", None, True, 1):
            with self.subTest(mode=value), self.assertRaises(ValueError):
                replace(self.config, control_mode=value)
        with self.assertRaises(ValueError):
            replace(self.config, activation_threshold=0.4, release_threshold=0.4)
        with self.assertRaises(ValueError):
            replace(self.config, confirm_s=0.8, max_gesture_s=0.8)
        with self.assertRaises(ValueError):
            replace(self.config, preview=1)
        with self.assertRaises(TypeError):
            FACE.FacialLocomotion({})

    def test_startup_requires_fresh_neutral_before_any_gesture(self) -> None:
        for _ in range(20):
            result = self.tick("forward")
            self.assertEqual(result.activation_count, 0)
        self.neutral()
        self.assertEqual(self.tick("forward").state, "confirming")

    def test_optional_calibration_defaults_preserve_shared_threshold(self) -> None:
        for name in ("wink_threshold", "jaw_threshold", "pucker_threshold", "mouth_side_threshold"):
            self.assertIsNone(getattr(self.config, name))
        for action in EXPRESSIONS:
            self.assertEqual(self.controller._threshold_for(action), self.config.activation_threshold)

    def test_optional_thresholds_reject_invalid_values_and_release_overlap(self) -> None:
        for name in ("wink_threshold", "jaw_threshold", "pucker_threshold", "mouth_side_threshold"):
            with self.subTest(name=name):
                self.assertIsNone(getattr(replace(self.config, **{name: None}), name))
            for value in (True, False, "0.5", math.nan, math.inf, 0.399, 1.001):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(self.config, **{name: value})
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(self.config, release_threshold=0.4, **{name: 0.4})

    def test_calibrated_thresholds_activate_each_expression_independently(self) -> None:
        cases = {
            "turn_left": {"EyesClosedL": 0.60, "EyesClosedR": 0.30},
            "turn_right": {"EyesClosedR": 0.60, "EyesClosedL": 0.16},
            "forward": {"JawDrop": 0.50},
            "back": {"LipPuckerL": 0.44, "LipPuckerR": 0.42},
            "strafe_left": {"MouthLeft": 0.70},
            "strafe_right": {"MouthRight": 0.70},
        }
        for action, values in cases.items():
            with self.subTest(action=action):
                self.setUp()
                self.use_config(
                    wink_threshold=0.55,
                    jaw_threshold=0.45,
                    pucker_threshold=0.40,
                    mouth_side_threshold=0.65,
                    release_threshold=0.35,
                )
                self.neutral()
                for _ in range(5):
                    result = self.tick(values)
                self.assertEqual(result.activated_action, action)
                self.assertEqual(result.activation_count, 1)
                self.assertTrue(any(result.velocity))
                self.assertFalse(any(self.tick().velocity))

    def test_lower_jaw_threshold_does_not_lower_eye_or_side_threshold(self) -> None:
        self.use_config(jaw_threshold=0.45)
        for values in ({"EyesClosedL": 0.5}, {"MouthLeft": 0.5}, {"LipPuckerL": 0.5, "LipPuckerR": 0.5}):
            self.neutral()
            for _ in range(8):
                self.assertEqual(self.tick(values).activation_count, 0)
            self.tick()
        self.neutral()
        for _ in range(5):
            result = self.tick({"JawDrop": 0.5})
        self.assertEqual(result.activated_action, "forward")

    def test_calibrated_bilateral_blink_and_cross_expression_still_rejected(self) -> None:
        self.use_config(wink_threshold=0.55, jaw_threshold=0.45, pucker_threshold=0.40, release_threshold=0.35)
        for values in ({"EyesClosedL": 0.60, "EyesClosedR": 0.6}, {"JawDrop": 0.5, "MouthRight": 0.6}):
            self.neutral()
            for _ in range(8):
                self.assertEqual(self.tick(values).activation_count, 0)
            self.tick()

    def test_calibrated_step_wink_still_waits_for_reopening(self) -> None:
        self.use_config(control_mode="step", wink_threshold=0.55, release_threshold=0.35)
        self.neutral()
        for _ in range(5):
            self.assertEqual(self.tick({"EyesClosedL": 0.6, "EyesClosedR": 0.3}).activation_count, 0)
        self.assertEqual(self.controller.command.state, "wink_release")
        self.assertEqual(self.tick().activated_action, "turn_left")

    def test_brief_expression_does_not_activate(self) -> None:
        for action in EXPRESSIONS:
            with self.subTest(action=action):
                self.setUp()
                self.neutral()
                self.tick(action)
                self.tick(action)
                self.assertEqual(self.tick().activation_count, 0)

    def test_six_hold_directions_have_correct_body_frame_sign_and_speed(self) -> None:
        expected = {
            "forward": (0.20, 0, 0),
            "back": (-0.15, 0, 0),
            "strafe_left": (0, 0.15, 0),
            "strafe_right": (0, -0.15, 0),
            "turn_left": (0, 0, 0.25),
            "turn_right": (0, 0, -0.25),
        }
        for action, velocity in expected.items():
            with self.subTest(action=action):
                self.setUp()
                self.assertEqual(self.activate(action).velocity, velocity)
                self.assertEqual(self.controller.command.activation_count, 1)

    def test_hold_continues_and_fires_once_until_relaxation(self) -> None:
        for action in EXPRESSIONS:
            with self.subTest(action=action):
                self.setUp()
                activation = self.activate(action)
                for _ in range(80):
                    result = self.tick(action)
                    self.assertEqual(result.velocity, activation.velocity)
                    self.assertIsNone(result.activated_action)
                    self.assertEqual(result.activation_count, 1)
                result = self.tick()
                self.assertEqual(result.velocity, (0, 0, 0))
                self.assertFalse(result.busy)
                self.assertFalse(result.stop_immediately)

    def test_hysteresis_keeps_expression_above_release_until_relaxed(self) -> None:
        self.activate("forward")
        self.assertGreater(self.tick({"JawDrop": 0.4}).velocity[0], 0)
        self.assertEqual(self.tick({"JawDrop": 0.24}).velocity, (0, 0, 0))

    def test_gradual_onset_crosses_threshold_without_neutral_deadlock(self) -> None:
        for action in EXPRESSIONS:
            with self.subTest(action=action):
                self.setUp()
                self.neutral()
                for weight in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9):
                    result = self.tick({name: weight for name in EXPRESSIONS[action]})
                self.assertEqual(result.activation_count, 1)
                self.assertTrue(any(result.velocity))

    def test_ordinary_bilateral_blinks_never_fire(self) -> None:
        self.neutral()
        for _ in range(12):
            result = self.tick({"EyesClosedL": 0.9, "EyesClosedR": 0.9})
            self.assertEqual(result.activation_count, 0)
        self.neutral()
        self.assertEqual(self.controller.command.activation_count, 0)

    def test_staggered_blinks_reject_closure_and_reopening_tails(self) -> None:
        for first, second in (("turn_left", "turn_right"), ("turn_right", "turn_left")):
            with self.subTest(first=first):
                self.setUp()
                self.neutral()
                self.tick(first)
                self.tick(first)
                self.tick({"EyesClosedL": 0.9, "EyesClosedR": 0.9})
                for _ in range(12):
                    result = self.tick(second)
                    self.assertEqual(result.activation_count, 0)
                self.neutral()
                self.assertEqual(self.activate(first).activation_count, 1)

    def test_bilateral_closure_during_turn_stops_immediately(self) -> None:
        self.activate("turn_left")
        result = self.tick({"EyesClosedL": 0.9, "EyesClosedR": 0.9})
        self.assertEqual(result.velocity, (0, 0, 0))
        self.assertTrue(result.stop_immediately)
        for _ in range(10):
            self.assertEqual(self.tick("turn_right").activation_count, 1)

    def test_mouth_and_eye_or_two_mouth_expressions_are_ignored(self) -> None:
        for expression in ({"JawDrop": 0.9, "EyesClosedL": 0.9}, {"JawDrop": 0.9, "MouthRight": 0.9}):
            with self.subTest(expression=expression):
                self.setUp()
                self.neutral()
                for _ in range(10):
                    self.assertEqual(self.tick(expression).activation_count, 0)

    def test_single_sided_pucker_does_not_move_back(self) -> None:
        self.neutral()
        for _ in range(10):
            self.assertEqual(self.tick({"LipPuckerL": 0.9}).activation_count, 0)

    def test_direct_gesture_switch_stops_and_requires_neutral(self) -> None:
        self.activate("forward")
        result = self.tick("strafe_left")
        self.assertTrue(result.stop_immediately)
        self.assertEqual(result.velocity, (0, 0, 0))
        for _ in range(10):
            self.assertEqual(self.tick("strafe_left").activation_count, 1)
        self.assertEqual(self.activate("strafe_left").activation_count, 2)

    def test_preview_recognizes_all_holds_and_always_zeroes_velocity(self) -> None:
        self.use_config(preview=True)
        self.pose = None
        for action in EXPRESSIONS:
            result = self.activate(action)
            self.assertEqual(result.state, "preview")
            for _ in range(10):
                result = self.tick(action)
                self.assertEqual(result.velocity, (0, 0, 0))
                self.assertTrue(result.busy)
                self.assertIsNone(result.activated_action)
            self.assertFalse(self.tick().busy)

    def test_repeated_sample_cannot_finish_confirmation_or_neutral(self) -> None:
        self.tick()
        for _ in range(10):
            self.assertFalse(self.tick(fresh=False).neutral_ready)
        self.neutral()
        self.tick("forward")
        for _ in range(10):
            self.assertEqual(self.tick("forward", fresh=False).activation_count, 0)

    def test_invalid_stale_missing_or_nonfinite_sample_stops(self) -> None:
        cases = [
            {"valid": False},
            {"age_s": 0.251},
            {"received_at": -1},
            {"received_at": math.nan},
            {"sequence": -1},
            {"sequence": True},
            {"weights": {}},
            {"weights": dict.fromkeys(FACE.REQUIRED_WEIGHTS, math.nan)},
            {"weights": dict.fromkeys(FACE.REQUIRED_WEIGHTS, True)},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                self.setUp()
                self.activate("forward")
                result = self.tick("forward", sample_changes=changes)
                self.assertEqual(result.velocity, (0, 0, 0))
                self.assertTrue(result.stop_immediately)
                for _ in range(10):
                    self.assertEqual(self.tick("forward").activation_count, 1)

    def test_recovery_requires_neutral_then_can_move_again(self) -> None:
        self.activate("forward")
        self.tick(valid=False)
        result = self.activate("back")
        self.assertEqual(result.activation_count, 2)

    def test_readiness_and_explicit_enable_cancel_motion(self) -> None:
        for field in ("ready", "enabled"):
            with self.subTest(field=field):
                self.setUp()
                self.activate("forward")
                result = self.tick("forward", **{field: False})
                self.assertTrue(result.stop_immediately)
                self.assertEqual(result.velocity, (0, 0, 0))

    def test_wall_gap_clock_reversal_and_sequence_restart_cancel(self) -> None:
        for arguments in ({"wall_dt": 0.3}, {"wall_dt": -0.01}, {"sample_changes": {"sequence": 0}}):
            with self.subTest(arguments=arguments):
                self.setUp()
                self.activate("forward")
                result = self.tick("forward", **arguments)
                self.assertEqual(result.velocity, (0, 0, 0))
                self.assertTrue(result.stop_immediately)

    def test_reset_cancels_fault_and_requires_neutral(self) -> None:
        self.activate("forward")
        self.assertEqual(self.tick("forward", pose=None).state, "fault")
        self.assertEqual(self.tick("forward").state, "fault")
        self.controller.reset()
        self.assertEqual(self.tick("forward").activation_count, 0)
        self.assertEqual(self.activate("forward").activation_count, 1)

    def test_invalid_pose_before_activation_never_counts_movement(self) -> None:
        self.pose = None
        self.neutral()
        for _ in range(10):
            result = self.tick("forward")
            self.assertEqual(result.activation_count, 0)
            self.assertEqual(result.velocity, (0, 0, 0))

    def test_hold_has_no_distance_or_time_deadline(self) -> None:
        self.activate("forward")
        for index in range(250):
            self.pose = (index * 0.03, 0, 0)
            result = self.tick("forward")
        self.assertEqual(result.state, "active")
        self.assertGreater(result.velocity[0], 0)

    def test_heading_correction_uses_current_robot_heading_for_side_step(self) -> None:
        self.activate("strafe_left")
        self.pose = (0, 0.02, -0.1)
        result = self.tick("strafe_left")
        self.assertGreater(result.velocity[2], 0)
        self.assertLessEqual(result.velocity[2], 0.15)
        self.assertEqual(result.velocity[:2], (0, 0.15))

    def test_step_wink_fires_only_when_reopened(self) -> None:
        self.use_config(control_mode="step")
        self.neutral()
        for _ in range(8):
            result = self.tick("turn_left")
            self.assertEqual(result.activation_count, 0)
        self.assertEqual(result.state, "wink_release")
        self.assertEqual(self.tick().activated_action, "turn_left")

    def test_step_long_wink_expires_without_turn(self) -> None:
        self.use_config(control_mode="step")
        self.neutral()
        for _ in range(45):
            self.assertEqual(self.tick("turn_left").activation_count, 0)
        self.assertEqual(self.tick().activation_count, 0)

    def test_step_relaxation_allows_move_to_finish_and_busy_ignores_new_gestures(self) -> None:
        self.use_config(control_mode="step")
        self.activate("forward")
        self.pose = (0.05, 0, 0)
        self.assertGreater(self.tick().velocity[0], 0)
        for _ in range(8):
            result = self.tick("turn_left")
            self.assertEqual(result.activation_count, 1)
            self.assertGreater(result.velocity[0], 0)
        self.pose = (0.15, 0, 0)
        result = self.tick()
        self.assertEqual(result.state, "stopping")
        self.assertEqual(result.velocity, (0, 0, 0))
        for _ in range(4):
            result = self.tick()
        self.assertFalse(result.busy)
        self.assertEqual(result.state, "awaiting_neutral")

    def test_step_side_distance_is_projected_along_robot_left_axis(self) -> None:
        for action, y in (("strafe_left", 0.15), ("strafe_right", -0.15)):
            with self.subTest(action=action):
                self.setUp()
                self.use_config(control_mode="step")
                self.activate(action)
                self.pose = (0.2, 0, 0)
                self.assertEqual(self.tick().state, "active")
                self.pose = (0.2, y, 0)
                self.assertEqual(self.tick().state, "stopping")

    def test_step_turn_wraps_angle_and_stops_at_ten_degrees(self) -> None:
        self.use_config(control_mode="step")
        self.pose = (0, 0, math.radians(175))
        self.activate("turn_left")
        self.pose = (0, 0, math.radians(-175))
        self.assertEqual(self.tick().state, "stopping")

    def test_step_timeout_and_unsettled_motion_latch_fault(self) -> None:
        self.use_config(control_mode="step")
        self.activate("forward")
        for _ in range(66):
            result = self.tick()
        self.assertEqual(result.state, "stopping")
        for _ in range(85):
            self.pose = (self.pose[0] + 0.02, 0, 0)
            result = self.tick(linear_speed_m_s=0.4)
        self.assertEqual(result.state, "fault")
        self.assertEqual(result.velocity, (0, 0, 0))

    def test_step_tracking_loss_cancels_despite_gaze_engine_step_independence(self) -> None:
        self.use_config(control_mode="step")
        self.activate("forward")
        result = self.tick(valid=False)
        self.assertEqual(result.velocity, (0, 0, 0))
        self.assertTrue(result.stop_immediately)


if __name__ == "__main__":
    unittest.main()
