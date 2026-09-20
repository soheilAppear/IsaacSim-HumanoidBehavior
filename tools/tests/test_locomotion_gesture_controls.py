# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise momentary V-sign walking intent without Isaac or tracking hardware."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
    / "locomotion_gesture.py"
)
SPEC = importlib.util.spec_from_file_location("locomotion_gesture_under_test", MODULE_PATH)
GESTURE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GESTURE
SPEC.loader.exec_module(GESTURE)

V_SIGN = {"index": 0.10, "middle": 0.10, "ring": 0.90, "little": 0.90}
NEUTRAL = (0.25, 0.20, -0.35)


class TestLocomotionGestureControls(unittest.TestCase):
    def setUp(self) -> None:
        self.gesture = GESTURE.LocomotionGesture()

    def update(self, now: float, point=NEUTRAL, curls=V_SIGN, valid=True):
        return self.gesture.update(curls=curls, position=point, tracking_valid=valid, now_s=now)

    def activate(self, start: float = 0.0, point=NEUTRAL):
        result = None
        for index in range(8):
            result = self.update(start + index * 0.1, point=point)
        self.assertEqual(result.state, "active")
        self.assertEqual((result.forward, result.turn), (0.0, 0.0))
        return result

    def assert_stopped(self, result) -> None:
        self.assertEqual(result.state, "idle")
        self.assertEqual((result.forward, result.turn), (0.0, 0.0))
        self.assertFalse(result.owns_hand)
        self.assertIsNone(result.neutral_position)

    def test_stable_v_requires_full_dwell_and_reserves_hand_without_early_motion(self) -> None:
        for index in range(7):
            result = self.update(index * 0.1)
            self.assertEqual(result.state, "acquiring")
            self.assertTrue(result.owns_hand)
            self.assertTrue(self.gesture.owns_hand)
            self.assertEqual((result.forward, result.turn), (0.0, 0.0))
            self.assertLess(result.acquisition_progress, 1.0)
        result = self.update(0.7)
        self.assertEqual(result.state, "active")
        self.assertEqual(result.neutral_position, NEUTRAL)
        self.assertEqual(result.acquisition_progress, 1.0)

    def test_ordinary_fist_open_hand_point_and_pinch_never_activate(self) -> None:
        shapes = (
            dict.fromkeys(V_SIGN, 1.0),
            dict.fromkeys(V_SIGN, 0.0),
            {"index": 0.0, "middle": 0.8, "ring": 0.8, "little": 0.8},
            {"index": 0.7, "middle": 0.2, "ring": 0.2, "little": 0.2},
            {"index": 0.4, "middle": 0.4, "ring": 0.4, "little": 0.4},
        )
        for shape in shapes:
            with self.subTest(shape=shape):
                for index in range(15):
                    self.assert_stopped(self.update(index * 0.1, curls=shape))

    def test_false_source_never_authorizes_even_with_perfect_v_measurements(self) -> None:
        for index in range(15):
            self.assert_stopped(self.update(index * 0.1, valid=False))

    def test_thumb_is_not_required_for_v_detection(self) -> None:
        for index in range(8):
            result = self.update(index * 0.1, curls={**V_SIGN, "thumb": math.nan})
        self.assertEqual(result.state, "active")

    def test_dwell_drift_restarts_timer_including_vertical_movement(self) -> None:
        for index in range(6):
            self.update(index * 0.1)
        shifted = (NEUTRAL[0], NEUTRAL[1], NEUTRAL[2] + 0.041)
        result = self.update(0.6, point=shifted)
        self.assertEqual(result.state, "acquiring")
        self.assertEqual(result.acquisition_progress, 0.0)
        for index in range(1, 7):
            result = self.update(0.6 + index * 0.1, point=shifted)
            self.assertEqual(result.state, "acquiring")
        result = self.update(1.3, point=shifted)
        self.assertEqual(result.state, "active")
        self.assertEqual(result.neutral_position, shifted)

    def test_small_dwell_jitter_anchors_current_wrist_without_activation_impulse(self) -> None:
        for index in range(8):
            point = (NEUTRAL[0] + 0.025 * (index % 2), NEUTRAL[1] - 0.015 * (index % 2), NEUTRAL[2])
            result = self.update(index * 0.1, point=point)
        self.assertEqual(result.state, "active")
        self.assertEqual(result.neutral_position, point)
        self.assertEqual((result.forward, result.turn), (0.0, 0.0))

    def test_deadzone_saturation_and_turn_signs(self) -> None:
        self.activate()
        for index, (offset, expected) in enumerate(
            (
                ((0.03, -0.03, 0.3), (0.0, 0.0)),
                ((0.11, 0.095, 0.0), (0.5, 0.5)),
                ((0.18, -0.15, 0.0), (1.0, -1.0)),
                ((2.0, 2.0, 0.0), (1.0, 1.0)),
            )
        ):
            point = tuple(value + delta for value, delta in zip(NEUTRAL, offset))
            result = self.update(0.8 + index * 0.1, point=point)
            self.assertAlmostEqual(result.forward, expected[0])
            self.assertAlmostEqual(result.turn, expected[1])

    def test_pulling_back_never_commands_reverse_but_turn_remains_available(self) -> None:
        self.activate()
        result = self.update(0.8, point=(NEUTRAL[0] - 0.5, NEUTRAL[1] - 0.15, NEUTRAL[2]))
        self.assertEqual(result.forward, 0.0)
        self.assertAlmostEqual(result.turn, -1.0)

    def test_releasing_either_extended_or_curled_fingers_stops_without_ramp(self) -> None:
        for role, value in (("index", 0.41), ("middle", 0.41), ("ring", 0.54), ("little", 0.54)):
            with self.subTest(role=role):
                self.gesture.reset()
                self.activate()
                result = self.update(0.8, point=(0.5, 0.5, -0.35))
                self.assertGreater(result.forward, 0.0)
                self.assert_stopped(self.update(0.9, curls={**V_SIGN, role: value}))

    def test_hysteresis_retains_held_v_but_cannot_start_a_partial_v(self) -> None:
        relaxed = {"index": 0.35, "middle": 0.35, "ring": 0.60, "little": 0.60}
        self.assert_stopped(self.update(0.0, curls=relaxed))
        self.activate(start=0.1)
        self.assertEqual(self.update(0.9, curls=relaxed).state, "active")

    def test_tracking_loss_clears_motion_and_requires_fresh_center_and_dwell(self) -> None:
        self.activate()
        result = self.update(0.8, point=(0.5, 0.5, -0.35))
        self.assertGreater(result.forward, 0.0)
        self.assert_stopped(self.update(0.9, valid=False))
        new_center = (0.65, -0.3, -0.2)
        result = self.update(1.0, point=new_center)
        self.assertEqual(result.state, "acquiring")
        self.assertEqual((result.forward, result.turn), (0.0, 0.0))
        for index in range(1, 8):
            result = self.update(1.0 + index * 0.1, point=new_center)
        self.assertEqual(result.state, "active")
        self.assertEqual(result.neutral_position, new_center)
        self.assertEqual((result.forward, result.turn), (0.0, 0.0))

    def test_incomplete_unknown_nonfinite_and_out_of_range_curls_stop(self) -> None:
        for role in V_SIGN:
            for value in (None, math.nan, math.inf, -0.01, 1.01, True, "0.1"):
                with self.subTest(role=role, value=value):
                    self.gesture.reset()
                    self.activate()
                    self.assert_stopped(self.update(0.8, curls={**V_SIGN, role: value}))
        for malformed in (None, {}, {"index": 0.1, "middle": 0.1, "ring": 0.9, "pinky": 0.9}, []):
            self.assert_stopped(self.update(0.8, curls=malformed))

    def test_missing_or_invalid_wrist_position_stops(self) -> None:
        for point in (
            None,
            (),
            (0.1, 0.2),
            (0.1, 0.2, 0.3, 0.4),
            (0.1, math.nan, 0.2),
            (0.1, 0.2, math.inf),
            (False, 0, 0),
        ):
            with self.subTest(point=point):
                self.gesture.reset()
                self.activate()
                self.assert_stopped(self.update(0.8, point=point))

    def test_time_gap_or_clock_discontinuity_cancels_and_requires_new_dwell(self) -> None:
        for now in (0.7, 0.6, 0.951, math.nan, math.inf, -1.0, True, "0.8"):
            with self.subTest(now=now):
                self.gesture.reset()
                self.activate()
                self.assert_stopped(self.update(now))
                result = self.update(1.0)
                self.assertEqual(result.state, "acquiring")
                self.assertEqual(result.acquisition_progress, 0.0)

    def test_irregular_valid_update_intervals_use_elapsed_time_not_frame_count(self) -> None:
        for now in (10.0, 10.03, 10.19, 10.32, 10.48, 10.64, 10.69):
            self.assertEqual(self.update(now).state, "acquiring")
        result = self.update(10.71)
        self.assertEqual(result.state, "active")

    def test_reset_is_immediate_and_idempotent(self) -> None:
        self.activate()
        self.assert_stopped(self.gesture.reset())
        self.assert_stopped(self.gesture.reset())
        self.assertEqual(self.update(100.0).state, "acquiring")

    def test_invalid_configuration_rejected(self) -> None:
        for configuration in (
            {"dwell_s": 0},
            {"dwell_radius_m": 0},
            {"deadzone_m": -0.1},
            {"full_forward_m": 0.04},
            {"full_turn_m": 0.03},
            {"max_gap_s": 0},
            {"dwell_s": math.nan},
            {"deadzone_m": True},
        ):
            with self.subTest(configuration=configuration), self.assertRaises(ValueError):
                GESTURE.LocomotionGesture(**configuration)


class TestGestureVelocityMapping(unittest.TestCase):
    def test_wrist_deadzone_is_zero_and_small_positive_travel_clears_balance_threshold(self) -> None:
        gesture = GESTURE.LocomotionGesture()
        for index in range(8):
            gesture.update(curls=V_SIGN, position=NEUTRAL, tracking_valid=True, now_s=index * 0.1)
        for index, displacement in enumerate((0.04, 0.05, 0.06, 0.18, 0.50)):
            with self.subTest(displacement=displacement):
                command = gesture.update(
                    curls=V_SIGN,
                    position=(NEUTRAL[0] + displacement, NEUTRAL[1], NEUTRAL[2]),
                    tracking_valid=True,
                    now_s=0.8 + index * 0.1,
                )
                velocity = GESTURE.map_gesture_to_velocity(command)
                if displacement == 0.04:
                    self.assertEqual(velocity, (0.0, 0.0, 0.0))
                else:
                    self.assertGreaterEqual(velocity[0], 0.08)
                    self.assertLessEqual(velocity[0], 0.25)
                    self.assertGreater(velocity[0], 0.05)
                    self.assertEqual(velocity[1:], (0.0, 0.0))
        self.assertEqual(velocity, (0.25, 0.0, 0.0))

    def test_forward_range_is_monotonic_and_steering_remains_signed(self) -> None:
        previous = 0.0
        for forward in (1e-9, 0.1, 0.5, 1.0):
            for turn in (-1.0, -0.5, 0.0, 0.5, 1.0):
                with self.subTest(forward=forward, turn=turn):
                    velocity = GESTURE.map_gesture_to_velocity(
                        GESTURE.GestureCommand(state="active", forward=forward, turn=turn)
                    )
                    self.assertGreater(velocity[0], previous)
                    self.assertAlmostEqual(velocity[0], 0.08 + 0.17 * forward)
                    self.assertEqual(velocity[1], 0.0)
                    self.assertAlmostEqual(velocity[2], 0.20 * turn)
            previous = velocity[0]

    def test_inactive_or_neutral_commands_never_walk_or_pivot(self) -> None:
        for state in ("idle", "acquiring", "active"):
            for turn in (-1.0, 0.0, 1.0):
                with self.subTest(state=state, turn=turn):
                    command = GESTURE.GestureCommand(state=state, forward=0.0, turn=turn)
                    self.assertEqual(GESTURE.map_gesture_to_velocity(command), (0.0, 0.0, 0.0))
        # Non-active states cannot authorize movement even if a stale axis leaks in.
        for state in ("idle", "acquiring"):
            command = GESTURE.GestureCommand(state=state, forward=1.0, turn=1.0)
            self.assertEqual(GESTURE.map_gesture_to_velocity(command), (0.0, 0.0, 0.0))

    def test_release_and_tracking_loss_clear_velocity_without_a_minimum_speed_leak(self) -> None:
        for loss in ("release", "invalid_tracking"):
            with self.subTest(loss=loss):
                gesture = GESTURE.LocomotionGesture()
                for index in range(8):
                    gesture.update(curls=V_SIGN, position=NEUTRAL, tracking_valid=True, now_s=index * 0.1)
                moved = (NEUTRAL[0] + 0.05, NEUTRAL[1] + 0.15, NEUTRAL[2])
                walking = gesture.update(curls=V_SIGN, position=moved, tracking_valid=True, now_s=0.8)
                self.assertGreater(GESTURE.map_gesture_to_velocity(walking)[0], 0.05)
                stopped = gesture.update(
                    curls=dict.fromkeys(V_SIGN, 0.0) if loss == "release" else V_SIGN,
                    position=moved,
                    tracking_valid=loss != "invalid_tracking",
                    now_s=0.9,
                )
                self.assertEqual(GESTURE.map_gesture_to_velocity(stopped), (0.0, 0.0, 0.0))

    def test_configured_speed_limits_are_used_without_exceeding_the_caps(self) -> None:
        for forward, expected in ((0.1, 0.105), (1.0, 0.15)):
            command = GESTURE.GestureCommand(state="active", forward=forward, turn=-1.0)
            velocity = GESTURE.map_gesture_to_velocity(
                command, min_forward_speed=0.10, max_forward_speed=0.15, max_yaw_speed=0.12
            )
            self.assertAlmostEqual(velocity[0], expected)
            self.assertEqual(velocity[1:], (0.0, -0.12))

    def test_invalid_speed_limits_and_active_inputs_are_rejected(self) -> None:
        command = GESTURE.GestureCommand(state="active", forward=0.5)
        for limits in (
            {"min_forward_speed": 0.0},
            {"min_forward_speed": -0.1},
            {"min_forward_speed": 0.3},
            {"max_forward_speed": math.nan},
            {"max_yaw_speed": -0.1},
            {"min_forward_speed": True},
            {"max_yaw_speed": "0.2"},
        ):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                GESTURE.map_gesture_to_velocity(command, **limits)
        for forward, turn in ((-0.1, 0.0), (1.1, 0.0), (0.1, 1.1), (0.1, -1.1), (math.nan, 0.0), (0.1, True)):
            with self.subTest(forward=forward, turn=turn), self.assertRaises(ValueError):
                GESTURE.map_gesture_to_velocity(GESTURE.GestureCommand(state="active", forward=forward, turn=turn))


if __name__ == "__main__":
    unittest.main()
