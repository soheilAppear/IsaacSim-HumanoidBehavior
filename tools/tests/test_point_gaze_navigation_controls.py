# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise destination selection and stop/restart behavior without Kit or XR."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
    / "point_gaze_navigation.py"
)
SPEC = importlib.util.spec_from_file_location("point_gaze_navigation_under_test", MODULE_PATH)
NAVIGATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NAVIGATION
SPEC.loader.exec_module(NAVIGATION)

POINT = {"index": 0.1, "middle": 0.9, "ring": 0.9, "little": 0.9}
OPEN = dict.fromkeys(POINT, 0.0)
FLOOR = (1.5, 0.0, 0.0)
ZERO = (0.0, 0.0, 0.0)


class TestPointGesture(unittest.TestCase):
    def test_point_requires_index_extended_and_other_fingers_curled(self) -> None:
        self.assertTrue(NAVIGATION.is_pointing(POINT))
        for curls in (OPEN, dict.fromkeys(POINT, 1.0), {**POINT, "middle": 0.1}):
            self.assertFalse(NAVIGATION.is_pointing(curls))

    def test_thumb_is_not_part_of_pointing_recognition(self) -> None:
        for thumb in (None, 0.0, 1.0, math.nan):
            self.assertTrue(NAVIGATION.is_pointing({**POINT, "thumb": thumb}))

    def test_hysteresis_does_not_authorize_partial_point_entry(self) -> None:
        curls = {"index": 0.35, "middle": 0.55, "ring": 0.55, "little": 0.55}
        self.assertFalse(NAVIGATION.is_pointing(curls))
        self.assertTrue(NAVIGATION.is_pointing(curls, retaining=True))

    def test_incomplete_nonfinite_out_of_range_or_non_numeric_curls_are_not_points(self) -> None:
        for role in POINT:
            for value in (None, True, "0.1", math.nan, math.inf, -0.01, 1.01):
                with self.subTest(role=role, value=value):
                    self.assertFalse(NAVIGATION.is_pointing({**POINT, role: value}))
            missing = dict(POINT)
            del missing[role]
            self.assertFalse(NAVIGATION.is_pointing(missing))
        for curls in (None, (), {}, "index"):
            self.assertFalse(NAVIGATION.is_pointing(curls))


class TestPointNavigationControls(unittest.TestCase):
    def setUp(self) -> None:
        self.navigation = NAVIGATION.PointNavigation()
        self.update(0.0, curls=OPEN)

    def update(self, now: float, **overrides: object) -> NAVIGATION.NavigationCommand:
        """Supply a measured pointing floor hit unless a test varies input."""
        values = {
            "curls": POINT,
            "hand_valid": True,
            "point_hit": FLOOR,
            "point_walkable": True,
            "path_clear": True,
            "robot_position": (0.0, 0.0, 0.8),
            "robot_yaw": 0.0,
            "now_s": now,
        }
        values.update(overrides)
        return self.navigation.update(**values)

    def activate(self, start: float = 0.1, **overrides: object) -> NAVIGATION.NavigationCommand:
        """Deliver a complete continuous dwell and assert physical intent starts."""
        result = None
        for index in range(7):
            result = self.update(start + index * 0.1, **overrides)
        self.assertEqual(result.state, "active")
        self.assertGreaterEqual(result.velocity[0], 0.08)
        return result

    def test_construction_and_reset_require_measured_release_before_any_dwell(self) -> None:
        for unused in range(2):
            self.navigation.reset()
            for index in range(10):
                result = self.update(index * 0.1)
                self.assertEqual(result.state, "awaiting_release")
                self.assertEqual(result.velocity, ZERO)
            self.assertEqual(self.update(1.0, curls=OPEN).state, "idle")
            self.activate(start=1.1)

    def test_full_pointing_dwell_has_no_early_motion_and_keeps_hand_tracking(self) -> None:
        for index in range(6):
            result = self.update(0.1 + index * 0.1)
            self.assertEqual(result.state, "acquiring")
            self.assertEqual(result.velocity, ZERO)
            self.assertLess(result.acquisition_progress, 1.0)
            self.assertFalse(result.owns_hand)
        result = self.update(0.7)
        self.assertEqual(result.state, "active")
        self.assertEqual(result.target, FLOOR)
        self.assertAlmostEqual(result.velocity[0], 0.60)
        self.assertEqual(result.velocity[1:], (0.0, 0.0))
        self.assertFalse(self.navigation.owns_hand)
        self.assertIs(self.navigation.command, result)

    def test_point_ray_must_hit_walkable_floor_and_route_must_be_clear(self) -> None:
        for invalid in (
            {"point_walkable": False},
            {"path_clear": False},
            {"point_hit": None},
        ):
            with self.subTest(invalid=invalid):
                self.navigation.reset(require_release=False)
                for index in range(12):
                    result = self.update(index * 0.1, **invalid)
                    self.assertEqual(result.state, "idle")
                    self.assertEqual(result.velocity, ZERO)

    def test_pointing_alone_selects_exact_blue_endpoint_after_stable_dwell(self) -> None:
        target = (1.5, 0.2, 0.0)
        result = self.activate(point_hit=target)
        self.assertEqual(result.target, target)
        self.assertGreater(result.velocity[2], 0.0)

    def test_candidate_drift_restarts_dwell_while_small_jitter_is_allowed(self) -> None:
        for index in range(5):
            self.update(0.1 + index * 0.1)
        shifted = (1.7, 0.0, 0.0)
        result = self.update(0.6, point_hit=shifted)
        self.assertEqual(result.state, "acquiring")
        self.assertEqual(result.acquisition_progress, 0.0)
        for index in range(1, 7):
            jittered = (1.7 + 0.04 * (index % 2), 0.0, 0.0)
            result = self.update(0.6 + index * 0.1, point_hit=jittered)
            if index < 6:
                self.assertEqual(result.state, "acquiring")
        self.assertEqual(result.state, "active")

    def test_point_hit_changes_after_lock_do_not_retarget_or_cancel(self) -> None:
        original = self.activate()
        result = self.update(
            0.8,
            point_hit=None,
            point_walkable=False,
        )
        self.assertEqual(result.state, "active")
        self.assertEqual(result.target, original.target)
        self.assertEqual(result.velocity, original.velocity)

    def test_far_floor_targets_through_twelve_meters_can_be_acquired(self) -> None:
        for distance in (4.01, 6.0, 8.0, 10.0, 12.0):
            with self.subTest(distance=distance):
                self.navigation.reset(require_release=False)
                target = (distance, 0.0, 0.0)
                result = self.activate(point_hit=target)
                self.assertEqual(result.target, target)
                self.assertEqual(result.velocity, (0.65, 0.0, 0.0))

    def test_far_dwell_tolerates_bounded_jitter_and_locks_exact_current_hit(self) -> None:
        self.update(0.1, point_hit=(6.0, 0.0, 0.0))
        for index in range(1, 7):
            candidate = (6.0, (-1) ** index * 0.23, 0.0)
            result = self.update(0.1 + index * 0.1, point_hit=candidate)
            if index < 6:
                self.assertEqual(result.state, "acquiring")
        self.assertEqual(result.state, "active")
        self.assertEqual(result.target, candidate)
        self.assertAlmostEqual(self.navigation._acquisition_radius_m, 0.24)

    def test_far_dwell_radius_is_capped_and_cannot_grow_with_drifting_ray(self) -> None:
        for initial, inside, outside, radius in ((6.0, 6.239, 6.249, 0.24), (7.9, 7.601, 7.599, 0.30)):
            with self.subTest(initial=initial):
                self.navigation.reset(require_release=False)
                self.update(0.0, point_hit=(initial, 0.0, 0.0))
                result = self.update(0.1, point_hit=(inside, 0.0, 0.0))
                self.assertGreater(result.acquisition_progress, 0.0)
                self.assertAlmostEqual(self.navigation._acquisition_radius_m, radius)
                result = self.update(0.2, point_hit=(outside, 0.0, 0.0))
                self.assertEqual(result.acquisition_progress, 0.0)
                self.assertEqual(self.navigation._acquisition_origin, (outside, 0.0, 0.0))

    def test_slow_ray_creep_restarts_dwell_instead_of_dragging_anchor(self) -> None:
        self.update(0.1, point_hit=(6.0, 0.0, 0.0))
        for index in range(1, 10):
            result = self.update(0.1 + index * 0.1, point_hit=(6.0 + 0.1 * index, 0.0, 0.0))
            self.assertEqual(result.state, "acquiring")
            self.assertEqual(result.velocity, ZERO)

    def test_far_tolerance_does_not_bypass_obstacle_or_surface_checks(self) -> None:
        for rejection in ({"path_clear": False}, {"point_walkable": False}, {"hand_valid": False}):
            with self.subTest(rejection=rejection):
                self.navigation.reset(require_release=False)
                for index in range(6):
                    self.update(index * 0.1, point_hit=(6.0, 0.0, 0.0))
                result = self.update(0.6, point_hit=(6.0, 0.2, 0.0), **rejection)
                self.assertNotEqual(result.state, "active")
                self.assertEqual(result.velocity, ZERO)

    def test_release_stops_immediately_and_allows_a_fresh_selection_dwell(self) -> None:
        self.activate()
        result = self.update(0.8, curls=OPEN)
        self.assertEqual(result.state, "idle")
        self.assertEqual(result.velocity, ZERO)
        self.assertIsNone(result.target)
        self.assertEqual(self.update(0.9).state, "acquiring")
        self.assertEqual(self.navigation.command.velocity, ZERO)

    def test_missing_tracking_or_digits_stops_and_cannot_clear_release_interlock(self) -> None:
        for invalid in ({"hand_valid": False}, {"curls": None}, {"curls": {"index": 0.1}}):
            with self.subTest(invalid=invalid):
                self.navigation.reset(require_release=False)
                self.activate()
                result = self.update(0.8, **invalid)
                self.assertEqual(result.state, "awaiting_release")
                self.assertEqual(result.velocity, ZERO)
                self.assertIsNone(result.target)
                for index in range(10):
                    self.assertEqual(self.update(0.9 + index * 0.1).state, "awaiting_release")
                self.assertEqual(self.update(1.9, curls=OPEN).state, "idle")
                self.assertEqual(self.update(2.0).state, "acquiring")

    def test_clock_discontinuity_requires_release_instead_of_resuming_held_point(self) -> None:
        for now in (0.7, 0.69, 1.151, -1.0, math.nan, math.inf, True, "0.8"):
            with self.subTest(now=now):
                self.navigation.reset(require_release=False)
                self.activate()
                result = self.update(now)
                self.assertEqual(result.state, "awaiting_release")
                self.assertEqual(result.velocity, ZERO)
                self.assertEqual(self.update(2.0).state, "awaiting_release")
                self.assertEqual(self.update(2.1, curls=OPEN).state, "idle")

    def test_slow_frame_jitter_uses_prior_cadence_and_survives_callback_bursts(self) -> None:
        self.activate()
        self.assertAlmostEqual(self.navigation.observation_gap_limit_s, 0.45)
        self.assertEqual(self.update(1.04).state, "active")
        self.assertEqual(self.navigation.observation_gap_limit_s, 0.75)
        for index in range(1, 21):
            self.assertEqual(self.update(1.04 + index * 0.0001).state, "active")
        self.assertEqual(self.update(1.442).state, "active")

    def test_fast_cadence_does_not_let_one_long_stall_expand_its_own_limit(self) -> None:
        self.navigation.reset(require_release=False)
        for index in range(61):
            self.update(index * 0.01)
        self.assertEqual(self.navigation.state, "active")
        self.assertEqual(self.navigation.observation_gap_limit_s, 0.25)
        result = self.update(0.851)
        self.assertEqual(result.state, "awaiting_release")
        self.assertEqual(result.velocity, ZERO)

    def test_open_hand_warmup_learns_first_slow_frame_then_requires_full_pointing_dwell(self) -> None:
        for now in (0.001, 0.002):
            self.assertEqual(self.update(now, curls=OPEN).state, "idle")
        interrupted = self.update(0.302, curls=OPEN)
        self.assertEqual(interrupted.state, "awaiting_release")
        self.assertEqual(interrupted.velocity, ZERO)
        self.assertEqual(self.navigation.observation_gap_limit_s, 0.75)
        self.assertEqual(self.update(0.303, curls=OPEN).state, "idle")
        for now in (0.304, 0.604, 0.605, 0.606):
            result = self.update(now)
            self.assertEqual(result.state, "acquiring")
            self.assertEqual(result.velocity, ZERO)
        self.assertEqual(self.update(0.906).state, "active")

    def test_learning_slow_frame_never_rearms_a_held_point_without_measured_release(self) -> None:
        self.navigation.reset(require_release=False)
        self.update(0.0)
        self.update(0.001)
        self.assertEqual(self.update(0.301).state, "awaiting_release")
        for now in (0.302, 0.602, 0.603, 0.903, 1.203):
            result = self.update(now)
            self.assertEqual(result.state, "awaiting_release")
            self.assertEqual(result.velocity, ZERO)
        self.assertEqual(self.update(1.204, curls=OPEN).state, "idle")
        self.assertEqual(self.update(1.205).state, "acquiring")

    def test_invalid_hand_during_slow_frame_cannot_train_the_gap_allowance(self) -> None:
        self.update(0.001, curls=OPEN)
        result = self.update(0.301, hand_valid=False)
        self.assertEqual(result.state, "awaiting_release")
        self.assertEqual(result.velocity, ZERO)
        self.assertEqual(self.navigation.observation_gap_limit_s, 0.25)
        self.assertEqual(self.update(0.302).state, "awaiting_release")

    def test_adaptive_gap_remains_bounded_and_resets_with_navigation(self) -> None:
        self.activate()
        self.update(0.9)
        self.assertEqual(self.navigation.observation_gap_limit_s, 0.75)
        self.assertEqual(self.update(1.65).state, "active")
        result = self.update(2.401)
        self.assertEqual(result.state, "awaiting_release")
        self.assertEqual(result.velocity, ZERO)
        self.assertEqual(self.navigation.observation_gap_limit_s, 0.25)

    def test_slow_frame_allowance_never_delays_hand_loss_or_point_release(self) -> None:
        for invalid in ({"hand_valid": False}, {"curls": None}, {"curls": OPEN}):
            with self.subTest(invalid=invalid):
                self.navigation.reset(require_release=False)
                self.activate()
                self.update(0.9)
                self.assertEqual(self.navigation.observation_gap_limit_s, 0.75)
                result = self.update(1.3, **invalid)
                self.assertEqual(result.velocity, ZERO)
                self.assertNotEqual(result.state, "active")

    def test_invalid_pose_stops_active_motion_without_fallback_origin(self) -> None:
        for invalid in (
            {"robot_position": None},
            {"robot_position": (1.0, 2.0)},
            {"robot_position": (0.0, math.nan, 0.8)},
            {"robot_position": (True, 0.0, 0.8)},
            {"robot_yaw": math.inf},
            {"robot_yaw": True},
            {"robot_yaw": "0.0"},
        ):
            with self.subTest(invalid=invalid):
                self.navigation.reset(require_release=False)
                self.activate()
                self.assertEqual(self.update(0.8, **invalid).velocity, ZERO)
                self.assertEqual(self.navigation.state, "awaiting_release")

    def test_target_range_and_forward_cone_reject_unreachable_or_rear_destinations(self) -> None:
        for target in ((0.49, 0.0, 0.0), (12.01, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.5, 1.0, 0.0)):
            with self.subTest(target=target):
                self.navigation.reset(require_release=False)
                for index in range(10):
                    result = self.update(index * 0.1, point_hit=target)
                    self.assertEqual(result.state, "idle")
                    self.assertEqual(result.velocity, ZERO)

    def test_invalid_hits_and_overflowing_distances_cannot_authorize_navigation(self) -> None:
        for target in ((), (1.0, 2.0), (math.nan, 0.0, 0.0), (False, 0.0, 0.0), (1e308, 1e308, 0.0)):
            with self.subTest(target=target):
                self.navigation.reset(require_release=False)
                result = self.update(0.0, point_hit=target)
                self.assertEqual(result.velocity, ZERO)
                self.assertEqual(result.state, "idle")

    def test_approach_slows_but_does_not_fall_under_the_walking_actor_threshold(self) -> None:
        self.activate()
        previous = 0.65
        for index, x in enumerate((0.4, 0.8, 1.0, 1.19)):
            result = self.update(0.8 + index * 0.1, robot_position=(x, 0.0, 0.8))
            self.assertEqual(result.state, "active")
            self.assertGreaterEqual(result.velocity[0], 0.08)
            self.assertLessEqual(result.velocity[0], previous)
            self.assertEqual(result.velocity[1:], (0.0, 0.0))
            previous = result.velocity[0]

    def test_faster_cruise_preserves_previous_near_target_approach_and_stop(self) -> None:
        """Increase longer-route speed without increasing commands near arrival."""
        target = (2.0, 0.0, 0.0)
        for heading in (-0.5, 0.0, 0.5):
            approaches = []
            cruises = []
            for navigation in (
                NAVIGATION.PointNavigation(max_forward_speed=0.45, slow_radius_m=1.20),
                NAVIGATION.PointNavigation(),
            ):
                self.navigation = navigation
                self.update(0.0, curls=OPEN)
                cruise = self.activate(robot_yaw=heading, point_hit=target)
                cruises.append(cruise.velocity[0])
                samples = []
                for index, distance in enumerate((1.20, 1.10, 0.80, 0.70, 0.60, 0.50, 0.46, 0.40, 0.31, 0.29)):
                    result = self.update(
                        0.8 + index * 0.1,
                        robot_position=(target[0] - distance, 0.0, 0.8),
                        robot_yaw=heading,
                    )
                    samples.append(result)
                approaches.append(samples)
            with self.subTest(heading=heading):
                self.assertAlmostEqual(cruises[1], cruises[0] * 0.65 / 0.45)
                for previous, faster in zip(*approaches):
                    self.assertEqual(faster.state, previous.state)
                    for current_velocity, previous_velocity in zip(faster.velocity, previous.velocity):
                        self.assertAlmostEqual(current_velocity, previous_velocity)
                self.assertEqual(approaches[1][-1].state, "arrived")
                self.assertEqual(approaches[1][-1].velocity, ZERO)

    def test_arrival_latches_zero_despite_drift_or_new_point_until_release(self) -> None:
        self.activate()
        result = self.update(0.8, robot_position=(1.21, 0.0, 0.8))
        self.assertEqual(result.state, "arrived")
        self.assertEqual(result.velocity, ZERO)
        self.assertEqual(result.target, FLOOR)
        result = self.update(0.9, point_hit=(3.0, 0.0, 0.0))
        self.assertEqual(result.state, "arrived")
        self.assertEqual(result.velocity, ZERO)
        self.assertEqual(self.update(1.0, curls=OPEN).state, "idle")
        self.assertEqual(self.update(1.1).state, "acquiring")

    def test_obstacle_appearance_stops_and_disappearance_does_not_auto_resume(self) -> None:
        self.activate()
        result = self.update(0.8, path_clear=False)
        self.assertEqual(result.state, "blocked")
        self.assertEqual(result.velocity, ZERO)
        result = self.update(0.9, path_clear=True)
        self.assertEqual(result.state, "blocked")
        self.assertEqual(result.velocity, ZERO)
        self.assertEqual(self.update(1.0, curls=OPEN).state, "idle")

    def test_heading_loss_blocks_instead_of_reverse_or_in_place_pivot(self) -> None:
        self.activate()
        result = self.update(0.8, robot_yaw=math.pi)
        self.assertEqual(result.state, "blocked")
        self.assertEqual(result.velocity, ZERO)

    def test_world_heading_rotation_preserves_forward_motion_and_signed_turns(self) -> None:
        for heading in (-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi):
            for angle in (-0.5, 0.0, 0.5):
                with self.subTest(heading=heading, angle=angle):
                    self.navigation.reset(require_release=False)
                    target = (2.0 * math.cos(heading + angle), 2.0 * math.sin(heading + angle), 0.0)
                    result = self.activate(robot_yaw=heading, point_hit=target)
                    self.assertGreater(result.velocity[0], 0.05)
                    self.assertLessEqual(result.velocity[0], 0.65)
                    self.assertEqual(result.velocity[1], 0.0)
                    self.assertLessEqual(abs(result.velocity[2]), 0.20)
                    self.assertAlmostEqual(result.velocity[2], max(-0.2, min(0.2, angle * 0.8)))

    def test_irregular_frame_intervals_require_elapsed_dwell_not_frame_count(self) -> None:
        for now in (0.03, 0.11, 0.19, 0.29, 0.41, 0.55, 0.62):
            self.assertEqual(self.update(now).state, "acquiring")
        self.assertEqual(self.update(0.64).state, "active")

    def test_nonboolean_validity_inputs_do_not_authorize_motion(self) -> None:
        for flag in ("hand_valid", "point_walkable", "path_clear"):
            for value in (1, "true", None):
                with self.subTest(flag=flag, value=value):
                    self.navigation.reset(require_release=False)
                    for index in range(10):
                        self.assertEqual(self.update(index * 0.1, **{flag: value}).velocity, ZERO)

    def test_constructor_rejects_nonfinite_and_inconsistent_configuration(self) -> None:
        for configuration in (
            {"dwell_s": 0.0},
            {"dwell_radius_m": math.nan},
            {"dwell_radius_m": True},
            {"max_dwell_radius_m": 0.17},
            {"dwell_radius_distance_scale": 0.0},
            {"dwell_radius_distance_scale": math.inf},
            {"max_gap_s": -1.0},
            {"max_adaptive_gap_s": 0.2},
            {"min_target_distance_m": 0.2},
            {"max_target_distance_m": 0.4},
            {"slow_radius_m": 0.2},
            {"max_forward_speed": 0.04},
            {"max_heading_rad": math.pi / 2},
            {"yaw_gain": "0.8"},
        ):
            with self.subTest(configuration=configuration), self.assertRaises(ValueError):
                NAVIGATION.PointNavigation(**configuration)


class TestFactoryPointNavigationControls(unittest.TestCase):
    def setUp(self) -> None:
        self.navigation = NAVIGATION.PointNavigation(
            allow_routing=True, allow_turn_in_place=True, max_target_distance_m=50
        )
        self.now = 0.0
        self.tick(curls=OPEN)

    def tick(self, **overrides):
        values = dict(
            curls=POINT,
            hand_valid=True,
            point_hit=(6, 0, 0),
            point_walkable=True,
            path_clear=True,
            robot_position=(0, 0, 0.8),
            robot_yaw=0.0,
            now_s=self.now,
            turn_clear=True,
        )
        values.update(overrides)
        self.now += 0.1
        return self.navigation.update(**values)

    def dwell(self, **overrides):
        for unused in range(7):
            result = self.tick(**overrides)
        return result

    def test_behind_target_turns_using_bounded_actor_yaw_and_never_moves_backward(self):
        result = self.dwell(point_hit=(-4, 0, 0))
        self.assertEqual(result.state, "active")
        self.assertEqual(result.velocity, (0, 0, 0.4))
        for heading in (1, 2, 2.8):
            result = self.tick(robot_yaw=heading)
            self.assertEqual(result.velocity[0], 0)
        result = self.tick(robot_yaw=3.0)
        self.assertGreater(result.velocity[0], 0)
        self.assertEqual(result.target, (-4, 0, 0))

    def test_turning_clearance_failure_latches_stop_until_release(self):
        result = self.dwell(point_hit=(0, 4, 0), turn_clear=False)
        self.assertEqual(result.state, "blocked")
        self.assertEqual(result.velocity, ZERO)
        self.assertEqual(self.tick(turn_clear=True).state, "blocked")
        self.assertEqual(self.tick(curls=OPEN).state, "idle")

    def test_stepping_turn_uses_separate_constant_speed_and_preserves_curve_limit(self):
        result = self.dwell(point_hit=(0, 4, 0))
        self.assertEqual(result.velocity, (0, 0, 0.4))
        result = self.tick(robot_yaw=math.pi / 2 - 0.3)
        self.assertEqual(result.velocity, (0, 0, 0.4))
        result = self.tick(robot_yaw=math.pi / 2 - 0.2)
        self.assertGreater(result.velocity[0], 0)
        self.assertAlmostEqual(result.velocity[2], 0.16)
        result = self.tick(robot_yaw=math.pi / 2 - 0.4)
        self.assertGreater(result.velocity[0], 0)
        self.assertEqual(result.velocity[2], 0.2)
        self.assertEqual(self.navigation.max_requested_yaw_speed, 0.4)
        self.assertEqual(self.navigation.max_yaw_speed, 0.2)

    def test_clockwise_stepping_turn_has_negative_sign_without_xy_command(self):
        result = self.dwell(point_hit=(0, -4, 0))
        self.assertEqual(result.velocity, (0, 0, -0.4))

    def test_unresponsive_turn_times_out_and_requires_release(self):
        self.dwell(point_hit=(0, 4, 0))
        for unused in range(152):
            result = self.tick()
        self.assertEqual(result.state, "blocked")
        self.assertEqual(result.velocity, ZERO)
        self.assertIn("no heading progress", result.detail)
        self.assertIsNone(self.navigation._turn_progress_time_s)
        self.assertEqual(self.tick(robot_yaw=1.5).state, "blocked")
        self.assertEqual(self.tick(curls=OPEN).state, "idle")

    def test_measured_heading_progress_renews_turn_watchdog(self):
        self.dwell(point_hit=(0, 4, 0))
        for index in range(350):
            # Slow but real progress: .05 rad every ten wall seconds. This
            # accommodates low real-time-factor rendering without a false stop.
            result = self.tick(robot_yaw=index * 0.0005)
            self.assertEqual(result.state, "active")
            self.assertEqual(result.velocity, (0, 0, 0.4))

    def test_heading_jitter_cannot_keep_renewing_watchdog_without_a_new_best(self):
        self.dwell(point_hit=(0, 4, 0))
        for index in range(160):
            result = self.tick(robot_yaw=0.06 if index % 2 else 0)
        self.assertEqual(result.state, "blocked")
        self.assertEqual(result.velocity, ZERO)

    def test_turn_watchdog_clears_on_phase_exit_release_reset_and_new_corner(self):
        self.dwell(point_hit=(0, 4, 0))
        self.assertIsNotNone(self.navigation._turn_progress_time_s)
        self.tick(robot_yaw=math.pi / 2)
        self.assertIsNone(self.navigation._turn_progress_time_s)
        self.tick(robot_yaw=0)
        self.assertIsNotNone(self.navigation._turn_progress_time_s)
        self.tick(curls=OPEN)
        self.assertIsNone(self.navigation._turn_progress_time_s)
        self.navigation.reset(require_release=False)
        self.dwell(path_clear=False)
        self.tick(route=((0, 2, 0), (6, 0, 0)))
        self.tick()
        self.assertIsNotNone(self.navigation._turn_progress_time_s)
        self.tick(robot_position=(0, 1.9, 0.8))
        self.assertIsNone(self.navigation._turn_progress_time_s)
        self.navigation.reset()
        self.assertIsNone(self.navigation._turn_progress_time_s)

    def test_forward_curved_steering_also_requires_fresh_body_turn_clearance(self):
        goal = (4 * math.cos(0.4), 4 * math.sin(0.4), 0)
        result = self.dwell(point_hit=goal)
        self.assertEqual(result.state, "active")
        self.assertGreater(result.velocity[0], 0)
        self.assertGreater(result.velocity[2], 0.01)
        result = self.tick(turn_clear=False)
        self.assertEqual(result.state, "blocked")
        self.assertEqual(result.velocity, ZERO)

    def test_aligned_straight_corridor_does_not_require_room_for_unused_full_turn(self):
        result = self.dwell(turn_clear=False)
        self.assertEqual(result.state, "active")
        self.assertGreater(result.velocity[0], 0)
        self.assertEqual(result.velocity[2], 0)

    def test_blocked_direct_route_locks_goal_while_waiting_with_zero_velocity(self):
        result = self.dwell(path_clear=False)
        self.assertEqual(result.state, "planning")
        self.assertEqual(result.target, (6, 0, 0))
        for unused in range(5):
            result = self.tick(point_hit=(0, 20, 0), path_clear=False)
            self.assertEqual(result.target, (6, 0, 0))
            self.assertEqual(result.velocity, ZERO)

    def test_route_install_and_corner_advance_wait_for_new_clearance(self):
        self.dwell(path_clear=False)
        route = ((2, 1, 0), (4, 1, 0), (6, 0, 0))
        result = self.tick(route=route, path_clear=False)
        self.assertEqual(result.state, "active")
        self.assertEqual(result.velocity, ZERO)
        self.assertEqual(self.navigation.waypoint, route[0])
        self.assertFalse(self.navigation.is_final_waypoint)
        result = self.tick(robot_position=(1.92, 1, 0.8), robot_yaw=0)
        self.assertEqual(result.velocity, ZERO)
        self.assertEqual(self.navigation.waypoint, route[1])
        result = self.tick(path_clear=False)
        self.assertEqual(result.state, "blocked")
        self.assertEqual(result.velocity, ZERO)

    def test_route_final_arrival_and_short_intermediate_tolerance(self):
        self.dwell(path_clear=False)
        self.tick(route=((2, 0, 0), (6, 0, 0)))
        self.tick(robot_position=(1.8, 0, 0.8))
        self.assertEqual(self.navigation.waypoint, (2, 0, 0))
        self.tick(robot_position=(1.9, 0, 0.8))
        self.assertEqual(self.navigation.waypoint, (6, 0, 0))
        self.assertTrue(self.navigation.is_final_waypoint)
        result = self.tick(robot_position=(5.72, 0, 0.8))
        self.assertEqual(result.state, "arrived")
        self.assertEqual(result.velocity, ZERO)

    def test_release_or_tracking_loss_cancels_planning_and_rejects_stale_route(self):
        for interruption in ({"curls": OPEN}, {"hand_valid": False}):
            self.navigation.reset(require_release=False)
            self.dwell(path_clear=False)
            result = self.tick(route=((6, 0, 0),), **interruption)
            self.assertIsNone(result.target)
            self.assertEqual(result.velocity, ZERO)
            self.assertEqual(self.navigation._route, ())

    def test_failed_or_invalid_plans_cannot_authorize_motion(self):
        for values in (
            {"route_failed": True},
            {"route": ()},
            {"route": ((5, 0, 0),)},
            {"route": ((math.nan, 0, 0), (6, 0, 0))},
        ):
            self.navigation.reset(require_release=False)
            self.dwell(path_clear=False)
            result = self.tick(**values)
            self.assertEqual(result.state, "blocked")
            self.assertEqual(result.velocity, ZERO)


if __name__ == "__main__":
    unittest.main()
