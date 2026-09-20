# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify pointing-only walking while preserving independent gaze and hand tracking."""

from __future__ import annotations

import math
import json
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from humanoid_test_support import CARB, HUMANOID, Array, pose
from pxr import Gf
from test_finger_controls import SkeletonDevice, skeleton_positions

POINT = {"index": 0.10, "middle": 0.90, "ring": 0.90, "little": 0.90}
OPEN = dict.fromkeys(POINT, 0.0)
FLOOR = Gf.Vec3d(1.5, 0.0, 0.0)


class OpticalSkeleton(SkeletonDevice):
    """Expose a complete virtual-world optical skeleton with a selectable source."""

    def __init__(self, example, curls=None):
        super().__init__(skeleton_positions(example, curls or POINT))
        self.flags = 0xA  # Position is valid and currently tracked, not extrapolated.
        self.source = "hand"

    def get_hand_tracking_data_source(self):
        return self.source

    def get_virtual_world_pose_desc(self, name=""):
        return super().get_virtual_world_pose_desc(name or "wrist")


class TestGestureIntegrationControls(unittest.TestCase):
    def setUp(self) -> None:
        with patch.dict(HUMANOID.os.environ, {"G1_GESTURE_WALKING": "1", "G1_WEBCAM_TRACKING": "0"}):
            self.ex = HUMANOID.HumanoidExample()
        self.left = OpticalSkeleton(self.ex)
        self.right = OpticalSkeleton(self.ex, {"index": 0.8})
        self.ex._xr_core = object()
        self.ex._get_xr_input_device = lambda handle: self.left if handle.endswith("left") else self.right
        self.ex._base_command = torch.zeros(3)
        self.ex._controller_command = torch.zeros(3)
        self.ex._keyboard_command = torch.zeros(3)
        self.ex._read_xr_controller_axes = Mock(return_value=(1.0, 1.0))
        self.ex._read_gamepad_controller_axes = Mock(side_effect=AssertionError("Gamepad cannot own gesture walking"))
        self.ex._update_headset_gait_command = Mock(return_value=1.0)
        self.ex._log_physics_step_error = Mock()
        self.ex._pause_faulted_articulation = Mock()
        # Existing root-watchdog tests isolate its original timing/magnitude
        # gates. Support-motion integration has dedicated native-pose tests below.
        self.ex._update_walking_support_motion = Mock(return_value=(0.11, 0.26, 0.16, 0.36))
        self.measured_curls = {"thumb": 0.1, "index": 0.2, "middle": 0.3, "ring": 0.4, "little": 0.5, "thumb_yaw": 0.2}
        robot = Mock()
        robot.is_physics_tensor_entity_valid.return_value = True
        robot.dof_names = ["waist_yaw_joint"]
        robot.get_dof_limits.return_value = (Array([[-1.0]]), Array([[1.0]]))
        robot.get_dof_positions.return_value = Array([[0.0]])
        robot.get_dof_velocities.return_value = Array([[0.0]])
        self.ex.g1 = types.SimpleNamespace(
            robot=robot,
            forward=Mock(),
            has_finger_control=lambda: True,
            get_finger_curls=Mock(side_effect=lambda side: dict(self.measured_curls)),
            set_finger_curls=Mock(),
            get_finger_joint_range=lambda side, role: (0.0, 1.0),
            invalidate_physics_state=Mock(),
        )
        self.ex._contact_reader = types.SimpleNamespace(
            read=Mock(return_value=({"left": [], "right": []}, "ok")), invalidate=Mock()
        )
        self.ex._get_g1_base_pose_for_arms = Mock(return_value=(Gf.Vec3d(0.0, 0.0, 0.8), 0.0))
        self.ex._navigation_scene = Mock()
        self.ex._navigation_scene.cast_ray.return_value = {
            "position": FLOOR,
            "path": "/World/GroundPlane/CollisionPlane",
            "walkable": True,
        }
        self.ex._navigation_scene.is_floor_point.return_value = True
        self.ex._navigation_scene.is_path_clear.return_value = True
        self.ex._navigation_scene.is_turn_clear.return_value = True
        self.ex._navigation_scene.last_error = None
        self.ex._navigation_scene.last_blocking_paths = ()
        self.gaze = types.SimpleNamespace(
            valid=True,
            hit_valid=True,
            source="eye_tracker",
            sample_time=0.0,
            origin=Gf.Vec3d(0, 0, 1.5),
            direction=Gf.Vec3d(1, 0, -1).GetNormalized(),
            hit_position=FLOOR,
            hit_object_path="/World/GroundPlane/CollisionPlane",
        )
        self.ex._eye_gaze_tracker = types.SimpleNamespace(latest=self.gaze, _elapsed_time=0.0, update=Mock())

    def refresh_gaze(self, now: float) -> None:
        """Expose a newly sampled measured gaze while leaving its validity configurable."""
        self.gaze.sample_time = now
        self.ex._eye_gaze_tracker._elapsed_time = now

    def tick(self, now: float, *, refresh_gaze: bool = True) -> None:
        if refresh_gaze:
            self.refresh_gaze(now)
        with patch.object(HUMANOID.time, "perf_counter", return_value=now):
            self.ex._update_controller_command(0.01)
        self.assertIsNone(self.ex._articulation_health_fault)

    def release(self, now: float = 0.0) -> None:
        self.left.positions = skeleton_positions(self.ex, OPEN)
        self.tick(now)

    def activate(self) -> None:
        self.ex._reset_gesture_locomotion()
        self.release()
        self.left.positions = skeleton_positions(self.ex, POINT)
        for index in range(1, 8):
            self.tick(index * 0.1)
        self.assertEqual(self.ex._gesture_command.state, "active")

    def physics_tick(self, now: float, *, check_balance: bool = False, check_motion: bool = False) -> None:
        self.ex._physics_ready = True
        if not check_balance:
            self.ex._check_gesture_balance = Mock(return_value=True)
        if not check_motion:
            self.ex._check_gesture_motion = Mock(return_value=True)
        self.ex._poll_webcam_tracking = Mock()
        self.ex._update_g1_fingers = Mock()
        self.ex._update_g1_arms_from_hand_tracking = Mock()
        self.ex._update_head_camera_view = Mock()
        self.ex._eye_gaze_tracker.update.reset_mock()
        self.ex._collect_all_behavioral_data = Mock()
        self.refresh_gaze(now)
        with patch.object(HUMANOID.time, "perf_counter", return_value=now):
            self.ex.on_physics_step(0.01, None)

    def test_blue_floor_ray_requires_dwell_then_requests_forward_without_wrist_travel(self) -> None:
        self.release()
        self.left.positions = skeleton_positions(self.ex, POINT)
        for index in range(1, 7):
            self.tick(index * 0.1)
            self.assertEqual(self.ex._gesture_command.state, "acquiring")
            np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])
        self.tick(0.7)
        self.assertEqual(self.ex._gesture_command.state, "active")
        np.testing.assert_allclose(self.ex._controller_command, [0.60, 0, 0])
        self.assertEqual(self.ex._gesture_command.target, tuple(FLOOR))
        self.ex._navigation_scene.cast_ray.assert_called()

    def test_locked_world_target_survives_gaze_changes_and_turns_with_measured_robot_heading(self) -> None:
        self.activate()
        self.gaze.hit_position = Gf.Vec3d(0, 3, 0)
        self.gaze.valid = False
        self.left.transform = pose((2.0, 1.0, 0.0), yaw=90)
        self.ex._get_g1_base_pose_for_arms.return_value = (Gf.Vec3d(0, 0, 0.8), 0.3)
        self.tick(0.8)
        self.assertEqual(self.ex._gesture_command.target, tuple(FLOOR))
        self.assertGreater(float(self.ex._controller_command[0]), 0.0)
        self.assertAlmostEqual(float(self.ex._controller_command[2]), -0.20, places=6)

    def test_scene_query_and_blue_line_use_the_same_filtered_ray(self) -> None:
        self.release()
        origin = Gf.Vec3d(0.1, 0.2, 1.2)
        direction = Gf.Vec3d(1.0, 0.0, -0.5).GetNormalized()
        self.ex._read_navigation_hand = Mock(return_value=(POINT, origin, origin - 0.08 * direction, origin))

        def floor_hit(ray_origin, ray_direction):
            distance = -ray_origin[2] / ray_direction[2]
            return {
                "position": ray_origin + distance * ray_direction,
                "distance": distance,
                "path": "/World/GroundPlane/CollisionPlane",
                "walkable": True,
            }

        self.ex._navigation_scene.cast_ray.side_effect = floor_hit
        self.tick(0.1)
        raw_origin = Gf.Vec3d(0.12, 0.22, 1.22)
        raw_direction = Gf.Vec3d(1.0, 0.12, -0.5).GetNormalized()
        self.ex._read_navigation_hand.return_value = (
            POINT,
            raw_origin,
            raw_origin - 0.08 * raw_direction,
            raw_origin,
        )
        self.tick(0.12)
        cast_origin, cast_direction = self.ex._navigation_scene.cast_ray.call_args.args
        drawn_origin, drawn_endpoint = self.ex._navigation_scene.draw.call_args.args
        self.assertGreater((cast_origin - raw_origin).GetLength(), 0.001)
        self.assertGreater((cast_direction - raw_direction).GetLength(), 0.001)
        np.testing.assert_allclose(drawn_origin, cast_origin, atol=1e-12)
        np.testing.assert_allclose((drawn_endpoint - drawn_origin).GetNormalized(), cast_direction, atol=1e-12)
        self.assertAlmostEqual(drawn_endpoint[2], 0.0)
        self.assertTrue(self.ex._navigation_scene.draw.call_args.kwargs["hit_valid"])
        np.testing.assert_allclose(self.ex._gesture_command.candidate, drawn_endpoint, atol=1e-12)

    def test_missing_collision_draws_only_ray_and_reports_no_hit(self) -> None:
        self.release()
        self.left.positions = skeleton_positions(self.ex, POINT)
        self.ex._navigation_scene.cast_ray.return_value = None
        self.ex._navigation_scene.last_error = "No collision within the pointing ray range"
        self.tick(0.1)
        origin, direction = self.ex._navigation_scene.cast_ray.call_args.args
        drawn_origin, endpoint = self.ex._navigation_scene.draw.call_args.args
        self.assertFalse(self.ex._navigation_scene.draw.call_args.kwargs["hit_valid"])
        self.assertIsNone(self.ex._navigation_scene.draw.call_args.kwargs["target"])
        np.testing.assert_allclose(drawn_origin, origin)
        np.testing.assert_allclose(endpoint, origin + direction * HUMANOID.PointNavigationScene.MAX_DISTANCE)
        status = self.ex.gesture_locomotion_status()
        self.assertIsNone(status["ray_hit_path"])
        self.assertFalse(status["ray_walkable"])
        self.assertIn("No collision", status["detail"])
        self.assertEqual(self.ex._gesture_command.state, "idle")
        np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])

    def test_corridor_query_cannot_overwrite_the_recorded_ray_error(self) -> None:
        self.activate()

        def ray_miss(*args):
            self.ex._navigation_scene.last_error = "No collision within the pointing ray range"
            return None

        self.ex._navigation_scene.cast_ray.side_effect = ray_miss
        for index, clear in enumerate((True, False)):
            with self.subTest(clear=clear):

                def corridor_query(*args, **kwargs):
                    self.ex._navigation_scene.last_error = None if clear else "Corridor query failed"
                    return clear

                self.ex._navigation_scene.is_path_clear.side_effect = corridor_query
                self.tick(0.8 + index * 0.1)
                status = self.ex.gesture_locomotion_status()
                self.assertEqual(status["ray_error"], "No collision within the pointing ray range")
                self.assertEqual(status["corridor_error"], None if clear else "Corridor query failed")
                self.assertEqual(status["state"], "active" if clear else "blocked")
                if not clear:
                    self.assertIn("Corridor query failed", status["detail"])
                    np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])

    def test_release_tracking_loss_and_reset_clear_filter_ray_and_movement(self) -> None:
        for interruption in ("release", "tracking_loss", "reset"):
            with self.subTest(interruption=interruption):
                self.left.flags = 0xA
                self.activate()
                self.assertIsNotNone(self.ex._navigation_ray_filter._last_time_s)
                self.assertIsNotNone(self.ex._navigation_ray)
                self.ex._base_command[:] = 0.2
                self.ex._navigation_scene.cast_ray.reset_mock()
                self.ex._navigation_scene.draw.reset_mock()
                self.ex._navigation_scene.hide.reset_mock()
                if interruption == "reset":
                    self.ex._reset_gesture_locomotion()
                elif interruption == "release":
                    self.release(0.8)
                else:
                    self.left.flags = 0x2
                    self.tick(0.8)
                self.assertIsNone(self.ex._navigation_ray_filter._last_time_s)
                self.assertIsNone(self.ex._navigation_ray_filter._origin)
                self.assertIsNone(self.ex._navigation_ray_filter._direction)
                self.assertIsNone(self.ex._navigation_ray)
                self.ex._navigation_scene.cast_ray.assert_not_called()
                self.ex._navigation_scene.draw.assert_not_called()
                self.ex._navigation_scene.hide.assert_called_once()
                np.testing.assert_allclose(self.ex._base_command, [0, 0, 0])
                np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])

    def test_duplicate_ray_sample_time_clears_visual_and_disarms_movement(self) -> None:
        self.activate()
        self.ex._navigation_scene.cast_ray.reset_mock()
        self.ex._navigation_scene.draw.reset_mock()
        self.ex._navigation_scene.hide.reset_mock()
        self.ex._base_command[:] = 0.2
        self.tick(self.ex._navigation_ray_filter._last_time_s)
        self.assertIsNone(self.ex._navigation_ray_filter._last_time_s)
        self.assertIsNone(self.ex._navigation_ray)
        self.assertEqual(self.ex._gesture_command.state, "awaiting_release")
        self.ex._navigation_scene.cast_ray.assert_not_called()
        self.ex._navigation_scene.draw.assert_not_called()
        self.ex._navigation_scene.hide.assert_called_once()
        np.testing.assert_allclose(self.ex._base_command, [0, 0, 0])
        np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])

    def test_missing_invalid_stale_and_mismatched_gaze_do_not_gate_selection(self) -> None:
        for condition in ("missing", "invalid", "stale", "mismatch", "head_fallback"):
            with self.subTest(condition=condition):
                self.ex._reset_gesture_locomotion()
                tracker = types.SimpleNamespace(latest=self.gaze, _elapsed_time=10.0, update=Mock())
                self.ex._eye_gaze_tracker = None if condition == "missing" else tracker
                self.gaze.valid = condition != "invalid"
                self.gaze.sample_time = -100.0 if condition == "stale" else 0.0
                self.gaze.source = "hmd_forward" if condition == "head_fallback" else "eye_tracker"
                self.gaze.hit_position = Gf.Vec3d(-3, 2, 1) if condition == "mismatch" else FLOOR
                self.left.positions = skeleton_positions(self.ex, OPEN)
                self.tick(0.0, refresh_gaze=False)
                self.left.positions = skeleton_positions(self.ex, POINT)
                for index in range(1, 8):
                    self.tick(index * 0.1, refresh_gaze=False)
                self.assertEqual(self.ex._gesture_command.state, "active")
                self.assertEqual(self.ex._gesture_command.target, tuple(FLOOR))
                self.assertEqual(self.ex.gesture_locomotion_status()["mode"], "point_only")
                self.assertGreater(float(self.ex._controller_command[0]), 0.05)
                self.assertIs(self.ex._eye_gaze_tracker, None if condition == "missing" else tracker)
                tracker.update.assert_not_called()

    def test_navigation_never_reads_or_updates_the_independent_gaze_tracker(self) -> None:
        class IndependentGaze:
            update = Mock()

            @property
            def latest(self):
                raise AssertionError("Pointing navigation must not read gaze")

        tracker = IndependentGaze()
        self.ex._eye_gaze_tracker = tracker
        self.left.positions = skeleton_positions(self.ex, OPEN)
        self.tick(0.0, refresh_gaze=False)
        self.left.positions = skeleton_positions(self.ex, POINT)
        for index in range(1, 8):
            self.tick(index * 0.1, refresh_gaze=False)
        self.assertEqual(self.ex._gesture_command.state, "active")
        self.assertIs(self.ex._eye_gaze_tracker, tracker)
        tracker.update.assert_not_called()

    def test_slow_physics_point_selection_has_no_gaze_sampling_dependency(self) -> None:
        self.ex._eye_gaze_tracker = None
        for wall_step in (0.05, 0.10):
            with self.subTest(real_time_factor=0.005 / wall_step):
                self.ex._reset_gesture_locomotion()
                self.ex._last_physics_dt = 0.005
                self.ex._teleop_input_dt = min(wall_step, 0.05)
                self.left.positions = skeleton_positions(self.ex, OPEN)
                self.tick(0.0, refresh_gaze=False)
                self.left.positions = skeleton_positions(self.ex, POINT)
                for index in range(1, 17):
                    self.tick(index * wall_step, refresh_gaze=False)
                    if self.ex._gesture_command.state == "active":
                        break
                self.assertEqual(self.ex._gesture_command.state, "active")
                self.assertGreater(float(self.ex._controller_command[0]), 0.05)

    def test_object_hit_and_obstacle_route_cannot_start(self) -> None:
        for invalid in ("object", "obstacle"):
            with self.subTest(invalid=invalid):
                self.ex._reset_gesture_locomotion()
                self.ex._navigation_scene.cast_ray.return_value["walkable"] = invalid != "object"
                self.ex._navigation_scene.is_path_clear.return_value = invalid != "obstacle"
                self.release()
                self.left.positions = skeleton_positions(self.ex, POINT)
                for index in range(1, 10):
                    self.tick(index * 0.1)
                self.assertNotEqual(self.ex._gesture_command.state, "active")
                np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])

    def test_factory_planner_is_wall_clock_bounded_and_route_adoption_needs_fresh_clearance(self) -> None:
        planner = types.SimpleNamespace(
            state="planning", route=(), target=tuple(FLOOR), detail="Finding detour", queries=0, expanded=0
        )

        def advance(max_queries):
            self.assertEqual(max_queries, 24)
            planner.queries += max_queries
            if planner.queries >= 48:
                planner.state = "ready"
                planner.route = ((0, 2, 0), tuple(FLOOR))

        planner.advance = Mock(side_effect=advance)
        self.ex._navigation_scene.is_path_clear.side_effect = lambda first, last, **kwargs: tuple(last) == (0, 2, 0)
        with patch.object(HUMANOID, "FactoryRoutePlanner", return_value=planner) as factory:
            self.release()
            self.left.positions = skeleton_positions(self.ex, POINT)
            for index in range(1, 8):
                self.tick(index * 0.1)
            self.assertEqual(self.ex._gesture_command.state, "planning")
            factory.assert_not_called()
            for now in (0.8, 0.81, 0.85, 0.89):
                self.tick(now)
                np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])
            factory.assert_called_once()
            self.assertIs(factory.call_args.args[3], self.ex._navigation_scene.is_route_segment_clear)
            self.assertIs(factory.call_args.kwargs["corner_clear"], self.ex._navigation_scene.is_route_corner_clear)
            self.assertEqual(planner.advance.call_count, 1)
            self.tick(0.91)
            self.assertEqual(planner.advance.call_count, 2)
            self.assertEqual(self.ex._gesture_command.state, "active")
            np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])
            self.tick(0.92)
            corridor = self.ex._navigation_scene.is_path_clear.call_args
            self.assertEqual(tuple(corridor.args[1]), (0, 2, 0))
            self.assertEqual(corridor.kwargs["stopping_shortfall_m"], 0)
            np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0.4])
            self.assertEqual(self.ex.gesture_locomotion_status()["target"], tuple(FLOOR))

    def test_planning_release_loss_reset_and_grasp_discard_plan_without_motion(self) -> None:
        for interruption in ("release", "loss", "reset", "grasp"):
            with self.subTest(interruption=interruption):
                self.ex._reset_gesture_locomotion()
                self.left.flags = 0xA
                self.ex._grabbed_objects_by_side["left"] = None
                self.ex._navigation_scene.is_path_clear.return_value = False
                self.release()
                self.left.positions = skeleton_positions(self.ex, POINT)
                for index in range(1, 8):
                    self.tick(index * 0.1)
                planner = types.SimpleNamespace(
                    state="planning",
                    route=(),
                    target=tuple(FLOOR),
                    detail="Finding detour",
                    queries=0,
                    expanded=0,
                    advance=Mock(),
                )
                self.ex._navigation_route_planner = planner
                if interruption == "release":
                    self.release(0.8)
                elif interruption == "loss":
                    self.left.flags = 0x2
                    self.tick(0.8)
                elif interruption == "reset":
                    self.ex._reset_gesture_locomotion()
                else:
                    self.ex._grabbed_objects_by_side["left"] = "/World/HeldBox"
                    self.tick(0.8)
                planner.advance.assert_not_called()
                self.assertIsNone(self.ex._navigation_route_planner)
                np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])

    def test_world_target_behind_robot_pivots_only_with_live_turn_clearance(self) -> None:
        self.ex._navigation_scene.cast_ray.return_value["position"] = Gf.Vec3d(-4, 0, 0)
        self.activate()
        np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0.4])
        self.ex._navigation_scene.is_turn_clear.return_value = False
        self.tick(0.8)
        self.assertEqual(self.ex._gesture_command.state, "blocked")
        np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])

    def prepare_object_approach(self):
        root = f"{self.ex._sample_box_root_path}/Box_06"
        self.ex._spawned_sample_box_paths = [root, f"{self.ex._sample_box_root_path}/Box_07"]
        self.ex._navigation_scene.cast_ray.return_value = {
            "position": Gf.Vec3d(2.0, 0, 0.3),
            "distance": 2.0,
            "path": root + "/Asset/Mesh",
            "walkable": False,
        }
        self.ex._navigation_scene.resolve_object_approach.return_value = {
            "target": (1.7, 0.0, 0.0),
            "source_object_root": root,
            "surface_point": (2.0, 0.0, 0.3),
            "floor_path": "/World/Floor",
            "standoff_m": 0.3,
            "detail": "Clear object approach",
        }
        return root

    def test_object_approach_preserves_blue_surface_hit_and_dwell_locks_floor_marker(self) -> None:
        root = self.prepare_object_approach()
        self.activate()
        self.assertEqual(self.ex._gesture_command.target, (1.7, 0, 0))
        self.assertGreater(float(self.ex._controller_command[0]), 0.0)
        _, ray_end = self.ex._navigation_scene.draw.call_args.args
        self.assertEqual(ray_end, Gf.Vec3d(2, 0, 0.3))
        self.assertEqual(self.ex._navigation_scene.draw.call_args.kwargs["target"], (1.7, 0, 0))
        status = self.ex.gesture_locomotion_status()
        self.assertEqual(status["destination_kind"], "object_approach")
        self.assertEqual(status["source_object_root"], root)
        self.assertFalse(status["ray_walkable"])
        self.assertAlmostEqual(status["stopping_shortfall_m"], 0.2)
        resolver = self.ex._navigation_scene.resolve_object_approach
        resolver.reset_mock()
        # Looking/pointing elsewhere during a held walk cannot redirect the goal.
        self.ex._navigation_scene.cast_ray.return_value["path"] = self.ex._spawned_sample_box_paths[1] + "/Mesh"
        self.ex._navigation_scene.cast_ray.return_value["position"] = Gf.Vec3d(4, 1, 0.5)
        self.tick(0.8)
        resolver.assert_not_called()
        self.assertEqual(self.ex._gesture_command.target, (1.7, 0, 0))
        self.assertEqual(self.ex.gesture_locomotion_status()["source_object_root"], root)

    def test_cached_object_candidate_never_reuses_clearance_authorization(self) -> None:
        self.prepare_object_approach()
        self.release()
        self.left.positions = skeleton_positions(self.ex, POINT)
        self.tick(0.1)
        self.tick(0.12)
        resolver = self.ex._navigation_scene.resolve_object_approach
        resolver.assert_called_once()
        self.assertEqual(self.ex._navigation_scene.is_path_clear.call_count, 2)
        self.assertEqual(self.ex._gesture_command.state, "acquiring")
        self.ex._navigation_scene.is_path_clear.return_value = False
        self.ex._navigation_scene.last_error = "A box moved into the route"
        self.ex._navigation_scene.last_blocking_paths = ("/World/NewObstacle",)
        self.tick(0.14)
        resolver.assert_called_once()
        self.assertEqual(self.ex._navigation_scene.is_path_clear.call_count, 3)
        # A blocked straight corridor now permits a stationary planning dwell;
        # cached floor selection still never authorizes movement through it.
        self.assertEqual(self.ex._gesture_command.state, "acquiring")
        self.assertEqual(self.ex.gesture_locomotion_status()["blocking_paths"], ("/World/NewObstacle",))
        np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])

    def test_object_search_cache_expires_on_time_surface_robot_or_object_change(self) -> None:
        for changed in ("time", "surface", "robot", "object"):
            with self.subTest(changed=changed):
                self.prepare_object_approach()
                self.ex._reset_gesture_locomotion()
                self.ex._get_g1_base_pose_for_arms.return_value = (Gf.Vec3d(0, 0, 0.8), 0.0)
                self.release()
                self.left.positions = skeleton_positions(self.ex, POINT)
                self.ex._navigation_scene.resolve_object_approach.reset_mock()
                self.tick(0.1)
                now = 0.12
                if changed == "time":
                    now = 0.151
                elif changed == "surface":
                    self.ex._navigation_scene.cast_ray.return_value["position"] = Gf.Vec3d(2.021, 0, 0.3)
                elif changed == "robot":
                    self.ex._get_g1_base_pose_for_arms.return_value = (Gf.Vec3d(0.021, 0, 0.8), 0.0)
                else:
                    self.ex._navigation_scene.cast_ray.return_value["path"] = self.ex._spawned_sample_box_paths[1]
                self.tick(now)
                self.assertEqual(self.ex._navigation_scene.resolve_object_approach.call_count, 2)

    def test_release_loss_and_reset_discard_object_candidate_cache(self) -> None:
        for interruption in ("release", "loss", "reset"):
            with self.subTest(interruption=interruption):
                self.prepare_object_approach()
                self.ex._reset_gesture_locomotion()
                self.left.flags = 0xA
                self.release()
                self.left.positions = skeleton_positions(self.ex, POINT)
                self.tick(0.1)
                self.assertIsNotNone(self.ex._navigation_object_approach_cache)
                if interruption == "reset":
                    self.ex._reset_gesture_locomotion()
                elif interruption == "release":
                    self.release(0.12)
                else:
                    self.left.flags = 0x2
                    self.tick(0.12)
                self.assertIsNone(self.ex._navigation_object_approach_cache)
                self.assertIsNone(self.ex._navigation_destination_kind)
                self.assertIsNone(self.ex._navigation_source_object_root)
                np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])

    def test_failed_object_search_stays_stopped_and_does_not_retry_every_physics_step(self) -> None:
        self.prepare_object_approach()
        self.ex._navigation_scene.resolve_object_approach.return_value = None
        self.ex._navigation_scene.last_error = "No clear standing spot before the object"
        self.ex._navigation_scene.last_blocking_paths = ("/World/Table",)
        self.release()
        self.left.positions = skeleton_positions(self.ex, POINT)
        for now in (0.1, 0.105, 0.11, 0.115):
            self.tick(now)
            self.assertEqual(self.ex._gesture_command.state, "idle")
            np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])
        self.ex._navigation_scene.resolve_object_approach.assert_called_once()
        self.ex._navigation_scene.is_path_clear.assert_not_called()
        self.assertIn("No clear standing spot", self.ex.gesture_locomotion_status()["detail"])
        self.assertEqual(self.ex.gesture_locomotion_status()["blocking_paths"], ("/World/Table",))

    def test_floor_and_object_corridors_share_arrival_shortfall_with_braking_margin(self) -> None:
        for kind in ("floor", "object"):
            with self.subTest(kind=kind):
                if kind == "object":
                    self.prepare_object_approach()
                self.ex._reset_gesture_locomotion()
                self.release()
                self.left.positions = skeleton_positions(self.ex, POINT)
                self.tick(0.1)
                self.assertAlmostEqual(
                    self.ex._navigation_scene.is_path_clear.call_args.kwargs["stopping_shortfall_m"], 0.2
                )
                if kind == "object":
                    self.assertAlmostEqual(
                        self.ex._navigation_scene.resolve_object_approach.call_args.kwargs["stopping_shortfall_m"], 0.2
                    )

    def test_left_contact_or_grasp_blocks_navigation_but_right_hand_stays_available(self) -> None:
        contact = types.SimpleNamespace(eligible_for_grasp=True)
        for blocked in ("left_contact", "left_held"):
            self.ex._reset_gesture_locomotion()
            self.ex._latest_hand_contacts = {"left": [contact] if blocked == "left_contact" else [], "right": []}
            self.ex._grabbed_objects_by_side["left"] = "/World/Cube" if blocked == "left_held" else None
            self.release()
            self.left.positions = skeleton_positions(self.ex, POINT)
            for index in range(1, 10):
                self.tick(index * 0.1)
            self.assertNotEqual(self.ex._gesture_command.state, "active")
            np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])
        self.ex._latest_hand_contacts = {"left": [], "right": [contact]}
        self.ex._grabbed_objects_by_side["left"] = None
        self.ex._grabbed_objects_by_side["right"] = "/World/Cube"
        self.activate()

    def test_incomplete_fingers_invalid_index_tip_and_controller_source_disarm(self) -> None:
        for invalid in ("index_intermediate", "index_tip", "wrist", "source"):
            with self.subTest(invalid=invalid):
                self.left.invalid.clear()
                self.left.source = "hand"
                self.activate()
                if invalid == "source":
                    self.left.source = "controller"
                else:
                    self.left.invalid.add(invalid)
                self.ex._base_command[:] = 0.2
                self.tick(0.8)
                self.assertEqual(self.ex._gesture_command.state, "awaiting_release")
                np.testing.assert_allclose(self.ex._base_command, [0, 0, 0])
                np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])

    def test_pointing_keeps_both_optical_hands_tracking_individual_fingers(self) -> None:
        self.activate()
        self.assertFalse(self.ex._gesture_command.owns_hand)
        self.ex._update_g1_fingers()
        self.assertEqual(self.ex._finger_curl_source["left"], "hand_tracking")
        self.assertEqual(self.ex._finger_curl_source["right"], "hand_tracking")
        self.assertLess(self.ex._applied_finger_curls["left"]["index"], 0.2)
        self.assertGreater(self.ex._applied_finger_curls["left"]["middle"], 0.8)
        self.assertGreater(self.ex._applied_finger_curls["right"]["index"], 0.7)
        self.release(0.8)
        for _ in range(30):
            self.ex._update_g1_fingers()
        self.assertLess(self.ex._applied_finger_curls["left"]["middle"], 0.2)

    def test_slow_frame_jitter_keeps_fresh_pointing_but_untracked_valid_pose_stops_immediately(self) -> None:
        self.activate()
        self.tick(1.04)
        self.assertEqual(self.ex._gesture_command.state, "active")
        for index in range(1, 21):
            self.tick(1.04 + index * 0.0001)
        self.tick(1.442)
        self.assertEqual(self.ex._gesture_command.state, "active")
        self.left.flags = 0x2
        self.tick(1.542)
        self.assertEqual(self.ex._gesture_command.state, "awaiting_release")
        np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])
        np.testing.assert_allclose(self.ex._base_command, [0, 0, 0])

    def test_open_hand_bootstrap_handles_first_slow_render_before_new_pointing_selection(self) -> None:
        self.release()
        self.tick(0.001)
        self.tick(0.002)
        self.tick(0.302)
        self.assertEqual(self.ex._gesture_command.state, "awaiting_release")
        np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])
        self.tick(0.303)
        self.assertEqual(self.ex._gesture_command.state, "idle")
        self.left.positions = skeleton_positions(self.ex, POINT)
        for now in (0.304, 0.604, 0.605, 0.606):
            self.tick(now)
            self.assertEqual(self.ex._gesture_command.state, "acquiring")
        self.tick(0.906)
        self.assertEqual(self.ex._gesture_command.state, "active")

    def test_decoupled_actor_keeps_both_arm_ik_paths_live_while_pointing(self) -> None:
        self.activate()
        ex = self.ex
        ex.g1.supports_arm_teleoperation = True
        positions = np.array([[0.4, -0.5, 0.1, 0.2]])
        ex.g1.robot.get_dof_positions.side_effect = lambda: Array(positions)
        ex._configure_g1_arm_dofs = Mock()
        ex._g1_arm_dofs_configured = True
        ex._g1_arm_dof_indices_by_side = {"left": {"elbow": 0, "wrist_yaw": 1}, "right": {"elbow": 2, "wrist_yaw": 3}}
        ex._g1_arm_joint_limits = {index: (-2, 2) for index in range(4)}
        ex._g1_arm_joint_defaults = {index: 0 for index in range(4)}
        ex._last_physics_dt = ex._teleop_input_dt = 0.01
        ex._contact_health = "ok"
        ex._get_optical_arm_pose = Mock(return_value=pose())
        ex._compute_arm_target_body_position = Mock(return_value=Gf.Vec3d(0.4, 0.1, 0.2))
        ex._compute_arm_target_orientation = Mock(return_value=None)
        ex._compute_arm_targets_from_body_position = Mock(
            side_effect=lambda side, *args, **kwargs: {0: 0.0, 1: 0.0} if side == "left" else {2: 0.0, 3: 0.0}
        )
        for method in (
            "_log_hand_tracking_status_once",
            "_set_grab_candidate",
            "_set_arm_rig_target_visible",
            "_release_grabbed_object",
            "_update_arm_rig_target",
            "_update_grabbed_object",
        ):
            setattr(ex, method, Mock())
        ex._update_g1_arms_from_hand_tracking()
        self.assertEqual([call.args[0] for call in ex._get_optical_arm_pose.call_args_list], ["left", "right"])
        self.assertEqual(
            [call.args[0] for call in ex._compute_arm_targets_from_body_position.call_args_list], ["left", "right"]
        )
        args, kwargs = ex.g1.robot.set_dof_position_targets.call_args
        targets = dict(zip(kwargs["dof_indices"], args[0]))
        self.assertLess(targets[0], positions[0, 0])
        self.assertLess(targets[2], positions[0, 2])

    def test_destination_request_clears_actor_threshold_using_bounded_wall_time_filter(self) -> None:
        self.activate()
        requested = self.ex._controller_command.clone()
        self.ex._teleop_input_dt = 0.05
        for _ in range(5):
            self.ex._smooth_base_command(requested, 0.005)
        self.assertGreater(float(self.ex._base_command[0]), 0.05)
        self.assertLess(float(self.ex._base_command[0]), float(requested[0]))
        status = self.ex.gesture_locomotion_status()
        self.assertEqual(status["state"], "active")
        self.assertEqual(status["requested_velocity"], requested.tolist())
        self.assertIn("Walking", status["detail"])
        self.release(0.8)
        np.testing.assert_allclose(self.ex._base_command, [0, 0, 0])

    def test_release_zeroes_actual_policy_command_before_smoothing(self) -> None:
        self.activate()
        self.physics_tick(0.8)
        self.assertGreater(float(self.ex.g1.forward.call_args.args[1][0]), 0.0)
        self.left.positions = skeleton_positions(self.ex, OPEN)
        self.physics_tick(0.9)
        np.testing.assert_allclose(self.ex.g1.forward.call_args.args[1], [0, 0, 0])
        self.ex._eye_gaze_tracker.update.assert_called_once_with(0.01)

    def test_recenter_requires_open_hand_then_new_pointing_dwell(self) -> None:
        self.activate()
        self.ex._base_command[:] = 0.2
        self.ex._controller_command[:] = 0.2
        self.ex._reset_operator_head_rotation = Mock()
        self.ex._update_head_camera_view = Mock()
        self.ex._log_xr_camera_state = Mock()
        self.ex._invalidate_webcam_calibration = Mock()
        self.ex._request_xr_recenter()
        self.assertEqual(self.ex._gesture_command.state, "awaiting_release")
        np.testing.assert_allclose(self.ex._base_command, [0, 0, 0])
        np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])
        self.tick(0.8)
        self.assertEqual(self.ex._gesture_command.state, "awaiting_release")
        self.release(0.9)
        self.left.positions = skeleton_positions(self.ex, POINT)
        self.tick(1.0)
        self.assertEqual(self.ex._gesture_command.state, "acquiring")

    def test_stop_invalidates_native_views_but_pause_only_disarms_navigation(self) -> None:
        events = types.SimpleNamespace(STOP=2, PAUSE=3, PLAY=1)
        for event_type in (events.STOP, events.PAUSE):
            with self.subTest(event_type=event_type):
                self.activate()
                self.ex._physics_ready = True
                self.ex._timeline_stopped = False
                self.ex.g1.invalidate_physics_state.reset_mock()
                self.ex._contact_reader.invalidate.reset_mock()
                self.ex._head_camera_mount_prim = object()
                self.ex._walking_support_pose_batch = object()
                self.ex._walking_support_samples = [(0.0, 0.0, 0.0, 0.0)]
                for cache in (self.ex._hand_link_prims, self.ex._grasp_link_prims, self.ex._object_rigid_prims):
                    cache["stale"] = object()
                with patch.object(
                    HUMANOID, "import_module", return_value=types.SimpleNamespace(TimelineEventType=events)
                ):
                    self.ex._on_navigation_timeline_event(types.SimpleNamespace(type=event_type))
                self.assertEqual(self.ex._gesture_command.state, "awaiting_release")
                self.assertIsNone(self.ex._walking_support_pose_batch)
                self.assertIsNone(self.ex._walking_support_samples)
                np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])
                if event_type == events.STOP:
                    self.assertFalse(self.ex._physics_ready)
                    self.assertTrue(self.ex._timeline_stopped)
                    self.ex.g1.invalidate_physics_state.assert_called_once()
                    self.ex._contact_reader.invalidate.assert_called_once()
                    self.assertIsNone(self.ex._head_camera_mount_prim)
                    self.assertFalse(self.ex._hand_link_prims)
                    self.assertFalse(self.ex._grasp_link_prims)
                    self.assertFalse(self.ex._object_rigid_prims)
                else:
                    self.assertTrue(self.ex._physics_ready)
                    self.assertFalse(self.ex._timeline_stopped)
                    self.ex.g1.invalidate_physics_state.assert_not_called()
                    self.ex._contact_reader.invalidate.assert_not_called()

    def test_policy_physics_uses_200_hz_but_recording_stays_100_hz_in_both_modes(self) -> None:
        collectors = (
            "_collect_behavioral_sample",
            "_collect_hand_tracking_sample",
            "_collect_gaze_sample",
            "_collect_object_state_sample",
        )
        for walking in (True, False):
            with self.subTest(walking=walking), patch.dict(
                HUMANOID.os.environ, {"G1_GESTURE_WALKING": "1" if walking else "0"}
            ):
                example = HUMANOID.HumanoidExample()
                expected_dt = 0.005 if walking else 0.01
                self.assertEqual(example._world_settings["physics_dt"], expected_dt)
                example._capture_eye_camera_frame = Mock()
                example._flush_behavioral_csvs = Mock()
                for name in collectors:
                    setattr(example, name, Mock())
                for _ in range(round(1.0 / expected_dt)):
                    example._collect_all_behavioral_data()
                for name in collectors:
                    self.assertEqual(getattr(example, name).call_count, 100)

    def test_legacy_command_smoothing_keeps_simulation_clock(self) -> None:
        self.ex._gesture_locomotion_enabled = False
        self.ex._teleop_input_dt = 0.05
        requested = self.ex._controller_command.clone()
        requested[0] = 0.1
        self.ex._smooth_base_command(requested, 0.005)
        self.assertAlmostEqual(float(self.ex._base_command[0]), 0.1 * (1 - math.exp(-0.005 / 0.3)), places=7)

    def test_no_xr_still_refreshes_physical_contacts_and_opens_fingers_in_both_modes(self) -> None:
        contacts = {"left": [types.SimpleNamespace(eligible_for_grasp=False)], "right": []}
        for walking in (True, False):
            with self.subTest(walking=walking):
                ex = self.ex
                ex._gesture_locomotion_enabled = walking
                ex._g1_locomotion = "policy" if walking else "stationary"
                ex._xr_core = None
                ex._webcam_enabled = False
                ex._contact_health = "physics_not_ready"
                ex._contact_reader = types.SimpleNamespace(read=Mock(return_value=(contacts, "ok")))
                ex._finger_curl_source["left"] = "hand_tracking"
                ex._latest_finger_curls["left"] = dict(POINT)
                ex.g1.set_finger_curls.reset_mock()
                ex._update_g1_fingers()
                ex._contact_reader.read.assert_called_once_with(ex._last_physics_dt)
                self.assertEqual(ex._contact_health, "ok")
                self.assertEqual(ex._latest_hand_contacts, contacts)
                self.assertEqual(ex._finger_curl_source, {})
                self.assertEqual(ex._latest_finger_curls, {})
                self.assertEqual(ex.g1.set_finger_curls.call_count, 2)
                for command in ex.g1.set_finger_curls.call_args_list:
                    self.assertEqual(command.args[1], dict.fromkeys(ex._finger_roles, 0.0))

    def test_no_xr_contact_refresh_prevents_inactive_arm_targets_following_gravity_droop(self) -> None:
        ex = self.ex
        ex._gesture_locomotion_enabled = False
        ex._g1_locomotion = "stationary"
        ex._xr_core = None
        ex._webcam_enabled = False
        ex._contact_health = "physics_not_ready"
        ex._contact_reader = types.SimpleNamespace(read=Mock(return_value=({"left": [], "right": []}, "ok")))
        initial = np.array([[0.4, -0.5, 0.1, 0.2]])
        measured = initial.copy()
        ex.g1.robot.get_dof_positions.side_effect = lambda: Array(measured)
        ex._configure_g1_arm_dofs = Mock()
        ex._g1_arm_dofs_configured = True
        ex._g1_arm_dof_indices_by_side = {"left": {"elbow": 0, "wrist_yaw": 1}, "right": {"elbow": 2, "wrist_yaw": 3}}
        ex._g1_arm_joint_limits = {index: (-2, 2) for index in range(4)}
        ex._g1_arm_joint_defaults = {index: 0 for index in range(4)}
        for method in ("_set_grab_candidate", "_set_arm_rig_target_visible", "_release_grabbed_object"):
            setattr(ex, method, Mock())
        for step in range(6):
            measured[:] = initial + step * np.array([[0.002, -0.002, 0.002, -0.002]])
            ex._update_g1_fingers()
            ex._update_g1_arms_from_hand_tracking()
            args, kwargs = ex.g1.robot.set_dof_position_targets.call_args
            targets = dict(zip(kwargs["dof_indices"], args[0]))
            np.testing.assert_allclose([targets[index] for index in range(4)], initial[0], atol=1e-7)
        self.assertEqual(ex._contact_reader.read.call_count, 6)
        self.assertIsNone(ex._articulation_health_fault)

    def test_keyboard_and_controller_axes_cannot_bypass_gesture_ownership(self) -> None:
        self.left.positions = skeleton_positions(self.ex)
        self.ex._keyboard_command[:] = 1.0
        self.ex._controller_command[:] = 1.0
        self.ex._base_command[:] = 1.0
        self.ex._sub_keyboard_event(types.SimpleNamespace(input="UP", type=CARB.input.KeyboardEventType.KEY_PRESS))
        self.physics_tick(0.0)
        np.testing.assert_allclose(self.ex.g1.forward.call_args.args[1], [0, 0, 0])
        self.ex._read_gamepad_controller_axes.assert_not_called()
        self.assertFalse(self.ex._pressed_keys)

    def test_unexpected_gesture_input_failure_pauses_before_policy_dispatch(self) -> None:
        self.ex._get_hand_tracking_finger_curls = Mock(side_effect=RuntimeError("broken skeleton reader"))
        self.physics_tick(0.0)
        self.ex.g1.forward.assert_not_called()
        self.ex._pause_faulted_articulation.assert_called()
        self.assertIn("gesture", self.ex._articulation_health_fault)
        self.assertEqual(self.ex._gesture_command.state, "awaiting_release")

    def test_policy_runtime_error_is_latched_and_stops_other_control_work(self) -> None:
        self.ex.g1.forward.side_effect = RuntimeError("bad policy output")
        self.physics_tick(0.0)
        self.assertIn("locomotion controller failed", self.ex._articulation_health_fault)
        self.ex._pause_faulted_articulation.assert_called_once()
        self.ex._update_g1_fingers.assert_not_called()
        self.ex._update_g1_arms_from_hand_tracking.assert_not_called()
        self.ex._eye_gaze_tracker.update.assert_not_called()
        self.ex.g1.forward.reset_mock()
        self.physics_tick(0.1)
        self.ex.g1.forward.assert_not_called()

    def test_balance_watchdog_accepts_normal_pose_yaw_and_normalizable_quaternion(self) -> None:
        self.activate()
        quaternions = (
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],  # Turning around is not a fall.
            [math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)],
            [math.cos(math.radians(15)), math.sin(math.radians(15)), 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        )
        for quaternion in quaternions:
            with self.subTest(quaternion=quaternion):
                self.ex.g1.robot.get_world_poses.return_value = (Array([[4.0, -3.0, 0.75]]), Array([quaternion]))
                self.assertTrue(self.ex._check_gesture_balance())
                self.assertEqual(self.ex._gesture_command.state, "active")
                self.assertIsNone(self.ex._articulation_health_fault)
        self.ex._pause_faulted_articulation.assert_not_called()

    def test_balance_watchdog_detects_low_pelvis_and_roll_or_pitch_falls(self) -> None:
        half_tilt = math.radians(46.0) / 2.0
        cases = (
            ([0.0, 0.0, 0.44], [1.0, 0.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.8], [math.cos(half_tilt), math.sin(half_tilt), 0.0, 0.0]),
            ([0.0, 0.0, 0.8], [math.cos(half_tilt), 0.0, -math.sin(half_tilt), 0.0]),
            ([0.0, 0.0, 0.8], [0.0, 1.0, 0.0, 0.0]),
        )
        for position, quaternion in cases:
            with self.subTest(position=position, quaternion=quaternion):
                self.ex._reset_gesture_locomotion()
                self.ex._articulation_health_fault = None
                self.activate()
                for command in (self.ex._base_command, self.ex._controller_command, self.ex._keyboard_command):
                    command[:] = 0.2
                self.ex._pause_faulted_articulation.reset_mock()
                self.ex.g1.robot.get_world_poses.return_value = (Array([position]), Array([quaternion]))
                self.assertFalse(self.ex._check_gesture_balance())
                self.assertTrue(self.ex._articulation_health_fault)
                self.assertEqual(self.ex._gesture_command.state, "awaiting_release")
                for command in (self.ex._base_command, self.ex._controller_command, self.ex._keyboard_command):
                    np.testing.assert_allclose(command, [0, 0, 0])
                self.ex._pause_faulted_articulation.assert_called_once()

    def test_balance_watchdog_rejects_nonfinite_or_degenerate_root_observations(self) -> None:
        observations = (
            (Array([[math.nan, 0.0, 0.8]]), Array([[1.0, 0.0, 0.0, 0.0]])),
            (Array([[0.0, math.inf, 0.8]]), Array([[1.0, 0.0, 0.0, 0.0]])),
            (Array([[0.0, 0.0, 0.8]]), Array([[math.nan, 0.0, 0.0, 0.0]])),
            (Array([[0.0, 0.0, 0.8]]), Array([[0.0, 0.0, 0.0, 0.0]])),
            (Array([[0.0, 0.8]]), Array([[1.0, 0.0, 0.0, 0.0]])),
            (None, None),
        )
        for observation in observations:
            with self.subTest(observation=observation):
                self.ex._articulation_health_fault = None
                self.ex._pause_faulted_articulation.reset_mock()
                self.ex.g1.robot.get_world_poses.return_value = observation
                self.assertFalse(self.ex._check_gesture_balance())
                self.assertTrue(self.ex._articulation_health_fault)
                self.ex._pause_faulted_articulation.assert_called_once()

    def test_balance_watchdog_fails_closed_on_read_error_but_stationary_never_reads(self) -> None:
        self.ex.g1.robot.get_world_poses.side_effect = RuntimeError("stale root tensor")
        self.assertFalse(self.ex._check_gesture_balance())
        self.assertTrue(self.ex._articulation_health_fault)
        self.ex.g1.robot.get_world_poses.reset_mock()
        self.ex._gesture_locomotion_enabled = False
        self.assertTrue(self.ex._check_gesture_balance())
        self.ex.g1.robot.get_world_poses.assert_not_called()

    def test_physics_fall_guard_pauses_before_gesture_policy_or_hand_updates(self) -> None:
        self.ex.g1.robot.get_world_poses.return_value = (Array([[0.0, 0.0, 0.3]]), Array([[1.0, 0.0, 0.0, 0.0]]))
        self.physics_tick(0.0, check_balance=True)
        self.assertTrue(self.ex._articulation_health_fault)
        self.ex._pause_faulted_articulation.assert_called_once()
        self.ex._read_xr_controller_axes.assert_not_called()
        self.ex.g1.forward.assert_not_called()
        self.ex._update_g1_fingers.assert_not_called()
        self.ex._update_g1_arms_from_hand_tracking.assert_not_called()

    def use_native_foot_support(self):
        """Supply existing world-body rows; execute the production reader/history."""
        del self.ex._update_walking_support_motion
        self.foot_rows = np.array([[0, 0.13, 0.035, 0, 0, 0, 1], [0, -0.13, 0.035, 0, 0, 0, 1]], dtype=float)
        self.foot_view = types.SimpleNamespace(
            prim_paths=["/World/G1/left_ankle_roll_link", "/World/G1/right_ankle_roll_link"],
            get_transforms=Mock(side_effect=lambda: Array(self.foot_rows.copy())),
        )
        self.simulation = types.SimpleNamespace(is_valid=True, create_rigid_body_view=Mock(return_value=self.foot_view))
        self.simulation_manager = types.SimpleNamespace(get_physics_simulation_view=Mock(return_value=self.simulation))
        dependency = patch.object(HUMANOID, "SimulationManager", self.simulation_manager, create=True)
        dependency.start()
        self.addCleanup(dependency.stop)

    def prime_stopped_support(self):
        self.ex.g1.robot.get_velocities.return_value = (Array([[0, 0, 0]]), Array([[0, 0, 0]]))
        for _ in range(300):
            self.assertTrue(self.ex._check_gesture_motion(0.005, [0, 0, 0]))

    def test_motion_watchdog_accepts_reaching_pelvis_with_planted_feet(self) -> None:
        self.use_native_foot_support()
        self.prime_stopped_support()
        self.ex.g1.robot.get_velocities.return_value = (Array([[0.15, 0, 0]]), Array([[0, 0, 0.35]]))
        for _ in range(400):
            self.assertTrue(self.ex._check_gesture_motion(0.005, [0, 0, 0]))
        self.ex._pause_faulted_articulation.assert_not_called()
        self.simulation.create_rigid_body_view.assert_called_once()
        self.assertLessEqual(len(self.ex._walking_support_samples), 82)

    def test_motion_watchdog_rejects_sustained_support_drift_after_stance_excursion(self) -> None:
        self.use_native_foot_support()
        initial = self.foot_rows.copy()
        for mode in ("both_slide", "repeated_steps", "sole_turn"):
            with self.subTest(mode=mode):
                self.ex._reset_gesture_locomotion()
                self.ex._articulation_health_fault = None
                self.ex._pause_faulted_articulation.reset_mock()
                self.foot_rows[:] = initial
                self.prime_stopped_support()
                linear = [0.15, 0, 0] if mode != "sole_turn" else [0, 0, 0]
                angular = [0, 0, 0.30] if mode == "sole_turn" else [0, 0, 0]
                self.ex.g1.robot.get_velocities.return_value = (Array([linear]), Array([angular]))
                expected_tick = {"both_slide": 273, "repeated_steps": 241, "sole_turn": 234}[mode]
                for index in range(1, expected_tick + 1):
                    elapsed = index * 0.005
                    if mode == "both_slide":
                        self.foot_rows[:, 0] = 0.11 * elapsed
                    elif mode == "repeated_steps":
                        # Four 5 cm advances with stationary gaps. The anchor
                        # must not follow each newly settled stance.
                        self.foot_rows[:, 0] = round(0.05 * int(elapsed / 0.4) + 0.25 * min(elapsed % 0.4, 0.2), 9)
                    else:
                        angle = 0.30 * elapsed
                        self.foot_rows[:, :2] = [
                            [-0.13 * math.sin(angle), 0.13 * math.cos(angle)],
                            [0.13 * math.sin(angle), -0.13 * math.cos(angle)],
                        ]
                        self.foot_rows[:, 3:] = [0, 0, math.sin(angle / 2), math.cos(angle / 2)]
                    self.assertEqual(self.ex._check_gesture_motion(0.005, [0, 0, 0]), index < expected_tick)
                self.ex._pause_faulted_articulation.assert_called_once()
                self.assertIn("foot support", self.ex._articulation_health_fault)

    def test_motion_watchdog_allows_bounded_single_foot_balance_step(self) -> None:
        self.use_native_foot_support()
        self.prime_stopped_support()
        anchor = self.ex._walking_support_anchor
        self.ex.g1.robot.get_velocities.return_value = (Array([[0.15, 0, 0]]), Array([[0, 0, 0.35]]))
        for index in range(1, 81):
            self.foot_rows[0, 0] = 0.25 * index * 0.005  # 10 cm foot / 5 cm centroid correction.
            self.assertTrue(self.ex._check_gesture_motion(0.005, [0, 0, 0]))
        self.assertEqual(self.ex._walking_support_anchor, anchor)
        self.assertAlmostEqual(self.ex._read_walking_support_pose()[2], 0.0)  # Stagger is not sole turning.
        self.assertGreaterEqual(self.ex._walking_residual_motion_time, 0.4 - 1e-9)
        self.ex._pause_faulted_articulation.assert_not_called()

    def test_motion_watchdog_both_recorded_balance_incidents_remain_running(self) -> None:
        self.use_native_foot_support()
        for filename in ("planted_foot_manipulation.json", "corrective_stance_step.json"):
            with self.subTest(filename=filename):
                self.ex._reset_gesture_locomotion()
                samples = json.loads((Path(__file__).parent / "fixtures" / filename).read_text())["samples"]
                for side_index, side in enumerate(("left", "right")):
                    self.foot_rows[side_index] = [*samples[0][side], *samples[0][side + "_xyzw"]]
                self.prime_stopped_support()
                anchor = self.ex._walking_support_anchor
                for previous, current in zip(samples, samples[1:]):
                    dt = current["time"] - previous["time"]
                    velocity = [(b - a) / dt for a, b in zip(previous["root"], current["root"])]
                    angle = current["root_yaw"] - previous["root_yaw"]
                    yaw = math.atan2(math.sin(angle), math.cos(angle)) / dt
                    self.ex.g1.robot.get_velocities.return_value = (Array([velocity]), Array([[0, 0, yaw]]))
                    for side_index, side in enumerate(("left", "right")):
                        self.foot_rows[side_index] = [*current[side], *current[side + "_xyzw"]]
                    self.assertTrue(self.ex._check_gesture_motion(dt, [0, 0, 0]))
                self.assertEqual(self.ex._walking_support_anchor, anchor)
        self.ex._pause_faulted_articulation.assert_not_called()

    def test_support_window_interpolates_actual_time_unwraps_heading_and_resets_backwards_clock(self) -> None:
        self.use_native_foot_support()
        for elapsed in (0, 0.07, 0.22, 0.50, 0.575, 0.725, 0.93):
            heading = math.pi - 0.01 + 0.30 * elapsed
            direction = np.array([math.cos(heading), math.sin(heading)]) * 0.13
            center = np.array([0.12 * elapsed, -0.16 * elapsed])
            self.foot_rows[:, :2] = [center + direction, center - direction]
            self.foot_rows[:, 3:] = [0, 0, math.sin(heading / 2), math.cos(heading / 2)]
            motion = self.ex._update_walking_support_motion(elapsed)
            if elapsed >= 0.4:
                np.testing.assert_allclose(motion[:2], [0.2, 0.30], atol=1e-12)
                self.assertAlmostEqual(self.ex._walking_support_samples[0][0], elapsed - 0.4)
            self.assertLessEqual(len(self.ex._walking_support_samples), 5)
        self.assertIsNone(self.ex._update_walking_support_motion(0.1))
        self.assertEqual(len(self.ex._walking_support_samples), 1)

    def test_support_reader_rebuilds_on_simulation_generation_and_rejects_invalid_existing_bodies(self) -> None:
        self.use_native_foot_support()
        self.ex._update_walking_support_motion(0.0)
        self.ex._update_walking_support_motion(0.5)
        self.simulation.create_rigid_body_view.assert_called_once()
        next_simulation = types.SimpleNamespace(is_valid=True, create_rigid_body_view=Mock(return_value=self.foot_view))
        self.simulation_manager.get_physics_simulation_view.return_value = next_simulation
        self.assertIsNone(self.ex._update_walking_support_motion(0.6))
        next_simulation.create_rigid_body_view.assert_called_once()
        self.assertEqual(len(self.ex._walking_support_samples), 1)
        self.foot_rows[0, 0] = math.nan
        self.ex.g1.robot.get_velocities.return_value = (Array([[0, 0, 0]]), Array([[0, 0, 0]]))
        self.assertFalse(self.ex._check_gesture_motion(0.005, [0, 0, 0]))
        self.assertIsNone(self.ex._walking_support_pose_batch)
        self.assertIsNone(self.ex._walking_support_samples)
        self.assertIn("Invalid walking foot pose", self.ex._articulation_health_fault)

    def test_support_anchor_survives_same_episode_view_recreation_across_yaw_seam(self) -> None:
        self.use_native_foot_support()
        start = math.pi - 0.01
        self.foot_rows[:, 3:] = [0, 0, math.sin(start / 2), math.cos(start / 2)]
        self.prime_stopped_support()
        anchor = self.ex._walking_support_anchor
        angle = -math.pi + 0.02
        self.foot_rows[:, 3:] = [0, 0, math.sin(angle / 2), math.cos(angle / 2)]
        self.ex._walking_support_pose_batch = None
        self.ex._walking_support_samples = None
        self.assertIsNone(self.ex._update_walking_support_motion(2.0))
        motion = self.ex._update_walking_support_motion(2.4)
        self.assertEqual(self.ex._walking_support_anchor, anchor)
        self.assertAlmostEqual(motion[3], 0.03, places=10)
        self.ex._walking_support_pose_batch = None
        self.ex._walking_support_samples = None
        self.ex._update_walking_support_motion(2.5)
        self.assertEqual(self.ex._walking_support_anchor, anchor)
        self.assertTrue(self.ex._check_gesture_motion(0.005, [0.2, 0, 0]))
        self.assertIsNone(self.ex._walking_support_anchor)

    def test_sole_heading_is_circular_mean_and_rejects_undefined_forward_projection(self) -> None:
        self.use_native_foot_support()
        for index, angle in enumerate((math.pi - 0.01, -math.pi + 0.01)):
            self.foot_rows[index, 3:] = [0, 0, 2 * math.sin(angle / 2), 2 * math.cos(angle / 2)]
        self.assertAlmostEqual(abs(self.ex._read_walking_support_pose()[2]), math.pi)
        self.foot_rows[0, 3:] = [0, math.sqrt(0.5), 0, math.sqrt(0.5)]
        with self.assertRaisesRegex(RuntimeError, "no horizontal forward"):
            self.ex._read_walking_support_pose()

    def test_motion_watchdog_requires_matching_root_and_support_motion_channels(self) -> None:
        self.use_native_foot_support()
        initial = self.foot_rows.copy()
        for root_turn in (False, True):
            with self.subTest(root_turn=root_turn):
                self.ex._reset_gesture_locomotion()
                self.foot_rows[:] = initial
                self.prime_stopped_support()
                linear = [0, 0, 0] if root_turn else [0.15, 0, 0]
                yaw = 0.35 if root_turn else 0.0
                self.ex.g1.robot.get_velocities.return_value = (Array([linear]), Array([[0, 0, yaw]]))
                for index in range(1, 401):
                    elapsed = index * 0.005
                    if root_turn:
                        self.foot_rows[:, 0] = 0.2 * elapsed
                    else:
                        angle = 0.4 * elapsed
                        self.foot_rows[:, 3:] = [0, 0, math.sin(angle / 2), math.cos(angle / 2)]
                    self.assertTrue(self.ex._check_gesture_motion(0.005, [0, 0, 0]))
        self.ex._pause_faulted_articulation.assert_not_called()

    def test_root_runaway_fault_does_not_depend_on_valid_feet(self) -> None:
        self.use_native_foot_support()
        self.ex._walking_runaway_time = 0.20
        self.foot_view.get_transforms.side_effect = AssertionError("Runaway cannot wait for foot evidence")
        self.ex.g1.robot.get_velocities.return_value = (Array([[0.9, 0, 0]]), Array([[0, 0, 0]]))
        self.assertFalse(self.ex._check_gesture_motion(0.05, [0, 0, 0]))
        self.foot_view.get_transforms.assert_not_called()
        self.assertIn("speed 0.900", self.ex._articulation_health_fault)

    def test_missing_or_stale_feet_fault_instead_of_using_root_as_support(self) -> None:
        self.use_native_foot_support()
        self.ex.g1.robot.get_velocities.return_value = (Array([[0, 0, 0]]), Array([[0, 0, 0]]))
        for invalid in ("missing_body", "stale_tensor", "missing_simulation"):
            with self.subTest(invalid=invalid):
                self.ex._reset_gesture_locomotion()
                self.ex._articulation_health_fault = None
                self.ex._pause_faulted_articulation.reset_mock()
                self.foot_view.prim_paths = ["/World/G1/left_ankle_roll_link", "/World/G1/right_ankle_roll_link"]
                self.foot_view.get_transforms.side_effect = lambda: Array(self.foot_rows.copy())
                self.simulation.is_valid = True
                if invalid == "missing_body":
                    self.foot_view.prim_paths.pop()
                elif invalid == "stale_tensor":
                    self.foot_view.get_transforms.side_effect = RuntimeError("stale tensor")
                else:
                    self.simulation.is_valid = False
                self.assertFalse(self.ex._check_gesture_motion(0.005, [0, 0, 0]))
                self.ex._pause_faulted_articulation.assert_called_once()
                self.assertIsNone(self.ex._walking_support_pose_batch)
                self.assertIsNone(self.ex._walking_support_samples)
                self.ex.g1.robot.get_world_poses.assert_not_called()

    def test_motion_watchdog_allows_startup_settling_then_rejects_sustained_uncommanded_travel(self) -> None:
        for linear, angular in (
            ([0.11, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([-0.08, 0.08, 0.0], [0.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, -0.26]),
        ):
            with self.subTest(linear=linear, angular=angular):
                self.ex._reset_gesture_locomotion()
                self.ex._articulation_health_fault = None
                self.ex._pause_faulted_articulation.reset_mock()
                self.activate()
                self.ex.g1.robot.get_velocities.return_value = (Array([linear]), Array([angular]))
                for _ in range(379):
                    self.assertTrue(self.ex._check_gesture_motion(0.005, [0, 0, 0]))
                self.ex._pause_faulted_articulation.assert_not_called()
                # One window after braking: no additional 0.4 s dwell is added.
                self.assertFalse(self.ex._check_gesture_motion(0.005, [0, 0, 0]))
                self.ex._pause_faulted_articulation.assert_called_once()
                self.assertTrue(self.ex._articulation_health_fault)
                self.assertEqual(self.ex._gesture_command.state, "awaiting_release")
                np.testing.assert_allclose(self.ex._base_command, [0, 0, 0])
                np.testing.assert_allclose(self.ex._controller_command, [0, 0, 0])

    def test_motion_watchdog_does_not_accumulate_separated_drift_and_new_commands_clear_history(self) -> None:
        robot = self.ex.g1.robot
        robot.get_velocities.return_value = (Array([[0.0, 0.0, 0.0]]), Array([[0.0, 0.0, 0.0]]))
        for _ in range(20):
            self.assertTrue(self.ex._check_gesture_motion(0.1, [0, 0, 0]))
        for _ in range(5):
            robot.get_velocities.return_value = (Array([[0.11, 0.0, 0.0]]), Array([[0.0, 0.0, 0.0]]))
            for _ in range(3):
                self.assertTrue(self.ex._check_gesture_motion(0.1, [0, 0, 0]))
            # Three centimeters of disturbance followed by a stationary window
            # must not accumulate into a later unintended-walking decision.
            robot.get_velocities.return_value = (Array([[0.0, 0.0, 0.0]]), Array([[0.0, 0.0, 0.0]]))
            for _ in range(4):
                self.assertTrue(self.ex._check_gesture_motion(0.1, [0, 0, 0]))
        robot.get_velocities.return_value = (Array([[0.11, 0.0, 0.0]]), Array([[0.0, 0.0, 0.0]]))
        for _ in range(3):
            self.assertTrue(self.ex._check_gesture_motion(0.1, [0, 0, 0]))
        self.assertTrue(self.ex._check_gesture_motion(0.1, [0.25, 0, 0.2]))
        self.assertEqual(self.ex._walking_stop_elapsed, 0.0)
        self.assertEqual(self.ex._walking_residual_motion_time, 0.0)
        self.assertEqual(self.ex._walking_residual_window_time, 0.0)
        self.assertIsNone(self.ex._walking_residual_motion_samples)
        self.assertIsNone(self.ex._walking_support_samples)
        for _ in range(14):
            self.assertTrue(self.ex._check_gesture_motion(0.1, [0, 0, 0]))
        self.ex._pause_faulted_articulation.assert_not_called()

    def test_motion_watchdog_brief_excursion_does_not_fault_from_window_mean_alone(self) -> None:
        self.ex._walking_stop_elapsed = 1.5
        robot = self.ex.g1.robot
        # The mean reaches 0.1125 m/s, but speed exceeded the threshold for
        # only 0.3 s. Keep the original continuous-speed gate as well.
        for velocity, ticks in ((0.15, 30), (0.0, 10)):
            robot.get_velocities.return_value = (Array([[velocity, 0.0, 0.0]]), Array([[0.0, 0.0, 0.0]]))
            for _ in range(ticks):
                self.assertTrue(self.ex._check_gesture_motion(0.01, [0, 0, 0]))
        self.assertGreater(
            sum(sample[0] * sample[1] for sample in self.ex._walking_residual_motion_samples) / 0.4,
            0.10,
        )
        self.ex._pause_faulted_articulation.assert_not_called()

    def test_motion_watchdog_allows_reversing_xy_and_yaw_sway_without_cancelling_runaway_limits(self) -> None:
        self.ex._walking_stop_elapsed = 1.5
        for index in range(1000):
            # Constant speed magnitudes exceed the residual limits throughout,
            # but signed components reverse and each 0.4 s window has zero drift.
            phase = index % 4
            linear = [0.2 * math.cos(phase * math.pi / 2), 0.2 * math.sin(phase * math.pi / 2), 0.0]
            yaw = 0.6 if index % 2 else -0.6
            self.ex.g1.robot.get_velocities.return_value = (Array([linear]), Array([[0.0, 0.0, yaw]]))
            self.assertTrue(self.ex._check_gesture_motion(0.01, [0, 0, 0]))
        self.assertIsNone(self.ex._articulation_health_fault)
        self.ex._pause_faulted_articulation.assert_not_called()

    def test_motion_watchdog_still_rejects_low_frequency_motion_with_real_net_travel(self) -> None:
        self.ex._walking_stop_elapsed = 1.5
        for index in range(40):
            # A slow reversal in the future cannot excuse the current 5.6 cm
            # displacement. This is real travel within the detection window.
            velocity = 0.15 * math.cos(2 * math.pi * ((index + 0.5) * 0.01) / 4.0)
            self.ex.g1.robot.get_velocities.return_value = (Array([[velocity, 0.0, 0.0]]), Array([[0.0, 0.0, 0.0]]))
            permitted = self.ex._check_gesture_motion(0.01, [0, 0, 0])
            self.assertEqual(permitted, index < 39)
        self.ex._pause_faulted_articulation.assert_called_once()

    def test_motion_watchdog_window_weights_actual_duration_and_trims_partial_old_samples(self) -> None:
        self.ex._walking_stop_elapsed = 1.5
        # The time integral is zero. An unweighted mean of the three velocity
        # samples is 0.167 m/s and would incorrectly stop the robot.
        for _ in range(100):
            for dt, velocity in ((0.05, 0.6), (0.15, 0.2), (0.20, -0.3)):
                self.ex.g1.robot.get_velocities.return_value = (
                    Array([[velocity, 0.0, 0.0]]),
                    Array([[0.0, 0.0, 0.0]]),
                )
                self.assertTrue(self.ex._check_gesture_motion(dt, [0, 0, 0]))
                self.assertLessEqual(self.ex._walking_residual_window_time, 0.4 + 1e-12)
                self.assertLessEqual(len(self.ex._walking_residual_motion_samples), 4)
        self.ex.g1.robot.get_velocities.return_value = (Array([[0.2, 0.0, 0.0]]), Array([[0.0, 0.0, 0.0]]))
        self.assertTrue(self.ex._check_gesture_motion(0.075, [0, 0, 0]))
        self.assertAlmostEqual(self.ex._walking_residual_motion_samples[0][0], 0.125)
        self.assertAlmostEqual(sum(sample[0] for sample in self.ex._walking_residual_motion_samples), 0.4)
        self.ex._pause_faulted_articulation.assert_not_called()

    def test_motion_watchdog_invalid_timebase_discards_prior_movement_evidence(self) -> None:
        for dt in (0.0, -0.005, math.nan, math.inf):
            with self.subTest(dt=dt):
                self.ex._reset_gesture_locomotion()
                self.ex._walking_stop_elapsed = 1.5
                self.ex.g1.robot.get_velocities.return_value = (
                    Array([[0.11, 0.0, 0.0]]),
                    Array([[0.0, 0.0, 0.0]]),
                )
                for _ in range(3):
                    self.assertTrue(self.ex._check_gesture_motion(0.1, [0, 0, 0]))
                self.assertTrue(self.ex._check_gesture_motion(dt, [0, 0, 0]))
                self.assertEqual(self.ex._walking_stop_elapsed, 0.0)
                self.assertEqual(self.ex._walking_residual_motion_time, 0.0)
                self.assertEqual(self.ex._walking_residual_window_time, 0.0)
                self.assertIsNone(self.ex._walking_residual_motion_samples)
                self.assertIsNone(self.ex._walking_support_pose_batch)
                self.assertIsNone(self.ex._walking_support_samples)
                self.assertTrue(self.ex._check_gesture_motion(0.1, [0, 0, 0]))
        self.ex._pause_faulted_articulation.assert_not_called()

    def test_motion_watchdog_detects_sustained_runaway_with_or_without_walk_commands(self) -> None:
        for command in ([0, 0, 0], [0.25, 0, 0.2]):
            for linear, angular in (([0.6, -0.6, 0.0], [0.0, 0.0, 0.0]), ([0.0, 0.0, 0.0], [0.0, 0.0, -1.3])):
                with self.subTest(command=command, linear=linear, angular=angular):
                    self.ex._reset_gesture_locomotion()
                    self.ex._articulation_health_fault = None
                    self.ex._pause_faulted_articulation.reset_mock()
                    self.ex.g1.robot.get_velocities.return_value = (Array([linear]), Array([angular]))
                    for _ in range(4):
                        self.assertTrue(self.ex._check_gesture_motion(0.05, command))
                    self.assertFalse(self.ex._check_gesture_motion(0.05, command))
                    self.assertTrue(self.ex._articulation_health_fault)
                    self.ex._pause_faulted_articulation.assert_called_once()

    def test_motion_watchdog_runaway_timer_resets_on_recovery_and_gesture_reset(self) -> None:
        fast = (Array([[0.9, 0.0, 0.0]]), Array([[0.0, 0.0, 0.0]]))
        self.ex.g1.robot.get_velocities.return_value = fast
        for _ in range(2):
            self.assertTrue(self.ex._check_gesture_motion(0.1, [0.25, 0, 0]))
        self.assertAlmostEqual(self.ex._walking_runaway_time, 0.2)
        self.ex.g1.robot.get_velocities.return_value = (Array([[0.25, 0.0, 0.0]]), Array([[0.0, 0.0, 0.0]]))
        self.assertTrue(self.ex._check_gesture_motion(0.1, [0.25, 0, 0]))
        self.assertEqual(self.ex._walking_runaway_time, 0.0)
        self.ex.g1.robot.get_velocities.return_value = fast
        for _ in range(2):
            self.assertTrue(self.ex._check_gesture_motion(0.1, [0, 0, 0]))
        self.ex._walking_residual_motion_time = 0.2
        self.ex._reset_gesture_locomotion()
        self.assertEqual(self.ex._walking_stop_elapsed, 0.0)
        self.assertEqual(self.ex._walking_residual_motion_time, 0.0)
        self.assertEqual(self.ex._walking_residual_window_time, 0.0)
        self.assertIsNone(self.ex._walking_residual_motion_samples)
        self.assertEqual(self.ex._walking_runaway_time, 0.0)
        self.ex._pause_faulted_articulation.assert_not_called()

    def test_motion_watchdog_rejects_invalid_velocity_or_command_observations_immediately(self) -> None:
        zero = Array([[0.0, 0.0, 0.0]])
        cases = (
            ((Array([[math.nan, 0.0, 0.0]]), zero), [0, 0, 0]),
            ((zero, Array([[0.0, 0.0, math.inf]])), [0, 0, 0]),
            ((Array([[0.0, 0.0]]), zero), [0, 0, 0]),
            ((zero, Array([[0.0, 0.0]])), [0, 0, 0]),
            ((None, None), [0, 0, 0]),
            ((zero, zero), [0, 0]),
            ((zero, zero), [0, math.nan, 0]),
        )
        for observation, command in cases:
            with self.subTest(observation=observation, command=command):
                self.ex._reset_gesture_locomotion()
                self.ex._articulation_health_fault = None
                self.ex._pause_faulted_articulation.reset_mock()
                self.ex.g1.robot.get_velocities.return_value = observation
                self.assertFalse(self.ex._check_gesture_motion(0.005, command))
                self.assertTrue(self.ex._articulation_health_fault)
                self.ex._pause_faulted_articulation.assert_called_once()

    def test_motion_watchdog_read_error_faults_but_stationary_never_reads_root_velocity(self) -> None:
        self.ex.g1.robot.get_velocities.side_effect = RuntimeError("stale velocity tensor")
        self.assertFalse(self.ex._check_gesture_motion(0.005, [0, 0, 0]))
        self.assertTrue(self.ex._articulation_health_fault)
        self.ex.g1.robot.get_velocities.reset_mock()
        self.ex._gesture_locomotion_enabled = False
        self.assertTrue(self.ex._check_gesture_motion(0.005, [0, 0, 0]))
        self.ex.g1.robot.get_velocities.assert_not_called()

    def test_physics_motion_guard_pauses_before_policy_or_hand_updates(self) -> None:
        self.ex.g1.robot.get_velocities.return_value = (Array([[math.nan, 0.0, 0.0]]), Array([[0.0, 0.0, 0.0]]))
        self.physics_tick(0.0, check_motion=True)
        self.assertTrue(self.ex._articulation_health_fault)
        self.ex._pause_faulted_articulation.assert_called_once()
        self.ex.g1.forward.assert_not_called()
        self.ex._update_g1_fingers.assert_not_called()
        self.ex._update_g1_arms_from_hand_tracking.assert_not_called()
        self.ex._eye_gaze_tracker.update.assert_not_called()

    def test_body_torque_refresh_consumes_current_tick_arm_and_finger_targets(self) -> None:
        self.ex.g1.uses_explicit_body_efforts = True

        def refresh():
            self.ex.g1.forward.assert_called_once()
            self.ex._update_g1_fingers.assert_called_once()
            self.ex._update_g1_arms_from_hand_tracking.assert_called_once()
            self.ex._update_head_camera_view.assert_not_called()

        self.ex.g1.apply_body_efforts = Mock(side_effect=refresh)
        self.physics_tick(0.0)
        self.ex.g1.apply_body_efforts.assert_called_once()
        self.assertIsNone(self.ex._articulation_health_fault)

    def test_body_torque_failure_pauses_before_camera_and_recording(self) -> None:
        self.ex.g1.uses_explicit_body_efforts = True
        self.ex.g1.apply_body_efforts = Mock(side_effect=ValueError("invalid measured motor velocity"))
        self.physics_tick(0.0)
        self.assertIn("body motor controller failed", self.ex._articulation_health_fault)
        self.ex._pause_faulted_articulation.assert_called_once()
        self.ex._update_head_camera_view.assert_not_called()
        self.ex._eye_gaze_tracker.update.assert_not_called()
        self.ex._collect_all_behavioral_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
