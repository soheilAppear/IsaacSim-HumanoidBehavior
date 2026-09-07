# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regress gaze provenance, loss/reacquisition, and collider filtering."""

from __future__ import annotations

import math
import types
import unittest
from unittest.mock import Mock

from humanoid_test_support import GAZE, HUMANOID, PHYSICS, USD, Device, pose
from pxr import Gf, Usd


class TestGazeControls(unittest.TestCase):
    def setUp(self):
        self.devices = {}
        self.xr = types.SimpleNamespace(
            get_input_device=lambda handle: self.devices.get(handle),
            get_all_input_devices=lambda: [],
        )
        self.tracker = GAZE.EyeGazeTracker(self.xr, draw_ray=False, highlight_gazed_boxes=False)
        self.tracker._update_ray_visual = Mock()
        self.tracker._update_highlight = Mock()
        self.tracker._report_gaze_target_change = Mock()
        self.tracker._log_source = Mock()
        self.tracker._log_missing_devices = Mock()
        self.tracker._raycast = Mock(return_value=(None, None, None))

    def test_head_fallback_is_not_labeled_as_eye_tracking(self):
        self.devices["/user/head"] = Device(pose((3, 4, 1.5)))
        self.tracker.update(0.01)
        self.assertEqual(self.tracker.latest.source, "hmd_forward")
        example = HUMANOID.HumanoidExample()
        example._eye_gaze_tracker = self.tracker
        example._collect_gaze_sample()
        self.assertEqual(example._gaze_records[-1]["gaze_source"], "hmd_forward")

    def test_eye_acquisition_and_loss_replace_source_without_stale_rows(self):
        self.devices["/user/head"] = Device(pose((3, 4, 1.5)))
        self.tracker.update(0.01)
        self.devices["/user/eye/unified"] = Device(pose((3, 4, 1.5), yaw=20))
        self.tracker.update(0.01)
        self.assertEqual(self.tracker.latest.source, "eye_tracker")
        self.devices.clear()
        self.tracker.update(0.01)
        self.assertFalse(self.tracker.latest.valid)
        self.assertIsNone(self.tracker.gaze_source)
        example = HUMANOID.HumanoidExample()
        example._eye_gaze_tracker = self.tracker
        example._last_headset_raw_position = Gf.Vec3d(3, 4, 1.5)
        example._last_headset_pose_matrix = pose((3, 4, 1.5))
        example._collect_gaze_sample()
        self.assertEqual(example._gaze_records[-1]["gaze_valid"], 0)

    def test_missing_eye_warning_also_runs_when_head_fallback_works(self):
        self.devices["/user/head"] = Device(pose((0, 0, 1.5)))
        for _ in range(320):
            self.tracker.update(0.01)
        self.tracker._log_missing_devices.assert_called_once()

    def test_physical_room_pose_is_never_used_as_world_gaze(self):
        self.devices["/user/eye/unified"] = types.SimpleNamespace(
            get_pose_names=lambda: ["gaze_ext"],
            get_virtual_world_pose=lambda name: None,
            get_pose=lambda name: pose((0, 1.5, 0)),
            get_raw_pose=lambda name: pose((0, 1.5, 0)),
        )
        self.assertIsNone(self.tracker._read_gaze_ray())

    def test_raycast_rate_is_50_hz_at_non_divisor_physics_rates(self):
        self.devices["/user/eye/unified"] = Device(pose((0, 0, 1.5)))
        for hz in (60, 90, 100, 120, 200):
            with self.subTest(hz=hz):
                tracker = GAZE.EyeGazeTracker(self.xr, draw_ray=False, highlight_gazed_boxes=False)
                tracker._log_source = Mock()
                tracker._update_highlight = Mock()
                tracker._report_gaze_target_change = Mock()
                tracker._raycast = Mock(return_value=(None, None, None))
                for _ in range(hz * 2):
                    tracker.update(1 / hz)
                self.assertEqual(tracker._raycast.call_count, 100)

    def test_unsorted_hits_skip_all_robot_links_and_keep_nearby_package(self):
        tracker = GAZE.EyeGazeTracker(self.xr, draw_ray=False)

        def raycast(origin, direction, distance, on_hit, both_sides):
            self.assertEqual(origin, (0.0, 0.0, 1.5))
            for path, hit_distance in (
                ("/World/G1_SampleBoxes/Box_2", 0.3),
                ("/World/G1/head", 0.01),
                ("/World/G1/shoulder", 0.05),
                ("/World/G1_SampleBoxes/Box_1", 0.1),
            ):
                self.assertTrue(
                    on_hit(
                        types.SimpleNamespace(rigid_body=path, distance=hit_distance, position=(hit_distance, 0, 1.5))
                    )
                )

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(raycast_all=raycast)
        position, distance, path = tracker._raycast(Gf.Vec3d(0, 0, 1.5), Gf.Vec3d(1, 0, 0))
        self.assertEqual(path, "/World/G1_SampleBoxes/Box_1")
        self.assertAlmostEqual(distance, 0.1)
        self.assertAlmostEqual(position[0], 0.1)

    def test_static_collision_path_and_nonfinite_rays(self):
        self.assertEqual(self.tracker._decode_hit_path({"rigid_body": 0, "collision": "/World/Floor"}), "/World/Floor")
        self.assertIsNone(self.tracker._matrix_to_ray(pose((math.nan, 0, 0))))

    def test_cleanup_resets_source_and_sampling_clock(self):
        stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: stage)
        self.devices["/user/eye/unified"] = Device(pose((0, 0, 1.5)))
        self.tracker.update(0.01)
        self.tracker.cleanup()
        self.assertFalse(self.tracker.latest.valid)
        self.assertIsNone(self.tracker.gaze_source)
        self.assertEqual(self.tracker._elapsed_time, 0)
