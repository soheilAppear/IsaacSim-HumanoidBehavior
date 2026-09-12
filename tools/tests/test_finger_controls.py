# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retarget individual optical fingers and thumb opposition to the G1 hand drives."""

from __future__ import annotations

import math
import types
import unittest
from unittest.mock import Mock

import numpy as np
from humanoid_test_support import G1, HUMANOID, Device, pose
from pxr import Gf


def skeleton_positions(example, curls=None, opposition=0.0, total_bends=None):
    """Construct a metric hand skeleton with independent, known joint bend angles."""
    curls = curls or {}
    total_bends = total_bends or {}
    positions = {"wrist": Gf.Vec3d(0, 0, 0), "palm": Gf.Vec3d(0, 0.025, 0)}
    lateral = {"index": 0.025, "middle": 0.0, "ring": -0.022, "little": -0.04}
    for role, chain in example._finger_curl_joint_chains.items():
        if role == "thumb":
            angle = math.radians(45.0 + 90.0 * opposition)
            direction = Gf.Vec3d(math.cos(angle), math.sin(angle), 0)
            current = Gf.Vec3d(0.035, 0.015, 0)
        else:
            direction = Gf.Vec3d(0, 1, 0)
            current = Gf.Vec3d(lateral[role], 0.02, 0)
        bend = total_bends.get(role, curls.get(role, 0.0) * example._finger_curl_full_flexion_rad[role])
        positions[chain[0]] = current
        for index, name in enumerate(chain[1:]):
            angle = bend * index / (len(chain) - 2)
            bone = direction * math.cos(angle) + Gf.Vec3d(0, 0, -1) * math.sin(angle)
            current = current + 0.035 * bone
            positions[name] = current
    return positions


class SkeletonDevice:
    def __init__(self, positions, transform=None, flags=2):
        self.positions = positions
        self.transform = transform if transform is not None else Gf.Matrix4d(1)
        self.flags = flags
        self.invalid = set()

    def get_hand_tracking_data_source(self):
        return "hand"

    def get_pose_names(self):
        return list(self.positions)

    def get_virtual_world_pose_desc(self, name=""):
        position = self.transform.Transform(self.positions[name])
        return types.SimpleNamespace(
            pose_matrix=pose(position), validity_flags=0 if name in self.invalid else self.flags
        )


class TestFingerControls(unittest.TestCase):
    def setUp(self):
        self.example = HUMANOID.HumanoidExample()

    def test_open_hand_is_acquired_without_grip_pinch_or_fist(self):
        ex = self.example
        device = SkeletonDevice(skeleton_positions(ex), flags=3)
        ex._get_xr_gesture_value = Mock(return_value=0.0)
        self.assertEqual(ex._get_hand_input_kind(device), "hand_tracking")
        self.assertIsNotNone(ex._get_hand_tracking_pose(device))
        curls = ex._get_hand_tracking_finger_curls(device)
        self.assertEqual(set(curls), {*ex._finger_roles, "thumb_yaw"})
        self.assertTrue(all(abs(value) < 1e-6 for value in curls.values()))
        ex._latest_finger_curls["right"] = curls
        ex._finger_curl_source["right"] = "hand_tracking"
        self.assertFalse(ex._is_hand_closed("right", device))
        ex._get_xr_gesture_value.assert_not_called()

    def test_individual_finger_updates_without_a_closed_hand_or_active_arm(self):
        ex = self.example
        device = SkeletonDevice(skeleton_positions(ex), flags=2)
        ex._xr_core = object()
        ex._get_xr_input_device = lambda handle: device
        ex._get_xr_gesture_value = Mock(return_value=0.0)
        ex.g1 = types.SimpleNamespace(
            has_finger_control=lambda: True,
            set_finger_curls=Mock(),
            get_finger_curls=lambda side: {role: 0.0 for role in (*ex._finger_roles, "thumb_yaw")},
            get_finger_joint_range=lambda side, role: (0.0, 1.0),
        )
        ex._contact_reader = types.SimpleNamespace(read=lambda dt: ({"left": [], "right": []}, "ok"))
        ex._update_g1_fingers()
        device.positions = skeleton_positions(ex, {"index": 0.7})
        for _ in range(15):
            ex._update_g1_fingers()
        for side in ("left", "right"):
            self.assertEqual(ex._finger_curl_source[side], "hand_tracking")
            self.assertGreater(ex._latest_finger_curls[side]["index"], 0.65)
            self.assertLess(ex._get_hand_closure(side), ex._finger_grab_threshold)
            self.assertNotIn(side, ex._arm_input_sources)
        ex.g1.set_finger_curls.assert_called()
        ex._get_xr_gesture_value.assert_not_called()

    def test_each_finger_moves_independently(self):
        ex = self.example
        for moving in ex._finger_roles:
            with self.subTest(finger=moving):
                device = SkeletonDevice(skeleton_positions(ex, {moving: 0.8}))
                curls = ex._get_hand_tracking_finger_curls(device)
                for role in ex._finger_roles:
                    self.assertAlmostEqual(curls[role], 0.8 if role == moving else 0.0)
                self.assertAlmostEqual(curls["thumb_yaw"], 0.0)

    def test_tightly_folded_finger_does_not_reopen_past_180_degrees(self):
        ex = self.example
        for degrees in (150, 180, 210, 240, 270):
            with self.subTest(degrees=degrees):
                device = SkeletonDevice(skeleton_positions(ex, total_bends={"index": math.radians(degrees)}))
                self.assertAlmostEqual(ex._get_hand_tracking_finger_curls(device)["index"], 1.0)

    def test_curls_and_opposition_ignore_hand_size_position_and_wrist_rotation(self):
        ex = self.example
        positions = skeleton_positions(ex, {"thumb": 0.65, "index": 0.3, "ring": 0.8}, opposition=0.7)
        baseline = ex._get_hand_tracking_finger_curls(SkeletonDevice(positions))
        for scale in (0.5, 1.0, 1.6):
            transform = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3), 143))
            transform.SetTranslateOnly(Gf.Vec3d(15, -6, 3))
            device = SkeletonDevice({name: point * scale for name, point in positions.items()}, transform)
            result = ex._get_hand_tracking_finger_curls(device)
            for role in result:
                self.assertAlmostEqual(result[role], baseline[role], places=6)

    def test_thumb_opposition_is_independent_of_thumb_flexion_and_hand_side(self):
        ex = self.example
        for mirrored in (False, True):
            for opposition in (0.0, 0.4, 1.0):
                for flexion in (0.0, 0.5, 1.0):
                    positions = skeleton_positions(ex, {"thumb": flexion}, opposition)
                    if mirrored:
                        positions = {name: Gf.Vec3d(-point[0], point[1], point[2]) for name, point in positions.items()}
                    result = ex._get_hand_tracking_finger_curls(SkeletonDevice(positions))
                    self.assertAlmostEqual(result["thumb"], flexion)
                    self.assertAlmostEqual(result["thumb_yaw"], opposition)

    def test_position_valid_joints_work_without_orientation(self):
        ex = self.example
        positions = skeleton_positions(ex, {"index": 0.7})
        self.assertAlmostEqual(ex._get_hand_tracking_finger_curls(SkeletonDevice(positions, flags=2))["index"], 0.7)
        self.assertIsNone(ex._get_hand_tracking_finger_curls(SkeletonDevice(positions, flags=1)))

    def test_missing_joint_relaxes_only_that_finger(self):
        ex = self.example
        device = SkeletonDevice(skeleton_positions(ex, {role: 0.8 for role in ex._finger_roles}))
        full = ex._get_hand_tracking_finger_curls(device)
        ex._smooth_finger_curls("right", full)
        device.invalid.add("index_intermediate")
        partial = ex._get_hand_tracking_finger_curls(device)
        self.assertNotIn("index", partial)
        self.assertAlmostEqual(partial["ring"], 0.8)
        for _ in range(50):
            result = ex._smooth_finger_curls("right", partial)
        self.assertLess(result["index"], 0.001)
        self.assertAlmostEqual(result["ring"], 0.8)
        self.assertIsNone(ex._get_controller_finger_curls(device))

    def test_missing_opposition_landmark_does_not_copy_thumb_flexion(self):
        ex = self.example
        device = SkeletonDevice(skeleton_positions(ex, {"thumb": 1.0}, opposition=1.0))
        device.invalid.add("wrist")
        result = ex._get_hand_tracking_finger_curls(device)
        self.assertAlmostEqual(result["thumb"], 1.0)
        self.assertEqual(result["thumb_yaw"], 0.0)

    def test_nonfinite_or_degenerate_joint_cannot_drive_fingers(self):
        ex = self.example
        for bad in (Gf.Vec3d(math.nan, 0, 0), Gf.Vec3d(math.inf, 0, 0)):
            device = SkeletonDevice(skeleton_positions(ex, {"index": 0.8}))
            device.positions["index_intermediate"] = bad
            self.assertNotIn("index", ex._get_hand_tracking_finger_curls(device))
        device = SkeletonDevice(skeleton_positions(ex))
        device.positions["index_intermediate"] = device.positions["index_proximal"]
        self.assertNotIn("index", ex._get_hand_tracking_finger_curls(device))

    def test_controller_fallback_closes_thumb_opposition_with_trigger(self):
        ex = self.example
        curls = ex._smooth_finger_curls("left", {role: 0.7 for role in ex._finger_roles})
        self.assertAlmostEqual(curls["thumb_yaw"], 0.7)

    def test_touch_controller_with_hand_label_still_uses_grip_clutch_and_trigger(self):
        ex = self.example
        device = Device(pose((0.25, -0.2, 1)), source="hand")
        device.get_pose_names = lambda: ["", "grip", "aim", "palm", "pinch", "poke"]
        device.get_input_names = lambda: ["thumbstick", "trigger", "squeeze", "a", "b"]
        ex._get_xr_gesture_value = lambda device, name, gesture: 0.75 if name in ("trigger", "squeeze") else 0.0
        self.assertEqual(ex._get_hand_input_kind(device), "controller")
        self.assertIsNone(ex._get_hand_tracking_pose(device))
        self.assertEqual(ex._get_controller_arm_pose("right", device), device.matrix)
        self.assertTrue(all(value == 0.75 for value in ex._get_controller_finger_curls(device).values()))
        ex._get_xr_gesture_value = lambda *args: 0.0
        self.assertIsNone(ex._get_controller_arm_pose("right", device))

    def test_hand_label_without_valid_grip_or_controller_actions_cannot_fall_back(self):
        ex = self.example
        device = Device(pose(), flags=0, source="hand")
        device.get_pose_names = lambda: ["", "grip", "aim", "palm", "pinch", "poke"]
        device.get_input_names = lambda: ["thumbstick", "trigger", "squeeze"]
        ex._get_xr_gesture_value = lambda *args: 1.0
        self.assertEqual(ex._get_hand_input_kind(device), "none")
        self.assertIsNone(ex._get_controller_arm_pose("right", device))
        self.assertIsNone(ex._get_controller_finger_curls(device))
        device.flags = 3
        device.get_input_names = lambda: ["pinch", "poke", "trigger"]
        self.assertEqual(ex._get_hand_input_kind(device), "none")
        self.assertIsNone(ex._get_controller_finger_curls(device))

    def test_partial_optical_skeleton_cannot_use_stale_controller_actions(self):
        ex = self.example
        device = Device(pose(), source="hand")
        device.get_pose_names = lambda: ["grip", "palm", "thumb_tip"]
        device.get_input_names = lambda: ["thumbstick", "trigger", "squeeze"]
        ex._get_xr_gesture_value = lambda *args: 1.0
        self.assertEqual(ex._get_hand_input_kind(device), "hand_tracking")
        self.assertIsNone(ex._get_controller_arm_pose("right", device))
        self.assertIsNone(ex._get_controller_finger_curls(device))

    def test_controller_tracking_loss_rejects_stale_trigger(self):
        ex = self.example
        device = Device(pose(), flags=0, source="controller")
        ex._get_xr_gesture_value = lambda *args: 1.0
        self.assertEqual(ex._get_hand_input_kind(device), "none")
        self.assertIsNone(ex._get_controller_finger_curls(device))

    def test_opposition_does_not_count_as_a_closed_hand(self):
        ex = self.example
        ex._latest_finger_curls["right"] = {"thumb_yaw": 1.0}
        ex._finger_curl_source["right"] = "hand_tracking"
        self.assertFalse(ex._is_hand_closed("right", None))

    def test_optical_finger_targets_reach_corresponding_robot_actuators(self):
        ex = self.example
        for side in ("left", "right"):
            robot = object.__new__(G1.G1TeleopRobot)
            indices = {role: index for index, role in enumerate((*ex._finger_roles, "thumb_yaw"))}
            robot._finger_dof_indices = {side: indices}
            robot._finger_open_closed = {side: {role: (0.1, 1.3) for role in indices}}
            robot.robot = types.SimpleNamespace(set_dof_position_targets=Mock())
            device = SkeletonDevice(skeleton_positions(ex, {"middle": 0.75}, opposition=0.5))
            curls = ex._get_hand_tracking_finger_curls(device)
            robot.set_finger_curls(side, curls)
            values = robot.robot.set_dof_position_targets.call_args.args[0]
            expected = [0.1 + 1.2 * curls[role] for role in indices]
            np.testing.assert_allclose(values, expected)
            self.assertGreater(values[indices["middle"]], values[indices["index"]])
            self.assertNotEqual(values[indices["thumb_yaw"]], values[indices["thumb"]])


if __name__ == "__main__":
    unittest.main()
