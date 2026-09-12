# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regress wall-time filtering and the legacy assisted return after tracking loss."""

from __future__ import annotations

import math
import types
import unittest
from unittest.mock import Mock

import numpy as np
from humanoid_test_support import HUMANOID, USD, Array, pose
from pxr import Gf, Usd
from test_arm_orientation_controls import PalmDevice


class TestArmResponseControls(unittest.TestCase):
    def setUp(self) -> None:
        self.example = HUMANOID.HumanoidExample()
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)

    def test_input_filters_have_same_wall_time_response_in_slow_simulation(self) -> None:
        responses = []
        for update_hz in (25, 50, 100):
            ex = HUMANOID.HumanoidExample()
            ex._arm_rig_smoothing = 0.02
            ex._finger_smoothing = 0.02
            ex._smoothed_arm_rig_targets["left"] = Gf.Vec3d(0)
            ex._smoothed_finger_curls["left"] = {"index": 0.0}
            ex._update_teleop_input_clock(0.01, now=0.0)
            for step in range(1, update_hz + 1):
                ex._update_teleop_input_clock(0.01, now=step / update_hz)
                arm = ex._smooth_arm_rig_target("left", Gf.Vec3d(1, 0, 0))
                fingers = ex._smooth_finger_curls("left", {"index": 1.0})
            responses.append([arm[0], fingers["index"]])
        np.testing.assert_allclose(responses, np.tile(responses[-1], (3, 1)), atol=1e-12)

    def test_pause_and_bad_clock_do_not_create_catch_up_command(self) -> None:
        ex = self.example
        for now in (0.0, 10.0, 9.0, math.nan, math.inf, 11.0):
            ex._update_teleop_input_clock(0.01, now=now)
            self.assertEqual(ex._teleop_input_dt, 0.01)
        ex._update_teleop_input_clock(0.01, now=11.2)
        self.assertEqual(ex._teleop_input_dt, 0.05)

    def test_wall_time_filter_does_not_raise_physics_joint_speed_limit(self) -> None:
        ex = self.example
        ex.g1 = types.SimpleNamespace(robot=types.SimpleNamespace(get_dof_positions=lambda: Array([[0.0]])))
        ex._g1_arm_joint_limits = {0: (-1.0, 1.0)}
        ex._last_physics_dt = 0.01
        ex._update_teleop_input_clock(0.01, now=0.0)
        ex._update_teleop_input_clock(0.01, now=0.04)
        previous = 0.0
        for _ in range(20):
            target = ex._smooth_and_clamp_arm_targets({0: 1.0})[0]
            self.assertLessEqual(target - previous, ex._arm_max_joint_speed * 0.01 + 1e-12)
            self.assertLessEqual(target, ex._arm_max_tracking_error)
            previous = target

    def test_assisted_tracking_loss_keeps_return_to_rest_command_progress(self) -> None:
        ex = self.example
        ex._grasp_mode = "assisted"
        ex.g1 = types.SimpleNamespace(robot=types.SimpleNamespace(get_dof_positions=lambda: Array([[0.0]])))
        ex._g1_arm_dof_indices_by_side = {"right": {"elbow": 0}}
        ex._g1_arm_joint_limits = {0: (-1.0, 1.0)}
        ex._arm_input_sources["right"] = "hand_tracking"
        ex._smoothed_arm_targets[0] = -0.15
        ex._deactivate_hand("right")
        self.assertNotIn(0, ex._smoothed_arm_targets)
        first = ex._smooth_and_clamp_arm_targets({0: 0.5})[0]
        for _ in range(20):
            ex._deactivate_hand("right")
            current = ex._smooth_and_clamp_arm_targets({0: 0.5})[0]
        self.assertGreater(current, first * 2)
        self.assertLessEqual(current, ex._arm_max_tracking_error)

    def test_fixed_palm_offset_rotates_with_wrist_without_reading_fingertips(self) -> None:
        ex = self.example
        ex._get_robot_palm_local_center = Mock(return_value=Gf.Vec3d(0.06, 0.0, 0.0))
        quaternion = pose(yaw=90).ExtractRotationQuat()
        ex._get_hand_link_prim = Mock(
            return_value=types.SimpleNamespace(
                get_world_poses=lambda: (
                    Array([[3, 4, 5]]),
                    Array([[quaternion.GetReal(), *quaternion.GetImaginary()]]),
                )
            )
        )
        ex._get_grasp_link_prims = Mock(side_effect=AssertionError("Moving fingers do not define the palm"))
        np.testing.assert_allclose(ex._palm_offset_world("right", [3, 4, 5]), [0, 0.06, 0], atol=1e-7)

    def test_position_valid_skeleton_acquires_arm_without_runtime_pose_orientation(self) -> None:
        ex = self.example
        for side in ("left", "right"):
            device = PalmDevice(side)
            target = ex._get_optical_arm_pose(side, device)
            self.assertIsNotNone(target)
            np.testing.assert_allclose(target.ExtractTranslation(), [0.05, 0, 0], atol=1e-7)
            device.invalid.add("index_proximal")
            self.assertIsNotNone(ex._get_optical_arm_pose(side, device))
            self.assertIsNone(ex._get_optical_palm_frame(side, device))
            for missing in ("wrist", "middle_proximal"):
                device.invalid = {missing}
                self.assertIsNone(ex._get_optical_arm_pose(side, device))


if __name__ == "__main__":
    unittest.main()
