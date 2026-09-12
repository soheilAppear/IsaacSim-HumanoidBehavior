# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Losing arm input must not sweep a physical hand through the work table."""

from __future__ import annotations

import types
import unittest
from unittest.mock import Mock, call

import numpy as np
from humanoid_test_support import HUMANOID, Array, pose
from pxr import Gf


class TestInactivePhysicalArms(unittest.TestCase):
    def setUp(self):
        self.ex = HUMANOID.HumanoidExample()
        self.positions = np.array([[0.4, -0.5, 0.1, 0.2]])
        self.robot = Mock()
        self.robot.get_dof_positions.side_effect = lambda: Array(self.positions)
        self.ex.g1 = types.SimpleNamespace(robot=self.robot, set_finger_curls=Mock())
        self.ex._xr_core = object()
        self.ex._contact_health = "ok"
        self.ex._g1_arm_dofs_configured = True
        self.ex._configure_g1_arm_dofs = Mock()
        self.ex._g1_arm_dof_indices_by_side = {
            "left": {"elbow": 0, "wrist_yaw": 1},
            "right": {"elbow": 2, "wrist_yaw": 3},
        }
        self.ex._g1_arm_joint_limits = {index: (-2.0, 2.0) for index in range(4)}
        self.ex._g1_arm_joint_defaults = {index: 0.0 for index in range(4)}
        self.ex._arm_smoothing = 1.0
        self.ex._last_physics_dt = self.ex._teleop_input_dt = 0.01
        self.ex._get_g1_base_pose_for_arms = Mock(return_value=(Gf.Vec3d(0, 0, 0.8), 0.0))
        self.ex._get_xr_input_device = Mock(return_value=object())
        self.active_optical = set()
        self.active_controller = set()
        self.ex._get_optical_arm_pose = lambda side, device: pose() if side in self.active_optical else None
        self.ex._get_controller_arm_pose = lambda side, device: pose() if side in self.active_controller else None
        self.ex._log_hand_tracking_status_once = Mock()
        self.ex._set_grab_candidate = Mock()
        self.ex._set_arm_rig_target_visible = Mock()
        self.ex._release_grabbed_object = Mock()
        self.ex._update_arm_rig_target = Mock()
        self.ex._update_grabbed_object = Mock()
        self.ex._is_hand_closed = Mock(return_value=False)
        self.ex._compute_arm_target_body_position = Mock(return_value=Gf.Vec3d(0.4, 0, 0.2))
        self.ex._compute_arm_target_orientation = Mock(return_value=None)
        self.ex._compute_arm_targets_from_body_position = lambda side, *args, **kwargs: {
            index: 0.0 for index in self.ex._g1_arm_dof_indices_by_side[side].values()
        }
        self.ex._pause_faulted_articulation = Mock()
        self.ex._log_physics_step_error = Mock()

    def tick(self):
        self.ex._update_g1_arms_from_hand_tracking()
        args, kwargs = self.robot.set_dof_position_targets.call_args
        return dict(zip(kwargs["dof_indices"], args[0]))

    def test_loss_captures_actual_pose_and_discards_remaining_moving_targets(self):
        self.ex._arm_input_sources = {"left": "hand_tracking", "right": "hand_tracking"}
        self.ex._smoothed_arm_targets = {index: 1.2 for index in range(4)}
        target = self.tick()
        np.testing.assert_allclose(list(target.values()), self.positions[0], atol=1e-7)
        self.assertEqual(set(self.ex._inactive_arm_hold_targets), {"left", "right"})
        self.assertEqual(self.ex._smoothed_arm_targets, target)

    def test_fixed_hold_does_not_recapture_joint_droop_each_tick(self):
        initial = self.tick()
        self.positions += np.array([[-0.03, 0.03, -0.03, -0.03]])
        for _ in range(10):
            target = self.tick()
            np.testing.assert_allclose(list(target.values()), list(initial.values()), atol=1e-7)
        self.assertFalse(self.ex._articulation_health_fault)

    def test_one_active_hand_does_not_send_other_hand_toward_rest(self):
        self.active_optical.add("left")
        self.ex._arm_input_sources["right"] = "hand_tracking"
        target = self.tick()
        self.assertLess(target[0], self.positions[0, 0])
        np.testing.assert_allclose([target[2], target[3]], self.positions[0, 2:], atol=1e-7)
        self.assertNotIn("left", self.ex._inactive_arm_hold_targets)
        self.assertIn("right", self.ex._inactive_arm_hold_targets)

    def test_controller_clutch_release_holds_arm_without_cancelling_valid_finger_input(self):
        self.ex._arm_input_sources["right"] = "controller"
        self.ex._finger_curl_source["right"] = "controller"
        target = self.tick()
        np.testing.assert_allclose([target[2], target[3]], self.positions[0, 2:], atol=1e-7)
        self.assertNotIn(call("right"), self.ex._release_grabbed_object.call_args_list)
        self.ex.g1.set_finger_curls.assert_not_called()

    def test_xr_loss_disabled_arm_input_and_missing_base_reference_all_hold(self):
        for unavailable in ("xr", "disabled", "base"):
            with self.subTest(unavailable=unavailable):
                self.ex._inactive_arm_hold_targets.clear()
                self.ex._xr_core = None if unavailable == "xr" else object()
                self.ex._hand_tracking_arm_control_enabled = unavailable != "disabled"
                self.ex._get_g1_base_pose_for_arms.return_value = (
                    None if unavailable == "base" else (Gf.Vec3d(0, 0, 0.8), 0.0)
                )
                self.active_optical.update(("left", "right"))
                target = self.tick()
                np.testing.assert_allclose(list(target.values()), self.positions[0], atol=1e-7)

    def test_reacquisition_releases_latch_and_next_loss_captures_new_pose(self):
        first = self.tick()
        self.active_optical.add("right")
        target = self.tick()
        self.assertNotIn("right", self.ex._inactive_arm_hold_targets)
        self.assertLess(target[2], first[2])
        self.assertLessEqual(first[2] - target[2], self.ex._arm_max_joint_speed * 0.01 + 1e-7)
        self.positions[0, 2:] = [0.3, 0.4]
        self.active_optical.clear()
        target = self.tick()
        np.testing.assert_allclose([target[2], target[3]], [0.3, 0.4], atol=1e-7)
        self.assertNotEqual(target[2], first[2])

    def test_reset_clears_hold_targets_and_clutch_recalibration_state(self):
        self.tick()
        self.ex._controller_arm_neutral_positions["right"] = Gf.Vec3d(1, 2, 3)
        self.ex._reset_teleoperation_state()
        self.assertEqual(self.ex._inactive_arm_hold_targets, {})
        self.assertEqual(self.ex._controller_arm_neutral_positions, {})

    def test_bad_measurement_pauses_without_writing_a_hold_target(self):
        self.positions[0, 0] = np.nan
        self.ex._update_g1_arms_from_hand_tracking()
        self.robot.set_dof_position_targets.assert_not_called()
        self.ex._pause_faulted_articulation.assert_called_once()
        self.assertIn("inactive arm", self.ex._articulation_health_fault)

    def test_assisted_mode_retains_return_to_default_posture(self):
        self.ex._grasp_mode = "assisted"
        target = self.tick()
        for index, angle in target.items():
            self.assertLess(abs(angle), abs(self.positions[0, index]))
        self.assertFalse(self.ex._inactive_arm_hold_targets)


if __name__ == "__main__":
    unittest.main()
