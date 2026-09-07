# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regress walking-policy timing, reset state, fallback, and arm Jacobian geometry."""

from __future__ import annotations

import math
import types
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from humanoid_test_support import G1, HUMANOID, PACKAGE, Array, pose
from pxr import Gf


def walking_robot():
    robot = G1.G1TeleopRobot.__new__(G1.G1TeleopRobot)
    robot.robot = types.SimpleNamespace(set_dof_position_targets=Mock())
    robot._walk_policy = Mock(return_value=torch.zeros(12))
    robot._walk_previous_action = torch.zeros(12)
    robot._walk_leg_defaults_tensor = torch.zeros(12)
    robot._walk_leg_lower_limits = torch.full((12,), -1.0)
    robot._walk_leg_upper_limits = torch.full((12,), 1.0)
    robot._leg_dof_indices = list(range(12))
    robot._reset_walk_policy_state()
    robot._station_keeping_command = lambda command: command
    robot._compute_walk_observation = lambda command: torch.zeros(47)
    return robot


class TestG1Controls(unittest.TestCase):
    def test_bundled_checkpoint_outputs_finite_actions_and_resets_deterministically(self):
        robot = walking_robot()
        robot.robot._device = "cpu"
        robot.robot.get_dof_limits = lambda **kwargs: (Array([[-3.0] * 12]), Array([[3.0] * 12]))
        robot._body_dof_indices = list(range(12))
        robot._walk_policy_path = str(PACKAGE / "robots/data/g1_unitree_motion.pt")
        robot._configure_walk_policy(list(robot.WALK_LEG_JOINT_ORDER))
        self.assertIsNotNone(robot._walk_policy)
        observation = torch.zeros(1, 47)
        observation[0, 5] = -1.0
        observation[0, 46] = 1.0
        with torch.no_grad():
            first = robot._walk_policy(observation).clone()
            for _ in range(25):
                output = robot._walk_policy(observation)
                self.assertEqual(output.numel(), 12)
                self.assertTrue(torch.isfinite(output).all())
            robot._reset_walk_policy_state()
            reset = robot._walk_policy(observation)
        torch.testing.assert_close(first, reset)

    def test_lstm_is_called_50_times_per_second_at_each_physics_rate(self):
        for hz in (60, 90, 100, 120, 200):
            with self.subTest(hz=hz):
                robot = walking_robot()
                for _ in range(hz * 2):
                    robot._step_walk_policy(1.0 / hz, [0.5, 0, 0])
                self.assertEqual(robot._walk_policy.call_count, 100)

    def test_long_frame_does_not_replay_the_lstm_on_one_observation(self):
        robot = walking_robot()
        robot._step_walk_policy(0.01, [0, 0, 0])
        robot._step_walk_policy(0.5, [0, 0, 0])
        self.assertEqual(robot._walk_policy.call_count, 2)
        self.assertGreater(robot._walk_next_control_time, robot._walk_time)

    def test_invalid_dt_does_not_advance_policy_or_corrupt_clock(self):
        robot = walking_robot()
        for dt in (0, -1, math.nan, math.inf):
            robot._step_walk_policy(dt, [0, 0, 0])
        self.assertEqual(robot._walk_time, 0)
        robot._walk_policy.assert_not_called()

    def test_invalid_action_preserves_last_valid_leg_target(self):
        robot = walking_robot()
        robot._walk_policy.return_value = torch.full((12,), math.nan)
        robot._step_walk_policy(0.02, [0, 0, 0])
        robot.robot.set_dof_position_targets.assert_not_called()
        self.assertTrue(torch.equal(robot._walk_previous_action, torch.zeros(12)))

    def test_action_targets_respect_live_joint_limits(self):
        robot = walking_robot()
        robot._walk_policy.return_value = torch.tensor([100, -100] * 6, dtype=torch.float32)
        robot._step_walk_policy(0.02, [0, 0, 0])
        targets = robot.robot.set_dof_position_targets.call_args.args[0].numpy()
        np.testing.assert_allclose(targets, [1, -1] * 6)

    def test_kinematic_fallback_sets_gravity_and_clears_policy_joint_ownership(self):
        robot = walking_robot()
        robot._body_dof_indices = list(range(29))
        robot._author_disabled_gravity = Mock()
        robot._activate_kinematic_fallback()
        self.assertEqual(robot._locomotion, "kinematic")
        self.assertTrue(robot._disable_gravity)
        self.assertIsNone(robot._walk_policy)
        self.assertEqual(robot._leg_dof_indices, [])
        self.assertEqual(robot._posture_dof_indices, list(range(29)))
        robot._author_disabled_gravity.assert_called_once()

    def test_initialize_registers_complete_spawn_reset_state(self):
        robot = G1.G1TeleopRobot.__new__(G1.G1TeleopRobot)
        names = ["left_shoulder_pitch_joint", "left_elbow_joint"]
        robot.robot = Mock()
        robot.robot.dof_names = names
        robot.robot._device = "cpu"
        robot.robot.num_links = 3
        robot.robot.get_dof_positions.return_value = Array([[0, 0]])
        robot._locomotion = "kinematic"
        robot._disable_gravity = True
        robot._spawn_position = [3, 2, 0.8]
        robot._spawn_orientation = [0.70710678, 0, 0, 0.70710678]
        robot._spawn_yaw = math.pi / 2
        robot._apply_body_joint_gains = Mock()
        robot._configure_finger_dofs = Mock()
        robot._apply_articulation_properties = Mock()
        robot.initialize()
        state = robot.robot.set_default_state.call_args.kwargs
        np.testing.assert_allclose(state["positions"], [[3, 2, 0.8]])
        np.testing.assert_allclose(state["orientations"], [robot._spawn_orientation])
        np.testing.assert_allclose(state["linear_velocities"], [[0, 0, 0]])
        np.testing.assert_allclose(state["angular_velocities"], [[0, 0, 0]])
        np.testing.assert_allclose(state["dof_positions"], [[0.2, 0.6]])

    def test_missing_policy_selects_avatar_posture_before_joint_defaults(self):
        robot = G1.G1TeleopRobot.__new__(G1.G1TeleopRobot)
        robot.robot = Mock()
        robot.robot.dof_names = ["left_shoulder_pitch_joint", "left_elbow_joint"]
        robot.robot._device = "cpu"
        robot.robot.num_links = 3
        robot.robot.get_dof_positions.return_value = Array([[0, 0]])
        robot._locomotion = "policy"
        robot._disable_gravity = False
        robot._spawn_position = [0, 0, 0.8]
        robot._spawn_orientation = [1, 0, 0, 0]
        robot._spawn_yaw = 0
        robot._author_disabled_gravity = Mock()
        robot._apply_body_joint_gains = Mock()
        robot._configure_finger_dofs = Mock()
        robot._apply_articulation_properties = Mock()
        # Actual configuration detects missing leg names and takes the complete fallback.
        robot.initialize()
        self.assertEqual(robot._locomotion, "kinematic")
        np.testing.assert_allclose(robot._default_dof_positions, [0.2, 0.6])
        self.assertTrue(robot._disable_gravity)

    def test_bad_command_values_cannot_poison_root_pose(self):
        for command in ([math.nan, 0, 0], [0, math.inf, 0], [1], None):
            self.assertEqual(G1.G1TeleopRobot._command_to_floats(command), (0.0, 0.0, 0.0))


class TestArmIKControls(unittest.TestCase):
    def make_example(self, fixed=False, yaw=0.0, palm_offset=(0, 0, 0)):
        example = HUMANOID.HumanoidExample()
        names = [*HUMANOID.ARM_IK_JOINT_ORDER, "wrist_roll", "wrist_pitch", "wrist_yaw"]
        example._g1_arm_dof_indices_by_side = {"right": dict(zip(names, range(7)))}
        example._get_arm_link_index = lambda side: 1
        example._get_g1_base_pose_for_arms = lambda: (Gf.Vec3d(0), yaw)
        hand = types.SimpleNamespace(get_world_poses=lambda: (Array([[0.0, 0, 0]]), Array([[1.0, 0, 0, 0]])))
        example._get_hand_link_prim = lambda side: hand
        example._palm_offset_world = lambda side, wrist: Gf.Vec3d(*palm_offset)
        offset = 0 if fixed else 6
        jacobian = torch.zeros((1, 1 if fixed else 2, 6, offset + 7))
        link_row = 0 if fixed else 1
        for axis in range(3):
            jacobian[0, link_row, axis, offset + axis] = 1.0
            jacobian[0, link_row, axis + 3, offset + axis + 4] = 1.0
        example.g1 = types.SimpleNamespace(
            robot=types.SimpleNamespace(
                num_dofs=7,
                num_links=2,
                is_physics_tensor_entity_valid=lambda: True,
                get_jacobian_matrices=lambda: Array(jacobian),
                get_dof_positions=lambda: Array([[0.0] * 7]),
            )
        )
        return example, jacobian

    def test_fixed_and_floating_base_use_correct_rows_and_columns(self):
        for fixed in (True, False):
            with self.subTest(fixed=fixed):
                ex, _ = self.make_example(fixed=fixed)
                targets = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0))
                self.assertIsNotNone(targets)
                self.assertGreater(targets[0], 0)
                self.assertAlmostEqual(targets[1], 0)

    def test_body_target_is_rotated_into_the_world_jacobian(self):
        ex, _ = self.make_example(yaw=math.pi / 2)
        targets = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0))
        self.assertGreater(targets[1], 0)
        self.assertAlmostEqual(targets[0], 0, places=7)

    def test_palm_offset_accounts_for_wrist_angular_velocity(self):
        ex, jacobian = self.make_example(palm_offset=(0.1, 0, 0))
        jacobian[0, 1, :3, :] = 0.0  # Only angular velocity can move the palm point.
        targets = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0.05, 0))
        self.assertGreater(targets[6], 0.0)

    def test_orientation_error_drives_the_wrist_in_the_correct_direction(self):
        ex, _ = self.make_example()
        targets = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0), pose(yaw=20))
        self.assertGreater(targets[6], 0.0)
        self.assertAlmostEqual(targets[0], 0.0)

    def test_conflicting_wrist_orientation_cannot_reverse_position_reach(self):
        ex, jacobian = self.make_example(fixed=True)
        # One joint both translates +X and rotates +Z. A requested -Z rotation
        # conflicts with reaching +X; the hand must keep reaching toward the object.
        jacobian[0, 0, 5, :] = 0.0
        jacobian[0, 0, 5, 0] = 1.0
        position_only = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.03, 0, 0))
        oriented = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.03, 0, 0), pose(yaw=-90))
        self.assertGreater(oriented[0], 0.0)
        self.assertAlmostEqual(oriented[0], position_only[0], places=7)

    def test_redundant_joint_returns_toward_default_without_changing_hand_tasks(self):
        ex, jacobian = self.make_example(fixed=True)
        current = np.array([0, 0, 0, 0.8, 0, 0, 0], dtype=float)
        ex.g1.robot.get_dof_positions = lambda: Array(current.reshape(1, -1))
        # Joint 3 affects neither palm position nor wrist orientation in this scene.
        ex._g1_arm_joint_defaults = dict(enumerate(current))
        baseline = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), pose(yaw=20))
        ex._g1_arm_joint_defaults[3] = 0.0
        relaxed = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), pose(yaw=20))
        self.assertGreater(relaxed[3], 0.0)
        self.assertLess(relaxed[3], current[3])
        task_change = jacobian[0, 0].numpy() @ np.array([relaxed[i] - baseline[i] for i in range(7)])
        np.testing.assert_allclose(task_change, np.zeros(6), atol=1e-8)

    def test_wrist_rotation_error_uses_world_axes_from_a_nontrivial_pose(self):
        ex, _ = self.make_example(fixed=True)
        current = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3).GetNormalized(), 73))
        quaternion = current.ExtractRotationQuat()
        values = [float(quaternion.GetReal()), *[float(v) for v in quaternion.GetImaginary()]]
        ex._get_hand_link_prim = lambda side: types.SimpleNamespace(
            get_world_poses=lambda: (Array([[0.0, 0, 0]]), Array([values]))
        )
        # Right multiplication of row-vector rotations requests a world-Z turn.
        desired = current * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 12))
        targets = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0), desired)
        self.assertGreater(targets[6], 0.0)
        self.assertAlmostEqual(targets[4], 0.0, places=6)
        self.assertAlmostEqual(targets[5], 0.0, places=6)
