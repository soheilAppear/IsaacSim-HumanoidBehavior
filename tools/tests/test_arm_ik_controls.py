# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numeric regressions for bounded task priority in the live arm IK solver."""

from __future__ import annotations

import math
import types
import unittest
from unittest import mock

import numpy as np
from humanoid_test_support import HUMANOID, Array, pose
from pxr import Gf


def example_with_jacobian(jacobian, current=None):
    """Provide real solver math with a controlled fixed-base tensor boundary."""
    example = HUMANOID.HumanoidExample()
    count = jacobian.shape[1]
    names = [*HUMANOID.ARM_IK_JOINT_ORDER, "wrist_roll", "wrist_pitch", "wrist_yaw"][:count]
    positions = np.zeros(count) if current is None else np.asarray(current, dtype=float)
    example._g1_arm_dof_indices_by_side = {"right": dict(zip(names, range(count)))}
    example._get_arm_link_index = lambda side: 1
    example._get_g1_base_pose_for_arms = lambda: (Gf.Vec3d(0), 0.0)
    example._get_hand_link_prim = lambda side: types.SimpleNamespace(
        get_world_poses=lambda: (Array([[0.0, 0, 0]]), Array([[1.0, 0, 0, 0]]))
    )
    example._palm_offset_world = lambda side, wrist: Gf.Vec3d(0)
    example.g1 = types.SimpleNamespace(
        robot=types.SimpleNamespace(
            num_dofs=count,
            num_links=2,
            is_physics_tensor_entity_valid=lambda: True,
            get_jacobian_matrices=lambda: Array(jacobian.reshape(1, 1, 6, count)),
            get_dof_positions=lambda: Array(positions.reshape(1, count)),
        )
    )
    return example


def independent_hand_tasks():
    jacobian = np.zeros((6, 7))
    jacobian[:3, :3] = np.eye(3)
    jacobian[3:, 4:] = np.eye(3)
    return jacobian


class TestArmPriorityControls(unittest.TestCase):
    def test_both_arms_share_one_physics_read_and_next_callback_refreshes(self):
        ex = example_with_jacobian(independent_hand_tasks())
        # The read-sharing contract is independent of the arms' DOF mapping.
        ex._g1_arm_dof_indices_by_side["left"] = dict(ex._g1_arm_dof_indices_by_side["right"])
        robot = ex.g1.robot
        robot.get_jacobian_matrices = mock.Mock(wraps=robot.get_jacobian_matrices)
        robot.get_dof_positions = mock.Mock(wraps=robot.get_dof_positions)
        snapshot = {}
        right = ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.1, 0, 0), physics_snapshot=snapshot)
        left = ex._compute_arm_targets_from_body_position("left", Gf.Vec3d(0.1, 0, 0), physics_snapshot=snapshot)
        self.assertGreater(right[0], 0.0)
        self.assertEqual(right, left)
        self.assertEqual(robot.get_jacobian_matrices.call_count, 1)
        self.assertEqual(robot.get_dof_positions.call_count, 1)

        # New simulation state must be observed when the caller starts the next
        # callback with a fresh dictionary. Change both tensor results so neither
        # a cached Jacobian nor cached positions can satisfy this assertion.
        next_jacobian = independent_hand_tasks()
        next_jacobian[0, 0] = 2.0
        next_positions = np.full(7, 0.1)
        robot.get_jacobian_matrices.return_value = Array(next_jacobian.reshape(1, 1, 6, 7))
        robot.get_dof_positions.return_value = Array(next_positions.reshape(1, 7))
        refreshed = ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.1, 0, 0), physics_snapshot={})
        expected = example_with_jacobian(next_jacobian, next_positions)._solve_arm_ik_jacobian(
            "right", Gf.Vec3d(0.1, 0, 0)
        )
        self.assertEqual(refreshed, expected)
        self.assertNotEqual(refreshed, right)
        self.assertEqual(robot.get_jacobian_matrices.call_count, 2)
        self.assertEqual(robot.get_dof_positions.call_count, 2)

        # Standalone calls omit the optional dictionary and always read afresh.
        ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0))
        ex._solve_arm_ik_jacobian("left", Gf.Vec3d(0.1, 0, 0))
        self.assertEqual(robot.get_jacobian_matrices.call_count, 4)
        self.assertEqual(robot.get_dof_positions.call_count, 4)

    def test_large_wrist_correction_cannot_consume_primary_reach_budget(self):
        ex = example_with_jacobian(independent_hand_tasks())
        primary = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0))
        oriented = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), pose(yaw=170))
        self.assertGreater(primary[0], 0.0)
        self.assertAlmostEqual(oriented[0], primary[0], places=10)
        self.assertGreater(oriented[6], 0.0)
        self.assertLessEqual(max(abs(value) for value in oriented.values()), ex._arm_ik_max_step + 1e-12)

    def test_large_posture_correction_cannot_slow_primary_reach(self):
        ex = example_with_jacobian(independent_hand_tasks())
        primary = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0))
        # Deliberately extreme preference on the redundant joint exercises budget
        # saturation. It must not rescale the independent positional correction.
        ex._g1_arm_joint_defaults = {3: -50.0}
        relaxed = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0))
        self.assertAlmostEqual(relaxed[0], primary[0], places=10)
        self.assertAlmostEqual(relaxed[3], -ex._arm_ik_max_step, places=10)

    def test_secondary_budget_preserves_reach_with_coupled_joint_axes(self):
        jacobian = np.zeros((6, 7))
        jacobian[0, :2] = (1.0, 1.0)
        jacobian[5, :2] = (1.0, -1.0)
        ex = example_with_jacobian(jacobian)
        primary = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0))
        oriented = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), pose(yaw=170))
        # Wrist motion moves the same two joints in opposite directions. Limiting
        # either joint independently would change the X reach given by their sum.
        self.assertAlmostEqual(oriented[0] + oriented[1], primary[0] + primary[1], places=10)
        self.assertGreater(oriented[0] - oriented[1], 0.0)
        self.assertLessEqual(max(abs(value) for value in oriented.values()), ex._arm_ik_max_step + 1e-12)

    def test_wrist_limit_preserves_reach_and_keeps_secondary_target_feasible(self):
        ex = example_with_jacobian(independent_hand_tasks())
        ex._g1_arm_joint_limits = {6: (-0.2, 0.04)}
        primary = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0))
        oriented = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), pose(yaw=170))
        self.assertAlmostEqual(oriented[0], primary[0], places=10)
        self.assertGreater(oriented[6], 0.0)
        self.assertLessEqual(oriented[6], 0.04 + 1e-12)

    def test_reach_resolves_through_free_joint_when_other_joint_hits_limit(self):
        for direction in (-1.0, 1.0):
            with self.subTest(direction=direction):
                jacobian = np.zeros((6, 7))
                jacobian[0, 0:2] = 1.0
                current = np.zeros(7)
                current[0] = direction * 0.2
                ex = example_with_jacobian(jacobian, current)
                ex._g1_arm_joint_limits = {0: (-0.2, 0.2), 1: (-1.0, 1.0)}
                targets = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(direction * 0.1, 0, 0))
                self.assertAlmostEqual(targets[0], current[0], places=10)
                self.assertGreater(direction * targets[1], 0.0)
                self.assertLessEqual(abs(targets[1]), ex._arm_ik_max_step)

    def test_singular_and_near_singular_tasks_produce_finite_bounded_targets(self):
        for small_axis in (0.0, 1e-9):
            with self.subTest(small_axis=small_axis):
                jacobian = independent_hand_tasks()
                jacobian[1, 1] = small_axis
                jacobian[2, 2] = 0.0
                ex = example_with_jacobian(jacobian)
                targets = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0.1, 0.1), pose(yaw=170))
                self.assertIsNotNone(targets)
                self.assertGreater(targets[0], 0.0)
                self.assertTrue(all(math.isfinite(value) for value in targets.values()))
                self.assertLessEqual(max(abs(value) for value in targets.values()), ex._arm_ik_max_step + 1e-12)

    def test_invalid_measurements_or_limits_do_not_issue_targets(self):
        jacobian = independent_hand_tasks()
        jacobian[0, 0] = math.inf
        ex = example_with_jacobian(jacobian)
        self.assertIsNone(ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0)))
        ex = example_with_jacobian(independent_hand_tasks(), [math.nan, 0, 0, 0, 0, 0, 0])
        self.assertIsNone(ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0)))
        ex = example_with_jacobian(independent_hand_tasks())
        ex._g1_arm_joint_limits = {0: (0.2, -0.2)}
        self.assertIsNone(ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0)))
