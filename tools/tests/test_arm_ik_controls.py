# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numeric regressions for bounded task priority in the live arm IK solver."""

from __future__ import annotations

import math
import types
import unittest
from unittest import mock

import numpy as np
from humanoid_test_support import GRASP, HUMANOID, Array, pose
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
        ex._contact_health = "ok"
        smoothed = ex._smooth_and_clamp_arm_targets(right, physics_snapshot=snapshot)
        self.assertEqual(ex._limit_arm_targets_at_contacts(smoothed, snapshot), smoothed)
        # Contact-reader failure must hold the same measured pose, without
        # fetching a different observation after the IK and filtering steps.
        ex._contact_health = "unavailable"
        self.assertEqual(ex._limit_arm_targets_at_contacts(smoothed, snapshot), {index: 0.0 for index in smoothed})
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

    def test_ik_reuses_positions_when_inactive_arm_already_captured_the_step(self):
        ex = example_with_jacobian(independent_hand_tasks())
        ex._g1_arm_dof_indices_by_side["left"] = dict(ex._g1_arm_dof_indices_by_side["right"])
        robot = ex.g1.robot
        robot.get_dof_positions = mock.Mock(wraps=robot.get_dof_positions)
        snapshot = {}
        held = ex._get_inactive_arm_targets("left", snapshot)
        self.assertTrue(held)
        self.assertIsNotNone(ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), physics_snapshot=snapshot))
        robot.get_dof_positions.assert_called_once_with()

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

    def test_orientation_uses_free_joint_when_coupled_wrist_reaches_either_limit(self) -> None:
        for direction in (-1.0, 1.0):
            for remaining in (0.0, 1e-6, 0.02):
                with self.subTest(direction=direction, remaining=remaining):
                    jacobian = np.zeros((6, 7))
                    jacobian[:3, :3] = np.eye(3)
                    # Both wrists can turn the palm, but also translate it. The
                    # remaining joints must compensate that translation exactly.
                    jacobian[0, 4] = 0.5
                    jacobian[0, 6] = -0.25
                    jacobian[5, (4, 6)] = 1.0
                    current = np.zeros(7)
                    current[4] = direction * (0.2 - remaining)
                    ex = example_with_jacobian(jacobian, current)
                    ex._g1_arm_joint_limits = {4: (-0.2, 0.2)}
                    primary = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0, 0, 0.05))
                    oriented = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0, 0, 0.05), pose(yaw=direction * 35))
                    self.assertIsNotNone(oriented)
                    primary_step = np.array([primary[index] for index in range(7)]) - current
                    step = np.array([oriented[index] for index in range(7)]) - current
                    np.testing.assert_allclose(jacobian[:3] @ step, jacobian[:3] @ primary_step, atol=1e-10)
                    self.assertGreater(direction * step[6], 0.15)
                    self.assertGreater(direction * (jacobian[5] @ step), 0.15)
                    self.assertLessEqual(abs(oriented[4]), 0.2 + 1e-12)
                    self.assertLessEqual(float(np.max(np.abs(step))), ex._arm_ik_max_step + 1e-12)

    def test_infeasible_orientation_exhausts_joint_budgets_without_reducing_reach(self) -> None:
        jacobian = independent_hand_tasks()
        current = np.zeros(7)
        current[4:] = 0.2
        ex = example_with_jacobian(jacobian, current)
        ex._g1_arm_joint_limits = {index: (-0.2, 0.2) for index in range(4, 7)}
        target = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 1, 1).GetNormalized(), 35))
        primary = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0))
        oriented = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), target)
        self.assertIsNotNone(oriented)
        np.testing.assert_allclose(list(oriented.values()), list(primary.values()), atol=1e-10)

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

    def test_reach_consumes_small_joint_budgets_then_uses_remaining_joints(self) -> None:
        for direction in (-1.0, 1.0):
            for remaining in (0.0, 0.0004, 0.02):
                with self.subTest(direction=direction, remaining=remaining):
                    jacobian = np.zeros((6, 7))
                    jacobian[0, :3] = 1.0
                    jacobian[1, 3] = jacobian[2, 4] = 1.0
                    jacobian[5, 5:] = (1.0, -1.0)
                    current = np.zeros(7)
                    current[:2] = direction * np.array([0.2 - remaining, 0.192])
                    ex = example_with_jacobian(jacobian, current)
                    ex._g1_arm_joint_limits = {0: (-0.2, 0.2), 1: (-0.2, 0.2)}
                    target = Gf.Vec3d(direction * 0.1, 0.03, 0.04)
                    primary = ex._solve_arm_ik_jacobian("right", target)
                    oriented = ex._solve_arm_ik_jacobian("right", target, pose(yaw=40))
                    self.assertIsNotNone(primary)
                    self.assertIsNotNone(oriented)
                    step = np.array([primary[index] for index in range(7)]) - current
                    oriented_step = np.array([oriented[index] for index in range(7)]) - current
                    # Two different travel limits are exhausted before the third
                    # joint completes the reach. A single scale stalls below 2.4 cm.
                    self.assertGreater(direction * float(jacobian[0] @ step), 0.05)
                    self.assertGreater(direction * step[2], 0.02)
                    self.assertAlmostEqual(primary[0], direction * 0.2, places=10)
                    self.assertAlmostEqual(primary[1], direction * 0.2, places=10)
                    np.testing.assert_allclose(jacobian[:3] @ oriented_step, jacobian[:3] @ step, atol=1e-10)
                    self.assertGreater(float(jacobian[5] @ oriented_step), 0.1)
                    for result in (primary, oriented):
                        self.assertTrue(all(abs(result[index]) <= 0.2 + 1e-12 for index in (0, 1)))
                        self.assertLessEqual(
                            max(abs(result[index] - current[index]) for index in range(7)),
                            ex._arm_ik_max_step + 1e-12,
                        )

    def test_unconstrained_primary_retains_single_damped_least_squares_step(self) -> None:
        jacobian = independent_hand_tasks()
        ex = example_with_jacobian(jacobian)
        target = Gf.Vec3d(0.08, -0.04, 0.06)
        result = ex._solve_arm_ik_jacobian("right", target)
        expected = jacobian[:3].T @ np.linalg.solve(
            jacobian[:3] @ jacobian[:3].T + ex._arm_ik_damping**2 * np.eye(3),
            np.asarray(target) * ex._arm_ik_gain,
        )
        np.testing.assert_allclose([result[index] for index in range(7)], expected, atol=1e-12)

    def test_exhausted_primary_joints_hold_without_blocking_independent_orientation(self) -> None:
        jacobian = independent_hand_tasks()
        current = np.zeros(7)
        current[:3] = 0.2
        ex = example_with_jacobian(jacobian, current)
        ex._g1_arm_joint_limits = {index: (-0.2, 0.2) for index in range(3)}
        result = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1), pose(yaw=35))
        self.assertIsNotNone(result)
        np.testing.assert_allclose([result[index] for index in range(3)], current[:3], atol=1e-12)
        self.assertGreater(result[6], 0.1)
        self.assertTrue(all(math.isfinite(value) for value in result.values()))

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


class TestWalkingArmIkCadence(unittest.TestCase):
    def example(self):
        ex = example_with_jacobian(independent_hand_tasks())
        ex._gesture_locomotion_enabled = True
        ex._last_physics_dt = ex._teleop_input_dt = 0.005
        ex._arm_input_sources["right"] = "hand_tracking"
        ex._solve_arm_ik_jacobian = mock.Mock(wraps=ex._solve_arm_ik_jacobian)
        ex.g1.robot.get_dof_positions = mock.Mock(wraps=ex.g1.robot.get_dof_positions)
        ex.g1.robot.get_jacobian_matrices = mock.Mock(wraps=ex.g1.robot.get_jacobian_matrices)
        return ex

    def test_walking_reuses_only_one_intervening_step_and_copies_raw_goals(self):
        ex = self.example()
        target = Gf.Vec3d(0.1, 0, 0)
        original = ex._compute_arm_targets_from_body_position("right", target)
        saved = dict(original)
        original[0] = 99.0
        # Repeated callers within the same step cannot mutate the saved result.
        repeated = ex._compute_arm_targets_from_body_position("right", target)
        self.assertEqual(repeated, saved)
        repeated.clear()
        for now, solves in ((0.005, 1), (0.010, 2), (0.015, 2), (0.020, 3)):
            ex._grab_time = now
            result = ex._compute_arm_targets_from_body_position("right", target)
            self.assertEqual(result, saved)
            self.assertEqual(ex._solve_arm_ik_jacobian.call_count, solves)
        self.assertEqual(ex.g1.robot.get_jacobian_matrices.call_count, 3)

    def test_cached_raw_goal_still_obeys_fresh_measured_lead_and_per_step_slew(self):
        ex = self.example()
        target = Gf.Vec3d(0.3, 0, 0)
        snapshot = {}
        raw = ex._compute_arm_targets_from_body_position("right", target, physics_snapshot=snapshot)
        first = ex._smooth_and_clamp_arm_targets(raw, snapshot)
        self.assertGreater(first[0], 0)
        self.assertLessEqual(first[0], ex._arm_max_joint_speed * 0.005)
        ex._grab_time = 0.005
        next_snapshot = {}
        reused = ex._compute_arm_targets_from_body_position("right", target, physics_snapshot=next_snapshot)
        self.assertEqual(reused, raw)
        self.assertEqual(next_snapshot, {})
        # An external load moved the measured joint. Reusing a raw goal must not
        # also reuse last step's pose or its allowed command lead.
        ex.g1.robot.get_dof_positions.return_value = Array([[0.2, 0, 0, 0, 0, 0, 0]])
        second = ex._smooth_and_clamp_arm_targets(reused, next_snapshot)
        self.assertGreaterEqual(second[0], 0.2 - ex._arm_max_tracking_error)
        self.assertEqual(ex.g1.robot.get_dof_positions.call_count, 2)
        self.assertEqual(ex._solve_arm_ik_jacobian.call_count, 1)
        self.assertEqual(ex._arm_ik_target_cache["right"]["targets"], raw)

    def test_new_contact_on_reused_step_reads_current_jacobian_and_blocks_inward_motion(self):
        ex = self.example()
        ex._contact_health = "ok"
        target = Gf.Vec3d(0, 0, 0.1)
        snapshot = {}
        raw = ex._compute_arm_targets_from_body_position("right", target, physics_snapshot=snapshot)
        first = ex._smooth_and_clamp_arm_targets(raw, snapshot)
        self.assertGreater(ex._limit_arm_targets_at_contacts(first, snapshot)[2], 0.0)

        ex._grab_time = 0.005
        ex.g1.robot.get_dof_positions.return_value = Array([[0, 0, 0.005, 0, 0, 0, 0]])
        ex._latest_hand_contacts["right"] = [
            GRASP.ContactSample("/World/Ceiling", "index", (0, 0, 0), (0, 0, -1), 1.0, eligible_for_grasp=False)
        ]
        snapshot = {}
        reused = ex._compute_arm_targets_from_body_position("right", target, physics_snapshot=snapshot)
        smoothed = ex._smooth_and_clamp_arm_targets(reused, snapshot)
        self.assertGreater(smoothed[2], 0.005)
        limited = ex._limit_arm_targets_at_contacts(smoothed, snapshot)
        self.assertAlmostEqual(limited[2], 0.005, places=9)
        self.assertTrue(ex._arm_contact_limited["right"])
        self.assertEqual(ex._solve_arm_ik_jacobian.call_count, 1)
        self.assertEqual(ex.g1.robot.get_dof_positions.call_count, 2)
        self.assertEqual(ex.g1.robot.get_jacobian_matrices.call_count, 2)

    def test_large_target_or_orientation_change_and_source_switch_refresh_immediately(self):
        for next_position, first_rotation, next_rotation, source in (
            ((0.121, 0, 0), pose(), pose(), "hand_tracking"),
            ((0.1, 0, 0), pose(), pose(yaw=6), "hand_tracking"),
            ((0.1, 0, 0), None, pose(), "hand_tracking"),
            ((0.1, 0, 0), pose(), None, "hand_tracking"),
            ((0.1, 0, 0), pose(), pose(), "controller"),
        ):
            with self.subTest(position=next_position, source=source):
                ex = self.example()
                ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.1, 0, 0), first_rotation)
                ex._grab_time = 0.005
                ex._arm_input_sources["right"] = source
                ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(*next_position), next_rotation)
                self.assertEqual(ex._solve_arm_ik_jacobian.call_count, 2)

    def test_small_target_change_reuses_goals_but_never_skips_two_steps(self):
        ex = self.example()
        ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.1, 0, 0), pose())
        ex._grab_time = 0.005
        ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.11, 0, 0), pose(yaw=3))
        self.assertEqual(ex._solve_arm_ik_jacobian.call_count, 1)
        # Even an unexpected extra callback before the nominal deadline can
        # reuse only the same intervening step, never a second distinct step.
        ex._grab_time = 0.0075
        ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.11, 0, 0), pose(yaw=3))
        self.assertEqual(ex._solve_arm_ik_jacobian.call_count, 2)

    def test_stationary_or_other_physics_rates_always_solve_fresh(self):
        for walking, dt in ((False, 0.005), (False, 0.010), (True, 0.010), (True, 0.0025)):
            with self.subTest(walking=walking, dt=dt):
                ex = self.example()
                ex._gesture_locomotion_enabled = walking
                ex._last_physics_dt = dt
                for now in (0, dt):
                    ex._grab_time = now
                    ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.1, 0, 0))
                self.assertEqual(ex._solve_arm_ik_jacobian.call_count, 2)
                self.assertFalse(ex._arm_ik_target_cache)

    def test_lost_source_and_teleoperation_reset_clear_saved_goals(self):
        ex = self.example()
        ex._release_grabbed_object = mock.Mock()
        ex._set_grab_candidate = mock.Mock()
        ex._set_arm_rig_target_visible = mock.Mock()
        ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.1, 0, 0))
        ex._deactivate_hand("right")
        self.assertNotIn("right", ex._arm_ik_target_cache)
        ex._arm_input_sources["right"] = "hand_tracking"
        ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.1, 0, 0))
        self.assertEqual(ex._solve_arm_ik_jacobian.call_count, 2)
        ex._reset_teleoperation_state()
        self.assertFalse(ex._arm_ik_target_cache)

    def test_invalid_or_empty_results_and_bad_clocks_are_never_reused(self):
        for result in ({}, {0: math.nan}, {0: math.inf}):
            with self.subTest(result=result):
                ex = self.example()
                ex._compute_raw_arm_targets_from_body_position = mock.Mock(return_value=result)
                for now in (0, 0.005):
                    ex._grab_time = now
                    ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.1, 0, 0))
                self.assertFalse(ex._arm_ik_target_cache)
                self.assertEqual(ex._compute_raw_arm_targets_from_body_position.call_count, 2)
        for now in (-0.005, math.nan, math.inf):
            with self.subTest(now=now):
                ex = self.example()
                ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.1, 0, 0))
                ex._grab_time = now
                ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.1, 0, 0))
                self.assertEqual(ex._solve_arm_ik_jacobian.call_count, 2)

    def test_invalid_physics_views_discard_old_goal_without_caching_fallback(self):
        ex = self.example()
        ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.1, 0, 0))
        ex.g1.robot.is_physics_tensor_entity_valid = lambda: False
        ex._grab_time = 0.005
        ex._compute_arm_targets_from_body_position("right", Gf.Vec3d(0.1, 0, 0))
        self.assertEqual(ex._solve_arm_ik_jacobian.call_count, 2)
        self.assertFalse(ex._arm_ik_target_cache)
