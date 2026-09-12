# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check arm contact projection against analytical constrained-motion cases."""

from __future__ import annotations

import types
import unittest
from unittest.mock import Mock

import numpy as np
from humanoid_test_support import ARM_CONTACT, GRASP, HUMANOID, Array


class TestArmContactProjection(unittest.TestCase):
    def project(self, step, rows, lower=None, upper=None):
        step = np.asarray(step, dtype=float)
        if lower is None:
            lower = np.full(step.shape, -1.0)
        if upper is None:
            upper = np.full(step.shape, 1.0)
        return ARM_CONTACT.project_contact_step(step, rows, lower, upper)

    def test_free_and_withdrawing_motion_remain_unchanged(self):
        for step, rows in (([0.2, -0.3], []), ([0.2, -0.3], [[1, 0]]), ([0.0, -0.3], [[1, 0]])):
            with self.subTest(step=step, rows=rows):
                np.testing.assert_allclose(self.project(step, rows), step, atol=1e-12)

    def test_table_contact_blocks_inward_translation_and_preserves_tangent(self):
        np.testing.assert_allclose(self.project([-0.2, 0.3], [[1, 0]]), [0, 0.3], atol=1e-12)

    def test_off_center_contact_blocks_inward_wrist_rotation(self):
        # A positive rotation lifts the contact point along its normal; the
        # other joint contributes pure tangential translation at that point.
        np.testing.assert_allclose(self.project([0.4, -0.2], [[0, 0.03]]), [0.4, 0], atol=1e-12)

    def test_oblique_normal_finds_nearest_tangent_not_whole_arm_stop(self):
        np.testing.assert_allclose(self.project([-1.0, 0.0], [[1, 1]]), [-0.5, 0.5], atol=1e-10)

    def test_opposing_contacts_preserve_shared_tangential_motion(self):
        np.testing.assert_allclose(self.project([0.4, 0.6], [[1, 0], [-1, 0]]), [0, 0.6], atol=1e-10)

    def test_joint_bounds_do_not_reintroduce_inward_motion(self):
        # Clipping the box once would produce [-0.4, 0.1], penetrating the
        # oblique plane. The closest jointly feasible point is [-0.1, 0.1].
        result = self.project([-0.4, 0.7], [[1, 1]], lower=[-0.5, -0.2], upper=[0.1, 0.1])
        np.testing.assert_allclose(result, [-0.1, 0.1], atol=1e-9)
        self.assertGreaterEqual(sum(result), -1e-9)

    def test_zero_rows_are_ignored_and_row_scale_cannot_change_motion(self):
        expected = self.project([-0.4, 0.2], [[1, 1]])
        for magnitude in (1e-250, 0.01, 4.0, 1e250):
            with self.subTest(magnitude=magnitude):
                np.testing.assert_allclose(
                    self.project([-0.4, 0.2], [[0, 0], [magnitude, magnitude]]), expected, atol=1e-10
                )
        np.testing.assert_allclose(self.project([2.0, -2.0], [[0, 0]]), [1, -1], atol=1e-12)

    def test_slow_convergence_falls_back_to_feasible_zero(self):
        result = self.project([-1.0, -1.0], [[1, 0], [-1, 1e-4]])
        np.testing.assert_array_equal(result, [0, 0])

    def test_projection_is_bounded_feasible_and_never_increases_norm(self):
        rng = np.random.default_rng(97)
        for _ in range(30):
            step = rng.uniform(-0.15, 0.15, 7)
            rows = rng.normal(size=(6, 7))
            lower = -rng.uniform(0.01, 0.15, 7)
            upper = rng.uniform(0.01, 0.15, 7)
            result = self.project(step, rows, lower, upper)
            self.assertTrue(np.all(rows @ result >= -1e-9))
            self.assertTrue(np.all(result >= lower - 1e-9))
            self.assertTrue(np.all(result <= upper + 1e-9))
            self.assertLessEqual(np.linalg.norm(result), np.linalg.norm(step) + 1e-9)

    def test_inputs_are_never_mutated(self):
        arrays = tuple(np.asarray(value, dtype=float) for value in ([-0.2, 0.3], [[1, 1]], [-1, -1], [1, 1]))
        before = tuple(value.copy() for value in arrays)
        result = ARM_CONTACT.project_contact_step(*arrays)
        for actual, expected in zip(arrays, before):
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.shares_memory(result, arrays[0]))

    def test_malformed_nonfinite_or_zero_excluding_bounds_are_rejected(self):
        invalid = (
            ([[0, 0]], [[1, 0]], [-1, -1], [1, 1]),
            ([], [], [], []),
            ([0, 0], [1, 0], [-1, -1], [1, 1]),
            ([0, 0], [[1]], [-1, -1], [1, 1]),
            ([0, 0], [[1, 0]], [-1], [1, 1]),
            ([0, 0], [[1, 0]], [0.01, -1], [1, 1]),
            ([0, 0], [[1, 0]], [-1, -1], [1, -0.01]),
            ([np.nan, 0], [[1, 0]], [-1, -1], [1, 1]),
            ([0, 0], [[np.inf, 0]], [-1, -1], [1, 1]),
            ([0, 0], [[1, 0]], [-np.inf, -1], [1, 1]),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ARM_CONTACT.project_contact_step(*values)


class TestArmContactIntegration(unittest.TestCase):
    def setUp(self):
        self.ex = HUMANOID.HumanoidExample()
        self.robot = Mock()
        self.robot.num_dofs = 4
        self.robot.num_links = 3
        self.positions = np.array([[0.0, 0.0, 0.0, 0.0]])
        self.jacobians = np.zeros((1, 2, 6, 4))
        self.jacobians[0, 1, 2, 1] = 1.0  # Joint 1 moves toward +Z.
        self.jacobians[0, 1, 0, 3] = 1.0  # Joint 3 slides along +X.
        self.robot.get_dof_positions.side_effect = lambda: Array(self.positions)
        self.robot.get_jacobian_matrices.side_effect = lambda: Array(self.jacobians)
        self.ex.g1 = types.SimpleNamespace(robot=self.robot)
        self.ex._g1_arm_dof_indices_by_side = {"right": {"shoulder_pitch": 1, "wrist_yaw": 3}}
        self.ex._g1_arm_joint_limits = {1: (-1.0, 1.0), 3: (-1.0, 1.0)}
        self.ex._get_arm_link_index = Mock(return_value=2)
        self.hand = types.SimpleNamespace(get_world_poses=lambda: (Array([[1.0, 2.0, 3.0]]), Array([[1, 0, 0, 0]])))
        self.ex._get_hand_link_prim = Mock(return_value=self.hand)
        self.ex._log_physics_step_error = Mock()
        self.ex._contact_health = "ok"
        self.ex._latest_hand_contacts = {"right": [self.contact()]}

    def contact(self, *, point=(1.0, 2.0, 3.0), normal=(0.0, 0.0, 1.0), eligible=False):
        return GRASP.ContactSample("/World/Table", "index", point, normal, 1.0, eligible_for_grasp=eligible)

    def test_fixed_base_contact_projection_preserves_unrelated_targets_and_updates_filter_cache(self):
        targets = {0: 0.12, 1: -0.1, 3: 0.05}
        self.ex._smoothed_arm_targets = dict(targets)
        result = self.ex._limit_arm_targets_at_contacts(targets, {})
        np.testing.assert_allclose([result[0], result[1], result[3]], [0.12, 0, 0.05], atol=1e-10)
        self.assertEqual(self.ex._smoothed_arm_targets, result)
        self.assertTrue(self.ex._arm_contact_limited["right"])
        self.assertEqual(targets, {0: 0.12, 1: -0.1, 3: 0.05})
        self.robot.set_dof_positions.assert_not_called()
        self.robot.set_world_poses.assert_not_called()

    def test_contact_point_shift_blocks_wrist_rotation_into_table(self):
        self.jacobians[:] = 0.0
        self.jacobians[0, 1, 0, 1] = 1.0  # Tangential translation.
        self.jacobians[0, 1, 4, 3] = 1.0  # Angular velocity about +Y.
        self.ex._latest_hand_contacts = {"right": [self.contact(point=(1.1, 2.0, 3.0))]}
        result = self.ex._limit_arm_targets_at_contacts({1: 0.05, 3: 0.1}, {})
        # omega_Y cross offset_X points down. The sign must block positive,
        # not negative, wrist rotation while preserving tangential translation.
        np.testing.assert_allclose([result[1], result[3]], [0.05, 0.0], atol=1e-10)
        result = self.ex._limit_arm_targets_at_contacts({1: 0.05, 3: -0.1}, {})
        np.testing.assert_allclose([result[1], result[3]], [0.05, -0.1], atol=1e-10)

    def test_predictive_table_force_blocks_motion_before_geometric_touch(self):
        # The live failure first reported a loaded contact at +8.6 mm. PhysX
        # applies this predictive load within the authored contact offset.
        self.ex._latest_hand_contacts = {
            "right": [
                GRASP.ContactSample(
                    "/World/Table",
                    "little",
                    (1.0, 2.0, 3.0),
                    (0.0, 0.0, 1.0),
                    0.65,
                    separation_m=0.0086,
                    eligible_for_grasp=False,
                )
            ]
        }
        result = self.ex._limit_arm_targets_at_contacts({1: -0.1, 3: 0.05}, {})
        np.testing.assert_allclose([result[1], result[3]], [0.0, 0.05], atol=1e-10)

    def test_floating_base_layout_ignores_six_root_columns(self):
        fixed = self.jacobians.copy()
        self.jacobians = np.full((1, 3, 6, 10), 37.0)
        self.jacobians[0, 2, :, 6:] = fixed[0, 1]
        result = self.ex._limit_arm_targets_at_contacts({1: -0.1, 3: 0.05}, {})
        np.testing.assert_allclose([result[1], result[3]], [0.0, 0.05], atol=1e-10)

    def test_uses_current_callback_snapshot_without_duplicate_tensor_reads(self):
        snapshot = {"arm_dof_positions": self.positions.reshape(-1), "arm_jacobians": self.jacobians}
        result = self.ex._limit_arm_targets_at_contacts({1: -0.1, 3: 0.05}, snapshot)
        self.assertAlmostEqual(result[1], 0.0)
        self.robot.get_dof_positions.assert_not_called()
        self.robot.get_jacobian_matrices.assert_not_called()
        self.ex._limit_arm_targets_at_contacts({1: -0.1, 3: 0.05}, {})
        self.robot.get_dof_positions.assert_called_once()
        self.robot.get_jacobian_matrices.assert_called_once()

    def test_dynamic_graspable_object_does_not_pin_arm_and_stale_limit_flag_clears(self):
        targets = {1: -0.1, 3: 0.05}
        self.ex._limit_arm_targets_at_contacts(targets, {})
        self.assertTrue(self.ex._arm_contact_limited["right"])
        self.ex._latest_hand_contacts = {"right": [self.contact(eligible=True)]}
        self.assertEqual(self.ex._limit_arm_targets_at_contacts(targets, {}), targets)
        self.assertFalse(self.ex._arm_contact_limited)
        self.ex._latest_hand_contacts = {"right": [self.contact()]}
        self.ex._grasp_mode = "assisted"
        self.assertEqual(self.ex._limit_arm_targets_at_contacts(targets, {}), targets)

    def test_invalid_contact_jacobian_holds_measured_arm_instead_of_original_command(self):
        self.positions[0, 1] = 0.2
        self.positions[0, 3] = -0.2
        self.jacobians = np.zeros((1, 2, 6, 9))
        result = self.ex._limit_arm_targets_at_contacts({1: 0.1, 3: -0.1}, {})
        np.testing.assert_allclose([result[1], result[3]], [0.2, -0.2], atol=1e-10)
        self.assertEqual(self.ex._smoothed_arm_targets, result)
        self.assertTrue(self.ex._arm_contact_limited["right"])
        self.ex._log_physics_step_error.assert_called_once()

    def test_finger_update_discards_old_contacts_before_any_early_return(self):
        self.ex._finger_control_enabled = False
        self.ex._update_g1_fingers()
        self.assertEqual(self.ex._latest_hand_contacts, {"left": [], "right": []})

    def test_unhealthy_contact_reader_holds_both_arms_and_recovers_on_fresh_read(self):
        self.ex._g1_arm_dof_indices_by_side["left"] = {"shoulder_pitch": 0}
        self.positions[:] = [[-0.1, 0.2, 0.0, -0.2]]
        self.ex._latest_hand_contacts = {"left": [], "right": []}
        self.ex._contact_health = "invalidated"
        requested = {0: 0.1, 1: 0.1, 3: -0.1}
        result = self.ex._limit_arm_targets_at_contacts(requested, {})
        self.assertEqual(result, {0: -0.1, 1: 0.2, 3: -0.2})
        self.assertEqual(self.ex._smoothed_arm_targets, result)
        self.assertEqual(self.ex._arm_contact_limited, {"left": True, "right": True})
        self.ex._contact_health = "ok"
        self.assertEqual(self.ex._limit_arm_targets_at_contacts(requested, {}), requested)
        self.assertFalse(self.ex._arm_contact_limited)

    def test_invalid_measurement_never_becomes_a_nan_holding_target(self):
        self.ex._pause_faulted_articulation = Mock()
        for health in ("ok", "invalidated"):
            with self.subTest(health=health):
                self.ex._contact_health = health
                self.positions[0, 1] = np.nan
                result = self.ex._limit_arm_targets_at_contacts({1: 0.1, 3: -0.1}, {})
                self.assertEqual(result, {})
                self.assertIsNotNone(self.ex._articulation_health_fault)
                self.ex._pause_faulted_articulation.assert_called()
                self.assertFalse(self.ex._smoothed_arm_targets)

    def test_failed_emergency_state_read_pauses_instead_of_reusing_unchecked_target(self):
        self.ex._pause_faulted_articulation = Mock()
        self.robot.get_dof_positions.side_effect = RuntimeError("physics tensor gone")
        self.assertEqual(self.ex._limit_arm_targets_at_contacts({1: 0.1, 3: -0.1}, {}), {})
        self.assertIn("physics tensor gone", self.ex._articulation_health_fault)
        self.ex._pause_faulted_articulation.assert_called_once()

    def test_own_robot_link_contact_is_not_mistaken_for_static_scenery(self):
        contact = GRASP.ContactSample(
            self.ex._g1_prim_path + "/torso_link",
            "index",
            (1.0, 2.0, 3.0),
            (0.0, 0.0, 1.0),
            1.0,
            eligible_for_grasp=False,
        )
        self.ex._latest_hand_contacts = {"right": [contact]}
        requested = {1: -0.1, 3: 0.05}
        self.assertEqual(self.ex._limit_arm_targets_at_contacts(requested, {}), requested)
        self.assertFalse(self.ex._arm_contact_limited)


if __name__ == "__main__":
    unittest.main()
