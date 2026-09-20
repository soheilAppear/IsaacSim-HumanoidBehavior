# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regress optional arm gravity indexing, ownership, saturation and failure handling."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
from humanoid_test_support import G1, Array
from test_g1_explicit_body_pd_controls import explicit_robot


def gravity_robot(*, fixed_base=False):
    """Use deliberately reordered native DOFs with fingers bracketing the body."""
    robot = explicit_robot()
    robot._arm_gravity_compensation = True
    robot.robot.num_dofs = len(robot.robot.dof_names)
    view = robot.robot._physics_articulation_view
    view.shared_metatype.fixed_base = fixed_base
    offset = 0 if fixed_base else 6
    # Large root/finger sentinels would saturate the wrong motors if indexed as DOFs.
    forces = np.full((1, robot.robot.num_dofs + offset), 1000.0)
    for slot, index in enumerate(robot._velocity_dof_indices):
        forces[0, index + offset] = (slot - 22) * 0.15
    view.get_gravity_compensation_forces.return_value = Array(forces)
    return robot, forces


class TestArmGravityCompensation(unittest.TestCase):
    def test_reordered_fixed_and_floating_tensors_add_only_named_arm_torques(self):
        for fixed in (False, True):
            with self.subTest(fixed_base=fixed):
                robot, _ = gravity_robot(fixed_base=fixed)
                policy = robot._velocity_policy
                target = np.linspace(-0.025, 0.025, 29).reshape(1, -1)
                robot.robot.get_dof_position_targets.return_value = Array(target)
                baseline = (policy.stiffness * target).astype(np.float32)
                robot.apply_body_efforts()
                call = robot.robot.set_dof_efforts.call_args
                actual = call.args[0]
                np.testing.assert_array_equal(actual[:, :15], baseline[:, :15])
                np.testing.assert_allclose(actual[0, 15:], baseline[0, 15:] + (np.arange(15, 29) - 22) * 0.15)
                self.assertEqual(call.kwargs["dof_indices"], list(range(29, 0, -1)))
                self.assertEqual(actual.dtype, np.float32)
                robot.robot.set_world_poses.assert_not_called()
                robot.robot.set_velocities.assert_not_called()
                robot.robot.set_dof_position_targets.assert_not_called()
                for session in policy._sessions.values():
                    session.run.assert_not_called()

    def test_combined_pd_and_gravity_obey_original_positive_and_negative_motor_limits(self):
        robot, forces = gravity_robot()
        policy = robot._velocity_policy
        target = np.zeros((1, 29))
        for slot, sign in ((20, 1), (21, -1), (27, -1), (28, 1)):
            target[0, slot] = sign * 4.5 / policy.stiffness[slot]
            forces[0, robot._velocity_dof_indices[slot] + 6] = sign * 1.0
        robot.robot._physics_articulation_view.get_gravity_compensation_forces.return_value = Array(forces)
        robot.robot.get_dof_position_targets.return_value = Array(target)
        robot.apply_body_efforts()
        actual = robot.robot.set_dof_efforts.call_args.args[0]
        np.testing.assert_array_equal(actual[0, [20, 21, 27, 28]], [5.0, -5.0, -5.0, 5.0])
        np.testing.assert_array_equal(actual[:, :15], np.zeros((1, 15)))
        np.testing.assert_array_equal(policy.effort_limits[[20, 21, 27, 28]], [5.0] * 4)

    def test_disabled_option_is_identical_pd_and_never_reads_gravity(self):
        robot, _ = gravity_robot()
        robot._arm_gravity_compensation = False
        target = np.linspace(-0.1, 0.1, 29).reshape(1, -1)
        robot.robot.get_dof_position_targets.return_value = Array(target)
        robot.apply_body_efforts()
        expected = np.clip(
            target * robot._velocity_policy.stiffness,
            -robot._velocity_policy.effort_limits,
            robot._velocity_policy.effort_limits,
        ).astype(np.float32)
        np.testing.assert_array_equal(robot.robot.set_dof_efforts.call_args.args[0], expected)
        robot.robot._physics_articulation_view.get_gravity_compensation_forces.assert_not_called()
        self.assertFalse(robot.arm_gravity_compensation_enabled)

    def test_every_effort_step_reads_fresh_gravity_without_accumulation(self):
        robot, forces = gravity_robot()
        view = robot.robot._physics_articulation_view
        robot.apply_body_efforts()
        first = robot.robot.set_dof_efforts.call_args.args[0].copy()
        changed = forces.copy()
        changed[0, robot._velocity_dof_indices[15] + 6] += 0.4
        view.get_gravity_compensation_forces.return_value = Array(changed)
        robot.apply_body_efforts()
        second = robot.robot.set_dof_efforts.call_args.args[0].copy()
        robot.apply_body_efforts()
        np.testing.assert_array_equal(robot.robot.set_dof_efforts.call_args.args[0], second)
        self.assertAlmostEqual(float(second[0, 15] - first[0, 15]), 0.4, places=6)
        self.assertEqual(view.get_gravity_compensation_forces.call_count, 3)

    def test_bad_gravity_zeroes_efforts_and_latches_until_explicit_reset(self):
        for bad in (np.zeros((1, 31)), np.zeros((2, 37)), np.full((1, 37), np.nan), np.full((1, 37), np.inf)):
            with self.subTest(shape=bad.shape, nonfinite=not np.isfinite(bad).all()):
                robot, valid = gravity_robot()
                reader = robot.robot._physics_articulation_view.get_gravity_compensation_forces
                reader.return_value = Array(bad)
                with self.assertRaisesRegex(RuntimeError, "invalid native gravity tensor"):
                    robot.apply_body_efforts()
                np.testing.assert_array_equal(robot.robot.set_dof_efforts.call_args.args[0], np.zeros((1, 29)))
                reader.return_value = Array(valid)
                with self.assertRaises(RuntimeError):
                    robot.apply_body_efforts()
                self.assertEqual(reader.call_count, 1)
                robot._reset_walk_policy_state()
                robot.apply_body_efforts()
                self.assertIsNone(robot.locomotion_fault)
                self.assertEqual(reader.call_count, 2)

    def test_invalid_mapping_or_unknown_base_layout_cannot_command_the_wrong_motor(self):
        def reorder_native(robot):
            robot.robot.dof_names[1], robot.robot.dof_names[2] = robot.robot.dof_names[2], robot.robot.dof_names[1]

        def duplicate_native(robot):
            robot.robot.dof_names[1] = robot.robot.dof_names[2]

        def replace_arm_with_finger(robot):
            names = list(robot._velocity_policy.joint_names)
            names[15] = robot.robot.dof_names[0]
            robot._velocity_policy.joint_names = tuple(names)
            robot._velocity_dof_indices[15] = 0

        for corrupt in (
            reorder_native,
            duplicate_native,
            replace_arm_with_finger,
            lambda robot: setattr(robot.robot._physics_articulation_view.shared_metatype, "fixed_base", None),
            lambda robot: setattr(robot.robot, "num_dofs", 30),
        ):
            with self.subTest(corruption=corrupt):
                robot, _ = gravity_robot()
                corrupt(robot)
                with self.assertRaisesRegex(RuntimeError, "Arm gravity compensation"):
                    robot.apply_body_efforts()
                np.testing.assert_array_equal(robot.robot.set_dof_efforts.call_args.args[0], np.zeros((1, 29)))
                robot.robot._physics_articulation_view.get_gravity_compensation_forces.assert_not_called()

    def test_backend_failure_cannot_reuse_previous_gravity_or_torque(self):
        robot, _ = gravity_robot()
        robot.apply_body_efforts()
        robot.robot._physics_articulation_view.get_gravity_compensation_forces.side_effect = RuntimeError(
            "expired view"
        )
        with self.assertRaisesRegex(RuntimeError, "expired view"):
            robot.apply_body_efforts()
        np.testing.assert_array_equal(robot.robot.set_dof_efforts.call_args.args[0], np.zeros((1, 29)))

    def test_option_requires_physical_mode_and_retains_tensor_lifecycle_guards(self):
        for mode in ("stationary", "kinematic", "policy"):
            with self.assertRaisesRegex(ValueError, "requires physical_locomotion"):
                G1.G1TeleopRobot("/World/G1", locomotion=mode, arm_gravity_compensation=True)
        for initialized, valid in ((False, True), (True, False)):
            robot, _ = gravity_robot()
            robot._initialized = initialized
            robot.robot.is_physics_tensor_entity_valid.return_value = valid
            robot.invalidate_physics_state = Mock()
            robot.apply_body_efforts()
            robot.robot.set_dof_efforts.assert_not_called()
            robot.robot._physics_articulation_view.get_gravity_compensation_forces.assert_not_called()


if __name__ == "__main__":
    unittest.main()
