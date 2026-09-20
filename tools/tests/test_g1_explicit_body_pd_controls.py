# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify decoupled motor PD, torque saturation, ownership, and reset behavior."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np
from humanoid_test_support import G1, Array
from test_g1_decoupled_policy_controls import decoupled_fixture
from test_g1_physical_policy_controls import physical_robot


def explicit_robot():
    robot = physical_robot()
    policy = decoupled_fixture()
    policy.load = Mock()
    robot._velocity_policy = policy
    robot.robot.dof_names = ["L_index_proximal_joint", *reversed(policy.joint_names), "R_index_proximal_joint"]
    robot.robot.get_dof_limits.return_value = (Array([[-3.0] * 15]), Array([[3.0] * 15]))
    robot._configure_velocity_policy(robot.robot.dof_names)
    robot.robot.get_dof_positions.return_value = Array(np.zeros((1, 29)))
    robot.robot.get_dof_velocities.return_value = Array(np.zeros((1, 29)))
    robot.robot.get_dof_position_targets.return_value = Array(np.zeros((1, 29)))
    return robot


class TestExplicitBodyPD(unittest.TestCase):
    def test_only_decoupled_authors_foot_contacts_before_articulation_construction(self):
        for physical, selection in ((True, "decoupled"), (False, "decoupled")):
            with self.subTest(physical=physical, selection=selection):
                order = []
                with (
                    patch.dict(G1.os.environ, {"G1_WALK_POLICY": selection}),
                    patch.object(G1, "get_assets_root_path", return_value="/unused", create=True),
                    patch.object(G1, "get_prim_at_path", return_value=Mock(), create=True),
                    patch.object(G1, "Articulation", side_effect=lambda **_: order.append("articulation"), create=True),
                    patch.object(
                        G1,
                        "configure_decoupled_foot_contacts",
                        side_effect=lambda *_: order.append("feet"),
                        create=True,
                    ),
                    patch.object(G1.G1TeleopRobot, "_select_variants"),
                    patch.object(G1.G1TeleopRobot, "_ensure_physx_articulation_api"),
                    patch.object(G1.G1TeleopRobot, "_author_disabled_gravity"),
                    patch.object(G1.G1TeleopRobot, "_set_spawn_transform"),
                    patch.object(G1.G1TeleopRobot, "_configure_stationary_anchor"),
                    patch.object(G1.G1TeleopRobot, "_author_stationary_arm_pose"),
                    patch.object(
                        G1.G1TeleopRobot, "_author_physical_walk_pose", side_effect=lambda: order.append("pose")
                    ),
                ):
                    G1.G1TeleopRobot(
                        "/World/G1", locomotion="policy" if physical else "stationary", physical_locomotion=physical
                    )
                expected = (
                    ["pose", "feet", "articulation"]
                    if physical and selection == "decoupled"
                    else ["pose", "articulation"]
                )
                self.assertEqual(order, expected)

    def test_position_and_velocity_terms_are_combined_then_motor_limited(self):
        robot = explicit_robot()
        policy = robot._velocity_policy
        q = np.linspace(-0.2, 0.2, 29).reshape(1, -1)
        target = np.linspace(3, -3, 29).reshape(1, -1)
        velocity = np.linspace(-2, 2, 29).reshape(1, -1)
        robot.robot.get_dof_positions.return_value = Array(q)
        robot.robot.get_dof_position_targets.return_value = Array(target)
        robot.robot.get_dof_velocities.return_value = Array(velocity)
        robot.apply_body_efforts()
        call = robot.robot.set_dof_efforts.call_args
        expected = np.clip(
            policy.stiffness * (target - q) - policy.damping * velocity,
            -policy.effort_limits,
            policy.effort_limits,
        )
        np.testing.assert_allclose(call.args[0], expected, atol=1e-5)
        self.assertEqual(call.kwargs["dof_indices"], list(range(29, 0, -1)))
        self.assertEqual(call.args[0].dtype, np.float32)
        self.assertEqual(call.args[0].shape, (1, 29))
        robot.robot.set_world_poses.assert_not_called()
        robot.robot.set_velocities.assert_not_called()
        robot.robot.set_dof_position_targets.assert_not_called()

    def test_second_call_uses_updated_arm_targets_without_accumulating_effort(self):
        robot = explicit_robot()
        robot.apply_body_efforts()
        target = np.zeros((1, 29))
        target[0, 15] = 0.1
        robot.robot.get_dof_position_targets.return_value = Array(target)
        robot.apply_body_efforts()
        expected = np.zeros((1, 29))
        expected[0, 15] = 10.0
        np.testing.assert_allclose(robot.robot.set_dof_efforts.call_args.args[0], expected)
        robot.apply_body_efforts()
        np.testing.assert_allclose(robot.robot.set_dof_efforts.call_args.args[0], expected)

    def test_efforts_refresh_at_physics_rate_between_actor_steps(self):
        robot = explicit_robot()
        for _ in range(200):
            robot.forward(0.005, [0, 0, 0])
        self.assertEqual(robot._velocity_policy._sessions["balance"].run.call_count, 50)
        self.assertEqual(robot.robot.set_dof_efforts.call_count, 200)
        self.assertEqual(robot.robot.set_dof_position_targets.call_count, 50)

    def test_deferred_efforts_use_latest_arm_targets_once_per_physics_step(self):
        robot = explicit_robot()
        for tick in range(200):
            robot.forward(0.005, [0, 0, 0], defer_body_efforts=True)
            # The caller owns the one motor update after this tick's arm targets.
            self.assertEqual(robot.robot.set_dof_efforts.call_count, tick)
            target = np.zeros((1, 29))
            target[0, 15] = 0.1 if tick % 2 else -0.1
            robot.robot.get_dof_position_targets.return_value = Array(target)
            robot.apply_body_efforts()
            expected = np.zeros((1, 29))
            expected[0, 15] = 10.0 if tick % 2 else -10.0
            np.testing.assert_allclose(robot.robot.set_dof_efforts.call_args.args[0], expected)
        self.assertEqual(robot._velocity_policy._sessions["balance"].run.call_count, 50)
        self.assertEqual(robot.robot.set_dof_efforts.call_count, 200)
        robot.robot.set_world_poses.assert_not_called()
        robot.robot.set_velocities.assert_not_called()

    def test_deferred_final_update_clears_bad_arm_target_and_latches_fault(self):
        robot = explicit_robot()
        robot.forward(0.005, [0, 0, 0], defer_body_efforts=True)
        target = np.zeros((1, 29))
        target[0, 15] = np.nan
        robot.robot.get_dof_position_targets.return_value = Array(target)
        with self.assertRaisesRegex(RuntimeError, "body effort update failed"):
            robot.apply_body_efforts()
        np.testing.assert_array_equal(robot.robot.set_dof_efforts.call_args.args[0], np.zeros((1, 29)))
        with self.assertRaisesRegex(RuntimeError, "body effort update failed"):
            robot.forward(0.005, [0, 0, 0], defer_body_efforts=True)
        self.assertEqual(robot.robot.set_dof_efforts.call_count, 1)

    def test_deferred_forward_preserves_initialization_and_tensor_validity_guards(self):
        for initialized, valid in ((False, True), (True, False)):
            with self.subTest(initialized=initialized, valid=valid):
                robot = explicit_robot()
                robot._initialized = initialized
                robot.robot.is_physics_tensor_entity_valid.return_value = valid
                robot.forward(0.005, [0, 0, 0], defer_body_efforts=True)
                self.assertFalse(robot._initialized)
                self.assertEqual(robot._velocity_policy._sessions["balance"].run.call_count, 0)
                robot.robot.set_dof_efforts.assert_not_called()
                robot.robot.set_dof_position_targets.assert_not_called()

    def test_effort_mode_disables_implicit_body_drives_only(self):
        robot = explicit_robot()
        robot._apply_body_joint_gains(robot.robot.dof_names)
        robot.robot.switch_dof_control_mode.assert_called_once_with("effort", dof_indices=list(range(29, 0, -1)))
        np.testing.assert_array_equal(robot.robot.set_dof_efforts.call_args.args[0], np.zeros((1, 29)))
        np.testing.assert_array_equal(
            robot.robot.set_dof_max_efforts.call_args.args[0][0], robot._velocity_policy.effort_limits
        )

    def test_bad_state_clears_motor_efforts_and_latches_fault_until_reset(self):
        for field, bad in (
            ("get_dof_position_targets", np.full((1, 29), np.nan)),
            ("get_dof_positions", np.full((1, 29), np.inf)),
            ("get_dof_velocities", np.zeros((1, 28))),
        ):
            with self.subTest(field=field):
                robot = explicit_robot()
                getattr(robot.robot, field).return_value = Array(bad)
                with self.assertRaisesRegex(RuntimeError, "body effort update failed"):
                    robot.apply_body_efforts()
                np.testing.assert_array_equal(robot.robot.set_dof_efforts.call_args.args[0], np.zeros((1, 29)))
                getattr(robot.robot, field).return_value = Array(np.zeros((1, 29)))
                with self.assertRaises(RuntimeError):
                    robot.apply_body_efforts()
                self.assertEqual(robot.robot.set_dof_efforts.call_count, 1)
                robot._reset_walk_policy_state()
                robot.apply_body_efforts()
                self.assertEqual(robot.robot.set_dof_efforts.call_count, 2)

    def test_reset_clears_previous_body_torques(self):
        robot = explicit_robot()
        robot._spawn_position = [0, 0, 0.8]
        robot._spawn_yaw = 0.0
        robot.post_reset()
        robot.robot.reset_to_default_state.assert_called_once_with()
        np.testing.assert_array_equal(robot.robot.set_dof_efforts.call_args.args[0], np.zeros((1, 29)))
        self.assertIsNone(robot.locomotion_fault)

    def test_initialize_restores_source_pose_and_effort_mode(self):
        robot = explicit_robot()
        robot._spawn_position = [0, 0, 0.8]
        robot._spawn_orientation = [1, 0, 0, 0]
        robot._spawn_yaw = 0.0
        robot.robot.get_dof_positions.return_value = Array(np.full((1, 31), 0.3))
        robot._configure_walk_policy = Mock()
        robot._configure_finger_dofs = Mock()
        robot._apply_articulation_properties = Mock()
        robot.initialize()
        self.assertEqual(robot.robot.switch_dof_control_mode.call_args.args[0], "effort")
        call = robot.robot.set_dof_position_targets.call_args
        self.assertEqual(call.kwargs["dof_indices"], list(range(29, 0, -1)))
        np.testing.assert_array_equal(call.args[0][0], robot._velocity_policy.initial_positions)
        defaults = robot.robot.set_default_state.call_args.kwargs["dof_positions"][0]
        np.testing.assert_array_equal(defaults[robot._velocity_dof_indices], robot._velocity_policy.initial_positions)
        np.testing.assert_allclose(defaults[[0, 30]], [0.3, 0.3])

    def test_legacy_actor_and_stationary_modes_never_write_explicit_efforts(self):
        robot = physical_robot()
        robot._physical_locomotion = False
        robot._locomotion = "policy"
        self.assertFalse(robot.uses_explicit_body_efforts)
        robot.apply_body_efforts()
        robot.robot.set_dof_efforts.assert_not_called()
        robot = explicit_robot()
        robot._physical_locomotion = False
        robot._locomotion = "stationary"
        self.assertFalse(robot.uses_explicit_body_efforts)
        robot.apply_body_efforts()
        robot.robot.set_dof_efforts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
