# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regress physical walking's root ownership, startup, and fail-closed behavior."""

from __future__ import annotations

import math
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from humanoid_test_support import DECOUPLED, G1, USD, Array
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics


def decoupled_fixture():
    policy = DECOUPLED.G1DecoupledPolicy()
    policy._sessions = {mode: Mock() for mode in ("balance", "walk")}
    for session in policy._sessions.values():
        session.run.return_value = [np.zeros((1, 15), dtype=np.float32)]
    return policy


def physical_robot():
    robot = G1.G1TeleopRobot.__new__(G1.G1TeleopRobot)
    robot._physical_locomotion = True
    robot._locomotion = "policy"
    robot._initialized = True
    robot._disable_gravity = False
    robot._velocity_policy = decoupled_fixture()
    robot._walk_policy = robot._velocity_policy
    robot._body_dof_indices = list(range(29))
    robot._velocity_dof_indices = list(range(29))
    robot._velocity_action_dof_indices = list(range(15))
    robot._posture_dof_indices = []
    robot._velocity_lower_limits = np.full(15, -3.0)
    robot._velocity_upper_limits = np.full(15, 3.0)
    robot._default_dof_positions = robot._velocity_policy.initial_positions.copy()
    robot.robot = Mock()
    robot.robot._device = "cpu"
    robot.robot.dof_names = list(robot._velocity_policy.joint_names)
    robot.robot.get_world_poses.return_value = (Array([[0, 0, 0.8]]), Array([[1.0, 0, 0, 0]]))
    robot.robot.get_velocities.return_value = (Array([[0, 0, 0]]), Array([[0, 0, 0]]))
    robot.robot.get_dof_positions.return_value = Array(robot._default_dof_positions.reshape(1, -1))
    robot.robot.get_dof_velocities.return_value = Array(np.zeros((1, 29)))
    robot.robot.get_dof_position_targets.return_value = Array(robot._default_dof_positions.reshape(1, -1))
    robot.robot.get_dof_limits.return_value = (Array([[-3.0] * 15]), Array([[3.0] * 15]))
    robot._author_disabled_gravity = Mock()
    robot._reset_walk_policy_state()
    return robot


class TestPhysicalWalking(unittest.TestCase):
    def test_constructor_rejects_nonpolicy_mode_before_scene_changes(self):
        for mode in ("stationary", "kinematic", "unknown"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "requires"):
                G1.G1TeleopRobot("/World/G1", locomotion=mode, physical_locomotion=True)

    def test_idle_translation_and_turning_never_write_root_or_call_assists(self):
        for command in ([0, 0, 0], [0.2, 0, 0], [0, 0, 0.2]):
            for hz in (100, 200):
                with self.subTest(command=command, hz=hz):
                    robot = physical_robot()
                    robot._update_idle_hold = Mock(side_effect=AssertionError("root pin called"))
                    robot._update_turn_in_place = Mock(side_effect=AssertionError("root pin called"))
                    robot._integrate_base_command = Mock(side_effect=AssertionError("avatar called"))
                    for _ in range(hz):
                        robot.forward(1.0 / hz, command)
                    self.assertEqual(
                        sum(session.run.call_count for session in robot._velocity_policy._sessions.values()), 50
                    )
                    self.assertEqual(robot.robot.set_dof_position_targets.call_count, 50)
                    robot.robot.set_world_poses.assert_not_called()
                    robot.robot.set_velocities.assert_not_called()

    def test_missing_actor_cannot_enter_kinematic_branch(self):
        robot = physical_robot()
        robot._walk_policy = None
        with self.assertRaisesRegex(RuntimeError, "no active policy"):
            robot.forward(0.02, [0.5, 0, 0])
        self.assertIsNotNone(robot.locomotion_fault)
        self.assertEqual(robot._locomotion, "policy")
        robot.robot.set_world_poses.assert_not_called()
        robot.robot.set_velocities.assert_not_called()

    def test_missing_joints_or_unloadable_model_cannot_disable_gravity(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                robot = physical_robot()
                robot._velocity_policy.load = Mock(side_effect=ValueError("bad model"))
                with self.assertRaises(RuntimeError):
                    robot._configure_walk_policy([] if missing else robot.robot.dof_names)
                self.assertIsNotNone(robot.locomotion_fault)
                self.assertEqual(robot._locomotion, "policy")
                self.assertFalse(robot._disable_gravity)
                robot._author_disabled_gravity.assert_not_called()

    def test_invalid_physics_limits_fail_before_policy_can_drive(self):
        for lower, upper in (([-math.inf] * 15, [1.0] * 15), ([1.0] * 15, [-1.0] * 15), ([-1.0], [1.0])):
            robot = physical_robot()
            robot.robot.get_dof_limits.return_value = (Array([lower]), Array([upper]))
            with self.assertRaisesRegex(RuntimeError, "joint limits"):
                robot._configure_walk_policy(robot.robot.dof_names)
            self.assertIsNotNone(robot.locomotion_fault)
            robot._author_disabled_gravity.assert_not_called()

    def test_runtime_fault_stays_latched_until_explicit_reset(self):
        for bad in (np.full((1, 15), math.nan), np.zeros((1, 14))):
            robot = physical_robot()
            session = robot._velocity_policy._sessions["walk"]
            session.run.return_value = [bad]
            with self.assertRaisesRegex(RuntimeError, "policy step failed"):
                robot.forward(0.02, [0.1, 0, 0])
            robot.robot.set_dof_position_targets.assert_not_called()
            session.run.return_value = [np.zeros((1, 15))]
            with self.assertRaises(RuntimeError):
                robot.forward(0.02, [0.1, 0, 0])
            self.assertEqual(session.run.call_count, 1)
            robot._reset_walk_policy_state()
            robot.forward(0.02, [0.1, 0, 0])
            self.assertIsNone(robot.locomotion_fault)
            self.assertEqual(session.run.call_count, 2)

    def test_base_measurements_use_body_axes_and_gravity_without_yaw_contamination(self):
        robot = physical_robot()
        robot.robot.get_world_poses.return_value = (
            Array([[0, 0, 0.8]]),
            Array([[math.sqrt(0.5), 0, 0, math.sqrt(0.5)]]),
        )
        robot.robot.get_velocities.return_value = (Array([[0, 0, 0]]), Array([[1.0, 2.0, 3.0]]))
        robot.forward(0.02, [0, 0, 0])
        frame = robot._velocity_policy.last_observation[0, -86:]
        np.testing.assert_allclose(frame[7:10], [1.0, -0.5, 1.5])
        np.testing.assert_allclose(frame[10:13], [0, 0, -1], atol=1e-7)

    def test_invalid_orientation_fails_before_output(self):
        for orientation in ([0, 0, 0, 0], [math.nan, 0, 0, 1]):
            robot = physical_robot()
            robot.robot.get_world_poses.return_value = (Array([[0, 0, 0.8]]), Array([orientation]))
            with self.assertRaisesRegex(RuntimeError, "orientation"):
                robot.forward(0.02, [0, 0, 0])
            for session in robot._velocity_policy._sessions.values():
                session.run.assert_not_called()

    def test_actor_targets_remain_within_live_limits_and_never_target_fingers(self):
        robot = physical_robot()
        robot._velocity_policy._sessions["balance"].run.return_value = [np.full((1, 15), 1000.0)]
        robot.forward(0.02, [0, 0, 0])
        call = robot.robot.set_dof_position_targets.call_args
        self.assertEqual(call.kwargs["dof_indices"], list(range(15)))
        np.testing.assert_array_equal(call.args[0], np.full((1, 15), 3.0))

    def test_actuator_gains_and_armatures_use_pinned_metadata(self):
        robot = physical_robot()
        robot._apply_body_joint_gains(robot.robot.dof_names)
        call = robot.robot.set_dof_gains.call_args.kwargs
        np.testing.assert_array_equal(call["stiffnesses"][0], robot._velocity_policy.stiffness)
        np.testing.assert_array_equal(call["dampings"][0], robot._velocity_policy.damping)
        self.assertEqual(call["dof_indices"], list(range(29)))
        np.testing.assert_allclose(robot.robot.set_dof_armatures.call_args.args[0], np.full((1, 29), 0.01))


class TestPhysicalWalkingStartup(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)
        UsdGeom.Xform.Define(self.stage, "/World/G1")
        self.robot = physical_robot()
        self.robot._prim_path = "/World/G1"
        self.names = [*self.robot._velocity_policy.joint_names, "L_index_proximal_joint", "L_index_intermediate_joint"]
        self.joints = {}
        for name in self.names:
            prim = UsdPhysics.RevoluteJoint.Define(self.stage, f"/World/G1/joints/{name}").GetPrim()
            state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
            state.CreatePositionAttr().Set(75.0)
            state.CreateVelocityAttr().Set(3.0)
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateStiffnessAttr().Set(9.0)
            drive.CreateTargetPositionAttr().Set(75.0)
            self.joints[name] = prim

    def test_physics_start_matches_metadata_pose_actuators_and_leaves_fingers_untouched(self):
        self.robot._author_physical_walk_pose()
        policy = self.robot._velocity_policy
        defaults = dict(zip(policy.joint_names, policy.initial_positions))
        for name, prim in self.joints.items():
            state = PhysxSchema.JointStateAPI(prim, "angular")
            finger = name.startswith("L_index")
            target = 75.0 if finger else math.degrees(defaults[name])
            self.assertAlmostEqual(state.GetPositionAttr().Get(), target, places=5)
            self.assertEqual(state.GetVelocityAttr().Get(), 3.0 if finger else 0.0)
            drive = UsdPhysics.DriveAPI(prim, "angular")
            self.assertAlmostEqual(drive.GetTargetPositionAttr().Get(), target, places=5)
            if finger:
                self.assertEqual(drive.GetStiffnessAttr().Get(), 9.0)
                self.assertFalse(prim.HasAPI(PhysxSchema.PhysxJointAPI))
            else:
                i = policy.joint_names.index(name)
                self.assertAlmostEqual(drive.GetStiffnessAttr().Get(), policy.stiffness[i] * math.pi / 180, places=5)
                self.assertAlmostEqual(PhysxSchema.PhysxJointAPI(prim).GetArmatureAttr().Get(), 0.01)

    def test_legacy_modes_are_unchanged(self):
        original = self.stage.GetRootLayer().ExportToString()
        self.robot._physical_locomotion = False
        for mode in ("stationary", "policy", "kinematic"):
            self.robot._locomotion = mode
            self.robot._author_physical_walk_pose()
            self.assertEqual(self.stage.GetRootLayer().ExportToString(), original)

    def test_registered_reset_state_matches_metadata_and_keeps_fingers(self):
        self.robot.robot.dof_names = self.names
        self.robot.robot.get_dof_positions.return_value = Array([[0.3] * len(self.names)])
        self.robot._spawn_position = [0, 0, 0.8]
        self.robot._spawn_orientation = [1, 0, 0, 0]
        self.robot._spawn_yaw = 0.0
        self.robot._configure_walk_policy = Mock()
        self.robot._apply_body_joint_gains = Mock()
        self.robot._configure_finger_dofs = Mock()
        self.robot._apply_articulation_properties = Mock()
        self.robot.initialize()
        positions = self.robot.robot.set_default_state.call_args.kwargs["dof_positions"][0]
        np.testing.assert_array_equal(positions[:29], self.robot._velocity_policy.initial_positions)
        np.testing.assert_allclose(positions[-2:], [0.3, 0.3])
        self.robot.robot.set_link_enabled_gravities.assert_not_called()

    def test_constructor_authors_state_before_creating_articulation(self):
        def create_view(**kwargs):
            elbow = PhysxSchema.JointStateAPI(self.joints["left_elbow_joint"], "angular")
            self.assertAlmostEqual(elbow.GetPositionAttr().Get(), 0.0, places=5)
            return Mock()

        with (
            patch.object(G1, "get_assets_root_path", return_value="/unused", create=True),
            patch.object(G1, "get_prim_at_path", side_effect=self.stage.GetPrimAtPath, create=True),
            patch.object(G1, "Articulation", side_effect=create_view, create=True),
            patch.object(G1.G1TeleopRobot, "_select_variants"),
            patch.object(G1.G1TeleopRobot, "_ensure_physx_articulation_api"),
            patch.object(G1.G1TeleopRobot, "_author_disabled_gravity") as gravity,
            patch.object(G1.G1TeleopRobot, "_configure_stationary_anchor"),
            patch.object(G1, "configure_decoupled_foot_contacts"),
            patch.dict(G1.os.environ, {}, clear=True),
        ):
            robot = G1.G1TeleopRobot("/World/G1", locomotion="policy", physical_locomotion=True)
        gravity.assert_called_once_with(False)
        self.assertFalse(robot._initialized)
        self.assertIsNone(robot.locomotion_fault)
        self.assertEqual(robot.walking_policy_name, "decoupled")


if __name__ == "__main__":
    unittest.main()
