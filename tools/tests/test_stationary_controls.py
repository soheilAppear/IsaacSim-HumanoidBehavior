# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stationary manipulation must reject locomotion without disabling hand/XR actions."""

from __future__ import annotations

import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from humanoid_test_support import CARB, G1, HUMANOID, USD
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


class TestStandingAnchor(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)
        UsdGeom.Xform.Define(self.stage, "/World/G1")
        self.pelvis = UsdGeom.Xform.Define(self.stage, "/World/G1/pelvis").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(self.pelvis)
        UsdPhysics.ArticulationRootAPI.Apply(self.pelvis)
        PhysxSchema.PhysxArticulationAPI.Apply(self.pelvis).CreateSolverPositionIterationCountAttr().Set(32)
        self.hand = UsdGeom.Xform.Define(self.stage, "/World/G1/hand").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(self.hand)
        self.wrist = UsdPhysics.RevoluteJoint.Define(self.stage, "/World/G1/wrist")
        self.wrist.CreateBody0Rel().SetTargets([self.pelvis.GetPath()])
        self.wrist.CreateBody1Rel().SetTargets([self.hand.GetPath()])
        self.box = UsdGeom.Xform.Define(self.stage, "/World/Box").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(self.box)

    def construct(self):
        def make_view(**kwargs):
            # Changing root topology after this view exists is too late.
            self.assertTrue(self.stage.GetPrimAtPath("/World/G1/G1_StandingAnchor").IsValid())
            return Mock()

        with (
            patch.object(G1, "get_assets_root_path", return_value="/unused", create=True),
            patch.object(G1, "get_prim_at_path", side_effect=self.stage.GetPrimAtPath, create=True),
            patch.object(G1, "Articulation", side_effect=make_view, create=True),
        ):
            return G1.G1TeleopRobot(
                "/World/G1", position=[2, 3, 0.8], orientation=[0.70710678, 0, 0, 0.70710678], locomotion="stationary"
            )

    def test_anchor_is_fixed_root_at_spawn_and_hand_joint_remains_dynamic(self):
        robot = self.construct()
        anchor = UsdPhysics.FixedJoint.Get(self.stage, "/World/G1/G1_StandingAnchor")
        roots = [prim.GetPath() for prim in self.stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        self.assertEqual(roots, [anchor.GetPath()])
        self.assertEqual(anchor.GetBody0Rel().GetTargets(), [])
        self.assertEqual(anchor.GetBody1Rel().GetTargets(), [self.pelvis.GetPath()])
        self.assertFalse(anchor.GetExcludeFromArticulationAttr().Get())
        self.assertTrue(anchor.GetJointEnabledAttr().Get())
        world = UsdGeom.XformCache().GetLocalToWorldTransform(self.pelvis)
        np.testing.assert_allclose(anchor.GetLocalPos0Attr().Get(), world.ExtractTranslation(), atol=1e-6)
        expected = Gf.Quatf(world.ExtractRotationQuat().GetNormalized())
        self.assertEqual(anchor.GetLocalRot0Attr().Get(), expected)
        self.assertEqual(PhysxSchema.PhysxArticulationAPI(anchor).GetSolverPositionIterationCountAttr().Get(), 32)
        self.assertTrue(robot._disable_gravity)
        self.assertIsNone(robot._walk_policy)
        self.assertTrue(self.wrist.GetJointEnabledAttr().Get())
        self.assertFalse(UsdPhysics.RigidBodyAPI(self.hand).GetKinematicEnabledAttr().Get())
        self.assertTrue(PhysxSchema.PhysxRigidBodyAPI(self.hand).GetDisableGravityAttr().Get())
        self.assertFalse(self.box.HasAPI(PhysxSchema.PhysxRigidBodyAPI))

    def test_anchor_setup_is_repeatable_and_can_restore_original_root(self):
        robot = self.construct()
        before = self.stage.GetRootLayer().ExportToString()
        robot._configure_stationary_anchor()
        self.assertEqual(before, self.stage.GetRootLayer().ExportToString())
        robot._locomotion = "kinematic"
        robot._configure_stationary_anchor()
        self.assertFalse(self.stage.GetPrimAtPath("/World/G1/G1_StandingAnchor").IsValid())
        self.assertTrue(self.pelvis.HasAPI(UsdPhysics.ArticulationRootAPI))
        self.assertEqual(PhysxSchema.PhysxArticulationAPI(self.pelvis).GetSolverPositionIterationCountAttr().Get(), 32)

    def test_stationary_forward_never_calls_actor_or_writes_root_motion(self):
        robot = self.construct()
        robot._initialized = True
        robot._body_dof_indices = [0, 1]
        robot._default_dof_positions = np.array([0.0, 0.6, 0.9])
        robot._walk_policy = Mock()
        robot._integrate_base_command = Mock()
        robot._write_base_pose = Mock()
        for _ in range(100):
            robot.forward(0.01, [10, -10, 10])
        robot._walk_policy.assert_not_called()
        robot._integrate_base_command.assert_not_called()
        robot._write_base_pose.assert_not_called()
        robot.robot.set_world_poses.assert_not_called()
        robot.robot.set_velocities.assert_not_called()
        self.assertEqual(robot._base_position, [2, 3, 0.8])
        target_call = robot.robot.set_dof_position_targets.call_args
        self.assertEqual(target_call.kwargs["dof_indices"], [0, 1])
        np.testing.assert_allclose(target_call.args[0], [[0, 0.6]])


class TestStationaryInputs(unittest.TestCase):
    def test_full_tick_blocks_stale_keyboard_gamepad_and_gait_but_runs_hand_and_gaze(self):
        example = HUMANOID.HumanoidExample()
        self.assertEqual(example._g1_locomotion, "stationary")
        example._physics_ready = True
        example.g1 = types.SimpleNamespace(robot=Mock(), forward=Mock())
        example.g1.robot.is_physics_tensor_entity_valid.return_value = True
        example._base_command = torch.ones(3)
        example._keyboard_command = torch.ones(3)
        example._controller_command = torch.ones(3)
        example._headset_gait_enabled = True
        example._read_xr_controller_axes = Mock(return_value=(1, 1))
        example._read_gamepad_controller_axes = Mock(side_effect=AssertionError("gamepad must be suppressed"))
        example._update_headset_gait_command = Mock(return_value=1)
        example._update_g1_fingers = Mock()
        example._update_g1_arms_from_hand_tracking = Mock()
        example._update_head_camera_view = Mock()
        example._eye_gaze_tracker = Mock()
        example._collect_all_behavioral_data = Mock()
        example.on_physics_step(0.01, None)
        np.testing.assert_allclose(example.g1.forward.call_args.args[1], [0, 0, 0])
        np.testing.assert_allclose(example._controller_command, [0, 0, 0])
        example._read_gamepad_controller_axes.assert_not_called()
        example._update_headset_gait_command.assert_called_once_with(0.01)
        example._update_g1_fingers.assert_called_once()
        example._update_g1_arms_from_hand_tracking.assert_called_once()
        example._eye_gaze_tracker.update.assert_called_once_with(0.01)
        example._collect_all_behavioral_data.assert_called_once()

    def test_stationary_xr_still_handles_drop_recenter_and_camera_once_per_press(self):
        example = HUMANOID.HumanoidExample()
        example._get_xr_input_device = lambda handle: handle
        example._log_xr_input_status_once = Mock()
        example._get_xr_stick_axis = lambda device, axis: 1.0
        example._get_xr_gesture_value = lambda device, name, gesture: 1.0
        example._get_xr_stick_click = lambda device: 1.0
        example._request_xr_recenter = Mock()
        example._drop_everything = Mock()
        example._cycle_xr_camera_mode = Mock()
        for _ in range(2):
            self.assertEqual(example._read_xr_controller_axes(), (0.0, 0.0))
            self.assertEqual(example._latest_stick_lateral, 0.0)
        example._request_xr_recenter.assert_called_once()
        example._drop_everything.assert_called_once()
        example._cycle_xr_camera_mode.assert_called_once()

    def test_keyboard_cannot_leave_motion_latched_in_stationary_mode(self):
        example = HUMANOID.HumanoidExample()
        example._keyboard_command = torch.ones(3)
        example._pressed_keys.add("UP")
        event = types.SimpleNamespace(input="RIGHT", type=CARB.input.KeyboardEventType.KEY_PRESS)
        example._sub_keyboard_event(event)
        self.assertEqual(example._pressed_keys, set())
        np.testing.assert_allclose(example._keyboard_command, [0, 0, 0])
