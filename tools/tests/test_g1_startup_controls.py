# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep stationary hands out of tabletop props before the first physics solve."""

from __future__ import annotations

import math
import types
import unittest
from unittest.mock import Mock, patch

from humanoid_test_support import G1, USD
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics


class TestG1StartingArmPose(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)
        UsdGeom.Xform.Define(self.stage, "/World/G1")
        self.robot = G1.G1TeleopRobot.__new__(G1.G1TeleopRobot)
        self.robot._prim_path = "/World/G1"
        self.robot._locomotion = "stationary"
        self.arms = {}
        for side in ("left", "right"):
            for role in self.robot.WALK_ARM_JOINT_ORDER:
                name = f"{side}_{role}_joint"
                prim = UsdPhysics.RevoluteJoint.Define(self.stage, f"/World/G1/joints/{name}").GetPrim()
                state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
                state.CreatePositionAttr().Set(75.0)
                state.CreateVelocityAttr().Set(3.0)
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTargetPositionAttr().Set(75.0)
                drive.CreateTargetVelocityAttr().Set(3.0)
                drive.CreateStiffnessAttr().Set(81.0)
                drive.CreateDampingAttr().Set(4.5)
                drive.CreateMaxForceAttr().Set(13.0)
                drive.CreateTypeAttr().Set("force")
                self.arms[name] = prim

    def test_all_fourteen_arm_states_use_degree_defaults_and_zero_velocity(self):
        self.robot._author_stationary_arm_pose()
        self.assertEqual(len(self.arms), 14)
        for name, prim in self.arms.items():
            with self.subTest(joint=name):
                expected = math.degrees(self.robot.DEFAULT_STANDING_POSE.get(name, 0.0))
                state = PhysxSchema.JointStateAPI(prim, "angular")
                self.assertAlmostEqual(state.GetPositionAttr().Get(), expected, places=5)
                self.assertEqual(state.GetVelocityAttr().Get(), 0.0)
                drive = UsdPhysics.DriveAPI(prim, "angular")
                self.assertAlmostEqual(drive.GetTargetPositionAttr().Get(), expected, places=5)
                self.assertEqual(drive.GetTargetVelocityAttr().Get(), 0.0)
                self.assertEqual(drive.GetStiffnessAttr().Get(), 81.0)
                self.assertEqual(drive.GetDampingAttr().Get(), 4.5)
                self.assertEqual(drive.GetMaxForceAttr().Get(), 13.0)
                self.assertEqual(drive.GetTypeAttr().Get(), "force")

    def test_constructor_authors_state_before_articulation_view_exists(self):
        def create_view(**kwargs):
            state = PhysxSchema.JointStateAPI(self.arms["left_elbow_joint"], "angular")
            self.assertAlmostEqual(state.GetPositionAttr().Get(), math.degrees(0.6), places=5)
            return Mock()

        with (
            patch.object(G1, "get_assets_root_path", return_value="/unused", create=True),
            patch.object(G1, "get_prim_at_path", side_effect=self.stage.GetPrimAtPath, create=True),
            patch.object(G1, "Articulation", side_effect=create_view, create=True),
            patch.object(G1.G1TeleopRobot, "_select_variants"),
            patch.object(G1.G1TeleopRobot, "_ensure_physx_articulation_api"),
            patch.object(G1.G1TeleopRobot, "_author_disabled_gravity"),
            patch.object(G1.G1TeleopRobot, "_configure_stationary_anchor"),
        ):
            robot = G1.G1TeleopRobot("/World/G1", locomotion="stationary")
        self.assertFalse(robot._initialized)
        robot.robot.set_dof_positions.assert_not_called()

    def test_finger_leg_waist_unknown_joint_and_other_robot_states_are_untouched(self):
        untouched = []
        for path in (
            "/World/G1/joints/L_index_proximal_joint",
            "/World/G1/joints/L_index_intermediate_joint",
            "/World/G1/joints/left_hip_pitch_joint",
            "/World/G1/joints/waist_yaw_joint",
            "/World/G1/joints/left_elbow_custom_joint",
            "/World/OtherRobot/left_elbow_joint",
        ):
            prim = UsdPhysics.RevoluteJoint.Define(self.stage, path).GetPrim()
            state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
            state.CreatePositionAttr().Set(19.0)
            state.CreateVelocityAttr().Set(2.0)
            untouched.append(prim)
        mimic = PhysxSchema.PhysxMimicJointAPI.Apply(untouched[1], "rotZ")
        mimic.CreateGearingAttr().Set(-1.0)
        self.robot._author_stationary_arm_pose()
        for prim in untouched:
            with self.subTest(joint=str(prim.GetPath())):
                state = PhysxSchema.JointStateAPI(prim, "angular")
                self.assertEqual(state.GetPositionAttr().Get(), 19.0)
                self.assertEqual(state.GetVelocityAttr().Get(), 2.0)
                self.assertFalse(prim.HasAPI(UsdPhysics.DriveAPI, "angular"))
        self.assertEqual(mimic.GetGearingAttr().Get(), -1.0)

    def test_only_stationary_mode_authors_a_starting_pose(self):
        original = self.stage.GetRootLayer().ExportToString()
        for mode in ("policy", "kinematic"):
            with self.subTest(mode=mode):
                self.robot._locomotion = mode
                self.robot._author_stationary_arm_pose()
                self.assertEqual(self.stage.GetRootLayer().ExportToString(), original)

    def test_missing_state_is_added_without_adding_a_drive(self):
        prim = self.arms["right_wrist_yaw_joint"]
        prim.RemoveAPI(PhysxSchema.JointStateAPI, "angular")
        prim.RemoveProperty("state:angular:physics:position")
        prim.RemoveProperty("state:angular:physics:velocity")
        prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
        self.robot._author_stationary_arm_pose()
        self.assertTrue(prim.HasAPI(PhysxSchema.JointStateAPI, "angular"))
        self.assertFalse(prim.HasAPI(UsdPhysics.DriveAPI, "angular"))
        self.assertEqual(PhysxSchema.JointStateAPI(prim, "angular").GetPositionAttr().Get(), 0.0)

    def test_non_revolute_joint_with_matching_name_is_not_modified(self):
        path = str(self.arms["left_wrist_roll_joint"].GetPath())
        self.stage.RemovePrim(path)
        prim = UsdPhysics.FixedJoint.Define(self.stage, path).GetPrim()
        self.robot._author_stationary_arm_pose()
        self.assertFalse(prim.HasAPI(PhysxSchema.JointStateAPI, "angular"))

    def test_authoring_is_idempotent_and_can_restore_a_saved_arm_state(self):
        self.robot._author_stationary_arm_pose()
        initial = self.stage.GetRootLayer().ExportToString()
        self.robot._author_stationary_arm_pose()
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), initial)
        state = PhysxSchema.JointStateAPI(self.arms["left_elbow_joint"], "angular")
        state.GetPositionAttr().Set(91.0)
        self.robot._author_stationary_arm_pose()
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), initial)


if __name__ == "__main__":
    unittest.main()
