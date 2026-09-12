# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify anatomical palm alignment separately from controller clutch rotation."""

from __future__ import annotations

import math
import types
import unittest
from unittest.mock import Mock

import numpy as np
from humanoid_test_support import HUMANOID, USD, Array, Device, pose
from pxr import Gf, Usd, UsdGeom, UsdPhysics


class PalmDevice:
    """Provide position-valid optical knuckles without arbitrary pose axes."""

    def __init__(self, side: str, transform: Gf.Matrix4d | None = None) -> None:
        radial = -1.0 if side == "left" else 1.0
        self.positions = {
            "wrist": Gf.Vec3d(0, 0, 0),
            "middle_proximal": Gf.Vec3d(0.1, 0, 0),
            "index_proximal": Gf.Vec3d(0.1, 0.025 * radial, 0),
            "little_proximal": Gf.Vec3d(0.1, -0.025 * radial, 0),
        }
        self.transform = transform if transform is not None else Gf.Matrix4d(1)
        self.invalid = set()

    def get_hand_tracking_data_source(self) -> str:
        return "hand"

    def get_pose_names(self) -> list[str]:
        return list(self.positions)

    def get_virtual_world_pose_desc(self, name: str = "") -> types.SimpleNamespace:
        if name not in self.positions:
            return types.SimpleNamespace(pose_matrix=pose(), validity_flags=0)
        position = self.transform.Transform(self.positions[name])
        return types.SimpleNamespace(pose_matrix=pose(position), validity_flags=0 if name in self.invalid else 2)


class TestArmOrientationControls(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)
        self.example = HUMANOID.HumanoidExample()
        self.example._g1_prim_path = "/World/G1"
        self.example._robot_palm_local_frames = {}
        self.example._robot_palm_local_centers = {}
        for side in ("left", "right"):
            letter = "L" if side == "left" else "R"
            hand_path = f"/World/G1/{side}_hand/{letter}_hand_base_link"
            UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(self.stage, hand_path).GetPrim())
            # The supplied Inspire asset extends along +X with the knuckles spread
            # along Z. Its neutral palm is vertical; the test operator palm is flat.
            for role, z in (("middle", 0.0), ("index", 0.025), ("pinky", -0.025)):
                path = f"/World/G1/{side}_hand/joints/{letter}_{role}_proximal_joint"
                joint = UsdPhysics.RevoluteJoint.Define(self.stage, path)
                joint.CreateBody0Rel().SetTargets([hand_path])
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.12, 0.0, z))

    def test_first_flat_optical_hand_corrects_vertical_robot_palm_for_both_sides(self) -> None:
        ex = self.example
        ex._get_hand_link_prim = Mock(side_effect=AssertionError("Optical axes cannot use an initial wrist anchor"))
        for side in ("left", "right"):
            with self.subTest(side=side):
                device = PalmDevice(side)
                target = ex._compute_arm_target_orientation(side, pose(yaw=37), 0.4, device)
                local = ex._get_robot_palm_local_frame(side)
                anatomical = local * target
                np.testing.assert_allclose(anatomical.TransformDir(Gf.Vec3d(1, 0, 0)), [1, 0, 0], atol=1e-7)
                np.testing.assert_allclose(anatomical.TransformDir(Gf.Vec3d(0, 0, 1)), [0, 0, -1], atol=1e-7)
                self.assertAlmostEqual(target.GetDeterminant(), 1.0)
                angle = abs(target.ExtractRotation().GetAngle()) % 360.0
                self.assertAlmostEqual(min(angle, 360.0 - angle), 90.0)
                self.assertNotIn(side, ex._arm_orientation_anchors)

    def test_world_rotation_and_translation_preserve_anatomical_alignment(self) -> None:
        ex = self.example
        rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3), 127))
        for side in ("left", "right"):
            baseline = ex._compute_arm_target_orientation(side, pose(), 0.0, PalmDevice(side))
            for scale in (0.6, 1.0, 1.5):
                with self.subTest(side=side, scale=scale):
                    transform = Gf.Matrix4d(rotation)
                    transform.SetTranslateOnly(Gf.Vec3d(11, -7, 3))
                    device = PalmDevice(side, transform)
                    device.positions = {name: value * scale for name, value in device.positions.items()}
                    target = ex._compute_arm_target_orientation(side, pose(yaw=-53), 1.3, device)
                    np.testing.assert_allclose(target, baseline * rotation, atol=1e-7)

    def test_missing_or_degenerate_optical_landmarks_do_not_invent_an_orientation(self) -> None:
        ex = self.example
        for name in ("wrist", "middle_proximal", "index_proximal", "little_proximal"):
            device = PalmDevice("right")
            device.invalid.add(name)
            self.assertIsNone(ex._compute_arm_target_orientation("right", pose(), 0.0, device))
            device.positions.pop(name)
            self.assertIsNone(ex._compute_arm_target_orientation("right", pose(), 0.0, device))
        for point in (Gf.Vec3d(0), Gf.Vec3d(math.nan, 0, 0), Gf.Vec3d(math.inf, 0, 0)):
            device = PalmDevice("right")
            device.positions["middle_proximal"] = point
            self.assertIsNone(ex._compute_arm_target_orientation("right", pose(), 0.0, device))
        device = PalmDevice("right")
        device.positions["index_proximal"] = Gf.Vec3d(0.1, 0, 0)
        device.positions["little_proximal"] = Gf.Vec3d(0.2, 0, 0)
        self.assertIsNone(ex._compute_arm_target_orientation("right", pose(), 0.0, device))
        device.positions["index_proximal"] = Gf.Vec3d(0.1, 0.0001, 0)
        self.assertIsNone(ex._compute_arm_target_orientation("right", pose(), 0.0, device))

    def test_reacquisition_uses_anatomy_instead_of_reanchoring_to_robot_posture(self) -> None:
        ex = self.example
        device = PalmDevice("right")
        first = ex._compute_arm_target_orientation("right", pose(), 0.0, device)
        device.invalid.add("middle_proximal")
        self.assertIsNone(ex._compute_arm_target_orientation("right", pose(), 0.0, device))
        ex._arm_orientation_anchors["right"] = (pose(yaw=-70), pose(yaw=43))
        device.invalid.clear()
        recovered = ex._compute_arm_target_orientation("right", pose(yaw=81), 0.8, device)
        np.testing.assert_allclose(recovered, first, atol=1e-7)

    def test_controller_clutch_still_preserves_initial_wrist_and_relative_rotation(self) -> None:
        ex = self.example
        robot = pose(yaw=31)
        quaternion = robot.ExtractRotationQuat()
        ex._get_hand_link_prim = Mock(
            return_value=types.SimpleNamespace(
                get_world_poses=lambda: (
                    Array([[0, 0, 0]]),
                    Array([[quaternion.GetReal(), *quaternion.GetImaginary()]]),
                )
            )
        )
        controller = Device(pose(yaw=-23))
        initial = ex._compute_arm_target_orientation("left", controller.matrix, 0.0, controller)
        np.testing.assert_allclose(initial, robot, atol=1e-7)
        controller.matrix = pose(yaw=17)
        moved = ex._compute_arm_target_orientation("left", controller.matrix, 0.0, controller)
        np.testing.assert_allclose(moved, robot * pose(yaw=40), atol=1e-7)
        ex._arm_orientation_anchors.clear()
        reacquired = ex._compute_arm_target_orientation("left", controller.matrix, 0.0, controller)
        np.testing.assert_allclose(reacquired, robot, atol=1e-7)

    def test_robot_calibration_uses_fixed_joint_anchors_and_stable_palm_center(self) -> None:
        ex = self.example
        frame = ex._get_robot_palm_local_frame("right")
        np.testing.assert_allclose(ex._get_robot_palm_local_center("right"), [0.06, 0, 0], atol=1e-7)
        # Moving downstream finger geometry must not redefine the rigid palm frame.
        finger = UsdGeom.Xform.Define(self.stage, "/World/G1/right_hand/R_middle_intermediate")
        finger.AddTranslateOp().Set(Gf.Vec3d(3, 4, -9))
        ex._robot_palm_local_frames.clear()
        np.testing.assert_allclose(ex._get_robot_palm_local_frame("right"), frame, atol=1e-7)
        np.testing.assert_allclose(ex._get_robot_palm_local_center("right"), [0.06, 0, 0], atol=1e-7)

    def test_missing_or_misparented_robot_anchor_disables_absolute_orientation(self) -> None:
        ex = self.example
        path = "/World/G1/right_hand/joints/R_index_proximal_joint"
        joint = UsdPhysics.Joint.Get(self.stage, path)
        joint.GetBody0Rel().SetTargets(["/World/G1/right_hand/R_index_proximal"])
        self.assertIsNone(ex._compute_arm_target_orientation("right", pose(), 0.0, PalmDevice("right")))
        self.assertIsNone(ex._get_robot_palm_local_center("right"))
        self.stage.RemovePrim(path)
        self.assertIsNone(ex._compute_arm_target_orientation("right", pose(), 0.0, PalmDevice("right")))


if __name__ == "__main__":
    unittest.main()
