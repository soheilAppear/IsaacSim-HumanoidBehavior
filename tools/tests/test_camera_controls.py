# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the stationary camera rigidly mounted while real-world headset poses change."""

from __future__ import annotations

import types
import unittest
from unittest.mock import Mock

import numpy as np
from humanoid_test_support import HUMANOID, USD, Array, pose
from pxr import Gf, Usd, UsdGeom, UsdPhysics


class TestRobotHeadCamera(unittest.TestCase):
    def setUp(self):
        self.example = HUMANOID.HumanoidExample()
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)
        self.example.g1 = types.SimpleNamespace(robot=Mock())
        self.example.g1.robot.is_physics_tensor_entity_valid.return_value = False
        self.local_mount = (
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(0.18, 0.02, 0.1), Gf.Vec3d(1.18, 0.02, -0.12), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        self.example._head_camera_mount_local = self.local_mount
        self.example._head_camera_mount_body_path = "/World/G1/head_link"
        self.body_pose = pose((0, 0, 1.25))
        self.example._read_camera_mount_body_pose = Mock(side_effect=lambda: self.body_pose)
        self.example._read_physical_head_pose = Mock(side_effect=AssertionError("Camera must not sample the HMD"))
        self.example._stabilize_camera_pose = Mock(side_effect=AssertionError("Rigid mounts must not lag body motion"))
        self.example._xr_core = Mock()

    def assert_matrix_close(self, actual, expected):
        self.assertIsNotNone(actual)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-10)

    def install_camera(self):
        camera = UsdGeom.Camera.Define(self.stage, self.example._head_camera_path)
        self.example._head_camera_transform_op = UsdGeom.Xformable(camera).AddTransformOp()

    def test_default_is_robot_head_and_pose_is_available_while_physics_is_paused(self):
        self.assertEqual(self.example._xr_camera_mode, "robot_head")
        self.example._physics_ready = False
        self.assert_matrix_close(self.example._get_head_camera_pose(), self.local_mount * self.body_pose)
        self.example._read_physical_head_pose.assert_not_called()
        self.example._stabilize_camera_pose.assert_not_called()

    def test_mount_preserves_full_body_translation_yaw_pitch_and_roll(self):
        for position, yaw, pitch, roll in (
            ((0, 0, 1.25), 0, 0, 0),
            ((1.3, -0.7, 1.1), 85, 0, 0),
            ((-2, 4, 0.8), -25, 32, -17),
            ((0.5, -3, 2), 170, -42, 63),
        ):
            with self.subTest(position=position, yaw=yaw, pitch=pitch, roll=roll):
                self.body_pose = (
                    Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), roll))
                    * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), pitch))
                    * pose(position, yaw)
                )
                camera_pose = self.example._get_robot_head_camera_pose()
                self.assert_matrix_close(camera_pose * self.body_pose.GetInverse(), self.local_mount)
        self.example._read_physical_head_pose.assert_not_called()
        self.example._stabilize_camera_pose.assert_not_called()

    def test_stationary_mode_suppresses_legacy_hmd_camera_routes(self):
        self.install_camera()
        legacy_routes = (
            "_schedule_composed_xr_camera",
            "_schedule_stage_anchor_xr_camera",
            "_configure_xr_custom_anchor",
        )
        for route in legacy_routes:
            setattr(self.example, route, Mock(side_effect=AssertionError("Legacy camera route must remain disabled")))
        for mode in ("robot_head", "head_compose", "stage_anchor", "custom_anchor", "camera_lock"):
            with self.subTest(mode=mode):
                self.example._xr_camera_mode = mode
                self.example._update_head_camera_view(force=True, dt=0.01)
                expected = self.local_mount * self.body_pose
                self.assert_matrix_close(self.example._head_camera_transform_op.Get(), expected)
                self.assert_matrix_close(self.example._xr_core.schedule_set_camera.call_args.args[0], expected)
        self.assertEqual(self.example._xr_core.schedule_set_camera.call_count, 5)
        for route in legacy_routes:
            getattr(self.example, route).assert_not_called()
        self.example._read_physical_head_pose.assert_not_called()

    def test_controller_mode_and_recenter_buttons_cannot_unlock_the_mount(self):
        self.example._update_head_camera_view = Mock()
        self.example._xr_calibrated = True
        self.example._xr_calibration_samples = ["existing calibration"]
        self.example._xr_camera_mode = "head_compose"
        self.example._cycle_xr_camera_mode()
        self.assertEqual(self.example._xr_camera_mode, "robot_head")
        self.example._request_xr_recenter()
        self.assertEqual(self.example._xr_camera_mode, "robot_head")
        self.assertTrue(self.example._xr_calibrated)
        self.assertEqual(self.example._xr_calibration_samples, ["existing calibration"])
        self.assertTrue(self.example._update_head_camera_view.called)
        self.example._read_physical_head_pose.assert_not_called()

    def test_render_callback_refreshes_the_mount_without_a_physics_step(self):
        self.example._physics_ready = False
        self.example._update_head_camera_view = Mock()
        self.example._on_robot_camera_update(types.SimpleNamespace(payload={"dt": 1.0 / 90.0}))
        self.example._update_head_camera_view.assert_called_once()
        self.example._read_physical_head_pose.assert_not_called()

    def test_reset_discards_dead_body_handle_and_keeps_the_authored_mount(self):
        self.example._head_camera_mount_prim = Mock()
        self.example._release_grabbed_object = Mock()
        self.example._set_grab_candidate = Mock()
        self.example._reset_teleoperation_state()
        self.assertIsNone(self.example._head_camera_mount_prim)
        self.assertEqual(self.example._head_camera_mount_body_path, "/World/G1/head_link")
        self.assert_matrix_close(self.example._head_camera_mount_local, self.local_mount)

    def test_visual_head_uses_its_rigid_torso_ancestor_as_the_mount(self):
        root = UsdGeom.Xform.Define(self.stage, "/World/G1")
        root.AddTransformOp().Set(pose((2, -1, 0.8), 37))
        torso = UsdGeom.Xform.Define(self.stage, "/World/G1/torso_link")
        torso.AddTranslateOp().Set(Gf.Vec3d(0.02, 0, 0.26))
        UsdPhysics.RigidBodyAPI.Apply(torso.GetPrim())
        head = UsdGeom.Xform.Define(self.stage, "/World/G1/torso_link/head_link")
        head.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.2))
        self.example._prepare_robot_head_camera_mount(self.stage)
        self.assertEqual(self.example._head_camera_mount_body_path, "/World/G1/torso_link")
        self.body_pose = UsdGeom.XformCache().GetLocalToWorldTransform(torso.GetPrim())
        camera = self.example._get_robot_head_camera_pose()
        camera_in_robot = camera * UsdGeom.XformCache().GetLocalToWorldTransform(root.GetPrim()).GetInverse()
        expected_eye = (
            self.example._first_person_head_forward_offset,
            0,
            self.example._first_person_eye_height_above_base + self.example._first_person_head_up_offset,
        )
        np.testing.assert_allclose(camera_in_robot.ExtractTranslation(), expected_eye, atol=1e-10)

    def test_active_body_uses_live_physics_pose_even_when_usd_is_stale(self):
        del self.example._read_camera_mount_body_pose
        body = UsdGeom.Xform.Define(self.stage, self.example._head_camera_mount_body_path)
        body.AddTranslateOp().Set(Gf.Vec3d(20, 30, 40))
        self.example.g1.robot.is_physics_tensor_entity_valid.return_value = True
        self.example._head_camera_mount_prim = Mock()
        rotation = Gf.Rotation(Gf.Vec3d(1, 2, 3).GetNormalized(), 42).GetQuat()
        position = (-0.4, 0.8, 1.3)
        self.example._head_camera_mount_prim.get_world_poses.return_value = (
            Array(np.array([position])),
            Array(np.array([[rotation.GetReal(), *rotation.GetImaginary()]])),
        )
        expected = Gf.Matrix4d().SetRotate(rotation)
        expected.SetTranslateOnly(Gf.Vec3d(*position))
        self.assert_matrix_close(self.example._read_camera_mount_body_pose(), expected)

    def test_invalid_live_pose_does_not_silently_jump_to_authored_body_pose(self):
        del self.example._read_camera_mount_body_pose
        UsdGeom.Xform.Define(self.stage, self.example._head_camera_mount_body_path)
        self.example.g1.robot.is_physics_tensor_entity_valid.return_value = True
        self.example._head_camera_mount_prim = Mock()
        self.example._head_camera_mount_prim.get_world_poses.return_value = (
            Array(np.array([[float("nan"), 0, 1]])),
            Array(np.array([[1, 0, 0, 0]])),
        )
        self.assertIsNone(self.example._read_camera_mount_body_pose())
        self.example._head_camera_mount_prim.get_world_poses.side_effect = RuntimeError("physics view reset")
        self.assertIsNone(self.example._read_camera_mount_body_pose())
        self.assertIsNone(self.example._head_camera_mount_prim)

    def test_authored_body_is_available_before_the_first_physics_step(self):
        del self.example._read_camera_mount_body_pose
        body = UsdGeom.Xform.Define(self.stage, self.example._head_camera_mount_body_path)
        body.AddTransformOp().Set(pose((0.2, -0.1, 1.3), 20))
        self.example._head_camera_mount_prim = Mock(side_effect=AssertionError("No live view exists before Play"))
        self.assert_matrix_close(self.example._read_camera_mount_body_pose(), pose((0.2, -0.1, 1.3), 20))
        self.example._head_camera_mount_prim.get_world_poses.assert_not_called()

    def test_cleanup_releases_frame_subscription_and_mount_handle(self):
        self.example._head_camera_update_sub = Mock()
        self.example._head_camera_mount_prim = Mock()
        self.example._release_grabbed_object = Mock()
        self.example._set_grab_candidate = Mock()
        self.example._save_behavioral_data = Mock()
        self.example._unsubscribe_keyboard = Mock()
        self.example._restore_physics_simulation_state = Mock()
        self.example.physics_cleanup()
        self.assertIsNone(self.example._head_camera_update_sub)
        self.assertIsNone(self.example._head_camera_mount_prim)


if __name__ == "__main__":
    unittest.main()
