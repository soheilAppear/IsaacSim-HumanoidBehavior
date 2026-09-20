# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify stabilized VR defaults and the explicit historical rigid-mount option."""

from __future__ import annotations

import asyncio
import math
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from humanoid_test_support import CARB, HUMANOID, USD, Array, BaseSample, pose
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from test_operator_head_controls import define_head


class TestPhysicalHeadPoseValidity(unittest.TestCase):
    def setUp(self):
        self.example = HUMANOID.HumanoidExample()
        self.example._xr_head_pose_max_age = 2
        self.device = types.SimpleNamespace()
        self.example._find_xr_head_device = Mock(side_effect=lambda: self.device)

    def descriptor(self, matrix=None, flags=3):
        return types.SimpleNamespace(pose_matrix=pose((0, 1.6, 0)) if matrix is None else matrix, validity_flags=flags)

    def test_raw_descriptor_is_preferred_without_reading_smoothed_or_virtual_pose(self):
        raw = pose((0, 1.6, 0), 30)
        self.device.get_raw_pose_desc = Mock(return_value=self.descriptor(raw))
        self.device.get_pose_desc = Mock(return_value=self.descriptor(pose((0, 1.6, 0), -30)))
        self.device.get_pose = Mock()
        self.device.get_virtual_world_pose = Mock()
        self.assertEqual(self.example._read_physical_head_pose(), raw)
        self.device.get_raw_pose_desc.assert_called_once_with("")
        self.device.get_pose_desc.assert_not_called()
        self.device.get_pose.assert_not_called()
        self.device.get_virtual_world_pose.assert_not_called()

    def test_valid_identity_and_inferred_valid_pose_are_accepted(self):
        self.device.get_raw_pose_desc = Mock(return_value=self.descriptor(Gf.Matrix4d(1), flags=3))
        self.assertEqual(self.example._read_physical_head_pose(), Gf.Matrix4d(1))
        self.assertEqual(self.example._xr_head_pose_age, 0)

    def test_invalid_raw_descriptor_never_replays_smoothed_or_matrix_fallback(self):
        self.device.get_pose_desc = Mock(return_value=self.descriptor())
        self.device.get_pose = Mock(return_value=pose((0, 1.6, 0)))
        self.device.get_raw_pose = Mock(return_value=pose((0, 1.6, 0)))
        for descriptor in (
            self.descriptor(flags=0),
            self.descriptor(flags=1),
            self.descriptor(flags=2),
            self.descriptor(flags=12),
            self.descriptor([[1, 0], [0, 1]]),
            self.descriptor(Gf.Matrix4d(0)),
            self.descriptor(pose((float("nan"), 0, 0))),
            self.descriptor(Gf.Matrix4d().SetScale(Gf.Vec3d(2, 1, 1))),
            RuntimeError("runtime lost"),
            AttributeError("descriptor missing matrix", name="pose_matrix"),
            AttributeError("runtime failed without a missing-method name"),
        ):
            with self.subTest(descriptor=descriptor):
                self.device.get_raw_pose_desc = Mock(
                    side_effect=descriptor if isinstance(descriptor, Exception) else None, return_value=descriptor
                )
                self.assertIsNone(self.example._read_physical_head_pose())
        self.device.get_pose_desc.assert_not_called()
        self.device.get_pose.assert_not_called()
        self.device.get_raw_pose.assert_not_called()

    def test_invalid_flags_age_last_good_pose_instead_of_refreshing_it(self):
        descriptor = self.descriptor()
        self.device.get_raw_pose_desc = Mock(return_value=descriptor)
        valid = self.example._read_physical_head_pose()
        descriptor.validity_flags = 0  # Runtime retains its last matrix after losing tracking.
        self.assertEqual(self.example._read_physical_head_pose(), valid)
        self.assertEqual(self.example._xr_head_pose_age, 1)
        self.assertEqual(self.example._read_physical_head_pose(), valid)
        self.assertEqual(self.example._xr_head_pose_age, 2)
        self.assertIsNone(self.example._read_physical_head_pose())
        self.assertIsNone(self.example._xr_last_head_pose)
        descriptor.validity_flags = 3
        descriptor.pose_matrix = pose((0.1, 1.6, 0), 25)
        self.assertEqual(self.example._read_physical_head_pose(), descriptor.pose_matrix)
        self.assertEqual(self.example._xr_head_pose_age, 0)

    def test_processed_descriptor_is_used_only_when_raw_descriptor_api_is_absent(self):
        self.device.get_pose_desc = Mock(return_value=self.descriptor())
        self.device.get_pose = Mock()
        self.assertEqual(self.example._read_physical_head_pose(), pose((0, 1.6, 0)))
        self.device.get_pose_desc.assert_called_once_with("")
        self.device.get_pose.assert_not_called()

    def test_raw_wrapper_with_missing_native_method_uses_validated_processed_descriptor(self):
        native = types.SimpleNamespace()
        self.device.get_raw_pose_desc = Mock(side_effect=lambda name: native.get_raw_pose_desc(name))
        descriptor = self.descriptor(flags=0x3000F)
        self.device.get_pose_desc = Mock(return_value=descriptor)
        self.device.get_pose = Mock(return_value=pose((5, 6, 7)))
        self.assertEqual(self.example._read_physical_head_pose(), descriptor.pose_matrix)
        self.device.get_raw_pose_desc.assert_called_once_with("")
        self.device.get_pose_desc.assert_called_once_with("")
        descriptor.validity_flags = 0
        self.example._xr_last_head_pose = None
        self.assertIsNone(self.example._read_physical_head_pose())
        self.device.get_pose.assert_not_called()

    def test_matrix_only_adapters_keep_legacy_sentinel_without_virtual_fallback(self):
        self.device.get_pose = Mock(return_value=Gf.Matrix4d(1))
        self.device.get_virtual_world_pose = Mock(return_value=pose((5, 6, 7)))
        self.assertIsNone(self.example._read_physical_head_pose())
        self.device.get_virtual_world_pose.assert_not_called()
        self.device.get_raw_pose = Mock(return_value=pose((0, 1.6, 0)))
        self.assertEqual(self.example._read_physical_head_pose(), pose((0, 1.6, 0)))


class TestRobotHeadCamera(unittest.TestCase):
    """Retain physical rigid-mount behavior when comfort stabilization is disabled."""

    def setUp(self):
        self.example = HUMANOID.HumanoidExample()
        self.example._head_camera_comfort_enabled = False
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
        self.example._read_physical_head_pose = Mock(return_value=None)
        self.example._stabilize_camera_pose = Mock(side_effect=AssertionError("Rigid mounts must not lag body motion"))
        self.example._xr_core = Mock()

    def assert_matrix_close(self, actual, expected):
        self.assertIsNotNone(actual)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-10)

    def install_camera(self):
        camera = UsdGeom.Camera.Define(self.stage, self.example._head_camera_path)
        self.example._head_camera_transform_op = UsdGeom.Xformable(camera).AddTransformOp()

    def set_head_pose(self, rotation=None, translation=(0.0, 1.6, 0.0)):
        head = Gf.Matrix4d(rotation) if rotation is not None else Gf.Matrix4d(1.0)
        head.SetTranslateOnly(Gf.Vec3d(*translation))
        self.example._read_physical_head_pose.return_value = head

    def enable_xr_scheduling(self):
        """Provide live physics and a fresh tracked headset for native XR writes."""
        self.example._physics_ready = True
        self.example._timeline_stopped = False
        self.example.g1.robot.is_physics_tensor_entity_valid.return_value = True
        self.example._xr_core.is_xr_enabled.return_value = True
        self.example._xr_core.is_xr_display_enabled.return_value = True
        self.example._xr_core.is_xr_viewport_enabled.return_value = True
        profile = self.example._xr_core.get_current_profile.return_value
        profile.get_name.return_value = "vr"
        profile.is_enabled.return_value = True
        self.example._find_xr_head_device = Mock(
            return_value=types.SimpleNamespace(
                get_pose_desc=lambda name: types.SimpleNamespace(pose_matrix=pose((0, 1.6, 0)), validity_flags=3)
            )
        )

    def test_default_is_robot_head_and_pose_is_available_while_physics_is_paused(self):
        self.assertEqual(self.example._xr_camera_mode, "robot_head")
        self.assertTrue(self.example._head_camera_operator_rotation_enabled)
        self.example._physics_ready = False
        self.assert_matrix_close(self.example._get_head_camera_pose(), self.local_mount * self.body_pose)
        self.example._read_physical_head_pose.assert_called_once()
        self.example._stabilize_camera_pose.assert_not_called()

    def test_head_turns_left_and_right_in_the_same_direction_at_a_fixed_eye_position(self):
        self.set_head_pose()
        mounted = self.example._get_head_camera_pose()
        for yaw, direction in ((30.0, 1), (-30.0, -1)):
            with self.subTest(yaw=yaw):
                self.set_head_pose(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), yaw)))
                camera = self.example._get_head_camera_pose()
                forward = camera.TransformDir(Gf.Vec3d(0, 0, -1))
                # Robot +X faces forward, +Y is its left. Positive physical
                # Y yaw turns OpenXR's -Z forward toward the operator's left.
                self.assertGreater(direction * forward[1], 0.0)
                np.testing.assert_allclose(camera.ExtractTranslation(), mounted.ExtractTranslation(), atol=1e-12)

    def test_physical_translation_never_changes_the_robot_mounted_camera(self):
        orientation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3).GetNormalized(), 38))
        self.set_head_pose(orientation)
        expected = self.example._get_head_camera_pose()
        for translation in ((0, 0.6, 0), (4, 2.1, -3), (-2, 1.2, 5)):
            with self.subTest(translation=translation):
                self.set_head_pose(orientation, translation)
                self.assert_matrix_close(self.example._get_head_camera_pose(), expected)

    def test_tilted_startup_calibrates_heading_only_and_upright_yaw_keeps_world_level(self):
        heading = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 63))
        tilt = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 32)) * Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -27)
        )
        self.set_head_pose(tilt * heading)
        mounted = self.local_mount * self.body_pose
        self.assert_matrix_close(self.example._get_head_camera_pose(), tilt * mounted)
        self.assert_matrix_close(self.example._head_camera_operator_neutral_rotation, heading)
        for axis, degrees in (
            ((0, 1, 0), -90),
            ((0, 1, 0), 0),
            ((0, 1, 0), 90),
            ((0, 1, 0), 180),
            ((1, 0, 0), 20),
            ((0, 0, -1), 25),
        ):
            with self.subTest(axis=axis, degrees=degrees):
                relative = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*axis), degrees))
                self.set_head_pose(relative * heading)
                self.assert_matrix_close(self.example._get_head_camera_pose(), relative * mounted)

    def test_vertical_startup_waits_for_defined_heading_then_tracks_vertical_look(self):
        vertical = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90))
        self.set_head_pose(vertical)
        mounted = self.local_mount * self.body_pose
        self.assert_matrix_close(self.example._get_head_camera_pose(), mounted)
        self.assertIsNone(self.example._head_camera_operator_neutral_rotation)
        self.set_head_pose()
        self.example._get_head_camera_pose()
        self.set_head_pose(vertical)
        self.assert_matrix_close(self.example._get_head_camera_pose(), vertical * mounted)

    def test_recenter_while_tilted_preserves_pitch_roll_and_does_not_change_gaze(self):
        self.install_camera()
        self.enable_xr_scheduling()
        gaze = object()
        self.example._eye_gaze_tracker = gaze
        self.set_head_pose()
        self.example._get_head_camera_pose()
        heading = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), -73))
        tilt = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), -19)) * Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), 24)
        )
        self.set_head_pose(tilt * heading)
        self.example._request_xr_recenter()
        self.assert_matrix_close(self.example._head_camera_operator_neutral_rotation, heading)
        self.assert_matrix_close(self.example._head_camera_transform_op.Get(), tilt * self.local_mount * self.body_pose)
        self.set_head_pose(heading)
        self.assert_matrix_close(self.example._get_head_camera_pose(), self.local_mount * self.body_pose)
        self.assertIs(self.example._eye_gaze_tracker, gaze)

    def test_head_rotation_keeps_following_live_body_pose_without_orbiting_the_eye(self):
        self.set_head_pose()
        self.example._get_head_camera_pose()
        relative = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), -40))
        self.set_head_pose(relative, (3, 2, -4))
        self.body_pose = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 17)) * pose((2, -3, 1.1), 65)
        mounted = self.example._get_robot_head_camera_pose()
        camera = self.example._get_head_camera_pose()
        self.assert_matrix_close(camera, relative * mounted)
        np.testing.assert_allclose(camera.ExtractTranslation(), mounted.ExtractTranslation(), atol=1e-12)

    def test_undefined_recenter_heading_preserves_view_and_previous_reference(self):
        self.set_head_pose(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 31)))
        self.example._get_head_camera_pose()
        reference = self.example._head_camera_operator_neutral_rotation
        for head in (
            Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90)) * reference,
            Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)) * reference,
            None,
        ):
            with self.subTest(head=head):
                self.example._read_physical_head_pose.return_value = head
                before = self.example._get_head_camera_pose()
                self.example._request_xr_recenter()
                self.assertIs(self.example._head_camera_operator_neutral_rotation, reference)
                self.assert_matrix_close(self.example._get_head_camera_pose(), before)

    def test_keyboard_b_recenters_once_for_bare_hands_in_stationary_and_gesture_modes(self):
        events = CARB.input.KeyboardEventType
        for locomotion, gesture in (("stationary", False), ("decoupled", True)):
            for key in ("B", types.SimpleNamespace(name="B")):
                with self.subTest(locomotion=locomotion, key=key):
                    self.example._g1_locomotion = locomotion
                    self.example._gesture_locomotion_enabled = gesture
                    self.example._keyboard_command = np.zeros(3)
                    self.example._request_xr_recenter = Mock()
                    for kind in (events.KEY_PRESS, events.KEY_REPEAT, events.KEY_RELEASE):
                        self.example._sub_keyboard_event(types.SimpleNamespace(input=key, type=kind))
                    self.example._request_xr_recenter.assert_called_once()
                    np.testing.assert_array_equal(self.example._keyboard_command, np.zeros(3))

    def test_head_tracking_loss_holds_last_rotation_and_reacquires_without_recalibration(self):
        self.set_head_pose()
        self.example._get_head_camera_pose()
        relative = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 35))
        self.set_head_pose(relative)
        self.example._get_head_camera_pose()
        self.example._read_physical_head_pose.return_value = None
        self.body_pose = pose((1, 2, 1.4), 52)
        expected = relative * self.local_mount * self.body_pose
        for _ in range(35):
            self.assert_matrix_close(self.example._get_head_camera_pose(), expected)
        self.set_head_pose()
        self.assert_matrix_close(self.example._get_head_camera_pose(), self.local_mount * self.body_pose)

    def test_invalid_head_rotation_holds_last_valid_view(self):
        self.set_head_pose()
        self.example._get_head_camera_pose()
        relative = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), -20))
        self.set_head_pose(relative)
        expected = self.example._get_head_camera_pose()
        for value in (float("nan"), float("inf")):
            invalid = Gf.Matrix4d(1.0)
            invalid[0, 0] = value
            self.example._read_physical_head_pose.return_value = invalid
            self.assert_matrix_close(self.example._get_head_camera_pose(), expected)
        self.example._read_physical_head_pose.return_value = Gf.Matrix4d(0.0)
        self.assert_matrix_close(self.example._get_head_camera_pose(), expected)

    def test_disabled_operator_rotation_retains_rigid_mount_without_sampling_the_headset(self):
        self.example._head_camera_operator_rotation_enabled = False
        self.assert_matrix_close(self.example._get_head_camera_pose(), self.local_mount * self.body_pose)
        self.example._read_physical_head_pose.assert_not_called()

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
        self.enable_xr_scheduling()
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
        self.assertEqual(self.example._read_physical_head_pose.call_count, 5)

    def test_native_camera_waits_for_xr_viewport_and_keeps_desktop_view_available(self):
        self.install_camera()
        self.enable_xr_scheduling()
        expected = self.local_mount * self.body_pose
        # The session and display flags may remain true while the native XR
        # viewport has no resources during teardown or initialization.
        self.example._xr_core.is_xr_viewport_enabled.return_value = False
        self.example._update_head_camera_view(force=True, dt=0)
        self.example._xr_core.schedule_set_camera.assert_not_called()
        self.assert_matrix_close(self.example._head_camera_transform_op.Get(), expected)
        self.example._find_xr_head_device.assert_not_called()

        self.example._xr_core.is_xr_viewport_enabled.return_value = True
        self.example._update_head_camera_view(force=True, dt=0)
        self.example._xr_core.schedule_set_camera.assert_called_once()
        self.assert_matrix_close(self.example._xr_core.schedule_set_camera.call_args.args[0], expected)
        self.assert_matrix_close(self.example._head_camera_transform_op.Get(), expected)

        # Readiness can disappear again without rebuilding the Python example.
        self.example._xr_core.is_xr_viewport_enabled.return_value = False
        self.example._update_head_camera_view(force=True, dt=0)
        self.example._xr_core.schedule_set_camera.assert_called_once()

    def test_controller_mode_and_recenter_buttons_cannot_unlock_the_mount(self):
        self.example._update_head_camera_view = Mock()
        self.set_head_pose()
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
        self.example._read_physical_head_pose.assert_called_once()

    def test_recenter_only_resets_head_orientation_without_touching_eye_gaze_or_legacy_calibration(self):
        self.install_camera()
        self.enable_xr_scheduling()
        tracker = object()
        self.example._eye_gaze_tracker = tracker
        self.example._xr_calibrated = True
        self.example._xr_calibration_samples = ["existing calibration"]
        self.set_head_pose()
        self.example._get_head_camera_pose()
        turned = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 55))
        self.set_head_pose(turned)
        self.example._get_head_camera_pose()
        self.example._request_xr_recenter()
        expected = self.local_mount * self.body_pose
        self.assert_matrix_close(self.example._head_camera_transform_op.Get(), expected)
        self.assert_matrix_close(self.example._xr_core.schedule_set_camera.call_args.args[0], expected)
        self.assertTrue(self.example._xr_calibrated)
        self.assertEqual(self.example._xr_calibration_samples, ["existing calibration"])
        self.assertIs(self.example._eye_gaze_tracker, tracker)
        self.assert_matrix_close(self.example._head_camera_operator_neutral_rotation, turned)
        self.set_head_pose(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 70)))
        self.assert_matrix_close(
            self.example._get_head_camera_pose(),
            Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 15)) * expected,
        )

    def test_render_callback_refreshes_the_mount_without_a_physics_step(self):
        self.example._physics_ready = False
        self.example._update_head_camera_view = Mock()
        self.example._on_robot_camera_update(types.SimpleNamespace(payload={"dt": 1.0 / 90.0}))
        self.example._update_head_camera_view.assert_called_once()
        self.example._read_physical_head_pose.assert_not_called()

    def test_reset_discards_dead_body_handle_and_keeps_the_authored_mount(self):
        self.example._head_camera_mount_prim = Mock()
        self.set_head_pose()
        self.example._get_head_camera_pose()
        self.set_head_pose(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 45)))
        self.example._get_head_camera_pose()
        self.example._release_grabbed_object = Mock()
        self.example._set_grab_candidate = Mock()
        self.example._reset_teleoperation_state()
        self.assertIsNone(self.example._head_camera_mount_prim)
        self.assertEqual(self.example._head_camera_mount_body_path, "/World/G1/head_link")
        self.assert_matrix_close(self.example._head_camera_mount_local, self.local_mount)
        self.assertIsNone(self.example._head_camera_operator_neutral_rotation)
        self.assertIsNone(self.example._head_camera_operator_delta_rotation)
        self.assert_matrix_close(self.example._get_head_camera_pose(), self.local_mount * self.body_pose)

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


class TestMeasuredVisualHeadCamera(unittest.TestCase):
    """Verify the historical visual-head attachment under explicit comfort opt-out."""

    def setUp(self):
        self.example = HUMANOID.HumanoidExample()
        self.example._head_camera_comfort_enabled = False
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)
        root = UsdGeom.Xform.Define(self.stage, "/World/G1")
        root.AddTransformOp().Set(pose((2.0, -1.0, 0.8), 37.0))
        self.torso = UsdGeom.Xform.Define(self.stage, "/World/G1/torso_link")
        self.torso_op = self.torso.AddTransformOp()
        self.torso_op.Set(pose((-0.004, 0.0, 0.044)))
        UsdPhysics.RigidBodyAPI.Apply(self.torso.GetPrim())
        UsdGeom.Xform.Define(self.stage, "/HeadRenderSource")
        define_head(self.stage, "/HeadRenderSource/head_link")
        self.visuals = UsdGeom.Xform.Define(self.stage, "/World/G1/torso_link/visuals").GetPrim()
        self.visuals.GetReferences().AddInternalReference("/HeadRenderSource")
        self.visuals.SetInstanceable(True)
        self.example.g1 = types.SimpleNamespace(robot=Mock())
        self.example.g1.robot.is_physics_tensor_entity_valid.return_value = False
        self.example._read_physical_head_pose = Mock(return_value=pose((0.0, 1.6, 0.0)))
        self.example._eye_gaze_tracker = object()
        self.gaze = self.example._eye_gaze_tracker
        self.example._prepare_robot_head_camera_mount(self.stage)
        self.assertTrue(self.example._operator_head_visual.ready, self.example._operator_head_visual.status)
        self.head = self.stage.GetPrimAtPath("/World/G1/torso_link/visuals/head_link")
        self.addCleanup(self.example._operator_head_visual.clear)

    def assert_matrix_close(self, actual, expected):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-10)

    def head_world(self):
        return UsdGeom.XformCache().GetLocalToWorldTransform(self.head)

    def test_neutral_camera_uses_measured_eye_with_clearance_and_level_forward(self):
        camera = self.example._get_head_camera_pose()
        torso = UsdGeom.XformCache().GetLocalToWorldTransform(self.torso.GetPrim())
        local_camera = camera * torso.GetInverse()
        np.testing.assert_allclose(local_camera.ExtractTranslation(), [0.099, 0.0, 0.386], atol=1e-7)
        np.testing.assert_allclose(local_camera.TransformDir(Gf.Vec3d(0, 0, -1)), [1, 0, 0], atol=1e-10)
        np.testing.assert_allclose(local_camera.TransformDir(Gf.Vec3d(0, 1, 0)), [0, 0, 1], atol=1e-10)
        self.assertEqual(self.example._head_camera_mount_body_path, "/World/G1/torso_link")
        self.assertIs(self.example._eye_gaze_tracker, self.gaze)

    def test_camera_to_visible_head_transform_stays_fixed_for_yaw_pitch_and_roll(self):
        camera = self.example._get_head_camera_pose()
        original_attachment = camera * self.head_world().GetInverse()
        original_eye = camera.ExtractTranslation()
        for axis, angle in (((0, 1, 0), -35.0), ((0, 1, 0), 35.0), ((1, 0, 0), 20.0), ((0, 0, -1), 25.0)):
            with self.subTest(axis=axis, angle=angle):
                rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*axis), angle))
                rotation.SetTranslateOnly(Gf.Vec3d(3.0, 2.0, -4.0))
                self.example._read_physical_head_pose.return_value = rotation
                camera = self.example._get_head_camera_pose()
                self.assert_matrix_close(camera * self.head_world().GetInverse(), original_attachment)
                self.assertGreater((camera.ExtractTranslation() - original_eye).GetLength(), 0.001)
                self.assertIs(self.example._eye_gaze_tracker, self.gaze)

    def test_physical_head_translation_cannot_move_visual_head_or_camera(self):
        neutral = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3).GetNormalized(), 38))
        self.example._read_physical_head_pose.return_value = neutral
        camera = self.example._get_head_camera_pose()
        head = self.head_world()
        for position in ((0, 0.6, 0), (4, 2.1, -3), (-2, 1.2, 5)):
            with self.subTest(position=position):
                translated = Gf.Matrix4d(neutral)
                translated.SetTranslateOnly(Gf.Vec3d(*position))
                self.example._read_physical_head_pose.return_value = translated
                self.assert_matrix_close(self.example._get_head_camera_pose(), camera)
                self.assert_matrix_close(self.head_world(), head)

    def test_tracking_loss_holds_head_and_eye_attachment_while_torso_moves(self):
        initial_camera = self.example._get_head_camera_pose()
        attachment = initial_camera * self.head_world().GetInverse()
        self.example._read_physical_head_pose.return_value = Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(0, 1, 0), -35)
        )
        self.example._get_head_camera_pose()
        head_local = UsdGeom.Xformable(self.head).GetLocalTransformation()
        self.example._read_physical_head_pose.return_value = None
        self.torso_op.Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 17)) * pose((1.0, 2.0, 0.1), 45))
        camera = self.example._get_head_camera_pose()
        self.assert_matrix_close(UsdGeom.Xformable(self.head).GetLocalTransformation(), head_local)
        self.assert_matrix_close(camera * self.head_world().GetInverse(), attachment)
        self.assertGreater((camera.ExtractTranslation() - initial_camera.ExtractTranslation()).GetLength(), 0.5)

    def test_reset_returns_visual_and_camera_to_neutral_without_changing_gaze(self):
        neutral_camera = self.example._get_head_camera_pose()
        neutral_head = self.head_world()
        self.example._read_physical_head_pose.return_value = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 45))
        self.example._get_head_camera_pose()
        self.example._reset_operator_head_rotation()
        self.assert_matrix_close(self.head_world(), neutral_head)
        self.assert_matrix_close(self.example._get_head_camera_pose(), neutral_camera)
        self.assertIs(self.example._eye_gaze_tracker, self.gaze)

    def test_disabling_operator_rotation_returns_both_view_and_head_to_authored_pose(self):
        neutral_camera = self.example._get_head_camera_pose()
        neutral_head = self.head_world()
        self.example._read_physical_head_pose.return_value = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 45))
        self.example._get_head_camera_pose()
        self.example._head_camera_operator_rotation_enabled = False
        self.example._read_physical_head_pose.reset_mock()
        self.assert_matrix_close(self.example._get_head_camera_pose(), neutral_camera)
        self.assert_matrix_close(self.head_world(), neutral_head)
        self.example._read_physical_head_pose.assert_not_called()

    def test_camera_recreation_preserves_neutral_mount_and_existing_head_attachment(self):
        self.example._get_head_camera_pose()
        self.example._read_physical_head_pose.return_value = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 45))
        camera = self.example._get_head_camera_pose()
        self.example._create_head_camera()
        self.assert_matrix_close(self.example._get_head_camera_pose(), camera)
        order = list(UsdGeom.Xformable(self.head).GetXformOpOrderAttr().Get())
        self.assertEqual(order.count("xformOp:transform:operatorHead"), 1)

    def test_physics_cleanup_restores_visual_reference_and_original_head_pose(self):
        self.example._get_head_camera_pose()
        self.example._read_physical_head_pose.return_value = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 45))
        self.example._get_head_camera_pose()
        self.example._eye_gaze_tracker = None
        for method in (
            "_release_grabbed_object",
            "_set_grab_candidate",
            "_save_behavioral_data",
            "_unsubscribe_keyboard",
            "_restore_physics_simulation_state",
        ):
            setattr(self.example, method, Mock())
        self.example.physics_cleanup()
        self.assertTrue(self.visuals.IsInstanceable())
        self.assertFalse(self.example._operator_head_visual.ready)
        head = self.stage.GetPrimAtPath("/World/G1/torso_link/visuals/head_link")
        self.assertFalse(head.GetAttribute("xformOp:transform:operatorHead").IsValid())


class TestComfortRobotHeadCamera(unittest.TestCase):
    """Exercise the actual main camera path with USD optical geometry and HMD poses."""

    def setUp(self):
        self.example = HUMANOID.HumanoidExample()
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)
        root = UsdGeom.Xform.Define(self.stage, "/World/G1")
        self.root_op = root.AddTransformOp()
        self.root_op.Set(pose((0.0, 0.0, 0.8)))
        self.torso = UsdGeom.Xform.Define(self.stage, "/World/G1/torso_link")
        self.torso_op = self.torso.AddTransformOp()
        self.torso_op.Set(pose((-0.004, 0.0, 0.044)))
        UsdPhysics.RigidBodyAPI.Apply(self.torso.GetPrim())
        UsdGeom.Xform.Define(self.stage, "/HeadRenderSource")
        define_head(self.stage, "/HeadRenderSource/head_link")
        mesh = UsdGeom.Mesh.Get(self.stage, "/HeadRenderSource/head_link/mesh")
        # Measured bounds of the loaded 6.1 asset, preserving the real 4.6 cm
        # difference between its bounding-box midpoint and optical sensor height.
        minimum = (-0.0663562194, -0.0777746066, 0.324795872)
        maximum = (0.0739840269, 0.0778166428, 0.530572593)
        mesh.GetPointsAttr().Set(
            [
                (x, y, z)
                for z in (minimum[2], maximum[2])
                for y in (minimum[1], maximum[1])
                for x in (minimum[0], maximum[0])
            ]
        )
        mesh.GetExtentAttr().Set([minimum, maximum])
        visuals = UsdGeom.Xform.Define(self.stage, "/World/G1/torso_link/visuals").GetPrim()
        visuals.GetReferences().AddInternalReference("/HeadRenderSource")
        visuals.SetInstanceable(True)
        sensor = UsdGeom.Xform.Define(self.stage, "/World/G1/torso_link/d435_link")
        sensor_pose = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 47.6))
        sensor_pose.SetTranslateOnly(Gf.Vec3d(0.0576234981, 0.0175299998, 0.4298700094))
        sensor.AddTransformOp().Set(sensor_pose)
        self.example.g1 = types.SimpleNamespace(robot=Mock(), _walk_time=0.0)
        self.example.g1.robot.is_physics_tensor_entity_valid.return_value = False
        self.example._physics_ready = False
        self.example._timeline_stopped = True
        self.example._read_physical_head_pose = Mock(return_value=pose((0.0, 1.6, 0.0)))
        self.example._base_command = np.zeros(3)
        self.example._eye_gaze_tracker = object()
        self.gaze = self.example._eye_gaze_tracker
        self.now_s = 0.0
        self.clock = patch.object(HUMANOID.time, "perf_counter", side_effect=lambda: self.now_s)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.example._prepare_robot_head_camera_mount(self.stage)
        self.addCleanup(self.example._operator_head_visual.clear)

    def assert_matrix_close(self, actual, expected):
        self.assertIsNotNone(actual)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-10)

    def advance(self, elapsed=0.02, dt=0.0):
        """Advance wall time independently of the supplied physics timestep."""
        self.now_s += elapsed
        return self.example._get_head_camera_pose(dt=dt)

    def head_world(self):
        head = self.stage.GetPrimAtPath(self.example._operator_head_visual.head_path)
        return UsdGeom.XformCache().GetLocalToWorldTransform(head)

    def assert_visible_head_anchored_to_neck(self, body_world=None):
        visual = self.example._operator_head_visual
        self.assertFalse(visual.first_person_hidden)
        self.assertNotEqual(
            UsdGeom.Imageable(self.stage.GetPrimAtPath(visual.head_path)).ComputeVisibility(), UsdGeom.Tokens.invisible
        )
        body = self.example._read_camera_mount_body_pose() if body_world is None else body_world
        # Reconstruct using the supplied measured body pose, as Fabric need not
        # publish current physical transforms into the USD test scene.
        local = UsdGeom.Xformable(self.stage.GetPrimAtPath(visual.head_path)).GetLocalTransformation()
        head_world = local * visual._parent_to_torso_transform * body
        np.testing.assert_allclose(
            head_world.Transform(visual.neck_pivot_local),
            body.Transform(visual.neck_pivot_in_torso),
            atol=1e-10,
        )
        return head_world

    def expected_hmd_view(self, neutral_camera, rotation):
        """Rotate the eye-to-neck vector in camera axes, independent of body pose."""
        pivot = self.example._head_camera_mount_local.GetInverse().Transform(
            self.example._operator_head_visual.neck_pivot_in_torso
        )
        # The neutral eye is the origin in these axes. Rotating that point
        # around the neck moves it by pivot - R(pivot), without using the
        # production articulated-view / measured-mount transform composition.
        turned_eye = pivot - rotation.TransformDir(pivot)
        expected = rotation * neutral_camera
        expected.SetTranslateOnly(neutral_camera.Transform(turned_eye))
        return expected

    def test_default_comfort_mount_uses_optical_height_and_ignores_sensor_downward_pitch(self):
        self.assertTrue(self.example._head_camera_comfort_enabled)
        camera = self.example._get_head_camera_pose()
        mount = self.example._head_camera_mount_local
        old_height = self.example._operator_head_visual.eye_mount_in_torso[2]
        self.assertAlmostEqual(mount.ExtractTranslation()[2] - old_height, 0.046185777, places=7)
        self.assertAlmostEqual(camera.ExtractTranslation()[2], 0.8 + 0.044 + 0.4298700094)
        self.assertAlmostEqual(mount.ExtractTranslation()[1], 0)
        np.testing.assert_allclose(camera.TransformDir(Gf.Vec3d(0, 0, -1)), (1, 0, 0), atol=1e-12)
        np.testing.assert_allclose(camera.TransformDir(Gf.Vec3d(0, 1, 0)), (0, 0, 1), atol=1e-12)
        self.assertEqual(self.example._head_camera_eye_height_source, "/World/G1/torso_link/d435_link")

    def test_torso_roll_pitch_bob_and_pelvis_bob_do_not_move_the_view(self):
        neutral = self.example._get_head_camera_pose()
        for sample in range(1, 101):
            wave = math.sin(sample * 0.2)
            self.torso_op.Set(
                Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), wave * 12))
                * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), wave * 9))
                * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), wave * 15))
                * pose((-0.004, 0, 0.044 + wave * 0.035))
            )
            self.root_op.Set(pose((0, 0, 0.8 + wave * 0.05)))
            self.assert_matrix_close(self.advance(), neutral)
            self.assert_visible_head_anchored_to_neck()
        self.assertIs(self.example._eye_gaze_tracker, self.gaze)

    def test_hmd_yaw_pitch_and_roll_immediately_rotate_the_eye_about_the_neck(self):
        neutral = self.example._get_head_camera_pose()
        neutral_eye = neutral.ExtractTranslation()
        neutral_head = self.assert_visible_head_anchored_to_neck()
        for axis, degrees in (((0, 1, 0), 30), ((0, 1, 0), -30), ((1, 0, 0), 20), ((0, 0, 1), 25)):
            with self.subTest(axis=axis, degrees=degrees):
                delta = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*axis), degrees))
                delta.SetTranslateOnly(Gf.Vec3d(4, 2, -3))
                self.example._read_physical_head_pose.return_value = delta
                rotation = Gf.Matrix4d().SetRotate(delta.ExtractRotationQuat())
                # Even at the exact same body-filter timestamp, the current HMD
                # rotation must reach this frame without any smoothing or delay.
                camera = self.example._get_head_camera_pose(dt=0)
                self.assert_matrix_close(camera, self.expected_hmd_view(neutral, rotation))
                self.assertGreater((camera.ExtractTranslation() - neutral_eye).GetLength(), 0.005)
                head = self.assert_visible_head_anchored_to_neck()
                self.assertFalse(np.allclose(np.asarray(head), np.asarray(neutral_head), atol=1e-10))
                if axis == (0, 1, 0):
                    self.assertGreater(camera.TransformDir(Gf.Vec3d(0, 0, -1))[1] * degrees, 0)
                if axis == (1, 0, 0):
                    self.assertGreater(camera.TransformDir(Gf.Vec3d(0, 0, -1))[2], 0)
                if axis == (0, 0, 1):
                    self.assertGreater(abs(camera.TransformDir(Gf.Vec3d(0, 1, 0))[1]), 0.1)

    def test_deep_downward_and_backward_looks_keep_head_geometry_behind_the_eye(self):
        neutral = self.example._get_head_camera_pose()
        neutral_attachment = self.head_world() * neutral.GetInverse()
        visual = self.example._operator_head_visual
        for yaw, pitch, roll in ((0, -60, 0), (0, -85, 0), (120, 0, 0), (-120, 0, 0), (180, 0, 0), (90, -60, 30)):
            with self.subTest(yaw=yaw, pitch=pitch, roll=roll):
                rotation = (
                    Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), roll))
                    * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), pitch))
                    * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), yaw))
                )
                self.example._read_physical_head_pose.return_value = rotation
                camera = self.example._get_head_camera_pose(dt=0)
                self.assert_matrix_close(camera, self.expected_hmd_view(neutral, rotation))
                head = self.assert_visible_head_anchored_to_neck()
                relative_head = head * camera.GetInverse()
                self.assert_matrix_close(relative_head, neutral_attachment)
                # This is a geometric clearance check on a level, static torso,
                # not a claim about native stereo rendering or arbitrary sway.
                for x in (visual.bounds_min[0], visual.bounds_max[0]):
                    for y in (visual.bounds_min[1], visual.bounds_max[1]):
                        for z in (visual.bounds_min[2], visual.bounds_max[2]):
                            self.assertGreater(relative_head.Transform(Gf.Vec3d(x, y, z))[2], 0.0)
                self.assertIs(self.example._eye_gaze_tracker, self.gaze)

    def test_camera_update_restores_head_hidden_by_old_session_without_moving_eye(self):
        expected = self.example._get_head_camera_pose()
        camera = UsdGeom.Camera.Define(self.stage, self.example._head_camera_path)
        self.example._head_camera_transform_op = camera.AddTransformOp()
        self.example._xr_camera_can_schedule = Mock(return_value=False)
        visual = self.example._operator_head_visual
        self.assertTrue(visual.set_first_person_hidden(True))
        self.example._update_head_camera_view(force=True, dt=0)
        self.assert_matrix_close(self.example._head_camera_transform_op.Get(), expected)
        self.assert_visible_head_anchored_to_neck()
        self.assertIs(self.example._eye_gaze_tracker, self.gaze)

    def test_comfort_and_rigid_mount_both_keep_the_visible_head_on_the_neck(self):
        self.example._get_head_camera_pose()
        for comfort_enabled in (True, False, True):
            with self.subTest(comfort_enabled=comfort_enabled):
                self.example._head_camera_comfort_enabled = comfort_enabled
                self.example._read_physical_head_pose.return_value = (
                    Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 45))
                    * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 20))
                    * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), -15))
                )
                self.example._get_head_camera_pose()
                self.assert_visible_head_anchored_to_neck()
                self.assertIs(self.example._eye_gaze_tracker, self.gaze)

    def test_combined_head_rotation_keeps_neck_on_live_torso_while_comfort_view_ignores_body_sway(self):
        neutral_camera = self.example._get_head_camera_pose()
        authored_torso = UsdGeom.XformCache().GetLocalToWorldTransform(self.torso.GetPrim())
        original_torso_local = Gf.Matrix4d(self.torso_op.Get())
        for yaw, pitch, roll, body_degrees in ((60, 25, -20, 18), (-55, -30, 22, -16), (0, 0, 0, 12)):
            with self.subTest(yaw=yaw, pitch=pitch, roll=roll, body_degrees=body_degrees):
                hmd = (
                    Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), yaw))
                    * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), pitch))
                    * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), roll))
                )
                self.example._read_physical_head_pose.return_value = hmd
                measured_torso = (
                    Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3).GetNormalized(), body_degrees))
                    * authored_torso
                )
                measured_torso.SetTranslateOnly(authored_torso.ExtractTranslation() + Gf.Vec3d(0.03, -0.02, 0.04))
                # Keep USD stale deliberately. The render correction must remain
                # attached when combined with the real measured torso pose.
                with patch.object(self.example, "_read_camera_mount_body_pose", return_value=measured_torso):
                    camera = self.advance()
                    self.assert_matrix_close(camera, self.expected_hmd_view(neutral_camera, hmd))
                    self.assert_visible_head_anchored_to_neck(measured_torso)
                self.assert_matrix_close(self.torso_op.Get(), original_torso_local)
                self.assertIs(self.example._eye_gaze_tracker, self.gaze)

    def test_physical_head_translation_cannot_move_the_stabilized_eye(self):
        self.example._get_head_camera_pose()
        rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -60)) * Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(0, 1, 0), 120)
        )
        self.example._read_physical_head_pose.return_value = rotation
        expected = self.example._get_head_camera_pose()
        expected_head = self.head_world()
        for translation in ((0, 0.6, 0), (4, 2.1, -3), (-2, 1.2, 5)):
            translated = Gf.Matrix4d(rotation)
            translated.SetTranslateOnly(Gf.Vec3d(*translation))
            self.example._read_physical_head_pose.return_value = translated
            self.assert_matrix_close(self.advance(), expected)
            self.assert_matrix_close(self.head_world(), expected_head)

    def test_application_zero_dt_does_not_restore_raw_robot_pose(self):
        self.example._get_head_camera_pose()
        self.example._base_command[:] = (0.4, 0, 0.2)
        self.root_op.Set(pose((1, 0, 1.5), 40))
        filtered = self.advance(elapsed=0.04, dt=0.005)
        self.root_op.Set(pose((5, 4, 2), -80))
        self.assert_matrix_close(self.example._get_head_camera_pose(dt=0), filtered)
        self.assertLess(filtered.ExtractTranslation()[0], 0.5)
        self.assertAlmostEqual(filtered.ExtractTranslation()[2], 1.2738700094)

    def test_walking_calibrates_once_after_settling_and_transitions_height(self):
        self.example._g1_locomotion = "policy"
        initial = self.example._get_head_camera_pose()
        self.example._physics_ready = True
        self.example._timeline_stopped = False
        self.example._get_g1_base_pose_for_arms = Mock(return_value=(Gf.Vec3d(0, 0, 0.8), 0))
        mounted = self.example._get_robot_head_camera_pose()
        mounted.SetTranslateOnly(mounted.ExtractTranslation() + Gf.Vec3d(0, 0, -0.1))
        self.example._get_robot_head_camera_pose = Mock(return_value=mounted)
        self.example.g1._walk_time = 0.49
        self.assert_matrix_close(self.advance(), initial)
        self.assertFalse(self.example._head_camera_comfort_height_calibrated)
        self.example.g1._walk_time = 0.51
        calibrated = self.advance()
        self.assertTrue(self.example._head_camera_comfort_height_calibrated)
        self.assertLess(calibrated.ExtractTranslation()[2], initial.ExtractTranslation()[2])
        self.assertGreater(calibrated.ExtractTranslation()[2], mounted.ExtractTranslation()[2])
        target = mounted.ExtractTranslation()[2]
        mounted.SetTranslateOnly(mounted.ExtractTranslation() + Gf.Vec3d(0, 0, 0.3))
        for unused in range(100):
            camera = self.advance(elapsed=0.04)
        self.assertAlmostEqual(camera.ExtractTranslation()[2], target, delta=1e-5)

    def test_reset_and_mount_recreation_discard_previous_calibration_and_history(self):
        self.example._get_head_camera_pose()
        self.example._head_camera_comfort_height_calibrated = True
        self.example._head_camera_comfort_filter.set_eye_height(9)
        self.example._release_grabbed_object = Mock()
        self.example._set_grab_candidate = Mock()
        self.example._reset_teleoperation_state()
        self.assertFalse(self.example._head_camera_comfort_height_calibrated)
        self.assertIsNone(self.example._head_camera_comfort_filter._last_time_s)
        self.assertIsNone(self.example._head_camera_comfort_filter._eye_height_target)
        self.example._get_head_camera_pose()
        self.example._head_camera_comfort_height_calibrated = True
        self.example._prepare_robot_head_camera_mount(self.stage)
        self.assertFalse(self.example._head_camera_comfort_height_calibrated)
        self.assertIsNone(self.example._head_camera_comfort_filter._last_time_s)

    def test_missing_running_base_never_falls_back_to_authored_pose(self):
        self.example._get_head_camera_pose()
        self.example._physics_ready = True
        self.example._timeline_stopped = False
        self.example._get_g1_base_pose_for_arms = Mock(return_value=None)
        self.assertIsNone(self.advance())


class TestCameraStageLifecycle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.example = HUMANOID.HumanoidExample()
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)
        self.example._get_head_camera_pose = Mock(side_effect=AssertionError("Accessed a replaced stage"))
        self.example._eye_gaze_tracker = object()

    def install_stale_handles(self):
        self.example._head_camera_update_sub = object()
        self.example._head_camera_transform_op = object()
        self.example._head_camera_mount_prim = object()
        self.example._head_camera_operator_neutral_rotation = object()
        self.example._head_camera_operator_delta_rotation = object()

    def assert_detached(self):
        self.assertIsNone(self.example._head_camera_update_sub)
        self.assertIsNone(self.example._head_camera_transform_op)
        self.assertIsNone(self.example._head_camera_mount_prim)
        self.assertIsNone(self.example._head_camera_operator_neutral_rotation)
        self.assertIsNone(self.example._head_camera_operator_delta_rotation)

    async def test_clear_and_load_detach_before_base_stage_replacement_and_queued_frame(self):
        tracker = self.example._eye_gaze_tracker
        for method in ("clear_async", "load_world_async"):
            with self.subTest(method=method):
                self.install_stale_handles()
                entered = []

                async def replace_stage(instance):
                    self.assertIs(instance, self.example)
                    self.assert_detached()  # Before the superclass's first await.
                    entered.append(True)
                    await asyncio.sleep(0)
                    # A queued frame can run after a new stage exists but before
                    # base cleanup. It must not inspect the previous stage's body.
                    self.stage = Usd.Stage.CreateInMemory()
                    instance._on_robot_camera_update(None)
                    self.assert_detached()

                with patch.object(BaseSample, method, replace_stage, create=True):
                    await getattr(self.example, method)()
                self.assertEqual(entered, [True])
                self.example._get_head_camera_pose.assert_not_called()
                self.assertIs(self.example._eye_gaze_tracker, tracker)

    async def test_load_preserves_new_camera_handles_created_by_base_lifecycle(self):
        self.install_stale_handles()
        fresh = (object(), object(), object())

        async def load_stage(instance):
            self.assert_detached()
            await asyncio.sleep(0)
            instance._head_camera_update_sub, instance._head_camera_transform_op, instance._head_camera_mount_prim = (
                fresh
            )

        with patch.object(BaseSample, "load_world_async", load_stage, create=True):
            await self.example.load_world_async()
        self.assertEqual(
            (
                self.example._head_camera_update_sub,
                self.example._head_camera_transform_op,
                self.example._head_camera_mount_prim,
            ),
            fresh,
        )

    async def test_failed_stage_load_keeps_old_handles_detached(self):
        self.install_stale_handles()

        async def failed_load(instance):
            self.assert_detached()
            await asyncio.sleep(0)
            raise RuntimeError("Stage load failed")

        with patch.object(BaseSample, "load_world_async", failed_load, create=True):
            with self.assertRaisesRegex(RuntimeError, "Stage load failed"):
                await self.example.load_world_async()
        self.assert_detached()
        self.example._on_robot_camera_update(None)
        self.example._get_head_camera_pose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
