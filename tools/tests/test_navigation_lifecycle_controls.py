# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Stop/Play recovery and reject XR writes during display transitions."""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from humanoid_test_support import HUMANOID, USD, Array, pose
from pxr import Gf, Usd, UsdGeom


class TestNavigationTimelineLifecycle(unittest.TestCase):
    """Disarm navigation without hiding an existing physics fault."""

    def setUp(self) -> None:
        self.ex = HUMANOID.HumanoidExample()
        self.ex.g1 = types.SimpleNamespace(
            robot=Mock(),
            invalidate_physics_state=Mock(),
            initialize=Mock(),
            post_reset=Mock(),
            forward=Mock(),
            _initialized=True,
        )
        self.ex.g1.robot.is_physics_tensor_entity_valid.return_value = True
        self.ex._reset_gesture_locomotion = Mock()
        self.ex._reset_teleoperation_state = Mock()
        self.ex._update_recorded_locomotion_mode = Mock()
        self.ex._configure_g1_arm_dofs = Mock()
        self.ex._update_head_camera_view = Mock()
        self.ex._log_physics_step_error = Mock()
        self.ex._pause_faulted_articulation = Mock()
        self.ex._contact_reader = Mock()
        self.ex._eye_gaze_tracker = object()
        self.ex._physics_ready = True
        self.ex._timeline_stopped = False
        self.types = types.SimpleNamespace(STOP=0, PAUSE=1, PLAY=2)
        timeline = types.SimpleNamespace(TimelineEventType=self.types)
        self.import_patch = patch.object(HUMANOID, "import_module", return_value=timeline)
        self.import_patch.start()
        self.addCleanup(self.import_patch.stop)

    def event(self, event_type: int) -> None:
        self.ex._on_navigation_timeline_event(types.SimpleNamespace(type=event_type))

    def test_stop_discards_native_handles_and_preserves_fault_and_gaze(self) -> None:
        gaze = self.ex._eye_gaze_tracker
        self.ex._articulation_health_fault = "Existing balance fault"
        self.ex._head_camera_mount_prim = object()
        self.ex._xr_last_head_pose = pose((0, 1.6, 0))
        self.ex._xr_head_pose_age = 2
        mappings = (self.ex._hand_link_prims, self.ex._grasp_link_prims, self.ex._object_rigid_prims)
        for mapping in mappings:
            mapping["old"] = object()

        self.event(self.types.STOP)

        self.ex._reset_gesture_locomotion.assert_called_once()
        self.ex.g1.invalidate_physics_state.assert_called_once()
        self.ex._contact_reader.invalidate.assert_called_once()
        self.assertFalse(self.ex._physics_ready)
        self.assertTrue(self.ex._timeline_stopped)
        self.assertIsNone(self.ex._head_camera_mount_prim)
        self.assertIsNone(self.ex._xr_last_head_pose)
        self.assertTrue(all(not mapping for mapping in mappings))
        self.assertEqual(self.ex._articulation_health_fault, "Existing balance fault")
        self.assertIs(self.ex._eye_gaze_tracker, gaze)
        self.ex.g1.robot.get_world_poses.assert_not_called()

    def test_pause_disarms_without_destroying_the_current_physics_episode(self) -> None:
        mount = self.ex._head_camera_mount_prim = object()
        hand = self.ex._hand_link_prims["left"] = object()
        self.event(self.types.PAUSE)
        self.ex._reset_gesture_locomotion.assert_called_once()
        self.assertTrue(self.ex._physics_ready)
        self.assertFalse(self.ex._timeline_stopped)
        self.assertIs(self.ex._head_camera_mount_prim, mount)
        self.assertIs(self.ex._hand_link_prims["left"], hand)
        self.ex.g1.invalidate_physics_state.assert_not_called()
        self.ex._contact_reader.invalidate.assert_not_called()

    def test_play_event_does_not_initialize_or_bypass_fault(self) -> None:
        self.ex._articulation_health_fault = "Previous failed episode"
        self.event(self.types.PLAY)
        self.assertEqual(self.ex._articulation_health_fault, "Previous failed episode")
        self.ex.g1.initialize.assert_not_called()
        self.ex._reset_gesture_locomotion.assert_not_called()

    def test_invalid_view_waits_without_initializing_or_sampling_old_episode(self) -> None:
        self.ex.g1.robot.is_physics_tensor_entity_valid.return_value = False
        self.ex.on_physics_step(0.005, None)
        self.assertFalse(self.ex._physics_ready)
        self.assertTrue(self.ex._timeline_stopped)
        self.ex.g1.invalidate_physics_state.assert_called_once()
        self.ex.g1.initialize.assert_not_called()
        self.ex.g1.forward.assert_not_called()
        self.ex._reset_teleoperation_state.assert_not_called()
        self.ex._update_head_camera_view.assert_not_called()
        self.assertIsNone(self.ex._articulation_health_fault)

    def test_stop_then_valid_step_reinitializes_before_any_motion(self) -> None:
        self.event(self.types.STOP)
        self.ex.on_physics_step(0.005, None)
        self.ex.g1.initialize.assert_called_once()
        self.ex.g1.post_reset.assert_called_once()
        self.ex._configure_g1_arm_dofs.assert_called_once()
        self.ex._update_head_camera_view.assert_called_once()
        self.ex.g1.forward.assert_not_called()
        self.assertTrue(self.ex._physics_ready)
        self.assertFalse(self.ex._timeline_stopped)

    def test_reset_post_hook_does_not_repeat_the_first_play_initialization(self) -> None:
        """Base reset awaits one Play update before it invokes the post hook."""
        self.event(self.types.STOP)
        self.ex.on_physics_step(0.005, None)
        asyncio.run(self.ex.setup_post_reset())
        self.ex._check_articulation_health = Mock(return_value=False)

        self.ex.on_physics_step(0.005, None)

        self.ex._check_articulation_health.assert_called_once()
        self.ex.g1.initialize.assert_called_once()
        self.ex.g1.post_reset.assert_called_once()
        self.ex._reset_teleoperation_state.assert_called_once()
        self.assertTrue(self.ex._physics_ready)
        self.assertFalse(self.ex._timeline_stopped)

    def test_reset_post_hook_before_first_physics_step_keeps_initialization_pending(self) -> None:
        self.event(self.types.STOP)
        asyncio.run(self.ex.setup_post_reset())
        self.assertFalse(self.ex._physics_ready)
        self.assertTrue(self.ex._timeline_stopped)

        self.ex.on_physics_step(0.005, None)

        self.ex.g1.initialize.assert_called_once()
        self.ex.g1.post_reset.assert_called_once()
        self.assertTrue(self.ex._physics_ready)

    def test_explicit_reset_clears_old_warning_deduplication_but_pause_play_preserve_it(self) -> None:
        self.ex._physics_step_error_logged = {"walking velocity"}
        self.event(self.types.PAUSE)
        self.event(self.types.PLAY)
        self.assertEqual(self.ex._physics_step_error_logged, {"walking velocity"})
        self.ex._eye_gaze_tracker = Mock()

        asyncio.run(self.ex.setup_pre_reset())

        self.assertEqual(self.ex._physics_step_error_logged, set())
        self.assertFalse(self.ex._physics_ready)
        self.assertTrue(self.ex._timeline_stopped)

    def test_failure_in_any_initialization_stage_is_contained_and_paused(self) -> None:
        for label in ("initialize", "post_reset", "recording", "arm setup", "camera"):
            with self.subTest(label=label):
                self.ex._physics_ready = False
                self.ex._articulation_health_fault = None
                operations = {
                    "initialize": self.ex.g1.initialize,
                    "post_reset": self.ex.g1.post_reset,
                    "recording": self.ex._update_recorded_locomotion_mode,
                    "arm setup": self.ex._configure_g1_arm_dofs,
                    "camera": self.ex._update_head_camera_view,
                }
                for operation in operations.values():
                    operation.reset_mock(side_effect=True)
                self.ex._pause_faulted_articulation.reset_mock()
                operations[label].side_effect = RuntimeError(f"Failed {label}")

                self.ex.on_physics_step(0.005, None)

                self.assertFalse(self.ex._physics_ready)
                self.assertIn(f"Failed {label}", self.ex._articulation_health_fault)
                self.ex._pause_faulted_articulation.assert_called_once()
                self.ex.g1.forward.assert_not_called()


class TestXRCameraLifecycleBoundary(unittest.TestCase):
    """Keep the authored mount while suppressing unsafe native XR teleport writes."""

    def setUp(self) -> None:
        self.ex = HUMANOID.HumanoidExample()
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)
        self.ex.g1 = types.SimpleNamespace(robot=Mock())
        self.ex.g1.robot.is_physics_tensor_entity_valid.return_value = True
        self.ex._physics_ready = True
        self.ex._timeline_stopped = False
        self.ex._xr_core = Mock()
        self.ex._xr_core.is_xr_enabled.return_value = True
        self.ex._xr_core.is_xr_display_enabled.return_value = True
        self.profile = Mock()
        self.profile.get_name.return_value = "vr"
        self.profile.is_enabled.return_value = True
        self.ex._xr_core.get_current_profile.return_value = self.profile
        self.head_pose = pose((0, 1.6, 0))
        self.head_flags = 3
        self.head = types.SimpleNamespace(
            get_pose_desc=lambda name: types.SimpleNamespace(pose_matrix=self.head_pose, validity_flags=self.head_flags)
        )
        self.ex._find_xr_head_device = Mock(return_value=self.head)
        self.camera_pose = pose((0.1, 0, 1.3), 10)
        self.ex._get_head_camera_pose = Mock(side_effect=lambda dt=None: self.camera_pose)
        camera = UsdGeom.Camera.Define(self.stage, self.ex._head_camera_path)
        self.ex._head_camera_transform_op = camera.AddTransformOp()

    def test_fresh_complete_session_schedules_the_identical_mounted_camera(self) -> None:
        self.ex._update_head_camera_view(force=True, dt=0.005)
        self.ex._xr_core.schedule_set_camera.assert_called_once()
        actual = self.ex._xr_core.schedule_set_camera.call_args.args[0]
        np.testing.assert_allclose(actual, self.camera_pose, atol=1e-12)
        np.testing.assert_allclose(self.ex._head_camera_transform_op.Get(), self.camera_pose, atol=1e-12)

    def test_session_or_physics_transition_blocks_xr_but_keeps_valid_usd_camera(self) -> None:
        mutations = (
            (self.ex, "_timeline_stopped", True),
            (self.ex, "_physics_ready", False),
            (self.ex.g1.robot.is_physics_tensor_entity_valid, "return_value", False),
            (self.ex._xr_core.is_xr_enabled, "return_value", False),
            (self.ex._xr_core.is_xr_display_enabled, "return_value", False),
        )
        for target, attribute, value in mutations:
            with self.subTest(attribute=attribute, target=target):
                before = getattr(target, attribute)
                setattr(target, attribute, value)
                try:
                    self.ex._update_head_camera_view(force=True, dt=0.005)
                finally:
                    setattr(target, attribute, before)
                self.ex._xr_core.schedule_set_camera.assert_not_called()
                np.testing.assert_allclose(self.ex._head_camera_transform_op.Get(), self.camera_pose, atol=1e-12)

    def test_invalid_fresh_head_cannot_use_cached_pose_to_schedule(self) -> None:
        self.ex._xr_last_head_pose = pose((0, 1.6, 0))
        for flags in (0, 1, 2):
            with self.subTest(flags=flags):
                self.head_flags = flags
                self.ex._update_head_camera_view(force=True, dt=0.005)
                self.ex._xr_core.schedule_set_camera.assert_not_called()
        self.head_flags = 3
        self.head_pose = Gf.Matrix4d(0.0)
        self.ex._update_head_camera_view(force=True, dt=0.005)
        self.ex._xr_core.schedule_set_camera.assert_not_called()

    def test_profile_disable_blocks_camera_while_native_display_flags_still_lag(self) -> None:
        gaze = self.ex._eye_gaze_tracker = object()
        self.profile.is_enabled.return_value = False
        self.ex._update_head_camera_view(force=True, dt=0.005)
        self.ex._xr_core.schedule_set_camera.assert_not_called()
        self.ex._find_xr_head_device.assert_not_called()
        np.testing.assert_allclose(self.ex._head_camera_transform_op.Get(), self.camera_pose, atol=1e-12)
        self.assertIs(self.ex._eye_gaze_tracker, gaze)
        self.assertTrue(self.ex._xr_core.is_xr_enabled())
        self.assertTrue(self.ex._xr_core.is_xr_display_enabled())
        self.profile.is_enabled.return_value = True
        self.ex._update_head_camera_view(force=True, dt=0.005)
        self.ex._xr_core.schedule_set_camera.assert_called_once()

    def test_empty_or_unavailable_profile_blocks_camera_without_altering_desktop_mount(self) -> None:
        for current in (None, self.profile):
            with self.subTest(current=current):
                self.profile.get_name.return_value = ""
                self.ex._xr_core.get_current_profile.return_value = current
                self.ex._update_head_camera_view(force=True, dt=0.005)
                self.ex._xr_core.schedule_set_camera.assert_not_called()
                np.testing.assert_allclose(self.ex._head_camera_transform_op.Get(), self.camera_pose, atol=1e-12)

    def test_unavailable_profile_state_fails_closed_during_native_transition(self) -> None:
        self.profile.is_enabled.side_effect = RuntimeError("profile transition")
        self.ex._update_head_camera_view(force=True, dt=0.005)
        self.ex._xr_core.schedule_set_camera.assert_not_called()
        np.testing.assert_allclose(self.ex._head_camera_transform_op.Get(), self.camera_pose, atol=1e-12)

    def test_invalid_camera_transform_is_not_written_to_usd_or_xr(self) -> None:
        self.ex._head_camera_transform_op.Set(self.camera_pose)
        previous = self.camera_pose
        for matrix in invalid_rigid_poses():
            with self.subTest(matrix=matrix):
                self.camera_pose = matrix
                self.ex._update_head_camera_view(force=True, dt=0.005)
                self.ex._xr_core.schedule_set_camera.assert_not_called()
                np.testing.assert_allclose(self.ex._head_camera_transform_op.Get(), previous, atol=1e-12)

    def test_rigid_pose_validation_rejects_scale_shear_reflection_and_nonaffine_values(self) -> None:
        self.assertTrue(self.ex._is_valid_rigid_pose(pose((3, -4, 2), -57)))
        for matrix in invalid_rigid_poses():
            with self.subTest(matrix=matrix):
                self.assertFalse(self.ex._is_valid_rigid_pose(matrix))
        for matrix in (None, [], [[1]], [[1, 0, 0]] * 3):
            with self.subTest(matrix=matrix):
                self.assertFalse(self.ex._is_valid_rigid_pose(matrix))

    def test_restarted_mount_view_is_recreated_before_native_pose_read(self) -> None:
        path = self.ex._head_camera_mount_body_path = "/World/G1/torso_link"
        UsdGeom.Xform.Define(self.stage, path)
        stale = self.ex._head_camera_mount_prim = Mock()
        stale.is_physics_tensor_entity_valid.return_value = False
        stale.get_world_poses.side_effect = AssertionError("Stale native view was read")
        fresh = Mock()
        fresh.is_physics_tensor_entity_valid.return_value = True
        fresh.get_world_poses.return_value = (Array([[0.2, 0.3, 1.2]]), Array([[1, 0, 0, 0]]))
        rigid = Mock(return_value=fresh)
        with patch.dict(sys.modules, {"isaacsim.core.experimental.prims": types.SimpleNamespace(RigidPrim=rigid)}):
            result = self.ex._read_camera_mount_body_pose()
        rigid.assert_called_once_with(paths=path)
        stale.get_world_poses.assert_not_called()
        np.testing.assert_allclose(result.ExtractTranslation(), [0.2, 0.3, 1.2], atol=1e-7)


def invalid_rigid_poses() -> list[Gf.Matrix4d]:
    """Return finite and nonfinite matrices that cannot describe a camera frame."""
    matrices = [Gf.Matrix4d(0.0)]
    for row, column, value in (
        (0, 0, float("nan")),
        (3, 0, float("inf")),
        (0, 0, 2.0),
        (0, 0, -1.0),
        (0, 1, 0.3),
        (0, 3, 0.2),
        (3, 3, 2.0),
    ):
        matrix = Gf.Matrix4d(1.0)
        matrix[row, column] = value
        matrices.append(matrix)
    return matrices


if __name__ == "__main__":
    unittest.main()
