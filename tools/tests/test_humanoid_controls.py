# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral regressions for tracking, reaching, pickup, and locomotion inputs."""

from __future__ import annotations

import math
import random
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from humanoid_test_support import HIGHLIGHTS, HUMANOID, POSE, USD, Array, Device, pose
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


class TestTrackingControls(unittest.TestCase):
    def test_rejects_invalid_descriptor_without_using_stale_matrix(self):
        device = Device(pose((2, 3, 4)), flags=0)
        device.get_virtual_world_pose = Mock(return_value=pose((2, 3, 4)))
        self.assertIsNone(POSE.read_world_pose(device))
        device.get_virtual_world_pose.assert_not_called()

    def test_valid_identity_is_accepted_but_legacy_identity_is_not(self):
        self.assertEqual(POSE.read_world_pose(Device(pose())), pose())
        self.assertIsNone(POSE.read_world_pose(types.SimpleNamespace(get_virtual_world_pose=lambda name: pose())))

    def test_malformed_and_nonfinite_tracking_are_rejected(self):
        for value in (None, [[1]], pose((math.nan, 0, 1)), Gf.Matrix4d(0)):
            with self.subTest(value=value):
                self.assertIsNone(POSE.read_world_pose(Device(value)))

    def test_filter_response_does_not_depend_on_physics_rate(self):
        results = []
        for hz in (60, 100, 120, 200):
            value = 0.0
            for _ in range(hz):
                value += (1.0 - value) * POSE.smoothing_alpha(0.01, 1.0 / hz)
            results.append(value)
        np.testing.assert_allclose(results, results[0], atol=1e-12)


class TestHumanoidControls(unittest.TestCase):
    def setUp(self):
        self.example = HUMANOID.HumanoidExample()
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)

    def test_referenced_package_fallback_authors_collision_on_nested_geometry(self) -> None:
        """Dynamic package colliders must use the same mesh's convex approximation."""
        asset = Usd.Stage.CreateInMemory()
        asset_root = UsdGeom.Xform.Define(asset, "/Package")
        asset.SetDefaultPrim(asset_root.GetPrim())
        asset_root.AddScaleOp().Set(Gf.Vec3f(0.01))
        UsdPhysics.RigidBodyAPI.Apply(asset_root.GetPrim())
        UsdGeom.Xform.Define(asset, "/Package/Geometry")
        mesh = UsdGeom.Mesh.Define(asset, "/Package/Geometry/Mesh")
        mesh.CreatePointsAttr([(-20, -20, 0), (20, -20, 0), (0, 20, 0), (0, 0, 40)])
        mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3])
        source_before = asset.GetRootLayer().ExportToString()

        def add_reference(usd_path: str, path: str) -> None:
            self.stage.DefinePrim(path).GetReferences().AddReference(usd_path)

        ex = self.example
        ex._package_usd_paths = (asset.GetRootLayer().identifier,)
        with (
            patch.object(HUMANOID, "get_assets_root_path", return_value="", create=True),
            patch.object(
                HUMANOID, "stage_utils", types.SimpleNamespace(add_reference_to_stage=add_reference), create=True
            ),
        ):
            ex._create_package_from_prop("/World/Box", random.Random(12), 1.0, 2.0, 0.8)

        package = self.stage.GetPrimAtPath("/World/Box")
        referenced_root = self.stage.GetPrimAtPath("/World/Box/Asset")
        collision_mesh = self.stage.GetPrimAtPath("/World/Box/Asset/Geometry/Mesh")
        self.assertTrue(package.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertFalse(referenced_root.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertFalse(referenced_root.HasAPI(UsdPhysics.CollisionAPI))
        self.assertTrue(collision_mesh.HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(UsdPhysics.MeshCollisionAPI(collision_mesh).GetApproximationAttr().Get(), "convexHull")
        self.assertEqual(referenced_root.GetAttribute("xformOp:scale").Get(), Gf.Vec3f(0.01))
        self.assertEqual(asset.GetRootLayer().ExportToString(), source_before)

    def test_controller_clutch_ignores_robot_translation_and_turn(self):
        ex = self.example
        ex._get_hand_link_prim = Mock(return_value=None)
        first = ex._compute_arm_target_body_position("right", pose((0.3, -0.2, 1)), Gf.Vec3d(0, 0, 0.8), 0, True)
        base = Gf.Vec3d(5, 2, 0.8)
        hand = ex._g1_body_point_to_stage(Gf.Vec3d(0.3, -0.2, 0.2), base, math.pi / 2)
        second = ex._compute_arm_target_body_position("right", pose(hand), base, math.pi / 2, True)
        np.testing.assert_allclose(second, first, atol=1e-12)
        hand += Gf.Vec3d(0, 0.1, 0)
        third = ex._compute_arm_target_body_position("right", pose(hand), base, math.pi / 2, True)
        self.assertAlmostEqual(third[0] - first[0], 0.13)

    def test_wrist_orientation_preserves_calibrated_axes_and_tracks_delta(self):
        ex = self.example
        ex._get_hand_link_prim = Mock(
            return_value=types.SimpleNamespace(get_world_poses=lambda: (Array([[0, 0, 0]]), Array([[1, 0, 0, 0]])))
        )
        initial = ex._compute_arm_target_orientation("right", pose(yaw=40), 0)
        changed = ex._compute_arm_target_orientation("right", pose(yaw=70), 0)
        np.testing.assert_allclose(initial, pose(), atol=1e-12)
        np.testing.assert_allclose(changed, pose(yaw=30), atol=1e-12)

    def test_arm_reacquisition_is_limited_from_measured_pose(self):
        ex = self.example
        ex.g1 = types.SimpleNamespace(robot=types.SimpleNamespace(get_dof_positions=lambda: Array([[0.0]])))
        ex._g1_arm_joint_limits = {0: (-0.5, 0.5)}
        ex._last_physics_dt = 0.01
        first = ex._smooth_and_clamp_arm_targets({0: 2.0})
        self.assertGreater(first[0], 0)
        self.assertLessEqual(first[0], 0.025)
        previous = first[0]
        for _ in range(50):  # A stalled joint may build only a bounded drive error.
            result = ex._smooth_and_clamp_arm_targets({0: 2.0})
            self.assertLessEqual(result[0], ex._arm_max_tracking_error)
            self.assertLessEqual(result[0] - previous, 0.025 + 1e-12)
            previous = result[0]
        self.assertGreater(result[0], 0.025)  # A position drive needs tracking error to move.
        self.assertEqual(ex._smooth_and_clamp_arm_targets({0: math.nan}), {})
        ex._last_physics_dt = 0.1
        self.assertLessEqual(ex._smooth_and_clamp_arm_targets({0: 2.0})[0], ex._arm_max_tracking_error)

    def test_partial_hand_skeleton_does_not_count_as_a_closed_hand(self):
        ex = self.example
        ex._latest_finger_curls["left"] = {"thumb": 1.0}
        ex._finger_curl_source["left"] = "hand_tracking"
        self.assertFalse(ex._is_hand_closed("left", None))
        self.assertAlmostEqual(ex._get_hand_closure("left"), 0.2)

    def test_controller_grip_clutch_leaves_fingers_open_until_trigger(self):
        ex = self.example
        ex._get_xr_gesture_value = lambda device, name, gesture: 1.0 if name in ("grip", "squeeze") else 0.0
        curls = ex._get_controller_finger_curls(Device(pose()))
        self.assertTrue(all(value == 0.0 for value in curls.values()))
        ex._get_xr_gesture_value = lambda device, name, gesture: 0.75 if name == "trigger" else 0.0
        self.assertTrue(all(value == 0.75 for value in ex._get_controller_finger_curls(Device(pose())).values()))

    def test_disabling_tracking_releases_objects_and_clutch_calibration(self):
        ex = self.example
        ex._grabbed_objects_by_side["right"] = "/World/Box"
        ex._controller_arm_neutral_positions["right"] = Gf.Vec3d(1, 2, 3)
        ex._xr_core = None
        ex._update_g1_arms_from_hand_tracking()
        self.assertEqual(ex._grabbed_objects_by_side, {})
        self.assertEqual(ex._controller_arm_neutral_positions, {})
        self.assertTrue(ex._grab_requires_release["right"])

    def test_grasp_hysteresis_requires_open_then_close(self):
        ex = self.example
        ex._get_xr_gesture_value = lambda device, input_name, gesture: device if gesture == "value" else 0
        self.assertTrue(ex._is_hand_closed("right", 0.7))
        self.assertTrue(ex._is_hand_closed("right", 0.5))
        self.assertFalse(ex._is_hand_closed("right", 0.2))
        self.assertFalse(ex._is_hand_closed("right", 0.5))

    def test_drop_cannot_regrab_while_trigger_is_still_held(self):
        ex = self.example
        ex._grabbed_objects_by_side["right"] = "/World/Box"
        ex._get_active_hand_world_position = Mock(return_value=Gf.Vec3d(0, 0, 1))
        ex._find_nearest_grabbable_object = Mock(return_value=(None, None))
        ex._set_grab_candidate = Mock()
        ex._drop_everything()
        ex._update_grabbed_object("right", True)
        self.assertNotIn("right", ex._grabbed_objects_by_side)
        ex._find_nearest_grabbable_object.assert_not_called()
        ex._update_grabbed_object("right", False)
        ex._update_grabbed_object("right", True)
        self.assertTrue(ex._find_nearest_grabbable_object.called)

    def test_grasp_joint_preserves_world_attachment_frames(self):
        ex = self.example
        hand_matrix = pose((1, 2, 3), yaw=75)
        object_matrix = pose((1.05, 2.1, 3.02), yaw=-25)
        for path in ("/World/Hand", "/World/Box"):
            UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(self.stage, path).GetPrim())
        quaternion = hand_matrix.ExtractRotationQuat()
        ex._get_hand_link_path = Mock(return_value="/World/Hand")
        ex._get_hand_link_prim = Mock(
            return_value=types.SimpleNamespace(
                get_world_poses=lambda: (
                    Array([list(hand_matrix.ExtractTranslation())]),
                    Array([[quaternion.GetReal(), *quaternion.GetImaginary()]]),
                )
            )
        )
        ex._get_object_world_pose = Mock(
            return_value=(object_matrix.ExtractTranslation(), object_matrix.ExtractRotationQuat())
        )
        self.assertTrue(ex._create_grasp_joint("right", "/World/Box"))
        joint = UsdPhysics.FixedJoint.Get(self.stage, ex._grasp_joints_by_side["right"])
        local0 = Gf.Matrix4d().SetRotate(Gf.Quatd(joint.GetLocalRot0Attr().Get()))
        local0.SetTranslateOnly(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
        local1 = Gf.Matrix4d().SetRotate(Gf.Quatd(joint.GetLocalRot1Attr().Get()))
        local1.SetTranslateOnly(Gf.Vec3d(joint.GetLocalPos1Attr().Get()))
        np.testing.assert_allclose(local0 * hand_matrix, local1 * object_matrix, atol=2e-7)
        self.assertTrue(joint.GetExcludeFromArticulationAttr().Get())
        ex._destroy_grasp_joint("right")
        self.assertFalse(self.stage.GetPrimAtPath("/World/G1_GraspJoints/right_grasp").IsValid())

    def test_failed_hand_pose_never_grasps_at_desired_marker(self):
        ex = self.example
        ex._active_g1_hand_target_matrices["right"] = pose((1, 2, 3))
        ex._get_hand_link_prim = Mock(return_value=None)
        self.assertIsNone(ex._get_active_hand_world_position("right"))

    def test_grab_search_uses_measured_palm_not_wrist_or_target(self):
        ex = self.example
        ex._active_g1_hand_target_matrices["right"] = pose((10, 20, 30))
        ex._get_hand_link_prim = Mock(
            return_value=types.SimpleNamespace(get_world_poses=lambda: (Array([[1, 2, 3]]), None))
        )
        ex._palm_offset_world = Mock(return_value=Gf.Vec3d(0.10, 0, -0.04))
        np.testing.assert_allclose(ex._get_active_hand_world_position("right"), (1.1, 2, 2.96))
        ex._active_g1_hand_target_matrices.clear()
        self.assertIsNone(ex._get_active_hand_world_position("right"))

    def test_brake_zeroes_translation_but_preserves_turning(self):
        ex = self.example
        ex._g1_locomotion = "kinematic"  # Exercise optional moving-avatar controls.
        ex._base_command = torch.tensor([0.5, 0.3, 0.0])
        ex._locomotion_brake = True
        ex._smooth_base_command(torch.tensor([0.0, 0.0, 1.0]), 0.01)
        self.assertEqual(ex._base_command[0], 0)
        self.assertEqual(ex._base_command[1], 0)
        self.assertGreater(ex._base_command[2], 0)

    def test_keyboard_repeat_and_unmatched_release_do_not_accumulate_motion(self):
        ex = self.example
        ex._g1_locomotion = "kinematic"
        ex._base_command = torch.zeros(3)
        ex._keyboard_command = torch.zeros(3)
        for kind in (1, 1, 3, 2, 2):
            ex._sub_keyboard_event(types.SimpleNamespace(input="UP", type=kind))
        self.assertTrue(torch.equal(ex._keyboard_command, torch.zeros(3)))
        for key, kind in (("DOWN", 1), ("NUMPAD_2", 1), ("DOWN", 3), ("DOWN", 2)):
            ex._sub_keyboard_event(types.SimpleNamespace(input=key, type=kind))
        self.assertTrue(ex._keyboard_brake)

    def test_stick_right_commands_body_right_and_trigger_does_not_walk(self):
        ex = self.example
        ex._g1_locomotion = "kinematic"
        ex._get_xr_input_device = lambda handle: handle
        ex._log_xr_input_status_once = Mock()
        ex._get_xr_gesture_value = lambda device, name, gesture: 1.0 if name == "trigger" else 0.0
        ex._get_xr_stick_axis = lambda device, axis: 1.0 if device.endswith("left") and axis == "x" else 0.0
        forward, yaw = ex._read_xr_controller_axes()
        self.assertEqual((forward, yaw), (0.0, 0.0))
        self.assertEqual(ex._latest_stick_lateral, -1.0)

    def test_reset_discards_stale_input_and_physics_handles(self):
        ex = self.example
        ex._base_command = torch.ones(3)
        ex._keyboard_command = torch.ones(3)
        ex._controller_command = torch.ones(3)
        ex._pressed_keys.add("UP")
        ex._hand_link_prims["left"] = object()
        ex._arm_ik_link_index["left"] = 10
        ex._controller_arm_neutral_positions["left"] = Gf.Vec3d(10, 0, 1)
        ex._headset_gait_time = 12.5
        ex._reset_teleoperation_state()
        self.assertEqual(ex._hand_link_prims, {})
        self.assertEqual(ex._arm_ik_link_index, {})
        self.assertEqual(ex._controller_arm_neutral_positions, {})
        self.assertEqual(ex._pressed_keys, set())
        self.assertFalse(ex._base_command.any())
        self.assertEqual(ex._headset_gait_time, 12.5)


class TestMaterialControls(unittest.TestCase):
    def test_overlapping_owners_restore_inherited_material_exactly(self):
        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/World/Box").GetPrim()
        mesh = UsdGeom.Cube.Define(stage, "/World/Box/Mesh").GetPrim()
        original = UsdShade.Material.Define(stage, "/World/Original")
        green = UsdShade.Material.Define(stage, "/World/Green")
        yellow = UsdShade.Material.Define(stage, "/World/Yellow")
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(original)
        before = stage.GetRootLayer().ExportToString()
        highlights = HIGHLIGHTS.MaterialHighlights()
        highlights.prepare(stage, [str(root.GetPath())])
        prepared_layers = list(stage.GetSessionLayer().subLayerPaths)
        highlights.set(stage, "gaze", str(root.GetPath()), yellow, 10)
        highlights.set(stage, "grab:left", str(root.GetPath()), green, 20)
        highlights.set(stage, "grab:right", str(root.GetPath()), green, 20)
        binding = UsdShade.MaterialBindingAPI(mesh)
        self.assertEqual(binding.ComputeBoundMaterial()[0].GetPath(), green.GetPath())
        highlights.set(stage, "grab:left", None, None)
        self.assertEqual(binding.ComputeBoundMaterial()[0].GetPath(), green.GetPath())
        highlights.set(stage, "grab:right", None, None)
        self.assertEqual(binding.ComputeBoundMaterial()[0].GetPath(), yellow.GetPath())
        highlights.set(stage, "gaze", None, None)
        self.assertEqual(binding.ComputeBoundMaterial()[0].GetPath(), original.GetPath())
        self.assertEqual(before, stage.GetRootLayer().ExportToString())
        self.assertEqual(list(stage.GetSessionLayer().subLayerPaths), prepared_layers)

    def test_runtime_highlights_never_resync_collision_prims(self):
        """A correct final USD tree alone misses destructive intermediate notices."""
        from pxr import Tf

        asset = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(asset, "/Package")
        mesh = UsdGeom.Cube.Define(asset, "/Package/Mesh").GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh)
        stage = Usd.Stage.CreateInMemory()
        targets = ["/World/Box_00", "/World/Box_01"]
        for target in targets:
            prim = UsdGeom.Xform.Define(stage, target).GetPrim()
            prim.GetReferences().AddReference(asset.GetRootLayer().identifier, "/Package")
            UsdPhysics.RigidBodyAPI.Apply(prim)
        green = UsdShade.Material.Define(stage, "/World/Green")
        yellow = UsdShade.Material.Define(stage, "/World/Yellow")
        before = stage.GetRootLayer().ExportToString()
        highlights = HIGHLIGHTS.MaterialHighlights()
        highlights.prepare(stage, targets)
        prepared_layers = list(stage.GetSessionLayer().subLayerPaths)
        resynced = []

        def record_notice(notice, sender):
            resynced.extend(notice.GetResyncedPaths())

        listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, record_notice, stage)
        try:
            for _ in range(3):
                highlights.set(stage, "gaze", targets[0], yellow, 10)
                highlights.set(stage, "grab:right", targets[0], green, 20)
                highlights.set(stage, "gaze", targets[1], yellow, 10)
                highlights.set(stage, "grab:right", None, None)
                highlights.clear()
                highlights.prepare(stage, targets)  # Reset may prepare existing roots again.
        finally:
            listener.Revoke()
        self.assertEqual([str(path) for path in resynced if path.IsPrimPath()], [])
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)
        self.assertEqual(list(stage.GetSessionLayer().subLayerPaths), prepared_layers)

    def test_unprepared_target_does_not_modify_stage_during_physics(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(stage, "/World/Unprepared")
        material = UsdShade.Material.Define(stage, "/World/Highlight")
        highlights = HIGHLIGHTS.MaterialHighlights()
        highlights.prepare(stage, [])
        before = highlights._layer.ExportToString()
        highlights.set(stage, "gaze", "/World/Unprepared", material, 10)
        self.assertEqual(highlights._layer.ExportToString(), before)
        self.assertEqual(highlights._owners, {})

    def test_cleanup_discards_an_invalid_closed_stage_wrapper(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(stage, "/World/Box")
        material = UsdShade.Material.Define(stage, "/World/Highlight")
        highlights = HIGHLIGHTS.MaterialHighlights()
        highlights.prepare(stage, ["/World/Box"])
        highlights.set(stage, "gaze", "/World/Box", material, 10)
        # A stage closed by Kit can retain its Python wrapper but fail Boost.Python
        # argument conversion. An incompatible wrapper exercises that same failure.
        highlights._stage = object()
        highlights.clear()
        self.assertIsNone(highlights._stage)
        self.assertIsNone(highlights._layer)
        self.assertEqual(highlights._owners, {})
        self.assertEqual(highlights._prepared, set())
        highlights.clear()  # Teardown callbacks may run more than once.
