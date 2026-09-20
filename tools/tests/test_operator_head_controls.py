# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify render-only G1 head rotation with real USD instances and colliders."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
    / "operator_head.py"
)
SPEC = importlib.util.spec_from_file_location("operator_head_under_test", MODULE_PATH)
HEAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HEAD)

ROBOT = "/World/G1"
TORSO = ROBOT + "/torso_link"
VISUALS = TORSO + "/visuals"
COLLISIONS = TORSO + "/collisions"
HEAD_PATH = VISUALS + "/head_link"


def define_head(stage: Usd.Stage, path: str) -> Usd.Prim:
    """Create a head-shaped box whose geometry is offset above its transform origin."""
    head = UsdGeom.Xform.Define(stage, path)
    head.AddTranslateOp().Set(Gf.Vec3d(0.004, 0.0, -0.044))
    mesh = UsdGeom.Mesh.Define(stage, path + "/mesh")
    minimum, maximum = (-0.08, -0.07, 0.32), (0.08, 0.07, 0.54)
    points = [
        (x, y, z) for z in (minimum[2], maximum[2]) for y in (minimum[1], maximum[1]) for x in (minimum[0], maximum[0])
    ]
    mesh.CreatePointsAttr(points)
    mesh.CreateExtentAttr([minimum, maximum])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return head.GetPrim()


class TestLevelHeadReference(unittest.TestCase):
    """Keep the physical gravity horizon independent of startup headset tilt."""

    @staticmethod
    def pose(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> Gf.Matrix4d:
        return (
            Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), roll))
            * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), pitch))
            * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), yaw))
        )

    def assert_matrix_close(self, actual: Gf.Matrix4d, expected: Gf.Matrix4d) -> None:
        self.assertIsNotNone(actual)
        for row in range(4):
            for column in range(4):
                self.assertAlmostEqual(actual[row][column], expected[row][column], places=10)

    def test_startup_sideways_then_worn_upright_does_not_inherit_tilt(self) -> None:
        for pitch, roll in ((25.0, 90.0), (-45.0, -90.0), (9.244, 3.846)):
            with self.subTest(pitch=pitch, roll=roll):
                reference = HEAD.compute_level_head_reference(self.pose(yaw=37.0, pitch=pitch, roll=roll))
                self.assert_matrix_close(reference, self.pose(yaw=37.0))
                worn_delta = self.pose(yaw=37.0) * reference.GetInverse()
                self.assert_matrix_close(worn_delta, Gf.Matrix4d(1.0))

    def test_calibration_preserves_current_physical_pitch_and_roll(self) -> None:
        physical = self.pose(yaw=-124.0, pitch=31.0, roll=-17.0)
        reference = HEAD.compute_level_head_reference(physical)
        self.assert_matrix_close(physical * reference.GetInverse(), self.pose(pitch=31.0, roll=-17.0))

    def test_turning_ninety_degrees_does_not_create_camera_pitch_or_roll(self) -> None:
        mount = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0), Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 0, 1)).GetInverse()
        reference = HEAD.compute_level_head_reference(self.pose(yaw=28.0, pitch=-33.0, roll=74.0))
        for yaw_delta in (-90.0, 0.0, 90.0, 179.0):
            with self.subTest(yaw_delta=yaw_delta):
                camera = self.pose(yaw=28.0 + yaw_delta) * reference.GetInverse() * mount
                self.assertLess((camera.TransformDir(Gf.Vec3d(0, 1, 0)) - Gf.Vec3d(0, 0, 1)).GetLength(), 1e-10)
                self.assertAlmostEqual(camera.TransformDir(Gf.Vec3d(0, 0, -1))[2], 0.0, places=10)

    def test_recenter_while_tilted_changes_heading_only(self) -> None:
        first_reference = HEAD.compute_level_head_reference(self.pose(yaw=10.0, roll=90.0))
        current = self.pose(yaw=85.0, pitch=24.0, roll=13.0)
        self.assert_matrix_close(current * first_reference.GetInverse(), self.pose(yaw=75.0, pitch=24.0, roll=13.0))
        recentered = HEAD.compute_level_head_reference(current)
        self.assert_matrix_close(current * recentered.GetInverse(), self.pose(pitch=24.0, roll=13.0))

    def test_near_vertical_startup_defers_until_heading_is_well_defined(self) -> None:
        for pitch in (-90.0, -89.0, -85.0, 85.0, 89.0, 90.0):
            with self.subTest(pitch=pitch):
                self.assertIsNone(HEAD.compute_level_head_reference(self.pose(yaw=47.0, pitch=pitch, roll=20.0)))
        for pitch in (-84.0, 0.0, 84.0):
            with self.subTest(pitch=pitch):
                self.assert_matrix_close(
                    HEAD.compute_level_head_reference(self.pose(yaw=47.0, pitch=pitch)), self.pose(yaw=47.0)
                )

    def test_captured_reference_preserves_later_vertical_look(self) -> None:
        reference = HEAD.compute_level_head_reference(self.pose(yaw=-63.0, pitch=15.0, roll=12.0))
        for pitch in (-90.0, 90.0):
            current = self.pose(yaw=-63.0, pitch=pitch, roll=27.0)
            self.assert_matrix_close(current * reference.GetInverse(), self.pose(pitch=pitch, roll=27.0))

    def test_translation_is_ignored_and_invalid_rotations_are_rejected(self) -> None:
        pose = self.pose(yaw=-41.0, pitch=22.0, roll=90.0)
        pose.SetTranslateOnly(Gf.Vec3d(9.0, 1.6, -20.0))
        self.assert_matrix_close(HEAD.compute_level_head_reference(pose), self.pose(yaw=-41.0))
        invalid = Gf.Matrix4d(1.0)
        invalid[0, 0] = math.nan
        for value in (
            None,
            Gf.Matrix4d(0.0),
            invalid,
            Gf.Matrix4d().SetScale(Gf.Vec3d(2.0, 1.0, 0.5)),
            Gf.Matrix4d().SetScale(Gf.Vec3d(-1.0, 1.0, 1.0)),
        ):
            with self.subTest(value=value):
                self.assertIsNone(HEAD.compute_level_head_reference(value))


class TestOperatorHeadControls(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(self.stage, ROBOT)
        root.AddTranslateOp().Set(Gf.Vec3d(0.5, -0.3, 0.8))
        self.torso = UsdGeom.Xform.Define(self.stage, TORSO)
        self.torso.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.0, 0.2))
        UsdPhysics.RigidBodyAPI.Apply(self.torso.GetPrim())
        UsdPhysics.MassAPI.Apply(self.torso.GetPrim()).CreateMassAttr(7.817)
        UsdGeom.Xform.Define(self.stage, "/RenderSource")
        define_head(self.stage, "/RenderSource/head_link")
        material = UsdShade.Material.Define(self.stage, "/World/Looks/HeadMaterial")
        UsdShade.MaterialBindingAPI.Apply(self.stage.GetPrimAtPath("/RenderSource/head_link/mesh")).Bind(material)
        self.visuals = UsdGeom.Xform.Define(self.stage, VISUALS).GetPrim()
        self.visuals.GetReferences().AddInternalReference("/RenderSource")
        self.visuals.SetInstanceable(True)
        UsdGeom.Xform.Define(self.stage, "/ColliderSource")
        define_head(self.stage, "/ColliderSource/head_link")
        collision = UsdPhysics.CollisionAPI.Apply(self.stage.GetPrimAtPath("/ColliderSource/head_link/mesh"))
        collision.CreateCollisionEnabledAttr(True)
        self.collisions = UsdGeom.Xform.Define(self.stage, COLLISIONS).GetPrim()
        self.collisions.GetReferences().AddInternalReference("/ColliderSource")
        self.collisions.SetInstanceable(True)
        self.visual = HEAD.RobotHeadVisual()
        self.addCleanup(self.visual.clear)

    def assert_matrix_close(self, actual: Gf.Matrix4d, expected: Gf.Matrix4d) -> None:
        for row in range(4):
            for column in range(4):
                self.assertAlmostEqual(actual[row][column], expected[row][column], places=10)

    def assert_vector_close(self, actual: Gf.Vec3d, expected: Gf.Vec3d) -> None:
        self.assertLess((actual - expected).GetLength(), 1e-7)

    def head_transform(self) -> Gf.Matrix4d:
        return UsdGeom.Xformable(self.stage.GetPrimAtPath(HEAD_PATH)).GetLocalTransformation()

    def test_preparation_edits_only_render_instance_and_preserves_material_and_collision(self) -> None:
        self.assertTrue(self.visuals.IsInstance())
        self.assertTrue(self.stage.GetPrimAtPath(HEAD_PATH).IsInstanceProxy())
        before = UsdGeom.Xformable(self.stage.GetPrimAtPath(COLLISIONS + "/head_link")).GetLocalTransformation()
        mass_before = UsdPhysics.MassAPI(self.torso.GetPrim()).GetMassAttr().Get()
        material = UsdShade.MaterialBindingAPI(self.stage.GetPrimAtPath(HEAD_PATH + "/mesh")).ComputeBoundMaterial()[0]
        self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
        self.assertFalse(self.visuals.IsInstanceable())
        self.assertFalse(self.stage.GetPrimAtPath(HEAD_PATH).IsInstanceProxy())
        self.assertTrue(self.collisions.IsInstance())
        self.assertTrue(self.stage.GetPrimAtPath(COLLISIONS + "/head_link").IsInstanceProxy())
        self.assertTrue(
            UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(COLLISIONS + "/head_link/mesh"))
            .GetCollisionEnabledAttr()
            .Get()
        )
        self.assert_matrix_close(
            UsdGeom.Xformable(self.stage.GetPrimAtPath(COLLISIONS + "/head_link")).GetLocalTransformation(), before
        )
        self.assertEqual(UsdPhysics.MassAPI(self.torso.GetPrim()).GetMassAttr().Get(), mass_before)
        self.assertEqual(
            UsdShade.MaterialBindingAPI(self.stage.GetPrimAtPath(HEAD_PATH + "/mesh"))
            .ComputeBoundMaterial()[0]
            .GetPath(),
            material.GetPath(),
        )
        self.assertFalse(self.stage.GetPrimAtPath(HEAD_PATH + "/mesh").HasAPI(UsdPhysics.CollisionAPI))

    def test_bounds_anchors_account_for_authored_head_translation(self) -> None:
        self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
        self.assert_vector_close(self.visual.bounds_min, Gf.Vec3d(-0.08, -0.07, 0.32))
        self.assert_vector_close(self.visual.bounds_max, Gf.Vec3d(0.08, 0.07, 0.54))
        self.assert_vector_close(self.visual.neck_pivot_local, Gf.Vec3d(0.0, 0.0, 0.32))
        self.assert_vector_close(self.visual.eye_mount_local, Gf.Vec3d(0.08, 0.0, 0.43))
        self.assert_vector_close(self.visual.neck_pivot_in_torso, Gf.Vec3d(0.004, 0.0, 0.276))
        self.assert_vector_close(self.visual.eye_mount_in_torso, Gf.Vec3d(0.084, 0.0, 0.386))

    def test_first_person_hides_only_render_head_and_restores_inherited_visibility(self) -> None:
        original_source = self.stage.GetPrimAtPath("/RenderSource/head_link")
        source_visibility = UsdGeom.Imageable(original_source).CreateVisibilityAttr(UsdGeom.Tokens.inherited)
        source_visibility.SetCustomData({"source": "unchanged"})
        source_custom_data = source_visibility.GetCustomData()
        self.assertTrue(self.visual.prepare(self.stage, ROBOT))
        attribute_path = Sdf.Path(HEAD_PATH + ".visibility")
        self.assertIsNone(self.stage.GetRootLayer().GetPropertyAtPath(attribute_path))
        self.assertTrue(self.visual.set_first_person_hidden(True))
        self.assertTrue(self.visual.first_person_hidden)
        self.assertEqual(
            UsdGeom.Imageable(self.stage.GetPrimAtPath(HEAD_PATH + "/mesh")).ComputeVisibility(),
            UsdGeom.Tokens.invisible,
        )
        self.assertEqual(
            UsdGeom.Imageable(self.stage.GetPrimAtPath(COLLISIONS + "/head_link/mesh")).ComputeVisibility(),
            UsdGeom.Tokens.inherited,
        )
        self.assertTrue(
            UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(COLLISIONS + "/head_link/mesh"))
            .GetCollisionEnabledAttr()
            .Get()
        )
        self.assertEqual(source_visibility.Get(), UsdGeom.Tokens.inherited)
        self.assertEqual(source_visibility.GetCustomData(), source_custom_data)
        self.assertTrue(self.visual.set_first_person_hidden(False))
        self.assertFalse(self.visual.first_person_hidden)
        self.assertIsNone(self.stage.GetRootLayer().GetPropertyAtPath(attribute_path))
        self.assertEqual(
            UsdGeom.Imageable(self.stage.GetPrimAtPath(HEAD_PATH + "/mesh")).ComputeVisibility(),
            UsdGeom.Tokens.inherited,
        )

    def test_first_person_restore_preserves_authored_animation_and_metadata_exactly(self) -> None:
        self.visuals.SetInstanceable(False)
        visibility = UsdGeom.Imageable(self.stage.GetPrimAtPath(HEAD_PATH)).CreateVisibilityAttr()
        visibility.Set(UsdGeom.Tokens.inherited)
        visibility.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(1))
        visibility.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(2))
        visibility.SetCustomData({"owner": "operator", "nested": {"index": 3}})
        visibility.SetDocumentation("Original authored visibility")
        backup = Sdf.Layer.CreateAnonymous()
        path = visibility.GetPath()
        Sdf.CreatePrimInLayer(backup, path.GetPrimPath())
        Sdf.CopySpec(self.stage.GetRootLayer(), path, backup, path)
        expected = backup.ExportToString()
        self.assertTrue(self.visual.prepare(self.stage, ROBOT))
        for end_with_clear in (False, True):
            self.assertTrue(self.visual.set_first_person_hidden(True))
            self.assertEqual(visibility.Get(Usd.TimeCode(2)), UsdGeom.Tokens.invisible)
            self.assertEqual(visibility.GetTimeSamples(), [])
            if end_with_clear:
                self.visual.clear()
            else:
                self.assertTrue(self.visual.set_first_person_hidden(False))
            restored = Sdf.Layer.CreateAnonymous()
            Sdf.CreatePrimInLayer(restored, path.GetPrimPath())
            Sdf.CopySpec(self.stage.GetRootLayer(), path, restored, path)
            self.assertEqual(restored.ExportToString(), expected)

    def test_first_person_state_is_idempotent_and_survives_rotation_reset(self) -> None:
        self.assertFalse(self.visual.set_first_person_hidden(True))
        self.assertTrue(self.visual.prepare(self.stage, ROBOT))
        self.assertFalse(self.visual.set_first_person_hidden("true"))
        self.assertTrue(self.visual.set_first_person_hidden(True))
        first = self.stage.GetRootLayer().ExportToString()
        for _ in range(5):
            self.assertTrue(self.visual.set_first_person_hidden(True))
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), first)
        self.visual.reset()
        self.assertTrue(self.visual.first_person_hidden)
        self.assertEqual(
            UsdGeom.Imageable(self.stage.GetPrimAtPath(HEAD_PATH)).ComputeVisibility(), UsdGeom.Tokens.invisible
        )
        self.visual.clear()
        self.assertTrue(self.visuals.IsInstanceable())
        self.assertEqual(
            UsdGeom.Imageable(self.stage.GetPrimAtPath(HEAD_PATH)).ComputeVisibility(), UsdGeom.Tokens.inherited
        )

    def test_first_person_visibility_uses_prepared_edit_target_and_restores_on_stage_change(self) -> None:
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.assertTrue(self.visual.prepare(self.stage, ROBOT))
        root_before = self.stage.GetRootLayer().ExportToString()
        self.assertTrue(self.visual.set_first_person_hidden(True))
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), root_before)
        self.assertIsNotNone(self.stage.GetSessionLayer().GetPropertyAtPath(Sdf.Path(HEAD_PATH + ".visibility")))
        self.assertFalse(self.visual.prepare(Usd.Stage.CreateInMemory(), ROBOT))
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), root_before)
        self.assertIsNone(self.stage.GetSessionLayer().GetPropertyAtPath(Sdf.Path(HEAD_PATH + ".visibility")))

    def test_first_person_hidden_cleanup_does_not_recreate_deleted_head(self) -> None:
        self.assertTrue(self.visual.prepare(self.stage, ROBOT))
        self.assertTrue(self.visual.set_first_person_hidden(True))
        self.stage.RemovePrim(ROBOT)
        self.visual.clear()
        self.assertFalse(self.stage.GetPrimAtPath(ROBOT).IsValid())

    def test_bounds_and_camera_anchor_do_not_inflate_when_whole_robot_rotates(self) -> None:
        root = UsdGeom.Xformable(self.stage.GetPrimAtPath(ROBOT))
        rotation_op = root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        for degrees in (0, 37, 65, -125):
            with self.subTest(degrees=degrees):
                self.visual.clear()
                rotation_op.Set(Gf.Rotation(Gf.Vec3d(1, 2, 3).GetNormalized(), degrees).GetQuat())
                self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
                self.assert_vector_close(self.visual.bounds_min, Gf.Vec3d(-0.08, -0.07, 0.32))
                self.assert_vector_close(self.visual.bounds_max, Gf.Vec3d(0.08, 0.07, 0.54))
                self.assert_vector_close(self.visual.eye_mount_in_torso, Gf.Vec3d(0.084, 0.0, 0.386))

    def test_head_rotates_about_neck_without_moving_collider_or_torso(self) -> None:
        self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
        original = self.visual.original_local_transform
        pivot_before = original.Transform(self.visual.neck_pivot_local)
        forehead = Gf.Vec3d(0.0, 0.0, 0.54)
        delta = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 30))
        collider_before = UsdGeom.XformCache().GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(COLLISIONS + "/head_link")
        )
        torso_before = UsdGeom.XformCache().GetLocalToWorldTransform(self.torso.GetPrim())
        self.assertTrue(self.visual.apply(delta, Gf.Matrix4d(1)))
        current = self.head_transform()
        self.assert_vector_close(current.Transform(self.visual.neck_pivot_local), pivot_before)
        self.assertGreater((current.Transform(forehead) - original.Transform(forehead)).GetLength(), 0.05)
        self.assert_matrix_close(
            UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(COLLISIONS + "/head_link")),
            collider_before,
        )
        self.assert_matrix_close(UsdGeom.XformCache().GetLocalToWorldTransform(self.torso.GetPrim()), torso_before)

    def test_camera_axis_conversion_maps_physical_left_yaw_to_robot_left(self) -> None:
        self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
        mount = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0), Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 0, 1)).GetInverse()
        for yaw in (-35.0, 35.0):
            with self.subTest(yaw=yaw):
                delta = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), yaw))
                self.assertTrue(self.visual.apply(delta, mount))
                forward = self.head_transform().TransformDir(Gf.Vec3d(1, 0, 0))
                self.assertGreater(math.copysign(1.0, yaw) * forward[1], 0.0)
                self.assertAlmostEqual(forward[2], 0.0, places=10)

    def test_camera_attachment_keeps_visible_bounds_behind_fixed_eye_for_all_head_axes(self) -> None:
        self.assertTrue(self.visual.prepare(self.stage, ROBOT))
        torso = UsdGeom.XformCache().GetLocalToWorldTransform(self.torso.GetPrim())
        # Put the initial optical mount inside the head to exercise automatic
        # render-only clearance instead of assuming a convenient asset shape.
        eye = Gf.Vec3d(0.0, 0.0, 0.43)
        mount = Gf.Matrix4d().SetLookAt(eye, eye + Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 0, 1)).GetInverse()
        neutral_camera = mount * torso
        collider_before = UsdGeom.Xformable(
            self.stage.GetPrimAtPath(COLLISIONS + "/head_link")
        ).GetLocalTransformation()
        material_before = UsdShade.MaterialBindingAPI(
            self.stage.GetPrimAtPath(HEAD_PATH + "/mesh")
        ).ComputeBoundMaterial()[0]
        attachment = None
        for yaw, pitch, roll in (
            (0, 0, 0),
            (90, 0, 0),
            (-90, 0, 0),
            (180, 0, 0),
            (0, 80, 0),
            (0, -80, 0),
            (70, 45, -35),
        ):
            with self.subTest(yaw=yaw, pitch=pitch, roll=roll):
                camera = TestLevelHeadReference.pose(yaw, pitch, roll) * neutral_camera
                camera_before = Gf.Matrix4d(camera)
                self.assertTrue(self.visual.apply_camera_attached(camera, mount, torso))
                self.assert_matrix_close(camera, camera_before)
                actual_attachment = self.head_transform() * torso * camera.GetInverse()
                if attachment is None:
                    attachment = Gf.Matrix4d(actual_attachment)
                self.assert_matrix_close(actual_attachment, attachment)
                for x in (self.visual.bounds_min[0], self.visual.bounds_max[0]):
                    for y in (self.visual.bounds_min[1], self.visual.bounds_max[1]):
                        for z in (self.visual.bounds_min[2], self.visual.bounds_max[2]):
                            self.assertGreaterEqual(actual_attachment.Transform(Gf.Vec3d(x, y, z))[2], 0.015 - 1e-10)
                self.assertEqual(
                    UsdGeom.Imageable(self.stage.GetPrimAtPath(HEAD_PATH + "/mesh")).ComputeVisibility(),
                    UsdGeom.Tokens.inherited,
                )
        self.assert_matrix_close(
            UsdGeom.Xformable(self.stage.GetPrimAtPath(COLLISIONS + "/head_link")).GetLocalTransformation(),
            collider_before,
        )
        self.assertEqual(
            UsdShade.MaterialBindingAPI(self.stage.GetPrimAtPath(HEAD_PATH + "/mesh"))
            .ComputeBoundMaterial()[0]
            .GetPath(),
            material_before.GetPath(),
        )
        self.assertFalse(self.stage.GetPrimAtPath(HEAD_PATH + "/mesh").HasAPI(UsdPhysics.CollisionAPI))

    def test_camera_attachment_compensates_live_torso_sway_without_sampling_stale_usd(self) -> None:
        self.assertTrue(self.visual.prepare(self.stage, ROBOT))
        torso = UsdGeom.XformCache().GetLocalToWorldTransform(self.torso.GetPrim())
        eye = self.visual.eye_mount_in_torso + Gf.Vec3d(0.015, 0, 0.04)
        mount = Gf.Matrix4d().SetLookAt(eye, eye + Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 0, 1)).GetInverse()
        camera = mount * torso
        self.assertTrue(self.visual.apply_camera_attached(camera, mount, torso))
        expected_world = self.head_transform() * torso
        for degrees in (-20, -7, 0, 11, 25):
            moving_torso = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3).GetNormalized(), degrees)) * torso
            moving_torso.SetTranslateOnly(torso.ExtractTranslation() + Gf.Vec3d(0.01, -0.02, 0.03))
            self.assertTrue(self.visual.apply_camera_attached(camera, mount, moving_torso))
            self.assert_matrix_close(self.head_transform() * moving_torso, expected_world)
        # The physical USD body was never reauthored to manufacture the result.
        self.assert_matrix_close(UsdGeom.XformCache().GetLocalToWorldTransform(self.torso.GetPrim()), torso)

    def test_camera_attachment_accounts_for_translated_rotated_render_parent(self) -> None:
        parent = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 38))
        parent.SetTranslateOnly(Gf.Vec3d(0.03, -0.06, 0.02))
        UsdGeom.Xformable(self.visuals).AddTransformOp().Set(parent)
        self.assertTrue(self.visual.prepare(self.stage, ROBOT))
        torso = UsdGeom.XformCache().GetLocalToWorldTransform(self.torso.GetPrim())
        mount = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.8, 0, 0.4), Gf.Vec3d(1.8, 0, 0.4), Gf.Vec3d(0, 0, 1)).GetInverse()
        camera = TestLevelHeadReference.pose(yaw=53, pitch=18, roll=-9) * mount * torso
        self.assertTrue(self.visual.apply_camera_attached(camera, mount, torso))
        expected = self.visual.original_local_transform * parent * mount.GetInverse() * camera
        actual = self.head_transform() * parent * torso
        self.assert_matrix_close(actual, expected)

    def test_camera_attachment_invalid_input_holds_previous_pose(self) -> None:
        self.assertTrue(self.visual.prepare(self.stage, ROBOT))
        valid = Gf.Matrix4d(1.0)
        self.assertTrue(self.visual.apply_camera_attached(valid, valid, valid))
        previous = self.head_transform()
        nonfinite = Gf.Matrix4d(1.0)
        nonfinite.SetTranslateOnly(Gf.Vec3d(math.nan, 0, 0))
        projective = Gf.Matrix4d(1.0)
        projective[0, 3] = 0.1
        for invalid in (None, nonfinite, projective, Gf.Matrix4d(0), Gf.Matrix4d().SetScale(Gf.Vec3d(2, 1, 1))):
            for index in range(3):
                args = [valid, valid, valid]
                args[index] = invalid
                self.assertFalse(self.visual.apply_camera_attached(*args))
                self.assert_matrix_close(self.head_transform(), previous)
        for invalid_clearance in (-0.01, math.nan, math.inf):
            self.assertFalse(self.visual.apply_camera_attached(valid, valid, valid, invalid_clearance))
            self.assert_matrix_close(self.head_transform(), previous)

    def test_camera_attachment_cache_tracks_mount_clearance_and_clears_with_lifecycle(self) -> None:
        self.assertTrue(self.visual.prepare(self.stage, ROBOT))
        original = self.visual.original_local_transform
        valid = Gf.Matrix4d(1)
        first_mount = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, 0.4))
        self.assertTrue(self.visual.apply_camera_attached(valid, first_mount, valid))
        first_attachment = Gf.Matrix4d(self.visual._head_to_camera)
        self.assertTrue(self.visual.apply_camera_attached(valid, first_mount, valid, clearance=0.05))
        self.assertGreater(
            self.visual._head_to_camera.ExtractTranslation()[2], first_attachment.ExtractTranslation()[2]
        )
        mount = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.1, -0.1, 0.2))
        self.assertTrue(self.visual.apply_camera_attached(valid, mount, valid, clearance=0.05))
        self.assert_matrix_close(self.visual._camera_attachment_mount, mount)
        self.visual.reset()
        self.assert_matrix_close(self.head_transform(), original)
        self.visual.clear()
        self.assertIsNone(self.visual._head_to_camera)
        self.assertIsNone(self.visual._parent_to_torso_transform)
        self.assert_matrix_close(self.head_transform(), original)
        self.assertFalse(self.visual.apply_camera_attached(valid, valid, valid))

    def test_switching_from_camera_attachment_back_to_legacy_restores_neck_pivot(self) -> None:
        self.assertTrue(self.visual.prepare(self.stage, ROBOT))
        original_pivot = self.visual.original_local_transform.Transform(self.visual.neck_pivot_local)
        camera = TestLevelHeadReference.pose(yaw=90.0, pitch=45.0)
        self.assertTrue(self.visual.apply_camera_attached(camera, Gf.Matrix4d(1), Gf.Matrix4d(1)))
        delta = TestLevelHeadReference.pose(yaw=30.0)
        self.assertTrue(self.visual.apply(delta, Gf.Matrix4d(1)))
        self.assert_vector_close(self.head_transform().Transform(self.visual.neck_pivot_local), original_pivot)

    def test_render_parent_rotation_is_accounted_for_in_torso_coordinates(self) -> None:
        rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 35))
        UsdGeom.Xformable(self.visuals).AddTransformOp().Set(rotation)
        self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
        delta = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 20))
        mount = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 15))
        self.assertTrue(self.visual.apply(delta, mount))
        authored_head_to_torso = self.visual.original_local_transform * rotation
        current_head_to_torso = self.head_transform() * rotation
        expected_direction = (authored_head_to_torso * mount.GetInverse() * delta * mount).TransformDir(
            Gf.Vec3d(0, 0, 1)
        )
        self.assert_vector_close(current_head_to_torso.TransformDir(Gf.Vec3d(0, 0, 1)), expected_direction)
        self.assert_vector_close(
            current_head_to_torso.Transform(self.visual.neck_pivot_local), self.visual.neck_pivot_in_torso
        )

    def test_reset_and_clear_restore_head_order_and_render_instanceability(self) -> None:
        original = self.head_transform()
        original_order = self.stage.GetPrimAtPath(HEAD_PATH).GetAttribute("xformOpOrder").Get()
        self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
        self.assertTrue(self.visual.apply(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 40)), Gf.Matrix4d(1)))
        self.visual.reset()
        self.assert_matrix_close(self.head_transform(), original)
        self.visual.clear()
        self.assertTrue(self.visuals.IsInstanceable())
        self.assertTrue(self.collisions.IsInstanceable())
        self.assert_matrix_close(self.head_transform(), original)
        self.assertEqual(self.stage.GetPrimAtPath(HEAD_PATH).GetAttribute("xformOpOrder").Get(), original_order)
        self.assertFalse(self.stage.GetPrimAtPath(HEAD_PATH).GetAttribute(HEAD.RobotHeadVisual._OP_NAME).IsValid())
        self.assertFalse(self.visual.ready)

    def test_clear_restores_local_authored_order_and_original_noninstance_state(self) -> None:
        self.visuals.SetInstanceable(False)
        xform = UsdGeom.Xformable(self.stage.GetPrimAtPath(HEAD_PATH))
        xform.AddRotateXOp().Set(8.0)
        original = self.head_transform()
        original_order = list(xform.GetXformOpOrderAttr().Get())
        self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
        self.visual.apply(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 15)), Gf.Matrix4d(1))
        self.visual.clear()
        self.assertFalse(self.visuals.IsInstanceable())
        self.assertEqual(list(xform.GetXformOpOrderAttr().Get()), original_order)
        self.assert_matrix_close(self.head_transform(), original)

    def test_repeated_prepare_does_not_accumulate_ops_and_stage_change_restores_old_stage(self) -> None:
        self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
        order = list(self.stage.GetPrimAtPath(HEAD_PATH).GetAttribute("xformOpOrder").Get())
        self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
        self.assertEqual(list(self.stage.GetPrimAtPath(HEAD_PATH).GetAttribute("xformOpOrder").Get()), order)
        empty_stage = Usd.Stage.CreateInMemory()
        self.assertFalse(self.visual.prepare(empty_stage, ROBOT))
        self.assertTrue(self.visuals.IsInstanceable())
        self.assertFalse(self.visual.ready)

    def test_invalid_rotation_and_tracking_loss_hold_last_valid_visual(self) -> None:
        self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
        self.visual.apply(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 15)), Gf.Matrix4d(1))
        last = self.head_transform()
        invalid = Gf.Matrix4d(1)
        invalid[0, 0] = math.nan
        for value in (None, Gf.Matrix4d(0), invalid, Gf.Matrix4d().SetScale(Gf.Vec3d(2, 1, 0.5))):
            with self.subTest(value=value):
                self.assertFalse(self.visual.apply(value, Gf.Matrix4d(1)))
                self.assertFalse(self.visual.apply(Gf.Matrix4d(1), value))
                self.assert_matrix_close(self.head_transform(), last)

    def test_collision_anywhere_in_render_subtree_refuses_without_editing_instances(self) -> None:
        for schema in (UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
            with self.subTest(schema=schema):
                unsafe = UsdGeom.Xform.Define(self.stage, "/RenderSource/UnsafeSibling").GetPrim()
                schema.Apply(unsafe)
                self.assertFalse(self.visual.prepare(self.stage, ROBOT))
                self.assertIn("contains physics", self.visual.status)
                self.assertTrue(self.visuals.IsInstanceable())
                self.assertTrue(self.collisions.IsInstanceable())
                self.stage.RemovePrim("/RenderSource/UnsafeSibling")

    def test_inactive_render_collider_also_refuses(self) -> None:
        unsafe = UsdGeom.Xform.Define(self.stage, "/RenderSource/InactiveCollider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(unsafe)
        unsafe.SetActive(False)
        self.assertFalse(self.visual.prepare(self.stage, ROBOT))
        self.assertTrue(self.visuals.IsInstanceable())

    def test_missing_mesh_rolls_back_instance_edit(self) -> None:
        self.stage.RemovePrim("/RenderSource/head_link/mesh")
        self.assertFalse(self.visual.prepare(self.stage, ROBOT))
        self.assertIn("bounds", self.visual.status)
        self.assertTrue(self.visuals.IsInstanceable())
        self.assertFalse(self.visual.ready)

    def test_detached_transform_stack_refuses_without_editing_instance(self) -> None:
        UsdGeom.Xformable(self.stage.GetPrimAtPath("/RenderSource/head_link")).SetResetXformStack(True)
        self.assertFalse(self.visual.prepare(self.stage, ROBOT))
        self.assertIn("detach", self.visual.status)
        self.assertTrue(self.visuals.IsInstanceable())

    def test_nonfinite_authored_transform_rolls_back_instance_edit(self) -> None:
        source = UsdGeom.Xformable(self.stage.GetPrimAtPath("/RenderSource/head_link"))
        source.GetOrderedXformOps()[0].Set(Gf.Vec3d(math.nan, 0, 0))
        self.assertFalse(self.visual.prepare(self.stage, ROBOT))
        self.assertIn("finite", self.visual.status)
        self.assertTrue(self.visuals.IsInstanceable())

    def test_cleanup_restores_owned_op_after_external_reinstancing(self) -> None:
        original = self.head_transform()
        self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
        self.visual.apply(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 15)), Gf.Matrix4d(1))
        self.visuals.SetInstanceable(True)
        self.visual.clear()
        self.assertTrue(self.visuals.IsInstanceable())
        self.assert_matrix_close(self.head_transform(), original)
        self.visuals.SetInstanceable(False)
        self.assertFalse(self.stage.GetPrimAtPath(HEAD_PATH).GetAttribute(HEAD.RobotHeadVisual._OP_NAME).IsValid())

    def test_deleted_stage_content_does_not_recreate_head_on_cleanup(self) -> None:
        self.assertTrue(self.visual.prepare(self.stage, ROBOT), self.visual.status)
        self.stage.RemovePrim(ROBOT)
        self.assertFalse(self.visual.apply(Gf.Matrix4d(1), Gf.Matrix4d(1)))
        self.visual.clear()
        self.assertFalse(self.stage.GetPrimAtPath(ROBOT).IsValid())


if __name__ == "__main__":
    unittest.main()
