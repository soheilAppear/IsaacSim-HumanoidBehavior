# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate contact geometry, instance handling, and asset preservation in USD."""

from __future__ import annotations

import unittest

from humanoid_test_support import FOOT_CONTACTS
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade


class TestDecoupledFootContacts(unittest.TestCase):
    def setUp(self) -> None:
        self.asset = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(self.asset, "/Collisions")
        spheres = UsdGeom.Xform.Define(self.asset, "/Collisions/spheres")
        spheres.AddScaleOp().Set(Gf.Vec3f(0.01))
        for index in range(4):
            sphere = UsdGeom.Sphere.Define(self.asset, f"/Collisions/spheres/OldPoint{index}")
            sphere.CreateRadiusAttr(0.5)
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim()).CreateCollisionEnabledAttr(index != 3)
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(self.stage, "/World/G1")
        for side in ("left", "right"):
            foot = UsdGeom.Xform.Define(self.stage, f"/World/G1/{side}_ankle_roll_link").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(foot)
            mass = UsdPhysics.MassAPI.Apply(foot)
            mass.CreateMassAttr(0.608)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0265, 0, -0.0164))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.00167, 0.00161, 0.000217))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
            scope = UsdGeom.Xform.Define(self.stage, foot.GetPath().AppendChild("collisions")).GetPrim()
            scope.GetReferences().AddReference(self.asset.GetRootLayer().identifier, "/Collisions")
            scope.SetInstanceable(True)
        other = UsdGeom.Cube.Define(self.stage, "/World/G1/left_ankle_pitch_link/OtherCollider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(other).CreateCollisionEnabledAttr(True)
        self.root_text = self.stage.GetRootLayer().ExportToString()
        self.asset_text = self.asset.GetRootLayer().ExportToString()
        self.mass_snapshot = FOOT_CONTACTS._body_mass_snapshot(self.stage.GetPrimAtPath("/World/G1"))

    def configure(self) -> dict:
        return FOOT_CONTACTS.configure_decoupled_foot_contacts(self.stage, "/World/G1")

    def assert_assets_unchanged(self) -> None:
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), self.root_text)
        self.assertEqual(self.asset.GetRootLayer().ExportToString(), self.asset_text)
        self.assertEqual(FOOT_CONTACTS._body_mass_snapshot(self.stage.GetPrimAtPath("/World/G1")), self.mass_snapshot)

    def test_instanced_contacts_become_exact_eight_source_spheres_without_mass_or_root_edits(self) -> None:
        self.assertTrue(
            self.stage.GetPrimAtPath("/World/G1/left_ankle_roll_link/collisions/spheres/OldPoint0").IsInstanceProxy()
        )
        report = self.configure()
        self.assertFalse(report["already_configured"])
        self.assertEqual(len(report["sphere_paths"]), 8)
        self.assertEqual(len(report["disabled_colliders"]), 8)
        self.assertEqual(len(report["instance_scopes"]), 2)
        for index, path in enumerate(report["sphere_paths"]):
            prim = self.stage.GetPrimAtPath(path)
            sphere = UsdGeom.Sphere(prim)
            self.assertEqual(sphere.GetRadiusAttr().Get(), 0.005)
            expected = FOOT_CONTACTS.FOOT_CONTACT_POINTS[index % 4]
            self.assertEqual(tuple(sphere.GetLocalTransformation().ExtractTranslation()), expected)
            self.assertTrue(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
            self.assertAlmostEqual(PhysxSchema.PhysxCollisionAPI(prim).GetContactOffsetAttr().Get(), 0.001)
            self.assertEqual(PhysxSchema.PhysxCollisionAPI(prim).GetRestOffsetAttr().Get(), 0.0)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            physics_material = UsdPhysics.MaterialAPI(material.GetPrim())
            self.assertEqual(physics_material.GetStaticFrictionAttr().Get(), 1.0)
            self.assertEqual(physics_material.GetDynamicFrictionAttr().Get(), 1.0)
            self.assertEqual(physics_material.GetRestitutionAttr().Get(), 0.0)
            self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertFalse(prim.HasAPI(UsdPhysics.MassAPI))
        other = self.stage.GetPrimAtPath("/World/G1/left_ankle_pitch_link/OtherCollider")
        self.assertTrue(UsdPhysics.CollisionAPI(other).GetCollisionEnabledAttr().Get())
        self.assertFalse(any(prim.IsA(UsdPhysics.Joint) for prim in self.stage.Traverse()))
        self.assert_assets_unchanged()

    def test_repeated_configuration_validates_without_adding_layers_or_duplicate_geometry(self) -> None:
        first = self.configure()
        layers = list(self.stage.GetSessionLayer().subLayerPaths)
        second = self.configure()
        self.assertTrue(second["already_configured"])
        self.assertEqual(first["sphere_paths"], second["sphere_paths"])
        self.assertEqual(list(self.stage.GetSessionLayer().subLayerPaths), layers)
        self.assert_assets_unchanged()

    def test_detaching_runtime_layer_restores_source_instances_and_original_collision_flags(self) -> None:
        self.configure()
        self.stage.GetSessionLayer().subLayerPaths.clear()
        for side in ("left", "right"):
            foot_path = f"/World/G1/{side}_ankle_roll_link"
            self.assertTrue(self.stage.GetPrimAtPath(foot_path + "/collisions").IsInstance())
            self.assertFalse(self.stage.GetPrimAtPath(foot_path + "/DecoupledFootContacts"))
            for index in range(4):
                prim = self.stage.GetPrimAtPath(foot_path + f"/collisions/spheres/OldPoint{index}")
                self.assertTrue(prim.IsInstanceProxy())
                self.assertEqual(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(), index != 3)
        self.assert_assets_unchanged()

    def test_invalid_asset_structure_fails_before_any_session_edit(self) -> None:
        self.asset.RemovePrim("/Collisions/spheres/OldPoint3")
        with self.assertRaisesRegex(ValueError, "four original sphere"):
            self.configure()
        self.assertFalse(self.stage.GetSessionLayer().subLayerPaths)
        for side in ("left", "right"):
            self.assertTrue(self.stage.GetPrimAtPath(f"/World/G1/{side}_ankle_roll_link/collisions").IsInstance())

    def test_missing_explicit_inertia_is_rejected_before_contact_mass_can_be_recomputed(self) -> None:
        foot = self.stage.GetPrimAtPath("/World/G1/left_ankle_roll_link")
        UsdPhysics.MassAPI(foot).GetDiagonalInertiaAttr().Clear()
        with self.assertRaisesRegex(ValueError, "inertia must be explicitly"):
            self.configure()
        self.assertFalse(self.stage.GetSessionLayer().subLayerPaths)

    def test_failed_stronger_collision_override_rolls_back_temporary_layer(self) -> None:
        # A user's stronger session opinion cannot be silently ignored or replaced.
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            scope = self.stage.GetPrimAtPath("/World/G1/left_ankle_roll_link/collisions")
            scope.SetInstanceable(False)
            original = self.stage.GetPrimAtPath(str(scope.GetPath()) + "/spheres/OldPoint0")
            UsdPhysics.CollisionAPI(original).CreateCollisionEnabledAttr(True)
        session_before = self.stage.GetSessionLayer().ExportToString()
        with self.assertRaisesRegex(ValueError, "still enabled"):
            self.configure()
        self.assertEqual(self.stage.GetSessionLayer().ExportToString(), session_before)
        self.assertFalse(self.stage.GetSessionLayer().subLayerPaths)
        self.assertFalse(self.stage.GetPrimAtPath("/World/G1/left_ankle_roll_link/DecoupledFootContacts"))
        self.assert_assets_unchanged()

    def test_changed_contact_geometry_is_rejected_on_repeat_instead_of_silently_accepted(self) -> None:
        report = self.configure()
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            UsdGeom.Sphere(self.stage.GetPrimAtPath(report["sphere_paths"][0])).CreateRadiusAttr(0.05)
        with self.assertRaisesRegex(ValueError, "geometry or offsets"):
            self.configure()
        self.assertEqual(len(self.stage.GetSessionLayer().subLayerPaths), 1)


if __name__ == "__main__":
    unittest.main()
