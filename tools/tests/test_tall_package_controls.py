# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check standing-height package geometry using real USD schemas and bounds."""

from __future__ import annotations

import unittest

from humanoid_test_support import GRASP, TALL_PACKAGES, load_source
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

CONTACTS = load_source("interactive/humanoid/grasp_contacts.py", ContactSample=GRASP.ContactSample)


class TestTallPackageControls(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        self.root = "/World/G1_SampleBoxes"
        self.paths = TALL_PACKAGES.create_tall_packages(self.stage, self.root, first_index=10)

    def bounds(self, path):
        return (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            .ComputeWorldBound(self.stage.GetPrimAtPath(path))
            .ComputeAlignedRange()
        )

    def test_tall_bodies_start_on_floor_and_handles_are_at_standing_hand_height(self):
        for path, dimensions, height in zip(
            self.paths, ((0.32, 0.32, 0.76), (0.40, 0.32, 0.88), (0.30, 0.38, 0.96)), (0.685, 0.805, 0.940)
        ):
            with self.subTest(path=path):
                body_bounds = self.bounds(path + "/Carton")
                for actual, expected in zip(body_bounds.GetSize(), dimensions):
                    self.assertAlmostEqual(actual, expected, places=6)
                self.assertAlmostEqual(body_bounds.GetMin()[2], 0.005, places=6)
                handle_bounds = self.bounds(path + "/HandleGrip")
                self.assertAlmostEqual(handle_bounds.GetMidpoint()[2], height, places=6)
                prim = self.stage.GetPrimAtPath(path)
                matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                local_center = prim.GetAttribute("g1:gripCenterLocal").Get()
                world_center = matrix.Transform(Gf.Vec3d(local_center))
                self.assertLess((world_center - handle_bounds.GetMidpoint()).GetLength(), 1e-6)

    def test_single_light_dynamic_body_owns_four_colliders_and_no_attachment_joints(self):
        self.assertEqual(
            [str(prim.GetPath()) for prim in self.stage.GetPrimAtPath(self.root).GetChildren()], self.paths
        )
        for path, mass in zip(self.paths, (0.25, 0.35, 0.45)):
            with self.subTest(path=path):
                prim = self.stage.GetPrimAtPath(path)
                subtree = list(Usd.PrimRange(prim))
                bodies = [str(item.GetPath()) for item in subtree if item.HasAPI(UsdPhysics.RigidBodyAPI)]
                self.assertEqual(bodies, [path])
                self.assertEqual(sum(item.HasAPI(UsdPhysics.CollisionAPI) for item in subtree), 4)
                self.assertFalse(any(item.IsA(UsdPhysics.Joint) for item in subtree))
                rigid = UsdPhysics.RigidBodyAPI(prim)
                self.assertTrue(rigid.GetRigidBodyEnabledAttr().Get())
                self.assertFalse(rigid.GetKinematicEnabledAttr().Get())
                self.assertAlmostEqual(UsdPhysics.MassAPI(prim).GetMassAttr().Get(), mass)
                self.assertAlmostEqual(prim.GetAttribute("g1:packageMass").Get(), mass)
                physics = PhysxSchema.PhysxRigidBodyAPI(prim)
                self.assertTrue(physics.GetEnableCCDAttr().Get())
                self.assertEqual(physics.GetSleepThresholdAttr().Get(), 0.0)
                self.assertEqual(physics.GetSolverPositionIterationCountAttr().Get(), 16)
                self.assertEqual(physics.GetSolverVelocityIterationCountAttr().Get(), 4)

    def test_handle_knuckle_gap_remains_open_in_compound_collision_geometry(self):
        for path in self.paths:
            with self.subTest(path=path):
                body = self.bounds(path + "/Carton")
                grip = self.bounds(path + "/HandleGrip")
                self.assertAlmostEqual(grip.GetSize()[0], 0.035, places=6)
                self.assertAlmostEqual(grip.GetSize()[1], 0.16, places=6)
                self.assertAlmostEqual(body.GetMin()[0] - grip.GetMax()[0], 0.07, places=6)
                gap_center = Gf.Vec3d(
                    (body.GetMin()[0] + grip.GetMax()[0]) / 2, grip.GetMidpoint()[1], grip.GetMidpoint()[2]
                )
                for prim in Usd.PrimRange(self.stage.GetPrimAtPath(path)):
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        self.assertFalse(self.bounds(str(prim.GetPath())).Contains(gap_center))
                # The side brackets deliberately intersect both body and grip.
                for name in ("HandleBracketLeft", "HandleBracketRight"):
                    bracket = self.bounds(path + "/" + name)
                    self.assertGreater(bracket.GetMax()[0], body.GetMin()[0])
                    self.assertLess(bracket.GetMin()[0], grip.GetMax()[0] - 0.01)
                    self.assertGreaterEqual(bracket.GetMin()[2], body.GetMin()[2])
                    self.assertLessEqual(bracket.GetMax()[2], body.GetMax()[2])

    def test_high_practice_handle_preserves_body_clearance_and_has_supported_brackets(self):
        path = self.paths[2]
        grip = self.bounds(path + "/HandleGrip")
        body = self.bounds(path + "/Carton")
        self.assertAlmostEqual(body.GetMax()[2] - grip.GetMax()[2], 0.0075, places=6)
        self.assertAlmostEqual(grip.GetMin()[0], 1.095, places=6)
        # The actual measured replay geometry puts the wrist behind the palm.
        # Raising only this handle brings that target into the 0.365 m arm span
        # while retaining the 0.38 m body footprint plus 0.10 m braking reserve.
        shoulder = Gf.Vec3d(0.610 + 0.013319, 0.050 - 0.089872, 1.059515)
        settled_grip = grip.GetMidpoint() - Gf.Vec3d(0, 0, 0.005)
        wrist = settled_grip + Gf.Vec3d(-0.1418707, -0.0420252, 0.0317728)
        self.assertLess((wrist - shoulder).GetLength(), 0.365377)
        self.assertGreater(grip.GetMin()[0] - 0.610, 0.38 + 0.10)

    def test_spec_rejects_nonfinite_geometry_and_handles_outside_body(self):
        defaults = {"position_xy": (0.0, 0.0), "dimensions": (0.3, 0.38, 0.96), "mass": 0.45}
        for override in (
            {"position_xy": (float("nan"), 0.0)},
            {"dimensions": (0.3, 0.38)},
            {"dimensions": (0.3, -0.38, 0.96)},
            {"mass": 0.0},
            {"handle_below_top": float("inf")},
            {"handle_below_top": 0.01},
            {"handle_below_top": 0.95},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                TALL_PACKAGES.TallPackageSpec(**(defaults | override))

    def test_friction_is_inherited_by_all_compound_colliders(self):
        for path in self.paths:
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(path)):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
                self.assertTrue(material)
                contact = UsdPhysics.MaterialAPI(material.GetPrim())
                self.assertAlmostEqual(contact.GetStaticFrictionAttr().Get(), 0.8)
                self.assertAlmostEqual(contact.GetDynamicFrictionAttr().Get(), 0.6)
                self.assertAlmostEqual(contact.GetRestitutionAttr().Get(), 0.0)
                self.assertIsNone(prim.GetRelationship("material:binding").GetTargets() or None)

    def test_layout_keeps_remote_stations_clear_and_places_high_handle_ahead_of_spawn(self):
        bounds = [self.bounds(path) for path in self.paths]
        for carton in bounds[:2]:
            self.assertGreater(carton.GetMin()[0], 3.15)
            self.assertTrue(carton.GetMin()[1] > 2.60 or carton.GetMax()[1] < -2.60)
            self.assertLess(carton.GetMax()[0], 4.80)
        practice = bounds[2]
        self.assertGreater(practice.GetMin()[0], 1.05)
        self.assertLess(practice.GetMax()[0], 1.55)
        self.assertAlmostEqual(practice.GetMidpoint()[1], 0.0)
        self.assertGreater(practice.GetMin()[0], 2 * 0.38 * 2**0.5)
        # A conservative turning footprint at the approach staging spots must
        # not overlap the old low cartons at (2.6, +/-1.15), even though the final
        # close reach still needs actual scene checks against the new carton.
        old_cartons = ((2.6, 1.15, 0.34), (2.6, -1.15, 0.40))
        for carton in bounds[:2]:
            staging = (carton.GetMin()[0] - 0.65, carton.GetMidpoint()[1] + 0.2)
            for x, y, size in old_cartons:
                gap = max(abs(staging[0] - x), abs(staging[1] - y)) - size / 2
                self.assertGreater(gap, 0.38 * 2**0.5)
        for index, first in enumerate(bounds):
            for second in bounds[index + 1 :]:
                gap = max(
                    max(first.GetMin()[axis] - second.GetMax()[axis], second.GetMin()[axis] - first.GetMax()[axis])
                    for axis in (0, 1)
                )
                self.assertGreater(gap, 0.60)

    def test_compound_handle_contacts_canonicalize_to_registered_package_roots(self):
        for side in ("L", "R"):
            for role in ("thumb", "index", "middle", "ring", "pinky", "hand_base_link"):
                prim = UsdGeom.Xform.Define(self.stage, f"/World/G1/{side}_{role}").GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(prim)
        reader = CONTACTS.ContactReader()
        reader.prepare(self.stage, "/World/G1", self.paths)
        self.assertEqual(set(reader.object_paths), set(self.paths))
        for path in self.paths:
            matches = [root for root in reader.object_paths if reader._within(path + "/HandleGrip", root)]
            self.assertEqual(matches, [path])

    def test_physical_support_at_upper_handle_does_not_depend_on_carton_center_proximity(self):
        path = self.paths[2]
        grip = self.bounds(path + "/HandleGrip").GetMidpoint()
        center = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path)).ExtractTranslation()
        self.assertGreater((grip - center).GetLength(), 0.4)
        # Measured opposing contacts on the actual handle are eligible through
        # their registered rigid root, even far above the carton body's center.
        contacts = [
            GRASP.ContactSample(path, "thumb", tuple(grip + Gf.Vec3d(-0.0175, 0, 0)), (-1, 0, 0), 3.0, 0.0),
            GRASP.ContactSample(path, "index", tuple(grip + Gf.Vec3d(0.0175, 0, 0)), (1, 0, 0), 3.0, 0.0),
        ]
        controller = GRASP.ContactGraspController()
        for _ in range(20):
            result = controller.update(
                dt=0.01,
                requested_curls={"thumb": 0.5, "index": 0.5},
                measured_curls={"thumb": 0.4, "index": 0.4},
                closing_lead={"thumb": 0.03, "index": 0.03},
                contacts=contacts,
            )
        self.assertTrue(result.contact_supported)
        self.assertEqual(result.object_path, path)
        self.assertEqual(result.mode, "pinch")

    def test_existing_objects_are_retained_and_duplicate_creation_fails_before_any_edit(self):
        original = self.stage.GetRootLayer().ExportToString()
        with self.assertRaisesRegex(ValueError, "unused"):
            TALL_PACKAGES.create_tall_packages(self.stage, self.root, first_index=10)
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), original)
        old = UsdGeom.Sphere.Define(self.stage, self.root + "/Box_00")
        old.CreateRadiusAttr(0.035)
        later = TALL_PACKAGES.create_tall_packages(self.stage, self.root, first_index=20)
        self.assertEqual(later, [self.root + f"/Box_{index:02d}" for index in range(20, 23)])
        self.assertEqual(old.GetRadiusAttr().Get(), 0.035)

    def test_invalid_names_or_indices_do_not_author_partial_packages(self):
        before = self.stage.GetRootLayer().ExportToString()
        for root, index in (("Relative", 30), ("/", 30), ("/World.attr", 30), (self.root, -1), (self.root, True)):
            with self.subTest(root=root, index=index):
                with self.assertRaises(ValueError):
                    TALL_PACKAGES.create_tall_packages(self.stage, root, first_index=index)
                self.assertEqual(self.stage.GetRootLayer().ExportToString(), before)


if __name__ == "__main__":
    unittest.main()
