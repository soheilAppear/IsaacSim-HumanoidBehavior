# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-USD checks for static shelves and independent two-hand practice cartons."""

from __future__ import annotations

import unittest

from humanoid_test_support import PACKAGE_SHELVES
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade


class TestPackageShelfControls(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        self.package_root = "/World/G1_SampleBoxes"
        self.shelf_root = "/World/G1_PackageShelves"
        self.paths = PACKAGE_SHELVES.create_shelf_packages(self.stage, self.package_root, first_index=10)

    def bounds(self, path):
        return (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            .ComputeWorldBound(self.stage.GetPrimAtPath(path))
            .ComputeAlignedRange()
        )

    def test_twelve_unique_cartons_use_separate_static_shelf_namespace(self):
        self.assertEqual(self.paths, [f"{self.package_root}/Box_{index:02d}" for index in range(10, 22)])
        self.assertEqual(
            [str(prim.GetPath()) for prim in self.stage.GetPrimAtPath(self.package_root).GetChildren()], self.paths
        )
        shelf_prims = list(Usd.PrimRange(self.stage.GetPrimAtPath(self.shelf_root)))
        self.assertFalse(any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in shelf_prims))
        self.assertEqual(sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in shelf_prims), 18)
        for shelf, expected in enumerate(((0.68, 0.94), (0.74, 1.02), (0.78, 1.06))):
            for level, height in enumerate(expected):
                self.assertAlmostEqual(
                    self.bounds(f"{self.shelf_root}/Shelf_{shelf:02d}/Deck_{level}").GetMax()[2], height, places=6
                )

    def test_each_carton_has_one_dynamic_body_and_one_collider_without_handles_or_joints(self):
        for path, spec in zip(self.paths, PACKAGE_SHELVES.SHELF_PACKAGE_SPECS):
            with self.subTest(path=path):
                prim = self.stage.GetPrimAtPath(path)
                descendants = list(Usd.PrimRange(prim))
                self.assertEqual(sum(item.HasAPI(UsdPhysics.RigidBodyAPI) for item in descendants), 1)
                self.assertEqual(sum(item.HasAPI(UsdPhysics.CollisionAPI) for item in descendants), 1)
                self.assertFalse(
                    any(item.IsA(UsdPhysics.Joint) or "handle" in item.GetName().lower() for item in descendants)
                )
                self.assertFalse(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())
                self.assertAlmostEqual(UsdPhysics.MassAPI(prim).GetMassAttr().Get(), spec.mass)
                for actual, expected in zip(self.bounds(path + "/Carton").GetSize(), spec.dimensions):
                    self.assertAlmostEqual(actual, expected, places=6)
                self.assertFalse(prim.GetAttribute("g1:gripCenterLocal").IsValid())
                self.assertEqual(prim.GetAttribute("g1:packageKind").Get(), "shelf_carton")
                self.assertTrue(prim.GetAttribute("g1:packageOnSurface").Get())
                physics = PhysxSchema.PhysxRigidBodyAPI(prim)
                self.assertTrue(physics.GetEnableCCDAttr().Get())
                self.assertEqual(physics.GetSleepThresholdAttr().Get(), 0)

    def test_cartons_rest_above_real_support_and_lower_decks_leave_lift_room(self):
        for path, spec in zip(self.paths, PACKAGE_SHELVES.SHELF_PACKAGE_SPECS):
            with self.subTest(path=path):
                support_paths = self.stage.GetPrimAtPath(path).GetRelationship("g1:supportSurface").GetTargets()
                self.assertEqual(len(support_paths), 1)
                deck = self.bounds(str(support_paths[0]))
                carton = self.bounds(path + "/Carton")
                self.assertAlmostEqual(carton.GetMin()[2] - deck.GetMax()[2], 0.003, places=6)
                self.assertAlmostEqual(carton.GetMin()[0] - deck.GetMin()[0], 0.005, places=6)
                for axis in (0, 1):
                    self.assertGreaterEqual(carton.GetMin()[axis], deck.GetMin()[axis])
                    self.assertLessEqual(carton.GetMax()[axis], deck.GetMax()[axis])
                if spec.level == 0:
                    ceiling = self.bounds(f"{self.shelf_root}/Shelf_{spec.shelf:02d}/Deck_1").GetMin()[2]
                    self.assertGreater(ceiling - carton.GetMax()[2], 0.09)

    def test_side_palms_have_room_between_cartons_and_posts(self):
        for offset in range(0, 12, 2):
            first = self.bounds(self.paths[offset] + "/Carton")
            second = self.bounds(self.paths[offset + 1] + "/Carton")
            self.assertGreater(second.GetMin()[1] - first.GetMax()[1], 0.28)
        for path, spec in zip(self.paths, PACKAGE_SHELVES.SHELF_PACKAGE_SPECS):
            carton = self.bounds(path + "/Carton")
            rack_center = PACKAGE_SHELVES.SHELF_SPECS[spec.shelf].position_xy[1]
            for side in (-1, 1):
                post_inner = rack_center + side * (PACKAGE_SHELVES.SHELF_WIDTH / 2 - PACKAGE_SHELVES.POST_WIDTH)
                edge = carton.GetMin()[1] if side < 0 else carton.GetMax()[1]
                self.assertGreater(side * (post_inner - edge), 0.34)

    def test_opposed_palm_metadata_lies_on_real_side_faces(self):
        for path, spec in zip(self.paths, PACKAGE_SHELVES.SHELF_PACKAGE_SPECS):
            prim = self.stage.GetPrimAtPath(path)
            for name, sign in (("left", 1), ("right", -1)):
                point = prim.GetAttribute(f"g1:{name}PalmContactLocal").Get()
                self.assertAlmostEqual(point[0], 0)
                self.assertAlmostEqual(point[1], sign * spec.dimensions[1] / 2)
                self.assertAlmostEqual(point[2], 0)

    def test_front_upper_practice_box_has_geometrically_reachable_side_contacts(self):
        # Measured G1 link span is 0.365m. This is a necessary geometry check
        # from the proposed nearby root, not evidence of navigation or lifting.
        path = f"{self.package_root}/Box_20"
        center = self.bounds(path + "/Carton").GetMidpoint() - Gf.Vec3d(0, 0, 0.003)
        self.assertLess((center - Gf.Vec3d(1.275, -0.28, 1.140)).GetLength(), 1e-6)
        for side in (-1, 1):
            shoulder = Gf.Vec3d(0.87 + 0.013319, -0.28 + side * 0.089872, 1.059515)
            wrist = center + Gf.Vec3d(-0.068855, side * 0.12, 0)
            self.assertLess((wrist - shoulder).GetLength(), 0.35)

    def test_friction_is_bound_to_shelf_and_carton_colliders(self):
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
                self.assertTrue(material)
                friction = UsdPhysics.MaterialAPI(material.GetPrim())
                self.assertAlmostEqual(friction.GetStaticFrictionAttr().Get(), 0.8)
                self.assertAlmostEqual(friction.GetDynamicFrictionAttr().Get(), 0.6)
                self.assertEqual(friction.GetRestitutionAttr().Get(), 0.0)

    def test_original_props_are_retained_and_duplicate_creation_is_atomic(self):
        stage = Usd.Stage.CreateInMemory()
        for index in range(10):
            original = UsdGeom.Sphere.Define(stage, f"{self.package_root}/Box_{index:02d}")
            original.CreateRadiusAttr(0.02 + index * 0.001)
        roots = PACKAGE_SHELVES.create_shelf_packages(stage, self.package_root, first_index=10)
        self.assertEqual(len(roots), 12)
        self.assertEqual(len(list(stage.GetPrimAtPath(self.package_root).GetChildren())), 22)
        for index in range(10):
            self.assertAlmostEqual(
                UsdGeom.Sphere(stage.GetPrimAtPath(f"{self.package_root}/Box_{index:02d}")).GetRadiusAttr().Get(),
                0.02 + index * 0.001,
            )
        before = stage.GetRootLayer().ExportToString()
        with self.assertRaises(ValueError):
            PACKAGE_SHELVES.create_shelf_packages(stage, self.package_root, first_index=10)
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)

    def test_invalid_specs_and_namespaces_reject_before_authoring(self):
        for dimensions, mass in (
            ((0.2, 0.9, 0.14), 0.2),
            ((0.4, 0.2, 0.14), 0.2),
            ((0.2, 0.2, -0.1), 0.2),
            ((0.2, 0.2, 0.1), 0),
        ):
            with self.assertRaises(ValueError):
                PACKAGE_SHELVES.ShelfPackageSpec(0, 0, 0.28, dimensions, mass)
        stage = Usd.Stage.CreateInMemory()
        before = stage.GetRootLayer().ExportToString()
        for package_root, shelf_root, index in (
            ("Relative", "/World/Shelf", 0),
            ("/World/Boxes", "/World/Boxes/Shelf", 0),
            ("/World/Boxes", "/World/Shelf", True),
        ):
            with self.assertRaises(ValueError):
                PACKAGE_SHELVES.create_shelf_packages(
                    stage, package_root, first_index=index, shelf_root_path=shelf_root
                )
            self.assertEqual(stage.GetRootLayer().ExportToString(), before)


if __name__ == "__main__":
    unittest.main()
