# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check opt-in walking scene authoring with real USD reference composition.

Small in-memory carton assets deliberately use centimetres, offset origins, and
inherited rigid bodies. They exercise the production asset wrapper without a
network connection, renderer, PhysX step, or assumption about a cached asset.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import Mock, patch

from humanoid_test_support import GRASP, HIGHLIGHTS, HUMANOID, USD, load_source
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

CONTACTS = load_source("interactive/humanoid/grasp_contacts.py", ContactSample=GRASP.ContactSample)


class TestWalkingSceneControls(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        self.context_patch = patch.object(
            USD, "get_context", return_value=types.SimpleNamespace(get_stage=lambda: self.stage)
        )
        self.context_patch.start()
        self.addCleanup(self.context_patch.stop)
        self.assets = []
        self.asset_text = []
        self.references = []
        for index in range(4):
            asset = Usd.Stage.CreateInMemory()
            root = UsdGeom.Xform.Define(asset, "/Cardboard")
            root.AddScaleOp().Set(Gf.Vec3f(0.01))
            asset.SetDefaultPrim(root.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            mesh = UsdGeom.Mesh.Define(asset, "/Cardboard/Mesh")
            lower = Gf.Vec3f(10.0, -20.0, 5.0)
            upper = Gf.Vec3f(50.0 + 5.0 * index, 10.0, 35.0 + 2.0 * index)
            mesh.CreatePointsAttr(
                [
                    Gf.Vec3f(x, y, z)
                    for z in (lower[2], upper[2])
                    for y in (lower[1], upper[1])
                    for x in (lower[0], upper[0])
                ]
            )
            mesh.CreateFaceVertexCountsAttr([4] * 6)
            mesh.CreateFaceVertexIndicesAttr([0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5])
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateExtentAttr([lower, upper])
            UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
            if index % 2:
                # The other fixtures require the production geometry fallback.
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
            self.assets.append(asset)
            self.asset_text.append(asset.GetRootLayer().ExportToString())
        self.reference_patch = patch.object(
            HUMANOID,
            "stage_utils",
            types.SimpleNamespace(add_reference_to_stage=self.add_reference),
            create=True,
        )
        self.reference_patch.start()
        self.addCleanup(self.reference_patch.stop)
        self.root_patch = patch.object(HUMANOID, "get_assets_root_path", return_value="/fixture-assets", create=True)
        self.asset_root = self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def add_reference(self, *, usd_path, path):
        index = len(self.references) % len(self.assets)
        self.references.append((usd_path, path))
        prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        prim.GetReferences().AddReference(self.assets[index].GetRootLayer().identifier)
        return prim

    def example(self, *, walking=True):
        with patch.dict(HUMANOID.os.environ, {"G1_GESTURE_WALKING": "1" if walking else "0"}):
            example = HUMANOID.HumanoidExample()
        center_x, center_y, yaw = example._work_surface_placements[0]
        example._get_work_surface_tops = Mock(
            return_value=[(center_x, center_y, yaw, example._front_work_surface_height, 0.918, 0.422)]
        )
        return example

    @staticmethod
    def bounds(prim):
        return (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            .ComputeWorldBound(prim)
            .ComputeAlignedRange()
        )

    def test_centimetre_assets_keep_native_scale_and_are_centred_above_floor(self):
        example = self.example()
        example._create_sample_boxes()
        expected = (
            ("A_01", 1.5, 1.0, 0.22),
            ("B_01", 1.5, -1.0, 0.28),
            ("C_01", 2.6, 1.15, 0.34),
            ("D_04", 2.6, -1.15, 0.40),
        )
        self.assertEqual(len(self.references), 4)
        for index, (asset_name, x, y, size) in enumerate(expected):
            path = example._spawned_sample_box_paths[6 + index]
            with self.subTest(path=path):
                self.assertEqual(
                    self.references[index],
                    (
                        f"/fixture-assets/Isaac/Environments/Simple_Warehouse/Props/SM_CardBox{asset_name}.usd",
                        path + "/Asset",
                    ),
                )
                prim = self.stage.GetPrimAtPath(path)
                bounds = self.bounds(prim)
                self.assertFalse(bounds.IsEmpty())
                self.assertAlmostEqual(bounds.GetMidpoint()[0], x, places=6)
                self.assertAlmostEqual(bounds.GetMidpoint()[1], y, places=6)
                self.assertAlmostEqual(bounds.GetMin()[2], 0.005, places=6)
                self.assertAlmostEqual(max(bounds.GetSize()[0], bounds.GetSize()[1]), size, places=6)
                asset = self.stage.GetPrimAtPath(path + "/Asset")
                self.assertEqual(asset.GetAttribute("xformOp:scale").Get(), Gf.Vec3f(0.01))
                self.assertTrue(prim.GetAttribute("xformOp:scale").HasAuthoredValueOpinion())
                self.assertEqual(self.assets[index].GetRootLayer().ExportToString(), self.asset_text[index])
        self.assertFalse(self.stage.GetCompositionErrors())

    def test_each_carton_has_one_dynamic_wrapper_and_convex_geometry_colliders(self):
        example = self.example()
        example._create_sample_boxes()
        for path, mass in zip(example._spawned_sample_box_paths[6:], (0.25, 0.35, 0.50, 0.70)):
            with self.subTest(path=path):
                prim = self.stage.GetPrimAtPath(path)
                bodies = [item for item in Usd.PrimRange(prim) if item.HasAPI(UsdPhysics.RigidBodyAPI)]
                self.assertEqual([str(body.GetPath()) for body in bodies], [path])
                rigid = UsdPhysics.RigidBodyAPI(prim)
                self.assertTrue(rigid.GetRigidBodyEnabledAttr().Get())
                self.assertFalse(rigid.GetKinematicEnabledAttr().Get())
                self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))
                mesh = self.stage.GetPrimAtPath(path + "/Asset/Mesh")
                self.assertTrue(mesh.HasAPI(UsdPhysics.CollisionAPI))
                self.assertEqual(UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get(), "convexHull")
                self.assertAlmostEqual(UsdPhysics.MassAPI(prim).GetMassAttr().Get(), mass)
                self.assertAlmostEqual(prim.GetAttribute("g1:packageMass").Get(), mass)
                self.assertFalse(prim.GetAttribute("g1:packageOnSurface").Get())
                self.assertAlmostEqual(
                    prim.GetAttribute("g1:packageSize").Get(), max(self.bounds(prim).GetSize()[:2]), places=6
                )
                physics = PhysxSchema.PhysxRigidBodyAPI(prim)
                self.assertTrue(physics.GetEnableCCDAttr().Get())
                self.assertEqual(physics.GetSolverPositionIterationCountAttr().Get(), 16)
                self.assertEqual(physics.GetSolverVelocityIterationCountAttr().Get(), 4)

    def test_floor_cartons_join_contact_and_highlight_roots_with_unique_recording_ids(self):
        example = self.example()
        example._create_sample_boxes()
        roots = example._spawned_sample_box_paths
        self.assertEqual(roots, [f"/World/G1_SampleBoxes/Box_{index:02d}" for index in range(13)])
        self.assertEqual([example._parse_sample_box_index(root) for root in roots], list(range(13)))
        robot_root = "/World/G1"
        UsdGeom.Xform.Define(self.stage, robot_root)
        for side, prefix in (("left", "L"), ("right", "R")):
            for name in (
                "thumb_proximal",
                "index_proximal",
                "middle_proximal",
                "ring_proximal",
                "pinky_proximal",
                "hand_base_link",
            ):
                prim = UsdGeom.Xform.Define(self.stage, f"{robot_root}/{side}_hand/{prefix}_{name}").GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(prim)
        reader = CONTACTS.ContactReader()
        reader.prepare(self.stage, robot_root, roots)
        self.assertEqual(set(reader.object_paths), set(roots))
        self.assertEqual(len(reader.sensor_paths), 12)
        highlights = HIGHLIGHTS.MaterialHighlights()
        highlights.prepare(self.stage, roots)
        material = UsdShade.Material.Define(self.stage, "/World/TestHighlight")
        for root in roots[6:]:
            # A selected child collider resolves to its one registered object.
            matches = [path for path in reader.object_paths if reader._within(root + "/Asset/Mesh", path)]
            self.assertEqual(matches, [root])
            highlights.set(self.stage, "gaze", root, material)
            self.assertEqual(
                UsdShade.Shader(self.stage.GetPrimAtPath(highlights._shader_paths[root])).GetInput("opacity").Get(), 1.0
            )
            self.assertFalse(self.stage.GetPrimAtPath(root).GetRelationship("material:binding:full"))
        highlights.clear()

    def test_walking_layout_keeps_spawn_and_table_clear_with_front_pickup_station(self):
        example = self.example()
        self.assertEqual(example._g1_locomotion, "policy")
        self.assertEqual(example._work_surface_placements[0], (2.0, -2.0, 90.0))
        example._create_sample_boxes()
        bounds = [self.bounds(self.stage.GetPrimAtPath(path)) for path in example._spawned_sample_box_paths]
        for carton in bounds[6:12]:
            self.assertGreater(carton.GetMin()[0], 1.0)
            # More than 1.5 m remains clear across the central walking aisle.
            self.assertTrue(carton.GetMin()[1] > 0.75 or carton.GetMax()[1] < -0.75)
            # The practice table occupies X=1.789..2.211, Y=-2.459..-1.541.
            # Later tall cartons can share its Y range because they are farther
            # down the aisle; the four original low cartons remain beside it.
            self.assertTrue(carton.GetMin()[1] > -1.541 + 0.05 or carton.GetMin()[0] > 2.211 + 0.05)
        # The final tall carton is deliberately on the central approach line.
        # Keep the start turn footprint empty without pretending the route past
        # this physical obstacle remains an unobstructed straight corridor.
        self.assertGreater(bounds[12].GetMin()[0], 2 * 0.38 * 2**0.5)
        self.assertAlmostEqual(bounds[12].GetMidpoint()[1], 0.0)
        for index, first in enumerate(bounds):
            for second in bounds[index + 1 :]:
                gap = max(
                    max(first.GetMin()[axis] - second.GetMax()[axis], second.GetMin()[axis] - first.GetMax()[axis])
                    for axis in (0, 1)
                )
                self.assertGreater(gap, 0.0)

    def test_stationary_scene_keeps_six_practice_objects_and_front_table_without_asset_fetch(self):
        example = self.example(walking=False)
        self.assertEqual(example._g1_locomotion, "stationary")
        self.assertFalse(example._gesture_locomotion_enabled)
        self.assertEqual(example._work_surface_placements[0], (0.46, 0.0, 90.0))
        example._create_sample_boxes()
        self.assertEqual(len(example._spawned_sample_box_paths), 6)
        self.assertEqual(self.references, [])
        self.asset_root.assert_not_called()
        self.assertFalse(self.stage.GetPrimAtPath("/World/G1_SampleBoxes/Box_06"))
        for path in example._spawned_sample_box_paths:
            prim = self.stage.GetPrimAtPath(path)
            self.assertTrue(prim.GetAttribute("g1:packageOnSurface").Get())
            self.assertAlmostEqual(self.bounds(prim).GetMin()[2], 0.923, places=6)

    def test_scene_reload_rebuilds_the_same_thirteen_roots_without_retaining_old_ids(self):
        example = self.example()
        example._create_sample_boxes()
        original_roots = tuple(example._spawned_sample_box_paths)
        self.stage = Usd.Stage.CreateInMemory()
        example._create_sample_boxes()
        self.assertEqual(tuple(example._spawned_sample_box_paths), original_roots)
        self.assertEqual(len(set(example._spawned_sample_box_paths)), 13)
        self.assertTrue(all(self.stage.GetPrimAtPath(path).IsValid() for path in original_roots))

    def test_missing_asset_root_fails_explicitly_without_registering_phantom_cartons(self):
        example = self.example()
        self.asset_root.return_value = None
        with self.assertRaisesRegex(RuntimeError, "warehouse carton assets"):
            example._create_sample_boxes()
        self.assertEqual(len(example._spawned_sample_box_paths), 6)
        self.assertEqual(self.references, [])


if __name__ == "__main__":
    unittest.main()
