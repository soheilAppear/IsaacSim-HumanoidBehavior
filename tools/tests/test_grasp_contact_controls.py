# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify USD sensor preparation and the PhysX packed contact-data boundary."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

PACKAGE = "contact_reader_test_package"
MODULE_DIR = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
)
package = types.ModuleType(PACKAGE)
package.__path__ = [str(MODULE_DIR)]
sys.modules[PACKAGE] = package
ContactReader = importlib.import_module(PACKAGE + ".grasp_contacts").ContactReader


class Array:
    def __init__(self, data):
        self.data = np.asarray(data)

    def numpy(self):
        return self.data


class ContactView:
    def __init__(self, sensor_paths, capacity=32):
        # Real Tensor views expose their own order; input order is not assumed.
        self.sensor_paths = tuple(reversed(sensor_paths))
        self.capacity = capacity
        self.records = []
        self.paths_by_id = {11: "/World/Cube/Asset/Mesh", 12: "/World/Cylinder", 13: "/World/Cube_Extra"}
        self.mapping_calls = 0
        self.last_dt = None
        self.mutate_raw = lambda data: data

    def get_raw_contact_data(self, dt):
        self.last_dt = dt
        forces = np.zeros((self.capacity, 1), dtype=float)
        points = np.zeros((self.capacity, 3), dtype=float)
        normals = np.zeros((self.capacity, 3), dtype=float)
        separations = np.zeros((self.capacity, 1), dtype=float)
        counts = np.zeros(len(self.sensor_paths), dtype=np.uint32)
        starts = np.zeros_like(counts)
        ids = np.zeros(self.capacity, dtype=np.uint64)
        index = 0
        for sensor, path in enumerate(self.sensor_paths):
            starts[sensor] = index
            for record in self.records:
                if record[0] != path:
                    continue
                _, actor, impulse, point, normal, gap = record
                forces[index] = impulse / dt
                points[index] = point
                normals[index] = normal
                separations[index] = gap
                ids[index] = actor
                index += 1
                counts[sensor] += 1
        data = self.mutate_raw([forces, points, normals, separations, counts, starts, ids])
        return tuple(Array(value) for value in data)

    def get_other_actor_paths_from_ids(self, ids):
        self.mapping_calls += 1
        if ids.data.dtype != np.uint64:
            raise TypeError("Actor IDs must remain uint64")
        return [self.paths_by_id[int(value)] for value in ids.numpy()]


class SimulationView:
    def __init__(self):
        self.created = []

    def create_rigid_contact_view(self, paths, **kwargs):
        view = ContactView(paths, kwargs["max_contact_data_count"])
        self.created.append(view)
        return view


def make_stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/G1")
    UsdGeom.Cube.Define(stage, "/World/Cube")
    UsdGeom.Cylinder.Define(stage, "/World/Cylinder")
    for side, prefix in (("left", "L"), ("right", "R")):
        for name in (
            "thumb_proximal",
            "index_proximal",
            "middle_proximal",
            "ring_proximal",
            "pinky_proximal",
            "hand_base_link",
        ):
            path = f"/World/G1/{side}_hand/{prefix}_{name}"
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdGeom.Cube.Define(stage, path + "/Collision")
    return stage


class TestGraspContactReader(unittest.TestCase):
    def setUp(self):
        self.stage = make_stage()
        self.reader = ContactReader(max_contact_count=32)
        self.reader.prepare(self.stage, "/World/G1", ["/World/Cube", "/World/Cylinder"])
        self.simulation = SimulationView()
        self.engine = "physx"
        manager = types.SimpleNamespace(
            get_active_physics_engine=lambda: self.engine,
            get_physics_simulation_view=lambda: self.simulation,
        )
        module = types.ModuleType("isaacsim.core.simulation_manager")
        module.SimulationManager = manager
        warp = types.ModuleType("warp")
        warp.uint64 = np.uint64

        def cpu_array(values, *, dtype, device):
            self.assertEqual(device, "cpu")
            return Array(np.array(values, dtype=dtype))

        warp.array = cpu_array
        self.modules = mock.patch.dict(sys.modules, {module.__name__: module, "warp": warp})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def ready(self):
        contacts, status = self.reader.read(0.01)
        self.assertEqual(status, "ok")
        self.assertEqual(contacts, {"left": [], "right": []})
        return self.simulation.created[-1]

    def test_prepare_only_reports_existing_rigid_links_and_pinky_maps_to_little(self):
        self.assertEqual(len(self.reader.sensor_paths), 12)
        for path in self.reader.sensor_paths:
            prim = self.stage.GetPrimAtPath(path)
            self.assertTrue(prim.HasAPI(PhysxSchema.PhysxContactReportAPI))
            self.assertEqual(PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr().Get(), 0.0)
            mesh = self.stage.GetPrimAtPath(path + "/Collision")
            self.assertFalse(mesh.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertFalse(mesh.HasAPI(PhysxSchema.PhysxContactReportAPI))
        self.assertEqual(self.reader._roles_by_path["/World/G1/left_hand/L_pinky_proximal"], ("left", "little"))
        self.assertEqual(self.reader._roles_by_path["/World/G1/right_hand/R_hand_base_link"], ("right", "palm"))

    def test_read_preserves_force_units_sensor_order_object_boundaries_and_normals(self):
        view = self.ready()
        view.records = [
            ("/World/G1/left_hand/L_thumb_proximal", 11, 0.0004, (1, 2, 3), (0, 0, 1), -0.001),
            ("/World/G1/right_hand/R_hand_base_link", 12, 0.002, (4, 5, 6), (1, 0, 0), 0.0),
            ("/World/G1/right_hand/R_index_proximal", 13, 0.01, (7, 8, 9), (-1, 0, 0), 0.0),
        ]
        original_layer = self.stage.GetRootLayer().ExportToString()
        contacts, status = self.reader.read(0.01)
        self.assertEqual(status, "ok")
        self.assertEqual(view.last_dt, 0.01)
        self.assertEqual(len(contacts["left"]), 1)
        self.assertEqual(len(contacts["right"]), 2)
        left, right = contacts["left"][0], contacts["right"][0]
        self.assertEqual((left.object_path, left.role, left.point_world), ("/World/Cube", "thumb", (1, 2, 3)))
        self.assertAlmostEqual(left.normal_force_n, 0.04)
        self.assertEqual(left.normal_world, (0, 0, 1))
        self.assertTrue(left.eligible_for_grasp)
        self.assertEqual((right.object_path, right.role), ("/World/Cylinder", "palm"))
        self.assertAlmostEqual(right.normal_force_n, 0.2)
        scenery = contacts["right"][1]
        self.assertEqual(scenery.object_path, "/World/Cube_Extra")
        self.assertFalse(scenery.eligible_for_grasp)
        self.assertAlmostEqual(scenery.normal_force_n, 1.0)
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), original_layer)
        self.reader.read(0.02)
        self.assertEqual(view.mapping_calls, 1)
        self.assertAlmostEqual(self.reader.read(0.02)[0]["left"][0].normal_force_n, 0.02)

    def test_full_or_out_of_bounds_buffers_fail_closed(self):
        view = self.ready()

        def full(data):
            data[4][0] = view.capacity
            return data

        view.mutate_raw = full
        self.assertEqual(self.reader.read(0.01), ({"left": [], "right": []}, "contact_buffer_full_or_overflow"))

        def invalid_range(data):
            data[4][0] = 2
            data[5][0] = view.capacity - 1
            return data

        view.mutate_raw = invalid_range
        self.assertEqual(self.reader.read(0.01), ({"left": [], "right": []}, "contact_buffer_overflow"))

    def test_overlapping_ranges_cannot_duplicate_contact_evidence(self):
        view = self.ready()

        def overlap(data):
            data[4][:2] = 1
            return data

        view.mutate_raw = overlap
        self.assertEqual(self.reader.read(0.01), ({"left": [], "right": []}, "overlapping_contact_ranges"))

    def test_nonfinite_force_and_invalid_normals_do_not_supply_contacts(self):
        view = self.ready()
        for force, normal in ((float("nan"), (1, 0, 0)), (0.1, (0, 0, 0)), (0.1, (float("inf"), 0, 0))):
            with self.subTest(force=force, normal=normal):
                view.records = [(self.reader.sensor_paths[0], 11, force, (1, 2, 3), normal, 0)]
                contacts, status = self.reader.read(0.01)
                self.assertEqual(contacts, {"left": [], "right": []})
                self.assertTrue(status.startswith("invalid_contact_samples"))

    def test_new_physics_view_clears_actor_id_cache_even_if_ids_are_reused(self):
        view = self.ready()
        record = (self.reader.sensor_paths[0], 11, 0.001, (0, 0, 0), (1, 0, 0), 0)
        view.records = [record]
        self.assertEqual(self.reader.read(0.01)[0]["left"][0].object_path, "/World/Cube")
        self.simulation = SimulationView()
        replacement = self.ready()
        replacement.paths_by_id[11] = "/World/Cylinder/Child"
        replacement.records = [record]
        self.assertEqual(self.reader.read(0.01)[0]["left"][0].object_path, "/World/Cylinder")
        self.assertEqual(replacement.mapping_calls, 1)

    def test_missing_physics_invalid_dt_and_backend_failure_do_not_replay_contacts(self):
        view = self.ready()
        view.records = [(self.reader.sensor_paths[0], 11, 0.001, (0, 0, 0), (1, 0, 0), 0)]
        self.assertEqual(self.reader.read(0.01)[1], "ok")
        for dt in (0.0, -1.0, float("nan"), float("inf")):
            self.assertEqual(self.reader.read(dt), ({"left": [], "right": []}, "invalid_physics_dt"))
        self.simulation = None
        self.assertEqual(self.reader.read(0.01), ({"left": [], "right": []}, "physics_not_ready"))
        self.assertFalse(self.reader._actor_objects)
        self.engine = "newton"
        self.assertEqual(self.reader.read(0.01), ({"left": [], "right": []}, "unsupported_physics_engine"))

    def test_reset_and_failed_prepare_cannot_reuse_old_sensors(self):
        self.ready()
        self.reader.reset()
        self.assertEqual(self.reader.read(0.01), ({"left": [], "right": []}, "not_prepared"))
        self.stage.RemovePrim("/World/G1/right_hand/R_thumb_proximal")
        with self.assertRaisesRegex(ValueError, "Missing right"):
            self.reader.prepare(self.stage, "/World/G1", ["/World/Cube"])
        self.assertFalse(self.reader.sensor_paths)

    def test_invalidate_retains_schema_preparation_and_recreates_view(self):
        self.ready()
        paths = self.reader.sensor_paths
        layer = self.stage.GetRootLayer().ExportToString()
        self.reader._actor_objects[11] = "/World/Cube"
        self.reader.invalidate()
        self.assertEqual(self.reader.sensor_paths, paths)
        self.assertFalse(self.reader._actor_objects)
        self.assertIsNone(self.reader._view)
        self.ready()
        self.assertEqual(len(self.simulation.created), 2)
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), layer)

    def test_tensor_exception_returns_health_and_drops_cached_view(self):
        view = self.ready()
        view.get_raw_contact_data = mock.Mock(side_effect=RuntimeError("destroyed tensor view"))
        contacts, status = self.reader.read(0.01)
        self.assertEqual(contacts, {"left": [], "right": []})
        self.assertIn("RuntimeError: destroyed tensor view", status)
        self.assertIsNone(self.reader._view)
        self.assertIsNone(self.reader._simulation_view)


if __name__ == "__main__":
    unittest.main()
