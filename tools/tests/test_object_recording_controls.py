# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch package observations without changing recorded rows or authoring rigid bodies."""

from __future__ import annotations

import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from humanoid_test_support import BIMANUAL_GRASP, HUMANOID, USD
from pxr import Gf, Usd, UsdGeom, UsdPhysics


class PackageSimulation:
    def __init__(self, rows):
        self.is_valid = True
        self.rows = rows
        self.views = []
        self.create_rigid_body_view = Mock(side_effect=self.create)

    def create(self, paths):
        # Native row ordering need not match USD sibling ordering.
        order = list(reversed(paths))
        view = types.SimpleNamespace(prim_paths=order)
        view.get_transforms = Mock(
            side_effect=lambda: types.SimpleNamespace(numpy=lambda: np.asarray([self.rows[path] for path in order]))
        )
        self.views.append(view)
        return view


class TestObjectRecordingControls(unittest.TestCase):
    def setUp(self):
        self.ex = HUMANOID.HumanoidExample()
        self.ex._sample_box_root_path = "/World/Packages"
        self.stage = self.make_stage()
        self.context = types.SimpleNamespace(get_stage=lambda: self.stage)
        self.addCleanup(patch.stopall)
        patch.object(USD, "get_context", return_value=self.context).start()
        patch.object(HUMANOID.time, "time", return_value=100.0).start()
        self.paths = ("/World/Packages/Box_00", "/World/Packages/Box_01")
        self.rows = {
            self.paths[0]: [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            self.paths[1]: [4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 1.0],
        }
        self.simulation = PackageSimulation(self.rows)
        self.manager = types.SimpleNamespace(get_physics_simulation_view=Mock(side_effect=lambda: self.simulation))
        patch.object(HUMANOID, "SimulationManager", self.manager, create=True).start()
        self.ex._get_object_world_pose = Mock(side_effect=self.individual_pose)
        self.ex._world_settings["physics_dt"] = 0.005
        self.ex._behavioral_data_log_every_n_steps = 2
        self.ex._behavioral_data_step_counter = 100
        self.ex._headset_gait_time = 4.0
        self.ex._grabbed_objects_by_side = {"left": self.paths[0]}
        self.ex._physical_grasp_results = {
            "left": types.SimpleNamespace(contact_supported=True, object_path=self.paths[0])
        }
        self.ex._contact_health = "ok"

    @staticmethod
    def make_stage():
        stage = Usd.Stage.CreateInMemory()
        for path in ("/World/Packages/Box_00", "/World/Packages/Box_01"):
            UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
        UsdGeom.Xform.Define(stage, "/World/Packages/DecorativeLabels")
        return stage

    def individual_pose(self, path):
        values = self.rows[path]
        return Gf.Vec3d(*values[:3]), Gf.Quatd(values[6], Gf.Vec3d(*values[3:6])).GetNormalized()

    def test_batch_matches_individual_rows_order_quaternions_and_velocities(self):
        self.ex._collect_object_state_sample()
        first = list(self.ex._object_state_records)
        self.assertEqual([row["object_path"] for row in first], list(self.paths))
        self.ex._get_object_world_pose.assert_not_called()
        self.simulation.create_rigid_body_view.assert_called_once_with(list(self.paths))
        self.assertEqual(self.simulation.views[0].get_transforms.call_count, 1)
        self.assertAlmostEqual(first[1]["qw"], 2**-0.5, places=6)
        self.assertAlmostEqual(first[1]["qz"], 2**-0.5, places=6)
        self.rows[self.paths[0]][0] += 0.01
        self.ex._collect_object_state_sample()
        second = self.ex._object_state_records[-2:]
        self.assertEqual(second[0]["vel_x"], 1.0)
        self.assertEqual(self.simulation.views[0].get_transforms.call_count, 2)
        self.assertEqual(self.simulation.create_rigid_body_view.call_count, 1)

        self.ex._object_state_records.clear()
        self.ex._object_state_prev_positions.clear()
        self.rows[self.paths[0]][0] -= 0.01
        with patch.object(self.ex, "_get_object_state_pose_batch", return_value=None):
            self.ex._collect_object_state_sample()
            self.rows[self.paths[0]][0] += 0.01
            self.ex._collect_object_state_sample()
        self.assertEqual(self.ex._object_state_records, first + second)

    def test_read_only_view_never_includes_decorations_or_authors_usd(self):
        before = self.stage.GetRootLayer().ExportToString()
        self.ex._collect_object_state_sample()
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), before)
        self.assertFalse(self.stage.GetPrimAtPath("/World/Packages/DecorativeLabels").HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertEqual(len(self.ex._object_state_records), 2)

    def test_cross_hand_support_is_recorded_without_extra_physics_reads(self):
        self.ex._physical_grasp_results.clear()
        self.ex._grabbed_objects_by_side.clear()
        self.ex._bimanual_grasp_result = BIMANUAL_GRASP.BimanualGraspResult(
            state="contact_supported", object_path=self.paths[0], contact_supported=True, opposing_contacts=True
        )
        self.ex._collect_object_state_sample()
        supported, other = self.ex._object_state_records
        self.assertEqual(supported["contact_supported"], 1)
        self.assertEqual(supported["contact_hand"], "left|right")
        self.assertEqual(supported["bimanual_contact_supported"], 1)
        self.assertEqual(supported["bimanual_opposing_contacts"], 1)
        self.assertEqual(supported["is_grabbed"], 0)
        self.assertEqual(other["contact_supported"], 0)
        self.assertEqual(other["contact_hand"], "")
        self.assertEqual(other["bimanual_contact_supported"], 0)
        self.assertEqual(other["bimanual_opposing_contacts"], 0)
        self.ex._get_object_world_pose.assert_not_called()
        self.assertEqual(self.simulation.views[0].get_transforms.call_count, 1)

    def test_new_opposing_contacts_do_not_report_sustained_support_early(self):
        self.ex._physical_grasp_results.clear()
        self.ex._bimanual_grasp_result = BIMANUAL_GRASP.BimanualGraspResult(
            state="contact_candidate", object_path=self.paths[0], opposing_contacts=True
        )
        self.ex._collect_object_state_sample()
        row = self.ex._object_state_records[0]
        self.assertEqual(row["contact_supported"], 0)
        self.assertEqual(row["bimanual_contact_supported"], 0)
        self.assertEqual(row["bimanual_opposing_contacts"], 1)

    def test_simulation_stage_and_membership_changes_rebuild_the_view(self):
        self.ex._collect_object_state_sample()
        original = self.simulation
        self.simulation = PackageSimulation(self.rows)
        self.ex._collect_object_state_sample()
        self.assertEqual(original.views[0].get_transforms.call_count, 1)
        self.assertEqual(self.simulation.create_rigid_body_view.call_count, 1)
        self.stage = self.make_stage()
        self.ex._collect_object_state_sample()
        self.assertEqual(self.simulation.create_rigid_body_view.call_count, 2)
        self.stage.RemovePrim(self.paths[1])
        self.ex._collect_object_state_sample()
        self.assertEqual(self.simulation.create_rigid_body_view.call_count, 3)
        self.simulation.create_rigid_body_view.assert_called_with([self.paths[0]])
        new_path = "/World/Packages/Box_02"
        self.rows[new_path] = [7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0]
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(self.stage, new_path).GetPrim())
        self.ex._collect_object_state_sample()
        self.simulation.create_rigid_body_view.assert_called_with([self.paths[0], new_path])

    def test_batch_failure_falls_back_without_dropping_any_valid_row(self):
        self.ex._collect_object_state_sample()
        view = self.simulation.views[0]
        for invalid in (RuntimeError("view expired"), np.zeros((1, 7))):
            with self.subTest(invalid=str(invalid)):
                view.get_transforms.side_effect = (
                    invalid if isinstance(invalid, Exception) else lambda: types.SimpleNamespace(numpy=lambda: invalid)
                )
                self.ex._object_state_pose_batch = {
                    "stage": self.stage,
                    "simulation": self.simulation,
                    "paths": self.paths,
                    "order": tuple(reversed(self.paths)),
                    "view": view,
                }
                self.ex._object_state_records.clear()
                self.ex._get_object_world_pose.reset_mock()
                self.ex._collect_object_state_sample()
                self.assertEqual(len(self.ex._object_state_records), 2)
                self.assertEqual(self.ex._get_object_world_pose.call_count, 2)
                self.assertIsNone(self.ex._object_state_pose_batch)

    def test_bad_row_retries_only_that_body_and_preserves_its_neighbors(self):
        self.ex._collect_object_state_sample()
        view = self.simulation.views[0]
        rows = np.asarray([self.rows[path] for path in reversed(self.paths)])
        rows[0, 0] = float("nan")
        view.get_transforms.side_effect = lambda: types.SimpleNamespace(numpy=lambda: rows)
        self.ex._object_state_records.clear()
        self.ex._collect_object_state_sample()
        self.ex._get_object_world_pose.assert_called_once_with(self.paths[1])
        self.assertEqual(len(self.ex._object_state_records), 2)

    def test_incomplete_or_duplicate_native_paths_fall_back(self):
        for paths in ([self.paths[0]], [self.paths[0], self.paths[0]], [self.paths[0], "/Wrong/Body"]):
            with self.subTest(paths=paths):
                self.ex._object_state_pose_batch = None
                self.simulation.create_rigid_body_view.side_effect = lambda requested: types.SimpleNamespace(
                    prim_paths=paths, get_transforms=Mock(side_effect=AssertionError("must not read mismatched view"))
                )
                self.ex._get_object_world_pose.reset_mock()
                self.ex._collect_object_state_sample()
                self.assertEqual(self.ex._get_object_world_pose.call_count, 2)

    def test_missing_stage_root_or_invalid_simulation_does_not_use_stale_view(self):
        self.ex._collect_object_state_sample()
        view = self.simulation.views[0]
        self.simulation.is_valid = False
        self.ex._collect_object_state_sample()
        self.assertEqual(view.get_transforms.call_count, 1)
        self.assertIsNone(self.ex._object_state_pose_batch)
        self.stage.RemovePrim("/World/Packages")
        self.ex._collect_object_state_sample()
        self.stage = None
        self.ex._collect_object_state_sample()
        self.assertIsNone(self.ex._object_state_pose_batch)

    def test_stop_and_teleoperation_reset_discard_native_batch(self):
        self.ex._reset_gesture_locomotion = Mock()
        self.ex._set_grab_candidate = Mock()
        self.ex._release_grabbed_object = Mock()
        self.ex._collect_object_state_sample()
        events = types.SimpleNamespace(STOP=2, PAUSE=3)
        with patch.object(HUMANOID, "import_module", return_value=types.SimpleNamespace(TimelineEventType=events)):
            self.ex._on_navigation_timeline_event(types.SimpleNamespace(type=events.STOP))
        self.assertIsNone(self.ex._object_state_pose_batch)
        self.ex._collect_object_state_sample()
        self.ex._reset_teleoperation_state()
        self.assertIsNone(self.ex._object_state_pose_batch)


if __name__ == "__main__":
    unittest.main()
