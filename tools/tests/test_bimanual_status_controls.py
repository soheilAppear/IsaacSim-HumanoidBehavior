# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report cross-hand evidence without changing the physical grasp or claiming a lift."""

import types
import unittest
from unittest.mock import Mock

from humanoid_test_support import BIMANUAL_GRASP, HUMANOID


class TestBimanualStatusControls(unittest.TestCase):
    def setUp(self):
        self.ex = HUMANOID.HumanoidExample()
        self.path = "/World/Packages/Box_20"
        self.support = BIMANUAL_GRASP.BimanualGraspResult(
            state="contact_supported",
            object_path=self.path,
            contact_supported=True,
            opposing_contacts=True,
            normal_force_n_by_side={"left": 2.0, "right": 3.0},
            estimated_total_vertical_capacity_n=3.0,
            required_vertical_support_n=1.4715,
            capacity_exceeds_weight=True,
        )

    def test_candidate_accepts_cross_hand_evidence_and_clears_when_lost(self):
        self.ex._grasp_mode = "physical"
        self.ex._physical_grasp_results.clear()
        self.ex._bimanual_grasp_result = self.support
        self.ex._set_grab_candidate = Mock()
        self.ex._get_active_hand_world_position = Mock(side_effect=AssertionError("must not assist physical grasp"))
        for side in ("left", "right"):
            self.ex._update_grabbed_object(side, True)
            self.ex._set_grab_candidate.assert_called_with(side, self.path)
        self.ex._bimanual_grasp_result = BIMANUAL_GRASP.BimanualGraspResult()
        self.ex._update_grabbed_object("right", True)
        self.ex._set_grab_candidate.assert_called_with("right", None)
        self.ex._physical_grasp_results["left"] = types.SimpleNamespace(object_path="/World/Packages/Small")
        self.ex._update_grabbed_object("left", True)
        self.ex._set_grab_candidate.assert_called_with("left", "/World/Packages/Small")
        self.ex._get_active_hand_world_position.assert_not_called()
        self.assertFalse(self.ex._grabbed_objects_by_side)

    def test_disabling_finger_control_discards_old_cross_hand_support(self):
        self.ex._bimanual_grasp_result = self.support
        self.ex._bimanual_grasp_monitor.result = self.support
        self.ex._finger_control_enabled = False
        self.ex._update_g1_fingers()
        self.assertFalse(self.ex._bimanual_grasp_result.contact_supported)
        self.assertIsNone(self.ex._bimanual_grasp_result.object_path)
        self.assertEqual(self.ex._latest_hand_contacts, {"left": [], "right": []})

    def test_hand_rows_keep_per_hand_and_bimanual_meanings_separate(self):
        self.ex._bimanual_grasp_result = self.support
        self.ex._get_xr_input_device = Mock(return_value=None)
        self.ex._get_raw_hand_or_controller_pose = Mock(return_value=None)
        self.ex._get_xr_gesture_value = Mock(return_value=0.0)
        self.ex._get_hand_closure = Mock(return_value=0.0)
        self.ex._physical_grasp_results = {
            side: types.SimpleNamespace(contact_supported=False, object_path=None, mode=None)
            for side in ("left", "right")
        }
        self.ex._collect_hand_tracking_sample()
        row = self.ex._hand_tracking_records[-1]
        for side in ("left", "right"):
            self.assertEqual(row[f"{side}_contact_supported"], 0)
            self.assertIsNone(row[f"{side}_contact_object"])
        self.assertEqual(row["bimanual_contact_supported"], 1)
        self.assertEqual(row["bimanual_object_path"], self.path)
        self.assertEqual(row["bimanual_left_normal_force_n"], 2.0)
        self.assertEqual(row["bimanual_right_normal_force_n"], 3.0)
        self.assertEqual(row["bimanual_capacity_exceeds_weight"], 1)
        self.assertEqual(row["bimanual_required_vertical_support_n"], 1.4715)
        self.assertTrue(all(type(value) in (str, int, float, bool, type(None)) for value in row.values()))
        # Later monitor updates cannot alter an accepted recording row.
        self.support.normal_force_n_by_side["left"] = 9.0
        self.ex._bimanual_grasp_result = BIMANUAL_GRASP.BimanualGraspResult()
        self.assertEqual(row["bimanual_left_normal_force_n"], 2.0)
        self.ex._collect_hand_tracking_sample()
        self.assertEqual(self.ex._hand_tracking_records[-1]["bimanual_contact_supported"], 0)
        self.assertIsNone(self.ex._hand_tracking_records[-1]["bimanual_capacity_exceeds_weight"])


if __name__ == "__main__":
    unittest.main()
