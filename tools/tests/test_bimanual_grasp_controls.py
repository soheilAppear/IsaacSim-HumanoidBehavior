# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify cross-hand contact evidence and force-capacity arithmetic without Kit."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
)


def load(name):
    spec = importlib.util.spec_from_file_location(name + "_bimanual_test", ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MONITOR = load("bimanual_grasp")
ContactSample = load("grasp_controller").ContactSample
CURVED = dict.fromkeys(("thumb", "index", "middle", "ring", "little"), 0.12)


class TestBimanualGraspControls(unittest.TestCase):
    def setUp(self):
        self.monitor = MONITOR.BimanualGraspMonitor()
        self.left = ContactSample("/Box", "palm", (0, 0.2, 0.7), (0, 1, 0), 4)
        self.right = ContactSample("/Box", "palm", (0, -0.2, 0.7), (0, -1, 0), 4)

    def update(self, **changes):
        values = dict(
            dt=0.01,
            contacts_by_side={"left": [self.left], "right": [self.right]},
            curls_by_side={"left": dict(CURVED), "right": dict(CURVED)},
            input_valid_by_side={"left": True, "right": True},
            object_masses_kg={"/Box": 0.45},
        )
        values.update(changes)
        return self.monitor.update(**values)

    def settle(self, **changes):
        result = None
        for _ in range(6):
            result = self.update(**changes)
        return result

    def test_opposing_palms_on_one_box_complete_dwell_without_individual_wraps(self):
        for _ in range(5):
            self.assertFalse(self.update().contact_supported)
        result = self.update()
        self.assertTrue(result.contact_supported)
        self.assertTrue(result.opposing_contacts)
        self.assertEqual(result.state, "contact_supported")
        self.assertEqual(result.object_path, "/Box")
        self.assertEqual(result.normal_force_n_by_side, {"left": 4, "right": 4})
        self.assertEqual(result.contact_count_by_side, {"left": 1, "right": 1})
        self.assertAlmostEqual(result.estimated_friction_support_n, 0.6 * (4 + 4))
        self.assertEqual(result.measured_normal_vertical_n, 0)
        self.assertAlmostEqual(result.required_vertical_support_n, 0.45 * 9.81)
        self.assertTrue(result.capacity_exceeds_weight)

    def test_contact_geometry_does_not_claim_sufficient_force_or_a_lift(self):
        self.left = replace(self.left, normal_force_n=1)
        self.right = replace(self.right, normal_force_n=1)
        result = self.settle()
        self.assertTrue(result.contact_supported)
        self.assertFalse(result.capacity_exceeds_weight)
        self.assertIn("not a confirmed lift", result.detail)

    def test_vertical_normal_force_has_object_sign_and_correct_friction_projection(self):
        self.left = replace(self.left, normal_world=(0, math.sqrt(0.75), 0.5))
        result = self.settle()
        self.assertTrue(result.contact_supported)
        self.assertAlmostEqual(result.measured_normal_vertical_n, -2)
        self.assertAlmostEqual(result.estimated_friction_support_n, 0.6 * (4 * math.sqrt(0.75) + 4))
        self.assertAlmostEqual(result.estimated_total_vertical_capacity_n, -2 + 0.6 * (4 * math.sqrt(0.75) + 4))
        self.assertFalse(result.capacity_exceeds_weight)

    def test_nonunit_normals_are_normalized_without_scaling_measured_force(self):
        self.left = replace(self.left, normal_world=(0, 8, 0))
        result = self.settle()
        self.assertTrue(result.contact_supported)
        self.assertEqual(result.normal_force_n_by_side["left"], 4)
        self.assertAlmostEqual(result.estimated_total_vertical_capacity_n, 4.8)

    def test_different_objects_are_never_combined_into_bimanual_support(self):
        self.right = replace(self.right, object_path="/Other")
        result = self.settle()
        self.assertFalse(result.contact_supported)
        self.assertEqual(sum(result.normal_force_n_by_side.values()), 4)
        self.assertEqual(sum(result.contact_count_by_side.values()), 1)

    def test_same_side_normals_and_coincident_points_do_not_support(self):
        original = self.right
        for modified in (
            replace(original, normal_world=(0, 1, 0)),
            replace(original, point_world=self.left.point_world),
        ):
            self.right = modified
            self.assertFalse(self.settle().contact_supported)

    def test_scenery_proximity_zero_force_and_invalid_contacts_cannot_support(self):
        original = self.left
        for modified in (
            replace(original, eligible_for_grasp=False),
            replace(original, separation_m=0.003),
            replace(original, normal_force_n=0),
            replace(original, normal_force_n=math.nan),
            replace(original, normal_world=(0, 0, 0)),
            replace(original, point_world=(0, math.inf, 0)),
            replace(original, role="robot_torso"),
        ):
            self.left = modified
            result = self.settle()
            self.assertFalse(result.contact_supported)
            self.assertEqual(result.support_elapsed_s, 0)

    def test_tracking_loss_contact_health_and_open_hand_reset_dwell(self):
        invalid_cases = (
            {"input_valid_by_side": {"left": True, "right": False}},
            {"contact_health": "capacity_exceeded"},
            {"curls_by_side": {"left": CURVED, "right": dict.fromkeys(CURVED, 0.0)}},
            {"contacts_by_side": {"left": [self.left], "right": []}},
        )
        for changes in invalid_cases:
            self.assertTrue(self.settle().contact_supported)
            self.assertFalse(self.update(**changes).contact_supported)
            self.assertFalse(self.update().contact_supported)

    def test_switching_objects_or_reset_requires_a_new_dwell(self):
        self.assertTrue(self.settle().contact_supported)
        self.left = replace(self.left, object_path="/Other")
        self.right = replace(self.right, object_path="/Other")
        self.assertFalse(self.update().contact_supported)
        self.assertTrue(self.settle().contact_supported)
        self.monitor.reset()
        self.assertFalse(self.update().contact_supported)

    def test_missing_mass_never_invents_weight_but_preserves_measured_contacts(self):
        for mass in (None, -1, math.nan, True, "0.45"):
            result = self.settle(object_masses_kg={"/Box": mass})
            self.assertTrue(result.contact_supported)
            self.assertIsNone(result.mass_kg)
            self.assertIsNone(result.required_vertical_support_n)
            self.assertIsNone(result.capacity_exceeds_weight)

    def test_delayed_step_cannot_manufacture_dwell_and_invalid_timestep_resets(self):
        self.assertFalse(self.update(dt=5).contact_supported)
        self.assertAlmostEqual(self.monitor.result.support_elapsed_s, 0.05)
        self.assertTrue(self.update().contact_supported)
        for dt in (0, -1, math.nan, True):
            self.assertEqual(self.update(dt=dt).state, "unavailable")
            self.assertFalse(self.update().contact_supported)

    def test_all_measured_manifold_forces_are_counted_once_for_selected_object(self):
        extra = replace(self.left, role="index", point_world=(0, 0.2, 0.72), normal_force_n=2)
        contacts = {"left": [self.left, extra], "right": [self.right]}
        before = dict(CURVED)
        result = self.settle(contacts_by_side=contacts)
        self.assertEqual(result.normal_force_n_by_side, {"left": 6, "right": 4})
        self.assertEqual(result.contact_count_by_side, {"left": 2, "right": 1})
        self.assertAlmostEqual(result.estimated_friction_support_n, 6)
        self.assertEqual(CURVED, before)
        self.assertEqual(contacts["left"], [self.left, extra])

    def test_contact_capacity_failure_is_explicit_and_does_not_truncate(self):
        result = self.update(contacts_by_side={"left": [self.left] * 129, "right": [self.right]})
        self.assertEqual(result.state, "unavailable")
        self.assertIn("capacity exceeded", result.detail)
        self.assertEqual(result.normal_force_n_by_side, {"left": 0, "right": 0})

    def test_small_measured_forces_are_reported_but_do_not_establish_opposition(self):
        self.left = replace(self.left, normal_force_n=0.01)
        result = self.settle()
        self.assertFalse(result.contact_supported)
        self.assertEqual(result.normal_force_n_by_side, {"left": 0.01, "right": 4})
        self.assertEqual(result.contact_count_by_side, {"left": 1, "right": 1})

    def test_invalid_curls_friction_and_overflow_fail_closed(self):
        for value in (math.nan, 1.1, -0.1, True):
            curls = dict(CURVED, index=value)
            self.assertEqual(self.update(curls_by_side={"left": curls, "right": CURVED}).state, "unavailable")
        for coefficient in (math.nan, -1, True):
            self.assertEqual(self.update(friction_coefficient=coefficient).state, "unavailable")
        self.left = replace(self.left, normal_force_n=1e308)
        self.right = replace(self.right, normal_force_n=1e308)
        self.assertEqual(self.update(friction_coefficient=2).state, "unavailable")


if __name__ == "__main__":
    unittest.main()
