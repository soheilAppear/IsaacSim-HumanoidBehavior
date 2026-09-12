# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure numeric tests for actual-contact evidence and compliant finger commands."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples"
    / "interactive/humanoid/grasp_controller.py"
)
SPEC = importlib.util.spec_from_file_location("contact_grasp_controller_test_source", SOURCE)
GRASP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GRASP
SPEC.loader.exec_module(GRASP)
ContactSample = GRASP.ContactSample
ContactGraspController = GRASP.ContactGraspController


def pinch_contacts(path="/World/Cube"):
    return [
        ContactSample(path, "thumb", (-0.02, 0, 0), (-1, 0, 0), 0.8),
        ContactSample(path, "index", (0.02, 0, 0), (1, 0, 0), 0.9),
    ]


class TestContactGraspControls(unittest.TestCase):
    def tick(self, controller, contacts=(), **overrides):
        args = dict(
            dt=0.01,
            requested_curls={"thumb": 0.35, "index": 0.45},
            measured_curls={role: 0.3 for role in GRASP.ACTUATOR_ROLES},
            closing_lead={role: 0.04 for role in GRASP.ACTUATOR_ROLES},
            contacts=contacts,
        )
        args.update(overrides)
        return controller.update(**args)

    def settle(self, controller, contacts=(), **overrides):
        for _ in range(10):
            result = self.tick(controller, contacts, **overrides)
        return result

    def test_pinch_needs_sustained_same_object_opposing_contacts_without_full_fist(self):
        controller = ContactGraspController()
        for _ in range(5):
            result = self.tick(controller, pinch_contacts())
            self.assertFalse(result.contact_supported)
        result = self.tick(controller, pinch_contacts())
        self.assertTrue(result.contact_supported)
        self.assertEqual(result.object_path, "/World/Cube")
        self.assertEqual(result.mode, "pinch")
        self.assertEqual(result.contact_roles, ("thumb", "index"))
        self.assertEqual(result.curls["middle"], 0.0)
        # A hand that starts curved can still make a physical grasp. No button or
        # fresh fist edge may be substituted for the observed physical contacts.
        self.assertFalse(result.closing_intent)

    def test_proximity_single_brush_and_same_surface_cannot_qualify(self):
        first, second = pinch_contacts()
        for contacts in ([], [first], [first, replace(second, normal_world=(-1, 0, 0))]):
            with self.subTest(contacts=contacts):
                result = self.settle(
                    ContactGraspController(), contacts, requested_curls={role: 1.0 for role in GRASP.ACTUATOR_ROLES}
                )
                self.assertFalse(result.contact_supported)
                self.assertIsNone(result.object_path)

    def test_opposition_on_different_objects_and_coincident_points_do_not_qualify(self):
        first, second = pinch_contacts()
        for contacts in (
            [first, replace(second, object_path="/World/Cylinder")],
            [first, replace(second, point_world=first.point_world)],
        ):
            result = self.settle(ContactGraspController(), contacts)
            self.assertFalse(result.contact_supported)

    def test_wrap_requires_two_distinct_finger_roles_against_thumb_or_palm(self):
        first, second = pinch_contacts()
        palm = replace(first, role="palm")
        middle = replace(second, role="middle", point_world=(0.02, 0.015, 0))
        for anchor in (first, palm):
            with self.subTest(anchor=anchor.role):
                result = self.settle(ContactGraspController(), [anchor, second, middle])
                self.assertTrue(result.contact_supported)
                self.assertEqual(result.mode, "wrap")
                self.assertEqual(result.contact_roles, (anchor.role, "index", "middle"))
        duplicate_index = replace(second, point_world=(0.02, 0.015, 0))
        self.assertFalse(self.settle(ContactGraspController(), [palm, second, duplicate_index]).contact_supported)

    def test_contact_compliance_uses_each_joint_range_and_opening_wins(self):
        controller = ContactGraspController()
        requested = {role: 0.9 for role in GRASP.ACTUATOR_ROLES}
        measured = {role: 0.4 for role in GRASP.ACTUATOR_ROLES}
        # Same 0.04 rad lead, normalized by different authored actuator travels.
        lead = {"thumb": 0.04 / 0.45, "thumb_yaw": 0.04 / 1.05, "index": 0.04 / 1.6}
        result = self.tick(
            controller, pinch_contacts(), requested_curls=requested, measured_curls=measured, closing_lead=lead
        )
        for role in ("thumb", "thumb_yaw", "index"):
            self.assertAlmostEqual(result.curls[role], measured[role] + lead[role])
        self.assertEqual(result.curls["middle"], 0.9)
        requested["index"] = 0.1
        result = self.tick(
            controller, pinch_contacts(), requested_curls=requested, measured_curls=measured, closing_lead=lead
        )
        self.assertEqual(result.curls["index"], 0.1)

    def test_compliance_applies_to_single_contact_without_supported_grasp(self):
        controller = ContactGraspController()
        result = self.tick(controller, [pinch_contacts()[1]])
        self.assertFalse(result.contact_supported)
        self.assertAlmostEqual(result.curls["index"], 0.34)

    def test_contact_anchor_does_not_chase_rolling_fingers_or_thumb_opposition(self):
        controller = ContactGraspController()
        requested = {role: 0.9 for role in GRASP.ACTUATOR_ROLES}
        first = self.tick(controller, pinch_contacts(), requested_curls=requested)
        for measured_curl in (0.32, 0.45, 0.65):
            result = self.tick(
                controller,
                pinch_contacts(),
                requested_curls=requested,
                measured_curls={role: measured_curl for role in GRASP.ACTUATOR_ROLES},
            )
            for role in ("index", "thumb", "thumb_yaw"):
                self.assertAlmostEqual(result.curls[role], first.curls[role])
            self.assertEqual(result.curls["middle"], 0.9)
        self.assertAlmostEqual(result.curls["index"], 0.34)

    def test_brief_contact_gap_preserves_cap_without_replaying_support_evidence(self):
        controller = ContactGraspController()
        requested = {role: 0.9 for role in GRASP.ACTUATOR_ROLES}
        self.assertTrue(self.settle(controller, pinch_contacts(), requested_curls=requested).contact_supported)
        # Missing reports cannot ratchet the command forward as the joint moves.
        for _ in range(3):
            result = self.tick(
                controller,
                requested_curls=requested,
                measured_curls={role: 0.6 for role in GRASP.ACTUATOR_ROLES},
            )
            self.assertAlmostEqual(result.curls["index"], 0.34)
            self.assertFalse(result.contact_supported)
            self.assertIsNone(result.object_path)
        result = self.tick(
            controller,
            pinch_contacts(),
            requested_curls=requested,
            measured_curls={role: 0.6 for role in GRASP.ACTUATOR_ROLES},
        )
        self.assertAlmostEqual(result.curls["index"], 0.34)
        self.assertFalse(result.contact_supported)
        # Four hundredths of actual simulation time without contact releases it.
        for _ in range(4):
            result = self.tick(controller, requested_curls=requested)
        self.assertEqual(result.curls["index"], 0.9)
        result = self.tick(
            controller,
            pinch_contacts(),
            requested_curls=requested,
            measured_curls={role: 0.6 for role in GRASP.ACTUATOR_ROLES},
        )
        self.assertAlmostEqual(result.curls["index"], 0.64)

    def test_external_contact_pushing_finger_open_retains_current_measured_lead_bound(self):
        controller = ContactGraspController()
        requested = {role: 0.9 for role in GRASP.ACTUATOR_ROLES}
        self.tick(controller, pinch_contacts(), requested_curls=requested)
        result = self.tick(
            controller,
            pinch_contacts(),
            requested_curls=requested,
            measured_curls={role: 0.1 for role in GRASP.ACTUATOR_ROLES},
        )
        for role in ("index", "thumb", "thumb_yaw"):
            self.assertAlmostEqual(result.curls[role], 0.14)
        # A temporary retreat must not erase the original preload budget.
        result = self.tick(controller, pinch_contacts(), requested_curls=requested)
        self.assertAlmostEqual(result.curls["index"], 0.34)

    def test_contact_during_input_ramp_preserves_first_measured_preload_ceiling(self):
        controller = ContactGraspController()
        measured = {role: 0.4 for role in GRASP.ACTUATOR_ROLES}
        lead = {"thumb": 0.08 / 0.45, "thumb_yaw": 0.08 / 1.05, "index": 0.08 / 1.6}
        first_request = {role: 0.42 for role in GRASP.ACTUATOR_ROLES}
        result = self.tick(
            controller,
            pinch_contacts(),
            requested_curls=first_request,
            measured_curls=measured,
            closing_lead=lead,
        )
        self.assertEqual(result.curls["thumb"], 0.42)
        self.assertEqual(result.curls["thumb_yaw"], 0.42)
        # Input keeps closing after the first force sample. It may use the whole
        # original measured+lead allowance, but cannot chase beyond that ceiling.
        result = self.tick(
            controller,
            pinch_contacts(),
            requested_curls={role: 0.9 for role in GRASP.ACTUATOR_ROLES},
            measured_curls={role: 0.65 for role in GRASP.ACTUATOR_ROLES},
            closing_lead=lead,
        )
        for role in ("thumb", "thumb_yaw", "index"):
            self.assertAlmostEqual(result.curls[role], measured[role] + lead[role])

    def test_isolated_thumb_opposition_obeys_contact_ceiling_with_straight_digits(self):
        controller = ContactGraspController()
        contact = replace(pinch_contacts()[0], normal_force_n=2.0)
        measured = {role: 0.2 for role in GRASP.ACTUATOR_ROLES}
        lead = {role: 0.1 for role in GRASP.ACTUATOR_ROLES}
        requested = {"thumb_yaw": 0.9}
        result = self.tick(controller, [contact], requested_curls=requested, measured_curls=measured, closing_lead=lead)
        self.assertAlmostEqual(result.curls["thumb_yaw"], 0.3)
        self.assertTrue(all(result.curls[role] == 0.0 for role in GRASP.FINGER_ROLES))
        # A straight-finger pose must not repeatedly reset the yaw anchor.
        result = self.tick(
            controller,
            [contact],
            requested_curls=requested,
            measured_curls={role: 0.6 for role in GRASP.ACTUATOR_ROLES},
            closing_lead=lead,
        )
        self.assertAlmostEqual(result.curls["thumb_yaw"], 0.3)
        # Partial opposition opening wins immediately without erasing its ceiling.
        result = self.tick(
            controller, [contact], requested_curls={"thumb_yaw": 0.1}, measured_curls=measured, closing_lead=lead
        )
        self.assertEqual(result.curls["thumb_yaw"], 0.1)
        result = self.tick(controller, [contact], requested_curls=requested, measured_curls=measured, closing_lead=lead)
        self.assertAlmostEqual(result.curls["thumb_yaw"], 0.3)
        # Opening all six actuators clears the old contact ceiling, even if the
        # sensor still reports a contact during the first opening frame.
        result = self.tick(controller, [contact], requested_curls={})
        self.assertTrue(all(value == 0.0 for value in result.curls.values()))
        result = self.tick(
            controller,
            [contact],
            requested_curls=requested,
            measured_curls={role: 0.6 for role in GRASP.ACTUATOR_ROLES},
            closing_lead=lead,
        )
        self.assertAlmostEqual(result.curls["thumb_yaw"], 0.7)

    def test_contact_gap_expiry_uses_physics_time_not_callback_count(self):
        for dt in (0.005, 0.01, 0.02):
            controller = ContactGraspController()
            self.tick(controller, pinch_contacts())
            for _ in range(round(0.04 / dt) - 1):
                result = self.tick(controller, dt=dt)
                self.assertAlmostEqual(result.curls["index"], 0.34)
            self.assertEqual(self.tick(controller, dt=dt).curls["index"], 0.45)

    def test_partial_opening_wins_and_whole_open_or_input_loss_rearms(self):
        for reset_with_input_loss in (False, True):
            with self.subTest(input_loss=reset_with_input_loss):
                controller = ContactGraspController()
                requested = {role: 0.9 for role in GRASP.ACTUATOR_ROLES}
                self.tick(controller, pinch_contacts(), requested_curls=requested)
                result = self.tick(controller, pinch_contacts(), requested_curls={**requested, "index": 0.1})
                self.assertEqual(result.curls["index"], 0.1)
                result = self.tick(controller, pinch_contacts(), requested_curls=requested)
                self.assertAlmostEqual(result.curls["index"], 0.34)
                result = self.tick(
                    controller, pinch_contacts(), requested_curls={}, input_valid=not reset_with_input_loss
                )
                self.assertTrue(all(value == 0.0 for value in result.curls.values()))
                self.assertFalse(result.contact_supported)
                result = self.tick(
                    controller,
                    pinch_contacts(),
                    requested_curls=requested,
                    measured_curls={role: 0.6 for role in GRASP.ACTUATOR_ROLES},
                )
                self.assertAlmostEqual(result.curls["index"], 0.64)

    def test_new_contact_body_discards_old_anchor_immediately(self):
        controller = ContactGraspController()
        requested = {role: 0.9 for role in GRASP.ACTUATOR_ROLES}
        self.tick(controller, pinch_contacts(), requested_curls=requested)
        result = self.tick(
            controller,
            pinch_contacts("/World/Cylinder"),
            requested_curls=requested,
            measured_curls={role: 0.6 for role in GRASP.ACTUATOR_ROLES},
        )
        for role in ("index", "thumb", "thumb_yaw"):
            self.assertAlmostEqual(result.curls[role], 0.64)
        self.assertFalse(result.contact_supported)

    def test_table_contact_also_anchors_without_limiting_free_fingers(self):
        controller = ContactGraspController()
        contact = replace(pinch_contacts("/World/Table")[1], eligible_for_grasp=False)
        requested = {role: 0.9 for role in GRASP.ACTUATOR_ROLES}
        self.tick(controller, [contact], requested_curls=requested)
        result = self.tick(
            controller,
            [contact],
            requested_curls=requested,
            measured_curls={role: 0.6 for role in GRASP.ACTUATOR_ROLES},
        )
        self.assertAlmostEqual(result.curls["index"], 0.34)
        self.assertEqual(result.curls["middle"], 0.9)
        self.assertEqual(result.curls["thumb_yaw"], 0.9)
        self.assertFalse(result.contact_supported)

    def test_table_contacts_limit_closure_but_never_become_grasp_support(self):
        controller = ContactGraspController()
        contacts = [replace(contact, eligible_for_grasp=False) for contact in pinch_contacts("/World/Table")]
        result = self.settle(controller, contacts)
        self.assertAlmostEqual(result.curls["index"], 0.34)
        self.assertAlmostEqual(result.curls["thumb"], 0.34)
        self.assertFalse(result.contact_supported)
        self.assertIsNone(result.object_path)
        # Eligibility must survive contact validation and remain per sample.
        contacts[1] = replace(contacts[1], eligible_for_grasp=True)
        self.assertFalse(self.settle(controller, contacts).contact_supported)

    def test_predictive_contact_force_limits_closure_without_declaring_support(self):
        # The hand/object contact offsets can generate a real solver force before
        # surfaces are within the stricter gap used to recognize grasp support.
        contacts = [replace(contact, separation_m=0.003) for contact in pinch_contacts()]
        result = self.settle(ContactGraspController(), contacts)
        self.assertAlmostEqual(result.curls["thumb"], 0.34)
        self.assertAlmostEqual(result.curls["index"], 0.34)
        self.assertFalse(result.contact_supported)
        self.assertIsNone(result.object_path)

    def test_nonphysical_contact_samples_never_authorize_support(self):
        first, second = pinch_contacts()
        invalid = (
            replace(second, normal_force_n=0.0),
            replace(second, normal_force_n=0.019),
            replace(second, normal_force_n=math.nan),
            replace(second, normal_world=(0, 0, 0)),
            replace(second, normal_world=(math.inf, 0, 0)),
            replace(second, point_world=(math.nan, 0, 0)),
            replace(second, separation_m=0.003),
            replace(second, separation_m=math.nan),
            replace(second, role="wrist"),
            replace(second, object_path=""),
        )
        for contact in invalid:
            with self.subTest(contact=contact):
                self.assertFalse(self.settle(ContactGraspController(), [first, contact]).contact_supported)

    def test_losing_contact_object_change_reset_and_tracking_loss_clear_dwell(self):
        controller = ContactGraspController()
        self.assertTrue(self.settle(controller, pinch_contacts()).contact_supported)
        self.assertFalse(self.tick(controller).contact_supported)
        self.assertFalse(self.tick(controller, pinch_contacts()).contact_supported)
        self.assertTrue(self.settle(controller, pinch_contacts()).contact_supported)
        self.assertFalse(self.tick(controller, pinch_contacts("/World/Cylinder")).contact_supported)
        controller.reset()
        self.assertFalse(self.tick(controller, pinch_contacts()).contact_supported)
        result = self.tick(controller, pinch_contacts(), input_valid=False)
        self.assertFalse(result.contact_supported)
        self.assertIsNone(result.object_path)
        self.assertTrue(all(curl == 0.0 for curl in result.curls.values()))
        self.assertFalse(self.tick(controller, pinch_contacts()).contact_supported)

    def test_invalid_clock_and_single_stalled_frame_cannot_manufacture_dwell(self):
        for dt in (0.0, -0.01, math.nan, math.inf, 10.0):
            with self.subTest(dt=dt):
                self.assertFalse(self.tick(ContactGraspController(), pinch_contacts(), dt=dt).contact_supported)

    def test_support_dwell_depends_on_elapsed_physics_time(self):
        for dt in (0.005, 0.01, 0.02):
            with self.subTest(dt=dt):
                controller = ContactGraspController()
                steps = round(controller.support_time_s / dt)
                for _ in range(steps - 1):
                    self.assertFalse(self.tick(controller, pinch_contacts(), dt=dt).contact_supported)
                self.assertTrue(self.tick(controller, pinch_contacts(), dt=dt).contact_supported)

    def test_intent_tracks_individual_digits_and_expires_without_blocking_support(self):
        controller = ContactGraspController()
        self.tick(controller, requested_curls={})
        result = self.tick(controller, requested_curls={"index": 0.2})
        self.assertTrue(result.closing_intent)
        for _ in range(45):
            result = self.tick(controller, requested_curls={"index": 0.2})
        self.assertFalse(result.closing_intent)
        self.assertTrue(self.settle(controller, pinch_contacts(), requested_curls={"index": 0.2}).contact_supported)

    def test_slow_deliberate_closing_reports_intent_and_opening_clears_it(self):
        controller = ContactGraspController()
        for step in range(21):
            result = self.tick(controller, requested_curls={"index": 0.005 * step})
        self.assertTrue(result.closing_intent)
        self.assertFalse(self.tick(controller, requested_curls={}).closing_intent)

    def test_invalid_commands_and_missing_contact_measurements_remain_bounded(self):
        result = self.tick(
            ContactGraspController(),
            pinch_contacts(),
            requested_curls={"thumb": math.nan, "index": 8.0, "middle": -1.0, "ring": math.inf},
            measured_curls={},
            closing_lead={},
        )
        self.assertTrue(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result.curls.values()))
        self.assertEqual(result.curls["index"], 0.0)


if __name__ == "__main__":
    unittest.main()
