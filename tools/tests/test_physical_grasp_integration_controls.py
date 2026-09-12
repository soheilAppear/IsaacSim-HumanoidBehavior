# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical grasp integration uses contacts and measured fingers without attaching objects.

XR input and the contact/actuator boundary are controlled here; production update
methods and grasp state logic are unchanged. Live frictional lifting is a separate
PhysX acceptance test, not something these supplied contact samples can prove.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import Mock, patch

from humanoid_test_support import GRASP, HUMANOID, USD
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics


class TestPhysicalGraspIntegrationControls(unittest.TestCase):
    def setUp(self):
        self.ex = HUMANOID.HumanoidExample()
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)
        self.ex._xr_core = object()
        self.ex._last_physics_dt = 0.01
        self.ex._finger_smoothing = 1.0
        self.roles = (*self.ex._finger_roles, "thumb_yaw")
        self.requested = {"thumb": 0.35, "index": 0.45, "thumb_yaw": 0.55}
        self.contacts = [
            GRASP.ContactSample("/World/Cube", "thumb", (-0.02, 0, 0), (-1, 0, 0), 0.8),
            GRASP.ContactSample("/World/Cube", "index", (0.02, 0, 0), (1, 0, 0), 0.9),
        ]
        self.ex._get_xr_input_device = Mock(return_value=object())
        self.ex._get_hand_tracking_finger_curls = Mock(side_effect=lambda device: self.requested.copy())
        self.ex._get_controller_finger_curls = Mock(return_value=None)
        self.ex._set_grab_candidate = Mock()
        self.ex._contact_reader = types.SimpleNamespace(
            read=Mock(side_effect=lambda dt: ({"left": [], "right": list(self.contacts)}, "ok"))
        )
        self.ex.g1 = types.SimpleNamespace(
            has_finger_control=lambda: True,
            set_finger_curls=Mock(),
            get_finger_curls=Mock(return_value={role: 0.3 for role in self.roles}),
            get_finger_joint_range=Mock(return_value=(0.0, 1.6)),
        )

    def tick(self, count=1):
        for _ in range(count):
            self.ex._update_g1_fingers()
            closed = self.ex._is_hand_closed("right", None)
            self.ex._update_grabbed_object("right", closed)
        return self.ex._physical_grasp_results.get("right")

    def test_default_physical_mode_never_creates_attachment_even_with_supported_contact(self):
        ex = self.ex
        self.assertEqual(ex._grasp_mode, "physical")
        self.assertFalse(ex._create_grasp_joint("right", "/World/Cube"))
        ex._create_grasp_joint = Mock(side_effect=AssertionError("Physical mode attempted an attachment"))
        ex._find_nearest_grabbable_object = Mock(side_effect=AssertionError("Proximity substituted for contact"))
        result = self.tick(10)
        self.assertTrue(result.contact_supported)
        ex._create_grasp_joint.assert_not_called()
        ex._find_nearest_grabbable_object.assert_not_called()
        self.assertFalse(ex._grabbed_objects_by_side)
        self.assertFalse(ex._grasp_joints_by_side)
        self.assertFalse(any(prim.IsA(UsdPhysics.FixedJoint) for prim in self.stage.Traverse()))

    def test_partial_pinch_qualifies_without_five_finger_fist_threshold(self):
        result = self.tick(10)
        self.assertLess(self.ex._get_hand_closure("right"), self.ex._finger_grab_threshold)
        self.assertFalse(self.ex._is_hand_closed("right", None))
        self.assertTrue(result.contact_supported)
        self.assertEqual(result.mode, "pinch")
        self.assertEqual(result.curls["middle"], 0.0)
        self.assertAlmostEqual(result.curls["index"], 0.3 + self.ex._finger_contact_lead_rad / 1.6)

    def test_drop_opens_fingers_and_prevents_reclosing_until_valid_open_input(self):
        self.assertTrue(self.tick(10).contact_supported)
        self.ex._drop_everything()
        for _, curls in (call.args for call in self.ex.g1.set_finger_curls.call_args_list[-2:]):
            self.assertTrue(all(value == 0.0 for value in curls.values()))
        result = self.tick(10)
        self.assertTrue(self.ex._grab_requires_release["right"])
        self.assertFalse(result.contact_supported)
        self.assertTrue(all(value == 0.0 for value in result.curls.values()))
        self.requested = {role: 0.0 for role in self.roles}
        self.contacts = []
        self.tick()
        self.assertFalse(self.ex._grab_requires_release["right"])
        self.requested = {"index": 0.4}
        self.tick()
        self.assertGreater(self.ex._applied_finger_curls["right"]["index"], 0.0)

    def test_relaxed_optical_open_rearms_before_smoothing_without_replaying_closed_targets(self):
        self.contacts = []
        self.requested = {role: 0.8 for role in self.roles}
        self.tick()
        self.ex._drop_everything()
        self.ex._finger_smoothing = 0.05
        self.requested = {role: 0.25 for role in self.ex._finger_roles}
        self.requested.update(thumb=0.5, thumb_yaw=0.8)

        # A naturally open tracked hand need not have a straight thumb. Release
        # uses this raw sample even while the filter still contains a closed fist.
        result = self.tick()
        self.assertGreater(self.ex._latest_finger_curls["right"]["index"], self.ex._finger_release_threshold)
        for side in ("left", "right"):
            self.assertFalse(self.ex._grab_requires_release[side])
            self.assertNotIn(side, self.ex._smoothed_finger_curls)
            self.assertTrue(all(value == 0.0 for value in self.ex._applied_finger_curls[side].values()))
        self.assertFalse(result.contact_supported)

        result = self.tick()
        for role, value in self.requested.items():
            self.assertAlmostEqual(result.curls[role], value)
        self.assertEqual(self.ex._finger_curl_source["right"], "hand_tracking")

    def test_missing_optical_digits_cannot_rearm_when_smoothing_fills_them_with_zero(self):
        self.contacts = []
        for missing in ("index", "middle", "ring", "little"):
            with self.subTest(missing=missing):
                self.requested = {role: 0.25 for role in self.roles if role != missing}
                self.requested.update(thumb=0.5, thumb_yaw=0.8)
                self.ex._drop_everything()
                result = self.tick(3)
                self.assertEqual(self.ex._latest_finger_curls["right"][missing], 0.0)
                self.assertTrue(self.ex._grab_requires_release["right"])
                self.assertTrue(all(value == 0.0 for value in result.curls.values()))
                self.requested[missing] = 0.25
                self.tick()
                self.assertFalse(self.ex._grab_requires_release["right"])

    def test_invalid_raw_optical_digits_cannot_rearm_after_smoothing_sanitizes_them(self):
        self.contacts = []
        for role in ("index", "middle", "ring", "little"):
            for invalid in (float("nan"), float("inf"), -float("inf"), -0.01):
                with self.subTest(role=role, invalid=invalid):
                    self.requested = {name: 0.0 for name in self.roles}
                    self.requested[role] = invalid
                    self.ex._drop_everything()
                    result = self.tick()
                    self.assertEqual(self.ex._latest_finger_curls["right"][role], 0.0)
                    self.assertTrue(self.ex._grab_requires_release["right"])
                    self.assertTrue(all(value == 0.0 for value in result.curls.values()))

    def test_optical_release_requires_all_four_fingers_below_release_threshold(self):
        self.contacts = []
        for role in ("index", "middle", "ring", "little"):
            with self.subTest(role=role):
                self.requested = {name: 0.0 for name in self.roles}
                self.requested[role] = self.ex._finger_release_threshold
                self.ex._drop_everything()
                self.tick()
                self.assertTrue(self.ex._grab_requires_release["right"])
                self.requested[role] -= 0.001
                self.tick()
                self.assertFalse(self.ex._grab_requires_release["right"])

    def test_held_controller_drop_stays_open_until_all_trigger_fingers_are_released(self):
        self.contacts = []
        self.requested = {role: 0.8 for role in self.ex._finger_roles}
        self.ex._get_hand_tracking_finger_curls.side_effect = lambda device: None
        self.ex._get_controller_finger_curls.side_effect = lambda device: self.requested.copy()
        self.tick()
        self.ex._drop_everything()
        result = self.tick(10)
        self.assertEqual(self.ex._finger_curl_source["right"], "controller")
        self.assertTrue(self.ex._grab_requires_release["right"])
        self.assertTrue(all(value == 0.0 for value in result.curls.values()))

        for role in self.ex._finger_roles:
            with self.subTest(unreleased=role):
                self.requested = {name: 0.0 for name in self.ex._finger_roles}
                self.requested[role] = 0.15
                self.ex._drop_everything()
                self.tick()
                self.assertTrue(self.ex._grab_requires_release["right"])
                self.requested[role] = 0.149
                result = self.tick()
                self.assertFalse(self.ex._grab_requires_release["right"])
                self.assertTrue(all(value == 0.0 for value in result.curls.values()))

        self.requested = {role: 0.8 for role in self.ex._finger_roles}
        result = self.tick()
        self.assertGreater(result.curls["index"], 0.0)

    def test_incomplete_or_invalid_controller_release_cannot_clear_drop_latch(self):
        self.contacts = []
        self.ex._get_hand_tracking_finger_curls.side_effect = lambda device: None
        self.ex._get_controller_finger_curls.side_effect = lambda device: self.requested.copy()
        for role in self.ex._finger_roles:
            for invalid in (None, float("nan"), float("inf"), -float("inf"), -0.01):
                with self.subTest(role=role, invalid=invalid):
                    self.requested = {name: 0.0 for name in self.ex._finger_roles}
                    if invalid is None:
                        self.requested.pop(role)
                    else:
                        self.requested[role] = invalid
                    self.ex._drop_everything()
                    result = self.tick()
                    self.assertTrue(self.ex._grab_requires_release["right"])
                    self.assertTrue(all(value == 0.0 for value in result.curls.values()))

    def test_valid_optical_release_cannot_rearm_without_healthy_contacts_and_full_measurements(self):
        self.contacts = []
        self.requested = {role: 0.0 for role in self.roles}
        self.ex._drop_everything()
        self.ex._contact_reader.read.side_effect = lambda dt: ({"left": [], "right": []}, "invalidated")
        result = self.tick()
        self.assertTrue(self.ex._grab_requires_release["right"])
        self.assertTrue(all(value == 0.0 for value in result.curls.values()))

        self.ex._contact_reader.read.side_effect = lambda dt: ({"left": [], "right": []}, "ok")
        for missing in self.roles:
            with self.subTest(missing_measurement=missing):
                self.ex.g1.get_finger_curls.return_value = {role: 0.3 for role in self.roles if role != missing}
                result = self.tick()
                self.assertTrue(self.ex._grab_requires_release["right"])
                self.assertTrue(all(value == 0.0 for value in result.curls.values()))
        self.ex.g1.get_finger_curls.return_value = {role: 0.3 for role in self.roles}
        result = self.tick()
        self.assertFalse(self.ex._grab_requires_release["right"])
        self.assertTrue(all(value == 0.0 for value in result.curls.values()))

    def test_tracking_loss_cannot_rearm_the_drop_latch(self):
        self.tick()
        self.ex._drop_everything()
        self.ex._get_hand_tracking_finger_curls.side_effect = lambda device: None
        result = self.tick()
        self.assertTrue(self.ex._grab_requires_release["right"])
        self.assertTrue(all(value == 0.0 for value in result.curls.values()))
        self.ex._get_hand_tracking_finger_curls.side_effect = lambda device: self.requested.copy()
        result = self.tick()
        self.assertFalse(result.contact_supported)
        self.assertTrue(all(value == 0.0 for value in result.curls.values()))

    def test_unhealthy_contact_reader_cannot_reuse_stale_support(self):
        self.assertTrue(self.tick(10).contact_supported)
        self.ex._contact_reader.read.side_effect = lambda dt: ({"left": [], "right": self.contacts}, "invalidated")
        result = self.tick()
        self.assertFalse(result.contact_supported)
        self.assertIsNone(result.object_path)
        self.assertTrue(all(value == 0.0 for value in result.curls.values()))
        self.ex._contact_reader.read.side_effect = lambda dt: ({"left": [], "right": self.contacts}, "ok")
        self.assertFalse(self.tick().contact_supported)

    def test_finger_tracking_loss_releases_contact_state_and_relaxes_hand(self):
        self.assertTrue(self.tick(10).contact_supported)
        self.ex._get_hand_tracking_finger_curls.side_effect = lambda device: None
        result = self.tick()
        self.assertEqual(self.ex._finger_curl_source["right"], "none")
        self.assertFalse(result.contact_supported)
        self.assertIsNone(result.object_path)
        self.assertTrue(all(value == 0.0 for value in result.curls.values()))
        self.ex._get_hand_tracking_finger_curls.side_effect = lambda device: self.requested.copy()
        self.assertFalse(self.tick().contact_supported)

    def test_arm_clutch_deactivation_does_not_reset_independent_finger_contact_dwell(self):
        self.ex._set_arm_rig_target_visible = Mock()
        for _ in range(10):
            self.tick()
            self.ex._deactivate_hand("right")
        self.assertTrue(self.ex._physical_grasp_results["right"].contact_supported)
        self.ex._finger_curl_source["right"] = "none"
        self.ex._deactivate_hand("right")
        self.assertNotIn("right", self.ex._physical_grasp_results)

    def test_missing_live_finger_measurement_cannot_issue_a_stale_holding_command(self):
        self.assertTrue(self.tick(10).contact_supported)
        self.ex.g1.get_finger_curls.return_value = {}
        result = self.tick()
        self.assertFalse(result.contact_supported)
        self.assertTrue(all(value == 0.0 for value in result.curls.values()))

    def test_partial_live_measurements_cannot_command_unobserved_free_fingers(self):
        self.assertTrue(self.tick(10).contact_supported)
        self.contacts = []
        self.requested = {role: 0.8 for role in self.roles}
        for missing in self.roles:
            with self.subTest(missing=missing):
                self.ex.g1.get_finger_curls.return_value = {role: 0.3 for role in self.roles if role != missing}
                result = self.tick()
                self.assertFalse(result.contact_supported)
                self.assertTrue(all(value == 0.0 for value in result.curls.values()))

    def test_physical_fixture_does_not_replace_assisted_layout_configuration_on_reload(self):
        ex = self.ex
        ex._get_work_surface_tops = Mock(return_value=[(0.5, -0.28, 0.0, 0.812, 1.0, 0.6)])
        ex._package_use_props = False
        with patch.object(HUMANOID, "get_assets_root_path", return_value=None, create=True):
            for configured_count in (10, 6):
                ex._sample_box_count = configured_count
                for mode in ("physical", "assisted", "physical"):
                    with self.subTest(configured_count=configured_count, mode=mode):
                        # Scene reload reuses the example and its settings, while
                        # replacing the USD stage and all spawned object roots.
                        self.stage = Usd.Stage.CreateInMemory()
                        ex._grasp_mode = mode
                        ex._create_sample_boxes()
                        self.assertEqual(ex._sample_box_count, configured_count)
                        roots = ex._spawned_sample_box_paths
                        self.assertEqual(len(roots), 6 if mode == "physical" else configured_count)
                        self.assertEqual(len(roots), len(set(roots)))
                        on_surface = sum(
                            bool(self.stage.GetPrimAtPath(path).GetAttribute("g1:packageOnSurface").Get())
                            for path in roots
                        )
                        self.assertEqual(on_surface, 6 if mode == "physical" else configured_count - 2)

    def test_larger_practice_shapes_keep_correct_bounds_clearance_light_mass_and_colors(self):
        ex = self.ex
        table_top = 0.812
        ex._get_work_surface_tops = Mock(return_value=[(0.5, -0.28, 0.0, table_top, 1.0, 0.6)])
        ex._create_sample_boxes()
        self.assertEqual(len(ex._spawned_sample_box_paths), 6)
        objects = (
            (UsdGeom.Cube, (0.06, 0.06, 0.06), 0.06, (0.15, 0.55, 0.9)),
            (UsdGeom.Cylinder, (0.06, 0.06, 0.10), 0.08, (0.95, 0.55, 0.1)),
            (UsdGeom.Sphere, (0.07, 0.07, 0.07), 0.05, (0.2, 0.75, 0.35)),
            (UsdGeom.Cylinder, (0.04, 0.04, 0.14), 0.06, (0.65, 0.3, 0.85)),
            (UsdGeom.Cube, (0.09, 0.06, 0.03), 0.07, (0.95, 0.8, 0.15)),
            (UsdGeom.Cone, (0.07, 0.07, 0.10), 0.05, (0.9, 0.25, 0.2)),
        )
        bounds = []
        for index, (schema, dimensions, mass, color) in enumerate(objects):
            with self.subTest(index=index):
                prim = self.stage.GetPrimAtPath(ex._spawned_sample_box_paths[index])
                self.assertTrue(prim.IsA(schema))
                if schema == UsdGeom.Cube:
                    self.assertAlmostEqual(UsdGeom.Cube(prim).GetSizeAttr().Get(), dimensions[0])
                else:
                    shape = schema(prim)
                    self.assertAlmostEqual(shape.GetRadiusAttr().Get() * 2, dimensions[0])
                    if schema != UsdGeom.Sphere:
                        self.assertAlmostEqual(shape.GetHeightAttr().Get(), dimensions[2])
                        self.assertEqual(shape.GetAxisAttr().Get(), "Z")
                lower, upper = UsdGeom.Boundable(prim).GetExtentAttr().Get()
                metadata_dimensions = prim.GetAttribute("g1:packageDimensions").Get()
                for axis, expected in enumerate(dimensions):
                    local_dimension = dimensions[0] if schema == UsdGeom.Cube else expected
                    self.assertAlmostEqual(upper[axis] - lower[axis], local_dimension)
                    self.assertAlmostEqual(metadata_dimensions[axis], expected)
                world_bounds = (
                    UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                    .ComputeWorldBound(prim)
                    .ComputeAlignedRange()
                )
                bounds.append(world_bounds)
                for actual, expected in zip(world_bounds.GetSize(), dimensions):
                    self.assertAlmostEqual(actual, expected)
                self.assertAlmostEqual(world_bounds.GetMin()[2] - table_top, 0.003)
                # The measured extended palm reaches about X=0.49 at this base
                # position; keep practice centers closer than that outer limit.
                self.assertLessEqual(world_bounds.GetMidpoint()[0], 0.47)
                # The live front table spans X=0.289..0.711, Y=-0.739..0.179.
                # Retain a margin from its edges and room for fingers between props.
                for axis, (minimum, maximum) in enumerate(((0.289, 0.711), (-0.739, 0.179))):
                    self.assertGreater(world_bounds.GetMin()[axis], minimum + 0.02)
                    self.assertLess(world_bounds.GetMax()[axis], maximum - 0.02)
                self.assertAlmostEqual(prim.GetAttribute("g1:packageSize").Get(), max(dimensions))
                self.assertAlmostEqual(prim.GetAttribute("g1:packageMass").Get(), mass)
                self.assertAlmostEqual(UsdPhysics.MassAPI(prim).GetMassAttr().Get(), mass)
                self.assertTrue(prim.GetAttribute("g1:packageOnSurface").Get())
                self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
                self.assertTrue(prim.HasAPI(UsdPhysics.RigidBodyAPI))
                self.assertFalse(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())
                self.assertAlmostEqual(PhysxSchema.PhysxCollisionAPI(prim).GetContactOffsetAttr().Get(), 0.002)
                self.assertEqual(PhysxSchema.PhysxCollisionAPI(prim).GetRestOffsetAttr().Get(), 0.0)
                self.assertEqual(PhysxSchema.PhysxRigidBodyAPI(prim).GetSleepThresholdAttr().Get(), 0.0)
                for actual, expected in zip(UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()[0], color):
                    self.assertAlmostEqual(actual, expected)
        for index, first in enumerate(bounds):
            for second in bounds[index + 1 :]:
                gap = max(
                    max(first.GetMin()[axis] - second.GetMax()[axis], second.GetMin()[axis] - first.GetMax()[axis])
                    for axis in (0, 1)
                )
                self.assertGreaterEqual(gap + 1e-7, 0.045)


class TestPhysicalHandCouplingControls(unittest.TestCase):
    def setUp(self):
        self.ex = HUMANOID.HumanoidExample()
        self.stage = Usd.Stage.CreateInMemory()
        self.hand_paths = [
            "/World/G1/left_hand/joints/L_thumb_intermediate_joint",
            "/World/G1/right_hand/joints/R_index_intermediate_joint",
        ]
        self.other_paths = [
            "/World/G1/joints/body_mimic",
            "/World/G1/right_hand_extra/joints/mimic",
            "/World/G1_Extra/left_hand/joints/mimic",
        ]
        for path in (*self.hand_paths, *self.other_paths):
            joint = UsdPhysics.RevoluteJoint.Define(self.stage, path)
            joint.CreateLowerLimitAttr(-30.0)
            joint.CreateUpperLimitAttr(120.0)
            joint.CreateAxisAttr("Z")
            reference = UsdPhysics.RevoluteJoint.Define(self.stage, path + "_reference")
            drive = UsdPhysics.DriveAPI.Apply(reference.GetPrim(), "angular")
            drive.CreateStiffnessAttr(20.0)
            drive.CreateDampingAttr(0.6)
            drive.CreateMaxForceAttr(10.0)
            coupling = PhysxSchema.PhysxMimicJointAPI.Apply(joint.GetPrim(), "rotZ")
            coupling.CreateNaturalFrequencyAttr(25.0)
            coupling.CreateDampingRatioAttr(0.005)
            coupling.CreateGearingAttr(-1.6)
            coupling.CreateOffsetAttr(0.1)
            coupling.CreateReferenceJointAxisAttr("rotX")
            coupling.CreateReferenceJointRel().SetTargets([reference.GetPath()])
        # Some assets author a disabled drive on a passive joint. Preserve that
        # exact state as well as the absence of any drive on the other hand.
        passive_drive = UsdPhysics.DriveAPI.Apply(self.stage.GetPrimAtPath(self.hand_paths[1]), "angular")
        passive_drive.CreateStiffnessAttr(0.0)
        passive_drive.CreateDampingAttr(0.0)
        passive_drive.CreateMaxForceAttr(0.0)
        self.revolute_hand_paths = [path for passive in self.hand_paths for path in (passive, passive + "_reference")]
        UsdPhysics.FixedJoint.Define(self.stage, "/World/G1/left_hand/joints/wrist_mount")

    def snapshot(self, prim):
        return {
            "schemas": prim.GetAppliedSchemas(),
            "attributes": {attribute.GetName(): attribute.Get() for attribute in prim.GetAuthoredAttributes()},
            "relationships": {
                relationship.GetName(): relationship.GetTargets() for relationship in prim.GetRelationships()
            },
        }

    def test_only_hand_compliance_and_inertia_change_without_passive_drives_or_velocity_caps(self):
        before = {str(prim.GetPath()): self.snapshot(prim) for prim in self.stage.Traverse()}
        self.ex._prepare_physical_hand_couplings(self.stage)
        self.assertEqual(set(self.ex._physical_hand_mimic_joint_paths), set(self.hand_paths))
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            expected = before[path]
            if path in self.hand_paths:
                expected["attributes"]["physxMimicJoint:rotZ:naturalFrequency"] = 500.0
                expected["attributes"]["physxMimicJoint:rotZ:dampingRatio"] = 1.0
            if path in self.revolute_hand_paths:
                expected["schemas"].append("PhysxJointAPI")
                expected["attributes"]["physxJoint:armature"] = PhysxSchema.PhysxJointAPI(prim).GetArmatureAttr().Get()
                self.assertAlmostEqual(expected["attributes"]["physxJoint:armature"], self.ex._finger_joint_armature)
            self.assertEqual(self.snapshot(prim), expected, path)
        self.assertEqual(set(self.ex._physical_hand_armature_joint_values), set(self.revolute_hand_paths))
        for path, armature in self.ex._physical_hand_armature_joint_values.items():
            self.assertEqual(
                armature, PhysxSchema.PhysxJointAPI(self.stage.GetPrimAtPath(path)).GetArmatureAttr().Get()
            )
        self.assertFalse(self.stage.GetPrimAtPath(self.hand_paths[0]).HasAPI(UsdPhysics.DriveAPI, "angular"))
        passive = UsdPhysics.DriveAPI(self.stage.GetPrimAtPath(self.hand_paths[1]), "angular")
        self.assertEqual(passive.GetStiffnessAttr().Get(), 0.0)
        self.assertEqual(passive.GetMaxForceAttr().Get(), 0.0)
        once = self.stage.GetRootLayer().ExportToString()
        self.ex._prepare_physical_hand_couplings(self.stage)
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), once)

    def test_tunable_armature_floor_preserves_stronger_authored_driven_and_passive_inertia(self):
        self.ex._finger_joint_armature = 1e-3
        for path, authored in zip(self.revolute_hand_paths, (0.0, 0.002, 0.003, 0.0)):
            PhysxSchema.PhysxJointAPI.Apply(self.stage.GetPrimAtPath(path)).CreateArmatureAttr(authored)
        self.ex._prepare_physical_hand_couplings(self.stage)
        for path, expected in zip(self.revolute_hand_paths, (1e-3, 0.002, 0.003, 1e-3)):
            self.assertAlmostEqual(self.ex._physical_hand_armature_joint_values[path], expected)

    def test_existing_axis_override_gets_floor_without_changing_other_axis_properties(self):
        for path, authored in zip(self.hand_paths, (0.0, 0.003)):
            prim = self.stage.GetPrimAtPath(path)
            self.assertTrue(prim.ApplyAPI("PhysxJointAxisAPI", "rotZ"))
            prim.GetAttribute("physxJointAxis:rotZ:armature").Set(authored)
            prim.GetAttribute("physxJointAxis:rotZ:maxJointVelocity").Set(432.0)
        self.ex._prepare_physical_hand_couplings(self.stage)
        for path, expected in zip(self.hand_paths, (self.ex._finger_joint_armature, 0.003)):
            prim = self.stage.GetPrimAtPath(path)
            self.assertAlmostEqual(prim.GetAttribute("physxJointAxis:rotZ:armature").Get(), expected)
            self.assertAlmostEqual(self.ex._physical_hand_armature_joint_values[path], expected)
            self.assertEqual(prim.GetAttribute("physxJointAxis:rotZ:maxJointVelocity").Get(), 432.0)

    def test_invalid_armature_floor_fails_before_any_scene_changes(self):
        before = self.stage.GetRootLayer().ExportToString()
        for value in (-0.001, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.ex._finger_joint_armature = value
                with self.assertRaisesRegex(ValueError, "finite, nonnegative"):
                    self.ex._prepare_physical_hand_couplings(self.stage)
                self.assertEqual(self.stage.GetRootLayer().ExportToString(), before)

    def test_assisted_and_other_hand_modes_retain_all_asset_properties(self):
        for mode, variant in (("assisted", "Inspire"), ("physical", "ThreeFinger")):
            with self.subTest(mode=mode, variant=variant):
                self.ex._grasp_mode = mode
                self.ex._g1_hand_variant = variant
                before = self.stage.GetRootLayer().ExportToString()
                self.ex._prepare_physical_hand_couplings(self.stage)
                self.assertEqual(self.stage.GetRootLayer().ExportToString(), before)
                self.assertEqual(self.ex._physical_hand_mimic_joint_paths, [])
                self.assertEqual(self.ex._physical_hand_armature_joint_values, {})


if __name__ == "__main__":
    unittest.main()
