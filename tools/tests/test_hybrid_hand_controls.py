# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real hand-source selection and clutch handover in the gaze profile.

The XR descriptors and physical joint/link measurements are supplied by fixtures.
Production input classification, palm mapping, finger retargeting, source reset,
and joint slew limits run unchanged; these checks do not simulate a loaded grasp.
"""

from __future__ import annotations

import os
import types
import unittest
from unittest.mock import Mock, call, patch

import numpy as np
from humanoid_test_support import HUMANOID, Array, Device, pose
from pxr import Gf
from test_finger_controls import SkeletonDevice, skeleton_positions


class ControllerDevice(Device):
    """Provide a valid tracked controller with independent grip/trigger actions."""

    def __init__(self, position, *, grip: float = 0.8, trigger: float = 0.7) -> None:
        super().__init__(pose(position), flags=15, source="controller")
        self.grip = grip
        self.trigger = trigger

    def get_input_names(self):
        return ["squeeze", "trigger", "thumbstick"]

    def has_input(self, name):
        return name in self.get_input_names()

    def has_input_gesture(self, name, gesture):
        return name in ("squeeze", "trigger") and gesture == "value"

    def get_input_gesture_value(self, name, gesture):
        assert gesture == "value"
        return self.grip if name == "squeeze" else self.trigger


class TestHybridHandControls(unittest.TestCase):
    """Check both hand paths without replacing locomotion or the arm output guards."""

    def setUp(self) -> None:
        environment = {
            "G1_GESTURE_WALKING": "1",
            "G1_LOCOMOTION_INPUT": "gaze",
            "G1_GAZE_PREVIEW": "0",
            "G1_GAZE_MODE": "hold",
            "G1_GAZE_DWELL_S": "0",
            "G1_GAZE_CONTROLLER_OVERRIDE": "1",
            "G1_OPTICAL_HANDS_ONLY": "0",
            "G1_GRASP_MODE": "physical",
            "G1_WEBCAM_ENABLED": "0",
        }
        with patch.dict(os.environ, environment):
            self.ex = HUMANOID.HumanoidExample()
        ex = self.ex
        self.positions = np.array([[0.4, -0.5, 0.1, 0.2]])
        self.palms = {"left": (0.3, 0.2, 1.0), "right": (0.3, -0.2, 1.0)}
        self.devices = {side: ControllerDevice(position) for side, position in self.palms.items()}
        robot = Mock()
        robot.get_dof_positions.side_effect = lambda: Array(self.positions)
        ex.g1 = types.SimpleNamespace(
            robot=robot,
            supports_arm_teleoperation=True,
            has_finger_control=lambda: True,
            set_finger_curls=Mock(),
            get_finger_curls=lambda side: dict.fromkeys((*ex._finger_roles, "thumb_yaw"), 0.7),
            get_finger_joint_range=lambda side, role: (0.0, 1.6),
        )
        ex._xr_core = object()
        ex._contact_reader = types.SimpleNamespace(read=lambda dt: ({"left": [], "right": []}, "ok"))
        ex._contact_health = "ok"
        ex._g1_arm_dofs_configured = True
        ex._configure_g1_arm_dofs = Mock()
        ex._g1_arm_dof_indices_by_side = {
            "left": {"elbow": 0, "wrist_yaw": 1},
            "right": {"elbow": 2, "wrist_yaw": 3},
        }
        ex._g1_arm_joint_limits = {index: (-2.0, 2.0) for index in range(4)}
        ex._g1_arm_joint_defaults = dict.fromkeys(range(4), 0.0)
        ex._arm_smoothing = ex._finger_smoothing = ex._arm_rig_smoothing = 1.0
        ex._last_physics_dt = ex._teleop_input_dt = 0.01
        ex._get_g1_base_pose_for_arms = Mock(return_value=(Gf.Vec3d(0, 0, 0.8), 0.0))
        ex._get_xr_input_device = lambda handle: self.devices[handle.rsplit("/", 1)[-1]]
        ex._get_hand_link_prim = lambda side: types.SimpleNamespace(
            get_world_poses=lambda: (Array([self.palms[side]]), Array([[1.0, 0.0, 0.0, 0.0]]))
        )
        ex._palm_offset_world = lambda side, wrist: Gf.Vec3d(0)
        ex._compute_arm_target_orientation = Mock(return_value=None)
        ex._compute_arm_targets_from_body_position = Mock(side_effect=self._joint_goals)
        for name in (
            "_log_hand_tracking_status_once",
            "_set_grab_candidate",
            "_set_arm_rig_target_visible",
            "_release_grabbed_object",
            "_update_arm_rig_target",
            "_update_grabbed_object",
            "_pause_faulted_articulation",
            "_log_physics_step_error",
        ):
            setattr(ex, name, Mock())
        ex._controller_command = np.array([0.4, 0.0, 0.0])
        ex._base_command = np.array([0.3, 0.0, 0.0])

    def _joint_goals(self, side, target_body, orientation, **kwargs):
        """Supply only the IK boundary; retain production mapping and output limits."""
        return {index: float(target_body[0]) for index in self.ex._g1_arm_dof_indices_by_side[side].values()}

    def optical(self, side: str, curl: float = 0.2, position=None) -> SkeletonDevice:
        """Return a measured optical skeleton with its palm at the requested point."""
        x, y, z = self.palms[side] if position is None else position
        return SkeletonDevice(
            skeleton_positions(self.ex, dict.fromkeys(self.ex._finger_roles, curl)),
            transform=pose((x, y - 0.0275, z)),
            flags=15,
        )

    def tick(self) -> dict[int, float]:
        """Run the production finger-before-arm ordering with current descriptors."""
        self.ex._update_g1_fingers()
        self.ex._update_g1_arms_from_hand_tracking()
        self.assertIsNone(self.ex._articulation_health_fault)
        args, kwargs = self.ex.g1.robot.set_dof_position_targets.call_args
        return dict(zip(kwargs["dof_indices"], args[0]))

    def test_controller_grip_and_trigger_fallback_preserve_gaze_request(self) -> None:
        self.tick()
        self.assertTrue(self.ex._controller_arm_control_enabled)
        self.assertEqual(self.ex._arm_input_sources, dict.fromkeys(self.palms, "controller"))
        self.assertEqual(self.ex._finger_curl_source, dict.fromkeys(self.palms, "controller"))
        for side in self.palms:
            for role in self.ex._finger_roles:
                self.assertAlmostEqual(self.ex._latest_finger_curls[side][role], 0.7)
        np.testing.assert_array_equal(self.ex._controller_command, (0.4, 0.0, 0.0))
        np.testing.assert_array_equal(self.ex._base_command, (0.3, 0.0, 0.0))

    def test_each_hand_can_use_a_different_source(self) -> None:
        self.devices["left"] = self.optical("left", curl=0.25)
        self.tick()
        self.assertEqual(self.ex._arm_input_sources, {"left": "hand_tracking", "right": "controller"})
        self.assertEqual(self.ex._finger_curl_source, {"left": "hand_tracking", "right": "controller"})
        self.assertAlmostEqual(self.ex._latest_finger_curls["left"]["index"], 0.25)
        self.assertAlmostEqual(self.ex._latest_finger_curls["right"]["index"], 0.7)

    def test_optical_input_is_preferred_without_reading_controller_actions(self) -> None:
        self.devices = {side: self.optical(side) for side in self.palms}
        self.ex._get_xr_gesture_value = Mock(side_effect=AssertionError("Optical input read a retained button"))
        self.ex._get_controller_arm_pose = Mock(wraps=self.ex._get_controller_arm_pose)
        self.ex._get_controller_finger_curls = Mock(wraps=self.ex._get_controller_finger_curls)
        self.tick()
        self.assertEqual(self.ex._arm_input_sources, dict.fromkeys(self.palms, "hand_tracking"))
        self.assertEqual(self.ex._finger_curl_source, dict.fromkeys(self.palms, "hand_tracking"))
        self.ex._get_controller_arm_pose.assert_not_called()
        self.ex._get_controller_finger_curls.assert_not_called()

    def test_releasing_grip_holds_arm_while_trigger_stays_closed(self) -> None:
        self.tick()
        self.ex._release_grabbed_object.reset_mock()
        self.devices["right"].grip = 0.0
        targets = self.tick()
        np.testing.assert_allclose([targets[2], targets[3]], self.positions[0, 2:])
        self.assertIn("right", self.ex._inactive_arm_hold_targets)
        self.assertNotIn("right", self.ex._arm_input_sources)
        self.assertEqual(self.ex._finger_curl_source["right"], "controller")
        self.assertAlmostEqual(self.ex._latest_finger_curls["right"]["index"], 0.7)
        self.assertNotIn(call("right"), self.ex._release_grabbed_object.call_args_list)

    def test_source_changes_recalibrate_controller_to_actual_palm_and_bound_joint_step(self) -> None:
        ex = self.ex
        self.devices["right"] = self.optical("right")
        self.tick()
        ex._controller_arm_neutral_positions["right"] = Gf.Vec3d(9)
        ex._controller_arm_neutral_targets["right"] = Gf.Vec3d(8)
        self.devices["right"] = ControllerDevice((2.0, -1.5, 2.0), trigger=0.8)
        targets = self.tick()
        np.testing.assert_allclose(ex._controller_arm_neutral_targets["right"], (0.3, -0.2, 0.2), atol=1e-7)
        self.assertEqual(ex._arm_input_sources["right"], "controller")
        self.assertEqual(ex._finger_curl_source["right"], "controller")
        step_limit = ex._arm_max_joint_speed * ex._last_physics_dt
        self.assertLessEqual(abs(targets[2] - self.positions[0, 2]), step_limit + 1e-7)
        # Optical targets are absolute, so returning to a distant optical palm
        # may request a new goal; the measured-joint slew bound must still apply.
        self.devices["right"] = self.optical("right", position=(0.7, -0.2, 1.0))
        targets = self.tick()
        self.assertEqual(ex._arm_input_sources["right"], "hand_tracking")
        self.assertNotIn("right", ex._controller_arm_neutral_positions)
        self.assertNotIn("right", ex._controller_arm_neutral_targets)
        self.assertLessEqual(abs(targets[2] - self.positions[0, 2]), step_limit + 1e-7)

    def test_tracking_loss_opens_fingers_and_reacquisition_reanchors_clutch(self) -> None:
        self.tick()
        self.devices["right"].flags = 0
        targets = self.tick()
        self.assertEqual(self.ex._finger_curl_source["right"], "none")
        self.assertAlmostEqual(self.ex._latest_finger_curls["right"]["index"], 0.0)
        np.testing.assert_allclose([targets[2], targets[3]], self.positions[0, 2:])
        self.assertNotIn("right", self.ex._controller_arm_neutral_positions)
        self.devices["right"].flags = 15
        self.devices["right"].matrix = pose((1.4, -0.9, 1.7))
        self.palms["right"] = (0.35, -0.24, 0.95)
        self.tick()
        np.testing.assert_allclose(self.ex._controller_arm_neutral_targets["right"], (0.35, -0.24, 0.15), atol=1e-7)

    def test_partial_optical_device_never_consumes_retained_controller_actions(self) -> None:
        device = self.optical("right")
        device.flags = 0
        device.get_input_names = lambda: ["thumbstick", "trigger", "squeeze"]
        self.devices["right"] = device
        self.devices["left"] = self.optical("left")
        self.ex._get_xr_gesture_value = Mock(side_effect=AssertionError("Optical loss consumed a stale button"))
        targets = self.tick()
        self.assertEqual(self.ex._get_hand_input_kind(device), "hand_tracking")
        self.assertEqual(self.ex._finger_curl_source["right"], "none")
        self.assertNotIn("right", self.ex._arm_input_sources)
        np.testing.assert_allclose([targets[2], targets[3]], self.positions[0, 2:])

    def test_explicit_optical_only_setting_rejects_controller_manipulation(self) -> None:
        with patch.dict(os.environ, {"G1_OPTICAL_HANDS_ONLY": "1", "G1_LOCOMOTION_INPUT": "gaze"}):
            ex = HUMANOID.HumanoidExample()
        self.assertFalse(ex._controller_arm_control_enabled)
        ex._get_xr_gesture_value = Mock(side_effect=AssertionError("Optical-only mode read a controller action"))
        self.assertIsNone(ex._get_controller_arm_pose("right", self.devices["right"]))
        self.assertIsNone(ex._get_controller_finger_curls(self.devices["right"]))


if __name__ == "__main__":
    unittest.main()
