# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify joystick walking coexists with controller and optical manipulation."""

from __future__ import annotations

import os
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from humanoid_test_support import HUMANOID, pose
from pxr import Gf


class StickDevice:
    def __init__(self) -> None:
        self.x = self.y = 0.0
        self.flags = 15
        self.actions_present = True

    def get_pose_desc(self, name):
        assert name == "grip"
        return types.SimpleNamespace(validity_flags=self.flags, pose_matrix=pose((0.2, 1.0, 0.3)))

    def get_virtual_world_pose_desc(self, name):
        return self.get_pose_desc(name)

    def get_pose_names(self):
        return ["grip"]

    def has_input(self, name):
        return self.actions_present and name == "thumbstick"

    def has_input_gesture(self, name, axis):
        return name == "thumbstick" and axis in ("x", "y")

    def get_input_gesture_value(self, name, axis):
        assert name == "thumbstick" and axis in ("x", "y"), "A non-stick action was consumed"
        return getattr(self, axis)


class TestControllerOwnership(unittest.TestCase):
    def setUp(self) -> None:
        with patch.dict(
            os.environ,
            {"G1_GESTURE_WALKING": "1", "G1_LOCOMOTION_INPUT": "controller", "G1_OPTICAL_HANDS_ONLY": "0"},
        ):
            self.ex = HUMANOID.HumanoidExample()
        self.ex._controller_command = np.zeros(3)
        self.ex._base_command = np.zeros(3)
        self.ex._keyboard_command = np.zeros(3)
        self.ex._update_headset_gait_command = Mock(return_value=1.0)
        self.ex._read_xr_controller_axes = Mock(side_effect=AssertionError("Legacy buttons must not be read"))
        self.ex._update_gesture_locomotion = Mock(side_effect=AssertionError("Blue-ray movement must remain disabled"))
        self.left, self.right = StickDevice(), StickDevice()
        self.ex._get_xr_input_device = lambda path: self.left if path.endswith("left") else self.right

    def tick(self) -> np.ndarray:
        self.ex._update_controller_command(0.005)
        return self.ex._controller_command.copy()

    def test_sticks_move_in_requested_directions_and_release_stops(self) -> None:
        self.assertFalse(self.ex._pointing_navigation_enabled)
        self.assertTrue(self.ex._controller_arm_control_enabled)
        self.tick()
        for left_x, left_y, right_x, expected in (
            (0, 1, 0, (0.65, 0, 0)),
            (0, -1, 0, (-0.35, 0, 0)),
            (1, 0, 0, (0, -0.30, 0)),
            (-1, 0, 0, (0, 0.30, 0)),
            (0, 0, 1, (0, 0, -0.40)),
            (0, 0, -1, (0, 0, 0.40)),
            (0, 0.575, 0, (0.325, 0, 0)),
            (0, -0.575, 0, (-0.175, 0, 0)),
            (0.575, 0, 0, (0, -0.15, 0)),
            (-0.575, 0, 0, (0, 0.15, 0)),
            (0, 0, 0.575, (0, 0, -0.20)),
            (0, 0, -0.575, (0, 0, 0.20)),
        ):
            self.left.x, self.left.y, self.right.x = left_x, left_y, right_x
            np.testing.assert_allclose(self.tick(), expected)
        self.ex._base_command[:] = self.ex._controller_command
        self.left.x = self.left.y = self.right.x = 0
        np.testing.assert_array_equal(self.tick(), (0, 0, 0))
        np.testing.assert_array_equal(self.ex._base_command, (0, 0, 0))

    def test_missing_tracking_stops_and_requires_neutral_on_reconnect(self) -> None:
        self.tick()
        self.left.y = 1
        self.tick()
        self.ex._base_command[:] = self.ex._controller_command
        self.right.flags = 3  # Valid retained pose is not currently tracked.
        np.testing.assert_array_equal(self.tick(), (0, 0, 0))
        np.testing.assert_array_equal(self.ex._base_command, (0, 0, 0))
        self.right.flags = 15
        np.testing.assert_array_equal(self.tick(), (0, 0, 0))
        self.left.y = 0
        self.tick()
        self.left.y = 1
        self.assertGreater(self.tick()[0], 0)

    def test_controller_commands_survive_the_final_physics_profile_clamps(self) -> None:
        self.tick()
        for axes in ((0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)):
            self.left.x, self.left.y, self.right.x = axes
            requested = self.tick()
            clamped = np.clip(
                requested,
                (-self.ex._max_backward_speed, -self.ex._max_lateral_speed, -self.ex._max_yaw_speed),
                (self.ex._max_forward_speed, self.ex._max_lateral_speed, self.ex._max_yaw_speed),
            )
            np.testing.assert_allclose(clamped, requested)
            self.assertGreater(np.linalg.norm(clamped), 0.1)

    def test_optical_mode_rejects_retained_stick_values_and_requires_neutral(self) -> None:
        self.ex._get_hand_input_kind = Mock(return_value="controller")
        self.tick()
        self.left.y = 1
        self.assertGreater(self.tick()[0], 0)
        self.ex._get_hand_input_kind.return_value = "hand_tracking"
        np.testing.assert_array_equal(self.tick(), (0, 0, 0))
        self.assertEqual(self.ex._controller_locomotion.command.state, "unavailable")
        self.ex._get_hand_input_kind.return_value = "controller"
        np.testing.assert_array_equal(self.tick(), (0, 0, 0))
        self.left.y = 0
        self.tick()
        self.left.y = 1
        self.assertGreater(self.tick()[0], 0)

    def test_bad_action_is_not_converted_to_a_valid_centered_stick(self) -> None:
        for value in (float("nan"), float("inf"), 1.2):
            self.left.x = value
            self.assertFalse(self.ex._read_tracked_controller_stick(self.left)[0])
        self.left.x = 0
        self.left.actions_present = False
        self.assertFalse(self.ex._read_tracked_controller_stick(self.left)[0])

    def test_optical_only_opt_in_still_disables_controller_manipulation(self) -> None:
        with patch.dict(
            os.environ,
            {"G1_GESTURE_WALKING": "1", "G1_LOCOMOTION_INPUT": "controller", "G1_OPTICAL_HANDS_ONLY": "1"},
        ):
            self.ex = HUMANOID.HumanoidExample()
        self.assertTrue(self.ex._controller_walking_enabled)
        self.assertFalse(self.ex._controller_arm_control_enabled)
        device = Mock()
        self.ex._get_hand_input_kind = Mock(return_value="controller")
        self.ex._get_xr_gesture_value = Mock(side_effect=AssertionError("Controller manipulation was read"))
        self.assertIsNone(self.ex._get_controller_arm_pose("left", device))
        self.assertIsNone(self.ex._get_controller_finger_curls(device))
        self.assertFalse(self.ex._is_hand_closed("left", device))
        self.ex._get_xr_gesture_value.assert_not_called()

    def test_controller_clutch_and_trigger_remain_independent_while_sticks_walk(self) -> None:
        self.ex._get_hand_input_kind = Mock(return_value="controller")
        grip, trigger = 0.8, 0.0
        self.ex._get_xr_gesture_value = lambda device, name, gesture: (
            grip
            if name == "squeeze" and gesture == "value"
            else trigger if name == "trigger" and gesture == "value" else 0.0
        )
        self.ex._get_hand_link_prim = Mock(return_value=None)
        self.tick()  # Neutral arms the sticks independently of the held clutch.
        self.left.y, self.right.x = 0.7, 0.3
        controller_pose = self.ex._get_controller_arm_pose("left", self.left)
        self.assertIsNotNone(controller_pose)
        self.assertTrue(all(value == 0.0 for value in self.ex._get_controller_finger_curls(self.left).values()))
        base = Gf.Vec3d(0, 0, 0.8)
        target = self.ex._compute_arm_target_body_position("left", controller_pose, base, 0.0, True)
        neutral = self.ex._controller_arm_neutral_positions["left"]
        trigger = 0.65
        for _ in range(5):
            walking = self.tick()
            self.assertGreater(walking[0], 0.0)
            self.assertLess(walking[2], 0.0)
            self.assertEqual(self.ex._controller_arm_neutral_positions["left"], neutral)
            self.assertEqual(self.ex._controller_arm_neutral_targets["left"], target)
            self.assertTrue(all(value == trigger for value in self.ex._get_controller_finger_curls(self.left).values()))
        # Releasing the arm clutch does not release fingers or zero the sticks.
        grip = 0.0
        self.assertIsNone(self.ex._get_controller_arm_pose("left", self.left))
        self.assertTrue(all(value == trigger for value in self.ex._get_controller_finger_curls(self.left).values()))
        self.assertGreater(self.tick()[0], 0.0)
        # Switching to optical input rejects all retained controller actions.
        self.ex._get_hand_input_kind.return_value = "hand_tracking"
        self.assertIsNone(self.ex._get_controller_arm_pose("left", self.left))
        self.assertIsNone(self.ex._get_controller_finger_curls(self.left))
        np.testing.assert_array_equal(self.tick(), (0, 0, 0))

    def test_optical_closure_still_controls_manipulation(self) -> None:
        self.ex._finger_curl_source["left"] = "hand_tracking"
        self.ex._latest_finger_curls["left"] = {role: 0.8 for role in self.ex._finger_roles}
        self.assertTrue(self.ex._is_hand_closed("left", self.left))
        self.ex._latest_finger_curls["left"] = {role: 0.0 for role in self.ex._finger_roles}
        self.assertFalse(self.ex._is_hand_closed("left", self.left))


if __name__ == "__main__":
    unittest.main()
