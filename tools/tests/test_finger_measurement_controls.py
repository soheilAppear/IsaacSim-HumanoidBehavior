# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual finger actuator measurements used for contact compliance."""

from __future__ import annotations

import math
import types
import unittest
from unittest.mock import Mock

from humanoid_test_support import G1, Array


def measured_hand(positions):
    hand = G1.G1TeleopRobot.__new__(G1.G1TeleopRobot)
    hand._finger_dof_indices = {"right": {"thumb": 3, "index": 1, "thumb_yaw": 0}}
    hand._finger_open_closed = {"right": {"thumb": (0.1, 0.5), "index": (0.2, -1.4), "thumb_yaw": (-0.1, 0.9)}}
    hand.robot = types.SimpleNamespace(
        is_physics_tensor_entity_valid=Mock(return_value=True),
        get_dof_positions=Mock(return_value=Array([positions])),
        set_dof_position_targets=Mock(),
    )
    return hand


class TestFingerMeasurementControls(unittest.TestCase):
    def test_actual_positions_use_one_read_and_preserve_reverse_joint_ranges(self):
        hand = measured_hand([0.4, -0.6, 99.0, 0.3])
        result = hand.get_finger_curls("right")
        for role in ("thumb", "index", "thumb_yaw"):
            self.assertAlmostEqual(result[role], 0.5, places=6)
        hand.robot.get_dof_positions.assert_called_once_with()
        self.assertEqual(hand.get_finger_joint_range("right", "index"), (0.2, -1.4))
        self.assertEqual(hand.get_finger_joint_range("right", "thumb"), (0.1, 0.5))

    def test_commands_are_not_reported_as_measured_motion_and_each_call_refreshes(self):
        hand = measured_hand([-0.1, 0.2, 0, 0.1])
        hand.set_finger_curls("right", {"thumb": 1.0, "index": 1.0, "thumb_yaw": 1.0})
        result = hand.get_finger_curls("right")
        self.assertTrue(all(abs(curl) < 1e-6 for curl in result.values()))
        hand.robot.get_dof_positions.return_value = Array([[0.9, -1.4, 0.0, 0.5]])
        result = hand.get_finger_curls("right")
        self.assertTrue(all(abs(curl - 1.0) < 1e-6 for curl in result.values()))
        self.assertEqual(hand.robot.get_dof_positions.call_count, 2)

    def test_contact_deflection_outside_command_ranges_is_clamped(self):
        hand = measured_hand([2.0, 1.0, 0.0, -1.0])
        self.assertEqual(hand.get_finger_curls("right"), {"thumb": 0.0, "index": 0.0, "thumb_yaw": 1.0})

    def test_invalid_individual_measurements_and_ranges_are_omitted(self):
        hand = measured_hand([0.4, math.nan, 0.0, 0.3])
        self.assertNotIn("index", hand.get_finger_curls("right"))
        hand._finger_open_closed["right"]["thumb"] = (0.3, 0.3)
        hand._finger_open_closed["right"]["thumb_yaw"] = (0.0, math.inf)
        self.assertIsNone(hand.get_finger_joint_range("right", "thumb"))
        self.assertIsNone(hand.get_finger_joint_range("right", "thumb_yaw"))
        self.assertEqual(hand.get_finger_curls("right"), {})

    def test_missing_hand_stale_view_failed_read_and_malformed_snapshot_are_empty(self):
        hand = measured_hand([0.4, -0.6, 0.0, 0.3])
        self.assertIsNone(hand.get_finger_joint_range("left", "index"))
        self.assertEqual(hand.get_finger_curls("left"), {})
        hand.robot.get_dof_positions.assert_not_called()
        hand.robot.is_physics_tensor_entity_valid.return_value = False
        self.assertEqual(hand.get_finger_curls("right"), {})
        hand.robot.get_dof_positions.assert_not_called()
        hand.robot.is_physics_tensor_entity_valid.return_value = True
        hand.robot.get_dof_positions.side_effect = RuntimeError("simulation stopped")
        self.assertEqual(hand.get_finger_curls("right"), {})
        hand.robot.get_dof_positions.side_effect = None
        hand.robot.get_dof_positions.return_value = Array([0.4, -0.6, 0.0, 0.3])
        self.assertEqual(hand.get_finger_curls("right"), {})


if __name__ == "__main__":
    unittest.main()
