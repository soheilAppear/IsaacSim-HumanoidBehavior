# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check episode invalidation without invoking a destroyed physics tensor view."""

from __future__ import annotations

import unittest

import numpy as np
from test_g1_physical_policy_controls import physical_robot


class TestG1Lifecycle(unittest.TestCase):
    """Keep stop/start independent from fault recovery and stale actor state."""

    def test_stop_discards_control_history_without_physics_access(self) -> None:
        robot = physical_robot()
        robot.forward(0.02, [0.2, 0, 0])
        self.assertEqual(robot._velocity_policy.active_mode, "walk")
        self.assertGreater(robot._walk_time, 0)
        robot.robot.reset_mock()

        robot.invalidate_physics_state()

        self.assertFalse(robot._initialized)
        self.assertEqual(robot._walk_time, 0)
        self.assertEqual(robot._walk_step_counter, 0)
        self.assertIsNone(robot._walk_next_control_time)
        self.assertIsNone(robot._velocity_policy.active_mode)
        self.assertIsNone(robot._velocity_policy.last_observation)
        np.testing.assert_array_equal(robot._velocity_policy.last_action, np.zeros(15))
        np.testing.assert_array_equal(robot._velocity_policy._history, np.zeros((6, 86)))
        self.assertEqual(robot.robot.mock_calls, [])

    def test_repeated_stop_retains_fault_and_cannot_resume_by_forward(self) -> None:
        robot = physical_robot()
        robot._locomotion_fault = "Walking balance fault"
        for _ in range(3):
            robot.invalidate_physics_state()
            self.assertEqual(robot.locomotion_fault, "Walking balance fault")
            with self.assertRaisesRegex(RuntimeError, "Walking balance fault"):
                robot.forward(0.005, [0.2, 0, 0])
        robot.robot.set_dof_position_targets.assert_not_called()
        robot.robot.set_dof_efforts.assert_not_called()

    def test_invalid_view_skips_initialization_before_dof_or_drive_access(self) -> None:
        robot = physical_robot()
        robot.robot.is_physics_tensor_entity_valid.return_value = False

        robot.initialize()

        self.assertFalse(robot._initialized)
        self.assertEqual(
            [call[0] for call in robot.robot.mock_calls],
            ["is_physics_tensor_entity_valid"],
        )
        for session in robot._velocity_policy._sessions.values():
            session.run.assert_not_called()

    def test_view_disappearing_prevents_actor_and_effort_reads(self) -> None:
        for method in ("forward", "apply_body_efforts"):
            with self.subTest(method=method):
                robot = physical_robot()
                robot.robot.is_physics_tensor_entity_valid.return_value = False
                if method == "forward":
                    robot.forward(0.005, [0.2, 0, 0])
                else:
                    robot.apply_body_efforts()

                self.assertFalse(robot._initialized)
                self.assertIsNone(robot.locomotion_fault)
                robot.robot.get_world_poses.assert_not_called()
                robot.robot.get_dof_positions.assert_not_called()
                robot.robot.get_dof_position_targets.assert_not_called()
                robot.robot.set_dof_position_targets.assert_not_called()
                robot.robot.set_dof_efforts.assert_not_called()

    def test_invalid_reset_does_not_write_native_state_or_clear_fault(self) -> None:
        robot = physical_robot()
        robot.robot.is_physics_tensor_entity_valid.return_value = False
        robot._locomotion_fault = "Previous failed episode"
        with self.assertRaisesRegex(RuntimeError, "valid physics articulation"):
            robot.post_reset()
        self.assertFalse(robot._initialized)
        self.assertEqual(robot.locomotion_fault, "Previous failed episode")
        robot.robot.reset_to_default_state.assert_not_called()
        robot.robot.set_dof_efforts.assert_not_called()

    def test_stop_requires_fresh_initialize_even_after_view_returns(self) -> None:
        robot = physical_robot()
        robot.invalidate_physics_state()
        robot.robot.is_physics_tensor_entity_valid.return_value = True

        robot.forward(0.005, [0.2, 0, 0])
        robot.apply_body_efforts()

        self.assertEqual(robot.robot.mock_calls, [])
        for session in robot._velocity_policy._sessions.values():
            session.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
