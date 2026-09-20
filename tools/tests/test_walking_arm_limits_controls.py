# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep walking arm commands inside the deployed GR00T teleoperation range."""

from __future__ import annotations

import types
import unittest

import numpy as np
import test_arm_contact_controls as contact_controls
from humanoid_test_support import HUMANOID, Array
from pxr import Gf
from test_arm_ik_controls import example_with_jacobian


class TestWalkingArmLimits(unittest.TestCase):
    def setUp(self):
        self.ex = HUMANOID.HumanoidExample()
        self.positions = np.array([[0.4, -0.4, 0.1]])
        self.ex.g1 = types.SimpleNamespace(
            walking_policy_name="decoupled",
            uses_explicit_body_efforts=True,
            robot=types.SimpleNamespace(get_dof_positions=lambda: Array(self.positions)),
        )
        self.ex._g1_arm_dof_indices_by_side = {
            "left": {"shoulder_roll": 0, "elbow": 2},
            "right": {"shoulder_roll": 1},
        }
        self.ex._g1_arm_joint_limits = {0: (-1.5882, 2.2515), 1: (-2.2515, 1.5882), 2: (-1.0, 2.0)}
        self.ex._last_physics_dt = self.ex._teleop_input_dt = 0.01

    def test_deployed_limits_apply_to_balance_without_active_navigation(self):
        self.assertEqual(self.ex._gesture_command.state, "awaiting_release")
        self.assertEqual(self.ex._gesture_command.velocity, (0.0, 0.0, 0.0))
        self.assertEqual(self.ex._get_arm_target_limits(0), (0.19, 2.2515))
        self.assertEqual(self.ex._get_arm_target_limits(1), (-2.2515, -0.19))
        self.assertEqual(self.ex._get_arm_target_limits(2), (-1.0, 2.0))
        # Narrower limits authored by the actual asset always win.
        self.ex._g1_arm_joint_limits[0] = (0.3, 1.4)
        self.assertEqual(self.ex._get_arm_target_limits(0), (0.3, 1.4))

    def test_stationary_and_other_actors_keep_existing_arm_commands(self):
        requested = {0: -1.0, 1: 1.0, 2: 1.0}
        results = []
        for name, explicit in (("stationary", False), ("kinematic", False), ("decoupled", False)):
            self.ex.g1.walking_policy_name = name
            self.ex.g1.uses_explicit_body_efforts = explicit
            self.ex._smoothed_arm_targets.clear()
            for index in requested:
                self.assertEqual(self.ex._get_arm_target_limits(index), self.ex._g1_arm_joint_limits[index])
            results.append(self.ex._smooth_and_clamp_arm_targets(requested))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_outward_goals_cannot_cross_either_policy_shoulder_limit(self):
        self.positions[0, :2] = (0.19, -0.19)
        for _ in range(20):
            targets = self.ex._smooth_and_clamp_arm_targets({0: -10.0, 1: 10.0, 2: 0.8})
            self.assertGreaterEqual(targets[0], 0.19)
            self.assertLessEqual(targets[1], -0.19)
            self.assertGreater(targets[2], 0.1)
            self.positions[0] = [targets[index] for index in range(3)]

    def test_measured_cross_body_posture_returns_gradually_without_snapping(self):
        self.positions[0, :2] = (-1.5, 1.5)
        for _ in range(160):
            previous = self.positions.copy()
            targets = self.ex._smooth_and_clamp_arm_targets({0: -10.0, 1: 10.0})
            for index, direction in ((0, 1), (1, -1)):
                change = direction * (targets[index] - previous[0, index])
                self.assertGreaterEqual(change, -1e-12)
                self.assertLessEqual(change, self.ex._arm_max_joint_speed * 0.01 + 1e-12)
                self.assertLessEqual(change, self.ex._arm_max_tracking_error + 1e-12)
                self.positions[0, index] = targets[index]
        np.testing.assert_allclose(self.positions[0, :2], [0.19, -0.19], atol=1e-6)

    def test_blocked_cross_body_arm_cannot_accumulate_a_distant_recovery_target(self):
        self.positions[0, :2] = (-1.5, 1.5)
        for _ in range(100):
            targets = self.ex._smooth_and_clamp_arm_targets({0: -10.0, 1: 10.0})
        np.testing.assert_allclose([targets[0], targets[1]], [-1.35, 1.35], atol=1e-7)

    def test_stale_outward_command_does_not_push_measured_arm_farther_out(self):
        self.positions[0, :2] = (-0.8, 0.8)
        self.ex._smoothed_arm_targets = {0: -1.5, 1: 1.5}
        targets = self.ex._smooth_and_clamp_arm_targets({0: -10.0, 1: 10.0})
        self.assertGreaterEqual(targets[0], -0.8)
        self.assertLessEqual(targets[1], 0.8)
        self.assertLessEqual(abs(targets[0] + 0.8), self.ex._arm_max_tracking_error)
        self.assertLessEqual(abs(targets[1] - 0.8), self.ex._arm_max_tracking_error)

    def test_live_ik_reassigns_reach_to_free_joint_at_shoulder_bound(self):
        for side, shoulder, direction in (("left", 0.19, -1), ("right", -0.19, 1)):
            with self.subTest(side=side):
                jacobian = np.zeros((6, 7))
                jacobian[0, :2] = 1.0
                current = np.zeros(7)
                current[1] = shoulder
                ex = example_with_jacobian(jacobian, current)
                ex._g1_arm_dof_indices_by_side = {side: ex._g1_arm_dof_indices_by_side["right"]}
                ex.g1.walking_policy_name = "decoupled"
                ex.g1.uses_explicit_body_efforts = True
                result = ex._solve_arm_ik_jacobian(side, Gf.Vec3d(direction * 0.1, 0, 0))
                self.assertIsNotNone(result)
                self.assertAlmostEqual(result[1], shoulder)
                self.assertGreater(direction * result[0], 0.0)

    def test_conflicting_asset_and_policy_limits_fail_before_issuing_targets(self):
        self.ex._g1_arm_joint_limits[0] = (-1.0, 0.1)
        with self.assertRaisesRegex(RuntimeError, "Invalid arm control limits"):
            self.ex._smooth_and_clamp_arm_targets({0: 0.5})
        self.assertFalse(self.ex._smoothed_arm_targets)

    def contact_example(self, current):
        fixture = contact_controls.TestArmContactIntegration()
        fixture.setUp()
        ex = fixture.ex
        ex.g1.walking_policy_name = "decoupled"
        ex.g1.uses_explicit_body_efforts = True
        ex._g1_arm_dof_indices_by_side["right"] = {"shoulder_roll": 1, "wrist_yaw": 3}
        ex._g1_arm_joint_limits[1] = (-2.2515, 1.5882)
        fixture.positions[0, 1] = current
        return fixture

    def test_contact_projection_cannot_reintroduce_cross_body_shoulder_command(self):
        fixture = self.contact_example(-0.19)
        # Positive shoulder travel is away from this surface, but outside the
        # walking range. The contact projection must honor both constraints.
        result = fixture.ex._limit_arm_targets_at_contacts({1: -0.09, 3: 0.05}, {})
        self.assertAlmostEqual(result[1], -0.19)
        self.assertAlmostEqual(result[3], 0.05)

    def test_contact_can_hold_out_of_range_arm_instead_of_forcing_recovery(self):
        fixture = self.contact_example(0.5)
        # Returning right shoulder toward -0.19 would move into this table.
        # Zero motion remains feasible even outside the narrower posture range.
        result = fixture.ex._limit_arm_targets_at_contacts({1: 0.475, 3: 0.05}, {})
        self.assertAlmostEqual(result[1], 0.5)
        self.assertAlmostEqual(result[3], 0.05)
        self.assertFalse(fixture.ex._articulation_health_fault)


if __name__ == "__main__":
    unittest.main()
