# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contain divergent physics before it reaches tracking, cameras, or recording."""

from __future__ import annotations

import math
import types
import unittest
from unittest.mock import Mock, patch

from humanoid_test_support import HEALTH, HUMANOID, Array


class TestArticulationStateBounds(unittest.TestCase):
    def check(self, position=0.0, velocity=0.0, limits=(-1.0, 1.0)):
        return HEALTH.find_articulation_fault(["waist_pitch_joint"], [position], [velocity], [limits])

    def test_accepts_normal_finger_speed_and_small_contact_overshoot(self):
        self.assertIsNone(self.check(position=1.1, velocity=8.0))
        self.assertIsNone(self.check(position=-1.1, velocity=-19.2))

    def test_rejects_finite_solver_explosion_even_with_unbounded_authored_limits(self):
        reason = self.check(position=1e33, limits=(-math.inf, math.inf))
        self.assertIn("waist_pitch_joint", reason)
        self.assertIn("impossible joint angle", reason)

    def test_rejects_each_nonfinite_position_and_velocity(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.assertIn("nonfinite", self.check(position=value))
                self.assertIn("nonfinite", self.check(velocity=value))

    def test_rejects_large_velocity_before_position_explodes(self):
        self.assertIn("velocity", self.check(velocity=101.0))
        self.assertIn("velocity", self.check(velocity=-101.0))

    def test_rejects_substantial_limit_violation_in_either_direction(self):
        self.assertIn("exceeds limits", self.check(position=1.36))
        self.assertIn("exceeds limits", self.check(position=-1.36))

    def test_checks_passive_finger_and_waist_not_only_arm_joints(self):
        names = ["right_shoulder_pitch_joint", "R_thumb_distal_joint", "waist_roll_joint"]
        for index in (1, 2):
            positions = [0.0, 0.0, 0.0]
            positions[index] = 1e34
            self.assertIn(names[index], HEALTH.find_articulation_fault(names, positions, [0.0] * 3, [(-2, 2)] * 3))

    def test_rejects_incomplete_or_malformed_state(self):
        self.assertIsNotNone(HEALTH.find_articulation_fault([], [], [], []))
        self.assertIsNotNone(HEALTH.find_articulation_fault(["joint"], [], [0.0], [(-1, 1)]))
        self.assertIsNotNone(HEALTH.find_articulation_fault(["joint"], None, [0.0], [(-1, 1)]))
        self.assertIsNotNone(self.check(limits=(1, -1)))
        self.assertIsNotNone(self.check(limits=(math.nan, 1)))


class TestArticulationFaultIntegration(unittest.TestCase):
    def setUp(self):
        self.ex = HUMANOID.HumanoidExample()
        self.robot = Mock()
        self.robot.dof_names = ["waist_pitch_joint", "R_index_intermediate_joint"]
        self.robot.get_dof_limits.return_value = (Array([[-1.0, -0.2]]), Array([[1.0, 2.0]]))
        self.robot.get_dof_positions.return_value = Array([[0.0, 0.5]])
        self.robot.get_dof_velocities.return_value = Array([[0.0, 8.0]])
        self.robot.is_physics_tensor_entity_valid.return_value = True
        self.ex.g1 = types.SimpleNamespace(robot=self.robot, forward=Mock())
        self.ex._physics_ready = True
        self.ex._log_physics_step_error = Mock()
        self.ex._update_g1_fingers = Mock()
        self.ex._update_g1_arms_from_hand_tracking = Mock()
        self.ex._update_head_camera_view = Mock()
        self.ex._eye_gaze_tracker = Mock()
        self.ex._collect_all_behavioral_data = Mock()
        self.timeline = Mock()
        self.timeline.is_playing.return_value = True
        self.timeline.pause.side_effect = lambda: setattr(self.timeline.is_playing, "return_value", False)
        self.timeline_module = types.SimpleNamespace(get_timeline_interface=lambda: self.timeline)
        self.import_patch = patch.object(HUMANOID, "import_module", return_value=self.timeline_module)
        self.import_patch.start()
        self.addCleanup(self.import_patch.stop)

    def test_fault_pauses_before_any_control_camera_or_recording_update(self):
        self.robot.get_dof_positions.return_value = Array([[1e33, 0.5]])
        self.ex.on_physics_step(0.01, None)
        self.timeline.pause.assert_called_once()
        self.ex.g1.forward.assert_not_called()
        self.ex._update_g1_fingers.assert_not_called()
        self.ex._update_g1_arms_from_hand_tracking.assert_not_called()
        self.ex._update_head_camera_view.assert_not_called()
        self.ex._eye_gaze_tracker.update.assert_not_called()
        self.ex._collect_all_behavioral_data.assert_not_called()
        message = str(self.ex._log_physics_step_error.call_args.args[1])
        self.assertIn("example's Reset or Load", message)
        self.assertIn("Stop/Play does not clear this fault", message)

    def test_latch_survives_better_readings_and_repauses_play_without_repeated_log(self):
        self.robot.get_dof_velocities.return_value = Array([[1e20, 8.0]])
        self.ex.on_physics_step(0.01, None)
        fault = self.ex._articulation_health_fault
        self.robot.get_dof_velocities.return_value = Array([[0.0, 8.0]])
        self.ex.on_physics_step(0.01, None)
        self.timeline.pause.assert_called_once()
        self.timeline.is_playing.return_value = True
        self.ex.on_physics_step(0.01, None)
        self.assertEqual(self.timeline.pause.call_count, 2)
        self.assertEqual(self.ex._articulation_health_fault, fault)
        self.ex._log_physics_step_error.assert_called_once()
        self.ex.g1.forward.assert_not_called()

    def test_healthy_state_caches_limits_but_reads_fresh_measurements_every_tick(self):
        self.assertTrue(self.ex._check_articulation_health())
        self.assertTrue(self.ex._check_articulation_health())
        self.robot.get_dof_limits.assert_called_once()
        self.assertEqual(self.robot.get_dof_positions.call_count, 2)
        self.assertEqual(self.robot.get_dof_velocities.call_count, 2)
        self.timeline.pause.assert_not_called()
        self.robot.get_dof_positions.return_value = Array([[0.0, 3.0]])
        self.assertFalse(self.ex._check_articulation_health())
        self.assertIn("R_index_intermediate_joint", self.ex._articulation_health_fault)

    def test_reset_discards_fault_and_old_limit_cache(self):
        self.robot.get_dof_positions.return_value = Array([[1e33, 0.0]])
        self.assertFalse(self.ex._check_articulation_health())
        self.ex._release_grabbed_object = Mock()
        self.ex._set_grab_candidate = Mock()
        self.ex._reset_teleoperation_state()
        self.assertIsNone(self.ex._articulation_health_fault)
        self.assertIsNone(self.ex._articulation_health_limits)
        self.assertIsNone(self.ex._articulation_health_names)
        self.robot.get_dof_positions.return_value = Array([[0.0, 0.0]])
        self.assertTrue(self.ex._check_articulation_health())
        self.assertEqual(self.robot.get_dof_limits.call_count, 2)

    def test_failed_state_read_cannot_continue_with_last_healthy_sample(self):
        self.assertTrue(self.ex._check_articulation_health())
        self.robot.get_dof_velocities.side_effect = RuntimeError("tensor unavailable")
        self.ex.on_physics_step(0.01, None)
        self.timeline.pause.assert_called_once()
        self.ex.g1.forward.assert_not_called()
        self.assertIn("tensor unavailable", self.ex._articulation_health_fault)


if __name__ == "__main__":
    unittest.main()
