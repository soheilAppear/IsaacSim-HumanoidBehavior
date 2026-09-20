# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the released lower-body actors and independent arm ownership offline."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from humanoid_test_support import DECOUPLED, G1, Array
from test_g1_physical_policy_controls import decoupled_fixture, physical_robot


def infer(policy, value=1.0, command=(0, 0, 0)):
    return policy.infer(
        policy.default_positions + value,
        np.full(29, 2 * value),
        [value, value * 2, value * 3],
        [0, 0, -1],
        command,
    )


class TestDecoupledPolicy(unittest.TestCase):
    def test_source_startup_pose_does_not_replace_observation_or_action_offsets(self):
        policy = decoupled_fixture()
        np.testing.assert_array_equal(policy.initial_positions[:15], np.zeros(15))
        expected_upper = np.zeros(14)
        expected_upper[[1, 8]] = [0.2, -0.2]
        np.testing.assert_allclose(policy.initial_positions[15:], expected_upper)
        expected_defaults = [-0.1, 0, 0, 0.3, -0.2, 0] * 2 + [0] * 17
        np.testing.assert_allclose(policy.default_positions, expected_defaults)

        target = policy.infer(policy.initial_positions, np.zeros(29), [0, 0, 0], [0, 0, -1], [0, 0, 0])
        # Source XML resets q to zero; deployment offsets remain the actor's
        # observation reference and its zero-action targets after activation.
        current_frame = policy.last_observation[0, -86:]
        np.testing.assert_allclose(current_frame[13:42], policy.initial_positions - policy.default_positions)
        np.testing.assert_array_equal(target, policy.default_positions[:15])

    def test_both_real_checkpoints_are_finite_and_reset_deterministic(self):
        policy = DECOUPLED.G1DecoupledPolicy()
        policy.load()
        for command, mode in (([0, 0, 0], "balance"), ([0.25, 0, 0], "walk")):
            policy.reset()
            first = infer(policy, 0, command)
            self.assertEqual(first.shape, (15,))
            self.assertTrue(np.isfinite(first).all())
            self.assertEqual(policy.active_mode, mode)
            for _ in range(10):
                infer(policy, 0.1, command)
            policy.reset()
            np.testing.assert_array_equal(first, infer(policy, 0, command))

    def test_first_frame_is_zero_padded_and_scales_match_deployment(self):
        policy = decoupled_fixture()
        infer(policy, command=(0.2, -0.1, 0.3))
        frame = np.concatenate(
            ([0.4, -0.2, 0.15, 0.74, 0, 0, 0, 0.5, 1, 1.5, 0, 0, -1], np.ones(29), np.full(29, 0.1), np.zeros(15))
        )
        np.testing.assert_array_equal(policy.last_observation[0, :430], np.zeros(430))
        np.testing.assert_allclose(policy.last_observation[0, 430:], frame, atol=1e-7)
        self.assertEqual(policy.last_observation.dtype, np.float32)

    def test_history_groups_whole_frames_and_keeps_only_lower_body_previous_actions(self):
        policy = decoupled_fixture()
        policy._sessions["balance"].run.return_value = [np.full((1, 15), 2, dtype=np.float32)]
        infer(policy, 1)
        previous = policy.last_observation[0, -86:].copy()
        policy._sessions["balance"].run.return_value = [np.full((1, 15), 3, dtype=np.float32)]
        target = infer(policy, 2)
        np.testing.assert_array_equal(policy.last_observation[0, -172:-86], previous)
        np.testing.assert_array_equal(policy.last_observation[0, -15:], np.full(15, 2))
        np.testing.assert_allclose(target, policy.default_positions[:15] + 0.75)
        self.assertEqual(policy.last_action.shape, (15,))

    def test_actor_switch_uses_unscaled_command_norm_and_preserves_history(self):
        policy = decoupled_fixture()
        infer(policy, command=(0, 0, 0.049))
        self.assertEqual(policy.active_mode, "balance")
        first = policy.last_observation[0, -86:].copy()
        infer(policy, command=(0, 0, 0.051))
        self.assertEqual(policy.active_mode, "walk")
        np.testing.assert_array_equal(policy.last_observation[0, -172:-86], first)
        self.assertEqual(policy._sessions["balance"].run.call_count, 1)
        self.assertEqual(policy._sessions["walk"].run.call_count, 1)

    def test_invalid_input_or_action_cannot_commit_state(self):
        for bad in (np.full((1, 15), np.nan), np.zeros((1, 29))):
            policy = decoupled_fixture()
            infer(policy)
            before = policy.last_observation.copy()
            policy._sessions["balance"].run.return_value = [bad]
            with self.assertRaises(ValueError):
                infer(policy, 2)
            np.testing.assert_array_equal(policy.last_observation, before)
            np.testing.assert_array_equal(policy.last_action, np.zeros(15))
        policy = decoupled_fixture()
        with self.assertRaises(ValueError):
            infer(policy, np.inf)
        policy._sessions["balance"].run.assert_not_called()

    def test_all_body_joint_measurements_are_resolved_by_name(self):
        policy = decoupled_fixture()
        names = ["L_index_proximal_joint", *reversed(policy.joint_names)]
        self.assertEqual(policy.resolve_joint_indices(names), list(range(29, 0, -1)))
        for invalid in (names[:-1], [*names, names[-1]]):
            with self.assertRaises(ValueError):
                policy.resolve_joint_indices(invalid)

    def test_checksum_mismatch_cannot_create_runtime_session(self):
        policy = DECOUPLED.G1DecoupledPolicy()
        with patch.object(DECOUPLED.Path, "read_bytes", return_value=b"wrong"), patch(
            "onnxruntime.InferenceSession"
        ) as create:
            with self.assertRaisesRegex(ValueError, "checksum"):
                policy.load()
            create.assert_not_called()
        self.assertEqual(policy._sessions, {})

    def test_invalid_selector_fails_before_creating_robot(self):
        for selection in ("invalid", "unitree29", ""):
            with self.subTest(selection=selection), patch.dict(G1.os.environ, {"G1_WALK_POLICY": selection}):
                with self.assertRaisesRegex(ValueError, "Unknown physical"):
                    G1.G1TeleopRobot("/World/G1", locomotion="policy", physical_locomotion=True)

    def test_wrong_onnx_interface_cannot_publish_sessions(self):
        for input_shape, output_shape, input_type in (
            ([1, 480], [1, 15], "tensor(float)"),
            ([1, 516], [1, 29], "tensor(float)"),
            ([1, 516], [1, 15], "tensor(double)"),
        ):
            policy = DECOUPLED.G1DecoupledPolicy()
            session = Mock()
            session.get_inputs.return_value = [SimpleNamespace(name="input", shape=input_shape, type=input_type)]
            session.get_outputs.return_value = [
                SimpleNamespace(name="output", shape=output_shape, type="tensor(float)")
            ]
            with patch("onnxruntime.InferenceSession", return_value=session), self.assertRaisesRegex(
                ValueError, "interface"
            ):
                policy.load()
            self.assertEqual(policy._sessions, {})

    def test_robot_observes_arms_but_never_overwrites_them_or_the_root(self):
        robot = physical_robot()
        policy = decoupled_fixture()
        robot._velocity_policy = policy
        self.assertTrue(robot.supports_arm_teleoperation)
        robot.robot.dof_names = ["L_index_proximal_joint", *reversed(policy.joint_names), "R_index_proximal_joint"]
        robot.robot.get_dof_limits.return_value = (Array([[-3] * 15]), Array([[3] * 15]))
        policy.load = Mock()
        robot._configure_velocity_policy(robot.robot.dof_names)
        robot.robot.get_dof_positions.return_value = Array(policy.initial_positions.reshape(1, -1))
        robot.robot.get_dof_position_targets.return_value = Array(policy.initial_positions.reshape(1, -1))
        robot.robot.get_dof_velocities.return_value = Array([np.zeros(29)])
        for _ in range(200):
            robot.forward(0.005, [0, 0, 0])
        self.assertEqual(policy._sessions["balance"].run.call_count, 50)
        call = robot.robot.set_dof_position_targets.call_args
        written = [robot.robot.dof_names[i] for i in call.kwargs["dof_indices"]]
        self.assertEqual(written, list(policy.action_joint_names))
        robot.robot.get_dof_positions.assert_called_with(dof_indices=list(range(29, 0, -1)))
        robot.robot.set_world_poses.assert_not_called()
        robot.robot.set_velocities.assert_not_called()

    def test_missing_physical_actor_does_not_advertise_arm_override_support(self):
        robot = physical_robot()
        robot._velocity_policy = None
        self.assertFalse(robot.supports_arm_teleoperation)
        robot._physical_locomotion = False
        robot._locomotion = "stationary"
        self.assertTrue(robot.supports_arm_teleoperation)


if __name__ == "__main__":
    unittest.main()
