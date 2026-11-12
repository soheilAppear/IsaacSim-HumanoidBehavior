# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise production facial ownership, gaze coexistence and reset safety."""

from __future__ import annotations

import os
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from humanoid_test_support import FACIAL_INPUT, FACIAL_INTEGRATION, HUMANOID, Array
from pxr import Gf


class TestFacialIntegrationControls(unittest.TestCase):
    def make_example(self, *, preview: bool = False, **overrides):
        """Build the real humanoid with deterministic facial and gaze settings."""
        environment = {
            "G1_LOCOMOTION_INPUT": "gaze",
            "G1_GESTURE_WALKING": "1",
            "G1_GAZE_PREVIEW": "1" if preview else "0",
            "G1_GAZE_MODE": "hold",
            "G1_GAZE_CONTROLLER_OVERRIDE": "0",
            "G1_GAZE_DWELL_S": "0",
            "G1_FACE_ENABLED": "1",
            "G1_FACE_PREVIEW": "1" if preview else "0",
            "G1_FACE_MODE": "hold",
            "G1_FACE_CONFIRM_S": "0.08",
            "G1_FACE_NEUTRAL_S": "0.10",
            "G1_FACE_FORWARD_SPEED": "0.20",
            "G1_FACE_BACKWARD_SPEED": "0.15",
            "G1_FACE_LATERAL_SPEED": "0.15",
            "G1_FACE_YAW_SPEED": "0.25",
            "G1_OPTICAL_HANDS_ONLY": "0",
            "G1_GRASP_MODE": "physical",
            "G1_WEBCAM_ENABLED": "0",
        }
        environment.update(overrides)
        with patch.dict(os.environ, environment):
            return HUMANOID.HumanoidExample()

    def setUp(self) -> None:
        self.ex = self.make_example()
        self.time = 10.0
        self.sequence = 0
        self.weights = {name: 0.0 for name in FACIAL_INPUT.EXPRESSION_NAMES}
        self.valid = True
        self.sample_age = 0.0
        self.ex._physics_ready = True
        self.ex._timeline_stopped = False
        self.ex._base_command = torch.zeros(3)
        self.ex._controller_command = torch.zeros(3)
        self.ex._keyboard_command = torch.zeros(3)
        self.ex._eye_gaze_tracker = types.SimpleNamespace(
            read_control_ray=Mock(return_value=None),
            control_session_ready=Mock(return_value=True),
            control_occlusion_distance=Mock(return_value=None),
            update=Mock(),
        )
        self.ex._gaze_controls = Mock()
        self.ex._gaze_controls.hit_test_face_toggle.return_value = False
        self.ex._navigation_scene = Mock()
        self.ex._get_g1_base_pose_for_arms = Mock(return_value=(Gf.Vec3d(0, 0, 0.8), 0.0))
        robot = Mock()
        robot.is_physics_tensor_entity_valid.return_value = True
        robot.get_velocities.return_value = (Array([[0.0, 0.0, 0.0]]), Array([[0.0, 0.0, 0.0]]))
        self.ex.g1 = types.SimpleNamespace(
            robot=robot,
            forward=Mock(),
            uses_explicit_body_efforts=True,
            apply_body_efforts=Mock(),
            invalidate_physics_state=Mock(),
        )
        self.ex._update_headset_gait_command = Mock(return_value=1.0)
        self.ex._log_physics_step_error = Mock()
        self.ex._pause_faulted_articulation = Mock()
        self.ex._facial_input.receiver = Mock()
        self.ex._facial_input.receiver.poll.side_effect = self.sample
        self.ex._facial_input.receiver.diagnostics.return_value = {"port": 9015, "started": True}
        self.ex._get_xr_input_device = Mock(side_effect=AssertionError("Face control cannot require hand devices"))
        self.ex._read_tracked_controller_stick = Mock(side_effect=AssertionError("Stick override is disabled"))
        self.ex.set_facial_locomotion_enabled(True)

    def sample(self, *, now_s: float):
        """Supply fresh hardware-shaped observations to the production state machine."""
        self.sequence += 1
        return FACIAL_INPUT.FaceTrackingSample(
            weights=dict(self.weights),
            valid=self.valid,
            reason="Fresh test face" if self.valid else "Face tracking is stale",
            received_at=now_s - self.sample_age,
            age_s=self.sample_age,
            confidence_upper=0.9,
            confidence_lower=0.9,
            sequence=self.sequence,
        )

    def tick(self, *, elapsed: float = 0.05, **weights):
        """Run actual humanoid dispatch with an independently controlled receipt clock."""
        self.weights = {name: 0.0 for name in FACIAL_INPUT.EXPRESSION_NAMES}
        self.weights.update(weights)
        self.time += elapsed
        with patch.object(FACIAL_INTEGRATION.time, "monotonic", return_value=self.time), patch.object(
            HUMANOID.time, "perf_counter", return_value=self.time
        ):
            self.ex._update_controller_command(0.005)
        return self.ex._facial_input.controller.command

    def activate(self, **weights):
        """Complete neutral and deliberate confirmation with distinct samples."""
        for _ in range(4):
            self.tick()
        for _ in range(3):
            command = self.tick(**weights)
        self.assertEqual(command.activation_count, 1)
        self.assertIsNone(self.ex._articulation_health_fault)
        return command

    def assert_zero_buffers(self) -> None:
        """Check every legacy body-intent buffer is cleared."""
        for value in (self.ex._base_command, self.ex._controller_command, self.ex._keyboard_command):
            np.testing.assert_array_equal(value.numpy(), (0, 0, 0))

    def test_face_owns_body_while_gaze_buttons_are_bypassed(self) -> None:
        self.ex._update_gaze_locomotion = Mock()
        command = self.activate(JawDrop=0.9)
        self.assertEqual(command.last_action, "forward")
        np.testing.assert_allclose(self.ex._controller_command.numpy(), (0.2, 0, 0))
        self.ex._update_gaze_locomotion.assert_not_called()
        self.ex._get_xr_input_device.assert_not_called()
        self.ex._read_tracked_controller_stick.assert_not_called()
        self.assertTrue(self.ex._hand_tracking_arm_control_enabled)
        self.assertTrue(self.ex._controller_arm_control_enabled)
        self.assertTrue(self.ex._finger_control_enabled)

    def test_disabled_face_falls_through_to_existing_gaze_dispatch(self) -> None:
        self.ex.set_facial_locomotion_enabled(False)
        self.ex._update_gaze_locomotion = Mock()
        self.tick(JawDrop=0.9)
        self.ex._update_gaze_locomotion.assert_called_once_with(0.005)
        self.assertFalse(self.ex._facial_input.enabled)

    def test_wink_can_turn_without_a_valid_open_eye_gaze_ray(self) -> None:
        self.ex._eye_gaze_tracker.read_control_ray.return_value = None
        command = self.activate(EyesClosedL=0.9)
        self.assertEqual(command.last_action, "turn_left")
        np.testing.assert_allclose(self.ex._controller_command.numpy(), (0, 0, 0.25))
        self.assertTrue(self.ex._eye_gaze_tracker.control_session_ready.called)

    def test_preview_reports_expression_and_clears_all_motion_intent(self) -> None:
        preview = self.make_example(preview=True)
        self.ex._facial_input.controller = preview._facial_input.controller
        self.ex._base_command[:] = 0.4
        self.ex._controller_command[:] = 0.4
        self.ex._keyboard_command[:] = 0.4
        command = self.activate(MouthRight=0.9)
        self.assertEqual(command.last_action, "strafe_right")
        self.assertEqual(command.velocity, (0, 0, 0))
        self.assert_zero_buffers()
        self.ex.g1.robot.get_velocities.assert_not_called()
        self.assertEqual(preview._g1_locomotion, "stationary")

    def test_relaxation_stops_hold_and_requires_neutral_before_next_gesture(self) -> None:
        self.activate(JawDrop=0.9)
        self.ex._base_command[:] = self.ex._controller_command
        command = self.tick()
        self.assertEqual(command.velocity, (0, 0, 0))
        np.testing.assert_array_equal(self.ex._controller_command.numpy(), (0, 0, 0))
        # A deliberate release keeps the established fast stopping filter;
        # invalid tracking bypasses it and clears the base immediately.
        self.ex._teleop_input_dt = 0.02
        values = []
        for _ in range(10):
            self.ex._smooth_base_command(self.ex._controller_command, 0.005)
            values.append(float(self.ex._base_command[0]))
        self.assertTrue(all(a > b > 0 for a, b in zip([0.2, *values], values)))
        self.assertLess(values[-1], 0.01)
        for _ in range(5):
            command = self.tick(MouthLeft=0.9)
        self.assertEqual(command.activation_count, 1)
        self.assertEqual(command.velocity, (0, 0, 0))

    def test_stale_invalid_and_lost_xr_session_cancel_active_face_motion(self) -> None:
        for failure in ("age", "valid", "session", "brake"):
            with self.subTest(failure=failure):
                self.setUp()
                self.activate(JawDrop=0.9)
                self.ex._base_command[:] = self.ex._controller_command
                if failure == "age":
                    self.sample_age = 0.3
                elif failure == "valid":
                    self.valid = False
                elif failure == "session":
                    self.ex._eye_gaze_tracker.control_session_ready.return_value = False
                else:
                    self.ex._keyboard_brake = True
                command = self.tick(JawDrop=0.9)
                self.assertEqual(command.velocity, (0, 0, 0))
                self.assert_zero_buffers()

    def test_mode_handoff_clears_previous_gaze_and_body_intent(self) -> None:
        for enabled in (False, True):
            for value in (self.ex._base_command, self.ex._controller_command, self.ex._keyboard_command):
                value[:] = 0.3
            self.ex.set_facial_locomotion_enabled(enabled)
            self.assert_zero_buffers()
            self.assertEqual(self.ex._gaze_locomotion.command.state, "awaiting_release")
            self.assertEqual(self.ex._facial_input.enabled, enabled)
            self.assertEqual(self.ex._facial_input.controller.command.state, "awaiting_neutral")

    def test_stick_override_disarms_face_with_zero_handoff_tick(self) -> None:
        self.activate(JawDrop=0.9)
        self.ex._gaze_controller_override_enabled = True
        self.ex._get_xr_input_device = Mock(side_effect=lambda path: path)
        self.ex._read_tracked_controller_stick = Mock(
            side_effect=lambda device: (True, 0.0, 0.8) if device.endswith("left") else (True, 0.0, 0.0)
        )
        self.ex._update_hybrid_locomotion = Mock()
        self.tick(JawDrop=0.9)
        self.assertFalse(self.ex._facial_input.enabled)
        self.assert_zero_buffers()
        self.ex._update_hybrid_locomotion.assert_not_called()
        self.tick()
        self.ex._update_hybrid_locomotion.assert_called_once()

    def test_gaze_toggle_disarms_once_and_preserves_zero_handoff_tick(self) -> None:
        self.activate(JawDrop=0.9)
        self.ex._eye_gaze_tracker.read_control_ray.return_value = (Gf.Vec3d(0, 0, 1.5), Gf.Vec3d(1, 0, 0))
        self.tick(JawDrop=0.9)
        self.ex._gaze_controls.hit_test_face_toggle.return_value = True
        for _ in range(15):
            self.tick(JawDrop=0.9)
        self.assertFalse(self.ex._facial_input.enabled)
        self.assert_zero_buffers()
        self.ex._update_gaze_locomotion = Mock()
        for _ in range(20):
            self.tick(JawDrop=0.9)
        self.assertFalse(self.ex._facial_input.enabled)

    def test_pause_and_stop_disarm_face_and_clear_intent(self) -> None:
        timeline = types.SimpleNamespace(TimelineEventType=types.SimpleNamespace(PAUSE=2, STOP=1))
        for event_type in (2, 1):
            with self.subTest(event_type=event_type):
                self.setUp()
                self.activate(JawDrop=0.9)
                with patch.object(HUMANOID, "import_module", return_value=timeline):
                    self.ex._on_navigation_timeline_event(types.SimpleNamespace(type=event_type))
                self.assertFalse(self.ex._facial_input.enabled)
                self.assert_zero_buffers()
                self.assertEqual(self.ex._facial_input.controller.command.state, "awaiting_neutral")

    def test_close_releases_receiver_and_forgets_sample(self) -> None:
        self.activate(JawDrop=0.9)
        self.ex._facial_input.close()
        self.ex._facial_input.receiver.close.assert_called_once()
        self.assertFalse(self.ex._facial_input.enabled)
        self.assertIsNone(self.ex._facial_input.sample)
        self.assertEqual(self.ex._facial_input.controller.command.velocity, (0, 0, 0))

    def test_bind_failure_keeps_gaze_available_and_reports_enable_error(self) -> None:
        face = self.ex._facial_input
        face.receiver.start.side_effect = OSError("Port already occupied")
        face.start()
        self.assertFalse(face.enabled)
        self.assertIn("Port already occupied", face.error)
        with self.assertRaises(RuntimeError):
            self.ex.set_facial_locomotion_enabled(True)
        self.ex._update_gaze_locomotion = Mock()
        self.tick()
        self.ex._update_gaze_locomotion.assert_called_once()

    def test_facial_dispatch_exception_zeros_intent_and_pauses_safely(self) -> None:
        self.activate(JawDrop=0.9)
        self.ex._base_command[:] = self.ex._controller_command
        self.ex._facial_input.receiver.poll.side_effect = RuntimeError("Receiver failed")
        self.tick()
        self.assertIn("Receiver failed", self.ex._articulation_health_fault)
        self.assert_zero_buffers()
        self.ex._pause_faulted_articulation.assert_called_once()
        self.assertFalse(self.ex._facial_input.enabled)

    def test_active_face_keeps_arm_finger_and_final_effort_order(self) -> None:
        self.activate(JawDrop=0.9)
        order = []
        self.ex._check_articulation_health = Mock(return_value=True)
        self.ex._check_gesture_balance = Mock(return_value=True)
        self.ex._check_gesture_motion = Mock(return_value=True)
        self.ex._poll_webcam_tracking = Mock()
        self.ex.g1.forward.side_effect = lambda *args, **kwargs: order.append("base")
        self.ex._update_g1_fingers = Mock(side_effect=lambda: order.append("fingers"))
        self.ex._update_g1_arms_from_hand_tracking = Mock(side_effect=lambda: order.append("arms"))
        self.ex.g1.apply_body_efforts.side_effect = lambda: order.append("efforts")
        self.ex._update_head_camera_view = Mock(side_effect=lambda **kwargs: order.append("camera"))
        self.ex._eye_gaze_tracker.update.side_effect = lambda dt: order.append("gaze")
        self.ex._collect_all_behavioral_data = Mock(side_effect=lambda: order.append("recording"))
        self.time += 0.05
        with patch.object(FACIAL_INTEGRATION.time, "monotonic", return_value=self.time), patch.object(
            HUMANOID.time, "perf_counter", return_value=self.time
        ):
            self.ex.on_physics_step(0.005, None)
        self.assertEqual(order, ["base", "fingers", "arms", "efforts", "camera", "gaze", "recording"])
        self.assertTrue(self.ex.g1.forward.call_args.kwargs["defer_body_efforts"])
        self.assertGreater(float(self.ex.g1.forward.call_args.args[1][0]), 0)
        self.assertIsNone(self.ex._articulation_health_fault)


if __name__ == "__main__":
    unittest.main()
