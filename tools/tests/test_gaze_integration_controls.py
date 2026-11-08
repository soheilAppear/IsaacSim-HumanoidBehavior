# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check production gaze input ownership, strict eye validity, and physics ordering."""

from __future__ import annotations

import math
import os
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from humanoid_test_support import GAZE, HUMANOID, USD, Array, Device, pose
from pxr import Gf, Usd, UsdGeom
from test_finger_controls import SkeletonDevice, skeleton_positions


class FreshRayTracker:
    """Expose fresh eye polling and fail if controls use the visualization cache."""

    def __init__(self) -> None:
        self.read_control_ray = Mock(return_value=(Gf.Vec3d(0, 0, 1.5), Gf.Vec3d(1, 0, 0)))
        self.control_session_ready = Mock(return_value=True)
        self.control_occlusion_distance = Mock(return_value=None)
        self.update = Mock()

    @property
    def latest(self):
        """Reject reuse of a throttled visualization snapshot as control input."""
        raise AssertionError("Gaze controls must not read tracker.latest")


class TestGazeIntegrationControls(unittest.TestCase):
    def make_example(self, preview: bool = False, **overrides):
        """Construct the real example with an explicit deterministic gaze profile."""
        environment = {
            "G1_LOCOMOTION_INPUT": "gaze",
            "G1_GESTURE_WALKING": "1",
            "G1_GAZE_PREVIEW": "1" if preview else "0",
            "G1_GAZE_MODE": "step",
            "G1_GAZE_DWELL_S": "0.3",
            "G1_GAZE_DISTANCE_M": "0.2",
            "G1_GAZE_TURN_DEG": "10",
            "G1_GAZE_FORWARD_SPEED": "0.2",
            "G1_GAZE_BACKWARD_SPEED": "0.15",
            "G1_GAZE_YAW_SPEED": "0.25",
            "G1_GAZE_MAX_GAP_S": "0.25",
            "G1_GAZE_HEADING_HOLD": "1",
            "G1_GAZE_HEADING_GAIN": "3.0",
            "G1_GAZE_HEADING_MAX_YAW_SPEED": "0.15",
            "G1_OPTICAL_HANDS_ONLY": "0",
            "G1_GRASP_MODE": "physical",
            "G1_WEBCAM_ENABLED": "0",
        }
        environment.update(overrides)
        with patch.dict(os.environ, environment):
            return HUMANOID.HumanoidExample()

    def setUp(self) -> None:
        self.ex = self.make_example()
        self.time = 0.0
        self.target = None
        self.ex._physics_ready = True
        self.ex._timeline_stopped = False
        self.ex._base_command = torch.zeros(3)
        self.ex._controller_command = torch.zeros(3)
        self.ex._keyboard_command = torch.zeros(3)
        self.ex._eye_gaze_tracker = FreshRayTracker()
        self.ex._gaze_controls = Mock()
        self.ex._gaze_controls.hit_test.side_effect = lambda *args, **kwargs: self.target
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
        for name in (
            "_read_xr_controller_axes",
            "_read_gamepad_controller_axes",
            "_read_tracked_controller_stick",
            "_get_xr_gesture_value",
            "_get_xr_input_device",
            "_update_joystick_locomotion",
            "_update_gesture_locomotion",
        ):
            setattr(self.ex, name, Mock(side_effect=AssertionError("Hand/controller input must not gate gaze")))

    def tick(self, target: str | None = None, dt: float = 0.005):
        """Run the production locomotion-input dispatch with current eye input."""
        self.target = target
        self.time += 0.1
        with patch.object(HUMANOID.time, "perf_counter", return_value=self.time):
            self.ex._update_controller_command(dt)
        self.assertIsNone(self.ex._articulation_health_fault)
        return self.ex._gaze_locomotion.command

    def activate(self, action: str = "forward"):
        """Release the interlock and dwell without any hand samples or button presses."""
        self.tick()
        for _ in range(4):
            result = self.tick(action)
        self.assertEqual(result.activated_action, action)
        return result

    def test_preview_anchors_robot_and_keeps_both_manipulation_paths_enabled(self) -> None:
        preview = self.make_example(preview=True)
        self.assertEqual(preview._g1_locomotion, "stationary")
        self.assertFalse(preview._gesture_locomotion_enabled)
        self.assertEqual(preview._world_settings["physics_dt"], 0.01)
        self.assertTrue(preview._gaze_walking_enabled)
        self.assertFalse(preview._controller_walking_enabled)
        self.assertFalse(preview._pointing_navigation_enabled)
        self.assertTrue(preview._hand_tracking_arm_control_enabled)
        self.assertTrue(preview._controller_arm_control_enabled)
        self.assertTrue(preview._finger_control_enabled)

    def test_gaze_desktop_camera_frames_side_controls_at_widescreen_aspect(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        self.ex._prepare_robot_head_camera_mount = Mock()
        with patch.object(USD, "get_context", return_value=types.SimpleNamespace(get_stage=lambda: stage)):
            self.ex._create_head_camera()
        camera = UsdGeom.Camera(stage.GetPrimAtPath(self.ex._head_camera_path))
        half_width = camera.GetHorizontalApertureAttr().Get() / (2.0 * camera.GetFocalLengthAttr().Get())
        half_height = half_width * 9.0 / 16.0
        # Include label/history margins, not just the selectable rectangles.
        self.assertGreater(half_width, 0.76)
        self.assertGreater(half_height, 0.54)

    def test_movement_uses_physical_profile_and_matching_final_command_caps(self) -> None:
        self.assertEqual(self.ex._g1_locomotion, "policy")
        self.assertTrue(self.ex._gesture_locomotion_enabled)
        self.assertEqual(self.ex._world_settings["physics_dt"], 0.005)
        self.assertTrue(self.ex._direct_walking_enabled)
        self.assertEqual(
            (self.ex._max_forward_speed, self.ex._max_backward_speed, self.ex._max_yaw_speed), (0.65, 0.35, 0.4)
        )
        for action in ("forward", "back", "turn_left", "turn_right"):
            self.ex._reset_gesture_locomotion()
            result = self.activate(action)
            bounds = ((-0.35, 0.65), (-0.3, 0.3), (-0.4, 0.4))
            self.assertTrue(any(result.velocity))
            for value, (lower, upper) in zip(result.velocity, bounds):
                self.assertLessEqual(lower, value)
                self.assertLessEqual(value, upper)

    def test_preview_and_movement_select_the_same_shelf_layout(self) -> None:
        preview = self.make_example(preview=True)
        self.assertEqual(preview._work_surface_placements, self.ex._work_surface_placements)
        for example in (preview, self.ex):
            stage = Usd.Stage.CreateInMemory()
            example._create_contact_grasp_objects = Mock()
            example._create_floor_cartons = Mock()
            with patch.object(
                USD, "get_context", return_value=types.SimpleNamespace(get_stage=lambda: stage)
            ), patch.object(
                HUMANOID, "create_shelf_packages", return_value=["/World/ShelfPackage"]
            ) as shelves, patch.object(
                HUMANOID, "create_tall_packages"
            ) as tall:
                example._create_sample_boxes()
            shelves.assert_called_once_with(stage, example._sample_box_root_path, first_index=0)
            tall.assert_not_called()
            self.assertEqual(example._spawned_sample_box_paths, ["/World/ShelfPackage"])

    def test_environment_rejects_speeds_outside_the_actual_walking_caps(self) -> None:
        for variable, value in (
            ("G1_GAZE_FORWARD_SPEED", "0.66"),
            ("G1_GAZE_BACKWARD_SPEED", "0.36"),
            ("G1_GAZE_YAW_SPEED", "0.41"),
            ("G1_GAZE_PREVIEW", "maybe"),
            ("G1_GAZE_MODE", "toggle"),
            ("G1_GAZE_HEADING_HOLD", "maybe"),
            ("G1_GAZE_HEADING_GAIN", "nan"),
            ("G1_GAZE_HEADING_MAX_YAW_SPEED", "0.41"),
        ):
            with self.subTest(variable=variable), self.assertRaises(ValueError):
                self.make_example(**{variable: value})

    def test_gaze_selection_has_no_legacy_stick_grip_or_hand_dependency(self) -> None:
        self.ex._keyboard_command[:] = 0.5
        result = self.activate()
        self.assertEqual(result.velocity, (0.2, 0, 0))
        np.testing.assert_array_equal(self.ex._keyboard_command.numpy(), (0, 0, 0))
        self.ex._read_xr_controller_axes.assert_not_called()
        self.ex._read_gamepad_controller_axes.assert_not_called()
        self.ex._get_xr_gesture_value.assert_not_called()
        self.ex._get_xr_input_device.assert_not_called()
        self.assertEqual(self.ex._update_headset_gait_command.call_count, 5)

    def test_every_input_tick_polls_current_strict_eyes_instead_of_latest_snapshot(self) -> None:
        tracker = self.ex._eye_gaze_tracker
        self.tick()
        self.tick("forward")
        tracker.read_control_ray.return_value = None
        result = self.tick("forward")
        self.assertEqual(result.state, "unavailable")
        self.assertFalse(self.ex._gaze_eye_valid)
        self.assertEqual(tracker.read_control_ray.call_count, 3)
        tracker.read_control_ray.return_value = (Gf.Vec3d(0, 0, 1.5), Gf.Vec3d(1, 0, 0))
        self.assertEqual(self.tick("forward").progress, 0)

    def test_occluded_button_cannot_begin_a_dwell(self) -> None:
        self.tick()
        self.target = "forward"
        self.ex._gaze_controls.hit_test.side_effect = lambda *args, **kwargs: (
            None if "max_distance" in kwargs else "forward"
        )
        for _ in range(6):
            result = self.tick("forward")
            self.assertEqual((result.state, result.activation_count), ("unavailable", 0))
        self.assertTrue(self.ex._eye_gaze_tracker.control_occlusion_distance.called)

    def test_occluding_a_selected_button_does_not_rearm_it_when_it_reappears(self) -> None:
        self.ex._gaze_locomotion = self.make_example(preview=True)._gaze_locomotion
        self.activate()
        self.ex._gaze_controls.hit_test.side_effect = lambda *args, **kwargs: (
            None if "max_distance" in kwargs else "forward"
        )
        self.assertEqual(self.tick("forward").state, "unavailable")
        self.ex._gaze_controls.hit_test.side_effect = lambda *args, **kwargs: self.target
        for _ in range(6):
            result = self.tick("forward")
            self.assertEqual((result.state, result.activation_count), ("preview", 1))
            self.assertIsNone(result.activated_action)
        self.assertEqual(self.activate().activation_count, 2)

    def test_xr_shutdown_cancels_active_motion_and_requires_valid_look_away_after_reconnect(self) -> None:
        self.activate()
        self.ex._base_command[:] = torch.tensor([0.2, 0, 0])
        tracker = self.ex._eye_gaze_tracker
        tracker.control_session_ready.return_value = False
        tracker.read_control_ray.return_value = None
        result = self.tick("forward")
        self.assertEqual((result.state, result.velocity), ("unavailable", (0, 0, 0)))
        for commands in (self.ex._base_command, self.ex._controller_command, self.ex._keyboard_command):
            np.testing.assert_array_equal(commands.numpy(), (0, 0, 0))
        tracker.control_session_ready.return_value = True
        tracker.read_control_ray.return_value = (Gf.Vec3d(0, 0, 1.5), Gf.Vec3d(1, 0, 0))
        for _ in range(6):
            result = self.tick("forward")
            self.assertEqual((result.state, result.activation_count), ("awaiting_release", 1))
        self.assertEqual(self.activate().activation_count, 2)

    def test_presentation_and_reset_display_failures_still_zero_motion_and_pause(self) -> None:
        self.activate()
        self.ex._base_command[:] = torch.tensor([0.2, 0, 0])
        self.ex._gaze_controls.update.side_effect = RuntimeError("USD presentation handle was invalidated")
        self.ex._pause_faulted_articulation = types.MethodType(
            HUMANOID.HumanoidExample._pause_faulted_articulation, self.ex
        )
        timeline = types.SimpleNamespace(is_playing=Mock(return_value=True), pause=Mock())
        native_timeline = types.SimpleNamespace(get_timeline_interface=lambda: timeline)
        self.time += 0.1
        with patch.object(HUMANOID, "import_module", return_value=native_timeline), patch.object(
            HUMANOID.time, "perf_counter", return_value=self.time
        ):
            self.ex._update_controller_command(0.005)
        timeline.pause.assert_called_once()
        self.assertIn("gaze walking input failed", self.ex._articulation_health_fault)
        for commands in (self.ex._base_command, self.ex._controller_command, self.ex._keyboard_command):
            np.testing.assert_array_equal(commands.numpy(), (0, 0, 0))
        reported = {call.args[0] for call in self.ex._log_physics_step_error.call_args_list}
        self.assertTrue({"gaze walking input", "gaze reset display", "gaze fault display"} <= reported)

    def test_preview_selection_clears_all_existing_velocity_buffers(self) -> None:
        self.ex._gaze_locomotion = self.make_example(preview=True)._gaze_locomotion
        self.ex._base_command[:] = 0.4
        self.ex._controller_command[:] = 0.4
        self.ex._keyboard_command[:] = 0.4
        result = self.activate("turn_left")
        self.assertEqual((result.state, result.last_action), ("preview", "turn_left"))
        for commands in (self.ex._base_command, self.ex._controller_command, self.ex._keyboard_command):
            np.testing.assert_array_equal(commands.numpy(), (0, 0, 0))
        self.ex.g1.robot.get_velocities.assert_not_called()

    def test_measured_target_completion_zeros_smoothed_base_buffer_immediately(self) -> None:
        self.activate()
        self.ex._base_command[:] = torch.tensor([0.2, 0, 0])
        self.ex._get_g1_base_pose_for_arms.return_value = (Gf.Vec3d(0.2, 0, 0.8), 0.0)
        result = self.tick("forward")
        self.assertEqual(result.state, "stopping")
        np.testing.assert_array_equal(self.ex._base_command.numpy(), (0, 0, 0))
        np.testing.assert_array_equal(self.ex._controller_command.numpy(), (0, 0, 0))

    def test_hold_release_uses_fast_existing_filter_without_restarting_dwell(self) -> None:
        self.ex._gaze_locomotion = self.make_example(G1_GAZE_MODE="hold")._gaze_locomotion
        self.activate()
        self.ex._base_command[:] = torch.tensor([0.2, 0, 0])
        result = self.tick(None)
        self.assertEqual(result.state, "idle")
        self.assertFalse(result.stop_immediately)
        np.testing.assert_allclose(self.ex._base_command.numpy(), (0.2, 0, 0))
        np.testing.assert_array_equal(self.ex._controller_command.numpy(), (0, 0, 0))
        self.ex._teleop_input_dt = 0.02
        values = []
        for _ in range(10):
            self.tick(None)
            self.ex._smooth_base_command(self.ex._controller_command, 0.005)
            values.append(float(self.ex._base_command[0]))
        self.assertTrue(all(a > b > 0 for a, b in zip([0.2, *values], values)))
        self.assertLess(values[-1], 0.01)
        for _ in range(4):
            result = self.tick("turn_left")
        self.assertEqual(result.activation_count, 2)
        self.assertEqual(result.activated_action, "turn_left")

    def test_hold_tracking_loss_bypasses_release_filter_immediately(self) -> None:
        self.ex._gaze_locomotion = self.make_example(G1_GAZE_MODE="hold")._gaze_locomotion
        self.activate()
        self.ex._base_command[:] = torch.tensor([0.2, 0, 0])
        self.ex._eye_gaze_tracker.read_control_ray.return_value = None
        result = self.tick("forward")
        self.assertEqual(result.state, "unavailable")
        self.assertTrue(result.stop_immediately)
        for commands in (self.ex._base_command, self.ex._controller_command, self.ex._keyboard_command):
            np.testing.assert_array_equal(commands.numpy(), (0, 0, 0))

    def test_gaze_profile_defaults_to_hold_and_retains_explicit_step(self) -> None:
        with patch.dict(os.environ, {"G1_LOCOMOTION_INPUT": "gaze"}):
            os.environ.pop("G1_GAZE_MODE", None)
            os.environ.pop("G1_GAZE_DWELL_S", None)
            example = HUMANOID.HumanoidExample()
        self.assertEqual(example._gaze_locomotion.config.control_mode, "hold")
        self.assertEqual(example._gaze_locomotion.config.dwell_s, 0)
        self.assertEqual(self.ex._gaze_locomotion.config.control_mode, "step")

    def test_zero_dwell_requests_movement_on_first_valid_button_sample(self) -> None:
        self.ex._gaze_locomotion = self.make_example(G1_GAZE_MODE="hold", G1_GAZE_DWELL_S="0")._gaze_locomotion
        self.tick()
        result = self.tick("forward")
        self.assertEqual((result.state, result.activated_action), ("active", "forward"))
        self.assertGreater(float(self.ex._controller_command[0]), 0)
        self.tick()
        result = self.tick("turn_left")
        self.assertEqual((result.state, result.activated_action, result.activation_count), ("active", "turn_left", 2))

    def test_step_retains_dwell_default_and_hold_accepts_positive_override(self) -> None:
        with patch.dict(os.environ, {"G1_LOCOMOTION_INPUT": "gaze", "G1_GAZE_MODE": "step"}):
            os.environ.pop("G1_GAZE_DWELL_S", None)
            example = HUMANOID.HumanoidExample()
        self.assertEqual(example._gaze_locomotion.config.dwell_s, 0.7)
        self.assertEqual(self.make_example(G1_GAZE_MODE="hold")._gaze_locomotion.config.dwell_s, 0.3)

    def test_hold_speed_defaults_are_faster_and_step_speeds_stay_bounded(self) -> None:
        for mode, speeds in (("hold", (0.4, 0.25, 0.4)), ("step", (0.2, 0.15, 0.25))):
            with self.subTest(mode=mode), patch.dict(os.environ, {"G1_LOCOMOTION_INPUT": "gaze", "G1_GAZE_MODE": mode}):
                for name in ("G1_GAZE_FORWARD_SPEED", "G1_GAZE_BACKWARD_SPEED", "G1_GAZE_YAW_SPEED"):
                    os.environ.pop(name, None)
                config = HUMANOID.HumanoidExample()._gaze_locomotion.config
            self.assertEqual((config.forward_speed, config.backward_speed, config.yaw_speed), speeds)
        explicit = self.make_example(G1_GAZE_MODE="hold")._gaze_locomotion.config
        self.assertEqual((explicit.forward_speed, explicit.backward_speed, explicit.yaw_speed), (0.2, 0.15, 0.25))

    def test_forward_heading_correction_reaches_base_and_eye_loss_clears_it(self) -> None:
        self.ex._gaze_locomotion = self.make_example(G1_GAZE_MODE="hold", G1_GAZE_DWELL_S="0")._gaze_locomotion
        self.tick()
        self.tick("forward")
        self.ex._get_g1_base_pose_for_arms.return_value = (Gf.Vec3d(0.1, 0, 0.8), -0.15)
        for _ in range(20):
            result = self.tick("forward", dt=0.01)
        self.assertEqual(result.velocity[:2], (0.2, 0))
        self.assertGreater(float(self.ex._controller_command[2]), 0)
        self.assertLessEqual(float(self.ex._controller_command[2]), 0.150001)
        status = self.ex.gaze_locomotion_status()["heading"]
        self.assertTrue(status["active"])
        self.ex._base_command[:] = self.ex._controller_command
        self.ex._eye_gaze_tracker.read_control_ray.return_value = None
        self.tick("forward")
        for command in (self.ex._base_command, self.ex._controller_command):
            np.testing.assert_array_equal(command.numpy(), (0, 0, 0))
        self.assertFalse(self.ex.gaze_locomotion_status()["heading"]["active"])

    def test_heading_feedback_can_be_disabled_without_disabling_gaze(self) -> None:
        example = self.make_example(G1_GAZE_HEADING_HOLD="0")
        self.assertTrue(example._gaze_walking_enabled)
        self.assertFalse(example._gaze_locomotion.config.heading_hold)

    def test_active_gaze_movement_keeps_fingers_and_arms_before_final_motor_efforts(self) -> None:
        self.activate()
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
        self.time += 0.1
        with patch.object(HUMANOID.time, "perf_counter", return_value=self.time):
            self.ex.on_physics_step(0.005, None)
        self.assertEqual(order, ["base", "fingers", "arms", "efforts", "camera", "gaze", "recording"])
        self.ex.g1.forward.assert_called_once()
        self.assertTrue(self.ex.g1.forward.call_args.kwargs["defer_body_efforts"])
        self.assertGreater(float(self.ex.g1.forward.call_args.args[1][0]), 0)
        self.assertEqual(self.ex._gaze_locomotion.command.state, "active")
        self.assertIsNone(self.ex._articulation_health_fault)

    def test_pause_and_stop_disarm_held_gaze_and_clear_velocity_buffers(self) -> None:
        timeline = types.SimpleNamespace(TimelineEventType=types.SimpleNamespace(PAUSE=2, STOP=1))
        for event_type in (2, 1):
            with self.subTest(event_type=event_type):
                self.setUp()
                self.activate()
                with patch.object(HUMANOID, "import_module", return_value=timeline):
                    self.ex._on_navigation_timeline_event(types.SimpleNamespace(type=event_type))
                for commands in (self.ex._base_command, self.ex._controller_command, self.ex._keyboard_command):
                    np.testing.assert_array_equal(commands.numpy(), (0, 0, 0))
                self.assertEqual(self.ex._gaze_locomotion.command.state, "awaiting_release")
                if event_type == 1:
                    self.assertFalse(self.ex._physics_ready)
                    self.ex.g1.invalidate_physics_state.assert_called_once()
                    self.ex._physics_ready, self.ex._timeline_stopped = True, False
                for _ in range(5):
                    self.assertEqual(self.tick("forward").state, "awaiting_release")

    def test_continuous_gaze_preserves_optical_finger_and_arm_updates_for_both_hands(self) -> None:
        """Exercise retargeting and IK with supplied sensors, without claiming a physical grasp."""
        self.ex._gaze_locomotion = self.make_example(G1_GAZE_MODE="hold")._gaze_locomotion
        self.activate()
        ex = self.ex
        stage = Usd.Stage.CreateInMemory()
        stage_context = patch.object(USD, "get_context", return_value=types.SimpleNamespace(get_stage=lambda: stage))
        stage_context.start()
        self.addCleanup(stage_context.stop)
        ex._xr_core = object()
        ex._finger_smoothing = ex._arm_rig_smoothing = 1.0
        ex.g1.supports_arm_teleoperation = True
        devices = {}
        hand_positions = {"left": (0.3, 0.2, 1.0), "right": (0.3, -0.2, 1.0)}
        for side, position in hand_positions.items():
            devices[side] = SkeletonDevice(
                skeleton_positions(ex, dict.fromkeys(ex._finger_roles, 0.7)),
                transform=pose((position[0], position[1] - 0.0275, position[2])),
                flags=0xF,
            )
        ex._get_xr_input_device = lambda handle: devices[handle.rsplit("/", 1)[-1]]
        ex._log_hand_tracking_status_once = Mock()
        ex._set_grab_candidate = Mock()
        ex._release_grabbed_object = Mock(side_effect=AssertionError("Gaze movement released a tracked hand"))
        ex._create_grasp_joint = Mock(side_effect=AssertionError("Physical mode created an attachment"))
        # Existing object references are continuity sentinels, not evidence of a
        # successful hold. No contact forces or simulated grasp success are supplied.
        held_references = {"left": "/World/ExistingLeftObject", "right": "/World/ExistingRightObject"}
        ex._grabbed_objects_by_side = dict(held_references)
        ex._contact_reader = types.SimpleNamespace(read=lambda dt: ({"left": [], "right": []}, "ok"))
        ex.g1.has_finger_control = lambda: True
        ex.g1.get_finger_curls = lambda side: dict.fromkeys((*ex._finger_roles, "thumb_yaw"), 0.7)
        ex.g1.get_finger_joint_range = lambda side, role: (0.0, 1.6)
        ex.g1.set_finger_curls = Mock()

        # Supply the tensor/link boundary while retaining production optical
        # pose extraction, arm mapping, IK math, filtering, and contact limiting.
        joint_roles = (*HUMANOID.ARM_IK_JOINT_ORDER, "wrist_roll", "wrist_pitch", "wrist_yaw")
        ex._g1_arm_dofs_configured = True
        ex._g1_arm_dof_indices_by_side = {
            "left": dict(zip(joint_roles, range(7))),
            "right": dict(zip(joint_roles, range(7, 14))),
        }
        ex._g1_arm_joint_limits = {index: (-2.0, 2.0) for index in range(14)}
        ex._g1_arm_joint_defaults = dict.fromkeys(range(14), 0.0)
        jacobians = np.zeros((1, 2, 6, 14))
        for link, offset in ((0, 0), (1, 7)):
            jacobians[0, link, :3, offset : offset + 3] = np.eye(3)
            jacobians[0, link, 3:, offset + 4 : offset + 7] = np.eye(3)
        ex.g1.robot.num_dofs, ex.g1.robot.num_links = 14, 3
        ex.g1.robot.get_dof_positions.return_value = Array(np.zeros((1, 14)))
        ex.g1.robot.get_jacobian_matrices.return_value = Array(jacobians)
        ex._get_arm_link_index = lambda side: 1 if side == "left" else 2
        ex._get_hand_link_prim = lambda side: types.SimpleNamespace(
            get_world_poses=lambda: (Array([hand_positions[side]]), Array([[1.0, 0, 0, 0]]))
        )
        ex._palm_offset_world = lambda side, wrist: Gf.Vec3d(0)
        ex._check_articulation_health = Mock(return_value=True)
        ex._check_gesture_balance = Mock(return_value=True)
        ex._check_gesture_motion = Mock(return_value=True)
        ex._poll_webcam_tracking = Mock()
        ex._update_head_camera_view = Mock()
        ex._collect_all_behavioral_data = Mock()

        baseline_targets = None
        for step in range(8):
            if step == 4:
                for side, device in devices.items():
                    position = hand_positions[side]
                    device.transform = pose((position[0] + 0.08, position[1] - 0.0275, position[2]))
                    device.positions = skeleton_positions(ex, dict.fromkeys(ex._finger_roles, 0.9))
            self.time += 0.01
            with patch.object(HUMANOID.time, "perf_counter", return_value=self.time):
                ex.on_physics_step(0.005, None)
            self.assertIsNone(ex._articulation_health_fault)
            self.assertEqual(ex._gaze_locomotion.command.state, "active")
            self.assertGreater(float(ex._controller_command[0]), 0)
            self.assertEqual(ex._arm_input_sources, {"left": "hand_tracking", "right": "hand_tracking"})
            self.assertEqual(ex._finger_curl_source, {"left": "hand_tracking", "right": "hand_tracking"})
            self.assertEqual(ex._grabbed_objects_by_side, held_references)
            args, kwargs = ex.g1.robot.set_dof_position_targets.call_args
            targets = dict(zip(kwargs["dof_indices"], args[0]))
            self.assertEqual(set(targets), set(range(14)))
            self.assertTrue(all(math.isfinite(value) for value in targets.values()))
            if step == 3:
                baseline_targets = targets.copy()
        for index in (0, 7):
            self.assertGreater(targets[index], baseline_targets[index])
        for side in ("left", "right"):
            self.assertGreater(ex._applied_finger_curls[side]["index"], 0.85)
            self.assertTrue(ex._is_hand_closed(side, devices[side]))
            self.assertFalse(ex._physical_grasp_results[side].contact_supported)
        self.assertEqual(ex.g1.set_finger_curls.call_count, 16)
        ex._get_xr_gesture_value.assert_not_called()
        ex._release_grabbed_object.assert_not_called()
        ex._create_grasp_joint.assert_not_called()

    def test_readiness_loss_and_status_snapshot_do_not_reuse_prior_motion(self) -> None:
        self.activate()
        self.ex._physics_ready = False
        result = self.tick("forward")
        self.assertEqual((result.state, result.velocity), ("unavailable", (0, 0, 0)))
        status = self.ex.gaze_locomotion_status()
        self.assertEqual(status["last_action"], "forward")
        self.assertEqual(status["activation_count"], 1)
        self.assertEqual(status["requested_velocity"], (0, 0, 0))
        self.assertFalse(status["preview"])


class TestStrictGazeControlRay(unittest.TestCase):
    def setUp(self) -> None:
        self.devices = {}
        self.profile = types.SimpleNamespace(get_name=Mock(return_value="VR"), is_enabled=Mock(return_value=True))
        self.xr = types.SimpleNamespace(
            get_current_profile=Mock(return_value=self.profile),
            is_xr_enabled=Mock(return_value=True),
            is_xr_display_enabled=Mock(return_value=True),
            is_xr_viewport_enabled=Mock(return_value=True),
            get_input_device=Mock(side_effect=lambda handle: self.devices.get(handle)),
            get_all_input_devices=lambda: [],
        )
        self.tracker = GAZE.EyeGazeTracker(self.xr, draw_ray=False, highlight_gazed_boxes=False)
        self.tracker._log_source = Mock()

    def test_profile_display_and_viewport_must_all_be_ready_before_reading_eyes(self) -> None:
        self.devices["/user/eye/unified"] = Device(pose((0, 0, 1.5)), flags=0xF)
        self.assertTrue(self.tracker.control_session_ready())
        for owner, name, value in (
            (self.xr, "get_current_profile", None),
            (self.profile, "get_name", ""),
            (self.profile, "is_enabled", False),
            (self.xr, "is_xr_enabled", False),
            (self.xr, "is_xr_display_enabled", False),
            (self.xr, "is_xr_viewport_enabled", False),
        ):
            with self.subTest(name=name), patch.object(owner, name, return_value=value):
                self.xr.get_input_device.reset_mock()
                self.assertFalse(self.tracker.control_session_ready())
                self.assertIsNone(self.tracker.read_control_ray())
                self.xr.get_input_device.assert_not_called()
        self.xr.get_current_profile.side_effect = RuntimeError("Profile is shutting down")
        self.assertFalse(self.tracker.control_session_ready())
        self.xr.get_current_profile.side_effect = None
        del self.xr.is_xr_display_enabled
        self.assertFalse(self.tracker.control_session_ready())

    def test_all_valid_and_tracked_bits_accept_current_world_eye_ray(self) -> None:
        device = Device(pose((2, 3, 1.5)), flags=0xF)
        self.devices["/user/eye/unified"] = device
        origin, direction = self.tracker.read_control_ray()
        self.assertEqual(tuple(origin), (2, 3, 1.5))
        self.assertEqual(tuple(direction), (0, 0, -1))
        device.matrix = pose((4, 5, 1.6))
        self.assertEqual(tuple(self.tracker.read_control_ray()[0]), (4, 5, 1.6))

    def test_valid_but_untracked_pose_is_visualizable_but_not_control_input(self) -> None:
        self.devices["/user/eye/unified"] = Device(pose((0, 0, 1.5)), flags=3)
        self.assertIsNotNone(self.tracker._read_gaze_ray())
        self.assertIsNone(self.tracker.read_control_ray())
        for flags in (0, 1, 2, 7, 11, 13, 14):
            with self.subTest(flags=flags):
                self.devices["/user/eye/unified"].flags = flags
                self.assertIsNone(self.tracker.read_control_ray())

    def test_head_forward_and_stereo_view_poses_never_substitute_for_eyes(self) -> None:
        for handle in ("/user/head", "/user/eye/left", "/user/eye/right"):
            self.devices[handle] = Device(pose((0, 0, 1.5)), flags=0xF)
        self.assertIsNotNone(self.tracker._read_gaze_ray())
        self.assertEqual(self.tracker.gaze_source, "hmd_forward")
        self.xr.get_input_device.reset_mock()
        self.assertIsNone(self.tracker.read_control_ray())
        self.assertTrue(
            all(call.args[0] in self.tracker.UNIFIED_DEVICES for call in self.xr.get_input_device.call_args_list)
        )

    def test_matrix_only_and_physical_room_poses_fail_strict_control(self) -> None:
        for device in (
            types.SimpleNamespace(get_virtual_world_pose=lambda name: pose((0, 0, 1.5))),
            types.SimpleNamespace(
                get_pose_desc=lambda name: types.SimpleNamespace(validity_flags=0xF, pose_matrix=pose((0, 1.5, 0)))
            ),
        ):
            with self.subTest(device=device):
                self.devices["/user/eye/unified"] = device
                self.assertIsNone(self.tracker.read_control_ray())

    def test_disabled_xr_or_nonfinite_eye_matrix_yields_no_control_ray(self) -> None:
        self.devices["/user/eye/unified"] = Device(pose((0, 0, 1.5)), flags=0xF)
        self.xr.is_xr_enabled.return_value = False
        self.assertIsNone(self.tracker.read_control_ray())
        self.xr.get_input_device.assert_not_called()
        self.xr.is_xr_enabled.return_value = True
        self.devices["/user/eye/unified"].matrix = pose((math.nan, 0, 1.5))
        self.assertIsNone(self.tracker.read_control_ray())


if __name__ == "__main__":
    unittest.main()
