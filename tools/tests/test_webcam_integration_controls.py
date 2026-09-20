# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional webcam input must preserve Quest authority and physical control limits."""

from __future__ import annotations

import dataclasses
import json
import math
import time
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from humanoid_test_support import HUMANOID, WEBCAM, Array, pose
from pxr import Gf
from test_arm_ik_controls import example_with_jacobian, independent_hand_tasks
from test_finger_controls import SkeletonDevice, skeleton_positions
from test_webcam_tracking_controls import packet


def webcam_frame(*, session="camera-a", request=0, age=0.0, hands=None):
    now = time.perf_counter_ns()
    data = packet(session=session, captured=now - int(age * 1e9))
    data["calibration_request"] = request
    if hands is not None:
        data["hand_world"] = hands
        data["hand_confidence"] = {side: 0.95 for side in hands}
    return WEBCAM.decode_frame(json.dumps(data).encode(), now_ns=now)


def calibrated_example():
    ex = HUMANOID.HumanoidExample()
    ex.enable_webcam_tracking()
    ex._webcam_frame = webcam_frame()
    ex._webcam_sender_session = ex._webcam_frame.sender_session
    targets = {"left": (0.3, 0.2, 0.1), "right": (0.3, -0.2, 0.1)}
    ex._webcam_calibration.calibrate(ex._webcam_frame, targets)
    ex._webcam_arm_targets = ex._webcam_calibration.map(ex._webcam_frame)
    return ex


def arm_example():
    ex = calibrated_example()
    robot = Mock()
    robot.get_dof_positions.return_value = Array([[0.2, 0.3, 0.4, 0.5]])
    ex.g1 = types.SimpleNamespace(robot=robot, set_finger_curls=Mock())
    ex._xr_core = object()
    ex._configure_g1_arm_dofs = Mock()
    ex._g1_arm_dofs_configured = True
    ex._g1_arm_dof_indices_by_side = {"left": {"elbow": 0, "wrist_yaw": 1}, "right": {"elbow": 2, "wrist_yaw": 3}}
    ex._g1_arm_joint_limits = {index: (-2.0, 2.0) for index in range(4)}
    ex._get_g1_base_pose_for_arms = Mock(return_value=(Gf.Vec3d(0, 0, 0.8), 0.0))
    ex._get_xr_input_device = lambda handle: handle.rsplit("/", 1)[-1]
    ex.optical = {}
    ex.controller = {}
    ex.kinds = {"left": "none", "right": "none"}
    ex._get_optical_arm_pose = lambda side, device: ex.optical.get(side)
    ex._get_controller_arm_pose = lambda side, device: ex.controller.get(side)
    ex._get_hand_input_kind = lambda device: ex.kinds.get(device, "none")
    ex._compute_arm_target_body_position = Mock(return_value=Gf.Vec3d(0.4, 0.1, 0.2))
    ex._compute_arm_target_orientation = Mock(return_value=None)
    ex._get_webcam_wrist_hold_rotation = Mock(return_value=None)
    ex._log_hand_tracking_status_once = Mock()
    ex._set_arm_rig_target_visible = Mock()
    ex._set_grab_candidate = Mock()
    ex._release_grabbed_object = Mock()
    ex._update_arm_rig_target = Mock()
    ex._update_grabbed_object = Mock()
    ex._is_hand_closed = Mock(return_value=False)
    ex._compute_arm_targets_from_body_position = Mock(return_value={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4})
    ex._smoothed_arm_rig_targets = {"left": Gf.Vec3d(0.3, 0.2, 0.1), "right": Gf.Vec3d(0.3, -0.2, 0.1)}
    ex._limit_arm_targets_at_contacts = Mock(side_effect=lambda targets, snapshot: targets)
    return ex


def use_real_quest_readers(ex, flags=2):
    """Exercise production optical validity/mapping instead of synthetic source labels."""
    devices = {
        side: SkeletonDevice(skeleton_positions(ex), transform=pose((0.3, sign * 0.2, 1.05)), flags=flags)
        for side, sign in (("left", 1), ("right", -1))
    }
    ex._get_xr_input_device = lambda handle: devices[handle.rsplit("/", 1)[-1]]
    for method in ("_get_hand_input_kind", "_get_optical_arm_pose", "_compute_arm_target_body_position"):
        setattr(ex, method, getattr(HUMANOID.HumanoidExample, method).__get__(ex))
    return devices


def spread_hand_landmarks(bends=None):
    """A flat, anatomically spread 21-point hand with controllable joint flexion."""
    bends = bends or {}
    points = [Gf.Vec3d(0)]
    thumb_base = Gf.Vec3d(0.035, 0.015, 0)
    thumb_direction = Gf.Vec3d(math.sqrt(0.5), math.sqrt(0.5), 0)
    points.extend(thumb_base + thumb_direction * (0.025 * index) for index in range(4))
    for role, lateral, spread in (
        ("index", 0.025, 25.0),
        ("middle", 0.0, 0.0),
        ("ring", -0.022, -12.0),
        ("little", -0.04, -30.0),
    ):
        direction = Gf.Vec3d(math.sin(math.radians(spread)), math.cos(math.radians(spread)), 0)
        current = Gf.Vec3d(lateral, 0.055, 0)
        points.append(current)
        total = 0.0
        for flexion in bends.get(role, (0.0, 0.0, 0.0)):
            total += math.radians(flexion)
            current = current + 0.03 * (direction * math.cos(total) + Gf.Vec3d(0, 0, -1) * math.sin(total))
            points.append(Gf.Vec3d(current))
    return points


class TestWebcamLandmarkFlexion(unittest.TestCase):
    def setUp(self):
        self.ex = HUMANOID.HumanoidExample()

    def curls(self, landmarks):
        return self.ex._get_hand_tracking_finger_curls(HUMANOID._WebcamHandDevice(landmarks))

    def test_flat_spread_fingers_are_open_without_false_mcp_flexion(self):
        result = self.curls(spread_hand_landmarks())
        self.assertEqual(set(result), {*self.ex._finger_roles, "thumb_yaw"})
        for value in result.values():
            self.assertAlmostEqual(value, 0.0, places=6)

    def test_mcp_only_flexion_is_preserved_at_rotations_and_hand_sizes(self):
        rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, -3), 137))
        rotation.SetTranslateOnly(Gf.Vec3d(2, -3, 4))
        for role in ("index", "middle", "ring", "little"):
            for degrees in (30.0, 60.0, 90.0, 120.0):
                for scale in (0.6, 1.0, 1.4):
                    with self.subTest(role=role, degrees=degrees, scale=scale):
                        points = spread_hand_landmarks({role: (degrees, 0.0, 0.0)})
                        points = [rotation.Transform(point * scale) for point in points]
                        result = self.curls(points)
                        self.assertAlmostEqual(result[role], degrees / 150.0, places=6)
                        for other in ("index", "middle", "ring", "little"):
                            if other != role:
                                self.assertAlmostEqual(result[other], 0.0, places=6)

    def test_pip_and_dip_bends_remain_independent_of_spread(self):
        points = spread_hand_landmarks({"index": (0.0, 60.0, 30.0), "little": (30.0, 45.0, 45.0)})
        result = self.curls(points)
        self.assertAlmostEqual(result["index"], 0.6, places=6)
        self.assertAlmostEqual(result["little"], 0.8, places=6)
        self.assertAlmostEqual(result["ring"], 0.0, places=6)
        self.assertAlmostEqual(result["thumb_yaw"], 0.0, places=6)

    def test_degenerate_palm_cannot_supply_webcam_fingers(self):
        ex = calibrated_example()
        points = [list(point) for point in spread_hand_landmarks()]
        points[9] = points[0]
        ex._webcam_frame = webcam_frame(hands={"right": points})
        self.assertIsNone(ex._get_webcam_finger_curls("right"))


class TestWebcamLifecycle(unittest.TestCase):
    def polling_example(self, frame):
        ex = HUMANOID.HumanoidExample()
        ex.enable_webcam_tracking()
        ex._webcam_receiver = types.SimpleNamespace(
            poll=Mock(return_value=frame), close=Mock(), last_error=None, is_open=True
        )
        ex._get_g1_base_pose_for_arms = Mock(return_value=(Gf.Vec3d(0), 0.0))
        ex._get_xr_input_device = lambda handle: handle
        ex._get_optical_arm_pose = lambda side, device: pose((0.3, 0.2 if side == "left" else -0.2, 0.1))
        ex._get_webcam_robot_arm_lengths = lambda: {"left": 0.37, "right": 0.37}
        return ex

    def test_disabled_and_constructor_do_not_bind_or_poll(self):
        with patch.dict(HUMANOID.os.environ, {"G1_WEBCAM_ENABLED": "0"}), patch.object(
            HUMANOID, "WebcamReceiver"
        ) as receiver:
            ex = HUMANOID.HumanoidExample()
            ex._poll_webcam_tracking()
            self.assertFalse(ex.webcam_tracking_status()["enabled"])
            receiver.assert_not_called()
            ex.enable_webcam_tracking()
            receiver.assert_not_called()

    def test_environment_enables_reception_without_binding(self):
        with patch.dict(HUMANOID.os.environ, {"G1_WEBCAM_ENABLED": "1"}), patch.object(
            HUMANOID, "WebcamReceiver"
        ) as receiver:
            ex = HUMANOID.HumanoidExample()
            self.assertTrue(ex.webcam_tracking_status()["enabled"])
            receiver.assert_not_called()

    def test_bind_failure_isolated_and_retry_requires_enable(self):
        ex = HUMANOID.HumanoidExample()
        ex.enable_webcam_tracking()
        with patch.object(HUMANOID, "WebcamReceiver", side_effect=OSError("port occupied")) as receiver:
            ex._poll_webcam_tracking()
            ex._poll_webcam_tracking()
            self.assertEqual(receiver.call_count, 1)
            self.assertIn("port occupied", ex.webcam_tracking_status()["receiver_error"])
            self.assertFalse(ex.webcam_tracking_status()["calibrated"])

    def test_status_reports_receive_socket_closed_without_outer_poll_exception(self):
        ex = self.polling_example(webcam_frame())
        self.assertTrue(ex.webcam_tracking_status()["listening"])
        ex._webcam_receiver.is_open = False
        ex._webcam_receiver.last_error = "socket disconnected"
        status = ex.webcam_tracking_status()
        self.assertFalse(status["listening"])
        self.assertIn("socket disconnected", status["receiver_error"])

    def test_no_automatic_calibration_until_explicit_request(self):
        ex = self.polling_example(webcam_frame())
        ex._poll_webcam_tracking()
        self.assertFalse(ex._webcam_calibration.is_calibrated)
        self.assertFalse(ex._webcam_arm_targets)
        ex.request_webcam_calibration()
        ex._poll_webcam_tracking()
        self.assertTrue(ex._webcam_calibration.is_calibrated)
        self.assertEqual(set(ex._webcam_arm_targets), {"left", "right"})

    def test_request_before_first_frame_survives_initial_session(self):
        ex = self.polling_example(webcam_frame())
        ex.request_webcam_calibration()
        ex._poll_webcam_tracking()
        self.assertTrue(ex._webcam_calibration.is_calibrated)

    def test_calibration_waits_for_both_optical_hands_and_fresh_sample(self):
        ex = self.polling_example(webcam_frame(request=1))
        optical = ex._get_optical_arm_pose
        ex._get_optical_arm_pose = lambda side, device: optical(side, device) if side == "left" else None
        ex._poll_webcam_tracking()
        self.assertTrue(ex._webcam_calibration_pending)
        self.assertFalse(ex._webcam_calibration.is_calibrated)
        ex._get_optical_arm_pose = optical
        ex._webcam_receiver.poll.return_value = webcam_frame(request=1, age=0.15)
        ex._poll_webcam_tracking()
        self.assertFalse(ex._webcam_calibration.is_calibrated)
        ex._webcam_receiver.poll.return_value = webcam_frame(request=1)
        ex._poll_webcam_tracking()
        self.assertTrue(ex._webcam_calibration.is_calibrated)
        self.assertFalse(ex._webcam_calibration_pending)

    def test_calibration_accepts_valid_quest_positions_after_tracked_flags_clear(self):
        ex = self.polling_example(webcam_frame(request=1))
        devices = use_real_quest_readers(ex, flags=2 | 8)
        ex._poll_webcam_tracking()
        self.assertTrue(ex._webcam_calibration.is_calibrated)
        for device in devices.values():
            device.flags = 2
        ex._webcam_receiver.poll.return_value = webcam_frame(request=2)
        ex._poll_webcam_tracking()
        self.assertTrue(ex._webcam_calibration.is_calibrated)
        self.assertFalse(ex._webcam_calibration_pending)

    def test_reset_does_not_replay_previous_sender_calibration_request(self):
        ex = self.polling_example(webcam_frame(request=1))
        ex._poll_webcam_tracking()
        self.assertTrue(ex._webcam_calibration.is_calibrated)
        ex._reset_webcam_tracking()
        ex._webcam_receiver.poll.return_value = webcam_frame(request=1)
        ex._poll_webcam_tracking()
        self.assertFalse(ex._webcam_calibration.is_calibrated)
        ex._webcam_receiver.poll.return_value = webcam_frame(request=2)
        ex._poll_webcam_tracking()
        self.assertTrue(ex._webcam_calibration.is_calibrated)

    def test_sender_restart_clears_then_new_explicit_request_realigns(self):
        ex = self.polling_example(webcam_frame(request=1))
        ex._poll_webcam_tracking()
        ex._webcam_receiver.poll.return_value = webcam_frame(session="new-camera", request=0)
        ex._poll_webcam_tracking()
        self.assertFalse(ex._webcam_calibration.is_calibrated)
        self.assertFalse(ex._webcam_arm_targets)
        ex._webcam_receiver.poll.return_value = webcam_frame(session="new-camera", request=1)
        ex._poll_webcam_tracking()
        self.assertTrue(ex._webcam_calibration.is_calibrated)

    def test_recenter_invalidates_without_replaying_C_and_disable_closes(self):
        ex = self.polling_example(webcam_frame(request=1))
        ex._poll_webcam_tracking()
        ex._update_head_camera_view = Mock()
        ex._log_xr_camera_state = Mock()
        ex._request_xr_recenter()
        ex._poll_webcam_tracking()
        self.assertFalse(ex._webcam_calibration.is_calibrated)
        receiver = ex._webcam_receiver
        ex.disable_webcam_tracking()
        receiver.close.assert_called_once()
        self.assertIsNone(ex._webcam_receiver)
        self.assertFalse(ex._webcam_enabled)

    def test_pending_request_expires_and_does_not_arm_when_hands_return(self):
        ex = self.polling_example(webcam_frame())
        ex.request_webcam_calibration()
        ex._webcam_calibration_deadline_ns = time.perf_counter_ns() - 1
        ex._poll_webcam_tracking()
        self.assertFalse(ex._webcam_calibration_pending)
        self.assertFalse(ex._webcam_calibration.is_calibrated)
        # Initial receiver discovery can replace the expiry message with setup
        # guidance, but neither path can preserve the expired request.
        ex._poll_webcam_tracking()
        self.assertFalse(ex._webcam_calibration.is_calibrated)

    def test_sdk_failure_in_optional_calibration_cannot_escape_poll(self):
        ex = self.polling_example(webcam_frame(request=1))
        ex._get_optical_arm_pose = Mock(side_effect=AttributeError("device disappeared"))
        ex._poll_webcam_tracking()
        self.assertFalse(ex._webcam_calibration.is_calibrated)
        self.assertIn("device disappeared", ex._webcam_receiver_error)


class TestWebcamArmAuthority(unittest.TestCase):
    def test_runtime_with_valid_only_flags_retains_quest_priority(self):
        ex = arm_example()
        use_real_quest_readers(ex, flags=2)
        ex._get_webcam_arm_target = Mock(side_effect=AssertionError("Valid Quest must not consult webcam authority"))
        ex._update_g1_arms_from_hand_tracking()
        self.assertEqual(ex._arm_input_sources, {"left": "hand_tracking", "right": "hand_tracking"})

    def test_valid_quest_remains_primary_after_previously_tracked_flag_clears(self):
        ex = arm_example()
        devices = use_real_quest_readers(ex, flags=2 | 8)
        ex._update_g1_arms_from_hand_tracking()
        self.assertEqual(ex._arm_input_sources["right"], "hand_tracking")
        devices["right"].flags = 2
        ex._update_g1_arms_from_hand_tracking()
        self.assertEqual(ex._arm_input_sources["right"], "hand_tracking")

    def test_stale_webcam_never_rejects_valid_quest_even_when_untracked(self):
        ex = arm_example()
        use_real_quest_readers(ex, flags=2)
        ex._webcam_frame = dataclasses.replace(
            ex._webcam_frame, capture_monotonic_ns=time.perf_counter_ns() - 500_000_000
        )
        ex._update_g1_arms_from_hand_tracking()
        self.assertEqual(ex._arm_input_sources["right"], "hand_tracking")

    def test_lowered_quest_hands_keep_moving_despite_webcam_loss_or_low_confidence(self):
        for camera_state in ("fresh", "missing", "stale", "low_body_confidence", "receiver_failure"):
            with self.subTest(camera_state=camera_state):
                ex = arm_example()
                devices = use_real_quest_readers(ex, flags=2)
                ex._update_g1_arms_from_hand_tracking()
                before = {
                    entry.args[0]: entry.args[1][2]
                    for entry in ex._compute_arm_targets_from_body_position.call_args_list
                }
                if camera_state in ("missing", "receiver_failure"):
                    ex._webcam_frame = None
                    ex._webcam_arm_targets.clear()
                    if camera_state == "receiver_failure":
                        ex._webcam_receiver_error = "camera disconnected"
                elif camera_state == "stale":
                    ex._webcam_frame = dataclasses.replace(
                        ex._webcam_frame, capture_monotonic_ns=time.perf_counter_ns() - 500_000_000
                    )
                elif camera_state == "low_body_confidence":
                    landmarks = [list(point) for point in ex._webcam_frame.pose_world]
                    for index in (11, 12, 23, 24):
                        landmarks[index][3:] = [0.0, 0.0]
                    ex._webcam_frame = dataclasses.replace(
                        ex._webcam_frame, pose_world=tuple(tuple(point) for point in landmarks)
                    )
                    ex._webcam_arm_targets = ex._webcam_calibration.map(ex._webcam_frame)
                    self.assertFalse(ex._webcam_arm_targets)
                for side, device in devices.items():
                    device.transform = pose((0.3, 0.2 if side == "left" else -0.2, 0.75))
                ex._compute_arm_targets_from_body_position.reset_mock()
                ex._update_g1_arms_from_hand_tracking()
                after = {
                    entry.args[0]: entry.args[1][2]
                    for entry in ex._compute_arm_targets_from_body_position.call_args_list
                }
                self.assertEqual(ex._arm_input_sources, {"left": "hand_tracking", "right": "hand_tracking"})
                self.assertFalse(ex._inactive_arm_hold_targets)
                for side in ("left", "right"):
                    self.assertLess(after[side], before[side] - 0.05)

    def test_actual_quest_position_loss_falls_back_only_for_that_hand(self):
        ex = arm_example()
        devices = use_real_quest_readers(ex, flags=2)
        ex._update_g1_arms_from_hand_tracking()
        devices["right"].invalid.add("wrist")
        ex._update_g1_arms_from_hand_tracking()
        self.assertEqual(ex._arm_input_sources, {"left": "hand_tracking", "right": "webcam_body"})
        devices["right"].invalid.clear()
        ex._update_g1_arms_from_hand_tracking()
        self.assertEqual(ex._arm_input_sources, {"left": "hand_tracking", "right": "hand_tracking"})

    def test_quest_optical_wins_per_hand_and_other_arm_falls_back(self):
        ex = arm_example()
        ex.optical["left"] = pose()
        ex.kinds["left"] = "hand_tracking"
        ex._update_g1_arms_from_hand_tracking()
        self.assertEqual(ex._arm_input_sources, {"left": "hand_tracking", "right": "webcam_body"})
        ex._limit_arm_targets_at_contacts.assert_called_once()
        self.assertFalse(ex._inactive_arm_hold_targets)

    def test_controller_clutch_release_holds_instead_of_webcam_takeover(self):
        ex = arm_example()
        ex.kinds["right"] = "controller"
        ex._update_g1_arms_from_hand_tracking()
        self.assertNotIn("right", ex._arm_input_sources)
        self.assertIn("right", ex._inactive_arm_hold_targets)
        ex.controller["right"] = pose()
        ex._update_g1_arms_from_hand_tracking()
        self.assertEqual(ex._arm_input_sources["right"], "controller")

    def test_fresh_calibrated_webcam_works_without_xr_and_stale_holds(self):
        ex = arm_example()
        ex._xr_core = None
        ex._update_g1_arms_from_hand_tracking()
        self.assertEqual(set(ex._arm_input_sources.values()), {"webcam_body"})
        ex._webcam_frame = dataclasses.replace(
            ex._webcam_frame, capture_monotonic_ns=time.perf_counter_ns() - 500_000_000
        )
        ex._update_g1_arms_from_hand_tracking()
        self.assertFalse(ex._arm_input_sources)
        self.assertEqual(set(ex._inactive_arm_hold_targets), {"left", "right"})

    def test_uncalibrated_stream_cannot_move_either_arm(self):
        ex = arm_example()
        ex._webcam_calibration.clear()
        ex._update_g1_arms_from_hand_tracking()
        ex._compute_arm_targets_from_body_position.assert_not_called()
        self.assertEqual(set(ex._inactive_arm_hold_targets), {"left", "right"})

    def test_source_transition_preserves_joint_filter_and_slews_palm(self):
        ex = arm_example()
        ex._teleop_input_dt = 0.01
        ex._arm_input_sources["right"] = "hand_tracking"
        previous = Gf.Vec3d(0.15, -0.2, 0.1)
        ex._smoothed_arm_rig_targets["right"] = previous
        ex._smoothed_arm_targets = {2: 0.4, 3: 0.5}
        ex._update_g1_arms_from_hand_tracking()
        requested = [
            entry.args[1]
            for entry in ex._compute_arm_targets_from_body_position.call_args_list
            if entry.args[0] == "right"
        ][0]
        self.assertLessEqual((requested - previous).GetLength(), 0.008 + 1e-9)
        self.assertIn("right", ex._webcam_handover_targets)
        ex.optical["right"] = pose()
        ex.kinds["right"] = "hand_tracking"
        ex._update_g1_arms_from_hand_tracking()
        self.assertEqual(ex._arm_input_sources["right"], "hand_tracking")
        self.assertIn("right", ex._webcam_handover_targets)

    def test_disabled_mode_retains_original_input_hold(self):
        ex = arm_example()
        ex.disable_webcam_tracking()
        ex._xr_core = None
        ex._update_g1_arms_from_hand_tracking()
        self.assertFalse(ex._arm_input_sources)
        ex._compute_arm_targets_from_body_position.assert_not_called()


class TestWebcamFingerAuthority(unittest.TestCase):
    def finger_example(self):
        ex = calibrated_example()
        ex._xr_core = object()
        ex._get_xr_input_device = lambda handle: None
        ex.g1 = types.SimpleNamespace(
            has_finger_control=lambda: True,
            set_finger_curls=Mock(),
            get_finger_curls=lambda side: {role: 0.0 for role in (*ex._finger_roles, "thumb_yaw")},
            get_finger_joint_range=lambda side, role: (0.0, 1.0),
        )
        ex._contact_reader = types.SimpleNamespace(read=lambda dt: ({"left": [], "right": []}, "ok"))
        return ex

    def test_optical_then_controller_then_webcam_priority(self):
        ex = self.finger_example()
        curls = {role: 0.6 for role in (*ex._finger_roles, "thumb_yaw")}
        ex._get_hand_tracking_finger_curls = Mock(return_value=curls)
        ex._get_controller_finger_curls = Mock(return_value=curls)
        ex._get_webcam_finger_curls = Mock(return_value=curls)
        ex._update_g1_fingers()
        self.assertEqual(set(ex._finger_curl_source.values()), {"hand_tracking"})
        ex._get_controller_finger_curls.assert_not_called()
        ex._get_webcam_finger_curls.assert_not_called()
        ex._get_hand_tracking_finger_curls.return_value = None
        ex._update_g1_fingers()
        self.assertEqual(set(ex._finger_curl_source.values()), {"controller"})
        ex._get_webcam_finger_curls.assert_not_called()
        ex._get_controller_finger_curls.return_value = None
        ex._update_g1_fingers()
        self.assertEqual(set(ex._finger_curl_source.values()), {"webcam_hand"})

    def test_valid_quest_fingers_remain_primary_after_tracked_flags_clear(self):
        ex = self.finger_example()
        devices = use_real_quest_readers(ex, flags=2 | 8)
        ex._get_webcam_finger_curls = Mock(return_value={role: 0.9 for role in (*ex._finger_roles, "thumb_yaw")})
        ex._update_g1_fingers()
        self.assertEqual(set(ex._finger_curl_source.values()), {"hand_tracking"})
        for camera_state in ("fresh", "missing", "stale"):
            with self.subTest(camera_state=camera_state):
                if camera_state == "fresh":
                    ex._webcam_frame = webcam_frame()
                elif camera_state == "missing":
                    ex._webcam_frame = None
                else:
                    ex._webcam_frame = dataclasses.replace(
                        webcam_frame(), capture_monotonic_ns=time.perf_counter_ns() - 500_000_000
                    )
                for device in devices.values():
                    device.flags = 2
                    device.positions = skeleton_positions(ex, {"index": 0.7})
                for _ in range(12):
                    ex._update_g1_fingers()
                self.assertEqual(set(ex._finger_curl_source.values()), {"hand_tracking"})
                self.assertGreater(ex._latest_finger_curls["right"]["index"], 0.65)
                self.assertLess(ex._latest_finger_curls["right"]["little"], 0.01)
        ex._get_webcam_finger_curls.assert_not_called()

    def test_finger_webcam_fallback_is_independent_per_hand_after_quest_loss(self):
        ex = self.finger_example()
        devices = use_real_quest_readers(ex, flags=2)
        devices["right"].flags = 0
        ex._get_webcam_finger_curls = Mock(return_value={role: 0.6 for role in (*ex._finger_roles, "thumb_yaw")})
        ex._update_g1_fingers()
        self.assertEqual(ex._finger_curl_source, {"left": "hand_tracking", "right": "webcam_hand"})
        ex._get_webcam_finger_curls.assert_called_once_with("right")

    def test_webcam_open_gesture_rearms_latch_and_loss_relaxes(self):
        ex = self.finger_example()
        ex._get_webcam_finger_curls = Mock(return_value={role: 0.1 for role in (*ex._finger_roles, "thumb_yaw")})
        ex._grab_requires_release = {"left": True, "right": True}
        ex._update_g1_fingers()
        self.assertFalse(any(ex._grab_requires_release.values()))
        ex._get_webcam_finger_curls.return_value = {role: 0.8 for role in (*ex._finger_roles, "thumb_yaw")}
        for _ in range(12):
            ex._update_g1_fingers()
        self.assertTrue(ex._is_hand_closed("right", None))
        ex._get_webcam_finger_curls.return_value = None
        for _ in range(20):
            ex._update_g1_fingers()
        self.assertEqual(ex._finger_curl_source["right"], "none")
        self.assertLess(max(ex._applied_finger_curls["right"].values()), 0.01)

    def test_detailed_hand_adapter_requires_confidence_freshness_and_calibration(self):
        ex = self.finger_example()
        positions = skeleton_positions(ex, {"index": 0.8, "thumb": 0.4})
        landmarks = [tuple(positions["wrist"])]
        for role in ("thumb", "index", "middle", "ring", "little"):
            chain = ex._finger_curl_joint_chains[role]
            landmarks.extend(tuple(positions[name]) for name in (chain if role == "thumb" else chain[1:]))
        ex._webcam_frame = webcam_frame(hands={"right": landmarks})
        curls = ex._get_webcam_finger_curls("right")
        self.assertGreater(curls["index"], 0.7)
        self.assertLess(curls["middle"], 0.1)
        self.assertIsNone(ex._get_webcam_finger_curls("left"))
        ex._webcam_frame = dataclasses.replace(ex._webcam_frame, hand_confidence={"right": 0.5})
        self.assertIsNone(ex._get_webcam_finger_curls("right"))
        ex._webcam_frame = webcam_frame(hands={"right": landmarks})
        ex._webcam_calibration.clear()
        self.assertIsNone(ex._get_webcam_finger_curls("right"))

    def test_deactivating_arm_preserves_valid_webcam_finger_contacts(self):
        ex = self.finger_example()
        ex._finger_curl_source["right"] = "webcam_hand"
        ex._set_arm_rig_target_visible = Mock()
        ex._set_grab_candidate = Mock()
        ex._release_grabbed_object = Mock()
        ex._deactivate_hand("right")
        ex._release_grabbed_object.assert_not_called()


class TestWebcamElbowPriority(unittest.TestCase):
    def test_webcam_elbow_never_changes_quest_or_controller_arm_solution(self):
        ex = example_with_jacobian(independent_hand_tasks())
        baseline = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), pose(yaw=30))
        ex._get_webcam_arm_target = Mock(return_value=WEBCAM.WebcamArmTarget((0.3, 0.2, -0.2), (0.5, 0.2, -0.4), 0.95))
        ex._get_webcam_robot_link = Mock(side_effect=AssertionError("Webcam cannot guide a primary Quest arm"))
        for source in ("hand_tracking", "controller"):
            with self.subTest(source=source):
                ex._arm_input_sources["right"] = source
                self.assertEqual(ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), pose(yaw=30)), baseline)
        ex._get_webcam_arm_target.assert_not_called()
        ex._get_webcam_robot_link.assert_not_called()

    def test_optional_elbow_tensor_failure_preserves_full_palm_solve(self):
        ex = example_with_jacobian(independent_hand_tasks())
        baseline = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), pose(yaw=30))
        ex._arm_input_sources["right"] = "webcam_body"
        ex._get_webcam_arm_target = lambda side: WEBCAM.WebcamArmTarget((0.1, 0, 0), (0.15, 0, 0), 0.95)
        ex._get_webcam_robot_link = Mock(side_effect=RuntimeError("elbow tensor lost"))
        self.assertEqual(ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), pose(yaw=30)), baseline)

    def test_elbow_guidance_only_changes_redundant_joint(self):
        ex = example_with_jacobian(independent_hand_tasks())
        base = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), pose(yaw=30))
        hand = independent_hand_tasks()
        elbow = np.zeros_like(hand)
        elbow[0, 0] = 1.0
        elbow[0, 3] = 1.0
        ex.g1.robot.num_links = 3
        ex.g1.robot.get_jacobian_matrices = lambda: Array(np.stack((hand, elbow)).reshape(1, 2, 6, 7))
        ex._arm_input_sources["right"] = "webcam_body"
        ex._get_webcam_arm_target = lambda side: WEBCAM.WebcamArmTarget((0.1, 0, 0), (0.15, 0, 0), 0.95)
        elbow_prim = types.SimpleNamespace(get_world_poses=lambda: (Array([[0.0, 0, 0]]), Array([[1.0, 0, 0, 0]])))
        ex._get_webcam_robot_link = lambda side, role: (2, elbow_prim)
        guided = ex._solve_arm_ik_jacobian("right", Gf.Vec3d(0.1, 0, 0), pose(yaw=30))
        for index in (0, 1, 2, 4, 5, 6):
            self.assertAlmostEqual(guided[index], base[index], places=9)
        self.assertNotEqual(guided[3], base[3])
        self.assertLessEqual(max(abs(value) for value in guided.values()), ex._arm_ik_max_step)
        self.assertTrue(all(math.isfinite(value) for value in guided.values()))


if __name__ == "__main__":
    unittest.main()
