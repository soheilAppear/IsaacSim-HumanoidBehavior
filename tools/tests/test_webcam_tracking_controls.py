# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise webcam protocol, anatomical frames, and explicit Quest alignment."""

from __future__ import annotations

import importlib.util
import json
import math
import socket
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/webcam_tracking.py"
)
SPEC = importlib.util.spec_from_file_location("webcam_tracking_under_test", MODULE_PATH)
WEB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WEB
SPEC.loader.exec_module(WEB)

NOW = 10_000_000_000


def packet(*, seq: int = 1, captured: int = NOW, session: str = "camera-a") -> dict:
    """Build an unmirrored, front-facing pose with two bent arms."""
    pose = [[0.0, 0.0, 0.0, 0.95, 0.95] for _ in range(33)]
    positions = {
        11: (0.2, -0.5, 0.0),
        12: (-0.2, -0.5, 0.0),
        23: (0.15, 0.0, 0.0),
        24: (-0.15, 0.0, 0.0),
        13: (0.2, -0.3, -0.12),
        14: (-0.2, -0.3, -0.12),
        15: (0.2, -0.15, -0.3),
        16: (-0.2, -0.15, -0.3),
    }
    for index, xyz in positions.items():
        pose[index][:3] = xyz
    return {
        "schema_version": 1,
        "sender_session": session,
        "seq": seq,
        "capture_monotonic_ns": captured,
        "pose_world": pose,
        "hand_world": {},
        "hand_confidence": {},
    }


def frame(data: dict | None = None, *, now: int = NOW):
    """Decode a test packet through the production boundary."""
    return WEB.decode_frame(json.dumps(packet() if data is None else data).encode(), now_ns=now)


class TestWebcamProtocol(unittest.TestCase):
    def test_valid_packet_preserves_identity_and_pose(self):
        result = frame()
        self.assertEqual(result.seq, 1)
        self.assertEqual(result.sender_session, "camera-a")
        self.assertEqual(len(result.pose_world), 33)
        self.assertEqual(result.calibration_request, 0)

    def test_explicit_tracking_loss_is_valid_and_clears_pose(self):
        data = packet()
        data["pose_world"] = None
        result = frame(data)
        self.assertIsNone(result.pose_world)
        self.assertIsNone(WEB.derive_body_pose(result))

    def test_hand_fields_require_21_landmarks_and_anatomical_side(self):
        data = packet()
        data["hand_world"] = {"left": [[0.01, 0.02, 0.03] for _ in range(21)]}
        data["hand_confidence"] = {"left": 0.9}
        data["calibration_request"] = 3
        result = frame(data)
        self.assertEqual(len(result.hand_world["left"]), 21)
        self.assertEqual(result.calibration_request, 3)
        data["hand_world"]["left"].pop()
        with self.assertRaises(ValueError):
            frame(data)
        data["hand_world"] = {"Left": [[0, 0, 0] for _ in range(21)]}
        with self.assertRaises(ValueError):
            frame(data)

    def test_rejects_malformed_oversized_and_deep_packets(self):
        for data in (b"", b"[]", b"null", b"\xff", b"{", b" " * 32769, b"[" * 2000 + b"]" * 2000):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                WEB.decode_frame(data, now_ns=NOW)

    def test_rejects_version_identity_and_integer_errors(self):
        values = {
            "schema_version": (2, True, 1.0, "1"),
            "sender_session": (None, "", "x" * 65, "camera\n", "\u00e9"),
            "seq": (-1, True, 1.5, 2**64),
            "capture_monotonic_ns": (-1, True, 1.5),
            "calibration_request": (-1, True, "1"),
        }
        for field, candidates in values.items():
            for value in candidates:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    data = packet()
                    data[field] = value
                    frame(data)

    def test_rejects_stale_and_future_data(self):
        for captured in (NOW - 250_000_001, NOW + 5_000_001):
            with self.subTest(captured=captured), self.assertRaises(ValueError):
                frame(packet(captured=captured))
        self.assertIsNotNone(frame(packet(captured=NOW - 250_000_000)))

    def test_capture_clock_uses_cross_version_performance_counter(self):
        # Isaac Python 3.12 uses GetTickCount64 for `monotonic_ns`, while host
        # Python 3.14 uses QPC. Only `perf_counter_ns` shares the clock origin.
        with patch.object(WEB.time, "monotonic_ns", return_value=NOW - 6_000_000_000), patch.object(
            WEB.time, "perf_counter_ns", return_value=NOW
        ):
            sample = WEB.decode_frame(json.dumps(packet()).encode())
            self.assertEqual(sample.age_s(), 0.0)
            alignment = WEB.WebcamCalibration()
            alignment.calibrate(sample, {"left": (0.3, 0.2, 0.1), "right": (0.3, -0.2, 0.1)})
            self.assertEqual(set(alignment.map(sample)), {"left", "right"})

    def test_rejects_nonfinite_out_of_range_and_bool_landmarks(self):
        for position, value in ((0, math.nan), (0, math.inf), (0, 5.1), (0, True), (0, 10**500), (3, -0.1), (4, 1.1)):
            with self.subTest(position=position, value=value), self.assertRaises(ValueError):
                data = packet()
                data["pose_world"][15][position] = value
                frame(data)

    def test_rejects_wrong_pose_dimensions(self):
        for pose in ([], [[0, 0, 0, 1, 1]] * 32, [[0, 0, 0]] * 33, "pose"):
            with self.subTest(pose_length=len(pose)), self.assertRaises(ValueError):
                data = packet()
                data["pose_world"] = pose
                frame(data)


class TestWebcamReceiver(unittest.TestCase):
    def setUp(self):
        self.receiver = WEB.WebcamReceiver(port=0)

    def tearDown(self):
        self.receiver.close()

    def accept(self, data):
        return self.receiver.accept(json.dumps(data).encode(), now_ns=NOW)

    def test_rejects_duplicate_out_of_order_and_timestamp_regression(self):
        self.assertTrue(self.accept(packet(seq=3, captured=NOW - 100)))
        for data in (
            packet(seq=3, captured=NOW),
            packet(seq=2, captured=NOW),
            packet(seq=4, captured=NOW - 101),
        ):
            self.assertFalse(self.accept(data))
        self.assertEqual(self.receiver.latest.seq, 3)
        self.assertEqual(self.receiver.rejected_packets, 3)
        self.assertTrue(self.accept(packet(seq=4)))

    def test_sender_restart_accepts_new_sequence_and_retires_old_session(self):
        self.assertTrue(self.accept(packet(seq=100, captured=NOW - 200)))
        self.assertTrue(self.accept(packet(seq=0, captured=NOW - 100, session="camera-b")))
        self.assertFalse(self.accept(packet(seq=101)))
        self.assertEqual(self.receiver.latest.sender_session, "camera-b")

    def test_missing_pose_supersedes_old_valid_tracking(self):
        self.assertTrue(self.accept(packet(captured=NOW - 100)))
        lost = packet(seq=2)
        lost["pose_world"] = None
        self.assertTrue(self.accept(lost))
        self.assertIsNone(self.receiver.poll(now_ns=NOW).pose_world)

    def test_poll_expires_cache_without_erasing_order_history(self):
        self.assertTrue(self.accept(packet()))
        self.assertIsNotNone(self.receiver.poll(now_ns=NOW))
        self.assertIsNone(self.receiver.poll(now_ns=NOW + 250_000_001))
        self.assertIsNotNone(self.receiver.latest)

    def test_backlogged_socket_yields_no_target_until_queue_is_drained(self):
        self.receiver._socket.close()
        fake_socket = Mock()
        datagrams = [
            (json.dumps(packet(seq=seq, captured=NOW - 200 + seq)).encode(), ("127.0.0.1", 55000)) for seq in range(129)
        ]
        fake_socket.recvfrom.side_effect = [*datagrams, BlockingIOError()]
        self.receiver._socket = fake_socket
        self.assertIsNone(self.receiver.poll(now_ns=NOW))
        self.assertEqual(fake_socket.recvfrom.call_count, 128)
        result = self.receiver.poll(now_ns=NOW)
        self.assertEqual(result.seq, 128)

    def test_port_conflict_fails_without_stealing_existing_receiver(self):
        with self.assertRaises(OSError):
            WEB.WebcamReceiver(port=self.receiver.port)
        self.assertTrue(self.accept(packet()))

    def test_local_udp_drains_to_latest_and_closes_idempotently(self):
        self.assertTrue(self.receiver.is_open)
        now = time.perf_counter_ns()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            for seq in range(3):
                sender.sendto(
                    json.dumps(packet(seq=seq, captured=now - 3 + seq)).encode(), ("127.0.0.1", self.receiver.port)
                )
        result = self.receiver.poll(now_ns=time.perf_counter_ns())
        self.assertIsNotNone(result)
        self.assertEqual(result.seq, 2)
        self.assertEqual(self.receiver.accepted_packets, 3)
        self.receiver.close()
        self.receiver.close()
        self.assertFalse(self.receiver.is_open)
        self.assertIsNone(self.receiver.poll())

    def test_receive_error_reports_closed_socket(self):
        self.receiver._socket.close()
        fake_socket = Mock()
        fake_socket.recvfrom.side_effect = OSError("socket disconnected")
        self.receiver._socket = fake_socket
        self.assertIsNone(self.receiver.poll(now_ns=NOW))
        self.assertFalse(self.receiver.is_open)
        self.assertIn("socket disconnected", self.receiver.last_error)
        fake_socket.close.assert_called_once()


class TestAnatomicalBodyFrame(unittest.TestCase):
    def test_unmirrored_front_view_maps_forward_left_and_up(self):
        data = packet()
        data["pose_world"][15][:3] = [0.28, -0.15, -0.30]
        pose = WEB.derive_body_pose(frame(data))
        for actual, expected in zip(pose.arms["left"].wrist, (0.30, 0.08, -0.35)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(pose.arms["right"].wrist[1], 0.0)

    def test_camera_translation_and_rotation_do_not_change_body_coordinates(self):
        original = WEB.derive_body_pose(frame())
        data = packet()
        angle = 0.45
        for point in data["pose_world"]:
            x, y, z = point[:3]
            point[:3] = [
                math.cos(angle) * x + math.sin(angle) * z + 0.4,
                y + 0.3,
                -math.sin(angle) * x + math.cos(angle) * z,
            ]
        rotated = WEB.derive_body_pose(frame(data))
        for side in WEB.SIDES:
            for actual, expected in zip(rotated.arms[side].wrist, original.arms[side].wrist):
                self.assertAlmostEqual(actual, expected)

    def test_low_torso_confidence_disables_all_arms(self):
        for index in (11, 12, 23, 24):
            with self.subTest(index=index):
                data = packet()
                data["pose_world"][index][3] = 0.64
                self.assertIsNone(WEB.derive_body_pose(frame(data)))

    def test_one_occluded_arm_does_not_disable_other_arm(self):
        data = packet()
        data["pose_world"][13][4] = 0.2
        body = WEB.derive_body_pose(frame(data))
        self.assertNotIn("left", body.arms)
        self.assertIn("right", body.arms)

    def test_degenerate_or_implausible_body_is_not_extrapolated(self):
        for index, point in ((12, (0.2, -0.5, 0)), (15, (0.2, -0.3, -0.12)), (13, (3, 2, 1))):
            with self.subTest(index=index):
                data = packet()
                data["pose_world"][index][:3] = point
                body = WEB.derive_body_pose(frame(data))
                self.assertTrue(body is None or "left" not in body.arms)


class TestExplicitWebcamCalibration(unittest.TestCase):
    def setUp(self):
        self.calibration = WEB.WebcamCalibration()
        self.targets = {"left": (0.3, 0.24, 0.1), "right": (0.3, -0.24, 0.1)}

    def calibrate(self, data=None):
        self.calibration.calibrate(frame(data), self.targets, now_ns=NOW)

    def test_no_automatic_mapping_before_explicit_calibration(self):
        self.assertEqual(self.calibration.map(frame(), now_ns=NOW), {})
        self.assertFalse(self.calibration.is_calibrated)

    def test_calibration_pairs_wrist_to_current_quest_target(self):
        self.calibrate()
        self.assertTrue(self.calibration.is_calibrated)
        result = self.calibration.map(frame(), now_ns=NOW)
        for side in WEB.SIDES:
            for actual, expected in zip(result[side].wrist_body, self.targets[side]):
                self.assertAlmostEqual(actual, expected)
            self.assertLess(result[side].elbow_body[0], result[side].wrist_body[0])

    def test_webcam_motion_has_correct_direction_and_calibrated_scale(self):
        self.calibrate()
        data = packet()
        data["pose_world"][15][2] -= 0.05
        moved = self.calibration.map(frame(data), now_ns=NOW)
        self.assertAlmostEqual(moved["left"].wrist_body[0] - 0.3, 0.05 * self.calibration.scales["left"])
        self.assertAlmostEqual(moved["right"].wrist_body[0], 0.3)

    def test_calibration_rejects_partial_missing_stale_and_bad_scale(self):
        with self.assertRaises(ValueError):
            self.calibration.calibrate(frame(), {"left": self.targets["left"]}, now_ns=NOW)
        data = packet()
        data["pose_world"][16][3] = 0.1
        with self.assertRaises(ValueError):
            self.calibrate(data)
        with self.assertRaises(ValueError):
            self.calibration.calibrate(frame(), self.targets, now_ns=NOW + 300_000_000)
        with self.assertRaises(ValueError):
            self.calibration.calibrate(frame(), self.targets, now_ns=NOW + 100_000_001)
        with self.assertRaises(ValueError):
            self.calibration.calibrate(frame(), self.targets, robot_arm_lengths={"left": 1.0, "right": 1.0}, now_ns=NOW)
        self.assertFalse(self.calibration.is_calibrated)

    def test_failed_recalibration_preserves_previous_explicit_mapping(self):
        self.calibrate()
        data = packet()
        data["pose_world"] = None
        with self.assertRaises(ValueError):
            self.calibrate(data)
        self.assertTrue(self.calibration.is_calibrated)

    def test_occlusion_and_staleness_disable_targets(self):
        self.calibrate()
        self.assertEqual(self.calibration.map(frame(), now_ns=NOW + 250_000_001), {})
        data = packet()
        data["pose_world"][15][4] = 0.1
        result = self.calibration.map(frame(data), now_ns=NOW)
        self.assertEqual(set(result), {"right"})

    def test_sender_restart_requires_explicit_recalibration(self):
        self.calibrate()
        self.assertEqual(self.calibration.map(frame(packet(session="camera-b")), now_ns=NOW), {})
        self.assertFalse(self.calibration.is_calibrated)
        self.assertEqual(self.calibration.map(frame(), now_ns=NOW), {})

    def test_clear_removes_both_mappings(self):
        self.calibrate()
        self.calibration.clear()
        self.assertFalse(self.calibration.is_calibrated)
        self.assertEqual(self.calibration.scales, {})
        self.assertEqual(self.calibration.offsets, {})


if __name__ == "__main__":
    unittest.main()
