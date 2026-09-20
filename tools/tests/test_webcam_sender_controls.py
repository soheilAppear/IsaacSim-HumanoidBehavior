# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the dependency-free webcam packet boundary and latest-frame capture."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import threading
import time
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "webcam_tracking.py"
SPEC = importlib.util.spec_from_file_location("webcam_tracking_under_test", MODULE_PATH)
SENDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SENDER)


def landmark(x: float = 0.0, y: float = 0.0, z: float = 0.0, confidence: float = 0.9) -> types.SimpleNamespace:
    """Create a MediaPipe-shaped landmark without importing camera dependencies."""
    return types.SimpleNamespace(x=x, y=y, z=z, visibility=confidence, presence=confidence)


def pose_fixture() -> types.SimpleNamespace:
    """Provide distinct anatomical wrists in a complete pose observation."""
    image = [landmark() for _ in range(33)]
    image[15] = landmark(0.8, 0.5)
    image[16] = landmark(0.2, 0.5)
    world = [landmark(index * 0.01, -index * 0.02, 0.1) for index in range(33)]
    return types.SimpleNamespace(pose_landmarks=[image], pose_world_landmarks=[world])


def hand_fixture(x: float, label: str = "Right") -> types.SimpleNamespace:
    """Create a hand near an image wrist, with an intentionally independent label."""
    return types.SimpleNamespace(
        hand_landmarks=[[landmark(x, 0.5) for _ in range(21)]],
        hand_world_landmarks=[[landmark(index * 0.001) for index in range(21)]],
        handedness=[[types.SimpleNamespace(category_name=label, score=0.95)]],
    )


class TestWebcamSenderControls(unittest.TestCase):
    def packet(self, pose=None, hand=None, **kwargs) -> dict:
        return SENDER.build_packet(
            pose, hand, sender_session="fixture", seq=7, capture_monotonic_ns=123456789, **kwargs
        )

    def test_packet_preserves_axes_confidence_and_original_capture_clock(self) -> None:
        packet = self.packet(pose_fixture(), calibration_request=3)
        self.assertEqual(packet["schema_version"], 1)
        self.assertEqual(packet["pose_world"][10], [0.1, -0.2, 0.1, 0.9, 0.9])
        self.assertEqual(packet["capture_monotonic_ns"], 123456789)
        self.assertEqual(packet["calibration_request"], 3)
        self.assertEqual(json.loads(SENDER.encode_packet(packet)), packet)

    def test_sent_timestamp_uses_shared_qpc_clock_not_python_specific_monotonic_epoch(self) -> None:
        with patch.object(SENDER.time, "perf_counter_ns", return_value=123456799), patch.object(
            SENDER.time, "monotonic_ns", side_effect=AssertionError("Do not use Python-specific clock epoch")
        ):
            packet = self.packet(pose_fixture())
        self.assertEqual(packet["capture_monotonic_ns"], 123456789)
        self.assertEqual(packet["sent_monotonic_ns"], 123456799)

    def test_packet_is_accepted_by_production_receiver(self) -> None:
        receiver_path = (
            MODULE_PATH.parents[1]
            / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
            / "webcam_tracking.py"
        )
        receiver_spec = importlib.util.spec_from_file_location("webcam_receiver_contract_test", receiver_path)
        receiver = importlib.util.module_from_spec(receiver_spec)
        sys.modules[receiver_spec.name] = receiver
        try:
            receiver_spec.loader.exec_module(receiver)
            packet = self.packet(pose_fixture(), hand_fixture(0.8), calibration_request=2)
            decoded = receiver.decode_frame(SENDER.encode_packet(packet), now_ns=123456790)
            self.assertEqual(decoded.capture_monotonic_ns, packet["capture_monotonic_ns"])
            self.assertEqual(decoded.calibration_request, 2)
            self.assertEqual(set(decoded.hand_world), {"left"})
            self.assertEqual(decoded.pose_world[10], tuple(packet["pose_world"][10]))
        finally:
            del sys.modules[receiver_spec.name]

    def test_no_detection_is_explicit_and_never_reuses_prior_pose(self) -> None:
        self.packet(pose_fixture())
        packet = self.packet(types.SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[]))
        self.assertIsNone(packet["pose_world"])
        self.assertIsNone(packet["pose_image"])
        self.assertEqual(packet["hand_world"], {})

    def test_nonfinite_or_incomplete_pose_is_omitted(self) -> None:
        pose = pose_fixture()
        pose.pose_world_landmarks[0][15].x = math.nan
        self.assertIsNone(self.packet(pose)["pose_world"])
        pose = pose_fixture()
        pose.pose_world_landmarks[0].pop()
        self.assertIsNone(self.packet(pose)["pose_world"])

    def test_hands_match_pose_wrists_instead_of_mirrored_handedness_label(self) -> None:
        packet = self.packet(pose_fixture(), hand_fixture(0.8, label="Right"))
        self.assertEqual(set(packet["hand_world"]), {"left"})
        self.assertAlmostEqual(packet["hand_confidence"]["left"], 0.9)
        self.assertEqual(len(packet["hand_world"]["left"]), 21)

    def test_ambiguous_far_or_unseen_body_hands_are_not_assigned(self) -> None:
        for pose, hand in ((pose_fixture(), hand_fixture(0.5)), (None, hand_fixture(0.8))):
            self.assertEqual(self.packet(pose, hand)["hand_world"], {})
        pose = pose_fixture()
        pose.pose_landmarks[0][16] = landmark(0.79, 0.5)
        self.assertEqual(self.packet(pose, hand_fixture(0.8))["hand_world"], {})

    def test_unknown_pose_wrist_confidence_cannot_assign_hand(self) -> None:
        pose = pose_fixture()
        pose.pose_landmarks[0][15].presence = None
        self.assertEqual(self.packet(pose, hand_fixture(0.8))["hand_world"], {})

    def test_json_disallows_nonfinite_and_oversized_payloads(self) -> None:
        for packet in ({"x": math.inf}, {"x": "x" * 32769}):
            with self.assertRaises(ValueError):
                SENDER.encode_packet(packet)

    def test_model_cache_avoids_network_and_missing_requires_explicit_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(SENDER.urllib.request, "urlopen") as network:
                with self.assertRaises(FileNotFoundError):
                    SENDER.prepare_models(directory, "lite", False, False)
                path = directory / "pose_landmarker_lite.task"
                with zipfile.ZipFile(path, "w") as model:
                    model.writestr("pose_landmarker.tflite", b"fixture")
                self.assertEqual(SENDER.prepare_models(directory, "lite", False, False), {"lite": path})
                network.assert_not_called()

    def test_cli_defaults_are_local_and_do_not_enable_hands_or_downloads(self) -> None:
        args = SENDER.create_parser().parse_args([])
        self.assertEqual(args.port, 8765)
        self.assertFalse(args.hands)
        self.assertFalse(args.download_models)
        self.assertFalse(args.download_only)

    def test_latest_capture_skips_buffered_frames_and_releases_camera(self) -> None:
        finished = threading.Event()
        capture = Mock()
        counter = iter(range(1, 51))

        def read():
            try:
                value = next(counter)
                if value == 50:
                    finished.set()
                return True, value
            except StopIteration:
                time.sleep(0.01)
                return False, None

        capture.read.side_effect = read
        with patch.object(SENDER.time, "perf_counter_ns", return_value=987654321), patch.object(
            SENDER.time, "monotonic_ns", side_effect=AssertionError("Do not use Python-specific clock epoch")
        ):
            frames = SENDER.LatestFrameCapture(capture)
            try:
                self.assertTrue(finished.wait(1.0))
                snapshot = frames.next_frame(0)
                self.assertEqual(snapshot[0], 50)
                self.assertEqual(snapshot[1], 987654321)
                self.assertEqual(snapshot[2], 50)
                self.assertIsNone(frames.next_frame(50, timeout=0.01))
            finally:
                frames.close()
        capture.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
