# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test Steam Link OSC parsing, independent freshness and bounded local transport."""

from __future__ import annotations

import importlib.util
import math
import socket
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
    / "face_tracking_input.py"
)
SPEC = importlib.util.spec_from_file_location("face_tracking_input_under_test", MODULE_PATH)
FACE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FACE
SPEC.loader.exec_module(FACE)


def osc_string(value: str) -> bytes:
    """Encode one padded test OSC string."""
    data = value.encode() + b"\0"
    return data + b"\0" * (-len(data) % 4)


def message(address: str, value: float) -> bytes:
    """Encode one standard OSC float message."""
    return osc_string(address) + osc_string(",f") + struct.pack(">f", value)


def bundle(*packets: bytes) -> bytes:
    """Encode an immediate OSC bundle containing the supplied elements."""
    return b"#bundle\0" + struct.pack(">Q", 1) + b"".join(struct.pack(">i", len(p)) + p for p in packets)


def frame(**overrides: float) -> bytes:
    """Construct real Steam Link addresses for every required control field."""
    weights = {name: 0.0 for name in FACE.EXPRESSION_NAMES}
    weights.update(overrides)
    return bundle(
        *(message(f"/sl/xrfb/facew/{name}", value) for name, value in weights.items()),
        message("/sl/xrfb/facec/UpperFace", 0.9),
        message("/sl/xrfb/facec/LowerFace", 0.95),
    )


class TestFaceOscParser(unittest.TestCase):
    def test_nested_split_bundles_preserve_separate_eyelids(self) -> None:
        receiver = FACE.SteamLinkFaceInput()
        receiver.ingest_packet(bundle(frame(EyesClosedL=0.9, EyesClosedR=0.05)), now_s=10.0)
        sample = receiver.snapshot(now_s=10.05)
        self.assertTrue(sample.valid)
        self.assertAlmostEqual(sample.weights["EyesClosedL"], 0.9)
        self.assertAlmostEqual(sample.weights["EyesClosedR"], 0.05)
        self.assertAlmostEqual(sample.age_s, 0.05)

    def test_single_messages_can_complete_a_frame(self) -> None:
        receiver = FACE.SteamLinkFaceInput()
        for address, args in FACE.decode_osc_packet(frame()):
            receiver.ingest_packet(message(address, args[0]), now_s=10.0)
        self.assertTrue(receiver.snapshot(now_s=10.0).valid)

    def test_rejects_truncated_bad_padding_and_trailing_bytes(self) -> None:
        good = message("/sl/xrfb/facew/JawDrop", 0.2)
        for bad in (b"", b"a", good[:-1], good + b"\0", b"#bundle\0", osc_string("address") + b",f\0\0"):
            with self.subTest(payload=bad), self.assertRaises(ValueError):
                FACE.decode_osc_packet(bad)
        with self.assertRaises(ValueError):
            FACE.decode_osc_packet(b"/x\0z,f\0\0" + struct.pack(">f", 0.0))

    def test_rejects_oversized_nested_and_excessive_packets(self) -> None:
        nested = frame()
        for _ in range(6):
            nested = bundle(nested)
        many = bundle(*(message("/x", 0.0) for _ in range(257)))
        for bad in (b"x" * 65508, nested, many):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                FACE.decode_osc_packet(bad)

    def test_rejects_invalid_bundle_length(self) -> None:
        for length in (-4, 0, 3, 10000):
            bad = b"#bundle\0" + struct.pack(">Q", 1) + struct.pack(">i", length) + b"1234"
            with self.subTest(length=length), self.assertRaises(ValueError):
                FACE.decode_osc_packet(bad)

    def test_supports_unrelated_standard_types_without_using_them(self) -> None:
        data = osc_string("/misc") + osc_string(",idhTFNIbs")
        data += struct.pack(">idq", 2, 1.25, 2**40) + struct.pack(">i", 3) + b"abc\0" + osc_string("hello")
        address, args = FACE.decode_osc_packet(data)[0]
        self.assertEqual(address, "/misc")
        self.assertEqual(args[:6], (2, 1.25, 2**40, True, False, None))
        self.assertEqual(args[6:], (math.inf, b"abc", "hello"))


class TestFaceInputFreshness(unittest.TestCase):
    def setUp(self) -> None:
        self.receiver = FACE.SteamLinkFaceInput()
        self.receiver.ingest_packet(frame(), now_s=10.0)

    def test_fresh_neutral_frame_is_valid(self) -> None:
        sample = self.receiver.snapshot(now_s=10.0)
        self.assertTrue(sample.valid)
        self.assertEqual(sample.sequence, 1)
        self.assertEqual(len(sample.weights), 7)

    def test_unrelated_packets_do_not_refresh_expired_face(self) -> None:
        self.receiver.ingest_packet(message("/tracking/eye/EyesClosedAmount", 0.0), now_s=10.3)
        sample = self.receiver.snapshot(now_s=10.3)
        self.assertFalse(sample.valid)
        self.assertEqual(sample.reason, "Face tracking is stale")

    def test_partial_update_does_not_refresh_other_fields(self) -> None:
        self.receiver.ingest_packet(message("/sl/xrfb/facew/JawDrop", 0.9), now_s=10.3)
        self.assertFalse(self.receiver.snapshot(now_s=10.3).valid)

    def test_sequence_requires_every_field_to_refresh(self) -> None:
        self.receiver.ingest_packet(message("/sl/xrfb/facew/JawDrop", 0.9), now_s=10.01)
        first = self.receiver.snapshot(now_s=10.01)
        self.assertEqual(first.sequence, 1)
        for index in range(20):
            self.receiver.ingest_packet(message("/sl/xrfb/facec/UpperFace", 0.9), now_s=10.01 + index * 0.01)
        self.assertEqual(self.receiver.snapshot(now_s=10.2).sequence, 1)
        for address, args in FACE.decode_osc_packet(frame(JawDrop=0.9)):
            if address not in ("/sl/xrfb/facew/JawDrop", "/sl/xrfb/facec/UpperFace"):
                self.receiver.ingest_packet(message(address, args[0]), now_s=10.21)
        self.assertEqual(self.receiver.snapshot(now_s=10.21).sequence, 2)
        self.receiver.ingest_packet(message("/sl/xrfb/facew/JawDrop", 0.9), now_s=10.22)
        self.assertEqual(self.receiver.snapshot(now_s=10.22).sequence, 2)

    def test_malformed_packet_discards_partial_refresh_cycle(self) -> None:
        self.receiver.ingest_packet(message("/sl/xrfb/facew/JawDrop", 0.9), now_s=10.01)
        self.receiver.ingest_packet(b"bad", now_s=10.02)
        for address, args in FACE.decode_osc_packet(frame()):
            if address != "/sl/xrfb/facew/JawDrop":
                self.receiver.ingest_packet(message(address, args[0]), now_s=10.03)
        sample = self.receiver.snapshot(now_s=10.03)
        self.assertFalse(sample.valid)
        self.assertEqual(sample.sequence, 1)

    def test_both_confidences_required_and_fresh(self) -> None:
        self.receiver.ingest_packet(message("/sl/xrfb/facec/LowerFace", 0.1), now_s=10.05)
        sample = self.receiver.snapshot(now_s=10.05)
        self.assertFalse(sample.valid)
        self.assertEqual(sample.reason, "Face tracking confidence is low")
        new = FACE.SteamLinkFaceInput()
        for name in FACE.EXPRESSION_NAMES:
            new.ingest_packet(message(f"/sl/xrfb/facew/{name}", 0.0), now_s=10.0)
        self.assertFalse(new.snapshot(now_s=10.0).valid)

    def test_invalid_weight_clears_cached_command(self) -> None:
        for value in (math.nan, math.inf, -0.1, 1.1):
            with self.subTest(value=value):
                self.receiver.ingest_packet(frame(JawDrop=0.9), now_s=10.0)
                self.assertFalse(self.receiver.ingest_packet(message("/sl/xrfb/facew/JawDrop", value), now_s=10.01))
                self.assertFalse(self.receiver.snapshot(now_s=10.01).valid)
                self.assertEqual(self.receiver.snapshot(now_s=10.01).weights, {})

    def test_boolean_or_extra_arguments_are_not_weights(self) -> None:
        for args in (osc_string(",T"), osc_string(",ff") + struct.pack(">ff", 0.1, 0.2)):
            bad = osc_string("/sl/xrfb/facew/JawDrop") + args
            self.assertFalse(self.receiver.ingest_packet(bad, now_s=10.0))
            self.assertFalse(self.receiver.snapshot(now_s=10.0).valid)

    def test_malformed_bundle_is_atomic(self) -> None:
        bad = bundle(frame(JawDrop=0.9), b"bad\0")
        self.assertFalse(self.receiver.ingest_packet(bad, now_s=10.1))
        self.assertEqual(self.receiver.snapshot(now_s=10.1).weights, {})

    def test_backward_or_invalid_clock_cannot_validate(self) -> None:
        self.assertFalse(self.receiver.snapshot(now_s=9.9).valid)
        with self.assertRaises(ValueError):
            self.receiver.poll(now_s=math.nan)

    def test_close_clears_cached_sample(self) -> None:
        self.receiver.close()
        self.assertFalse(self.receiver.snapshot(now_s=10.1).valid)

    def test_invalid_configuration_is_rejected(self) -> None:
        for kwargs in (
            {"port": True},
            {"port": -1},
            {"port": 65536},
            {"stale_after_s": math.nan},
            {"stale_after_s": 0},
            {"min_confidence": 0},
            {"min_confidence": 1.1},
            {"max_packets_per_poll": 0},
            {"max_packets_per_poll": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FACE.SteamLinkFaceInput(**kwargs)


class TestFaceInputSocket(unittest.TestCase):
    def test_short_callback_burst_keeps_only_latest_complete_face(self) -> None:
        for count in (75, 100):
            with self.subTest(packets=count):
                receiver = FACE.SteamLinkFaceInput()
                receiver._last_poll_s = 10.0
                receiver._socket = Mock()
                receiver._socket.recvfrom.side_effect = [
                    *((frame(JawDrop=0.9), ("127.0.0.1", 1234)) for _ in range(count - 1)),
                    (frame(MouthRight=0.6), ("127.0.0.1", 1234)),
                    BlockingIOError(),
                ]
                sample = receiver.poll(now_s=10.188)
                self.assertTrue(sample.valid)
                self.assertAlmostEqual(sample.weights["JawDrop"], 0.0)
                self.assertAlmostEqual(sample.weights["MouthRight"], 0.6)
                self.assertEqual(sample.sequence, count)
                self.assertEqual(receiver.diagnostics()["backlog_events"], 0)
                self.assertEqual(receiver._socket.recvfrom.call_count, count + 1)

    def test_long_callback_burst_is_discarded_until_a_fresh_following_poll(self) -> None:
        for count in (75, 100):
            with self.subTest(packets=count):
                receiver = FACE.SteamLinkFaceInput()
                receiver._last_poll_s = 10.0
                receiver._socket = Mock()
                receiver.ingest_packet(frame(), now_s=10.0)
                receiver._socket.recvfrom.side_effect = [
                    *((frame(JawDrop=0.9), ("127.0.0.1", 1234)) for _ in range(count)),
                    BlockingIOError(),
                ]
                sample = receiver.poll(now_s=10.251)
                self.assertFalse(sample.valid)
                self.assertEqual(sample.weights, {})
                self.assertEqual(sample.sequence, 1)
                self.assertIn("callback interruption", sample.reason)
                receiver._socket.recvfrom.side_effect = [(frame(), ("127.0.0.1", 1234)), BlockingIOError()]
                fresh = receiver.poll(now_s=10.26)
                self.assertTrue(fresh.valid)
                self.assertEqual(fresh.sequence, 2)
                self.assertEqual(fresh.weights["JawDrop"], 0.0)

    def test_default_packet_cap_and_persistent_backlog_remain_invalid(self) -> None:
        receiver = FACE.SteamLinkFaceInput()
        self.assertEqual(receiver.max_packets_per_poll, 128)
        receiver._last_poll_s = 10.0
        receiver._socket = Mock()
        receiver._socket.recvfrom.return_value = (frame(JawDrop=0.9), ("127.0.0.1", 1234))
        for poll_index in range(3):
            sample = receiver.poll(now_s=10.01 + poll_index * 0.01)
            self.assertFalse(sample.valid)
            self.assertEqual(sample.weights, {})
            self.assertEqual(sample.sequence, 128)
            self.assertEqual(receiver._socket.recvfrom.call_count, 128 * (poll_index + 1))
            self.assertEqual(receiver.diagnostics()["backlog_events"], poll_index + 1)
        receiver._socket.recvfrom.side_effect = [(frame(), ("127.0.0.1", 1234)), BlockingIOError()]
        self.assertFalse(receiver.poll(now_s=10.04).valid)
        receiver._socket.recvfrom.side_effect = [(frame(), ("127.0.0.1", 1234)), BlockingIOError()]
        self.assertTrue(receiver.poll(now_s=10.05).valid)

    def test_exact_default_batch_is_conservatively_invalidated(self) -> None:
        receiver = FACE.SteamLinkFaceInput()
        receiver._last_poll_s = 10.0
        receiver._socket = Mock()
        receiver._socket.recvfrom.side_effect = [
            *((frame(JawDrop=0.9), ("127.0.0.1", 1234)) for _ in range(128)),
            BlockingIOError(),
        ]
        self.assertFalse(receiver.poll(now_s=10.188).valid)
        self.assertFalse(receiver.poll(now_s=10.20).valid)
        receiver._socket.recvfrom.side_effect = [(frame(), ("127.0.0.1", 1234)), BlockingIOError()]
        self.assertTrue(receiver.poll(now_s=10.21).valid)

    def test_short_burst_still_requires_all_fields_and_final_confidence(self) -> None:
        missing_jaw = bundle(
            *(
                message(address, args[0])
                for address, args in FACE.decode_osc_packet(frame())
                if not address.endswith("JawDrop")
            )
        )
        low_confidence = bundle(frame(JawDrop=0.9), message("/sl/xrfb/facec/LowerFace", 0.1))
        for payload in (missing_jaw, low_confidence):
            with self.subTest(payload=payload):
                receiver = FACE.SteamLinkFaceInput()
                receiver._last_poll_s = 10.0
                receiver._socket = Mock()
                receiver._socket.recvfrom.side_effect = [
                    *((payload, ("127.0.0.1", 1234)) for _ in range(100)),
                    BlockingIOError(),
                ]
                sample = receiver.poll(now_s=10.188)
                self.assertFalse(sample.valid)
                self.assertEqual(receiver.diagnostics()["backlog_events"], 0)

    def test_real_loopback_bind_poll_and_close(self) -> None:
        receiver = FACE.SteamLinkFaceInput(port=0)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            receiver.start()
            self.assertFalse(receiver.poll().valid)
            sender.sendto(frame(MouthLeft=0.8), ("127.0.0.1", receiver.port))
            sample = receiver.poll()
            self.assertTrue(sample.valid)
            self.assertAlmostEqual(sample.weights["MouthLeft"], 0.8)
            other = FACE.SteamLinkFaceInput(port=receiver.port)
            with self.assertRaises(OSError):
                other.start()
            other.close()
        finally:
            receiver.close()
            sender.close()

    def test_callback_stall_discards_queued_gesture(self) -> None:
        receiver = FACE.SteamLinkFaceInput()
        receiver._last_poll_s = 10.0
        receiver._socket = Mock()
        receiver._socket.recvfrom.side_effect = [(frame(JawDrop=0.9), ("127.0.0.1", 1234)), BlockingIOError()]
        self.assertFalse(receiver.poll(now_s=10.4).valid)
        receiver._socket.recvfrom.side_effect = [(frame(), ("127.0.0.1", 1234)), BlockingIOError()]
        self.assertTrue(receiver.poll(now_s=10.41).valid)

    def test_bounded_backlog_discards_until_empty(self) -> None:
        receiver = FACE.SteamLinkFaceInput(max_packets_per_poll=2)
        receiver._last_poll_s = 10.0
        receiver._socket = Mock()
        receiver._socket.recvfrom.return_value = (frame(JawDrop=0.9), ("127.0.0.1", 1234))
        self.assertFalse(receiver.poll(now_s=10.01).valid)
        self.assertEqual(receiver._socket.recvfrom.call_count, 2)
        receiver._socket.recvfrom.side_effect = [(frame(), ("127.0.0.1", 1234)), BlockingIOError()]
        self.assertFalse(receiver.poll(now_s=10.02).valid)
        receiver._socket.recvfrom.side_effect = [(frame(), ("127.0.0.1", 1234)), BlockingIOError()]
        self.assertTrue(receiver.poll(now_s=10.03).valid)


if __name__ == "__main__":
    unittest.main()
