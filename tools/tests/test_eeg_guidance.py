# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify spoken-rest timing and stream checks without audio or EEG hardware."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eeg.guidance import find_live_session, read_stream_state, run_rest


class FakeClock:
    """Advance deterministic time without sleeping or speaking."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestGuidedRest(unittest.TestCase):
    """Exercise saved-session selection and non-actuating rest guidance."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.sessions = self.root / "sessions"
        self.output = self.root / "guidance"
        self.clock = FakeClock()
        self.session = self.make_session("active")

    def make_session(self, name: str, *, receipt: float = 99.0, source: str = "brainbit") -> Path:
        path = self.sessions / name
        path.mkdir(parents=True)
        metadata = {
            "schema_version": 1,
            "source": source,
            "complete": False,
            "sample_rate_hz": 250,
            "channels": ["O1", "O2", "T3", "T4"],
            "packet_modulus": 2**32,
            "robot_output_enabled": False,
            "config": {"targets_hz": None},
        }
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with (path / "eeg.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "source",
                    "sample_index",
                    "segment",
                    "packet",
                    "marker",
                    "sample_time_estimate_monotonic_s",
                    "received_monotonic_s",
                    "received_unix_ns",
                    "missing_before",
                    "sequence_status",
                    "label",
                    "O1_v",
                    "O2_v",
                    "T3_v",
                    "T4_v",
                ]
            )
            writer.writerow(
                [
                    source,
                    7,
                    0,
                    65_537,
                    0,
                    receipt,
                    receipt,
                    1_800_000_000_000_000_000,
                    0,
                    "ok",
                    "unlabeled",
                    1e-6,
                    2e-6,
                    3e-6,
                    4e-6,
                ]
            )
        return path

    def read_events(self, folder: Path) -> list[dict]:
        return [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    def live_state(self, session: Path, *, now: float | None = None) -> dict:
        self.assertEqual(session, self.session.resolve())
        received = self.clock() - 1.0
        return {
            "sample_index": int(received * 250),
            "packet": int(received * 250),
            "received_monotonic_s": received,
            "file_sample_age_s": 1.0,
        }

    def speak(self, text: str) -> dict:
        self.clock.sleep(2.0)
        return {"backend": "mock", "finished_at": self.clock()}

    def test_read_state_ignores_partial_trailing_csv_row(self) -> None:
        with (self.session / "eeg.csv").open("a", encoding="utf-8") as handle:
            handle.write("brainbit,8,0,65538,0,")
        state = read_stream_state(self.session, now=100.0)
        self.assertEqual(state["sample_index"], 7)
        self.assertEqual(state["packet"], 65_537)
        self.assertEqual(state["received_monotonic_s"], 99.0)
        self.assertEqual(state["file_sample_age_s"], 1.0)

    def test_read_state_rejects_stale_future_synthetic_and_closed_sessions(self) -> None:
        for name, receipt, source in (
            ("stale", 94.9, "brainbit"),
            ("future", 100.1, "brainbit"),
            ("synthetic", 99.0, "synthetic"),
        ):
            with self.subTest(name=name), self.assertRaises((ValueError, RuntimeError)):
                read_stream_state(self.make_session(name, receipt=receipt, source=source), now=100.0)
        for name, fields in (
            ("completed", {"complete": True}),
            ("failed", {"complete": False, "ended_utc": "2026-11-03T04:30:00+00:00"}),
        ):
            path = self.make_session(name)
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            (path / "metadata.json").write_text(json.dumps({**metadata, **fields}), encoding="utf-8")
            with self.subTest(name=name), self.assertRaises((ValueError, RuntimeError)):
                read_stream_state(path, now=100.0)

    def test_find_session_requires_exactly_one_fresh_active_recorder(self) -> None:
        self.make_session("old", receipt=80.0)
        with patch("eeg.guidance.time.monotonic", return_value=100.0):
            self.assertEqual(find_live_session(self.sessions), self.session)
            self.make_session("second_active")
            with self.assertRaises((ValueError, RuntimeError)):
                find_live_session(self.sessions)
            (self.root / "empty").mkdir()
            with self.assertRaises((ValueError, RuntimeError)):
                find_live_session(self.root / "empty")

    def test_invalid_durations_never_request_speech(self) -> None:
        speaker = Mock()
        for duration, settle in (
            (4, 0),
            (301, 0),
            (math.nan, 0),
            (math.inf, 0),
            (True, 0),
            (30, -1),
            (30, 61),
            (30, math.nan),
            (30, True),
        ):
            with self.subTest(duration=duration, settle=settle), self.assertRaises(ValueError):
                run_rest(
                    self.session,
                    self.output,
                    speaker,
                    duration=duration,
                    settle=settle,
                    clock=self.clock,
                    sleep=self.clock.sleep,
                )
        speaker.assert_not_called()

    def test_quiet_interval_excludes_speech_and_waits_for_recorded_end(self) -> None:
        with patch("eeg.guidance.read_stream_state", side_effect=self.live_state):
            folder = run_rest(
                self.session, self.output, self.speak, duration=30, settle=10, clock=self.clock, sleep=self.clock.sleep
            )
        events = self.read_events(folder)
        started = next(event for event in events if event["type"] == "interval_started")
        finished = next(event for event in events if event["type"] == "interval_finished")
        speech_completed = [event for event in events if event["type"] == "speech_completed"]
        speech_requested = [event for event in events if event["type"] == "speech_requested"]
        self.assertEqual(started["monotonic_s"], 114.0)
        self.assertEqual(started["monotonic_s"], speech_completed[1]["monotonic_s"])
        self.assertEqual(finished["monotonic_s"] - started["monotonic_s"], 30.0)
        self.assertEqual(finished["elapsed_s"], 30.0)
        self.assertEqual(started["requested_label"], "rest")
        self.assertFalse(
            any(started["monotonic_s"] < event["monotonic_s"] < finished["monotonic_s"] for event in speech_requested)
        )
        self.assertGreaterEqual(speech_requested[-1]["monotonic_s"], finished["monotonic_s"] + 1.0)
        self.assertEqual(events[-1]["type"], "completed")
        self.assertGreaterEqual(events[-1]["final_sample"]["received_monotonic_s"], finished["monotonic_s"])
        self.assertFalse(any(event["type"] == "aborted" for event in events))

    def test_stale_stream_aborts_without_claiming_completion_or_editing_recording(self) -> None:
        before = (self.session / "eeg.csv").read_bytes()

        def stale_mid_interval(session: Path, *, now: float | None = None) -> dict:
            if self.clock() >= 106.0:
                raise ValueError("Recording is stale")
            return self.live_state(session, now=now)

        with patch("eeg.guidance.read_stream_state", side_effect=stale_mid_interval):
            with self.assertRaisesRegex(ValueError, "stale"):
                run_rest(
                    self.session,
                    self.output,
                    self.speak,
                    duration=5,
                    settle=0,
                    clock=self.clock,
                    sleep=self.clock.sleep,
                )
        events = self.read_events(next(self.output.iterdir()))
        self.assertEqual(events[-1]["type"], "aborted")
        self.assertFalse(any(event["type"] == "completed" for event in events))
        self.assertEqual((self.session / "eeg.csv").read_bytes(), before)

    def test_cancel_file_aborts_during_preparation_pause(self) -> None:
        def cancel_on_first_wait(seconds: float) -> None:
            self.clock.sleep(seconds)
            (next(self.output.iterdir()) / "cancel").touch()

        with patch("eeg.guidance.read_stream_state", side_effect=self.live_state):
            with self.assertRaisesRegex(InterruptedError, "cancelled"):
                run_rest(
                    self.session,
                    self.output,
                    self.speak,
                    duration=5,
                    settle=10,
                    clock=self.clock,
                    sleep=cancel_on_first_wait,
                )
        events = self.read_events(next(self.output.iterdir()))
        self.assertEqual(events[-1]["type"], "aborted")
        self.assertFalse(any(event["type"] in ("interval_started", "completed") for event in events))

    def test_speech_failure_records_one_abort_and_no_quiet_interval(self) -> None:
        speaker = Mock(side_effect=RuntimeError("mock audio unavailable"))
        with patch("eeg.guidance.read_stream_state", side_effect=self.live_state):
            with self.assertRaisesRegex(RuntimeError, "audio unavailable"):
                run_rest(
                    self.session, self.output, speaker, duration=5, settle=0, clock=self.clock, sleep=self.clock.sleep
                )
        events = self.read_events(next(self.output.iterdir()))
        self.assertEqual(sum(event["type"] == "aborted" for event in events), 1)
        self.assertFalse(
            any(event["type"] in ("speech_completed", "interval_started", "completed") for event in events)
        )

    def test_recorded_end_must_be_visible_before_success_is_announced(self) -> None:
        def held_back_tail(session: Path, *, now: float | None = None) -> dict:
            state = self.live_state(session, now=now)
            state["received_monotonic_s"] = min(state["received_monotonic_s"], 108.9)
            state["file_sample_age_s"] = self.clock() - state["received_monotonic_s"]
            if state["file_sample_age_s"] > 5.0:
                raise ValueError("Recording is stale while waiting for final flush")
            return state

        with patch("eeg.guidance.read_stream_state", side_effect=held_back_tail):
            with self.assertRaisesRegex((RuntimeError, ValueError), "flush"):
                run_rest(
                    self.session,
                    self.output,
                    self.speak,
                    duration=5,
                    settle=0,
                    clock=self.clock,
                    sleep=self.clock.sleep,
                )
        events = self.read_events(next(self.output.iterdir()))
        self.assertTrue(any(event["type"] == "interval_finished" for event in events))
        self.assertEqual(events[-1]["type"], "aborted")
        self.assertEqual(sum(event["type"] == "speech_requested" for event in events), 2)
        self.assertFalse(any(event["type"] == "completed" for event in events))


if __name__ == "__main__":
    unittest.main()
