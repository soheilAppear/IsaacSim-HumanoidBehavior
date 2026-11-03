# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check raw recording integrity and application lifecycle without EEG hardware."""

from __future__ import annotations

import csv
import json
import math
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eeg.app import PreviewApp
from eeg.session import CHANNELS, SessionRecorder


def sample_event(packets: list[int], received: float, *, first_index: int = 0) -> dict:
    """Build known four-channel volt samples with a stable 12 Hz component."""
    samples = []
    for index, packet in enumerate(packets, start=first_index):
        carrier = math.sin(2 * math.pi * 12 * index / 250)
        samples.append(
            {"packet": packet, "marker": 7, "values_v": [carrier * level for level in (25e-6, 22e-6, 5e-6, 4e-6)]}
        )
    return {
        "type": "samples",
        "received_monotonic": received,
        "received_unix_ns": 1_800_000_000_123_456_789,
        "samples": samples,
    }


def read_rows(path: Path) -> list[dict]:
    """Read finalized rows through the public CSV schema."""
    with (path / "eeg.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_metadata(path: Path) -> dict:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


class TestSessionRecorder(unittest.TestCase):
    """Verify persisted units, sequence accounting, labels, and completion state."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.recorder = SessionRecorder(Path(self.directory.name), "synthetic", {"test": True})
        self.addCleanup(self.recorder.close, outcome="failed", error="test cleanup")

    def test_preserves_raw_volts_receipt_clocks_and_channel_order(self) -> None:
        event = sample_event([10, 11], 100.0)
        event["samples"][0]["values_v"] = [1.25e-6, -2.5e-6, 3.75e-6, -4e-6]
        self.assertFalse(read_metadata(self.recorder.path)["complete"])
        self.recorder.write_samples(event)
        self.recorder.close(outcome="completed")
        rows = read_rows(self.recorder.path)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["sample_index"] for row in rows], ["0", "1"])
        self.assertEqual([float(rows[0][f"{channel}_v"]) for channel in CHANNELS], event["samples"][0]["values_v"])
        self.assertEqual([row["source"] for row in rows], ["synthetic", "synthetic"])
        self.assertEqual([row["marker"] for row in rows], ["7", "7"])
        self.assertEqual([int(row["received_unix_ns"]) for row in rows], [event["received_unix_ns"]] * 2)
        self.assertEqual([float(row["received_monotonic_s"]) for row in rows], [100.0, 100.0])
        metadata = read_metadata(self.recorder.path)
        self.assertEqual(metadata["raw_unit"], "V")
        self.assertEqual(metadata["channels"], list(CHANNELS))
        self.assertEqual(metadata["sample_count"], 2)
        self.assertEqual(metadata["sample_rate_hz"], 250)
        self.assertTrue(metadata["complete"])
        self.assertFalse(metadata["robot_output_enabled"])
        self.assertIn("ended_utc", metadata)

    def test_counter_wrap_keeps_four_millisecond_spacing(self) -> None:
        self.recorder.write_samples(sample_event([2046, 2047, 0, 1], 100.0))
        self.recorder.close(outcome="completed")
        rows = read_rows(self.recorder.path)
        for row, expected in zip(rows, [99.988, 99.992, 99.996, 100.0]):
            self.assertAlmostEqual(float(row["sample_time_estimate_monotonic_s"]), expected)
        self.assertEqual([row["segment"] for row in rows], ["0"] * 4)
        self.assertEqual([row["sequence_status"] for row in rows], ["start", "ok", "ok", "ok"])
        self.assertEqual(read_metadata(self.recorder.path)["missing_samples"], 0)

    def test_brainbit2_device_details_configure_uint32_counter_before_samples(self) -> None:
        info = {
            "family": "BrainBit2",
            "sample_rate_hz": 250,
            "packet_modulus": 2**32,
            "channels": list(CHANNELS),
        }
        self.recorder.configure_device(info)
        info["channels"][0] = "mutated after configuration"
        self.assertEqual(self.recorder.packet_modulus, 2**32)
        self.recorder.write_samples(sample_event([65_534, 65_535, 65_536], 100.0))
        self.recorder.close(outcome="completed")
        rows = read_rows(self.recorder.path)
        self.assertEqual([int(row["packet"]) for row in rows], [65_534, 65_535, 65_536])
        self.assertEqual([row["sequence_status"] for row in rows], ["start", "ok", "ok"])
        metadata = read_metadata(self.recorder.path)
        self.assertEqual(metadata["packet_modulus"], 2**32)
        self.assertEqual(metadata["device"]["channels"], list(CHANNELS))
        self.assertEqual(metadata["device"]["family"], "BrainBit2")

    def test_uint32_wrap_and_first_batch_gap_have_correct_sample_times(self) -> None:
        self.recorder.configure_device({"sample_rate_hz": 250, "packet_modulus": 2**32, "channels": list(CHANNELS)})
        packets = [2**32 - 3, 2**32 - 1, 0, 1]
        self.recorder.write_samples(sample_event(packets, 100.0))
        self.recorder.close(outcome="completed")
        rows = read_rows(self.recorder.path)
        self.assertEqual([int(row["packet"]) for row in rows], packets)
        self.assertEqual([row["sequence_status"] for row in rows], ["start", "gap", "ok", "ok"])
        for row, expected in zip(rows, [99.984, 99.992, 99.996, 100.0]):
            self.assertAlmostEqual(float(row["sample_time_estimate_monotonic_s"]), expected)
        self.assertEqual(read_metadata(self.recorder.path)["missing_samples"], 1)
        self.assertEqual(read_metadata(self.recorder.path)["sequence_anomalies"], 0)

    def test_uint32_backwards_jump_starts_segment_and_duplicate_is_preserved(self) -> None:
        self.recorder.configure_device({"sample_rate_hz": 250, "packet_modulus": 2**32, "channels": list(CHANNELS)})
        self.recorder.write_samples(sample_event([65_536, 65_536, 2, 3], 100.0))
        self.recorder.close(outcome="completed")
        rows = read_rows(self.recorder.path)
        self.assertEqual(
            [row["sequence_status"] for row in rows], ["start", "duplicate", "restart_or_out_of_order", "ok"]
        )
        self.assertEqual([int(row["segment"]) for row in rows], [0, 0, 1, 1])
        self.assertAlmostEqual(float(rows[2]["sample_time_estimate_monotonic_s"]), 99.996)
        self.assertAlmostEqual(float(rows[3]["sample_time_estimate_monotonic_s"]), 100.0)
        self.assertEqual(read_metadata(self.recorder.path)["missing_samples"], 0)

    def test_constructor_accepts_uint32_counter_and_rejects_invalid_moduli(self) -> None:
        recorder = SessionRecorder(Path(self.directory.name), "brainbit", {}, packet_modulus=2**32)
        self.addCleanup(recorder.close, outcome="failed")
        recorder.write_samples(sample_event([2**32 - 1], 100.0))
        recorder.close(outcome="completed")
        self.assertEqual(read_metadata(recorder.path)["packet_modulus"], 2**32)
        for invalid in (True, False, 1, 0, -1, 2**32 + 1, 2048.0, "2048", None):
            with self.subTest(modulus=invalid), self.assertRaises(ValueError):
                SessionRecorder(Path(self.directory.name), "brainbit", {}, packet_modulus=invalid)

    def test_device_configuration_rejects_invalid_format_without_changing_metadata(self) -> None:
        baseline = {"sample_rate_hz": 250, "packet_modulus": 2**32, "channels": list(CHANNELS)}
        cases = [
            {**baseline, "packet_modulus": True},
            {**baseline, "packet_modulus": 1},
            {**baseline, "packet_modulus": 2**32 + 1},
            {**baseline, "sample_rate_hz": 500},
            {**baseline, "sample_rate_hz": "250"},
            {**baseline, "channels": ["T3", "T4", "O1", "O2"]},
            {**baseline, "channels": None},
            {**baseline, "battery": math.nan},
            {},
        ]
        for info in cases:
            with self.subTest(info=info), self.assertRaises(ValueError):
                self.recorder.configure_device(info)
        self.assertEqual(self.recorder.packet_modulus, 2048)
        metadata = read_metadata(self.recorder.path)
        self.assertEqual(metadata["packet_modulus"], 2048)
        self.assertNotIn("device", metadata)

    def test_device_configuration_cannot_change_after_samples_or_close(self) -> None:
        info = {"sample_rate_hz": 250, "packet_modulus": 2**32, "channels": list(CHANNELS)}
        self.recorder.write_samples(sample_event([0], 100.0))
        with self.assertRaises(ValueError):
            self.recorder.configure_device(info)
        self.recorder.close(outcome="completed")
        with self.assertRaises(ValueError):
            self.recorder.configure_device(info)
        self.assertEqual(read_metadata(self.recorder.path)["packet_modulus"], 2048)

    def test_packet_range_uses_exclusive_configured_modulus(self) -> None:
        self.recorder.configure_device({"sample_rate_hz": 250, "packet_modulus": 2**32, "channels": list(CHANNELS)})
        for invalid in (-1, 2**32, True):
            with self.subTest(packet=invalid), self.assertRaises(ValueError):
                self.recorder.write_samples(sample_event([invalid], 100.0))
        self.assertEqual(self.recorder.sample_count, 0)

    def test_gaps_duplicates_and_restart_are_retained_and_counted(self) -> None:
        self.recorder.write_samples(sample_event([100, 101], 10.0))
        self.recorder.write_samples(sample_event([104, 104, 5, 6], 11.0))
        self.recorder.close(outcome="completed")
        rows = read_rows(self.recorder.path)
        self.assertEqual([row["packet"] for row in rows], ["100", "101", "104", "104", "5", "6"])
        self.assertEqual(
            [row["sequence_status"] for row in rows],
            ["start", "ok", "gap", "duplicate", "restart_or_out_of_order", "ok"],
        )
        self.assertEqual([int(row["missing_before"]) for row in rows], [0, 0, 2, 0, 0, 0])
        self.assertEqual([int(row["segment"]) for row in rows], [0, 0, 0, 0, 1, 1])
        for row, expected in zip(rows, [9.996, 10.0, 10.012, 10.012, 10.996, 11.0]):
            self.assertAlmostEqual(float(row["sample_time_estimate_monotonic_s"]), expected)
        metadata = read_metadata(self.recorder.path)
        self.assertEqual(metadata["missing_samples"], 2)
        self.assertEqual(metadata["sequence_anomalies"], 2)
        self.assertEqual(metadata["sample_count"], 6)

    def test_first_batch_gap_uses_counter_span_for_arrival_anchor(self) -> None:
        self.recorder.write_samples(sample_event([100, 103], 10.0))
        self.recorder.close(outcome="completed")
        times = [float(row["sample_time_estimate_monotonic_s"]) for row in read_rows(self.recorder.path)]
        self.assertAlmostEqual(times[0], 9.988)
        self.assertAlmostEqual(times[1], 10.0)

    def test_first_batch_duplicate_keeps_same_estimated_acquisition_time(self) -> None:
        self.recorder.write_samples(sample_event([100, 100], 10.0))
        self.recorder.close(outcome="completed")
        rows = read_rows(self.recorder.path)
        self.assertEqual([float(row["sample_time_estimate_monotonic_s"]) for row in rows], [10.0, 10.0])
        self.assertEqual(rows[1]["sequence_status"], "duplicate")

    def test_receipt_wall_clock_changes_do_not_change_sample_spacing(self) -> None:
        self.recorder.write_samples(sample_event([10], 100.0))
        second = sample_event([11], 100.2)
        second["received_unix_ns"] -= 5_000_000_000
        self.recorder.write_samples(second)
        self.recorder.close(outcome="completed")
        rows = read_rows(self.recorder.path)
        self.assertAlmostEqual(float(rows[1]["sample_time_estimate_monotonic_s"]), 100.004)
        self.assertEqual(int(rows[1]["received_unix_ns"]), second["received_unix_ns"])

    def test_labels_are_explicit_annotations_and_default_is_unlabeled(self) -> None:
        self.recorder.write_samples(sample_event([0], 100.0))
        self.recorder.set_label("left")
        self.recorder.write_samples(sample_event([1], 100.004))
        with self.assertRaises(ValueError):
            self.recorder.set_label("inferred_forward")
        self.recorder.close(outcome="completed")
        self.assertEqual([row["label"] for row in read_rows(self.recorder.path)], ["unlabeled", "left"])
        events = [json.loads(line) for line in (self.recorder.path / "events.jsonl").read_text().splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "annotation")
        self.assertEqual(events[0]["label"], "left")
        self.assertEqual(events[0]["next_sample_index"], 1)

    def test_failed_close_is_idempotent_and_never_becomes_completed(self) -> None:
        self.recorder.close(outcome="failed", error="device disconnected")
        self.recorder.close(outcome="completed")
        metadata = read_metadata(self.recorder.path)
        self.assertFalse(metadata["complete"])
        self.assertEqual(metadata["outcome"], "failed")
        self.assertEqual(metadata["error"], "device disconnected")


class TestPreviewAppLifecycle(unittest.TestCase):
    """Exercise the shared main-thread processor with real preview evaluation."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        with patch("eeg.app.time.monotonic", return_value=100.0):
            self.app = PreviewApp("synthetic", Path(self.directory.name), {"forward": 12.0, "left": 15.0}, {})
        self.addCleanup(self.app.recorder.close, outcome="failed", error="test cleanup")

    def qualify_forward(self) -> None:
        self.app.events.put({"type": "status", "state": "streaming", "message": "Synthetic fixture"})
        self.app.events.put(sample_event(list(range(500)), 100.0))
        with patch("eeg.app.time.monotonic", return_value=100.0):
            self.app.pump()
        for index in range(3):
            first = 500 + index * 50
            now = 100.2 + index * 0.2
            self.app.events.put(sample_event(list(range(first, first + 50)), now, first_index=first))
            with patch("eeg.app.time.monotonic", return_value=now):
                self.app.pump()
        self.assertEqual(self.app.snapshot["command"], "forward")

    def test_user_label_does_not_set_classifier_command(self) -> None:
        self.app.recorder.set_label("right")
        self.qualify_forward()
        self.assertEqual(self.app.recorder.label, "right")
        with patch("eeg.app.time.monotonic", return_value=100.6):
            self.assertEqual(self.app.finish(), 0)
        rows = read_rows(self.app.recorder.path)
        self.assertEqual(len(rows), 650)
        self.assertTrue(all(row["label"] == "right" for row in rows))

    def test_device_configuration_preserves_line_setting_and_records_it(self) -> None:
        for line_frequency in (50.0, None):
            with self.subTest(line_frequency=line_frequency):
                with patch("eeg.app.time.monotonic", return_value=100.0):
                    app = PreviewApp("brainbit", Path(self.directory.name), None, {"line_frequency_hz": line_frequency})
                    try:
                        app.events.put(
                            {
                                "type": "device",
                                "info": {
                                    "sample_rate_hz": 250,
                                    "packet_modulus": 2**32,
                                    "channels": list(CHANNELS),
                                },
                            }
                        )
                        app.pump()
                        self.assertEqual(app.engine.line_frequency_hz, line_frequency)
                        self.assertEqual(app.recorder.metadata["config"]["line_frequency_hz"], line_frequency)
                    finally:
                        app.finish()

    def test_conditioning_preserves_raw_recorded_voltages(self) -> None:
        event = sample_event(list(range(500)), 100.0)
        for index, sample in enumerate(event["samples"]):
            for channel in range(4):
                sample["values_v"][channel] += 0.02 + 0.001 * np.sin(2 * np.pi * 60 * index / 250)
        expected = [list(sample["values_v"]) for sample in event["samples"]]
        self.app.events.put({"type": "status", "state": "streaming", "message": "Conditioning fixture"})
        self.app.events.put(event)
        with patch("eeg.app.time.monotonic", return_value=100.0):
            self.app.pump()
            self.assertEqual(self.app.snapshot["quality"], "interference")
            self.assertEqual(self.app.snapshot["command"], "none")
            self.app.finish()
        rows = read_rows(self.app.recorder.path)
        actual = [[float(row[f"{channel}_v"]) for channel in CHANNELS] for row in rows]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_stale_input_clears_previously_qualified_command(self) -> None:
        self.qualify_forward()
        with patch("eeg.app.time.monotonic", return_value=101.2):
            self.app.pump()
        self.assertEqual(self.app.snapshot["command"], "none")
        self.assertEqual(self.app.snapshot["candidate"], "none")
        self.assertEqual(self.app.snapshot["quality"], "stale")

    def test_disconnect_overrides_qualified_command_and_fails_session(self) -> None:
        self.qualify_forward()
        self.app.events.put({"type": "status", "state": "disconnected", "message": "Headset disconnected"})
        with patch("eeg.app.time.monotonic", return_value=100.6):
            self.app.pump()
            self.assertEqual(self.app.snapshot["command"], "none")
            self.assertEqual(self.app.snapshot["candidate"], "none")
            self.assertTrue(self.app.stop.is_set())
            self.assertEqual(self.app.finish(), 1)
        metadata = read_metadata(self.app.recorder.path)
        self.assertFalse(metadata["complete"])
        self.assertIn("disconnected", metadata["error"])

    def test_finish_waits_for_worker_and_records_its_final_batch(self) -> None:
        def emit_last_batch(events: queue.Queue, stop: threading.Event) -> None:
            stop.wait(2.0)
            events.put(sample_event([42, 43], 100.0))

        self.app.start(emit_last_batch)
        with patch("eeg.app.time.monotonic", return_value=100.0):
            self.assertEqual(self.app.finish(), 0)
        self.assertFalse(self.app.worker.is_alive())
        self.assertEqual([row["packet"] for row in read_rows(self.app.recorder.path)], ["42", "43"])
        self.assertTrue(read_metadata(self.app.recorder.path)["complete"])

    def test_unexpected_worker_exit_disarms_app(self) -> None:
        def exit_without_status(events: queue.Queue, stop: threading.Event) -> None:
            return

        self.app.start(exit_without_status)
        self.app.worker.join(timeout=1.0)
        with patch("eeg.app.time.monotonic", return_value=100.0):
            self.app.pump()
            self.assertTrue(self.app.stop.is_set())
            self.assertEqual(self.app.snapshot["command"], "none")
            self.assertEqual(self.app.finish(), 1)
        self.assertIn("unexpectedly", read_metadata(self.app.recorder.path)["error"])

    def test_worker_exception_is_recorded_as_failure(self) -> None:
        def fail_during_connection(events: queue.Queue, stop: threading.Event) -> None:
            raise RuntimeError("connection refused by test device")

        self.app.start(fail_during_connection)
        self.app.worker.join(timeout=1.0)
        with patch("eeg.app.time.monotonic", return_value=100.0):
            self.assertEqual(self.app.finish(), 1)
        self.assertIn("connection refused", read_metadata(self.app.recorder.path)["error"])

    def test_final_drain_error_records_failed_metadata(self) -> None:
        with patch.object(self.app, "pump", side_effect=OSError("disk full")):
            self.assertEqual(self.app.finish(), 1)
        metadata = read_metadata(self.app.recorder.path)
        self.assertFalse(metadata["complete"])
        self.assertEqual(metadata["outcome"], "failed")
        self.assertIn("disk full", metadata["error"])

    def test_finish_can_be_called_twice_without_writing_closed_files(self) -> None:
        with patch("eeg.app.time.monotonic", return_value=100.0):
            self.assertEqual(self.app.finish(), 0)
        with patch("eeg.app.time.monotonic", return_value=101.0):
            self.assertEqual(self.app.finish(), 0)


if __name__ == "__main__":
    unittest.main()
