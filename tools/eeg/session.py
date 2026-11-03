# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss-auditable EEG CSV recording with host clock anchors and user markers."""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

CHANNELS = ("O1", "O2", "T3", "T4")


class SessionRecorder:
    """Write a new recording directory without overwriting previous sessions.

    Args:
        output_root: Parent directory for unique session folders.
        source: Explicit acquisition source, saved in metadata and every raw row.
        config: Reproduction settings such as target frequencies and sample rate.
        packet_modulus: Exclusive packet-counter limit for the acquisition device.
    """

    def __init__(self, output_root: Path, source: str, config: dict, *, packet_modulus: int = 2048) -> None:
        self.packet_modulus = self._validate_packet_modulus(packet_modulus)
        self.started_monotonic = time.monotonic()
        self.started_unix_ns = time.time_ns()
        self.source = source
        self.path = output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        self._raw = (self.path / "eeg.csv").open("x", newline="", encoding="utf-8")
        try:
            self._events = (self.path / "events.jsonl").open("x", encoding="utf-8")
        except OSError:
            self._raw.close()
            raise
        self._writer = csv.writer(self._raw)
        self._writer.writerow(
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
                *[f"{channel}_v" for channel in CHANNELS],
            ]
        )
        self.sample_count = 0
        self.missing_samples = 0
        self.sequence_anomalies = 0
        self.label = "unlabeled"
        self._last_packet: int | None = None
        self._sample_time = 0.0
        self._segment = 0
        self._last_flush = self.started_monotonic
        self._closed = False
        self.metadata = {
            "schema_version": 1,
            "source": source,
            "sample_rate_hz": 250,
            "packet_modulus": self.packet_modulus,
            "channels": CHANNELS,
            "raw_unit": "V",
            "robot_output_enabled": False,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "host_clock_anchor": {"monotonic_s": self.started_monotonic, "unix_ns": self.started_unix_ns},
            "timing_note": "Sample times are estimates from a 250 Hz counter anchored to first batch arrival. "
            "They are not hardware timestamps or verified robot synchronization. "
            "Each callback retains its host receipt clocks; gaps are retained, never interpolated.",
            "config": config,
            "complete": False,
        }
        self._write_metadata()

    def configure_device(self, info: dict) -> None:
        """Save the validated acquisition format before accepting any raw samples.

        Args:
            info: JSON-compatible device details containing sample_rate_hz,
                packet_modulus, and canonical O1/O2/T3/T4 channel ordering.

        Raises:
            ValueError: If recording has begun, is closed, or the format is unsupported.
        """
        if self._closed or self.sample_count:
            raise ValueError("Device configuration must precede all recorded samples")
        if not isinstance(info, dict):
            raise ValueError("Device configuration must be a dictionary")
        modulus = self._validate_packet_modulus(info.get("packet_modulus"))
        sample_rate = info.get("sample_rate_hz")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)) or sample_rate != 250:
            raise ValueError("Only 250 Hz acquisition is supported")
        channels = info.get("channels")
        if not isinstance(channels, (list, tuple)) or tuple(channels) != CHANNELS:
            raise ValueError("Device channels must be ordered O1, O2, T3, T4")
        try:
            device = json.loads(json.dumps(info, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("Device details must contain finite JSON-compatible values") from error
        self.packet_modulus = modulus
        self.metadata.update(device=device, packet_modulus=modulus)
        self._write_metadata()

    def write_samples(self, event: dict) -> None:
        """Append raw rows, preserving counter wrap, gaps, duplicates and restarts."""
        samples = event["samples"]
        arrival = float(event["received_monotonic"])
        unix_ns = int(event["received_unix_ns"])
        if not math.isfinite(arrival):
            raise ValueError("Nonfinite sample receipt timestamp")
        for offset, sample in enumerate(samples):
            packet = sample["packet"]
            values = sample["values_v"]
            if type(packet) is not int or not 0 <= packet < self.packet_modulus or len(values) != 4:
                raise ValueError("Malformed four-channel EEG packet")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError("Nonfinite raw EEG value")
            missing = 0
            status = "ok"
            if self._last_packet is None:
                self._sample_time = arrival - self._remaining_ticks(samples, offset) / 250.0
                status = "start"
            else:
                delta = (packet - self._last_packet) % self.packet_modulus
                if delta == 0:
                    status = "duplicate"
                    self.sequence_anomalies += 1
                elif delta > self.packet_modulus // 2:
                    self._segment += 1
                    self._sample_time = arrival - self._remaining_ticks(samples, offset) / 250.0
                    status = "restart_or_out_of_order"
                    self.sequence_anomalies += 1
                else:
                    missing = delta - 1
                    self.missing_samples += missing
                    self._sample_time += delta / 250.0
                    if missing:
                        status = "gap"
            self._writer.writerow(
                [
                    self.source,
                    self.sample_count,
                    self._segment,
                    packet,
                    sample.get("marker", 0),
                    f"{self._sample_time:.9f}",
                    f"{arrival:.9f}",
                    unix_ns,
                    missing,
                    status,
                    self.label,
                    *values,
                ]
            )
            self.sample_count += 1
            self._last_packet = packet
        self.flush_if_due()

    def write_event(self, event: dict) -> None:
        """Save a status, resistance snapshot, annotation or preview decision."""
        self._events.write(json.dumps({"recorded_monotonic": time.monotonic(), **event}, allow_nan=False) + "\n")
        self.flush_if_due()

    def set_label(self, label: str) -> None:
        """Mark task intent manually; labels never set decoder output."""
        if label not in ("unlabeled", "rest", "forward", "left", "right", "stop"):
            raise ValueError("Unknown task label")
        self.label = label
        self.write_event({"type": "annotation", "label": label, "next_sample_index": self.sample_count})

    def flush_if_due(self) -> None:
        """Flush buffered records at least once per second while processing events."""
        if time.monotonic() - self._last_flush >= 1.0:
            self._raw.flush()
            self._events.flush()
            self._last_flush = time.monotonic()

    def close(self, *, outcome: str, error: str | None = None) -> None:
        """Finalize once, marking interrupted or failed sessions explicitly."""
        if self._closed:
            return
        self._closed = True
        try:
            self._raw.close()
        finally:
            self._events.close()
        self.metadata.update(
            complete=outcome == "completed",
            outcome=outcome,
            error=error,
            sample_count=self.sample_count,
            missing_samples=self.missing_samples,
            sequence_anomalies=self.sequence_anomalies,
            ended_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._write_metadata()

    def _write_metadata(self) -> None:
        temporary = self.path / "metadata.json.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.path / "metadata.json")

    def _remaining_ticks(self, samples: list[dict], offset: int) -> int:
        ticks = 0
        for previous, following in zip(samples[offset:], samples[offset + 1 :]):
            delta = (following["packet"] - previous["packet"]) % self.packet_modulus
            ticks += delta if delta <= self.packet_modulus // 2 else 1
        return ticks

    @staticmethod
    def _validate_packet_modulus(value: object) -> int:
        if type(value) is not int or not 2 <= value <= 2**32:
            raise ValueError("Packet modulus must be an integer from 2 through 2**32")
        return value
