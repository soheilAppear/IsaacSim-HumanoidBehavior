# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main-thread processing shared by the EEG desktop panel and headless recorder."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path

from eeg.preview import PreviewEngine
from eeg.session import SessionRecorder


class PreviewApp:
    """Consume acquisition events, record raw input, and expose shadow decisions.

    Args:
        source: Acquisition name saved on every raw sample.
        output: Parent folder for the session.
        targets: Optional command-to-frequency map; absent means monitoring only.
        config: Acquisition arguments saved for reproduction.
    """

    def __init__(self, source: str, output: Path, targets: dict | None, config: dict) -> None:
        self.events: queue.Queue = queue.Queue(maxsize=1024)
        self.stop = threading.Event()
        self.line_frequency_hz = config.get("line_frequency_hz", 60.0)
        self.engine = PreviewEngine(targets=targets, line_frequency_hz=self.line_frequency_hz)
        self.targets = targets
        self.recorder = SessionRecorder(
            output,
            source,
            {**config, "targets_hz": targets, "line_frequency_hz": self.line_frequency_hz},
        )
        self.source = source
        self.state = "starting"
        self.message = "Starting acquisition"
        self.error: str | None = None
        self.resistance: list[float] | None = None
        self.resistance_time: float | None = None
        self.trace: deque = deque(maxlen=500)
        self.snapshot = self.engine.evaluate(time.monotonic())
        self.worker: threading.Thread | None = None
        self._last_preview = 0.0
        self._finished = False
        self._result = 1

    def start(self, target: object, **kwargs: object) -> None:
        """Start the acquisition thread; all file and UI work stays on this thread."""

        def acquire() -> None:
            try:
                target(self.events, self.stop, **kwargs)
            except Exception as error:
                # Thread-boundary containment: SDKs may raise undocumented exception types.
                event = {"type": "status", "state": "error", "message": f"Acquisition worker failed: {error}"}
                try:
                    self.events.put_nowait(event)
                except queue.Full:
                    discarded = self.events.get_nowait()
                    event["discarded_samples"] = len(discarded.get("samples", []))
                    event["message"] += "; full event queue forced one queued event to be discarded"
                    self.events.put_nowait(event)
                self.stop.set()

        self.worker = threading.Thread(target=acquire, name="BrainBit acquisition", daemon=True)
        self.worker.start()

    def pump(self) -> None:
        """Drain a bounded amount of acquisition work and refresh the preview."""
        for _ in range(1024):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event["type"] == "samples":
                self.recorder.write_samples(event)
                values = [sample["values_v"] for sample in event["samples"]]
                packets = [sample["packet"] for sample in event["samples"]]
                self.engine.add_samples(values, packets, event["received_monotonic"])
                self.trace.extend(values)
            else:
                self.recorder.write_event(event)
                if event["type"] == "device":
                    self.recorder.configure_device(event["info"])
                    self.engine = PreviewEngine(
                        targets=self.targets,
                        packet_modulus=event["info"]["packet_modulus"],
                        line_frequency_hz=self.line_frequency_hz,
                    )
                elif event["type"] == "status":
                    self.state, self.message = event["state"], event["message"]
                    if self.state in ("error", "disconnected"):
                        self.error = self.message
                        self.stop.set()
                elif event["type"] == "resistance":
                    self.resistance = event["values_ohm"]
                    self.resistance_time = event["received_monotonic"]
        if self.worker is not None and not self.worker.is_alive() and not self.stop.is_set() and self.events.empty():
            self.error = "Acquisition worker stopped unexpectedly"
            self.state, self.message = "error", self.error
            self.stop.set()
        now = time.monotonic()
        self.snapshot = self.engine.evaluate(now)
        if self.state != "streaming" or self.stop.is_set():
            self.snapshot = {**self.snapshot, "command": "none", "candidate": "none", "reason": self.message}
        if now - self._last_preview >= 0.2:
            self.recorder.write_event({"type": "preview", **self.snapshot, "robot_output_enabled": False})
            self._last_preview = now

    def finish(self) -> int:
        """Stop callbacks, drain their final events, then finalize the recording."""
        if self._finished:
            return self._result
        self._finished = True
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=6.0)
            if self.worker.is_alive():
                self.error = "Acquisition did not stop within 6 seconds; session incomplete"
        try:
            self.pump()
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            self.error = f"Final recording drain failed: {error}"
        finally:
            self.recorder.close(outcome="failed" if self.error else "completed", error=self.error)
        self._result = 1 if self.error else 0
        return self._result
