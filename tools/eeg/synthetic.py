# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic artificial EEG for the preview, never presented as a brain recording."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

DEMO_TARGETS = {"forward": 12.0, "left": 15.0, "right": 20.0}


def run_synthetic(events: queue.Queue, stop: threading.Event) -> None:
    """Generate paced 250 Hz signal batches cycling rest/forward/left/right."""
    rng = np.random.default_rng(42)
    index = 0
    origin = time.monotonic()
    events.put({"type": "status", "state": "streaming", "message": "SYNTHETIC demo — no headset connected"})
    while not stop.is_set():
        elapsed = time.monotonic() - origin
        available = int(elapsed * 250) - index
        if available < 10:
            stop.wait(0.01)
            continue
        count = min(available, 25)
        samples = []
        for sample_index in range(index, index + count):
            phase = (sample_index // 1250) % 4
            frequency = (0.0, 12.0, 15.0, 20.0)[phase]
            values = rng.normal(0, 3e-6, 4)
            if frequency:
                carrier = np.sin(2 * np.pi * frequency * sample_index / 250.0)
                values += carrier * np.array([25e-6, 22e-6, 5e-6, 4e-6])
            samples.append({"packet": sample_index % 2048, "marker": 0, "values_v": values.tolist()})
        try:
            events.put(
                {
                    "type": "samples",
                    "received_monotonic": time.monotonic(),
                    "received_unix_ns": time.time_ns(),
                    "samples": samples,
                },
                timeout=0.2,
            )
        except queue.Full as error:
            raise RuntimeError(f"Synthetic event queue overflow; {len(samples)} samples were not queued") from error
        index += count
