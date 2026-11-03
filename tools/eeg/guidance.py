# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guide a resting recording by voice without opening the EEG device or Isaac."""

from __future__ import annotations

import csv
import io
import json
import math
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

MAX_FILE_AGE_SECONDS = 5.0


def read_stream_state(session: Path, *, now: float | None = None) -> dict:
    """Read the latest complete CSV row from an active real-device recording.

    The five-second limit accommodates disk buffering; it is not a robot-control
    watchdog or a statement about EEG quality. No acquisition settings change.

    Args:
        session: Existing session directory containing metadata and raw CSV.
        now: Host monotonic time used for file freshness validation.

    Returns:
        Latest sample counter and host receipt time, with the disk-visible age.

    Raises:
        ValueError: If the session is synthetic, ended, malformed, or stale.
        OSError: If recording files cannot be read.
    """
    now = time.monotonic() if now is None else now
    metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("source") != "brainbit"
        or metadata.get("complete") is not False
        or metadata.get("outcome")
        or metadata.get("ended_utc")
    ):
        raise ValueError("An active real BrainBit recording is required")
    with (session / "eeg.csv").open("rb") as stream:
        header_text = stream.readline().decode("utf-8").strip()
        if not header_text:
            raise ValueError("Recording has no CSV header yet")
        header = next(csv.reader([header_text]))
        body_start = stream.tell()
        stream.seek(0, 2)
        size = stream.tell()
        start = max(body_start, size - 65536)
        stream.seek(start)
        tail = stream.read(size - start)
    if start > body_start:
        tail = tail[tail.find(b"\n") + 1 :]
    end = tail.rfind(b"\n")
    if end < 0:
        raise ValueError("Recording has no complete sample rows yet")
    rows = list(csv.reader(io.StringIO(tail[: end + 1].decode("utf-8"))))
    if not rows or len(rows[-1]) != len(header):
        raise ValueError("Recording has no valid final sample row")
    row = dict(zip(header, rows[-1]))
    try:
        received = float(row["received_monotonic_s"])
        sample_index = int(row["sample_index"])
        packet = int(row["packet"])
        source = row["source"]
    except (KeyError, ValueError) as error:
        raise ValueError("Recording sample clock or counter is invalid") from error
    age = now - received
    if (
        source != "brainbit"
        or not math.isfinite(age)
        or not 0 <= age <= MAX_FILE_AGE_SECONDS
        or min(sample_index, packet) < 0
    ):
        raise ValueError("Recording is stale, has an invalid clock, or is not real BrainBit data")
    return {"sample_index": sample_index, "packet": packet, "received_monotonic_s": received, "file_sample_age_s": age}


def find_live_session(root: Path) -> Path:
    """Find exactly one fresh real-device recording, requiring explicit selection otherwise."""
    candidates = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            read_stream_state(path)
        except (OSError, ValueError, StopIteration):
            continue
        candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"Expected one active BrainBit session; found {len(candidates)}. Use --session to select one")
    return candidates[0].resolve()


def run_rest(
    session: Path,
    output_root: Path,
    speaker: Callable[[str], dict],
    *,
    duration: float = 30.0,
    settle: float = 10.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Speak instructions, mark a quiet interval, then announce completion.

    Markers go to a separate local log; the running recorder is never edited or
    restarted. Speech timestamps measure host calls, not acoustic onset. Rest is
    an instruction, not verified stillness or a trained brain command. The caller
    must first confirm that the operator can hear the chosen audio output.

    Args:
        session: Pinned active BrainBit session.
        output_root: Parent directory for a unique guidance event log.
        speaker: Blocking speech function that returns playback metadata.
        duration: Requested quiet interval in seconds, from 5 through 300.
        settle: Preparation delay in seconds, from 0 through 60.
        clock: Monotonic host clock shared with the recorder.
        sleep: Wait function for polling and test injection.

    Returns:
        Directory containing the separate guidance event log.

    Raises:
        ValueError: If arguments or the recording are invalid.
        OSError: If logging fails or a cancellation file appears.
        RuntimeError: If speech fails or recorded samples do not advance.
    """
    for value, minimum, maximum, name in ((duration, 5, 300, "Duration"), (settle, 0, 60, "Settle delay")):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite numeric seconds")
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum} seconds")
    session = session.resolve()
    initial = read_stream_state(session, now=clock())
    previous = initial
    path = output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    print(f"Guidance log: {path.resolve()}", flush=True)
    with (path / "events.jsonl").open("x", encoding="utf-8") as log:

        def write_event(kind: str, **fields: object) -> None:
            """Flush a host-timestamped event without touching the acquisition files."""
            event = {"type": kind, "monotonic_s": clock(), "unix_ns": time.time_ns(), **fields}
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()

        def check_stream() -> dict:
            """Reject cancellation, stale data, and replacement of the pinned stream."""
            nonlocal previous
            if (path / "cancel").exists():
                raise InterruptedError("Voice guidance cancelled")
            state = read_stream_state(session, now=clock())
            if (
                state["sample_index"] < previous["sample_index"]
                or state["received_monotonic_s"] < previous["received_monotonic_s"]
            ):
                raise RuntimeError("Recording sample counter or receipt clock moved backwards")
            previous = state
            return state

        def wait_until(deadline: float) -> None:
            """Keep checking the recording throughout a silent interval."""
            while clock() < deadline:
                check_stream()
                sleep(min(0.25, max(0.0, deadline - clock())))
            check_stream()

        def announce(message: str) -> None:
            """Log synchronous speech outside the measured quiet interval."""
            check_stream()
            write_event("speech_requested", text=message)
            playback = speaker(message)
            write_event("speech_completed", playback=playback)
            check_stream()

        write_event(
            "started",
            session=str(session),
            initial_sample=initial,
            duration_s=duration,
            settle_s=settle,
            robot_output_enabled=False,
            actual_stillness_verified=False,
            timing_note="Host software clocks only; audio onset and EEG acquisition are not hardware synchronized.",
        )
        try:
            announce(
                f"Keep your headset on. Look at a fixed point and leave the controllers at rest. "
                f"The quiet check lasts {duration:g} seconds. It will begin after a {settle:g} second preparation pause. "
                "Wait for the start instruction."
            )
            wait_until(clock() + settle)
            announce("Stay still and keep looking at the same point. The quiet interval starts now.")
            started = clock()
            write_event("interval_started", requested_label="rest", sample=check_stream())
            wait_until(started + duration)
            finished = clock()
            write_event("interval_finished", elapsed_s=finished - started, sample=check_stream())
            flush_deadline = clock() + MAX_FILE_AGE_SECONDS
            while check_stream()["received_monotonic_s"] < finished:
                if clock() >= flush_deadline:
                    raise RuntimeError("Recording did not flush samples covering the quiet interval")
                sleep(0.25)
            announce("The quiet check is finished. You can relax and keep the headset on while I check the recording.")
            write_event("completed", final_sample=check_stream())
        except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
            write_event("aborted", error=str(error) or type(error).__name__)
            try:
                speaker("The check has stopped. You can relax. The recording needs attention.")
            except (OSError, ValueError, RuntimeError):
                pass
            raise
    return path
