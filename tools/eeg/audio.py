# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speak offline Windows SAPI cues through an explicitly selected audio output.

The companion script receives UTF-8 JSON on stdin. Speech text is never inserted
into shell commands or interpreted as XML. Volume applies to this voice instance;
the backend does not change the OS default endpoint or its volume. Returned
timestamps describe software calls, not measured sound onset or user audibility.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta
from numbers import Real
from pathlib import Path

__all__ = ["AudioBackendError", "list_audio_outputs", "speak"]


class AudioBackendError(RuntimeError):
    """Indicate unavailable output, failed speech, or an invalid backend response."""


def _powershell_executable() -> Path:
    if sys.platform != "win32":
        raise AudioBackendError("Offline SAPI audio requires Windows")
    executable = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    executable = executable / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if not executable.is_file():
        raise AudioBackendError("Windows PowerShell was not found in the Windows system directory")
    return executable


def _run_backend(request: dict, timeout: float) -> dict:
    if (
        not isinstance(timeout, Real)
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or not 0 < timeout <= 600
    ):
        raise ValueError("Audio timeout must be finite, positive, and at most 600 seconds")
    executable = _powershell_executable()
    script = Path(__file__).with_name("speak.ps1").resolve()
    if not script.is_file():
        raise AudioBackendError("The companion SAPI script is missing")
    arguments = [
        str(executable),
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
    ]
    try:
        process = subprocess.run(
            arguments,
            input=json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            timeout=float(timeout),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise AudioBackendError("Audio backend timed out; successful playback completion is unconfirmed") from error
    except OSError as error:
        raise AudioBackendError(f"Could not start the offline audio backend: {error}") from error
    try:
        response = json.loads(process.stdout.decode("utf-8-sig"))
    except (UnicodeError, ValueError, TypeError, AttributeError) as error:
        detail = process.stderr.decode("utf-8", errors="replace").strip()[:1000]
        raise AudioBackendError(f"Invalid audio backend response (exit {process.returncode}): {detail}") from error
    if not isinstance(response, dict) or response.get("ok") is not True or process.returncode != 0:
        detail = response.get("error") if isinstance(response, dict) else None
        if not isinstance(detail, str) or not detail:
            detail = f"Audio backend failed with exit code {process.returncode}"
        raise AudioBackendError(detail)
    return response


def _validate_device(value: object) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("id"), str)
        or not value["id"]
        or not isinstance(value.get("name"), str)
        or not value["name"]
    ):
        raise AudioBackendError("Audio backend returned invalid device metadata")
    return {"id": value["id"], "name": value["name"]}


def list_audio_outputs(*, timeout: float = 10.0) -> list[dict[str, str]]:
    """List the exact SAPI output token IDs and their display names.

    This operation neither plays audio nor changes the default output device.

    Args:
        timeout: Maximum backend process duration in seconds.

    Returns:
        List of device dictionaries with id and name strings. An empty list means
        SAPI exposed no audio outputs.

    Raises:
        ValueError: If timeout is invalid.
        AudioBackendError: If enumeration fails or its response is malformed.
    """
    response = _run_backend({"action": "list"}, timeout)
    outputs = response.get("outputs")
    if not isinstance(outputs, list):
        raise AudioBackendError("Audio backend returned an invalid output list")
    devices = [_validate_device(output) for output in outputs]
    if len({device["id"] for device in devices}) != len(devices):
        raise AudioBackendError("Audio backend returned duplicated output IDs")
    return devices


def speak(
    text: str,
    device_id: str | None = None,
    *,
    volume: int = 60,
    timeout: float = 90.0,
) -> dict:
    """Speak plain text synchronously using an offline Windows SAPI voice.

    The device ID must exactly match an enumerated token; an unavailable ID
    fails without falling back. Omitting it uses SAPI's current default output.
    The volume is a per-voice percentage and does not alter the OS volume.
    Completion means SAPI's synchronous call returned, not that the user heard
    the cue. No hardware onset timing is measured.

    Args:
        text: Nonempty plain text, at most 4000 characters, without NUL characters.
        device_id: Exact output token ID, or None for the SAPI default.
        volume: Integer per-voice volume from 0 through 100.
        timeout: Maximum backend process duration in seconds.

    Returns:
        Dictionary containing selected_device, volume, host_play_requested_utc,
        host_play_completed_utc, and timing_basis. UTC timestamps surround the
        synchronous SAPI call inside the backend process.

    Raises:
        ValueError: If text, device ID, volume, or timeout is invalid.
        AudioBackendError: If the output is unavailable, playback fails or times
            out, or the backend response is malformed.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 4000 or "\0" in text:
        raise ValueError("Speech text must contain 1 to 4000 characters and no NUL")
    if device_id is not None and (not isinstance(device_id, str) or not device_id or "\0" in device_id):
        raise ValueError("Audio device ID must be a nonempty string without NUL, or None")
    if not isinstance(volume, int) or isinstance(volume, bool) or not 0 <= volume <= 100:
        raise ValueError("Speech volume must be an integer from 0 through 100")
    response = _run_backend({"action": "speak", "text": text, "device_id": device_id, "volume": volume}, timeout)
    selected = _validate_device(response.get("selected_device"))
    if device_id is not None and selected["id"] != device_id:
        raise AudioBackendError("Audio backend did not confirm the exact requested output ID")
    if type(response.get("volume")) is not int or response["volume"] != volume:
        raise AudioBackendError("Audio backend did not confirm the requested voice volume")
    times = []
    for key in ("host_play_requested_utc", "host_play_completed_utc"):
        try:
            value = response[key]
            if not isinstance(value, str):
                raise ValueError("Timestamp must be text")
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if timestamp.utcoffset() != timedelta(0):
                raise ValueError("Timestamp must include UTC timezone")
            times.append(timestamp)
        except (KeyError, ValueError, TypeError) as error:
            raise AudioBackendError(f"Audio backend returned invalid {key}") from error
    if times[1] < times[0]:
        raise AudioBackendError("Audio backend timestamps reversed; completion timing is unconfirmed")
    return {
        "selected_device": selected,
        "volume": volume,
        "host_play_requested_utc": response["host_play_requested_utc"],
        "host_play_completed_utc": response["host_play_completed_utc"],
        "timing_basis": "Host synchronous SAPI call; no measured hardware onset or audibility",
    }
