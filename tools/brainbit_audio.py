# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speak local EEG instructions and guide a quiet recording while the user wears VR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg.audio import list_audio_outputs, speak
from eeg.guidance import find_live_session, run_rest

ROOT = Path(__file__).resolve().parents[1]


def select_device(value: str | None) -> str | None:
    """Resolve an exact output name or SAPI token ID without changing Windows settings."""
    if value is None:
        return None
    matches = [item for item in list_audio_outputs() if value in (item["id"], item["name"])]
    if len(matches) != 1:
        raise ValueError("Audio output must match exactly one name or ID from the devices command")
    return matches[0]["id"]


def main(argv: list[str] | None = None) -> int:
    """Run audio discovery, a spoken message, or a voice-guided resting interval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="Exact output name or ID; omitted uses the current Windows speech default")
    parser.add_argument("--volume", type=int, default=60, help="Speech volume 1–100; does not change system volume")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("devices", help="List the available speech audio outputs")
    commands.add_parser("test", help="Play a short audibility check before using timed guidance")
    say = commands.add_parser("say", help="Speak a literal message locally")
    say.add_argument("text")
    rest = commands.add_parser("rest", help="Guide rest with spoken start/end cues; requires an existing live recorder")
    rest.add_argument("--session", type=Path, help="Explicit active session folder; otherwise require exactly one")
    rest.add_argument("--seconds", type=float, default=30)
    rest.add_argument("--settle", type=float, default=10)
    args = parser.parse_args(argv)
    if not 1 <= args.volume <= 100:
        parser.error("Volume must be between 1 and 100")
    try:
        if args.command == "devices":
            print(json.dumps(list_audio_outputs(), indent=2))
            return 0
        device = select_device(args.device)

        def speak_message(message: str) -> dict:
            """Use the chosen output for this invocation only."""
            return speak(message, device_id=device, volume=args.volume)

        if args.command == "test":
            print(json.dumps(speak_message("Headset audio test. If you hear this, tell me audio works."), indent=2))
        elif args.command == "say":
            print(json.dumps(speak_message(args.text), indent=2))
        else:
            session = args.session or find_live_session(ROOT / ".eeg" / "sessions")
            path = run_rest(
                session,
                ROOT / ".eeg" / "guidance",
                speak_message,
                duration=args.seconds,
                settle=args.settle,
            )
            print(f"Saved voice guidance markers: {path}")
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        parser.exit(1, f"Audio guidance stopped: {error or type(error).__name__}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
