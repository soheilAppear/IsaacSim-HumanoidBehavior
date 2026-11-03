# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record BrainBit EEG and preview experimental commands without robot actuation.

Examples:
    python tools/brainbit_teleop.py --source synthetic
    python tools/brainbit_teleop.py --source synthetic --headless --duration 20
    python tools/brainbit_teleop.py --source brainbit --address AA:BB:CC:DD:EE:FF

Live mode monitors only unless --target NAME=HZ is supplied for external, timed
SSVEP stimuli. The panel does not generate flicker or train an intent classifier.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_targets(values: list[str]) -> dict[str, float] | None:
    """Parse explicit target-frequency pairs without allowing duplicate commands."""
    targets = {}
    for value in values:
        try:
            name, frequency = value.split("=", maxsplit=1)
            if name in targets:
                raise ValueError(f"Duplicate target: {name}")
            targets[name] = float(frequency)
        except ValueError as error:
            raise ValueError(f"Invalid target {value!r}; use forward=12, left=15, etc.") from error
    return targets or None


def main(argv: list[str] | None = None) -> int:
    """Run a live or synthetic session with guaranteed recording finalization."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("brainbit", "synthetic"), required=True)
    parser.add_argument("--address", help="Select one headset by its Bluetooth address")
    parser.add_argument("--scan-seconds", type=float, default=8.0)
    parser.add_argument("--resistance-seconds", type=float, default=3.0)
    parser.add_argument(
        "--duration", type=float, default=0.0, help="Wall-clock seconds including setup; 0 runs until stopped"
    )
    parser.add_argument("--headless", action="store_true", help="Record with console status and no desktop window")
    parser.add_argument("--output", type=Path, default=ROOT / ".eeg" / "sessions")
    parser.add_argument(
        "--line-frequency",
        choices=("50", "60", "off"),
        default="60",
        help="Mains-frequency fit for preview diagnostics; raw EEG is always recorded unchanged",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="COMMAND=HZ",
        help="Optional external-stimulus frequency; repeat for 2+ targets. Live default: no decoder",
    )
    args = parser.parse_args(argv)
    if any(
        not math.isfinite(value) or value < 0 for value in (args.duration, args.scan_seconds, args.resistance_seconds)
    ):
        parser.error("Durations must be finite and nonnegative")
    if args.scan_seconds == 0 or args.resistance_seconds == 0:
        parser.error("Scan and contact-check durations must be positive")
    if args.source == "synthetic" and args.address:
        parser.error("--address applies only to a real BrainBit source")
    try:
        from eeg.app import PreviewApp
        from eeg.synthetic import DEMO_TARGETS, run_synthetic

        targets = parse_targets(args.target)
        if args.source == "synthetic" and not targets:
            targets = DEMO_TARGETS
        app = PreviewApp(
            args.source,
            args.output,
            targets,
            {
                "scan_seconds": args.scan_seconds,
                "resistance_seconds": args.resistance_seconds,
                "line_frequency_hz": None if args.line_frequency == "off" else float(args.line_frequency),
            },
        )
    except ImportError as error:
        parser.exit(2, f"Missing dependency: {error}. Run py -3.11 tools/brainbit_setup.py first.\n")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Cannot start session: {error}\n")
    print(f"Session: {app.recorder.path}")
    print(f"Source: {args.source.upper()}. Preview only; no robot output.")
    result = 1
    try:
        if args.source == "brainbit":
            from eeg.brainbit import run_brainbit

            app.start(
                run_brainbit,
                address=args.address,
                scan_seconds=args.scan_seconds,
                resistance_seconds=args.resistance_seconds,
            )
        else:
            app.start(run_synthetic)
        if args.headless:
            started = time.monotonic()
            last_status = 0.0
            while not app.stop.is_set():
                app.pump()
                now = time.monotonic()
                if now - last_status >= 1.0:
                    print(
                        f"{app.state}: {app.recorder.sample_count} samples | "
                        f"{app.snapshot['quality']} | {app.snapshot['command']} | {app.snapshot['reason']}"
                    )
                    last_status = now
                if args.duration > 0 and now - started >= args.duration:
                    break
                if app.worker is not None and not app.worker.is_alive():
                    if not app.stop.is_set():
                        app.error = "Acquisition worker stopped unexpectedly"
                    break
                app.stop.wait(0.04)
        else:
            from eeg.panel import run_panel

            run_panel(app, args.duration)
    except KeyboardInterrupt:
        print("Stopping recording")
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        app.error = str(error)
    finally:
        try:
            result = app.finish()
        except (OSError, ValueError) as error:
            print(f"Recording could not be finalized: {error}", file=sys.stderr)
            result = 1
    if app.error:
        print(f"Acquisition failed: {app.error}", file=sys.stderr)
    print(f"Saved {app.recorder.sample_count} samples to {app.recorder.path}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
