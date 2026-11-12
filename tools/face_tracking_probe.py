# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect Steam Link facial signals locally without starting or moving Isaac.

In SteamVR's advanced Steam Link settings enable OSC, eye sharing, face sharing,
and the 9015 Custom/ALT output. Enable eye and face tracking and permissions on
the Quest Pro. Run this probe before the facial Isaac launcher, then let it exit
to release the exclusive port. Close other receivers such as VRCFaceTracking.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
    / "face_tracking_input.py"
)


def main(argv: list[str] | None = None) -> int:
    """Report current weights and aggregate ranges, returning nonzero if no valid input arrives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15.0, help="Local observation duration")
    parser.add_argument("--port", type=int, default=9015, help="Steam Link custom OSC output port")
    parser.add_argument("--interval", type=float, default=0.5, help="Console update interval")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--json-output", type=Path, help="Optional local summary JSON, without per-frame capture")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0.1 <= args.seconds <= 3600:
        parser.error("Seconds must be between 0.1 and 3600")
    if not math.isfinite(args.interval) or not 0.05 <= args.interval <= 60:
        parser.error("Interval must be between 0.05 and 60")
    spec = importlib.util.spec_from_file_location("face_tracking_probe_input", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        receiver = module.SteamLinkFaceInput(port=args.port, min_confidence=args.min_confidence)
        receiver.start()
    except (OSError, ValueError) as error:
        print(f"Face receiver could not start: {error}", file=sys.stderr)
        print("Close Isaac's facial receiver and VRCFaceTracking before running this probe.", file=sys.stderr)
        return 2
    print(f"Listening only on 127.0.0.1:{receiver.port}. No robot commands are sent.", flush=True)
    start = time.monotonic()
    next_print = start
    valid_polls = 0
    total_polls = 0
    ranges: dict[str, list[float]] = {}
    confidence_ranges: dict[str, list[float]] = {}
    final_sample = receiver.poll()
    try:
        while time.monotonic() - start < args.seconds:
            now = time.monotonic()
            final_sample = receiver.poll(now_s=now)
            total_polls += 1
            if final_sample.valid:
                valid_polls += 1
                for name, value in final_sample.weights.items():
                    bounds = ranges.setdefault(name, [value, value])
                    bounds[0], bounds[1] = min(bounds[0], value), max(bounds[1], value)
                for name, value in (
                    ("upper", final_sample.confidence_upper),
                    ("lower", final_sample.confidence_lower),
                ):
                    bounds = confidence_ranges.setdefault(name, [value, value])
                    bounds[0], bounds[1] = min(bounds[0], value), max(bounds[1], value)
            if now >= next_print:
                print(
                    json.dumps(
                        {
                            "elapsed_s": round(now - start, 2),
                            "valid": final_sample.valid,
                            "status": final_sample.reason,
                            "upper_confidence": round(final_sample.confidence_upper, 3),
                            "lower_confidence": round(final_sample.confidence_lower, 3),
                            "weights": {name: round(value, 3) for name, value in final_sample.weights.items()},
                        }
                    ),
                    flush=True,
                )
                next_print = now + args.interval
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Probe interrupted. Releasing the local OSC port.", flush=True)
    finally:
        diagnostics = receiver.diagnostics()
        receiver.close()
    report = {
        "duration_s": round(time.monotonic() - start, 3),
        "valid_polls": valid_polls,
        "total_polls": total_polls,
        "expression_ranges": ranges,
        "confidence_ranges": confidence_ranges,
        "last_status": final_sample.reason,
        "transport": diagnostics,
        "limitation": "Local receipt freshness does not prove hardware capture freshness or gesture reliability.",
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if valid_polls else 1


if __name__ == "__main__":
    raise SystemExit(main())
