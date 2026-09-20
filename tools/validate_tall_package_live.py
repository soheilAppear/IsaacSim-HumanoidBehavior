#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attempt one real contact lift of a tall package in the loaded walking scene.

First walk the G1 beside the package with its right hand facing the front handle.
The measured arm span must cover the approach and lift targets. This replay does
not teleport the robot or package, reload the scene, constrain the base, change
gains, or attach a grasp joint. GR00T balance remains active while navigation
receives open/no-point input. Only the right optical hand is synthesized; eye
tracking and the real left-hand device are retained.

The default grasp endpoint is an experimental handle profile, not a validated
pickup. Success requires measured opposing contacts, at least 2.5 cm sustained
carton AND handle lift, and at least 1.5 cm drop after opening. Failed attempts
remain failures in the JSON report. Every exit restores input methods, zeros the
walking request, and leaves the timeline paused. Overall wall time is bounded.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import uuid
from pathlib import Path

from kit_exec import DEFAULT_HOST, DEFAULT_PORT, execute
from validate_contact_grasp_live import build_live_code


def make_config(
    *,
    object_root: str,
    output: Path,
    timeout: float = 300.0,
    offset: tuple[float, float, float] = (0.0, 0.008, 0.012),
    curl: float = 0.8,
    thumb_curl: float = 0.55,
    opposition: float = 0.5,
    lift: float = 0.07,
    hold: float = 1.0,
    trace_contact_steps: bool = False,
) -> dict:
    """Validate bounded replay parameters before contacting Isaac Sim.

    Args:
        object_root: Exact registered package root with handle metadata.
        output: Local JSON progress and final result path.
        timeout: Overall wall deadline including cleanup, in seconds.
        offset: Small world-space correction to the measured grasp feature.
        curl: Four-finger closure endpoint in normalized units.
        thumb_curl: Thumb flexion endpoint in normalized units.
        opposition: Thumb opposition endpoint in normalized units.
        lift: Requested vertical hand travel in metres.
        hold: Required hold duration in simulated seconds.
        trace_contact_steps: Include packed contact observations for diagnosis.

    Returns:
        Trusted configuration for the shared contact replay.

    Raises:
        ValueError: An input is invalid or exceeds this replay's bounds.

    Example:

    .. code-block:: python

        config = make_config(object_root="/World/G1_SampleBoxes/Box_10", output=Path("attempt.json"))
        print(config["min_lift"])  # 0.025
    """
    if not isinstance(object_root, str) or not re.fullmatch(r"(?:/[A-Za-z_][A-Za-z0-9_]*)+", object_root):
        raise ValueError("Object root must be an absolute USD prim path")
    values = (timeout, *offset, curl, thumb_curl, opposition, lift, hold)
    if len(offset) != 3 or any(isinstance(value, bool) or not math.isfinite(value) for value in values):
        raise ValueError("Require three finite offsets and finite numeric endpoints")
    if not 30.0 <= timeout <= 600.0:
        raise ValueError("Timeout must be between 30 and 600 seconds")
    if math.hypot(*offset) > 0.08:
        raise ValueError("Handle offset must remain within 8 cm of the measured grasp feature")
    if not all(0.0 <= value <= 1.0 for value in (curl, thumb_curl, opposition)):
        raise ValueError("Finger endpoints must be normalized to [0, 1]")
    if not 0.03 <= lift <= 0.12 or not 1.0 <= hold <= 3.0:
        raise ValueError("Require lift between 3 and 12 cm and hold between 1 and 3 seconds")
    profile = {
        "offset": tuple(float(value) for value in offset),
        "curl": float(curl),
        "thumb_curl": float(thumb_curl),
        "opposition": float(opposition),
        "base_profile": "experimental_right_horizontal_handle",
        "side": "right",
        "overrides": ["offset", "curl", "thumb_curl", "opposition"],
    }
    return {
        "run_id": uuid.uuid4().hex,
        "fixture": "tall_package",
        "timeout": float(timeout),
        "trace_contact_steps": bool(trace_contact_steps),
        "side": "right",
        "paths": [object_root],
        "profiles": {object_root: profile},
        "finger_direction": (1.0, 0.0, 0.0),
        "palm_normal": (0.0, 0.0, -1.0),
        "lift": float(lift),
        "min_lift": 0.025,
        "min_release_drop": 0.015,
        "hold": float(hold),
        "output": str(output.resolve()),
    }


def main() -> int:
    """Launch a bounded simulator task and poll its versioned JSON report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--object-root", default="/World/G1_SampleBoxes/Box_10")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--offset", nargs=3, type=float, default=(0.0, 0.008, 0.012), metavar=("X", "Y", "Z"))
    parser.add_argument("--curl", type=float, default=0.8)
    parser.add_argument("--thumb-curl", type=float, default=0.55)
    parser.add_argument("--opposition", type=float, default=0.5)
    parser.add_argument("--lift", type=float, default=0.07)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--trace-contact-steps", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "_compat61/tall-package-live.json"
    )
    args = parser.parse_args()
    try:
        config = make_config(**{key: value for key, value in vars(args).items() if key not in ("host", "port")})
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = execute(build_live_code(config), args.host, args.port, min(30.0, args.timeout))
    if status:
        return status
    last_phase = None
    while time.monotonic() - started < args.timeout:
        try:
            report = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
        if report.get("run_id") == config["run_id"]:
            if report.get("phase") != last_phase:
                last_phase = report.get("phase")
                print("Phase: " + str(last_phase), flush=True)
            if report.get("status") != "running":
                print(json.dumps(report, indent=2, sort_keys=True))
                return 0 if report["status"] == "passed" else 1
        time.sleep(0.2)
    print("Host deadline elapsed; inspect " + str(output) + " for server cleanup and partial results.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
