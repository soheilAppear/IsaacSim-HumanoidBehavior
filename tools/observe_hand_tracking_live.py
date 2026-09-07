#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe real optical hand input and measured robot fingers in the running Kit app.

Run while the Humanoid example is playing and the operator moves bare fingers in
the headset cameras' view. This observer does not supply synthetic devices, change
tracking settings, touch gaze, or start/pause/reset the simulation. It samples for
wall-clock seconds so a slow VR frame rate cannot stall observation indefinitely.
"""

from __future__ import annotations

import argparse

from kit_exec import DEFAULT_HOST, DEFAULT_PORT, execute

LIVE_CODE = r"""
import asyncio


async def _observe_real_hand_tracking(ex, seconds):
    import json
    import math
    import time

    import omni.timeline

    assert ex is not None and ex.g1 is not None, "Load the Humanoid example first"
    assert ex._xr_core is not None, "XRCore is unavailable"
    timeline = omni.timeline.get_timeline_interface()
    roles = tuple(ex._finger_roles)
    actuators = (*roles, "thumb_yaw")
    required = dict(ex._finger_curl_joint_chains)
    required["thumb_yaw"] = (
        "wrist", "middle_proximal", "index_proximal", "little_proximal", "thumb_metacarpal", "thumb_proximal"
    )
    known_landmarks = {name for chain in required.values() for name in chain}
    started = time.monotonic()
    physics_start = float(ex._headset_gait_time)
    deadline = started + seconds
    report = {
        "input": "unmodified real XR devices",
        "timeline_changed": False,
        "playing_at_start": timeline.is_playing(),
        "sample_count": 0,
        "target_motion_threshold": 0.10,
        "measured_motion_threshold": 0.08,
        "minimum_samples_for_motion": 3,
        "hands": {},
        "read_errors": {},
        "max_physics_callback_error_count": 0,
    }
    callback_errors = set()
    observed_names = {side: set() for side in ("left", "right")}
    for side in observed_names:
        report["hands"][side] = {
            "device_samples": 0,
            "optical_samples": 0,
            "raw_source_counts": {},
            "control_source_counts": {},
            "max_pose_count": 0,
            "max_valid_skeletal_pose_count": 0,
            "fingers": {
                role: {"samples": 0, "target_min": None, "target_max": None, "measured_min": None, "measured_max": None}
                for role in actuators
            },
        }

    def note_error(label, error):
        key = f"{label}: {type(error).__name__}: {error}"
        report["read_errors"][key] = report["read_errors"].get(key, 0) + 1

    def count(mapping, value):
        mapping[value] = mapping.get(value, 0) + 1

    def extend_range(record, name, value):
        value = float(value)
        if not math.isfinite(value):
            return
        lower, upper = name + "_min", name + "_max"
        record[lower] = value if record[lower] is None else min(record[lower], value)
        record[upper] = value if record[upper] is None else max(record[upper], value)

    while time.monotonic() < deadline:
        report["sample_count"] += 1
        errors = tuple(ex._physics_step_error_logged)
        callback_errors.update(str(error) for error in errors)
        report["max_physics_callback_error_count"] = max(report["max_physics_callback_error_count"], len(errors))
        positions = None
        try:
            if ex.g1.robot.is_physics_tensor_entity_valid():
                positions = ex.g1.robot.get_dof_positions().numpy()[0]
        except Exception as error:
            note_error("robot joint read", error)

        for side, state in report["hands"].items():
            control_source = ex._finger_curl_source.get(side, "none")
            count(state["control_source_counts"], control_source)
            try:
                # Read XRCore directly, bypassing even the example's input reader.
                # This observer never creates or replaces an input device.
                device = ex._xr_core.get_input_device("/user/hand/" + side)
                if device is None:
                    count(state["raw_source_counts"], "no_device")
                    continue
                state["device_samples"] += 1
                source = str(device.get_hand_tracking_data_source())
                count(state["raw_source_counts"], source or "empty")
                names = {str(name) for name in device.get_pose_names()}
                observed_names[side].update(names)
                state["max_pose_count"] = max(state["max_pose_count"], len(names))
                valid_landmarks = set()
                for name in names.intersection(known_landmarks):
                    try:
                        descriptor = device.get_virtual_world_pose_desc(name)
                        if int(descriptor.validity_flags) & 2 == 0:
                            continue
                        point = descriptor.pose_matrix.ExtractTranslation()
                        if all(math.isfinite(float(value)) for value in point):
                            valid_landmarks.add(name)
                    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                        # Native poses can disappear between enumeration and reading.
                        note_error(side + " landmark " + name, error)
                state["max_valid_skeletal_pose_count"] = max(
                    state["max_valid_skeletal_pose_count"], len(valid_landmarks)
                )
                if source != "hand" or not valid_landmarks:
                    continue
                state["optical_samples"] += 1
                if control_source != "hand_tracking":
                    continue  # The physics loop has not yet consumed this real input.
                targets = dict(ex._latest_finger_curls.get(side, {}))
                indices = ex.g1._finger_dof_indices.get(side, {})
                ranges = ex.g1._finger_open_closed.get(side, {})
                for role, record in state["fingers"].items():
                    if not set(required[role]).issubset(valid_landmarks) or role not in targets:
                        continue
                    record["samples"] += 1
                    extend_range(record, "target", targets[role])
                    if positions is not None and role in indices and role in ranges:
                        opened, closed = ranges[role]
                        if abs(closed - opened) > 1e-8:
                            measured = (float(positions[indices[role]]) - opened) / (closed - opened)
                            extend_range(record, "measured", measured)
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                note_error(side + " device", error)
        await asyncio.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    total = 0
    for side, state in report["hands"].items():
        state["pose_names"] = sorted(observed_names[side])
        state["skeletal_pose_names"] = sorted(observed_names[side].intersection(known_landmarks))
        for role, record in state["fingers"].items():
            for label in ("target", "measured"):
                low, high = record[label + "_min"], record[label + "_max"]
                record[label + "_range"] = None if low is None or high is None else high - low
            record["optical_motion_observed"] = bool(
                record["samples"] >= report["minimum_samples_for_motion"]
                and (record["target_range"] or 0.0) >= report["target_motion_threshold"]
                and (record["measured_range"] or 0.0) >= report["measured_motion_threshold"]
            )
            if role in roles and record["optical_motion_observed"]:
                total += 1
            for key, value in tuple(record.items()):
                if isinstance(value, float):
                    record[key] = round(value, 5)
        state["finger_motion_count"] = sum(state["fingers"][role]["optical_motion_observed"] for role in roles)
    report.update(
        wall_seconds=round(time.monotonic() - started, 3),
        simulated_seconds=round(float(ex._headset_gait_time) - physics_start, 3),
        playing_at_end=timeline.is_playing(),
        physics_callback_errors=sorted(callback_errors),
        optical_fingers_with_motion=total,
        optical_fingers_expected=2 * len(roles),
        all_fingers_observed=total == 2 * len(roles),
        interpretation="Motion requires real optical targets and corresponding measured robot-joint variation; no motion may mean no gesture or insufficient visible tracking.",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


async def _run_hand_observer_with_timeout():
    await asyncio.wait_for(_observe_real_hand_tracking(EX, __SECONDS__), timeout=__SECONDS__ + 10.0)


await asyncio.ensure_future(_run_hand_observer_with_timeout())
"""


def main() -> int:
    """Observe real hands over the existing Python server without changing the scene."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seconds", type=float, default=20.0, help="Wall-clock observation interval")
    args = parser.parse_args()
    if not 0.1 <= args.seconds <= 600.0:
        parser.error("--seconds must be between 0.1 and 600")
    code = LIVE_CODE.replace("__SECONDS__", repr(args.seconds))
    return execute(code, args.host, args.port, args.seconds + 20.0)


if __name__ == "__main__":
    raise SystemExit(main())
