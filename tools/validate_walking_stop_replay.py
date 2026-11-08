#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay recorded walking/optical inputs in an owned headless physical scene.

Run with Isaac Sim's python.bat, --recording-dir LOCAL_SESSION, --stop-feedback
0 or 1, --output-dir OUTPUT, and the same --ext-path/--enable Kit arguments used
by validate_gaze_walking.py. Recorded files are read in place, never copied into
the repository. The real actor, arm IK, finger/contact control and health guards
run at the production 200 Hz physics rate. No root, joint, object state, contact
or measured robot velocity is injected.

This is an approximate input replay, not reconstruction of the original physics.
The CSV's already-filtered cmd_* values are replayed as requests and filtered
again. Logged runtime palm poses substitute for unrecorded anatomical landmark
frames, and applied finger targets substitute for raw optical curls. Palm poses
are transformed from the recorded root's yaw frame to the current root's frame.
The original gaze selector and hardware XR transport are not being tested.
Contacts and payload motion in the report are newly measured simulation results.

Replay ends at at most 60.745 recorded simulation seconds, then requests zero
for eight seconds while holding the last valid palms/curls. Reports retain
failures and distinguish completing a replay from reproducing its contact fault.
Only this process's SimulationApp is closed. A native blocking startup cannot
be interrupted by the Python wall-clock deadlines.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import csv
import importlib
import json
import math
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
ROLES = ("thumb", "index", "middle", "ring", "little", "thumb_yaw")


def _require(condition: bool, message: str) -> None:
    """Fail an explicit replay contract independently of Python assert settings."""
    if not condition:
        raise RuntimeError(message)


def _write_report(path: Path, report: dict) -> None:
    """Persist measured output without including the source input rows."""
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _number(value: str | None) -> float | None:
    """Treat missing/nonfinite optional optical fields as unavailable."""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _stop_acceptance(samples: list[dict], final: dict) -> dict:
    """Measure sustained end-of-tail drift separately from merely avoiding a fault."""
    cutoff = final["replay_sim_time_s"] - 2.0
    first = min(samples, key=lambda sample: abs(sample["replay_sim_time_s"] - cutoff))
    span = final["replay_sim_time_s"] - first["replay_sim_time_s"]
    _require(span >= 1.8, "Insufficient final zero-command measurement window")
    window = [
        sample
        for sample in samples
        if first["replay_sim_time_s"] <= sample["replay_sim_time_s"] <= final["replay_sim_time_s"]
    ]
    window.append(final)
    for sample in window:
        requested = sample.get("raw_recorded_request")
        _require(
            isinstance(requested, (tuple, list))
            and len(requested) == 3
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in requested),
            "Final stop window lacks finite raw-command evidence",
        )
    max_raw_request = max(abs(value) for sample in window for value in sample["raw_recorded_request"])
    root_angles = [window[0]["root_yaw_rad"]]
    support_angles = [window[0]["support_world_xy_yaw"][2]]
    for previous, current in zip(window, window[1:]):
        root_angles.append(
            root_angles[-1] + (current["root_yaw_rad"] - previous["root_yaw_rad"] + math.pi) % (2 * math.pi) - math.pi
        )
        support_angles.append(
            support_angles[-1]
            + (current["support_world_xy_yaw"][2] - previous["support_world_xy_yaw"][2] + math.pi) % (2 * math.pi)
            - math.pi
        )
    root_ranges = [
        max(sample["root_position_m"][i] for sample in window) - min(sample["root_position_m"][i] for sample in window)
        for i in (0, 1)
    ]
    support_ranges = [
        max(sample["support_world_xy_yaw"][i] for sample in window)
        - min(sample["support_world_xy_yaw"][i] for sample in window)
        for i in (0, 1)
    ]
    root_delta = [final["root_position_m"][i] - first["root_position_m"][i] for i in (0, 1)]
    support_delta = [final["support_world_xy_yaw"][i] - first["support_world_xy_yaw"][i] for i in (0, 1)]
    root_yaw = root_angles[-1] - root_angles[0]
    support_yaw = support_angles[-1] - support_angles[0]
    checks = (
        ("max_absolute_raw_request", max_raw_request, 0.0),
        ("root_net_speed_m_s", math.hypot(*root_delta) / span, 0.05),
        ("support_net_speed_m_s", math.hypot(*support_delta) / span, 0.05),
        ("root_net_yaw_rad_s", abs(root_yaw) / span, 0.10),
        ("support_net_yaw_rad_s", abs(support_yaw) / span, 0.10),
        ("final_native_root_speed_m_s", math.hypot(*final["root_velocity_world_m_s"][:2]), 0.10),
        ("final_native_root_yaw_rad_s", abs(final["root_angular_velocity_world_rad_s"][2]), 0.25),
        ("root_xy_envelope_m", math.hypot(*root_ranges), 0.10),
        ("support_xy_envelope_m", math.hypot(*support_ranges), 0.10),
        ("root_yaw_envelope_rad", max(root_angles) - min(root_angles), 0.20),
        ("support_yaw_envelope_rad", max(support_angles) - min(support_angles), 0.20),
    )
    failures = [name for name, value, limit in checks if value > limit]
    return {
        "passed": not failures,
        "window_s": span,
        "measurements": {name: value for name, value, _ in checks},
        "limits": {name: limit for name, _, limit in checks},
        "failed_checks": failures,
        "meaning": "Prototype bounded-stop acceptance; not proof of exact replay or successful loaded transport",
    }


class RecordedInputs:
    """Read only required robot and optical fields, interpolated in simulation time."""

    def __init__(self, directory: Path) -> None:
        robot_fields = (
            "robot_pos_x",
            "robot_pos_y",
            "robot_pos_z",
            "robot_yaw",
            "cmd_forward",
            "cmd_lateral",
            "cmd_yaw",
        )
        hand_fields = tuple(
            field
            for side in SIDES
            for field in (
                f"{side}_hand_valid",
                *(f"{side}_hand_pos_{axis}" for axis in "xyz"),
                *(f"{side}_hand_q{axis}" for axis in "wxyz"),
                *(f"{side}_finger_target_{role}" for role in ROLES),
                f"{side}_grip",
                f"{side}_trigger",
            )
        )
        self.robot = self._read(directory / "behavior.csv", robot_fields, required=True)
        self.hands = self._read(directory / "hand_tracking.csv", hand_fields, required=False)
        self.robot_times = [row["sim_time"] for row in self.robot]
        self.hand_times = [row["sim_time"] for row in self.hands]
        self.end_s = min(60.745, self.robot_times[-1], self.hand_times[-1])
        _require(
            self.end_s >= 60.0, "The recording must cover the requested late arm/contact episode through 60 seconds"
        )

    @staticmethod
    def _read(path: Path, fields: tuple[str, ...], *, required: bool) -> list[dict]:
        """Reject missing schemas/backwards clocks; tolerate unavailable optical values."""
        rows = []
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            _require(
                set(("sim_time", *fields)).issubset(reader.fieldnames or ()), f"Missing required columns in {path.name}"
            )
            for raw in reader:
                row = {key: _number(raw.get(key)) for key in ("sim_time", *fields)}
                _require(row["sim_time"] is not None and row["sim_time"] >= 0, f"Invalid clock in {path.name}")
                if required:
                    _require(all(row[key] is not None for key in fields), f"Invalid robot input in {path.name}")
                _require(
                    not rows or row["sim_time"] >= rows[-1]["sim_time"], f"Backwards recording clock in {path.name}"
                )
                if rows and row["sim_time"] == rows[-1]["sim_time"]:
                    rows[-1] = row
                else:
                    rows.append(row)
        _require(len(rows) >= 2, f"Insufficient samples in {path.name}")
        return rows

    @staticmethod
    def _bracket(rows: list[dict], times: list[float], elapsed: float) -> tuple[dict, dict, float]:
        """Clamp endpoints without extrapolating a missing pose or quaternion."""
        index = max(0, min(len(rows) - 2, bisect.bisect_right(times, elapsed) - 1))
        a, b = rows[index : index + 2]
        fraction = max(0.0, min(1.0, (elapsed - a["sim_time"]) / (b["sim_time"] - a["sim_time"])))
        return a, b, fraction

    def sample(self, elapsed: float) -> tuple[dict, dict]:
        """Interpolate robot pose/commands and each independently valid optical hand."""
        a, b, fraction = self._bracket(self.robot, self.robot_times, elapsed)
        robot = {key: a[key] + fraction * (b[key] - a[key]) for key in a}
        robot["robot_yaw"] = a["robot_yaw"] + fraction * (
            (b["robot_yaw"] - a["robot_yaw"] + math.pi) % (2 * math.pi) - math.pi
        )
        a, b, fraction = self._bracket(self.hands, self.hand_times, elapsed)
        result = {}
        for side in SIDES:
            fields = [*(f"{side}_hand_pos_{axis}" for axis in "xyz"), *(f"{side}_hand_q{axis}" for axis in "wxyz")]
            valid = a[f"{side}_hand_valid"] == 1 and b[f"{side}_hand_valid"] == 1
            valid = valid and all(a[key] is not None and b[key] is not None for key in fields)
            hand = {"valid": valid, "grip": a[f"{side}_grip"] or 0.0, "trigger": a[f"{side}_trigger"] or 0.0}
            if valid:
                hand["position"] = tuple(a[key] + fraction * (b[key] - a[key]) for key in fields[:3])
                qa, qb = tuple(a[key] for key in fields[3:]), tuple(b[key] for key in fields[3:])
                sign = -1 if sum(x * y for x, y in zip(qa, qb)) < 0 else 1
                q = tuple(x + fraction * (sign * y - x) for x, y in zip(qa, qb))
                norm = math.sqrt(sum(value * value for value in q))
                hand["valid"] = norm > 1e-8
                if hand["valid"]:
                    hand["quaternion"] = tuple(value / norm for value in q)
                curls = {}
                for role in ROLES:
                    key = f"{side}_finger_target_{role}"
                    if a[key] is not None and b[key] is not None:
                        curls[role] = max(0.0, min(1.0, a[key] + fraction * (b[key] - a[key])))
                hand["curls"] = curls if len(curls) == len(ROLES) else None
            result[side] = hand
        return robot, result


async def _replay(example: object, inputs: RecordedInputs, report: dict) -> None:
    """Replace input readers only and measure the production physical response."""
    import numpy as np
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdPhysics

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    await example.load_world_async()
    while not example._physics_ready:
        _require(not example._articulation_health_fault, str(example._articulation_health_fault))
        await app.next_update_async()
    report["test_started_wall_monotonic_s"] = time.perf_counter()
    timeline.pause()
    _require(example._gaze_walking_enabled and example._g1_locomotion == "policy", "Expected the physical gaze profile")
    _require(abs(example._world_settings["physics_dt"] - 0.005) < 1e-9, "Replay requires 200 Hz physics")
    _require(hasattr(example, "_walking_stop_regulator"), "Stop-feedback integration is not loaded")
    example._walking_stop_feedback_enabled = report["stop_feedback"]
    native_xr = example._xr_core
    names = (
        "_poll_webcam_tracking",
        "_get_xr_input_device",
        "_get_hand_input_kind",
        "_get_optical_arm_pose",
        "_get_optical_palm_frame",
        "_get_hand_tracking_finger_curls",
        "_update_controller_command",
    )
    saved = {name: (name in example.__dict__, example.__dict__.get(name)) for name in names}
    saved_webcam = example._webcam_enabled
    fixture = {"elapsed_s": 0.0, "hands": {}, "palms": {}, "last_valid": {}, "raw": (0.0, 0.0, 0.0), "ticks": 0}
    original_arm = example._update_g1_arms_from_hand_tracking
    original_fingers = example._update_g1_fingers

    class OpticalDevice:
        """Expose recorded grip values without allowing them to gate optical targets."""

        def __init__(self, side: str) -> None:
            self.side = side

        def get_name(self) -> str:
            return f"recorded_optical_{self.side}"

        def get_pose_names(self) -> tuple:
            return ()

        def has_input(self, name: str) -> bool:
            return name in ("squeeze", "grip", "trigger")

        def has_input_gesture(self, name: str, gesture: str) -> bool:
            return self.has_input(name) and gesture in ("value", "click")

        def get_input_gesture_value(self, name: str, gesture: str) -> float:
            value = fixture["hands"].get(self.side, {}).get("trigger" if name == "trigger" else "grip", 0.0)
            return float(value >= 0.5) if gesture == "click" else value

    devices = {side: OpticalDevice(side) for side in SIDES}

    def optical(side: str, device: object) -> object:
        pose = fixture["palms"].get(side)
        return Gf.Matrix4d(pose) if device is devices.get(side) and pose is not None else None

    def orientation(side: str, device: object) -> object:
        pose = optical(side, device)
        return Gf.Matrix4d().SetRotate(pose.ExtractRotationQuat()) if pose is not None else None

    def curls(device: object) -> dict | None:
        side = getattr(device, "side", None)
        value = fixture["hands"].get(side, {}).get("curls")
        return dict(value) if value is not None else None

    def command(dt: float) -> None:
        fixture["elapsed_s"] += float(dt)
        fixture["ticks"] += 1
        elapsed = fixture["elapsed_s"]
        recorded, hands = inputs.sample(min(elapsed, inputs.end_s))
        if elapsed <= inputs.end_s:
            for side, hand in hands.items():
                if hand["valid"]:
                    fixture["last_valid"][side] = (dict(recorded), dict(hand))
        fixture["hands"] = hands
        fixture["palms"] = {}
        base = example._get_g1_base_pose_for_arms()
        if base is not None:
            current_position, current_yaw = base
            current_rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(current_yaw)))
            for side in SIDES:
                source, hand = recorded, hands[side]
                if elapsed > inputs.end_s and side in fixture["last_valid"]:
                    source, hand = fixture["last_valid"][side]
                    fixture["hands"][side] = hand
                if not hand["valid"]:
                    continue
                recorded_rotation = Gf.Matrix4d().SetRotate(
                    Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(source["robot_yaw"]))
                )
                recorded_position = Gf.Vec3d(*(source[f"robot_pos_{axis}"] for axis in "xyz"))
                relative = recorded_rotation.GetInverse().TransformDir(Gf.Vec3d(*hand["position"]) - recorded_position)
                q = hand["quaternion"]
                matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
                matrix = matrix * recorded_rotation.GetInverse() * current_rotation
                matrix.SetTranslateOnly(current_position + current_rotation.TransformDir(relative))
                fixture["palms"][side] = matrix
        requested = tuple(recorded[key] for key in ("cmd_forward", "cmd_lateral", "cmd_yaw"))
        requested = tuple(0.0 if elapsed > inputs.end_s or abs(value) < 1e-5 else value for value in requested)
        fixture["raw"] = requested
        example._keyboard_command[:] = 0.0
        example._locomotion_brake = False
        for axis, value in enumerate(requested):
            example._controller_command[axis] = value
        if not any(requested):
            example._base_command[:] = 0.0

    def vector(value: object) -> list[float]:
        result = np.asarray(example._first_pose_value(value), dtype=float).reshape(-1)
        _require(bool(np.isfinite(result).all()), "Nonfinite native measurement")
        return result.tolist()

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(example._sample_box_root_path)
    paths = tuple(str(prim.GetPath()) for prim in root.GetChildren() if prim.HasAPI(UsdPhysics.RigidBodyAPI))
    object_summary = report["objects"] = {}
    previous_support = None

    def snapshot() -> dict:
        nonlocal previous_support
        positions, quaternions = example.g1.robot.get_world_poses()
        linear, angular = example.g1.robot.get_velocities()
        p, q, v, w = map(vector, (positions, quaternions, linear, angular))
        support = tuple(float(value) for value in example._read_walking_support_pose())
        rate = None
        if previous_support is not None and fixture["elapsed_s"] > previous_support[0]:
            dt = fixture["elapsed_s"] - previous_support[0]
            rate = [(support[i] - previous_support[1][i]) / dt for i in (0, 1)]
            rate.append(((support[2] - previous_support[1][2] + math.pi) % (2 * math.pi) - math.pi) / dt)
        previous_support = (fixture["elapsed_s"], support)
        bimanual = example._bimanual_grasp_result
        contacts = {}
        for side in SIDES:
            sampled = example._latest_hand_contacts.get(side, ())
            contacts[side] = {
                "count": len(sampled),
                "normal_force_sum_n": sum(float(contact.normal_force_n) for contact in sampled),
                "objects": sorted({str(contact.object_path) for contact in sampled}),
                "arm_source": example._arm_input_sources.get(side),
                "recorded_grip": fixture["hands"].get(side, {}).get("grip", 0.0),
                "finger_curls": example.g1.get_finger_curls(side),
            }
        poses = example._get_object_state_pose_batch(stage, paths) or {}
        moved = {}
        for path, (position, _) in poses.items():
            position = tuple(float(value) for value in position)
            item = object_summary.setdefault(
                path,
                {
                    "initial_position_m": position,
                    "max_displacement_m": 0.0,
                    "height_min_m": position[2],
                    "height_max_m": position[2],
                    "bimanual_supported_samples": 0,
                },
            )
            distance = math.dist(position, item["initial_position_m"])
            item.update(
                max_displacement_m=max(item["max_displacement_m"], distance),
                height_min_m=min(item["height_min_m"], position[2]),
                height_max_m=max(item["height_max_m"], position[2]),
                final_position_m=position,
            )
            supported = bool(bimanual.contact_supported and bimanual.object_path == path)
            item["bimanual_supported_samples"] += int(supported)
            if distance > 0.001 or supported:
                moved[path] = {"position_m": position, "bimanual_contact_supported": supported}
        regulator_status = dict(example._walking_stop_regulator.status)
        filtered = [float(value) for value in example._base_command]
        effective = regulator_status.get("effective", filtered) if report["stop_feedback"] else filtered
        return {
            "replay_sim_time_s": fixture["elapsed_s"],
            "phase": report["phase"],
            "root_position_m": p,
            "root_quaternion_wxyz": q,
            "root_yaw_rad": math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)),
            "root_velocity_world_m_s": v,
            "root_angular_velocity_world_rad_s": w,
            "support_world_xy_yaw": support,
            "support_snapshot_difference_xy_yaw_rate": rate,
            "raw_recorded_request": fixture["raw"],
            "filtered_command": filtered,
            "effective_actor_command": effective,
            "actor_active_mode": getattr(getattr(example.g1, "_velocity_policy", None), "active_mode", None),
            "stop_regulator": regulator_status,
            "contact_health": example._contact_health,
            "contacts": contacts,
            "moved_objects": moved,
            "bimanual_state": str(bimanual.state),
            "bimanual_object": bimanual.object_path,
            "bimanual_contact_supported": bool(bimanual.contact_supported),
            "fault": example._articulation_health_fault,
        }

    try:
        example._webcam_enabled = True  # Existing headless optical-input gate; no webcam is opened.
        example._poll_webcam_tracking = lambda: None
        example._get_xr_input_device = lambda handle: devices.get(handle.rsplit("/", 1)[-1])
        example._get_hand_input_kind = lambda device: "hand_tracking" if device in devices.values() else "none"
        example._get_optical_arm_pose = optical
        example._get_optical_palm_frame = orientation
        example._get_hand_tracking_finger_curls = curls
        example._update_controller_command = command
        report["phase"] = "replaying_recorded_inputs"
        report["samples"] = [snapshot()]
        timeline.play()
        while fixture["elapsed_s"] < inputs.end_s + 8.0:
            await app.next_update_async()
            if fixture["elapsed_s"] > inputs.end_s:
                report["phase"] = "zero_request_holding_last_palms"
            if fixture["elapsed_s"] - report["samples"][-1]["replay_sim_time_s"] >= 0.1:
                report["samples"].append(snapshot())
            _require(not example._articulation_health_fault, str(example._articulation_health_fault))
            _require(not example.g1.locomotion_fault, str(example.g1.locomotion_fault))
            _require(not example._physics_step_error_logged, str(example._physics_step_error_logged))
            _require(timeline.is_playing(), "Timeline paused before replay completion")
        report["final"] = snapshot()
        report["replay_completed"] = True
        report["stop_acceptance"] = _stop_acceptance(report["samples"], report["final"])
        _require(report["stop_acceptance"]["passed"], "Final zero-command drift exceeds replay acceptance limits")
        report["passed"] = True
        report["phase"] = "complete"
    finally:
        report["physics_input_ticks"] = fixture["ticks"]
        report["last_replay_sim_time_s"] = fixture["elapsed_s"]
        report["fault"] = example._articulation_health_fault
        report["fault_snapshot"] = getattr(example, "_last_walking_fault_snapshot", None)
        report["normal_updaters_preserved"] = (
            example._update_g1_arms_from_hand_tracking == original_arm
            and example._update_g1_fingers == original_fingers
        )
        try:
            report["final"] = snapshot()
        except Exception as error:
            report["final_snapshot_error"] = repr(error)
        late_samples = [sample for sample in report.get("samples", ()) if 59.0 <= sample["replay_sim_time_s"] <= 61.0]
        report["late_contact_observed"] = any(
            sample["contacts"][side]["count"] > 0 for sample in late_samples for side in SIDES
        )
        report["late_bimanual_supported_observed"] = any(
            sample["bimanual_contact_supported"] for sample in late_samples
        )
        report["late_both_arm_sources_active"] = any(
            all(sample["contacts"][side]["arm_source"] == "hand_tracking" for side in SIDES) for sample in late_samples
        )
        timeline.pause()
        for name in ("_keyboard_command", "_controller_command", "_base_command"):
            getattr(example, name)[:] = 0.0
        for name, (existed, value) in saved.items():
            if existed:
                setattr(example, name, value)
            else:
                example.__dict__.pop(name, None)
        example._webcam_enabled = saved_webcam
        report["native_xr_preserved"] = example._xr_core is native_xr
        report["fixture_restored"] = True


def main() -> int:
    """Own an isolated application and retain partial evidence on failures/timeouts."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--stop-feedback", choices=("0", "1"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--load-timeout", type=float, default=180.0)
    parser.add_argument("--test-timeout", type=float, default=240.0)
    args, kit_args = parser.parse_known_args()
    for name in ("load_timeout", "test_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    inputs = RecordedInputs(args.recording_dir.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"walking-stop-{args.stop_feedback}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    report_path = output / f"{run_id}.json"
    report = {
        "passed": False,
        "phase": "starting_application",
        "run_id": run_id,
        "stop_feedback": args.stop_feedback == "1",
        "recording_end_s": inputs.end_s,
        "tail_s": 8.0,
        "input_row_counts": {"robot": len(inputs.robot), "hands": len(inputs.hands)},
        "hardware_tracking_tested": False,
        "original_fault_reproduction_established": False,
        "approximations": [
            "Recorded filtered commands replayed as requests through the normal command filter a second time",
            "Recorded raw runtime palm frames stand in for unrecorded anatomical landmark midpoint/orientation",
            "Recorded applied finger targets replayed as optical curls through normal smoothing/contact limits",
            "Recorded root is used only to express palm inputs relative to the current root; no state injection",
            "Input timing follows simulation time; gaze selection and original wall-time transport are not replayed",
            "Completion alone does not establish reproduction of the original contact sequence or fix acceptance",
        ],
    }
    _write_report(report_path, report)
    print(f"Walking stop replay report: {report_path}", flush=True)
    os.environ.update(
        {
            "G1_GESTURE_WALKING": "1",
            "G1_LOCOMOTION_INPUT": "gaze",
            "G1_POINTING_NAVIGATION": "0",
            "G1_OPTICAL_HANDS_ONLY": "1",
            "G1_WEBCAM_ENABLED": "0",
            "G1_GAZE_PREVIEW": "0",
            "G1_GAZE_MODE": "hold",
            "G1_GAZE_DWELL_S": "0",
            "G1_GAZE_FORWARD_SPEED": "0.40",
            "G1_GAZE_BACKWARD_SPEED": "0.25",
            "G1_GAZE_YAW_SPEED": "0.40",
            "G1_GAZE_HEADING_HOLD": "1",
            "G1_GAZE_HEADING_GAIN": "3.0",
            "G1_GAZE_HEADING_MAX_YAW_SPEED": "0.15",
            "G1_GAZE_STOP_FEEDBACK": args.stop_feedback,
        }
    )
    sys.argv = [sys.argv[0], *kit_args]
    simulation_app = task = example = None
    started = time.perf_counter()
    try:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": True, "width": 640, "height": 480, "renderer": "RayTracedLighting"})
        module = importlib.import_module("isaacsim.robot.policy.examples.interactive.humanoid.humanoid_example")
        expected = ROOT / "source/extensions/isaacsim.robot.policy.examples"
        _require(Path(module.__file__).resolve().is_relative_to(expected), "Isaac loaded a different policy extension")
        report["loaded_module"] = module.__file__
        example = module.HumanoidExample()
        example._behavioral_data_enabled = False
        report["phase"] = "loading_scene"
        task = asyncio.ensure_future(_replay(example, inputs, report))
        last_progress = 0.0
        while not task.done():
            now = time.perf_counter()
            test_start = report.get("test_started_wall_monotonic_s")
            deadline = test_start + args.test_timeout if test_start is not None else started + args.load_timeout
            if now > deadline:
                task.cancel()
                raise TimeoutError(f"Wall deadline exceeded during {report['phase']}")
            if now - last_progress >= 5.0:
                report["wall_elapsed_s"] = now - started
                _write_report(report_path, report)
                print(f"Walking stop replay: {report['phase']} ({now - started:.1f} wall seconds)", flush=True)
                last_progress = now
            simulation_app.update()
        task.result()
    except BaseException as error:
        report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
    finally:
        if task is not None and not task.done():
            task.cancel()
            if simulation_app is not None:
                try:
                    for _ in range(3):
                        simulation_app.update()
                        if task.done():
                            break
                except Exception as error:
                    report["cancellation_error"] = repr(error)
        if example is not None:
            try:
                example.physics_cleanup()
            except Exception as error:
                report.update(passed=False, cleanup_error=repr(error))
        report["wall_elapsed_s"] = time.perf_counter() - started
        _write_report(report_path, report)
        print(f"Walking stop replay {'completed' if report['passed'] else 'FAILED'}: {report_path}", flush=True)
        if simulation_app is not None:
            try:
                simulation_app.close(exit_code=0 if report["passed"] else 1)
            except Exception as error:
                report.update(passed=False, close_error=repr(error))
                _write_report(report_path, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
