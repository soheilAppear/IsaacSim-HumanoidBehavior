#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate facial locomotion in one owned, headless Isaac Sim process.

Run with Isaac Sim's python.bat and this checkout's entire policy extension::

    python.bat tools/validate_facial_walking.py --mode preview --output-dir OUTPUT \
        --ext-path POLICY_EXTENSION --enable isaacsim.robot.policy.examples-5.2.11 \
        --/app/vulkan=false --/metricsAssembler/changeListenerEnabled=false

Use a separate --mode move invocation for the physical GR00T scene. Real scene,
walking actor, smoothing, measured stopping guards and arm/finger updater methods
remain active. Only receiver samples and XR session readiness are synthetic.
The facial input callback alone receives a monotonic clock advanced by physics dt.
This validates neither OSC transport timing nor a person's face/hand tracking or
box carrying. Captures and JSON preserve partial failures. Only this process's
SimulationApp is closed. A blocking native load cannot be preempted by Python.
"""

from __future__ import annotations

import argparse
import asyncio
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
from types import SimpleNamespace

from validate_gaze_walking import _capture_camera, _require, _write_report

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("back", "strafe_left", "strafe_right", "turn_left", "turn_right", "forward")
EXPRESSIONS = {
    "forward": {"JawDrop": 0.9},
    "back": {"LipPuckerL": 0.9, "LipPuckerR": 0.9},
    "strafe_left": {"MouthLeft": 0.9},
    "strafe_right": {"MouthRight": 0.9},
    "turn_left": {"EyesClosedL": 0.9},
    "turn_right": {"EyesClosedR": 0.9},
}


async def _validate(example: object, report: dict, output: Path, run_id: str) -> None:
    """Feed synthetic expressions through live body arbitration and inspect physics."""
    import numpy as np
    import omni.kit.app
    import omni.timeline
    from isaacsim.robot.policy.examples.interactive.humanoid.face_tracking_input import (
        EXPRESSION_NAMES,
        FaceTrackingSample,
    )

    module = importlib.import_module(
        "isaacsim.robot.policy.examples.interactive.humanoid.facial_locomotion_integration"
    )
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    await example.load_world_async()
    while not example._physics_ready:
        _require(not example._articulation_health_fault, str(example._articulation_health_fault))
        await app.next_update_async()
    report["test_started_wall_monotonic_s"] = time.perf_counter()
    report["phase"] = "installing_fixture"
    face = example._facial_input
    preview = report["mode"] == "preview"
    _require(face is not None, "Facial profile was not loaded")
    _require(face.controller.config.preview is preview, "Wrong preview configuration")
    _require(face.controller.config.control_mode == "hold", "This fixture validates held expressions")
    _require(example._g1_locomotion == ("stationary" if preview else "policy"), "Wrong base topology")
    _require(example._optical_hands_only, "Optical hand profile was not retained")
    _require(not example._pointing_navigation_enabled, "Pointing navigation unexpectedly owns body input")
    report["effective_heading"] = {
        "enabled": face.controller.config.heading_hold,
        "gain": face.controller.config.heading_gain,
        "max_yaw_speed": face.controller.config.heading_max_yaw_speed,
        "filter_s": face.controller.config.heading_filter_s,
    }
    report["expression_thresholds"] = face.status()["thresholds"]
    fixture = {"elapsed_s": 0.0, "expression": None, "valid": True, "stale": False, "sequence": 0}
    counts = {"facial_updates": 0, "arm_updates": 0, "finger_updates": 0}
    report["updater_calls"] = counts
    report["motion_samples"] = []
    report["state_transitions"] = []
    saved = []
    native_xr = example._xr_core
    real_time = module.time
    selection_clock = SimpleNamespace(monotonic=lambda: fixture["elapsed_s"])

    def install(owner: object, name: str, replacement: object) -> None:
        saved.append((owner, name, name in vars(owner), vars(owner).get(name)))
        setattr(owner, name, replacement)

    def vector(value: object, size: int) -> list[float]:
        values = np.asarray(example._first_pose_value(value), dtype=float).reshape(-1)
        _require(values.shape == (size,) and bool(np.isfinite(values).all()), "Invalid native physics vector")
        return values.tolist()

    def snapshot() -> dict:
        positions, quaternions = example.g1.robot.get_world_poses()
        linear, angular = example.g1.robot.get_velocities()
        p, q, v, w = vector(positions, 3), vector(quaternions, 4), vector(linear, 3), vector(angular, 3)
        norm = math.sqrt(sum(component * component for component in q))
        _require(norm > 1e-6, "Invalid measured root quaternion")
        qw, qx, qy, qz = (component / norm for component in q)
        return {
            "control_sim_time_s": fixture["elapsed_s"],
            "position_m": p,
            "yaw_rad": math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)),
            "quaternion_wxyz": q,
            "linear_velocity_m_s": v,
            "angular_velocity_rad_s": w,
            "status": example.facial_locomotion_status(),
            "filtered_command": [float(value) for value in example._base_command],
        }

    def poll(*, now_s: float | None = None) -> FaceTrackingSample:
        fixture["sequence"] += 1
        values = dict.fromkeys(EXPRESSION_NAMES, 0.0)
        expression = fixture["expression"]
        if isinstance(expression, str):
            values.update(EXPRESSIONS[expression])
        elif expression is not None:
            values.update(expression)
        age = 0.5 if fixture["stale"] else 0.0
        return FaceTrackingSample(
            weights=values,
            valid=fixture["valid"],
            reason="Synthetic native-validation input",
            received_at=fixture["elapsed_s"] - age,
            age_s=age,
            confidence_upper=1.0,
            confidence_lower=1.0,
            sequence=fixture["sequence"],
        )

    original_face = face.update
    original_arms = example._update_g1_arms_from_hand_tracking
    original_fingers = example._update_g1_fingers
    previous_state = None

    def update_face(dt: float) -> bool:
        nonlocal previous_state
        fixture["elapsed_s"] += float(dt)
        counts["facial_updates"] += 1
        module.time = selection_clock
        try:
            owned = original_face(dt)
        finally:
            module.time = real_time
        state = face.controller.command.state
        if state != previous_state:
            report["state_transitions"].append(snapshot())
            previous_state = state
        if preview:
            _require(not any(face.controller.command.velocity), "Preview emitted physical face intent")
            _require(not any(float(value) for value in example._base_command), "Preview retained filtered intent")
        return owned

    def update_arms() -> None:
        counts["arm_updates"] += 1
        original_arms()

    def update_fingers() -> None:
        counts["finger_updates"] += 1
        original_fingers()

    def check_health() -> None:
        _require(example._xr_core is native_xr, "Native XR ownership changed")
        _require(not example._articulation_health_fault, str(example._articulation_health_fault))
        _require(not example.g1.locomotion_fault, str(example.g1.locomotion_fault))
        _require(
            not example._physics_step_error_logged,
            "Physics callback errors: " + str(example._physics_step_error_logged),
        )
        _require(example.g1.robot.is_physics_tensor_entity_valid(), "Native physics views became invalid")
        _require(timeline.is_playing(), "Timeline unexpectedly stopped or paused")

    async def advance() -> None:
        await app.next_update_async()
        check_health()
        samples = report["motion_samples"]
        if not samples or fixture["elapsed_s"] - samples[-1]["control_sim_time_s"] >= 0.1:
            samples.append(snapshot())

    async def wait_sim(seconds: float) -> None:
        end = fixture["elapsed_s"] + seconds
        while fixture["elapsed_s"] + 1e-9 < end:
            await advance()

    async def neutral(seconds: float = 0.35) -> None:
        fixture.update(expression=None, valid=True, stale=False)
        await wait_sim(seconds)
        _require(face.controller.command.neutral_ready, "Fresh neutral expression did not arm recognition")

    def preflight(action: str, start: dict, duration: float) -> None:
        if preview:
            return
        position, yaw = start["position_m"], start["yaw_rad"]
        if action.startswith("turn"):
            _require(example._navigation_scene.is_route_corner_clear(position), "Turn fixture lacks body clearance")
            return
        angle = yaw + (math.pi / 2 if action.startswith("strafe") else 0)
        sign = -1 if action in ("back", "strafe_right") else 1
        speed = {
            "forward": face.controller.config.forward_speed,
            "back": face.controller.config.backward_speed,
            "strafe_left": face.controller.config.lateral_speed,
            "strafe_right": face.controller.config.lateral_speed,
        }[action]
        distance = max(0.60, speed * (duration + face.controller.config.confirm_s) + 0.35)
        goal = (position[0] + sign * distance * math.cos(angle), position[1] + sign * distance * math.sin(angle), 0.0)
        _require(example._navigation_scene.floor_at(goal) is not None, "Translation lacks verified floor")
        _require(
            example._navigation_scene.is_path_clear(position, goal),
            "Translation corridor is blocked: " + str(example._navigation_scene.last_error),
        )

    try:
        install(face.receiver, "poll", poll)
        install(
            face,
            "_session_ready",
            lambda: bool(
                example._physics_ready and not example._timeline_stopped and example._articulation_health_fault is None
            ),
        )
        install(face, "update", update_face)
        install(example, "_update_g1_arms_from_hand_tracking", update_arms)
        install(example, "_update_g1_fingers", update_fingers)
        example.set_facial_locomotion_enabled(True)
        await neutral()
        report["start"] = snapshot()
        report["captures"].append(await _capture_camera(example, output / f"{run_id}-ready.png"))
        report["phase"] = "blink_rejection"
        count = face.controller.command.activation_count
        fixture["expression"] = "turn_left"
        await wait_sim(0.05)
        await neutral()
        _require(face.controller.command.activation_count == count, "Brief unilateral closure activated")
        fixture["expression"] = {"EyesClosedL": 0.9, "EyesClosedR": 0.9}
        await wait_sim(0.30)
        await neutral()
        _require(face.controller.command.activation_count == count, "Bilateral blink activated")
        report["blink_rejection_passed"] = True

        report["phase"] = "weak_wink_mouth_rejection"
        # This observed combination is a partial right-eye closure with strong
        # accompanying mouth motion. It must not be interpreted as a sidestep.
        fixture["expression"] = {"EyesClosedL": 0.110, "EyesClosedR": 0.223, "MouthRight": 0.908}
        await wait_sim(face.controller.config.confirm_s + 0.10)
        report["weak_wink_mouth"] = snapshot()
        _require(face.controller.command.activation_count == count, "Weak wink selected a mouth command")
        _require(not any(face.controller.command.velocity), "Weak wink emitted facial velocity")
        _require(not any(float(value) for value in example._base_command), "Weak wink retained filtered intent")
        fixture["expression"] = {"MouthRight": 0.908}
        await wait_sim(face.controller.config.confirm_s + 0.10)
        report["weak_wink_mouth_after_reopening"] = snapshot()
        _require(face.controller.command.activation_count == count, "Reopening promoted wink overlap to a sidestep")
        _require(not any(face.controller.command.velocity), "Wink mouth overlap moved after eye reopening")
        _require(not any(float(value) for value in example._base_command), "Reopening retained filtered intent")
        await neutral()
        report["weak_wink_mouth_rejection_passed"] = True

        report["phase"] = "post_blink_wink"
        # A brief uneven reopening remains a blink. The wearer may deliberately
        # hold one eye closed afterward for the longer configured confirmation.
        fixture["expression"] = {"EyesClosedL": 0.9, "EyesClosedR": 0.9}
        await wait_sim(0.30)
        _require(face.controller.command.activation_count == count, "Bilateral closure selected a turn")
        fixture["expression"] = "turn_left"
        post_blink_confirm = face.controller.config.post_blink_wink_confirm_s
        await wait_sim(post_blink_confirm / 2)
        report["post_blink_short_unilateral"] = snapshot()
        _require(face.controller.command.activation_count == count, "Brief post-blink closure selected a turn")
        _require(not any(face.controller.command.velocity), "Brief post-blink closure emitted velocity")
        _require(not any(float(value) for value in example._base_command), "Brief post-blink closure retained intent")
        await wait_sim(post_blink_confirm / 2 + 0.05)
        report["post_blink_wink_activated"] = snapshot()
        _require(face.controller.command.activation_count == count + 1, "Held post-blink wink did not activate once")
        _require(
            face.controller.command.busy and face.controller.command.target == "turn_left",
            "Held post-blink wink did not select the left turn",
        )
        if not preview:
            _require(face.controller.command.velocity[2] > 0, "Held post-blink wink selected the wrong turn direction")
        await wait_sim(0.10)
        _require(face.controller.command.activation_count == count + 1, "Held post-blink wink repeated activation")
        fixture["expression"] = None
        await wait_sim(0.02)
        _require(not any(face.controller.command.velocity), "Relaxing a post-blink wink retained facial velocity")
        await neutral(5.0 if not preview else 0.35)
        _require(
            max(abs(float(value)) for value in example._base_command) < 1e-5,
            "Post-blink turn did not clear filtered intent after relaxation",
        )
        report["post_blink_wink_stopped"] = snapshot()
        report["post_blink_wink_passed"] = True

        for action in ACTIONS:
            report["phase"] = action
            await neutral()
            duration = report["back_hold_s"] if action == "back" else report["hold_s"]
            entry = {"action": action, "passed": False, "start": snapshot(), "hold_s": duration}
            report["actions"].append(entry)
            preflight(action, entry["start"], duration)
            counts_before = dict(counts)
            baseline = face.controller.command.activation_count
            fixture["expression"] = action
            await wait_sim(face.controller._confirmation_for(action) + 0.05)
            _require(
                face.controller.command.activation_count == baseline + 1, "Expression did not activate exactly once"
            )
            _require(face.controller.command.last_action == action, "Wrong facial action selected")
            entry["activated"] = snapshot()
            if action == "turn_right":
                # Real Quest winks can move the mouth on the same side. This must
                # not cancel an already qualified turn or select a sidestep.
                fixture["expression"] = {"EyesClosedR": 0.9, "MouthRight": 1.0}
                await wait_sim(face.controller.config.confirm_s + 0.05)
                entry["mouth_overlap"] = snapshot()
                command = face.controller.command
                _require(command.busy and command.target == "turn_right", "Mouth overlap cancelled the right wink")
                _require(command.activation_count == baseline + 1, "Mouth overlap repeated the right-turn activation")
                if not preview:
                    _require(command.velocity[2] < 0, "Mouth overlap changed the right-turn direction")
                    _require(not any(command.velocity[:2]), "Mouth overlap added translation to a right turn")
                entry["mouth_overlap_passed"] = True
                # A confirmed calibrated wink tolerates small noise in the open
                # eye without repeating the half-second activation interval.
                fixture["expression"] = {"EyesClosedL": 0.17, "EyesClosedR": 0.31, "MouthRight": 1.0}
                await wait_sim(0.10)
                command = face.controller.command
                _require(command.busy and command.target == "turn_right", "Opposite-eye noise interrupted right turn")
                _require(command.activation_count == baseline + 1, "Opposite-eye noise repeated turn activation")
                fixture["expression"] = {"EyesClosedR": 0.9, "MouthRight": 1.0}
                entry["right_wink_hysteresis_passed"] = True
            await wait_sim(duration)
            entry["held"] = snapshot()
            _require(face.controller.command.activation_count == baseline + 1, "Held expression repeated activation")
            _require(face.controller.command.busy, "Held expression stopped before release")
            _require(counts["arm_updates"] > counts_before["arm_updates"], "Arm updater stopped during locomotion")
            _require(
                counts["finger_updates"] > counts_before["finger_updates"], "Finger updater stopped during locomotion"
            )
            if action == "forward":
                # An ordinary blink pauses the request. Holding the same mouth
                # expression through reopening resumes it after confirmation,
                # without requiring full facial neutral or counting a new action.
                fixture["expression"] = {"JawDrop": 0.9, "EyesClosedL": 0.9, "EyesClosedR": 0.9}
                await wait_sim(0.05)
                entry["blink_paused"] = snapshot()
                command = face.controller.command
                _require(command.state == "blink_pause", "Held forward did not enter blink pause")
                _require(command.target == "forward", "Blink pause discarded the held mouth target")
                _require(not any(command.velocity), "Blink pause retained facial velocity")
                _require(
                    not any(float(value) for value in example._base_command), "Blink pause retained filtered intent"
                )
                _require(command.activation_count == baseline + 1, "Blink pause counted a new activation")
                await wait_sim(0.25)
                fixture["expression"] = "forward"
                await wait_sim(0.05)
                entry["blink_reconfirming"] = snapshot()
                command = face.controller.command
                _require(command.state == "confirming", "Open eyes did not reconfirm the held mouth expression")
                _require(not any(command.velocity), "Blink recovery moved before confirmation")
                await wait_sim(face.controller.config.confirm_s + 0.05)
                entry["blink_resumed"] = snapshot()
                command = face.controller.command
                _require(command.busy and command.target == "forward", "Held forward did not resume after the blink")
                _require(command.activation_count == baseline + 1, "Blink recovery counted a new forward activation")
                if not preview:
                    _require(command.velocity[0] > 0, "Blink recovery did not restore forward intent")
                    before_heading = entry["held"]["status"]["heading"]["target_yaw_rad"]
                    resumed_heading = entry["blink_resumed"]["status"]["heading"]["target_yaw_rad"]
                    _require(
                        abs((resumed_heading - before_heading + math.pi) % (2 * math.pi) - math.pi) < 1e-6,
                        "Blink recovery changed the original forward heading",
                    )
                entry["blink_pause_resume_passed"] = True
            # The last translation exercises stale input while still holding.
            fixture["stale"] = action == "forward"
            if action != "forward":
                fixture["expression"] = None
            await wait_sim(0.02)
            entry["released"] = snapshot()
            _require(not any(face.controller.command.velocity), "Released/stale expression retained intent")
            if fixture["stale"]:
                _require(
                    not any(float(value) for value in example._base_command), "Stale input did not zero the filter"
                )
            await wait_sim(0.35)
            fixture.update(expression=None, valid=True, stale=False)
            await wait_sim(4.65 if not preview else 0.35)
            entry["end"] = snapshot()
            _require(not any(face.controller.command.velocity), "Stopped facial input retained intent")
            _require(
                max(abs(float(value)) for value in example._base_command) < 1e-5,
                "Filtered intent did not settle to zero",
            )
            start, held, end = entry["activated"], entry["held"], entry["end"]
            dx, dy = held["position_m"][0] - start["position_m"][0], held["position_m"][1] - start["position_m"][1]
            forward = dx * math.cos(start["yaw_rad"]) + dy * math.sin(start["yaw_rad"])
            lateral = -dx * math.sin(start["yaw_rad"]) + dy * math.cos(start["yaw_rad"])
            yaw_delta = (held["yaw_rad"] - start["yaw_rad"] + math.pi) % (2 * math.pi) - math.pi
            entry["held_forward_m"] = forward
            entry["held_lateral_m"] = lateral
            entry["held_yaw_rad"] = yaw_delta
            entry["signed_progress"] = {
                "forward": forward,
                "back": -forward,
                "strafe_left": lateral,
                "strafe_right": -lateral,
                "turn_left": yaw_delta,
                "turn_right": -yaw_delta,
            }[action]
            if not preview:
                if action in ("forward", "back", "strafe_left", "strafe_right"):
                    active_samples = [
                        start,
                        held,
                        *[
                            sample
                            for sample in report["motion_samples"]
                            if start["control_sim_time_s"] <= sample["control_sim_time_s"] <= held["control_sim_time_s"]
                        ],
                    ]
                    entry["held_heading_error_degrees"] = math.degrees(yaw_delta)
                    entry["max_sampled_heading_error_degrees"] = max(
                        abs(math.degrees((s["yaw_rad"] - start["yaw_rad"] + math.pi) % (2 * math.pi) - math.pi))
                        for s in active_samples
                    )
                    direction = start["yaw_rad"] + (math.pi / 2 if action.startswith("strafe") else 0)
                    entry["max_sampled_cross_track_m"] = max(
                        abs(
                            -(s["position_m"][0] - start["position_m"][0]) * math.sin(direction)
                            + (s["position_m"][1] - start["position_m"][1]) * math.cos(direction)
                        )
                        for s in active_samples
                    )
                    _require(
                        abs(entry["held_heading_error_degrees"]) <= 3.0,
                        "Held translation ended more than 3 degrees off heading",
                    )
                    _require(
                        entry["max_sampled_heading_error_degrees"] <= 6.0,
                        "Held translation exceeded 6 degrees sampled yaw error",
                    )
                    _require(
                        entry["max_sampled_cross_track_m"] <= 0.05,
                        "Held translation veered more than 5 cm off its starting line",
                    )
                _require(
                    entry["signed_progress"] > 0.01,
                    "Physical movement did not make progress in the requested direction",
                )
                _require(
                    math.dist(start["position_m"][:2], end["position_m"][:2]) < report["speed_scale"],
                    "One action exceeded the scaled test corridor",
                )
                samples = [
                    sample
                    for sample in report["motion_samples"]
                    if sample["control_sim_time_s"] >= end["control_sim_time_s"] - 1.0
                ]
                _require(len(samples) >= 5, "Insufficient final stopping-window samples")
                entry["stop_window_net_drift_m"] = math.dist(samples[0]["position_m"][:2], end["position_m"][:2])
                entry["stop_window_yaw_drift_rad"] = abs(
                    (end["yaw_rad"] - samples[0]["yaw_rad"] + math.pi) % (2 * math.pi) - math.pi
                )
                _require(entry["stop_window_net_drift_m"] < 0.04, "Physical base kept drifting after relaxation")
                _require(entry["stop_window_yaw_drift_rad"] < 0.08, "Physical base kept turning after relaxation")
                _require(
                    math.hypot(*end["linear_velocity_m_s"][:2]) < 0.10, "Residual physical speed exceeded stop limit"
                )
            entry["passed"] = True
            print(f"Facial validation: {action} passed ({fixture['elapsed_s']:.2f} simulated seconds)", flush=True)

        report["phase"] = "reset_disarms"
        example._reset_gesture_locomotion()
        _require(not face.enabled, "Scene reset did not disarm facial locomotion")
        _require(not any(float(value) for value in example._base_command), "Reset retained base intent")
        report["reset_disarms_passed"] = True
        report["final"] = snapshot()
        report["captures"].append(await _capture_camera(example, output / f"{run_id}-final.png"))
        check_health()
        if preview:
            observed = [*report["motion_samples"], report["final"]]
            origin = report["start"]
            displacement = max(math.dist(sample["position_m"], origin["position_m"]) for sample in observed)
            rotation = max(
                abs((sample["yaw_rad"] - origin["yaw_rad"] + math.pi) % (2 * math.pi) - math.pi) for sample in observed
            )
            report["preview_max_sampled_position_delta_m"] = displacement
            report["preview_max_sampled_yaw_delta_rad"] = rotation
            _require(displacement <= 1e-4 and rotation <= 1e-4, "Anchored preview moved")
        report.update(passed=True, phase="complete")
    finally:
        module.time = real_time
        try:
            timeline.pause()
            example._reset_gesture_locomotion()
        finally:
            for owner, name, existed, value in reversed(saved):
                if existed:
                    setattr(owner, name, value)
                else:
                    vars(owner).pop(name, None)
            report["fixture_restored"] = module.time is real_time and example._xr_core is native_xr


def main() -> int:
    """Launch one isolated native check, preserve evidence, and close its process."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("preview", "move"), default="preview")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hold-s", type=float, default=1.5)
    parser.add_argument("--back-hold-s", type=float, default=2.5)
    parser.add_argument("--forward-speed", type=float, default=0.20)
    parser.add_argument("--backward-speed", type=float, default=0.15)
    parser.add_argument("--lateral-speed", type=float, default=0.15)
    parser.add_argument("--yaw-speed", type=float, default=0.25)
    parser.add_argument(
        "--speed-scale", type=float, default=1.0, help="Multiply base facial/gaze requests once, in [1,3]"
    )
    parser.add_argument("--load-timeout", type=float, default=240.0)
    parser.add_argument("--test-timeout", type=float, default=480.0)
    args, kit_args = parser.parse_known_args()
    if not math.isfinite(args.speed_scale) or not 1.0 <= args.speed_scale <= 3.0:
        parser.error("--speed-scale must be finite and in [1,3]")
    for name, upper in (
        ("hold_s", 10),
        ("back_hold_s", 10),
        ("forward_speed", 0.65),
        ("backward_speed", 0.45),
        ("lateral_speed", 0.45),
        ("yaw_speed", 0.75),
    ):
        value = getattr(args, name)
        effective = value * args.speed_scale if name.endswith("_speed") else value
        if not math.isfinite(value) or not 0 < effective <= upper:
            parser.error(f"--{name.replace('_', '-')} after speed scaling must be finite and in (0, {upper}]")
    for name in ("load_timeout", "test_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"face-{args.mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    report_path = output / f"{run_id}.json"
    report = {
        "passed": False,
        "phase": "starting_application",
        "mode": args.mode,
        "control_mode": "hold",
        "speed_scale": args.speed_scale,
        "requested_speeds": {
            "forward_m_s": args.forward_speed * args.speed_scale,
            "backward_m_s": args.backward_speed * args.speed_scale,
            "lateral_m_s": args.lateral_speed * args.speed_scale,
            "yaw_rad_s": args.yaw_speed * args.speed_scale,
        },
        "run_id": run_id,
        "hold_s": args.hold_s,
        "back_hold_s": args.back_hold_s,
        "fixture": "Real warehouse, gait and body arbitration; synthetic face samples and XR readiness",
        "clock": "Synthetic monotonic time advanced by physics dt inside facial input callback only",
        "hardware_tracking_tested": False,
        "osc_transport_tested": False,
        "payload_tested": False,
        "headset_visual_acceptance_tested": False,
        "rendered_image_content_reviewed": False,
        "actions": [],
        "captures": [],
    }
    _write_report(report_path, report)
    print(f"Facial validation report: {report_path}", flush=True)
    preview = "1" if args.mode == "preview" else "0"
    os.environ.update(
        {
            "G1_GESTURE_WALKING": "1",
            "G1_LOCOMOTION_INPUT": "gaze",
            "G1_TELEOP_SPEED_SCALE": str(args.speed_scale),
            "G1_POINTING_NAVIGATION": "0",
            "G1_OPTICAL_HANDS_ONLY": "1",
            "G1_WEBCAM_ENABLED": "0",
            "G1_GAZE_PREVIEW": preview,
            "G1_GAZE_MODE": "hold",
            "G1_GAZE_DWELL_S": "0",
            "G1_FACE_ENABLED": "1",
            "G1_FACE_PREVIEW": preview,
            "G1_FACE_MODE": "hold",
            "G1_FACE_OSC_PORT": "0",
            "G1_FACE_FORWARD_SPEED": str(args.forward_speed),
            "G1_FACE_BACKWARD_SPEED": str(args.backward_speed),
            "G1_FACE_LATERAL_SPEED": str(args.lateral_speed),
            "G1_FACE_YAW_SPEED": str(args.yaw_speed),
            "G1_FACE_CONFIRM_S": "0.18",
            "G1_FACE_NEUTRAL_S": "0.25",
            "G1_FACE_ACTIVATION_THRESHOLD": "0.65",
            "G1_FACE_RELEASE_THRESHOLD": "0.35",
            "G1_FACE_WINK_THRESHOLD": "0.45",
            "G1_FACE_RIGHT_WINK_THRESHOLD": "0.30",
            "G1_FACE_RIGHT_WINK_RELEASE_THRESHOLD": "0.20",
            "G1_FACE_RIGHT_WINK_CONFIRM_S": "0.50",
            "G1_FACE_RIGHT_MOUTH_THRESHOLD": "0.55",
            "G1_FACE_JAW_THRESHOLD": "0.45",
            "G1_FACE_PUCKER_THRESHOLD": "0.40",
            "G1_FACE_MOUTH_SIDE_THRESHOLD": "0.45",
            "G1_FACE_EYE_OPEN_THRESHOLD": "0.15",
        }
    )
    sys.argv = [sys.argv[0], *kit_args]
    simulation_app = example = task = None
    started = time.perf_counter()
    try:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp(
            {"headless": True, "width": 1280, "height": 960, "renderer": "RayTracedLighting"}
        )
        module = importlib.import_module("isaacsim.robot.policy.examples.interactive.humanoid.humanoid_example")
        expected = ROOT / "source/extensions/isaacsim.robot.policy.examples"
        _require(Path(module.__file__).resolve().is_relative_to(expected), "Isaac loaded a different policy extension")
        report["loaded_module"] = module.__file__
        example = module.HumanoidExample()
        example._behavioral_data_enabled = False
        report["phase"] = "loading_scene"
        task = asyncio.ensure_future(_validate(example, report, output, run_id))
        last_progress = 0.0
        while not task.done():
            now = time.perf_counter()
            test_start = report.get("test_started_wall_monotonic_s")
            deadline = test_start + args.test_timeout if test_start is not None else started + args.load_timeout
            if now > deadline:
                task.cancel()
                raise TimeoutError(f"Wall deadline exceeded during {report['phase']}")
            if now - last_progress >= 5:
                report["wall_elapsed_s"] = now - started
                _write_report(report_path, report)
                print(f"Facial validation: {report['phase']} ({now - started:.1f} wall seconds)", flush=True)
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
                    report.update(passed=False, cancellation_error=repr(error))
        if example is not None:
            try:
                example.physics_cleanup()
            except Exception as error:
                report.update(passed=False, cleanup_error=repr(error))
        report["wall_elapsed_s"] = time.perf_counter() - started
        _write_report(report_path, report)
        print(f"Facial validation {'passed' if report['passed'] else 'FAILED'}: {report_path}", flush=True)
        if simulation_app is not None:
            try:
                simulation_app.close(exit_code=0 if report["passed"] else 1)
            except Exception as error:
                report.update(passed=False, close_error=repr(error))
                _write_report(report_path, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
