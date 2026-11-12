#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate gaze controls in an owned headless Isaac process with synthetic eyes.

Run with the installed Isaac Sim python.bat and this checkout's entire extension:

    python.bat tools/validate_gaze_walking.py --mode preview --output-dir OUTPUT \
        --ext-path POLICY_EXTENSION --enable isaacsim.robot.policy.examples-5.2.11 \
        --/app/vulkan=false --/metricsAssembler/changeListenerEnabled=false

Use a fresh invocation with --mode move for the physical GR00T scene. Add
--control-mode hold to verify sustained fixation and look-away/eye-loss stops;
--control-mode step retains the original short-action checks. The hold
profile defaults to .60/.35 m/s forward/reverse and .40 rad/s yaw. The first
back action holds for --back-hold-s 2.5 to create actual clearance; the remaining
actions use --hold-s 1.5 to retain the one-meter measured travel bound. Explicit
--forward-speed, --backward-speed and --yaw-speed overrides are supported;
step defaults retain .20/.15 m/s and .25 rad/s. Larger/longer requests still
must pass real corridor clearance and the one-meter measured travel bound. The
production warehouse, camera, panel hit tests, occlusion checks, actor, motor loop,
and health guards remain active. No robot root/joint state or payload is injected.
The eye descriptors are synthetic, and only the synchronous gaze-selection call
receives a simulated wall clock advanced by physics dt. This avoids confusing a
slow headless render with an eye-input gap; it does NOT validate real wall-time
freshness, XR transport, human fixation, optical hand tracking, or carrying a box.
Hand/finger updater call counts establish execution only, not successful tracking.

The output directory is required. Unique JSON and PNG filenames retain partial
failure evidence. Only this script's SimulationApp is closed. Native application
startup or a blocking asset load cannot be preempted by a Python wall deadline.
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

ROOT = Path(__file__).resolve().parents[1]
# Back away from the front shelf before applying the conservative full-turn
# clearance check. Every relocation still goes through a real gaze command.
ACTIONS = ("back", "turn_left", "turn_right", "forward")


def _require(condition: bool, message: str) -> None:
    """Fail an acceptance condition even when Python assertions are disabled."""
    if not condition:
        raise RuntimeError(message)


def _write_report(path: Path, report: dict) -> None:
    """Persist a bounded progress/failure report without serializing native objects."""
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


async def _capture_camera(example: object, path: Path) -> dict:
    """Capture the actual authored head camera with a viewport or render product."""
    import omni.kit.app
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()
    if viewport is not None:
        viewport.camera_path = example._head_camera_path
        viewport.resolution = (1280, 960)
        for _ in range(8):
            await omni.kit.app.get_app().next_update_async()
        capture = capture_viewport_to_file(viewport, str(path))
        await capture.wait_for_result()
        backend = "viewport"
    else:
        import numpy as np
        import omni.replicator.core as rep
        from PIL import Image

        product = rep.create.render_product(example._head_camera_path, (1280, 960))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        try:
            annotator.attach(product)
            await rep.orchestrator.step_async(rt_subframes=4, pause_timeline=False, delta_time=0.0)
            pixels = np.asarray(annotator.get_data(do_array_copy=True))
            _require(pixels.ndim == 3 and pixels.shape[:2] == (960, 1280), "Invalid rendered camera dimensions")
            Image.fromarray(pixels).save(path)
        finally:
            annotator.detach()
            product.destroy()
        backend = "replicator_render_product"
    # The native capture future can complete before its asynchronous PNG write.
    # Keep pumping Kit and wait for a complete image, bounded by wall time.
    from PIL import Image

    deadline = time.perf_counter() + 10.0
    while True:
        try:
            with Image.open(path) as rendered:
                rendered.verify()
            break
        except (OSError, SyntaxError):
            _require(time.perf_counter() < deadline, "Camera capture did not finish writing a PNG")
            await omni.kit.app.get_app().next_update_async()
    return {"path": str(path), "camera": example._head_camera_path, "backend": backend, "resolution": [1280, 960]}


async def _validate(example: object, module: object, report: dict, output: Path, run_id: str) -> None:
    """Load the real scene, inject only eye input, and measure the resulting actions."""
    import numpy as np
    import omni.kit.app
    import omni.timeline
    from isaacsim.robot.policy.examples.interactive.humanoid.eye_gaze_tracker import EyeGazeTracker
    from pxr import Gf

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    await example.load_world_async()
    while not example._physics_ready:
        _require(not example._articulation_health_fault, str(example._articulation_health_fault))
        await app.next_update_async()
    report["test_started_wall_monotonic_s"] = time.perf_counter()
    report["phase"] = "initializing_fixture"
    _require(example._gaze_walking_enabled, "The loaded scene is not the gaze profile")
    _require(example._optical_hands_only, "The gaze fixture must leave both hands optical-only")
    preview = report["mode"] == "preview"
    hold = report["control_mode"] == "hold"
    dwell_s = example._gaze_locomotion.config.dwell_s
    config = example._gaze_locomotion.config
    report["effective_profile"] = {
        "speed_scale": example._teleop_speed_scale,
        "forward_m_s": config.forward_speed,
        "backward_m_s": config.backward_speed,
        "yaw_rad_s": config.yaw_speed,
        "heading_gain": config.heading_gain,
        "heading_max_yaw_speed": config.heading_max_yaw_speed,
        "command_attack_s": example._command_attack_time,
        "command_release_s": example._command_release_time,
    }
    _require(example._gaze_locomotion.config.preview is preview, "Unexpected preview configuration")
    _require(example._gaze_locomotion.config.control_mode == report["control_mode"], "Wrong gaze control mode")
    _require(example._g1_locomotion == ("stationary" if preview else "policy"), "Wrong robot base topology")
    _require(not example._pointing_navigation_enabled, "Pointing input must be disabled")

    fixture = {"target": None, "eye_valid": True, "elapsed_s": 0.0, "last_state": None}
    counts = {"gaze_updates": 0, "arm_updates": 0, "finger_updates": 0}
    report["updater_calls"] = counts
    report["state_transitions"] = []
    report["motion_samples"] = []
    saved_tracker = example._eye_gaze_tracker
    native_xr = example._xr_core
    saved_methods = {}
    real_time = module.time
    selection_clock = SimpleNamespace(
        perf_counter=lambda: fixture["elapsed_s"], time=time.time, monotonic=time.monotonic
    )

    def vector(value: object, size: int) -> list[float]:
        values = np.asarray(example._first_pose_value(value), dtype=float).reshape(-1)
        _require(values.shape == (size,) and bool(np.isfinite(values).all()), "Invalid measured physics vector")
        return values.tolist()

    def snapshot() -> dict:
        positions, quaternions = example.g1.robot.get_world_poses()
        linear, angular = example.g1.robot.get_velocities()
        p, q, v, w = (vector(positions, 3), vector(quaternions, 4), vector(linear, 3), vector(angular, 3))
        norm = math.sqrt(sum(component * component for component in q))
        _require(norm > 1e-6, "Invalid measured root quaternion")
        qw, qx, qy, qz = (component / norm for component in q)
        return {
            "control_sim_time_s": fixture["elapsed_s"],
            "wall_monotonic_s": time.perf_counter(),
            "position_m": p,
            "quaternion_wxyz": q,
            "yaw_rad": math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)),
            "linear_velocity_m_s": v,
            "angular_velocity_rad_s": w,
            "status": example.gaze_locomotion_status(),
            "filtered_command": [float(value) for value in example._base_command],
        }

    class SyntheticEye:
        """Supply a tracked combined-eye descriptor aimed at displayed panel geometry."""

        def get_name(self) -> str:
            return "synthetic_combined_eye"

        def get_pose_names(self) -> tuple[str, ...]:
            return ("gaze_ext",)

        def get_virtual_world_pose_desc(self, name: str) -> SimpleNamespace:
            if name != "gaze_ext" or not fixture["eye_valid"]:
                return SimpleNamespace(validity_flags=0, pose_matrix=None)
            camera = Gf.Matrix4d(example._head_camera_transform_op.Get())
            target = fixture["target"]
            x, y = example._gaze_controls.BUTTON_CENTERS[target] if target is not None else (0.0, 0.0)
            direction = camera.TransformDir(Gf.Vec3d(x, y, -example._gaze_controls.DEPTH)).GetNormalized()
            pose = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, -1), direction))
            pose.SetTranslateOnly(camera.ExtractTranslation())
            return SimpleNamespace(validity_flags=0xF, pose_matrix=pose)

    eye = SyntheticEye()

    class SyntheticXR:
        """Advertise a synthetic active session only to the isolated eye tracker."""

        def is_xr_enabled(self) -> bool:
            return True

        def is_xr_display_enabled(self) -> bool:
            return True

        def is_xr_viewport_enabled(self) -> bool:
            return True

        def get_current_profile(self) -> SimpleNamespace:
            return SimpleNamespace(is_enabled=lambda: True, get_name=lambda: "synthetic_gaze_validation")

        def get_input_device(self, handle: str) -> SyntheticEye | None:
            return eye if handle in EyeGazeTracker.UNIFIED_DEVICES else None

        def get_all_input_devices(self) -> list[SyntheticEye]:
            return [eye]

    def install(name: str, replacement: object) -> None:
        saved_methods[name] = (name in vars(example), vars(example).get(name))
        setattr(example, name, replacement)

    original_gaze = example._update_gaze_locomotion
    original_arms = example._update_g1_arms_from_hand_tracking
    original_fingers = example._update_g1_fingers

    def update_gaze(dt: float) -> None:
        fixture["elapsed_s"] += float(dt)
        counts["gaze_updates"] += 1
        previous_time = module.time
        module.time = selection_clock
        try:
            original_gaze(dt)
        finally:
            module.time = previous_time
        state = example._gaze_locomotion.command.state
        if state != fixture["last_state"]:
            report["state_transitions"].append(snapshot())
            fixture["last_state"] = state
        if preview:
            _require(not any(example._gaze_locomotion.command.velocity), "Preview emitted a movement request")
            _require(not any(float(value) for value in example._base_command), "Preview retained a base command")

    def update_arms() -> None:
        counts["arm_updates"] += 1
        original_arms()

    def update_fingers() -> None:
        counts["finger_updates"] += 1
        original_fingers()

    def check_health() -> None:
        _require(example._xr_core is native_xr, "The fixture replaced native XR ownership")
        _require(not example._articulation_health_fault, str(example._articulation_health_fault))
        _require(not example.g1.locomotion_fault, str(example.g1.locomotion_fault))
        _require(
            not example._physics_step_error_logged,
            "Physics callback errors: " + str(example._physics_step_error_logged),
        )
        _require(example.g1.robot.is_physics_tensor_entity_valid(), "Physics tensor views became invalid")
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

    def preflight(action: str, start: dict) -> float | None:
        """Require real floor/corridor space for requested travel plus braking margin."""
        if preview:
            return None
        scene = example._navigation_scene
        position = start["position_m"]
        if action.startswith("turn_"):
            _require(scene.is_route_corner_clear(position), "Turn fixture lacks body/stepping clearance")
            return None
        else:
            config = example._gaze_locomotion.config
            speed = config.forward_speed if action == "forward" else config.backward_speed
            duration = report["back_hold_s"] if action == "back" else report["hold_s"]
            requested_distance = speed * (duration + 0.15) if hold else config.distance_m
            distance = max(0.60, requested_distance + 0.35)
            sign = 1 if action == "forward" else -1
            yaw = start["yaw_rad"]
            goal = (position[0] + sign * distance * math.cos(yaw), position[1] + sign * distance * math.sin(yaw), 0.0)
            _require(scene.floor_at(goal) is not None, "Translation fixture lacks verified floor")
            _require(
                scene.is_path_clear(position, goal), "Translation fixture corridor is blocked: " + str(scene.last_error)
            )
            return distance

    try:
        if saved_tracker is not None:
            saved_tracker.cleanup()
        example._eye_gaze_tracker = EyeGazeTracker(
            SyntheticXR(), draw_ray=True, highlights=example._material_highlights
        )
        install("_update_gaze_locomotion", update_gaze)
        install("_update_g1_arms_from_hand_tracking", update_arms)
        install("_update_g1_fingers", update_fingers)
        example._reset_gesture_locomotion()
        await wait_sim(0.30)
        report["start"] = snapshot()
        report["captures"].append(await _capture_camera(example, output / f"{run_id}-ready.png"))
        check_health()

        for action in ACTIONS:
            report["phase"] = action
            entry = {"action": action, "passed": False, "start": snapshot()}
            entry["requested_hold_s"] = (
                (report["back_hold_s"] if action == "back" else report["hold_s"]) if hold else None
            )
            report["actions"].append(entry)
            entry["preflight_translation_clearance_m"] = preflight(action, entry["start"])
            baseline_count = example._gaze_locomotion.command.activation_count
            fixture["target"] = None
            await wait_sim(0.10)
            _require(example._gaze_locomotion.command.state == "idle", "Valid look-away did not arm selection")

            fixture["target"] = action
            transition_start = len(report["state_transitions"])
            if dwell_s > 0:
                await wait_sim(dwell_s * 0.4)
                _require(
                    example._gaze_locomotion.command.activation_count == baseline_count,
                    "A short glance bypassed the configured dwell",
                )
                entry["short_glance"] = snapshot()
                fixture["target"] = None
                await wait_sim(0.10)
                fixture["target"] = action
                await wait_sim(dwell_s + 0.10)
            else:
                await advance()
            command = example._gaze_locomotion.command
            _require(
                command.activation_count == baseline_count + 1 and command.last_action == action,
                "Selection did not fire exactly once at the configured delay; inspect occlusion/tracking status",
            )
            entry["activated"] = snapshot()
            await wait_sim(entry["requested_hold_s"] if hold else 1.0)
            _require(
                example._gaze_locomotion.command.activation_count == baseline_count + 1,
                "Holding a gaze button repeated the action",
            )
            entry["held"] = snapshot()
            if hold and not preview:
                _require(example._gaze_locomotion.command.state == "active", "Held gaze did not continue moving")
                _require(any(example._gaze_locomotion.command.velocity), "Held gaze lost its movement request")
            # Exercise real invalid-eye handling as well as ordinary look-away.
            lose_eye = hold and action == "forward"
            fixture["eye_valid"] = not lose_eye
            fixture["target"] = None
            entry["release_reason"] = "eye_tracking_lost" if lose_eye else "looked_away"
            if hold and not preview:
                await wait_sim(0.01)
                entry["released"] = snapshot()
                _require(not any(example._gaze_locomotion.command.velocity), "Look-away retained a movement request")
                if lose_eye:
                    _require(
                        not any(float(value) for value in example._base_command), "Eye loss did not stop immediately"
                    )
            end_sim = fixture["elapsed_s"] + 12.0
            while example._gaze_locomotion.command.state in ("active", "stopping"):
                _require(fixture["elapsed_s"] < end_sim, "Movement failed to finish before the fixture deadline")
                await advance()
            fixture["eye_valid"] = True
            # Hold release is an input event, not proof that the pelvis settled.
            # Continue real physics beyond the former four-second pause deadline
            # while the independent balance, foot-support drift and runaway guards run.
            await wait_sim(5.0 if hold and not preview else 0.10)
            entry["end"] = snapshot()
            transitions = report["state_transitions"][transition_start:]
            entry["states"] = [sample["status"]["state"] for sample in transitions]
            _require(not any(example._gaze_locomotion.command.velocity), "Finished action retained a movement request")
            _require(
                max(abs(float(value)) for value in example._base_command) < 1e-5,
                "Finished action retained meaningful filtered movement",
            )
            if not preview:
                _require(
                    "active" in entry["states"] and (hold or "stopping" in entry["states"]),
                    "Missing physical action/stop states",
                )
                if not hold:
                    _require(
                        any("Movement target reached" in sample["status"]["detail"] for sample in transitions),
                        "The physical action timed out instead of reaching its measured target",
                    )
                _require(
                    not any("timed out" in sample["status"]["detail"] for sample in transitions),
                    "The physical action reported a timeout",
                )
                entry["motion_origin"] = next(sample for sample in transitions if sample["status"]["state"] == "active")
                entry["goal_sample"] = (
                    entry["held"]
                    if hold
                    else next(
                        sample for sample in transitions if "Movement target reached" in sample["status"]["detail"]
                    )
                )
                entry["settled_sample"] = next(
                    (
                        sample
                        for sample in transitions
                        if sample["status"]["state"] == "awaiting_release"
                        and "Movement target reached" in sample["status"]["detail"]
                    ),
                    None,
                )
                if not hold:
                    _require(entry["settled_sample"] is not None, "No measured settling completion was observed")
                else:
                    stop = entry["end"]["status"]["last_stop"]
                    _require(
                        stop.get("mode") == "hold"
                        and stop.get("action") == action
                        and stop.get("decision") == "zero_requested_external_motion_guard_active"
                        and stop.get("settling_evaluated") is False,
                        "Current hold stop diagnostics were lost or incorrectly claim measured settling",
                    )
                start, end = entry["motion_origin"], entry["end"]
                dx = end["position_m"][0] - start["position_m"][0]
                dy = end["position_m"][1] - start["position_m"][1]
                yaw_change = (end["yaw_rad"] - start["yaw_rad"] + math.pi) % (2 * math.pi) - math.pi
                entry["planar_displacement_m"] = math.hypot(dx, dy)
                entry["yaw_change_rad"] = yaw_change
                entry["forward_displacement_m"] = dx * math.cos(start["yaw_rad"]) + dy * math.sin(start["yaw_rad"])
                entry["signed_final_progress"] = {
                    "forward": entry["forward_displacement_m"],
                    "back": -entry["forward_displacement_m"],
                    "turn_left": yaw_change,
                    "turn_right": -yaw_change,
                }[action]
                _require(entry["signed_final_progress"] > 0.0, "Final physical movement has the wrong sign")
                if hold:
                    held = entry["held"]
                    held_dx = held["position_m"][0] - start["position_m"][0]
                    held_dy = held["position_m"][1] - start["position_m"][1]
                    held_forward = held_dx * math.cos(start["yaw_rad"]) + held_dy * math.sin(start["yaw_rad"])
                    held_yaw = (held["yaw_rad"] - start["yaw_rad"] + math.pi) % (2 * math.pi) - math.pi
                    entry["held_heading_error_degrees"] = math.degrees(held_yaw)
                    entry["held_cross_track_m"] = -held_dx * math.sin(start["yaw_rad"]) + held_dy * math.cos(
                        start["yaw_rad"]
                    )
                    entry["signed_held_progress"] = {
                        "forward": held_forward,
                        "back": -held_forward,
                        "turn_left": held_yaw,
                        "turn_right": -held_yaw,
                    }[action]
                    config = example._gaze_locomotion.config
                    requested_speed = {
                        "forward": config.forward_speed,
                        "back": config.backward_speed,
                        "turn_left": config.yaw_speed,
                        "turn_right": config.yaw_speed,
                    }[action]
                    duration = held["control_sim_time_s"] - start["control_sim_time_s"]
                    _require(duration > 0, "Held interval has no measured simulation duration")
                    entry["held_duration_s"] = duration
                    entry["requested_speed"] = requested_speed
                    entry["speed_units"] = "m/s" if action in ("forward", "back") else "rad/s"
                    entry["mean_signed_held_speed"] = entry["signed_held_progress"] / duration
                    entry["mean_speed_request_ratio"] = entry["mean_signed_held_speed"] / requested_speed
                    _require(
                        entry["mean_speed_request_ratio"] > 0.20,
                        "Mean signed physical progress was below 20 percent of the requested speed",
                    )
                    entry["final_native_speed_m_s"] = math.hypot(*end["linear_velocity_m_s"][:2])
                    entry["final_native_yaw_speed_rad_s"] = abs(end["angular_velocity_rad_s"][2])
                    _require(
                        entry["final_native_speed_m_s"] <= 0.10 and entry["final_native_yaw_speed_rad_s"] <= 0.25,
                        "Physical residual speed remained above the fixture stop limits after five seconds",
                    )
                    old_goal = (
                        example._gaze_locomotion.config.distance_m
                        if action in ("forward", "back")
                        else math.radians(example._gaze_locomotion.config.turn_degrees)
                    )
                    _require(entry["signed_held_progress"] > old_goal, "Held movement stopped at the old step limit")
                    if action in ("forward", "back"):
                        active_samples = [
                            sample
                            for sample in report["motion_samples"]
                            if start["control_sim_time_s"] <= sample["control_sim_time_s"] <= held["control_sim_time_s"]
                        ]
                        active_samples.extend((start, held))
                        entry["max_sampled_heading_error_degrees"] = max(
                            abs(
                                math.degrees((sample["yaw_rad"] - start["yaw_rad"] + math.pi) % (2 * math.pi) - math.pi)
                            )
                            for sample in active_samples
                        )
                        entry["max_sampled_heading_correction_rad_s"] = max(
                            abs(sample["status"]["heading"]["correction_rad_s"]) for sample in active_samples
                        )
                        if report["heading_hold"]:
                            _require(
                                abs(entry["held_heading_error_degrees"]) <= 3.0,
                                "Translation finished its held interval more than 3 degrees from the starting heading",
                            )
                            _require(
                                entry["max_sampled_heading_error_degrees"] <= 6.0,
                                "Translation exceeded the 6 degree sampled heading envelope",
                            )
                _require(
                    entry["planar_displacement_m"] < report["speed_scale"],
                    "Short-action fixture exceeded its scaled test corridor",
                )
            entry["passed"] = True
            print(f"Gaze validation: {action} passed ({fixture['elapsed_s']:.2f} simulated seconds)", flush=True)

        report["phase"] = "capturing_final"
        report["final"] = snapshot()
        report["captures"].append(await _capture_camera(example, output / f"{run_id}-final.png"))
        # Record the actual scene masks and a second, settled render so temporal
        # reconstruction artifacts can be distinguished from incorrect USD state.
        report["status_opacities"] = {
            name: float(value.Get())
            for name, value in example._gaze_controls._opacity_inputs.items()
            if name.startswith("status_")
        }
        for _ in range(60):
            await app.next_update_async()
        report["captures"].append(await _capture_camera(example, output / f"{run_id}-settled.png"))
        check_health()
        _require(counts["arm_updates"] > 0 and counts["finger_updates"] > 0, "Hand/finger updater execution was lost")
        if preview:
            start = report["start"]
            observed = [*report["motion_samples"], report["final"]]
            displacement = max(math.dist(sample["position_m"], start["position_m"]) for sample in observed)
            rotation = max(
                2
                * math.acos(
                    min(
                        1.0,
                        abs(
                            float(np.dot(sample["quaternion_wxyz"], start["quaternion_wxyz"]))
                            / (np.linalg.norm(sample["quaternion_wxyz"]) * np.linalg.norm(start["quaternion_wxyz"]))
                        ),
                    )
                )
                for sample in observed
            )
            report["preview_max_sampled_position_delta_m"] = displacement
            report["preview_max_sampled_rotation_delta_rad"] = rotation
            _require(displacement <= 1e-4 and rotation <= 1e-4, "Anchored preview moved beyond numerical tolerance")
        report["passed"] = True
        report["phase"] = "complete"
    finally:
        module.time = real_time
        fixture["target"] = None
        try:
            timeline.pause()
            example._reset_gesture_locomotion()
        finally:
            for name, (owned, value) in saved_methods.items():
                if owned:
                    setattr(example, name, value)
                else:
                    vars(example).pop(name, None)
            try:
                if example._eye_gaze_tracker is not saved_tracker:
                    example._eye_gaze_tracker.cleanup()
            finally:
                example._eye_gaze_tracker = saved_tracker
                report["fixture_restored"] = module.time is real_time and example._xr_core is native_xr


def main() -> int:
    """Own one bounded application run and preserve its report on every Python exit."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("preview", "move"), default="preview")
    parser.add_argument("--control-mode", choices=("step", "hold"), default="step")
    parser.add_argument("--dwell-s", type=float, default=None, help="Default: 0 for hold, 0.7 for step")
    parser.add_argument("--heading-hold", choices=("0", "1"), default="1", help="Use 0 for an unassisted baseline")
    parser.add_argument("--heading-gain", type=float, default=3.0, help="Heading feedback gain in 1/s")
    parser.add_argument("--forward-speed", type=float, default=None, help="m/s; default hold .60, step .20")
    parser.add_argument("--backward-speed", type=float, default=None, help="m/s; default hold .35, step .15")
    parser.add_argument("--yaw-speed", type=float, default=None, help="rad/s; default hold .40, step .25")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Multiply base gaze requests once, in [1,3]")
    parser.add_argument(
        "--hold-s", type=float, default=1.5, help="Hold-mode forward/turn fixation after activation, in sim seconds"
    )
    parser.add_argument("--back-hold-s", type=float, default=2.5, help="Initial hold-mode back fixation for clearance")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--load-timeout", type=float, default=180.0)
    parser.add_argument("--test-timeout", type=float, default=180.0)
    args, kit_args = parser.parse_known_args()
    if not math.isfinite(args.speed_scale) or not 1.0 <= args.speed_scale <= 3.0:
        parser.error("--speed-scale must be finite and in [1,3]")
    if args.dwell_s is None:
        args.dwell_s = 0.0 if args.control_mode == "hold" else 0.7
    defaults = (0.60, 0.35, 0.40) if args.control_mode == "hold" else (0.20, 0.15, 0.25)
    for name, default, lower, upper in zip(
        ("forward_speed", "backward_speed", "yaw_speed"), defaults, (0.01, 0.0, 0.01), (0.65, 0.35, 0.40)
    ):
        if getattr(args, name) is None:
            setattr(args, name, default)
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0 or value < lower or value > upper:
            parser.error(f"--{name.replace('_', '-')} must be positive and within [{lower}, {upper}]")
    for name in ("hold_s", "back_hold_s"):
        if not math.isfinite(getattr(args, name)) or not 0 < getattr(args, name) <= 10:
            parser.error(f"--{name.replace('_', '-')} must be finite and in (0, 10]")
    if not math.isfinite(args.dwell_s) or not 0 <= args.dwell_s <= 5:
        parser.error("--dwell-s must be finite and between 0 and 5")
    if not math.isfinite(args.heading_gain) or not 0 <= args.heading_gain <= 5:
        parser.error("--heading-gain must be finite and between 0 and 5")
    for name in ("load_timeout", "test_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"gaze-{args.mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    report_path = output / f"{run_id}.json"
    report = {
        "passed": False,
        "phase": "starting_application",
        "mode": args.mode,
        "control_mode": args.control_mode,
        "speed_scale": args.speed_scale,
        "dwell_s": args.dwell_s,
        "heading_hold": args.heading_hold == "1",
        "heading_gain": args.heading_gain,
        "forward_speed": args.forward_speed,
        "backward_speed": args.backward_speed,
        "yaw_speed": args.yaw_speed,
        "hold_s": args.hold_s,
        "back_hold_s": args.back_hold_s,
        "run_id": run_id,
        "fixture": "Real warehouse/robot/actor; synthetic tracked eye descriptors; no injected hands or payload",
        "dwell_clock": "Synthetic monotonic selection time advanced by physics dt only inside gaze update",
        "wall_gap_behavior_tested": False,
        "hardware_tracking_tested": False,
        "payload_tested": False,
        "headset_visual_acceptance_tested": False,
        "rendered_image_content_reviewed": False,
        "actions": [],
        "captures": [],
        "load_timeout_s": args.load_timeout,
        "test_timeout_s": args.test_timeout,
    }
    _write_report(report_path, report)
    print(f"Gaze validation report: {report_path}", flush=True)
    os.environ.update(
        {
            "G1_GESTURE_WALKING": "1",
            "G1_LOCOMOTION_INPUT": "gaze",
            "G1_TELEOP_SPEED_SCALE": str(args.speed_scale),
            "G1_POINTING_NAVIGATION": "0",
            "G1_OPTICAL_HANDS_ONLY": "1",
            "G1_WEBCAM_ENABLED": "0",
            "G1_GAZE_PREVIEW": "1" if args.mode == "preview" else "0",
            "G1_GAZE_MODE": args.control_mode,
            "G1_GAZE_DWELL_S": str(args.dwell_s),
            "G1_GAZE_HEADING_HOLD": args.heading_hold,
            "G1_GAZE_HEADING_GAIN": str(args.heading_gain),
            "G1_GAZE_HEADING_MAX_YAW_SPEED": "0.15",
            "G1_GAZE_DISTANCE_M": "0.20",
            "G1_GAZE_TURN_DEG": "10",
            "G1_GAZE_FORWARD_SPEED": str(args.forward_speed),
            "G1_GAZE_BACKWARD_SPEED": str(args.backward_speed),
            "G1_GAZE_YAW_SPEED": str(args.yaw_speed),
            "G1_GAZE_MAX_GAP_S": "0.25",
        }
    )
    sys.argv = [sys.argv[0], *kit_args]
    simulation_app = None
    task = None
    example = None
    started = time.perf_counter()
    try:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp(
            {"headless": True, "width": 1280, "height": 960, "renderer": "RayTracedLighting"}
        )
        import carb.settings

        render_settings = carb.settings.get_settings()
        report["render_settings"] = {
            key: render_settings.get(key)
            for key in (
                "/rtx/post/aa/op",
                "/rtx/post/dlss/execMode",
                "/rtx/dldenoiser/responsiveDenoising",
                "/rtx/indirectDiffuse/enabled",
                "/rtx/post/motionblur/enabled",
            )
        }
        module = importlib.import_module("isaacsim.robot.policy.examples.interactive.humanoid.humanoid_example")
        expected = ROOT / "source/extensions/isaacsim.robot.policy.examples"
        _require(Path(module.__file__).resolve().is_relative_to(expected), "Isaac loaded a different policy extension")
        report["loaded_module"] = module.__file__
        example = module.HumanoidExample()
        example._behavioral_data_enabled = False
        report["phase"] = "loading_scene"
        task = asyncio.ensure_future(_validate(example, module, report, output, run_id))
        last_progress = 0.0
        while not task.done():
            now = time.perf_counter()
            test_start = report.get("test_started_wall_monotonic_s")
            deadline = (test_start + args.test_timeout) if test_start is not None else (started + args.load_timeout)
            if now > deadline:
                task.cancel()
                raise TimeoutError(f"Wall deadline exceeded during {report['phase']}")
            if now - last_progress >= 5.0:
                report["wall_elapsed_s"] = now - started
                _write_report(report_path, report)
                print(f"Gaze validation: {report['phase']} ({now - started:.1f} wall seconds)", flush=True)
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
        print(f"Gaze validation {'passed' if report['passed'] else 'FAILED'}: {report_path}", flush=True)
        if simulation_app is not None:
            try:
                # Isaac 6.1 fast shutdown can terminate the process here. Pass
                # the result explicitly; code after this call may never execute.
                simulation_app.close(exit_code=0 if report["passed"] else 1)
            except Exception as error:
                report.update(passed=False, close_error=repr(error))
                _write_report(report_path, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
