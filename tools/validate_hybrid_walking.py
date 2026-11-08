#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate controller takeover of gaze in an owned headless Isaac process.

Run with the installed Isaac python.bat, this checkout's policy extension, and
--mode preview or --mode move --output-dir OUTPUT. Pass the same --ext-path and
--enable arguments used by validate_gaze_walking.py. The real scene, panel ray
tests, arbitration, actor, arm IK, finger/contact logic and health guards run.
Only eye/controller input descriptors are synthetic. No robot, joint or object
state is injected. Grip/trigger tests establish controller hand input processing,
not a successful grasp, real optical tracking, hardware freshness or VR comfort.
The hold profile uses .60/.35 m/s forward/back, .30 m/s lateral and .40 rad/s yaw.
Both full-stick lateral directions are held 1.2 simulated seconds; each full-stick
diagonal is held 1.0 second. Radial normalization is checked before physical caps,
and actual signed lateral travel is required without relaxing the one-meter bound.

The synchronous input update uses a simulated selection clock advanced by physics
dt, avoiding false controller/eye gaps during slow headless rendering. Actual
wall-time freshness is not tested. This script closes only its own SimulationApp.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import math
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from validate_gaze_walking import ROOT, _require, _write_report


async def _validate(example: object, module: object, report: dict) -> None:
    """Exercise real input arbitration and physics using current synthetic poses."""
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
    preview = report["mode"] == "preview"
    _require(example._gaze_walking_enabled and example._gaze_controller_override_enabled, "Hybrid mode is disabled")
    _require(not example._optical_hands_only, "Controller hand fallback is disabled")
    _require(example._gaze_locomotion.config.preview is preview, "Unexpected preview configuration")
    _require(example._gaze_locomotion.config.control_mode == "hold", "Hybrid fixture requires hold mode")
    _require(example._xr_core is not None, "Enable the native XR extension for the normal hand-updater gate")
    _require(not example._pointing_navigation_enabled, "Pointing locomotion must remain disabled")

    fixture = dict(
        elapsed_s=0.0,
        target=None,
        valid=True,
        left_x=0.0,
        left_y=0.0,
        right_x=0.0,
        tracked=True,
        grip=1.0,
        trigger=0.0,
        hand_raise=0.0,
        expect_zero=True,
    )
    saved_methods = {}
    saved_tracker, native_xr, real_time = example._eye_gaze_tracker, example._xr_core, module.time
    counts = dict(input=0, arms=0, fingers=0)
    report["updater_calls"] = counts
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
        p, q, v, w = vector(positions, 3), vector(quaternions, 4), vector(linear, 3), vector(angular, 3)
        norm = math.sqrt(sum(x * x for x in q))
        _require(norm > 1e-6, "Invalid measured root quaternion")
        qw, qx, qy, qz = (x / norm for x in q)
        status = example.gaze_locomotion_status()
        return dict(
            sim_s=fixture["elapsed_s"],
            position_m=p,
            quaternion_wxyz=q,
            yaw_rad=math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)),
            linear_velocity_m_s=v,
            angular_velocity_rad_s=w,
            status=status,
            requested=[float(x) for x in example._controller_command],
            filtered=[float(x) for x in example._base_command],
            arm_sources=dict(example._arm_input_sources),
            finger_sources=dict(example._finger_curl_source),
            finger_curls={side: dict(values) for side, values in example._latest_finger_curls.items()},
            arm_targets={
                side: {
                    name: float(example._smoothed_arm_targets[index])
                    for name, index in indices.items()
                    if index in example._smoothed_arm_targets
                }
                for side, indices in example._g1_arm_dof_indices_by_side.items()
            },
            actor_mode=getattr(getattr(example.g1, "_velocity_policy", None), "active_mode", None),
            synthetic_input={
                name: fixture[name] for name in ("target", "tracked", "left_x", "left_y", "right_x", "grip", "trigger")
            },
        )

    class Eye:
        """Aim a strict tracked eye descriptor at actual authored panel geometry."""

        def get_name(self):
            return "synthetic_hybrid_eye"

        def get_pose_names(self):
            return ("gaze_ext",)

        def get_virtual_world_pose_desc(self, name):
            if name != "gaze_ext" or not fixture["valid"]:
                return SimpleNamespace(validity_flags=0, pose_matrix=None)
            camera = Gf.Matrix4d(example._head_camera_transform_op.Get())
            target = fixture["target"]
            x, y = example._gaze_controls.BUTTON_CENTERS[target] if target else (0.0, 0.0)
            direction = camera.TransformDir(Gf.Vec3d(x, y, -example._gaze_controls.DEPTH)).GetNormalized()
            pose = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, -1), direction))
            pose.SetTranslateOnly(camera.ExtractTranslation())
            return SimpleNamespace(validity_flags=0xF, pose_matrix=pose)

    eye = Eye()

    class EyeXR:
        """Supply session flags to the isolated eye tracker, never native XR ownership."""

        def is_xr_enabled(self):
            return True

        is_xr_display_enabled = is_xr_enabled
        is_xr_viewport_enabled = is_xr_enabled

        def get_current_profile(self):
            return SimpleNamespace(is_enabled=lambda: True, get_name=lambda: "synthetic_hybrid_validation")

        def get_input_device(self, handle):
            return eye if handle in EyeGazeTracker.UNIFIED_DEVICES else None

        def get_all_input_devices(self):
            return [eye]

    class Controller:
        """Expose the same tracked grip and action interfaces as a real controller."""

        def __init__(self, side):
            self.side = side

        def get_name(self):
            return "synthetic_" + self.side + "_controller"

        def get_hand_tracking_data_source(self):
            return "controller"

        def get_pose_names(self):
            return ("grip", "aim")

        def get_pose_desc(self, name):
            if not fixture["tracked"] or name not in self.get_pose_names():
                return SimpleNamespace(validity_flags=0, pose_matrix=None)
            root, yaw = example._get_g1_base_pose_for_arms()
            pose = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw)))
            local = Gf.Vec3d(0.35, 0.24 if self.side == "left" else -0.24, 0.20 + fixture["hand_raise"])
            pose.SetTranslateOnly(root + pose.TransformDir(local))
            return SimpleNamespace(validity_flags=0xF, pose_matrix=pose)

        get_virtual_world_pose_desc = get_pose_desc

        def get_input_names(self):
            return ("thumbstick", "squeeze", "grip", "trigger")

        def has_input(self, name):
            return name in self.get_input_names()

        def has_input_gesture(self, name, gesture):
            return self.has_input(name) and gesture in (("x", "y") if name == "thumbstick" else ("value", "click"))

        def get_input_gesture_value(self, name, gesture):
            if name == "thumbstick":
                return (
                    fixture[self.side + "_x"] if gesture == "x" else fixture["left_y"] if self.side == "left" else 0.0
                )
            value = fixture["trigger"] if name == "trigger" else fixture["grip"]
            return float(value >= 0.5) if gesture == "click" else value

    controllers = {"/user/hand/" + side: Controller(side) for side in ("left", "right")}

    def install(name, method):
        saved_methods[name] = (name in vars(example), vars(example).get(name))
        setattr(example, name, method)

    original_input = example._update_controller_command
    original_devices = example._get_xr_input_device
    original_arms, original_fingers = example._update_g1_arms_from_hand_tracking, example._update_g1_fingers

    def update_input(dt):
        fixture["elapsed_s"] += float(dt)
        counts["input"] += 1
        previous = module.time
        module.time = selection_clock
        try:
            original_input(dt)
        finally:
            module.time = previous
        if fixture["expect_zero"]:
            _require(not any(float(x) for x in example._controller_command), "An interlocked interval emitted intent")
        if preview:
            _require(not any(float(x) for x in example._controller_command), "Preview emitted locomotion intent")
            _require(not any(float(x) for x in example._base_command), "Preview retained a filtered command")

    def update_arms():
        counts["arms"] += 1
        original_arms()

    def update_fingers():
        counts["fingers"] += 1
        original_fingers()

    def health():
        _require(example._xr_core is native_xr, "Native XR ownership changed")
        _require(not example._articulation_health_fault, str(example._articulation_health_fault))
        _require(not example.g1.locomotion_fault, str(example.g1.locomotion_fault))
        _require(not example._physics_step_error_logged, str(example._physics_step_error_logged))
        _require(example.g1.robot.is_physics_tensor_entity_valid(), "Invalid physics tensors")
        _require(timeline.is_playing(), "Timeline unexpectedly paused")
        forward, lateral, yaw = example.gaze_locomotion_status()["controller_requested_velocity"]
        config = example._gaze_locomotion.config
        forward_cap = config.forward_speed if forward >= 0 else config.backward_speed
        _require(abs(yaw) <= config.yaw_speed + 1e-6, "Controller yaw exceeded its cap")
        _require(
            math.hypot(forward / forward_cap, lateral / 0.30) <= 1.0 + 1e-6,
            "Controller translation exceeded the normalized planar caps",
        )
        if preview:
            feedback = example._walking_stop_regulator.status
            _require(not any(feedback.get("effective", ())), "Preview emitted stop-feedback movement")

    async def wait(seconds):
        end = fixture["elapsed_s"] + seconds
        while fixture["elapsed_s"] + 1e-9 < end:
            await app.next_update_async()
            health()
            samples = report["samples"]
            if not samples or fixture["elapsed_s"] - samples[-1]["sim_s"] >= 0.1:
                samples.append(snapshot())

    def checkpoint(name):
        report["phase"] = name
        value = snapshot()
        report["checks"][name] = value
        return value

    def stopped(message):
        _require(not any(float(x) for x in example._controller_command), message)
        _require(example.gaze_locomotion_status()["hybrid"]["source"] != "controller", message)

    def forward_delta(start, end):
        dx, dy = (end["position_m"][i] - start["position_m"][i] for i in (0, 1))
        return dx * math.cos(start["yaw_rad"]) + dy * math.sin(start["yaw_rad"])

    def lateral_delta(start, end):
        dx, dy = (end["position_m"][i] - start["position_m"][i] for i in (0, 1))
        return -dx * math.sin(start["yaw_rad"]) + dy * math.cos(start["yaw_rad"])

    def clearance(forward, lateral, duration):
        """Require floor and a real collision-free corridor along the intended translation."""
        if preview:
            return
        current = snapshot()
        position, yaw = current["position_m"], current["yaw_rad"]
        speed = math.hypot(forward, lateral)
        distance = speed * (duration + 0.15) + 0.35
        dx, dy = forward / speed * distance, lateral / speed * distance
        goal = (
            position[0] + dx * math.cos(yaw) - dy * math.sin(yaw),
            position[1] + dx * math.sin(yaw) + dy * math.cos(yaw),
            0.0,
        )
        scene = example._navigation_scene
        _require(
            scene.floor_at(goal) is not None and scene.is_path_clear(position, goal),
            "No clear lateral/diagonal fixture corridor: " + str(scene.last_error),
        )

    def gaze_active(value, action):
        status = value["status"]
        _require(status["hybrid"]["source"] == "gaze", "Gaze did not own locomotion")
        _require(
            status["state"] == ("preview" if preview else "active") and status["last_action"] == action,
            "Fresh gaze failed to activate the intended button",
        )

    try:
        if saved_tracker is not None:
            saved_tracker.cleanup()
        example._eye_gaze_tracker = EyeGazeTracker(EyeXR(), draw_ray=True, highlights=example._material_highlights)
        install("_get_xr_input_device", lambda handle: controllers.get(handle) or original_devices(handle))
        install("_update_controller_command", update_input)
        install("_update_g1_arms_from_hand_tracking", update_arms)
        install("_update_g1_fingers", update_fingers)
        example._reset_gesture_locomotion()
        await wait(0.4)
        start = checkpoint("ready")

        if not preview:
            position, yaw = start["position_m"], start["yaw_rad"]
            scene = example._navigation_scene
            goal = (position[0] - 0.65 * math.cos(yaw), position[1] - 0.65 * math.sin(yaw), 0.0)
            _require(
                scene.floor_at(goal) is not None and scene.is_path_clear(position, goal),
                "No clear backward fixture corridor",
            )
        fixture.update(target="back", expect_zero=False)
        await wait(0.9)
        away = checkpoint("gaze_back_clearance")
        gaze_active(away, "back")
        if not preview:
            _require(forward_delta(start, away) < -0.015, "Gaze did not physically back away")
        fixture.update(target=None, expect_zero=True)
        await wait(0.4)
        if not preview:
            current = snapshot()
            position, yaw = current["position_m"], current["yaw_rad"]
            goal = (position[0] + 0.60 * math.cos(yaw), position[1] + 0.60 * math.sin(yaw), 0.0)
            _require(
                scene.floor_at(goal) is not None and scene.is_path_clear(position, goal),
                "No clear forward fixture corridor after backing away",
            )
        fixture.update(target="forward", expect_zero=False)
        await wait(0.4)
        gaze = checkpoint("gaze_forward")
        gaze_active(gaze, "forward")
        fixture.update(left_y=-0.575, trigger=0.75, hand_raise=0.04)
        await wait(0.02)
        takeover = checkpoint("stick_takeover")
        _require(takeover["status"]["hybrid"]["source"] == "controller", "Stick failed to take priority")
        if not preview:
            _require(takeover["requested"][0] < 0 and takeover["requested"][1] == 0, "Back override has wrong axes")
        await wait(1.2)
        back = checkpoint("controller_back")
        if not preview:
            _require(forward_delta(takeover, back) < -0.015, "Controller override did not physically reverse")
        for side in ("left", "right"):
            _require(back["arm_sources"].get(side) == "controller", "Normal controller arm fallback was not active")
            _require(back["finger_sources"].get(side) == "controller", "Normal trigger finger fallback was not active")
            _require(back["finger_curls"].get(side, {}).get("index", 0) > 0.3, "Trigger did not curl the fingers")
            earlier, current = gaze["arm_targets"].get(side, {}), back["arm_targets"].get(side, {})
            _require(
                any(abs(value - earlier.get(name, value)) > 0.001 for name, value in current.items()),
                "Tracked grip movement did not change normal IK targets",
            )
        fixture.update(left_y=0.0, expect_zero=True)
        await wait(0.6)
        stopped("Held gaze resumed after stick neutral without a fresh look-away")
        checkpoint("neutral_blocks_held_gaze")

        for name, stick, sign in (("left", -1.0, 1.0), ("right", 1.0, -1.0)):
            clearance(0.0, sign * 0.30, 1.2)
            origin = checkpoint("lateral_" + name + "_start")
            fixture.update(left_x=stick, expect_zero=False)
            await wait(1.2)
            lateral = checkpoint("lateral_" + name)
            command = lateral["status"]["controller_requested_velocity"]
            _require(lateral["status"]["hybrid"]["source"] == "controller", "Left-X failed to acquire locomotion")
            _require(
                abs(command[0]) < 1e-6 and abs(command[2]) < 1e-6 and abs(command[1] - sign * 0.30) < 1e-6,
                "Full lateral stick did not produce the intended capped direction",
            )
            lateral["signed_lateral_progress_m"] = sign * lateral_delta(origin, lateral)
            lateral["held_duration_s"] = lateral["sim_s"] - origin["sim_s"]
            if not preview:
                _require(
                    lateral["signed_lateral_progress_m"] > 0.025,
                    "Measured lateral movement has insufficient or wrong-sign progress",
                )
            fixture.update(left_x=0.0, expect_zero=True)
            await wait(0.6)
            stopped("Held gaze resumed after lateral stick release")
            checkpoint("lateral_" + name + "_neutral")

        config = example._gaze_locomotion.config
        for name, stick_y, stick_x in (("forward_right", 1.0, 1.0), ("back_left", -1.0, -1.0)):
            forward = stick_y * (config.forward_speed if stick_y > 0 else config.backward_speed) / math.sqrt(2)
            lateral = -stick_x * 0.30 / math.sqrt(2)
            clearance(forward, lateral, 1.0)
            origin = checkpoint("diagonal_" + name + "_start")
            fixture.update(left_y=stick_y, left_x=stick_x, expect_zero=False)
            await wait(1.0)
            diagonal = checkpoint("diagonal_" + name)
            command = diagonal["status"]["controller_requested_velocity"]
            _require(diagonal["status"]["hybrid"]["source"] == "controller", "Diagonal stick failed to acquire control")
            _require(
                abs(command[0] - forward) < 1e-6 and abs(command[1] - lateral) < 1e-6 and abs(command[2]) < 1e-6,
                "Full-stick diagonal was not normalized before directional scaling",
            )
            diagonal["measured_forward_m"] = forward_delta(origin, diagonal)
            diagonal["measured_lateral_m"] = lateral_delta(origin, diagonal)
            diagonal["held_duration_s"] = diagonal["sim_s"] - origin["sim_s"]
            if not preview:
                _require(
                    stick_y * diagonal["measured_forward_m"] > 0.01
                    and -stick_x * diagonal["measured_lateral_m"] > 0.01,
                    "Physical diagonal did not progress along both intended axes",
                )
            fixture.update(left_y=0.0, left_x=0.0, expect_zero=True)
            await wait(0.6)
            stopped("Held gaze resumed after diagonal stick release")
            checkpoint("diagonal_" + name + "_neutral")

        if not preview:
            _require(example._navigation_scene.is_route_corner_clear(snapshot()["position_m"]), "No turn clearance")
        fixture.update(right_x=0.575, expect_zero=False)
        turn_start = snapshot()
        await wait(1.0)
        turn = checkpoint("controller_turn")
        _require(turn["status"]["hybrid"]["source"] == "controller", "Right stick did not acquire turn control")
        if not preview:
            yaw = math.atan2(
                math.sin(turn["yaw_rad"] - turn_start["yaw_rad"]), math.cos(turn["yaw_rad"] - turn_start["yaw_rad"])
            )
            _require(
                turn["requested"][2] < 0 and turn["requested"][1] == 0 and yaw < -0.015, "Right-X did not turn right"
            )
        fixture.update(right_x=0.0, expect_zero=True)
        await wait(0.4)
        stopped("Held gaze resumed after turn neutral")
        fixture["target"] = None
        await wait(0.1)
        fixture.update(target="back", expect_zero=False)
        await wait(0.3)
        resumed = checkpoint("fresh_gaze_resumed")
        gaze_active(resumed, "back")
        fixture["right_x"] = 0.575
        await wait(0.1)
        _require(example.gaze_locomotion_status()["hybrid"]["source"] == "controller", "Second takeover failed")
        fixture.update(tracked=False, expect_zero=True)
        await wait(0.02)
        stopped("Controller tracking loss retained a command or resumed held gaze")
        _require(not any(float(x) for x in example._base_command), "Tracking loss failed to clear filtered intent")
        checkpoint("controller_loss")
        fixture.update(tracked=True, right_x=0.0)
        await wait(0.4)
        stopped("Controller recovery resumed held gaze before look-away")
        checkpoint("recovery_blocks_held_gaze")
        fixture.update(target=None, grip=0.0, trigger=0.0)
        await wait(5.0 if not preview else 0.2)
        final = checkpoint("final")
        _require(counts["arms"] > 0 and counts["fingers"] > 0, "Normal hand updates were skipped")
        if preview:
            displacement = max(math.dist(x["position_m"], start["position_m"]) for x in report["samples"])
            rotation = max(
                2
                * math.acos(
                    min(
                        1.0,
                        abs(float(np.dot(x["quaternion_wxyz"], start["quaternion_wxyz"])))
                        / (np.linalg.norm(x["quaternion_wxyz"]) * np.linalg.norm(start["quaternion_wxyz"])),
                    )
                )
                for x in report["samples"]
            )
            _require(displacement <= 1e-4 and rotation <= 1e-4, "Anchored preview moved")
            report["preview_max_position_delta_m"] = displacement
            report["preview_max_rotation_delta_rad"] = rotation
        else:
            _require(math.hypot(*final["linear_velocity_m_s"][:2]) <= 0.10, "Final physical stop speed remains high")
            _require(abs(final["angular_velocity_rad_s"][2]) <= 0.25, "Final physical yaw speed remains high")
            _require(
                max(math.dist(x["position_m"][:2], start["position_m"][:2]) for x in report["samples"]) < 1.0,
                "Hybrid fixture exceeded one meter of travel",
            )
        report.update(passed=True, phase="complete", normal_hand_updaters_preserved=True)
    finally:
        module.time = real_time
        try:
            try:
                report["last_observation"] = snapshot()
            except Exception as error:
                report["last_observation_error"] = repr(error)
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
    """Own one bounded headless run and preserve partial failure evidence."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("preview", "move"), default="preview")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--load-timeout", type=float, default=180.0)
    parser.add_argument("--test-timeout", type=float, default=180.0)
    args, kit_args = parser.parse_known_args()
    for name in ("load_timeout", "test_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name + " must be finite and positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"hybrid-{args.mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    path = output / (run_id + ".json")
    report = dict(
        passed=False,
        phase="starting_application",
        mode=args.mode,
        run_id=run_id,
        checks={},
        samples=[],
        hardware_tracking_tested=False,
        optical_tracking_tested=False,
        payload_tested=False,
        wall_gap_behavior_tested=False,
        controller_layout="left Y forward/back, left X opposite body-lateral, right X opposite body-yaw; normalized planar sticks",
        directional_caps=dict(forward_m_s=0.60, backward_m_s=0.35, lateral_m_s=0.30, yaw_rad_s=0.40),
    )
    _write_report(path, report)
    print(f"Hybrid validation report: {path}", flush=True)
    os.environ.update(
        {
            "G1_GESTURE_WALKING": "1",
            "G1_LOCOMOTION_INPUT": "gaze",
            "G1_POINTING_NAVIGATION": "0",
            "G1_OPTICAL_HANDS_ONLY": "0",
            "G1_WEBCAM_ENABLED": "0",
            "G1_GAZE_CONTROLLER_OVERRIDE": "1",
            "G1_GAZE_PREVIEW": "1" if args.mode == "preview" else "0",
            "G1_GAZE_MODE": "hold",
            "G1_GAZE_DWELL_S": "0",
            "G1_GAZE_FORWARD_SPEED": "0.60",
            "G1_GAZE_BACKWARD_SPEED": "0.35",
            "G1_GAZE_YAW_SPEED": "0.40",
            "G1_GAZE_HEADING_HOLD": "1",
            "G1_GAZE_HEADING_GAIN": "3.0",
            "G1_GAZE_HEADING_MAX_YAW_SPEED": "0.15",
            "G1_GAZE_MAX_GAP_S": "0.25",
            "G1_GAZE_STOP_FEEDBACK": "1",
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
        _require(Path(module.__file__).resolve().is_relative_to(expected), "A different policy extension was loaded")
        report["loaded_module"] = module.__file__
        example = module.HumanoidExample()
        example._behavioral_data_enabled = False
        report["phase"] = "loading_scene"
        task = asyncio.ensure_future(_validate(example, module, report))
        last_progress = 0.0
        while not task.done():
            now = time.perf_counter()
            test_start = report.get("test_started_wall_monotonic_s")
            deadline = test_start + args.test_timeout if test_start is not None else started + args.load_timeout
            if now > deadline:
                task.cancel()
                raise TimeoutError("Wall deadline exceeded during " + report["phase"])
            if now - last_progress >= 5.0:
                report["wall_elapsed_s"] = now - started
                _write_report(path, report)
                print(f"Hybrid validation: {report['phase']} ({now - started:.1f}s wall)", flush=True)
                last_progress = now
            simulation_app.update()
        task.result()
    except BaseException as error:
        report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
    finally:
        if task is not None and not task.done():
            task.cancel()
            if simulation_app is not None:
                for _ in range(3):
                    try:
                        simulation_app.update()
                    except Exception as error:
                        report.update(passed=False, cancellation_error=repr(error))
                        break
                    if task.done():
                        break
        if example is not None:
            try:
                example.physics_cleanup()
            except Exception as error:
                report.update(passed=False, cleanup_error=repr(error))
        report["wall_elapsed_s"] = time.perf_counter() - started
        _write_report(path, report)
        print(f"Hybrid validation {'passed' if report['passed'] else 'FAILED'}: {path}", flush=True)
        if simulation_app is not None:
            try:
                simulation_app.close(exit_code=0 if report["passed"] else 1)
            except Exception as error:
                report.update(passed=False, close_error=repr(error))
                _write_report(path, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
