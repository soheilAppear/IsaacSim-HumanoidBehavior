#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check three Stop/Play and two Pause/Play cycles in a loaded walking example.

Run with ordinary host Python while the humanoid is healthy, playing, and idle.
Both pointing-only and controller-stick walking profiles are supported. Leave
navigation released and all mapped controller axes centered (or unavailable).
The test uses the real timeline and callbacks, requires navigation to disarm,
and never toggles XR or replaces gaze, tracking, policy inputs, or safety guards.
Healthy completion leaves Play running; failures pause the simulation. Scene
Reset is never called. A temporary initialize-call counter is restored afterward.

Example:
    python tools/validate_point_navigation_lifecycle_live.py --output .webcam/lifecycle.json

Short TCP polls retrieve a bounded asynchronous Kit task. The host report is
updated after each poll. A sibling ``*.journal.json`` is also flushed by Kit
before each timeline action, preserving the last phase if Kit crashes. This
journal assumes host and Kit run on the same PC. ``--resume JOB_ID`` only polls.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import time
import uuid
from pathlib import Path

from kit_exec import DEFAULT_HOST, DEFAULT_PORT, execute

JOB_PREFIX = "POINT_NAVIGATION_LIFECYCLE_JOB "

LIVE_CODE = r"""
import asyncio
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.timeline

_POINT_NAVIGATION_LIFECYCLE_JOBS = globals().setdefault("_POINT_NAVIGATION_LIFECYCLE_JOBS", {})
for _registry_name, _registry in list(globals().items()):
    if _registry_name.endswith("_JOBS") and isinstance(_registry, dict):
        for _other_id, _other in _registry.items():
            _task = _other.get("task") if isinstance(_other, dict) else None
            if _task is not None and not _task.done():
                raise RuntimeError("Another live validation is running: " + str(_other_id))
if OPTIONS["job_id"] in _POINT_NAVIGATION_LIFECYCLE_JOBS:
    raise RuntimeError("Lifecycle job already exists")
for _old_id in list(_POINT_NAVIGATION_LIFECYCLE_JOBS)[:-7]:
    del _POINT_NAVIGATION_LIFECYCLE_JOBS[_old_id]
_job = {
    "job_id": OPTIONS["job_id"], "done": False,
    "report": {"passed": False, "phase": "starting", "phases": [], "initialize_calls": 0,
               "hardware_tracking_tested": False, "xr_toggled_by_test": False},
}
_POINT_NAVIGATION_LIFECYCLE_JOBS[OPTIONS["job_id"]] = _job

def _write_point_lifecycle_journal(job, options):
    report = dict(job["report"], job_id=job["job_id"], done=job["done"])
    path = Path(options["journal"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write before the dangerous native operation, not only after its return.
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

async def _run_point_navigation_lifecycle(ex, job, options):
    report = job["report"]
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    started = time.perf_counter()
    gaze = getattr(ex, "_eye_gaze_tracker", None)
    errors_before = set(getattr(ex, "_physics_step_error_logged", ()))
    g1 = getattr(ex, "g1", None)
    original_initialize = getattr(g1, "initialize", None)
    had_override = g1 is not None and "initialize" in g1.__dict__
    own_initialize = g1.__dict__.get("initialize") if g1 is not None else None
    wrapped = False
    mode = None

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def check_health():
        require(ex.g1 is g1, "Robot instance changed during lifecycle validation")
        require(ex._eye_gaze_tracker is gaze, "Gaze tracker changed during lifecycle validation")
        require(not ex._articulation_health_fault, str(ex._articulation_health_fault))
        require(not g1.locomotion_fault, str(g1.locomotion_fault))
        errors = set(ex._physics_step_error_logged) - errors_before
        require(not errors, "New callback errors: " + repr(sorted(errors)))

    def commands_are_zero():
        for name in ("_base_command", "_keyboard_command", "_controller_command"):
            value = getattr(ex, name, None)
            require(value is not None, name + " is unavailable")
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            array = np.asarray(value, dtype=float)
            require(array.shape == (3,) and np.isfinite(array).all(), name + " is invalid")
            require(np.max(np.abs(array)) <= 1e-7, name + " requested movement during lifecycle test")

    def check_navigation(*, after_transition=False):
        # The unused pointing controller must not retain a destination when
        # joystick walking owns input. Real XR readers stay installed throughout.
        require(ex._gesture_command.target is None, "Destination survived a lifecycle transition")
        require(ex._gesture_command.candidate is None, "Pointing candidate survived a lifecycle transition")
        if mode == "controller_sticks":
            helper = ex._controller_locomotion
            command = helper.command
            require(not ex._pointing_navigation_enabled, "Pointing navigation is active in controller mode")
            require(all(abs(float(value)) <= 1e-7 for value in command.velocity),
                    "Controller input helper retained a movement request")
            if after_transition:
                require(command.state == "awaiting_neutral" and not helper._armed,
                        "Stop/Pause did not disarm controller input and require neutral")
            else:
                require(command.state in ("idle", "awaiting_neutral", "unavailable"),
                        "Controller navigation unexpectedly became active")
            samples = {}
            for side in ("left", "right"):
                valid, x, y = ex._read_tracked_controller_stick(ex._get_xr_input_device("/user/hand/" + side))
                axes = (x, y) if side == "left" else (x,)
                if valid:
                    require(all(np.isfinite(value) and abs(value) <= helper.deadzone for value in axes),
                            "Center the real " + side + " controller before lifecycle testing")
                samples[side] = {"valid": bool(valid), "mapped_axes": [float(value) for value in axes]}
            return {"state": command.state, "armed": bool(helper._armed), "real_controller_samples": samples}
        allowed = ("awaiting_release",) if after_transition else ("idle", "awaiting_release")
        require(ex._gesture_command.state in allowed, "Navigation did not remain released/disarmed")
        return None

    def snapshot(label, *, read_physics=True):
        check_health()
        commands_are_zero()
        controller = check_navigation()
        item = {
            "phase": label, "wall_seconds": time.perf_counter() - started,
            "playing": bool(timeline.is_playing()), "stopped": bool(ex._timeline_stopped),
            "physics_ready": bool(ex._physics_ready), "initialized": bool(g1._initialized),
            "tensor_valid": bool(g1.robot.is_physics_tensor_entity_valid()),
            "walk_time": float(g1._walk_time), "initialize_calls": report["initialize_calls"],
            "navigation": ex.gesture_locomotion_status(), "gaze_identity_retained": ex._eye_gaze_tracker is gaze,
            "finger_sources": dict(ex._finger_curl_source),
            "xr_enabled": bool(ex._xr_core.is_xr_enabled()) if ex._xr_core is not None else False,
            "xr_display_enabled": bool(ex._xr_core.is_xr_display_enabled()) if ex._xr_core is not None else False,
            "camera_can_schedule": bool(ex._xr_camera_can_schedule()),
        }
        if controller is not None:
            item["controller_interlock"] = controller
        if read_physics and item["tensor_valid"]:
            position, quaternion = g1.robot.get_world_poses()
            position = np.asarray(ex._first_pose_value(position), dtype=float)
            quaternion = np.asarray(ex._first_pose_value(quaternion), dtype=float)
            require(position.shape == (3,) and quaternion.shape == (4,), "Unexpected root pose dimensions")
            require(np.isfinite(position).all() and np.isfinite(quaternion).all(), "Nonfinite root pose")
            require(0.8 < np.linalg.norm(quaternion) < 1.2, "Invalid root quaternion")
            item.update(position=position.tolist(), quaternion=quaternion.tolist())
        return item

    def journal(label, *, read_physics=True):
        report["phase"] = label
        report["phases"].append(snapshot(label, read_physics=read_physics))
        _write_point_lifecycle_journal(job, options)

    async def frames(count):
        for _ in range(count):
            await app.next_update_async()
            check_health()

    async def wait_for_running(target_time):
        play_deadline = time.perf_counter() + 5.0
        observed_playing = False
        while True:
            # Timeline.play() queues a transition; Kit commits it on an app
            # update. Requiring is_playing synchronously tests the queue rather
            # than the lifecycle and falsely fails a healthy first restart.
            await app.next_update_async()
            check_health()
            commands_are_zero()
            check_navigation()
            if not timeline.is_playing():
                require(not observed_playing and time.perf_counter() < play_deadline,
                        "Timeline failed to enter Play or stopped unexpectedly")
                continue
            observed_playing = True
            if (ex._physics_ready and g1._initialized
                    and g1.robot.is_physics_tensor_entity_valid() and g1._walk_time >= target_time):
                break
        check_navigation()

    def count_initialize(*args, **kwargs):
        report["initialize_calls"] += 1
        return original_initialize(*args, **kwargs)

    try:
        require(ex is not None and g1 is not None, "Load the humanoid example first")
        require(ex._gesture_locomotion_enabled, "This validation requires physical walking mode")
        require(timeline.is_playing() and ex._physics_ready and g1._initialized, "Start a healthy playing scene first")
        require(gaze is not None, "Gaze tracker must already exist to verify identity preservation")
        mode = ex.gesture_locomotion_status().get("mode")
        require(mode in ("point_only", "controller_sticks"), "Unsupported walking input profile")
        report["input_mode"] = mode
        check_navigation()
        check_health()
        commands_are_zero()
        g1.initialize = count_initialize
        wrapped = True
        journal("initial")
        for cycle in range(1, 4):
            prefix = "stop_play_" + str(cycle)
            count = report["initialize_calls"]
            journal(prefix + ":before_stop")
            timeline.stop()
            await frames(4)
            require(ex._timeline_stopped and not ex._physics_ready and not g1._initialized, "Stop retained ready state")
            check_navigation(after_transition=True)
            journal(prefix + ":before_play", read_physics=False)
            timeline.play()
            await wait_for_running(0.25)
            require(report["initialize_calls"] == count + 1, "Stop/Play did not initialize exactly once")
            journal(prefix + ":complete")
        for cycle in range(1, 3):
            prefix = "pause_play_" + str(cycle)
            count = report["initialize_calls"]
            journal(prefix + ":before_pause")
            timeline.pause()
            await frames(4)
            require(not timeline.is_playing(), "Pause did not pause the timeline")
            require(ex._physics_ready and g1._initialized, "Pause discarded the physics episode")
            check_navigation(after_transition=True)
            paused_time = float(g1._walk_time)
            journal(prefix + ":before_play")
            timeline.play()
            await wait_for_running(paused_time + 0.25)
            require(report["initialize_calls"] == count, "Pause/Play unexpectedly reinitialized the robot")
            journal(prefix + ":complete")
        report["passed"] = True
        journal("complete")
    except BaseException as error:
        report["passed"] = False
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if not report["passed"]:
            timeline.pause()
            pause_deadline = time.perf_counter() + 5.0
            while timeline.is_playing() and time.perf_counter() < pause_deadline:
                await app.next_update_async()
        if wrapped:
            if had_override:
                g1.initialize = own_initialize
            else:
                g1.__dict__.pop("initialize", None)
        report["wall_seconds"] = time.perf_counter() - started
        report["final_playing"] = bool(timeline.is_playing())
        report["fault"] = getattr(ex, "_articulation_health_fault", None)
        report["new_callback_errors"] = sorted(set(getattr(ex, "_physics_step_error_logged", ())) - errors_before)
        transitions = [item for item in report["phases"] if item["phase"].endswith((":before_stop", ":before_pause"))]
        report["xr_display_active_for_all_transitions"] = len(transitions) == 5 and all(
            item["xr_enabled"] and item["xr_display_enabled"] for item in transitions
        )
        _write_point_lifecycle_journal(job, options)

async def _point_navigation_lifecycle_runner(ex, job, options):
    try:
        await asyncio.wait_for(_run_point_navigation_lifecycle(ex, job, options), timeout=options["timeout"])
    except BaseException as error:
        job["report"]["passed"] = False
        job["report"]["job_error"] = repr(error)
    finally:
        job["done"] = True
        _write_point_lifecycle_journal(job, options)

_job["task"] = asyncio.ensure_future(_point_navigation_lifecycle_runner(EX, _job, OPTIONS))
print("POINT_NAVIGATION_LIFECYCLE_JOB " + json.dumps({"job_id": OPTIONS["job_id"], "done": False, "report": _job["report"]}))
"""

POLL_CODE = r"""
import json
_job = _POINT_NAVIGATION_LIFECYCLE_JOBS[JOB_ID]
_task = _job["task"]
if _task.done() and not _job["done"]:
    _job["done"] = True
    _job["report"]["passed"] = False
    _job["report"]["job_error"] = "Cancelled" if _task.cancelled() else repr(_task.exception())
print("POINT_NAVIGATION_LIFECYCLE_JOB " + json.dumps({key: value for key, value in _job.items() if key != "task"}, allow_nan=False))
"""


def request_job(code: str, host: str, port: int, timeout: float) -> tuple[int, dict | None]:
    """Send a short Kit request and recover its structured job snapshot."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = execute(code, host, port, timeout)
    lines = [line[len(JOB_PREFIX) :] for line in output.getvalue().splitlines() if line.startswith(JOB_PREFIX)]
    if status or not lines:
        print(output.getvalue(), end="")
        return status or 1, None
    return 0, json.loads(lines[-1])


def main() -> int:
    """Start or recover a journaled lifecycle test without opening another Kit."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=180.0, help="Kit task deadline, at most 180 seconds")
    parser.add_argument("--output", type=Path, default=Path(".webcam/point-navigation-lifecycle.json"))
    parser.add_argument("--resume", metavar="JOB_ID", help="Poll an existing job without touching the timeline")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 180:
        parser.error("--timeout must be finite and between 0 and 180 seconds")
    try:
        job_id = str(uuid.UUID(args.resume)) if args.resume else str(uuid.uuid4())
    except ValueError:
        parser.error("--resume requires the UUID from the start message")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = output.with_suffix(".journal.json")
    deadline = time.monotonic() + args.timeout + 10
    print(f"Lifecycle job {job_id}; journal {journal}", flush=True)
    if not args.resume:
        options = {"job_id": job_id, "timeout": args.timeout, "journal": str(journal)}
        status, reply = request_job("OPTIONS = " + repr(options) + "\n" + LIVE_CODE, args.host, args.port, 10)
        if status:
            print(f"Start response failed; recover an accepted job with --resume {job_id}.")
            return status
    last_phase = None
    poll = "JOB_ID = " + repr(job_id) + "\n" + POLL_CODE
    while time.monotonic() < deadline:
        status, reply = request_job(poll, args.host, args.port, min(10, max(0.1, deadline - time.monotonic())))
        if status:
            print(f"Kit connection/poll failed. Inspect {journal}; resume with --resume {job_id}.")
            return status
        if reply.get("job_id") != job_id:
            raise RuntimeError("Kit returned an unexpected job ID")
        report = dict(reply["report"], job_id=job_id, done=reply["done"])
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if report.get("phase") != last_phase:
            last_phase = report.get("phase")
            print(f"Phase: {last_phase}", flush=True)
        if reply["done"]:
            print(json.dumps(report, sort_keys=True, allow_nan=False))
            return 0 if report.get("passed") else 1
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    print(f"Host wait expired. Kit owns timeout cleanup; inspect {journal} or --resume {job_id}.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
