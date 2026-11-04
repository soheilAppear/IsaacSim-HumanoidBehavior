#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay physical walking and reach acceptance in an already-loaded Isaac Sim.

The historical filename is retained for existing validation workflows.
Start tools/launch_gesture_walking.bat (an alias for controller walking) and load
Humanoid, then run this host CLI with ordinary Python. The production actor,
arm IK, finger control, contacts, and safety guards run in real PhysX. Synthetic
commands exercise walking; the reach suite also injects right-palm targets and
curls. This is not a Quest hardware test, a payload test, or a test of gesture
recognition itself.

--reset explicitly resets the scene before the first suite. --suite all always
resets between baseline and reach so the tests start independently. Input
substitutions are restored and the timeline is paused even when a test fails.
No gaze tracker, body force controller, foot geometry, or root pose is replaced.
A persistent Kit task runs the replay; short host polls retrieve its report.
--output writes JSON on the host. --resume JOB_ID retrieves an existing job after
a host/network interruption without starting or resetting another replay.
--speed sets forward/arc and reach-while-walking requests in meters per simulation
second. Its default preserves the original .25 m/s fixtures. Baseline reports
distinguish requested velocity, measured simulation velocity, and simulation/wall-time ratio.
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

REPORT_PREFIX = "GESTURE_WALKING_LIVE_REPORT "
JOB_PREFIX = "GESTURE_WALKING_LIVE_JOB "

# Everything in this string executes in Kit, not in the host interpreter.
LIVE_CODE = r"""
async def _validate_gesture_walking_live(ex, options):
    import asyncio
    import json
    import math
    import time

    import numpy as np
    import omni.kit.app
    import omni.timeline
    from pxr import Gf
    from isaacsim.robot.policy.examples.interactive.humanoid.locomotion_gesture import GestureCommand

    timeline = omni.timeline.get_timeline_interface()
    app = omni.kit.app.get_app()
    report = {
        "suite": options["suite"], "reset_requested": options["reset"],
        "baseline_requested_speed_m_s": options["speed"] if options["suite"] in ("baseline", "all") else None,
        "reach_requested_speed_m_s": options["speed"] if options["suite"] in ("reach", "all") else None,
        "passed": False, "hardware_tracking_tested": False,
        "payload_tested": False, "suites": {},
    }
    gaze = getattr(ex, "_eye_gaze_tracker", None)

    def own_overrides(names):
        return {name: (name in ex.__dict__, ex.__dict__.get(name)) for name in names}

    def restore_overrides(saved):
        for name, (was_overridden, value) in saved.items():
            if was_overridden:
                setattr(ex, name, value)
            else:
                ex.__dict__.pop(name, None)

    def clear_commands():
        for name in ("_base_command", "_keyboard_command", "_controller_command"):
            value = getattr(ex, name, None)
            if value is not None:
                value[:] = 0

    def check_health(errors_before, check_gaze=True):
        if ex._articulation_health_fault:
            raise RuntimeError(ex._articulation_health_fault)
        new_errors = set(ex._physics_step_error_logged) - errors_before
        if new_errors:
            raise RuntimeError("New physics callback errors: " + repr(sorted(new_errors)))
        if check_gaze:
            assert ex._eye_gaze_tracker is gaze, "Gaze tracker was replaced during replay"

    def vector(value):
        return np.asarray(ex._first_pose_value(value), dtype=float)

    def snapshot(label):
        positions, orientations = ex.g1.robot.get_world_poses()
        velocities, angular = ex.g1.robot.get_velocities()
        p, q, v, w = map(vector, (positions, orientations, velocities, angular))
        return {
            "phase": label, "time": ex.g1._walk_time, "position": p.tolist(),
            "quaternion": q.tolist(), "velocity": v.tolist(), "angular": w.tolist(),
            "command": ex._base_command.tolist(), "up_dot": float(1 - 2 * (q[1]**2 + q[2]**2)),
        }

    async def prepare(reset, errors_before):
        nonlocal gaze
        timeline.pause()
        clear_commands()
        ex._reset_gesture_locomotion()
        if reset:
            await ex.reset_async()
        timeline.play()
        deadline = time.perf_counter() + 60.0
        while not ex._physics_ready or ex.g1._walk_time < .25:
            assert time.perf_counter() < deadline, "Physics did not initialize within 60 seconds"
            await app.next_update_async()
            check_health(errors_before, check_gaze=False)
        assert ex.g1.robot.is_physics_tensor_entity_valid(), "Invalid articulation tensor after initialization"
        # A requested scene reset may rebuild lifecycle objects. Identity must
        # stay fixed during each replay, not across an intentional reset.
        gaze = ex._eye_gaze_tracker
        check_health(errors_before)

    async def baseline(reset):
        speed = options["speed"]
        result = {"fixture": "synthetic commands; production arm/finger callbacks and physical actor",
                  "requested_speed_m_s": speed,
                  "samples": [], "phases": [], "passed": False}
        report["suites"]["baseline"] = result
        saved = own_overrides(("_update_controller_command",))
        errors_before = set(ex._physics_step_error_logged)
        requested = [0., 0., 0.]

        def command(dt):
            ex._keyboard_command[:] = 0
            for index, value in enumerate(requested):
                ex._controller_command[index] = value
            if not any(requested):
                ex._base_command[:] = 0

        async def phase(name, duration, velocity):
            requested[:] = velocity
            start = snapshot(name)
            sample_start = len(result["samples"])
            wall_start = time.perf_counter()
            until = ex.g1._walk_time + duration
            while ex.g1._walk_time < until:
                await asyncio.sleep(.04)
                check_health(errors_before)
                if not result["samples"] or ex.g1._walk_time - result["samples"][-1]["time"] >= .10:
                    result["samples"].append(snapshot(name))
            end = snapshot(name)
            elapsed = end["time"] - start["time"]
            measurements = [start, *result["samples"][sample_start:], end]
            intervals = np.diff([sample["time"] for sample in measurements])
            velocities = np.array([sample["velocity"] for sample in measurements])
            planar_speeds = np.linalg.norm(velocities[:, :2], axis=1)
            headings = np.array([yaw(sample["quaternion"]) for sample in measurements])
            forward_speeds = velocities[:, 0] * np.cos(headings) + velocities[:, 1] * np.sin(headings)
            assert elapsed > 0 and np.all(intervals >= 0), "Non-monotonic simulation timestamps"
            average_velocity = np.sum((velocities[:-1] + velocities[1:]) * .5 * intervals[:, None], axis=0) / elapsed
            average_planar_speed = float(np.sum((planar_speeds[:-1] + planar_speeds[1:]) * .5 * intervals) / elapsed)
            average_forward_speed = float(np.sum((forward_speeds[:-1] + forward_speeds[1:]) * .5 * intervals) / elapsed)
            wall_seconds = time.perf_counter() - wall_start
            entry = {"name": name, "start": start, "end": end,
                     "requested_velocity": list(velocity),
                     "command_reached": bool(np.allclose(end["command"], velocity, rtol=0, atol=1e-4)),
                     "displacement": (np.array(end["position"]) - start["position"]).tolist(),
                     "simulation_seconds": elapsed, "wall_seconds": wall_seconds,
                     "real_time_factor": elapsed / wall_seconds,
                     "average_world_velocity_m_s": average_velocity.tolist(),
                     "average_planar_speed_m_s": average_planar_speed,
                     "average_body_forward_speed_m_s": average_forward_speed}
            result["phases"].append(entry)
            if velocity[0] > 0:
                assert entry["command_reached"], f"{name}: applied command did not reach requested {velocity}"
            return entry

        def yaw(q):
            return math.atan2(2 * (q[0]*q[3] + q[1]*q[2]), 1 - 2 * (q[2]**2 + q[3]**2))

        def yaw_change(entry):
            delta = yaw(entry["end"]["quaternion"]) - yaw(entry["start"]["quaternion"])
            entry["yaw_change"] = math.atan2(math.sin(delta), math.cos(delta))
            return entry["yaw_change"]

        try:
            assert math.isfinite(speed) and 0 < speed <= .65, "Baseline speed must be finite and in (0, .65] m/s"
            runtime_cap = float(ex._max_forward_speed)
            result["runtime_forward_cap_m_s"] = runtime_cap
            assert math.isfinite(runtime_cap) and runtime_cap >= speed, (
                f"Runtime forward cap {runtime_cap} m/s is below requested {speed} m/s; reload the updated example"
            )
            ex._update_controller_command = command
            await prepare(reset, errors_before)
            zero = await phase("zero", 6., [0., 0., 0.])
            assert math.hypot(*zero["displacement"][:2]) <= .20, "zero command drift exceeded .20m in 6 seconds"
            forward = await phase("forward", 4., [speed, 0., 0.])
            assert forward["displacement"][0] >= .20, "forward command did not advance along initial +X"
            stop = await phase("stop", 4., [0., 0., 0.])
            assert math.hypot(*stop["end"]["velocity"][:2]) <= .10, "final stopped speed exceeded .10m/s"
            assert math.hypot(*stop["displacement"][:2]) <= .25, "stop distance exceeded .25m"
            turn = await phase("turn_left", 5., [speed, 0., .20])
            assert yaw_change(turn) >= .15, "left walking turn did not rotate at least .15 rad"
            stop = await phase("stop_after_turn", 4., [0., 0., 0.])
            assert math.hypot(*stop["end"]["velocity"][:2]) <= .10, "final stopped speed after turning exceeded .10m/s"
            turn = await phase("turn_right", 5., [speed, 0., -.20])
            assert yaw_change(turn) <= -.15, "right walking turn did not rotate at least .15 rad"
            stop = await phase("stop_after_right", 4., [0., 0., 0.])
            assert math.hypot(*stop["end"]["velocity"][:2]) <= .10, "stopped speed after right steering exceeded .10m/s"
            result["passed"] = True
        except Exception as error:
            result["error"] = repr(error)
            raise
        finally:
            timeline.pause()
            clear_commands()
            restore_overrides(saved)
            result["fault"] = ex._articulation_health_fault
            result["callback_errors"] = sorted(set(ex._physics_step_error_logged) - errors_before)

    async def reach(reset):
        speed = options["speed"]
        result = {"fixture": "synthetic anatomical right-palm targets and curls; production IK, fingers and physical actor",
                  "requested_speed_m_s": speed, "phases": [], "passed": False}
        report["suites"]["reach"] = result
        names = ("_poll_webcam_tracking", "_get_xr_input_device", "_get_hand_input_kind",
                 "_get_optical_arm_pose", "_compute_arm_target_orientation",
                 "_get_hand_tracking_finger_curls", "_update_controller_command")
        saved = own_overrides(names)
        errors_before = set(ex._physics_step_error_logged)
        webcam = ex._webcam_enabled
        device = object()
        active, curl, velocity = True, 0., 0.
        point = Gf.Vec3d(.15, -.22, .20)

        def optical(side, input_device):
            if side != "right" or not active:
                return None
            base, yaw = ex._get_g1_base_pose_for_arms()
            return Gf.Matrix4d().SetTranslate(ex._g1_body_point_to_stage(point, base, yaw))

        def command(dt):
            ex._gesture_command = GestureCommand(state="active", forward=float(velocity > 0))
            ex._keyboard_command[:] = 0
            ex._controller_command[:] = 0
            ex._controller_command[0] = velocity
            if velocity == 0:
                ex._base_command[:] = 0

        async def phase(name, duration, target, new_curl, speed):
            nonlocal point, curl, velocity
            point, curl, velocity = Gf.Vec3d(*target), new_curl, speed
            until = ex.g1._walk_time + duration
            while ex.g1._walk_time < until:
                await asyncio.sleep(.03)
                check_health(errors_before)
            positions, orientations = ex.g1.robot.get_world_poses()
            velocities, _ = ex.g1.robot.get_velocities()
            base, yaw = ex._get_g1_base_pose_for_arms()
            palm = Gf.Vec3d(*map(float, vector(ex._get_hand_link_prim("right").get_world_poses()[0])))
            palm += ex._palm_offset_world("right", palm)
            expected = ex._g1_body_point_to_stage(point, base, yaw)
            entry = {"name": name, "requested_speed_m_s": speed, "root": vector(positions).tolist(),
                     "quaternion": vector(orientations).tolist(), "velocity": vector(velocities).tolist(),
                     "palm_error_m": (palm - expected).GetLength(), "curls": ex.g1.get_finger_curls("right"),
                     "sources": dict(ex._arm_input_sources), "contact_health": ex._contact_health}
            result["phases"].append(entry)
            return entry

        try:
            assert math.isfinite(speed) and 0 < speed <= .65, "Reach walking speed must be finite and in (0, .65] m/s"
            runtime_cap = float(ex._max_forward_speed)
            result["runtime_forward_cap_m_s"] = runtime_cap
            assert math.isfinite(runtime_cap) and runtime_cap >= speed, (
                f"Runtime forward cap {runtime_cap} m/s is below requested {speed} m/s; reload the updated example"
            )
            # This enables the common input path in a headless/no-headset session;
            # all pose/curl samples come from the explicit synthetic optical fixture.
            ex._webcam_enabled = True
            ex._poll_webcam_tracking = lambda: None
            ex._get_xr_input_device = lambda handle: device if active and handle.endswith("right") else None
            ex._get_hand_input_kind = lambda value: "hand_tracking" if value is device else "none"
            ex._get_optical_arm_pose = optical
            ex._compute_arm_target_orientation = lambda *args, **kwargs: None
            ex._get_hand_tracking_finger_curls = lambda value: {role: curl for role in ex._finger_roles} if value is device else None
            ex._update_controller_command = command
            await prepare(reset, errors_before)
            ex._gesture_finger_hold = ex.g1.get_finger_curls("left")
            await phase("raise", 3., (.16, -.22, .35), 0., 0.)
            extended = await phase("reach_and_walk", 4., (.43, -.18, .30), 0., speed)
            assert extended["palm_error_m"] < .10, extended
            closed = await phase("close_during_walk", 2., (.43, -.18, .30), .8, speed)
            assert closed["curls"]["index"] > .5, closed
            stopped = await phase("stop_with_arm_extended", 4., (.43, -.18, .30), .8, 0.)
            assert np.linalg.norm(stopped["velocity"][:2]) < .1, stopped
            opened = await phase("retract_and_open", 3., (.15, -.22, .20), 0., 0.)
            assert opened["curls"]["index"] < .2, opened
            active = False
            lost = await phase("tracking_loss", 3., (.15, -.22, .20), 0., 0.)
            assert not lost["sources"], lost
            assert np.linalg.norm(lost["velocity"][:2]) < .1, lost
            check_health(errors_before)
            result["passed"] = True
        except Exception as error:
            result["error"] = repr(error)
            raise
        finally:
            timeline.pause()
            clear_commands()
            restore_overrides(saved)
            ex._webcam_enabled = webcam
            ex._reset_gesture_locomotion()
            result["fault"] = ex._articulation_health_fault
            result["callback_errors"] = sorted(set(ex._physics_step_error_logged) - errors_before)

    try:
        assert ex is not None and ex.g1 is not None, "Load the Humanoid example first"
        assert ex._gesture_locomotion_enabled and ex.g1._physical_locomotion, "Load gesture walking mode first"
        if options["suite"] in ("baseline", "all"):
            await baseline(options["reset"])
        if options["suite"] in ("reach", "all"):
            await reach(options["reset"] or options["suite"] == "all")
        report["gaze_tracker_preserved_within_replays"] = ex._eye_gaze_tracker is gaze
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        timeline.pause()
        if ex is not None:
            clear_commands()
        _GESTURE_WALKING_LIVE_JOBS[options["job_id"]]["report"] = report
        print("GESTURE_WALKING_LIVE_REPORT " + json.dumps(report, sort_keys=True))

async def _gesture_walking_job_runner(ex, options, job):
    try:
        # PythonServer's top-level-await driver is not an asyncio Task. This
        # wrapper is scheduled below, so wait_for owns a real cancellation scope.
        await asyncio.wait_for(_validate_gesture_walking_live(ex, options), timeout=options["timeout"])
    except BaseException as error:
        job["error"] = repr(error)
        job["traceback"] = traceback.format_exc()
    finally:
        job["done"] = True

_job = _GESTURE_WALKING_LIVE_JOBS[OPTIONS["job_id"]]
_job["task"] = asyncio.ensure_future(_gesture_walking_job_runner(EX, OPTIONS, _job))
print("GESTURE_WALKING_LIVE_JOB " + json.dumps({"job_id": OPTIONS["job_id"], "done": False}))
"""


START_GUARD = r"""
import asyncio
import json
import traceback

_GESTURE_WALKING_LIVE_JOBS = globals().setdefault("_GESTURE_WALKING_LIVE_JOBS", {})
for _active_id, _active_job in _GESTURE_WALKING_LIVE_JOBS.items():
    _active_task = _active_job.get("task")
    if _active_task is not None and not _active_task.done():
        raise RuntimeError("Walking validation already running: " + _active_id)
if OPTIONS["job_id"] in _GESTURE_WALKING_LIVE_JOBS:
    raise RuntimeError("Walking validation job ID already exists")
# Keep completed reports for recovery without growing the interactive session forever.
for _old_id in list(_GESTURE_WALKING_LIVE_JOBS)[:-7]:
    del _GESTURE_WALKING_LIVE_JOBS[_old_id]
_GESTURE_WALKING_LIVE_JOBS[OPTIONS["job_id"]] = {
    "job_id": OPTIONS["job_id"], "done": False, "report": None, "error": None,
}
"""

POLL_CODE = r"""
import json

_jobs = globals().get("_GESTURE_WALKING_LIVE_JOBS", {})
if JOB_ID not in _jobs:
    raise RuntimeError("Walking validation job not found: " + JOB_ID)
_job = _jobs[JOB_ID]
_task = _job.get("task")
if _task is not None and _task.done():
    # Retrieve any unexpected task exception as well as errors caught by the runner.
    if _task.cancelled():
        _job["error"] = _job.get("error") or "Validation task was cancelled"
    else:
        _error = _task.exception()
        if _error is not None:
            _job["error"] = _job.get("error") or repr(_error)
    _job["done"] = True
_reply = {key: value for key, value in _job.items() if key != "task"}
print("GESTURE_WALKING_LIVE_JOB " + json.dumps(_reply, sort_keys=True))
"""


def request_job(code: str, host: str, port: int, timeout: float) -> tuple[int, dict | None]:
    """Use the shared Kit transport for one short, structured job request."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        status = execute(code, host, port, timeout)
    output = captured.getvalue()
    messages = [line[len(JOB_PREFIX) :] for line in output.splitlines() if line.startswith(JOB_PREFIX)]
    if status or not messages:
        if output:
            print(output, end="")
        return status or 1, None
    return 0, json.loads(messages[-1])


def main() -> int:
    """Start or resume a bounded Kit job, polling without a long-lived socket."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--suite", choices=("baseline", "reach", "all"), default="all")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.25,
        help="Forward/arc and reach-walking speed in m/s, greater than 0 and at most 0.65",
    )
    parser.add_argument("--reset", action="store_true", help="Reset the loaded scene before the first suite")
    parser.add_argument("--timeout", type=float, default=480.0, help="Maximum replay/wait time in seconds")
    parser.add_argument("--output", type=Path, help="Write the validation JSON report on this host")
    parser.add_argument("--resume", metavar="JOB_ID", help="Poll an existing job without starting or resetting it")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if not math.isfinite(args.speed) or not 0 < args.speed <= 0.65:
        parser.error("--speed must be finite, greater than 0, and at most 0.65 m/s")
    if args.resume and args.reset:
        parser.error("--resume cannot be combined with --reset")
    try:
        job_id = str(uuid.UUID(args.resume)) if args.resume else str(uuid.uuid4())
    except ValueError:
        parser.error("--resume requires the UUID printed when the job started")
    deadline = time.monotonic() + args.timeout + 15.0
    if not args.resume:
        options = {
            "suite": args.suite,
            "speed": args.speed,
            "reset": args.reset,
            "timeout": args.timeout,
            "job_id": job_id,
        }
        code = "OPTIONS = " + repr(options) + "\n" + START_GUARD + LIVE_CODE
        status, reply = request_job(code, args.host, args.port, 10.0)
        if status:
            print(f"Start response failed. If Kit accepted the job, recover it with --resume {job_id}.")
            return status
        if reply.get("job_id") != job_id:
            raise RuntimeError("Kit returned a different validation job ID")
    print(f"Walking validation job {job_id}; polling {args.host}:{args.port}", flush=True)
    poll = "JOB_ID = " + repr(job_id) + "\n" + POLL_CODE
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        status, reply = request_job(poll, args.host, args.port, min(10.0, max(0.1, remaining)))
        if status == 1:
            print(f"Kit rejected the poll for job {job_id}; verify its UUID and server port.")
            return 1
        if not status:
            if reply.get("job_id") != job_id:
                raise RuntimeError("Kit returned a different validation job ID")
            if reply.get("done"):
                report = reply.get("report") or {"passed": False, "error": "Job ended before a report was created"}
                report["job_id"] = job_id
                if reply.get("error"):
                    report["job_error"] = reply["error"]
                if reply.get("traceback"):
                    report["job_traceback"] = reply["traceback"]
                print(REPORT_PREFIX + json.dumps(report, sort_keys=True))
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                return 0 if report.get("passed") and not reply.get("error") else 1
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    print(f"Report wait expired. Kit owns the bounded replay and cleanup; recover with --resume {job_id}.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
