#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay controller-stick walking through the real GR00T actor in loaded Isaac.

Start launch_controller_walking.bat and initialize G1 on clear, level floor.
This script never changes the robot root, joints, gravity, scene or safety guards.
Only the stick reader is synthetic; side proxies delegate all other hand-device
methods to their original devices. Gaze and the existing hand-input pipeline
remain unchanged; the replay never fabricates optical fingers. A runtime that
switches between controllers and optical hands still has that hardware limitation.
Hardware tracking, payload handling and Stop/Play lifecycle are separate tests.

Each direction runs for 2.5 simulation seconds, then settles for four seconds.
The six cases cover forward/backward, both strafes and both turns. Right turning
also exercises controller loss and neutral rearming. Each phase has a 48-second
wall limit; the overall Kit task is bounded to 180 seconds by default. Current
body/floor geometry must pass preflight; move the scene's spawn before loading
if it does not. The validator will never teleport G1 to make a fixture pass.

Reports and a sibling .progress.json remain in the repository. Partial failures
are preserved. --resume retrieves an existing job after a host disconnect. Every
exit restores method ownership, clears movement, and leaves physics paused.
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

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "CONTROLLER_WALKING_JOB "
PHASES = ("forward", "backward", "strafe_left", "strafe_right", "yaw_left", "yaw_right")

LIVE_CODE = r"""
import asyncio
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import UsdGeom

_CONTROLLER_WALKING_JOBS = globals().setdefault("_CONTROLLER_WALKING_JOBS", {})
for _name, _registry in list(globals().items()):
    if _name.endswith("_JOBS") and isinstance(_registry, dict):
        for _other_id, _other in _registry.items():
            _task = _other.get("task") if isinstance(_other, dict) else None
            if _task is not None and not _task.done():
                raise RuntimeError("Another live validator is running: " + str(_other_id))
if OPTIONS["job_id"] in _CONTROLLER_WALKING_JOBS:
    raise RuntimeError("Controller walking job already exists")
for _old_id in list(_CONTROLLER_WALKING_JOBS)[:-7]:
    del _CONTROLLER_WALKING_JOBS[_old_id]
_job = {"job_id":OPTIONS["job_id"], "done":False, "report":{
    "passed":False, "phase":"prepare", "phases":[], "samples":[],
    "hardware_tracking_tested":False, "payload_tested":False, "lifecycle_tested":False,
    "fixture":"synthetic thumbsticks; production input policy, actor, unmodified hand readers and safety guards",
}}
_CONTROLLER_WALKING_JOBS[OPTIONS["job_id"]] = _job

def _write_controller_walking(job, options):
    path = Path(options["progress_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(job["report"], job_id=job["job_id"], done=job["done"]),
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")

async def _controller_walking_replay(ex, job, options):
    report = job["report"]
    timeline = omni.timeline.get_timeline_interface()
    app = omni.kit.app.get_app()
    started = time.perf_counter()
    gaze = getattr(ex, "_eye_gaze_tracker", None)
    old_errors = set(getattr(ex, "_physics_step_error_logged", ()))
    saved = {}
    modified = False
    last_flush = -math.inf
    current = {"left":(True,0.,0.), "right":(True,0.,0.)}

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def flush(force=False):
        nonlocal last_flush
        now = time.perf_counter()
        if force or now-last_flush >= 1:
            _write_controller_walking(job, options)
            last_flush = now

    def vector(value):
        result = np.asarray(ex._first_pose_value(value), dtype=float).reshape(-1)
        require(np.isfinite(result).all(), "Nonfinite physics observation")
        return result

    def yaw(quaternion):
        w,x,y,z = quaternion
        return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))

    def wrap(angle):
        return (angle+math.pi) % (2*math.pi)-math.pi

    def check_health():
        require(not ex._articulation_health_fault, str(ex._articulation_health_fault))
        require(not ex.g1.locomotion_fault, str(ex.g1.locomotion_fault))
        require(ex._eye_gaze_tracker is gaze, "Gaze tracker changed")
        require(not (set(ex._physics_step_error_logged)-old_errors), "New physics callback errors")
        require(ex.g1.robot.is_physics_tensor_entity_valid(), "Articulation tensor became invalid")

    def snapshot(label):
        p,q = ex.g1.robot.get_world_poses()
        v,w = ex.g1.robot.get_velocities()
        p,q,v,w = map(vector,(p,q,v,w))
        body_positions = vector(ex.g1.robot.get_dof_positions())[:21]
        body_velocities = vector(ex.g1.robot.get_dof_velocities())[:21]
        require(len(body_positions)==21 and len(body_velocities)==21, "Missing first 21 body DOFs")
        command = ex._controller_locomotion.command
        return {"phase":label, "sim_time":float(ex.g1._walk_time), "wall_time":time.perf_counter(),
                "position":p.tolist(), "yaw":yaw(q), "velocity":v.tolist(), "angular":w.tolist(),
                "up_dot":float(1-2*(q[1]**2+q[2]**2)),
                "state":command.state, "normalized":list(command.velocity),
                "request":ex._controller_command.tolist(), "filtered":ex._base_command.tolist(),
                "body_position_21":body_positions.tolist(), "body_velocity_21":body_velocities.tolist()}

    async def advance(label):
        await app.next_update_async()
        check_health()
        if not report["samples"] or ex.g1._walk_time-report["samples"][-1]["sim_time"] >= .25:
            report["samples"].append(snapshot(label))
        flush()

    async def run_interval(label, duration, *, assert_stopped=False, wall_limit=48.):
        report["phase"] = label
        end_sim = ex.g1._walk_time+duration
        deadline = time.perf_counter()+wall_limit
        while ex.g1._walk_time < end_sim:
            require(time.perf_counter() < deadline, label+": simulation did not advance before deadline")
            await advance(label)
            if assert_stopped:
                require(not any(ex._controller_locomotion.command.velocity), label+": normalized command is nonzero")
                require(max(abs(float(v)) for v in ex._controller_command.tolist()) < 1e-8,
                        label+": movement request is nonzero")
                require(max(abs(float(v)) for v in ex._base_command.tolist()) < 1e-8,
                        label+": filtered movement is nonzero")
        return snapshot(label)

    async def observe_first_physics_step(label):
        before = ex.g1._walk_time
        deadline = time.perf_counter()+5.
        while ex.g1._walk_time <= before:
            require(time.perf_counter() < deadline, label+": no input-processing physics step")
            await advance(label)
        return snapshot(label)

    def zero_requests(sample, label):
        require(max(abs(v) for v in (*sample["normalized"], *sample["request"], *sample["filtered"])) < 1e-8,
                label+": release/loss did not immediately clear movement requests")

    def neutral():
        current.update(left=(True,0.,0.), right=(True,0.,0.))

    def check_blue_hidden():
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(ex._navigation_scene.ROOT)
        if prim.IsValid():
            require(UsdGeom.Imageable(prim).ComputeVisibility()==UsdGeom.Tokens.invisible,
                    "Disabled blue navigation is still visible")

    def preflight(label, start):
        position, heading = start["position"], start["yaw"]
        if label.startswith("yaw"):
            require(ex._navigation_scene.is_route_corner_clear(position),
                    label+": fixture lacks full-turn and stepping-drift clearance")
            return
        direction = {"forward":(1,0), "backward":(-1,0), "strafe_left":(0,1), "strafe_right":(0,-1)}[label]
        dx = math.cos(heading)*direction[0]-math.sin(heading)*direction[1]
        dy = math.sin(heading)*direction[0]+math.cos(heading)*direction[1]
        # Bound this short fixture with a full 2.5m body corridor. It is a test
        # precondition, never an obstacle waiver or a root-position correction.
        target = (position[0]+2.5*dx, position[1]+2.5*dy, 0.)
        require(ex._navigation_scene.floor_at(target) is not None, label+": fixture has no verified floor")
        require(ex._navigation_scene.is_path_clear(position,target),
                label+": fixture corridor blocked: "+str(ex._navigation_scene.last_error))

    try:
        require(ex is not None and ex.g1 is not None and ex._physics_ready and ex.g1._initialized,
                "Load and initialize the controller walking scene first")
        require(ex._controller_walking_enabled and ex._gesture_locomotion_enabled, "Controller GR00T walking mode is required")
        require(ex._g1_locomotion != "stationary", "Robot must use the real physical walking actor")
        require(ex._optical_hands_only and not ex._pointing_navigation_enabled,
                "Optical-only hands and disabled pointing navigation are required")
        check_health()
        timeline.pause()
        modified = True
        for _ in range(3):
            await app.next_update_async()
        original_device_reader = ex._get_xr_input_device
        for name in ("_get_xr_input_device", "_read_tracked_controller_stick"):
            saved[name] = (name in vars(ex), vars(ex).get(name))

        class SideDevice:
            def __init__(self, side):
                self.side = side
                self.delegate = None
            def __getattr__(self, name):
                return getattr(self.delegate,name)

        proxies = {side:SideDevice(side) for side in ("left","right")}
        def device_reader(handle):
            side = {"/user/hand/left":"left", "/user/hand/right":"right"}.get(handle)
            original = original_device_reader(handle)
            if side is None:
                return original
            proxies[side].delegate = original
            return proxies[side]
        def stick_reader(device):
            return current[device.side] if isinstance(device,SideDevice) else (False,0.,0.)
        ex._get_xr_input_device = device_reader
        ex._read_tracked_controller_stick = stick_reader
        ex._reset_gesture_locomotion()
        current["left"] = (True,0.,1.)
        timeline.play()
        await run_interval("startup_held_stick",.5,assert_stopped=True)
        require(ex._controller_locomotion.command.state=="awaiting_neutral", "Held startup stick bypassed neutral interlock")
        neutral()
        await run_interval("startup_centered",.25,assert_stopped=True)
        require(ex._controller_locomotion.command.state=="idle", "Centered sticks did not arm")
        report["neutral_startup_passed"] = True
        check_blue_hidden()

        cases = (("forward", (True,0.,1.), (True,0.,0.), 0,1,.08),
                 ("backward", (True,0.,-1.), (True,0.,0.), 0,-1,.03),
                 ("strafe_left", (True,-1.,0.), (True,0.,0.), 1,1,.03),
                 ("strafe_right", (True,1.,0.), (True,0.,0.), 1,-1,.03),
                 ("yaw_left", (True,0.,0.), (True,-1.,0.), 2,1,.15),
                 ("yaw_right", (True,0.,0.), (True,1.,0.), 2,-1,.15))
        for label,left,right,axis,sign,minimum in cases:
            if options["phase"] != "all" and options["phase"] != label:
                continue
            report["phase"] = label+"_preflight"
            start = snapshot(label)
            entry = {"name":label, "start":start, "passed":False}
            report["phases"].append(entry)
            flush(True)
            preflight(label,start)
            current.update(left=left,right=right)
            end = await run_interval(label,2.5)
            entry["end"] = end
            dx,dy = end["position"][0]-start["position"][0],end["position"][1]-start["position"][1]
            heading = start["yaw"]
            body_displacement = (dx*math.cos(heading)+dy*math.sin(heading),
                                 -dx*math.sin(heading)+dy*math.cos(heading))
            progress = sign*(wrap(end["yaw"]-heading) if axis==2 else body_displacement[axis])
            entry.update(body_displacement_m=body_displacement, yaw_change_rad=wrap(end["yaw"]-heading),
                         signed_progress=progress, minimum_progress=minimum,
                         planar_displacement_m=math.hypot(dx,dy))
            flush(True)
            require(end["state"]=="active", label+": input did not remain active")
            require(sign*end["request"][axis] > .01, label+": requested axis has wrong sign")
            require(progress >= minimum, label+": physical movement failed signed progress: "+str(progress))
            require(math.hypot(dx,dy) < 2.5, label+": exceeded bounded fixture travel")
            if label=="yaw_right":
                current["right"] = (False,1.,0.)
                stopped = await observe_first_physics_step("turn_controller_loss")
                zero_requests(stopped,"turn_controller_loss")
                require(stopped["state"]=="unavailable", "Lost controller did not invalidate input")
                current["right"] = (True,1.,0.)
                await run_interval("reconnected_held_stick",.25,assert_stopped=True)
                require(ex._controller_locomotion.command.state=="awaiting_neutral", "Reconnect bypassed neutral interlock")
                report["tracking_loss_and_rearm_passed"] = True
                entry["loss_stop"] = stopped
            neutral()
            stopped = await observe_first_physics_step(label+"_release")
            zero_requests(stopped,label+"_release")
            entry["release_stop"] = stopped
            settled = await run_interval(label+"_settle",4.,assert_stopped=True)
            entry["settled"] = settled
            require(math.hypot(*settled["velocity"][:2]) < .1, label+": planar motion did not settle")
            require(math.sqrt(sum(v*v for v in settled["angular"])) < .25, label+": angular motion did not settle")
            check_blue_hidden()
            entry["passed"] = True
            flush(True)
        report.update(passed=True,phase="complete")
    except BaseException as error:
        report.update(passed=False,error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if modified:
            neutral()
            ex._reset_gesture_locomotion()
            for name in ("_controller_command","_keyboard_command","_base_command"):
                value = getattr(ex,name,None)
                if value is not None: value[:] = 0.
            timeline.pause()
            try:
                # Pause is queued until an application update. Keep the neutral
                # synthetic reader until it takes effect, then restore real input.
                pause_deadline = time.perf_counter()+5.
                while timeline.is_playing() and time.perf_counter() < pause_deadline:
                    await app.next_update_async()
            finally:
                for name,(owned,value) in saved.items():
                    if owned: setattr(ex,name,value)
                    else: vars(ex).pop(name,None)
            report["input_readers_restored"] = all(
                (vars(ex).get(name) is value) if owned else name not in vars(ex)
                for name,(owned,value) in saved.items())
            report["final_paused"] = not timeline.is_playing()
            if not report["input_readers_restored"] or not report["final_paused"]:
                report.update(passed=False,cleanup_error="Input ownership or Pause was not restored")
        report.update(wall_seconds=time.perf_counter()-started,
                      gaze_tracker_preserved=getattr(ex,"_eye_gaze_tracker",None) is gaze,
                      fault=getattr(ex,"_articulation_health_fault",None),
                      new_callback_errors=sorted(set(getattr(ex,"_physics_step_error_logged",()))-old_errors))
        flush(True)

async def _controller_walking_runner(ex,job,options):
    try:
        await asyncio.wait_for(_controller_walking_replay(ex,job,options),timeout=options["timeout"])
    except BaseException as error:
        job["report"].update(passed=False,job_error=repr(error))
    finally:
        job["done"] = True
        _write_controller_walking(job,options)

_job["task"] = asyncio.ensure_future(_controller_walking_runner(EX,_job,OPTIONS))
print("CONTROLLER_WALKING_JOB "+json.dumps({key:value for key,value in _job.items() if key!="task"}))
"""

POLL_CODE = r"""
import json
_job = _CONTROLLER_WALKING_JOBS[JOB_ID]
print("CONTROLLER_WALKING_JOB "+json.dumps({key:value for key,value in _job.items() if key!="task"},allow_nan=False))
"""


def request_job(code: str, host: str, port: int, timeout: float) -> tuple[int, dict | None]:
    """Send one bounded server request and parse the persistent job snapshot."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = execute(code, host, port, timeout)
    messages = [line[len(PREFIX) :] for line in output.getvalue().splitlines() if line.startswith(PREFIX)]
    if status or not messages:
        print(output.getvalue(), end="")
        return status or 1, None
    return 0, json.loads(messages[-1])


def main() -> int:
    """Run or retrieve a bounded physical walking replay in the existing process."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=180.0, help="Overall Kit deadline, from 1 to 180 seconds")
    parser.add_argument(
        "--phase", choices=("all", *PHASES), default="all", help="Select one direction for an isolated replay"
    )
    parser.add_argument("--output", type=Path, default=ROOT / ".webcam/controller-walking-live.json")
    parser.add_argument("--resume", metavar="JOB_ID", help="Retrieve an existing replay without starting movement")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 1 <= args.timeout <= 180:
        parser.error("--timeout must be finite and between 1 and 180 seconds")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("--output must stay inside this repository")
    try:
        job_id = str(uuid.UUID(args.resume)) if args.resume else str(uuid.uuid4())
    except ValueError:
        parser.error("--resume requires the UUID from the start message")
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout + 15
    print(f"Controller walking job {job_id}", flush=True)
    if not args.resume:
        options = dict(
            job_id=job_id,
            timeout=args.timeout,
            phase=args.phase,
            progress_path=str(output.with_suffix(".progress.json")),
        )
        status, _ = request_job("OPTIONS = " + repr(options) + "\n" + LIVE_CODE, args.host, args.port, 10)
        if status:
            print(f"Recover an accepted job with --resume {job_id}; Kit owns cleanup.")
            return status
    code = "JOB_ID = " + repr(job_id) + "\n" + POLL_CODE
    phase = None
    try:
        while time.monotonic() < deadline:
            status, job = request_job(code, args.host, args.port, 10)
            if status:
                print(f"Poll failed; Kit owns cleanup. Resume with --resume {job_id}.")
                return status
            if job.get("job_id") != job_id:
                raise RuntimeError("Kit returned a different replay job")
            report = dict(job["report"], job_id=job_id, done=job["done"])
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if report.get("phase") != phase:
                phase = report.get("phase")
                print(f"Phase: {phase}", flush=True)
            if job["done"]:
                print(json.dumps({key: value for key, value in report.items() if key not in ("samples", "phases")}))
                return 0 if report.get("passed") else 1
            time.sleep(min(1, max(0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        request_job(
            "_CONTROLLER_WALKING_JOBS[" + repr(job_id) + ']["task"].cancel()\n' + code, args.host, args.port, 10
        )
        return 130
    print(f"Host wait expired; Kit owns cleanup. Resume with --resume {job_id}.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
