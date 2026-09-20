#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe pointing geometry in the loaded humanoid scene without moving actors.

Run with ordinary host Python and the existing Isaac Python Server. The bounded
Kit job pauses a healthy, initialized scene, queries production raycasts and
object approaches, then restores the prior Play/Pause state. It never changes
actor transforms, controller inputs, gaze, XR, camera, or authored collision.
Pause naturally disarms walking; release the pointing gesture before resuming.

The report distinguishes real clear routes from occlusion, corridor rejection,
and missing scene coverage. Far targets are sampled at 4, 8, 10, and 12 meters
over 24 bearings. Results outside the controller's forward cone are labelled;
they demonstrate scene geometry, not an authorized reverse-walking command.
Object aim points use local asset bounds carried by measured physics poses,
and only real hits on the intended registered object count as successful aims.

Example:
    python tools/validate_point_targets_live.py --output .webcam/point-targets-live.json

This is a geometry test, not a bare-hand or physical walking acceptance test.
Short socket polls retrieve a bounded asynchronous task. A sibling journal is
written by Kit, so --resume JOB_ID can recover a disconnected host poll.
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

JOB_PREFIX = "POINT_TARGETS_JOB "

LIVE_CODE = r"""
import asyncio
import json
import math
import time
import traceback
from pathlib import Path

import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.robot.policy.examples.interactive.humanoid.point_navigation_scene import PointNavigationScene

_POINT_TARGETS_JOBS = globals().setdefault("_POINT_TARGETS_JOBS", {})
for _registry_name, _registry in list(globals().items()):
    if _registry_name.endswith("_JOBS") and isinstance(_registry, dict):
        for _other_id, _other in _registry.items():
            _task = _other.get("task") if isinstance(_other, dict) else None
            if _task is not None and not _task.done():
                raise RuntimeError("Another live validation is running: " + str(_other_id))
if OPTIONS["job_id"] in _POINT_TARGETS_JOBS:
    raise RuntimeError("Point target job already exists")
for _old_id in list(_POINT_TARGETS_JOBS)[:-7]:
    del _POINT_TARGETS_JOBS[_old_id]
_job = {
    "job_id": OPTIONS["job_id"], "done": False,
    "report": {
        "passed": False, "phase": "starting", "geometry_only": True,
        "hardware_tracking_tested": False, "actors_commanded": False,
        "far_floor": [], "objects": [], "near_floor": [], "unregistered_hits": [],
    },
}
_POINT_TARGETS_JOBS[OPTIONS["job_id"]] = _job

def _write_point_targets_journal(job, options):
    path = Path(options["journal"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(job["report"], job_id=job["job_id"], done=job["done"]), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

async def _run_point_targets(ex, job, options):
    report = job["report"]
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    started = time.perf_counter()
    original_playing = bool(timeline.is_playing())
    original_stopped = bool(timeline.is_stopped())
    report.update(initial_playing=original_playing, initial_stopped=original_stopped)
    gaze = getattr(ex, "_eye_gaze_tracker", None)
    errors_before = set(getattr(ex, "_physics_step_error_logged", ()))
    g1 = getattr(ex, "g1", None)
    changed_timeline = False

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def check_health():
        require(time.perf_counter() - started < options["timeout"], "Geometry query deadline exceeded")
        require(ex._eye_gaze_tracker is gaze, "Gaze tracker identity changed during geometry validation")
        require(not getattr(ex, "_articulation_health_fault", None), "Articulation health fault")
        require(set(getattr(ex, "_physics_step_error_logged", ())) <= errors_before, "New physics callback error")
        require(g1.robot.is_physics_tensor_entity_valid(), "Physics tensor view became invalid")

    def vector(value, count):
        result = tuple(float(axis) for axis in value)
        require(len(result) == count and all(math.isfinite(axis) for axis in result), "Invalid measured pose")
        return result

    def root_pose():
        positions, orientations = g1.robot.get_world_poses()
        position = vector(ex._first_pose_value(positions), 3)
        quaternion = vector(ex._first_pose_value(orientations), 4)
        require(0.8 < math.sqrt(sum(axis * axis for axis in quaternion)) < 1.2, "Invalid root quaternion")
        return position, quaternion

    def hit_record(hit):
        if hit is None:
            return None
        return {**hit, "position": list(vector(hit["position"], 3))}

    def under(path, root):
        return path == root or path.startswith(root + "/")

    def registered(path):
        return next((root for root in roots if under(path, root)), None)

    async def progress(phase):
        report["phase"] = phase
        _write_point_targets_journal(job, options)
        await app.next_update_async()
        check_health()
        require(not timeline.is_playing() and not timeline.is_stopped(), "Timeline changed during paused queries")

    try:
        require(ex is not None and g1 is not None, "Load the humanoid example first")
        require(not original_stopped and ex._physics_ready and g1._initialized, "Start an initialized scene first")
        require(ex._gesture_locomotion_enabled, "This check requires pointing walking mode")
        require(ex._gesture_command.state in ("idle", "awaiting_release"), "Release pointing before running this check")
        check_health()
        if original_playing:
            changed_timeline = True
            timeline.pause()
            for _ in range(3):
                await app.next_update_async()
                if not timeline.is_playing():
                    break
        require(not timeline.is_playing() and not timeline.is_stopped(), "Timeline did not enter Pause")
        check_health()
        base, quaternion = root_pose()
        qw, qx, qy, qz = quaternion
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        # This measured-base-relative probe origin deliberately does not use or
        # override the HMD, hand, or camera pose. It is labelled in the report.
        origin = (base[0], base[1], max(1.25, base[2] + 0.65))
        controller = ex._locomotion_gesture
        shortfall = max(0.0, controller.arrival_radius_m - 0.10)
        scene = PointNavigationScene(ex._g1_prim_path)
        stage = omni.usd.get_context().get_stage()
        require(stage is not None, "No current USD stage")
        roots = tuple(dict.fromkeys(str(path) for path in ex._spawned_sample_box_paths))
        require(0 < len(roots) <= 32, "Expected one to 32 registered practice objects")
        report.update(
            robot_position=list(base), robot_quaternion=list(quaternion), robot_yaw_rad=yaw,
            query_origin=list(origin), query_origin_source="measured robot base + vertical offset",
            registered_objects=list(roots), stopping_shortfall_m=shortfall,
            loaded_max_target_distance_m=controller.max_target_distance_m,
            loaded_max_heading_rad=controller.max_heading_rad, ray_max_distance_m=scene.MAX_DISTANCE,
        )
        initial_object_poses = {}
        object_handles = {}
        bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])

        def clearance(target, shortfall_distance=shortfall):
            clear = scene.is_path_clear(base, target, stopping_shortfall_m=shortfall_distance)
            return {"clear": bool(clear), "error": scene.last_error, "blockers": list(scene.last_blocking_paths)}

        def reject_unregistered(hit, label):
            if hit is None or hit["walkable"] or registered(hit["path"]) or len(report["unregistered_hits"]) >= 12:
                return
            result = scene.resolve_object_approach(base, hit, roots, stopping_shortfall_m=shortfall)
            report["unregistered_hits"].append({
                "probe": label, "hit": hit_record(hit), "refused": result is None, "reason": scene.last_error,
            })

        await progress("far_floor")
        for distance in (4.0, 8.0, 10.0, 12.0):
            for index in range(24):
                bearing = -math.pi + index * math.tau / 24
                angle = yaw + bearing
                requested = (base[0] + distance * math.cos(angle), base[1] + distance * math.sin(angle), 0.0)
                direction = tuple(requested[i] - origin[i] for i in range(3))
                hit = scene.cast_ray(origin, direction)
                error = scene.last_error
                matches = bool(hit and hit["walkable"] and math.hypot(
                    hit["position"][0] - requested[0], hit["position"][1] - requested[1],
                ) <= 0.05)
                route = clearance(hit["position"]) if matches else None
                report["far_floor"].append({
                    "distance_m": distance, "bearing_deg": round(math.degrees(bearing), 3),
                    "inside_forward_cone": abs(bearing) <= controller.max_heading_rad + 1e-9,
                    "within_loaded_range": distance <= controller.max_target_distance_m,
                    "requested_floor": list(requested), "hit": hit_record(hit), "ray_error": error,
                    "hit_matches_requested_floor": matches, "corridor": route,
                })
                reject_unregistered(hit, "far_" + str(distance) + "_" + str(index))
                # Bound work per Kit update; no physics steps occur while paused.
                await progress("far_floor_" + str(int(distance)) + "m")

        await progress("registered_objects")
        for path in roots:
            item = {"object_root": path, "aims": [], "resolved": False}
            report["objects"].append(item)
            try:
                prim = stage.GetPrimAtPath(path)
                require(prim.IsValid(), "Registered object prim is missing")
                on_surface_attr = prim.GetAttribute("g1:packageOnSurface")
                item["floor_carton"] = bool(on_surface_attr and on_surface_attr.Get() is False)
                handle = ex._object_rigid_prims.get(path)
                if handle is None:
                    handle = RigidPrim(paths=path)
                object_handles[path] = handle
                positions, orientations = handle.get_world_poses()
                position = vector(ex._first_pose_value(positions), 3)
                rotation = vector(ex._first_pose_value(orientations), 4)
                initial_object_poses[path] = (position, rotation)
                item["physics_position"] = list(position)
                item["physics_quaternion"] = list(rotation)
                local_range = bounds_cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
                require(not local_range.IsEmpty(), "Object has no local geometry bound")
                local_matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
                if isinstance(local_matrix, tuple):
                    local_matrix = local_matrix[0]
                scale = vector(Gf.Transform(local_matrix).GetScale(), 3)
                measured_rotation = Gf.Matrix4d(1.0)
                measured_rotation.SetRotate(Gf.Quatd(rotation[0], Gf.Vec3d(*rotation[1:])).GetNormalized())

                def physical_point(local):
                    scaled = Gf.Vec3d(*(float(local[i]) * scale[i] for i in range(3)))
                    return Gf.Vec3d(*position) + measured_rotation.TransformDir(scaled)

                local_center = local_range.GetMidpoint()
                item["aim_source"] = "asset-local bounds + measured rigid pose; actual collider hit required"
                # Three interior heights accommodate hollow or irregular props.
                hit = None
                for fraction in (0.5, 0.25, 0.75):
                    local = Gf.Vec3d(local_center)
                    local[2] = local_range.GetMin()[2] + fraction * local_range.GetSize()[2]
                    aim = physical_point(local)
                    candidate = scene.cast_ray(origin, aim - Gf.Vec3d(*origin))
                    item["aims"].append({
                        "point": list(aim), "hit": hit_record(candidate), "error": scene.last_error,
                    })
                    reject_unregistered(candidate, path)
                    if candidate is not None and under(candidate["path"], path) and not candidate["walkable"]:
                        hit = candidate
                        break
                if hit is None:
                    item["reason"] = "No unoccluded physical hit on the intended object from the probe origin"
                else:
                    item["direct_hit"] = hit_record(hit)
                    resolved = scene.resolve_object_approach(base, hit, roots, stopping_shortfall_m=shortfall)
                    item["approach_error"] = scene.last_error
                    item["approach_blockers"] = list(scene.last_blocking_paths)
                    item["approach"] = resolved
                    # A second production cast must still stop at the same object:
                    # approach resolution cannot turn the blue ray into a floor hit.
                    repeat = scene.cast_ray(origin, Gf.Vec3d(*hit["position"]) - Gf.Vec3d(*origin))
                    item["blue_hit_preserved"] = bool(
                        repeat and repeat["path"] == hit["path"] and not repeat["walkable"]
                        and (repeat["position"] - hit["position"]).GetLength() <= 0.01
                    )
                    if resolved:
                        target = resolved["target"]
                        floor_hit = scene.cast_ray(
                            (target[0], target[1], max(base[2] + 1.0, hit["position"][2] + 0.5)),
                            (0.0, 0.0, -1.0),
                        )
                        item["verified_floor_hit"] = hit_record(floor_hit)
                        floor_verified = bool(floor_hit and floor_hit["walkable"] and
                            (Gf.Vec3d(*target) - floor_hit["position"]).GetLength() <= 0.05)
                        route = clearance(target)
                        item["fresh_corridor"] = route
                        item["resolved"] = floor_verified and route["clear"] and item["blue_hit_preserved"]

                    if item["floor_carton"]:
                        surface = hit["position"]
                        planar = math.hypot(surface[0] - base[0], surface[1] - base[1])
                        if planar > 0.5:
                            for offset in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
                                if planar - offset < 0.5:
                                    continue
                                fraction = (planar - offset) / planar
                                x = base[0] + (surface[0] - base[0]) * fraction
                                y = base[1] + (surface[1] - base[1]) * fraction
                                down = scene.cast_ray((x, y, max(base[2] + 1.0, surface[2] + 0.5)), (0, 0, -1))
                                near = {"object_root": path, "surface_offset_m": offset, "floor_hit": hit_record(down)}
                                if down and down["walkable"]:
                                    # Also verify the visible blue ray reaches this floor
                                    # rather than accepting a patch occluded by the object.
                                    visible = scene.cast_ray(origin, down["position"] - Gf.Vec3d(*origin))
                                    near["visible_floor"] = bool(visible and visible["walkable"] and
                                        (visible["position"] - down["position"]).GetLength() <= 0.05)
                                    near["full_corridor"] = clearance(down["position"], 0.0)
                                    near["stopping_corridor"] = clearance(down["position"])
                                    near["endpoint_false_rejection_removed"] = bool(
                                        near["visible_floor"] and not near["full_corridor"]["clear"]
                                        and near["stopping_corridor"]["clear"]
                                    )
                                else:
                                    near["reason"] = scene.last_error
                                report["near_floor"].append(near)
            except Exception as error:
                item["probe_error"] = repr(error)
            await progress("registered_objects")

        final_base, final_quaternion = root_pose()
        report["paused_root_unchanged"] = (
            max(abs(a - b) for a, b in zip(base, final_base)) <= 1e-6
            and max(abs(a - b) for a, b in zip(quaternion, final_quaternion)) <= 1e-6
        )
        moved_objects = []
        for path, (position, rotation) in initial_object_poses.items():
            positions, orientations = object_handles[path].get_world_poses()
            final_position = vector(ex._first_pose_value(positions), 3)
            final_rotation = vector(ex._first_pose_value(orientations), 4)
            if (max(abs(a - b) for a, b in zip(position, final_position)) > 1e-6
                    or max(abs(a - b) for a, b in zip(rotation, final_rotation)) > 1e-6):
                moved_objects.append(path)
        report["objects_changed_while_paused"] = moved_objects
        report["far_summary"] = {}
        for distance in (4.0, 8.0, 10.0, 12.0):
            samples = [item for item in report["far_floor"] if item["distance_m"] == distance]
            clear = [item for item in samples if item["corridor"] and item["corridor"]["clear"]]
            report["far_summary"][str(int(distance)) + "m"] = {
                "samples": len(samples), "clear_bearings_deg": [item["bearing_deg"] for item in clear],
                "forward_clear_bearings_deg": [item["bearing_deg"] for item in clear if item["inside_forward_cone"]],
            }
        direct = [item for item in report["objects"] if item.get("direct_hit")]
        criteria = {
            "loaded_range_reaches_12m": controller.max_target_distance_m >= 12.0,
            "clear_route_at_8m": bool(report["far_summary"]["8m"]["clear_bearings_deg"]),
            "clear_route_at_10m": bool(report["far_summary"]["10m"]["clear_bearings_deg"]),
            "clear_route_at_12m": bool(report["far_summary"]["12m"]["clear_bearings_deg"]),
            "floor_carton_has_real_clear_approach": any(item.get("floor_carton") and item["resolved"] for item in direct),
            "direct_object_blue_hits_preserved": bool(direct) and all(item["blue_hit_preserved"] for item in direct),
            "unregistered_obstacles_refused": bool(report["unregistered_hits"]) and
                all(item["refused"] for item in report["unregistered_hits"]),
            "near_floor_endpoint_improvement_observed": any(
                item.get("endpoint_false_rejection_removed", False) for item in report["near_floor"]),
            "paused_actors_unchanged": report["paused_root_unchanged"] and not moved_objects,
            "object_probes_completed": not any(item.get("probe_error") for item in report["objects"]),
        }
        report["criteria"] = criteria
        report["not_demonstrated"] = [key for key, value in criteria.items() if not value]
        report["passed"] = all(criteria.values())
        report["phase"] = "geometry_complete"
        check_health()
    except BaseException as error:
        report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
    finally:
        # No Stop/Reset and no commands are injected. Restore only the state this
        # job changed, even on an incomplete geometry report or task timeout.
        try:
            if changed_timeline:
                timeline.play()
                for _ in range(3):
                    await app.next_update_async()
                    if timeline.is_playing():
                        break
            restored = bool(timeline.is_playing()) == original_playing and bool(timeline.is_stopped()) == original_stopped
            report["timeline_restored"] = restored
            if not restored:
                report["passed"] = False
        except BaseException as error:
            report.update(passed=False, timeline_restored=False, restore_error=repr(error))
        report["gaze_identity_retained"] = getattr(ex, "_eye_gaze_tracker", None) is gaze
        report["new_callback_errors"] = sorted(set(getattr(ex, "_physics_step_error_logged", ())) - errors_before)
        report["fault"] = getattr(ex, "_articulation_health_fault", None)
        if report["new_callback_errors"] or report["fault"] or not report["gaze_identity_retained"]:
            report["passed"] = False
        report["wall_seconds"] = time.perf_counter() - started
        report["final_playing"] = bool(timeline.is_playing())
        report["phase"] = "complete"
        _write_point_targets_journal(job, options)

async def _point_targets_runner(ex, job, options):
    try:
        await asyncio.wait_for(_run_point_targets(ex, job, options), timeout=options["timeout"])
    except BaseException as error:
        job["report"].update(passed=False, job_error=repr(error))
    finally:
        job["done"] = True
        _write_point_targets_journal(job, options)

_job["task"] = asyncio.ensure_future(_point_targets_runner(EX, _job, OPTIONS))
print("POINT_TARGETS_JOB " + json.dumps({key: value for key, value in _job.items() if key != "task"}))
"""

POLL_CODE = r"""
import json
_job = _POINT_TARGETS_JOBS[JOB_ID]
_task = _job["task"]
if _task.done() and not _job["done"]:
    _job["done"] = True
    _job["report"].update(passed=False, job_error="Cancelled" if _task.cancelled() else repr(_task.exception()))
print("POINT_TARGETS_JOB " + json.dumps({key: value for key, value in _job.items() if key != "task"}, allow_nan=False))
"""


def request_job(code: str, host: str, port: int, timeout: float) -> tuple[int, dict | None]:
    """Send one short socket request and recover the structured job snapshot."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = execute(code, host, port, timeout)
    lines = [line[len(JOB_PREFIX) :] for line in output.getvalue().splitlines() if line.startswith(JOB_PREFIX)]
    if status or not lines:
        print(output.getvalue(), end="")
        return status or 1, None
    return 0, json.loads(lines[-1])


def main() -> int:
    """Run or resume geometry probes using the already loaded Isaac instance."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=90.0, help="Bounded Kit deadline, at most 180 seconds")
    parser.add_argument("--output", type=Path, default=Path(".webcam/point-targets-live.json"))
    parser.add_argument("--resume", metavar="JOB_ID", help="Poll an existing job without touching the scene")
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
    print(f"Point geometry job {job_id}; journal {journal}", flush=True)
    if not args.resume:
        options = {"job_id": job_id, "timeout": args.timeout, "journal": str(journal)}
        status, _ = request_job("OPTIONS = " + repr(options) + "\n" + LIVE_CODE, args.host, args.port, 10)
        if status:
            print(f"Start response failed; recover an accepted job with --resume {job_id}.")
            return status
    last_phase = None
    poll = "JOB_ID = " + repr(job_id) + "\n" + POLL_CODE
    while time.monotonic() < deadline:
        status, reply = request_job(poll, args.host, args.port, min(10, max(0.1, deadline - time.monotonic())))
        if status:
            print(f"Kit poll failed. Inspect {journal}; resume with --resume {job_id}.")
            return status
        if reply.get("job_id") != job_id:
            raise RuntimeError("Kit returned an unexpected job ID")
        report = dict(reply["report"], job_id=job_id, done=reply["done"])
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if report.get("phase") != last_phase:
            last_phase = report.get("phase")
            print(f"Phase: {last_phase}", flush=True)
        if reply["done"]:
            summary = {
                key: report.get(key)
                for key in (
                    "passed",
                    "criteria",
                    "far_summary",
                    "not_demonstrated",
                    "timeline_restored",
                    "error",
                    "job_error",
                )
            }
            print(json.dumps(summary, indent=2, allow_nan=False))
            print(f"Full report: {output}")
            return 0 if report.get("passed") else 1
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    print(f"Host wait expired. Kit owns timeout cleanup; inspect {journal} or --resume {job_id}.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
