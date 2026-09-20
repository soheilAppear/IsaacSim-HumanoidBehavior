#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay pointing-only destination walking in an already loaded Isaac Sim.

Run with ordinary host Python after loading the walking example. Only the
navigation hand reader receives synthetic input. Production destination
selection, physical walking, scene raycasts, clearance, blue visuals, arm/finger
callbacks, and safety guards remain active. Existing eye tracking and red gaze
visuals remain unchanged. This does not test Quest hardware or human gait quality.

The bounded replay runs as a persistent Kit task and always restores the hand
reader, zeros navigation, and pauses on completion or failure. It never resets
the scene or changes the robot root pose. Short polls save a recoverable report;
--resume JOB_ID reconnects to a job after a host connection interruption. Reports
must remain inside the repository. A sibling .progress.json is flushed by Kit
while the replay runs, including when the host is disconnected.
The 600-second wall-time allowance accommodates slow VR rendering. Velocity,
arrival, stopping, and tracking-loss limits remain enforced; route duration also
allows physical stepping turns and detours, whose progress is checked per waypoint.
--distance sets the first destination's distance in meters, default 0.9.
--bearing sets its angle in degrees relative to the initial robot heading,
positive left, default 0. Floor and body-clearance checks still apply. The factory
profile may physically turn toward targets behind it; legacy profiles retain their
forward cone. --allow-route permits a blocked first floor corridor to use the
production incremental planner, with a 60-second planning/dwell timeout. It never
ignores obstacles. Release/loss checks keep separate clear 0.9-meter destinations.
--object-root /World/... replaces only the first floor destination with a ray at
a registered object's measured physical center. It requires an actual object hit,
a separately verified clear floor approach, and a permitted controller heading.
The blue ray must end on the object while the destination marker stays on floor.
--distance and --bearing apply only when --object-root is omitted.
--approach-only validates arrival, stopping, an open-hand release, and final balance.
Use it for an object approach whose stopping location has no clear forward exit
for the separate moving release/tracking-loss fixtures. The default full suite
still runs those fixtures; approach-only reports deadman_motion_tested=false.
--turn-release-only replaces the approach with left/right stepping-turn deadman
checks. Its fixed +90/-90 degree floor fixtures use --distance 0.9 (default) or
1.5 meters and must each have a clear real index ray and body corridor. Each turn
must rotate at least 0.25 rad before open-hand release (left) or tracking loss
(right), followed by three simulation seconds of measured braking. The body may
drift while stepping; zero requested translation is checked separately.
The legacy filename and job identifiers are retained for command compatibility;
this replay requires the current point_only navigation mode.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import time
import uuid
from pathlib import Path

from kit_exec import DEFAULT_HOST, DEFAULT_PORT, execute

JOB_PREFIX = "POINT_GAZE_WALKING_JOB "
REPORT_PREFIX = "POINT_GAZE_WALKING_REPORT "
ROOT = Path(__file__).resolve().parents[1]

LIVE_CODE = r"""
async def _validate_point_gaze_walking(ex, options, job):
    import json
    import math
    import time
    from pathlib import Path

    import numpy as np
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    timeline = omni.timeline.get_timeline_interface()
    app = omni.kit.app.get_app()
    report = {
        "job_id": options["job_id"], "passed": False,
        "hardware_tracking_tested": False, "payload_tested": False,
        "mode": "point_only",
        "requested_destination_distance_m": options["distance"],
        "requested_destination_bearing_deg": options["bearing"],
        "requested_object_root": options.get("object_root"),
        "allow_route": options.get("allow_route", False),
        "initial_direct_corridor_clear": None, "initial_corridor_error": None,
        "planned_route_at_activation": [], "max_abs_requested_yaw_rad_s": 0.0,
        "initial_pivot": {"samples": 0, "no_translation_requested": True,
                          "maximum_planar_drift_m": 0.0, "duration_sim_s": 0.0},
        "suite": ("turn_release_only" if options.get("turn_release_only", False)
                  else "approach_only" if options.get("approach_only", False) else "full"),
        "deadman_motion_tested": False,
        "fixture": "synthetic navigation hand only; production scene queries, visuals and physical controller; gaze untouched",
        "phases": [], "samples": [], "phase": "prepare", "input_readers_restored": False,
        "target_checks": [],
    }
    job["report"] = report
    report_path = Path(options["progress_path"])
    saved = {}
    gaze = getattr(ex, "_eye_gaze_tracker", None)
    old_errors = set(getattr(ex, "_physics_step_error_logged", ()))
    current = {"hand": "open", "ray_target": None, "destination": None, "object_root": None,
               "tip": None, "knuckle": None, "wrist": None}
    last_flush = -math.inf

    def flush(force=False):
        nonlocal last_flush
        now = time.perf_counter()
        if force or now - last_flush >= 1.0:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            last_flush = now

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def check_health():
        require(not ex._articulation_health_fault, str(ex._articulation_health_fault))
        new_errors = sorted(set(ex._physics_step_error_logged) - old_errors)
        require(not new_errors, "New physics callback errors: " + repr(new_errors))
        require(ex._eye_gaze_tracker is gaze, "The existing eye gaze tracker changed")
        require(ex.g1.robot.is_physics_tensor_entity_valid(), "Articulation tensor became invalid")

    def vector(value):
        result = np.asarray(ex._first_pose_value(value), dtype=float)
        require(bool(np.isfinite(result).all()), "Nonfinite physics observation")
        return result

    def snapshot(label):
        positions, orientations = ex.g1.robot.get_world_poses()
        velocities, angular = ex.g1.robot.get_velocities()
        position, quaternion, velocity, turn = map(vector, (positions, orientations, velocities, angular))
        return {
            "phase": label, "sim_time": float(ex.g1._walk_time),
            "wall_time": time.perf_counter(),
            "input_gap_limit_s": ex._locomotion_gesture.observation_gap_limit_s,
            "position": position.tolist(), "quaternion": quaternion.tolist(),
            "velocity": velocity.tolist(), "angular": turn.tolist(),
            "state": ex._gesture_command.state, "request": list(ex._gesture_command.velocity),
            "filtered": ex._base_command.tolist(), "target": ex._gesture_command.target,
            "comfort_eye_height_m": float(ex._head_camera_transform_op.Get().ExtractTranslation()[2])
                if ex._head_camera_comfort_enabled else None,
        }

    async def advance(label):
        await app.next_update_async()
        check_health()
        report["max_abs_requested_yaw_rad_s"] = max(
            report["max_abs_requested_yaw_rad_s"], abs(float(ex._gesture_command.velocity[2])))
        if not report["samples"] or ex.g1._walk_time - report["samples"][-1]["sim_time"] >= 0.1:
            report["samples"].append(snapshot(label))
        flush()

    async def wait_for_state(expected, label, wall_timeout=10.0):
        report["phase"] = label
        deadline = time.perf_counter() + wall_timeout
        while ex._gesture_command.state != expected:
            require(time.perf_counter() < deadline,
                    "Expected " + expected + ": " + str(ex.gesture_locomotion_status()))
            await advance(label)

    async def settle(label, duration):
        report["phase"] = label
        until = ex.g1._walk_time + duration
        while ex.g1._walk_time < until:
            await advance(label)
        end = snapshot(label)
        report["phases"].append({"name": label, "end": end})
        flush(True)
        return end

    def read_hand():
        if current["hand"] == "invalid":
            return None, None, None, None
        curls = {"index": 0.05, "middle": 0.9, "ring": 0.9, "little": 0.9, "thumb": 0.1}
        if current["hand"] == "open":
            curls = dict.fromkeys(curls, 0.0)
        return curls, current["tip"], current["knuckle"], current["wrist"]

    def configure_floor_target(distance, bearing_deg=0.0, *, allow_route=False, record_initial=False):
        base, yaw = ex._get_g1_base_pose_for_arms()
        heading = yaw + math.radians(bearing_deg)
        forward = Gf.Vec3d(math.cos(heading), math.sin(heading), 0)
        left = Gf.Vec3d(-math.sin(heading), math.cos(heading), 0)
        candidate = Gf.Vec3d(base[0], base[1], 0) + distance * forward
        hit = ex._navigation_scene.cast_ray(candidate + Gf.Vec3d(0, 0, 2), Gf.Vec3d(0, 0, -1))
        require(hit is not None and hit["walkable"], "Test destination is not a measured walkable floor hit")
        target = Gf.Vec3d(*hit["position"])
        clear = ex._navigation_scene.is_path_clear(base, target)
        corridor_error = ex._navigation_scene.last_error
        if record_initial:
            report["initial_direct_corridor_clear"] = bool(clear)
            report["initial_corridor_error"] = corridor_error
        require(clear or allow_route, "Test route is blocked: " + str(corridor_error))
        if allow_route:
            require(ex._locomotion_gesture.allow_routing, "The loaded controller does not enable factory routing")
        tip = Gf.Vec3d(base[0], base[1], 1.05) + 0.35 * forward + 0.18 * left
        direction = (target - tip).GetNormalized()
        current.update(tip=tip, knuckle=tip - 0.07 * direction, wrist=tip - 0.15 * direction)
        # Verify the same ray used by production navigation actually reaches this
        # floor patch, rather than merely assuming an unoccluded infinite ray.
        ray_hit = ex._navigation_scene.cast_ray(tip, direction)
        require(ray_hit is not None and ray_hit["walkable"], "Synthetic index ray was occluded before the floor")
        require((Gf.Vec3d(*ray_hit["position"]) - target).GetLength() <= 0.02,
                "Synthetic index ray reached a different surface")
        current.update(ray_target=Gf.Vec3d(*ray_hit["position"]),
                       destination=Gf.Vec3d(*ray_hit["position"]), object_root=None)
        return target, forward

    def configure_object_target(object_root):
        from isaacsim.core.experimental.prims import RigidPrim

        stage = omni.usd.get_context().get_stage()
        require(object_root in ex._spawned_sample_box_paths, "Object root is not a registered practice object")
        prim = stage.GetPrimAtPath(object_root)
        require(prim.IsValid(), "Requested object root is missing")
        handle = ex._object_rigid_prims.get(object_root)
        if handle is None:
            handle = RigidPrim(paths=object_root)
        positions, orientations = handle.get_world_poses()
        position, quaternion = vector(positions), vector(orientations)
        require(position.shape == (3,) and quaternion.shape == (4,), "Object physics pose has invalid dimensions")
        require(0.8 < np.linalg.norm(quaternion) < 1.2, "Object physics quaternion is invalid")
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        local_range = bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
        require(not local_range.IsEmpty(), "Requested object has no asset-local geometry bound")
        local_matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
        if isinstance(local_matrix, tuple):
            local_matrix = local_matrix[0]
        scale = Gf.Transform(local_matrix).GetScale()
        center = local_range.GetMidpoint()
        scaled_center = Gf.Vec3d(*(float(center[axis]) * float(scale[axis]) for axis in range(3)))
        rotation = Gf.Matrix4d().SetRotate(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*quaternion[1:])).GetNormalized())
        physical_center = Gf.Vec3d(*position) + rotation.TransformDir(scaled_center)
        require(all(math.isfinite(value) for value in physical_center), "Object physical center is nonfinite")
        base, yaw = ex._get_g1_base_pose_for_arms()
        toward = Gf.Vec3d(physical_center[0] - base[0], physical_center[1] - base[1], 0)
        require(toward.GetLength() >= 0.5, "Object is less than 0.5 meters from the robot")
        toward.Normalize()
        left = Gf.Vec3d(-toward[1], toward[0], 0)
        tip = Gf.Vec3d(base[0], base[1], 1.05) + 0.35 * toward + 0.18 * left
        direction = (physical_center - tip).GetNormalized()
        hit = ex._navigation_scene.cast_ray(tip, direction)
        require(hit is not None and not hit["walkable"], "Object center ray did not hit a physical object")
        hit_path = str(hit["path"])
        require(hit_path == object_root or hit_path.startswith(object_root + "/"),
                "Object center ray was occluded by another collider: " + hit_path)
        shortfall = max(0.0, ex._locomotion_gesture.arrival_radius_m - 0.10)
        approach = ex._navigation_scene.resolve_object_approach(
            base, hit, ex._spawned_sample_box_paths, stopping_shortfall_m=shortfall
        )
        require(approach is not None, "Object approach is blocked: " + str(ex._navigation_scene.last_error))
        require(approach["source_object_root"] == object_root, "Approach resolver selected another object")
        target = Gf.Vec3d(*approach["target"])
        floor_hit = ex._navigation_scene.cast_ray(
            (target[0], target[1], max(base[2] + 1.0, hit["position"][2] + 0.5)), (0, 0, -1)
        )
        require(floor_hit is not None and floor_hit["walkable"]
                and (Gf.Vec3d(*floor_hit["position"]) - target).GetLength() <= 0.02,
                "Object approach target is not an independently measured walkable floor hit")
        require(ex._navigation_scene.is_path_clear(base, target, stopping_shortfall_m=shortfall),
                "Object approach corridor is blocked: " + str(ex._navigation_scene.last_error))
        planar = Gf.Vec3d(target[0] - base[0], target[1] - base[1], 0)
        distance = planar.GetLength()
        require(math.isfinite(distance) and 0.5 <= distance <= ex._locomotion_gesture.max_target_distance_m,
                "Object approach floor target is outside the allowed distance")
        heading = math.atan2(planar[1], planar[0])
        bearing = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        require(ex._locomotion_gesture.allow_turn_in_place or abs(bearing) <= ex._locomotion_gesture.max_heading_rad,
                "Object approach floor target is outside the controller's forward cone")
        current.update(tip=tip, knuckle=tip - 0.07 * direction, wrist=tip - 0.15 * direction,
                       ray_target=Gf.Vec3d(*hit["position"]), destination=target, object_root=object_root)
        report["object_approach"] = {
            "object_root": object_root, "physics_position": position.tolist(),
            "physics_quaternion": quaternion.tolist(), "physical_center": list(physical_center),
            "aim_source": "asset-local bounds and measured rigid-body pose",
            "hit_path": hit_path, "hit_position": list(hit["position"]), "hit_distance_m": hit["distance"],
            "floor_path": floor_hit["path"], "approach_target": list(target),
            "approach_distance_m": distance, "approach_bearing_deg": math.degrees(bearing),
            "standoff_m": approach["standoff_m"], "stopping_shortfall_m": shortfall,
            "fresh_floor_verified": True, "fresh_corridor_clear": True,
        }
        return target, planar.GetNormalized()

    def validate_locked_target(label):
        locked = ex._gesture_command.target
        require(locked is not None, "Navigation activated without a locked destination")
        error = (Gf.Vec3d(*locked) - current["destination"]).GetLength()
        require(math.isfinite(error) and error <= 0.02,
                "Locked destination differs from the verified floor destination")
        item = {"phase": label, "ray_hit": list(current["ray_target"]),
                "expected_floor_target": list(current["destination"]), "locked_target": list(locked), "error_m": error}
        if current["object_root"] is None:
            item["ray_floor_hit"] = list(current["ray_target"])
        else:
            status = ex._navigation_query_status
            require(status.get("destination_kind") == "object_approach", "Production did not lock an object approach")
            require(status.get("source_object_root") == current["object_root"], "Production locked a different object")
            item.update(destination_kind="object_approach", source_object_root=current["object_root"])
        report["target_checks"].append(item)

    def validate_visuals():
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(ex._navigation_scene.ROOT)
        ray = stage.GetPrimAtPath(ex._navigation_scene.ROOT + "/Ray")
        require(root.IsValid() and ray.IsValid(), "Blue pointing ray prim is missing")
        require(UsdGeom.Imageable(ray).ComputeVisibility() != UsdGeom.Tokens.invisible,
                "Blue pointing ray is hidden during navigation")
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(ray)
        require(all(math.isfinite(matrix[row][column]) for row in range(4) for column in range(4)),
                "Blue pointing ray has a nonfinite transform")
        for prim in Usd.PrimRange(root):
            require(not prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI),
                    "Navigation visualization unexpectedly affects physics: " + str(prim.GetPath()))
        material, relationship = UsdShade.MaterialBindingAPI(ray).ComputeBoundMaterial()
        require(material and str(material.GetPath()).endswith("/Blue"), "Pointing ray material is not blue")
        result = {"ray_path": str(ray.GetPath()), "visible": True, "finite_transform": True,
                  "noncolliding": True, "material": str(material.GetPath())}
        if current["object_root"] is not None:
            endpoint = matrix.Transform(Gf.Vec3d(0, 0, 0.5))
            hit_error = (endpoint - current["ray_target"]).GetLength()
            require(hit_error <= 0.02, "Blue ray endpoint moved away from the physical object hit")
            marker = stage.GetPrimAtPath(ex._navigation_scene.ROOT + "/Target")
            require(marker.IsValid() and UsdGeom.Imageable(marker).ComputeVisibility() != UsdGeom.Tokens.invisible,
                    "Object approach floor marker is hidden")
            marker_position = UsdGeom.XformCache().GetLocalToWorldTransform(marker).ExtractTranslation()
            expected_marker = Gf.Vec3d(*ex._gesture_command.target) + Gf.Vec3d(0, 0, 0.01)
            marker_error = (marker_position - expected_marker).GetLength()
            require(marker_error < 1e-5, "Destination marker is not on the separately locked floor target")
            separation = (endpoint - Gf.Vec3d(*ex._gesture_command.target)).GetLength()
            require(separation >= 0.10, "Object ray endpoint and floor destination were incorrectly merged")
            result.update(object_root=current["object_root"], blue_endpoint=list(endpoint),
                          physical_hit_error_m=hit_error, floor_marker=list(marker_position),
                          floor_marker_error_m=marker_error, ray_floor_target_separation_m=separation)
        return result

    def validate_carton_clearance():
        stage = omni.usd.get_context().get_stage()
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        base, yaw = ex._get_g1_base_pose_for_arms()
        checked = []
        for path in ex._spawned_sample_box_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or prim.GetAttribute("g1:packageOnSurface").Get() is not False:
                continue
            box = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                continue
            center = box.GetMidpoint()
            clear = ex._navigation_scene.is_path_clear(base, Gf.Vec3d(center[0], center[1], 0))
            checked.append({"path": path, "corridor_clear": bool(clear)})
            require(not clear, "A physical floor carton did not block its approach corridor")
        require(bool(checked), "No floor cartons were available for obstacle validation")
        return checked

    def validate_camera_height():
        heights = [sample["comfort_eye_height_m"] for sample in report["samples"]
                   if sample.get("comfort_eye_height_m") is not None]
        if heights:
            report["walking_camera_height_peak_to_peak_m"] = max(heights) - min(heights)
            require(report["walking_camera_height_peak_to_peak_m"] < 0.0005,
                    "Walking body bob reached the calibrated comfort-camera height")

    async def validate_turn_deadman():
        require(ex._locomotion_gesture.allow_turn_in_place, "Stepping turns are disabled in the current controller")
        require(options["distance"] in (.9, 1.5), "Turn-release fixture distance must be 0.9 or 1.5 meters")
        report["turn_deadman_checks"] = []
        for bearing, loss in ((90.0, "release"), (-90.0, "tracking_loss")):
            label = "left_turn_release" if bearing > 0 else "right_turn_tracking_loss"
            current["hand"] = "open"
            await wait_for_state("idle", label + "_rearm")
            target, unused_forward = configure_floor_target(options["distance"], bearing)
            base, start_yaw = ex._get_g1_base_pose_for_arms()
            require(ex._navigation_scene.is_turn_clear(base),
                    "Turn-release fixture has insufficient turning clearance: " + str(ex._navigation_scene.last_error))
            before = snapshot(label)
            current["hand"] = "point"
            await wait_for_state("active", label + "_dwell")
            validate_locked_target(label)
            sign = 1.0 if bearing > 0 else -1.0
            started_sim = ex.g1._walk_time
            deadline_sim = started_sim + 10.0
            deadline_wall = time.perf_counter() + 60.0
            yaw_progress = 0.0
            drift = 0.0
            observations = 0
            report["phase"] = label
            while yaw_progress < .25:
                require(ex._gesture_command.state == "active",
                        "Stepping turn stopped before physical progress: " + str(ex.gesture_locomotion_status()))
                request = ex._gesture_command.velocity
                require(request[0] == 0.0 and request[1] == 0.0,
                        "Turn-release fixture requested translation before the turn completed")
                require(sign * request[2] > 0.0, "Stepping-turn yaw request has the wrong sign")
                require(abs(request[2]) <= ex._locomotion_gesture.max_requested_yaw_speed + 1e-8,
                        "Stepping-turn request exceeded the configured yaw cap")
                require(ex.g1._walk_time < deadline_sim and time.perf_counter() < deadline_wall,
                        "Stepping turn did not achieve 0.25 rad measured heading progress within its bounded deadline")
                await advance(label)
                measured, current_yaw = ex._get_g1_base_pose_for_arms()
                yaw_progress = sign * math.atan2(math.sin(current_yaw - start_yaw), math.cos(current_yaw - start_yaw))
                drift = max(drift, math.dist(tuple(measured)[:2], tuple(base)[:2]))
                observations += 1
            moving = snapshot(label)
            require(moving["state"] == "active" and sign * moving["request"][2] > 0.0,
                    "Turn stopped independently before the injected release/tracking loss")
            require(moving["request"][0] == 0.0 and moving["request"][1] == 0.0,
                    "Turn changed to forward travel before the injected release/tracking loss")
            command_change_wall = time.perf_counter()
            current["hand"] = "open" if loss == "release" else "invalid"
            await wait_for_state("idle" if loss == "release" else "awaiting_release", label + "_stop")
            stopped_request = snapshot(label + "_stop")
            require(stopped_request["request"] == [0.0, 0.0, 0.0], loss + " did not clear the turn request")
            require(all(value == 0.0 for value in stopped_request["filtered"]),
                    loss + " did not bypass residual turn filtering")
            await settle(label + "_braking", 3.0)
            stopped = snapshot(label + "_braking")
            planar_speed = math.hypot(*stopped["velocity"][:2])
            angular_speed = float(np.linalg.norm(stopped["angular"]))
            require(planar_speed < .1, "Turn stopping retained planar speed >=0.1 m/s after three simulation seconds")
            require(angular_speed < .25, "Turn stopping retained angular speed >=0.25 rad/s after three simulation seconds")
            check_health()
            result = {
                "name": label, "bearing_deg": bearing, "target": list(target), "start": before,
                "moving": moving, "immediate_stop_observation": stopped_request, "braked": stopped,
                "measured_heading_progress_rad": yaw_progress, "maximum_natural_planar_drift_m": drift,
                "zero_translation_requested": True, "turn_observations": observations,
                "stop_state_observed_after_wall_s": stopped_request["wall_time"] - command_change_wall,
                "stop_planar_speed_m_s": planar_speed, "stop_angular_speed_rad_s": angular_speed,
                "turn_duration_sim_s": moving["sim_time"] - started_sim,
            }
            report["turn_deadman_checks"].append(result)
            report["phases"].append({"name": label, "start": before, "end": stopped})
            flush(True)
        report["deadman_motion_tested"] = True

    try:
        require(ex is not None and ex.g1 is not None, "Load the humanoid example before validation")
        require(ex._gesture_locomotion_enabled and ex.g1._physical_locomotion,
                "The loaded example must use physical gesture walking")
        require(ex.gesture_locomotion_status().get("mode") == "point_only",
                "The loaded example must use the current point_only navigation mode")
        require(ex._physics_ready and not ex._timeline_stopped, "Initialize the loaded example with Play first")
        check_health()
        timeline.pause()
        for name in ("_read_navigation_hand",):
            require(callable(getattr(ex, name, None)), "Missing current navigation measurement seam: " + name)
            saved[name] = (name in ex.__dict__, ex.__dict__.get(name))
        ex._read_navigation_hand = read_hand
        ex._reset_gesture_locomotion()
        require(math.isfinite(options["distance"]) and 0.5 <= options["distance"] <= 50.0,
                "Destination distance must be finite and in [0.5, 50.0] meters")
        require(math.isfinite(options["bearing"]) and -180.0 <= options["bearing"] <= 180.0,
                "Destination bearing must be finite and in [-180.0, 180.0] degrees")
        object_root = options.get("object_root")
        target, forward = (configure_floor_target(options["distance"], 90.0, record_initial=True)
                           if options.get("turn_release_only") else configure_object_target(object_root) if object_root
                           else configure_floor_target(options["distance"], options["bearing"],
                                                       allow_route=options.get("allow_route", False), record_initial=True))
        report["initial_target"] = list(target)
        report["initial_target_direction"] = list(forward)
        report["obstacle_clearance"] = validate_carton_clearance()
        timeline.play()
        await wait_for_state("idle", "measured_release")
        # Multiple callbacks may run together before the first slow render
        # interval is observable. Keep the hand open through a full wall second
        # so timing can settle without authorizing a held point after a stall.
        release_until = time.perf_counter() + 1.0
        while time.perf_counter() < release_until:
            await advance("measured_release")
        await wait_for_state("idle", "measured_release")
        if options.get("turn_release_only"):
            await validate_turn_deadman()
            await settle("final_balance", 3.0)
            check_health()
            validate_camera_height()
            report["passed"] = True
            return
        if object_root:
            # Re-measure after the open-hand settling period; never teleport an
            # object or assume its initial USD pose survived physical settling.
            target, forward = configure_object_target(object_root)
            report["initial_target"] = list(target)
            report["initial_target_direction"] = list(forward)
        start = snapshot("destination_approach")
        current["hand"] = "point"
        await wait_for_state("active", "point_dwell", wall_timeout=60.0 if options.get("allow_route") else 10.0)
        status = ex.gesture_locomotion_status()
        report["planned_route_at_activation"] = list(status.get("route_waypoints", ()))
        report["route_queries_at_activation"] = status.get("route_queries", 0)
        report["route_cells_at_activation"] = status.get("route_expanded", 0)
        validate_locked_target("destination_approach")
        # Ensure the independent visual update has run after acquisition.
        draw_until = time.perf_counter() + 0.08
        while time.perf_counter() < draw_until:
            await advance("point_dwell")
        report["blue_visuals"] = validate_visuals()
        report["phase"] = "destination_approach"
        initial_distance = math.hypot(start["position"][0] - target[0], start["position"][1] - target[1])
        report["initial_destination_distance_m"] = initial_distance
        travel = max(0.0, initial_distance - ex._locomotion_gesture.arrival_radius_m)
        # Budget the whole route at half the minimum requested walking speed,
        # plus six seconds for startup and corrective stepping. Progress checks
        # below still reject stalled or diverging approaches promptly.
        route_points = [start["position"], *report["planned_route_at_activation"]]
        route_length = sum(math.dist(first[:2], last[:2]) for first, last in zip(route_points, route_points[1:]))
        turn_allowance = math.pi / ex._locomotion_gesture.max_requested_yaw_speed * max(1, len(route_points) - 1)
        approach_budget = 2.0 * max(travel, route_length) / ex._locomotion_gesture.min_forward_speed + 6.0 + turn_allowance
        report["approach_budget_sim_seconds"] = approach_budget
        approach_start = ex.g1._walk_time
        deadline_sim = approach_start + approach_budget
        progress_time = approach_start + 2.0
        progress_distance = None
        progress_key = None
        initial_pivot_started = None
        initial_pivot_position = None
        initial_pivot_finished = False
        while ex._gesture_command.state != "arrived":
            require(ex.g1._walk_time < deadline_sim,
                    f"Destination was not reached within {approach_budget:.2f} simulation seconds")
            require(ex._gesture_command.state == "active",
                    "Navigation stopped before arrival: " + str(ex.gesture_locomotion_status()))
            await advance("destination_approach")
            if ex._gesture_command.state == "arrived":
                break
            request = ex._gesture_command.velocity
            pivoting = abs(request[0]) < 1e-8 and abs(request[1]) < 1e-8 and abs(request[2]) > .01
            if not initial_pivot_finished:
                if pivoting:
                    observed, unused_yaw = ex._get_g1_base_pose_for_arms()
                    if initial_pivot_started is None:
                        initial_pivot_started = ex.g1._walk_time
                        initial_pivot_position = tuple(observed)
                    report["initial_pivot"]["samples"] += 1
                    report["initial_pivot"]["no_translation_requested"] &= request[0] == 0 and request[1] == 0
                    report["initial_pivot"]["maximum_planar_drift_m"] = max(
                        report["initial_pivot"]["maximum_planar_drift_m"],
                        math.dist(tuple(observed)[:2], initial_pivot_position[:2]))
                    report["initial_pivot"]["duration_sim_s"] = ex.g1._walk_time - initial_pivot_started
                elif request[0] > 0:
                    initial_pivot_finished = True
            if ex.g1._walk_time >= progress_time:
                measured, measured_yaw = ex._get_g1_base_pose_for_arms()
                waypoint = ex._locomotion_gesture.waypoint
                key = (tuple(waypoint), "turn" if pivoting else "walk")
                heading = math.atan2(waypoint[1] - measured[1], waypoint[0] - measured[0]) - measured_yaw
                measure = (abs(math.atan2(math.sin(heading), math.cos(heading))) if pivoting
                           else math.hypot(measured[0] - waypoint[0], measured[1] - waypoint[1]))
                if progress_distance is not None and key == progress_key:
                    require(progress_distance - measure >= (0.03 if pivoting else 0.01),
                            "Approach did not make waypoint/heading progress over three simulation seconds")
                progress_distance, progress_key = measure, key
                progress_time = ex.g1._walk_time + 3.0
        arrived = snapshot("destination_approach")
        displacement = np.asarray(arrived["position"]) - start["position"]
        progress = float(np.dot(displacement, np.asarray(forward)))
        remaining = math.hypot(arrived["position"][0] - target[0], arrived["position"][1] - target[1])
        require(progress >= 0.25, "Destination walking advanced less than 0.25 meters")
        require(remaining <= 0.35, "Arrival was outside the 0.35-meter acceptance radius")
        require(report["max_abs_requested_yaw_rad_s"] <= ex._locomotion_gesture.max_requested_yaw_speed + 1e-8,
                "Requested yaw exceeded the configured actor cap")
        require(report["initial_pivot"]["no_translation_requested"], "Initial pivot requested translation")
        report["phases"].append({"name": "destination_approach", "start": start, "end": arrived,
                                  "forward_distance_m": progress, "remaining_distance_m": remaining})
        flush(True)
        stopped = await settle("arrival_stop", 3.0)
        require(math.hypot(*stopped["velocity"][:2]) < 0.1, "Speed remained above 0.1 m/s after arrival stop")

        if options.get("approach_only", False):
            # Arrival already stopped the actor. This checks release/rearming
            # without claiming a moving deadman test or inventing a clear exit.
            current["hand"] = "open"
            await wait_for_state("idle", "approach_open_hand_release")
            released = snapshot("approach_open_hand_release")
            require(released["request"] == [0.0, 0.0, 0.0], "Open hand did not clear the arrived navigation request")
            require(all(value == 0.0 for value in released["filtered"]), "Open hand retained filtered movement")
            report["phases"].append({"name": "approach_open_hand_release", "end": released})
            report["stationary_release_tested"] = True
            flush(True)
        else:
            # Verify both deadman paths on real physics ticks. Each retry must
            # expose an open hand, then complete the normal pointing dwell.
            for loss in ("release", "tracking_loss"):
                current["hand"] = "open"
                await wait_for_state("idle", loss + "_rearm")
                configure_floor_target(0.9)
                current["hand"] = "point"
                await wait_for_state("active", loss + "_dwell")
                validate_locked_target(loss)
                before = snapshot(loss)
                current["hand"] = "open" if loss == "release" else "invalid"
                await wait_for_state("idle" if loss == "release" else "awaiting_release", loss)
                after = snapshot(loss)
                require(after["request"] == [0.0, 0.0, 0.0], loss + " did not clear the navigation request")
                require(all(value == 0.0 for value in after["filtered"]), loss + " did not bypass residual smoothing")
                report["phases"].append({"name": loss, "before": before, "after": after})
                flush(True)
            report["deadman_motion_tested"] = True
        await settle("final_balance", 3.0)
        check_health()
        validate_camera_height()
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        timeline.pause()
        for name, (was_overridden, previous) in saved.items():
            if was_overridden:
                setattr(ex, name, previous)
            else:
                ex.__dict__.pop(name, None)
        report["input_readers_restored"] = all(
            (name in ex.__dict__) == was_overridden
            and (not was_overridden or ex.__dict__[name] is previous)
            for name, (was_overridden, previous) in saved.items()
        )
        if ex is not None:
            ex._reset_gesture_locomotion()
            report["fault"] = ex._articulation_health_fault
            report["callback_errors"] = sorted(set(ex._physics_step_error_logged) - old_errors)
            report["gaze_tracker_preserved"] = ex._eye_gaze_tracker is gaze
        pause_deadline = time.perf_counter() + 2.0
        while timeline.is_playing() and time.perf_counter() < pause_deadline:
            await app.next_update_async()
        report["timeline_paused"] = not timeline.is_playing()
        if not report["timeline_paused"]:
            report["passed"] = False
            report["cleanup_error"] = "Timeline did not acknowledge Pause within two seconds"
        report["phase"] = "complete" if report["passed"] else "failed"
        flush(True)

async def _point_gaze_job_runner(ex, options, job):
    try:
        await asyncio.wait_for(_validate_point_gaze_walking(ex, options, job), timeout=options["timeout"])
    except BaseException as error:
        job["error"] = repr(error)
        job["traceback"] = traceback.format_exc()
    finally:
        job["done"] = True

_job = _POINT_GAZE_WALKING_JOBS[OPTIONS["job_id"]]
_job["task"] = asyncio.ensure_future(_point_gaze_job_runner(EX, OPTIONS, _job))
print("POINT_GAZE_WALKING_JOB " + json.dumps({"job_id": OPTIONS["job_id"], "done": False}))
"""

START_GUARD = r"""
import asyncio
import json
import traceback

_POINT_GAZE_WALKING_JOBS = globals().setdefault("_POINT_GAZE_WALKING_JOBS", {})
for _registry in (_POINT_GAZE_WALKING_JOBS, globals().get("_GESTURE_WALKING_LIVE_JOBS", {})):
    for _active_id, _active_job in _registry.items():
        _task = _active_job.get("task")
        if _task is not None and not _task.done():
            raise RuntimeError("A walking validation is already running: " + _active_id)
if OPTIONS["job_id"] in _POINT_GAZE_WALKING_JOBS:
    raise RuntimeError("Validation job ID already exists")
for _old_id in list(_POINT_GAZE_WALKING_JOBS)[:-7]:
    del _POINT_GAZE_WALKING_JOBS[_old_id]
_POINT_GAZE_WALKING_JOBS[OPTIONS["job_id"]] = {
    "job_id": OPTIONS["job_id"], "done": False, "report": None, "error": None,
}
"""

POLL_CODE = r"""
import json

_jobs = globals().get("_POINT_GAZE_WALKING_JOBS", {})
if JOB_ID not in _jobs:
    raise RuntimeError("Pointing-only validation job not found: " + JOB_ID)
_job = _jobs[JOB_ID]
_task = _job.get("task")
if _task is not None and _task.done():
    if _task.cancelled():
        _job["error"] = _job.get("error") or "Validation task cancelled"
    else:
        _error = _task.exception()
        if _error is not None:
            _job["error"] = _job.get("error") or repr(_error)
    _job["done"] = True
print("POINT_GAZE_WALKING_JOB " + json.dumps({key: value for key, value in _job.items() if key != "task"}))
"""


def request_job(code: str, host: str, port: int, timeout: float) -> tuple[int, dict | None]:
    """Make one short request through the existing Kit Python Server transport."""
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
    """Start or resume one bounded live replay and persist its final report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="Maximum wall seconds; default 600 accommodates slow VR rendering"
    )
    parser.add_argument(
        "--distance", type=float, default=0.9, help="First destination distance in meters, from 0.5 through 50.0"
    )
    parser.add_argument(
        "--bearing",
        type=float,
        default=0.0,
        help="First destination angle relative to initial robot heading in degrees, positive left, from -180 through 180",
    )
    parser.add_argument("--object-root", help="Registered absolute USD object prim path for the first approach only")
    parser.add_argument(
        "--allow-route",
        action="store_true",
        help="Allow a blocked first floor corridor to use collision-checked factory routing; keep other fixtures direct",
    )
    parser.add_argument(
        "--approach-only",
        action="store_true",
        help="Check approach/stop/open-hand release/balance; omit the separate moving release and tracking-loss fixtures",
    )
    parser.add_argument(
        "--turn-release-only",
        action="store_true",
        help="Check +90-degree left turn/release and -90-degree right turn/tracking loss, at --distance 0.9 or 1.5",
    )
    parser.add_argument("--output", type=Path, default=ROOT / ".webcam/point-only-walking-live.json")
    parser.add_argument("--resume", metavar="JOB_ID", help="Reconnect to an existing job without starting another")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("--timeout must be finite and positive")
    if not math.isfinite(args.distance) or not 0.5 <= args.distance <= 50.0:
        parser.error("--distance must be finite and between 0.5 and 50.0 meters")
    if not math.isfinite(args.bearing) or not -180.0 <= args.bearing <= 180.0:
        parser.error("--bearing must be finite and between -180 and 180 degrees")
    if args.turn_release_only:
        if args.approach_only or args.allow_route or args.object_root or args.bearing != 0:
            parser.error(
                "--turn-release-only cannot combine with --approach-only, --allow-route, --object-root or --bearing"
            )
        if args.distance not in (0.9, 1.5):
            parser.error("--turn-release-only requires --distance 0.9 or 1.5")
    if args.object_root is not None and not re.fullmatch(
        r"/(?:[A-Za-z_][A-Za-z_0-9]*)(?:/[A-Za-z_][A-Za-z_0-9]*)*", args.object_root
    ):
        parser.error("--object-root must be an absolute USD prim path without property, variant, or parent segments")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("--output must be inside this repository")
    try:
        job_id = str(uuid.UUID(args.resume)) if args.resume else str(uuid.uuid4())
    except ValueError:
        parser.error("--resume requires the UUID printed when the job started")
    if not args.resume:
        options = {
            "job_id": job_id,
            "timeout": args.timeout,
            "distance": args.distance,
            "bearing": args.bearing,
            "object_root": args.object_root,
            "approach_only": args.approach_only,
            "allow_route": args.allow_route,
            "turn_release_only": args.turn_release_only,
            "progress_path": str(output.with_suffix(".progress.json")),
        }
        status, reply = request_job(
            "OPTIONS = " + repr(options) + "\n" + START_GUARD + LIVE_CODE, args.host, args.port, 10
        )
        if status:
            print(f"Start response failed. If Kit accepted the job, recover with --resume {job_id}.")
            return status
    print(f"Pointing-only validation job {job_id}; polling {args.host}:{args.port}", flush=True)
    deadline = time.monotonic() + args.timeout + 15.0
    poll = "JOB_ID = " + repr(job_id) + "\n" + POLL_CODE
    last_phase = None
    while time.monotonic() < deadline:
        status, reply = request_job(poll, args.host, args.port, min(10.0, max(0.1, deadline - time.monotonic())))
        if status == 1:
            print(f"Kit rejected job {job_id}; verify its UUID and server port.")
            return 1
        if not status:
            if reply.get("job_id") != job_id:
                raise RuntimeError("Kit returned a different validation job ID")
            report = reply.get("report") or {}
            if report.get("phase") != last_phase:
                last_phase = report.get("phase")
                print(f"Validation phase: {last_phase}", flush=True)
            if reply.get("done"):
                report = report or {"passed": False, "error": "Job ended without a report"}
                report["job_id"] = job_id
                if reply.get("error"):
                    report["job_error"] = reply["error"]
                if reply.get("traceback"):
                    report["job_traceback"] = reply["traceback"]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(REPORT_PREFIX + json.dumps(report, sort_keys=True))
                return 0 if report.get("passed") and not reply.get("error") else 1
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    print(f"Report wait expired. Kit owns replay cleanup; recover with --resume {job_id}.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
