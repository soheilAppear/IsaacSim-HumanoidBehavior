#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attempt a physical two-palm shelf-box lift through synthetic optical hands.

Requires a loaded physical walking scene already positioned within measured arm
reach. The robot and box are never teleported, constrained, or attached. Both
optical hands replace the operator's hand input for this bounded replay; neutral
stick input keeps the production leg balance policy active. Eye gaze is untouched.
Every exit restores input/contact readers, zeros motion commands, and pauses.
Success requires independent opposing hand contacts, production bimanual support,
3 cm box lift throughout a one-second hold, and 1.5 cm drop after release.
Failed attempts remain failed reports; this does not validate headset hardware.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import textwrap
import time
import uuid
from pathlib import Path

from kit_exec import DEFAULT_HOST, DEFAULT_PORT, execute
from validate_contact_grasp_live import LIVE_CODE as CONTACT_REPLAY_CODE


def make_config(
    *,
    object_root: str,
    output: Path,
    timeout: float = 300.0,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    side_gap: float = 0.015,
    squeeze: float = 0.018,
    curl: float = 0.5,
    thumb_curl: float = 0.0,
    opposition: float = 0.0,
    lift: float = 0.06,
    hold: float = 1.0,
    clearance_only: bool = False,
) -> dict:
    """Validate small, declared pose adjustments without changing physics gains."""
    if not isinstance(object_root, str) or not re.fullmatch(r"(?:/[A-Za-z_][A-Za-z0-9_]*)+", object_root):
        raise ValueError("Object root must be an absolute USD prim path")
    if not isinstance(clearance_only, bool):
        raise ValueError("Clearance-only mode must be a boolean")
    values = (timeout, *offset, side_gap, squeeze, curl, thumb_curl, opposition, lift, hold)
    if len(offset) != 3 or any(isinstance(value, bool) or not math.isfinite(value) for value in values):
        raise ValueError("All replay parameters must be finite numbers")
    if not 30 <= timeout <= 900 or math.hypot(*offset) > 0.06:
        raise ValueError("Use a 30–900 second timeout and offset within 6 cm")
    if not 0.006 <= side_gap <= 0.04 or not 0 <= squeeze <= 0.02:
        raise ValueError("Side gap must be 6–40 mm; inward travel is limited to 20 mm per hand")
    if not all(0 <= value <= 1 for value in (curl, thumb_curl, opposition)):
        raise ValueError("Curl and opposition endpoints must be normalized")
    if not 0.04 <= lift <= 0.10 or not 1 <= hold <= 2:
        raise ValueError("Lift must be 4–10 cm and hold 1–2 simulation seconds")
    return dict(
        run_id=uuid.uuid4().hex,
        object_root=object_root,
        output=str(output.resolve()),
        timeout=float(timeout),
        offset=tuple(offset),
        side_gap=float(side_gap),
        squeeze=float(squeeze),
        curl=float(curl),
        thumb_curl=float(thumb_curl),
        opposition=float(opposition),
        lift=float(lift),
        hold=float(hold),
        clearance_only=clearance_only,
        min_lift=0.03,
        min_release_drop=0.015,
        fixture="existing_scene_no_teleports",
        profile="experimental_palm_first_side_grasp",
    )


LIVE_CODE = r"""
import asyncio


async def _await_replay_playing(timeline, app, timeout_s=10.0):
    # Wait for queued Play before applying strict replay health checks.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not timeline.is_playing():
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise RuntimeError("Simulation did not start within the replay startup timeout")
        try:
            await asyncio.wait_for(app.next_update_async(), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Simulation did not start within the replay startup timeout") from exc


def _bimanual_approach_ready(hands, max_error_m=.04, max_angle_deg=15.0):
    return all(not hand["scenery_motion_limited"] and hand["palm_error_m"] <= max_error_m
               and hand["palm_angle_error_deg"] <= max_angle_deg for hand in hands.values())


def _bimanual_above_shelf_ready(hands, post_top_z):
    return _bimanual_approach_ready(hands) and all(hand["actual_palm"][2] >= post_top_z + .10 for hand in hands.values())


def _bimanual_front_entry_corridor(starts, ends, obstacles, motion="forward"):
    # Declared, conservative fixture envelope for the aligned open hand/wrist.
    # This is a target-path check, not a proof of the full IK forearm trajectory
    # or of clearance under arbitrary measured tracking/orientation error.
    import math
    lower, upper, margin = (-.12, -.06, -.07), (.16, .06, .09), .002
    result = dict(clear=True, scope="Declared aligned hand/wrist swept AABB versus static rack and measured carton AABBs; not full-arm motion planning",
                  palm_relative_min_m=list(lower), palm_relative_max_m=list(upper), contact_margin_m=margin,
                  hands={}, obstacles=obstacles)
    if not obstacles:
        raise RuntimeError("Front-entry corridor requires measured collision obstacles")
    for side in ("left", "right"):
        start, end = starts[side], ends[side]
        if any(not math.isfinite(float(value)) for value in (*start, *end)):
            raise RuntimeError("Nonfinite front-entry waypoint")
        if abs(start[2] - end[2]) > 1e-6:
            raise RuntimeError("Shelf entry must remain at the nominal contact height")
        if motion == "forward":
            if abs(start[1] - end[1]) > 1e-6 or end[0] <= start[0]:
                raise RuntimeError("Front entry must move forward at fixed lateral position")
        elif motion == "lateral":
            if abs(start[0] - end[0]) > 1e-6:
                raise RuntimeError("Lateral entry must keep its forward position fixed")
        else:
            raise RuntimeError("Unknown bounded shelf-entry motion")
        low = [min(start[axis], end[axis]) + lower[axis] for axis in range(3)]
        high = [max(start[axis], end[axis]) + upper[axis] for axis in range(3)]
        separations = {}
        for obstacle in obstacles:
            bounds = (*obstacle["min"], *obstacle["max"])
            if len(bounds) != 6 or any(not math.isfinite(float(value)) for value in bounds):
                raise RuntimeError("Invalid front-entry collision bounds")
            if any(obstacle["min"][axis] > obstacle["max"][axis] for axis in range(3)):
                raise RuntimeError("Inverted front-entry collision bounds")
            # One separating axis suffices for disjoint AABBs. Positive values
            # are conservative clearance from this enclosing swept volume.
            separation = max(max(low[axis] - obstacle["max"][axis], obstacle["min"][axis] - high[axis]) for axis in range(3))
            separations[obstacle["path"]] = separation
        blocked = [path for path, separation in separations.items() if separation <= margin]
        result["hands"][side] = dict(start_palm=list(start), end_palm=list(end), swept_min_m=low,
                                    swept_max_m=high, separating_axis_clearance_m=separations, blocked_by=blocked)
        result["clear"] = result["clear"] and not blocked
    return result


def _bimanual_open_approach_contact_free(stats):
    # Support eligibility and predictive separation are irrelevant here: an
    # open approach must not load the target, a neighboring carton, or scenery.
    return not stats["max_external_contact_force_n"]


def _bimanual_side_contacts_opposed(measured, outward_normals, palm_only=False):
    # The measured box's local +/-Y faces are the intended squeeze surfaces.
    # Contacts hooked around its back (+X) cannot qualify as side-palm support.
    import math
    candidates = {}
    for side in ("left", "right"):
        candidates[side] = []
        for sample in measured[side]:
            if palm_only and sample.role != "palm":
                continue
            length = math.hypot(*sample.normal_world)
            if not math.isfinite(length) or length <= 1e-9:
                continue
            normal = tuple(value / length for value in sample.normal_world)
            if sum(a * b for a, b in zip(normal, outward_normals[side])) >= .5:
                candidates[side].append((sample.point_world, normal))
    return any(sum(a * b for a, b in zip(left_normal, right_normal)) < -.5
               and math.dist(left_point, right_point) >= .03
               for left_point, left_normal in candidates["left"]
               for right_point, right_normal in candidates["right"])


def _bimanual_contact_dwell_ready(stats, palm_only=False):
    key = "current_opposing_palm_seconds" if palm_only else "current_opposing_side_seconds"
    return stats[key] + 1e-12 >= .06 and not stats["max_scenery_force_n"]


async def _validate_bimanual_box(ex, config):
    import dataclasses
    import json
    import math
    import os
    import time
    import types
    import numpy as np
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()
    started = time.monotonic()
    initial_sim = float(ex._headset_gait_time)
    path = config["object_root"]
    report = dict(run_id=config["run_id"], status="running", phase="preconditions", configuration=config,
                  input="synthetic optical landmarks for both hands; production IK/fingers/PhysX/balance",
                  objects_teleported_or_attached=False, base_constrained=False, gaze_modified=False,
                  restored=False, phases={}, max_root_translation_m=0.0, contact_health_counts={})
    overrides = []
    phase_stats = None
    hands = {}
    root_start = None
    gaze_before = ex._eye_gaze_tracker
    original_contact_read = None
    provenance_path = None
    provenance = None

    def vector(values):
        return Gf.Vec3d(*[float(value) for value in values[:3]])

    def replace(owner, name, value):
        overrides.append((owner, name, name in vars(owner), vars(owner).get(name)))
        setattr(owner, name, value)

    async def save():
        report["wall_seconds"] = time.monotonic() - started
        report["simulated_seconds"] = float(ex._headset_gait_time) - initial_sim
        temporary = config["output"] + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        for attempt in range(25):
            try:
                os.replace(temporary, config["output"])
                return
            except PermissionError:
                if attempt == 24:
                    raise
                await asyncio.sleep(.01)

    def object_pose():
        result = ex._get_object_world_pose(path)
        if result is None:
            raise RuntimeError("Missing measured box pose")
        return vector(result[0]), result[1]

    def root_pose():
        position, orientation = ex.g1.robot.get_world_poses()
        p = np.asarray(ex._first_pose_value(position), dtype=float)
        q = np.asarray(ex._first_pose_value(orientation), dtype=float)
        if not np.isfinite(p).all() or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            raise RuntimeError("Invalid measured root pose")
        return p, q / np.linalg.norm(q)

    def palm_pose(side):
        positions, orientations = ex._get_hand_link_prim(side).get_world_poses()
        wrist = vector(ex._first_pose_value(positions))
        q = ex._first_pose_value(orientations)
        rotation = Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[0]), vector(q[1:])))
        center = wrist + rotation.TransformDir(ex._get_robot_palm_local_center(side))
        return center, ex._get_robot_palm_local_frame(side) * rotation, wrist

    def ensure_free():
        body = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path))
        if not body or body.GetKinematicEnabledAttr().Get():
            raise RuntimeError("Target is not a free dynamic body")
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                bodies = [str(item) for relation in (joint.GetBody0Rel(), joint.GetBody1Rel()) for item in relation.GetTargets()]
                if any(item == path or item.startswith(path + "/") for item in bodies):
                    raise RuntimeError("A joint constrains the target box")
        if ex._grabbed_objects_by_side or ex._grasp_joints_by_side:
            raise RuntimeError("Assisted grasp state is active")

    def health():
        if ex._articulation_health_fault or ex._physics_step_error_logged:
            raise RuntimeError("Production health/callback fault: " + str(ex._articulation_health_fault or ex._physics_step_error_logged))
        if not timeline.is_playing():
            raise RuntimeError("Simulation stopped during replay")
        if ex._eye_gaze_tracker is not gaze_before:
            raise RuntimeError("Gaze tracker changed during replay")
        position, orientation = root_pose()
        displacement = float(np.linalg.norm(position - root_start[0]))
        up = float(1 - 2 * (orientation[1] ** 2 + orientation[2] ** 2))
        linear, angular = ex.g1.robot.get_velocities()
        velocity = vector(ex._first_pose_value(linear))
        spin = vector(ex._first_pose_value(angular))
        report["max_root_translation_m"] = max(report["max_root_translation_m"], displacement)
        report["minimum_pelvis_height_m"] = min(report.get("minimum_pelvis_height_m", 10), float(position[2]))
        report["minimum_up_dot"] = min(report.get("minimum_up_dot", 1), up)
        if not all(math.isfinite(value) for value in (*velocity, *spin, up)):
            raise RuntimeError("Nonfinite root velocity or orientation")
        if displacement > .25 or position[2] < .45 or up < math.cos(math.radians(35)):
            raise RuntimeError("Root drift, height, or tilt guard failed")
        if math.hypot(*velocity[:2]) > .8 or abs(spin[2]) > 1.2:
            raise RuntimeError("Root speed guard failed")
        if any(abs(float(value)) > 1e-6 for value in ex._base_command):
            raise RuntimeError("Walking request became nonzero during grasp replay")
        return dict(position=position.tolist(), orientation_wxyz=orientation.tolist(),
                    linear_velocity=list(velocity), angular_velocity=list(spin), up_dot=up)

    def capture_contacts(dt):
        contacts, status = original_contact_read(dt)
        report["contact_health_counts"][status] = report["contact_health_counts"].get(status, 0) + 1
        if phase_stats is not None:
            measured = {side: [sample for sample in contacts.get(side, ()) if sample.object_path == path
                               and sample.eligible_for_grasp and sample.normal_force_n >= .02
                               and sample.separation_m <= .002] for side in ("left", "right")}
            scenery = {side: [sample for sample in contacts.get(side, ()) if not sample.eligible_for_grasp
                              and not sample.object_path.startswith(ex._g1_prim_path + "/")
                              and sample.normal_force_n >= .02] for side in ("left", "right")}
            opposed = any(Gf.Dot(vector(a.normal_world).GetNormalized(), vector(b.normal_world).GetNormalized()) < -.5
                          and (vector(a.point_world) - vector(b.point_world)).GetLength() >= .03
                          for a in measured["left"] for b in measured["right"])
            position, rotation = object_pose()
            palm_opposed = side_opposed = False
            if status == "ok" and measured["left"] and measured["right"]:
                box_frame = Gf.Matrix4d().SetRotate(rotation)
                outward_normals = {side: box_frame.TransformDir(Gf.Vec3d(0, sign, 0)).GetNormalized()
                                   for side, sign in (("left", 1), ("right", -1))}
                palm_opposed = _bimanual_side_contacts_opposed(measured, outward_normals, palm_only=True)
                side_opposed = _bimanual_side_contacts_opposed(measured, outward_normals)
            elapsed = min(.05, max(0.0, float(dt))) if math.isfinite(float(dt)) else 0.0
            for key, qualifying in (("current_opposing_palm_seconds", palm_opposed),
                                    ("current_opposing_side_seconds", side_opposed)):
                phase_stats[key] = phase_stats.get(key, 0.0) + elapsed if qualifying and elapsed > 0 else 0.0
            phase_stats["physics_samples"] += 1
            phase_stats["opposing_contact_seconds"] += float(dt) if opposed else 0.0
            phase_stats["object_z_min"] = min(phase_stats["object_z_min"], float(position[2]))
            phase_stats["object_z_max"] = max(phase_stats["object_z_max"], float(position[2]))
            for side in ("left", "right"):
                for item in contacts.get(side, ()):
                    if item.object_path == ex._g1_prim_path or item.object_path.startswith(ex._g1_prim_path + "/"):
                        continue
                    if item.normal_force_n >= .02:
                        key = side + ":" + item.object_path
                        phase_stats["max_external_contact_force_n"][key] = max(
                            phase_stats["max_external_contact_force_n"].get(key, 0.0), float(item.normal_force_n))
                total = sum(float(item.normal_force_n) for item in measured[side])
                phase_stats["max_normal_force_n_by_side"][side] = max(phase_stats["max_normal_force_n_by_side"][side], total)
                for item in scenery[side]:
                    key = side + ":" + item.object_path
                    phase_stats["max_scenery_force_n"][key] = max(phase_stats["max_scenery_force_n"].get(key, 0.0), float(item.normal_force_n))
            if len(phase_stats["contact_trace"]) < 2500:
                phase_stats["contact_trace"].append(dict(sim_time=float(ex._headset_gait_time), object_position=list(position),
                    opposed=opposed, palm_opposed=palm_opposed, side_opposed=side_opposed,
                    current_opposing_palm_seconds=phase_stats["current_opposing_palm_seconds"],
                    current_opposing_side_seconds=phase_stats["current_opposing_side_seconds"],
                    contacts={side: [dict(role=item.role, point=list(item.point_world),
                    normal_on_hand=list(item.normal_world), normal_force_n=float(item.normal_force_n),
                    separation_m=float(item.separation_m)) for item in measured[side][:24]] for side in ("left", "right")},
                    scenery_contacts={side: [dict(object_path=item.object_path, role=item.role,
                    point=list(item.point_world), normal_on_hand=list(item.normal_world),
                    normal_force_n=float(item.normal_force_n), separation_m=float(item.separation_m))
                    for item in scenery[side][:24]] for side in ("left", "right")}))
        return contacts, status

__OPTICAL_HAND_FACTORY__

    def reach_preflight(targets, frames):
        report["reach_preflight"] = {}
        measured_joints = np.asarray(ex._first_pose_value(ex.g1.robot.get_dof_positions()), dtype=float)
        for side in ("left", "right"):
            points = []
            for role in ("shoulder", "elbow", "wrist"):
                entry = ex._get_webcam_robot_link(side, role)
                if entry is None:
                    raise RuntimeError("Cannot measure " + side + " " + role)
                points.append(vector(ex._first_pose_value(entry[1].get_world_poses()[0])))
            span = (points[1] - points[0]).GetLength() + (points[2] - points[1]).GetLength()
            offset = (ex._get_robot_palm_local_frame(side).GetInverse() * frames[side]).TransformDir(ex._get_robot_palm_local_center(side))
            initial_offset = (ex._get_robot_palm_local_frame(side).GetInverse() * hands[side].frame).TransformDir(ex._get_robot_palm_local_center(side))
            offsets = {name: initial_offset if name in ("natural_prelift", "withdraw", "spread_in_front") else offset for name in targets}
            measured = {name: dict(palm=list(target[side]), wrist=list(target[side] - offsets[name]),
                                  wrist_distance_m=float((target[side] - offsets[name] - points[0]).GetLength()))
                        for name, target in targets.items()}
            report["reach_preflight"][side] = dict(arm_span_m=float(span), shoulder=list(points[0]), targets=measured)
            elbow_index = ex._g1_arm_dof_indices_by_side[side]["elbow"]
            elbow_limits = ex._get_arm_target_limits(elbow_index)
            report["reach_preflight"][side].update(elbow_position_rad=float(measured_joints[elbow_index]),
                elbow_limits_rad=[float(value) for value in elbow_limits], minimum_waypoint_radius_m=.18,
                minimum_radius_meaning="Conservative fixture guard against tightly folded waypoints, not full IK feasibility")
            if not .2 <= span <= .7 or any(item["wrist_distance_m"] > span + .025 for item in measured.values()):
                raise RuntimeError("Requested " + side + " palm approach/lift lies beyond measured arm span")
            if any(measured[name]["wrist_distance_m"] < .18 for name in ("withdraw", "raise_in_front") if name in measured):
                raise RuntimeError("Requested " + side + " clearance waypoint folds too close to its shoulder")
            if "spread_in_front" in targets:
                # A reachable endpoint does not guarantee a reachable straight
                # interpolation. Check the closest shoulder distance on each
                # retreat/raise segment analytically. Low withdrawal retains the
                # measured hand orientation; alignment occurs only after spread.
                segments = []
                previous = Gf.Vec3d(hands[side].center) - initial_offset
                sequence = ["natural_prelift", "spread_in_front", "withdraw", "align_in_front", "raise_in_front"]
                if "wide_in_front" in targets:
                    sequence.extend(("wide_in_front", "front_entry", "side_approach"))
                for name in sequence:
                    endpoint = targets[name][side] - offsets[name]
                    delta = endpoint - previous
                    denominator = Gf.Dot(delta, delta)
                    fraction = 0.0 if denominator < 1e-12 else max(0.0, min(1.0, -Gf.Dot(previous - points[0], delta) / denominator))
                    nearest = previous + delta * fraction
                    minimum = float((nearest - points[0]).GetLength())
                    maximum = max(float((previous - points[0]).GetLength()), float((endpoint - points[0]).GetLength()))
                    if name == "align_in_front":
                        # Any rotation of the palm offset fits inside this sphere.
                        # Keep the same radius guard through alignment, not just
                        # at its two endpoint wrist positions.
                        palm_radius = float((targets[name][side] - points[0]).GetLength())
                        minimum = max(0.0, palm_radius - offset.GetLength())
                        maximum = palm_radius + offset.GetLength()
                    segments.append(dict(to=name, minimum_wrist_radius_m=minimum, maximum_wrist_radius_m=maximum))
                    previous = endpoint
                report["reach_preflight"][side]["clearance_segments"] = segments
                if any(item["minimum_wrist_radius_m"] < .18 or item["maximum_wrist_radius_m"] > span + .025 for item in segments):
                    raise RuntimeError("Requested " + side + " retreat/raise path leaves the bounded arm reach band")

    def arm_joint_snapshot():
        # One articulation read of each array per phase, never per object or
        # physics step. Native drive targets distinguish servo lag from IK error.
        measured = np.asarray(ex._first_pose_value(ex.g1.robot.get_dof_positions()), dtype=float)
        commanded = np.asarray(ex._first_pose_value(ex.g1.robot.get_dof_position_targets()), dtype=float)
        velocities = np.asarray(ex._first_pose_value(ex.g1.robot.get_dof_velocities()), dtype=float)
        indices = sorted({index for values in ex._g1_arm_dof_indices_by_side.values() for index in values.values()})
        optional, errors = {}, {}
        for label, method in (("gravity_wrapper_unverified_value", "get_dof_gravity_compensation_forces"),
                              ("max_effort_nm", "get_dof_max_efforts")):
            try:
                values = np.asarray(ex._first_pose_value(getattr(ex.g1.robot, method)(dof_indices=indices)), dtype=float)
                if len(values) != len(indices) or not np.isfinite(values).all():
                    raise RuntimeError("Invalid arm force diagnostic shape or values")
                optional[label] = dict(zip(indices, values))
            except Exception as error:
                optional[label] = {}
                errors[label] = type(error).__name__ + ": " + str(error)
        result = {}
        for side in ("left", "right"):
            result[side] = {}
            for role, index in ex._g1_arm_dof_indices_by_side[side].items():
                value = ex._smoothed_arm_targets.get(index)
                result[side][role] = dict(index=int(index), measured_rad=float(measured[index]),
                    native_target_rad=float(commanded[index]), target_minus_measured_rad=float(commanded[index] - measured[index]),
                    velocity_rad_s=float(velocities[index]), smoothed_target_rad=None if value is None else float(value),
                    limits_rad=[float(limit) for limit in ex._get_arm_target_limits(index)], gravity_joint_model_nm=None)
                result[side][role].update({label: float(values[index]) if index in values else None for label, values in optional.items()})
        result["optional_read_errors"] = errors
        result["gravity_mapping"] = dict(status="unverified_wrapper_values",
            detail="Experimental wrapper slices gravity columns by DOF index without accounting for a floating root; do not interpret selected values as arm torque",
            num_dofs=int(ex.g1.robot.num_dofs))
        try:
            view = ex.g1.robot._physics_articulation_view
            native = view.get_gravity_compensation_forces()
            raw = np.asarray(native.numpy() if hasattr(native, "numpy") else native, dtype=float)
            if raw.ndim != 2 or raw.shape[0] != 1 or not np.isfinite(raw).all():
                raise RuntimeError("Unexpected native gravity force shape or values")
            result["gravity_mapping"].update(native_shape=list(raw.shape), native_values=raw[0].tolist())
            fixed_base = view.shared_metatype.fixed_base
            if not isinstance(fixed_base, (bool, np.bool_)):
                raise RuntimeError("Native gravity mapping requires the measured fixed-base flag")
            offset = 0 if fixed_base else 6
            if raw.shape != (1, int(ex.g1.robot.num_dofs) + offset):
                raise RuntimeError("Native gravity shape contradicts the fixed-base flag/DOF count")
            for side in ("left", "right"):
                for joint in result[side].values():
                    joint["gravity_joint_model_nm"] = float(raw[0, joint["index"] + offset])
            result["gravity_mapping"].update(status="verified_native_layout", fixed_base=bool(fixed_base), joint_offset=offset,
                wrapper_values_status="unverified_wrapper_values", detail="Floating-root force/torque precedes joint-space torques; selected wrapper values remain diagnostic only",
                source="https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html#gravity-compensation-force")
        except Exception as error:
            errors["gravity_native"] = type(error).__name__ + ": " + str(error)
        return result

    async def phase(name, seconds, targets=None, curl=None, thumb_curl=None, opposition=None, require_contact_free=False):
        nonlocal phase_stats
        ensure_free()
        position, _ = object_pose()
        phase_stats = dict(physics_samples=0, opposing_contact_seconds=0.0, supported_samples=0, samples=0,
                           current_opposing_palm_seconds=0.0, current_opposing_side_seconds=0.0,
                           object_z_min=float(position[2]), object_z_max=float(position[2]), contact_trace=[],
                           max_normal_force_n_by_side={"left": 0.0, "right": 0.0}, max_scenery_force_n={},
                           max_external_contact_force_n={})
        report["phases"][name] = phase_stats
        report["phase"] = name
        await save()
        start_sim = float(ex._headset_gait_time)
        last_sim, last_advance = start_sim, time.monotonic()
        initial = {side: (Gf.Vec3d(hand.center), hand.curl, hand.thumb_curl or 0.0, hand.opposition)
                   for side, hand in hands.items()}
        while float(ex._headset_gait_time) - start_sim < seconds:
            progress = min(1.0, (float(ex._headset_gait_time) - start_sim) / max(.01, seconds * .7))
            progress = progress * progress * (3 - 2 * progress)
            for side, hand in hands.items():
                old = initial[side]
                if targets is not None:
                    hand.center = old[0] + (targets[side] - old[0]) * progress
                for attribute, value, before in (("curl", curl, old[1]), ("thumb_curl", thumb_curl, old[2]), ("opposition", opposition, old[3])):
                    if value is not None:
                        setattr(hand, attribute, before + (value - before) * progress)
            await asyncio.sleep(.01)
            phase_stats["root_balance"] = health()
            now_sim = float(ex._headset_gait_time)
            if now_sim > last_sim:
                last_sim, last_advance = now_sim, time.monotonic()
            elif time.monotonic() - last_advance > 15:
                raise RuntimeError("Simulation clock stopped advancing")
            result = ex._bimanual_grasp_result
            phase_stats["samples"] += 1
            phase_stats["supported_samples"] += int(result.contact_supported and result.object_path == path)
            if require_contact_free and not _bimanual_open_approach_contact_free(phase_stats):
                # Break to capture the same measured pose/joint diagnostics as a
                # normal endpoint, then fail before any subsequent grip phase.
                break
        phase_stats["simulated_seconds"] = float(ex._headset_gait_time) - start_sim
        phase_stats["bimanual_result"] = dataclasses.asdict(ex._bimanual_grasp_result)
        phase_stats["arm_joints"] = arm_joint_snapshot()
        phase_stats["hands"] = {}
        for side, hand in hands.items():
            center, frame, wrist = palm_pose(side)
            cosine = (sum(float(frame[row][col]) * float(hand.frame[row][col]) for row in range(3) for col in range(3)) - 1.0) / 2.0
            phase_stats["hands"][side] = dict(actual_palm=list(center), requested_palm=list(hand.center), wrist=list(wrist),
                palm_error_m=float((center - hand.center).GetLength()), actual_frame=[[float(frame[r][c]) for c in range(3)] for r in range(3)],
                requested_frame=[[float(hand.frame[r][c]) for c in range(3)] for r in range(3)],
                palm_angle_error_deg=math.degrees(math.acos(max(-1.0, min(1.0, cosine)))),
                scenery_motion_limited=bool(ex._arm_contact_limited.get(side, False)),
                measured_curls=ex.g1.get_finger_curls(side), applied_curls=dict(ex._applied_finger_curls.get(side, {})))
        position, rotation = object_pose()
        phase_stats["final_box_pose"] = dict(position=list(position), orientation_wxyz=[float(rotation.GetReal()), *list(rotation.GetImaginary())])
        ensure_free()
        await save()
        if require_contact_free and not _bimanual_open_approach_contact_free(phase_stats):
            raise RuntimeError("Unexpected shelf or target contact during open-hand approach; refusing to grip")

    try:
        await save()
        if ex._g1_locomotion != "policy" or ex._grasp_mode != "physical" or ex.g1.walking_policy_name != "decoupled":
            raise RuntimeError("Requires real physical grasping and the decoupled walking/balance policy")
        if not ex._physics_ready or not ex.g1.robot.is_physics_tensor_entity_valid() or not ex.g1.has_finger_control():
            raise RuntimeError("Robot physics and finger drives must already be initialized")
        if path not in ex._spawned_sample_box_paths or ex._contact_reader is None or ex._xr_core is None:
            raise RuntimeError("Requires registered target, live contact reader, and XR input")
        if not hasattr(ex, "_bimanual_grasp_result"):
            raise RuntimeError("Load the build containing the bimanual contact monitor")
        ensure_free()
        timeline.pause()
        root_start = root_pose()
        report["root_start"] = root_start[0].tolist()
        report["scene_fixture"] = getattr(ex, "_grasp_validation_fixture", {"kind": config["fixture"]})
        center, rotation = object_pose()
        report["box_start_position"] = list(center)
        box_rotation = Gf.Matrix4d().SetRotate(rotation)
        frames, targets, wide_targets, squeeze_targets, lift_targets = {}, {}, {}, {}, {}
        for side, sign in (("left", 1), ("right", -1)):
            attribute = stage.GetPrimAtPath(path).GetAttribute("g1:" + side + "PalmContactLocal")
            if not attribute or attribute.Get() is None:
                raise RuntimeError("Target lacks real side-palm contact metadata")
            normal = box_rotation.TransformDir(Gf.Vec3d(0, -sign, 0)).GetNormalized()
            forward = box_rotation.TransformDir(Gf.Vec3d(1, 0, 0)).GetNormalized()
            transverse = Gf.Cross(normal, forward).GetNormalized()
            frames[side] = Gf.Matrix4d(*forward, 0, *transverse, 0, *normal, 0, 0, 0, 0, 1)
            surface = center + box_rotation.TransformDir(vector(attribute.Get())) + vector(config["offset"])
            targets[side] = surface - normal * config["side_gap"]
            wide_targets[side] = surface - normal * .07
            squeeze_targets[side] = targets[side] + normal * config["squeeze"]
            lift_targets[side] = squeeze_targets[side] + Gf.Vec3d(0, 0, config["lift"])
            actual, frame, initial_wrist = palm_pose(side)
            report.setdefault("initial_measured_hands", {})[side] = dict(palm=list(actual), wrist=list(initial_wrist),
                frame=[[float(frame[row][col]) for col in range(3)] for row in range(3)])
            hands[side] = make_hand(side, actual, frame)
        support = stage.GetPrimAtPath(path).GetRelationship("g1:supportSurface").GetTargets()
        if len(support) != 1 or not stage.GetPrimAtPath(support[0]):
            raise RuntimeError("Replay requires one authored shelf support surface")
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            stage.GetPrimAtPath(support[0])).ComputeAlignedRange()
        deck_front_x, deck_top_z = float(bounds.GetMin()[0]), float(bounds.GetMax()[2])
        if not math.isfinite(deck_front_x) or not math.isfinite(deck_top_z) or forward[0] < .98 or abs(forward[2]) > .03:
            raise RuntimeError("Shelf replay requires the front-facing, approximately upright practice carton")
        # Withdraw the whole open hand before rising: a diagonal reach from
        # below the deck hits its underside and should be stopped by production.
        clear_x = deck_front_x - .18
        prelift_targets, withdrawn_targets, spread_targets, raised_targets = {}, {}, {}, {}
        for side, sign in (("left", 1), ("right", -1)):
            shoulder_link = ex._get_webcam_robot_link(side, "shoulder")
            if shoulder_link is None:
                raise RuntimeError("Cannot measure the shoulder for " + side + " shelf-clearance waypoint")
            shoulder = vector(ex._first_pose_value(shoulder_link[1].get_world_poses()[0]))
            # Lift slightly in the measured hand orientation before withdrawing:
            # the unraised wrist drooped into the lower deck in a physical replay.
            # This tentative path is guarded by live contact, not claimed clear
            # from its initial enclosing AABB alone.
            low_z = hands[side].center[2] + .04
            prelift_targets[side] = Gf.Vec3d(hands[side].center[0], hands[side].center[1], low_z)
            outward_y = shoulder[1] + sign * .24
            # Establish lateral reach while still forward, then withdraw at
            # that fixed wide Y. Retreating inward first produced a tightly
            # folded transient and a lower-deck strike in the measured replay.
            current_outward = sign * (hands[side].center[1] - shoulder[1])
            withdrawal_y = shoulder[1] + sign * max(.20, current_outward)
            withdrawn_targets[side] = Gf.Vec3d(clear_x, withdrawal_y, low_z)
            spread_targets[side] = Gf.Vec3d(hands[side].center[0], withdrawal_y, low_z)
            raised_targets[side] = Gf.Vec3d(clear_x, outward_y, wide_targets[side][2])
        if any(target[2] < deck_top_z + .065 for target in raised_targets.values()):
            raise RuntimeError("Target height does not leave room for a side-oriented palm above the deck")
        report["shelf_clearance"] = dict(support_path=str(support[0]), front_x=deck_front_x, top_z=deck_top_z,
                                         withdrawal_palm_x=clear_x, distal_clearance_allowance_m=.18,
                                         natural_frame_prelift_m=.04, prelift_position_tolerance_m=.015,
                                         minimum_withdrawal_outward_from_shoulder_m=.20,
                                         spread_from_shoulder_m=.20, outward_from_shoulder_m=.24,
                                         low_approach_order="natural_prelift_spread_withdraw_align",
                                         low_path_clearance_scope="Contact-guarded tentative path; initial enclosing hand bounds are not a full collision-clearance proof",
                                         desired_frame_alignment_after="withdraw_in_front")
        preflight_targets = dict(natural_prelift=prelift_targets, withdraw=withdrawn_targets, spread_in_front=spread_targets,
                                 align_in_front=withdrawn_targets, raise_in_front=raised_targets, side_approach=wide_targets,
                                 contact=targets, squeezed=squeeze_targets, lift=lift_targets)
        if not config["clearance_only"]:
            rack = stage.GetPrimAtPath(support[0]).GetParent()
            posts = [prim for prim in rack.GetChildren() if prim.GetName().startswith("Post_")]
            if len(posts) != 4:
                raise RuntimeError("Full shelf replay requires all four measured rack posts")
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            outboard_front_targets, outboard_forward_targets = {}, {}
            for side, sign in (("left", 1), ("right", -1)):
                lateral = float(root_start[0][1]) + sign * .34
                outboard_front_targets[side] = Gf.Vec3d(clear_x, lateral, wide_targets[side][2])
                outboard_forward_targets[side] = Gf.Vec3d(wide_targets[side][0], lateral, wide_targets[side][2])
            preflight_targets.update(wide_in_front=outboard_front_targets, front_entry=outboard_forward_targets)
            obstacles = []
            for prim in rack.GetChildren():
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    obstacle_bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    obstacles.append(dict(path=str(prim.GetPath()), source="static_usd_collider",
                                          min=list(obstacle_bounds.GetMin()), max=list(obstacle_bounds.GetMax())))
            if len(obstacles) != 6:
                raise RuntimeError("Front-entry corridor requires the two decks and four colliding posts")
            # Dynamic carton transforms come from the physics reader, not USD's
            # potentially stale authored transform. This runs only at preflight.
            shelf_index = stage.GetPrimAtPath(path).GetAttribute("g1:shelfIndex").Get()
            for object_path in ex._spawned_sample_box_paths:
                prim = stage.GetPrimAtPath(object_path)
                if prim.GetAttribute("g1:shelfIndex").Get() != shelf_index:
                    continue
                dimensions = prim.GetAttribute("g1:packageDimensions").Get()
                measured_pose = ex._get_object_world_pose(object_path)
                if dimensions is None or measured_pose is None:
                    raise RuntimeError("Cannot measure neighboring carton collision bounds")
                position = vector(measured_pose[0])
                orientation = Gf.Matrix4d().SetRotate(measured_pose[1])
                extent = Gf.Vec3d(*(sum(abs(orientation[row][column]) * dimensions[row] * .5 for row in range(3)) for column in range(3)))
                obstacles.append(dict(path=object_path, source="measured_dynamic_carton", min=list(position-extent), max=list(position+extent)))
            corridors = {
                "wide_in_front": _bimanual_front_entry_corridor(raised_targets, outboard_front_targets, obstacles, "lateral"),
                "front_entry": _bimanual_front_entry_corridor(outboard_front_targets, outboard_forward_targets, obstacles),
                "side_approach": _bimanual_front_entry_corridor(outboard_forward_targets, wide_targets, obstacles, "lateral"),
            }
            report["front_entry_corridors"] = corridors
            if any(not corridor["clear"] for corridor in corridors.values()):
                raise RuntimeError("Open-hand front-entry planning envelope intersects rack or carton; refusing replay")
            report["shelf_clearance"].update(approach_path="nominal_height_wide_forward_then_narrow",
                                             forward_outboard_from_initial_root_y_m=.34,
                                             forward_wide_palms={side: list(target) for side, target in outboard_forward_targets.items()})
        reach_preflight(preflight_targets, frames)
        report["input_preflight"] = {}
        for side, hand in hands.items():
            observations = []
            for endpoint in ((0, 0, 0), (config["curl"], config["thumb_curl"], config["opposition"]), (0, 0, 0)):
                hand.curl, hand.thumb_curl, hand.opposition = endpoint
                parsed = ex._get_hand_tracking_finger_curls(hand)
                expected = {role: endpoint[0] for role in ex._finger_roles}
                expected.update(thumb=endpoint[1], thumb_yaw=endpoint[2])
                if parsed is None or any(abs(parsed.get(role, -1) - value) > 1e-5 for role, value in expected.items()):
                    raise RuntimeError("Optical landmark round trip failed for " + side)
                observations.append(parsed)
            report["input_preflight"][side] = observations
            ex._deactivate_hand(side)
        reader = ex._get_xr_input_device
        replace(ex, "_get_xr_input_device", lambda handle: hands.get(handle.rsplit("/", 1)[-1]) if handle in ("/user/hand/left", "/user/hand/right") else reader(handle))
        replace(ex, "_read_navigation_hand", lambda: (None, None, None, None))
        replace(ex, "_read_tracked_controller_stick", lambda device: (True, 0.0, 0.0))
        original_contact_read = ex._contact_reader.read
        replace(ex._contact_reader, "read", capture_contacts)
        ex._reset_gesture_locomotion()
        for name in ("_base_command", "_keyboard_command", "_controller_command"):
            getattr(ex, name)[:] = 0.0
        session = getattr(ex, "_behavioral_session_dir", None)
        if session:
            provenance_path = os.path.join(str(session), "validation_" + config["run_id"] + ".json")
            provenance = dict(source="synthetic_bimanual_hand_replay", run_id=config["run_id"],
                              start_unix_seconds=time.time(), start_simulation_time=float(ex._headset_gait_time),
                              scope="Only this interval; other session rows may be real operator input")
            with open(provenance_path, "w", encoding="utf-8") as handle:
                json.dump(provenance, handle, indent=2)
            report["recording_provenance"] = provenance_path
        timeline.play()
        report["phase"] = "starting_simulation"
        await save()
        await _await_replay_playing(timeline, omni.kit.app.get_app())
        health()
        await phase("initialize_open", .6, curl=0, thumb_curl=0, opposition=0, require_contact_free=True)
        await phase("natural_prelift", 1.5, prelift_targets, require_contact_free=True)
        await phase("natural_prelift_settle", 1.0, require_contact_free=True)
        if not _bimanual_approach_ready(report["phases"]["natural_prelift_settle"]["hands"], max_error_m=.015):
            raise RuntimeError("Natural-frame prelift did not settle within 15 mm before withdrawal")
        await phase("spread_in_front", 2.0, spread_targets, require_contact_free=True)
        if any(item["palm_error_m"] > .05 for item in report["phases"]["spread_in_front"]["hands"].values()):
            raise RuntimeError("Palms did not establish reachable lateral clearance before raising")
        await phase("spread_in_front_settle", 1.0, require_contact_free=True)
        if not _bimanual_approach_ready(report["phases"]["spread_in_front_settle"]["hands"]):
            raise RuntimeError("Natural-frame spread did not settle before wide withdrawal")
        await phase("withdraw_in_front", 3.0, withdrawn_targets, require_contact_free=True)
        if any(item["palm_error_m"] > .05 for item in report["phases"]["withdraw_in_front"]["hands"].values()):
            raise RuntimeError("Palms did not withdraw clear of the shelf before raising")
        await phase("withdraw_in_front_settle", 1.0, require_contact_free=True)
        if not _bimanual_approach_ready(report["phases"]["withdraw_in_front_settle"]["hands"]):
            raise RuntimeError("Natural-frame withdrawal did not settle before palm alignment")
        for side in hands:
            hands[side].frame = frames[side]
        await phase("align_in_front", 2.0, require_contact_free=True)
        if not _bimanual_approach_ready(report["phases"]["align_in_front"]["hands"]):
            raise RuntimeError("Open palms did not align after low outward spreading")
        await phase("raise_in_front", 1.5, raised_targets, require_contact_free=True)
        if any(item["palm_error_m"] > .05 for item in report["phases"]["raise_in_front"]["hands"].values()):
            raise RuntimeError("Palms did not reach the clear approach height before advancing")
        if config["clearance_only"]:
            await phase("clearance_settle", 1.0, require_contact_free=True)
            report.update(status="diagnostic_complete", validation_scope="clearance_only_no_grasp_attempt",
                          grasp_validated=False)
            return
        await phase("raise_in_front_settle", 2.0, require_contact_free=True)
        if not _bimanual_approach_ready(report["phases"]["raise_in_front_settle"]["hands"]):
            raise RuntimeError("Palms did not settle within 4 cm and 15 degrees in front of the shelf")
        await phase("wide_in_front", 1.5, outboard_front_targets, require_contact_free=True)
        await phase("wide_in_front_settle", 1.0, require_contact_free=True)
        if not _bimanual_approach_ready(report["phases"]["wide_in_front_settle"]["hands"]):
            raise RuntimeError("Wide nominal-height palms did not settle before advancing")
        # Advance with the hands outboard, then narrow only at the forward pose.
        # This follows the successful free-space ordering and avoids the bound
        # shoulder/wrist branch caused by narrowing directly above the shoulders.
        await phase("front_entry", 3.0, outboard_forward_targets, require_contact_free=True)
        if not _bimanual_approach_ready(report["phases"]["front_entry"]["hands"], max_error_m=.08, max_angle_deg=25.0):
            raise RuntimeError("Open-hand approach is blocked or misaligned; refusing to curl or squeeze into the box")
        await phase("front_entry_settle", 1.0, require_contact_free=True)
        if not _bimanual_approach_ready(report["phases"]["front_entry_settle"]["hands"]):
            raise RuntimeError("Forward wide palms did not settle before narrowing")
        await phase("side_approach", 2.0, wide_targets, require_contact_free=True)
        await phase("approach_settle", .75, require_contact_free=True)
        if not _bimanual_approach_ready(report["phases"]["approach_settle"]["hands"]):
            raise RuntimeError("Open palms did not settle within 4 cm and 15 degrees of the side approach before closure")
        # Establish palm contact before flexing fingers, otherwise the long
        # little fingers can hook the back face while the palms remain separated.
        await phase("touch_sides", 1.5, targets, curl=0, thumb_curl=0, opposition=0)
        if any(item["palm_error_m"] > .08 for item in report["phases"]["touch_sides"]["hands"].values()):
            raise RuntimeError("Measured palms missed requested side contact positions by more than 8 cm")
        await phase("close_together", 1.0, squeeze_targets)
        await phase("palm_squeeze_settle", 1.0)
        if not _bimanual_contact_dwell_ready(report["phases"]["palm_squeeze_settle"], palm_only=True):
            raise RuntimeError("Opposing side PALM contacts did not persist for the latest 60 ms; refusing finger closure")
        # Targets=None holds both requested palm positions fixed during closure.
        await phase("curl_after_palm_contact", 1.5, curl=config["curl"], thumb_curl=config["thumb_curl"], opposition=config["opposition"])
        await phase("curl_settle", 1.0)
        if not _bimanual_contact_dwell_ready(report["phases"]["curl_settle"]):
            raise RuntimeError("Current opposing two-hand side contact was lost after curling; refusing lift")
        await phase("lift", 1.0, lift_targets)
        await phase("hold", config["hold"])
        held = report["phases"]["hold"]
        release_targets = {side: lift_targets[side] + (wide_targets[side] - squeeze_targets[side]) for side in hands}
        await phase("release", 1.2, release_targets, curl=0, thumb_curl=0, opposition=0)
        report["minimum_hold_lift_m"] = held["object_z_min"] - report["box_start_position"][2]
        report["release_drop_m"] = held["object_z_min"] - report["phases"]["release"]["object_z_min"]
        report["checks"] = dict(actual_opposing_contacts=held["opposing_contact_seconds"] >= config["hold"] * .8,
            production_bimanual_support=held["supported_samples"] >= max(1, held["samples"] * .8),
            box_lifted_and_held=report["minimum_hold_lift_m"] >= config["min_lift"],
            box_dropped_after_release=report["release_drop_m"] >= config["min_release_drop"],
            hold_duration=held["simulated_seconds"] >= config["hold"])
        report["status"] = "passed" if all(report["checks"].values()) else "failed"
    except asyncio.CancelledError:
        report["status"] = "timed_out"
        report["error"] = "Bounded replay deadline reached"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
    finally:
        async def cleanup():
            errors = []
            try:
                timeline.pause()
            except Exception as error:
                errors.append("pause: " + str(error))
            for owner, name, had_own, value in reversed(overrides):
                try:
                    setattr(owner, name, value) if had_own else vars(owner).pop(name, None)
                except Exception as error:
                    errors.append("restore " + name + ": " + str(error))
            for side in hands:
                try:
                    ex._deactivate_hand(side)
                except Exception as error:
                    errors.append("deactivate " + side + ": " + str(error))
            try:
                ex._reset_gesture_locomotion()
                for name in ("_base_command", "_keyboard_command", "_controller_command"):
                    getattr(ex, name)[:] = 0.0
                deadline = time.monotonic() + 5
                while timeline.is_playing() and time.monotonic() < deadline:
                    await asyncio.wait_for(omni.kit.app.get_app().next_update_async(), max(.01, deadline - time.monotonic()))
            except Exception as error:
                errors.append("pause/command cleanup: " + str(error))
            report["restore_errors"] = errors
            report["playing_at_end"] = timeline.is_playing()
            report["restored"] = not errors
            report["callback_errors"] = sorted(str(value) for value in ex._physics_step_error_logged)
            report["articulation_health_fault"] = ex._articulation_health_fault
            if errors or report["playing_at_end"] or report["callback_errors"] or report["articulation_health_fault"]:
                report["status"] = "failed"
            if provenance is not None:
                provenance.update(end_unix_seconds=time.time(), end_simulation_time=float(ex._headset_gait_time), status=report["status"])
                try:
                    with open(provenance_path, "w", encoding="utf-8") as handle:
                        json.dump(provenance, handle, indent=2)
                except OSError as error:
                    report["provenance_write_error"] = str(error)
            await save()
        cleanup_task = asyncio.create_task(cleanup())
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                continue
        cleanup_task.result()


_bimanual_config = __CONFIG__
if "CONTACT_GRASP_VALIDATION_TASK" in globals() and not CONTACT_GRASP_VALIDATION_TASK.done():
    raise RuntimeError("A grasp replay is already running")
CONTACT_GRASP_VALIDATION_TASK = asyncio.ensure_future(asyncio.wait_for(
    _validate_bimanual_box(EX, _bimanual_config), timeout=_bimanual_config["timeout"] - 8))
print("Started bimanual box replay: " + _bimanual_config["output"])
"""


def build_live_code(config: dict) -> str:
    """Reuse the established optical landmark generator, with a closure per hand."""
    tree = ast.parse(CONTACT_REPLAY_CODE.replace("__CONFIG__", "{}"))
    hand = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "OpticalHand")
    factory = "    def make_hand(side, center, frame):\n" + textwrap.indent(ast.unparse(hand), "        ")
    factory += "\n        return OpticalHand(center, frame)\n"
    return LIVE_CODE.replace("__OPTICAL_HAND_FACTORY__", factory).replace("__CONFIG__", repr(config))


def main() -> int:
    """Start one asynchronous bounded replay and read its journal until completion."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--object-root", default="/World/G1_SampleBoxes/Box_20")
    parser.add_argument("--output", type=Path, default=Path(".webcam/bimanual-box-live.json"))
    parser.add_argument(
        "--clearance-only", action="store_true", help="Record free-space arm diagnostics and stop before box approach"
    )
    parser.add_argument("--offset", nargs=3, type=float, default=(0, 0, 0), metavar=("X", "Y", "Z"))
    for name, default in (
        ("timeout", 300),
        ("side-gap", 0.015),
        ("squeeze", 0.018),
        ("curl", 0.5),
        ("thumb-curl", 0.0),
        ("opposition", 0.0),
        ("lift", 0.06),
        ("hold", 1),
    ):
        parser.add_argument("--" + name, type=float, default=default)
    args = parser.parse_args()
    try:
        config = make_config(**{name: value for name, value in vars(args).items() if name not in ("host", "port")})
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = execute(build_live_code(config), args.host, args.port, min(30, args.timeout))
    if status:
        return status
    previous_phase = None
    while time.monotonic() - started < args.timeout:
        try:
            report = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
        if report.get("run_id") == config["run_id"]:
            if report.get("phase") != previous_phase:
                previous_phase = report.get("phase")
                print("Phase: " + str(previous_phase), flush=True)
            if report.get("status") != "running":
                print(json.dumps(report, indent=2))
                return 0 if report["status"] in ("passed", "diagnostic_complete") else 1
        time.sleep(0.2)
    print("Host deadline reached; check server cleanup in " + str(output))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
