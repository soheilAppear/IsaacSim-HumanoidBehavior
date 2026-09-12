#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a bounded hand/table contact test in an initialized Humanoid scene.

The input is synthetic optical landmarks, parsed by the production hand tracker.
Production IK, finger drives, contacts and PhysX remain active. The hand approaches
an empty area of the front table, lowers gently, presses slightly, holds and
retracts. Every contact read records joint positions/velocities and forces. Unsafe
joint state pauses the timeline immediately. Original inputs are restored and
the scene is left paused on success, failure or cancellation.

This script does not validate a physical headset, modify gaze, move objects,
reload the scene, or change physics settings. Use a fresh initialized scene.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import textwrap
import time
import uuid
from pathlib import Path

from kit_exec import DEFAULT_HOST, DEFAULT_PORT, execute
from validate_contact_grasp_live import LIVE_CODE as GRASP_LIVE_CODE


def optical_hand_source() -> str:
    """Reuse the tested optical landmark generator without executing a replay."""
    tree = ast.parse(GRASP_LIVE_CODE)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "OpticalHand"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one OpticalHand class in the grasp replay")
    return textwrap.indent(ast.unparse(matches[0]), "    ")


LIVE_CODE = r"""
import asyncio


async def _validate_table_contact(ex, config):
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
    app = omni.kit.app.get_app()
    started = time.monotonic()
    initial_sim = float(getattr(ex, "_headset_gait_time", 0.0))
    side = config["side"]
    report = {
        "run_id": config["run_id"], "status": "running", "phase": "preconditions",
        "input": "synthetic optical hand landmarks; real production controls and PhysX",
        "configuration": config, "gaze_modified": False, "scene_reloaded": False,
        "objects_teleported_or_attached": False, "physics_settings_modified": False,
        "phases": {}, "steps": [], "contact_health_counts": {}, "restored": False,
        "peak_abs_q_rad": 0.0, "peak_abs_qd_rad_s": 0.0,
        "max_root_translation_m": 0.0, "max_root_rotation_rad": 0.0,
        "max_waist_deviation_rad": 0.0,
        "scenery_force_max_n": 0.0, "object_force_max_n": 0.0,
    }
    original_reader = original_contact_read = contact_reader = hand = None
    reader_had_override = contact_had_override = False
    root_start = None
    fault = None
    provenance = provenance_path = None

__OPTICAL_HAND__

    async def save():
        report["wall_seconds"] = time.monotonic() - started
        report["simulated_seconds"] = float(getattr(ex, "_headset_gait_time", initial_sim)) - initial_sim
        temporary = config["output"] + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        for attempt in range(25):
            try:
                os.replace(temporary, config["output"])
                return
            except PermissionError:
                if attempt == 24:
                    raise
                await asyncio.sleep(0.01)

    def write_provenance(ended=False):
        nonlocal provenance, provenance_path
        from isaacsim.core.simulation_manager import SimulationManager

        session = getattr(ex, "_behavioral_session_dir", None)
        if not session:
            return
        report["recording_session"] = str(session)
        if provenance is None:
            provenance_path = os.path.join(str(session), "validation_" + config["run_id"] + ".json")
            provenance = {
                "source": "synthetic_table_contact_replay", "run_id": config["run_id"],
                "scope": "Only this interval; other samples may contain real user input",
                "start_unix_seconds": time.time(), "start_simulation_time": float(ex._headset_gait_time),
                "start_physics_step": SimulationManager.get_num_physics_steps(),
            }
        if ended:
            provenance.update(end_unix_seconds=time.time(), end_simulation_time=float(ex._headset_gait_time),
                              end_physics_step=SimulationManager.get_num_physics_steps(), status=report["status"])
        with open(provenance_path, "w", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
        report["recording_provenance"] = provenance_path

    def vector(value):
        return Gf.Vec3d(*[float(component) for component in value[:3]])

    def root_pose():
        position, orientation = ex.g1.robot.get_world_poses()
        p = np.asarray(ex._first_pose_value(position), dtype=float)
        q = np.asarray(ex._first_pose_value(orientation), dtype=float)
        if not np.isfinite(p).all() or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            raise RuntimeError("Invalid measured root pose")
        return p, q / np.linalg.norm(q)

    def palm_pose():
        position, orientation = ex._get_hand_link_prim(side).get_world_poses()
        wrist = vector(ex._first_pose_value(position))
        q = ex._first_pose_value(orientation)
        rotation = Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[0]), vector(q[1:])))
        return (wrist + rotation.TransformDir(ex._get_robot_palm_local_center(side)),
                ex._get_robot_palm_local_frame(side) * rotation)

    def hand_lowest_z():
        # Measure local collision bounds, then use current physics link poses.
        # Authored world transforms may lag tensor state with USD updates disabled.
        # Collision meshes are commonly invisible/proxy-purpose geometry. Visual
        # filtering must not remove the very meshes whose contact height we need.
        from isaacsim.core.experimental.prims import RigidPrim

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
            False,
            True,
        )
        lowest = math.inf
        hand_root = str(ex._g1_prim_path) + "/" + side + "_hand/"
        sensor_paths = [str(path) for path in ex._contact_reader.sensor_paths if str(path).startswith(hand_root)]
        if not sensor_paths:
            raise RuntimeError("No physical hand sensor links available for collision bounds")
        for path in sensor_paths:
            # Use the contact reader's full rigid-body list, including pinky
            # links omitted by the legacy grasp-link name filter.
            link = RigidPrim(paths=path)
            prim = stage.GetPrimAtPath(path)
            position, orientation = link.get_world_poses()
            p = vector(ex._first_pose_value(position))
            q = ex._first_pose_value(orientation)
            rotation = Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[0]), vector(q[1:])))
            for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                bounds = cache.ComputeRelativeBound(child, prim).ComputeAlignedRange()
                if bounds.IsEmpty():
                    continue
                lower, upper = bounds.GetMin(), bounds.GetMax()
                for x in (lower[0], upper[0]):
                    for y in (lower[1], upper[1]):
                        for z in (lower[2], upper[2]):
                            lowest = min(lowest, float((p + rotation.TransformDir(Gf.Vec3d(x, y, z)))[2]))
        if not math.isfinite(lowest):
            raise RuntimeError("Could not measure current hand collision lower extent")
        return lowest

    def stop(reason):
        nonlocal fault
        if fault is None:
            fault = reason
            report["early_stop"] = reason
            report["early_stop_simulation_time"] = float(ex._headset_gait_time)
        timeline.pause()

    def capture_contacts(dt):
        # This wrapper observes exactly the production read, once per physics
        # callback. It returns the original result without editing contacts.
        contacts, status = original_contact_read(dt)
        try:
            q = np.asarray(ex.g1.robot.get_dof_positions().numpy()[0], dtype=float)
            qd = np.asarray(ex.g1.robot.get_dof_velocities().numpy()[0], dtype=float)
            if not np.isfinite(q).all() or not np.isfinite(qd).all():
                stop("Nonfinite joint position or velocity")
                return contacts, status
            names = report["dof_names"]
            qi, vi = int(np.argmax(np.abs(q))), int(np.argmax(np.abs(qd)))
            p, root_q = root_pose()
            translation = float(np.linalg.norm(p - root_start[0]))
            rotation = 2.0 * math.acos(min(1.0, abs(float(np.dot(root_q, root_start[1])))))
            samples = [sample for values in contacts.values() for sample in values]
            scenery = sum(float(sample.normal_force_n) for sample in samples if not sample.eligible_for_grasp)
            objects = sum(float(sample.normal_force_n) for sample in samples if sample.eligible_for_grasp)
            entry = {
                "simulation_time": float(ex._headset_gait_time), "dt": float(dt), "phase": report["phase"],
                "contact_health": status, "q_rad": q.tolist(), "qd_rad_s": qd.tolist(),
                "peak_abs_q_rad": float(abs(q[qi])), "peak_q_joint": names[qi],
                "peak_abs_qd_rad_s": float(abs(qd[vi])), "peak_qd_joint": names[vi],
                "waist_q_rad": {name: float(q[index]) for index, name in enumerate(names) if "waist" in name},
                "root_position": p.tolist(), "root_orientation_wxyz": root_q.tolist(),
                "root_translation_m": translation, "root_rotation_rad": rotation,
                "palm_position": list(palm_pose()[0]), "requested_palm_position": list(hand.center),
                "requested_curls": dict(ex._latest_finger_curls.get(side, {})),
                "previous_applied_curls": dict(ex._applied_finger_curls.get(side, {})),
                "scenery_force_sum_n": scenery, "object_force_sum_n": objects,
                "contacts": [
                    {"side": contact_side, "role": sample.role, "path": sample.object_path,
                     "eligible_for_grasp": sample.eligible_for_grasp, "force_n": float(sample.normal_force_n),
                     "normal_world": list(sample.normal_world), "point_world": list(sample.point_world),
                     "separation_m": float(sample.separation_m)}
                    for contact_side, values in contacts.items() for sample in values
                ],
            }
            waist_delta = max((abs(value - report["initial_waist_q_rad"][name])
                               for name, value in entry["waist_q_rad"].items()), default=0.0)
            entry["max_waist_deviation_rad"] = waist_delta
            report["steps"].append(entry)
            report["contact_health_counts"][status] = report["contact_health_counts"].get(status, 0) + 1
            for field, value in (("peak_abs_q_rad", entry["peak_abs_q_rad"]),
                                 ("peak_abs_qd_rad_s", entry["peak_abs_qd_rad_s"]),
                                 ("max_root_translation_m", translation), ("max_root_rotation_rad", rotation),
                                 ("max_waist_deviation_rad", waist_delta),
                                 ("scenery_force_max_n", scenery), ("object_force_max_n", objects)):
                report[field] = max(report[field], value)
            if abs(q[qi]) > config["max_q"] or abs(qd[vi]) > config["max_qd"]:
                stop(f"Joint safety limit: {names[qi]} q={q[qi]:.6g} rad; {names[vi]} qd={qd[vi]:.6g} rad/s")
            if translation > 0.01 or rotation > 0.05:
                stop("Stationary root displacement exceeded test limit")
            if waist_delta > config["max_waist_delta"]:
                stop(f"Waist deviated {waist_delta:.6g} rad from its initial stationary pose")
            if status != "ok":
                stop("Unhealthy physical contact reader: " + str(status))
        except Exception as error:
            stop(type(error).__name__ + ": " + str(error))
        return contacts, status

    def health():
        if fault is not None:
            raise RuntimeError(fault)
        articulation_fault = getattr(ex, "_articulation_health_fault", None)
        if articulation_fault:
            raise RuntimeError("Production articulation health guard: " + str(articulation_fault))
        if ex._physics_step_error_logged:
            raise RuntimeError("Physics callback errors: " + str(ex._physics_step_error_logged))

    async def phase(name, seconds, target=None, curl=None, thumb_curl=None, opposition=None):
        report["phase"] = name
        start = float(ex._headset_gait_time)
        entry = {"start_simulation_time": start, "first_step_index": len(report["steps"])}
        report["phases"][name] = entry
        origin = Gf.Vec3d(hand.center)
        old = (hand.curl, hand.thumb_curl or 0.0, hand.opposition)
        await save()
        while float(ex._headset_gait_time) - start < seconds:
            health()
            alpha = min(1.0, (float(ex._headset_gait_time) - start) / max(0.01, seconds * 0.85))
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            if target is not None:
                hand.center = origin + (target - origin) * alpha
            if curl is not None:
                hand.curl = old[0] + (curl - old[0]) * alpha
            if thumb_curl is not None:
                hand.thumb_curl = old[1] + (thumb_curl - old[1]) * alpha
            if opposition is not None:
                hand.opposition = old[2] + (opposition - old[2]) * alpha
            await app.next_update_async()
        health()
        entry.update(end_simulation_time=float(ex._headset_gait_time), last_step_index=len(report["steps"]),
                     final_palm_position=list(palm_pose()[0]), requested_palm_position=list(hand.center),
                     measured_curls=ex.g1.get_finger_curls(side))
        await save()

    try:
        await save()
        if ex is None or ex.g1 is None or not ex.g1.robot.is_physics_tensor_entity_valid():
            raise RuntimeError("Load and initialize the Humanoid before running this test")
        if ex._g1_locomotion != "stationary" or ex._grasp_mode != "physical":
            raise RuntimeError("Requires a stationary physical-grasp scene")
        if ex._xr_core is None or ex._contact_reader is None or not ex.g1.has_finger_control():
            raise RuntimeError("XR, fingers and physical contacts must be initialized")
        health()
        if ex._grabbed_objects_by_side or ex._grasp_joints_by_side:
            raise RuntimeError("Assisted grasp state is active")
        timeline.pause()
        report["dof_names"] = list(ex.g1.robot.dof_names)
        initial_q = np.asarray(ex.g1.robot.get_dof_positions().numpy()[0], dtype=float)
        report["initial_waist_q_rad"] = {
            name: float(initial_q[index]) for index, name in enumerate(report["dof_names"]) if "waist" in name
        }
        report["actual_hand_mimic_parameters"] = {
            str(prim.GetPath()): {attr.GetName(): attr.Get() for attr in prim.GetAttributes()
                                 if "physxMimicJoint:" in attr.GetName() and
                                 attr.GetName().rsplit(":", 1)[-1] in ("naturalFrequency", "dampingRatio")}
            for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)
            and str(prim.GetPath()).startswith(str(ex._g1_prim_path) + "/")
            and any("physxMimicJoint:" in attr.GetName() for attr in prim.GetAttributes())
        }
        root_start = root_pose()
        center, frame = palm_pose()
        hand = OpticalHand(center, frame)
        report["input_preflight"] = []
        for curl, thumb, opposition in ((0.0, 0.0, 0.0),
                                       (config["curl"], config["thumb_curl"], config["opposition"]),
                                       (0.0, 0.0, 0.0)):
            hand.curl, hand.thumb_curl, hand.opposition = curl, thumb, opposition
            parsed = ex._get_hand_tracking_finger_curls(hand)
            expected = {role: curl for role in ex._finger_roles}
            expected.update(thumb=thumb, thumb_yaw=opposition)
            report["input_preflight"].append({"expected": expected, "parsed": parsed})
            if parsed is None or any(abs(parsed.get(role, -10) - value) > 1e-5 for role, value in expected.items()):
                raise RuntimeError("Optical landmark parser preflight failed")
        surfaces = ex._get_work_surface_tops()
        if not surfaces:
            raise RuntimeError("Front work table is missing")
        table_z = float(surfaces[0][3])
        report["table_surface"] = list(surfaces[0])
        reader_had_override = "_get_xr_input_device" in vars(ex)
        original_reader = ex._get_xr_input_device
        ex._get_xr_input_device = lambda handle: (
            hand if handle == "/user/hand/" + side else
            None if handle in ("/user/hand/left", "/user/hand/right") else original_reader(handle)
        )
        contact_reader = ex._contact_reader
        contact_had_override = "read" in vars(contact_reader)
        original_contact_read = contact_reader.read
        contact_reader.read = capture_contacts
        write_provenance()
        timeline.play()
        await phase("initialize_open", 0.6)
        # Starting palms can be below tabletop height. Raise in place with the
        # current orientation before reaching forward, otherwise a diagonal
        # approach tests collision with the table's underside instead of a
        # gentle top-surface touch. The requested palm rotation follows the lift.
        clearance = Gf.Vec3d(hand.center[0], hand.center[1], table_z + config["hover"])
        await phase("raise_clearance", 1.0, target=clearance)
        direction = vector(config["finger_direction"]).GetNormalized()
        normal = vector(config["palm_normal"])
        normal -= direction * Gf.Dot(normal, direction)
        normal.Normalize()
        transverse = Gf.Cross(normal, direction).GetNormalized()
        hand.frame = Gf.Matrix4d(*direction, 0.0, *transverse, 0.0, *normal, 0.0, 0.0, 0.0, 0.0, 1.0)
        hover = Gf.Vec3d(*config["target_xy"], table_z + config["hover"])
        await phase("approach_open", 2.5, target=hover)
        await phase("configure_fingers", 0.6, curl=config["curl"],
                    thumb_curl=config["thumb_curl"], opposition=config["opposition"])
        actual_center = palm_pose()[0]
        lowest = predicted_contact_z = None
        contact_z = config["contact_palm_z"]
        if contact_z is None:
            lowest = hand_lowest_z()
            predicted_contact_z = table_z + (float(actual_center[2]) - lowest)
            contact_z = predicted_contact_z
        report["hand_extent_calibration"] = {
            "palm_center": list(actual_center), "lowest_collision_z": lowest,
            "predicted_contact_palm_z": predicted_contact_z, "used_contact_palm_z": contact_z,
            "note": ("Explicit palm-height override; collision-bounds measurement skipped" if lowest is None else
                     "Collision AABB lower bound; actual contacts determine whether table touch occurred"),
        }
        above = Gf.Vec3d(*config["target_xy"], contact_z + 0.015)
        contact = Gf.Vec3d(*config["target_xy"], contact_z)
        press = Gf.Vec3d(*config["target_xy"], contact_z - config["penetration"])
        await phase("lower_to_clearance", 1.8, target=above)
        await phase("touch_table", 1.5, target=contact)
        await phase("slight_press", 1.5, target=press)
        # Loaded opening is a different constraint transient from a static
        # curled-hand hold. Exercise both directions without moving the table.
        for cycle in range(config.get("close_open_cycles", 0)):
            await phase(f"cycle_{cycle}_close", 0.4, curl=0.9, thumb_curl=0.7, opposition=0.7)
            await phase(f"cycle_{cycle}_open", 0.4, curl=0.0, thumb_curl=0.0, opposition=0.0)
        await phase("hold", config["hold"])
        await phase("retract", 1.8, target=hover, curl=0.0, thumb_curl=0.0, opposition=0.0)
        relevant = [entry for entry in report["steps"] if entry["phase"] in ("touch_table", "slight_press", "hold")]
        table_root = str(ex._work_surface_root_path) + "/Surface_00"
        def has_selected_table_contact(entry):
            return any(contact["side"] == side and contact["force_n"] >= 0.02 and
                       (contact["path"] == table_root or contact["path"].startswith(table_root + "/"))
                       for contact in entry["contacts"])

        duration = sum(entry["dt"] for entry in relevant if has_selected_table_contact(entry))
        report["table_contact_duration_s"] = duration
        report["table_contact_observed"] = duration >= 0.05
        retract_samples = [entry for entry in report["steps"] if entry["phase"] == "retract"]
        last_time = retract_samples[-1]["simulation_time"] if retract_samples else 0.0
        retract_tail = [entry for entry in retract_samples if entry["simulation_time"] >= last_time - 0.1]
        report["retract_contact_cleared"] = bool(retract_tail) and not any(
            has_selected_table_contact(entry) for entry in retract_tail
        )
        report["retract_palm_height_error_m"] = abs(float(palm_pose()[0][2]) - float(hover[2]))
        report["retract_height_recovered"] = report["retract_palm_height_error_m"] <= 0.03
        report["status"] = "passed"
        if duration < 0.05:
            report["status"] = "inconclusive"
            report["error"] = "No actual front-table contact observed; trajectory is not a contact stability pass"
        elif not report["retract_contact_cleared"] or not report["retract_height_recovered"]:
            report["status"] = "failed"
            report["error"] = "Hand did not clear the table and recover its hover height on retraction"
    except asyncio.CancelledError:
        report["status"] = "failed"
        report["error"] = "Replay cancelled or reached its wall deadline"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
    finally:
        async def cleanup():
            errors = []
            try:
                timeline.pause()
            except Exception as error:
                errors.append("Pause: " + str(error))
            if original_reader is not None:
                try:
                    if reader_had_override:
                        ex._get_xr_input_device = original_reader
                    else:
                        delattr(ex, "_get_xr_input_device")
                    ex._drop_everything()
                    for hand_side in ("left", "right"):
                        ex._deactivate_hand(hand_side)
                except Exception as error:
                    errors.append("Input cleanup: " + str(error))
            if original_contact_read is not None:
                try:
                    if contact_had_override:
                        contact_reader.read = original_contact_read
                    else:
                        delattr(contact_reader, "read")
                except Exception as error:
                    errors.append("Contact cleanup: " + str(error))
            try:
                deadline = time.monotonic() + 5.0
                for _ in range(4):
                    if not timeline.is_playing():
                        break
                    await asyncio.wait_for(app.next_update_async(), max(0.01, deadline - time.monotonic()))
            except Exception as error:
                errors.append("Pause wait: " + str(error))
            report["restore_errors"] = errors
            report["restored"] = not errors
            report["playing_at_end"] = timeline.is_playing()
            report["callback_errors"] = sorted(str(value) for value in getattr(ex, "_physics_step_error_logged", ()))
            # A production arm guard can latch during the queued pause without
            # adding a callback-error entry. Preserve that final fault as well.
            report["articulation_health_fault"] = getattr(ex, "_articulation_health_fault", None)
            if (errors or report["playing_at_end"] or report["callback_errors"]
                    or report["articulation_health_fault"] or fault is not None):
                report["status"] = "failed"
            if provenance is not None:
                try:
                    write_provenance(ended=True)
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


_table_contact_config = __CONFIG__
for _name in ("TABLE_CONTACT_VALIDATION_TASK", "CONTACT_GRASP_VALIDATION_TASK"):
    if _name in globals() and not globals()[_name].done():
        raise RuntimeError("A humanoid validation is already running: " + _name)
TABLE_CONTACT_VALIDATION_TASK = asyncio.ensure_future(
    asyncio.wait_for(_validate_table_contact(EX, _table_contact_config), timeout=_table_contact_config["timeout"] - 8.0)
)
print("Started table contact validation: " + _table_contact_config["output"], flush=True)
"""


def build_live_code(config: dict) -> str:
    """Embed validated arguments and the shared optical generator in live code."""
    return LIVE_CODE.replace("__OPTICAL_HAND__", optical_hand_source()).replace("__CONFIG__", repr(config))


def main() -> int:
    """Start a bounded server replay and poll its JSON progress artifact."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--target-xy", nargs=2, type=float, default=(0.32, -0.26), metavar=("X", "Y"))
    parser.add_argument("--finger-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    parser.add_argument("--palm-normal", nargs=3, type=float, default=(0.0, 0.0, -1.0))
    parser.add_argument("--curl", type=float, default=0.0)
    parser.add_argument("--thumb-curl", type=float, default=0.0)
    parser.add_argument("--opposition", type=float, default=0.0)
    parser.add_argument(
        "--close-open-cycles",
        type=int,
        default=0,
        help="Repeat loaded finger closing/opening at the table (0..30; 0.8 simulated seconds per cycle)",
    )
    parser.add_argument(
        "--penetration", type=float, default=0.01, help="Downward command beyond measured contact, metres"
    )
    parser.add_argument("--contact-palm-z", type=float, help="Override collision-bound estimate of contact palm height")
    parser.add_argument("--hover", type=float, default=0.15, help="Approach palm height above the table, metres")
    parser.add_argument("--hold", type=float, default=1.5, help="Simulated seconds holding the slight downward command")
    parser.add_argument("--max-q", type=float, default=10.0, help="Immediate pause above absolute joint position, rad")
    parser.add_argument("--max-qd", type=float, default=100.0, help="Immediate pause above absolute joint speed, rad/s")
    parser.add_argument(
        "--max-waist-delta", type=float, default=0.1, help="Immediate pause above waist pose deviation, rad"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "_compat61/table-contact-live.json"
    )
    args = parser.parse_args()
    numeric = [
        args.timeout,
        *args.target_xy,
        *args.finger_direction,
        *args.palm_normal,
        args.curl,
        args.thumb_curl,
        args.opposition,
        args.penetration,
        args.hover,
        args.hold,
        args.max_q,
        args.max_qd,
        args.max_waist_delta,
    ]
    if args.contact_palm_z is not None:
        numeric.append(args.contact_palm_z)
    if not all(math.isfinite(value) for value in numeric):
        parser.error("All numeric arguments must be finite")
    if args.timeout <= 15 or not 0 <= args.penetration <= 0.03 or not 0.1 <= args.hold <= 5:
        parser.error("Require timeout > 15, penetration 0..0.03 m and hold 0.1..5 simulated seconds")
    if not 0.08 <= args.hover <= 0.3 or not 0 < args.max_q <= 10 or not 0 < args.max_qd <= 100:
        parser.error("Require hover 0.08..0.3 m, max-q 0..10 rad and max-qd 0..100 rad/s")
    if not 0 < args.max_waist_delta <= 0.35:
        parser.error("Require max-waist-delta above 0 and at most 0.35 rad")
    if not 0 <= args.close_open_cycles <= 30:
        parser.error("Require close-open-cycles in 0..30")
    if any(not 0 <= value <= 1 for value in (args.curl, args.thumb_curl, args.opposition)):
        parser.error("Finger endpoints must lie in [0, 1]")
    a, b = args.finger_direction, args.palm_normal
    cross = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    if sum(value * value for value in cross) < 1e-10:
        parser.error("Finger direction and palm normal must be nonzero and nonparallel")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(run_id=uuid.uuid4().hex, output=str(args.output))
    started = time.monotonic()
    status = execute(build_live_code(config), args.host, args.port, min(30.0, args.timeout))
    if status:
        return status
    last_phase = None
    while time.monotonic() - started < args.timeout:
        try:
            report = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
        if report.get("run_id") == config["run_id"]:
            if report.get("phase") != last_phase:
                last_phase = report.get("phase")
                print("Phase: " + str(last_phase), flush=True)
            if report.get("status") != "running":
                print(
                    json.dumps(
                        {key: value for key, value in report.items() if key != "steps"}, indent=2, sort_keys=True
                    )
                )
                print(f"Recorded {len(report.get('steps', []))} physics reads in {args.output}")
                return 0 if report["status"] == "passed" else 1
        time.sleep(0.2)
    print("Host deadline elapsed; inspect " + str(args.output) + " for server cleanup and partial results.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
