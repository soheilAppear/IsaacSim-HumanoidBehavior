#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attempt physical cube/cylinder grasps in an already-loaded stationary Humanoid.

This intentionally moves the simulated arm and fingers with synthetic optical
landmarks. Production IK, finger drives, contact sensing and PhysX remain active;
gaze is untouched. No object is teleported or attached, and no scene is reloaded.
Use a fresh physical-mode scene with both objects on the front table. The test
leaves it paused, restores live inputs, and records each phase to a local JSON
file. Results validate this synthetic trajectory, not headset hardware input.

The TCP request starts a bounded background task and returns immediately. The host
polls the JSON artifact, avoiding a long TCP response on slow VR sessions. Right
hand defaults use separate cube/cylinder profiles with the palm facing down and
fingers forward. The cube profile bends and opposes the thumb far enough to place
its contact surface opposite the index finger, rather than pinching oblique cube
corners. These are synthetic operator pose cues, not increased force settings.
Explicit offset, thumb curl and opposition each override only
that endpoint. Left-hand runs require all three custom endpoints. A failed grasp
is retained with measured contacts and poses, never reported as an assisted pass.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from pathlib import Path

from kit_exec import DEFAULT_HOST, DEFAULT_PORT, execute


def choose_profile(
    object_name: str,
    *,
    side: str = "right",
    offset: tuple[float, float, float] | None = None,
    thumb_curl: float | None = None,
    opposition: float | None = None,
    curl: float = 0.75,
) -> dict:
    """Resolve one object's grasp endpoints without changing simulator state.

    Args:
        object_name: Cube or cylinder profile name.
        side: Hand used by the replay. Left-hand endpoints must all be explicit.
        offset: World offset overriding the measured grasp-space target.
        thumb_curl: Thumb flexion overriding this object's profile endpoint.
        opposition: Thumb opposition overriding this object's profile endpoint.
        curl: Flexion endpoint for the four remaining fingers.

    Returns:
        Selected profile, applied overrides, and resolved endpoints.

    Raises:
        ValueError: The profile or endpoints are invalid or left-hand endpoints
            are incomplete.

    Example:

    .. code-block:: python

        print(choose_profile("cube", thumb_curl=0.4))
    """
    defaults = {
        "cube": {"offset": (0.008, 0.013, 0.040), "thumb_curl": 0.45, "opposition": 0.40},
        "cylinder": {"offset": (0.003, 0.015, 0.060), "thumb_curl": 0.55, "opposition": 0.50},
    }
    if object_name not in defaults or side not in ("left", "right"):
        raise ValueError("Choose cube/cylinder and left/right")
    overrides = {"offset": offset, "thumb_curl": thumb_curl, "opposition": opposition}
    if side == "left" and any(value is None for value in overrides.values()):
        raise ValueError("Left-hand runs require explicit --offset, --thumb-curl and --opposition")
    profile = dict(defaults[object_name]) if side == "right" else {}
    profile.update({name: value for name, value in overrides.items() if value is not None})
    profile["offset"] = tuple(float(value) for value in profile["offset"])
    profile["curl"] = float(curl)
    values = (*profile["offset"], profile["curl"], profile["thumb_curl"], profile["opposition"])
    if len(profile["offset"]) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("Grasp offsets require three finite values; all finger endpoints must be finite")
    if any(not 0.0 <= profile[name] <= 1.0 for name in ("curl", "thumb_curl", "opposition")):
        raise ValueError("Curl, thumb-curl and opposition must lie in [0, 1]")
    profile["base_profile"] = f"validated_right_{object_name}" if side == "right" else "custom_left"
    profile["side"] = side
    profile["overrides"] = [name for name, value in overrides.items() if value is not None]
    if curl != 0.75:
        profile["overrides"].append("curl")
    return profile


LIVE_CODE = r'''
import asyncio


async def _validate_contact_grasp(ex, config):
    import json
    import math
    import os
    import time
    import types

    import numpy as np
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdPhysics

    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()
    started = time.monotonic()
    report = {
        "run_id": config["run_id"], "status": "running", "phase": "preconditions",
        "input": "synthetic optical hand landmarks; real production controls and PhysX",
        "gaze_modified": False, "objects_teleported_or_attached": False,
        "scene_reloaded": False, "cases": {}, "contact_health_counts": {},
        "actual_contact_lead_rad": float(getattr(ex, "_finger_contact_lead_rad", 0.0)),
        "scene_fixture": getattr(ex, "_grasp_validation_fixture", {"kind": "existing_loaded_scene"}),
        "max_root_translation_m": 0.0, "max_root_rotation_rad": 0.0,
        "configuration": config, "restored": False,
    }
    def snapshot_hand_mimics():
        # Record authored values independently of any fixture's claimed settings.
        # This catches stale preparation hooks after an extension is reloaded.
        hand_roots = tuple(str(ex._g1_prim_path) + "/" + hand + "_hand/" for hand in ("left", "right"))
        return {
            str(prim.GetPath()): {
                attribute.GetName(): attribute.Get() for attribute in prim.GetAttributes()
                if "physxMimicJoint:" in attribute.GetName()
                and attribute.GetName().rsplit(":", 1)[-1] in ("naturalFrequency", "dampingRatio")
            }
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(hand_roots) and prim.IsA(UsdPhysics.Joint)
            and any("physxMimicJoint:" in attribute.GetName() for attribute in prim.GetAttributes())
        }
    original_reader = None
    contact_reader = None
    original_contact_read = None
    reader_had_override = False
    contact_had_override = False
    root_start = None
    initial_sim = float(getattr(ex, "_headset_gait_time", 0.0))
    phase_stats = None
    active_path = None
    input_hand = None
    provenance = None
    provenance_path = None
    side = config["side"]

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
                "source": "synthetic_hand_replay", "run_id": config["run_id"],
                "scope": "Only this interval; other samples in the recording may contain real user input",
                "start_unix_seconds": time.time(), "start_simulation_time": float(ex._headset_gait_time),
                "start_physics_step": SimulationManager.get_num_physics_steps(),
            }
        if ended:
            provenance.update(end_unix_seconds=time.time(), end_simulation_time=float(ex._headset_gait_time),
                              end_physics_step=SimulationManager.get_num_physics_steps(), status=report["status"])
        with open(provenance_path, "w", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
        report["recording_provenance"] = provenance_path

    async def save():
        report["wall_seconds"] = time.monotonic() - started
        report["simulated_seconds"] = float(getattr(ex, "_headset_gait_time", initial_sim)) - initial_sim
        temporary = config["output"] + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        # Windows readers may briefly hold a non-delete-sharing file handle.
        # Let Kit continue advancing while the host finishes its JSON read.
        for attempt in range(25):
            try:
                os.replace(temporary, config["output"])
                break
            except PermissionError:
                if attempt == 24:
                    raise
                await asyncio.sleep(0.01)

    def vector(value):
        return Gf.Vec3d(*[float(component) for component in value[:3]])

    def root_pose():
        position, orientation = ex.g1.robot.get_world_poses()
        p = np.asarray(ex._first_pose_value(position), dtype=float)
        q = np.asarray(ex._first_pose_value(orientation), dtype=float)
        if not np.isfinite(p).all() or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            raise RuntimeError("Invalid measured robot root pose")
        return p, q / np.linalg.norm(q)

    def object_position(path):
        pose = ex._get_object_world_pose(path)
        if pose is None:
            raise RuntimeError("Missing live object pose: " + path)
        return vector(pose[0])

    def palm_pose():
        position, orientation = ex._get_hand_link_prim(side).get_world_poses()
        wrist = vector(ex._first_pose_value(position))
        q = ex._first_pose_value(orientation)
        rotation = Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[0]), vector(q[1:])))
        local_frame = ex._get_robot_palm_local_frame(side)
        center = wrist + rotation.TransformDir(ex._get_robot_palm_local_center(side))
        return center, local_frame * rotation

    def ensure_objects_free():
        for path in config["paths"]:
            body = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path))
            if not body or body.GetKinematicEnabledAttr().Get():
                raise RuntimeError("Test object must remain a dynamic rigid body: " + path)
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            bodies = [str(path) for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()) for path in rel.GetTargets()]
            if any(body == path or body.startswith(path + "/") for body in bodies for path in config["paths"]):
                raise RuntimeError("A joint constrains a test object: " + str(prim.GetPath()))
        if ex._grabbed_objects_by_side or ex._grasp_joints_by_side:
            raise RuntimeError("Assisted grasp state is active")

    def health():
        articulation_fault = getattr(ex, "_articulation_health_fault", None)
        if articulation_fault:
            raise RuntimeError("Production articulation health guard: " + str(articulation_fault))
        if ex._physics_step_error_logged:
            raise RuntimeError("Physics callback errors: " + str(ex._physics_step_error_logged))
        if root_start is not None:
            position, orientation = root_pose()
            displacement = float(np.linalg.norm(position - root_start[0]))
            rotation = 2.0 * math.acos(min(1.0, abs(float(np.dot(orientation, root_start[1])))))
            report["max_root_translation_m"] = max(report["max_root_translation_m"], displacement)
            report["max_root_rotation_rad"] = max(report["max_root_rotation_rad"], rotation)
            if displacement > 0.001 or rotation > 0.001:
                raise RuntimeError("Stationary robot root moved")
        if phase_stats is not None and active_path is not None:
            position = object_position(active_path)
            phase_stats["object_z_min"] = min(phase_stats["object_z_min"], float(position[2]))
            phase_stats["object_z_max"] = max(phase_stats["object_z_max"], float(position[2]))
            result = ex._physical_grasp_results.get(side)
            phase_stats["sample_count"] += 1
            if result is not None and result.contact_supported and result.object_path == active_path:
                phase_stats["supported_samples"] += 1

    def capture_contacts(dt):
        contacts, status = original_contact_read(dt)
        report["contact_health_counts"][status] = report["contact_health_counts"].get(status, 0) + 1
        if phase_stats is not None:
            if config.get("trace_contact_steps") and report["phase"].rsplit("/", 1)[-1] in ("close", "lift", "hold"):
                controller = ex._physical_grasp_controllers[side]
                result = ex._physical_grasp_results.get(side)
                pose = ex._get_object_world_pose(active_path)
                phase_stats.setdefault("contact_step_trace", []).append({
                    "simulation_time": float(ex._headset_gait_time), "dt": float(dt),
                    "state_timing": "contacts/measured now; anchors/caps precede this step's controller update",
                    "requested": dict(ex._latest_finger_curls.get(side, {})),
                    "measured": ex.g1.get_finger_curls(side),
                    "previous_capped": dict(result.curls) if result is not None else {},
                    "previous_applied": dict(ex._applied_finger_curls.get(side, {})),
                    "previous_anchors": {
                        role: {path: {"limit": anchor.limit, "missing_s": anchor.missing_s}
                               for path, anchor in anchors.items()}
                        for role, anchors in controller._contact_anchors.items()
                    },
                    "object_position": list(pose[0]),
                    "object_orientation_wxyz": [float(pose[1].GetReal()), *list(pose[1].GetImaginary())],
                    "contacts": [
                        {"actor_or_canonical_object_path": sample.object_path,
                         "eligible_for_grasp": sample.eligible_for_grasp, "role": sample.role,
                         "point_world": list(sample.point_world), "normal_on_hand_world": list(sample.normal_world),
                         "normal_force_n": sample.normal_force_n, "separation_m": sample.separation_m}
                        for sample in contacts.get(side, ())
                    ],
                })
            phase_stats["physics_samples"] += 1
            matching = [sample for sample in contacts.get(side, ()) if sample.object_path == active_path]
            phase_stats["contact_points_max"] = max(phase_stats["contact_points_max"], len(matching))
            phase_stats["normal_force_sum_max_n"] = max(
                phase_stats["normal_force_sum_max_n"], sum(sample.normal_force_n for sample in matching)
            )
            if float(ex._headset_gait_time) >= phase_stats["settled_start_simulation_time"]:
                force_sum = sum(sample.normal_force_n for sample in matching)
                phase_stats["settled_force_samples"] += 1
                phase_stats["settled_force_sum_total_n"] += force_sum
                previous_min = phase_stats["settled_force_sum_min_n"]
                phase_stats["settled_force_sum_min_n"] = force_sum if previous_min is None else min(previous_min, force_sum)
                phase_stats["settled_force_sum_mean_n"] = (
                    phase_stats["settled_force_sum_total_n"] / phase_stats["settled_force_samples"]
                )
            if matching:
                phase_stats["contact_samples"] += 1
                phase_stats["roles"] = sorted(set(phase_stats["roles"]) | {sample.role for sample in matching})
                phase_stats["latest_contacts"] = [
                    {"role": sample.role, "point_world": list(sample.point_world),
                     "normal_on_hand_world": list(sample.normal_world), "normal_force_n": sample.normal_force_n,
                     "separation_m": sample.separation_m}
                    for sample in matching[:12]
                ]
                if "first_contacts" not in phase_stats and any(sample.normal_force_n >= 0.02 for sample in matching):
                    phase_stats["first_contacts"] = list(phase_stats["latest_contacts"])
                    phase_stats["first_contact_simulation_time"] = float(ex._headset_gait_time)
                    phase_stats["first_contact_measured_curls"] = ex.g1.get_finger_curls(side)
                    phase_stats["first_contact_applied_curls"] = dict(ex._applied_finger_curls.get(side, {}))
        return contacts, status

    class OpticalHand:
        """Supply a movable rigid palm and independently bent optical landmarks."""

        def __init__(self, center, frame):
            self.center = Gf.Vec3d(center)
            self.frame = Gf.Matrix4d(frame)
            self.curl = 0.0
            self.thumb_curl = None  # Preserve shared curl until an independent endpoint is supplied.
            self.opposition = 0.0

        def positions(self):
            points = {"wrist": Gf.Vec3d(0, 0, 0), "palm": Gf.Vec3d(0, 0.01, 0)}
            lateral = {"index": 0.025, "middle": 0.0, "ring": -0.022, "little": -0.04}
            for role, chain in ex._finger_curl_joint_chains.items():
                if role == "thumb":
                    angle = math.radians(45.0 + 90.0 * self.opposition)
                    direction = Gf.Vec3d(math.cos(angle), math.sin(angle), 0)
                    current = Gf.Vec3d(0.035, 0.015, 0)
                else:
                    direction = Gf.Vec3d(0, 1, 0)
                    current = Gf.Vec3d(lateral[role], 0.02, 0)
                curl = self.thumb_curl if role == "thumb" and self.thumb_curl is not None else self.curl
                bend = curl * math.radians(95.0 if role == "thumb" else 150.0)
                points[chain[0]] = current
                for index, name in enumerate(chain[1:]):
                    angle = bend * index / (len(chain) - 2)
                    current = current + 0.035 * (direction * math.cos(angle) + Gf.Vec3d(0, 0, -math.sin(angle)))
                    points[name] = Gf.Vec3d(current)
            if side == "right":
                points = {name: Gf.Vec3d(-point[0], point[1], point[2]) for name, point in points.items()}
            source_frame = ex._build_palm_frame(side, *(points[name] for name in
                                                      ("wrist", "middle_proximal", "index_proximal", "little_proximal")))
            transform = source_frame.GetInverse() * self.frame
            center = (points["wrist"] + points["middle_proximal"]) * 0.5
            return {name: self.center + transform.TransformDir(point - center) for name, point in points.items()}

        def get_name(self):
            return "/user/hand/" + side

        def get_hand_tracking_data_source(self):
            return "hand"

        def get_pose_names(self):
            return list(self.positions())

        def get_virtual_world_pose_desc(self, name=""):
            point = self.positions().get(name)
            return types.SimpleNamespace(
                pose_matrix=Gf.Matrix4d(1.0).SetTranslate(point if point is not None else Gf.Vec3d(0)),
                validity_flags=2 if point is not None else 0,
            )

        def get_input_names(self):
            return []

        def has_input(self, name):
            return False

        def has_input_gesture(self, name, gesture):
            return False

        def get_input_gesture_value(self, name, gesture):
            return 0.0

    def preflight_profiles(hand):
        """Check every selected endpoint through the real parser before replay."""
        report["input_preflight"] = []
        for path in config["paths"]:
            profile = config["profiles"][path]
            for phase_name, curl, thumb_curl, opposition in (
                ("open", 0.0, 0.0, 0.0),
                ("close", profile["curl"], profile["thumb_curl"], profile["opposition"]),
                ("open_reset", 0.0, 0.0, 0.0),
            ):
                hand.curl, hand.thumb_curl, hand.opposition = curl, thumb_curl, opposition
                parsed = ex._get_hand_tracking_finger_curls(hand)
                report["input_preflight"].append({
                    "path": path, "base_profile": profile["base_profile"], "phase": phase_name,
                    "curl": curl, "thumb_curl": thumb_curl, "opposition": opposition, "parsed": parsed,
                })
                expected = {role: curl for role in ex._finger_roles}
                expected["thumb"] = thumb_curl
                expected["thumb_yaw"] = opposition
                if parsed is None or any(abs(parsed.get(role, -10.0) - value) > 1e-5 for role, value in expected.items()):
                    raise RuntimeError("Synthetic hand preflight failed the optical curl/opposition round trip: " + path)

    async def phase(case, name, seconds, target=None, curl=None, opposition=None, thumb_curl=None):
        nonlocal phase_stats
        ensure_objects_free()
        position = object_position(active_path)
        phase_stats = {
            "sample_count": 0, "supported_samples": 0, "physics_samples": 0, "contact_samples": 0,
            "contact_points_max": 0, "normal_force_sum_max_n": 0.0, "roles": [],
            "object_z_min": float(position[2]), "object_z_max": float(position[2]),
            "settled_start_simulation_time": float(ex._headset_gait_time) + (0.0 if name == "hold" else seconds * 0.65),
            "settled_force_samples": 0, "settled_force_sum_total_n": 0.0,
            "settled_force_sum_min_n": None, "settled_force_sum_mean_n": None,
        }
        case["phases"][name] = phase_stats
        report["phase"] = case["name"] + "/" + name
        await save()
        start_sim = float(ex._headset_gait_time)
        initial = Gf.Vec3d(input_hand.center)
        old_curl, old_opposition = input_hand.curl, input_hand.opposition
        old_thumb_curl = old_curl if input_hand.thumb_curl is None else input_hand.thumb_curl
        if thumb_curl is None:
            thumb_curl = curl
        while float(ex._headset_gait_time) - start_sim < seconds:
            alpha = min(1.0, (float(ex._headset_gait_time) - start_sim) / max(0.01, seconds * 0.65))
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            if target is not None:
                input_hand.center = initial + (target - initial) * alpha
            if curl is not None:
                input_hand.curl = old_curl + (curl - old_curl) * alpha
            if thumb_curl is not None:
                input_hand.thumb_curl = old_thumb_curl + (thumb_curl - old_thumb_curl) * alpha
            if opposition is not None:
                input_hand.opposition = old_opposition + (opposition - old_opposition) * alpha
            await asyncio.sleep(0.02)
            health()
        phase_stats["final_object_position"] = list(object_position(active_path))
        if config.get("trace_contact_steps"):
            names = ex.g1.robot.dof_names
            prefix = "R_" if side == "right" else "L_"
            indices = [index for index, name in enumerate(names) if name.startswith(prefix)]
            stiffness, damping = ex.g1.robot.get_dof_gains(dof_indices=indices)
            phase_stats["hand_joint_readback"] = {
                "names": [names[index] for index in indices],
                "position_rad": ex.g1.robot.get_dof_positions(dof_indices=indices).numpy()[0].tolist(),
                "target_rad": ex.g1.robot.get_dof_position_targets(dof_indices=indices).numpy()[0].tolist(),
                "velocity_rad_s": ex.g1.robot.get_dof_velocities(dof_indices=indices).numpy()[0].tolist(),
                "projected_joint_force_nm": ex.g1.robot.get_dof_projected_joint_forces(dof_indices=indices).numpy()[0].tolist(),
                "stiffness": stiffness.numpy()[0].tolist(), "damping": damping.numpy()[0].tolist(),
            }
        measured_center, measured_frame = palm_pose()
        phase_stats["final_palm_position"] = list(measured_center)
        phase_stats["requested_palm_position"] = list(input_hand.center)
        phase_stats["palm_position_error_m"] = float((measured_center - input_hand.center).GetLength())
        actual_rotation = np.array([[measured_frame[row][col] for col in range(3)] for row in range(3)])
        target_rotation = np.array([[input_hand.frame[row][col] for col in range(3)] for row in range(3)])
        cosine = float((np.trace(actual_rotation @ target_rotation.T) - 1.0) * 0.5)
        phase_stats["palm_orientation_error_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        phase_stats["measured_palm_frame_world"] = actual_rotation.tolist()
        phase_stats["requested_palm_frame_world"] = target_rotation.tolist()
        phase_stats["parsed_input_curls"] = ex._get_hand_tracking_finger_curls(input_hand)
        phase_stats["applied_finger_curls"] = dict(ex._applied_finger_curls.get(side, {}))
        phase_stats["measured_finger_curls"] = ex.g1.get_finger_curls(side)
        phase_stats["simulated_seconds"] = float(ex._headset_gait_time) - start_sim
        ensure_objects_free()
        await save()

    try:
        await save()
        if ex is None or ex.g1 is None or not ex.g1.robot.is_physics_tensor_entity_valid():
            raise RuntimeError("Load the Humanoid and initialize physics before running this test")
        report["actual_hand_mimic_parameters"] = snapshot_hand_mimics()
        if ex._g1_locomotion != "stationary" or ex._grasp_mode != "physical":
            raise RuntimeError("This test requires stationary mode with physical grasping")
        if ex._xr_core is None or ex._contact_reader is None:
            raise RuntimeError("XR input and the physical contact reader must be initialized")
        if not ex.g1.has_finger_control():
            raise RuntimeError("Finger drives are not initialized")
        prepared_paths = getattr(ex, "_physical_hand_mimic_joint_paths", None)
        observed = report["actual_hand_mimic_parameters"]
        expected = {
            "naturalFrequency": getattr(ex, "_finger_mimic_natural_frequency", None),
            "dampingRatio": getattr(ex, "_finger_mimic_damping_ratio", None),
        }
        report["expected_hand_mimic_parameters"] = expected
        expected_valid = (
            all(isinstance(value, (int, float)) and math.isfinite(value) for value in expected.values())
            and expected["naturalFrequency"] > 0.0 and expected["dampingRatio"] >= 1.0
        )
        def coupling_matches(values):
            parameters = {name.rsplit(":", 1)[-1]: value for name, value in values.items()}
            return (len(values) == 2 and set(parameters) == set(expected)
                    and all(isinstance(value, (int, float)) and math.isfinite(value)
                            and math.isclose(value, expected[name], rel_tol=1e-6, abs_tol=0.0)
                            for name, value in parameters.items()))

        # Exact authored values and preparation paths must match this production
        # instance. A fixture label cannot substitute for reading the live USD,
        # and stale hard (0/0) couplings must not pass the compliant-grip replay.
        if (not prepared_paths or len(prepared_paths) != 12 or len(observed) != 12 or set(prepared_paths) != set(observed)
                or not expected_valid or any(not coupling_matches(values) for values in observed.values())):
            raise RuntimeError("Reload the latest Humanoid and initialize physics: all 12 physical hand mimic couplings must match the configured finite compliant parameters (frequency > 0, damping ratio >= 1)")
        report["prepared_hand_mimic_joint_paths"] = list(prepared_paths)
        health()
        ensure_objects_free()
        timeline.pause()
        root_start = root_pose()
        center, initial_frame = palm_pose()
        input_hand = OpticalHand(center, initial_frame)
        # Verify the input generator against the real parser before any control
        # override. A malformed skeleton must never turn an open-hand phase into
        # a closed-fist collision test.
        preflight_profiles(input_hand)
        await save()
        reader_had_override = "_get_xr_input_device" in vars(ex)
        original_reader = ex._get_xr_input_device
        ex._get_xr_input_device = lambda handle: (
            input_hand if handle == "/user/hand/" + side else
            None if handle in ("/user/hand/left", "/user/hand/right") else original_reader(handle)
        )
        contact_reader = ex._contact_reader
        contact_had_override = "read" in vars(contact_reader)
        original_contact_read = contact_reader.read
        contact_reader.read = capture_contacts
        write_provenance()
        timeline.play()

        for path in config["paths"]:
            active_path = path
            profile = config["profiles"][path]
            requested_curls = {role: profile["curl"] for role in ex._finger_roles}
            requested_curls.update(thumb=profile["thumb_curl"], thumb_yaw=profile["opposition"])
            case = {
                "name": "cube" if path.endswith("00") else "cylinder", "path": path, "phases": {},
                "profile": profile, "requested_finger_curls": requested_curls,
            }
            report["cases"][path] = case
            await phase(case, "initialize_open_hand", 0.6, curl=0.0, opposition=0.0, thumb_curl=0.0)
            if any(value > 0.20 for value in ex.g1.get_finger_curls(side).values()):
                raise RuntimeError("Open-hand precondition failed; measured fingers remain closed before reaching")
            start_position = object_position(path)
            case["start_position"] = list(start_position)
            current_center, current_frame = palm_pose()
            letter = "L" if side == "left" else "R"
            landmarks = {}
            for prim in ex._get_grasp_link_prims(side):
                name = str(prim.paths[0]).rsplit("/", 1)[-1]
                if name in (letter + "_thumb_distal", letter + "_index_intermediate"):
                    landmarks[name] = vector(ex._first_pose_value(prim.get_world_poses()[0]))
            if len(landmarks) != 2:
                raise RuntimeError("Cannot measure thumb/index grasp space")
            gap_offset = (sum(landmarks.values(), Gf.Vec3d(0)) * 0.5) - current_center
            gap_local = current_frame.GetInverse().TransformDir(gap_offset)
            direction = Gf.Vec3d(*config["finger_direction"]).GetNormalized()
            normal = Gf.Vec3d(*config["palm_normal"])
            normal -= direction * Gf.Dot(normal, direction)
            normal.Normalize()
            transverse = Gf.Cross(normal, direction).GetNormalized()
            desired_frame = Gf.Matrix4d(
                *direction, 0.0, *transverse, 0.0, *normal, 0.0, 0.0, 0.0, 0.0, 1.0
            )
            grasp_target = start_position - desired_frame.TransformDir(gap_local) + Gf.Vec3d(*profile["offset"])
            case["measured_gap_offset_local"] = list(gap_local)
            case["grasp_palm_target"] = list(grasp_target)
            case["palm_normal_world"] = list(normal)
            case["finger_direction_world"] = list(direction)
            clearance = max(float(start_position[2]) + 0.20, 1.06)
            await phase(case, "raise_clear", 1.7, Gf.Vec3d(current_center[0], current_center[1], clearance))
            input_hand.frame = desired_frame
            await phase(case, "align_above_object", 2.0, Gf.Vec3d(grasp_target[0], grasp_target[1], clearance))
            await phase(case, "approach", 2.0, grasp_target)
            await phase(case, "close", 1.5, curl=profile["curl"], opposition=profile["opposition"], thumb_curl=profile["thumb_curl"])
            await phase(case, "lift", 2.0, grasp_target + Gf.Vec3d(0, 0, config["lift"]))
            await phase(case, "hold", config["hold"])
            held_position = object_position(path)
            case["held_position"] = list(held_position)
            await phase(case, "release", 0.8, curl=0.0, opposition=0.0, thumb_curl=0.0)
            await phase(case, "drop_observation", 1.0)
            final_position = object_position(path)
            hold_stats = case["phases"]["hold"]
            case["lift_m"] = float(held_position[2] - start_position[2])
            case["minimum_hold_lift_m"] = hold_stats["object_z_min"] - float(start_position[2])
            case["release_drop_m"] = float(held_position[2] - final_position[2])
            case["hold_supported_fraction"] = hold_stats["supported_samples"] / max(1, hold_stats["sample_count"])
            case["checks"] = {
                "actual_object_contact": case["phases"]["close"]["contact_samples"] > 0,
                "opposing_contacts_during_hold": case["hold_supported_fraction"] >= 0.5,
                "lifted_and_held": case["minimum_hold_lift_m"] >= config["min_lift"],
                "fell_after_opening": case["release_drop_m"] >= 0.025,
                "contact_released": case["phases"]["drop_observation"]["supported_samples"] == 0,
            }
            case["status"] = "passed" if all(case["checks"].values()) else "failed"
            await phase(case, "retract", 1.0, Gf.Vec3d(current_center[0], current_center[1], clearance))
            await save()
        report["status"] = "passed" if all(case["status"] == "passed" for case in report["cases"].values()) else "failed"
    except asyncio.CancelledError:
        report["status"] = "timed_out"
        report["error"] = "Server deadline elapsed; inspect the last completed phase"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
    finally:
        async def cleanup():
            """Restore inputs and publish the result despite replay cancellation."""
            restore_errors = []
            try:
                timeline.pause()
            except Exception as error:
                restore_errors.append("Timeline pause: " + str(error))
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
                    restore_errors.append("XR input/hand cleanup: " + str(error))
            if original_contact_read is not None:
                try:
                    if contact_had_override:
                        contact_reader.read = original_contact_read
                    else:
                        delattr(contact_reader, "read")
                except Exception as error:
                    restore_errors.append("Contact reader restore: " + str(error))
            report["restored"] = not restore_errors
            report["restore_errors"] = restore_errors
            # Timeline commands are queued. Bound this wait independently from
            # the replay deadline, leaving time for provenance and the JSON save.
            import omni.kit.app
            pause_deadline = time.monotonic() + 5.0
            try:
                for _ in range(4):
                    if not timeline.is_playing():
                        break
                    await asyncio.wait_for(
                        omni.kit.app.get_app().next_update_async(),
                        timeout=max(0.0, pause_deadline - time.monotonic()),
                    )
            except Exception as error:
                report["pause_wait_error"] = type(error).__name__ + ": " + str(error)
            report["playing_at_end"] = timeline.is_playing()
            report["callback_errors"] = sorted(str(value) for value in getattr(ex, "_physics_step_error_logged", ()))
            report["articulation_health_fault"] = getattr(ex, "_articulation_health_fault", None)
            if (restore_errors or report["playing_at_end"] or report["callback_errors"]
                    or report["articulation_health_fault"] or "pause_wait_error" in report):
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
                # The replay deadline may land while its queued pause or final
                # write is pending. Keep this separately bounded cleanup alive.
                continue
        cleanup_task.result()


_contact_grasp_config = __CONFIG__
if "CONTACT_GRASP_VALIDATION_TASK" in globals() and not CONTACT_GRASP_VALIDATION_TASK.done():
    raise RuntimeError("A contact grasp validation is already running")
CONTACT_GRASP_VALIDATION_TASK = asyncio.ensure_future(
    asyncio.wait_for(_validate_contact_grasp(EX, _contact_grasp_config), timeout=_contact_grasp_config["timeout"] - 8.0)
)
print("Started contact grasp validation: " + _contact_grasp_config["output"], flush=True)
'''


def build_live_code(config: dict) -> str:
    """Embed trusted CLI parameters into the simulator-side replay source."""
    return LIVE_CODE.replace("__CONFIG__", repr(config))


def main() -> int:
    """Start one bounded live replay and report its progress artifact."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=300.0, help="Overall wall timeout including server cleanup")
    parser.add_argument(
        "--trace-contact-steps",
        action="store_true",
        help="Include per-physics-step close/lift/hold contact diagnostics",
    )
    parser.add_argument("--object", choices=("cube", "cylinder", "both"), default="both")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--offset",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Override the per-object world offset from the measured thumb/index grasp-space target, metres",
    )
    parser.add_argument("--finger-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--palm-normal",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Anatomical palm normal; defaults to -Z for right-hand profiles or -Y for custom left-hand runs",
    )
    parser.add_argument("--curl", type=float, default=0.75)
    parser.add_argument("--thumb-curl", type=float, help="Override per-object thumb flexion: cube .45, cylinder .55")
    parser.add_argument("--opposition", type=float, help="Override per-object thumb opposition: cube .40, cylinder .50")
    parser.add_argument("--lift", type=float, default=0.12, help="Requested vertical hand motion in metres")
    parser.add_argument("--min-lift", type=float, default=0.04, help="Minimum measured lift sustained throughout hold")
    parser.add_argument("--hold", type=float, default=2.0, help="Hold duration in simulated seconds")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "_compat61/contact-grasp-live.json"
    )
    args = parser.parse_args()
    objects = {"cube": ["cube"], "cylinder": ["cylinder"], "both": ["cube", "cylinder"]}[args.object]
    try:
        profiles = {
            f"/World/G1_SampleBoxes/Box_{['cube', 'cylinder'].index(name):02d}": choose_profile(
                name,
                side=args.side,
                offset=args.offset,
                thumb_curl=args.thumb_curl,
                opposition=args.opposition,
                curl=args.curl,
            )
            for name in objects
        }
    except ValueError as error:
        parser.error(str(error))
    normal = args.palm_normal or ((0.0, 0.0, -1.0) if args.side == "right" else (0.0, -1.0, 0.0))
    values = [
        args.timeout,
        args.lift,
        args.min_lift,
        args.hold,
        *args.finger_direction,
        *normal,
    ]
    if not all(math.isfinite(value) for value in values):
        parser.error("All numeric arguments must be finite")
    if args.timeout <= 15:
        parser.error("Timeout must exceed 15 seconds")
    if not 0 < args.min_lift <= args.lift <= 0.3 or not 0.2 <= args.hold <= 10:
        parser.error("Require 0 < min-lift <= lift <= 0.3 metres and hold between 0.2 and 10 seconds")
    direction = args.finger_direction
    cross = (
        normal[1] * direction[2] - normal[2] * direction[1],
        normal[2] * direction[0] - normal[0] * direction[2],
        normal[0] * direction[1] - normal[1] * direction[0],
    )
    if sum(value * value for value in cross) < 1e-10:
        parser.error("Finger direction and palm normal must be nonzero and nonparallel")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = dict(
        run_id=uuid.uuid4().hex,
        timeout=args.timeout,
        trace_contact_steps=args.trace_contact_steps,
        side=args.side,
        paths=list(profiles),
        profiles=profiles,
        finger_direction=args.finger_direction,
        palm_normal=normal,
        lift=args.lift,
        min_lift=args.min_lift,
        hold=args.hold,
        output=str(args.output),
    )
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
                print(json.dumps(report, indent=2, sort_keys=True))
                return 0 if report["status"] == "passed" else 1
        time.sleep(0.2)
    print("Host deadline elapsed; inspect " + str(args.output) + " for server cleanup and partial results.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
