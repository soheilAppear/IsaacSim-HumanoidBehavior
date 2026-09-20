#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate render-head rotation and the native XR camera in a loaded humanoid.

Run with ordinary host Python while Isaac is healthy and Quest XR is active::

    python tools/validate_robot_head_camera_live.py --output .webcam/robot-head-camera.json

The bounded Kit task pauses physics, supplies synthetic physical-head rotations
and translations, and checks the production camera against the actual XR device's
virtual pose. It checks the selected physical/comfort attachment mode and unchanged
collision opinions. Both modes retain the visible head's fixed neck pivot;
comfort mode follows the HMD-driven eye arc around the neck while rejecting
body sway. Physical HMD translation remains ignored.
Signed head bounds depths are reported as diagnostics, not an occlusion pass.
Comfort mode also injects torso-only shake into
the camera reader and requires complete rejection; the real actor stays paused.
Startup and recenter cases use production calibration while the synthetic headset
is pitched and rolled, then require an upright yaw sweep to keep a level horizon.
It does not replace XRCore, move the robot, or
change gaze. This is an integration replay, not operator tracking-quality testing.

The physical-head reader, rotation calibration, and original Play/Pause state are
restored in finally, including cancellation. Short TCP polls write a JSON report;
Kit also maintains a sibling journal so a lost host connection does not own cleanup.
Use --resume JOB_ID to retrieve an existing test without starting another replay.
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

JOB_PREFIX = "ROBOT_HEAD_CAMERA_JOB "

LIVE_CODE = r"""
import asyncio
import hashlib
import json
import copy
import time
import traceback
from pathlib import Path

import omni.kit.app
import omni.timeline
import omni.usd
from omni.kit.xr.core import XRSettings
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

_ROBOT_HEAD_CAMERA_JOBS = globals().setdefault("_ROBOT_HEAD_CAMERA_JOBS", {})
for _registry_name, _registry in list(globals().items()):
    if _registry_name.endswith("_JOBS") and isinstance(_registry, dict):
        for _other_id, _other in _registry.items():
            _task = _other.get("task") if isinstance(_other, dict) else None
            if _task is not None and not _task.done():
                raise RuntimeError("Another live validation is running: " + str(_other_id))
if OPTIONS["job_id"] in _ROBOT_HEAD_CAMERA_JOBS:
    raise RuntimeError("Head-camera job already exists")
for _old_id in list(_ROBOT_HEAD_CAMERA_JOBS)[:-7]:
    del _ROBOT_HEAD_CAMERA_JOBS[_old_id]
_job = {
    "job_id": OPTIONS["job_id"], "done": False,
    "report": {"passed": False, "phase": "starting", "poses": [],
               "hardware_tracking_quality_tested": False, "native_xr_pose_tested": True,
               "first_person_occlusion_tested": False,
               "visible_head_attachment": "torso_neck_pivot_in_all_camera_modes",
               "comfort_eye_attachment": "hmd_neck_arc_in_stabilized_robot_frame"},
}
_ROBOT_HEAD_CAMERA_JOBS[OPTIONS["job_id"]] = _job

def _write_head_camera_journal(job, options):
    path = Path(options["journal"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(job["report"], job_id=job["job_id"], done=job["done"]),
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")

def _freeze_collision_value(value):
    '''Keep complete typed USD metadata, independent of Python wrapper addresses.'''
    kind = type(value).__module__ + "." + type(value).__qualname__
    if value is None or isinstance(value, (str, bool, int)):
        return (kind, value)
    if isinstance(value, float):
        # Preserve signed zero, infinities and NaN bits while retaining reflexive
        # equality. Comparing raw NaNs would spuriously report changed opinions.
        import struct
        return (kind, struct.pack("!d", value).hex())
    if isinstance(value, bytes):
        return (kind, value.hex())
    if isinstance(value, dict):
        items = [(_freeze_collision_value(key), _freeze_collision_value(item)) for key, item in value.items()]
        return (kind, tuple(sorted(items, key=lambda item: repr(item[0]))))
    if kind.startswith("pxr.Sdf.") and type(value).__name__.endswith("ListOp"):
        # Explicit and nonexplicit operations differ even when applying them to
        # an empty list produces the same result. Preserve every authored bucket
        # and its order; do not substitute GetAppliedItems() or repr(wrapper).
        fields = ("explicitItems", "addedItems", "deletedItems", "prependedItems", "appendedItems", "orderedItems")
        return (kind, ("isExplicit", bool(value.isExplicit)),
                tuple((field, _freeze_collision_value(getattr(value, field))) for field in fields))
    if isinstance(value, Sdf.Reference):
        return (kind, _freeze_collision_value(value.assetPath), _freeze_collision_value(value.primPath),
                _freeze_collision_value(value.layerOffset), _freeze_collision_value(value.customData))
    if isinstance(value, Sdf.Payload):
        return (kind, _freeze_collision_value(value.assetPath), _freeze_collision_value(value.primPath),
                _freeze_collision_value(value.layerOffset))
    if isinstance(value, Sdf.LayerOffset):
        return (kind, _freeze_collision_value(value.offset), _freeze_collision_value(value.scale))
    if isinstance(value, Sdf.AssetPath):
        return (kind, value.path, value.resolvedPath)
    if isinstance(value, Sdf.Path):
        return (kind, str(value))
    if isinstance(value, Sdf.ValueTypeName):
        return (kind, str(value), str(value.role), value.type.typeName)
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        return (kind, _freeze_collision_value(value.GetReal()), _freeze_collision_value(value.GetImaginary()))
    if kind.startswith("pxr.Vt."):
        try:
            view = memoryview(value)
        except TypeError:
            view = None  # String/token arrays still require their complete values.
        if view is not None and view.format.lstrip("@=<>!") in ("?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "e", "f", "d"):
            # Fingerprint every numeric byte directly. Expanding million-vertex
            # meshes into per-scalar Python tuples stalls the application frame
            # and can exhaust the replay deadline. Preserve the type and complete
            # buffer layout; no sampling or shortened display repr is involved.
            data = view if view.c_contiguous else view.tobytes(order="C")
            return (kind, ("numeric_buffer_sha256", view.format, view.itemsize,
                           tuple(view.shape), tuple(view.strides), view.nbytes,
                           hashlib.sha256(data).hexdigest()))
    if isinstance(value, (list, tuple)) or (
        kind.startswith(("pxr.Gf.", "pxr.Vt.")) and hasattr(value, "__len__") and hasattr(value, "__getitem__")
    ):
        # Iterate the complete Vt array/Gf matrix: display repr can abbreviate
        # large values, hiding an actual edit beyond the printed prefix/suffix.
        return (kind, tuple(_freeze_collision_value(item) for item in value))
    representation = repr(value)
    if " at 0x" in representation or " object at " in representation:
        raise TypeError("Unsupported opaque collision metadata: " + kind)
    # USD enums, value-type names and value-block markers have stable value repr.
    # Unknown address-based wrappers fail the check rather than dropping fields.
    return (kind, representation)

async def _run_robot_head_camera(ex, job, options):
    import math

    report = job["report"]
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    started = time.perf_counter()
    was_playing = bool(timeline.is_playing())
    original_reader = getattr(ex, "__dict__", {}).get("_read_physical_head_pose")
    had_reader = "_read_physical_head_pose" in getattr(ex, "__dict__", {})
    original_neutral = getattr(ex, "_head_camera_operator_neutral_rotation", None)
    original_delta = getattr(ex, "_head_camera_operator_delta_rotation", None)
    comfort = bool(getattr(ex, "_head_camera_comfort_enabled", False))
    original_filter = copy.deepcopy(ex._head_camera_comfort_filter) if comfort else None
    original_calibrated = getattr(ex, "_head_camera_comfort_height_calibrated", False)
    had_body_reader = "_read_camera_mount_body_pose" in vars(ex)
    original_body_reader = vars(ex).get("_read_camera_mount_body_pose")
    report["comfort_view"] = comfort
    gaze = getattr(ex, "_eye_gaze_tracker", None)
    errors_before = set(getattr(ex, "_physics_step_error_logged", ()))
    modified = False
    collisions_before = None
    fake_head = None

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def matrix_error(left, right):
        return max(abs(float(left[row][column] - right[row][column]))
                   for row in range(4) for column in range(4))

    def rotation_error_degrees(left, right):
        left_quat = left.ExtractRotationQuat().GetNormalized()
        right_quat = right.ExtractRotationQuat().GetNormalized()
        dot = abs(float(left_quat.GetReal() * right_quat.GetReal()
                        + Gf.Dot(left_quat.GetImaginary(), right_quat.GetImaginary())))
        return math.degrees(2.0 * math.acos(min(1.0, dot)))

    def check_health():
        require(ex._eye_gaze_tracker is gaze, "Gaze tracker was replaced")
        require(not ex._articulation_health_fault, str(ex._articulation_health_fault))
        require(not ex.g1.locomotion_fault, str(ex.g1.locomotion_fault))
        errors = set(ex._physics_step_error_logged) - errors_before
        require(not errors, "New callback errors: " + repr(sorted(errors)))
        require(ex._xr_core.is_xr_enabled() and ex._xr_core.is_xr_display_enabled(), "XR became inactive")

    async def frames(count):
        for _ in range(count):
            await app.next_update_async()
            check_health()

    def collision_opinions():
        # Compare authored specs, including weaker referenced layers, rather than
        # only current physics poses. The render-head op must not touch any of them.
        result = {}
        root = stage.GetPrimAtPath(ex._g1_prim_path)
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) and "/collisions" not in str(prim.GetPath()):
                continue
            specs = []
            for spec in prim.GetPrimStack():
                properties = []
                for prop in spec.properties:
                    properties.append((str(prop.path), sorted((key, _freeze_collision_value(prop.GetInfo(key)))
                                                             for key in prop.ListInfoKeys())))
                specs.append((spec.layer.identifier, str(spec.path),
                              sorted((key, _freeze_collision_value(spec.GetInfo(key))) for key in spec.ListInfoKeys()),
                              sorted(properties)))
            result[str(prim.GetPath())] = specs
        require(bool(result), "No collision prims were available for the invariance check")
        return result

    def describe_collision_changes(before, after):
        '''Report differing leaf opinions without dumping entire mesh arrays.'''
        changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        details = []

        def find_leaves(left, right, location, leaves):
            if left == right or len(leaves) >= 5:
                return
            if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right):
                if len(left) == 2 and isinstance(left[0], str) and left[0] == right[0]:
                    location += "(" + left[0] + ")"
                for index, (old, new) in enumerate(zip(left, right)):
                    find_leaves(old, new, location + "[" + str(index) + "]", leaves)
            else:
                old, new = repr(left), repr(right)
                leaves.append({"location": location, "before": old[:4000], "after": new[:4000],
                               "before_characters": len(old), "after_characters": len(new)})

        for path in changed[:5]:
            leaves = []
            find_leaves(before.get(path), after.get(path), "specs", leaves)
            details.append({"prim_path": path, "differences": leaves})
        return {"changed_paths": changed, "first_five_changed_prim_details": details,
                "before_prim_count": len(before), "after_prim_count": len(after)}

    def render_head_world():
        # USD may lag Fabric for a simulated torso. Read the render transform
        # relative to the torso, then compose with the real live rigid-body pose.
        cache = UsdGeom.XformCache()
        head = stage.GetPrimAtPath(visual.head_path)
        torso = stage.GetPrimAtPath(ex._head_camera_mount_body_path)
        local = cache.GetLocalToWorldTransform(head) * cache.GetLocalToWorldTransform(torso).GetInverse()
        body = ex._read_camera_mount_body_pose()
        require(body is not None, "Camera mount lost its real rigid-body pose")
        return local * body, body

    def snapshot(label, expected_rotation, baseline_body, camera_to_head):
        camera = Gf.Matrix4d(ex._head_camera_transform_op.Get())
        device = ex._find_xr_head_device()
        require(device is not None, "Native XR head device is unavailable")
        native = Gf.Matrix4d(device.get_virtual_world_pose_desc("").pose_matrix)
        head, body = render_head_world()
        require(all(math.isfinite(float(matrix[row][column]))
                    for matrix in (camera, native, head, body)
                    for row in range(4) for column in range(4)), "Nonfinite camera or head transform")
        body_error = matrix_error(body, baseline_body)
        camera_local = camera * body.GetInverse()
        mounted_rotation = Gf.Matrix4d().SetRotate(ex._head_camera_mount_local.ExtractRotationQuat())
        if comfort:
            mounted_rotation = Gf.Matrix4d().SetRotate((baseline_camera * baseline_body.GetInverse()).ExtractRotationQuat())
        desired_local = expected_rotation * mounted_rotation
        local_rotation_error = rotation_error_degrees(camera_local, desired_local)
        native_position_error = (native.ExtractTranslation() - camera.ExtractTranslation()).GetLength()
        native_rotation_error = rotation_error_degrees(native, camera)
        native_matrix_error = matrix_error(native, camera)
        attachment_error = matrix_error(camera * head.GetInverse(), camera_to_head)
        pivot = head.Transform(visual.neck_pivot_local)
        expected_pivot = body.Transform(visual.neck_pivot_in_torso)
        pivot_error = (pivot - expected_pivot).GetLength()
        head_to_camera = head * camera.GetInverse()
        minimum_depth = min(head_to_camera.Transform(Gf.Vec3d(x,y,z))[2]
                            for x in (visual.bounds_min[0],visual.bounds_max[0])
                            for y in (visual.bounds_min[1],visual.bounds_max[1])
                            for z in (visual.bounds_min[2],visual.bounds_max[2]))
        require(body_error < 1e-5, "Robot body moved while physics was paused")
        require(pivot_error < 1e-5, "Visible render head moved away from the torso neck pivot")
        require(not visual.first_person_hidden, "Camera mode globally hid the render head")
        require(UsdGeom.Imageable(stage.GetPrimAtPath(visual.head_path)).ComputeVisibility()
                != UsdGeom.Tokens.invisible, "Render head is not visible to observer cameras")
        require(local_rotation_error < 0.01, "Camera rotation disagrees with the synthetic head input")
        require(native_position_error < 0.002, "Native XR camera position differs from the robot camera")
        require(native_rotation_error < 0.25, "Native XR camera rotation differs from the robot camera")
        require(native_matrix_error < 1e-3, "Native XR camera matrix differs from the robot camera")
        if comfort:
            # Geometric oracle: in neutral camera axes the eye is the origin.
            # Rotating it around the measured neck moves it by p - R(p).
            # This does not reuse the production articulated-view composition
            # or any measured torso-world transform used by that composition.
            pivot_in_camera = ex._head_camera_mount_local.GetInverse().Transform(visual.neck_pivot_in_torso)
            eye_arc = pivot_in_camera - expected_rotation.TransformDir(pivot_in_camera)
            expected_eye = baseline_camera.Transform(eye_arc)
            eye_error = (camera.ExtractTranslation() - expected_eye).GetLength()
            require(eye_error < 1e-5, "Comfort eye did not follow the HMD neck arc")
            expected_camera = expected_rotation * baseline_camera
            expected_camera.SetTranslateOnly(expected_eye)
            comfort_matrix_error = matrix_error(camera, expected_camera)
            require(comfort_matrix_error < 1e-5, "Comfort camera disagrees with the independent neck-pivot model")
        else:
            eye_error = 0.0
            comfort_matrix_error = None
            require(attachment_error < 1e-5, "Camera moved relative to the render head")
        # Camera local -Z is forward and +Y is up. These explicit signs catch
        # mirrored yaw/pitch/roll without depending only on matrix multiplication.
        relative = camera_local * mounted_rotation.GetInverse()
        forward = relative.TransformDir(Gf.Vec3d(0, 0, -1))
        up = relative.TransformDir(Gf.Vec3d(0, 1, 0))
        if label == "yaw_left": require(forward[0] < -0.45, "Left head turn points right")
        if label == "yaw_right": require(forward[0] > 0.45, "Right head turn points left")
        if label == "pitch_up": require(forward[1] > 0.30, "Up head tilt points down")
        if label.startswith("pitch_down"): require(forward[1] < -0.30, "Down head tilt points up")
        if label == "roll_left": require(up[0] < -0.30, "Left head roll is mirrored")
        if label == "roll_right": require(up[0] > 0.30, "Right head roll is mirrored")
        if label.startswith("upright_"):
            require(abs(forward[1]) < 1e-5 and (up - Gf.Vec3d(0,1,0)).GetLength() < 1e-5,
                    "Upright head acquired pitch or roll after tilted calibration")
        return {
            "label": label, "native_position_error_m": float(native_position_error),
            "native_rotation_error_degrees": native_rotation_error, "native_matrix_error": native_matrix_error,
            "camera_head_attachment_matrix_error": attachment_error, "comfort_eye_position_error_m": float(eye_error), "neck_pivot_error_m": float(pivot_error),
            "comfort_eye_position_reference": "independent_hmd_neck_arc" if comfort else None,
            "comfort_camera_matrix_error": comfort_matrix_error,
            "visible_head_minimum_camera_depth_m": float(minimum_depth),
            "visible_head_bounds_entirely_behind_eye": bool(minimum_depth >= 0.0),
            "head_depth_diagnostic_only": True, "first_person_occlusion_tested": False,
            "neck_pivot_fixed_required": True,
            "camera_input_rotation_error_degrees": local_rotation_error, "paused_body_matrix_error": body_error,
            "camera_world": [list(row) for row in camera], "head_world": [list(row) for row in head],
            "head_relative_to_camera": [list(row) for row in head_to_camera],
            "relative_forward": list(forward), "relative_up": list(up),
            "neck_pivot_world": list(pivot), "direction_sign_passed": True,
        }

    async def record_pose(label, rotation, reference, translation=(0,0,0)):
        nonlocal fake_head
        report["phase"] = label
        _write_head_camera_journal(job, options)
        fake_head = rotation * reference
        fake_head.SetTranslateOnly(position + Gf.Vec3d(*translation))
        await frames(8)
        samples = []
        for _ in range(3):
            await frames(1)
            samples.append(snapshot(label, rotation, baseline_body, camera_to_head))
        item = samples[-1]
        item["settled_samples"] = len(samples)
        for key in ("native_position_error_m", "native_rotation_error_degrees", "native_matrix_error"):
            item[key] = max(sample[key] for sample in samples)
        if label.startswith("translation"):
            translation_error = matrix_error(ex._head_camera_transform_op.Get(), baseline_camera)
            item["translation_only_camera_matrix_error"] = translation_error
            require(translation_error < 1e-5, "Physical translation moved the robot-mounted camera")
        report["poses"].append(item)

    async def initialize_tilted(label, yaw_degrees, pitch_degrees, roll_degrees):
        nonlocal fake_head
        reference = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,1,0), yaw_degrees))
        # In row-vector order roll around the head's forward axis first, then
        # pitch, then world yaw. The horizontal forward bearing remains exactly
        # the known reference yaw; this oracle does not call the helper under test.
        tilt = (Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1), roll_degrees))
                * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0), pitch_degrees)))
        fake_head = tilt * reference
        fake_head.SetTranslateOnly(position)
        # This is the head-calibration reset used by startup and recenter. Calling
        # the broader B-button handler would also reset navigation/webcam state.
        ex._reset_operator_head_rotation()
        ex._update_head_camera_view(force=True, dt=0.0)
        await record_pose(label, tilt, reference)
        actual_reference = ex._head_camera_operator_neutral_rotation
        require(actual_reference is not None, "Production did not initialize its head reference")
        reference_error = matrix_error(actual_reference, reference)
        require(reference_error < 1e-5, "Tilted calibration stored pitch or roll in the yaw reference")
        report.setdefault("tilted_calibrations", []).append({
            "label":label, "startup_yaw_degrees":yaw_degrees,
            "startup_pitch_degrees":pitch_degrees, "startup_roll_degrees":roll_degrees,
            "reference_matrix_error":reference_error,
            "production_reset_used":True,
        })
        return reference

    try:
        require(ex is not None and ex.g1 is not None, "Load a healthy humanoid example first")
        require(ex._physics_ready and ex.g1._initialized, "Start the simulation before this validation")
        require(ex._xr_camera_mode == "robot_head", "Robot-head camera mode is required")
        require(ex._head_camera_operator_rotation_enabled, "Operator head rotation is disabled")
        require(ex._head_camera_update_sub is not None, "Missing application-frame camera updates")
        require(ex._xr_core is not None and ex._xr_camera_can_schedule(), "Active XR camera scheduling is required")
        visual = ex._operator_head_visual
        require(visual.ready, "Render head is unavailable: " + str(visual.status))
        require(not visual.first_person_hidden, "First person still globally hides the render head")
        stage = omni.usd.get_context().get_stage()
        require(stage is not None, "No USD stage")
        require(UsdGeom.Imageable(stage.GetPrimAtPath(visual.head_path)).ComputeVisibility()
                != UsdGeom.Tokens.invisible, "Render head is not visible")
        report["render_head_visible"] = True
        physical = ex._read_physical_head_pose()
        require(physical is not None, "No valid physical headset pose")
        # Use a known level heading instead of assigning the real tilted HMD
        # rotation as neutral, which previously hid startup pitch/roll errors.
        neutral = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,1,0), 37.0))
        position = physical.ExtractTranslation()
        check_health()
        report.update(initial_playing=was_playing, head_path=visual.head_path,
                      mount_body=ex._head_camera_mount_body_path, neck_pivot_local=list(visual.neck_pivot_local),
                      eye_mount_local=list(visual.eye_mount_local))
        timeline.pause()
        modified = True
        await frames(4)
        require(not timeline.is_playing(), "Physics did not pause")
        collisions_before = collision_opinions()
        if comfort:
            ex._head_camera_comfort_filter.reset()
            ex._head_camera_comfort_height_calibrated = True
            report["eye_height_source"] = ex._head_camera_eye_height_source
        fake_head = Gf.Matrix4d(neutral)
        fake_head.SetTranslateOnly(position)
        ex._read_physical_head_pose = lambda: Gf.Matrix4d(fake_head)
        ex._reset_operator_head_rotation()
        ex._update_head_camera_view(force=True, dt=0.0)
        await frames(8)
        require(ex._head_camera_operator_neutral_rotation is not None
                and matrix_error(ex._head_camera_operator_neutral_rotation, neutral) < 1e-5,
                "Production did not initialize the known level heading")
        baseline_camera = Gf.Matrix4d(ex._head_camera_transform_op.Get())
        baseline_head, baseline_body = render_head_world()
        camera_to_head = baseline_camera * baseline_head.GetInverse()
        # XR has its own clipping setting; matching the desktop USD camera is
        # insufficient. Resolve the active profile through the installed public
        # API, including fallback to the global preference if no override exists.
        near_token = "profile/persistent/render/nearPlane"
        xr_settings = XRSettings.get_singleton()
        native_near = xr_settings.get_setting(near_token)
        camera_schema = UsdGeom.Camera.Get(stage, ex._head_camera_path)
        usd_clip = camera_schema.GetClippingRangeAttr().Get()
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        usd_near_m = float(usd_clip[0]) * meters_per_unit
        expected_near_m = float(ex._head_camera_near_clip_m)
        require(isinstance(native_near, (int, float)) and math.isfinite(native_near),
                "Native XR near clipping plane is unavailable or nonfinite")
        require(abs(float(native_near) - expected_near_m) < 1e-7,
                "Native XR near clipping plane does not match the robot camera configuration")
        require(abs(usd_near_m - float(native_near)) < 1e-7,
                "Desktop USD and native XR near clipping planes disagree")
        report["near_clipping"] = {
            "configured_m": expected_near_m, "usd_m": usd_near_m,
            "native_setting_m": float(native_near),
            "native_setting_path": xr_settings.get_setting_location(near_token),
            "native_default_m": xr_settings.get_default_setting(near_token),
            "usd_stage_meters_per_unit": meters_per_unit,
            "native_rendered_clip_plane_tested": False,
        }
        cases = (
            ("neutral", (0, 1, 0), 0, (0, 0, 0)),
            ("yaw_left", (0, 1, 0), 30, (0, 0, 0)),
            ("yaw_right", (0, 1, 0), -30, (0, 0, 0)),
            ("pitch_up", (1, 0, 0), 20, (0, 0, 0)),
            ("pitch_down", (1, 0, 0), -20, (0, 0, 0)),
            ("pitch_down_60", (1, 0, 0), -60, (0, 0, 0)),
            ("pitch_down_85", (1, 0, 0), -85, (0, 0, 0)),
            ("yaw_left_120", (0, 1, 0), 120, (0, 0, 0)),
            ("yaw_right_120", (0, 1, 0), -120, (0, 0, 0)),
            ("yaw_back_180", (0, 1, 0), 180, (0, 0, 0)),
            ("roll_left", (0, 0, 1), 20, (0, 0, 0)),
            ("roll_right", (0, 0, 1), -20, (0, 0, 0)),
            ("translation_a", (0, 1, 0), 0, (0.6, -0.4, 0.3)),
            ("translation_b", (0, 1, 0), 0, (-0.7, 0.3, -0.2)),
        )
        for label, axis, degrees, translation in cases:
            rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*axis), degrees))
            await record_pose(label, rotation, neutral, translation)
        tilted_reference = await initialize_tilted("tilted_startup", 37.0, 18.0, -11.0)
        for label, degrees in (("upright_after_tilted_startup",0),
                               ("upright_yaw_left_75",75), ("upright_yaw_left_30",30),
                               ("upright_yaw_right_30",-30), ("upright_yaw_right_75",-75),
                               ("upright_yaw_neutral",0)):
            await record_pose(label, Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,1,0), degrees)), tilted_reference)
        recentered_reference = await initialize_tilted("tilted_recenter", -48.0, -23.0, 9.0)
        for label, degrees in (("upright_after_tilted_recenter",0),
                               ("upright_recenter_yaw_left",55), ("upright_recenter_yaw_right",-55)):
            await record_pose(label, Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,1,0), degrees)), recentered_reference)
        if comfort:
            report["phase"] = "torso_shake_rejection"
            fake_head = Gf.Matrix4d(physical)
            ex._update_head_camera_view(force=True, dt=0.0)
            await frames(4)
            reference = Gf.Matrix4d(ex._head_camera_transform_op.Get())
            jitter_pose = Gf.Matrix4d(baseline_body)
            ex._read_camera_mount_body_pose = lambda: Gf.Matrix4d(jitter_pose)
            worst = 0.0
            worst_attachment = 0.0
            worst_pivot = 0.0
            minimum_head_depth = None
            for index in range(48):
                angle = 2.0 * math.pi * index / 12.0
                jitter_pose = (Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0), 8*math.sin(angle)))
                               * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,1,0), 6*math.cos(angle)))
                               * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1), 7*math.sin(angle)))
                               * baseline_body)
                jitter_pose.SetTranslateOnly(baseline_body.ExtractTranslation() + Gf.Vec3d(0,0,.04*math.sin(angle)))
                ex._update_head_camera_view(force=True, dt=0.0)
                await frames(1)
                worst = max(worst, matrix_error(ex._head_camera_transform_op.Get(), reference))
                attached_head, observed_body = render_head_world()
                worst_attachment = max(worst_attachment,
                                       matrix_error(ex._head_camera_transform_op.Get() * attached_head.GetInverse(), camera_to_head))
                neck = attached_head.Transform(visual.neck_pivot_local)
                expected_neck = observed_body.Transform(visual.neck_pivot_in_torso)
                worst_pivot = max(worst_pivot, float((neck - expected_neck).GetLength()))
                relative_head = attached_head * Gf.Matrix4d(ex._head_camera_transform_op.Get()).GetInverse()
                depth = min(relative_head.Transform(Gf.Vec3d(x,y,z))[2]
                            for x in (visual.bounds_min[0],visual.bounds_max[0])
                            for y in (visual.bounds_min[1],visual.bounds_max[1])
                            for z in (visual.bounds_min[2],visual.bounds_max[2]))
                minimum_head_depth = float(depth) if minimum_head_depth is None else min(minimum_head_depth, float(depth))
                require(not visual.first_person_hidden and
                        UsdGeom.Imageable(stage.GetPrimAtPath(visual.head_path)).ComputeVisibility()
                        != UsdGeom.Tokens.invisible, "Torso motion hid the render head")
            report["injected_torso_jitter"] = {"roll_degrees":8,"pitch_degrees":6,"yaw_degrees":7,"height_m":.04,
                                                "camera_matrix_max_error":worst,
                                                "head_attachment_max_error":worst_attachment,
                                                "head_attachment_error_diagnostic_only":True,
                                                "neck_pivot_max_error_m":worst_pivot,
                                                "visible_head_minimum_camera_depth_m":minimum_head_depth,
                                                "head_depth_diagnostic_only":True,
                                                "first_person_occlusion_tested":False,
                                                "physics_actor_moved":False}
            require(worst < 1e-5, "Torso yaw, roll, pitch or bob reached the comfort view")
            require(worst_pivot < 1e-5, "Injected torso motion separated the render head from its neck")
            if had_body_reader: ex._read_camera_mount_body_pose = original_body_reader
            else: ex.__dict__.pop("_read_camera_mount_body_pose",None)
        collisions_after = collision_opinions()
        if collisions_after != collisions_before:
            report["collision_opinion_changes"] = describe_collision_changes(collisions_before, collisions_after)
        require(collisions_after == collisions_before, "Head motion changed collision prim opinions")
        report.update(collision_prim_count=len(collisions_before), collision_opinions_unchanged=True,
                      collision_opinions_sha256=hashlib.sha256(repr(collisions_before).encode()).hexdigest(),
                      gaze_tracker_preserved=ex._eye_gaze_tracker is gaze, passed=True, phase="complete")
    except BaseException as error:
        report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if modified:
            try:
                if had_reader:
                    ex._read_physical_head_pose = original_reader
                else:
                    ex.__dict__.pop("_read_physical_head_pose", None)
                if had_body_reader: ex._read_camera_mount_body_pose = original_body_reader
                else: ex.__dict__.pop("_read_camera_mount_body_pose",None)
                if comfort:
                    ex._head_camera_comfort_filter = original_filter
                    ex._head_camera_comfort_height_calibrated = original_calibrated
                ex._head_camera_operator_neutral_rotation = original_neutral
                ex._head_camera_operator_delta_rotation = original_delta
                report["neutral_calibration_restored"] = ex._head_camera_operator_neutral_rotation is original_neutral
                ex._update_head_camera_view(force=True, dt=0.0)
                if was_playing:
                    timeline.play()
                else:
                    timeline.pause()
                await frames(4)
                report["reader_restored"] = (
                    ex.__dict__.get("_read_physical_head_pose") is original_reader if had_reader
                    else "_read_physical_head_pose" not in ex.__dict__
                )
                report["timeline_restored"] = bool(timeline.is_playing()) == was_playing
                require(report["reader_restored"] and report["timeline_restored"], "Input or timeline was not restored")
                require(report["neutral_calibration_restored"], "Head calibration was not restored")
            except BaseException as error:
                report.update(passed=False, cleanup_error=repr(error))
        report.update(wall_seconds=time.perf_counter() - started, final_playing=bool(timeline.is_playing()),
                      gaze_tracker_preserved=getattr(ex, "_eye_gaze_tracker", None) is gaze,
                      fault=getattr(ex, "_articulation_health_fault", None),
                      new_callback_errors=sorted(set(getattr(ex, "_physics_step_error_logged", ())) - errors_before))
        if ex is not None and ex._xr_core is not None:
            report.update(final_xr_enabled=bool(ex._xr_core.is_xr_enabled()),
                          final_xr_display_enabled=bool(ex._xr_core.is_xr_display_enabled()),
                          final_physics_ready=bool(ex._physics_ready))
        _write_head_camera_journal(job, options)

async def _robot_head_camera_runner(ex, job, options):
    try:
        await asyncio.wait_for(_run_robot_head_camera(ex, job, options), timeout=options["timeout"])
    except BaseException as error:
        job["report"].update(passed=False, job_error=repr(error))
    finally:
        job["done"] = True
        _write_head_camera_journal(job, options)

_job["task"] = asyncio.ensure_future(_robot_head_camera_runner(EX, _job, OPTIONS))
print("ROBOT_HEAD_CAMERA_JOB " + json.dumps({key: value for key, value in _job.items() if key != "task"}))
"""

POLL_CODE = r"""
import json
_job = _ROBOT_HEAD_CAMERA_JOBS[JOB_ID]
_task = _job["task"]
if _task.done() and not _job["done"]:
    _job["done"] = True
    _job["report"].update(passed=False, job_error="Cancelled" if _task.cancelled() else repr(_task.exception()))
print("ROBOT_HEAD_CAMERA_JOB " + json.dumps({key: value for key, value in _job.items() if key != "task"}, allow_nan=False))
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
    """Run or retrieve a bounded head-camera test in the existing Isaac process."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=90.0, help="Kit task deadline, at most 120 seconds")
    parser.add_argument("--output", type=Path, default=Path(".webcam/robot-head-camera.json"))
    parser.add_argument("--resume", metavar="JOB_ID", help="Poll an existing job without starting a replay")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 120:
        parser.error("--timeout must be finite and between 0 and 120 seconds")
    try:
        job_id = str(uuid.UUID(args.resume)) if args.resume else str(uuid.uuid4())
    except ValueError:
        parser.error("--resume requires the UUID from the start message")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = output.with_suffix(".journal.json")
    deadline = time.monotonic() + args.timeout + 15
    print(f"Head-camera job {job_id}; journal {journal}", flush=True)
    if not args.resume:
        options = {"job_id": job_id, "timeout": args.timeout, "journal": str(journal)}
        status, _ = request_job("OPTIONS = " + repr(options) + "\n" + LIVE_CODE, args.host, args.port, 10)
        if status:
            print(f"Start response failed; recover an accepted job with --resume {job_id}.")
            return status
    last_phase = None
    poll = "JOB_ID = " + repr(job_id) + "\n" + POLL_CODE
    try:
        while time.monotonic() < deadline:
            status, reply = request_job(poll, args.host, args.port, min(10, max(0.1, deadline - time.monotonic())))
            if status:
                print(f"Kit poll failed. Kit owns timeout cleanup; inspect {journal} or --resume {job_id}.")
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
    except KeyboardInterrupt:
        cancel = "_ROBOT_HEAD_CAMERA_JOBS[" + repr(job_id) + ']["task"].cancel()\n' + poll
        request_job(cancel, args.host, args.port, 10)
        print(f"Cancellation requested; Kit restores inputs and timeline. Retrieve with --resume {job_id}.")
        return 130
    print(f"Host wait expired. Kit owns timeout cleanup; inspect {journal} or --resume {job_id}.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
