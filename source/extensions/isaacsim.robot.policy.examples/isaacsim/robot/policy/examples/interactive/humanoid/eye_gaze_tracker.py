# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Meta Quest Pro eye-gaze tracking for the H1 humanoid VR example.

Reads the runtime's combined ("unified") OpenXR eye-gaze pose through Kit's
XRCore input-device API, raycasts it into the PhysX scene, and draws a thin
RED RAY from the user's eyes to the gazed point with a large BLOOD-RED marker
sphere at the collision. Sample boxes being looked at are tinted yellow, and
every change of gazed object is printed live to the terminal as
"[EyeGaze] looking at ..." so you can watch what the user looks at without
opening the CSVs. The same data feeds gaze.csv through the `latest` GazeSample
(see humanoid_example._collect_gaze_sample).

Verified working setup (Quest Pro, July 2026 — see HUMANOID_VR_CONTROL.md):
  1. Headset: Settings -> Movement tracking -> Eye tracking: ON
     (grant the permission and run the eye calibration once).
  2. Headset: give the *Steam Link* app the eye-tracking permission
     (Settings -> Privacy & Safety -> App permissions -> Eye tracking).
  3. Steam Link app settings: "Share eye tracking data to other apps on this
     PC": ON. This is OFF by default and is the switch people miss.
  4. PC: SteamVR set as the active OpenXR runtime; connect via Steam Link.
     Quest Link / Air Link NEVER forward XR_EXT_eye_gaze_interaction, so the
     Meta desktop app cannot deliver eye gaze no matter how it is configured.

Device naming: Kit registers the XR_EXT_eye_gaze_interaction device under its
bindings-manifest name "/user/eye/unified" with pose "gaze" (not the raw OpenXR
path "/user/eyes_ext", which is kept only as a fallback for other runtimes).

Per-eye devices ("/user/eye/left" / "/user/eye/right") are deliberately NOT
fused into the ray: on SteamVR + Steam Link their default poses carry a
head-like orientation rather than a true per-eye gaze, which silently biased
the ray toward the view center whenever the user looked near straight ahead.
The unified gaze IS the runtime's own calibrated binocular fusion of both
eyes, so using it alone is both the accurate and the correct "two eyes"
treatment.

This module is imported lazily by humanoid_example.py inside a try/except, so a
failure here can never remove the Humanoid example from the Isaac Examples menu.
"""

from dataclasses import dataclass

import carb
import omni.usd
from pxr import Gf, UsdGeom


@dataclass
class GazeSample:
    """Latest eye-gaze reading, in stage (virtual-world) coordinates.

    valid=False means no usable eye pose this update (device missing, pose not
    tracked yet, or eye tracking disabled on the headset). hit_* fields are only
    meaningful when hit_valid is True.
    """

    valid: bool = False
    origin: Gf.Vec3d | None = None
    direction: Gf.Vec3d | None = None
    hit_valid: bool = False
    hit_position: Gf.Vec3d | None = None
    hit_distance: float | None = None
    hit_object_path: str | None = None


class EyeGazeTracker:
    """Poll OpenXR eye gaze, raycast it, and draw a red gaze ray to the hit point.

    Usage from the humanoid example (all calls are cheap and safe to skip):
        tracker = EyeGazeTracker(xr_core)          # once, after XRCore is up
        tracker.update(dt)                         # every physics step
        sample = tracker.latest                    # read from the data logger
        tracker.cleanup()                          # on scene clear / teardown
    """

    # Combined-gaze devices, best candidate first. Kit maps the
    # XR_EXT_eye_gaze_interaction profile to the device name "/user/eye/unified"
    # with pose "gaze" (verified with SteamVR + Steam Link on a Quest Pro); the
    # remaining handles are fallbacks for other runtimes. Per-eye devices are
    # intentionally not used — see the module docstring.
    UNIFIED_DEVICES = ("/user/eye/unified", "/user/eyes_ext", "/user/eyes", "/user/gaze", "/user/eye_gaze")
    POSE_CANDIDATES = ("gaze", "gaze_ext", "")

    def __init__(self, xr_core, draw_ray: bool = True, highlight_gazed_boxes: bool = True):
        self._xr_core = xr_core
        self._draw_ray = draw_ray
        self._highlight_gazed_boxes = highlight_gazed_boxes

        # --- tunables -------------------------------------------------------
        self._max_distance = 20.0            # m: raycast + drawn-ray length cap
        self._ray_start_offset = 0.35        # m: skip forward past the robot's own head collision
        self._update_every_n_calls = 4       # update() is called at 200 Hz; do work at ~50 Hz
        self._ray_radius = 0.004             # m: red ray cylinder thickness
        self._ray_visual_start_offset = 0.6  # m: drawn ray starts this far ahead of the eyes (comfort)
        self._hit_marker_radius = 0.09       # m: large sphere marking the gaze collision point
        self._ray_color = Gf.Vec3f(1.0, 0.05, 0.05)          # red
        self._hit_marker_color = Gf.Vec3f(0.45, 0.0, 0.02)   # blood red
        self._highlight_color = Gf.Vec3f(1.0, 0.85, 0.1)     # gazed sample-box tint
        self._robot_root_path = "/World/H1"                  # own-body hits are re-cast past
        self._sample_box_root_marker = "/H1_SampleBoxes/"    # only these prims get tinted
        self._ray_root_path = "/World/H1_EyeGazeRay"

        # --- state ----------------------------------------------------------
        self.latest = GazeSample()
        self._call_counter = 0
        self._status_logged = False
        self._missing_warned = False
        self._failed_updates = 0
        self._visuals_created = False
        self._ray_xform_op = None
        self._marker_translate_op = None
        self._highlighted_path = None
        self._highlight_original_color = None
        self._raycast_error_logged = False
        # Sentinel that never equals a real target (None = "no hit"), so the
        # first resolved gaze target always produces a terminal line.
        self._last_reported_target: object = "<unset>"

    # ------------------------------------------------------------------ pose

    def _describe_input_devices(self) -> str:
        """One-line inventory of the session's XR input devices for the missing-eye-device warning.

        Answers "is the headset even connected, and does the runtime expose an eye
        device?" directly in the log, so no Script Editor probing is needed.
        """
        try:
            devices = self._xr_core.get_all_input_devices()
        except Exception as e:
            return f"<unavailable: {e}>"
        if not devices:
            return "<none - is the VR session running?>"
        parts = []
        for device in devices:
            try:
                name = str(device.get_name())
            except Exception:
                name = "<unnamed>"
            try:
                poses = ",".join(str(p) for p in device.get_pose_names())
            except Exception:
                poses = "?"
            parts.append(f"{name}[poses:{poses}]")
        return "; ".join(parts)

    def _read_pose_matrix(self, handles: tuple) -> Gf.Matrix4d | None:
        """Return the first valid pose matrix from the given device handles, or None.

        get_virtual_world_pose is preferred because the XR origin follows the
        robot head (schedule_set_camera), and the virtual-world frame is the one
        the scene raycast needs. An exact-identity matrix is treated as "not
        tracked yet" — OpenXR runtimes commonly report identity before the eye
        tracker locks on or while the user has eye tracking disabled.
        """
        for handle in handles:
            try:
                device = self._xr_core.get_input_device(handle)
            except Exception:
                device = None
            if device is None:
                continue
            try:
                pose_names = {str(name) for name in device.get_pose_names()}
            except Exception:
                pose_names = set()
            for reader_name in ("get_virtual_world_pose", "get_pose", "get_raw_pose"):
                reader = getattr(device, reader_name, None)
                if reader is None:
                    continue
                for pose_name in self.POSE_CANDIDATES:
                    if pose_name and pose_names and pose_name not in pose_names:
                        continue
                    try:
                        pose = reader(pose_name) if pose_name else reader()
                    except Exception:
                        continue
                    if pose is None or Gf.Matrix4d(pose) == Gf.Matrix4d(1.0):
                        continue
                    return Gf.Matrix4d(pose)
        return None

    @staticmethod
    def _matrix_to_ray(pose: Gf.Matrix4d):
        """Split a pose matrix into (origin, unit direction); OpenXR gaze looks down -Z."""
        origin = Gf.Vec3d(pose.ExtractTranslation())
        direction = pose.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
        length = direction.GetLength()
        if length <= 1e-6:
            return None
        return origin, direction / length

    def _read_gaze_ray(self):
        """Return the (origin, direction) gaze ray from the unified device, or None.

        Only the runtime's combined "unified" gaze is used — it is the pose the
        XR_EXT_eye_gaze_interaction profile guarantees to be an actual gaze ray,
        and it is already the runtime's calibrated fusion of both eyes. Manually
        mixing in the per-eye devices was tried and removed: their default poses
        carry a head-like orientation on SteamVR + Steam Link, which biased the
        ray toward the view center whenever the gaze was near straight ahead.
        """
        unified_pose = self._read_pose_matrix(self.UNIFIED_DEVICES)
        if unified_pose is None:
            return None
        ray = self._matrix_to_ray(unified_pose)
        if ray is None:
            return None
        if not self._status_logged:
            self._status_logged = True
            carb.log_info(
                f"EyeGazeTracker: using unified eye gaze (tried {', '.join(self.UNIFIED_DEVICES)})"
            )
        return ray

    # --------------------------------------------------------------- raycast

    def _raycast(self, origin: Gf.Vec3d, direction: Gf.Vec3d):
        """Raycast the gaze ray against scene colliders, skipping the robot's own body.

        The sample boxes and the ground plane both carry UsdPhysics.CollisionAPI,
        so they are valid gaze targets. The gaze origin sits at the robot's head
        (the XR origin follows it), so the cast starts _ray_start_offset metres
        along the ray, and a hit on the robot itself is retried once from just
        past that hit.

        Returns (hit_position, hit_distance_from_eye, hit_object_path) or
        (None, None, None) when nothing is hit within range.
        """
        try:
            import omni.physics.core

            query = omni.physics.core.get_physics_scene_query_interface()
        except Exception as e:
            self._warn_raycast_broken_once(e)
            return None, None, None

        start = origin + direction * self._ray_start_offset
        remaining = self._max_distance - self._ray_start_offset
        for _ in range(2):  # initial cast + at most one retry past a self-hit
            if remaining <= 0.0:
                break
            start_t = (float(start[0]), float(start[1]), float(start[2]))
            direction_t = (float(direction[0]), float(direction[1]), float(direction[2]))
            try:
                try:
                    # omni.physics.core requires the both_sides argument; omitting
                    # it raises TypeError, which used to be swallowed silently and
                    # disabled every gaze collision (hit marker, box tint, CSV hits).
                    ret, hit = query.raycast_closest(start_t, direction_t, remaining, False)
                except TypeError:
                    ret, hit = query.raycast_closest(start_t, direction_t, remaining)
            except Exception as e:
                self._warn_raycast_broken_once(e)
                return None, None, None
            if not ret:
                break

            hit_path = self._decode_hit_path(hit)
            hit_distance = float(getattr(hit, "distance", 0.0))
            # Match the robot prim SUBTREE only ("/World/H1" or "/World/H1/...").
            # A bare startswith("/World/H1") also matched "/World/H1_SampleBoxes/..."
            # and silently discarded every box hit as a self-hit.
            if hit_path is not None and (
                hit_path == self._robot_root_path or hit_path.startswith(self._robot_root_path + "/")
            ):
                # Own-body hit: continue the ray from just past it.
                start = start + direction * (hit_distance + 0.05)
                remaining -= hit_distance + 0.05
                continue

            hit_pos = getattr(hit, "position", None)
            hit_position = Gf.Vec3d(float(hit_pos[0]), float(hit_pos[1]), float(hit_pos[2])) if hit_pos is not None else None
            distance_from_eye = float((start - origin).GetLength()) + hit_distance if hit_position is None else float(
                (hit_position - origin).GetLength()
            )
            return hit_position, distance_from_eye, hit_path
        return None, None, None

    def _warn_raycast_broken_once(self, error: Exception) -> None:
        """Surface a broken scene-query API once instead of failing silently forever."""
        if self._raycast_error_logged:
            return
        self._raycast_error_logged = True
        carb.log_warn(
            f"EyeGazeTracker: gaze raycast failed ({error!r}) — collision info (hit marker, "
            "box highlight, gaze_hit_* columns) is disabled until this is fixed"
        )

    def _decode_hit_path(self, hit) -> str | None:
        """PhysX reports the hit body as an Sdf path string or an encoded int."""
        rigid_body = getattr(hit, "rigid_body", None)
        if isinstance(rigid_body, str):
            return rigid_body
        if rigid_body is not None:
            try:
                from pxr import PhysicsSchemaTools  # lazy: registered by the PhysX extension

                return str(PhysicsSchemaTools.intToSdfPath(rigid_body))
            except Exception:
                return None
        return None

    # --------------------------------------------------------------- visuals

    def _ensure_visuals(self) -> None:
        """Create the red gaze-ray cylinder and hit-marker sphere prims once.

        Neither prim gets a CollisionAPI, so the gaze raycast (and the grab
        system) can never hit the visualization itself.
        """
        if self._visuals_created or not self._draw_ray:
            return
        stage = omni.usd.get_context().get_stage()

        UsdGeom.Xform.Define(stage, self._ray_root_path)

        ray = UsdGeom.Cylinder.Define(stage, f"{self._ray_root_path}/Ray")
        ray.CreateAxisAttr("Z")
        ray.CreateHeightAttr(1.0)  # unit height: scaled to the actual ray length every update
        ray.CreateRadiusAttr(self._ray_radius)
        ray.CreateDisplayColorAttr().Set([self._ray_color])
        xformable = UsdGeom.Xformable(ray.GetPrim())
        xformable.ClearXformOpOrder()
        self._ray_xform_op = xformable.AddTransformOp()

        marker = UsdGeom.Sphere.Define(stage, f"{self._ray_root_path}/HitMarker")
        marker.CreateRadiusAttr(self._hit_marker_radius)
        marker.CreateDisplayColorAttr().Set([self._hit_marker_color])
        marker.ClearXformOpOrder()
        self._marker_translate_op = marker.AddTranslateOp()

        self._visuals_created = True
        carb.log_info(f"EyeGazeTracker: created gaze-ray visuals under {self._ray_root_path}")

    def _set_visuals_visible(self, visible: bool) -> None:
        if not self._visuals_created:
            return
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._ray_root_path)
        if not prim.IsValid():
            self._visuals_created = False  # stage was cleared under us; recreate next update
            return
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _update_ray_visual(self, origin: Gf.Vec3d, end: Gf.Vec3d, hit_valid: bool) -> None:
        """Stretch the unit cylinder from the eyes to the gazed point."""
        if not self._draw_ray:
            return
        self._ensure_visuals()
        if self._ray_xform_op is None:
            return

        direction = end - origin
        length = direction.GetLength()
        if length <= 1e-6:
            self._set_visuals_visible(False)
            return
        direction = direction / length
        # Start the drawn cylinder ahead of the eyes (capped at half the ray
        # length) so the beam never sits inside the user's face in first person.
        start_offset = min(self._ray_visual_start_offset, length * 0.5)
        start = origin + direction * start_offset
        segment_length = length - start_offset
        midpoint = start + direction * (segment_length * 0.5)

        # Unit-height Z cylinder -> scale to length, rotate +Z onto the gaze
        # direction, then translate to the segment midpoint (Gf row-vector
        # convention: leftmost matrix applies first).
        scale_m = Gf.Matrix4d(1.0).SetScale(Gf.Vec3d(1.0, 1.0, segment_length))
        rot_m = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), direction))
        trans_m = Gf.Matrix4d(1.0).SetTranslate(midpoint)
        try:
            self._ray_xform_op.Set(scale_m * rot_m * trans_m)
            if self._marker_translate_op is not None:
                self._marker_translate_op.Set(end)
            marker_prim = omni.usd.get_context().get_stage().GetPrimAtPath(f"{self._ray_root_path}/HitMarker")
            if marker_prim.IsValid():
                imageable = UsdGeom.Imageable(marker_prim)
                imageable.MakeVisible() if hit_valid else imageable.MakeInvisible()
            self._set_visuals_visible(True)
        except Exception:
            # Stage prims can vanish during a clear; recreate on the next update.
            self._visuals_created = False
            self._ray_xform_op = None
            self._marker_translate_op = None

    # ------------------------------------------------------------- highlight

    def _update_highlight(self, hit_path: str | None) -> None:
        """Tint the gazed sample box; restore the previous box when gaze moves on."""
        if not self._highlight_gazed_boxes:
            return
        target = hit_path if (hit_path and self._sample_box_root_marker in hit_path) else None
        if target == self._highlighted_path:
            return
        stage = omni.usd.get_context().get_stage()

        # Restore the previously highlighted box.
        if self._highlighted_path is not None:
            try:
                prim = stage.GetPrimAtPath(self._highlighted_path)
                if prim.IsValid() and self._highlight_original_color is not None:
                    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([self._highlight_original_color])
            except Exception:
                pass
            self._highlighted_path = None
            self._highlight_original_color = None

        # Tint the newly gazed box.
        if target is not None:
            try:
                prim = stage.GetPrimAtPath(target)
                if prim.IsValid():
                    color_attr = UsdGeom.Gprim(prim).GetDisplayColorAttr()
                    existing = color_attr.Get()
                    self._highlight_original_color = existing[0] if existing else None
                    color_attr.Set([self._highlight_color])
                    self._highlighted_path = target
            except Exception:
                self._highlighted_path = None
                self._highlight_original_color = None

    # -------------------------------------------------------- terminal events

    def _friendly_object_name(self, hit_path: str | None) -> str:
        """Human-readable label for a gaze target, for the terminal event line."""
        if not hit_path:
            return "nothing"
        if self._sample_box_root_marker in hit_path:
            return f"sample box {hit_path.rsplit('/', 1)[-1]}"
        if "ground" in hit_path.lower():
            return "ground"
        return hit_path

    def _report_gaze_target_change(
        self, hit_valid: bool, hit_path: str | None, hit_position: Gf.Vec3d | None, hit_distance: float | None
    ) -> None:
        """Print one terminal line whenever the gazed object changes.

        Fires only on transitions (box -> ground -> nothing -> other box...), never
        per-frame, so the terminal stays readable while still showing live what the
        user is looking at. print() is used instead of carb.log_info so the line is
        visible at the default console verbosity.
        """
        target = hit_path if hit_valid else None
        if target == self._last_reported_target:
            return
        self._last_reported_target = target

        if hit_valid and hit_position is not None:
            distance_text = f"{hit_distance:.2f} m away" if hit_distance is not None else "distance unknown"
            print(
                f"[EyeGaze] looking at {self._friendly_object_name(hit_path)} "
                f"@ ({hit_position[0]:.2f}, {hit_position[1]:.2f}, {hit_position[2]:.2f}) m, {distance_text}",
                flush=True,
            )
        else:
            print(f"[EyeGaze] gaze hits nothing within {self._max_distance:.0f} m", flush=True)

    # ---------------------------------------------------------------- update

    def update(self, dt: float) -> None:
        """Poll eye gaze, raycast, refresh the ray, and report target changes.

        Call once per physics step. Work is internally rate-limited to ~50 Hz;
        between working updates the previous `latest` sample stays available to
        the data logger, so gaze.csv rows are never starved by the rate limit.
        """
        self._call_counter += 1
        if self._call_counter % self._update_every_n_calls != 0:
            return

        ray = self._read_gaze_ray()
        if ray is None:
            self._failed_updates += 1
            # ~10 s of no eye data at 50 Hz working rate: tell the user once why.
            if not self._missing_warned and not self._status_logged and self._failed_updates > 500:
                self._missing_warned = True
                carb.log_warn(
                    "EyeGazeTracker: no OpenXR eye-gaze device found "
                    f"(tried {', '.join(self.UNIFIED_DEVICES)}). The streaming app must forward "
                    "eye tracking as XR_EXT_eye_gaze_interaction: Quest Link / Air Link never do; "
                    "use Steam Link ('Share eye tracking data' enabled) or Virtual Desktop "
                    "('Forward tracking data' enabled) with SteamVR as the OpenXR runtime. "
                    "Falling back to HMD-forward gaze. "
                    f"XR input devices visible this session: {self._describe_input_devices()}"
                )
            if self.latest.valid:
                # Tracking just dropped out: clear state so re-acquisition reports fresh.
                self.latest = GazeSample()
                self._set_visuals_visible(False)
                self._update_highlight(None)
                self._last_reported_target = "<unset>"
            return

        origin, direction = ray
        hit_position, hit_distance, hit_path = self._raycast(origin, direction)
        hit_valid = hit_position is not None

        self.latest = GazeSample(
            valid=True,
            origin=origin,
            direction=direction,
            hit_valid=hit_valid,
            hit_position=hit_position,
            hit_distance=hit_distance,
            hit_object_path=hit_path,
        )

        ray_end = hit_position if hit_valid else origin + direction * self._max_distance
        self._update_ray_visual(origin, ray_end, hit_valid)
        self._update_highlight(hit_path if hit_valid else None)
        self._report_gaze_target_change(hit_valid, hit_path, hit_position, hit_distance)

    # --------------------------------------------------------------- cleanup

    def cleanup(self) -> None:
        """Remove the ray visuals and restore any tinted box. Safe to call twice."""
        self._update_highlight(None)
        try:
            stage = omni.usd.get_context().get_stage()
            if stage is not None and stage.GetPrimAtPath(self._ray_root_path).IsValid():
                stage.RemovePrim(self._ray_root_path)
        except Exception:
            pass
        self._visuals_created = False
        self._ray_xform_op = None
        self._marker_translate_op = None
        self._last_reported_target = "<unset>"
        self.latest = GazeSample()
