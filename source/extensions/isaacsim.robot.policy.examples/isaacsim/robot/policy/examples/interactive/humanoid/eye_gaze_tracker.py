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

"""Validated world-space gaze, scene selection, and visualization for humanoid VR.

The runtime's combined eye-gaze device supplies the ray when available. A
separately labeled HMD-forward ray is optional; it does not measure eye motion.
Pose validity is checked every physics tick and scene queries run at 50 Hz.
The nearest non-robot hit feeds visuals, shared package highlights, and the
recorder's latest GazeSample, including its actual source and sampling time.

Kit's binding manifest may name the device /user/eye/unified and its pose
'gaze_ext'; other combined-device/pose names are tried for compatible runtimes.
Per-eye view transforms are not treated as eye-gaze measurements. Runtime
support, streaming permissions, and headset calibration must be checked in a
live session; extension availability alone does not establish valid tracking.
See docs/humanoid-control.md for diagnostics and the validation procedure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import carb
import omni.usd
from isaacsim.robot.policy.examples.interactive.humanoid.material_highlights import MaterialHighlights
from isaacsim.robot.policy.examples.interactive.humanoid.xr_pose import read_world_pose
from pxr import Gf, Sdf, UsdGeom, UsdShade


@dataclass
class GazeSample:
    """Latest eye-gaze reading, in stage (virtual-world) coordinates.

    valid=False means neither a usable eye ray nor an enabled HMD fallback.
    source distinguishes eye_tracker from hmd_forward; sample_time records the
    simulation time of the query. hit_* fields require hit_valid=True.
    """

    valid: bool = False
    origin: Gf.Vec3d | None = None
    direction: Gf.Vec3d | None = None
    hit_valid: bool = False
    hit_position: Gf.Vec3d | None = None
    hit_distance: float | None = None
    hit_object_path: str | None = None
    source: str | None = None
    sample_time: float = 0.0


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
    #: "gaze_ext" first because that is the pose name in the bindings manifest this Kit
    #: build ships (xrmanifests/openxr_controller_bindings/eye_gaze_interaction_bindings.json:
    #: devicePath /user/eyes_ext -> deviceName /user/eye/unified, pose "gaze_ext").
    #: "gaze" stays as a fallback for runtimes that publish the short name.
    POSE_CANDIDATES = ("gaze_ext", "gaze", "")
    #: NOT a fallback: ``/user/eye/left`` and ``/user/eye/right`` exist in every session,
    #: with or without eye tracking, because XRCore creates them as the stereo DISPLAY
    #: view poses -- the log lists them as the input device handles of ``hmdLeft`` and
    #: ``hmdRight``. Captured together they carry the SAME rotation and differ only by the
    #: interpupillary distance, so using them would silently report head direction while
    #: claiming to be eye tracking. The head fallback below at least says what it is.
    #: Used when no eye device exists, so a ray is always drawn.
    HEAD_DEVICES = ("/user/head",)
    HMD_FALLBACK_ENABLED = True

    def __init__(
        self,
        xr_core,
        draw_ray: bool = True,
        highlight_gazed_boxes: bool = True,
        highlights: MaterialHighlights | None = None,
    ):
        self._highlights = highlights if highlights is not None else MaterialHighlights()
        self._xr_core = xr_core
        self._draw_ray = draw_ray
        self._highlight_gazed_boxes = highlight_gazed_boxes

        # --- tunables -------------------------------------------------------
        self._max_distance = 20.0  # m: raycast + drawn-ray length cap
        self._update_period = 1.0 / 50.0  # seconds between raycasts, independent of physics rate
        # 4 mm was invisible in practice. The ray IS drawn -- the session logs show it
        # resolving targets continuously -- but a 4 mm unlit cylinder in a dim warehouse,
        # seen from the robot's own head, is a hairline you will never notice. It is now
        # thick enough to read and, more importantly, emissive (see _ensure_glow_material).
        self._ray_radius = 0.012  # m: red ray cylinder thickness
        self._ray_visual_start_offset = 0.6  # m: drawn ray starts this far ahead of the eyes (comfort)
        # 9 cm was a balloon: rendered, it covered half the work surface and the 6x
        # emissive washed it from red to pink. A gaze indicator has to be findable
        # WITHOUT hiding the thing you are looking at.
        self._hit_marker_radius = 0.028  # m: sphere marking the gaze collision point
        # The red ball used to be hidden whenever the ray hit nothing, so looking at open
        # space made it vanish and the tracker looked broken. It now always rides the gaze --
        # parked at this distance when there is nothing to land on.
        self._marker_always_visible = True
        self._no_hit_marker_distance = 4.0  # m along the gaze ray when nothing is hit
        self._ray_color = Gf.Vec3f(1.0, 0.05, 0.05)  # red
        self._hit_marker_color = Gf.Vec3f(1.0, 0.08, 0.08)  # bright red; it is emissive now,
        # so a dark colour just reads black
        self._highlight_color = Gf.Vec3f(1.0, 0.85, 0.1)  # gazed sample-box tint
        self._robot_root_path = "/World/G1"  # own-body hits are re-cast past
        self._sample_box_root_marker = "/G1_SampleBoxes/"  # only these prims get tinted
        self._ray_root_path = "/World/G1_EyeGazeRay"

        # --- state ----------------------------------------------------------
        self.latest = GazeSample()
        self.gaze_source = None  # "eye_tracker" | "hmd_forward" | None
        self._call_counter = 0
        self._elapsed_time = 0.0
        self._next_update_time = 0.0
        self._missing_eye_time = 0.0
        self._logged_source = "<unset>"  # last source announced on the terminal
        self._missing_warned = False
        self._failed_updates = 0
        self._visuals_created = False
        self._ray_xform_op = None
        self._marker_translate_op = None
        self._highlighted_path = None
        self._highlight_original_color = None
        self._highlight_previous_material = None
        self._highlight_material_path = "/World/G1_GazeHighlightMaterial"
        self._ray_material_path = "/World/G1_GazeRayMaterial"
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

    def _read_pose_matrix(self, handles: tuple[str, ...]) -> Gf.Matrix4d | None:
        """Read a valid stage-space pose; physical-room coordinates are never raycast."""
        for handle in handles:
            try:
                device = self._xr_core.get_input_device(handle)
            except Exception:
                continue
            if device is None:
                continue
            candidates = self.POSE_CANDIDATES if handles == self.UNIFIED_DEVICES else ("", "head", "hmd")
            try:
                names = {str(name) for name in device.get_pose_names()}
            except Exception:
                names = set()
            for pose_name in candidates:
                if pose_name and names and pose_name not in names:
                    continue
                pose = read_world_pose(device, pose_name)
                if pose is not None:
                    return pose
        return None

    @staticmethod
    def _matrix_to_ray(pose: Gf.Matrix4d):
        """Split a pose matrix into (origin, unit direction); OpenXR gaze looks down -Z."""
        origin = Gf.Vec3d(pose.ExtractTranslation())
        direction = pose.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
        length = direction.GetLength()
        if not all(math.isfinite(float(v)) for v in (*origin, *direction, length)) or length <= 1e-6:
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
        self.gaze_source = None
        unified_pose = self._read_pose_matrix(self.UNIFIED_DEVICES)
        ray = self._matrix_to_ray(unified_pose) if unified_pose is not None else None
        if ray is not None:
            self.gaze_source = "eye_tracker"
            self._log_source("real eye tracking (unified gaze device)")
            return ray

        # Fall back to where the head is pointing. Without this the ray was simply not
        # drawn when no eye device was present -- the docstring promised a fallback but
        # only gaze.csv had one, so a user without eye tracking saw nothing at all and
        # had no way to tell whether the feature was broken or just unavailable.
        if not self.HMD_FALLBACK_ENABLED:
            return None
        head_pose = self._read_pose_matrix(self.HEAD_DEVICES)
        ray = self._matrix_to_ray(head_pose) if head_pose is not None else None
        if ray is None:
            return None
        self.gaze_source = "hmd_forward"
        self._log_source(
            "HMD-forward direction (no eye-gaze device found -- the ray follows your head, "
            "not your eyes). XR devices: " + self._describe_input_devices()
        )
        return ray

    def _log_missing_devices(self) -> None:
        """Report absent eye measurements without assuming a particular hardware fault."""
        print(
            "[EyeGaze] no valid combined eye-gaze pose; source=" + str(self.gaze_source) + ".\n"
            "[EyeGaze] Check headset eye calibration, streaming-app permissions, the active\n"
            "[EyeGaze] OpenXR runtime, and whether Kit receives a valid unified gaze pose.\n"
            "[EyeGaze] HMD-forward, when available, follows the head and does not measure eye motion.",
            flush=True,
        )
        print(f"[EyeGaze] XR devices this session: {self._describe_input_devices()}", flush=True)

    def describe(self) -> str:
        """One-line status for the live Kit console (isaacsim.code_editor.python_server).

        Returns what is needed to tell "eye tracking is not reaching Kit" apart from
        "eye tracking works but the ray is drawn in the wrong place".
        """
        latest = self.latest
        origin = tuple(round(float(v), 3) for v in latest.origin) if latest.origin else None
        direction = tuple(round(float(v), 3) for v in latest.direction) if latest.direction else None
        return (
            f"source={self.gaze_source} valid={latest.valid} origin={origin} dir={direction} "
            f"hit={latest.hit_object_path} visuals={self._visuals_created} "
            f"failed_updates={self._failed_updates} devices={self._describe_input_devices()}"
        )

    def _log_source(self, description: str) -> None:
        """Announce where the drawn gaze ray comes from, once per CHANGE of source.

        This used to latch on the first report, which hid the case that matters most on
        this rig: the session starts before the eye tracker has locked on, logs
        "HMD-forward", then silently upgrades to real eye tracking (or silently drops
        back to the head). The terminal then disagreed with what the ray was doing.
        """
        if self.gaze_source == self._logged_source:
            return
        self._logged_source = self.gaze_source
        message = f"[EyeGaze] ray source: {description}"
        print(message, flush=True)
        carb.log_info(message)

    # --------------------------------------------------------------- raycast

    def _raycast(self, origin: Gf.Vec3d, direction: Gf.Vec3d):
        """Select the closest non-robot collider from an all-hit scene query.

        Cast from the eye, not 35 cm in front of it. Filtering every returned hit
        preserves nearby objects and works through multiple self-colliders without
        advancing the ray into or through a valid target. Results are not sorted.
        """
        best = (None, None, None)

        def on_hit(hit):
            nonlocal best
            path = self._decode_hit_path(hit)
            if path and (path == self._robot_root_path or path.startswith(self._robot_root_path + "/")):
                return True
            field = hit.get if isinstance(hit, dict) else lambda name, default=None: getattr(hit, name, default)
            try:
                distance = float(field("distance", math.inf))
                if not math.isfinite(distance) or not 0.0 <= distance <= self._max_distance:
                    return True
                value = field("position")
                position = Gf.Vec3d(*[float(v) for v in value]) if value is not None else origin + direction * distance
                if not all(math.isfinite(float(v)) for v in position):
                    return True
                if best[1] is None or distance < best[1]:
                    best = (position, distance, path)
            except (TypeError, ValueError, RuntimeError):
                pass
            return True

        try:
            from omni.physics.core import get_physics_scene_query_interface

            query = get_physics_scene_query_interface()
            query.raycast_all(tuple(origin), tuple(direction), self._max_distance, on_hit, False)
        except Exception as error:
            self._warn_raycast_broken_once(error)
        return best

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
        """Decode a rigid body path, or the collision path for static scene geometry."""
        field = hit.get if isinstance(hit, dict) else lambda name, default=None: getattr(hit, name, default)
        for name in ("rigid_body", "collision"):
            value = field(name)
            if isinstance(value, (str, Sdf.Path)) and str(value):
                return str(value)
            if value:
                try:
                    from pxr import PhysicsSchemaTools

                    path = str(PhysicsSchemaTools.intToSdfPath(value))
                    if path:
                        return path
                except (TypeError, ValueError, RuntimeError):
                    continue
        return None

    # --------------------------------------------------------------- visuals

    def _ensure_glow_material(self):
        """Create (once) the unlit-looking emissive material the ray and marker are bound to.

        ``displayColor`` alone is not enough: in the RTX renderer a gprim with no bound
        material still gets a default lit one, so the ray was being shaded by the
        warehouse lighting and came out near-black. The gaze highlight already had to
        learn this; the ray itself never did, which is why it was drawn every frame and
        still could not be seen.
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._ray_material_path)
        if prim.IsValid():
            return UsdShade.Material(prim)
        try:
            material = UsdShade.Material.Define(stage, self._ray_material_path)
            shader = UsdShade.Shader.Define(stage, f"{self._ray_material_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(self._ray_color)
            # Emissive well above 1.0 so the ray reads as a glowing beam rather than a
            # painted rod, in bright and dim parts of the warehouse alike.
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(self._ray_color[0] * 2.0, self._ray_color[1] * 2.0, self._ray_color[2] * 2.0)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return material
        except Exception as e:
            carb.log_warn(f"EyeGazeTracker: could not create the gaze-ray material: {e}")
            return None

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

        glow = self._ensure_glow_material()
        if glow is not None:
            for target in (ray.GetPrim(), marker.GetPrim()):
                try:
                    UsdShade.MaterialBindingAPI(target).Bind(glow)
                except Exception as e:
                    carb.log_warn(f"EyeGazeTracker: could not bind the gaze-ray material: {e}")

        self._visuals_created = True
        print(
            f"[EyeGaze] gaze ray visuals ready under {self._ray_root_path} "
            f"(radius {self._ray_radius * 1000:.0f} mm, marker {self._hit_marker_radius * 100:.0f} cm, emissive)",
            flush=True,
        )
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
        marker_visible = hit_valid or self._marker_always_visible
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
                imageable.MakeVisible() if marker_visible else imageable.MakeInvisible()
            self._set_visuals_visible(True)
        except Exception:
            # Stage prims can vanish during a clear; recreate on the next update.
            self._visuals_created = False
            self._ray_xform_op = None
            self._marker_translate_op = None

    # ------------------------------------------------------------- highlight

    def _ensure_highlight_material(self):
        """Create (once) the emissive material used to mark the gazed package.

        Tinting via ``displayColor`` alone stopped working when the packages became real
        warehouse crates: those carry bound PBR materials, and a bound material wins over
        displayColor in the RTX renderer. Overriding the *binding* is the only tint that
        survives regardless of what the asset ships with.
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._highlight_material_path)
        if prim.IsValid():
            return UsdShade.Material(prim)
        try:
            material = UsdShade.Material.Define(stage, self._highlight_material_path)
            shader = UsdShade.Shader.Define(stage, f"{self._highlight_material_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(self._highlight_color)
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(self._highlight_color)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return material
        except Exception as e:
            carb.log_warn(f"EyeGazeTracker: could not create the highlight material: {e}")
            return None

    def _update_highlight(self, hit_path: str | None) -> None:
        """Highlight the package root, including hits on its referenced child meshes."""
        target = None
        if self._highlight_gazed_boxes and hit_path and self._sample_box_root_marker in hit_path:
            prefix, suffix = hit_path.split(self._sample_box_root_marker, 1)
            target = prefix + self._sample_box_root_marker + suffix.split("/", 1)[0]
        stage = omni.usd.get_context().get_stage()
        material = self._ensure_highlight_material() if target and stage is not None else None
        self._highlights.set(stage, "gaze", target, material, priority=10)
        self._highlighted_path = target

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
        if not math.isfinite(dt) or dt <= 0.0:
            return
        self._elapsed_time += dt
        # Check validity every step even when the expensive raycast is not due.
        ray = self._read_gaze_ray()
        if self.gaze_source == "eye_tracker":
            self._missing_eye_time = 0.0
            self._missing_warned = False
        else:
            self._missing_eye_time += dt
            if not self._missing_warned and self._missing_eye_time >= 3.0:
                self._missing_warned = True
                self._log_missing_devices()
        if ray is None:
            self._failed_updates += 1
            # ~3 s of no gaze data at the 50 Hz working rate: say why, on the terminal. This
            # used to need 10 s AND no source ever having been logged, so the case that
            # matters most -- nothing works at all -- stayed silent the longest.
            if self.latest.valid:
                # Tracking just dropped out: clear state so re-acquisition reports fresh.
                self.latest = GazeSample()
                self._set_visuals_visible(False)
                self._update_highlight(None)
                self._last_reported_target = "<unset>"
            self._next_update_time = self._elapsed_time
            return

        self._failed_updates = 0
        source_changed = self.latest.source != self.gaze_source
        if self._elapsed_time + 1e-9 < self._next_update_time and not source_changed:
            return
        # Retain the fractional remainder at non-divisor physics rates (e.g. 120 Hz).
        periods = max(1, math.floor((self._elapsed_time - self._next_update_time + 1e-9) / self._update_period) + 1)
        if self._next_update_time == 0.0 or source_changed:
            self._next_update_time = self._elapsed_time + self._update_period
        else:
            self._next_update_time += periods * self._update_period

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
            source=self.gaze_source,
            sample_time=self._elapsed_time,
        )

        # A no-hit ray used to run the full 20 m, which put the ball so far away it was a
        # dot on the horizon. Park it at arm's-reach-plus instead, so "where am I looking"
        # always reads clearly.
        no_hit_distance = self._no_hit_marker_distance if self._marker_always_visible else self._max_distance
        ray_end = hit_position if hit_valid else origin + direction * no_hit_distance
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
        self.gaze_source = None
        self._elapsed_time = 0.0
        self._next_update_time = 0.0
        self._missing_eye_time = 0.0
        self._missing_warned = False
        self._logged_source = "<unset>"
