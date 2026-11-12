# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extend the existing stereo gaze panel with an explicit facial-mode toggle."""

from __future__ import annotations

import math

from isaacsim.robot.policy.examples.interactive.humanoid.gaze_controls_scene import GazeControlsScene
from pxr import Gf, Usd, UsdGeom, UsdShade


class FacialControlsScene(GazeControlsScene):
    """Keep the four gaze buttons and add a separate, gaze-operated face switch.

    The switch sits beneath the right-hand buttons and leaves the center clear.
    Geometry is created once. Runtime updates only change preauthored materials,
    transforms, and ring topology, following the existing headset HUD strategy.
    """

    FACE_CENTER = (0.55, -0.49)
    FACE_HALF_HEIGHT = 0.085
    _STATUS_DESCRIPTIONS = {
        **GazeControlsScene._STATUS_DESCRIPTIONS,
        "face_idle": ("FACE READY", "ready"),
        "face_neutral": ("RELAX FACE", "ready"),
        "face_confirm": ("HOLD FACE", "dwelling"),
        "face_unavailable": ("NO FACE DATA", "no_eye"),
        "blink_pause": ("OPEN EYES", "dwelling"),
        "face_pause_turn_left": ("OPEN R EYE", "dwelling"),
        "face_pause_turn_right": ("OPEN L EYE", "dwelling"),
        "face_confirm_turn_left": ("HOLD L WINK", "dwelling"),
        "face_confirm_turn_right": ("HOLD R WINK", "dwelling"),
        "face_confirm_forward": ("HOLD MOUTH", "dwelling"),
        "face_confirm_back": ("HOLD PUCKER", "dwelling"),
        "face_confirm_strafe_left": ("HOLD L LIPS", "dwelling"),
        "face_confirm_strafe_right": ("HOLD R LIPS", "dwelling"),
    }
    _ACTION_LABELS = {
        **GazeControlsScene._ACTION_LABELS,
        "strafe_left": "STEP LEFT",
        "strafe_right": "STEP RIGHT",
    }

    def create(self, stage: Usd.Stage) -> None:
        """Create gaze controls plus preauthored face on/off labels and dwell ring."""
        if self._stage is stage and self._is_created():
            return
        super().create(stage)
        self._face_display = None
        self._face_ring_count = 0
        # Use an independent material for button text. Recoloring the base
        # `text` material would also dim the toggle and other operating hints.
        self._materials["face_gaze_text"] = self._make_material("face_gaze_text", (0.9, 0.95, 1.0))
        for button in self.BUTTON_CENTERS:
            if button != "back" or self.backwards_enabled:
                label = stage.GetPrimAtPath(f"{self.root_path}/{button}/Label")
                UsdShade.MaterialBindingAPI(label).Bind(self._materials["face_gaze_text"])
        self._face_button_color_inputs = {}
        for name, color in {**self._BUTTON_COLORS, "face_gaze_text": (0.9, 0.95, 1.0)}.items():
            shader = UsdShade.Shader(stage.GetPrimAtPath(f"{self._materials[name].GetPath()}/Shader"))
            self._face_button_color_inputs[name] = (
                (shader.GetInput("diffuseColor"), shader.GetInput("emissiveColor")),
                Gf.Vec3f(*color),
            )
        x, y = self.FACE_CENTER
        self._backing_plate(self.root_path + "/FaceToggle", x, y, self.HALF_WIDTH, self.FACE_HALF_HEIGHT)
        for enabled, label in ((False, "FACE OFF"), (True, "FACE ON")):
            name = "face_on" if enabled else "face_off"
            self._materials[name] = self._make_material(
                name, (0.2, 1.0, 0.35) if enabled else (0.9, 0.95, 1.0), opacity=float(not enabled)
            )
            mesh = self._text(f"{self.root_path}/{name}", label, x, y + 0.022, 0.005, name)
            self._register_mask_transform(mesh, name)
        self._text(self.root_path + "/FaceToggleHint", "LOOK TO SWITCH", x, y - 0.023, 0.0034, "text")
        self._face_ring = self._mesh(
            self.root_path + "/FaceToggleProgress", self._ring_points(x, y, 0.092, 0.098), [], "dwelling"
        )
        self.update_face_toggle(False, 0.0)

    def hit_test_face_toggle(self, origin: object, direction: object, max_distance: float | None = None) -> bool:
        """Test the same displayed camera frame and occlusion rules as the gaze buttons."""
        if self._camera_inverse is None or not self._is_created():
            return False
        try:
            if len(origin) != 3 or len(direction) != 3:
                return False
            origin, direction = Gf.Vec3d(*origin), Gf.Vec3d(*direction)
            if not all(math.isfinite(value) for value in (*origin, *direction)):
                return False
            length = direction.GetLength()
            if length <= 1e-9:
                return False
            if max_distance is not None and (not math.isfinite(max_distance) or max_distance <= 0):
                return False
            local_origin = self._camera_inverse.Transform(origin)
            local_direction = self._camera_inverse.TransformDir(direction / length)
            if local_direction[2] >= -1e-9:
                return False
            distance = (-self.DEPTH - local_origin[2]) / local_direction[2]
            if distance <= 0 or (max_distance is not None and distance >= max_distance):
                return False
            hit = local_origin + distance * local_direction
            return bool(
                abs(hit[0] - self.FACE_CENTER[0]) <= self.HALF_WIDTH
                and abs(hit[1] - self.FACE_CENTER[1]) <= self.FACE_HALF_HEIGHT
            )
        except (TypeError, ValueError, RuntimeError):
            return False

    def update_face_toggle(self, enabled: bool, progress: float) -> None:
        """Display the actual mode and switch dwell progress without repeated stage edits."""
        if not self._is_created():
            return
        if self._face_display != enabled:
            self._set_opacity("face_on", enabled)
            self._set_opacity("face_off", not enabled)
            for inputs, original_color in self._face_button_color_inputs.values():
                color = Gf.Vec3f(*self._STATE_COLORS["disabled"]) if enabled else original_color
                for shader_input in inputs:
                    if shader_input.Get() != color:
                        shader_input.Set(color)
            self._face_display = enabled
        count = int(max(0.0, min(1.0, progress)) * self.RING_SEGMENTS) if math.isfinite(progress) else 0
        if count != self._face_ring_count:
            self._face_ring.GetFaceVertexCountsAttr().Set([4] * count)
            self._face_ring.GetFaceVertexIndicesAttr().Set(list(range(4 * count)))
            self._face_ring_count = count

    def update_facial(self, command: object) -> None:
        """Show facial state independently of eye closure, which is itself a control."""
        state = {
            "idle": "face_idle",
            "awaiting_neutral": "face_neutral",
            "awaiting_release": "face_neutral",
            "confirming": "face_confirm",
            "unavailable": "face_unavailable",
        }.get(command.state, command.state)
        if command.state == "confirming" and command.target in self._ACTION_LABELS:
            state = f"face_confirm_{command.target}"
        elif command.state == "blink_pause" and command.target in ("turn_left", "turn_right"):
            state = f"face_pause_{command.target}"
        self.update(
            {
                "state": state,
                "target": command.target,
                "progress": command.progress,
                "last_action": command.last_action,
            },
            eye_valid=True,
        )
