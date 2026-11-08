# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draw stereo-visible gaze buttons without introducing physics or XR input bindings.

The caller supplies the final robot-head camera transform and validated eye ray.
All hit tests use the same cached transform as the displayed geometry. This is
depth-tested scene geometry, not an always-on-top overlay; an optional nearest
scene-hit distance prevents activating a button hidden behind an object.
"""

from __future__ import annotations

import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

__all__ = ["GazeControlsScene"]

# A small, dependency-free block font. Each label is one mesh, not one prim per pixel.
_FONT = {
    "A": (14, 17, 17, 31, 17, 17, 17),
    "B": (30, 17, 17, 30, 17, 17, 30),
    "C": (14, 17, 16, 16, 16, 17, 14),
    "D": (30, 17, 17, 17, 17, 17, 30),
    "E": (31, 16, 16, 30, 16, 16, 31),
    "F": (31, 16, 16, 30, 16, 16, 16),
    "G": (14, 17, 16, 23, 17, 17, 15),
    "H": (17, 17, 17, 31, 17, 17, 17),
    "I": (31, 4, 4, 4, 4, 4, 31),
    "J": (7, 2, 2, 2, 18, 18, 12),
    "K": (17, 18, 20, 24, 20, 18, 17),
    "L": (16, 16, 16, 16, 16, 16, 31),
    "M": (17, 27, 21, 21, 17, 17, 17),
    "N": (17, 25, 21, 19, 17, 17, 17),
    "O": (14, 17, 17, 17, 17, 17, 14),
    "P": (30, 17, 17, 30, 16, 16, 16),
    "Q": (14, 17, 17, 17, 21, 18, 13),
    "R": (30, 17, 17, 30, 20, 18, 17),
    "S": (15, 16, 16, 14, 1, 1, 30),
    "T": (31, 4, 4, 4, 4, 4, 4),
    "U": (17, 17, 17, 17, 17, 17, 14),
    "V": (17, 17, 17, 17, 17, 10, 4),
    "W": (17, 17, 17, 21, 21, 21, 10),
    "X": (17, 17, 10, 4, 10, 17, 17),
    "Y": (17, 17, 10, 4, 4, 4, 4),
    "Z": (31, 1, 2, 4, 8, 16, 31),
    " ": (0, 0, 0, 0, 0, 0, 0),
}


class GazeControlsScene:
    """Create four large side buttons in the supplied camera's local frame.

    Centers lie about 29 degrees either side of the view, with an unobstructed
    central region. Local camera axes are +X right, +Y up, and -Z forward.
    This helper never issues robot commands or accepts head-forward fallback.

    Args:
        backwards_enabled: Whether Back can be selected; otherwise it remains visibly off.
        preview: Whether the heading and last-action feedback describe non-actuating preview.
        root_path: Dedicated absolute prim path owned by this helper.

    Raises:
        ValueError: The root path is not an absolute non-root prim path.

    Example:

        .. code-block:: python

            scene = GazeControlsScene(backwards_enabled=False)
            scene.create(stage)
            scene.set_camera_pose(camera_pose)
    """

    BUTTON_CENTERS = {
        "forward": (-0.55, 0.22),
        "back": (-0.55, -0.22),
        "turn_left": (0.55, 0.22),
        "turn_right": (0.55, -0.22),
    }
    #: Camera-local centers, in meters, with forward/back on the left and turns on the right.
    DEPTH = 1.0
    #: Positive distance ahead of the camera, in meters.
    HALF_WIDTH = 0.17
    HALF_HEIGHT = 0.155
    RING_SEGMENTS = 64
    #: Smooth progress resolution; no new prims are authored during dwell updates.
    _BUTTON_COLORS = {
        "forward": (0.15, 0.78, 0.95),
        "back": (0.95, 0.55, 0.15),
        "turn_left": (0.65, 0.4, 0.95),
        "turn_right": (0.2, 0.55, 1.0),
    }
    _STATE_COLORS = {
        "ready": (0.25, 0.7, 0.85),
        "preview": (0.2, 0.55, 1.0),
        "dwelling": (1.0, 0.7, 0.15),
        "active": (0.2, 1.0, 0.35),
        "stopping": (1.0, 0.55, 0.15),
        "disabled": (0.28, 0.3, 0.33),
        "no_eye": (1.0, 0.25, 0.2),
        "fault": (1.0, 0.2, 0.2),
    }
    _STATUS_DESCRIPTIONS = {
        "idle": ("READY", "ready"),
        "dwelling": ("DWELL", "dwelling"),
        "preview": ("PREVIEW", "preview"),
        "active": ("RUNNING", "active"),
        "stopping": ("STOPPING", "stopping"),
        "awaiting_release": ("LOOK AWAY", "ready"),
        "unavailable": ("DISABLED", "disabled"),
        "fault": ("FAULT", "fault"),
        "no_eye": ("NO EYE", "no_eye"),
        "controller": ("STICKS", "active"),
        "controller_preview": ("STICK PREVIEW", "preview"),
        "controller_wait": ("CENTER STICKS", "ready"),
    }
    _ACTION_LABELS = {"forward": "FORWARD", "back": "BACK", "turn_left": "TURN LEFT", "turn_right": "TURN RIGHT"}

    def __init__(
        self, *, backwards_enabled: bool = True, preview: bool = True, root_path: str = "/World/G1_GazeControls"
    ) -> None:
        path = Sdf.Path(root_path)
        if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
            raise ValueError("Gaze controls require an absolute non-root prim path")
        self.backwards_enabled = bool(backwards_enabled)
        self.preview = bool(preview)
        self.root_path = str(path)
        self._stage = None
        self._transform_op = None
        self._camera_inverse = None
        self._materials = {}
        self._opacity_inputs = {}
        self._mask_transform_ops = {}
        self._rings = {}
        self._status_label = None
        self._status_meshes = {}
        self._status_cards = {}
        self._feedback_label = None
        self._feedback_meshes = {}
        self._feedback_prefix = None
        self._feedback_plate = None
        self._last_display = None
        self._last_feedback = None
        self._ring_counts = {}
        self._ring_materials = {}

    def create(self, stage: Usd.Stage) -> None:
        """Author all nonphysical meshes and materials before updating their state.

        Args:
            stage: Stage that will render the buttons in both headset eyes.

        Raises:
            ValueError: The stage is missing or the owned path already exists.

        Example:

            .. code-block:: python

                scene.create(stage)
        """
        if stage is None:
            raise ValueError("A valid USD stage is required")
        if self._stage is stage and self._is_created():
            return
        self.cleanup()
        if stage.GetPrimAtPath(self.root_path).IsValid():
            raise ValueError("Gaze controls root already belongs to another scene")
        self._stage = stage
        root = UsdGeom.Xform.Define(stage, self.root_path)
        self._transform_op = root.AddTransformOp()
        UsdGeom.Imageable(root).MakeInvisible()
        colors = {**self._STATE_COLORS, **self._BUTTON_COLORS, "panel": (0.025, 0.04, 0.065), "text": (0.9, 0.95, 1.0)}
        for name, color in colors.items():
            self._materials[name] = self._make_material(name, color)
        for button, (x, y) in self.BUTTON_CENTERS.items():
            path = f"{self.root_path}/{button}"
            UsdGeom.Xform.Define(stage, path)
            self._mesh(
                path + "/Panel",
                [
                    (x - self.HALF_WIDTH, y - self.HALF_HEIGHT, -self.DEPTH),
                    (x + self.HALF_WIDTH, y - self.HALF_HEIGHT, -self.DEPTH),
                    (x + self.HALF_WIDTH, y + self.HALF_HEIGHT, -self.DEPTH),
                    (x - self.HALF_WIDTH, y + self.HALF_HEIGHT, -self.DEPTH),
                ],
                [4],
                "panel",
            )
            disabled = button == "back" and not self.backwards_enabled
            self._text(
                path + "/Label", self._ACTION_LABELS[button], x, y - 0.092, 0.0055, "disabled" if disabled else "text"
            )
            self._text(
                path + "/Arrow",
                {"forward": "F", "back": "B", "turn_left": "L", "turn_right": "R"}[button],
                x,
                y + 0.027,
                0.008,
                "disabled" if disabled else button,
            )
            points = self._ring_points(x, y + 0.027, 0.071, 0.082)
            self._mesh(path + "/Ring", points, [4] * self.RING_SEGMENTS, "disabled" if disabled else button)
            progress = self._mesh(path + "/Progress", [(a, b, c + 0.001) for a, b, c in points], [], "dwelling")
            self._rings[button] = progress
            self._ring_counts[button] = 0
            self._ring_materials[button] = "dwelling"
            if disabled:
                self._text(path + "/Disabled", "REVERSE OFF", x, y + 0.122, 0.0035, "disabled")
        self._backing_plate(self.root_path + "/TitleBacking", -0.55, 0.465, 0.18, 0.04)
        self._backing_plate(self.root_path + "/StatusBacking", 0.55, 0.465, 0.18, 0.04)
        self._materials["feedback_backing"] = self._make_material("feedback_backing", colors["panel"], opacity=0.0)
        self._feedback_plate = self._backing_plate(
            self.root_path + "/FeedbackBacking", -0.55, -0.49, 0.18, 0.06, material="feedback_backing"
        )
        self._register_mask_transform(self._feedback_plate, "feedback_backing")
        self._text(
            self.root_path + "/Title",
            "PREVIEW ONLY" if self.preview else "LIVE WALK",
            -0.55,
            0.465,
            0.005,
            "preview" if self.preview else "active",
        )
        # Fabric/RTX can retain stale topology or visibility for scene text.
        # Keep every label's geometry and binding fixed. Independent masked
        # materials use the same shader-input updates as `MaterialHighlights`.
        # Inactive labels also move behind the camera through preauthored ops.
        status_root = UsdGeom.Xform.Define(stage, self.root_path + "/Status")
        self._status_label = status_root.GetPrim().CreateAttribute(
            "gazeControls:label", Sdf.ValueTypeNames.String, custom=True
        )
        self._status_label.Set("NO EYE")
        UsdGeom.Xform.Define(stage, self.root_path + "/StatusCards")
        for state, (label, material) in self._STATUS_DESCRIPTIONS.items():
            # A paired opaque card changes geometry across the full status area,
            # including pixels vacated when the previous word was longer.
            card_material = f"status_card_{state}"
            self._materials[card_material] = self._make_material(
                card_material, colors["panel"], opacity=1.0 if state == "no_eye" else 0.0
            )
            card = self._backing_plate(
                f"{self.root_path}/StatusCards/{state}",
                0.55,
                0.465,
                0.18,
                0.04,
                material=card_material,
                depth_offset=0.001,
            )
            self._register_mask_transform(card, card_material)
            self._status_cards[state] = card
            material_name = f"status_{state}"
            self._materials[material_name] = self._make_material(
                material_name, colors[material], opacity=1.0 if state == "no_eye" else 0.0
            )
            mesh = self._text(f"{self.root_path}/Status/{state}", label, 0.55, 0.465, 0.0055, material_name)
            self._register_mask_transform(mesh, material_name)
            self._status_meshes[state] = mesh
        self._last_display = self._STATUS_DESCRIPTIONS["no_eye"]
        feedback_material = "preview" if self.preview else "text"
        self._materials["feedback_prefix"] = self._make_material(
            "feedback_prefix", colors[feedback_material], opacity=0.0
        )
        self._feedback_prefix = self._text(
            self.root_path + "/FeedbackPrefix",
            "WOULD" if self.preview else "LAST",
            -0.55,
            -0.465,
            0.0055,
            "feedback_prefix",
        )
        self._register_mask_transform(self._feedback_prefix, "feedback_prefix")
        feedback_root = UsdGeom.Xform.Define(stage, self.root_path + "/FeedbackAction")
        self._feedback_label = feedback_root.GetPrim().CreateAttribute(
            "gazeControls:label", Sdf.ValueTypeNames.String, custom=True
        )
        self._feedback_label.Set("")
        for action, label in self._ACTION_LABELS.items():
            material_name = f"feedback_{action}"
            self._materials[material_name] = self._make_material(material_name, colors[feedback_material], opacity=0.0)
            mesh = self._text(f"{self.root_path}/FeedbackAction/{action}", label, -0.55, -0.515, 0.0055, material_name)
            self._register_mask_transform(mesh, material_name)
            self._feedback_meshes[action] = mesh
        self._last_feedback = ""

    def set_camera_pose(self, camera_pose: Gf.Matrix4d) -> None:
        """Place visuals and cache the identical rigid transform for hit tests.

        Args:
            camera_pose: Final camera-to-world matrix, including comfort/head rotation.

        Raises:
            ValueError: The matrix is nonfinite, nonrigid, or malformed.
            RuntimeError: The scene has not been created or has been deleted.

        Example:

            .. code-block:: python

                scene.set_camera_pose(camera_pose)
        """
        if not self._is_created():
            self._camera_inverse = None
            raise RuntimeError("Create the gaze controls before setting their camera")
        visibility = UsdGeom.Imageable(self._stage.GetPrimAtPath(self.root_path)).GetVisibilityAttr()
        try:
            if len(camera_pose) != 4 or any(len(row) != 4 for row in camera_pose):
                raise ValueError("Camera transform must be 4 by 4")
            pose = Gf.Matrix4d(camera_pose)
            if not all(math.isfinite(pose[i][j]) for i in range(4) for j in range(4)):
                raise ValueError("Camera transform must be finite")
            rotation = pose.ExtractRotationMatrix()
            if abs(rotation.GetDeterminant() - 1.0) > 1e-5:
                raise ValueError("Camera transform must be rigid")
            if (
                any(
                    abs(sum(rotation[i][k] * rotation[j][k] for k in range(3)) - (i == j)) > 1e-5
                    for i in range(3)
                    for j in range(3)
                )
                or any(abs(pose[i][3]) > 1e-8 for i in range(3))
                or abs(pose[3][3] - 1) > 1e-8
            ):
                raise ValueError("Camera transform must be rigid and affine")
            if self._transform_op.Get() != pose:
                self._transform_op.Set(pose)
            self._camera_inverse = pose.GetInverse()
            if visibility.Get() != UsdGeom.Tokens.inherited:
                visibility.Set(UsdGeom.Tokens.inherited)
        except (TypeError, ValueError, RuntimeError) as error:
            self._camera_inverse = None
            if visibility.Get() != UsdGeom.Tokens.invisible:
                visibility.Set(UsdGeom.Tokens.invisible)
            if isinstance(error, ValueError):
                raise
            raise ValueError("Invalid gaze control camera transform") from error

    def hit_test(self, origin: object, direction: object, max_distance: float | None = None) -> str | None:
        """Intersect a world-space eye ray with visible, enabled button rectangles.

        Args:
            origin: Eye ray origin in stage meters.
            direction: Nonzero eye ray direction; normalization is handled here.
            max_distance: Nearest occluder distance from that origin, in meters.

        Returns:
            Button ID, or None for invalid, occluded, backward, disabled, or gap hits.

        Example:

            .. code-block:: python

                button = scene.hit_test(eye_origin, eye_direction, nearest_object_distance)
        """
        if self._camera_inverse is None or not self._is_created():
            return None
        try:
            if len(origin) != 3 or len(direction) != 3:
                return None
            ray_origin, ray_direction = Gf.Vec3d(*origin), Gf.Vec3d(*direction)
            if not all(math.isfinite(v) for v in (*ray_origin, *ray_direction)):
                return None
            length = ray_direction.GetLength()
            if length <= 1e-9:
                return None
            if max_distance is not None and (not math.isfinite(max_distance) or max_distance <= 0):
                return None
            local_origin = self._camera_inverse.Transform(ray_origin)
            local_direction = self._camera_inverse.TransformDir(ray_direction / length)
            if local_direction[2] >= -1e-9:
                return None
            distance = (-self.DEPTH - local_origin[2]) / local_direction[2]
            if distance <= 0 or (max_distance is not None and distance >= max_distance):
                return None
            hit = local_origin + distance * local_direction
            for button, (x, y) in self.BUTTON_CENTERS.items():
                if button == "back" and not self.backwards_enabled:
                    continue
                if abs(hit[0] - x) <= self.HALF_WIDTH and abs(hit[1] - y) <= self.HALF_HEIGHT:
                    return button
        except (TypeError, ValueError, RuntimeError):
            return None
        return None

    def update(self, status: object, eye_valid: bool = True) -> None:
        """Show controller state and dwell/action progress without issuing commands.

        Args:
            status: Controller object or dict with state, target, progress, and last_action fields.
            eye_valid: Whether a usable real eye sample exists. Running, stopping, and fault
                feedback remain visible because an accepted action may outlive its eye sample.

        Example:

            .. code-block:: python

                scene.update({"state": "dwelling", "target": "forward", "progress": 0.5})
        """
        if not self._is_created():
            return
        read = status.get if isinstance(status, dict) else lambda key, default=None: getattr(status, key, default)
        state, target = str(read("state", "unavailable")), read("target")
        try:
            progress = float(read("progress", 0.0))
        except (TypeError, ValueError):
            progress = 0.0
        progress = min(1.0, max(0.0, progress)) if math.isfinite(progress) else 0.0
        if not eye_valid and state not in (
            "active",
            "stopping",
            "fault",
            "controller",
            "controller_preview",
            "controller_wait",
        ):
            state, target, progress = "no_eye", None, 0.0
        if target == "back" and not self.backwards_enabled:
            target, progress = None, 0.0
        display_state = state if state in self._STATUS_DESCRIPTIONS else "unavailable"
        label, material = self._STATUS_DESCRIPTIONS[display_state]
        if state not in ("dwelling", "preview", "active", "stopping"):
            target, progress = None, 0.0
        display = (label, material)
        if display != self._last_display:
            for name in self._status_meshes:
                self._set_opacity(f"status_{name}", name == display_state)
                self._set_opacity(f"status_card_{name}", name == display_state)
            self._status_label.Set(label)
            self._last_display = display
        # `last_action` belongs to the controller, so its reset also clears this history.
        action = read("last_action")
        action_label = self._ACTION_LABELS.get(action, "")
        if action_label != self._last_feedback:
            for name in self._feedback_meshes:
                self._set_opacity(f"feedback_{name}", name == action)
            for name in ("feedback_backing", "feedback_prefix"):
                self._set_opacity(name, bool(action_label))
            self._feedback_label.Set(action_label)
            self._last_feedback = action_label
        for button, mesh in self._rings.items():
            count = min(self.RING_SEGMENTS, int(progress * self.RING_SEGMENTS)) if button == target else 0
            if count != self._ring_counts[button]:
                mesh.GetFaceVertexCountsAttr().Set([4] * count)
                mesh.GetFaceVertexIndicesAttr().Set(list(range(4 * count)))
                self._ring_counts[button] = count
            if count and self._ring_materials[button] != material:
                UsdShade.MaterialBindingAPI(mesh).Bind(self._materials[material])
                self._ring_materials[button] = material

    def cleanup(self) -> None:
        """Remove only the owned visual subtree and invalidate all cached handles.

        Example:

            .. code-block:: python

                scene.cleanup()
        """
        if self._stage is not None and self._stage.GetPrimAtPath(self.root_path).IsValid():
            self._stage.RemovePrim(self.root_path)
        self._stage = None
        self._transform_op = None
        self._camera_inverse = None
        self._materials.clear()
        self._opacity_inputs.clear()
        self._mask_transform_ops.clear()
        self._rings.clear()
        self._ring_counts.clear()
        self._ring_materials.clear()
        self._status_label = None
        self._status_meshes.clear()
        self._status_cards.clear()
        self._feedback_label = None
        self._feedback_meshes.clear()
        self._feedback_prefix = None
        self._feedback_plate = None
        self._last_display = None
        self._last_feedback = None

    def _is_created(self) -> bool:
        """Check whether the authored root still exists after stage changes."""
        return self._stage is not None and self._stage.GetPrimAtPath(self.root_path).IsValid()

    def _make_material(
        self, name: str, color: tuple[float, float, float], *, opacity: float = 1.0
    ) -> UsdShade.Material:
        """Create a prebound emissive material with an explicit binary opacity mask."""
        path = f"{self.root_path}/Materials/{name}"
        material = UsdShade.Material.Define(self._stage, path)
        shader = UsdShade.Shader.Define(self._stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        opacity_input = shader.CreateInput("opacity", Sdf.ValueTypeNames.Float)
        opacity_input.Set(opacity)
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
        self._opacity_inputs[name] = opacity_input
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _register_mask_transform(self, mesh: UsdGeom.Mesh, material: str) -> None:
        """Preauthor one translation for a uniquely masked visual."""
        transform = UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp()
        self._mask_transform_ops[material] = transform
        visible = self._opacity_inputs[material].Get() >= 0.5
        transform.Set(Gf.Vec3d(0.0, 0.0, 0.0 if visible else 2.0 * self.DEPTH))

    def _set_opacity(self, material: str, visible: bool) -> None:
        """Mask inactive geometry and translate it behind the camera."""
        opacity_input = self._opacity_inputs[material]
        opacity = 1.0 if visible else 0.0
        if opacity_input.Get() != opacity:
            opacity_input.Set(opacity)
        transform = self._mask_transform_ops[material]
        translation = Gf.Vec3d(0.0, 0.0, 0.0 if visible else 2.0 * self.DEPTH)
        if transform.Get() != translation:
            transform.Set(translation)

    def _mesh(self, path: str, points: list, counts: list[int], material: str) -> UsdGeom.Mesh:
        """Write a double-sided visual mesh with no collision or rigid-body schema."""
        mesh = UsdGeom.Mesh.Define(self._stage, path)
        mesh.CreatePointsAttr().Set(points)
        mesh.CreateFaceVertexCountsAttr().Set(counts)
        mesh.CreateFaceVertexIndicesAttr().Set(list(range(sum(counts))))
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr().Set(True)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self._materials[material])
        return mesh

    def _backing_plate(
        self,
        path: str,
        x: float,
        y: float,
        half_width: float,
        half_height: float,
        *,
        material: str = "panel",
        depth_offset: float = 0.0,
    ) -> UsdGeom.Mesh:
        """Place an unselectable dark plate behind side text, leaving button bounds unchanged."""
        z = -self.DEPTH + depth_offset
        return self._mesh(
            path,
            [
                (x - half_width, y - half_height, z),
                (x + half_width, y - half_height, z),
                (x + half_width, y + half_height, z),
                (x - half_width, y + half_height, z),
            ],
            [4],
            material,
        )

    def _text(self, path: str, label: str, x: float, y: float, pixel: float, material: str) -> UsdGeom.Mesh:
        """Build centered readable block lettering as one mesh ahead of the panel."""
        points = []
        left = x - (len(label) * 6 - 1) * pixel / 2
        for index, char in enumerate(label):
            for row, bits in enumerate(_FONT[char]):
                for column in range(5):
                    if bits & (1 << (4 - column)):
                        a, b = left + (index * 6 + column) * pixel, y + (3.5 - row) * pixel
                        z = -self.DEPTH + 0.004
                        points.extend([(a, b - pixel, z), (a + pixel, b - pixel, z), (a + pixel, b, z), (a, b, z)])
        mesh = self._mesh(path, points, [4] * (len(points) // 4), material)
        mesh.GetPrim().CreateAttribute("gazeControls:label", Sdf.ValueTypeNames.String, custom=True).Set(label)
        return mesh

    def _ring_points(self, x: float, y: float, inner: float, outer: float) -> list[tuple[float, float, float]]:
        """Return clockwise annulus quads beginning at twelve o'clock."""
        points = []
        for index in range(self.RING_SEGMENTS):
            a, b = (math.pi / 2 - 2 * math.pi * value / self.RING_SEGMENTS for value in (index, index + 1))
            points.extend(
                [
                    (x + radius * math.cos(angle), y + radius * math.sin(angle), -self.DEPTH + 0.002)
                    for radius, angle in ((inner, a), (inner, b), (outer, b), (outer, a))
                ]
            )
        return points
