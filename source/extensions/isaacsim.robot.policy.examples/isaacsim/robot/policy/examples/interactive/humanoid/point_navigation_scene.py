# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene queries and independent blue pointing visuals for floor navigation.

Call ``create(stage)`` while physics is stopped. Runtime drawing changes only
prepared attributes and material relationships; it never creates collision or
rigid-body prims. The existing eye-gaze tracker and its red visuals are untouched.
"""

from __future__ import annotations

import math

import carb
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


class PointNavigationScene:
    """Provide conservative flat-floor selection, clearance, and pointing feedback.

    Args:
        robot_root: Robot subtree excluded from pointing and clearance queries.
    """

    ROOT = "/World/G1_PointNavigation"
    MAX_DISTANCE = 60.0
    FLOOR_TOLERANCE = 0.08
    CORRIDOR_HALF_WIDTH = 0.38
    CORRIDOR_HEIGHT = 1.5
    CORRIDOR_BOTTOM = 0.06
    ROUTE_TRACKING_MARGIN = 0.15
    ROUTE_TURN_DRIFT_RESERVE = 0.50

    def __init__(self, robot_root: str) -> None:
        self.robot_root = str(robot_root).rstrip("/")
        self.last_error: str | None = None
        self.last_blocking_paths: tuple[str, ...] = ()
        self._stage = None
        self._ray_op = None
        self._marker_op = None
        self._target_op = None
        self._target_radius = None

    @staticmethod
    def _vector(value: object) -> Gf.Vec3d | None:
        """Validate coordinates before passing them to native USD/physics APIs."""
        try:
            if len(value) != 3:
                return None
            values = tuple(float(axis) for axis in value)
            return Gf.Vec3d(*values) if all(math.isfinite(axis) for axis in values) else None
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _field(hit: object, name: str, default: object = None) -> object:
        """Read scene-query fields from either dictionary or attribute-style hits."""
        return hit.get(name, default) if isinstance(hit, dict) else getattr(hit, name, default)

    @classmethod
    def _hit_path(cls, hit: object) -> str | None:
        """Accept both Python test hits and native encoded PhysX collision paths."""
        for name in ("collision", "rigid_body"):
            value = cls._field(hit, name)
            if isinstance(value, (str, Sdf.Path)) and str(value).startswith("/"):
                return str(value)
            if value:
                try:
                    from pxr import PhysicsSchemaTools

                    decoded = str(PhysicsSchemaTools.intToSdfPath(value))
                    if decoded.startswith("/"):
                        return decoded
                except (TypeError, ValueError, RuntimeError):
                    continue
        return None

    def _is_robot(self, path: str) -> bool:
        """Match the robot root or its descendants without matching sibling prefixes."""
        return path == self.robot_root or path.startswith(self.robot_root + "/")

    @staticmethod
    def _current_stage() -> Usd.Stage | None:
        """Return the current USD stage, tolerating an unavailable Kit context."""
        try:
            return omni.usd.get_context().get_stage()
        except (AttributeError, RuntimeError):
            return None

    def is_floor_point(self, path: str | None, point: object) -> bool:
        """Accept static colliders near this scene's flat ground, never rigid boxes.

        Collision and rigid-body APIs can live above a mesh, so inspect ancestors.
        A path name containing ``ground`` is not sufficient evidence of a floor.
        """
        return self._floor_rejection_reason(path, point) is None

    def _floor_rejection_reason(self, path: str | None, point: object) -> str | None:
        """Explain an ineligible hit without inventing a floor beyond a collider."""
        position = self._vector(point)
        stage = self._current_stage()
        if position is None or stage is None or not path or self._is_robot(str(path)):
            return "The pointing hit has no valid non-robot scene collider"
        if abs(position[2]) > self.FLOOR_TOLERANCE:
            return "The pointing ray hit a surface above or below the walking floor"
        try:
            prim = stage.GetPrimAtPath(path)
            collision = False
            while prim.IsValid() and not prim.IsPseudoRoot():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    return "The pointing ray hit a movable object"
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                    collision = collision or enabled is not False
                prim = prim.GetParent()
            return None if collision else "The pointing hit has no enabled floor collider"
        except (TypeError, ValueError, RuntimeError):
            return "The pointing floor collider could not be inspected"

    def cast_ray(self, origin: object, direction: object) -> dict | None:
        """Return the nearest non-robot hit; malformed or failed queries select nothing."""
        self.last_error = None
        self.last_blocking_paths = ()
        start, axis = self._vector(origin), self._vector(direction)
        length = axis.GetLength() if axis is not None else math.nan
        if start is None or axis is None or not math.isfinite(length) or length <= 1e-8:
            self.last_error = "Pointing ray is invalid"
            return None
        axis.Normalize()
        best = None
        malformed = False

        def on_hit(hit: object) -> bool:
            """Keep the nearest nonrobot surface, including obstacles before a floor."""
            nonlocal best, malformed
            path = self._hit_path(hit)
            if path and self._is_robot(path):
                return True
            try:
                distance = float(self._field(hit, "distance", math.nan))
                position = self._vector(self._field(hit, "position"))
                normal = self._vector(self._field(hit, "normal"))
                if (
                    not path
                    or position is None
                    or not math.isfinite(distance)
                    or not 0 <= distance <= self.MAX_DISTANCE
                ):
                    malformed = True
                    return True
                if best is None or distance < best["distance"]:
                    floor_normal = normal is not None and normal.GetLength() > 1e-8 and normal.GetNormalized()[2] >= 0.9
                    rejection = (
                        self._floor_rejection_reason(path, position)
                        if floor_normal
                        else "The pointing ray hit a steep or downward-facing surface"
                    )
                    best = {
                        "position": position,
                        "path": path,
                        "distance": distance,
                        "walkable": rejection is None,
                        "detail": rejection or "Pointing ray hit walkable floor",
                    }
            except (TypeError, ValueError, RuntimeError, OverflowError):
                malformed = True
            return True

        try:
            from omni.physics.core import get_physics_scene_query_interface

            get_physics_scene_query_interface().raycast_all(tuple(start), tuple(axis), self.MAX_DISTANCE, on_hit, False)
        except Exception as error:
            self.last_error = f"Pointing scene query unavailable: {error}"
            return None
        if malformed:
            self.last_error = "Pointing scene query returned an invalid collider"
            return None
        if best is None:
            self.last_error = f"No collider within {self.MAX_DISTANCE:g} meters along the pointing ray"
        elif not best["walkable"]:
            self.last_error = best["detail"]
        return best

    def floor_at(self, point: object) -> tuple[float, float, float] | None:
        """Find real walking floor below a proposed node, never project through scenery.

        The nearest downward hit must be flat static floor. A box, mezzanine,
        missing collider, or unknown query result makes the node unavailable.
        """
        position = self._vector(point)
        if position is None:
            self.last_error = "Route floor query has invalid coordinates"
            return None
        hit = self.cast_ray((position[0], position[1], max(1.7, position[2] + 0.9)), (0.0, 0.0, -1.0))
        if hit is None or hit["walkable"] is not True:
            return None
        floor = self._vector(hit["position"])
        if floor is None or math.hypot(floor[0] - position[0], floor[1] - position[1]) > 0.05:
            self.last_error = "Route floor collision does not match the downward query"
            return None
        return tuple(float(axis) for axis in floor)

    def is_turn_clear(self, position: object) -> bool:
        """Check the enclosing footprint for every yaw of the current body box.

        The straight-path footprint is a 0.76 m square. A full turn needs its
        circumscribed radius, so the query grows by sqrt(2), not merely a change
        in requested yaw. Existing floor and robot collision exclusions remain.
        """
        point = self._vector(position)
        if point is None:
            self.last_error = "Turning position is invalid"
            return False
        floor = (float(point[0]), float(point[1]), 0.0)
        return self.is_path_clear(floor, floor, turning_clearance=True)

    def is_route_segment_clear(self, start: object, target: object) -> bool:
        """Reserve steering room and modest tracking error along a planned detour.

        Runtime turning uses a world-axis-aligned square. Project that entire
        square onto the rotated corridor axes: at 45 degrees its half-extent is
        sqrt(2) larger than on an axis-aligned route. Using the unprojected extent
        can mark a diagonal clear even while the runtime square hits a table.
        Add 15 cm of tracking reserve after projection. Current-pose guards and
        direct close-object approaches keep their existing collision queries.
        """
        first, last = self._vector(start), self._vector(target)
        margin = self.ROUTE_TRACKING_MARGIN
        if first is not None and last is not None:
            yaw = math.atan2(last[1] - first[1], last[0] - first[0])
            projection = abs(math.cos(yaw)) + abs(math.sin(yaw))
            turn_half_width = self.CORRIDOR_HALF_WIDTH * math.sqrt(2.0)
            margin += turn_half_width * max(0.0, projection - 1.0)
        return self.is_path_clear(start, target, turning_clearance=True, clearance_margin_m=margin)

    def is_route_corner_clear(self, position: object) -> bool:
        """Reserve open space for measured stepping drift at actual route corners.

        A physical 180-degree turn drifted 0.493 m. Reserve 0.50 m beyond the
        rotating body footprint at the start and selected intermediate corners.
        This is planning clearance, not a guarantee or a relaxation of runtime
        guards. Straight sections do not all need this larger corner footprint.
        """
        point = self._vector(position)
        if point is None:
            self.last_error = "Route corner coordinates are invalid"
            return False
        floor = (float(point[0]), float(point[1]), 0.0)
        return self.is_path_clear(
            floor, floor, turning_clearance=True, clearance_margin_m=self.ROUTE_TURN_DRIFT_RESERVE
        )

    def is_path_clear(
        self,
        start: object,
        target: object,
        *,
        stopping_shortfall_m: float = 0.0,
        turning_clearance: bool = False,
        clearance_margin_m: float = 0.0,
    ) -> bool:
        """Reject collisions throughout a body-width direct corridor to the target.

        This is a conservative straight-path check, not obstacle routing. The box
        starts above the floor and includes clearance around both endpoints.
        Unknown hits and query errors block motion instead of implying free space.

        Args:
            start: Measured robot position in world meters.
            target: Selected floor point in world meters.
            stopping_shortfall_m: Distance by which the checked center route ends
                before the selected point. The caller may subtract a validated
                braking margin from its arrival radius; zero preserves the full
                route. The body clearance padding is never reduced, even when
                the remaining route is shorter than this distance.
            turning_clearance: Expand the footprint to enclose every yaw angle.
                This is used for stationary turn checks, not to reduce margins.
            clearance_margin_m: Extra nonnegative lateral and endpoint reserve.
                The planner uses this to allow tracking error and stepping drift.
        """
        self.last_error = None
        self.last_blocking_paths = ()
        try:
            shortfall = float(stopping_shortfall_m)
            if isinstance(stopping_shortfall_m, bool) or not math.isfinite(shortfall) or shortfall < 0.0:
                raise ValueError("Invalid stopping shortfall")
        except (TypeError, ValueError, OverflowError):
            self.last_error = "Walking stopping shortfall must be finite and nonnegative"
            return False
        try:
            margin = float(clearance_margin_m)
            if isinstance(clearance_margin_m, bool) or not math.isfinite(margin) or margin < 0.0:
                raise ValueError("Invalid clearance margin")
        except (TypeError, ValueError, OverflowError):
            self.last_error = "Walking clearance margin must be finite and nonnegative"
            return False
        first, last = self._vector(start), self._vector(target)
        if first is None or last is None or abs(last[2]) > self.FLOOR_TOLERANCE:
            self.last_error = "Walking corridor coordinates are invalid"
            return False
        dx, dy = last[0] - first[0], last[1] - first[1]
        length = math.hypot(dx, dy)
        if not math.isfinite(length) or length > self.MAX_DISTANCE:
            self.last_error = "Walking corridor is out of range"
            return False
        yaw = math.atan2(dy, dx)
        checked_length = max(0.0, length - shortfall)
        fraction = checked_length / length if length > 0.0 else 0.0
        dx, dy = dx * fraction, dy * fraction
        center = (first[0] + dx * 0.5, first[1] + dy * 0.5, self.CORRIDOR_BOTTOM + self.CORRIDOR_HEIGHT / 2)
        width = self.CORRIDOR_HALF_WIDTH * (math.sqrt(2.0) if turning_clearance else 1.0) + margin
        extents = (checked_length / 2 + width, width, self.CORRIDOR_HEIGHT / 2)
        blocked = False
        blockers = []

        def on_hit(hit: object) -> bool:
            """Treat unknown and nonrobot overlap hits as blockers and retain a short trace."""
            nonlocal blocked
            path = self._hit_path(hit)
            if path is None or not self._is_robot(path):
                blocked = True
                name = path or "<unknown collider>"
                if len(blockers) < 4 and name not in blockers:
                    blockers.append(name)
            return True

        try:
            from omni.physics.core import get_physics_scene_query_interface

            get_physics_scene_query_interface().overlap_box(
                carb.Float3(*extents),
                carb.Float3(*center),
                carb.Float4(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
                on_hit,
            )
        except Exception as error:
            self.last_blocking_paths = tuple(blockers)
            self.last_error = f"Walking clearance query unavailable: {error}"
            return False
        self.last_blocking_paths = tuple(blockers)
        if blocked:
            self.last_error = f"An obstacle intersects the walking corridor: {blockers[0]}"
        return not blocked

    def resolve_object_approach(
        self,
        robot_position: object,
        hit: object,
        object_roots: object,
        *,
        stopping_shortfall_m: float = 0.20,
        allow_routing: bool = False,
    ) -> dict | None:
        """Find real clear floor before a pointed registered practice object.

        ``hit`` must be the nearest hit returned by ``cast_ray``. Its original
        surface remains the blue ray endpoint: this method performs separate
        downward collision queries at seven bounded standing candidates, never
        projecting through an object or changing the pointing ray. Only the
        caller's registered object roots are eligible; scenery stays an obstacle.

        Candidates start 0.30 m before the hit in the robot-to-object direction
        and retreat in 0.15 m increments, at most 1.20 m. Each needs a real
        walkable floor hit and the same full body-clearance corridor used for
        ordinary floor navigation. Tabletop objects are eligible, but the table
        itself must remain outside that corridor.

        With ``allow_routing=True``, a blocked direct route may instead accept a
        candidate with full turning clearance for the route planner. A clear
        direct approach keeps its existing graspable stopping distance; it does
        not require unused turning room at the clicked point. The caller must
        plan and validate all detour segments without ignoring any obstacle.
        """
        self.last_error = None
        self.last_blocking_paths = ()
        position = self._vector(robot_position)
        surface = self._vector(self._field(hit, "position"))
        path = self._field(hit, "path")
        try:
            shortfall = float(stopping_shortfall_m)
            ray_distance = float(self._field(hit, "distance", math.nan))
            if isinstance(stopping_shortfall_m, bool) or not math.isfinite(shortfall) or shortfall < 0.0:
                raise ValueError("Invalid stopping shortfall")
            if not math.isfinite(ray_distance) or not 0.0 <= ray_distance <= self.MAX_DISTANCE:
                raise ValueError("Invalid object collision distance")
            if not isinstance(path, (str, Sdf.Path)) or isinstance(object_roots, (str, bytes)):
                raise ValueError("Invalid object paths")
            path = str(path)
            roots = tuple(
                str(root).rstrip("/")
                for root in object_roots
                if isinstance(root, (str, Sdf.Path)) and str(root).startswith("/") and str(root).rstrip("/")
            )
            matches = [root for root in roots if path == root or path.startswith(root + "/")]
        except (TypeError, ValueError, OverflowError):
            self.last_error = (
                "Object approach requires valid coordinates, registered object paths, and stopping distance"
            )
            return None
        if position is None or surface is None or not matches or self._is_robot(path):
            self.last_error = "Point at a registered practice object to select a nearby standing spot"
            return None
        object_root = max(matches, key=len)
        dx, dy = surface[0] - position[0], surface[1] - position[1]
        distance = math.hypot(dx, dy)
        if not math.isfinite(distance) or distance < 0.80:
            self.last_error = "Object is too close for a separate walking approach"
            return None
        direction = (dx / distance, dy / distance)
        ray_height = max(position[2] + 1.0, surface[2] + 0.5)
        last_reason = "No clear floor standing spot was found"
        for attempt in range(7):
            standoff = 0.30 + 0.15 * attempt
            candidate_distance = distance - standoff
            if candidate_distance < 0.5:
                break
            candidate_xy = (
                position[0] + direction[0] * candidate_distance,
                position[1] + direction[1] * candidate_distance,
            )
            floor_hit = self.cast_ray((*candidate_xy, ray_height), (0.0, 0.0, -1.0))
            if floor_hit is None or not floor_hit["walkable"]:
                last_reason = self.last_error or "The standing spot has no visible walkable floor"
                continue
            floor = self._vector(floor_hit["position"])
            if floor is None or math.hypot(floor[0] - candidate_xy[0], floor[1] - candidate_xy[1]) > 0.05:
                last_reason = "The standing-floor collision did not match the downward query"
                continue
            floor_distance = math.hypot(floor[0] - position[0], floor[1] - position[1])
            if not 0.5 <= floor_distance <= distance:
                last_reason = "The standing floor is too close or beyond the selected object"
                continue
            clear = self.is_path_clear(position, floor, stopping_shortfall_m=shortfall)
            if not clear and allow_routing:
                clear = self.is_turn_clear(floor)
            if not clear:
                last_reason = self.last_error or "The route to the standing spot is blocked"
                continue
            return {
                "target": tuple(float(axis) for axis in floor),
                "source_object_root": object_root,
                "surface_point": tuple(float(axis) for axis in surface),
                "floor_path": floor_hit["path"],
                "standoff_m": standoff,
                "detail": f"Approach {object_root} from the clear floor in front of it",
            }
        self.last_error = f"No clear standing spot before {object_root}: {last_reason}"
        return None

    def create(self, stage: Usd.Stage) -> None:
        """Prepare reusable noncolliding visuals before starting physics."""
        if stage is None:
            return
        if stage == self._stage and self._visuals_valid():
            return
        self._drop_handles()
        self._stage = stage
        UsdGeom.Xform.Define(stage, self.ROOT)
        colors = {"Blue": (0.02, 0.2, 1.0), "Cyan": (0.0, 0.8, 1.0), "Green": (0.05, 1.0, 0.15)}
        for name, color in colors.items():
            material = UsdShade.Material.Define(stage, f"{self.ROOT}/Materials/{name}")
            shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[2 * v for v in color]))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        ray = UsdGeom.Cylinder.Define(stage, f"{self.ROOT}/Ray")
        ray.CreateAxisAttr("Z")
        ray.CreateRadiusAttr(0.004)
        ray.CreateHeightAttr(1.0)
        marker = UsdGeom.Sphere.Define(stage, f"{self.ROOT}/HitMarker")
        marker.CreateRadiusAttr(0.025)
        target = UsdGeom.Cylinder.Define(stage, f"{self.ROOT}/Target")
        target.CreateAxisAttr("Z")
        target.CreateHeightAttr(0.005)
        self._target_radius = target.CreateRadiusAttr(0.14)
        for shape in (ray, marker, target):
            shape.CreateDisplayColorAttr().Set([Gf.Vec3f(*colors["Blue"])])
            shape.ClearXformOpOrder()
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
                UsdShade.Material.Get(stage, f"{self.ROOT}/Materials/Blue")
            )
        self._ray_op = ray.AddTransformOp()
        self._marker_op = marker.AddTranslateOp()
        self._target_op = target.AddTranslateOp()
        self.hide()

    def _visuals_valid(self) -> bool:
        """Check that all retained visual transform attributes remain usable."""
        try:
            return self._stage is not None and all(
                op is not None and op.GetAttr().IsValid() for op in (self._ray_op, self._marker_op, self._target_op)
            )
        except RuntimeError:
            return False

    def draw(
        self,
        origin: object,
        end: object,
        target: object = None,
        progress: float = 0.0,
        state: str = "idle",
        *,
        hit_valid: bool = True,
    ) -> None:
        """Draw a ray and show a hit marker only at a confirmed scene collision.

        A ray that misses still shows its direction, without a floating sphere
        that could be mistaken for a selectable floor hit. A previously locked
        destination keeps its independent marker while the operator points.
        Stage replacement requires a new ``create``; drawing creates no prims.
        """
        first, last = self._vector(origin), self._vector(end)
        if self._current_stage() != self._stage or not self._visuals_valid():
            self._drop_handles()
            return
        if first is None or last is None:
            self.hide()
            return
        delta = last - first
        length = delta.GetLength()
        if not math.isfinite(length) or not 1e-6 < length <= self.MAX_DISTANCE:
            self.hide()
            return
        try:
            scale = Gf.Matrix4d().SetScale(Gf.Vec3d(1, 1, length))
            rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), delta / length))
            translation = Gf.Matrix4d().SetTranslate(first + delta * 0.5)
            # Gf uses row vectors: scale, then rotate, then translate.
            self._ray_op.Set(scale * rotation * translation)
            self._marker_op.Set(last)
            UsdGeom.Imageable(self._stage.GetPrimAtPath(self.ROOT)).MakeVisible()
            hit_marker = UsdGeom.Imageable(self._stage.GetPrimAtPath(f"{self.ROOT}/HitMarker"))
            if hit_valid is True:
                hit_marker.MakeVisible()
            else:
                hit_marker.MakeInvisible()
            target_prim = self._stage.GetPrimAtPath(f"{self.ROOT}/Target")
            selected = self._vector(target)
            if selected is None:
                UsdGeom.Imageable(target_prim).MakeInvisible()
            else:
                self._target_op.Set(selected + Gf.Vec3d(0, 0, 0.01))
                fraction = float(progress)
                fraction = min(1.0, max(0.0, fraction)) if math.isfinite(fraction) else 0.0
                self._target_radius.Set(0.07 + 0.07 * fraction)
                color = (
                    "Green" if state in ("active", "walking", "arrived") else "Cyan" if state == "acquiring" else "Blue"
                )
                UsdShade.MaterialBindingAPI(target_prim).Bind(
                    UsdShade.Material.Get(self._stage, f"{self.ROOT}/Materials/{color}")
                )
                UsdGeom.Imageable(target_prim).MakeVisible()
        except (TypeError, ValueError, RuntimeError, OverflowError):
            self.hide()
            self._drop_handles()

    def hide(self) -> None:
        """Hide prepared feedback without modifying scene topology."""
        if self._stage is None:
            return
        try:
            root = self._stage.GetPrimAtPath(self.ROOT)
            if root.IsValid():
                UsdGeom.Imageable(root).MakeInvisible()
        except RuntimeError:
            pass

    def _drop_handles(self) -> None:
        """Forget cached stage and visual handles without authoring or removing prims."""
        self._stage = None
        self._ray_op = None
        self._marker_op = None
        self._target_op = None
        self._target_radius = None

    def cleanup(self, remove: bool = False) -> None:
        """Hide and release cached handles; remove prims only during scene teardown."""
        self.hide()
        if remove and self._stage is not None:
            try:
                if self._stage.GetPrimAtPath(self.ROOT).IsValid():
                    self._stage.RemovePrim(self.ROOT)
            except RuntimeError:
                pass
        self._drop_handles()
        self.last_error = None
        self.last_blocking_paths = ()
