# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate level HMD heading and animate the G1 render-only head."""

from __future__ import annotations

import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

__all__ = ["RobotHeadVisual", "compute_level_head_reference"]


def _rotation_matrix(value: Gf.Matrix4d | None) -> Gf.Matrix4d | None:
    """Extract a finite proper rotation, ignoring translation deliberately."""
    try:
        if value is None:
            return None
        matrix = Gf.Matrix4d(value).ExtractRotationMatrix()
        if not all(math.isfinite(matrix[row][column]) for row in range(3) for column in range(3)):
            return None
        if abs(matrix.GetDeterminant() - 1.0) > 1e-5:
            return None
        if any(
            abs(sum(matrix[row][axis] * matrix[column][axis] for axis in range(3)) - float(row == column)) > 1e-5
            for row in range(3)
            for column in range(3)
        ):
            return None
        return Gf.Matrix4d(matrix, Gf.Vec3d(0.0))
    except (TypeError, ValueError, RuntimeError):
        return None


def compute_level_head_reference(pose: Gf.Matrix4d | None) -> Gf.Matrix4d | None:
    """Capture physical HMD heading without treating its tilt as the horizon.

    XR physical space is Y-up, with local -Z pointing out of the headset. Only
    the heading of that forward vector on the XZ floor plane is calibrated.
    Thus ``current_rotation * reference.GetInverse()`` removes initial yaw,
    while preserving physical pitch and roll relative to gravity. A headset
    resting sideways at startup cannot permanently tilt the camera horizon.

    A forward vector within about 5.7 degrees of vertical has an ambiguous
    floor heading: return None so the caller can defer initial calibration.
    Once a reference is captured, callers retain it until explicit recentering;
    looking vertically afterward must still use the current HMD orientation.

    Args:
        pose: Proper physical HMD rotation, optionally with translation.

    Returns:
        A translation-free rotation about physical +Y, or None for invalid
        rotation or an initially near-vertical forward direction.
    """
    rotation = _rotation_matrix(pose)
    if rotation is None:
        return None
    forward = rotation.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
    if math.hypot(float(forward[0]), float(forward[2])) < 0.1:
        return None
    yaw_degrees = math.degrees(math.atan2(-float(forward[0]), -float(forward[2])))
    return Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), yaw_degrees))


class RobotHeadVisual:
    """Rotate a verified render-only head around a measured geometric neck pivot.

    The G1 actor has no neck joint. This optional display feature leaves its
    original torso/head collision shapes, rigid bodies, masses, sensors, and
    policy joints untouched. Only the ``torso_link/visuals`` reference may be
    made editable. If it contains physics, preparation refuses the entire edit.

    Bounds and anchors are measured in the original head-local frame. The neck
    pivot is the bounding box's bottom center; the eye mount is its front (+X)
    face center. These are geometric attachment references, not measurements of
    an anatomical eye. Callers may add a camera clearance offset separately.

    ``apply()`` accepts the existing camera-local HMD rotation delta and the
    camera mount's rotation relative to the torso. It never reads XR or gaze.
    ``reset()`` returns the render head to its authored pose; ``clear()`` also
    restores the previous transform order and render-reference instanceability.

    Both normal and comfort viewing use ``apply()`` so the neck remains attached
    to the torso. Camera stabilization is independent of this visual transform.
    ``apply_camera_attached()`` is retained only for old diagnostic callers; it
    can separate the neck from the body and is not used by robot teleoperation.
    The legacy explicit visibility helper remains reversible for old sessions;
    it changes stage-wide visibility and is not a per-camera rendering mask.
    """

    _OP_NAME = "xformOp:transform:operatorHead"

    def __init__(self) -> None:
        self.status = "Not prepared"
        self.head_path: str | None = None
        self.bounds_min: Gf.Vec3d | None = None
        self.bounds_max: Gf.Vec3d | None = None
        self.neck_pivot_local: Gf.Vec3d | None = None
        self.eye_mount_local: Gf.Vec3d | None = None
        self.neck_pivot_in_torso: Gf.Vec3d | None = None
        self.eye_mount_in_torso: Gf.Vec3d | None = None
        self._stage: Usd.Stage | None = None
        self._edit_target: Usd.EditTarget | None = None
        self._visuals_path: str | None = None
        self._visuals_instance_changed = False
        self._visuals_had_instance_opinion = False
        self._visuals_instance_opinion = False
        self._original_order_backup: Sdf.Layer | None = None
        self._original_order_spec_path: Sdf.Path | None = None
        self._original_order_was_authored = False
        self._op: UsdGeom.XformOp | None = None
        self._original_local: Gf.Matrix4d | None = None
        self._parent_to_torso_rotation: Gf.Matrix4d | None = None
        self._parent_to_torso_transform: Gf.Matrix4d | None = None
        self._pivot_in_parent: Gf.Vec3d | None = None
        self._camera_attachment_mount: Gf.Matrix4d | None = None
        self._camera_attachment_clearance: float | None = None
        self._head_to_camera: Gf.Matrix4d | None = None
        self._first_person_hidden = False
        self._visibility_backup: Sdf.Layer | None = None
        self._visibility_spec_path: Sdf.Path | None = None

    @staticmethod
    def _contains_physics(root: Usd.Prim) -> bool:
        """Reject physics even in inactive descendants or instance proxies."""
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if prim.IsA(UsdPhysics.Joint) or any(
                prim.HasAPI(schema)
                for schema in (
                    UsdPhysics.RigidBodyAPI,
                    UsdPhysics.CollisionAPI,
                    UsdPhysics.MeshCollisionAPI,
                    UsdPhysics.MassAPI,
                    UsdPhysics.ArticulationRootAPI,
                )
            ):
                return True
        return False

    @staticmethod
    def _measure_head_bounds(head: Usd.Prim, cache: UsdGeom.XformCache) -> Gf.Range3d:
        """Union native geometry extents directly in the original head frame.

        An aggregate Xform bounding box may already be aligned to world axes.
        Transforming that box back into head coordinates inflates its dimensions
        as the robot turns. Transform each leaf geometry's local extent instead.
        """
        result = Gf.Range3d()
        world_to_head = cache.GetLocalToWorldTransform(head).GetInverse()
        for prim in Usd.PrimRange(head, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Boundable):
                continue
            if UsdGeom.Imageable(prim).ComputePurpose() not in (
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ):
                continue
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), Usd.TimeCode.Default())
            if extent is None or len(extent) != 2:
                continue
            native_range = Gf.Range3d(Gf.Vec3d(extent[0]), Gf.Vec3d(extent[1]))
            geometry_to_head = cache.GetLocalToWorldTransform(prim) * world_to_head
            result.UnionWith(Gf.BBox3d(native_range, geometry_to_head).ComputeAlignedRange())
        return result

    @property
    def ready(self) -> bool:
        """Return whether the saved stage still contains the owned visual op."""
        return self._stage is not None and self._op is not None and self._op.GetAttr().IsValid()

    @property
    def original_local_transform(self) -> Gf.Matrix4d | None:
        """Return a copy of the authored head transform before operator rotation."""
        return Gf.Matrix4d(self._original_local) if self._original_local is not None else None

    @property
    def first_person_hidden(self) -> bool:
        """Return whether this helper currently owns a hidden visibility opinion."""
        return self._first_person_hidden

    def set_first_person_hidden(self, hidden: bool) -> bool:
        """Hide only the render head, or restore its exact prior visibility.

        This stage-wide opinion affects concurrent observer cameras too. Call
        with False on leaving first person to restore the external appearance.
        Repeated calls with the same state do not reauthor USD each frame. Head
        rotation reset deliberately retains this state; ``clear()`` restores it.

        Args:
            hidden: Whether the comfort view needs the head mesh hidden.

        Returns:
            True on success. Invalid state or a failed USD edit returns False.
        """
        if not isinstance(hidden, bool) or not self.ready:
            return False
        if hidden == self._first_person_hidden:
            return True
        try:
            with Usd.EditContext(self._stage, self._edit_target):
                layer = self._edit_target.GetLayer()
                if hidden:
                    head = self._stage.GetPrimAtPath(self.head_path)
                    path = self._edit_target.MapToSpecPath(head.GetPath().AppendProperty("visibility"))
                    self._visibility_spec_path = path
                    self._visibility_backup = None
                    # Copy the entire local property, including time samples,
                    # metadata and blocks. An inherited opinion must remain
                    # inherited when we remove our temporary stronger opinion.
                    if layer.GetAttributeAtPath(path) is not None:
                        backup = Sdf.Layer.CreateAnonymous("operator-head-visibility-backup")
                        Sdf.CreatePrimInLayer(backup, path.GetPrimPath())
                        Sdf.CopySpec(layer, path, backup, path)
                        self._visibility_backup = backup
                        # Remove local animation before authoring the temporary
                        # default; otherwise an old visible time sample wins.
                        layer.GetPrimAtPath(path.GetPrimPath()).RemoveProperty(layer.GetPropertyAtPath(path))
                    result = UsdGeom.Imageable(head).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                    if not result:
                        return False
                    self._first_person_hidden = True
                else:
                    path = self._visibility_spec_path
                    property_spec = layer.GetPropertyAtPath(path)
                    if property_spec is not None:
                        layer.GetPrimAtPath(path.GetPrimPath()).RemoveProperty(property_spec)
                    if self._visibility_backup is not None:
                        Sdf.CopySpec(self._visibility_backup, path, layer, path)
                    self._first_person_hidden = False
                    self._visibility_backup = None
                    self._visibility_spec_path = None
            return True
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

    def prepare(self, stage: Usd.Stage, robot_root: str) -> bool:
        """Prepare a render-only head and save reversible stage-local edits.

        Args:
            stage: Loaded stage containing the G1 render and collision subtrees.
            robot_root: Robot root prim path, usually ``/World/G1``.

        Returns:
            True when the visual head is ready, or False with ``status`` describing
            missing geometry or an unsafe asset. Failed preparation restores edits.
        """
        head_path = robot_root.rstrip("/") + "/torso_link/visuals/head_link"
        if self._stage is stage and self.head_path == head_path and self.ready:
            return True
        self.clear()
        torso_path = robot_root.rstrip("/") + "/torso_link"
        visuals_path = torso_path + "/visuals"
        if stage is None:
            self.status = "No stage for the render head"
            return False
        torso = stage.GetPrimAtPath(torso_path)
        visuals = stage.GetPrimAtPath(visuals_path)
        head = stage.GetPrimAtPath(head_path)
        if not torso.IsValid() or not visuals.IsValid() or not head.IsValid() or not head.IsA(UsdGeom.Xformable):
            self.status = "Missing G1 render head at " + head_path
            return False
        if self._contains_physics(visuals):
            self.status = "Refused head animation because the render subtree contains physics"
            return False
        if UsdGeom.Xformable(visuals).GetResetXformStack() or any(
            prim.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(prim).GetResetXformStack()
            for prim in Usd.PrimRange(head, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
        ):
            self.status = "Refused head animation because render transforms detach from their parent"
            return False
        if head.GetAttribute(self._OP_NAME).IsValid():
            self.status = "Refused to replace an existing operator head transform"
            return False

        self._stage = stage
        self._edit_target = stage.GetEditTarget()
        self._visuals_path = visuals_path
        self.head_path = head_path
        try:
            # An instance proxy cannot receive local transform opinions. Only
            # this render reference is made editable; its collision sibling is
            # neither de-instanced nor reparented.
            if visuals.IsInstanceable():
                spec = self._edit_target.GetPrimSpecForScenePath(visuals.GetPath())
                self._visuals_had_instance_opinion = bool(spec and spec.HasInfo("instanceable"))
                if self._visuals_had_instance_opinion:
                    self._visuals_instance_opinion = spec.GetInfo("instanceable")
                visuals.SetInstanceable(False)
                self._visuals_instance_changed = True
            head = stage.GetPrimAtPath(head_path)
            if head.IsInstanceProxy() or self._contains_physics(head):
                raise ValueError("Head render geometry remains instanced or contains physics")
            xform = UsdGeom.Xformable(head)
            self._original_local = Gf.Matrix4d(xform.GetLocalTransformation())
            if not all(math.isfinite(self._original_local[row][column]) for row in range(4) for column in range(4)):
                raise ValueError("Head render transform must be finite")
            cache = UsdGeom.XformCache()
            parent_to_torso = (
                cache.GetLocalToWorldTransform(head.GetParent()) * cache.GetLocalToWorldTransform(torso).GetInverse()
            )
            self._parent_to_torso_rotation = _rotation_matrix(parent_to_torso)
            if self._parent_to_torso_rotation is None:
                raise ValueError("Head render parent must have a rigid transform relative to the torso")
            self._parent_to_torso_transform = Gf.Matrix4d(parent_to_torso)
            extent = self._measure_head_bounds(head, cache)
            minimum, maximum = Gf.Vec3d(extent.GetMin()), Gf.Vec3d(extent.GetMax())
            if extent.IsEmpty() or not all(
                math.isfinite(minimum[axis]) and math.isfinite(maximum[axis]) and maximum[axis] > minimum[axis]
                for axis in range(3)
            ):
                raise ValueError("Head render geometry has no finite three-dimensional bounds")
            center = (minimum + maximum) * 0.5
            self.bounds_min, self.bounds_max = minimum, maximum
            self.neck_pivot_local = Gf.Vec3d(center[0], center[1], minimum[2])
            self.eye_mount_local = Gf.Vec3d(maximum[0], center[1], center[2])
            self._pivot_in_parent = self._original_local.Transform(self.neck_pivot_local)
            head_to_torso = self._original_local * parent_to_torso
            self.neck_pivot_in_torso = head_to_torso.Transform(self.neck_pivot_local)
            self.eye_mount_in_torso = head_to_torso.Transform(self.eye_mount_local)

            # Save only this edit target's original order opinion. Restoring an
            # inherited order as a new local override would change composition.
            layer = self._edit_target.GetLayer()
            order_path = self._edit_target.MapToSpecPath(head.GetPath().AppendProperty("xformOpOrder"))
            self._original_order_spec_path = order_path
            self._original_order_was_authored = bool(layer.GetAttributeAtPath(order_path))
            if self._original_order_was_authored:
                self._original_order_backup = Sdf.Layer.CreateAnonymous("operator-head-order-backup")
                Sdf.CreatePrimInLayer(self._original_order_backup, order_path.GetPrimPath())
                Sdf.CopySpec(layer, order_path, self._original_order_backup, order_path)
            original_ops = xform.GetOrderedXformOps()
            resets_stack = xform.GetResetXformStack()
            self._op = xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "operatorHead")
            self._op.Set(Gf.Matrix4d(1.0))
            # USD evaluates its op order right-to-left for Gf row vectors. Put
            # the correction first so the result is original_local * correction.
            xform.SetXformOpOrder([self._op, *original_ops], resets_stack)
            self.status = "Render head ready; physical head remains fixed to the torso"
            return True
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            self.clear()
            self.status = "Render head unavailable: " + str(error)
            return False

    def apply(self, delta_rotation: Gf.Matrix4d | None, mount_rotation: Gf.Matrix4d | None) -> bool:
        """Apply a camera-local HMD delta without translating the neck pivot.

        Args:
            delta_rotation: Valid relative HMD orientation in camera coordinates.
                None or invalid input holds the last valid visual orientation.
            mount_rotation: Camera-to-torso orientation used by the existing view.

        Returns:
            True when a finite rotation was applied; False leaves the visual alone.
        """
        if not self.ready:
            return False
        delta, mount = _rotation_matrix(delta_rotation), _rotation_matrix(mount_rotation)
        if delta is None or mount is None:
            return False
        torso_delta = mount.GetInverse() * delta * mount
        parent_rotation = self._parent_to_torso_rotation
        parent_delta = parent_rotation * torso_delta * parent_rotation.GetInverse()
        correction = (
            Gf.Matrix4d().SetTranslate(-self._pivot_in_parent)
            * parent_delta
            * Gf.Matrix4d().SetTranslate(self._pivot_in_parent)
        )
        try:
            with Usd.EditContext(self._stage, self._edit_target):
                return bool(self._op.Set(correction))
        except (RuntimeError, ValueError):
            return False

    def apply_camera_attached(
        self,
        camera_world: Gf.Matrix4d | None,
        mount_in_torso: Gf.Matrix4d | None,
        torso_world: Gf.Matrix4d | None,
        clearance: float = 0.015,
    ) -> bool:
        """Keep the visible render head behind a separately stabilized eye.

        The authored head and neutral optical mount define a fixed head-to-camera
        transform. If necessary, move only the render head backward until all
        eight measured bounds corners are at least ``clearance`` behind the
        camera plane (+Z in camera coordinates). Rotating both with the same HMD
        orientation preserves that separation for yaw, pitch, and roll, including
        stereo eyes offset laterally from the camera center.

        The head is intentionally stabilized with the eye instead of retaining a
        fixed neck pivot. Large head turns and filtered torso motion can create a
        cosmetic neck gap or overlap. Camera matrices, skeletons, materials, and
        colliders are untouched. Teleoperation uses ``apply()`` to retain the
        torso's neck pivot instead of this legacy camera-relative placement.

        Args:
            camera_world: Final comfort view, including current HMD orientation.
            mount_in_torso: Original neutral camera mount measured at preparation.
            torso_world: Current physical torso pose, supplied by the caller so
                Fabric's possibly stale USD body transforms are never sampled.
            clearance: Minimum render-bounds distance behind the eye, in metres.

        Returns:
            True on success; invalid input leaves the prior render pose intact.
        """
        if not self.ready or self._parent_to_torso_transform is None:
            return False
        try:
            clearance = float(clearance)
            if not math.isfinite(clearance) or clearance < 0.0:
                return False
            matrices = []
            for value in (camera_world, mount_in_torso, torso_world):
                if value is None:
                    return False
                matrix = Gf.Matrix4d(value)
                if (
                    _rotation_matrix(matrix) is None
                    or not all(math.isfinite(matrix[row][column]) for row in range(4) for column in range(4))
                    or any(abs(matrix[row][3]) > 1e-8 for row in range(3))
                    or abs(matrix[3][3] - 1.0) > 1e-8
                ):
                    return False
                matrices.append(matrix)
            camera, mount, torso = matrices
            if (
                self._head_to_camera is None
                or self._camera_attachment_mount != mount
                or self._camera_attachment_clearance != clearance
            ):
                attachment = self._original_local * self._parent_to_torso_transform * mount.GetInverse()
                minimum_depth = min(
                    attachment.Transform(Gf.Vec3d(x, y, z))[2]
                    for x in (self.bounds_min[0], self.bounds_max[0])
                    for y in (self.bounds_min[1], self.bounds_max[1])
                    for z in (self.bounds_min[2], self.bounds_max[2])
                )
                position = attachment.ExtractTranslation()
                position[2] += max(0.0, clearance - minimum_depth)
                attachment.SetTranslateOnly(position)
                self._head_to_camera = attachment
                self._camera_attachment_mount = Gf.Matrix4d(mount)
                self._camera_attachment_clearance = clearance
            parent_world = self._parent_to_torso_transform * torso
            desired_local = self._head_to_camera * camera * parent_world.GetInverse()
            correction = self._original_local.GetInverse() * desired_local
            if not all(math.isfinite(correction[row][column]) for row in range(4) for column in range(4)):
                return False
            with Usd.EditContext(self._stage, self._edit_target):
                return bool(self._op.Set(correction))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

    def compute_camera_forward_extent(self, camera_world: Gf.Matrix4d, torso_world: Gf.Matrix4d) -> float | None:
        """Measure the head's farthest forward depth in the current camera frame.

        Positive depth is in front of the eye; negative depth is behind it.
        Transform all eight cached head-bound corners using the current local
        render transform and the caller's measured torso pose. Reading a USD
        world transform here would miss live movement when Fabric owns physics.
        This query never changes geometry, visibility, or collision opinions.

        Args:
            camera_world: Final stabilized camera, including HMD rotation.
            torso_world: The same measured body pose used to compute that view.

        Returns:
            Maximum forward depth in stage units, or None for unavailable or
            invalid geometry/poses. Bounds are conservative, not a ray test.
        """
        if not self.ready or self._parent_to_torso_transform is None:
            return None
        try:
            for matrix in (camera_world, torso_world):
                if (
                    _rotation_matrix(matrix) is None
                    or not all(math.isfinite(matrix[row][column]) for row in range(4) for column in range(4))
                    or any(abs(matrix[row][3]) > 1e-8 for row in range(3))
                    or abs(matrix[3][3] - 1.0) > 1e-8
                ):
                    return None
            head = self._stage.GetPrimAtPath(self.head_path)
            local = UsdGeom.Xformable(head).GetLocalTransformation()
            head_to_camera = local * self._parent_to_torso_transform * torso_world * camera_world.GetInverse()
            depths = [
                -float(head_to_camera.Transform(Gf.Vec3d(x, y, z))[2])
                for x in (self.bounds_min[0], self.bounds_max[0])
                for y in (self.bounds_min[1], self.bounds_max[1])
                for z in (self.bounds_min[2], self.bounds_max[2])
            ]
            return max(depths) if all(math.isfinite(value) for value in depths) else None
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None

    def reset(self) -> None:
        """Return the visual to its authored pose while retaining prepared geometry."""
        if self.ready:
            with Usd.EditContext(self._stage, self._edit_target):
                self._op.Set(Gf.Matrix4d(1.0))

    def clear(self) -> None:
        """Restore owned stage opinions and discard references to the previous stage."""
        if self._first_person_hidden:
            self.set_first_person_hidden(False)
        if self._stage is not None and self._edit_target is not None:
            with Usd.EditContext(self._stage, self._edit_target):
                head = self._stage.GetPrimAtPath(self.head_path) if self.head_path else None
                if head and head.IsValid() and self._op is not None:
                    # Work at the saved layer so cleanup also succeeds if an
                    # external edit already re-instanced the render reference.
                    layer = self._edit_target.GetLayer()
                    for name in (self._OP_NAME, "xformOpOrder"):
                        path = self._edit_target.MapToSpecPath(head.GetPath().AppendProperty(name))
                        property_spec = layer.GetPropertyAtPath(path)
                        if property_spec is not None:
                            layer.GetPrimAtPath(path.GetPrimPath()).RemoveProperty(property_spec)
                    if self._original_order_was_authored and self._original_order_backup is not None:
                        Sdf.CopySpec(
                            self._original_order_backup,
                            self._original_order_spec_path,
                            self._edit_target.GetLayer(),
                            self._original_order_spec_path,
                        )
                visuals = self._stage.GetPrimAtPath(self._visuals_path) if self._visuals_path else None
                if visuals and visuals.IsValid() and self._visuals_instance_changed:
                    if self._visuals_had_instance_opinion:
                        visuals.SetInstanceable(self._visuals_instance_opinion)
                    else:
                        visuals.ClearInstanceable()
        self.__init__()
