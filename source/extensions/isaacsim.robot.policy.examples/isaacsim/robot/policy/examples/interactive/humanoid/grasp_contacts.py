# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read actual PhysX hand contacts without changing joint or object behavior.

Prepare the reader before initializing physics. Each physics callback may then
read the preceding solver's contact data once, using that step's duration. The
Isaac Sim 6.1 raw-contact API supplies world-space normals pointing in the force
direction on the HAND sensor; forces on the contacted object have the opposite
direction. No proximity estimate, attachment, or synthetic contact is produced.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .grasp_controller import ContactSample


class ContactReader:
    """Retain one contact-only tensor view for both Inspire hands.

    Args:
        max_contact_count: Total packed contact capacity across all hand links.
            A full buffer is rejected conservatively because truncation cannot
            establish complete opposing-contact evidence.

    Example:

    .. code-block:: python

        reader = ContactReader()
        reader.prepare(stage, "/World/G1", ["/World/GraspCube"])
        # Initialize physics before reading its measured contact forces.
        contacts, health = reader.read(0.01)
    """

    def __init__(self, *, max_contact_count: int = 2048) -> None:
        if isinstance(max_contact_count, bool) or not isinstance(max_contact_count, int) or max_contact_count < 1:
            raise ValueError("Contact capacity must be a positive integer")
        self.max_contact_count = max_contact_count
        self.reset()

    def reset(self) -> None:
        """Release tensor views and actor identities without editing USD schemas.

        Call before clearing/replacing the scene, then call ``prepare`` for the
        new stage. Existing contact-report schemas remain owned by that stage.

        Example:

        .. code-block:: python

            reader.reset()
        """
        self.sensor_paths: tuple[str, ...] = ()
        self.object_paths: tuple[str, ...] = ()
        self._roles_by_path: dict[str, tuple[str, str]] = {}
        self._view_roles: tuple[tuple[str, str], ...] = ()
        self._view: Any = None
        self._simulation_view: Any = None
        self._actor_objects: dict[int, tuple[str, bool]] = {}

    def invalidate(self) -> None:
        """Drop physics views and actor IDs while retaining prepared stage paths.

        Use for a timeline reset of the same loaded scene. Use ``reset`` when
        clearing the stage, and ``prepare`` before playing a replacement scene.

        Example:

        .. code-block:: python

            reader.invalidate()
        """
        self._release_view()

    @staticmethod
    def _hand_role(name: str) -> tuple[str, str] | None:
        """Recognize actual Inspire rigid-link names, including the palm."""
        tokens = name.lower().split("_")
        if len(tokens) < 2 or tokens[0] not in ("l", "r"):
            return None
        side = "left" if tokens[0] == "l" else "right"
        role = tokens[1]
        if role == "hand" and "base" in tokens:
            role = "palm"
        elif role == "pinky":
            role = "little"
        if role in ("thumb", "index", "middle", "ring", "little", "palm"):
            return side, role
        return None

    @staticmethod
    def _within(path: str, root: str) -> bool:
        """Match an exact USD prim or descendant, never a similar name prefix."""
        return path == root or path.startswith(root + "/")

    def prepare(self, stage: Any, robot_root: str, object_paths: Iterable[str]) -> None:
        """Discover existing hand rigid bodies and author contact reports once.

        Args:
            stage: USD stage containing the robot and grasp objects.
            robot_root: Existing robot root prim path.
            object_paths: Exact eligible object roots. Contacting child actors
                are canonicalized to these roots, with the deepest match first.

        Raises:
            ValueError: A root is missing or either hand lacks required links.

        Example:

        .. code-block:: python

            reader.prepare(stage, "/World/G1", ["/World/GraspCube", "/World/GraspCylinder"])
        """
        from pxr import PhysxSchema, UsdPhysics

        self.reset()
        robot_root = str(robot_root).rstrip("/")
        roots = tuple(sorted({str(path).rstrip("/") for path in object_paths}, key=lambda path: (-len(path), path)))
        if not robot_root.startswith("/") or not stage.GetPrimAtPath(robot_root).IsValid():
            raise ValueError("The robot root must be an existing absolute USD prim path")
        if not roots or any(not path.startswith("/") or not stage.GetPrimAtPath(path).IsValid() for path in roots):
            raise ValueError("Every grasp object must be an existing absolute USD prim path")

        sensors = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not self._within(path, robot_root) or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if (role := self._hand_role(prim.GetName())) is not None:
                sensors.append((path, role, prim))
        required = {"thumb", "index", "middle", "ring", "little", "palm"}
        for side in ("left", "right"):
            missing = required - {role[1] for _, role, _ in sensors if role[0] == side}
            if missing:
                raise ValueError(f"Missing {side} hand rigid contact links: {', '.join(sorted(missing))}")

        # This is the only USD-authoring boundary. In particular, never apply a
        # rigid-body API to a child collision mesh or author schemas in `read`.
        for _, _, prim in sensors:
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        self.sensor_paths = tuple(path for path, _, _ in sensors)
        self.object_paths = roots
        self._roles_by_path = {path: role for path, role, _ in sensors}

    def read(self, dt: float) -> tuple[dict[str, list[ContactSample]], str]:
        """Return current contacts and a health description for the caller.

        Args:
            dt: Duration in seconds of the solver step producing these contacts.
                PhysX converts impulse to newtons with this value; do not divide
                the returned force a second time.

        Returns:
            Contacts grouped by hand and ``ok`` on a complete valid read.
            Unavailable views, invalid data, and potentially truncated buffers
            return both hands empty with a descriptive non-``ok`` health value.
            A valid read with no contacts is still ``ok``. Scenery and robot
            contacts carry their actual actor paths with ``eligible_for_grasp``
            false, so they limit finger travel without establishing a grasp.

        Example:

        .. code-block:: python

            contacts, health = reader.read(physics_dt)
            print(health, len(contacts["right"]))
        """
        empty = {"left": [], "right": []}
        try:
            dt = float(dt)
            if not math.isfinite(dt) or dt <= 0.0:
                return empty, "invalid_physics_dt"
            if not self.sensor_paths:
                return empty, "not_prepared"

            from isaacsim.core.simulation_manager import SimulationManager

            if SimulationManager.get_active_physics_engine() != "physx":
                self._release_view()
                return empty, "unsupported_physics_engine"
            simulation_view = SimulationManager.get_physics_simulation_view()
            if simulation_view is None:
                self._release_view()
                return empty, "physics_not_ready"
            if simulation_view is not self._simulation_view:
                self._release_view()
                self._simulation_view = simulation_view
            if self._view is None:
                self._create_view(simulation_view)
            return self._read_contacts(dt)
        except Exception as error:
            # Tensor backends raise several Python/C++ exception classes. A bad
            # sensor read must not interrupt arm control or imply a secure grasp.
            self._release_view()
            return empty, f"contact_read_error: {type(error).__name__}: {str(error)[:180]}"

    def _release_view(self) -> None:
        """Discard physics-owned state while retaining stage preparation."""
        self._view = None
        self._simulation_view = None
        self._view_roles = ()
        self._actor_objects.clear()

    def _create_view(self, simulation_view: Any) -> None:
        """Create a read-only contact view and honor its actual sensor order."""
        # Use the same underlying view as experimental `RigidPrim`, avoiding its
        # constructor's USD API checks and `flush_changes` inside a callback.
        # Raw contacts need no filter patterns, so equal sensor/object counts do
        # not accidentally select one object per sensor instead of all objects.
        view = simulation_view.create_rigid_contact_view(
            list(self.sensor_paths), max_contact_data_count=self.max_contact_count
        )
        paths = tuple(str(path) for path in view.sensor_paths)
        if len(paths) != len(self.sensor_paths) or set(paths) != set(self.sensor_paths):
            raise RuntimeError("The physics contact view does not contain every prepared rigid hand link")
        self._view_roles = tuple(self._roles_by_path[path] for path in paths)
        self._view = view

    def _read_contacts(self, dt: float) -> tuple[dict[str, list[ContactSample]], str]:
        """Decode only occupied packed-buffer ranges and eligible actor IDs."""
        result: dict[str, list[ContactSample]] = {"left": [], "right": []}
        raw = self._view.get_raw_contact_data(dt)
        forces, points, normals, separations, counts, starts, actor_ids = [np.asarray(value.numpy()) for value in raw]
        counts, starts = counts.reshape(-1), starts.reshape(-1)
        if len(counts) != len(self._view_roles) or len(starts) != len(counts):
            return result, "invalid_contact_sensor_layout"
        if counts.dtype.kind not in "iu" or starts.dtype.kind not in "iu":
            return result, "invalid_contact_indices"
        forces, separations, actor_ids = forces.reshape(-1), separations.reshape(-1), actor_ids.reshape(-1)
        if actor_ids.dtype.kind != "u" or actor_ids.dtype.itemsize != 8:
            return result, "invalid_contact_actor_ids"
        capacity = len(forces)
        if (
            points.shape != (capacity, 3)
            or normals.shape != (capacity, 3)
            or len(separations) != capacity
            or len(actor_ids) != capacity
        ):
            return result, "invalid_contact_buffer_layout"
        total = sum(int(value) for value in counts)
        if total >= capacity or total >= self.max_contact_count:
            return result, "contact_buffer_full_or_overflow"
        occupied = []
        for sensor, (count, start) in enumerate(zip(counts, starts)):
            count, start = int(count), int(start)
            if count < 0 or (count and (start < 0 or start + count > capacity)):
                return result, "contact_buffer_overflow"
            occupied.extend((sensor, index) for index in range(start, start + count))
        if len({index for _, index in occupied}) != total:
            return result, "overlapping_contact_ranges"
        unknown = sorted({int(actor_ids[index]) for _, index in occupied} - self._actor_objects.keys() - {0})
        if unknown:
            import warp as wp

            ids_cpu = wp.array(unknown, dtype=wp.uint64, device="cpu")
            paths = self._view.get_other_actor_paths_from_ids(ids_cpu)
            if len(paths) != len(unknown):
                return result, "invalid_contact_actor_mapping"
            for actor_id, actor_path in zip(unknown, paths):
                if not actor_path:
                    return {"left": [], "right": []}, "unresolved_contact_actor"
                object_root = next((root for root in self.object_paths if self._within(str(actor_path), root)), None)
                self._actor_objects[actor_id] = (object_root or str(actor_path), object_root is not None)

        invalid = 0
        for sensor, index in occupied:
            actor = self._actor_objects.get(int(actor_ids[index]))
            if actor is None:
                continue
            object_path, eligible = actor
            force, separation = float(forces[index]), float(separations[index])
            point, normal = points[index], normals[index]
            length = float(np.linalg.norm(normal))
            if (
                not math.isfinite(force)
                or not math.isfinite(separation)
                or not np.isfinite(point).all()
                or not math.isfinite(length)
                or length < 1e-8
                or force < 0.0
            ):
                invalid += 1
                continue
            if force == 0.0:
                continue
            side, role = self._view_roles[sensor]
            result[side].append(
                ContactSample(
                    object_path=object_path,
                    role=role,
                    point_world=tuple(float(value) for value in point),
                    normal_world=tuple(float(value) for value in normal / length),
                    normal_force_n=force,
                    separation_m=separation,
                    eligible_for_grasp=eligible,
                )
            )
        if invalid:
            return {"left": [], "right": []}, f"invalid_contact_samples: {invalid}"
        return result, "ok"
