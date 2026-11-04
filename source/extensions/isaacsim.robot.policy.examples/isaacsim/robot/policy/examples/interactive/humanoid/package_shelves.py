# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author handle-free shipping cartons on open shelves for two-hand practice.

Every carton is a free dynamic body with one box collider. Static shelf decks
and legs provide physical support and remain navigation obstacles. Decorative
tape and shipping labels add no collision shapes, handles, or attachment joints.
The light masses are practice settings, not loaded warehouse carton models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

__all__ = ["SHELF_PACKAGE_SPECS", "SHELF_SPECS", "create_shelf_packages"]

SHELF_DEPTH = 0.34
# Leave room for open hands on both carton sides without reaching around posts.
SHELF_WIDTH = 1.60
DECK_THICKNESS = 0.02
POST_WIDTH = 0.035
PACKAGE_FLOOR_GAP = 0.003
MIN_LIFT_CLEARANCE = 0.09


@dataclass(frozen=True)
class ShelfSpec:
    """Floor location and two deck-top heights, in metres."""

    position_xy: tuple[float, float]
    deck_heights: tuple[float, float]

    def __post_init__(self) -> None:
        """Require finite shelf placement and two ordered standing-practice decks."""
        if len(self.position_xy) != 2 or len(self.deck_heights) != 2:
            raise ValueError("Each shelf needs two floor coordinates and two deck heights")
        if not all(math.isfinite(value) for value in (*self.position_xy, *self.deck_heights)):
            raise ValueError("Shelf coordinates and heights must be finite")
        if not 0.5 <= self.deck_heights[0] < self.deck_heights[1] <= 1.2:
            raise ValueError("Shelf heights must be increasing and inside the standing practice band")


@dataclass(frozen=True)
class ShelfPackageSpec:
    """One carton slot: shelf index, deck index, lateral offset, XYZ size, mass."""

    shelf: int
    level: int
    lateral_offset: float
    dimensions: tuple[float, float, float]
    mass: float

    def __post_init__(self) -> None:
        """Validate deck selection, practice mass, and room for side-palm access."""
        if (
            type(self.shelf) is not int
            or not 0 <= self.shelf < 3
            or type(self.level) is not int
            or self.level not in (0, 1)
        ):
            raise ValueError("Package requires a valid shelf and deck index")
        if len(self.dimensions) != 3 or not all(
            math.isfinite(value) for value in (*self.dimensions, self.lateral_offset, self.mass)
        ):
            raise ValueError("Package dimensions, offset, and mass must be finite")
        if any(value <= 0 for value in self.dimensions) or not 0 < self.mass <= 0.5:
            raise ValueError("Package dimensions must be positive and practice mass at most 0.5 kg")
        width, depth, _ = self.dimensions
        if width > SHELF_DEPTH - 0.01 or abs(self.lateral_offset) + depth / 2 + 0.08 > SHELF_WIDTH / 2 - POST_WIDTH:
            raise ValueError("Package must leave room for side palms between the shelf posts")


SHELF_SPECS = (
    ShelfSpec((3.5, 2.8), (0.68, 0.94)),
    ShelfSpec((3.5, -2.8), (0.74, 1.02)),
    ShelfSpec((1.35, 0.0), (0.78, 1.06)),
)

SHELF_PACKAGE_SPECS = (
    ShelfPackageSpec(0, 0, -0.28, (0.16, 0.20, 0.12), 0.10),
    ShelfPackageSpec(0, 0, 0.28, (0.22, 0.24, 0.14), 0.15),
    ShelfPackageSpec(0, 1, -0.28, (0.20, 0.26, 0.18), 0.18),
    ShelfPackageSpec(0, 1, 0.28, (0.24, 0.28, 0.22), 0.25),
    ShelfPackageSpec(1, 0, -0.28, (0.20, 0.24, 0.14), 0.14),
    ShelfPackageSpec(1, 0, 0.28, (0.16, 0.20, 0.12), 0.10),
    ShelfPackageSpec(1, 1, -0.28, (0.24, 0.28, 0.20), 0.24),
    ShelfPackageSpec(1, 1, 0.28, (0.18, 0.22, 0.16), 0.16),
    ShelfPackageSpec(2, 0, -0.28, (0.18, 0.22, 0.12), 0.12),
    ShelfPackageSpec(2, 0, 0.28, (0.22, 0.26, 0.14), 0.17),
    ShelfPackageSpec(2, 1, -0.28, (0.18, 0.24, 0.16), 0.15),
    ShelfPackageSpec(2, 1, 0.28, (0.22, 0.28, 0.20), 0.22),
)


def _box(stage, path, dimensions, center, color, *, collision=True):
    """Create an analytic cuboid with matching explicit visual/collision bounds."""
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.0)
    shape.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
    shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    shape.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim = shape.GetPrim()
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
        contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        contact.CreateContactOffsetAttr(0.002)
        contact.CreateRestOffsetAttr(0.0)
    return prim


def _validate_layout() -> None:
    """Keep lower cartons clear of the next deck during a short lift."""
    for package in SHELF_PACKAGE_SPECS:
        shelf = SHELF_SPECS[package.shelf]
        if package.level == 0:
            ceiling = shelf.deck_heights[1] - DECK_THICKNESS
            carton_top = shelf.deck_heights[0] + PACKAGE_FLOOR_GAP + package.dimensions[2]
            if ceiling - carton_top < MIN_LIFT_CLEARANCE:
                raise ValueError("Lower carton needs at least 9 cm of free lift/finger space below the upper deck")


def create_shelf_packages(
    stage: Usd.Stage,
    root_path: str,
    *,
    first_index: int,
    shelf_root_path: str = "/World/G1_PackageShelves",
) -> list[str]:
    """Create three static racks and twelve free cartons before physics starts.

    Return only carton roots for the existing contact, gaze, recording, and
    object-navigation registration. Shelves stay in their own static namespace.
    Side-palm contact metadata identifies the real carton faces; it applies no
    motion, force, attachment, or grasp assistance. Physical pickup needs live
    reach/contact validation and operator positioning beside each shelf.
    """
    _validate_layout()
    paths = [Sdf.Path(value) for value in (root_path, shelf_root_path)]
    if any(not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath for path in paths):
        raise ValueError("Package and shelf roots must be absolute prim paths")
    if paths[0].HasPrefix(paths[1]) or paths[1].HasPrefix(paths[0]):
        raise ValueError("Package and static shelf namespaces must be separate")
    if type(first_index) is not int or first_index < 0:
        raise ValueError("First package index must be a nonnegative integer")
    roots = [f"{root_path}/Box_{first_index + index:02d}" for index in range(len(SHELF_PACKAGE_SPECS))]
    if stage.GetPrimAtPath(shelf_root_path).IsValid() or any(stage.GetPrimAtPath(path).IsValid() for path in roots):
        raise ValueError("Shelf and package output paths must be unused; existing props were not changed")

    UsdGeom.Xform.Define(stage, root_path)
    UsdGeom.Xform.Define(stage, shelf_root_path)
    material_path = str(paths[1].AppendChild("ContactMaterial"))
    material = UsdShade.Material.Define(stage, material_path)
    contact = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    contact.CreateStaticFrictionAttr(0.8)
    contact.CreateDynamicFrictionAttr(0.6)
    contact.CreateRestitutionAttr(0.0)
    for index, shelf in enumerate(SHELF_SPECS):
        rack_path = f"{shelf_root_path}/Shelf_{index:02d}"
        rack = UsdGeom.Xform.Define(stage, rack_path)
        rack.AddTranslateOp().Set(Gf.Vec3d(*shelf.position_xy, 0))
        UsdShade.MaterialBindingAPI.Apply(rack.GetPrim()).Bind(material, materialPurpose="physics")
        for level, top in enumerate(shelf.deck_heights):
            _box(
                stage,
                f"{rack_path}/Deck_{level}",
                (SHELF_DEPTH, SHELF_WIDTH, DECK_THICKNESS),
                (0, 0, top - DECK_THICKNESS / 2),
                (0.30, 0.35, 0.40),
            )
        post_height = shelf.deck_heights[-1] + 0.025
        for post, (x_sign, y_sign) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            _box(
                stage,
                f"{rack_path}/Post_{post}",
                (POST_WIDTH, POST_WIDTH, post_height),
                (x_sign * (SHELF_DEPTH - POST_WIDTH) / 2, y_sign * (SHELF_WIDTH - POST_WIDTH) / 2, post_height / 2),
                (0.78, 0.30, 0.08),
            )

    for index, (root, spec) in enumerate(zip(roots, SHELF_PACKAGE_SPECS)):
        width, depth, height = spec.dimensions
        shelf = SHELF_SPECS[spec.shelf]
        # Put every carton just inside the open front edge, instead of recessing
        # it behind the shelf supports where the short G1 arms cannot reach.
        center = (
            shelf.position_xy[0] - SHELF_DEPTH / 2 + 0.005 + width / 2,
            shelf.position_xy[1] + spec.lateral_offset,
            shelf.deck_heights[spec.level] + PACKAGE_FLOOR_GAP + height / 2,
        )
        wrapper = UsdGeom.Xform.Define(stage, root)
        wrapper.AddTranslateOp().Set(Gf.Vec3d(*center))
        prim = wrapper.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(spec.mass)
        physics = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physics.CreateEnableCCDAttr(True)
        physics.CreateSolverPositionIterationCountAttr(16)
        physics.CreateSolverVelocityIterationCountAttr(4)
        physics.CreateSleepThresholdAttr(0.0)
        color = (0.58 + (index % 3) * 0.025, 0.37 + (index % 2) * 0.025, 0.19)
        _box(stage, root + "/Carton", spec.dimensions, (0, 0, 0), color)
        _box(
            stage,
            root + "/PackingTape",
            (width, 0.035, 0.001),
            (0, 0, height / 2 + 0.0005),
            (0.25, 0.19, 0.12),
            collision=False,
        )
        _box(
            stage,
            root + "/ShippingLabel",
            (0.001, 0.085, 0.04),
            (-width / 2 - 0.0005, 0, 0.005),
            (0.90, 0.88, 0.81),
            collision=False,
        )
        for stripe in range(7):
            _box(
                stage,
                f"{root}/Barcode_{stripe}",
                (0.001, 0.003 + 0.001 * (stripe % 2), 0.022),
                (-width / 2 - 0.0015, (stripe - 3) * 0.009, 0.005),
                (0.09, 0.09, 0.08),
                collision=False,
            )
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
        )
        for name, value_type, value in (
            ("g1:packageSize", Sdf.ValueTypeNames.Float, max(spec.dimensions)),
            ("g1:packageDimensions", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*spec.dimensions)),
            ("g1:packageMass", Sdf.ValueTypeNames.Float, spec.mass),
            ("g1:packageOnSurface", Sdf.ValueTypeNames.Bool, True),
            ("g1:packageKind", Sdf.ValueTypeNames.String, "shelf_carton"),
            ("g1:shelfIndex", Sdf.ValueTypeNames.Int, spec.shelf),
            ("g1:shelfLevel", Sdf.ValueTypeNames.Int, spec.level),
            ("g1:leftPalmContactLocal", Sdf.ValueTypeNames.Float3, Gf.Vec3f(0, depth / 2, 0)),
            ("g1:rightPalmContactLocal", Sdf.ValueTypeNames.Float3, Gf.Vec3f(0, -depth / 2, 0)),
        ):
            prim.CreateAttribute(name, value_type).Set(value)
        prim.CreateRelationship("g1:supportSurface").SetTargets(
            [f"{shelf_root_path}/Shelf_{spec.shelf:02d}/Deck_{spec.level}"]
        )
    return roots
