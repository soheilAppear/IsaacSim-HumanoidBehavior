# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author lightweight, tall practice cartons with physically graspable handles.

The carton and its handle form one compound rigid body. The open space behind
the handle remains open in collision geometry, so fingers can wrap around it.
There are no attachment joints, scripted object motion, or grasp force changes.
These are deliberately light practice props, not calibrated loaded cartons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

__all__ = ["TALL_PACKAGE_SPECS", "create_tall_packages"]

_FLOOR_GAP = 0.005
_GRIP_DIAMETER = 0.035
_GRIP_LENGTH = 0.16
_GRIP_CLEARANCE = 0.07


@dataclass(frozen=True)
class TallPackageSpec:
    """Describe a standing carton in the warehouse's metre coordinate system."""

    position_xy: tuple[float, float]
    #: Floor position of the pickup station.
    dimensions: tuple[float, float, float]
    #: Carton body width, depth, and height; excludes the protruding handle.
    mass: float
    #: Total mass in kilograms, including the integrated handle.
    handle_below_top: float = 0.08
    #: Vertical distance from the carton top to the handle center.

    def __post_init__(self) -> None:
        """Reject invalid props before USD authoring or physics initialization."""
        if len(self.position_xy) != 2 or len(self.dimensions) != 3:
            raise ValueError("Package needs two position coordinates and three dimensions")
        values = (*self.position_xy, *self.dimensions, self.mass, self.handle_below_top)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Package measurements must be finite")
        if self.mass <= 0 or any(dimension <= 0 for dimension in self.dimensions):
            raise ValueError("Package dimensions and mass must be positive")
        radius = _GRIP_DIAMETER / 2
        if not radius <= self.handle_below_top <= self.dimensions[2] - radius:
            raise ValueError("Handle and brackets must fit within the carton's vertical extent")


TALL_PACKAGE_SPECS = (
    # Separate the pickup stations from the original low floor cartons. Their
    # earlier positions left those obstacles between the robot and the handles.
    TallPackageSpec((3.5, 2.8), (0.32, 0.32, 0.76), 0.25),
    TallPackageSpec((3.5, -2.8), (0.40, 0.32, 0.88), 0.35),
    # The tallest handle is the first practice station, directly ahead of spawn.
    # It is a real obstacle; navigation must approach it or route around it.
    # A higher grip keeps the short G1 arm nearer shoulder height, allowing a
    # reach from outside the existing body footprint and braking clearance.
    TallPackageSpec((1.35, 0.0), (0.30, 0.38, 0.96), 0.45, handle_below_top=0.025),
)

_BODY_COLOR = Gf.Vec3f(0.55, 0.33, 0.16)
_GRIP_COLOR = Gf.Vec3f(0.12, 0.22, 0.30)


def _add_collision(prim: Usd.Prim) -> None:
    """Keep primitive colliders separate so the handle opening is not filled."""
    UsdPhysics.CollisionAPI.Apply(prim)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(0.002)
    collision.CreateRestOffsetAttr(0.0)


def _add_box(
    stage: Usd.Stage,
    path: str,
    dimensions: tuple[float, float, float],
    center: tuple[float, float, float],
    color: Gf.Vec3f,
    *,
    collidable: bool = True,
) -> Usd.Prim:
    """Author an explicit-stage box with matching visual and collision bounds."""
    # The experimental wrappers operate on the active Kit stage. This compound
    # authoring helper also supports isolated USD stages, so use schema authoring
    # at this boundary instead of changing the app's active stage as a side effect.
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.0)
    shape.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
    shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    shape.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    shape.CreateDisplayColorAttr([color])
    if collidable:
        _add_collision(shape.GetPrim())
    return shape.GetPrim()


def create_tall_packages(stage: Usd.Stage, root_path: str, *, first_index: int) -> list[str]:
    """Create three floor cartons with narrow handles at standing hand height.

    Call before initializing physics, then register the returned roots with the
    existing contact reader, navigation object selector, and gaze highlighter.
    Existing objects are never replaced. Handle centers are approximately
    0.685, 0.805, and 0.940 m above the floor when authored; settling the 5 mm
    initial floor gap places them at 0.680, 0.800, and 0.935 m.

    Args:
        stage: Stage to author before Play.
        root_path: Existing or new absolute Xform containing sample packages.
        first_index: First unused ``Box_<nn>`` index in that namespace.

    Returns:
        Exact dynamic-body roots, in deterministic creation order.

    Raises:
        ValueError: A path/index is invalid or an output prim already exists.

    Example:

    .. code-block:: python

        roots = create_tall_packages(stage, "/World/G1_SampleBoxes", first_index=10)
        print(len(roots))  # 3
    """
    path = Sdf.Path(root_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError("Package root must be an absolute prim path")
    if isinstance(first_index, bool) or not isinstance(first_index, int) or first_index < 0:
        raise ValueError("Package index must be a nonnegative integer")
    roots = [f"{root_path}/Box_{first_index + index:02d}" for index in range(len(TALL_PACKAGE_SPECS))]
    if any(stage.GetPrimAtPath(root).IsValid() for root in roots):
        raise ValueError("Tall package paths must be unused; existing objects were not changed")

    UsdGeom.Xform.Define(stage, root_path)
    # Keep non-package prims outside the package root: recording and assisted-mode
    # search iterate its direct children as rigid objects.
    material_path = str(path.GetParentPath().AppendChild(path.name + "_TallPackageContactMaterial"))
    material = UsdShade.Material.Define(stage, material_path)
    contact = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    contact.CreateStaticFrictionAttr(0.8)
    contact.CreateDynamicFrictionAttr(0.6)
    contact.CreateRestitutionAttr(0.0)
    for root, spec in zip(roots, TALL_PACKAGE_SPECS):
        width, depth, height = spec.dimensions
        wrapper = UsdGeom.Xform.Define(stage, root)
        wrapper.AddTranslateOp().Set(Gf.Vec3d(*spec.position_xy, _FLOOR_GAP + height / 2))
        prim = wrapper.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(spec.mass)
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body.CreateEnableCCDAttr(True)
        body.CreateSolverPositionIterationCountAttr(16)
        body.CreateSolverVelocityIterationCountAttr(4)
        # Static holds must continue producing measured fingertip contact forces.
        body.CreateSleepThresholdAttr(0.0)
        _add_box(stage, root + "/Carton", spec.dimensions, (0.0, 0.0, 0.0), _BODY_COLOR)

        grip_x = -width / 2 - _GRIP_CLEARANCE - _GRIP_DIAMETER / 2
        grip_z = height / 2 - spec.handle_below_top
        grip = UsdGeom.Cylinder.Define(stage, root + "/HandleGrip")
        grip.CreateAxisAttr("Y")
        grip.CreateRadiusAttr(_GRIP_DIAMETER / 2)
        grip.CreateHeightAttr(_GRIP_LENGTH)
        extent = Gf.Vec3f(_GRIP_DIAMETER / 2, _GRIP_LENGTH / 2, _GRIP_DIAMETER / 2)
        grip.CreateExtentAttr([-extent, extent])
        grip.AddTranslateOp().Set(Gf.Vec3d(grip_x, 0.0, grip_z))
        grip.CreateDisplayColorAttr([_GRIP_COLOR])
        _add_collision(grip.GetPrim())
        for sign, name in ((-1, "HandleBracketLeft"), (1, "HandleBracketRight")):
            _add_box(
                stage,
                f"{root}/{name}",
                (_GRIP_CLEARANCE + _GRIP_DIAMETER, 0.025, _GRIP_DIAMETER),
                (-width / 2 - _GRIP_CLEARANCE / 2, sign * _GRIP_LENGTH / 2, grip_z),
                _GRIP_COLOR,
            )
        # Packing tape is visual-only and contributes neither mass nor contacts.
        _add_box(
            stage,
            root + "/PackingTape",
            (width, 0.045, 0.001),
            (0.0, 0.0, height / 2 + 0.0005),
            Gf.Vec3f(0.82, 0.65, 0.35),
            collidable=False,
        )
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
        )
        prim.CreateAttribute("g1:packageSize", Sdf.ValueTypeNames.Float).Set(max(spec.dimensions))
        prim.CreateAttribute("g1:packageDimensions", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*spec.dimensions))
        prim.CreateAttribute("g1:packageMass", Sdf.ValueTypeNames.Float).Set(spec.mass)
        prim.CreateAttribute("g1:packageOnSurface", Sdf.ValueTypeNames.Bool).Set(False)
        prim.CreateAttribute("g1:packageKind", Sdf.ValueTypeNames.String).Set("tall_handled_carton")
        prim.CreateAttribute("g1:gripCenterLocal", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(grip_x, 0.0, grip_z))
        prim.CreateAttribute("g1:gripDiameter", Sdf.ValueTypeNames.Float).Set(_GRIP_DIAMETER)
    return roots
