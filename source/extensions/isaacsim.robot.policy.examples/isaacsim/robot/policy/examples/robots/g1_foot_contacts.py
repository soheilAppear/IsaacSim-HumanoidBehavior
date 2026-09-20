# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure the released decoupled actor's four contact points on each foot.

The packaged USD's shortened heel differs from the paired NVIDIA deployment
XML. This explicit compatibility adjustment is only called for physical
decoupled walking, before physics initializes. It changes contact geometry in
an anonymous session layer; authored rigid-body mass properties and referenced
asset files remain untouched. No anchors or root-motion constraints are added.
"""

from __future__ import annotations

import math

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

FOOT_CONTACT_POINTS = ((-0.05, 0.025, -0.03), (-0.05, -0.025, -0.03), (0.12, 0.03, -0.03), (0.12, -0.03, -0.03))
FOOT_CONTACT_RADIUS = 0.005
_CONTAINER_NAME = "DecoupledFootContacts"
_MARKER = "isaacsim:decoupledFootContacts"
_MASS_ATTRIBUTES = ("physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes")


def _body_mass_snapshot(root: Usd.Prim) -> dict:
    """Read authored rigid-body properties, excluding collider-only mass APIs."""
    return {
        str(prim.GetPath()): {name: prim.GetAttribute(name).Get() for name in _MASS_ATTRIBUTES}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.HasAPI(UsdPhysics.MassAPI)
    }


def _validate_contacts(stage: Usd.Stage, feet: list[Usd.Prim]) -> dict:
    """Reject incomplete or changed contact configurations on repeated calls."""
    report = {"sphere_paths": [], "disabled_colliders": [], "instance_scopes": []}
    for foot in feet:
        parent = stage.GetPrimAtPath(foot.GetPath().AppendChild(_CONTAINER_NAME))
        marker = parent.GetCustomDataByKey(_MARKER) if parent else None
        if not isinstance(marker, dict) or marker.get("version") != 1:
            raise ValueError("Incomplete decoupled foot-contact configuration")
        sources = marker.get("source_colliders", {})
        if len(sources) != 4:
            raise ValueError("Decoupled foot contacts require four original colliders per foot")
        for path in sources:
            prim = stage.GetPrimAtPath(path)
            if not prim or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                raise ValueError(f"Original foot collider is still enabled: {path}")
            report["disabled_colliders"].append(path)
        scope = stage.GetPrimAtPath(foot.GetPath().AppendChild("collisions"))
        if scope:
            if scope.IsInstanceable():
                raise ValueError("Decoupled foot collision scope unexpectedly became instanceable")
            report["instance_scopes"].append(str(scope.GetPath()))
        for index, position in enumerate(FOOT_CONTACT_POINTS):
            prim = stage.GetPrimAtPath(parent.GetPath().AppendChild(f"Point{index}"))
            if not prim or not prim.IsA(UsdGeom.Sphere):
                raise ValueError("Decoupled foot-contact sphere is missing")
            sphere = UsdGeom.Sphere(prim)
            transform = sphere.GetLocalTransformation()
            expected = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*position))
            collision = PhysxSchema.PhysxCollisionAPI(prim)
            if (
                not math.isclose(sphere.GetRadiusAttr().Get(), FOOT_CONTACT_RADIUS, abs_tol=1e-9)
                or sphere.GetResetXformStack()
                or any(abs(transform[i][j] - expected[i][j]) > 1e-9 for i in range(4) for j in range(4))
                or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True
                or not math.isclose(collision.GetContactOffsetAttr().Get(), 0.001, abs_tol=1e-9)
                or collision.GetRestOffsetAttr().Get() != 0.0
                or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or prim.HasAPI(UsdPhysics.MassAPI)
            ):
                raise ValueError("Decoupled foot-contact geometry or offsets changed")
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            physics_material = UsdPhysics.MaterialAPI(material.GetPrim()) if material else None
            if not physics_material or (
                physics_material.GetStaticFrictionAttr().Get(),
                physics_material.GetDynamicFrictionAttr().Get(),
                physics_material.GetRestitutionAttr().Get(),
            ) != (1.0, 1.0, 0.0):
                raise ValueError("Decoupled foot-contact physics material changed")
            report["sphere_paths"].append(str(prim.GetPath()))
    return report


def configure_decoupled_foot_contacts(stage: Usd.Stage, robot_prim_path: str) -> dict:
    """Replace shortened asset foot contacts with the paired actor's geometry.

    Args:
        stage: Loaded USD stage, before creating the physics articulation.
        robot_prim_path: Robot root whose two ankle-roll rigid bodies own the feet.

    Returns:
        Paths of new contact spheres, disabled source colliders, modified instance
        scopes, and whether this call found an already valid configuration.

    Raises:
        ValueError: The robot, mass properties, or collider structure is unsupported.
        RuntimeError: An override could not be applied without changing mass data.
    """
    root = stage.GetPrimAtPath(robot_prim_path) if stage is not None else None
    if not root or not root.IsValid():
        raise ValueError("Decoupled foot contacts require a loaded robot prim")
    feet = [stage.GetPrimAtPath(root.GetPath().AppendChild(f"{side}_ankle_roll_link")) for side in ("left", "right")]
    if not all(foot and foot.HasAPI(UsdPhysics.RigidBodyAPI) and not foot.IsInstanceProxy() for foot in feet):
        raise ValueError("Decoupled foot contacts require two editable ankle-roll rigid bodies")
    existing = [bool(stage.GetPrimAtPath(foot.GetPath().AppendChild(_CONTAINER_NAME))) for foot in feet]
    if any(existing):
        if not all(existing):
            raise ValueError("Incomplete decoupled foot-contact configuration")
        return {**_validate_contacts(stage, feet), "already_configured": True}

    masses = _body_mass_snapshot(root)
    for foot in feet:
        properties = masses.get(str(foot.GetPath()), {})
        if (
            not properties
            or any(value is None for value in properties.values())
            or any(not foot.GetAttribute(name).HasAuthoredValueOpinion() for name in _MASS_ATTRIBUTES)
            or not math.isfinite(properties["physics:mass"])
            or properties["physics:mass"] <= 0.0
            or any(not math.isfinite(v) or v <= 0.0 for v in properties["physics:diagonalInertia"])
            or any(not math.isfinite(v) for v in properties["physics:centerOfMass"])
        ):
            raise ValueError("Foot mass, COM, principal axes, and inertia must be explicitly resolved")
    sources_by_foot = {}
    for foot in feet:
        sources = {
            str(prim.GetPath()): bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
            for prim in Usd.PrimRange(foot, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        }
        if len(sources) != 4 or any(not stage.GetPrimAtPath(path).IsA(UsdGeom.Sphere) for path in sources):
            raise ValueError("Expected exactly four original sphere colliders under each ankle-roll body")
        sources_by_foot[str(foot.GetPath())] = sources

    # Editing a temporary composition layer is schema-specific and deliberately
    # avoids authoring opinions into the source asset or current user edit target.
    layer = Sdf.Layer.CreateAnonymous("g1-decoupled-foot-contacts.usda")
    session = stage.GetSessionLayer()
    session.subLayerPaths.insert(0, layer.identifier)
    try:
        with Usd.EditContext(stage, layer):
            for foot in feet:
                scope = stage.GetPrimAtPath(foot.GetPath().AppendChild("collisions"))
                if scope:
                    scope.SetInstanceable(False)
                sources = sources_by_foot[str(foot.GetPath())]
                for path in sources:
                    prim = stage.GetPrimAtPath(path)
                    if prim.IsInstanceProxy():
                        raise ValueError("Foot colliders are instanced outside the supported collisions scope")
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                parent_path = foot.GetPath().AppendChild(_CONTAINER_NAME)
                parent = UsdGeom.Xform.Define(stage, parent_path)
                parent.GetPrim().SetCustomDataByKey(_MARKER, {"version": 1, "source_colliders": sources})
                material = UsdShade.Material.Define(stage, parent_path.AppendChild("ContactMaterial"))
                physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                # XML inherits MuJoCo sliding friction=1, condim=3. PhysX's two
                # coefficients approximate it; no solver-softness equivalence is
                # claimed. Keep the contact offsets explicit for tiny spheres.
                physics_material.CreateStaticFrictionAttr(1.0)
                physics_material.CreateDynamicFrictionAttr(1.0)
                physics_material.CreateRestitutionAttr(0.0)
                for index, position in enumerate(FOOT_CONTACT_POINTS):
                    sphere = UsdGeom.Sphere.Define(stage, parent_path.AppendChild(f"Point{index}"))
                    sphere.CreateRadiusAttr(FOOT_CONTACT_RADIUS)
                    sphere.AddTranslateOp().Set(Gf.Vec3d(*position))
                    sphere.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                    prim = sphere.GetPrim()
                    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    collision.CreateContactOffsetAttr(0.001)
                    collision.CreateRestOffsetAttr(0.0)
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                        material, UsdShade.Tokens.strongerThanDescendants, "physics"
                    )
        report = _validate_contacts(stage, feet)
        if _body_mass_snapshot(root) != masses:
            raise RuntimeError("Decoupled foot configuration changed authored body mass properties")
        return {**report, "already_configured": False}
    except Exception:
        session.subLayerPaths.remove(layer.identifier)
        raise
