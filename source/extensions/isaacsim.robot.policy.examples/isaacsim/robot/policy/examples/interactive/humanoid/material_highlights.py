# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate gaze/hand colors without editing a running rigid body's USD subtree."""

from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


class MaterialHighlights:
    """Preauthor invisible render shells, then change only external shader inputs.

    Even a render-purpose material binding change on a live carton can make PhysX
    recreate it from its stale USD pose while Fabric owns the simulated transform.
    Each prepared carton therefore gets nonphysical copies of its visible geometry
    and a permanently bound material outside the body's subtree. Opacity reveals
    the shell in the selected owner's color; removing selection reveals the untouched
    original visuals. The shell inherits the body's Fabric motion without pose writes.
    """

    MATERIAL_ROOT = "/World/G1_HighlightMaterials"
    SHELL_CHILD = "G1_HighlightShells"
    SHELL_SCALE = 1.005

    def __init__(self) -> None:
        self._stage = None
        self._layer = None
        self._owners = {}
        self._prepared = set()
        self._active_bindings = {}
        self._shader_paths = {}

    def prepare(self, stage: Usd.Stage, targets: list[str]) -> None:
        """Copy visible geometry and author materials once, before physics starts.

        Copies contain geometry attributes and a baked body-relative transform,
        never source references, relationships or physics APIs. Repeated preparation
        performs no edits. Direct USD is needed here to copy arbitrary Gprim schemas
        while deliberately excluding physics schemas and composition arcs.
        """
        if stage != self._stage:
            self._discard_stage()
            self._stage = stage
        if self._layer is None:
            self._layer = Sdf.Layer.CreateAnonymous("g1_highlights.usda")
            session = stage.GetSessionLayer()
            session.subLayerPaths = [self._layer.identifier, *session.subLayerPaths]
        with Usd.EditContext(stage, self._layer):
            for target in targets:
                target = str(target)
                root = stage.GetPrimAtPath(target)
                if target in self._prepared or not root.IsValid():
                    continue
                # Enumerate before creating shells so a later traversal cannot
                # accidentally duplicate our own overlay geometry.
                sources = [
                    prim
                    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                    if prim.IsA(UsdGeom.Gprim)
                    and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
                    and UsdGeom.Imageable(prim).ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
                ]
                shell_root = f"{target}/{self.SHELL_CHILD}"
                if stage.GetPrimAtPath(shell_root).IsValid():
                    raise ValueError(f"Highlight shell path is already used: {shell_root}")
                material_path = f"{self.MATERIAL_ROOT}/Package_{len(self._prepared):04d}"
                material = UsdShade.Material.Define(stage, material_path)
                shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0))
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.0)
                shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                UsdGeom.Xform.Define(stage, shell_root)
                root_inverse = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
                bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                for index, source in enumerate(sources):
                    self._copy_shell(
                        stage, source, f"{shell_root}/Geometry_{index:04d}", root_inverse, bounds, material
                    )
                self._shader_paths[target] = str(shader.GetPath())
                self._prepared.add(target)

    @classmethod
    def _copy_shell(cls, stage, source, path, root_inverse, bounds, material) -> None:
        """Copy render geometry only; source collider and rigid-body APIs stay behind."""
        schema = getattr(UsdGeom, str(source.GetTypeName()), None)
        if schema is None or not hasattr(schema, "GetSchemaAttributeNames"):
            raise ValueError(f"Unsupported highlight geometry type: {source.GetTypeName()}")
        geometry_names = set(schema.GetSchemaAttributeNames())
        shell = stage.DefinePrim(path, source.GetTypeName())
        for attr in source.GetAttributes():
            name = attr.GetName()
            if name == "xformOpOrder" or name.startswith("xformOp:"):
                continue
            if name not in geometry_names and not name.startswith("primvars:"):
                continue
            # Keep schema fallbacks as fallbacks. In particular a Cube's default
            # extent is for size 2; authoring it on a size-0.2 copy would give the
            # renderer/bounds consumers a box ten times larger than its geometry.
            if not attr.HasAuthoredValueOpinion():
                continue
            value = attr.Get()
            if value is None:
                continue
            copied = shell.CreateAttribute(name, attr.GetTypeName(), custom=False, variability=attr.GetVariability())
            copied.Set(value)
            for key in ("interpolation", "elementSize"):
                if attr.HasAuthoredMetadata(key):
                    copied.SetMetadata(key, attr.GetMetadata(key))
        # Expand about the geometry's own bounds center, not its potentially
        # distant imported origin. A 0.5% shell avoids coplanar depth flicker.
        box = bounds.ComputeUntransformedBound(source).ComputeAlignedRange()
        center = Gf.Vec3d(0.0) if box.IsEmpty() else box.GetMidpoint()
        expansion = (
            Gf.Matrix4d().SetTranslate(-center)
            * Gf.Matrix4d().SetScale(Gf.Vec3d(cls.SHELL_SCALE))
            * Gf.Matrix4d().SetTranslate(center)
        )
        relative = UsdGeom.Xformable(source).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * root_inverse
        UsdGeom.Xformable(shell).AddTransformOp().Set(expansion * relative)
        UsdShade.MaterialBindingAPI.Apply(shell).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose=UsdShade.Tokens.full
        )

    def set(
        self,
        stage: Usd.Stage | None,
        owner: str,
        target: str | None,
        material: UsdShade.Material | None,
        priority: int = 0,
    ) -> None:
        """Select an owner's color without authoring anything below a rigid body."""
        if stage != self._stage:
            self._discard_stage()
            return
        if stage is None or self._layer is None:
            return
        if target and (target not in self._prepared or not material or not stage.GetPrimAtPath(target).IsValid()):
            target = None
        request = (target, str(material.GetPath()), priority) if target else None
        old = self._owners.get(owner)
        if old == request:
            return
        self._owners.pop(owner, None)
        if request is not None:
            self._owners[owner] = request
        for path in {entry[0] for entry in (old, request) if entry is not None}:
            candidates = [entry for entry in self._owners.values() if entry[0] == path]
            selected = max(candidates, key=lambda entry: (entry[2], entry[1]))[1] if candidates else None
            if self._active_bindings.get(path) == selected:
                continue
            with Usd.EditContext(stage, self._layer):
                shader = UsdShade.Shader(stage.GetPrimAtPath(self._shader_paths[path]))
                if selected is not None:
                    source, _, _ = UsdShade.Material(stage.GetPrimAtPath(selected)).ComputeSurfaceSource()
                    for name in ("diffuseColor", "emissiveColor"):
                        color = source.GetInput(name).Get() if source and source.GetInput(name) else Gf.Vec3f(1.0)
                        if shader.GetInput(name).Get() != color:
                            shader.GetInput(name).Set(color)
                    shader.GetInput("opacity").Set(1.0)
                    self._active_bindings[path] = selected
                else:
                    shader.GetInput("opacity").Set(0.0)
                    self._active_bindings.pop(path, None)

    def clear(self) -> None:
        """Hide every shell using only our independent layer, even after stage close."""
        if self._layer is not None:
            for target in self._active_bindings:
                attribute = self._layer.GetAttributeAtPath(
                    Sdf.Path(self._shader_paths[target]).AppendProperty("inputs:opacity")
                )
                if attribute is not None:
                    attribute.default = 0.0
        self._owners.clear()
        self._active_bindings.clear()

    def _discard_stage(self) -> None:
        """Forget a replaced stage without invoking its possibly closed wrapper."""
        self._owners.clear()
        self._active_bindings.clear()
        self._shader_paths.clear()
        self._prepared.clear()
        self._stage = None
        self._layer = None
