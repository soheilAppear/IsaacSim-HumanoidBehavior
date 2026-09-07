# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate gaze and both hands' temporary material highlights."""

from __future__ import annotations

from pxr import Sdf, Usd, UsdShade


class MaterialHighlights:
    """Own temporary bindings without changing the live physics scene topology.

    Multiple hands and gaze can select the same object. A shared owner registry avoids
    restoring a stale yellow/green binding when another owner leaves. Call ``prepare``
    before starting physics: prim overrides, API schemas and the session layer must
    remain stable while tensor views exist. Removing/recreating an override prim can
    make PhysX reparse its collision descendants and invalidate *all* tensor views.
    Runtime changes affect only a material relationship's targets and metadata.
    """

    def __init__(self) -> None:
        self._stage = None
        self._layer = None
        self._owners = {}
        self._prepared = set()

    def prepare(self, stage: Usd.Stage, targets: list[str]) -> None:
        """Prepare package roots before physics creates its tensor views.

        The dedicated layer stays attached for the lifetime of this stage. Empty
        relationship opinions inherit the original scene's bindings, including child
        materials. Repeated preparation of the same roots performs no USD edits.
        """
        if stage != self._stage:
            self.clear()
            self._stage = stage
            self._layer = None
            self._prepared.clear()
        if self._layer is None:
            self._layer = Sdf.Layer.CreateAnonymous("g1_highlights.usda")
            session = stage.GetSessionLayer()
            session.subLayerPaths = [self._layer.identifier, *session.subLayerPaths]
        with Usd.EditContext(stage, self._layer):
            for target in targets:
                target = str(target)
                if target in self._prepared or not stage.GetPrimAtPath(target).IsValid():
                    continue
                prim = stage.OverridePrim(target)
                UsdShade.MaterialBindingAPI.Apply(prim)
                prim.CreateRelationship("material:binding", custom=False)
                self._prepared.add(target)

    def set(
        self,
        stage: Usd.Stage | None,
        owner: str,
        target: str | None,
        material: UsdShade.Material | None,
        priority: int = 0,
    ) -> None:
        """Set or clear one owner's highlight request.

        Args:
            stage: Active scene stage, or None after a scene clear.
            owner: Stable requester name, such as gaze or grab:left.
            target: Prepared package root path, or None to release this owner's request.
            material: Material to bind when target is provided.
            priority: Larger values win when several owners select the same object.
        """
        if stage != self._stage:
            self.clear()
            return
        if stage is None or self._layer is None:
            return
        # Never add schemas/overrides here: set() runs inside the physics callback.
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
            with Usd.EditContext(stage, self._layer):
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    continue
                relationship = prim.GetRelationship("material:binding")
                candidates = [entry for entry in self._owners.values() if entry[0] == path]
                if candidates:
                    selected = max(candidates, key=lambda entry: (entry[2], entry[1]))
                    relationship.SetTargets([Sdf.Path(selected[1])])
                    relationship.SetMetadata("bindMaterialAs", UsdShade.Tokens.strongerThanDescendants)
                else:
                    self._clear_binding(relationship)

    @staticmethod
    def _clear_binding(relationship: Usd.Relationship) -> None:
        """Restore weaker scene opinions while preserving the prepared prim/spec."""
        relationship.ClearTargets(False)
        relationship.ClearMetadata("bindMaterialAs")

    def clear(self) -> None:
        """Release every highlight without removing layers, prims, or API schemas.

        This is safe while physics is running. Keeping empty override specs is
        intentional: layer removal or ``RemovePrim`` emits structural USD notices.
        """
        try:
            if self._stage is not None and self._layer is not None:
                with Usd.EditContext(self._stage, self._layer):
                    for target in {entry[0] for entry in self._owners.values()}:
                        prim = self._stage.GetPrimAtPath(target)
                        if prim.IsValid():
                            self._clear_binding(prim.GetRelationship("material:binding"))
        except (ReferenceError, RuntimeError, TypeError):
            # Kit may close the stage before calling example cleanup. Its surviving
            # Python wrapper then fails USD's C++ argument conversion (TypeError),
            # even though it is not None. There is no live scene left to restore.
            self._stage = None
            self._layer = None
            self._prepared.clear()
        finally:
            self._owners.clear()
