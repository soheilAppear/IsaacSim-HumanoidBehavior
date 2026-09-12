# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact-based finger compliance and grasp diagnostics, independent of simulator APIs.

The caller supplies measured PhysX contacts for one hand and joint angles converted
to the same normalized ranges as its finger commands. This controller never moves
objects, creates constraints, or guarantees a successful lift. ``contact_supported``
means sustained opposing contacts were observed; the object can still slip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

FINGER_ROLES = ("thumb", "index", "middle", "ring", "little")
ACTUATOR_ROLES = (*FINGER_ROLES, "thumb_yaw")
CONTACT_ROLES = (*FINGER_ROLES, "palm")


@dataclass(frozen=True)
class ContactSample:
    """One actual contact point, with world axes and force in newtons.

    ``object_path`` identifies the contacted rigid body, not an aggregate wildcard.
    Normals point along force on the hand, consistently for every sensor. Positive separation
    denotes a gap; proximity observations without a measured force do not qualify.
    Set ``eligible_for_grasp=False`` for scenery or other robot links: their
    contacts still limit closing travel but cannot establish object support.
    """

    object_path: str
    role: str
    point_world: tuple[float, float, float]
    normal_world: tuple[float, float, float]
    normal_force_n: float
    separation_m: float = 0.0
    eligible_for_grasp: bool = True


@dataclass(frozen=True)
class GraspResult:
    """Finger commands and contact evidence; no attachment or object-control action."""

    curls: dict[str, float]
    contact_supported: bool = False
    object_path: str | None = None
    mode: str | None = None
    contact_roles: tuple[str, ...] = ()
    closing_intent: bool = False


@dataclass
class _ContactAnchor:
    """Maximum curl retained from one actuator's first contact with a body."""

    limit: float
    missing_s: float = 0.0


class ContactGraspController:
    """Limit contact-loaded finger travel and recognize a pinch or wrap.

    Use one instance per hand. Preparation of sensors, object eligibility, joint
    effort limits, friction, and drop-button inhibition belong to the caller.
    Only current contacts should be supplied: stale frames must not be replayed.
    """

    def __init__(
        self,
        *,
        support_time_s: float = 0.06,
        min_force_n: float = 0.02,
        max_separation_m: float = 0.002,
        min_point_spacing_m: float = 0.008,
        opposing_normal_dot: float = -0.5,
        contact_grace_s: float = 0.04,
    ) -> None:
        values = (
            support_time_s,
            min_force_n,
            max_separation_m,
            min_point_spacing_m,
            opposing_normal_dot,
            contact_grace_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Contact thresholds must be finite")
        if support_time_s <= 0 or min_force_n <= 0 or max_separation_m < 0 or min_point_spacing_m <= 0:
            raise ValueError("Contact time, force, and point spacing must be positive; separation must be nonnegative")
        if not -1.0 <= opposing_normal_dot < 0.0:
            raise ValueError("Opposing normal threshold must be in [-1, 0)")
        if contact_grace_s < 0:
            raise ValueError("Contact grace time must be nonnegative")
        self.support_time_s = support_time_s
        self.min_force_n = min_force_n
        self.max_separation_m = max_separation_m
        self.min_point_spacing_m = min_point_spacing_m
        self.opposing_normal_dot = opposing_normal_dot
        self.contact_grace_s = contact_grace_s
        self.reset()

    def reset(self) -> None:
        """Discard contact anchors, dwell, and intent after reset or input loss."""
        self._candidate_path: str | None = None
        self._support_elapsed = 0.0
        self._previous_requested: dict[str, float] = {}
        self._closing_intent_remaining = 0.0
        self._contact_anchors: dict[str, dict[str, _ContactAnchor]] = {}

    def _contact_limited_curls(
        self,
        requested: Mapping[str, float],
        measured_curls: Mapping[str, float],
        closing_lead: Mapping[str, float],
        contacts: list[ContactSample],
        elapsed: float,
    ) -> dict[str, float]:
        paths_by_role: dict[str, set[str]] = {role: set() for role in ACTUATOR_ROLES}
        for contact in contacts:
            if contact.role in paths_by_role:
                paths_by_role[contact.role].add(contact.object_path)
            if contact.role == "thumb":
                paths_by_role["thumb_yaw"].add(contact.object_path)
        curls = dict(requested)
        for role, paths in paths_by_role.items():
            anchors = self._contact_anchors.get(role, {})
            measured = self._curl(measured_curls.get(role, 0.0))
            lead = self._curl(closing_lead.get(role, 0.0))
            limit = min(1.0, measured + lead)
            if paths:
                # A different body establishes a fresh contact pose. Preserve
                # only bodies still touching this actuator, including scenery.
                anchors = {path: anchors.get(path, _ContactAnchor(limit)) for path in paths}
                for anchor in anchors.values():
                    anchor.missing_s = 0.0
            elif anchors:
                for anchor in anchors.values():
                    anchor.missing_s += elapsed
                anchors = {
                    path: anchor for path, anchor in anchors.items() if anchor.missing_s + 1e-12 < self.contact_grace_s
                }
            if anchors:
                self._contact_anchors[role] = anchors
                # Keep the original preload budget separate from the changing
                # request and current joint state. Neither a smoothed input ramp
                # nor a solver excursion may permanently ratchet it downward.
                curls[role] = min(requested[role], limit, *(anchor.limit for anchor in anchors.values()))
            else:
                self._contact_anchors.pop(role, None)
        return curls

    @staticmethod
    def _curl(value, default: float = 0.0) -> float:
        try:
            value = float(value)
            return min(1.0, max(0.0, value)) if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    def _valid_contact(self, contact: ContactSample) -> ContactSample | None:
        try:
            if not isinstance(contact.object_path, str) or not contact.object_path or contact.role not in CONTACT_ROLES:
                return None
            point = tuple(float(value) for value in contact.point_world)
            normal = tuple(float(value) for value in contact.normal_world)
            force = float(contact.normal_force_n)
            separation = float(contact.separation_m)
            if len(point) != 3 or len(normal) != 3:
                return None
            if not all(math.isfinite(value) for value in (*point, *normal, force, separation)):
                return None
            if force < self.min_force_n:
                return None
            length = math.hypot(*normal)
            if not math.isfinite(length) or length <= 1e-9:
                return None
            return ContactSample(
                contact.object_path,
                contact.role,
                point,
                tuple(v / length for v in normal),
                force,
                separation,
                eligible_for_grasp=contact.eligible_for_grasp is True,
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None

    def _opposed(self, first: ContactSample, second: ContactSample) -> bool:
        normal_dot = sum(a * b for a, b in zip(first.normal_world, second.normal_world))
        spacing = math.dist(first.point_world, second.point_world)
        return normal_dot <= self.opposing_normal_dot and spacing >= self.min_point_spacing_m

    def _classify(self, contacts: list[ContactSample]) -> tuple[str, tuple[str, ...]] | None:
        by_role = {role: [contact for contact in contacts if contact.role == role] for role in CONTACT_ROLES}

        def opposed_roles(first: str, second: str) -> bool:
            return any(self._opposed(a, b) for a in by_role[first] for b in by_role[second])

        # A wrap needs two distinct digits against a common thumb/palm surface.
        # Multiple manifold points or two links on one finger count as one digit.
        for anchor in ("thumb", "palm"):
            digits = tuple(role for role in FINGER_ROLES[1:] if opposed_roles(anchor, role))
            if len(digits) >= 2:
                return "wrap", (anchor, *digits)
        for digit in ("index", "middle"):
            if opposed_roles("thumb", digit):
                return "pinch", ("thumb", digit)
        return None

    def update(
        self,
        *,
        dt: float,
        requested_curls: Mapping[str, float],
        measured_curls: Mapping[str, float],
        contacts: Iterable[ContactSample],
        closing_lead: Mapping[str, float],
        input_valid: bool = True,
    ) -> GraspResult:
        """Compute one finger command and assess current opposing contact evidence.

        ``closing_lead`` is per actuator in normalized curl units. Convert a desired
        radian lead using that actuator's actual open-to-closed travel. First
        contact anchors the maximum curl to the measured pose plus this lead.
        That ceiling stays fixed while touching the same body. Commands also stay
        within the current measured pose plus lead, so an external force pushing
        the finger open cannot create a large closing error. Neither that current
        bound nor the input ramp changes the first-contact ceiling. Opening is
        unrestricted and free fingers follow the input. A brief
        contact-report gap retains only the command limit, never grasp evidence.
        These limits create compliance through the caller's effort-limited
        position drives; they do not command a contact force or guarantee a hold.

        ``input_valid`` refers to the finger input, independently of arm clutch state.
        Recent increasing curls indicate closing intent, but are not a requirement
        for physical support: an already curved hand may encounter an object later.
        """
        if not input_valid:
            self.reset()
            return GraspResult({role: 0.0 for role in ACTUATOR_ROLES})
        requested = {role: self._curl(requested_curls.get(role, 0.0)) for role in ACTUATOR_ROLES}
        if all(requested[role] < 0.08 for role in ACTUATOR_ROLES):
            self.reset()
            self._previous_requested = requested.copy()
            return GraspResult(requested)
        # A stalled or invalid frame cannot manufacture a completed dwell interval.
        valid_dt = isinstance(dt, (int, float)) and math.isfinite(dt) and dt > 0
        elapsed = min(float(dt), 0.05) if valid_dt else 0.0
        self._closing_intent_remaining = max(0.0, self._closing_intent_remaining - elapsed)
        if valid_dt and any(
            requested[role] >= 0.08
            and requested[role] - self._previous_requested.get(role, requested[role]) >= max(0.0005, 0.05 * elapsed)
            for role in FINGER_ROLES
        ):
            self._closing_intent_remaining = 0.4
        self._previous_requested = requested.copy()

        valid_contacts = [sample for contact in contacts if (sample := self._valid_contact(contact)) is not None]
        curls = self._contact_limited_curls(requested, measured_curls, closing_lead, valid_contacts, elapsed)

        by_object: dict[str, list[ContactSample]] = {}
        for contact in valid_contacts:
            # Predictive contacts at a positive gap and contacts with scenery
            # still load the fingers, so they receive compliance above. Only
            # sufficiently close contacts with eligible objects establish support.
            if contact.eligible_for_grasp and contact.separation_m <= self.max_separation_m:
                by_object.setdefault(contact.object_path, []).append(contact)
        candidates = {}
        for object_path, object_contacts in by_object.items():
            classification = self._classify(object_contacts)
            if classification is not None:
                candidates[object_path] = (*classification, sum(contact.normal_force_n for contact in object_contacts))
        if not valid_dt or not candidates:
            self._candidate_path = None
            self._support_elapsed = 0.0
            return GraspResult(curls, closing_intent=self._closing_intent_remaining > 0.0)

        # Retain the same valid candidate through changes in its manifold points.
        # Otherwise choose deterministically by measured force and then body path.
        path = self._candidate_path
        if path not in candidates:
            path = max(sorted(candidates), key=lambda item: candidates[item][2])
            self._candidate_path = path
            self._support_elapsed = 0.0
        self._support_elapsed += elapsed
        mode, roles, _ = candidates[path]
        return GraspResult(
            curls=curls,
            contact_supported=self._support_elapsed + 1e-12 >= self.support_time_s,
            object_path=path,
            mode=mode,
            contact_roles=roles,
            closing_intent=self._closing_intent_remaining > 0.0,
        )
