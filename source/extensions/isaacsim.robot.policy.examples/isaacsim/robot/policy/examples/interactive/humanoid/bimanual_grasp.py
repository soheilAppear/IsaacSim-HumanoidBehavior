# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe two-hand contact support without commanding fingers or object forces.

Opposing contacts may belong to different palms/fingers on the same object. The
single-hand grasp classifiers cannot recognize that configuration independently.
This diagnostic never closes a hand, changes drives, adds adhesion, or attaches
an object. Sustained contact evidence does not establish a successful lift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:
    from .grasp_controller import ContactSample

FINGER_ROLES = ("thumb", "index", "middle", "ring", "little")
CONTACT_ROLES = (*FINGER_ROLES, "palm")
SIDES = ("left", "right")


@dataclass(frozen=True)
class BimanualGraspResult:
    """Current contact evidence and a friction-capacity estimate, not pressure.

    Forces are newtons, not pressure per area. ``contact_supported`` means the
    opposing-contact dwell passed. ``capacity_exceeds_weight`` compares an upper
    vertical force capacity with gravity only: it excludes torque balance,
    acceleration, slip, and other object contacts. Neither flag proves a lift.
    """

    state: str = "idle"
    object_path: str | None = None
    contact_supported: bool = False
    opposing_contacts: bool = False
    normal_force_n_by_side: dict[str, float] = field(default_factory=lambda: dict.fromkeys(SIDES, 0.0))
    contact_count_by_side: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SIDES, 0))
    support_elapsed_s: float = 0.0
    mass_kg: float | None = None
    friction_coefficient: float | None = None
    required_vertical_support_n: float | None = None
    estimated_friction_support_n: float = 0.0
    measured_normal_vertical_n: float = 0.0
    estimated_total_vertical_capacity_n: float = 0.0
    capacity_exceeds_weight: bool | None = None
    detail: str = "Waiting for valid two-hand object contacts"


@dataclass(frozen=True)
class _MeasuredContact:
    """Retain a validated world contact with a unit normal and force in newtons."""

    path: str
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    force: float


class BimanualGraspMonitor:
    """Recognize current opposing cross-hand contacts after a short simulation dwell.

    The caller supplies complete fresh contact samples and optical finger intent.
    Eligible-object membership comes from the existing contact reader. Unknown
    mass leaves weight comparison unavailable; it does not invent an object mass.

    Args:
        support_time_s: Simulation-time dwell required for opposing contacts.
        min_force_n: Minimum force on each side of an opposing contact pair.
        max_separation_m: Largest contact gap included in the diagnostic.
        min_point_spacing_m: Minimum distance between opposing contact points.
        opposing_normal_dot: Upper dot-product threshold for opposing unit normals.
        min_curl_intent: Minimum curl reached by at least one finger on each hand.
        max_contacts_per_hand: Input capacity; exceeding it invalidates the update.
        gravity_m_s2: Acceleration used to estimate the object's weight.

    Raises:
        ValueError: A threshold or contact capacity is outside its permitted range.
    """

    def __init__(
        self,
        *,
        support_time_s: float = 0.06,
        min_force_n: float = 0.02,
        max_separation_m: float = 0.002,
        min_point_spacing_m: float = 0.008,
        opposing_normal_dot: float = -0.5,
        min_curl_intent: float = 0.08,
        max_contacts_per_hand: int = 128,
        gravity_m_s2: float = 9.81,
    ) -> None:
        for name, value in (
            ("support_time_s", support_time_s),
            ("min_force_n", min_force_n),
            ("min_point_spacing_m", min_point_spacing_m),
            ("gravity_m_s2", gravity_m_s2),
        ):
            if not self._finite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))
        if not self._finite(max_separation_m) or max_separation_m < 0:
            raise ValueError("Maximum contact separation must be finite and nonnegative")
        if not self._finite(opposing_normal_dot) or not -1 <= opposing_normal_dot < 0:
            raise ValueError("Opposing normal threshold must be in [-1, 0)")
        if not self._finite(min_curl_intent) or not 0 < min_curl_intent <= 1:
            raise ValueError("Minimum curl intent must be in (0, 1]")
        if (
            isinstance(max_contacts_per_hand, bool)
            or not isinstance(max_contacts_per_hand, int)
            or max_contacts_per_hand < 1
        ):
            raise ValueError("Contact capacity must be a positive integer")
        self.max_separation_m = float(max_separation_m)
        self.opposing_normal_dot = float(opposing_normal_dot)
        self.min_curl_intent = float(min_curl_intent)
        self.max_contacts_per_hand = max_contacts_per_hand
        self.reset()

    @staticmethod
    def _finite(value: object) -> bool:
        """Accept finite real measurements while excluding boolean flags."""
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)

    def reset(self) -> None:
        """Discard dwell; previous contact samples can never establish a new hold."""
        self._candidate_path: str | None = None
        self._support_elapsed_s = 0.0
        self.result = BimanualGraspResult()

    def _unavailable(self, detail: str) -> BimanualGraspResult:
        """Clear contact dwell and report why this update cannot provide evidence."""
        self.reset()
        self.result = BimanualGraspResult(state="unavailable", detail=detail)
        return self.result

    def _collect(self, values: Iterable[ContactSample]) -> dict[str, list[_MeasuredContact]]:
        """Validate eligible contacts and group positive-force samples by object.

        Invalid eligible measurements or an exceeded input capacity raise rather
        than silently producing a diagnostic from incomplete evidence.
        """
        grouped: dict[str, list[_MeasuredContact]] = {}
        for index, sample in enumerate(values):
            if index >= self.max_contacts_per_hand:
                raise ValueError("Contact diagnostic capacity exceeded; no contacts were truncated")
            if sample.eligible_for_grasp is not True:
                continue
            if not isinstance(sample.object_path, str) or not sample.object_path.startswith("/"):
                raise ValueError("Eligible contact has an invalid object path")
            if sample.role not in CONTACT_ROLES:
                raise ValueError("Eligible contact has an invalid hand role")
            point, normal = tuple(sample.point_world), tuple(sample.normal_world)
            force, separation = sample.normal_force_n, sample.separation_m
            if (
                len(point) != 3
                or len(normal) != 3
                or not all(self._finite(v) for v in (*point, *normal, force, separation))
            ):
                raise ValueError("Eligible contact contains invalid measurements")
            if force < 0:
                raise ValueError("Contact force magnitude must be nonnegative")
            length = math.hypot(*normal)
            if not math.isfinite(length) or length <= 1e-9:
                raise ValueError("Contact normal has no finite direction")
            if force == 0 or separation > self.max_separation_m:
                continue
            grouped.setdefault(sample.object_path, []).append(
                _MeasuredContact(sample.object_path, point, tuple(axis / length for axis in normal), float(force))
            )
        return grouped

    def _opposed(self, left: list[_MeasuredContact], right: list[_MeasuredContact]) -> bool:
        """Require one separated cross-hand pair with sufficient opposing forces."""
        return any(
            first.force >= self.min_force_n
            and second.force >= self.min_force_n
            and sum(a * b for a, b in zip(first.normal, second.normal)) < self.opposing_normal_dot
            and math.dist(first.point, second.point) >= self.min_point_spacing_m
            for first in left
            for second in right
        )

    def update(
        self,
        *,
        dt: float,
        contacts_by_side: Mapping[str, Iterable[ContactSample]],
        curls_by_side: Mapping[str, Mapping[str, float]],
        input_valid_by_side: Mapping[str, bool],
        object_masses_kg: Mapping[str, float],
        contact_health: str = "ok",
        friction_coefficient: float = 0.6,
    ) -> BimanualGraspResult:
        """Return evidence only; the caller keeps all physical contact safeguards.

        The default friction coefficient matches the configured practice material;
        pass the actual material coefficient when it differs. For each measured
        hand-normal force N and unit normal n (force direction ON THE HAND), the
        force on the object has vertical component ``-N*n_z``. Coulomb friction
        can supply at most ``mu*N*sqrt(1-n_z**2)`` additional upward force. For
        opposing side palms, this becomes ``mu*(N_left+N_right) >= mass*g``.
        This is a capacity estimate, not measured friction, pressure, slip or lift
        success. A common object and sustained opposing geometry are separate
        evidence. Invalid input/contact health resets dwell immediately.
        """
        if contact_health != "ok":
            return self._unavailable("Contact reader is not healthy: " + str(contact_health))
        if not self._finite(dt) or dt <= 0:
            return self._unavailable("Contact diagnostic requires a positive finite simulation timestep")
        if not self._finite(friction_coefficient) or friction_coefficient < 0:
            return self._unavailable("Friction coefficient must be finite and nonnegative")
        try:
            for side in SIDES:
                if input_valid_by_side.get(side) is not True:
                    return self._unavailable("Both optical hand inputs must be valid")
                curls = curls_by_side[side]
                values = [curls[role] for role in FINGER_ROLES]
                if not all(self._finite(v) and 0 <= v <= 1 for v in values):
                    return self._unavailable("Finger intent must contain finite normalized curls")
                if not any(value >= self.min_curl_intent for value in values):
                    self.reset()
                    self.result = BimanualGraspResult(detail="Both hands need curved-finger intent for a bimanual wrap")
                    return self.result
            grouped = {side: self._collect(contacts_by_side[side]) for side in SIDES}
            common = grouped["left"].keys() & grouped["right"].keys()
            opposed = {path for path in common if self._opposed(grouped["left"][path], grouped["right"][path])}
            candidates = opposed or common or (grouped["left"].keys() | grouped["right"].keys())
            if not candidates:
                self.reset()
                return self.result
            path = (
                self._candidate_path
                if self._candidate_path in opposed
                else min(
                    candidates,
                    key=lambda item: (
                        -sum(sample.force for side in SIDES for sample in grouped[side].get(item, ())),
                        item,
                    ),
                )
            )
            mass = object_masses_kg.get(path)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
            return self._unavailable("Invalid bimanual contact input: " + str(error))

        if path in opposed:
            if path != self._candidate_path:
                self._support_elapsed_s = 0.0
            self._candidate_path = path
            # A delayed callback cannot manufacture the entire sustained dwell.
            self._support_elapsed_s = min(self.support_time_s, self._support_elapsed_s + min(float(dt), 0.05))
        else:
            self._candidate_path = None
            self._support_elapsed_s = 0.0
        supported = path in opposed and self._support_elapsed_s + 1e-12 >= self.support_time_s
        samples = [sample for side in SIDES for sample in grouped[side].get(path, ())]
        measured_vertical = sum(-sample.force * sample.normal[2] for sample in samples)
        friction = sum(
            friction_coefficient * sample.force * math.sqrt(max(0.0, 1.0 - sample.normal[2] ** 2)) for sample in samples
        )
        mass = float(mass) if self._finite(mass) and mass > 0 else None
        required = mass * self.gravity_m_s2 if mass is not None else None
        capacity = measured_vertical + friction
        forces = {side: sum(sample.force for sample in grouped[side].get(path, ())) for side in SIDES}
        if not all(math.isfinite(value) for value in (*forces.values(), measured_vertical, friction, capacity)) or (
            required is not None and not math.isfinite(required)
        ):
            return self._unavailable("Contact capacity calculation exceeded finite measurement limits")
        state = "contact_supported" if supported else "contact_dwell" if path in opposed else "not_opposed"
        self.result = BimanualGraspResult(
            state=state,
            object_path=path,
            contact_supported=supported,
            opposing_contacts=path in opposed,
            normal_force_n_by_side=forces,
            contact_count_by_side={side: len(grouped[side].get(path, ())) for side in SIDES},
            support_elapsed_s=self._support_elapsed_s,
            mass_kg=mass,
            friction_coefficient=float(friction_coefficient),
            required_vertical_support_n=required,
            estimated_friction_support_n=friction,
            measured_normal_vertical_n=measured_vertical,
            estimated_total_vertical_capacity_n=capacity,
            capacity_exceeds_weight=capacity >= required if required is not None else None,
            detail=(
                "Sustained opposing two-hand contacts; capacity is an estimate, not a confirmed lift"
                if supported
                else "Waiting for sustained opposing contacts on the same object"
            )
            + ("; object mass unavailable" if mass is None else ""),
        )
        return self.result
