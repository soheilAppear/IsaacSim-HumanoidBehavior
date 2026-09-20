# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stabilize measured pointing geometry before both scene queries and drawing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

Vector = tuple[float, float, float]
Ray = tuple[Vector, Vector]

__all__ = ["PointRayFilter"]


def _read_vector(value: Sequence[float] | None) -> Vector | None:
    """Reject malformed or nonfinite geometry without retaining the old ray."""
    try:
        if value is None or len(value) != 3:
            return None
        if any(isinstance(component, bool) or not isinstance(component, Real) for component in value):
            return None
        vector = tuple(float(component) for component in value)
        return vector if all(math.isfinite(component) for component in vector) else None
    except (TypeError, ValueError, OverflowError):
        return None


class PointRayFilter:
    """Reduce optical finger jitter while following deliberate aiming promptly.

    Angular noise is amplified where a shallow pointing ray meets distant floor.
    Smooth its direction independently from its origin, using elapsed wall time
    so physics bursts do not count as additional tracking observations. Larger
    aiming changes shorten the angular time constant; a small angular deadband
    suppresses tiny tremor without changing collision geometry or selecting a
    different surface. No prediction or remembered scene hit is used.

    Pass the returned origin and unit direction to the actual scene query and
    draw that same ray. Call ``reset()`` on gesture release or timeline changes.
    This filter does not grant permission to walk or replace navigation's fresh
    input, dwell, obstruction, and stop checks.

    Args:
        origin_tau_s: Wall-time smoothing constant for the ray origin.
        direction_tau_s: Wall-time smoothing constant for small angular changes.
        fast_direction_tau_s: Shorter smoothing constant for large aiming changes.
        deadband_rad: Angular difference below which the direction remains still.
        adaptation_start_rad: Angular difference where faster following begins.
        adaptation_end_rad: Angular difference where the fastest following applies.
        max_gap_s: Gap after which current measured geometry replaces filter history.

    Raises:
        ValueError: If the filter constants are nonfinite, nonpositive, or inconsistent.
    """

    def __init__(
        self,
        *,
        origin_tau_s: float = 0.018,
        direction_tau_s: float = 0.100,
        fast_direction_tau_s: float = 0.022,
        deadband_rad: float = math.radians(0.25),
        adaptation_start_rad: float = math.radians(2.0),
        adaptation_end_rad: float = math.radians(12.0),
        max_gap_s: float = 0.35,
    ) -> None:
        settings = {
            "origin_tau_s": origin_tau_s,
            "direction_tau_s": direction_tau_s,
            "fast_direction_tau_s": fast_direction_tau_s,
            "deadband_rad": deadband_rad,
            "adaptation_start_rad": adaptation_start_rad,
            "adaptation_end_rad": adaptation_end_rad,
            "max_gap_s": max_gap_s,
        }
        for name, value in settings.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite real number")
            setattr(self, name, float(value))
        if self.fast_direction_tau_s > self.direction_tau_s:
            raise ValueError("The fast direction time constant must not exceed the normal one")
        if not self.deadband_rad < self.adaptation_start_rad < self.adaptation_end_rad < math.pi:
            raise ValueError("Angular thresholds must satisfy 0 < deadband < adaptation start < end < pi")
        self.reset()

    def reset(self) -> None:
        """Discard the ray so a new gesture starts at its current measured pose."""
        self._last_time_s: float | None = None
        self._origin: Vector | None = None
        self._direction: Vector | None = None

    def update(
        self,
        *,
        origin: Sequence[float] | None,
        direction: Sequence[float] | None,
        now_s: float,
        tracked: bool = True,
    ) -> Ray | None:
        """Return filtered measured geometry, or clear it when input is unavailable.

        Args:
            origin: Current fingertip position in world meters.
            direction: Current index pointing vector in the same world frame.
            now_s: Monotonic wall time for this observation, in seconds.
            tracked: Whether the hand and required pointing landmarks are tracked.

        Returns:
            A world origin and unit direction, or None for unavailable tracking,
            invalid geometry, or a duplicate/backwards clock. A long positive
            gap starts a new filter from the current measurement; the caller's
            navigation state machine must still validate its own timing.

        Example:

        .. code-block:: python

            ray = PointRayFilter().update(origin=(0, 0, 1), direction=(1, 0, -1), now_s=0.0)
        """
        measured_origin = _read_vector(origin)
        measured_direction = _read_vector(direction)
        valid_time = not isinstance(now_s, bool) and isinstance(now_s, Real) and math.isfinite(now_s) and now_s >= 0.0
        if tracked is not True or measured_origin is None or measured_direction is None or not valid_time:
            self.reset()
            return None
        length = math.hypot(*measured_direction)
        if not math.isfinite(length) or length < 1e-8:
            self.reset()
            return None
        measured_direction = tuple(component / length for component in measured_direction)
        now_s = float(now_s)
        interval = now_s - self._last_time_s if self._last_time_s is not None else None
        if interval is not None and interval <= 0.0:
            self.reset()
            return None
        if interval is None or interval > self.max_gap_s:
            self._origin, self._direction = measured_origin, measured_direction
        else:
            origin_alpha = -math.expm1(-interval / self.origin_tau_s)
            self._origin = tuple(
                previous + origin_alpha * (current - previous)
                for previous, current in zip(self._origin, measured_origin)
            )
            dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(self._direction, measured_direction))))
            angle = math.acos(dot)
            if angle > self.deadband_rad:
                blend = min(
                    1.0,
                    max(
                        0.0, (angle - self.adaptation_start_rad) / (self.adaptation_end_rad - self.adaptation_start_rad)
                    ),
                )
                tau = self.direction_tau_s + blend * (self.fast_direction_tau_s - self.direction_tau_s)
                alpha = -math.expm1(-interval / tau)
                # Spherical interpolation preserves angular speed and unit
                # length. An antipodal optical discontinuity has no unique
                # interpolation arc, so use its current measured direction.
                sine = math.sin(angle)
                if sine < 1e-8:
                    self._direction = measured_direction
                else:
                    previous_weight = math.sin((1.0 - alpha) * angle) / sine
                    measured_weight = math.sin(alpha * angle) / sine
                    interpolated = tuple(
                        previous_weight * previous + measured_weight * current
                        for previous, current in zip(self._direction, measured_direction)
                    )
                    norm = math.hypot(*interpolated)
                    self._direction = tuple(component / norm for component in interpolated)
        self._last_time_s = now_s
        return self._origin, self._direction
