# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect unusable G1 joint states before teleoperation consumes them.

These bounds contain solver divergence; they are not actuator specifications or a
collision controller. The G1 has bounded revolute joints. A 0.35 rad joint-limit
margin tolerates small contact overshoot, while the 100 rad/s velocity ceiling is
over five times even the thumb mimic speed (2.4 times the configured 8 rad/s).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def find_articulation_fault(
    names: Sequence[str],
    positions: Sequence[float],
    velocities: Sequence[float],
    limits: Sequence[tuple[float, float]],
    *,
    limit_margin_rad: float = 0.35,
    max_position_rad: float = 20.0,
    max_velocity_rad_s: float = 100.0,
) -> str | None:
    """Return the first invalid joint observation, or None for plausible state.

    All values are measured articulation state in radians and radians per second.
    Infinite authored limits are allowed, but the absolute G1 bounds still apply.
    Check every joint, including passive finger joints and the waist; a fixed
    pelvis alone cannot establish that the rest of the articulation is healthy.
    """
    try:
        count = len(names)
        if count == 0 or any(len(values) != count for values in (positions, velocities, limits)):
            return "incomplete articulation joint state"
        for index, name in enumerate(names):
            position = float(positions[index])
            velocity = float(velocities[index])
            lower, upper = (float(value) for value in limits[index])
            if not math.isfinite(position) or not math.isfinite(velocity):
                return f"{name}: nonfinite joint state (q={position}, qd={velocity})"
            if abs(position) > max_position_rad:
                return f"{name}: impossible joint angle {position:.6g} rad"
            if abs(velocity) > max_velocity_rad_s:
                return f"{name}: excessive joint velocity {velocity:.6g} rad/s"
            if math.isnan(lower) or math.isnan(upper) or lower > upper:
                return f"{name}: invalid authored joint limits"
            if position < lower - limit_margin_rad or position > upper + limit_margin_rad:
                return f"{name}: joint angle {position:.6g} rad exceeds limits [{lower:.6g}, {upper:.6g}]"
    except (TypeError, ValueError, IndexError, OverflowError):
        return "malformed articulation joint state"
    return None
