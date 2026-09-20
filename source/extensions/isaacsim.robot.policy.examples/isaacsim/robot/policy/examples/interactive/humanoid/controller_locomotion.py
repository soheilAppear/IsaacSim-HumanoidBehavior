# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert only controller thumbsticks into bounded walking intent.

This module owns no XR devices, arm/finger input, scene queries, or physics. The
caller supplies currently valid analog samples, checks the physical walking
guards, and scales the normalized command to the existing learned actor's limits.
Center both sticks after startup, reset, tracking loss, or an input interruption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


@dataclass(frozen=True)
class ControllerCommand:
    """Normalized forward, body-left and yaw-left intent, never joint targets."""

    state: str = "awaiting_neutral"
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    detail: str = "Center both controller sticks to enable walking"


class ControllerLocomotion:
    """Apply deadzones and a neutral interlock to the two-stick layout.

    Left Y controls forward/backward, left X controls sideways movement, and
    right X controls turning. OpenXR stick-right has the opposite sign from
    robot body-left and yaw-left. Simultaneous diagonal translation stays within
    unit magnitude. Grips, triggers, buttons and clicks are intentionally absent.

    Args:
        deadzone: Axis magnitude that counts as centered, in [0, 1).
        max_sample_gap_s: Maximum wall-clock update gap before requiring neutral.
    """

    def __init__(self, *, deadzone: float = 0.15, max_sample_gap_s: float = 0.25) -> None:
        if not self._finite(deadzone) or not 0 <= deadzone < 1:
            raise ValueError("Stick deadzone must be finite and in [0, 1)")
        if not self._finite(max_sample_gap_s) or max_sample_gap_s <= 0:
            raise ValueError("Stick sample-gap limit must be finite and positive")
        self.deadzone = float(deadzone)
        self.max_sample_gap_s = float(max_sample_gap_s)
        self.reset()

    @staticmethod
    def _finite(value: object) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)

    def reset(self) -> None:
        """Stop and require centered sticks before accepting a new command."""
        self._armed = False
        self._last_time_s: float | None = None
        self.command = ControllerCommand()

    def _stop(self, state: str, detail: str) -> ControllerCommand:
        self._armed = False
        self.command = ControllerCommand(state=state, detail=detail)
        return self.command

    def _axis(self, value: float) -> float:
        """Map usable stick travel linearly to speed; full deflection stays 1.0.

        With the default 0.15 deadzone, 0.575 deflection requests half the
        configured directional speed. The caller applies physical speed caps.
        """
        if abs(value) <= self.deadzone:
            return 0.0
        return math.copysign((abs(value) - self.deadzone) / (1.0 - self.deadzone), value)

    def update(
        self,
        *,
        left_x: float,
        left_y: float,
        right_x: float,
        left_valid: bool,
        right_valid: bool,
        now_s: float,
        speed_scale: float = 1.0,
    ) -> ControllerCommand:
        """Read one callback's samples; invalid input stops all axes immediately.

        Device/action validity is the caller's responsibility. ``now_s`` is the
        current monotonic wall time, not an invented XR sample timestamp. Repeated
        equal timestamps are permitted for physics steps within one app frame.
        ``speed_scale`` may reduce all axes for a carrying profile, but may not
        increase the base actor's configured limits. A zero scale requires neutral
        again before a later profile can resume movement.
        """
        if not self._finite(now_s) or now_s < 0:
            self._last_time_s = None
            return self._stop("unavailable", "Invalid controller update clock; center both sticks")
        previous = self._last_time_s
        self._last_time_s = float(now_s)
        if left_valid is not True or right_valid is not True:
            return self._stop("unavailable", "Both tracked controller sticks are required; center them to resume")
        if any(not self._finite(value) or not -1 <= value <= 1 for value in (left_x, left_y, right_x)):
            return self._stop("unavailable", "Invalid controller axis sample; center both sticks")
        if not self._finite(speed_scale) or not 0 <= speed_scale <= 1:
            return self._stop("unavailable", "Invalid controller speed cap; center both sticks")
        if previous is not None and (now_s < previous or now_s - previous > self.max_sample_gap_s):
            return self._stop("awaiting_neutral", "Controller input was interrupted; center both sticks")
        if speed_scale == 0:
            return self._stop("awaiting_neutral", "Walking is inhibited; center both sticks before resuming")
        forward, lateral, yaw = self._axis(left_y), -self._axis(left_x), -self._axis(right_x)
        if forward == lateral == yaw == 0.0:
            self._armed = True
            self.command = ControllerCommand(state="idle", detail="Sticks centered; walking stopped")
            return self.command
        if not self._armed:
            return self._stop("awaiting_neutral", "Center both controller sticks to enable walking")
        magnitude = math.hypot(forward, lateral)
        if magnitude > 1.0:
            forward, lateral = forward / magnitude, lateral / magnitude
        self.command = ControllerCommand(
            state="active",
            velocity=(forward * speed_scale, lateral * speed_scale, yaw * speed_scale),
            detail="Left stick moves; right stick turns; release sticks to stop",
        )
        return self.command
