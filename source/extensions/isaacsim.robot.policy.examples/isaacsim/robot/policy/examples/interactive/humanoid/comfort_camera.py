# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stabilize robot motion beneath an independently tracked VR view.

This filter never receives a headset pose. Its output contains only the robot's
horizontal position, heading, and an explicitly calibrated eye height. Compose
the latest HMD rotation after this stage, without filtering that rotation.
"""

from __future__ import annotations

import math
from numbers import Real

from pxr import Gf


def _finite_number(value: object) -> bool:
    """Reject malformed numeric input, including booleans and numeric strings."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _wrap_angle(angle: float) -> float:
    """Wrap a heading into [-pi, pi] without changing its direction."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class ComfortCameraFilter:
    """Filter robot travel while rejecting gait bob and idle balance tremor.

    The 0.18 s XY and 0.25 s heading time constants reduce body oscillation while
    retaining continuous travel. After one second without a movement command,
    independent position and heading deadbands hold small balance corrections.
    Larger actual displacements still follow through the filter. Robot height
    is deliberately excluded; this mode assumes the factory's level floor.

    Args:
        position_tau_s: Horizontal position filter time constant in wall seconds.
        yaw_tau_s: Heading filter time constant in wall seconds.
        height_tau_s: Time constant for explicit eye-height recalibration.
        max_dt_s: Maximum elapsed interval used by one update after a stall.
        idle_delay_s: Coast-following time before applying idle deadbands.
        idle_position_deadband_m: Maximum held horizontal displacement when idle.
        idle_yaw_deadband_rad: Maximum held heading displacement when idle.

    Raises:
        ValueError: If a tuning is not finite and strictly positive.

    Example:
        .. code-block:: python

            camera_filter = ComfortCameraFilter()
            eye, yaw = camera_filter.update((0, 0, 0.8), 0, 1.22, 0.0, False)
    """

    def __init__(
        self,
        *,
        position_tau_s: float = 0.18,
        yaw_tau_s: float = 0.25,
        height_tau_s: float = 0.30,
        max_dt_s: float = 0.10,
        idle_delay_s: float = 1.0,
        idle_position_deadband_m: float = 0.04,
        idle_yaw_deadband_rad: float = 0.10,
    ) -> None:
        for name, value in (
            ("position_tau_s", position_tau_s),
            ("yaw_tau_s", yaw_tau_s),
            ("height_tau_s", height_tau_s),
            ("max_dt_s", max_dt_s),
            ("idle_delay_s", idle_delay_s),
            ("idle_position_deadband_m", idle_position_deadband_m),
            ("idle_yaw_deadband_rad", idle_yaw_deadband_rad),
        ):
            if not _finite_number(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))
        self.reset()

    def reset(self) -> None:
        """Clear pose, timing, and calibration when the robot or stage resets.

        Example:
            .. code-block:: python

                camera_filter.reset()
        """
        self._position: Gf.Vec3d | None = None
        self._yaw: float | None = None
        self._eye_height_target: float | None = None
        self._last_time_s: float | None = None
        self._idle_since_s: float | None = None

    def set_eye_height(self, height: float) -> bool:
        """Accept an explicit eye-height calibration without an abrupt jump.

        Args:
            height: Measured world-space sensor height in meters. This is a
                one-time calibration, not the body's changing gait height.

        Returns:
            Whether a finite target was accepted. Before initialization it is
            used as the initial height; later updates approach it smoothly.

        Example:
            .. code-block:: python

                camera_filter.set_eye_height(1.22)
        """
        if not _finite_number(height):
            return False
        self._eye_height_target = float(height)
        return True

    def update(
        self,
        position: Gf.Vec3d | tuple[float, float, float],
        yaw: float,
        eye_height: float,
        now_s: float,
        moving: bool,
    ) -> tuple[Gf.Vec3d, float] | None:
        """Return the filtered eye center and robot heading for a level floor.

        Args:
            position: Robot position in stage coordinates. Its Z component is
                validated but never drives the camera height.
            yaw: Robot heading around stage +Z, in radians.
            eye_height: Initial absolute eye height. After initialization only
                ``set_eye_height`` can change it; gait-varying values are ignored.
            now_s: Monotonic wall-clock timestamp, shared by all callback sources.
            moving: Whether a movement command is active, independent of head motion.

        Returns:
            A copied eye position and heading, or None before the first valid
            sample. Invalid input or non-increasing timestamps hold the previous
            filtered result without consuming time or bypassing stabilization.

        Example:
            .. code-block:: python

                eye, yaw = camera_filter.update((0.1, 0, 0.8), 0, 1.22, 0.02, True)
        """
        try:
            valid_position = len(position) == 3 and all(_finite_number(value) for value in position)
        except TypeError:
            valid_position = False
        if (
            not valid_position
            or not all(_finite_number(value) for value in (yaw, eye_height, now_s))
            or now_s < 0
            or not isinstance(moving, bool)
        ):
            return self._read_result()
        if self._last_time_s is not None and now_s <= self._last_time_s:
            return self._read_result()

        yaw = _wrap_angle(float(yaw))
        now_s = float(now_s)
        if self._position is None:
            if self._eye_height_target is None:
                self._eye_height_target = float(eye_height)
            self._position = Gf.Vec3d(float(position[0]), float(position[1]), self._eye_height_target)
            self._yaw = yaw
            self._last_time_s = now_s
            self._idle_since_s = None if moving else now_s
            return self._read_result()

        dt = min(now_s - self._last_time_s, self.max_dt_s)
        self._last_time_s = now_s
        if moving:
            self._idle_since_s = None
        elif self._idle_since_s is None:
            self._idle_since_s = now_s
        hold_idle = self._idle_since_s is not None and now_s - self._idle_since_s >= self.idle_delay_s

        dx, dy = float(position[0]) - self._position[0], float(position[1]) - self._position[1]
        yaw_delta = _wrap_angle(yaw - self._yaw)
        # Compare against the held frame, not the preceding raw observation, so
        # accumulated drift beyond the deadband remains visible and is followed.
        if not hold_idle or math.hypot(dx, dy) > self.idle_position_deadband_m:
            position_alpha = -math.expm1(-dt / self.position_tau_s)
            self._position[0] += dx * position_alpha
            self._position[1] += dy * position_alpha
        if not hold_idle or abs(yaw_delta) > self.idle_yaw_deadband_rad:
            yaw_alpha = -math.expm1(-dt / self.yaw_tau_s)
            self._yaw = _wrap_angle(self._yaw + yaw_delta * yaw_alpha)
        height_alpha = -math.expm1(-dt / self.height_tau_s)
        self._position[2] += (self._eye_height_target - self._position[2]) * height_alpha
        return self._read_result()

    def _read_result(self) -> tuple[Gf.Vec3d, float] | None:
        """Copy the last filtered result so callers cannot mutate filter history."""
        return None if self._position is None else (Gf.Vec3d(self._position), self._yaw)
