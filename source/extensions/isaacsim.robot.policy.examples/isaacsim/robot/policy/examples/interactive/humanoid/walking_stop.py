# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded foot-support feedback after the operator requests zero walking velocity.

This experimental regulator returns actor commands only. It owns no simulation,
root pose, motor, hand, or grasp state. The caller must supervise actual motion
against the original operator request; corrections must never reset that guard.
A delayed stance anchor and signed support-motion window avoid treating normal
pelvis sway as continuing walking. Physical validation under load is required.
"""

from __future__ import annotations

import math
from collections import deque
from numbers import Real

__all__ = ["WalkingStopRegulator"]
ZERO_COMMAND = (0.0, 0.0, 0.0)


def _number(value: object, name: str) -> float:
    """Require a finite real scalar, excluding booleans."""
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _vector(value: object, size: int, name: str) -> tuple[float, ...]:
    """Require a finite vector of the documented dimension."""
    try:
        if len(value) != size:
            raise ValueError(f"{name} must contain {size} values")
        return tuple(_number(component, name) for component in value)
    except TypeError as error:
        raise ValueError(f"{name} must contain {size} values") from error


class WalkingStopRegulator:
    """Regulate foot-centroid position and mean sole heading through actor input.

    Nonzero operator intent passes its ordinary filtered command unchanged.
    At zero intent, allow normal braking before capturing a fixed support anchor.
    Estimate signed support velocity over an exact interpolated pose window,
    then combine position error and velocity damping with continuous deadbands.
    Low-pass the capped correction in world coordinates before rotating XY into
    body-forward/body-left axes. No pelvis velocity or root-state write is used.

    Args:
        linear_gain: Foot-position error gain in inverse seconds.
        yaw_gain: Mean sole-heading error gain in inverse seconds.
        linear_damping: Signed foot-centroid velocity damping gain.
        yaw_damping: Signed mean sole-heading rate damping gain.
        braking_s: Unassisted zero-intent interval before anchor capture.
        window_s: Positive support-velocity measurement window in seconds.
        filter_s: Positive correction-filter time constant in simulation seconds.
        max_linear_speed: Maximum planar correction magnitude in meters/second.
        max_yaw_speed: Maximum yaw correction magnitude in radians/second.
        position_deadband: Foot-centroid displacement deadband in meters.
        heading_deadband: Mean sole-heading displacement deadband in radians.
        linear_deadband: Support-speed deadband in meters/second.
        yaw_deadband: Support yaw-rate deadband in radians/second.
        tail_threshold: Ordinary command-tail threshold per planar/yaw channel.
    """

    INTENT_EPSILON = 1e-6

    def __init__(
        self,
        *,
        linear_gain: float = 1.5,
        yaw_gain: float = 1.5,
        linear_damping: float = 0.7,
        yaw_damping: float = 0.7,
        braking_s: float = 1.0,
        window_s: float = 0.4,
        filter_s: float = 0.15,
        max_linear_speed: float = 0.20,
        max_yaw_speed: float = 0.30,
        position_deadband: float = 0.02,
        heading_deadband: float = 0.05,
        linear_deadband: float = 0.02,
        yaw_deadband: float = 0.05,
        tail_threshold: float = 0.02,
    ) -> None:
        parameters = dict(
            linear_gain=linear_gain,
            yaw_gain=yaw_gain,
            linear_damping=linear_damping,
            yaw_damping=yaw_damping,
            braking_s=braking_s,
            window_s=window_s,
            filter_s=filter_s,
            max_linear_speed=max_linear_speed,
            max_yaw_speed=max_yaw_speed,
            position_deadband=position_deadband,
            heading_deadband=heading_deadband,
            linear_deadband=linear_deadband,
            yaw_deadband=yaw_deadband,
            tail_threshold=tail_threshold,
        )
        for name, value in parameters.items():
            value = _number(value, name)
            positive = name in ("filter_s", "window_s")
            if value < 0.0 or (positive and value == 0.0):
                raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
            setattr(self, name, value)
        self.reset()

    @property
    def status(self) -> dict:
        """Return commands and support evidence as a defensive snapshot."""
        return dict(self._status)

    def reset(self) -> None:
        """Discard the anchor, history and correction on lifecycle/input changes."""
        self._elapsed_s = self._filter_time_s = 0.0
        self._velocity = (0.0, 0.0)
        self._yaw_rate = 0.0
        self._support_pose = self._anchor = None
        self._samples = deque()
        self._correction_world = ZERO_COMMAND
        self._position_error = (0.0, 0.0)
        self._heading_error = 0.0
        self._publish("reset")

    def _publish(
        self,
        phase: str,
        requested: tuple = ZERO_COMMAND,
        filtered: tuple = ZERO_COMMAND,
        correction: tuple = ZERO_COMMAND,
        effective: tuple = ZERO_COMMAND,
    ) -> tuple[float, float, float]:
        self._status = {
            "phase": phase,
            "active": phase == "regulating",
            "requested": requested,
            "filtered": filtered,
            "correction": correction,
            "effective": effective,
            "elapsed_s": self._elapsed_s,
            "filter_time_s": self._filter_time_s,
            "velocity_source": "foot_support_window",
            "filtered_world_velocity": self._velocity,
            "filtered_yaw_rate": self._yaw_rate,
            "support_pose": self._support_pose,
            "support_anchor": self._anchor,
            "position_error_world": self._position_error,
            "heading_error_rad": self._heading_error,
            "correction_world": self._correction_world,
            "linear_gain": self.linear_gain,
            "yaw_gain": self.yaw_gain,
            "linear_damping": self.linear_damping,
            "yaw_damping": self.yaw_damping,
            "braking_s": self.braking_s,
            "window_s": self.window_s,
            "filter_s": self.filter_s,
            "max_linear_speed": self.max_linear_speed,
            "max_yaw_speed": self.max_yaw_speed,
            "position_deadband": self.position_deadband,
            "heading_deadband": self.heading_deadband,
            "linear_deadband": self.linear_deadband,
            "yaw_deadband": self.yaw_deadband,
            "tail_threshold": self.tail_threshold,
        }
        return effective

    @staticmethod
    def _radial_deadband(vector: tuple, deadband: float) -> tuple[float, float]:
        length = math.hypot(*vector)
        scale = max(0.0, length - deadband) / length if length > 0.0 else 0.0
        return tuple(value * scale for value in vector)

    @staticmethod
    def _signed_deadband(value: float, deadband: float) -> float:
        return math.copysign(max(0.0, abs(value) - deadband), value)

    def _sample_support(self, pose: tuple) -> bool:
        """Measure signed motion over an exact window, unwrapping sole yaw."""
        heading = pose[2]
        if self._samples:
            previous = self._samples[-1][3]
            heading = previous + (heading - previous + math.pi) % (2.0 * math.pi) - math.pi
        self._support_pose = (pose[0], pose[1], heading)
        self._samples.append((self._elapsed_s, *self._support_pose))
        cutoff = self._elapsed_s - self.window_s
        while len(self._samples) > 1 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        if len(self._samples) < 2 or self._samples[0][0] > cutoff + 1e-9:
            return False
        first, second = self._samples[0], self._samples[1]
        fraction = max(0.0, (cutoff - first[0]) / (second[0] - first[0]))
        boundary = (cutoff, *(first[i] + fraction * (second[i] - first[i]) for i in (1, 2, 3)))
        self._samples[0] = boundary
        current = self._samples[-1]
        self._velocity = tuple((current[i] - boundary[i]) / self.window_s for i in (1, 2))
        self._yaw_rate = (current[3] - boundary[3]) / self.window_s
        return True

    def update(
        self,
        *,
        dt: float,
        operator_command: object,
        filtered_command: object,
        support_pose: object,
        yaw: float,
        ready: bool = True,
    ) -> tuple[float, float, float]:
        """Return actor input while leaving operator buffers and motion guards intact.

        Support pose is world foot-centroid x/y plus mean sole yaw in radians;
        yaw is the measured root heading used only to rotate the correction.
        Not-ready input returns zero. Invalid required input resets and raises
        ValueError for the caller's fault latch. A new operator movement does not
        depend on support measurements. The filtered command must be ordinary
        operator-filter output, never this method's previous effective output.
        """
        try:
            if type(ready) is not bool:
                raise ValueError("ready must be a boolean")
            if not ready:
                self.reset()
                return self._publish("unavailable")
            dt = _number(dt, "dt")
            if dt <= 0.0:
                raise ValueError("dt must be positive")
            requested = _vector(operator_command, 3, "operator_command")
            filtered = _vector(filtered_command, 3, "filtered_command")
            if any(abs(value) > self.INTENT_EPSILON for value in requested):
                self.reset()
                return self._publish("operator", requested, filtered, effective=filtered)
            self._elapsed_s += dt
            if not math.isfinite(self._elapsed_s):
                raise ValueError("Walking stop feedback clock became nonfinite")
            pose = _vector(support_pose, 3, "support_pose")
            yaw = _number(yaw, "yaw")
            window_ready = self._sample_support(pose)
            if math.hypot(*filtered[:2]) > self.tail_threshold or abs(filtered[2]) > self.tail_threshold:
                self._correction_world = ZERO_COMMAND
                self._filter_time_s = 0.0
                return self._publish("command_tail", requested, filtered, effective=filtered)
            if self._elapsed_s + 1e-9 < self.braking_s:
                return self._publish("braking", requested, filtered, effective=filtered)
            if self._anchor is None:
                self._anchor = self._support_pose
            if not window_ready:
                return self._publish("measuring_support", requested, filtered, effective=filtered)
            self._position_error = tuple(self._anchor[i] - self._support_pose[i] for i in (0, 1))
            self._heading_error = (self._anchor[2] - self._support_pose[2] + math.pi) % (2.0 * math.pi) - math.pi
            position = self._radial_deadband(self._position_error, self.position_deadband)
            velocity = self._radial_deadband(self._velocity, self.linear_deadband)
            target = tuple(self.linear_gain * p - self.linear_damping * v for p, v in zip(position, velocity))
            length = math.hypot(*target)
            if length > self.max_linear_speed:
                target = tuple(value * self.max_linear_speed / length for value in target)
            angular = self.yaw_gain * self._signed_deadband(self._heading_error, self.heading_deadband)
            angular -= self.yaw_damping * self._signed_deadband(self._yaw_rate, self.yaw_deadband)
            target = (*target, max(-self.max_yaw_speed, min(self.max_yaw_speed, angular)))
            alpha = -math.expm1(-dt / self.filter_s)
            self._correction_world = tuple(
                old + alpha * (new - old) for old, new in zip(self._correction_world, target)
            )
            self._filter_time_s += dt
            if not all(math.isfinite(v) for v in (*self._velocity, self._yaw_rate, *self._correction_world)):
                raise ValueError("Walking stop feedback state became nonfinite")
            world_x, world_y, angular = self._correction_world
            cosine, sine = math.cos(yaw), math.sin(yaw)
            correction = (cosine * world_x + sine * world_y, -sine * world_x + cosine * world_y, angular)
            return self._publish("regulating", requested, filtered, correction, correction)
        except ValueError:
            self.reset()
            self._publish("invalid")
            raise
