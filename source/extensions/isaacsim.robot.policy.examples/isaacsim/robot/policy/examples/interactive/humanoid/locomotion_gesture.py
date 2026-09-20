# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interpret a held left-hand V sign as a momentary walking joystick.

The detector returns normalized commands; a separate helper maps them to bounded
physical velocity requests. The caller supplies fresh Quest bare-hand observations
in an operator-relative frame and applies robot policy control and left-arm
reservation. Fingers remain tracked. Webcam estimates, controller poses, head
rotation, and gaze must not authorize this gesture.

Index and middle fingers are extended, ring and little fingers are curled, and
the thumb is ignored. A stable dwell establishes the joystick center. Releasing
the gesture or losing trustworthy observations immediately returns zero; there
is no toggle latch, backward command, or smoothing that can prolong walking.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

__all__ = ["GestureCommand", "LocomotionGesture", "map_gesture_to_velocity"]

GestureState = Literal["idle", "acquiring", "active"]
Position = tuple[float, float, float]
FINGER_ROLES = ("index", "middle", "ring", "little")


@dataclass(frozen=True)
class GestureCommand:
    """Normalized momentary command and gesture ownership for the caller."""

    state: GestureState = "idle"
    #: Current state; acquiring reserves the left arm but does not authorize walking.
    forward: float = 0.0
    #: Forward command in [0, 1]; pulling the wrist back never commands reverse.
    turn: float = 0.0
    #: Signed turn in [-1, 1], positive for a leftward wrist displacement.
    acquisition_progress: float = 0.0
    #: Stable dwell fraction, capped at one after activation.
    neutral_position: Position | None = None
    #: Joystick center in the supplied forward/left/up frame after activation.

    @property
    def owns_hand(self) -> bool:
        """Return whether navigation reserves the left arm; fingers remain tracked."""
        return self.state != "idle"


def _finite_number(value: object) -> float | None:
    """Accept real finite measurements without interpreting strings or booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def map_gesture_to_velocity(
    command: GestureCommand,
    *,
    min_forward_speed: float = 0.08,
    max_forward_speed: float = 0.25,
    max_yaw_speed: float = 0.20,
) -> tuple[float, float, float]:
    """Convert active wrist input into forward, lateral, and yaw velocity requests.

    A positive forward input starts above the balance actor's 0.05 m/s switching
    threshold. Scaling from zero would add an invisible second wrist deadzone:
    the hand would be reserved for navigation while the actor stayed in balance.
    The caller may ramp the resulting request, but releases must remain immediate.

    Args:
        command: Current normalized gesture state, including its explicit deadzone.
        min_forward_speed: Smallest positive forward request in meters per second.
        max_forward_speed: Forward request at full wrist travel in meters per second.
        max_yaw_speed: Maximum signed steering request in radians per second.

    Returns:
        A bounded velocity request. Inactive gestures and zero forward input always
        return three exact zeros; sideways input alone never starts motion.

    Raises:
        ValueError: If speed limits or active normalized inputs are invalid.
    """
    limits = tuple(_finite_number(value) for value in (min_forward_speed, max_forward_speed, max_yaw_speed))
    if any(value is None for value in limits):
        raise ValueError("Gesture speed limits must be finite real numbers")
    minimum, maximum, yaw_limit = limits
    if minimum <= 0.0 or maximum < minimum or yaw_limit < 0.0:
        raise ValueError("Gesture forward speeds must satisfy 0 < minimum <= maximum; yaw must be nonnegative")
    if command.state != "active":
        return 0.0, 0.0, 0.0
    forward, turn = _finite_number(command.forward), _finite_number(command.turn)
    if forward is None or turn is None or not 0.0 <= forward <= 1.0 or not -1.0 <= turn <= 1.0:
        raise ValueError("Active gesture inputs must be finite and normalized")
    if forward == 0.0:
        return 0.0, 0.0, 0.0
    return minimum + (maximum - minimum) * forward, 0.0, yaw_limit * turn


def _read_position(position: Sequence[float] | None) -> Position | None:
    """Require all three finite wrist coordinates; never substitute an origin."""
    try:
        if position is None or len(position) != 3:
            return None
        values = tuple(_finite_number(value) for value in position)
        if any(value is None for value in values):
            return None
        return values
    except TypeError:
        return None


def _read_curls(curls: Mapping[str, float] | None) -> tuple[float, ...] | None:
    """Require four complete normalized digit curls; unknown digits are not open."""
    if not isinstance(curls, Mapping):
        return None
    values = tuple(_finite_number(curls.get(role)) for role in FINGER_ROLES)
    if any(value is None or not 0.0 <= value <= 1.0 for value in values):
        return None
    return values


class LocomotionGesture:
    """Require a stable V-sign dwell before interpreting wrist displacement.

    Args:
        dwell_s: Continuous stationary gesture duration required for activation.
        dwell_radius_m: Maximum distance from the acquisition starting point.
        deadzone_m: Neutral half-width of both joystick axes.
        full_forward_m: Forward displacement from neutral that reaches full input.
        full_turn_m: Sideways displacement from neutral that reaches full input.
        max_gap_s: Longest permitted interval between fresh update observations.

    Raises:
        ValueError: If timing or geometric settings are nonfinite or inconsistent.
    """

    def __init__(
        self,
        *,
        dwell_s: float = 0.7,
        dwell_radius_m: float = 0.04,
        deadzone_m: float = 0.04,
        full_forward_m: float = 0.18,
        full_turn_m: float = 0.15,
        max_gap_s: float = 0.25,
    ) -> None:
        values = tuple(
            _finite_number(value)
            for value in (dwell_s, dwell_radius_m, deadzone_m, full_forward_m, full_turn_m, max_gap_s)
        )
        if any(value is None for value in values):
            raise ValueError("Gesture timing and geometry must be finite real numbers")
        if (
            dwell_s <= 0.0
            or dwell_radius_m <= 0.0
            or deadzone_m < 0.0
            or full_forward_m <= deadzone_m
            or full_turn_m <= deadzone_m
            or max_gap_s <= 0.0
        ):
            raise ValueError("Gesture dwell/radius/gap must be positive and full travel must exceed its deadzone")
        self.dwell_s, self.dwell_radius_m, self.deadzone_m = values[:3]
        self.full_forward_m, self.full_turn_m, self.max_gap_s = values[3:]
        self.reset()

    @property
    def state(self) -> GestureState:
        """Return the latest state without advancing gesture timing."""
        return self._command.state

    @property
    def owns_hand(self) -> bool:
        """Return whether acquisition or walking reserves the left arm, excluding fingers."""
        return self._command.owns_hand

    def reset(self) -> GestureCommand:
        """Stop immediately and discard the old dwell and joystick center."""
        self._command = GestureCommand()
        self._last_time_s: float | None = None
        self._acquisition_start_s: float | None = None
        self._acquisition_position: Position | None = None
        self._neutral_position: Position | None = None
        return self._command

    def update(
        self,
        *,
        curls: Mapping[str, float] | None,
        position: Sequence[float] | None,
        tracking_valid: bool,
        now_s: float,
    ) -> GestureCommand:
        """Consume a fresh left Quest hand sample and return a bounded command.

        Args:
            curls: Index/middle/ring/little curls, zero open and one fully curled.
            position: Wrist (forward, left, up) in meters in a stable operator frame.
            tracking_valid: True only for fresh optical Quest hand tracking, never
                inferred webcam input, controller input, or an old cached sample.
            now_s: Absolute monotonic time in seconds from one clock. Duplicate,
                backwards, nonfinite, or overly separated updates cancel walking.

        Returns:
            The momentary command. Both axes are zero during acquisition, on the
            activation frame, and immediately after every release or invalid input.
        """
        now = _finite_number(now_s)
        point = _read_position(position)
        digits = _read_curls(curls)
        if tracking_valid is not True or now is None or now < 0.0 or point is None or digits is None:
            return self.reset()
        if self._last_time_s is not None:
            interval = now - self._last_time_s
            if interval <= 0.0 or interval > self.max_gap_s:
                return self.reset()
        self._last_time_s = now

        # Entry is deliberately stricter than retention to reject partial grasps
        # without flickering when a held V sign has small curl-estimation jitter.
        extended_limit, curled_limit = (0.25, 0.70) if self.state == "idle" else (0.40, 0.55)
        index, middle, ring, little = digits
        if index > extended_limit or middle > extended_limit or ring < curled_limit or little < curled_limit:
            return self.reset()

        if self.state == "idle":
            self._begin_acquisition(now, point)
            return self._command
        if self.state == "acquiring":
            distance = math.dist(point, self._acquisition_position)
            if distance > self.dwell_radius_m:
                self._begin_acquisition(now, point)
                return self._command
            elapsed = now - self._acquisition_start_s
            if elapsed + 1e-12 < self.dwell_s:
                self._command = GestureCommand(state="acquiring", acquisition_progress=elapsed / self.dwell_s)
                return self._command
            # Center at the current wrist, so permitted dwell jitter cannot cause
            # a forward/turn impulse on the exact activation frame.
            self._neutral_position = point
            self._command = GestureCommand(state="active", acquisition_progress=1.0, neutral_position=point)
            return self._command

        forward = self._axis(point[0] - self._neutral_position[0], self.full_forward_m)
        turn = self._axis(point[1] - self._neutral_position[1], self.full_turn_m)
        self._command = GestureCommand(
            state="active",
            forward=max(0.0, forward),
            turn=turn,
            acquisition_progress=1.0,
            neutral_position=self._neutral_position,
        )
        return self._command

    def _begin_acquisition(self, now_s: float, position: Position) -> None:
        """Reserve the hand and restart a stationary dwell at its current position."""
        self._acquisition_start_s = now_s
        self._acquisition_position = position
        self._neutral_position = None
        self._command = GestureCommand(state="acquiring")

    def _axis(self, displacement: float, full_travel: float) -> float:
        """Map displacement through a continuous deadzone and saturate at one."""
        magnitude = abs(displacement)
        if magnitude <= self.deadzone_m:
            return 0.0
        amount = min(1.0, (magnitude - self.deadzone_m) / (full_travel - self.deadzone_m))
        return math.copysign(amount, displacement)
