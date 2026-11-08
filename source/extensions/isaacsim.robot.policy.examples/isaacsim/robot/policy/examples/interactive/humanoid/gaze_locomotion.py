# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select bounded base movements with gaze dwell, independent of hand controls.

The caller resolves eye-ray hits, supplies measured robot motion, and applies
the returned body-frame velocity through its existing locomotion safeguards.
This module owns no XR, scene, physics, arm, finger, or grip input. Gaze selects
an action; looking away or losing eye tracking does not cancel an active action.
Readiness loss does cancel it. A valid look outside every button is required
after startup, reset, and each completed action before another dwell can begin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

__all__ = ["GazeLocomotion", "GazeLocomotionCommand", "GazeLocomotionConfig"]

ACTIONS = ("forward", "back", "turn_left", "turn_right")
ZERO_VELOCITY = (0.0, 0.0, 0.0)


def _finite(value: object) -> bool:
    """Accept finite real measurements while excluding boolean flags."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class GazeLocomotionConfig:
    """Configure a short gaze-selected step or turn, with preview enabled by default.

    Args:
        dwell_s: Continuous wall-time fixation required, from 0.1 to 5 seconds.
        distance_m: Requested projected travel, from 0.01 to 1 meter.
        turn_degrees: Requested signed yaw change, from 1 to 90 degrees.
        forward_speed: Positive body-forward speed, at most 1 meter per second.
        backward_speed: Reverse speed magnitude, at most 1 meter per second; zero disables reverse.
        yaw_speed: Positive yaw-speed magnitude, at most 1.5 radians per second.
        preview: Emit selection events and status while keeping every velocity zero.
        max_gaze_gap_s: Largest permitted wall-time gap within a dwell, from 0.05 to 1 second.

    Raises:
        ValueError: A setting is nonfinite, has the wrong type, or is outside its range.
    """

    dwell_s: float = 0.7
    distance_m: float = 0.20
    turn_degrees: float = 10.0
    forward_speed: float = 0.20
    backward_speed: float = 0.15
    yaw_speed: float = 0.25
    preview: bool = True
    max_gaze_gap_s: float = 0.25

    def __post_init__(self) -> None:
        """Reject invalid configuration before any input can produce a command."""
        for name, lower, upper in (
            ("dwell_s", 0.1, 5.0),
            ("distance_m", 0.01, 1.0),
            ("turn_degrees", 1.0, 90.0),
            ("forward_speed", 0.01, 1.0),
            ("backward_speed", 0.0, 1.0),
            ("yaw_speed", 0.01, 1.5),
            ("max_gaze_gap_s", 0.05, 1.0),
        ):
            value = getattr(self, name)
            if not _finite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
        if type(self.preview) is not bool:
            raise ValueError("preview must be a boolean")


@dataclass(frozen=True)
class GazeLocomotionCommand:
    """Return a status snapshot and body-forward/body-left/yaw-left velocity.

    ``activated_action`` is populated on the activation update only, including
    preview. ``last_action`` and ``activation_count`` persist until reset.
    ``progress`` is dwell fraction during selection and measured goal fraction
    during movement. All speeds use meters per second or radians per second.
    """

    state: str = "awaiting_release"
    target: str | None = None
    progress: float = 0.0
    last_action: str | None = None
    activation_count: int = 0
    velocity: tuple[float, float, float] = ZERO_VELOCITY
    detail: str = "Look outside all gaze buttons to enable selection"
    activated_action: str | None = None


class GazeLocomotion:
    """Dwell once, finish a measured movement, settle, then require look-away.

    Dwell and sample-gap checks use monotonic wall time. Travel and stopping
    timeouts use simulation ``dt`` so slower rendering does not shorten a move.
    A movement timeout requests zero and enters settling even if the robot has
    stalled. Failing to settle within 4 simulation seconds latches a fault
    until ``reset``. Preview does not require a robot pose and never commands motion.

    Args:
        config: Validated selection/movement settings, or None for preview defaults.
    """

    SETTLE_LINEAR_SPEED_M_S = 0.03
    SETTLE_YAW_SPEED_RAD_S = 0.06
    SETTLE_DWELL_S = 0.20
    # The physical actor needs time to damp post-step balance sway. This only
    # extends the zero-command waiting window; the stop-speed criteria and
    # the sample's independent runaway/balance guards remain unchanged.
    SETTLE_TIMEOUT_S = 4.0

    def __init__(self, config: GazeLocomotionConfig | None = None) -> None:
        if config is not None and not isinstance(config, GazeLocomotionConfig):
            raise TypeError("config must be a GazeLocomotionConfig")
        self.config = config if config is not None else GazeLocomotionConfig()
        self.reset()

    def reset(self) -> None:
        """Cancel motion, clear faults and counts, and require a new valid look-away."""
        self._released = False
        self._last_time_s: float | None = None
        self._candidate: str | None = None
        self._dwell_start_s: float | None = None
        self._action: str | None = None
        self._start_pose: tuple[float, float, float] | None = None
        self._previous_yaw = 0.0
        self._yaw_travel = 0.0
        self._move_elapsed_s = 0.0
        self._stop_elapsed_s = 0.0
        self._settled_s = 0.0
        self._move_timeout_s = 0.0
        self._motion_progress = 0.0
        self._stop_detail = ""
        self._fault: str | None = None
        self._last_action: str | None = None
        self._activation_count = 0
        self.command = GazeLocomotionCommand()

    @staticmethod
    def _pose(value: object) -> tuple[float, float, float] | None:
        """Accept a finite planar world pose with yaw in radians."""
        try:
            if len(value) == 3 and all(_finite(component) for component in value):
                return tuple(float(component) for component in value)
        except (TypeError, ValueError):
            pass
        return None

    def _publish(
        self,
        state: str,
        detail: str,
        *,
        target: str | None = None,
        progress: float = 0.0,
        velocity: tuple[float, float, float] = ZERO_VELOCITY,
        activated_action: str | None = None,
    ) -> GazeLocomotionCommand:
        """Publish immutable status, enforcing the preview zero-velocity invariant."""
        self.command = GazeLocomotionCommand(
            state=state,
            target=target,
            progress=min(1.0, max(0.0, progress)),
            last_action=self._last_action,
            activation_count=self._activation_count,
            velocity=ZERO_VELOCITY if self.config.preview else velocity,
            detail=detail,
            activated_action=activated_action,
        )
        return self.command

    def _cancel(self, detail: str, *, fault: bool = False) -> GazeLocomotionCommand:
        """Drop pending and active actions without treating cancellation as look-away."""
        self._candidate = self._dwell_start_s = self._action = None
        self._released = False
        if fault:
            self._fault = detail
        return self._publish("fault" if fault else "unavailable", detail)

    def _velocity(self, action: str) -> tuple[float, float, float]:
        """Map one selected action to its configured body-frame speed."""
        return {
            "forward": (self.config.forward_speed, 0.0, 0.0),
            "back": (-self.config.backward_speed, 0.0, 0.0),
            "turn_left": (0.0, 0.0, self.config.yaw_speed),
            "turn_right": (0.0, 0.0, -self.config.yaw_speed),
        }[action]

    def _description(self, action: str) -> str:
        """Describe the bounded action for preview and movement status."""
        if action in ("forward", "back"):
            direction = "forward" if action == "forward" else "backward"
            return f"move {direction} {self.config.distance_m:.2f} m"
        direction = "left" if action == "turn_left" else "right"
        return f"turn {direction} {self.config.turn_degrees:g} degrees"

    def _activate(self, action: str, pose: tuple[float, float, float] | None) -> GazeLocomotionCommand:
        """Latch one selection and establish its measured-motion origin."""
        self._released = False
        self._candidate = self._dwell_start_s = None
        self._last_action = action
        self._activation_count += 1
        if self.config.preview:
            return self._publish(
                "preview",
                f"Would {self._description(action)}; look away to select again",
                target=action,
                progress=1.0,
                activated_action=action,
            )
        self._action = action
        self._start_pose = pose
        self._previous_yaw = pose[2]
        self._yaw_travel = self._move_elapsed_s = self._motion_progress = 0.0
        velocity = self._velocity(action)
        nominal_time = (
            self.config.distance_m / abs(velocity[0])
            if action in ("forward", "back")
            else math.radians(self.config.turn_degrees) / abs(velocity[2])
        )
        self._move_timeout_s = min(10.0, max(1.5, 3.0 * nominal_time + 1.0))
        return self._publish(
            "active",
            f"Requested {self._description(action)}",
            target=action,
            velocity=velocity,
            activated_action=action,
        )

    def _advance_motion(
        self, dt: float, pose: tuple[float, float, float], linear_speed: float, yaw_speed: float
    ) -> GazeLocomotionCommand:
        """Measure travel and stop/settle without consuming any new gaze selection."""
        action = self._action
        if self.command.state == "stopping":
            self._stop_elapsed_s += dt
            if abs(linear_speed) <= self.SETTLE_LINEAR_SPEED_M_S and abs(yaw_speed) <= self.SETTLE_YAW_SPEED_RAD_S:
                self._settled_s += dt
            else:
                self._settled_s = 0.0
            if self._settled_s + 1e-12 >= self.SETTLE_DWELL_S:
                self._action = None
                return self._publish("awaiting_release", self._stop_detail + "; look outside all buttons to rearm")
            if self._stop_elapsed_s + 1e-12 >= self.SETTLE_TIMEOUT_S:
                return self._cancel("Robot did not settle after zero command; reset required", fault=True)
            return self._publish(
                "stopping",
                self._stop_detail + "; waiting for measured stop",
                target=action,
                progress=self._motion_progress,
            )

        self._move_elapsed_s += dt
        if action in ("forward", "back"):
            start_x, start_y, start_yaw = self._start_pose
            projected = (pose[0] - start_x) * math.cos(start_yaw) + (pose[1] - start_y) * math.sin(start_yaw)
            travel = projected if action == "forward" else -projected
            goal = self.config.distance_m
        else:
            increment = (pose[2] - self._previous_yaw + math.pi) % (2.0 * math.pi) - math.pi
            self._previous_yaw = pose[2]
            self._yaw_travel += increment
            travel = self._yaw_travel if action == "turn_left" else -self._yaw_travel
            goal = math.radians(self.config.turn_degrees)
        self._motion_progress = min(1.0, max(0.0, travel / goal))
        if travel + 1e-12 >= goal or self._move_elapsed_s + 1e-12 >= self._move_timeout_s:
            self._stop_elapsed_s = self._settled_s = 0.0
            self._stop_detail = (
                "Movement target reached" if travel + 1e-12 >= goal else "Movement timed out before target"
            )
            return self._publish(
                "stopping",
                self._stop_detail + "; requesting zero velocity",
                target=action,
                progress=self._motion_progress,
            )
        return self._publish(
            "active",
            f"Requested {self._description(action)}",
            target=action,
            progress=self._motion_progress,
            velocity=self._velocity(action),
        )

    def update(
        self,
        *,
        target: str | None,
        eye_valid: bool,
        now_s: float,
        dt: float,
        pose: tuple[float, float, float] | None,
        linear_speed_m_s: float,
        yaw_speed_rad_s: float,
        ready: bool = True,
    ) -> GazeLocomotionCommand:
        """Consume current selection and measured motion without any hand dependency.

        ``target=None`` with valid eye tracking means a verified look outside all
        buttons. Unknown target IDs cannot rearm. First fixation samples contribute
        zero dwell; equal timestamps are allowed but add no time. An invalid eye
        sample, target change, or excessive callback gap clears unfinished dwell.
        Readiness loss cancels motion immediately; the caller remains responsible
        for its physical stop/fault guards. Invalid motion measurements while
        moving latch a fault rather than continuing open loop.
        """
        if self._fault is not None:
            return self._publish("fault", self._fault)
        moving = self._action is not None
        if not _finite(now_s) or now_s < 0 or not _finite(dt) or dt <= 0:
            self._last_time_s = None
            return self._cancel("Invalid gaze update clock or simulation timestep", fault=moving)
        previous = self._last_time_s
        self._last_time_s = float(now_s)
        if ready is not True:
            return self._cancel("Gaze locomotion is not ready; look away after it becomes ready")
        measured_pose = self._pose(pose)
        if not self.config.preview and (
            measured_pose is None or not _finite(linear_speed_m_s) or not _finite(yaw_speed_rad_s)
        ):
            return self._cancel("Valid measured robot pose and speeds are required", fault=moving)
        if moving:
            return self._advance_motion(float(dt), measured_pose, float(linear_speed_m_s), float(yaw_speed_rad_s))
        if previous is not None and (now_s < previous or now_s - previous > self.config.max_gaze_gap_s):
            self._candidate = self._dwell_start_s = None
            return self._publish(
                "idle" if self._released else "awaiting_release", "Gaze update interrupted; dwell cleared"
            )
        if eye_valid is not True:
            self._candidate = self._dwell_start_s = None
            return self._publish("unavailable", "Valid eye tracking is required for selection")
        if target is not None and (not isinstance(target, str) or target not in ACTIONS):
            self._candidate = self._dwell_start_s = None
            return self._publish("unavailable", "Unknown gaze button; look outside the buttons to rearm")
        if target is None:
            self._candidate = self._dwell_start_s = None
            self._released = True
            return self._publish("idle", "Look at one gaze button to begin dwell")
        if not self._released:
            if self.config.preview and self._last_action is not None:
                return self._publish(
                    "preview",
                    f"Would {self._description(self._last_action)}; look away to select again",
                    target=self._last_action,
                    progress=1.0,
                )
            return self._publish("awaiting_release", "Look outside all gaze buttons before another selection")
        if target == "back" and self.config.backward_speed == 0:
            self._candidate = self._dwell_start_s = None
            return self._publish("unavailable", "Backward movement is disabled", target=target)
        if target != self._candidate:
            self._candidate = target
            self._dwell_start_s = float(now_s)
        progress = (now_s - self._dwell_start_s) / self.config.dwell_s
        if progress + 1e-12 >= 1.0:
            return self._activate(target, measured_pose)
        return self._publish(
            "dwelling", f"Keep looking to {self._description(target)}", target=target, progress=progress
        )
