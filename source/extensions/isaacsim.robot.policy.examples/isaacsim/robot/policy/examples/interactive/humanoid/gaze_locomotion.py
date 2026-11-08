# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select stepped or gaze-held base movement independently of hand controls.

The caller resolves eye-ray hits, supplies measured robot motion, and applies
the returned body-frame velocity through its existing locomotion safeguards.
This module owns no XR, scene, physics, arm, finger, or grip input. Step mode
finishes its selected action independently of gaze; hold mode moves only while
the same button has valid, uninterrupted gaze. A valid look outside every button
is required after startup/reset and between actions. In hold mode the look-away
that requests zero also permits another selection; physical motion supervision
belongs to the caller's support-aware stop and runaway guards.
"""

from __future__ import annotations

import math
from collections import deque
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
    """Configure gaze-selected steps or held movement, with preview enabled by default.

    Args:
        dwell_s: Continuous wall-time fixation required, from 0 to 5 seconds; zero selects immediately.
        distance_m: Requested projected travel, from 0.01 to 1 meter.
        turn_degrees: Requested signed yaw change, from 1 to 90 degrees.
        forward_speed: Positive body-forward speed, at most 1 meter per second.
        backward_speed: Reverse speed magnitude, at most 1 meter per second; zero disables reverse.
        yaw_speed: Positive yaw-speed magnitude, at most 1.5 radians per second.
        preview: Emit selection events and status while keeping every velocity zero.
        max_gaze_gap_s: Largest permitted wall-time gap within a dwell, from 0.05 to 1 second.
        control_mode: ``step`` finishes a measured distance/angle; ``hold`` requires continued fixation.
        heading_hold: Correct measured heading drift during live forward/back translation.
        heading_gain: Proportional heading-error gain, from 0 to 5 inverse seconds.
        heading_max_yaw_speed: Correction cap, from 0 to 0.4 radians per second.

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
    control_mode: str = "step"
    heading_hold: bool = True
    heading_gain: float = 3.0
    heading_max_yaw_speed: float = 0.15

    def __post_init__(self) -> None:
        """Reject invalid configuration before any input can produce a command."""
        for name, lower, upper in (
            ("dwell_s", 0.0, 5.0),
            ("distance_m", 0.01, 1.0),
            ("turn_degrees", 1.0, 90.0),
            ("forward_speed", 0.01, 1.0),
            ("backward_speed", 0.0, 1.0),
            ("yaw_speed", 0.01, 1.5),
            ("max_gaze_gap_s", 0.05, 1.0),
            ("heading_gain", 0.0, 5.0),
            ("heading_max_yaw_speed", 0.0, 0.4),
        ):
            value = getattr(self, name)
            if not _finite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
        if type(self.preview) is not bool:
            raise ValueError("preview must be a boolean")
        if type(self.heading_hold) is not bool:
            raise ValueError("heading_hold must be a boolean")
        if not isinstance(self.control_mode, str) or self.control_mode not in ("step", "hold"):
            raise ValueError("control_mode must be 'step' or 'hold'")


@dataclass(frozen=True)
class GazeLocomotionCommand:
    """Return a status snapshot and body-forward/body-left/yaw-left velocity.

    ``activated_action`` is populated on the activation update only, including
    preview. ``last_action`` and ``activation_count`` persist until reset.
    ``progress`` is dwell fraction during selection and measured goal fraction
    during stepped movement, and one during held movement. All speeds use meters
    per second or radians per second. ``stop_immediately`` forbids the caller's
    normal release smoothing after tracking/readiness failure or a fault. Otherwise
    the caller applies its existing bounded start/release smoothing once.
    """

    state: str = "awaiting_release"
    target: str | None = None
    progress: float = 0.0
    last_action: str | None = None
    activation_count: int = 0
    velocity: tuple[float, float, float] = ZERO_VELOCITY
    detail: str = "Look outside all gaze buttons to enable selection"
    activated_action: str | None = None
    stop_immediately: bool = False


class GazeLocomotion:
    """Select movement and require verified look-away between actions.

    Dwell and sample-gap checks use monotonic wall time. Travel and stopping
    timeouts use simulation ``dt`` so slower rendering does not shorten a move.
    A movement timeout requests zero and enters settling even if the robot has
    stalled. Step settling accepts sustained low speeds or a bounded pose window
    with low net drift; failure within 4 simulation seconds latches a fault.
    Hold mode has no travel/time goal, requests zero when fixation ends, and does
    not equate a zero request with measured physical rest. Its caller must keep
    physical stop/runaway supervision active during and after release. Preview
    does not require a robot pose and never commands motion.

    Args:
        config: Validated selection/movement settings, or None for preview defaults.
    """

    SETTLE_LINEAR_SPEED_M_S = 0.03
    SETTLE_YAW_SPEED_RAD_S = 0.06
    SETTLE_DWELL_S = 0.20
    SETTLE_TIMEOUT_S = 4.0
    SETTLE_WINDOW_S = 1.0
    SETTLE_POSITION_ENVELOPE_M = 0.04
    SETTLE_YAW_ENVELOPE_RAD = 0.08
    SETTLE_DRIFT_M_S = 0.02
    SETTLE_YAW_DRIFT_RAD_S = 0.04
    SETTLE_PEAK_LINEAR_SPEED_M_S = 0.15
    SETTLE_PEAK_YAW_SPEED_RAD_S = 0.30
    HEADING_DEADBAND_RAD = math.radians(0.5)
    HEADING_FILTER_S = 0.15

    def __init__(self, config: GazeLocomotionConfig | None = None) -> None:
        if config is not None and not isinstance(config, GazeLocomotionConfig):
            raise TypeError("config must be a GazeLocomotionConfig")
        self.config = config if config is not None else GazeLocomotionConfig()
        self.reset()

    def reset(self) -> None:
        """Cancel motion and require look-away, retaining the last stop's diagnostic snapshot."""
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
        self._sway_settled_s = 0.0
        self._stop_window = deque()
        self._stop_previous_yaw = self._stop_yaw = 0.0
        self._stop_immediately = False
        self._move_timeout_s = 0.0
        self._motion_progress = 0.0
        self._stop_detail = ""
        self._fault: str | None = None
        self._last_action: str | None = None
        self._activation_count = 0
        self._stop_diagnostics = getattr(self, "_stop_diagnostics", {})
        self._clear_heading()
        self.command = GazeLocomotionCommand()

    @property
    def heading_status(self) -> dict:
        """Return current translation feedback; this never changes a root pose."""
        return dict(self._heading_status)

    def _clear_heading(self) -> None:
        """Discard the filter and correction when no translation owns heading."""
        self._filtered_heading_error = 0.0
        self._heading_status = {
            "enabled": self.config.heading_hold,
            "active": False,
            "target_yaw_rad": None,
            "error_rad": 0.0,
            "filtered_error_rad": 0.0,
            "correction_rad_s": 0.0,
        }

    @property
    def stop_diagnostics(self) -> dict:
        """Return the last stop's evidence, preserved through fault/reset until the next stop.

        Hold entries describe a zero request, not a finding of physical rest.
        Step entries include the most recent settling-window gates and decision.
        Values are scalars or immutable tuples; callers receive a defensive copy.
        """
        return dict(self._stop_diagnostics)

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
        stop_immediately: bool = False,
    ) -> GazeLocomotionCommand:
        """Publish immutable status, enforcing the preview zero-velocity invariant."""
        if state != "active":
            self._clear_heading()
        self.command = GazeLocomotionCommand(
            state=state,
            target=target,
            progress=min(1.0, max(0.0, progress)),
            last_action=self._last_action,
            activation_count=self._activation_count,
            velocity=ZERO_VELOCITY if self.config.preview else velocity,
            detail=detail,
            activated_action=activated_action,
            stop_immediately=stop_immediately,
        )
        return self.command

    def _cancel(self, detail: str, *, fault: bool = False) -> GazeLocomotionCommand:
        """Drop pending and active actions without treating cancellation as look-away."""
        self._candidate = self._dwell_start_s = self._action = None
        self._released = False
        self._stop_window.clear()
        if fault:
            self._fault = detail
        return self._publish("fault" if fault else "unavailable", detail, stop_immediately=True)

    def _velocity(self, action: str) -> tuple[float, float, float]:
        """Map one selected action to its configured body-frame speed."""
        return {
            "forward": (self.config.forward_speed, 0.0, 0.0),
            "back": (-self.config.backward_speed, 0.0, 0.0),
            "turn_left": (0.0, 0.0, self.config.yaw_speed),
            "turn_right": (0.0, 0.0, -self.config.yaw_speed),
        }[action]

    def _active_velocity(self, action: str, pose: tuple[float, float, float], dt: float) -> tuple[float, float, float]:
        """Steer translation toward its starting yaw through bounded actor input.

        Filter wrapped error in simulation time to suppress gait oscillation.
        The small continuous deadband avoids correcting normal heading noise.
        Reverse keeps the same yaw-error sign; explicit turns own their own yaw.
        """
        velocity = self._velocity(action)
        if not self.config.heading_hold or action not in ("forward", "back") or self.config.preview:
            self._clear_heading()
            return velocity
        target_yaw = self._start_pose[2]
        error = (target_yaw - pose[2] + math.pi) % (2.0 * math.pi) - math.pi
        increment = (error - self._filtered_heading_error + math.pi) % (2.0 * math.pi) - math.pi
        alpha = -math.expm1(-dt / self.HEADING_FILTER_S)
        self._filtered_heading_error = (self._filtered_heading_error + alpha * increment + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        outside_deadband = math.copysign(
            max(0.0, abs(self._filtered_heading_error) - self.HEADING_DEADBAND_RAD), self._filtered_heading_error
        )
        cap = self.config.heading_max_yaw_speed
        correction = max(-cap, min(cap, self.config.heading_gain * outside_deadband))
        self._heading_status = {
            "enabled": True,
            "active": True,
            "target_yaw_rad": target_yaw,
            "error_rad": error,
            "filtered_error_rad": self._filtered_heading_error,
            "correction_rad_s": correction,
        }
        return velocity[0], velocity[1], correction

    def _description(self, action: str) -> str:
        """Describe the selected action and its mode-specific stopping rule."""
        if action in ("forward", "back"):
            direction = "forward" if action == "forward" else "backward"
            if self.config.control_mode == "hold":
                return f"move {direction} while looking at this button"
            return f"move {direction} {self.config.distance_m:.2f} m"
        direction = "left" if action == "turn_left" else "right"
        if self.config.control_mode == "hold":
            return f"turn {direction} while looking at this button"
        return f"turn {direction} {self.config.turn_degrees:g} degrees"

    def _activate(self, action: str, pose: tuple[float, float, float] | None) -> GazeLocomotionCommand:
        """Latch one selection and establish its measured-motion origin."""
        self._released = False
        self._candidate = self._dwell_start_s = None
        self._last_action = action
        self._activation_count += 1
        self._clear_heading()
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
        if self.config.control_mode == "hold":
            self._motion_progress = 1.0
        velocity = self._active_velocity(action, pose, 0.0)
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
            progress=1.0 if self.config.control_mode == "hold" else 0.0,
            velocity=velocity,
            activated_action=action,
        )

    def _begin_stop(
        self,
        detail: str,
        pose: tuple[float, float, float],
        linear_speed: float,
        yaw_speed: float,
        *,
        immediate: bool = False,
    ) -> GazeLocomotionCommand:
        """Request zero once and collect fresh settling evidence after that request."""
        self._stop_elapsed_s = self._settled_s = self._sway_settled_s = 0.0
        self._stop_detail = detail
        self._stop_immediately = immediate
        self._stop_previous_yaw = self._stop_yaw = pose[2]
        self._stop_window.clear()
        self._stop_window.append((0.0, *pose, abs(linear_speed), abs(yaw_speed)))
        self._stop_diagnostics = {
            "mode": self.config.control_mode,
            "action": self._action,
            "reason": detail,
            "decision": "measuring_step_stop",
            "pose": pose,
            "linear_speed_m_s": abs(linear_speed),
            "yaw_speed_rad_s": abs(yaw_speed),
            "stop_immediately": immediate,
            "elapsed_s": 0.0,
        }
        return self._publish(
            "stopping",
            detail + "; requesting zero velocity",
            target=self._action,
            progress=self._motion_progress,
            stop_immediately=immediate,
        )

    def _release_hold(
        self,
        detail: str,
        pose: tuple[float, float, float],
        linear_speed: float,
        yaw_speed: float,
        *,
        immediate: bool,
        verified_outside: bool,
    ) -> GazeLocomotionCommand:
        """Request zero without imposing a second pelvis-only physical stop guard.

        Verified outside input can release and rearm on this same update. Other
        buttons and invalid/occluded input cannot queue a new action. The caller
        still applies normal release smoothing or immediate zero as requested,
        and independently monitors measured foot support, drift and runaway.
        """
        self._stop_diagnostics = {
            "mode": "hold",
            "action": self._action,
            "reason": detail,
            "decision": "zero_requested_external_motion_guard_active",
            "pose": pose,
            "linear_speed_m_s": abs(linear_speed),
            "yaw_speed_rad_s": abs(yaw_speed),
            "stop_immediately": immediate,
            "verified_outside": verified_outside,
            "settling_evaluated": False,
        }
        self._candidate = self._dwell_start_s = self._action = None
        self._released = verified_outside
        self._stop_window.clear()
        state = "idle" if verified_outside else "unavailable" if immediate else "awaiting_release"
        next_selection = "ready for another selection" if verified_outside else "look outside all buttons to rearm"
        return self._publish(state, detail + "; zero requested; " + next_selection, stop_immediately=immediate)

    def _bounded_sway(self, pose: tuple[float, float, float], linear_speed: float, yaw_speed: float) -> bool:
        """Distinguish small standing sway from continuing travel over a full pose window.

        Both endpoint and fitted drift must be small; the envelope rejects large
        out-and-back motion and the speed peaks reject violent motion with low net
        displacement. Wrapped incremental yaw keeps the window valid across pi.
        This supplements, rather than replaces, the caller's balance/health guards.
        """
        self._stop_yaw += (pose[2] - self._stop_previous_yaw + math.pi) % (2.0 * math.pi) - math.pi
        self._stop_previous_yaw = pose[2]
        self._stop_window.append(
            (self._stop_elapsed_s, pose[0], pose[1], self._stop_yaw, abs(linear_speed), abs(yaw_speed))
        )
        cutoff = self._stop_elapsed_s - self.SETTLE_WINDOW_S
        while len(self._stop_window) > 2 and self._stop_window[1][0] <= cutoff:
            self._stop_window.popleft()
        window = self._stop_window
        span = window[-1][0] - window[0][0]
        self._stop_diagnostics.update(
            elapsed_s=self._stop_elapsed_s,
            pose=pose,
            linear_speed_m_s=abs(linear_speed),
            yaw_speed_rad_s=abs(yaw_speed),
            window_span_s=span,
            window_samples=len(window),
        )
        if len(window) < 5 or span + 1e-12 < self.SETTLE_WINDOW_S:
            self._stop_diagnostics["failed_checks"] = ("window_not_ready",)
            return False
        coordinates = tuple(tuple(row[index] for row in window) for index in (1, 2, 3))
        ranges = tuple(max(values) - min(values) for values in coordinates)
        endpoint = tuple((values[-1] - values[0]) / span for values in coordinates)
        mean_time = sum(row[0] for row in window) / len(window)
        centered_time = tuple(row[0] - mean_time for row in window)
        denominator = sum(value * value for value in centered_time)
        fitted = tuple(
            sum(t * (value - values[0]) for t, value in zip(centered_time, values)) / denominator
            for values in coordinates
        )
        checks = (
            ("peak_linear_speed_m_s", max(row[4] for row in window), self.SETTLE_PEAK_LINEAR_SPEED_M_S),
            ("peak_yaw_speed_rad_s", max(row[5] for row in window), self.SETTLE_PEAK_YAW_SPEED_RAD_S),
            ("position_envelope_m", math.hypot(*ranges[:2]), self.SETTLE_POSITION_ENVELOPE_M),
            ("yaw_envelope_rad", ranges[2], self.SETTLE_YAW_ENVELOPE_RAD),
            ("endpoint_drift_m_s", math.hypot(*endpoint[:2]), self.SETTLE_DRIFT_M_S),
            ("endpoint_yaw_drift_rad_s", abs(endpoint[2]), self.SETTLE_YAW_DRIFT_RAD_S),
            ("fitted_drift_m_s", math.hypot(*fitted[:2]), self.SETTLE_DRIFT_M_S),
            ("fitted_yaw_drift_rad_s", abs(fitted[2]), self.SETTLE_YAW_DRIFT_RAD_S),
        )
        self._stop_diagnostics.update((name, value) for name, value, _ in checks)
        failed = tuple(name for name, value, limit in checks if value > limit)
        self._stop_diagnostics["failed_checks"] = failed
        return not failed

    def _advance_stop(
        self, dt: float, pose: tuple[float, float, float], linear_speed: float, yaw_speed: float
    ) -> GazeLocomotionCommand:
        """Keep requesting zero until low motion is measured or the stop deadline faults."""
        self._stop_elapsed_s += dt
        if abs(linear_speed) <= self.SETTLE_LINEAR_SPEED_M_S and abs(yaw_speed) <= self.SETTLE_YAW_SPEED_RAD_S:
            self._settled_s += dt
        else:
            self._settled_s = 0.0
        self._sway_settled_s = self._sway_settled_s + dt if self._bounded_sway(pose, linear_speed, yaw_speed) else 0.0
        self._stop_diagnostics.update(low_speed_confirm_s=self._settled_s, sway_confirm_s=self._sway_settled_s)
        if max(self._settled_s, self._sway_settled_s) + 1e-12 >= self.SETTLE_DWELL_S:
            self._stop_diagnostics["decision"] = "measured_step_settled"
            self._action = None
            return self._publish(
                "awaiting_release",
                self._stop_detail + "; look outside all buttons to rearm",
                stop_immediately=self._stop_immediately,
            )
        if self._stop_elapsed_s + 1e-12 >= self.SETTLE_TIMEOUT_S:
            self._stop_diagnostics["decision"] = "step_settle_timeout_fault"
            return self._cancel("Robot did not settle after zero command; reset required", fault=True)
        return self._publish(
            "stopping",
            self._stop_detail + "; waiting for measured stop",
            target=self._action,
            progress=self._motion_progress,
            stop_immediately=self._stop_immediately,
        )

    def _advance_motion(
        self, dt: float, pose: tuple[float, float, float], linear_speed: float, yaw_speed: float
    ) -> GazeLocomotionCommand:
        """Measure travel and stop/settle without consuming any new gaze selection."""
        action = self._action
        if self.command.state == "stopping":
            return self._advance_stop(dt, pose, linear_speed, yaw_speed)
        if self.config.control_mode == "hold":
            return self._publish(
                "active",
                f"Requested {self._description(action)}",
                target=action,
                progress=1.0,
                velocity=self._active_velocity(action, pose, dt),
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
            return self._begin_stop(
                "Movement target reached" if travel + 1e-12 >= goal else "Movement timed out before target",
                pose,
                linear_speed,
                yaw_speed,
            )
        return self._publish(
            "active",
            f"Requested {self._description(action)}",
            target=action,
            progress=self._motion_progress,
            velocity=self._active_velocity(action, pose, dt),
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
        In hold mode a changed button or verified look-away requests a normally
        smoothed stop; eye loss, unknown/occluded hits, interrupted callbacks and
        readiness loss demand immediate zero. A verified outside sample can end
        hold and rearm selection on the same update; hold imposes no physical
        settling delay. Step mode continues independently of gaze, then settles
        without consuming a look-away for the next action. The caller remains
        responsible for physical stop/fault guards and velocity smoothing.
        Invalid motion measurements while moving latch a fault.
        """
        if self._fault is not None:
            return self._publish("fault", self._fault, stop_immediately=True)
        moving = self._action is not None
        if not _finite(now_s) or now_s < 0 or not _finite(dt) or dt <= 0:
            self._last_time_s = None
            return self._cancel("Invalid gaze update clock or simulation timestep", fault=moving)
        previous = self._last_time_s
        self._last_time_s = float(now_s)
        hold_moving = moving and self.config.control_mode == "hold"
        if ready is not True and not hold_moving:
            return self._cancel("Gaze locomotion is not ready; look away after it becomes ready")
        measured_pose = self._pose(pose)
        if not self.config.preview and (
            measured_pose is None or not _finite(linear_speed_m_s) or not _finite(yaw_speed_rad_s)
        ):
            return self._cancel("Valid measured robot pose and speeds are required", fault=moving)
        interrupted = previous is not None and (now_s < previous or now_s - previous > self.config.max_gaze_gap_s)
        if hold_moving:
            unknown_target = target is not None and (not isinstance(target, str) or target not in ACTIONS)
            immediate = ready is not True or eye_valid is not True or interrupted or unknown_target
            if immediate or target != self._action:
                if ready is not True:
                    detail = "Gaze locomotion readiness lost"
                elif interrupted:
                    detail = "Gaze update interrupted"
                elif eye_valid is not True:
                    detail = "Eye tracking lost"
                elif unknown_target:
                    detail = "Gaze button unavailable or occluded"
                else:
                    detail = "Stopped looking at the selected button"
                return self._release_hold(
                    detail,
                    measured_pose,
                    float(linear_speed_m_s),
                    float(yaw_speed_rad_s),
                    immediate=immediate,
                    verified_outside=not immediate and target is None,
                )
        if moving:
            return self._advance_motion(float(dt), measured_pose, float(linear_speed_m_s), float(yaw_speed_rad_s))
        if interrupted:
            self._candidate = self._dwell_start_s = None
            if self.config.control_mode == "hold":
                self._released = False
            return self._publish(
                "idle" if self._released else "awaiting_release",
                "Gaze update interrupted; dwell cleared",
                stop_immediately=self.config.control_mode == "hold",
            )
        if eye_valid is not True:
            self._candidate = self._dwell_start_s = None
            if self.config.control_mode == "hold":
                self._released = False
            return self._publish(
                "unavailable",
                "Valid eye tracking is required for selection",
                stop_immediately=self.config.control_mode == "hold",
            )
        if target is not None and (not isinstance(target, str) or target not in ACTIONS):
            self._candidate = self._dwell_start_s = None
            if self.config.control_mode == "hold":
                self._released = False
            return self._publish(
                "unavailable",
                "Unknown gaze button; look outside the buttons to rearm",
                stop_immediately=self.config.control_mode == "hold",
            )
        if target is None:
            self._candidate = self._dwell_start_s = None
            self._released = True
            return self._publish(
                "idle",
                (
                    "Look at one gaze button to move"
                    if self.config.dwell_s == 0
                    else "Look at one gaze button to begin dwell"
                ),
            )
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
        if self.config.dwell_s == 0:
            return self._activate(target, measured_pose)
        if target != self._candidate:
            self._candidate = target
            self._dwell_start_s = float(now_s)
        progress = (now_s - self._dwell_start_s) / self.config.dwell_s
        if progress + 1e-12 >= 1.0:
            return self._activate(target, measured_pose)
        return self._publish(
            "dwelling", f"Keep looking to {self._description(target)}", target=target, progress=progress
        )
