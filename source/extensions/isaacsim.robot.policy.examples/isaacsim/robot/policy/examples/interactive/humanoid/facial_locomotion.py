# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize deliberate face gestures and request supervised walking intent.

Face expressions provide intent only. The existing bounded gaze movement engine
measures travel, heading and settling, while the caller retains the physical gait,
balance guards and all hand controls. No expression updates a robot root pose.
Hold mode moves only while an expression remains held and stops upon relaxation.
Optional step mode commits winks on reopening and mouth gestures on confirmation,
then finishes one short movement independently of expression. A bilateral blink
poisons the entire cycle until neutral. Stale input or lost readiness cancels all
intent. This module owns no sockets, XR devices, scene prims or simulator callbacks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

from isaacsim.robot.policy.examples.interactive.humanoid.gaze_locomotion import GazeLocomotion, GazeLocomotionConfig

if TYPE_CHECKING:
    from isaacsim.robot.policy.examples.interactive.humanoid.face_tracking_input import FaceTrackingSample

__all__ = ["FacialLocomotion", "FacialLocomotionCommand", "FacialLocomotionConfig"]

REQUIRED_WEIGHTS = ("EyesClosedL", "EyesClosedR", "JawDrop", "MouthLeft", "MouthRight", "LipPuckerL", "LipPuckerR")
ZERO_VELOCITY = (0.0, 0.0, 0.0)


def _finite(value: object) -> bool:
    """Accept finite real measurements while excluding boolean flags."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class FacialLocomotionConfig:
    """Configure expression thresholds and short physical movements.

    Args:
        preview: Publish recognition events with zero physical velocity.
        control_mode: ``hold`` moves while expressive; ``step`` completes a short movement.
        distance_m: Requested projected distance for a forward, back or side step.
        turn_degrees: Requested signed yaw change for a completed wink.
        forward_speed: Body-forward command, within the supported walking cap.
        backward_speed: Reverse magnitude, within the supported walking cap.
        lateral_speed: Sidestep magnitude, within the supported walking cap.
        yaw_speed: Turning magnitude in radians per second.
        activation_threshold: Minimum expression weight to begin confirmation.
        wink_threshold: Optional eye-closure activation threshold; None uses the shared threshold.
        jaw_threshold: Optional jaw-opening activation threshold; None uses the shared threshold.
        pucker_threshold: Optional bilateral-pucker activation threshold; None uses the shared threshold.
        mouth_side_threshold: Optional left/right mouth activation threshold; None uses the shared threshold.
        release_threshold: Maximum weight permitted in neutral and unused channels.
        confirm_s: Sustained wall-time confirmation on distinct input samples.
        neutral_s: Sustained neutral interval required before a new gesture.
        max_sample_age_s: Maximum accepted receive age and update interruption.
        max_gesture_s: Maximum duration for completing a unilateral wink.

    Raises:
        ValueError: A value is nonfinite, has the wrong type or is outside its range.
    """

    preview: bool = True
    control_mode: str = "hold"
    distance_m: float = 0.15
    turn_degrees: float = 10.0
    forward_speed: float = 0.20
    backward_speed: float = 0.15
    lateral_speed: float = 0.15
    yaw_speed: float = 0.25
    activation_threshold: float = 0.65
    wink_threshold: float | None = None
    jaw_threshold: float | None = None
    pucker_threshold: float | None = None
    mouth_side_threshold: float | None = None
    release_threshold: float = 0.25
    confirm_s: float = 0.18
    neutral_s: float = 0.25
    max_sample_age_s: float = 0.25
    max_gesture_s: float = 2.0

    def __post_init__(self) -> None:
        """Validate settings before any gesture can request movement."""
        if type(self.preview) is not bool:
            raise ValueError("preview must be a boolean")
        if self.control_mode not in ("hold", "step"):
            raise ValueError("control_mode must be hold or step")
        for name, lower, upper in (
            ("distance_m", 0.01, 1.0),
            ("turn_degrees", 1.0, 45.0),
            ("forward_speed", 0.01, 0.65),
            ("backward_speed", 0.01, 0.35),
            ("lateral_speed", 0.01, 0.30),
            ("yaw_speed", 0.01, 0.40),
            ("activation_threshold", 0.40, 1.0),
            ("release_threshold", 0.0, 0.50),
            ("confirm_s", 0.08, 1.0),
            ("neutral_s", 0.10, 1.0),
            ("max_sample_age_s", 0.05, 1.0),
            ("max_gesture_s", 0.40, 5.0),
        ):
            value = getattr(self, name)
            if not _finite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
        if self.release_threshold >= self.activation_threshold:
            raise ValueError("release_threshold must be smaller than activation_threshold")
        for name in ("wink_threshold", "jaw_threshold", "pucker_threshold", "mouth_side_threshold"):
            value = getattr(self, name)
            if value is None:
                continue
            if not _finite(value) or not 0.40 <= value <= 1.0:
                raise ValueError(f"{name} must be None or finite and in [0.4, 1.0]")
            if self.release_threshold >= value:
                raise ValueError(f"release_threshold must be smaller than {name}")
        if self.max_gesture_s <= self.confirm_s:
            raise ValueError("max_gesture_s must exceed confirm_s")


@dataclass(frozen=True)
class FacialLocomotionCommand:
    """Publish recognition and body-forward/body-left/yaw-left intent.

    ``activated_action`` is an event on one update only. ``busy`` covers physical
    travel and measured settling; new expressions during that period are ignored.
    ``neutral_ready`` permits a new expression after a valid neutral interval.
    ``stop_immediately`` asks the caller to clear the velocity smoothing filter.
    """

    state: str = "awaiting_neutral"
    target: str | None = None
    progress: float = 0.0
    last_action: str | None = None
    activation_count: int = 0
    velocity: tuple[float, float, float] = ZERO_VELOCITY
    detail: str = "Relax your face with both eyes open to enable gestures"
    activated_action: str | None = None
    stop_immediately: bool = False
    busy: bool = False
    neutral_ready: bool = False


class FacialLocomotion:
    """Convert deliberate expressions into supervised base movement.

    Both eyes must remain open for mouth commands. A wink requires the opposite
    eye and all mouth channels to remain neutral throughout its recognition cycle.
    Any conflicting expression requires a fresh neutral interval. The caller must
    explicitly enable this controller before speaking or eating can create intent.

    Args:
        config: Validated gesture and movement configuration, or preview defaults.

    Example:

        .. code-block:: python

            controller = FacialLocomotion()
            print(controller.command.velocity)  # (0.0, 0.0, 0.0)
    """

    def __init__(self, config: FacialLocomotionConfig | None = None) -> None:
        if config is not None and not isinstance(config, FacialLocomotionConfig):
            raise TypeError("config must be a FacialLocomotionConfig")
        self.config = config if config is not None else FacialLocomotionConfig()
        self.reset()

    @property
    def stop_diagnostics(self) -> dict:
        """Return the bounded motion engine's most recent stop evidence."""
        return self._motion.stop_diagnostics

    @property
    def heading_status(self) -> dict:
        """Return heading correction evidence in the selected movement frame."""
        return self._motion.heading_status

    def reset(self) -> None:
        """Cancel intent and require a new neutral interval, including after faults."""
        self._motion = self._make_motion(lateral=False)
        self._action: str | None = None
        self._last_action: str | None = None
        self._activation_count = 0
        self._last_time_s: float | None = None
        self._last_sequence: int | None = None
        self._fault: str | None = None
        self._clear_recognition()
        self.command = FacialLocomotionCommand()

    def _make_motion(self, *, lateral: bool) -> GazeLocomotion:
        """Reuse the measured step engine with supported directional speed limits."""
        config = self.config
        return GazeLocomotion(
            GazeLocomotionConfig(
                dwell_s=0.0,
                distance_m=config.distance_m,
                turn_degrees=config.turn_degrees,
                forward_speed=config.lateral_speed if lateral else config.forward_speed,
                backward_speed=config.lateral_speed if lateral else config.backward_speed,
                yaw_speed=config.yaw_speed,
                preview=config.preview,
                max_gaze_gap_s=config.max_sample_age_s,
                control_mode=config.control_mode,
            )
        )

    def _clear_recognition(self) -> None:
        """Discard pending gestures without allowing held expressions to rearm."""
        self._neutral_ready = False
        self._neutral_since_s: float | None = None
        self._candidate: str | None = None
        self._candidate_since_s: float | None = None
        self._wink_confirmed = False

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
    ) -> FacialLocomotionCommand:
        """Publish a fresh immutable command and enforce zero intent in preview."""
        self.command = FacialLocomotionCommand(
            state=state,
            detail=detail,
            target=target,
            progress=max(0.0, min(1.0, progress)),
            velocity=ZERO_VELOCITY if self.config.preview else velocity,
            last_action=self._last_action,
            activation_count=self._activation_count,
            activated_action=activated_action,
            stop_immediately=stop_immediately,
            busy=self._action is not None,
            neutral_ready=self._neutral_ready,
        )
        return self.command

    def _inhibit(self, detail: str, *, fault: bool = False) -> FacialLocomotionCommand:
        """Cancel all intent immediately while preserving the recognition count."""
        self._action = None
        self._motion.reset()
        self._clear_recognition()
        if fault:
            self._fault = detail
        return self._publish("fault" if fault else "unavailable", detail, stop_immediately=True)

    def _read_sample(self, sample: FaceTrackingSample, now_s: float) -> dict[str, float] | None:
        """Check every required scalar instead of trusting a sender's validity flag."""
        if getattr(sample, "valid", False) is not True:
            return None
        age = getattr(sample, "age_s", None)
        received = getattr(sample, "received_at", None)
        sequence = getattr(sample, "sequence", None)
        if (
            not _finite(age)
            or not 0 <= age <= self.config.max_sample_age_s
            or not _finite(received)
            or not 0 <= now_s - received <= self.config.max_sample_age_s
            or type(sequence) is not int
            or sequence < 0
        ):
            return None
        values = getattr(sample, "weights", None)
        if not isinstance(values, dict):
            return None
        if any(not _finite(values.get(name)) or not 0 <= values[name] <= 1 for name in REQUIRED_WEIGHTS):
            return None
        return values

    def _threshold_for(self, action: str) -> float:
        """Resolve one expression's optional wearer calibration with shared fallback."""
        name = {
            "turn_left": "wink_threshold",
            "turn_right": "wink_threshold",
            "forward": "jaw_threshold",
            "back": "pucker_threshold",
            "strafe_left": "mouth_side_threshold",
            "strafe_right": "mouth_side_threshold",
        }[action]
        value = getattr(self.config, name)
        return self.config.activation_threshold if value is None else value

    def _classify(self, values: dict[str, float]) -> tuple[str | None, bool]:
        """Return one unambiguous gesture or neutral, blocking all partial conflicts."""
        low = self.config.release_threshold
        left, right = values["EyesClosedL"], values["EyesClosedR"]
        mouth = tuple(values[name] for name in REQUIRED_WEIGHTS[2:])
        if left <= low and right <= low and max(mouth) <= low:
            return None, True
        if max(mouth) <= low:
            if left >= self._threshold_for("turn_left") and right <= low:
                return "turn_left", False
            if right >= self._threshold_for("turn_right") and left <= low:
                return "turn_right", False
        if max(left, right) <= low:
            for name, action in (("JawDrop", "forward"), ("MouthLeft", "strafe_left"), ("MouthRight", "strafe_right")):
                if values[name] >= self._threshold_for(action) and all(
                    values[other] <= low for other in REQUIRED_WEIGHTS[2:] if other != name
                ):
                    return action, False
            if min(values["LipPuckerL"], values["LipPuckerR"]) >= self._threshold_for("back") and max(mouth[:3]) <= low:
                return "back", False
        return None, False

    def _recognize(self, values: dict[str, float], now_s: float) -> str | None:
        """Accumulate a deliberate gesture only across distinct fresh samples."""
        action, neutral = self._classify(values)
        if self._candidate in ("turn_left", "turn_right"):
            candidate = self._candidate
            elapsed = now_s - self._candidate_since_s
            if elapsed > self.config.max_gesture_s:
                self._clear_recognition()
                self._publish("awaiting_neutral", "Wink took too long; relax both eyes before retrying")
                return None
            # Check the opposite eye even in the closing/opening threshold band.
            # A staggered bilateral blink must not become a wink on reopening.
            opposite = "EyesClosedR" if candidate == "turn_left" else "EyesClosedL"
            conflicts = values[opposite] > self.config.release_threshold or any(
                values[name] > self.config.release_threshold for name in REQUIRED_WEIGHTS[2:]
            )
            if conflicts:
                self._clear_recognition()
                self._publish("awaiting_neutral", "Blink or conflicting expression ignored; return to neutral")
                return None
            if neutral:
                confirmed = self._wink_confirmed
                self._clear_recognition()
                if confirmed:
                    return candidate
                self._publish("awaiting_neutral", "Brief eye closure ignored; return to neutral")
                return None
            if action == candidate:
                if elapsed + 1e-12 >= self.config.confirm_s:
                    self._wink_confirmed = True
                    if self.config.control_mode == "hold":
                        self._clear_recognition()
                        return candidate
                self._publish(
                    "wink_release" if self._wink_confirmed else "confirming",
                    "Reopen the winked eye to turn" if self._wink_confirmed else "Confirming deliberate wink",
                    target=candidate,
                    progress=elapsed / self.config.confirm_s,
                )
                return None
            # Do not accumulate confirmation through a partial closure.
            if not self._wink_confirmed:
                self._candidate_since_s = now_s
            self._publish("confirming", "Complete the deliberate wink", target=candidate)
            return None
        if neutral:
            self._candidate = self._candidate_since_s = None
            if self._neutral_since_s is None:
                self._neutral_since_s = now_s
            if now_s - self._neutral_since_s + 1e-12 >= self.config.neutral_s:
                self._neutral_ready = True
            self._publish(
                "idle" if self._neutral_ready else "awaiting_neutral",
                "Ready for one deliberate expression" if self._neutral_ready else "Relax your face with both eyes open",
                progress=(now_s - self._neutral_since_s) / self.config.neutral_s,
            )
            return None
        self._neutral_since_s = None
        if not self._neutral_ready:
            self._publish("awaiting_neutral", "Relax your face before another expression")
            return None
        if action is None or (self._candidate is not None and action != self._candidate):
            partial = self._partial_action(values) if action is None else None
            if partial is not None and self._candidate in (None, partial):
                # Facial weights rise gradually. A single expression crossing
                # the threshold band must not invalidate its preceding neutral.
                self._candidate = self._candidate_since_s = None
                self._publish("confirming", "Make one deliberate expression", target=partial)
                return None
            self._clear_recognition()
            self._publish("awaiting_neutral", "Ambiguous expression ignored; return to neutral")
            return None
        if self._candidate is None:
            self._candidate, self._candidate_since_s = action, now_s
        elapsed = now_s - self._candidate_since_s
        if action not in ("turn_left", "turn_right") and elapsed + 1e-12 >= self.config.confirm_s:
            self._clear_recognition()
            return action
        self._publish(
            "confirming",
            "Hold one deliberate expression briefly",
            target=action,
            progress=elapsed / self.config.confirm_s,
        )
        return None

    def _partial_action(self, values: dict[str, float]) -> str | None:
        """Identify an isolated rising expression without counting confirmation."""
        low = self.config.release_threshold
        for action, names in (
            ("turn_left", ("EyesClosedL",)),
            ("turn_right", ("EyesClosedR",)),
            ("forward", ("JawDrop",)),
            ("back", ("LipPuckerL", "LipPuckerR")),
            ("strafe_left", ("MouthLeft",)),
            ("strafe_right", ("MouthRight",)),
        ):
            if max(values[name] for name in names) > low and all(
                values[name] <= low for name in REQUIRED_WEIGHTS if name not in names
            ):
                return action
        return None

    def _held_expression(self, values: dict[str, float]) -> tuple[bool, bool]:
        """Apply release hysteresis while rejecting a second active expression."""
        low = self.config.release_threshold
        if max(values[name] for name in REQUIRED_WEIGHTS) <= low:
            return False, False
        names = {
            "turn_left": ("EyesClosedL",),
            "turn_right": ("EyesClosedR",),
            "forward": ("JawDrop",),
            "back": ("LipPuckerL", "LipPuckerR"),
            "strafe_left": ("MouthLeft",),
            "strafe_right": ("MouthRight",),
        }[self._action]
        held = min(values[name] for name in names) > low and all(
            values[name] <= low for name in REQUIRED_WEIGHTS if name not in names
        )
        return held, not held

    @staticmethod
    def _movement_pose(
        pose: tuple[float, float, float] | None, action: str | None
    ) -> tuple[float, float, float] | None:
        """Rotate the virtual movement frame for measured left/right sidesteps."""
        measured = GazeLocomotion._pose(pose)
        if measured is not None and action in ("strafe_left", "strafe_right"):
            return measured[0], measured[1], measured[2] + math.pi / 2
        return measured

    def _advance_motion(
        self,
        *,
        now_s: float,
        dt: float,
        pose: tuple[float, float, float] | None,
        linear_speed_m_s: float,
        yaw_speed_rad_s: float,
        activating: bool = False,
        held: bool = True,
        release_immediately: bool = False,
    ) -> FacialLocomotionCommand:
        """Apply the existing bounded-distance, turn and measured-stop logic."""
        action = self._action
        target = {"strafe_left": "forward", "strafe_right": "back"}.get(action, action)
        arguments = dict(
            eye_valid=not release_immediately,
            now_s=now_s,
            dt=dt,
            pose=self._movement_pose(pose, action),
            linear_speed_m_s=linear_speed_m_s,
            yaw_speed_rad_s=yaw_speed_rad_s,
            ready=True,
        )
        if activating:
            self._motion = self._make_motion(lateral=action in ("strafe_left", "strafe_right"))
            self._motion.update(target=None, **arguments)
        keep_target = activating or (self.config.control_mode == "hold" and held)
        result = self._motion.update(target=target if keep_target else None, **arguments)
        if result.state == "fault":
            return self._inhibit(result.detail, fault=True)
        if activating and result.activated_action is None:
            return self._inhibit("Valid measured robot pose and speeds are required")
        if activating:
            self._last_action = action
            self._activation_count += 1
        velocity = result.velocity
        if action in ("strafe_left", "strafe_right"):
            velocity = (0.0, velocity[0], velocity[2])
        preview_held = self.config.preview and self.config.control_mode == "hold" and held
        if result.state not in ("active", "stopping") and not preview_held:
            self._action = None
            self._clear_recognition()
        state = "awaiting_neutral" if result.state == "awaiting_release" else result.state
        if state == "preview":
            detail = (
                f"Would {action.replace('_', ' ')}; relax to stop"
                if self.config.control_mode == "hold"
                else (f"Would {action.replace('_', ' ')}; relax before another expression")
            )
        elif state == "active":
            detail = (
                f"Moving {action.replace('_', ' ')}; relax to stop"
                if self.config.control_mode == "hold"
                else (f"Completing {action.replace('_', ' ')}; relax while the movement finishes")
            )
        elif self.config.control_mode == "hold" and not held:
            state = "awaiting_neutral"
            detail = "Relax your face before another expression"
        elif state == "awaiting_neutral":
            detail = "Movement stopped; relax your face before another expression"
        else:
            detail = result.detail
        return self._publish(
            state,
            detail,
            target=action,
            progress=result.progress,
            velocity=velocity,
            activated_action=action if activating else None,
            stop_immediately=result.stop_immediately,
        )

    def update(
        self,
        *,
        sample: FaceTrackingSample,
        now_s: float,
        dt: float,
        pose: tuple[float, float, float] | None,
        linear_speed_m_s: float,
        yaw_speed_rad_s: float,
        ready: bool = True,
        enabled: bool = True,
    ) -> FacialLocomotionCommand:
        """Consume a fresh face sample and measured motion without reading hands.

        Args:
            sample: Transport-validated face weights with receipt age and sequence.
            now_s: Monotonic wall time shared with the receiver.
            dt: Positive physics timestep in seconds.
            pose: Measured world X, Y and heading, or None in preview.
            linear_speed_m_s: Measured planar speed magnitude.
            yaw_speed_rad_s: Measured yaw-speed magnitude.
            ready: Valid simulation and XR session with no articulation fault.
            enabled: Explicit facial-control arm state owned by the caller.

        Returns:
            Current recognition state and supported body-frame velocity intent.
        """
        if self._fault is not None:
            return self._publish("fault", self._fault, stop_immediately=True)
        if not _finite(now_s) or now_s < 0 or not _finite(dt) or dt <= 0:
            self._last_time_s = None
            return self._inhibit("Invalid input clock or simulation timestep", fault=self._action is not None)
        previous = self._last_time_s
        self._last_time_s = now_s
        if enabled is not True or ready is not True:
            return self._inhibit(
                "Facial controls disabled" if enabled is not True else "Facial locomotion is not ready"
            )
        if previous is not None and (now_s < previous or now_s - previous > self.config.max_sample_age_s):
            return self._inhibit("Face update interrupted; return to neutral")
        values = self._read_sample(sample, now_s)
        if values is None:
            return self._inhibit("Fresh valid face tracking is required; return to neutral after recovery")
        sequence = sample.sequence
        if self._last_sequence is not None and sequence < self._last_sequence:
            self._last_sequence = sequence
            return self._inhibit("Face stream sequence restarted; return to neutral")
        fresh = sequence != self._last_sequence
        self._last_sequence = sequence
        arguments = dict(
            now_s=now_s,
            dt=dt,
            pose=pose,
            linear_speed_m_s=linear_speed_m_s,
            yaw_speed_rad_s=yaw_speed_rad_s,
        )
        if self._action is not None:
            if self.config.control_mode == "hold":
                held, immediate = self._held_expression(values)
                return self._advance_motion(held=held, release_immediately=immediate, **arguments)
            return self._advance_motion(**arguments)
        if not fresh:
            return self._publish(
                self.command.state,
                self.command.detail,
                target=self.command.target,
                progress=self.command.progress,
            )
        action = self._recognize(values, now_s)
        if action is not None:
            self._action = action
            return self._advance_motion(activating=True, **arguments)
        return self.command
