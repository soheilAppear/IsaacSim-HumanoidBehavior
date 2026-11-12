# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize deliberate face gestures and request supervised walking intent.

Face expressions provide intent only. The existing bounded gaze movement engine
measures travel, heading and settling, while the caller retains the physical gait,
balance guards and all hand controls. No expression updates a robot root pose.
Hold mode moves only while an expression remains held and stops upon relaxation.
Optional step mode commits winks on reopening and mouth gestures on confirmation,
then finishes one short movement independently of expression. Hold mode handles
eye and mouth intent separately: winks take priority over incidental mouth motion,
and a blink pauses a held mouth request until the eyes reopen and it is reconfirmed.
Stale input or lost readiness cancels all
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

    Speed bounds accept experimental requests, not a claimed trained gait
    envelope. The host separately validates the physical profile and supervises
    measured balance, speed and stopping. Default requests remain conservative.

    Args:
        preview: Publish recognition events with zero physical velocity.
        control_mode: ``hold`` moves while expressive; ``step`` completes a short movement.
        distance_m: Requested projected distance for a forward, back or side step.
        turn_degrees: Requested signed yaw change for a completed wink.
        forward_speed: Body-forward request, at most 0.65 meters per second.
        backward_speed: Reverse request magnitude, at most 0.45 meters per second.
        lateral_speed: Sidestep request magnitude, at most 0.45 meters per second.
        yaw_speed: Turning request magnitude, at most 0.75 radians per second.
        activation_threshold: Minimum expression weight to begin confirmation.
        wink_threshold: Optional eye-closure activation threshold; None uses the shared threshold.
        right_wink_threshold: Optional right-eye activation override; a lower calibration requires the left eye clearly open.
        right_wink_release_threshold: Optional right-eye release override; None uses release_threshold.
        right_wink_confirm_s: Optional right-wink confirmation duration; None uses confirm_s.
        jaw_threshold: Optional jaw-opening activation threshold; None uses the shared threshold.
        pucker_threshold: Optional bilateral-pucker activation threshold; None uses the shared threshold.
        mouth_side_threshold: Optional left/right mouth activation threshold; None uses the shared threshold.
        right_mouth_threshold: Optional right-mouth activation override; None uses mouth_side_threshold.
        release_threshold: Maximum weight permitted in neutral and unused channels.
        eye_open_threshold: Maximum closure per eye for hold-mode mouth motion; independent of wink detection.
        confirm_s: Sustained wall-time confirmation on distinct input samples.
        post_blink_wink_confirm_s: Longer hold confirmation for a unilateral wink following a bilateral blink.
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
    right_wink_threshold: float | None = None
    right_wink_release_threshold: float | None = None
    right_wink_confirm_s: float | None = None
    jaw_threshold: float | None = None
    pucker_threshold: float | None = None
    mouth_side_threshold: float | None = None
    right_mouth_threshold: float | None = None
    release_threshold: float = 0.25
    eye_open_threshold: float = 0.15
    confirm_s: float = 0.18
    post_blink_wink_confirm_s: float = 0.5
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
            ("backward_speed", 0.01, 0.45),
            ("lateral_speed", 0.01, 0.45),
            ("yaw_speed", 0.01, 0.75),
            ("activation_threshold", 0.40, 1.0),
            ("release_threshold", 0.0, 0.50),
            ("eye_open_threshold", 0.0, 0.35),
            ("confirm_s", 0.08, 1.0),
            ("post_blink_wink_confirm_s", 0.08, 5.0),
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
        for name, lower, upper in (
            ("right_wink_threshold", 0.25, 1.0),
            ("right_wink_release_threshold", 0.0, 0.50),
            ("right_wink_confirm_s", 0.08, 1.0),
            ("right_mouth_threshold", 0.40, 1.0),
        ):
            value = getattr(self, name)
            if value is not None and (not _finite(value) or not lower <= value <= upper):
                raise ValueError(f"{name} must be None or finite and in [{lower}, {upper}]")
        right_activation = self.right_wink_threshold
        if right_activation is None:
            right_activation = self.wink_threshold if self.wink_threshold is not None else self.activation_threshold
        right_release = (
            self.right_wink_release_threshold
            if self.right_wink_release_threshold is not None
            else self.release_threshold
        )
        if right_release >= right_activation:
            raise ValueError("Effective right-eye release must be smaller than its activation threshold")
        if self.right_mouth_threshold is not None and self.release_threshold >= self.right_mouth_threshold:
            raise ValueError("release_threshold must be smaller than right_mouth_threshold")
        if self.max_gesture_s <= self.confirm_s:
            raise ValueError("max_gesture_s must exceed confirm_s")
        if self.right_wink_confirm_s is not None and self.max_gesture_s <= self.right_wink_confirm_s:
            raise ValueError("max_gesture_s must exceed right_wink_confirm_s")
        if self.post_blink_wink_confirm_s < self.confirm_s:
            raise ValueError("post_blink_wink_confirm_s must be at least confirm_s")


@dataclass(frozen=True)
class FacialLocomotionCommand:
    """Publish recognition and body-forward/body-left/yaw-left intent.

    ``activated_action`` is an event on one update only. ``busy`` covers physical
    travel and measured settling. Step mode ignores new expressions during that
    period; hold mode can stop and reconfirm a different direction.
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

    Startup and tracking recovery require a neutral face. In hold mode, a qualified
    unilateral wink owns the request even when the mouth moves with it. Short
    asymmetric blink tails do not become turns. A clearly unilateral closure
    held beyond the longer post-blink interval can deliberately start a turn. A held
    mouth command pauses during eye closure and resumes after open-eye confirmation.
    Clean direction changes reconfirm without requiring the entire face to relax.
    The caller must explicitly enable this controller before expressions create intent.

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
        """Reuse the measured step engine with the configured directional requests."""
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
        self._blink_blocked = False
        self._hold_paused_action: str | None = None
        self._mouth_armed = False

    def _clear_hold_candidate(self) -> None:
        """Restart confirmation without discarding the preceding neutral arming."""
        self._candidate = self._candidate_since_s = None
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
        override = {"turn_right": "right_wink_threshold", "strafe_right": "right_mouth_threshold"}.get(action)
        if override is not None and getattr(self.config, override) is not None:
            return getattr(self.config, override)
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

    def _release_for(self, name: str) -> float:
        """Apply right-eye hysteresis consistently to holding, neutral and conflicts."""
        if name == "EyesClosedR" and self.config.right_wink_release_threshold is not None:
            return self.config.right_wink_release_threshold
        return self.config.release_threshold

    def _confirmation_for(self, action: str) -> float:
        """Retain the shared duration unless a right-wink calibration overrides it."""
        if action == "turn_right" and self.config.right_wink_confirm_s is not None:
            return self.config.right_wink_confirm_s
        return self.config.confirm_s

    def _opposite_eye_limit(self, action: str) -> float:
        """Keep a lowered right-wink activation distinct from partial bilateral closure.

        Shared/default thresholds already separate activation from the opposite
        eye's release band. A lower independent right threshold can overlap that
        band, so its opposite left eye must instead pass the clear-eye gate.
        """
        opposite = "EyesClosedR" if action == "turn_left" else "EyesClosedL"
        limit = self._release_for(opposite)
        if (
            action == "turn_right"
            and self.config.right_wink_threshold is not None
            and self.config.right_wink_threshold < self._threshold_for("turn_left")
        ):
            # Retain a nonoverlapping band even if the configurable eye-open
            # threshold is later raised above the calibrated activation.
            limit = min(limit, self.config.eye_open_threshold, self._release_for("EyesClosedR"))
        return limit

    def _eyes_clearly_open(self, values: dict[str, float]) -> bool:
        """Require both the eye-open gate and each eye's release threshold."""
        return all(
            values[name] <= min(self.config.eye_open_threshold, self._release_for(name))
            for name in ("EyesClosedL", "EyesClosedR")
        )

    def _classify(self, values: dict[str, float]) -> tuple[str | None, bool]:
        """Return one unambiguous gesture or neutral, blocking all partial conflicts."""
        low = self.config.release_threshold
        left_low, right_low = self._release_for("EyesClosedL"), self._release_for("EyesClosedR")
        left, right = values["EyesClosedL"], values["EyesClosedR"]
        mouth = tuple(values[name] for name in REQUIRED_WEIGHTS[2:])
        if left <= left_low and right <= right_low and max(mouth) <= low:
            return None, True
        if max(mouth) <= low:
            if left >= self._threshold_for("turn_left") and right <= self._opposite_eye_limit("turn_left"):
                return "turn_left", False
            if right >= self._threshold_for("turn_right") and left <= self._opposite_eye_limit("turn_right"):
                return "turn_right", False
        if left <= left_low and right <= right_low:
            for name, action in (("JawDrop", "forward"), ("MouthLeft", "strafe_left"), ("MouthRight", "strafe_right")):
                if values[name] >= self._threshold_for(action) and all(
                    values[other] <= low for other in REQUIRED_WEIGHTS[2:] if other != name
                ):
                    return action, False
            if min(values["LipPuckerL"], values["LipPuckerR"]) >= self._threshold_for("back") and max(mouth[:3]) <= low:
                return "back", False
        return None, False

    def _recognize(self, values: dict[str, float], now_s: float) -> str | None:
        """Accumulate a step-mode gesture only across distinct fresh samples."""
        action, neutral = self._classify(values)
        if self._candidate in ("turn_left", "turn_right"):
            candidate = self._candidate
            confirmation_s = self._confirmation_for(candidate)
            elapsed = now_s - self._candidate_since_s
            if elapsed > self.config.max_gesture_s:
                self._clear_recognition()
                self._publish("awaiting_neutral", "Wink took too long; relax both eyes before retrying")
                return None
            # Check the opposite eye even in the closing/opening threshold band.
            # A staggered bilateral blink must not become a wink on reopening.
            opposite = "EyesClosedR" if candidate == "turn_left" else "EyesClosedL"
            conflicts = values[opposite] > self._opposite_eye_limit(candidate) or any(
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
                if elapsed + 1e-12 >= confirmation_s:
                    self._wink_confirmed = True
                self._publish(
                    "wink_release" if self._wink_confirmed else "confirming",
                    "Reopen the winked eye to turn" if self._wink_confirmed else "Confirming deliberate wink",
                    target=candidate,
                    progress=elapsed / confirmation_s,
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
        for action, names in (
            ("turn_left", ("EyesClosedL",)),
            ("turn_right", ("EyesClosedR",)),
            ("forward", ("JawDrop",)),
            ("back", ("LipPuckerL", "LipPuckerR")),
            ("strafe_left", ("MouthLeft",)),
            ("strafe_right", ("MouthRight",)),
        ):
            if action in ("turn_left", "turn_right"):
                opposite = "EyesClosedR" if action == "turn_left" else "EyesClosedL"
                if values[opposite] > self._opposite_eye_limit(action):
                    continue
            if any(values[name] > self._release_for(name) for name in names) and all(
                values[name] <= self._release_for(name) for name in REQUIRED_WEIGHTS if name not in names
            ):
                return action
        return None

    def _update_blink_latch(self, values: dict[str, float]) -> None:
        """Require longer wink confirmation after a bilateral closure.

        A fresh clearly-both-open sample clears the latch for normal confirmation.
        Otherwise a subsequent unilateral closure must complete the configured
        longer hold before the caller clears this latch and starts turning.
        """
        left, right = values["EyesClosedL"], values["EyesClosedR"]
        if left >= self._threshold_for("turn_left") and right >= self._threshold_for("turn_right"):
            self._blink_blocked = True
        elif self._eyes_clearly_open(values):
            self._blink_blocked = False

    def _hold_wink(self, values: dict[str, float], *, continuing: str | None = None) -> str | None:
        """Select only a clear unilateral closure, independently of lower-face cross-talk."""
        for action, closed, opened in (
            ("turn_left", "EyesClosedL", "EyesClosedR"),
            ("turn_right", "EyesClosedR", "EyesClosedL"),
        ):
            threshold = self._release_for(closed) if continuing == action else self._threshold_for(action)
            closed_enough = values[closed] > threshold if continuing == action else values[closed] >= threshold
            if closed_enough and values[opened] <= self._opposite_eye_limit(action):
                return action
        return None

    def _hold_mouth(self, values: dict[str, float], *, continuing: str | None = None) -> str | None:
        """Require one mouth command while ignoring eye channels for blink-paused intent."""
        low = self.config.release_threshold
        for action, names in (
            ("forward", ("JawDrop",)),
            ("back", ("LipPuckerL", "LipPuckerR")),
            ("strafe_left", ("MouthLeft",)),
            ("strafe_right", ("MouthRight",)),
        ):
            threshold = low if continuing == action else self._threshold_for(action)
            weight = min(values[name] for name in names)
            expressive = weight > threshold if continuing == action else weight >= threshold
            if expressive and all(values[name] <= low for name in REQUIRED_WEIGHTS[2:] if name not in names):
                return action
        return None

    def _update_hold(self, values: dict[str, float], *, fresh: bool, **motion) -> FacialLocomotionCommand:
        """Resolve held gestures without allowing a transient to permanently lock them out."""
        now_s = motion["now_s"]
        low = self.config.release_threshold
        # A relaxed mouth can tolerate more weight than a clearly open eye. Using
        # its release threshold here misreads a weak wink plus cheek motion as a
        # sidestep, even though the wink never reaches its activation threshold.
        eyes_open = self._eyes_clearly_open(values)
        mouth_neutral = max(values[name] for name in REQUIRED_WEIGHTS[2:]) <= low
        if fresh:
            self._update_blink_latch(values)
            if mouth_neutral:
                self._mouth_armed = True
        if not self._neutral_ready:
            if fresh and eyes_open and mouth_neutral:
                if self._neutral_since_s is None:
                    self._neutral_since_s = now_s
                if now_s - self._neutral_since_s + 1e-12 >= self.config.neutral_s:
                    self._neutral_ready = True
            elif fresh:
                self._neutral_since_s = None
            return self._publish(
                "idle" if self._neutral_ready else "awaiting_neutral",
                "Ready for one deliberate expression" if self._neutral_ready else "Relax your face with both eyes open",
            )

        current = self._action or self._hold_paused_action
        wink = self._hold_wink(values, continuing=self._action)
        mouth = self._hold_mouth(values, continuing=current)
        preserve_mouth = current is not None and not current.startswith("turn") and mouth == current and not eyes_open
        if not eyes_open and not mouth_neutral and not preserve_mouth:
            # A newly arriving mouth signal during eye closure may be wink
            # cross-talk. Do not let it turn into translation on reopening. An
            # established mouth hold is exempt so ordinary blinks can resume it.
            self._mouth_armed = False
        # A clear wink has explicit priority. Natural cheek/lip movement during a
        # right wink must not turn it into a sidestep, including after release.
        if wink is not None:
            desired = wink
            if self._action in ("turn_left", "turn_right"):
                self._mouth_armed = mouth_neutral
            if preserve_mouth:
                # One eyelid often leads an ordinary blink. Retain the mouth
                # request until a deliberate wink actually completes confirmation.
                self._hold_paused_action = current
        elif preserve_mouth:
            if self._action is not None:
                self._advance_motion(held=False, release_immediately=True, **motion)
                if self.command.state == "fault":
                    return self.command
            self._hold_paused_action = current
            self._clear_hold_candidate()
            return self._publish(
                "blink_pause", "Open both eyes to resume the held mouth command", target=current, stop_immediately=True
            )
        elif eyes_open and self._mouth_armed:
            desired = mouth
        else:
            desired = None

        if self._action is not None and desired == self._action:
            return self._advance_motion(**motion)
        old_action = self._action
        if old_action is not None:
            # Clear the previous velocity before confirming a different direction.
            self._advance_motion(held=False, release_immediately=desired is not None or not eyes_open, **motion)
            if self.command.state == "fault":
                return self.command
        resuming = desired is not None and desired == self._hold_paused_action
        if not resuming and not preserve_mouth:
            self._hold_paused_action = None
        if desired is None:
            self._clear_hold_candidate()
            state = "idle" if eyes_open and mouth_neutral else "confirming"
            detail = (
                "Ready for another expression" if eyes_open and mouth_neutral else "Waiting for one clear expression"
            )
            if self._blink_blocked:
                state = "blink_pause"
                detail = "Open both eyes once after a blink before another wink"
            elif not eyes_open and self._hold_paused_action is None:
                state = "blink_pause"
                detail = "Open both eyes before a new mouth command, or make one clear wink"
            elif eyes_open and not self._mouth_armed:
                detail = "Relax your mouth after eye closure before a new mouth command"
            return self._publish(
                state,
                detail,
                stop_immediately=old_action is not None and not (eyes_open and mouth_neutral),
            )
        if not fresh:
            return self._publish("confirming", "Waiting for a fresh expression sample", target=desired)
        if self._candidate != desired:
            self._candidate = desired
            self._candidate_since_s = now_s
        elapsed = now_s - self._candidate_since_s
        post_blink_wink = self._blink_blocked and desired.startswith("turn")
        confirmation_s = self._confirmation_for(desired)
        if post_blink_wink:
            confirmation_s = max(confirmation_s, self.config.post_blink_wink_confirm_s)
        if elapsed + 1e-12 < confirmation_s:
            return self._publish(
                "confirming",
                (
                    "Hold this unilateral wink longer after the blink"
                    if post_blink_wink
                    else "Reconfirming the held mouth command" if resuming else "Hold one deliberate expression briefly"
                ),
                target=desired,
                progress=elapsed / confirmation_s,
                stop_immediately=old_action is not None,
            )
        self._action = desired
        self._hold_paused_action = None
        self._clear_hold_candidate()
        if desired.startswith("turn"):
            self._blink_blocked = False
            self._mouth_armed = mouth_neutral
        return self._advance_motion(activating=True, count_activation=not resuming, **motion)

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
        count_activation: bool = True,
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
        if activating and count_activation:
            self._last_action = action
            self._activation_count += 1
        velocity = result.velocity
        if action in ("strafe_left", "strafe_right"):
            velocity = (0.0, velocity[0], velocity[2])
        preview_held = self.config.preview and self.config.control_mode == "hold" and held
        if result.state not in ("active", "stopping") and not preview_held:
            self._action = None
            if self.config.control_mode == "hold":
                self._clear_hold_candidate()
            else:
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
            activated_action=action if activating and count_activation else None,
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
        if self.config.control_mode == "hold":
            return self._update_hold(values, fresh=fresh, **arguments)
        if self._action is not None:
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
