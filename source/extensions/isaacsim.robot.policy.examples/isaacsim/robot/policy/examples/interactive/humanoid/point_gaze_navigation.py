# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select floor destinations with a held optical left-hand point.

The caller raycasts the index pointing ray against the scene and supplies its
world-space floor hit. Scene queries, blue pointing-ray visualization, and the
physical walking policy stay with the caller. Eye gaze is not a navigation input.

Only a complete optical hand sample can authorize motion. A stable pointing
dwell locks a world-space destination; subsequent ray motion does not steer.
Point release or tracking loss requests an immediate stop. The command is a
bounded forward/yaw velocity for a balance controller, never a root transform.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

__all__ = ["NavigationCommand", "PointNavigation", "PointGazeNavigation", "is_pointing"]

Position = tuple[float, float, float]
Velocity = tuple[float, float, float]
NavigationState = Literal["awaiting_release", "idle", "acquiring", "planning", "active", "arrived", "blocked"]
_ZERO_VELOCITY = (0.0, 0.0, 0.0)
_FINGER_ROLES = ("index", "middle", "ring", "little")


def _finite_number(value: object) -> float | None:
    """Accept finite real measurements without converting strings or booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_position(position: Sequence[float] | None) -> Position | None:
    """Require exactly three finite world coordinates without an origin fallback."""
    try:
        if position is None or len(position) != 3:
            return None
        values = tuple(_finite_number(value) for value in position)
    except TypeError:
        return None
    return None if any(value is None for value in values) else values


def _read_curls(curls: Mapping[str, float] | None) -> tuple[float, ...] | None:
    """Require four measured digit curls; missing digits do not count as open."""
    if not isinstance(curls, Mapping):
        return None
    values = tuple(_finite_number(curls.get(role)) for role in _FINGER_ROLES)
    return None if any(value is None or not 0.0 <= value <= 1.0 for value in values) else values


def is_pointing(curls: Mapping[str, float] | None, *, retaining: bool = False) -> bool:
    """Recognize an extended index with the other three measured fingers curled.

    Args:
        curls: Complete normalized index, middle, ring, and little curls. Thumb
            position is deliberately irrelevant.
        retaining: Allow modest measurement jitter after acquisition starts.

    Returns:
        Whether the complete finite measurements satisfy the pointing pose.
    """
    digits = _read_curls(curls)
    if digits is None:
        return False
    extended_limit, curled_limit = (0.40, 0.50) if retaining else (0.25, 0.65)
    return digits[0] <= extended_limit and all(value >= curled_limit for value in digits[1:])


@dataclass(frozen=True)
class NavigationCommand:
    """Expose destination selection, immediate velocity, and operator feedback."""

    state: NavigationState = "awaiting_release"
    #: Selection state; only `active` authorizes a nonzero velocity.
    velocity: Velocity = _ZERO_VELOCITY
    #: Body forward and lateral meters per second, followed by positive-left yaw radians per second.
    target: Position | None = None
    #: Locked world-space destination; subsequent pointing cannot change it.
    candidate: Position | None = None
    #: Pointed floor patch while acquiring, suitable for a destination marker.
    acquisition_progress: float = 0.0
    #: Continuous pointing dwell fraction in [0, 1].
    detail: str = "Open your left hand before pointing"
    #: Human-readable selection or stop reason.

    @property
    def owns_hand(self) -> bool:
        """Keep arm and finger tracking live so the pointing pose stays visible."""
        return False


class PointNavigation:
    """Convert a stable pointing floor selection into forward walking intent.

    A measured non-point gesture must occur after construction, reset, arrival,
    obstacle rejection, or tracking loss before a new selection can begin. This
    prevents a held pointing pose from resuming motion across Stop/Play. By
    default targets outside the forward cone are rejected. The factory profile
    explicitly enables collision-checked routes and physical stepping turns;
    reverse motion and sidesteps remain disabled.

    The higher cruise speed starts slowing farther from the destination. The
    default approach slope remains 0.5 inverse seconds, preserving the previous
    velocity requests inside 1.2 meters while increasing speed on longer routes.
    Farther pointing can drift slightly more during dwell, within a radius fixed
    at the initial hit. This prevents a slowly moving ray from accumulating an
    arbitrarily large destination change while the dwell timer keeps running.

    Args:
        dwell_s: Required uninterrupted pointing duration on a stable floor patch.
        dwell_radius_m: Minimum allowed drift from the initial floor hit during dwell.
        max_dwell_radius_m: Maximum allowed drift for more distant floor selections.
        dwell_radius_distance_scale: Additional drift allowance per meter of initial target distance.
        min_target_distance_m: Minimum planar distance for a new destination.
        max_target_distance_m: Maximum planar distance for a new destination.
        arrival_radius_m: Planar stopping distance from the locked destination.
        slow_radius_m: Distance at which approach velocity reaches its maximum.
        min_forward_speed: Smallest positive velocity, above the actor's walking threshold.
        max_forward_speed: Maximum forward velocity in meters per second.
        max_yaw_speed: Maximum curved forward-steering yaw in radians per second.
        stepping_turn_speed: Constant yaw magnitude for zero-forward stepping turns.
            The physical actor may translate slightly while stepping; no root pose is held.
        turn_progress_timeout_s: Maximum wall seconds without measured heading improvement.
        turn_progress_rad: Required new best heading improvement to renew the turn watchdog.
        yaw_gain: Heading-error gain in inverse seconds.
        max_heading_rad: Maximum target angle from robot forward, at entry and while walking.
        max_gap_s: Baseline wall-time gap allowance between tracked hand observations.
        max_adaptive_gap_s: Absolute gap limit when the observed frame cadence is slow.
        allow_routing: Lock blocked direct destinations while a caller plans a validated route.
        allow_turn_in_place: Permit bounded actor yaw when fresh turning clearance is true.
        waypoint_radius_m: Intermediate corner tolerance; smaller than final arrival radius.

    Raises:
        ValueError: If timing, geometry, or velocity limits are nonfinite or inconsistent.
    """

    def __init__(
        self,
        *,
        dwell_s: float = 0.6,
        dwell_radius_m: float = 0.18,
        max_dwell_radius_m: float = 0.30,
        dwell_radius_distance_scale: float = 0.04,
        min_target_distance_m: float = 0.5,
        max_target_distance_m: float = 12.0,
        arrival_radius_m: float = 0.30,
        slow_radius_m: float = 1.60,
        min_forward_speed: float = 0.08,
        max_forward_speed: float = 0.65,
        max_yaw_speed: float = 0.20,
        stepping_turn_speed: float = 0.40,
        turn_progress_timeout_s: float = 15.0,
        turn_progress_rad: float = 0.05,
        yaw_gain: float = 0.8,
        max_heading_rad: float = math.pi / 3.0,
        max_gap_s: float = 0.25,
        max_adaptive_gap_s: float = 0.75,
        allow_routing: bool = False,
        allow_turn_in_place: bool = False,
        waypoint_radius_m: float = 0.15,
    ) -> None:
        settings = {
            "dwell_s": dwell_s,
            "dwell_radius_m": dwell_radius_m,
            "max_dwell_radius_m": max_dwell_radius_m,
            "dwell_radius_distance_scale": dwell_radius_distance_scale,
            "min_target_distance_m": min_target_distance_m,
            "max_target_distance_m": max_target_distance_m,
            "arrival_radius_m": arrival_radius_m,
            "slow_radius_m": slow_radius_m,
            "min_forward_speed": min_forward_speed,
            "max_forward_speed": max_forward_speed,
            "max_yaw_speed": max_yaw_speed,
            "stepping_turn_speed": stepping_turn_speed,
            "turn_progress_timeout_s": turn_progress_timeout_s,
            "turn_progress_rad": turn_progress_rad,
            "yaw_gain": yaw_gain,
            "max_heading_rad": max_heading_rad,
            "max_gap_s": max_gap_s,
            "max_adaptive_gap_s": max_adaptive_gap_s,
            "waypoint_radius_m": waypoint_radius_m,
        }
        for name, raw_value in settings.items():
            value = _finite_number(raw_value)
            if value is None or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite real number")
            setattr(self, name, value)
        if not self.arrival_radius_m < self.min_target_distance_m <= self.max_target_distance_m:
            raise ValueError("Target distances must satisfy arrival < minimum <= maximum")
        if self.max_dwell_radius_m < self.dwell_radius_m:
            raise ValueError("Maximum dwell radius must not be smaller than its minimum")
        if self.slow_radius_m <= self.arrival_radius_m:
            raise ValueError("The approach slowing radius must exceed the arrival radius")
        if self.min_forward_speed > self.max_forward_speed:
            raise ValueError("Minimum forward speed must not exceed maximum forward speed")
        if self.max_heading_rad >= math.pi / 2.0:
            raise ValueError("The forward heading cone must remain smaller than 90 degrees")
        if self.max_adaptive_gap_s < self.max_gap_s:
            raise ValueError("The adaptive gap limit must not be smaller than the baseline gap")
        if self.stepping_turn_speed < self.max_yaw_speed:
            raise ValueError("Stepping-turn speed must not be smaller than the curved-steering limit")
        if self.turn_progress_rad >= 0.25:
            raise ValueError("Turn progress threshold must be smaller than the turn-exit angle")
        if not isinstance(allow_routing, bool) or not isinstance(allow_turn_in_place, bool):
            raise ValueError("Routing and turning options must be booleans")
        if self.waypoint_radius_m >= self.arrival_radius_m:
            raise ValueError("Intermediate waypoint tolerance must be smaller than final arrival tolerance")
        self.allow_routing = allow_routing
        self.allow_turn_in_place = allow_turn_in_place
        self.reset()

    @property
    def command(self) -> NavigationCommand:
        """Return the last command without advancing input timing."""
        return self._command

    @property
    def state(self) -> NavigationState:
        """Return the current selection state."""
        return self._command.state

    @property
    def target(self) -> Position | None:
        """Return the locked destination, including after arrival or blockage."""
        return self._command.target

    @property
    def owns_hand(self) -> bool:
        """Allow ordinary arm and finger tracking during pointing and walking."""
        return False

    @property
    def waypoint(self) -> Position | None:
        """Return the next route corner for a fresh collision query, or the final goal."""
        return self._route[self._route_index] if self._route_index < len(self._route) else self.target

    @property
    def is_final_waypoint(self) -> bool:
        """Permit stopping short only on the final approach, never at route corners."""
        return not self._route or self._route_index >= len(self._route) - 1

    @property
    def max_requested_yaw_speed(self) -> float:
        """Expose the physical clamp without raising the curved-steering speed."""
        return self.stepping_turn_speed if self.allow_turn_in_place else self.max_yaw_speed

    @property
    def observation_gap_limit_s(self) -> float:
        """Allow bounded frame jitter using previously observed tracked-input cadence."""
        return min(self.max_adaptive_gap_s, max(self.max_gap_s, 4.0 * self._observed_interval_s + 0.05))

    def reset(self, *, require_release: bool = True) -> NavigationCommand:
        """Stop and discard all timing, optionally requiring a measured release.

        Args:
            require_release: Require a valid non-point pose before acquiring again.
                Keep this true for timeline resets and tracking interruptions.

        Returns:
            A zero-velocity command with no retained destination.
        """
        self._last_time_s: float | None = None
        self._observed_interval_s = 0.0
        self._acquisition_start_s: float | None = None
        self._acquisition_origin: Position | None = None
        self._acquisition_radius_m = self.dwell_radius_m
        self._route: tuple[Position, ...] = ()
        self._route_index = 0
        self._turning = False
        self._turn_progress_time_s: float | None = None
        self._turn_progress_heading_rad: float | None = None
        self._command = NavigationCommand(
            state="awaiting_release" if require_release else "idle",
            detail=("Open your left hand before pointing" if require_release else "Point at a clear floor spot"),
        )
        return self._command

    def update(
        self,
        *,
        curls: Mapping[str, float] | None,
        hand_valid: bool,
        point_hit: Sequence[float] | None,
        point_walkable: bool,
        path_clear: bool,
        robot_position: Sequence[float] | None,
        robot_yaw: float,
        now_s: float,
        route: Sequence[Sequence[float]] | None = None,
        route_failed: bool = False,
        turn_clear: bool = False,
    ) -> NavigationCommand:
        """Evaluate fresh input and return a bounded, momentary movement request.

        Args:
            curls: Complete optical left-hand digit curls; never inferred or cached.
            hand_valid: True only for a currently tracked measured hand pose and
                finger sample. Valid-but-untracked extrapolated poses are false.
            point_hit: Validated floor destination from the index ray or object approach, in world meters.
            point_walkable: Whether the supplied destination is verified walkable floor.
            path_clear: Whether the robot-width route to the candidate, or to the
                locked target while active, is clear of obstacles.
            robot_position: Measured robot world position; only XY controls approach distance.
            robot_yaw: Measured world heading in radians, with positive yaw turning left.
            now_s: Monotonic wall time of this fresh observation. Duplicate,
                backwards, invalid, or overly separated times stop immediately.
            route: Fully collision-checked floor corners supplied only while planning.
                The last point must match the locked goal. Adoption requests zero
                velocity so the next callback can check its first segment freshly.
            route_failed: Latch a stopped result when the bounded planner cannot find a route.
            turn_clear: Fresh collision query of the full-turn body footprint.

        Returns:
            Selection feedback and forward/lateral/yaw velocity. Non-active states
            always return exact zero; pointing-ray hits are ignored after target lock.
        """
        now = _finite_number(now_s)
        position = _read_position(robot_position)
        yaw = _finite_number(robot_yaw)
        digits = _read_curls(curls)
        if hand_valid is not True or digits is None:
            return self._interrupt("Left-hand tracking unavailable; open your hand to rearm")
        if now is None or now < 0.0 or position is None or yaw is None:
            return self._interrupt("Invalid input or robot pose; open your hand to rearm")
        if self._last_time_s is not None:
            interval = now - self._last_time_s
            gap_limit = self.observation_gap_limit_s
            if not 0.0 < interval <= gap_limit:
                stopped = self._interrupt(
                    f"Input timing interrupted ({interval:.3f}s; limit {gap_limit:.3f}s); open your hand to rearm"
                )
                if 0.0 < interval <= self.max_adaptive_gap_s:
                    # A few fast callbacks may precede the first slow render
                    # frame. Learn its cadence only after stopping; a measured
                    # release and new dwell remain mandatory before any motion.
                    self._last_time_s = now
                    self._observed_interval_s = interval
                return stopped
            # Physics callbacks can arrive in render-frame bursts. Retain the
            # recent slow-frame interval instead of mistaking the tiny intervals
            # within a burst for the expected sensor cadence. A new long stall
            # cannot increase its own allowance: validation above uses history.
            self._observed_interval_s = max(interval, self._observed_interval_s * math.exp(-interval / 2.0))
        self._last_time_s = now

        # Release must itself be a valid measurement. Missing fingers, a missing
        # hand, and stale frames cannot clear the stop/restart interlock.
        retaining = self.state in ("acquiring", "planning", "active", "arrived", "blocked", "awaiting_release")
        if not is_pointing(curls, retaining=retaining):
            return self._set_idle("Point at a clear floor spot")
        if self.state in ("awaiting_release", "arrived", "blocked"):
            return self._command

        if self.state == "planning":
            if route_failed:
                return self._stop_at_target("blocked", "No clear floor route found; open your hand and select again")
            if route is None:
                return self._command
            try:
                points = tuple(_read_position(point) for point in route)
            except TypeError:
                points = ()
            if (
                not points
                or len(points) > 8192
                or any(point is None for point in points)
                or math.dist(points[-1], self.target) > 0.05
            ):
                return self._stop_at_target("blocked", "Invalid planned route; open your hand and select again")
            self._route = points
            self._route_index = 0
            # Do not reuse the previous frame's clearance for a newly delivered
            # route. The caller must query the first waypoint before any motion.
            self._command = NavigationCommand(
                state="active", target=self.target, detail="Checking the first route segment"
            )
            return self._command

        if self.state == "active":
            if path_clear is not True:
                return self._stop_at_target("blocked", "Route blocked; open your hand and select a clear floor spot")
            return self._approach(position, yaw, turn_clear=turn_clear)

        candidate = _read_position(point_hit)
        if candidate is None or point_walkable is not True:
            return self._set_idle("Point at a clear floor surface")
        distance, heading = self._relative_target(candidate, position, yaw)
        if not math.isfinite(distance) or not self.min_target_distance_m <= distance <= self.max_target_distance_m:
            return self._set_idle(
                f"Select a floor spot {self.min_target_distance_m:g} to {self.max_target_distance_m:g} meters away"
            )
        if abs(heading) > self.max_heading_rad and not self.allow_turn_in_place:
            return self._set_idle("Select a floor spot in front of the robot")
        if path_clear is not True and not self.allow_routing:
            return self._set_idle("Choose a floor spot with a clear route")
        if self.state != "acquiring" or math.dist(candidate, self._acquisition_origin) > self._acquisition_radius_m:
            self._acquisition_start_s = now
            self._acquisition_origin = candidate
            # Freeze both the initial hit and its allowance for this dwell. Ray
            # drift cannot enlarge its own tolerance or drag the anchor along.
            self._acquisition_radius_m = max(
                self.dwell_radius_m, min(self.max_dwell_radius_m, self.dwell_radius_distance_scale * distance)
            )
        elapsed = now - self._acquisition_start_s
        if elapsed + 1e-12 < self.dwell_s:
            self._command = NavigationCommand(
                state="acquiring",
                candidate=candidate,
                acquisition_progress=min(1.0, elapsed / self.dwell_s),
                detail="Hold pointing at this floor spot",
            )
            return self._command
        if self.allow_routing and path_clear is not True:
            self._command = NavigationCommand(
                state="planning",
                target=candidate,
                acquisition_progress=1.0,
                detail="Finding a clear floor route; keep pointing, or open your hand to cancel",
            )
            return self._command
        self._command = NavigationCommand(state="active", target=candidate, acquisition_progress=1.0)
        return self._approach(position, yaw, turn_clear=turn_clear)

    def _set_idle(self, detail: str) -> NavigationCommand:
        """Clear selection while retaining the clock for duplicate-frame detection."""
        self._acquisition_origin = None
        self._acquisition_start_s = None
        self._acquisition_radius_m = self.dwell_radius_m
        self._route = ()
        self._route_index = 0
        self._turning = False
        self._turn_progress_time_s = None
        self._turn_progress_heading_rad = None
        self._command = NavigationCommand(state="idle", detail=detail)
        return self._command

    def _interrupt(self, detail: str) -> NavigationCommand:
        """Discard interrupted motion and require an explicit measured release."""
        self.reset()
        self._command = NavigationCommand(detail=detail)
        return self._command

    def _stop_at_target(self, state: NavigationState, detail: str) -> NavigationCommand:
        """Keep the destination marker but latch a zero request until release."""
        self._turn_progress_time_s = None
        self._turn_progress_heading_rad = None
        self._command = NavigationCommand(state=state, target=self.target, detail=detail)
        return self._command

    def _relative_target(self, target: Position, position: Position, yaw: float) -> tuple[float, float]:
        """Return planar distance and wrapped heading error to a world target."""
        dx, dy = target[0] - position[0], target[1] - position[1]
        heading = math.atan2(dy, dx) - yaw
        return math.hypot(dx, dy), math.atan2(math.sin(heading), math.cos(heading))

    def _approach(self, position: Position, yaw: float, *, turn_clear: bool = False) -> NavigationCommand:
        """Follow validated corners, stepping to turn before large heading changes."""
        distance, heading = self._relative_target(self.waypoint, position, yaw)
        if not math.isfinite(distance):
            return self._interrupt("Invalid destination distance; open your hand to rearm")
        final = self.is_final_waypoint
        radius = self.arrival_radius_m if final else self.waypoint_radius_m
        if distance <= radius and not final:
            self._route_index += 1
            self._turn_progress_time_s = None
            self._turn_progress_heading_rad = None
            # A corner is never cut using the old segment's clearance result.
            # A zero request lets the caller validate the next segment first.
            self._command = NavigationCommand(
                state="active", target=self.target, detail="Checking the next route segment"
            )
            return self._command
        if distance <= radius:
            return self._stop_at_target("arrived", "Destination reached; open your hand before choosing another")
        if abs(heading) > self.max_heading_rad and not self.allow_turn_in_place:
            return self._stop_at_target("blocked", "Destination is no longer ahead; open your hand and select again")
        # Hysteresis avoids switching between a stepping turn and forward walking
        # on every noisy heading sample. Turning still uses the physical actor.
        self._turning = self.allow_turn_in_place and abs(heading) > (0.25 if self._turning else 0.55)
        turn = (
            math.copysign(self.stepping_turn_speed, heading)
            if self._turning
            else max(-self.max_yaw_speed, min(self.max_yaw_speed, self.yaw_gain * heading))
        )
        # Forward steering also rotates the body outside a corridor aligned to
        # the destination. Enforce the freshly checked turning footprint for
        # meaningful yaw commands, not only for a zero-forward pure pivot.
        if self.allow_turn_in_place and abs(turn) > 0.01 and turn_clear is not True:
            return self._stop_at_target(
                "blocked", "Insufficient body clearance to turn; open your hand and select again"
            )
        if self._turning:
            magnitude = abs(heading)
            if (
                self._turn_progress_time_s is None
                or self._turn_progress_heading_rad - magnitude >= self.turn_progress_rad
            ):
                self._turn_progress_time_s = self._last_time_s
                self._turn_progress_heading_rad = magnitude
            elif self._last_time_s - self._turn_progress_time_s >= self.turn_progress_timeout_s:
                return self._stop_at_target(
                    "blocked", "Stepping turn made no heading progress; open your hand and select again"
                )
            # This is a measured policy-driven stepping turn, not a fixed-position
            # pivot. The caller rechecks the body footprint and route from the
            # current measured position on every callback, including any drift.
            self._command = NavigationCommand(
                state="active",
                velocity=(0.0, 0.0, turn),
                target=self.target,
                acquisition_progress=1.0,
                detail="Stepping to turn toward the route; open your left hand to stop",
            )
            return self._command
        self._turn_progress_time_s = None
        self._turn_progress_heading_rad = None
        fraction = min(1.0, (distance - radius) / (self.slow_radius_m - radius))
        # Preserve the actor's walking threshold even for slow approaches or
        # gentle curves, avoiding a second hidden no-motion region.
        forward = max(self.min_forward_speed, self.max_forward_speed * fraction * math.cos(heading) ** 2)
        self._command = NavigationCommand(
            state="active",
            velocity=(forward, 0.0, turn),
            target=self.target,
            acquisition_progress=1.0,
            detail="Walking to the selected floor spot; open your left hand to stop",
        )
        return self._command


# Preserve imports from earlier checkouts; this alias uses the pointing-only API.
PointGazeNavigation = PointNavigation
