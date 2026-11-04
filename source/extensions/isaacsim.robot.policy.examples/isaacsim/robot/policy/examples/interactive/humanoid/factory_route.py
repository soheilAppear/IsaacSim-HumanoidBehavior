# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, incremental flat-floor routing using caller-owned collision queries.

This module never controls the robot. Each ``advance`` executes a limited number
of scene queries; the caller schedules it at a modest rate while locomotion is
stopped. A weighted A* grid finds detours and collision-checked string pulling
removes needless corners. Every edge and shortcut requires real floor samples,
body-width clearance, and enough clearance at corners for a physical turn.
Moving obstacles still require fresh segment checks while following the result.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Generator, Sequence
from numbers import Real

Position = tuple[float, float, float]


def _position(value: object) -> Position | None:
    """Read finite coordinates without accepting booleans or string coercion."""
    try:
        if len(value) != 3 or any(isinstance(axis, bool) or not isinstance(axis, Real) for axis in value):
            return None
        result = tuple(float(axis) for axis in value)
        return result if all(math.isfinite(axis) for axis in result) else None
    except (TypeError, ValueError, OverflowError):
        return None


class FactoryRoutePlanner:
    """Plan a flat-floor route without blocking a physics callback.

    Args:
        start: Current robot world position; its height is not a floor assumption.
        target: Independently raycast and validated world-space floor target.
        floor_at: Query returning a floor position at the supplied XY, or None.
        segment_clear: Fresh body-width collision check for the entire segment.
        turn_clear: Fresh full-turn footprint collision check at a point.
        corner_clear: Optional larger stepping-drift reserve at the start and
            actual intermediate corners. It is not required at every grid node.
        resolution_m: Grid spacing; floor is additionally sampled between nodes.
        max_nodes: Hard limit on expanded cell/heading states, including dead ends.
        margin_m: Search distance outside the rectangle joining start and goal.

    Attributes:
        state: ``planning``, ``ready``, or ``failed``.
        route: Validated corners excluding start, ending at the exact floor goal.
        detail: Progress or terminal explanation suitable for operator feedback.
        queries: Total number of caller scene-query invocations.
        expanded: Total expanded search nodes.
    """

    def __init__(
        self,
        start: Sequence[float],
        target: Sequence[float],
        floor_at: Callable[[Position], Position | None],
        segment_clear: Callable[[Position, Position], bool],
        turn_clear: Callable[[Position], bool],
        *,
        corner_clear: Callable[[Position], bool] | None = None,
        resolution_m: float = 0.6,
        max_nodes: int = 6000,
        margin_m: float = 12.0,
    ) -> None:
        self.start, self.target = _position(start), _position(target)
        if self.start is None or self.target is None:
            raise ValueError("Route endpoints must be finite 3D coordinates")
        if not all(callable(callback) for callback in (floor_at, segment_clear, turn_clear)):
            raise ValueError("Route planning requires real floor and collision query callbacks")
        if corner_clear is not None and not callable(corner_clear):
            raise ValueError("Route corner clearance must be a query callback")
        if (
            isinstance(resolution_m, bool)
            or not isinstance(resolution_m, Real)
            or not math.isfinite(resolution_m)
            or not 0.2 <= resolution_m <= 1.0
            or isinstance(max_nodes, bool)
            or not isinstance(max_nodes, int)
            or not 1 <= max_nodes <= 50000
            or isinstance(margin_m, bool)
            or not isinstance(margin_m, Real)
            or not math.isfinite(margin_m)
            or not 1.0 <= margin_m <= 30.0
        ):
            raise ValueError("Invalid bounded route-planning limits")
        self.resolution_m, self.max_nodes, self.margin_m = float(resolution_m), max_nodes, float(margin_m)
        self._floor_at, self._segment_clear, self._turn_clear = floor_at, segment_clear, turn_clear
        self._corner_clear = corner_clear if corner_clear is not None else turn_clear
        self.state = "planning"
        self.route: tuple[Position, ...] = ()
        self.detail = "Finding a collision-checked route"
        self.queries = 0
        self.expanded = 0
        self._work = self._plan()

    def advance(self, max_queries: int = 24) -> str:
        """Run at most ``max_queries`` callbacks; return the current planner state.

        Query failures fail closed. A cancelled planner can simply be discarded;
        this generator owns no native resources or background threads.
        """
        if isinstance(max_queries, bool) or not isinstance(max_queries, int) or not 1 <= max_queries <= 256:
            raise ValueError("Query budget must be an integer between 1 and 256")
        if self.state != "planning":
            return self.state
        try:
            for unused in range(max_queries):
                next(self._work)
        except StopIteration:
            if self.state == "planning":
                self.state = "failed"
                self.detail = "No clear floor route within the bounded search area"
        except Exception as error:
            self.state = "failed"
            self.route = ()
            self.detail = f"Route query failed: {error}"
        return self.state

    def _query(self, callback: Callable, *args: object) -> Generator[None, None, object]:
        """Yield after exactly one callback, allowing a strict per-update budget."""
        result = callback(*args)
        self.queries += 1
        yield
        return result

    def _floor(self, point: Position) -> Generator[None, None, Position | None]:
        """Verify floor height and a support patch within the shared query budget."""
        result = _position((yield from self._query(self._floor_at, point)))
        if result is None or math.hypot(result[0] - point[0], result[1] - point[1]) > 0.05:
            return None
        if abs(result[2] - self.target[2]) > 0.08:
            return None
        # A center ray alone permits a shortcut to skim a hole between samples.
        # Check a small support patch at every ≤0.3 m centerline sample. These
        # individual queries consume the same strict incremental budget.
        for dx, dy in ((0.16, 0.16), (0.16, -0.16), (-0.16, 0.16), (-0.16, -0.16)):
            support = (point[0] + dx, point[1] + dy, point[2])
            floor = _position((yield from self._query(self._floor_at, support)))
            if (
                floor is None
                or math.hypot(floor[0] - support[0], floor[1] - support[1]) > 0.05
                or abs(floor[2] - self.target[2]) > 0.08
            ):
                return None
        return result

    def _edge(self, first: Position, last: Position) -> Generator[None, None, bool]:
        """Check collision first, then sample floor at ≤0.3 m along the centerline."""
        if (yield from self._query(self._segment_clear, first, last)) is not True:
            return False
        length = math.hypot(last[0] - first[0], last[1] - first[1])
        steps = max(1, math.ceil(length / 0.3))
        for index in range(1, steps):
            fraction = index / steps
            point = tuple(first[axis] + fraction * (last[axis] - first[axis]) for axis in range(3))
            if (yield from self._floor(point)) is None:
                return False
        return True

    def _plan(self) -> Generator[None, None, None]:
        """Find a verified direct route or search heading-aware grid alternatives."""
        start = yield from self._floor(self.start)
        goal = yield from self._floor(self.target)
        if start is None or goal is None or math.dist(goal, self.target) > 0.08:
            self.detail = "The start or destination has no verified walking floor"
            self.state = "failed"
            return
        # At an already close object approach the goal is a body-center point,
        # not the touched object's surface. Do not waive its body clearance.
        for point in (start, goal):
            if (yield from self._query(self._turn_clear, point)) is not True:
                self.detail = "The start or destination has insufficient body turning clearance"
                self.state = "failed"
                return
        if (yield from self._query(self._corner_clear, start)) is not True:
            self.state, self.detail = "failed", "The start has insufficient stepping-turn drift clearance"
            return
        if (yield from self._edge(start, goal)):
            self.route, self.state, self.detail = (goal,), "ready", "Direct floor route ready"
            return
        spacing = self.resolution_m
        minimum = tuple(min(start[axis], goal[axis]) - self.margin_m for axis in range(2))
        maximum = tuple(max(start[axis], goal[axis]) + self.margin_m for axis in range(2))
        # Heading is part of a search state: a narrow straight passage may be
        # traversable without offering enough space to change direction there.
        origin = (0, 0, -1)
        positions = {(0, 0): start}
        visited = set()
        parents = {}
        costs = {origin: 0.0}
        frontier = [(math.dist(start[:2], goal[:2]), 0.0, origin)]
        neighbors = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        corner_space = {(0, 0): True}

        def corner_allowed(
            key: tuple[int, int, int], point: Position, outgoing: tuple[float, float]
        ) -> Generator[None, None, bool]:
            """Require cached corner clearance when the search changes heading."""
            if key[2] < 0:
                return True
            incoming = neighbors[key[2]]
            if (
                abs(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]) < 1e-8
                and incoming[0] * outgoing[0] + incoming[1] * outgoing[1] >= 0
            ):
                return True
            xy = key[:2]
            if xy not in corner_space:
                corner_space[xy] = (yield from self._query(self._corner_clear, point)) is True
            return corner_space[xy]

        end = None
        while frontier and self.expanded < self.max_nodes:
            unused_score, cost, key = heapq.heappop(frontier)
            if key in visited or cost != costs.get(key):
                continue
            visited.add(key)
            point = positions[key[:2]]
            self.expanded += 1
            self.detail = f"Finding a clear floor route ({self.expanded} route nodes checked)"
            if (
                math.dist(point[:2], goal[:2]) <= spacing * 1.5
                and (yield from corner_allowed(key, point, (goal[0] - point[0], goal[1] - point[1])))
                and (yield from self._edge(point, goal))
            ):
                end = key
                break
            for direction_index, offset in enumerate(neighbors):
                neighbor_xy = key[0] + offset[0], key[1] + offset[1]
                neighbor = (*neighbor_xy, direction_index)
                if neighbor in visited:
                    continue
                next_cost = cost + spacing * math.hypot(*offset)
                if next_cost >= costs.get(neighbor, math.inf):
                    continue
                candidate = (start[0] + spacing * neighbor_xy[0], start[1] + spacing * neighbor_xy[1], goal[2])
                if any(not minimum[axis] <= candidate[axis] <= maximum[axis] for axis in range(2)):
                    continue
                if not (yield from corner_allowed(key, point, offset)):
                    continue
                if neighbor_xy not in positions:
                    floor = yield from self._floor(candidate)
                    if floor is None or (yield from self._query(self._turn_clear, floor)) is not True:
                        positions[neighbor_xy] = None
                    else:
                        positions[neighbor_xy] = floor
                candidate = positions[neighbor_xy]
                if candidate is None or not (yield from self._edge(point, candidate)):
                    continue
                costs[neighbor], parents[neighbor] = next_cost, key
                # A modest weighted heuristic bounds practical search time in
                # warehouses. The route need not be globally shortest to be safe.
                score = next_cost + 1.15 * math.dist(candidate[:2], goal[:2])
                heapq.heappush(frontier, (score, next_cost, neighbor))
        if end is None:
            self.state = "failed"
            self.detail = "No clear floor route within the bounded search area or cell budget"
            return
        raw = [goal]
        while end != origin:
            raw.append(positions[end[:2]])
            end = parents[end]
        raw.append(start)
        raw.reverse()
        route = []
        index = 0
        while index < len(raw) - 1:
            following = len(raw) - 1
            while following > index + 1:
                corner_ok = (
                    following == len(raw) - 1 or (yield from self._query(self._corner_clear, raw[following])) is True
                )
                if corner_ok and (yield from self._edge(raw[index], raw[following])):
                    break
                following -= 1
            # Even a formerly clear adjacent edge is rechecked. Obstacles may
            # have moved while the incremental search was running.
            if following == index + 1:
                if (
                    following != len(raw) - 1
                    and (yield from self._query(self._corner_clear, raw[following])) is not True
                ):
                    self.state, self.detail = "failed", "A route corner has insufficient stepping-turn drift clearance"
                    return
                if not (yield from self._edge(raw[index], raw[following])):
                    self.state, self.detail = "failed", "The route changed while planning; select again"
                    return
            route.append(raw[following])
            index = following
        self.route, self.state, self.detail = tuple(route), "ready", f"Clear floor route ready ({len(route)} segments)"
