# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise incremental collision-checked routing without Kit or native queries."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/factory_route.py"
)
SPEC = importlib.util.spec_from_file_location("factory_route_under_test", MODULE_PATH)
FACTORY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FACTORY
SPEC.loader.exec_module(FACTORY)


class TestFactoryRouteControls(unittest.TestCase):
    @staticmethod
    def floor(point):
        return point[0], point[1], 0.0

    @staticmethod
    def clear(first, last):
        return True

    @staticmethod
    def turn(point):
        return True

    def planner(self, start=(0, 0, 0.8), target=(6, 0, 0), **overrides):
        arguments = dict(floor_at=self.floor, segment_clear=self.clear, turn_clear=self.turn)
        arguments.update(overrides)
        return FACTORY.FactoryRoutePlanner(start, target, **arguments)

    def finish(self, planner, budget=17):
        for unused in range(10000):
            before = planner.queries
            state = planner.advance(budget)
            self.assertLessEqual(planner.queries - before, budget)
            if state != "planning":
                return state
        self.fail("Bounded planner did not terminate")

    def test_long_clear_route_is_single_segment_and_strictly_query_bounded(self):
        planner = self.planner(target=(48, 0, 0))
        self.assertEqual(self.finish(planner, budget=1), "ready")
        self.assertEqual(planner.route, ((48.0, 0.0, 0.0),))
        self.assertGreater(planner.queries, 160)
        self.assertEqual(planner.expanded, 0)

    def test_body_obstacle_gets_a_clear_detour_and_no_corner_cutting(self):
        # The fake query is already inflated by the robot footprint, matching
        # production overlap-box semantics. Sample densely to test shortcuts.
        def collision(point):
            return 1.6 <= point[0] <= 3.8 and -0.8 <= point[1] <= 0.8

        def segment(first, last):
            count = max(1, math.ceil(math.dist(first[:2], last[:2]) / 0.02))
            return not any(
                collision(tuple(first[axis] + (last[axis] - first[axis]) * index / count for axis in range(3)))
                for index in range(count + 1)
            )

        planner = self.planner(segment_clear=segment, turn_clear=lambda point: not collision(point))
        self.assertEqual(self.finish(planner), "ready")
        self.assertGreater(len(planner.route), 1)
        self.assertLess(len(planner.route), 6)
        previous = (0, 0, 0)
        for point in planner.route:
            self.assertTrue(segment(previous, point))
            previous = point
        self.assertEqual(planner.route[-1], (6, 0, 0))

    def test_reverse_heading_target_is_a_geometry_query_without_forward_cone(self):
        planner = self.planner(target=(-8, 2, 0))
        self.assertEqual(self.finish(planner), "ready")
        self.assertEqual(planner.route[-1], (-8, 2, 0))

    def test_floor_hole_is_not_bridged_by_direct_edge_or_smoothing(self):
        def floor(point):
            if 1.1 < point[0] < 2.5 and abs(point[1]) < 0.65:
                return None
            return self.floor(point)

        planner = self.planner(target=(4, 0, 0), floor_at=floor)
        self.assertEqual(self.finish(planner), "ready")
        self.assertGreater(len(planner.route), 1)
        previous = (0, 0, 0)
        for point in planner.route:
            steps = max(1, math.ceil(math.dist(point[:2], previous[:2]) / 0.03))
            for index in range(steps + 1):
                sample = tuple(previous[axis] + (point[axis] - previous[axis]) * index / steps for axis in range(3))
                self.assertIsNotNone(floor(sample))
            previous = point

    def test_closed_wall_exhausts_bounded_search_without_authorizing_a_route(self):
        planner = self.planner(
            target=(3, 0, 0),
            margin_m=1,
            max_nodes=100,
            segment_clear=lambda first, last: (first[0] < 1 and last[0] < 1) or (first[0] > 2 and last[0] > 2),
        )
        self.assertEqual(self.finish(planner), "failed")
        self.assertEqual(planner.route, ())
        self.assertLessEqual(planner.expanded, 100)

    def test_missing_or_misaligned_start_or_goal_floor_fails_closed(self):
        for floor in (lambda point: None, lambda point: (point[0] + 0.2, point[1], 0), lambda point: (*point[:2], 0.5)):
            planner = self.planner(floor_at=floor)
            self.assertEqual(self.finish(planner), "failed")
            self.assertIn("floor", planner.detail)

    def test_start_or_goal_without_turning_clearance_is_rejected(self):
        for turn in (lambda point: point[0] != 0, lambda point: point[0] != 6):
            planner = self.planner(turn_clear=turn)
            self.assertEqual(self.finish(planner), "failed")
            self.assertEqual(planner.route, ())
            self.assertIn("turning clearance", planner.detail)

    def test_scene_query_exception_is_terminal_and_no_route_is_retained(self):
        def failure(point):
            raise RuntimeError("physics restarting")

        planner = self.planner(floor_at=failure)
        self.assertEqual(self.finish(planner), "failed")
        self.assertIn("physics restarting", planner.detail)
        count = planner.queries
        planner.advance()
        self.assertEqual(planner.queries, count)

    def test_malformed_configuration_and_budget_are_rejected(self):
        for arguments in (
            {"start": (math.nan, 0, 0)},
            {"target": (True, 0, 0)},
            {"max_nodes": True},
            {"max_nodes": 0},
            {"resolution_m": 0.01},
            {"resolution_m": math.nan},
            {"margin_m": math.inf},
            {"floor_at": None},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.planner(**arguments)
        planner = self.planner()
        for budget in (True, 0, 257, 3.5):
            with self.assertRaises(ValueError):
                planner.advance(budget)

    def test_start_requires_stepping_drift_room_but_final_goal_does_not(self):
        planner = self.planner(corner_clear=lambda point: point[0] > 0.1)
        self.assertEqual(self.finish(planner), "failed")
        self.assertIn("start", planner.detail)
        planner = self.planner(corner_clear=lambda point: point[0] < 0.1)
        self.assertEqual(self.finish(planner), "ready")
        self.assertEqual(planner.route[-1], (6, 0, 0))

    def test_search_reserves_drift_at_turns_but_can_cross_a_narrow_straight_section(self):
        # The center corridor excludes the already-inflated body obstacle. Only
        # the ends of this narrow aisle have enough room for a stepping turn.
        def collision(point):
            return 1.6 <= point[0] <= 3.8 and abs(point[1]) <= 0.8

        def segment(first, last):
            count = max(1, math.ceil(math.dist(first[:2], last[:2]) / 0.02))
            return not any(
                collision(tuple(first[axis] + (last[axis] - first[axis]) * index / count for axis in range(3)))
                for index in range(count + 1)
            )

        def corner(point):
            return not (1.1 <= point[0] <= 4.3 and abs(point[1]) <= 1.3)

        def floor(point):
            return self.floor(point) if abs(point[1]) <= 1.4 else None

        planner = self.planner(
            segment_clear=segment, turn_clear=lambda point: not collision(point), corner_clear=corner, floor_at=floor
        )
        self.assertEqual(self.finish(planner), "ready")
        self.assertGreater(len(planner.route), 1)
        self.assertTrue(all(corner(point) for point in planner.route[:-1]))
        previous = (0, 0, 0)
        crosses_narrow_section = False
        for point in planner.route:
            self.assertTrue(segment(previous, point))
            if previous[0] < 1.1 and point[0] > 4.3:
                crosses_narrow_section = True
            previous = point
        self.assertTrue(crosses_narrow_section)

    def test_noncallable_corner_clearance_is_rejected(self):
        with self.assertRaises(ValueError):
            self.planner(corner_clear=True)


if __name__ == "__main__":
    unittest.main()
