# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise floor selection, conservative clearance, and real USD pointing visuals."""

from __future__ import annotations

import math
import types
import unittest
from unittest.mock import patch

from humanoid_test_support import CARB, PHYSICS, USD, load_source
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

SCENE = load_source("interactive/humanoid/point_navigation_scene.py")


class TestPointNavigationSceneControls(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = Usd.Stage.CreateInMemory()
        USD.get_context.return_value = types.SimpleNamespace(get_stage=lambda: self.stage)
        floor = UsdGeom.Plane.Define(self.stage, "/World/Floor")
        floor.CreateAxisAttr("Z")
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        box = UsdGeom.Cube.Define(self.stage, "/World/Box")
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
        self.scene = SCENE.PointNavigationScene("/World/G1")
        self.float3 = patch.object(CARB, "Float3", lambda *values: tuple(values), create=True)
        self.float4 = patch.object(CARB, "Float4", lambda *values: tuple(values), create=True)
        self.float3.start()
        self.float4.start()
        self.addCleanup(self.float3.stop)
        self.addCleanup(self.float4.stop)

    @staticmethod
    def hit(path: str, distance: float, position: tuple = (1, 0, 0), normal: tuple = (0, 0, 1)) -> dict:
        return {"collision": path, "distance": distance, "position": position, "normal": normal}

    def ray_hits(self, hits: list) -> None:
        def raycast(origin, direction, maximum, callback, both_sides):
            self.assertAlmostEqual(Gf.Vec3d(*direction).GetLength(), 1.0)
            self.assertFalse(both_sides)
            for hit in hits:
                self.assertTrue(callback(hit))

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(raycast_all=raycast)

    def test_nearest_hit_ignores_robot_and_does_not_select_floor_through_a_box(self) -> None:
        self.ray_hits(
            [
                self.hit("/World/Floor", 3),
                self.hit("/World/G1/left_hand", 0.01),
                self.hit("/World/Box", 1, (1, 0, 0.2)),
            ]
        )
        result = self.scene.cast_ray((0, 0, 1), (2, 0, -1))
        self.assertEqual(result["path"], "/World/Box")
        self.assertFalse(result["walkable"])
        self.assertEqual(result["distance"], 1)

    def test_floor_requires_static_collision_ancestry_height_and_upward_normal(self) -> None:
        mesh = UsdGeom.Mesh.Define(self.stage, "/World/Floor/Mesh")
        self.assertTrue(self.scene.is_floor_point(str(mesh.GetPath()), (1, 0, 0.01)))
        self.assertFalse(self.scene.is_floor_point("/World/Box", (1, 0, 0)))
        self.assertFalse(self.scene.is_floor_point("/World/Floor", (1, 0, 0.5)))
        self.assertFalse(self.scene.is_floor_point("/World/NotPresent", (1, 0, 0)))
        UsdGeom.Mesh.Define(self.stage, "/World/Box/Mesh")
        self.assertFalse(self.scene.is_floor_point("/World/Box/Mesh", (1, 0, 0)))
        self.ray_hits([self.hit("/World/Floor", 2)])
        self.assertTrue(self.scene.cast_ray((0, 0, 1), (1, 0, -1))["walkable"])
        self.ray_hits([self.hit("/World/Floor", 2, normal=(1, 0, 0))])
        self.assertFalse(self.scene.cast_ray((0, 0, 1), (1, 0, -1))["walkable"])

    def test_invalid_or_failed_ray_queries_fail_closed(self) -> None:
        self.ray_hits([self.hit("/World/Floor", 2), self.hit("/World/Box", math.nan)])
        self.assertIsNone(self.scene.cast_ray((0, 0, 1), (1, 0, -1)))
        self.assertIsNotNone(self.scene.last_error)
        self.assertIsNone(self.scene.cast_ray((math.nan, 0, 1), (1, 0, -1)))
        self.assertIsNone(self.scene.cast_ray((0, 0, 1), (0, 0, 0)))
        PHYSICS.get_physics_scene_query_interface.side_effect = RuntimeError("query unavailable")
        try:
            self.assertIsNone(self.scene.cast_ray((0, 0, 1), (1, 0, -1)))
            self.assertIn("unavailable", self.scene.last_error)
        finally:
            PHYSICS.get_physics_scene_query_interface.side_effect = None

    def test_distant_floor_collision_is_not_limited_by_navigation_selection_range(self) -> None:
        # Selection distance is controlled separately. A real floor collision
        # beyond the former four-meter destination limit still ends the blue ray.
        self.ray_hits([self.hit("/World/Floor", 12.1, (12, 0, 0))])
        result = self.scene.cast_ray((0, 0, 1), (12, 0, -1))
        self.assertEqual(result["position"], Gf.Vec3d(12, 0, 0))
        self.assertTrue(result["walkable"])
        self.assertIsNone(self.scene.last_error)

    def test_missing_floor_is_not_replaced_with_a_synthetic_collision(self) -> None:
        self.ray_hits([self.hit("/World/G1/left_hand", 0.01)])
        self.assertIsNone(self.scene.cast_ray((0, 0, 1), (1, 0, 0)))
        self.assertIn("No collider within 60 meters", self.scene.last_error)

    def test_nonwalkable_hit_explains_height_normal_and_movable_object(self) -> None:
        for hit, expected in (
            (self.hit("/World/Box", 1, (1, 0, 0)), "movable object"),
            (self.hit("/World/Floor", 1, (1, 0, 0.5)), "above or below"),
            (self.hit("/World/Floor", 1, normal=(1, 0, 0)), "steep or downward-facing"),
        ):
            with self.subTest(expected=expected):
                self.ray_hits([hit, self.hit("/World/Floor", 10, (10, 0, 0))])
                result = self.scene.cast_ray((0, 0, 1), (10, 0, -1))
                self.assertEqual(result["distance"], 1)
                self.assertFalse(result["walkable"])
                self.assertIn(expected, result["detail"])
                self.assertEqual(self.scene.last_error, result["detail"])

    def test_corridor_volume_has_width_and_ignores_only_robot(self) -> None:
        hits = [{"rigid_body": "/World/G1/left_ankle"}]

        def overlap(extents, center, rotation, callback):
            self.assertEqual(extents, (1.38, 0.38, 0.75))
            self.assertEqual(center, (1.0, 0.0, 0.81))
            self.assertEqual(rotation, (0, 0, 0, 1))
            for hit in hits:
                self.assertTrue(callback(hit))

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(overlap_box=overlap)
        self.assertTrue(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0)))
        hits.append({"collision": "/World/Box"})
        self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0)))
        self.assertIn("obstacle", self.scene.last_error)
        hits[:] = [{}]
        self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0)))

    def test_clearance_errors_and_nonfinite_coordinates_block_motion(self) -> None:
        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace()
        self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0)))
        self.assertIn("unavailable", self.scene.last_error)
        self.assertFalse(self.scene.is_path_clear((math.inf, 0, 0.8), (2, 0, 0)))
        self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 1)))

    def test_stopping_shortfall_removes_only_unused_endpoint_space_and_keeps_body_width(self) -> None:
        # Axis-aligned synthetic colliders exercise the actual overlap volume,
        # rather than returning a hit irrespective of the query coordinates.
        obstacle = [2.20, 2.40, -0.10, 0.10]
        queries = []

        def overlap(extents, center, rotation, callback):
            self.assertEqual(rotation, (0, 0, 0, 1))
            queries.append((extents, center))
            x_min, x_max, y_min, y_max = obstacle
            if (
                center[0] - extents[0] <= x_max
                and x_min <= center[0] + extents[0]
                and center[1] - extents[1] <= y_max
                and y_min <= center[1] + extents[1]
            ):
                callback({"collision": "/World/Box"})

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(overlap_box=overlap)
        self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0)))
        self.assertTrue(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0), stopping_shortfall_m=0.20))
        self.assertEqual(queries[-1][0], (1.28, 0.38, 0.75))
        self.assertEqual(queries[-1][1], (0.9, 0, 0.81))
        self.assertEqual(self.scene.last_blocking_paths, ())

        # This box enters the retained endpoint/body margin and must still block.
        obstacle[:] = [2.17, 2.40, -0.10, 0.10]
        self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0), stopping_shortfall_m=0.20))
        # The lateral clearance is unchanged even though the route is shorter.
        obstacle[:] = [1.5, 1.7, 0.37, 0.45]
        self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0), stopping_shortfall_m=0.20))
        obstacle[:] = [1.5, 1.7, 0.39, 0.45]
        self.assertTrue(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0), stopping_shortfall_m=0.20))

    def test_shortfall_never_removes_body_clearance_when_remaining_route_is_zero(self) -> None:
        def overlap(extents, center, rotation, callback):
            self.assertEqual(extents, (0.38, 0.38, 0.75))
            self.assertEqual(center, (0.0, 0.0, 0.81))
            self.assertEqual(rotation, (0, 0, 0, 1))
            callback({"collision": "/World/Box"})

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(overlap_box=overlap)
        self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (0.1, 0, 0), stopping_shortfall_m=0.20))
        self.assertEqual(self.scene.last_blocking_paths, ("/World/Box",))

    def test_invalid_shortfall_fails_closed_before_native_query(self) -> None:
        calls = []
        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(
            overlap_box=lambda *args: calls.append(args)
        )
        for value in (-0.1, math.nan, math.inf, None, True, "invalid"):
            with self.subTest(value=value):
                self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0), stopping_shortfall_m=value))
                self.assertIn("finite and nonnegative", self.scene.last_error)
                self.assertEqual(self.scene.last_blocking_paths, ())
        self.assertEqual(calls, [])

    def test_blocker_diagnostics_are_unique_bounded_and_cleared_by_next_query(self) -> None:
        paths = ["/World/G1/ankle", "/World/Box0", "/World/Box0", *[f"/World/Box{i}" for i in range(1, 8)]]

        def overlap(extents, center, rotation, callback):
            for path in paths:
                callback({"collision": path} if path is not None else {})

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(overlap_box=overlap)
        self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0)))
        self.assertEqual(self.scene.last_blocking_paths, tuple(f"/World/Box{i}" for i in range(4)))
        self.assertIn("/World/Box0", self.scene.last_error)
        paths[:] = [None]
        self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0)))
        self.assertEqual(self.scene.last_blocking_paths, ("<unknown collider>",))
        paths.clear()
        self.assertTrue(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0)))
        self.assertEqual(self.scene.last_blocking_paths, ())
        paths[:] = ["/World/Box"]
        self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0)))
        self.ray_hits([self.hit("/World/Floor", 2)])
        self.scene.cast_ray((0, 0, 1), (1, 0, -1))
        self.assertEqual(self.scene.last_blocking_paths, ())

    @staticmethod
    def object_hit(path="/World/Box/Mesh", position=(2.0, 0.0, 0.4)) -> dict:
        return {"path": path, "position": position, "distance": 2.0, "walkable": False}

    def test_registered_object_approach_uses_separate_real_floor_and_body_queries(self) -> None:
        rays = []
        corridors = []

        def raycast(origin, direction, maximum, callback, both_sides):
            rays.append((origin, direction, maximum))
            self.assertEqual(direction, (0, 0, -1))
            self.assertEqual(maximum, 60.0)
            self.assertFalse(both_sides)
            callback(self.hit("/World/Floor", origin[2], (origin[0], origin[1], 0)))

        def overlap(extents, center, rotation, callback):
            corridors.append((extents, center))

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(
            raycast_all=raycast, overlap_box=overlap
        )
        # A registered tabletop object can be approached, but is never itself
        # classified as floor. The original hit must stay unchanged for drawing.
        source = self.object_hit(position=(2, 0, 1.4))
        original = dict(source)
        result = self.scene.resolve_object_approach((0, 0, 0.8), source, ["/World/Box"])
        self.assertEqual(result["target"], (1.7, 0, 0))
        self.assertEqual(result["source_object_root"], "/World/Box")
        self.assertEqual(result["surface_point"], (2, 0, 1.4))
        self.assertEqual(result["floor_path"], "/World/Floor")
        self.assertEqual(result["standoff_m"], 0.3)
        self.assertEqual(source, original)
        self.assertEqual(len(rays), 1)
        self.assertAlmostEqual(rays[0][0][2], 1.9)
        self.assertEqual(len(corridors), 1)
        self.assertAlmostEqual(corridors[0][0][0], 1.13)
        self.assertEqual(corridors[0][0][1:], (0.38, 0.75))
        self.assertIsNone(self.scene.last_error)

    def test_object_approach_retreats_to_a_clear_standing_location_without_ignoring_nearer_obstacle(self) -> None:
        ray_x = []
        blocked = []

        def raycast(origin, direction, maximum, callback, both_sides):
            ray_x.append(origin[0])
            callback(self.hit("/World/Floor", origin[2], (origin[0], origin[1], 0)))

        def overlap(extents, center, rotation, callback):
            if center[0] + extents[0] >= 1.60:
                blocked.append(True)
                callback({"collision": "/World/NearerObstacle"})

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(
            raycast_all=raycast, overlap_box=overlap
        )
        result = self.scene.resolve_object_approach((0, 0, 0.8), self.object_hit(), ["/World/Box"])
        self.assertEqual(len(blocked), 2)
        self.assertEqual(len(ray_x), 3)
        self.assertAlmostEqual(result["target"][0], 1.4)
        self.assertAlmostEqual(result["standoff_m"], 0.6)
        self.assertEqual(self.scene.last_blocking_paths, ())

    def test_object_approach_does_not_look_through_an_object_to_the_floor(self) -> None:
        queries = []
        corridors = []

        def raycast(origin, direction, maximum, callback, both_sides):
            queries.append(origin)
            callback(self.hit("/World/Floor", origin[2], (origin[0], origin[1], 0)))
            callback(self.hit("/World/Box", origin[2] - 0.1, (origin[0], origin[1], 0.1)))

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(
            raycast_all=raycast, overlap_box=lambda *args: corridors.append(args)
        )
        self.assertIsNone(self.scene.resolve_object_approach((0, 0, 0.8), self.object_hit(), ["/World/Box"]))
        self.assertEqual(len(queries), 7)
        self.assertEqual(corridors, [])
        self.assertIn("No clear standing spot", self.scene.last_error)

    def test_object_approach_all_routes_blocked_is_bounded_and_reports_obstacle(self) -> None:
        counts = [0, 0]

        def raycast(origin, direction, maximum, callback, both_sides):
            counts[0] += 1
            callback(self.hit("/World/Floor", origin[2], (origin[0], origin[1], 0)))

        def overlap(extents, center, rotation, callback):
            counts[1] += 1
            callback({"collision": "/World/NearerObstacle"})

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(
            raycast_all=raycast, overlap_box=overlap
        )
        self.assertIsNone(self.scene.resolve_object_approach((0, 0, 0.8), self.object_hit(), ["/World/Box"]))
        self.assertEqual(counts, [7, 7])
        self.assertEqual(self.scene.last_blocking_paths, ("/World/NearerObstacle",))
        self.assertIn("/World/NearerObstacle", self.scene.last_error)

    def test_object_approach_rejects_scenery_prefix_collisions_and_malformed_inputs_without_queries(self) -> None:
        queries = []
        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(
            raycast_all=lambda *args: queries.append(args), overlap_box=lambda *args: queries.append(args)
        )
        for position, hit, roots, shortfall in (
            ((0, 0, 0.8), self.object_hit(path="/World/Table"), ["/World/Box"], 0.2),
            ((0, 0, 0.8), self.object_hit(path="/World/Box12"), ["/World/Box"], 0.2),
            ((0, 0, 0.8), self.object_hit(), ["/"], 0.2),
            ((0, 0, 0.8), self.object_hit(), "/World/Box", 0.2),
            ((0, 0, 0.8), self.object_hit(), None, 0.2),
            ((math.nan, 0, 0.8), self.object_hit(), ["/World/Box"], 0.2),
            ((0, 0, 0.8), self.object_hit(position=(math.nan, 0, 0)), ["/World/Box"], 0.2),
            ((0, 0, 0.8), self.object_hit(position=(0.7, 0, 0)), ["/World/Box"], 0.2),
            ((0, 0, 0.8), None, ["/World/Box"], 0.2),
            ((0, 0, 0.8), {**self.object_hit(), "distance": math.nan}, ["/World/Box"], 0.2),
            ((0, 0, 0.8), self.object_hit(), ["/World/Box"], math.nan),
            ((0, 0, 0.8), self.object_hit(), ["/World/Box"], -0.1),
        ):
            with self.subTest(position=position, hit=hit, roots=roots, shortfall=shortfall):
                self.assertIsNone(
                    self.scene.resolve_object_approach(position, hit, roots, stopping_shortfall_m=shortfall)
                )
                self.assertIsNotNone(self.scene.last_error)
        self.assertEqual(queries, [])

    def test_object_approach_rejects_misaligned_floor_hits_and_standing_too_close(self) -> None:
        corridors = []

        def displaced_floor(origin, direction, maximum, callback, both_sides):
            callback(self.hit("/World/Floor", origin[2], (origin[0] + 0.051, origin[1], 0)))

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(
            raycast_all=displaced_floor, overlap_box=lambda *args: corridors.append(args)
        )
        self.assertIsNone(self.scene.resolve_object_approach((0, 0, 0.8), self.object_hit(), ["/World/Box"]))
        self.assertIn("did not match", self.scene.last_error)

        def too_close_floor(origin, direction, maximum, callback, both_sides):
            callback(self.hit("/World/Floor", origin[2], (origin[0] - 0.02, origin[1], 0)))

        PHYSICS.get_physics_scene_query_interface.return_value.raycast_all = too_close_floor
        self.assertIsNone(
            self.scene.resolve_object_approach((0, 0, 0.8), self.object_hit(position=(0.8, 0, 0.4)), ["/World/Box"])
        )
        self.assertIn("too close", self.scene.last_error)
        self.assertEqual(corridors, [])

    def test_turning_clearance_uses_circumscribed_body_radius_and_unknown_hits_block(self) -> None:
        hits = []

        def overlap(extents, center, rotation, callback):
            self.assertAlmostEqual(extents[0], 0.38 * math.sqrt(2))
            self.assertAlmostEqual(extents[1], 0.38 * math.sqrt(2))
            self.assertEqual(center, (2, 3, 0.81))
            self.assertEqual(rotation, (0, 0, 0, 1))
            for hit in hits:
                callback(hit)

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(overlap_box=overlap)
        self.assertTrue(self.scene.is_turn_clear((2, 3, 0.8)))
        hits.append({})
        self.assertFalse(self.scene.is_turn_clear((2, 3, 0.8)))
        self.assertFalse(self.scene.is_turn_clear((math.nan, 0, 0)))

    def test_route_floor_query_needs_aligned_real_nearest_floor(self) -> None:
        self.ray_hits([self.hit("/World/Floor", 1.7, (4, 3, 0))])
        self.assertEqual(self.scene.floor_at((4, 3, 0.8)), (4, 3, 0))
        self.assertIsNone(self.scene.floor_at((8, 3, 0.8)))
        self.ray_hits([self.hit("/World/Box", 1.5, (4, 3, 0.2)), self.hit("/World/Floor", 1.7, (4, 3, 0))])
        self.assertIsNone(self.scene.floor_at((4, 3, 0.8)))
        self.assertIsNone(self.scene.floor_at((math.inf, 0, 0)))

    def test_planned_segments_cover_full_turn_footprint_and_tracking_error(self) -> None:
        volumes = []

        def overlap(extents, center, rotation, callback):
            volumes.append((extents, center))
            # An obstacle 0.60 m beside the centerline clears the old .38 m
            # corridor but enters the new steering-plus-tracking reserve.
            if extents[1] >= 0.60:
                callback({"collision": "/World/TableEdge"})

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(overlap_box=overlap)
        self.assertTrue(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0)))
        self.assertFalse(self.scene.is_route_segment_clear((0, 0, 0.8), (2, 0, 0)))
        width = 0.38 * math.sqrt(2) + 0.15
        self.assertAlmostEqual(volumes[-1][0][0], 1 + width)
        self.assertAlmostEqual(volumes[-1][0][1], width)
        self.assertEqual(volumes[-1][1], (1, 0, 0.81))
        self.assertEqual(self.scene.last_blocking_paths, ("/World/TableEdge",))

    def test_planned_corner_reserves_half_meter_drift_without_changing_runtime_guard(self) -> None:
        widths = []

        def overlap(extents, center, rotation, callback):
            widths.append(extents[0])
            if extents[0] >= 0.9:
                callback({"collision": "/World/Rack"})

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(overlap_box=overlap)
        self.assertTrue(self.scene.is_turn_clear((0, 0, 0.8)))
        self.assertFalse(self.scene.is_route_corner_clear((0, 0, 0.8)))
        self.assertAlmostEqual(widths[0], 0.38 * math.sqrt(2))
        self.assertAlmostEqual(widths[1], 0.38 * math.sqrt(2) + 0.50)

    def test_diagonal_planned_corridor_contains_runtime_square_corners(self) -> None:
        # The obstacle is inside the runtime world-axis turn square at (1,1),
        # but outside an incorrectly unprojected .687m diagonal corridor.
        obstacle = (1.5, 0.5)
        volumes = []

        def overlap(extents, center, rotation, callback):
            volumes.append((extents, center, rotation))
            yaw = 2 * math.atan2(rotation[2], rotation[3])
            dx, dy = obstacle[0] - center[0], obstacle[1] - center[1]
            along = dx * math.cos(yaw) + dy * math.sin(yaw)
            across = -dx * math.sin(yaw) + dy * math.cos(yaw)
            if abs(along) <= extents[0] and abs(across) <= extents[1]:
                callback({"collision": "/World/TableCorner"})

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(overlap_box=overlap)
        self.assertFalse(self.scene.is_turn_clear((1, 1, 0.8)))
        self.assertTrue(
            self.scene.is_path_clear((0, 0, 0.8), (2, 2, 0), turning_clearance=True, clearance_margin_m=0.15)
        )
        self.assertFalse(self.scene.is_route_segment_clear((0, 0, 0.8), (2, 2, 0)))
        self.assertAlmostEqual(volumes[-1][0][1], 0.91)
        self.assertAlmostEqual(volumes[-1][0][0], math.sqrt(2) + 0.91)
        self.assertEqual(self.scene.last_blocking_paths, ("/World/TableCorner",))

    def test_every_planned_heading_encloses_runtime_square_and_tracking_offset(self) -> None:
        volumes = []
        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(
            overlap_box=lambda extents, center, rotation, callback: volumes.append((extents, center, rotation))
        )
        radius = 0.38 * math.sqrt(2)
        for degrees in range(-180, 180, 15):
            angle = math.radians(degrees)
            forward = (math.cos(angle), math.sin(angle))
            side = (-forward[1], forward[0])
            target = (4 * forward[0], 4 * forward[1], 0)
            self.assertTrue(self.scene.is_route_segment_clear((0, 0, 0.8), target))
            extents, center, rotation = volumes[-1]
            # Check start, middle, and end, including both signs of a 15cm
            # transverse tracking error and all four runtime square corners.
            for progress in (0, 0.5, 1):
                for tracking in (-0.15, 0.15):
                    for ox, oy in ((-radius, -radius), (-radius, radius), (radius, -radius), (radius, radius)):
                        px = target[0] * progress + tracking * side[0] + ox - center[0]
                        py = target[1] * progress + tracking * side[1] + oy - center[1]
                        self.assertLessEqual(abs(px * forward[0] + py * forward[1]), extents[0] + 1e-9)
                        self.assertLessEqual(abs(px * side[0] + py * side[1]), extents[1] + 1e-9)

    def test_invalid_extra_clearance_margin_never_reaches_native_query(self) -> None:
        calls = []
        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(
            overlap_box=lambda *args: calls.append(args)
        )
        for margin in (-0.1, math.nan, math.inf, True, None, "invalid"):
            self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), (2, 0, 0), clearance_margin_m=margin))
            self.assertIn("clearance margin", self.scene.last_error)
        self.assertEqual(calls, [])

    def test_routed_object_approach_checks_standing_clearance_without_demanding_direct_path(self) -> None:
        corridors = []

        def raycast(origin, direction, maximum, callback, both_sides):
            callback(self.hit("/World/Floor", origin[2], (origin[0], origin[1], 0)))

        def overlap(extents, center, rotation, callback):
            corridors.append((extents, center))
            # Direct corridor intersects this unrelated obstacle at x=.5;
            # a valid standing node at x=1.7 can be approached by a detour.
            if center[0] - extents[0] < 0.6:
                callback({"collision": "/World/BlockingRack"})

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(
            raycast_all=raycast, overlap_box=overlap
        )
        result = self.scene.resolve_object_approach((0, 0, 0.8), self.object_hit(), ["/World/Box"], allow_routing=True)
        self.assertEqual(result["target"], (1.7, 0, 0))
        self.assertEqual(len(corridors), 2)
        self.assertAlmostEqual(corridors[1][0][0], 0.38 * math.sqrt(2))
        self.assertEqual(corridors[1][1][0], 1.7)
        self.assertFalse(self.scene.is_path_clear((0, 0, 0.8), result["target"]))

    def test_routed_object_cannot_relax_standing_collision_or_look_through_object(self) -> None:
        def raycast(origin, direction, maximum, callback, both_sides):
            callback(self.hit("/World/Floor", origin[2], (origin[0], origin[1], 0)))

        def overlap(extents, center, rotation, callback):
            callback({"collision": "/World/Box"})

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(
            raycast_all=raycast, overlap_box=overlap
        )
        self.assertIsNone(
            self.scene.resolve_object_approach((0, 0, 0.8), self.object_hit(), ["/World/Box"], allow_routing=True)
        )
        self.assertIn("/World/Box", self.scene.last_error)

    def test_factory_direct_object_approach_keeps_close_graspable_standoff(self) -> None:
        corridors = []

        def raycast(origin, direction, maximum, callback, both_sides):
            callback(self.hit("/World/Floor", origin[2], (origin[0], origin[1], 0)))

        def overlap(extents, center, rotation, callback):
            corridors.append((extents, center))
            # The object begins at x=2. The .20 m shortened direct corridor
            # ends at x=1.88, while an unnecessary pivot at x=1.7 would overlap.
            if center[0] + extents[0] >= 2:
                callback({"collision": "/World/Box"})

        PHYSICS.get_physics_scene_query_interface.return_value = types.SimpleNamespace(
            raycast_all=raycast, overlap_box=overlap
        )
        result = self.scene.resolve_object_approach((0, 0, 0.8), self.object_hit(), ["/World/Box"], allow_routing=True)
        self.assertEqual(result["target"], (1.7, 0, 0))
        self.assertEqual(result["standoff_m"], 0.3)
        self.assertEqual(len(corridors), 1)
        self.assertFalse(self.scene.is_turn_clear(result["target"]))

    def test_prepared_visuals_are_idempotent_finite_and_noncolliding(self) -> None:
        self.scene.create(self.stage)
        paths = [str(prim.GetPath()) for prim in self.stage.Traverse()]
        self.scene.create(self.stage)
        self.assertEqual(paths, [str(prim.GetPath()) for prim in self.stage.Traverse()])
        self.scene.draw((0, 0, 1), (2, 0, 0), target=(2, 0, 0), progress=0.5, state="acquiring")
        for suffix in ("Ray", "HitMarker", "Target"):
            prim = self.stage.GetPrimAtPath(f"{self.scene.ROOT}/{suffix}")
            self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))
            self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertEqual(len(UsdGeom.Xformable(prim).GetOrderedXformOps()), 1)
            matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
            self.assertTrue(all(math.isfinite(matrix[row][col]) for row in range(4) for col in range(4)))
            self.assertNotEqual(UsdGeom.Imageable(prim).ComputeVisibility(), UsdGeom.Tokens.invisible)
        target = self.stage.GetPrimAtPath(f"{self.scene.ROOT}/Target")
        binding = UsdShade.MaterialBindingAPI(target).GetDirectBindingRel().GetTargets()
        self.assertEqual(str(binding[0]), f"{self.scene.ROOT}/Materials/Cyan")
        self.scene.draw((0, 0, 1), (2, 0, 0), target=(2, 0, 0), progress=math.nan, state="active")
        self.assertEqual(
            str(UsdShade.MaterialBindingAPI(target).GetDirectBindingRel().GetTargets()[0]),
            f"{self.scene.ROOT}/Materials/Green",
        )
        self.assertEqual(paths, [str(prim.GetPath()) for prim in self.stage.Traverse()])

    def test_invalid_draw_hides_and_deleted_stage_never_recreates_during_runtime(self) -> None:
        self.scene.create(self.stage)
        self.scene.draw((0, 0, 1), (1, 0, 0))
        self.scene.draw((math.nan, 0, 1), (1, 0, 0))
        root = self.stage.GetPrimAtPath(self.scene.ROOT)
        self.assertEqual(UsdGeom.Imageable(root).ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.stage.RemovePrim(self.scene.ROOT)
        self.scene.draw((0, 0, 1), (1, 0, 0))
        self.assertFalse(self.stage.GetPrimAtPath(self.scene.ROOT).IsValid())
        self.scene.create(self.stage)
        self.assertTrue(self.stage.GetPrimAtPath(self.scene.ROOT).IsValid())
        self.scene.cleanup()
        self.assertTrue(self.stage.GetPrimAtPath(self.scene.ROOT).IsValid())
        self.scene.create(self.stage)
        self.scene.cleanup(remove=True)
        self.assertFalse(self.stage.GetPrimAtPath(self.scene.ROOT).IsValid())

    def test_ray_miss_hides_only_collision_marker_and_next_hit_restores_it(self) -> None:
        self.scene.create(self.stage)
        self.scene.draw((0, 0, 1), (4, 0, 1), target=(2, 0, 0), state="active", hit_valid=False)
        marker = UsdGeom.Imageable(self.stage.GetPrimAtPath(f"{self.scene.ROOT}/HitMarker"))
        ray = UsdGeom.Imageable(self.stage.GetPrimAtPath(f"{self.scene.ROOT}/Ray"))
        target = UsdGeom.Imageable(self.stage.GetPrimAtPath(f"{self.scene.ROOT}/Target"))
        self.assertEqual(marker.ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.assertNotEqual(ray.ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.assertNotEqual(target.ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.scene.draw((0, 0, 1), (6, 0, 0), hit_valid=True)
        self.assertNotEqual(marker.ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.assertEqual(target.ComputeVisibility(), UsdGeom.Tokens.invisible)


if __name__ == "__main__":
    unittest.main()
