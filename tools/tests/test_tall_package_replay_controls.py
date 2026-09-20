# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify bounded handle replay configuration, reach checks, and cleanup."""

from __future__ import annotations

import ast
import asyncio
import json
import math
import re
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import test_contact_replay_controls as contact_replay
from pxr import Gf

SOURCE = Path(__file__).resolve().parents[1] / "validate_tall_package_live.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_config")
namespace = dict(math=math, re=re, uuid=uuid, Path=Path)
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
make_config = namespace["make_config"]


def replay_function(name, namespace):
    """Load an unchanged runtime helper from the shared replay source."""
    function = next(
        node for node in contact_replay.replay.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class TestTallPackageReplayControls(unittest.TestCase):
    def test_profile_requires_real_lift_and_release_with_an_explicit_experimental_label(self):
        config = make_config(object_root="/World/G1_SampleBoxes/Box_10", output=Path("attempt.json"))
        self.assertEqual(config["fixture"], "tall_package")
        self.assertEqual(config["side"], "right")
        self.assertEqual(config["min_lift"], 0.025)
        self.assertEqual(config["min_release_drop"], 0.015)
        self.assertGreaterEqual(config["hold"], 1.0)
        self.assertEqual(config["finger_direction"], (1, 0, 0))
        self.assertEqual(config["palm_normal"], (0, 0, -1))
        profile = config["profiles"][config["paths"][0]]
        self.assertEqual(profile["base_profile"], "experimental_right_horizontal_handle")
        self.assertTrue(Path(config["output"]).is_absolute())

    def test_invalid_profile_or_unbounded_endpoint_rejects_before_connecting(self):
        cases = (
            {"object_root": "relative/Box_10"},
            {"object_root": "/World/Box_10.attr"},
            {"offset": (0, 0)},
            {"offset": (math.nan, 0, 0)},
            {"offset": (0.09, 0, 0)},
            {"timeout": 601},
            {"timeout": 15},
            {"hold": 0.5},
            {"lift": 0.13},
            {"lift": 0.02},
            {"opposition": 1.1},
            {"thumb_curl": -0.1},
            {"curl": math.inf},
        )
        for override in cases:
            with self.subTest(override=override), self.assertRaises(ValueError):
                make_config(**{"object_root": "/World/Box_10", "output": Path("attempt.json"), **override})

    def test_reach_uses_measured_shoulder_span_and_converts_palm_to_wrist(self):
        positions = {"shoulder": (0, 0, 1), "elbow": (0.2, 0, 1), "wrist": (0.4, 0, 1)}
        ex = types.SimpleNamespace(
            _get_webcam_robot_link=lambda side, role: (
                0,
                types.SimpleNamespace(get_world_poses=lambda: ([positions[role]], None)),
            ),
            _first_pose_value=lambda value: value[0],
            _get_robot_palm_local_frame=lambda side: Gf.Matrix4d(1),
            _get_robot_palm_local_center=lambda side: Gf.Vec3d(0.08, 0, 0),
        )
        namespace = dict(ex=ex, side="right", Gf=Gf, math=math, report={}, vector=lambda value: Gf.Vec3d(*value))
        check = replay_function("check_handle_reach", namespace)
        check({"approach": Gf.Vec3d(0.46, 0, 1)}, Gf.Matrix4d(1))
        self.assertAlmostEqual(namespace["report"]["reach_preflight"][0]["wrist_distance_m"], 0.38)
        with self.assertRaisesRegex(RuntimeError, "walk closer"):
            check({"lift": Gf.Vec3d(0.65, 0, 1)}, Gf.Matrix4d(1))
        ex._get_webcam_robot_link = lambda side, role: None
        with self.assertRaisesRegex(RuntimeError, "Cannot measure"):
            check({"approach": Gf.Vec3d(0.3, 0, 1)}, Gf.Matrix4d(1))

    def test_walking_grasp_health_rejects_motion_falls_commands_and_gaze_replacement(self):
        gaze = object()
        ex = types.SimpleNamespace(
            _articulation_health_fault=None,
            _physics_step_error_logged=set(),
            _base_command=np.zeros(3),
            _eye_gaze_tracker=gaze,
            _first_pose_value=lambda value: value[0],
            g1=types.SimpleNamespace(robot=types.SimpleNamespace(get_velocities=lambda: ([[0, 0, 0]], [[0, 0, 0]]))),
        )
        position, quaternion = np.array([0, 0, 0.78]), np.array([1, 0, 0, 0])
        namespace = dict(
            ex=ex,
            config={"fixture": "tall_package"},
            root_start=(position.copy(), quaternion.copy()),
            root_pose=lambda: (position, quaternion),
            phase_stats=None,
            active_path=None,
            report={"max_root_translation_m": 0, "max_root_rotation_rad": 0},
            gaze_before=gaze,
            vector=lambda value: Gf.Vec3d(*value),
            np=np,
            math=math,
        )
        health = replay_function("health", namespace)
        health()
        position[:] = (0.26, 0, 0.78)
        with self.assertRaisesRegex(RuntimeError, "balance drift"):
            health()
        position[:] = (0, 0, 0.40)
        with self.assertRaisesRegex(RuntimeError, "height"):
            health()
        position[:] = (0, 0, 0.78)
        ex._base_command[0] = 0.1
        with self.assertRaisesRegex(RuntimeError, "nonzero"):
            health()
        ex._base_command[:] = 0
        ex._eye_gaze_tracker = object()
        with self.assertRaisesRegex(RuntimeError, "Gaze tracker"):
            health()
        ex._eye_gaze_tracker = gaze
        ex.g1.robot.get_velocities = lambda: ([[0.9, 0, 0]], [[0, 0, 0]])
        with self.assertRaisesRegex(RuntimeError, "velocity"):
            health()


class TestTallPackageReplayCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_restores_navigation_and_zeroes_commands_before_final_result(self):
        fixture = contact_replay.TestContactReplayCleanup()
        with tempfile.TemporaryDirectory() as directory:
            namespace, modules, entered, release, output, _ = fixture.make_cleanup(directory)
            ex = namespace["ex"]
            original_navigation = mock.Mock(return_value="original navigation")
            ex._read_navigation_hand = mock.Mock(return_value=(None, None, None, None))
            ex._reset_gesture_locomotion = mock.Mock()
            for name in ("_base_command", "_keyboard_command", "_controller_command"):
                setattr(ex, name, np.ones(3))
            namespace["config"]["fixture"] = "tall_package"
            namespace["locomotion_overrides"] = {"_read_navigation_hand": (True, original_navigation)}
            with mock.patch.dict("sys.modules", modules):
                task = asyncio.create_task(namespace["run_cleanup"]())
                await asyncio.wait_for(entered.wait(), timeout=1)
                task.cancel()
                release.set()
                await asyncio.wait_for(task, timeout=1)
            self.assertIs(ex._read_navigation_hand, original_navigation)
            ex._reset_gesture_locomotion.assert_called_once()
            for name in ("_base_command", "_keyboard_command", "_controller_command"):
                np.testing.assert_array_equal(getattr(ex, name), np.zeros(3))
            report = json.loads(output.read_text())
            self.assertTrue(report["restored"])
            self.assertFalse(report["playing_at_end"])


if __name__ == "__main__":
    unittest.main()
