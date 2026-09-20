# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate bounded replay configuration and both anatomical optical generators."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np
from humanoid_test_support import HUMANOID, PACKAGE_SHELVES, ROOT
from pxr import Gf, Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(ROOT / "tools"))
try:
    spec = importlib.util.spec_from_file_location("bimanual_replay_test", ROOT / "tools/validate_bimanual_box_live.py")
    REPLAY = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = REPLAY
    spec.loader.exec_module(REPLAY)
finally:
    sys.path.pop(0)


class TestBimanualReplayControls(unittest.TestCase):
    def config(self, **overrides):
        return REPLAY.make_config(
            **dict(object_root="/World/G1_SampleBoxes/Box_20", output=Path(".webcam/replay-unit.json"), **overrides)
        )

    def test_bounds_reject_unlimited_squeeze_offset_hold_and_invalid_paths(self):
        for overrides in (
            {"squeeze": 0.021},
            {"squeeze": -0.001},
            {"side_gap": 0.002},
            {"offset": (0.061, 0, 0)},
            {"offset": (0, 0)},
            {"curl": 1.1},
            {"thumb_curl": float("nan")},
            {"opposition": True},
            {"hold": 0.5},
            {"lift": 0.2},
            {"timeout": 901},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.config(**overrides)
        with self.assertRaises(ValueError):
            REPLAY.make_config(object_root="/World/Box.attr", output=Path("result.json"))

    def test_default_fixture_requires_measured_lift_hold_and_release(self):
        config = self.config()
        self.assertEqual(config["fixture"], "existing_scene_no_teleports")
        self.assertEqual(config["min_lift"], 0.03)
        self.assertEqual(config["min_release_drop"], 0.015)
        self.assertEqual(config["hold"], 1.0)
        self.assertEqual((config["curl"], config["thumb_curl"], config["opposition"]), (0.5, 0.0, 0.0))
        self.assertFalse(config["clearance_only"])
        self.assertEqual(config["timeout"], 300.0)
        self.assertEqual(self.config(timeout=900)["timeout"], 900.0)
        self.assertTrue(self.config(clearance_only=True)["clearance_only"])
        self.assertLessEqual(config["squeeze"], 0.02)
        compile(REPLAY.build_live_code(config), "<bimanual-server-code>", "exec")

    def test_startup_waits_for_queued_play_and_times_out_without_weakening_health(self):
        code = ast.parse(REPLAY.build_live_code(self.config()))
        helper = next(
            node
            for node in code.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_await_replay_playing"
        )
        namespace = {"asyncio": asyncio}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "<startup-wait>", "exec"), namespace)
        wait = namespace["_await_replay_playing"]

        async def exercise():
            state = dict(playing=False, updates=0)
            timeline = types.SimpleNamespace(is_playing=lambda: state["playing"])

            async def update():
                await asyncio.sleep(0)
                state["updates"] += 1
                state["playing"] = state["updates"] >= 3

            await wait(timeline, types.SimpleNamespace(next_update_async=update), timeout_s=1)
            self.assertEqual(state["updates"], 3)
            # Already playing never unnecessarily advances the scene.
            await wait(timeline, types.SimpleNamespace(next_update_async=update), timeout_s=1)
            self.assertEqual(state["updates"], 3)
            state["playing"] = False
            with self.assertRaisesRegex(RuntimeError, "did not start"):
                await wait(
                    timeline, types.SimpleNamespace(next_update_async=lambda: asyncio.sleep(60)), timeout_s=0.001
                )

        asyncio.run(exercise())

    def test_approach_must_align_both_palms_before_closing(self):
        code = ast.parse(REPLAY.build_live_code(self.config()))
        helper = next(
            node for node in code.body if isinstance(node, ast.FunctionDef) and node.name == "_bimanual_approach_ready"
        )
        namespace = {}
        high_helper = next(
            node
            for node in code.body
            if isinstance(node, ast.FunctionDef) and node.name == "_bimanual_above_shelf_ready"
        )
        exec(compile(ast.Module(body=[helper, high_helper], type_ignores=[]), "<approach-gate>", "exec"), namespace)
        hands = {
            side: dict(scenery_motion_limited=False, palm_error_m=0.025, palm_angle_error_deg=4)
            for side in ("left", "right")
        }
        ready = namespace["_bimanual_approach_ready"]
        self.assertTrue(ready(hands))
        for changed in ({"palm_error_m": 0.30}, {"palm_angle_error_deg": 26}, {"scenery_motion_limited": True}):
            sample = {side: dict(value) for side, value in hands.items()}
            sample["left"].update(changed)
            self.assertFalse(ready(sample))
        sample = {side: dict(value) for side, value in hands.items()}
        sample["right"]["palm_error_m"] = 0.05
        self.assertTrue(ready(sample, max_error_m=0.08, max_angle_deg=25))
        self.assertFalse(ready(sample))
        for hand in hands.values():
            hand["actual_palm"] = [1.27, -0.09, 1.205]
        self.assertTrue(namespace["_bimanual_above_shelf_ready"](hands, 1.085))
        hands["right"]["actual_palm"][2] = 1.18
        self.assertFalse(namespace["_bimanual_above_shelf_ready"](hands, 1.085))

    def test_phase_joint_diagnostics_preserve_native_commands_and_optional_read_errors(self):
        code = ast.parse(REPLAY.build_live_code(self.config()))
        helper = next(
            node for node in ast.walk(code) if isinstance(node, ast.FunctionDef) and node.name == "arm_joint_snapshot"
        )
        calls = []

        def gravity(*, dof_indices):
            calls.append(dof_indices)
            return [[1.5, -2.5]]

        def unavailable(*, dof_indices):
            raise RuntimeError("unavailable fixture capability")

        robot = types.SimpleNamespace(
            num_dofs=2,
            get_dof_positions=lambda: [[0.1, 0.2]],
            get_dof_position_targets=lambda: [[0.3, 0.1]],
            get_dof_velocities=lambda: [[0.01, 0.02]],
            get_dof_gravity_compensation_forces=gravity,
            get_dof_max_efforts=unavailable,
        )
        example = types.SimpleNamespace(
            g1=types.SimpleNamespace(robot=robot),
            _first_pose_value=lambda values: values[0],
            _g1_arm_dof_indices_by_side={"left": {"elbow": 0}, "right": {"elbow": 1}},
            _smoothed_arm_targets={0: 0.29, 1: 0.11},
            _get_arm_target_limits=lambda index: (-1.0, 2.0),
        )
        namespace = {"np": np, "ex": example}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "<joint-diagnostics>", "exec"), namespace)
        result = namespace["arm_joint_snapshot"]()
        self.assertEqual(calls, [[0, 1]])
        self.assertEqual(result["left"]["elbow"]["native_target_rad"], 0.3)
        self.assertAlmostEqual(result["left"]["elbow"]["target_minus_measured_rad"], 0.2)
        self.assertEqual(result["right"]["elbow"]["gravity_wrapper_unverified_value"], -2.5)
        self.assertEqual(result["gravity_mapping"]["status"], "unverified_wrapper_values")
        self.assertIsNone(result["right"]["elbow"]["max_effort_nm"])
        self.assertIn("unavailable fixture capability", result["optional_read_errors"]["max_effort_nm"])
        for fixed_base, raw, expected_offset in (
            (False, [[100, 101, 102, 103, 104, 105, 7, -3]], 6),
            (True, [[7, -3]], 0),
        ):
            robot._physics_articulation_view = types.SimpleNamespace(
                shared_metatype=types.SimpleNamespace(fixed_base=fixed_base),
                get_gravity_compensation_forces=lambda: raw,
            )
            result = namespace["arm_joint_snapshot"]()
            self.assertEqual(result["gravity_mapping"]["status"], "verified_native_layout")
            self.assertEqual(result["gravity_mapping"]["joint_offset"], expected_offset)
            self.assertEqual(result["left"]["elbow"]["gravity_joint_model_nm"], 7)
            self.assertEqual(result["right"]["elbow"]["gravity_joint_model_nm"], -3)
            self.assertEqual(result["right"]["elbow"]["gravity_wrapper_unverified_value"], -2.5)
        robot._physics_articulation_view.shared_metatype.fixed_base = False
        result = namespace["arm_joint_snapshot"]()
        self.assertEqual(result["gravity_mapping"]["status"], "unverified_wrapper_values")
        self.assertIsNone(result["left"]["elbow"]["gravity_joint_model_nm"])
        self.assertIn("contradicts", result["optional_read_errors"]["gravity_native"])

    def test_shelf_waypoints_withdraw_before_rising_and_enter_reach_preflight(self):
        stage = Usd.Stage.CreateInMemory()
        PACKAGE_SHELVES.create_shelf_packages(stage, "/World/Packages", first_index=10)
        code = ast.parse(REPLAY.build_live_code(self.config()))
        validator = next(
            node
            for node in code.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_validate_bimanual_box"
        )
        body = next(node for node in validator.body if isinstance(node, ast.Try)).body
        start = next(
            index
            for index, node in enumerate(body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "support" for target in node.targets)
        )
        end = next(
            index
            for index, node in enumerate(body[start:], start)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "reach_preflight"
        )
        captures = []
        shoulders = {"left": Gf.Vec3d(0.95, -0.14, 1.05), "right": Gf.Vec3d(0.95, -0.42, 1.05)}

        def shoulder_link(side, role):
            self.assertEqual(role, "shoulder")
            return ("shoulder", types.SimpleNamespace(get_world_poses=lambda: ([shoulders[side]], None)))

        namespace = {
            "Gf": Gf,
            "Usd": Usd,
            "UsdGeom": UsdGeom,
            "UsdPhysics": UsdPhysics,
            "math": math,
            "vector": lambda values: Gf.Vec3d(*values),
            "ex": types.SimpleNamespace(
                _get_webcam_robot_link=shoulder_link,
                _first_pose_value=lambda values: values[0],
                _spawned_sample_box_paths=[f"/World/Packages/Box_{index:02d}" for index in range(10, 22)],
                _get_object_world_pose=lambda path: (
                    UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation(),
                    Gf.Quatd(1),
                ),
            ),
            "stage": stage,
            "path": "/World/Packages/Box_20",
            "forward": Gf.Vec3d(1, 0, 0),
            "report": {},
            "root_start": (Gf.Vec3d(0.9322, -0.28, 0.745), Gf.Quatd(1)),
            "config": {"clearance_only": False},
            "hands": {
                side: types.SimpleNamespace(center=Gf.Vec3d(1.234, y, 0.85), frame=Gf.Matrix4d(1))
                for side, y in (("left", -0.114), ("right", -0.449))
            },
            "wide_targets": {side: Gf.Vec3d(1.275, y, 1.14) for side, y in (("left", -0.09), ("right", -0.47))},
            "targets": {},
            "squeeze_targets": {},
            "lift_targets": {},
            "frames": {},
            "reach_preflight": lambda targets, frames: captures.append(targets),
        }
        corridor_helper = next(
            node
            for node in code.body
            if isinstance(node, ast.FunctionDef) and node.name == "_bimanual_front_entry_corridor"
        )
        exec(compile(ast.Module(body=[corridor_helper], type_ignores=[]), "<corridor>", "exec"), namespace)
        exec(compile(ast.Module(body=body[start : end + 1], type_ignores=[]), "<shelf-waypoints>", "exec"), namespace)
        self.assertAlmostEqual(namespace["report"]["shelf_clearance"]["front_x"], 1.18)
        for side in ("left", "right"):
            withdrawn = captures[0]["withdraw"][side]
            raised = captures[0]["raise_in_front"][side]
            self.assertAlmostEqual(withdrawn[0], 1.0)
            self.assertAlmostEqual(withdrawn[2], 0.89)
            prelift = captures[0]["natural_prelift"][side]
            self.assertEqual(tuple(prelift[:2]), tuple(namespace["hands"][side].center[:2]))
            self.assertAlmostEqual(prelift[2], 0.89)
            self.assertAlmostEqual(raised[0], 1.0)
            self.assertAlmostEqual(raised[2], 1.14)
            sign = 1 if side == "left" else -1
            initial_gap = sign * (namespace["hands"][side].center[1] - shoulders[side][1])
            self.assertAlmostEqual(withdrawn[1], shoulders[side][1] + sign * max(0.20, initial_gap))
            self.assertAlmostEqual(raised[1], shoulders[side][1] + sign * 0.24)
            self.assertAlmostEqual(captures[0]["spread_in_front"][side][1], shoulders[side][1] + sign * 0.20)
            self.assertAlmostEqual(captures[0]["spread_in_front"][side][0], namespace["hands"][side].center[0])
            self.assertEqual(captures[0]["align_in_front"][side], withdrawn)
            wrist = withdrawn - Gf.Vec3d(0.068855, 0, 0)
            self.assertGreater((wrist - shoulders[side]).GetLength(), 0.18)
            self.assertGreater(raised[2] - namespace["report"]["shelf_clearance"]["top_z"], 0.065)
            self.assertAlmostEqual(captures[0]["front_entry"][side][2], 1.14)
            pivot = captures[0]["wide_in_front"][side]
            self.assertAlmostEqual(pivot[0], 1.0)
            self.assertAlmostEqual(pivot[1], captures[0]["front_entry"][side][1])
            self.assertAlmostEqual(pivot[1], -0.28 + sign * 0.34)
            self.assertAlmostEqual(pivot[2], 1.14)
        self.assertEqual(namespace["report"]["shelf_clearance"]["minimum_withdrawal_outward_from_shoulder_m"], 0.20)
        self.assertEqual(namespace["report"]["shelf_clearance"]["desired_frame_alignment_after"], "withdraw_in_front")
        corridor = namespace["report"]["front_entry_corridors"]["front_entry"]
        self.assertTrue(corridor["clear"])
        self.assertEqual(len(corridor["obstacles"]), 10)
        self.assertEqual(sum(item["source"] == "measured_dynamic_carton" for item in corridor["obstacles"]), 4)
        namespace["config"]["clearance_only"] = True
        captures.clear()
        exec(compile(ast.Module(body=body[start : end + 1], type_ignores=[]), "<shelf-waypoints>", "exec"), namespace)
        self.assertNotIn("front_entry", captures[0])
        self.assertNotIn("wide_in_front", captures[0])
        self.assertAlmostEqual(captures[0]["raise_in_front"]["left"][2], 1.14)

    def test_reach_preflight_checks_the_whole_clearance_interpolation(self):
        code = ast.parse(REPLAY.build_live_code(self.config()))
        helper = next(
            node for node in ast.walk(code) if isinstance(node, ast.FunctionDef) and node.name == "reach_preflight"
        )
        shoulders = {side: Gf.Vec3d(0.95, sign * 0.14, 1.05) for side, sign in (("left", 1), ("right", -1))}

        def link(side, role):
            position = shoulders[side] + Gf.Vec3d(0, 0, {"shoulder": 0, "elbow": -0.18, "wrist": -0.36}[role])
            return (role, types.SimpleNamespace(get_world_poses=lambda: ([position], None)))

        example = types.SimpleNamespace(
            g1=types.SimpleNamespace(robot=types.SimpleNamespace(get_dof_positions=lambda: [[0, 0]])),
            _first_pose_value=lambda values: values[0],
            _get_webcam_robot_link=link,
            _get_robot_palm_local_frame=lambda side: Gf.Matrix4d(1),
            _get_robot_palm_local_center=lambda side: Gf.Vec3d(0.068855, 0, 0),
            _g1_arm_dof_indices_by_side={"left": {"elbow": 0}, "right": {"elbow": 1}},
            _get_arm_target_limits=lambda index: (-1.047, 2.094),
        )
        hands, targets = {}, {
            name: {} for name in ("natural_prelift", "withdraw", "spread_in_front", "align_in_front", "raise_in_front")
        }
        for side, sign in (("left", 1), ("right", -1)):
            y = shoulders[side][1]
            hands[side] = types.SimpleNamespace(
                center=Gf.Vec3d(1.2, y + sign * 0.03, 0.85),
                frame=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), -30)),
            )
            targets["natural_prelift"][side] = hands[side].center + Gf.Vec3d(0, 0, 0.04)
            targets["withdraw"][side] = Gf.Vec3d(1, y + sign * 0.20, 0.89)
            targets["spread_in_front"][side] = Gf.Vec3d(1.2, y + sign * 0.20, 0.89)
            targets["align_in_front"][side] = targets["withdraw"][side]
            targets["raise_in_front"][side] = Gf.Vec3d(1, y + sign * 0.24, 1.14)
        namespace = {
            "Gf": Gf,
            "np": np,
            "ex": example,
            "hands": hands,
            "report": {},
            "vector": lambda values: Gf.Vec3d(*values),
        }
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "<reach-preflight>", "exec"), namespace)
        frames = {side: Gf.Matrix4d(1) for side in hands}
        namespace["reach_preflight"](targets, frames)
        self.assertEqual(len(namespace["report"]["reach_preflight"]["left"]["clearance_segments"]), 5)
        withdrawal = namespace["report"]["reach_preflight"]["left"]["targets"]["withdraw"]
        self.assertLess(withdrawal["wrist"][2], withdrawal["palm"][2] - 0.03)
        for name in ("wide_in_front", "front_entry", "side_approach"):
            targets[name] = {}
        for side, sign in (("left", 1), ("right", -1)):
            y = shoulders[side][1]
            targets["wide_in_front"][side] = Gf.Vec3d(1, sign * 0.34, 1.14)
            targets["front_entry"][side] = Gf.Vec3d(1.275, sign * 0.34, 1.14)
            targets["side_approach"][side] = Gf.Vec3d(1.275, y + sign * 0.05, 1.14)
        namespace["reach_preflight"](targets, frames)
        segments = namespace["report"]["reach_preflight"]["left"]["clearance_segments"]
        self.assertEqual(len(segments), 8)
        self.assertEqual(segments[-2]["to"], "front_entry")
        self.assertEqual(segments[-1]["to"], "side_approach")
        # Keep lateral room until full forward reach, then narrow at constant Z.
        self.assertGreater(segments[-1]["minimum_wrist_radius_m"], 0.18)
        self.assertLess(segments[-1]["maximum_wrist_radius_m"], 0.36)
        targets["align_in_front"] = {side: Gf.Vec3d(1, shoulders[side][1], 1.0) for side in hands}
        with self.assertRaisesRegex(RuntimeError, "path leaves"):
            namespace["reach_preflight"](targets, frames)

    def test_low_spread_precedes_withdrawal_and_preserves_measured_frame_until_clear(self):
        code = ast.parse(REPLAY.build_live_code(self.config()))
        validator = next(
            node
            for node in code.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_validate_bimanual_box"
        )
        body = next(node for node in validator.body if isinstance(node, ast.Try)).body

        def phase_name(node):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Await):
                call = node.value.value
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "phase":
                    return call.args[0].value
            return None

        first = next(index for index, node in enumerate(body) if phase_name(node) == "initialize_open")
        last = next(index for index, node in enumerate(body) if phase_name(node) == "raise_in_front")
        helper = ast.parse("async def exercise():\n    pass").body[0]
        helper.body = body[first : last + 1]
        hands = {side: types.SimpleNamespace(frame=Gf.Matrix4d(1)) for side in ("left", "right")}
        frames = {side: Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90)) for side in hands}
        captures, report = {}, {"phases": {}}

        async def phase(name, seconds, *args, **kwargs):
            captures[name] = dict(
                frames={side: Gf.Matrix4d(hand.frame) for side, hand in hands.items()}, options=kwargs, duration=seconds
            )
            report["phases"][name] = {
                "hands": {
                    side: dict(palm_error_m=0.0, palm_angle_error_deg=0.0, scenery_motion_limited=False)
                    for side in hands
                }
            }

        ready = next(
            node for node in code.body if isinstance(node, ast.FunctionDef) and node.name == "_bimanual_approach_ready"
        )
        namespace = dict(
            hands=hands,
            frames=frames,
            phase=phase,
            report=report,
            prelift_targets={},
            withdrawn_targets={},
            spread_targets={},
            raised_targets={},
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[ready, helper], type_ignores=[])), "<withdraw-order>", "exec"
            ),
            namespace,
        )
        asyncio.run(namespace["exercise"]())
        self.assertLess(list(captures).index("spread_in_front"), list(captures).index("withdraw_in_front"))
        self.assertEqual(captures["withdraw_in_front"]["duration"], 3.0)
        for side in hands:
            self.assertEqual(captures["natural_prelift"]["frames"][side], Gf.Matrix4d(1))
            self.assertEqual(captures["withdraw_in_front"]["frames"][side], Gf.Matrix4d(1))
            self.assertEqual(captures["spread_in_front"]["frames"][side], Gf.Matrix4d(1))
            self.assertEqual(captures["align_in_front"]["frames"][side], frames[side])
        self.assertTrue(captures["align_in_front"]["options"]["require_contact_free"])
        self.assertTrue(captures["natural_prelift"]["options"]["require_contact_free"])
        self.assertTrue(captures["natural_prelift_settle"]["options"]["require_contact_free"])
        # A 16 mm endpoint miss cannot pass the tighter 15 mm clearance gate.
        old_phase = namespace["phase"]

        async def missed_prelift(name, seconds, *args, **kwargs):
            await old_phase(name, seconds, *args, **kwargs)
            if name == "natural_prelift_settle":
                report["phases"][name]["hands"]["right"]["palm_error_m"] = 0.016

        namespace["phase"] = missed_prelift
        with self.assertRaisesRegex(RuntimeError, "within 15 mm"):
            asyncio.run(namespace["exercise"]())

    def test_early_open_contact_stops_before_later_phases_can_forget_it(self):
        code = ast.parse(REPLAY.build_live_code(self.config()))
        validator = next(
            node
            for node in code.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_validate_bimanual_box"
        )
        body = next(node for node in validator.body if isinstance(node, ast.Try)).body

        def phase_name(node):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Await):
                call = node.value.value
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "phase":
                    return call.args[0].value
            return None

        first = next(index for index, node in enumerate(body) if phase_name(node) == "initialize_open")
        last = next(index for index, node in enumerate(body) if phase_name(node) == "touch_sides")
        helper = ast.parse("async def exercise():\n    pass").body[0]
        helper.body = body[first:last]
        helpers = [
            node
            for node in code.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_bimanual_approach_ready", "_bimanual_open_approach_contact_free")
        ]
        for contact_phase in (
            "initialize_open",
            "natural_prelift",
            "natural_prelift_settle",
            "withdraw_in_front",
            "withdraw_in_front_settle",
            "spread_in_front",
            "spread_in_front_settle",
            "raise_in_front",
            "raise_in_front_settle",
            "clearance_settle",
        ):
            with self.subTest(contact_phase=contact_phase):
                visited, report = [], {"phases": {}}
                hands = {side: types.SimpleNamespace(frame=Gf.Matrix4d(1)) for side in ("left", "right")}
                namespace = dict(
                    hands=hands,
                    frames={side: Gf.Matrix4d(1) for side in hands},
                    report=report,
                    config={"clearance_only": contact_phase == "clearance_settle"},
                    **{
                        name: {}
                        for name in (
                            "prelift_targets",
                            "withdrawn_targets",
                            "spread_targets",
                            "raised_targets",
                            "outboard_front_targets",
                            "outboard_forward_targets",
                            "wide_targets",
                        )
                    },
                )

                async def phase(name, seconds, *args, **kwargs):
                    visited.append(name)
                    # Model phase-local accumulation: a later phase starts empty.
                    stats = dict(max_external_contact_force_n={})
                    if name == contact_phase:
                        stats["max_external_contact_force_n"]["left:/World/NeighborCarton"] = 0.03
                    if kwargs.get("require_contact_free") and not namespace["_bimanual_open_approach_contact_free"](
                        stats
                    ):
                        raise RuntimeError("Unexpected contact in " + name)
                    report["phases"][name] = {
                        "hands": {
                            side: dict(palm_error_m=0.0, palm_angle_error_deg=0.0, scenery_motion_limited=False)
                            for side in hands
                        }
                    }

                namespace["phase"] = phase
                exec(
                    compile(
                        ast.fix_missing_locations(ast.Module(body=[*helpers, helper], type_ignores=[])),
                        "<all-open-contact-gates>",
                        "exec",
                    ),
                    namespace,
                )
                with self.assertRaisesRegex(RuntimeError, "Unexpected contact in " + contact_phase):
                    asyncio.run(namespace["exercise"]())
                self.assertEqual(visited[-1], contact_phase)

    def test_front_entry_envelope_rejects_old_posts_low_palms_and_unintended_contacts(self):
        code = ast.parse(REPLAY.build_live_code(self.config()))
        names = ("_bimanual_front_entry_corridor", "_bimanual_open_approach_contact_free")
        helpers = [node for node in code.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "<front-corridor>", "exec"), namespace)
        check = namespace[names[0]]
        stage = Usd.Stage.CreateInMemory()
        PACKAGE_SHELVES.create_shelf_packages(stage, "/World/Packages", first_index=10)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        obstacles = []
        for prim in stage.GetPrimAtPath("/World/G1_PackageShelves/Shelf_02").GetChildren():
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            obstacles.append(dict(path=str(prim.GetPath()), min=list(bounds.GetMin()), max=list(bounds.GetMax())))
        starts = {side: (1.0, y, 1.14) for side, y in (("left", 0.06), ("right", -0.62))}
        ends = {side: (1.275, point[1], 1.14) for side, point in starts.items()}
        result = check(starts, ends, obstacles)
        self.assertTrue(result["clear"])
        self.assertIn("not full-arm", result["scope"])
        right_clearances = result["hands"]["right"]["separating_axis_clearance_m"]
        self.assertGreater(min(gap for path, gap in right_clearances.items() if "/Post_" in path), 0.08)
        # The former narrow negative-Y posts intrude into the open-hand envelope.
        old_obstacles = []
        for obstacle in obstacles:
            item = dict(obstacle, min=list(obstacle["min"]), max=list(obstacle["max"]))
            if "/Post_" in item["path"]:
                delta = 0.15 if item["max"][1] < 0 else -0.15
                item["min"][1] += delta
                item["max"][1] += delta
            old_obstacles.append(item)
        self.assertFalse(check(starts, ends, old_obstacles)["clear"])
        low_ends = {side: (point[0], point[1], 1.12) for side, point in ends.items()}
        low_starts = {side: (point[0], point[1], 1.12) for side, point in starts.items()}
        self.assertFalse(check(low_starts, low_ends, obstacles)["clear"])
        with self.assertRaisesRegex(RuntimeError, "fixed lateral"):
            check(starts, dict(ends, right=(1.275, -0.45, 1.14)), obstacles)
        contact_free = namespace[names[1]]
        stats = dict(max_external_contact_force_n={})
        self.assertTrue(contact_free(stats))
        stats["max_external_contact_force_n"]["left:/World/NeighborCarton"] = 0.02
        self.assertFalse(contact_free(stats))

    def test_open_approach_rejects_neighbor_and_predictive_target_contacts_without_changing_support(self):
        code = ast.parse(REPLAY.build_live_code(self.config()))
        names = ("capture_contacts", "_bimanual_open_approach_contact_free")
        helpers = [node for node in ast.walk(code) if isinstance(node, ast.FunctionDef) and node.name in names]
        target = "/World/Packages/Box_20"
        for object_path, separation, eligible, expect_blocked in (
            ("/World/Packages/Box_21", 0.0, True, True),
            (target, 0.01, True, True),
            ("/World/Shelf/Deck", 0.01, False, True),
            ("/World/G1/palm", 0.0, False, False),
            ("/World/G1", 0.0, False, False),
        ):
            with self.subTest(path=object_path, separation=separation):
                sample = types.SimpleNamespace(
                    object_path=object_path,
                    separation_m=separation,
                    eligible_for_grasp=eligible,
                    normal_force_n=0.03,
                    role="palm",
                    point_world=(0, 0, 0),
                    normal_world=(0, 1, 0),
                )
                stats = dict(
                    physics_samples=0,
                    opposing_contact_seconds=0,
                    object_z_min=1.14,
                    object_z_max=1.14,
                    max_normal_force_n_by_side=dict(left=0.0, right=0.0),
                    max_scenery_force_n={},
                    max_external_contact_force_n={},
                    contact_trace=[],
                )
                contacts = dict(left=[sample], right=[])
                namespace = dict(
                    Gf=Gf,
                    math=math,
                    phase_stats=stats,
                    path=target,
                    ex=types.SimpleNamespace(_g1_prim_path="/World/G1", _headset_gait_time=1.0),
                    report={"contact_health_counts": {}},
                    object_pose=lambda: (Gf.Vec3d(0, 0, 1.14), Gf.Quatd(1)),
                    original_contact_read=lambda dt: (contacts, "fresh"),
                    vector=lambda values: Gf.Vec3d(*values),
                )
                exec(compile(ast.Module(body=helpers, type_ignores=[]), "<contact-guard>", "exec"), namespace)
                returned, status = namespace["capture_contacts"](0.005)
                self.assertIs(returned, contacts)
                self.assertEqual(status, "fresh")
                self.assertEqual(namespace["_bimanual_open_approach_contact_free"](stats), not expect_blocked)
                self.assertEqual(stats["max_normal_force_n_by_side"], dict(left=0.0, right=0.0))
                self.assertEqual(stats["opposing_contact_seconds"], 0)

    def test_palm_dwell_is_current_contiguous_and_aligned_with_measured_box_sides(self):
        code = ast.parse(REPLAY.build_live_code(self.config()))
        names = ("capture_contacts", "_bimanual_side_contacts_opposed", "_bimanual_contact_dwell_ready")
        helpers = [node for node in ast.walk(code) if isinstance(node, ast.FunctionDef) and node.name in names]
        target = "/World/Packages/Box_20"
        contacts = {}
        stats = dict(
            physics_samples=0,
            opposing_contact_seconds=0.0,
            object_z_min=1.14,
            object_z_max=1.14,
            max_normal_force_n_by_side=dict(left=0.0, right=0.0),
            max_scenery_force_n={},
            max_external_contact_force_n={},
            contact_trace=[],
        )
        state = dict(status="ok", rotation=Gf.Quatd(1))

        def pair(role="palm", axis=(0, 1, 0)):
            for side, sign in (("left", 1), ("right", -1)):
                contacts[side] = [
                    types.SimpleNamespace(
                        object_path=target,
                        role=role,
                        normal_force_n=1.0,
                        eligible_for_grasp=True,
                        separation_m=0.0,
                        point_world=tuple(sign * 0.12 * value for value in axis),
                        normal_world=tuple(sign * value for value in axis),
                    )
                ]

        namespace = dict(
            Gf=Gf,
            math=math,
            phase_stats=stats,
            path=target,
            ex=types.SimpleNamespace(_g1_prim_path="/World/G1", _headset_gait_time=1.0),
            report={"contact_health_counts": {}},
            object_pose=lambda: (Gf.Vec3d(0, 0, 1.14), state["rotation"]),
            original_contact_read=lambda dt: (contacts, state["status"]),
            vector=lambda values: Gf.Vec3d(*values),
        )
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "<palm-dwell>", "exec"), namespace)
        capture, ready = namespace["capture_contacts"], namespace["_bimanual_contact_dwell_ready"]
        pair()
        capture(0.03)
        self.assertFalse(ready(stats, palm_only=True))
        capture(0.03)
        self.assertTrue(ready(stats, palm_only=True))
        contacts["right"] = []
        capture(0.005)
        self.assertEqual(stats["current_opposing_palm_seconds"], 0.0)
        pair()
        capture(0.03)
        self.assertFalse(ready(stats, palm_only=True))
        capture(0.03)
        self.assertTrue(ready(stats, palm_only=True))
        # Opposite back/front normals are not the intended carton side faces.
        pair(axis=(1, 0, 0))
        capture(0.04)
        capture(0.04)
        self.assertFalse(ready(stats))
        # The side axis follows the measured box, rather than assuming world Y.
        state["rotation"] = Gf.Rotation(Gf.Vec3d(0, 0, 1), -90).GetQuat()
        capture(0.03)
        capture(0.03)
        self.assertTrue(ready(stats, palm_only=True))
        pair(role="little", axis=(1, 0, 0))
        capture(0.03)
        capture(0.03)
        self.assertTrue(ready(stats))
        self.assertFalse(ready(stats, palm_only=True))
        state["status"] = "stale"
        capture(0.03)
        self.assertFalse(ready(stats))

    def test_closure_waits_for_palms_and_lift_waits_for_current_side_support(self):
        code = ast.parse(REPLAY.build_live_code(self.config()))
        validator = next(
            node
            for node in code.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_validate_bimanual_box"
        )
        body = next(node for node in validator.body if isinstance(node, ast.Try)).body

        def name(node):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Await):
                call = node.value.value
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "phase":
                    return call.args[0].value
            return None

        first = next(index for index, node in enumerate(body) if name(node) == "touch_sides")
        last = next(index for index, node in enumerate(body) if name(node) == "lift")
        helper = ast.parse("async def exercise():\n    pass").body[0]
        helper.body = body[first : last + 1]
        ready = next(
            node
            for node in code.body
            if isinstance(node, ast.FunctionDef) and node.name == "_bimanual_contact_dwell_ready"
        )
        for palms, sides, failure in (
            (0.055, 0.06, "refusing finger closure"),
            (0.06, 0.055, "refusing lift"),
            (0.06, 0.06, None),
        ):
            with self.subTest(palms=palms, sides=sides):
                visited, report = {}, {"phases": {}}

                async def phase(label, seconds, *targets, **options):
                    visited[label] = dict(targets=targets, options=options)
                    report["phases"][label] = dict(
                        current_opposing_palm_seconds=palms,
                        current_opposing_side_seconds=sides,
                        max_scenery_force_n={},
                        hands={side: dict(palm_error_m=0.0) for side in ("left", "right")},
                    )

                namespace = dict(
                    phase=phase, report=report, config=self.config(), targets={}, squeeze_targets={}, lift_targets={}
                )
                exec(
                    compile(
                        ast.fix_missing_locations(ast.Module(body=[ready, helper], type_ignores=[])),
                        "<palm-first-sequence>",
                        "exec",
                    ),
                    namespace,
                )
                if failure:
                    with self.assertRaisesRegex(RuntimeError, failure):
                        asyncio.run(namespace["exercise"]())
                else:
                    asyncio.run(namespace["exercise"]())
                self.assertEqual(visited["touch_sides"]["options"], dict(curl=0, thumb_curl=0, opposition=0))
                if palms < 0.06:
                    self.assertNotIn("curl_after_palm_contact", visited)
                else:
                    self.assertEqual(visited["curl_after_palm_contact"]["targets"], ())
                    self.assertEqual(
                        visited["curl_after_palm_contact"]["options"], dict(curl=0.5, thumb_curl=0, opposition=0)
                    )
                self.assertEqual("lift" in visited, failure is None)

    def test_both_generated_hands_recover_requested_palm_pose_and_independent_finger_curls(self):
        code = ast.parse(REPLAY.build_live_code(self.config()))
        factory = next(
            node for node in ast.walk(code) if isinstance(node, ast.FunctionDef) and node.name == "make_hand"
        )
        example = HUMANOID.HumanoidExample()
        namespace = {"ex": example, "Gf": Gf, "math": math, "types": types}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), "<optical-factory>", "exec"
            ),
            namespace,
        )
        target = Gf.Vec3d(1.275, -0.28, 1.14)
        hands = {}
        for side, sign in (("left", 1), ("right", -1)):
            forward, normal = Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, -sign, 0)
            transverse = Gf.Cross(normal, forward)
            frame = Gf.Matrix4d(*forward, 0, *transverse, 0, *normal, 0, 0, 0, 0, 1)
            hand = namespace["make_hand"](side, target, frame)
            hand.curl, hand.thumb_curl, hand.opposition = 0.25, 0.15, 0.3
            hands[side] = hand
            points = hand.positions()
            measured = example._build_palm_frame(
                side, *(points[name] for name in ("wrist", "middle_proximal", "index_proximal", "little_proximal"))
            )
            self.assertLess(((points["wrist"] + points["middle_proximal"]) * 0.5 - target).GetLength(), 1e-9)
            self.assertTrue(
                all(abs(measured[row][col] - frame[row][col]) < 1e-9 for row in range(4) for col in range(4))
            )
            curls = example._get_hand_tracking_finger_curls(hand)
            for role in ("index", "middle", "ring", "little"):
                self.assertAlmostEqual(curls[role], 0.25)
            self.assertAlmostEqual(curls["thumb"], 0.15)
            self.assertAlmostEqual(curls["thumb_yaw"], 0.3)
            self.assertEqual(hand.get_name(), "/user/hand/" + side)
        # Each factory call closes over its own side; creating the right hand
        # must not change the left hand's mirror convention or device identity.
        self.assertEqual(hands["left"].get_name(), "/user/hand/left")
        self.assertEqual(hands["right"].get_name(), "/user/hand/right")


if __name__ == "__main__":
    unittest.main()
