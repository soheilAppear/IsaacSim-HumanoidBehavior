# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the live grasp replay's optical skeleton against the real hand parser."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import math
import os
import tempfile
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

from humanoid_test_support import HUMANOID
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

SOURCE = Path(__file__).resolve().parents[1] / "validate_contact_grasp_live.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
profile_function = next(
    node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "choose_profile"
)
profile_namespace = {"math": math}
exec(compile(ast.Module(body=[profile_function], type_ignores=[]), str(SOURCE), "exec"), profile_namespace)
choose_profile = profile_namespace["choose_profile"]
main_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
MAIN_CODE = compile(ast.Module(body=[main_function], type_ignores=[]), str(SOURCE), "exec")
live_source = next(
    node.value.value
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "LIVE_CODE" for target in node.targets)
)
replay = next(
    node
    for node in ast.parse(live_source).body
    if isinstance(node, ast.AsyncFunctionDef) and node.name == "_validate_contact_grasp"
)
REPLAY_CODE = compile(ast.Module(body=[replay], type_ignores=[]), str(SOURCE), "exec")
hand_class = next(node for node in replay.body if isinstance(node, ast.ClassDef) and node.name == "OpticalHand")
HAND_CODE = compile(ast.Module(body=[hand_class], type_ignores=[]), str(SOURCE), "exec")
preflight_function = next(
    node for node in replay.body if isinstance(node, ast.FunctionDef) and node.name == "preflight_profiles"
)
PREFLIGHT_CODE = compile(ast.Module(body=[preflight_function], type_ignores=[]), str(SOURCE), "exec")
save_function = next(node for node in replay.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "save")
SAVE_CODE = compile(ast.Module(body=[save_function], type_ignores=[]), str(SOURCE), "exec")
replay_try = next(node for node in replay.body if isinstance(node, ast.Try))
cleanup_wrapper = ast.parse("async def run_cleanup():\n    pass\n").body[0]
cleanup_wrapper.body = replay_try.finalbody
CLEANUP_CODE = compile(
    ast.fix_missing_locations(ast.Module(body=[cleanup_wrapper], type_ignores=[])), str(SOURCE), "exec"
)


class TestContactReplayProfiles(unittest.TestCase):
    def test_right_hand_defaults_are_independent_per_object(self):
        cube, cylinder = choose_profile("cube"), choose_profile("cylinder")
        self.assertEqual(
            (cube["offset"], cube["curl"], cube["thumb_curl"], cube["opposition"]),
            ((0.008, 0.013, 0.040), 0.75, 0.45, 0.40),
        )
        self.assertEqual(
            (cylinder["offset"], cylinder["curl"], cylinder["thumb_curl"], cylinder["opposition"]),
            ((0.003, 0.015, 0.060), 0.75, 0.55, 0.50),
        )
        self.assertEqual(cube["base_profile"], "validated_right_cube")
        self.assertEqual(cylinder["base_profile"], "validated_right_cylinder")
        self.assertFalse(cube["overrides"])
        cube["offset"] = (9, 9, 9)
        self.assertEqual(choose_profile("cube")["offset"], (0.008, 0.013, 0.040))

    def test_partial_overrides_preserve_other_endpoints_and_explicit_zeros(self):
        for name in ("cube", "cylinder"):
            defaults = choose_profile(name)
            for key, value in (("offset", (0.0, 0.0, 0.0)), ("thumb_curl", 0.0), ("opposition", 0.0)):
                with self.subTest(object=name, override=key):
                    selected = choose_profile(name, **{key: value})
                    self.assertEqual(selected[key], value)
                    self.assertEqual(selected["overrides"], [key])
                    for other in ("offset", "curl", "thumb_curl", "opposition"):
                        if other != key:
                            self.assertEqual(selected[other], defaults[other])

    def test_left_requires_all_custom_endpoints_without_a_validated_label(self):
        values = {"offset": (0.0, 0.01, 0.0), "thumb_curl": 0.0, "opposition": 0.0}
        for missing in values:
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "Left-hand runs require"):
                choose_profile("cube", side="left", **{key: value for key, value in values.items() if key != missing})
        selected = choose_profile("cylinder", side="left", **values)
        self.assertEqual(selected["base_profile"], "custom_left")
        self.assertEqual(selected["offset"], values["offset"])
        self.assertEqual(selected["thumb_curl"], 0.0)
        self.assertEqual(selected["opposition"], 0.0)

    def test_cli_both_selects_profiles_and_applies_only_explicit_overrides(self):
        for overrides in ([], ["--thumb-curl", "0", "--offset", "0", "0.01", "0"]):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                build = mock.Mock(return_value="offline code")
                namespace = dict(
                    argparse=argparse,
                    math=math,
                    time=time,
                    uuid=uuid,
                    Path=Path,
                    __doc__="Offline profile test",
                    __file__=str(SOURCE),
                    DEFAULT_HOST="127.0.0.1",
                    DEFAULT_PORT=8226,
                    choose_profile=choose_profile,
                    build_live_code=build,
                    execute=mock.Mock(return_value=19),
                )
                exec(MAIN_CODE, namespace)
                argv = [str(SOURCE), "--object", "both", "--output", str(Path(directory) / "result.json"), *overrides]
                with mock.patch("sys.argv", argv):
                    self.assertEqual(namespace["main"](), 19)
                config = build.call_args.args[0]
                self.assertEqual(config["hold"], 2.0)
                self.assertEqual(config["palm_normal"], (0.0, 0.0, -1.0))
                self.assertEqual(config["finger_direction"], (1.0, 0.0, 0.0))
                self.assertEqual(len(config["paths"]), 2)
                for index, name in enumerate(("cube", "cylinder")):
                    path = f"/World/G1_SampleBoxes/Box_{index:02d}"
                    expected = (
                        choose_profile(name, offset=(0, 0.01, 0), thumb_curl=0) if overrides else choose_profile(name)
                    )
                    self.assertEqual(config["profiles"][path], expected)

    def test_all_profile_endpoints_round_trip_before_replay_and_leave_the_hand_open(self):
        ex = HUMANOID.HumanoidExample()
        profiles = {
            f"/World/G1_SampleBoxes/Box_{index:02d}": choose_profile(name)
            for index, name in enumerate(("cube", "cylinder"))
        }
        for angle in (0.0, 50.0, 130.0):
            with self.subTest(angle=angle):
                report = {}
                namespace = dict(
                    ex=ex,
                    side="right",
                    Gf=Gf,
                    math=math,
                    types=types,
                    report=report,
                    config={"paths": list(profiles), "profiles": profiles},
                )
                exec(HAND_CODE, namespace)
                exec(PREFLIGHT_CODE, namespace)
                frame = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3).GetNormalized(), angle))
                hand = namespace["OpticalHand"](Gf.Vec3d(0.4, -0.2, 0.9), frame)
                namespace["preflight_profiles"](hand)
                self.assertEqual(len(report["input_preflight"]), 6)
                closed = [sample for sample in report["input_preflight"] if sample["phase"] == "close"]
                self.assertEqual([sample["thumb_curl"] for sample in closed], [0.45, 0.55])
                self.assertEqual([sample["opposition"] for sample in closed], [0.40, 0.50])
                for sample in closed:
                    self.assertAlmostEqual(sample["parsed"]["thumb"], sample["thumb_curl"], delta=1e-6)
                    self.assertAlmostEqual(sample["parsed"]["thumb_yaw"], sample["opposition"], delta=1e-6)
                self.assertEqual((hand.curl, hand.thumb_curl, hand.opposition), (0.0, 0.0, 0.0))


class TestContactReplayControls(unittest.TestCase):
    def test_progress_save_retries_windows_read_sharing_without_blocking_kit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "progress.json")
            replace = mock.Mock(side_effect=[PermissionError("reader holds file"), None])
            pause = mock.AsyncMock()
            namespace = dict(
                report={"status": "running"},
                started=time.monotonic(),
                time=time,
                ex=types.SimpleNamespace(_headset_gait_time=1.0),
                initial_sim=0.0,
                config={"output": output},
                json=json,
                os=types.SimpleNamespace(replace=replace),
                asyncio=types.SimpleNamespace(sleep=pause),
            )
            exec(SAVE_CODE, namespace)
            asyncio.run(namespace["save"]())
            self.assertEqual(replace.call_count, 2)
            pause.assert_awaited_once_with(0.01)
            self.assertEqual(json.loads(Path(output + ".tmp").read_text())["status"], "running")

    def test_synthetic_optical_curls_round_trip_for_both_hands_and_world_frames(self):
        ex = HUMANOID.HumanoidExample()
        for side in ("left", "right"):
            namespace = dict(ex=ex, side=side, Gf=Gf, math=math, types=types)
            exec(HAND_CODE, namespace)
            for angle in (0.0, 50.0, 130.0):
                frame = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3).GetNormalized(), angle))
                center = Gf.Vec3d(0.4, -0.2, 0.9)
                hand = namespace["OpticalHand"](center, frame)
                for curl, opposition in ((0.0, 0.0), (0.2, 0.8), (0.75, 0.65), (1.0, 1.0), (0.0, 0.0)):
                    with self.subTest(side=side, angle=angle, curl=curl, opposition=opposition):
                        hand.curl, hand.opposition = curl, opposition
                        parsed = ex._get_hand_tracking_finger_curls(hand)
                        self.assertIsNotNone(parsed)
                        for role in ex._finger_roles:
                            # `acos` near a straight finger amplifies roundoff.
                            self.assertAlmostEqual(parsed[role], curl, delta=1e-6)
                        self.assertAlmostEqual(parsed["thumb_yaw"], opposition, places=8)
                        palm = ex._get_optical_arm_pose(side, hand)
                        self.assertLess((palm.ExtractTranslation() - center).GetLength(), 1e-8)
                        measured_frame = ex._get_optical_palm_frame(side, hand)
                        for row in range(3):
                            for col in range(3):
                                self.assertAlmostEqual(measured_frame[row][col], frame[row][col], places=8)

    def test_independent_thumb_curl_round_trip_and_open_reset_for_both_hands_and_world_frames(self):
        ex = HUMANOID.HumanoidExample()
        for side in ("left", "right"):
            namespace = dict(ex=ex, side=side, Gf=Gf, math=math, types=types)
            exec(HAND_CODE, namespace)
            for angle in (0.0, 50.0, 130.0):
                frame = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3).GetNormalized(), angle))
                center = Gf.Vec3d(0.4, -0.2, 0.9)
                hand = namespace["OpticalHand"](center, frame)
                for curl, thumb_curl, opposition in ((0.75, 0.35, 0.30), (0.0, 0.0, 0.0), (0.75, None, 0.30)):
                    with self.subTest(side=side, angle=angle, curl=curl, thumb_curl=thumb_curl):
                        hand.curl, hand.thumb_curl, hand.opposition = curl, thumb_curl, opposition
                        parsed = ex._get_hand_tracking_finger_curls(hand)
                        self.assertIsNotNone(parsed)
                        for role in ex._finger_roles:
                            expected = thumb_curl if role == "thumb" and thumb_curl is not None else curl
                            self.assertAlmostEqual(parsed[role], expected, delta=1e-6)
                        self.assertAlmostEqual(parsed["thumb_yaw"], opposition, places=8)
                        palm = ex._get_optical_arm_pose(side, hand)
                        self.assertLess((palm.ExtractTranslation() - center).GetLength(), 1e-8)
                        measured_frame = ex._get_optical_palm_frame(side, hand)
                        for row in range(3):
                            for col in range(3):
                                self.assertAlmostEqual(measured_frame[row][col], frame[row][col], places=8)


class TestContactReplayMimicPreflight(unittest.IsolatedAsyncioTestCase):
    async def run_preflight(
        self,
        directory,
        *,
        metadata=True,
        frequency=1000.0,
        damping=1.0,
        configured_frequency=1000.0,
        configured_damping=1.0,
        stop_after_guard=False,
        health_fault=None,
        duplicate_path=False,
        example_state="loaded",
    ):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World/G1")
        paths = []
        for side in ("left", "right"):
            for index in range(6):
                path = f"/World/G1/{side}_hand/mimic_{index}"
                prim = UsdPhysics.RevoluteJoint.Define(stage, path).GetPrim()
                mimic = PhysxSchema.PhysxMimicJointAPI.Apply(prim, "rotX")
                mimic.CreateNaturalFrequencyAttr().Set(frequency if not paths else configured_frequency)
                mimic.CreateDampingRatioAttr().Set(damping if not paths else configured_damping)
                paths.append(path)
        live_reader = mock.Mock(return_value="live input")
        contact_read = mock.Mock(return_value=({}, "ok"))
        ex = types.SimpleNamespace(
            _g1_prim_path="/World/G1",
            _g1_locomotion="stationary",
            _grasp_mode="physical",
            _xr_core=object(),
            _contact_reader=types.SimpleNamespace(read=contact_read),
            _get_xr_input_device=live_reader,
            _headset_gait_time=0.0,
            _grasp_validation_fixture={"kind": "claims_matching_compliant_couplings"},
            _finger_mimic_natural_frequency=configured_frequency,
            _finger_mimic_damping_ratio=configured_damping,
            _articulation_health_fault=health_fault,
            _physics_step_error_logged={"preflight_test_stop"} if stop_after_guard else set(),
            _drop_everything=mock.Mock(),
            _deactivate_hand=mock.Mock(),
            g1=types.SimpleNamespace(
                has_finger_control=mock.Mock(return_value=True),
                robot=types.SimpleNamespace(is_physics_tensor_entity_valid=mock.Mock(return_value=True)),
            ),
        )
        if metadata:
            ex._physical_hand_mimic_joint_paths = paths
        if duplicate_path:
            stage.RemovePrim(paths[-1])
            ex._physical_hand_mimic_joint_paths = paths[:-1] + [paths[0]]
        if example_state != "loaded":
            stage = None
        if example_state == "unloaded":
            ex.g1 = None
        replay_ex = None if example_state == "missing" else ex
        timeline = types.SimpleNamespace(pause=mock.Mock(), play=mock.Mock(), is_playing=mock.Mock(return_value=False))
        omni = types.ModuleType("omni")
        omni.timeline = types.ModuleType("omni.timeline")
        omni.timeline.get_timeline_interface = lambda: timeline
        omni.usd = types.ModuleType("omni.usd")
        omni.usd.get_context = lambda: types.SimpleNamespace(get_stage=lambda: stage)
        omni.kit = types.ModuleType("omni.kit")
        omni.kit.app = types.ModuleType("omni.kit.app")
        omni.kit.app.get_app = mock.Mock()
        modules = {
            "omni": omni,
            "omni.timeline": omni.timeline,
            "omni.usd": omni.usd,
            "omni.kit": omni.kit,
            "omni.kit.app": omni.kit.app,
        }
        output = Path(directory) / "preflight.json"
        namespace = {"asyncio": asyncio}
        exec(REPLAY_CODE, namespace)
        with mock.patch.dict("sys.modules", modules):
            await namespace["_validate_contact_grasp"](
                replay_ex, {"run_id": "mimic_preflight_test", "side": "right", "output": str(output)}
            )
        # These calls only occur after installing synthetic input. Guard rejection
        # must leave the original readers untouched without relying on restoration.
        self.assertIs(ex._get_xr_input_device, live_reader)
        self.assertIs(ex._contact_reader.read, contact_read)
        ex._drop_everything.assert_not_called()
        ex._deactivate_hand.assert_not_called()
        timeline.play.assert_not_called()
        live_reader.assert_not_called()
        contact_read.assert_not_called()
        result = json.loads(output.read_text())
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["restored"])
        self.assertFalse(result["playing_at_end"])
        self.assertNotIn("input_preflight", result)
        return result, paths

    async def test_missing_preparation_metadata_rejects_before_synthetic_override(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = await self.run_preflight(directory, metadata=False)
        self.assertEqual(set(result["actual_hand_mimic_parameters"]), set(paths))
        self.assertIn("must match the configured finite compliant parameters", result["error"])
        self.assertNotIn("prepared_hand_mimic_joint_paths", result)

    async def test_stale_or_mismatched_coupling_rejects_despite_fixture_claim_and_matching_paths(self):
        for frequency, damping in ((0.0, 0.0), (10.0, 1.0), (1000.0, 0.5), (1000.0, 2.0)):
            with self.subTest(frequency=frequency, damping=damping), tempfile.TemporaryDirectory() as directory:
                result, paths = await self.run_preflight(directory, frequency=frequency, damping=damping)
                observed = result["actual_hand_mimic_parameters"][paths[0]]
                self.assertEqual(observed["physxMimicJoint:rotX:naturalFrequency"], frequency)
                self.assertEqual(observed["physxMimicJoint:rotX:dampingRatio"], damping)
                self.assertIn("must match the configured finite compliant parameters", result["error"])
                self.assertNotIn("prepared_hand_mimic_joint_paths", result)

    async def test_twelve_matching_compliant_couplings_reach_the_next_preflight_check(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = await self.run_preflight(directory, stop_after_guard=True)
        self.assertEqual(result["prepared_hand_mimic_joint_paths"], paths)
        self.assertEqual(len(result["actual_hand_mimic_parameters"]), 12)
        self.assertEqual(result["expected_hand_mimic_parameters"], {"naturalFrequency": 1000.0, "dampingRatio": 1.0})
        self.assertIn("preflight_test_stop", result["error"])
        self.assertNotIn("must match the configured finite compliant parameters", result["error"])

    async def test_configured_compliance_must_be_finite_positive_and_at_least_critically_damped(self):
        for frequency, damping in ((0.0, 0.0), (-1.0, 1.0), (1000.0, 0.9), (math.inf, 1.0), (1000.0, math.nan)):
            with self.subTest(frequency=frequency, damping=damping), tempfile.TemporaryDirectory() as directory:
                result, _ = await self.run_preflight(
                    directory,
                    frequency=frequency,
                    damping=damping,
                    configured_frequency=frequency,
                    configured_damping=damping,
                )
                self.assertIn("must match the configured finite compliant parameters", result["error"])
                self.assertNotIn("prepared_hand_mimic_joint_paths", result)

    async def test_matching_float_roundoff_within_relative_tolerance_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = await self.run_preflight(directory, frequency=1000.0005, stop_after_guard=True)
        self.assertEqual(result["prepared_hand_mimic_joint_paths"], paths)
        self.assertIn("preflight_test_stop", result["error"])

    async def test_latched_articulation_fault_stops_before_synthetic_override(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = await self.run_preflight(directory, health_fault="R_index_intermediate speed exceeded")
        self.assertEqual(result["prepared_hand_mimic_joint_paths"], paths)
        self.assertIn("Production articulation health guard", result["error"])
        self.assertEqual(result["articulation_health_fault"], "R_index_intermediate speed exceeded")

    async def test_duplicate_metadata_cannot_hide_a_missing_observed_coupling(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = await self.run_preflight(directory, duplicate_path=True)
        self.assertEqual(len(result["actual_hand_mimic_parameters"]), 11)
        self.assertIn("must match the configured finite compliant parameters", result["error"])
        self.assertNotIn("prepared_hand_mimic_joint_paths", result)

    async def test_missing_or_unloaded_example_still_publishes_final_failure(self):
        for state in ("missing", "unloaded"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                result, _ = await self.run_preflight(directory, example_state=state)
                self.assertIn("Load the Humanoid and initialize physics", result["error"])
                self.assertNotIn("actual_hand_mimic_parameters", result)


class TestContactReplayCleanup(unittest.IsolatedAsyncioTestCase):
    def make_cleanup(self, directory, *, overrides=False):
        class Example:
            def _get_xr_input_device(self, handle):
                return ("live", handle)

        class Reader:
            def read(self, dt):
                return ({"left": [], "right": []}, "ok")

        ex, reader = Example(), Reader()
        ex._drop_everything = mock.Mock()
        ex._deactivate_hand = mock.Mock()
        ex._physics_step_error_logged = set()
        ex._headset_gait_time = 1.0
        if overrides:
            ex._get_xr_input_device = mock.Mock(return_value="original override")
            reader.read = mock.Mock(return_value=({}, "original override"))
        original_reader, original_contact_read = ex._get_xr_input_device, reader.read
        ex._get_xr_input_device = mock.Mock(return_value="synthetic input")
        reader.read = mock.Mock(return_value=({}, "synthetic contacts"))
        entered, release = asyncio.Event(), asyncio.Event()
        timeline = types.SimpleNamespace(pause=mock.Mock(), is_playing=mock.Mock(return_value=True))

        async def next_update():
            entered.set()
            await release.wait()
            timeline.is_playing.return_value = False

        omni = types.ModuleType("omni")
        omni.kit = types.ModuleType("omni.kit")
        omni.kit.app = types.ModuleType("omni.kit.app")
        omni.kit.app.get_app = lambda: types.SimpleNamespace(next_update_async=next_update)
        modules = {"omni": omni, "omni.kit": omni.kit, "omni.kit.app": omni.kit.app}
        output = Path(directory) / "result.json"
        marker = Path(directory) / "provenance.json"
        report = {"status": "passed"}

        def write_provenance(*, ended):
            marker.write_text(json.dumps({"ended": ended, "status": report["status"]}), encoding="utf-8")

        namespace = dict(
            asyncio=asyncio,
            time=time,
            os=os,
            json=json,
            ex=ex,
            contact_reader=reader,
            timeline=timeline,
            report=report,
            started=time.monotonic(),
            initial_sim=0.0,
            config={"output": str(output)},
            provenance={},
            write_provenance=write_provenance,
            original_reader=original_reader,
            original_contact_read=original_contact_read,
            reader_had_override=overrides,
            contact_had_override=overrides,
        )
        exec(SAVE_CODE, namespace)
        exec(CLEANUP_CODE, namespace)
        return namespace, modules, entered, release, output, marker

    def assert_restored(self, namespace, *, overrides):
        ex, reader = namespace["ex"], namespace["contact_reader"]
        self.assertEqual("_get_xr_input_device" in vars(ex), overrides)
        self.assertEqual("read" in vars(reader), overrides)
        if overrides:
            self.assertIs(ex._get_xr_input_device, namespace["original_reader"])
            self.assertIs(reader.read, namespace["original_contact_read"])
        else:
            self.assertIs(ex._get_xr_input_device.__func__, namespace["original_reader"].__func__)
            self.assertIs(reader.read.__func__, namespace["original_contact_read"].__func__)
        ex._drop_everything.assert_called_once_with()
        self.assertEqual(ex._deactivate_hand.call_args_list, [mock.call("left"), mock.call("right")])

    async def test_deadline_during_queued_pause_preserves_cleanup_and_original_methods(self):
        for overrides in (False, True):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                namespace, modules, entered, release, output, marker = self.make_cleanup(directory, overrides=overrides)
                with mock.patch.dict("sys.modules", modules):
                    task = asyncio.create_task(namespace["run_cleanup"]())
                    await asyncio.wait_for(entered.wait(), timeout=1.0)
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    release.set()
                    await asyncio.wait_for(task, timeout=1.0)
                self.assert_restored(namespace, overrides=overrides)
                result = json.loads(output.read_text())
                self.assertEqual(result["status"], "passed")
                self.assertTrue(result["restored"])
                self.assertFalse(result["playing_at_end"])
                self.assertEqual(json.loads(marker.read_text()), {"ended": True, "status": "passed"})

    async def test_deadline_during_final_save_retry_still_publishes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace, modules, _, release, output, marker = self.make_cleanup(directory)
            release.set()
            write_started = asyncio.Event()
            original_replace = os.replace

            def replace(source, destination):
                if not write_started.is_set():
                    write_started.set()
                    raise PermissionError("reader holds file")
                original_replace(source, destination)

            with mock.patch.dict("sys.modules", modules), mock.patch.object(os, "replace", side_effect=replace):
                task = asyncio.create_task(namespace["run_cleanup"]())
                await asyncio.wait_for(write_started.wait(), timeout=1.0)
                task.cancel()
                await asyncio.wait_for(task, timeout=1.0)
            self.assert_restored(namespace, overrides=False)
            self.assertEqual(json.loads(output.read_text())["status"], "passed")
            self.assertTrue(json.loads(marker.read_text())["ended"])

    async def test_callback_error_while_pause_is_pending_fails_final_report(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace, modules, entered, release, output, marker = self.make_cleanup(directory)
            with mock.patch.dict("sys.modules", modules):
                task = asyncio.create_task(namespace["run_cleanup"]())
                await asyncio.wait_for(entered.wait(), timeout=1.0)
                namespace["ex"]._physics_step_error_logged.add("late physics failure")
                release.set()
                await asyncio.wait_for(task, timeout=1.0)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["callback_errors"], ["late physics failure"])
            self.assertEqual(json.loads(marker.read_text())["status"], "failed")

    async def test_unacknowledged_pause_is_bounded_and_publishes_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace, modules, _, _, output, marker = self.make_cleanup(directory)
            with mock.patch.dict("sys.modules", modules):
                await asyncio.wait_for(namespace["run_cleanup"](), timeout=6.0)
            self.assert_restored(namespace, overrides=False)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["restored"])
            self.assertTrue(result["playing_at_end"])
            self.assertIn("TimeoutError", result["pause_wait_error"])
            self.assertEqual(json.loads(marker.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
