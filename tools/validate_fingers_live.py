#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay optical fingers in an already-running Isaac Sim and measure real joint motion.

Load the stationary Humanoid example first. This test bends each finger on each hand,
opposes each thumb, tests partial tracking loss, and switches to controller triggers.
Only XR hand input is replayed; production finger control and PhysX drives remain live.
Gaze is untouched. The timeline is left paused and live device input is restored.
This is not a validation of the headset's cameras or OpenXR transport.
Use --controllers-only for a short replay of controller fallback and tracking loss.
"""

from __future__ import annotations

import argparse

from kit_exec import DEFAULT_HOST, DEFAULT_PORT, execute

LIVE_CODE = r'''
import asyncio


async def _validate_fingers_live(ex, controllers_only):
    import json
    import math
    import types

    import numpy as np
    import omni.timeline
    from pxr import Gf

    class HandReplay:
        """Generate independent joint positions, including position-only landmarks."""

        def __init__(self, side):
            self.side = side
            self.source = "hand"
            self.curls = {}
            self.opposition = 0.0
            self.invalid = set()
            self.trigger = 0.0
            self.present = True
            self.controller_actions = False

        def positions(self):
            points = {"wrist": Gf.Vec3d(0, 0, 0), "palm": Gf.Vec3d(0, 0.025, 0)}
            lateral = {"index": 0.025, "middle": 0.0, "ring": -0.022, "little": -0.04}
            for role, chain in ex._finger_curl_joint_chains.items():
                if role == "thumb":
                    angle = math.radians(45.0 + 90.0 * self.opposition)
                    direction = Gf.Vec3d(math.cos(angle), math.sin(angle), 0)
                    current = Gf.Vec3d(0.035, 0.015, 0)
                else:
                    direction = Gf.Vec3d(0, 1, 0)
                    current = Gf.Vec3d(lateral[role], 0.02, 0)
                bend = self.curls.get(role, 0.0) * math.radians(95.0 if role == "thumb" else 150.0)
                points[chain[0]] = current
                for index, name in enumerate(chain[1:]):
                    angle = bend * index / (len(chain) - 2)
                    current = current + 0.035 * (
                        direction * math.cos(angle) + Gf.Vec3d(0, 0, -1) * math.sin(angle)
                    )
                    points[name] = current
            if self.side == "right":
                points = {name: Gf.Vec3d(-point[0], point[1], point[2]) for name, point in points.items()}
            return points

        def get_name(self):
            return "/user/hand/" + self.side

        def get_hand_tracking_data_source(self):
            return self.source

        def get_pose_names(self):
            return (
                ["", "grip", "aim", "palm", "pinch", "poke"]
                if self.controller_actions else list(self.positions())
            )

        def get_virtual_world_pose_desc(self, name=""):
            point = self.positions().get(name, Gf.Vec3d(0, 0, 0))
            # Finger positions remain usable without orientation. No wrist pose is
            # activated, so the arms stay at their standing pose during this test.
            flags = 3 if self.controller_actions else 2
            if name in self.invalid:
                flags = 0
            matrix = Gf.Matrix4d(1).SetTranslate(point + Gf.Vec3d(0.25, 0, 1))
            return types.SimpleNamespace(pose_matrix=matrix, validity_flags=flags)

        def get_input_names(self):
            return ["squeeze", "trigger", "thumbstick"] if self.controller_actions else []

        def has_input(self, name):
            return self.controller_actions and name in ("squeeze", "grip", "trigger", "thumbstick")

        def has_input_gesture(self, name, gesture):
            return self.has_input(name) and gesture in ("value", "click")

        def get_input_gesture_value(self, name, gesture):
            return self.trigger if name == "trigger" else 0.0

    timeline = omni.timeline.get_timeline_interface()
    hands = {side: HandReplay(side) for side in ("left", "right")}
    if controllers_only:
        for hand in hands.values():
            hand.source = "controller"
            hand.controller_actions = True
    original_reader = None
    report = {"finger_motion": {}, "thumb_opposition": {}}
    start_time = 0.0
    root_start = None

    def root_position():
        positions, _ = ex.g1.robot.get_world_poses()
        return np.asarray(ex._first_pose_value(positions), dtype=float)

    def health(label):
        assert not ex._physics_step_error_logged, f"{label}: callback errors {ex._physics_step_error_logged}"
        displacement = float(np.linalg.norm(root_position() - root_start))
        assert displacement < 0.001, f"{label}: stationary robot moved {displacement} m"

    async def wait_sim(seconds, label):
        target = ex._headset_gait_time + seconds
        while ex._headset_gait_time + 1e-9 < target:
            await asyncio.sleep(0.02)
            health(label)

    def measured_curls(side):
        roles = ex.g1._finger_dof_indices[side]
        positions = ex.g1.robot.get_dof_positions().numpy()[0]
        result = {}
        for role, index in roles.items():
            opened, closed = ex.g1._finger_open_closed[side][role]
            result[role] = (float(positions[index]) - opened) / (closed - opened)
        assert all(math.isfinite(value) for value in result.values()), "Nonfinite live finger joint"
        return result

    def read_device(handle):
        for side, hand in hands.items():
            if handle == "/user/hand/" + side:
                return hand if hand.present else None
        return original_reader(handle)

    try:
        timeline.pause()
        assert ex is not None and ex.g1 is not None, "Load the Humanoid example first"
        assert ex._g1_locomotion == "stationary", "Use stationary mode for this replay"
        assert ex._xr_core is not None, "Run the XR VR application"
        assert not ex._grabbed_objects_by_side, "Release held objects before validation"
        assert not ex._physics_step_error_logged, "Reload after existing callback errors"
        assert all(len(ex.g1.get_finger_roles(side)) == 6 for side in hands), "Use the five-finger Inspire asset"
        original_reader = ex._get_xr_input_device
        ex._get_xr_input_device = read_device
        start_time = ex._headset_gait_time
        root_start = root_position()
        timeline.play()
        await wait_sim(1.2, "open hands")

        optical_hands = {} if controllers_only else hands
        for side, hand in optical_hands.items():
            report["finger_motion"][side] = {}
            for role in ex._finger_roles:
                hand.curls = {role: 0.7}
                await wait_sim(0.65, f"{side} {role} bend")
                assert ex._finger_curl_source.get(side) == "hand_tracking", f"{side}: optical source lost"
                measured = measured_curls(side)
                assert measured[role] > 0.4, f"{side} {role} failed to bend: {measured}"
                for other in ex._finger_roles:
                    if other != role:
                        assert abs(measured[other]) < 0.2, f"{side} {role} also moved {other}: {measured}"
                report["finger_motion"][side][role] = round(measured[role], 4)
                hand.curls = {}
                await wait_sim(0.45, f"{side} {role} reopen")
            hand.opposition = 0.7
            await wait_sim(0.65, f"{side} thumb opposition")
            measured = measured_curls(side)
            assert measured["thumb_yaw"] > 0.4, f"{side} thumb opposition did not move: {measured}"
            assert abs(measured["thumb"]) < 0.2, f"{side} opposition also flexed thumb: {measured}"
            report["thumb_opposition"][side] = round(measured["thumb_yaw"], 4)
            hand.opposition = 0.0
            hand.curls = {"middle": 0.7, "ring": 0.7}
            await wait_sim(0.65, f"{side} tracking loss setup")
            hand.invalid.add("ring_intermediate")
            await wait_sim(0.65, f"{side} partial tracking loss")
            measured = measured_curls(side)
            assert measured["middle"] > 0.4 and abs(measured["ring"]) < 0.2, f"Bad partial-loss response: {measured}"
            hand.curls = {}
            hand.invalid.clear()
            await wait_sim(0.45, f"{side} reset after tracking loss")

        for hand in hands.values():
            hand.source = "controller"
            hand.controller_actions = True
            hand.trigger = 0.7
        await wait_sim(0.8, "controller fallback")
        for side in hands:
            assert ex._finger_curl_source.get(side) == "controller", f"{side}: no controller fallback"
            measured = measured_curls(side)
            assert all(measured[role] > 0.3 for role in ex._finger_roles), f"{side}: controller fingers failed {measured}"
        # Match the real SteamVR/Kit observation after enabling hand tracking:
        # metadata says hand, while only Touch interaction poses/actions exist.
        for hand in hands.values():
            hand.source = "hand"
            hand.trigger = 0.5
        await wait_sim(0.8, "Touch controller with premature hand metadata")
        for side in hands:
            assert ex._finger_curl_source.get(side) == "controller", f"{side}: hand label blocked Touch inputs"
            measured = measured_curls(side)
            assert all(measured[role] > 0.3 for role in ex._finger_roles), f"{side}: mislabeled Touch failed {measured}"
        for hand in hands.values():
            hand.present = False
        await wait_sim(0.8, "full tracking loss")
        for side in hands:
            assert ex._finger_curl_source.get(side) == "none", f"{side}: stale input source"
            measured = measured_curls(side)
            assert all(abs(value) < 0.2 for value in measured.values()), f"{side}: lost tracking froze fingers {measured}"
        health("completed finger replay")
        report.update(
            status="passed",
            simulated_seconds=round(ex._headset_gait_time - start_time, 3),
            root_translation_m=float(np.linalg.norm(root_position() - root_start)),
            optical_finger_sequence="skipped (--controllers-only)" if controllers_only else "passed both hands",
            optical_thumb_opposition="skipped (--controllers-only)" if controllers_only else "passed both hands",
            partial_tracking_loss="skipped (--controllers-only)" if controllers_only else "passed both hands",
            controller_fallback="passed both hands",
            premature_hand_metadata="passed both Touch controllers",
            full_tracking_loss="passed both hands",
            input=(
                "synthetic controller poses/actions; real production callbacks and PhysX finger joints"
                if controllers_only else "synthetic joint positions/controllers; real production callbacks and PhysX finger joints"
            ),
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    finally:
        timeline.pause()
        if original_reader is not None:
            ex._get_xr_input_device = original_reader
            ex._drop_everything()
            ex._latest_finger_curls.clear()
            ex._smoothed_finger_curls.clear()
            ex._finger_curl_source.clear()
            for side in hands:
                ex._deactivate_hand(side)
                ex.g1.set_finger_curls(side, {role: 0.0 for role in (*ex._finger_roles, "thumb_yaw")})


async def _run_finger_validation_with_timeout():
    await asyncio.wait_for(_validate_fingers_live(EX, __CONTROLLERS_ONLY__), timeout=__EXECUTION_TIMEOUT__)


await asyncio.ensure_future(_run_finger_validation_with_timeout())
'''


def main() -> int:
    """Send the optical replay to the existing Kit process through its Python server."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--controllers-only",
        action="store_true",
        help="Skip optical fingers, thumb opposition, and partial skeleton loss",
    )
    args = parser.parse_args()
    if args.timeout <= 5.0:
        parser.error("--timeout must be greater than five seconds")
    code = LIVE_CODE.replace("__EXECUTION_TIMEOUT__", repr(args.timeout - 5.0)).replace(
        "__CONTROLLERS_ONLY__", repr(args.controllers_only)
    )
    return execute(code, args.host, args.port, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
