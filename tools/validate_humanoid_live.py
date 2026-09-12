#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay stationary humanoid controls inside an already-running Isaac Sim.

Run this host CLI with ordinary Python, after loading the Humanoid example and
enabling isaacsim.code_editor.python_server (tools/launch_isaac_vr.bat does this).
The test moves the simulated right arm, picks up Box_00, and releases it. It leaves
the timeline paused. --reload-example explicitly clears/rebuilds the example for
fresh package positions; save any stage edits first. It does not hot-reload modules.

Only the XR input-device boundary and gaze-highlight selection are replayed. Arm
retargeting, joint drives, fixed-joint pickup, collisions, and physics callbacks run
in the actual Kit application. This does not validate headset hardware tracking.
"""

from __future__ import annotations

import argparse

from kit_exec import DEFAULT_HOST, DEFAULT_PORT, execute

# Kept as source because imports below belong to Kit's Python environment, not to
# the ordinary host interpreter. All replay state is local to the async function.
LIVE_CODE = r'''
import asyncio


async def _validate_humanoid_live(ex, reload_example):
    import json
    import math
    import types

    import numpy as np
    import omni.timeline
    from pxr import Gf

    class ControllerReplay:
        """Supply validated stage-space controller poses and analog buttons."""

        def __init__(self, side):
            self.side = side
            self.grip = 0.0
            self.trigger = 0.0
            self.pose = Gf.Matrix4d(1.0).SetTranslate(
                Gf.Vec3d(0.3, -0.3 if side == "right" else 0.3, 1.0)
            )

        def get_name(self):
            return "/user/hand/" + self.side

        def get_hand_tracking_data_source(self):
            return "controller"

        def get_pose_names(self):
            return ["grip", "aim"]

        def get_virtual_world_pose_desc(self, name=""):
            return types.SimpleNamespace(pose_matrix=self.pose, validity_flags=3)

        def has_input(self, name):
            return name in ("squeeze", "grip", "trigger")

        def has_input_gesture(self, name, gesture):
            return gesture in ("value", "click")

        def get_input_gesture_value(self, name, gesture):
            return self.trigger if name == "trigger" else self.grip

        def get_input_names(self):
            return ["squeeze", "trigger"]

    timeline = omni.timeline.get_timeline_interface()
    original_reader = None
    tracker = None
    original_update = None
    left, right = ControllerReplay("left"), ControllerReplay("right")
    ticks = 0
    simulated_time = 0.0
    highlight_start = None
    root_start = None
    max_root_translation = 0.0
    max_root_rotation = 0.0
    box_path = "/World/G1_SampleBoxes/Box_00"
    report = {}

    def root_pose():
        positions, orientations = ex.g1.robot.get_world_poses()
        position = np.asarray(ex._first_pose_value(positions), dtype=float)
        orientation = np.asarray(ex._first_pose_value(orientations), dtype=float)
        assert position.shape == (3,) and orientation.shape == (4,), "Invalid robot root pose shape"
        assert np.isfinite(position).all() and np.isfinite(orientation).all(), "Nonfinite robot root pose"
        norm = float(np.linalg.norm(orientation))
        assert norm > 1e-12, "Invalid zero-length robot root quaternion"
        orientation /= norm
        return position, orientation

    def check_health(label):
        nonlocal max_root_translation, max_root_rotation
        assert not ex._physics_step_error_logged, f"{label}: callback errors {ex._physics_step_error_logged}"
        if root_start is None:
            return
        position, orientation = root_pose()
        translation = float(np.linalg.norm(position - root_start[0]))
        alignment = min(1.0, abs(float(np.dot(orientation, root_start[1]))))
        rotation = 2.0 * math.acos(alignment)
        max_root_translation = max(max_root_translation, translation)
        max_root_rotation = max(max_root_rotation, rotation)
        assert translation < 0.001, f"{label}: stationary root moved {translation:.6f} m"
        assert rotation < 0.001, f"{label}: stationary root rotated {rotation:.6f} rad"

    async def wait_sim(seconds, label):
        start_ticks = ticks
        target = simulated_time + seconds
        while simulated_time + 1e-9 < target:
            await asyncio.sleep(0.02)
            check_health(label)
        assert ticks > start_ticks, f"{label}: physics did not advance"

    def replay_update(dt):
        nonlocal ticks, simulated_time
        ticks += 1
        simulated_time += float(dt)
        if highlight_start is None:
            original_update(dt)
            return
        transition = ticks - highlight_start
        # Run inside the normal physics callback. Back-to-back host USD writes can
        # coalesce notices and miss the original collision-prim invalidation bug.
        if transition == 1:
            tracker._update_highlight("/World/G1_SampleBoxes/Box_03/Asset")
        elif transition == 20:
            tracker._update_highlight(box_path + "/Asset")
        elif transition == 40:
            ex._set_grab_candidate("right", box_path)
        elif transition == 60:
            tracker._update_highlight(None)
            ex._set_grab_candidate("right", None)

    def object_position():
        result = ex._get_object_world_pose(box_path)
        assert result is not None, f"Missing live package pose: {box_path}"
        return result[0]

    try:
        timeline.pause()
        assert ex is not None, "Open the Humanoid example in the examples browser first"
        assert ex._grasp_mode == "assisted", (
            "This legacy replay requires assisted mode selected before LOAD. "
            "Use tools/validate_contact_grasp_live.py for default physical grasping."
        )
        if reload_example:
            await ex.clear_async()
            await ex.load_world_async()
        assert ex.g1 is not None, "LOAD the Humanoid example or pass --reload-example"
        assert ex._g1_locomotion == "stationary", "This replay requires the default stationary mode"
        assert ex._xr_core is not None, "XRCore is unavailable; run the XR VR application"
        tracker = ex._eye_gaze_tracker
        assert tracker is not None, "The example's gaze tracker must be initialized"
        assert not ex._grabbed_objects_by_side, "Release currently held objects before validation"
        assert not ex._physics_step_error_logged, "Existing callback errors: reload the example first"
        original_reader = ex._get_xr_input_device
        original_update = tracker.update
        ex._get_xr_input_device = lambda handle: (
            right if handle == "/user/hand/right"
            else left if handle == "/user/hand/left"
            else original_reader(handle)
        )
        tracker.update = replay_update
        timeline.play()
        await wait_sim(0.5, "initialization")
        root_start = root_pose()

        right.grip = 1.0
        await wait_sim(0.8, "grip hold")
        right.pose.SetTranslateOnly(Gf.Vec3d(0.4, -0.3, 1.08))
        await wait_sim(1.0, "controller movement")
        right.grip = 0.0
        await wait_sim(0.4, "grip release")
        assert "right" not in ex._arm_input_sources, "Grip release left arm tracking active"
        assert not ex._grabbed_objects_by_side, "Grip-only replay unexpectedly attached an object"
        print("PASS grip hold/movement/release", flush=True)

        highlight_start = ticks
        while ticks - highlight_start < 65:
            await asyncio.sleep(0.02)
            check_health("highlight transitions")
        report["highlight_ticks"] = ticks - highlight_start
        highlight_start = None
        print("PASS in-callback gaze/grip highlight transitions", flush=True)

        right.pose.SetTranslateOnly(Gf.Vec3d(0.3, -0.3, 1.0))
        right.grip = 1.0
        await wait_sim(0.2, "pickup clutch")
        assert "right" in ex._controller_arm_neutral_targets, "Controller clutch did not calibrate"
        neutral_controller = right.pose.ExtractTranslation()
        base_position, yaw = ex._get_g1_base_pose_for_arms()
        neutral_palm = ex._g1_body_point_to_stage(
            ex._controller_arm_neutral_targets["right"], base_position, yaw
        )

        def set_palm_target(target):
            delta = ex._stage_vector_to_g1_body(target - neutral_palm, yaw)
            controller_delta = Gf.Vec3d(delta[0] / 1.3, delta[1] / 1.2, delta[2] / 1.5)
            world_delta = ex._g1_body_point_to_stage(controller_delta, Gf.Vec3d(0), yaw)
            right.pose.SetTranslateOnly(neutral_controller + world_delta)

        target_object = object_position()
        set_palm_target(Gf.Vec3d(neutral_palm[0], neutral_palm[1], 1.02))
        await wait_sim(1.8, "raise hand")
        set_palm_target(Gf.Vec3d(target_object[0], target_object[1], 1.02))
        await wait_sim(1.8, "move above package")
        set_palm_target(target_object + Gf.Vec3d(0, 0, 0.015))
        await wait_sim(2.0, "reach package")
        report["finger_gap_m"] = float(ex._grasp_contact_gap("right", object_position()))
        right.trigger = 1.0
        await wait_sim(0.8, "trigger close")
        assert ex._grabbed_objects_by_side.get("right") == box_path, "Controller did not attach Box_00"
        attached_z = float(object_position()[2])
        set_palm_target(target_object + Gf.Vec3d(0, 0, 0.18))
        await wait_sim(2.0, "lift package")
        lifted_z = float(object_position()[2])
        report["lift_m"] = lifted_z - attached_z
        assert report["lift_m"] > 0.03, f"Attached package did not lift: {report['lift_m']:.4f} m"
        right.trigger = 0.0
        await wait_sim(0.8, "trigger release")
        assert not ex._grabbed_objects_by_side, "Trigger release did not drop the package"
        assert not ex._grasp_joints_by_side, "Trigger release left a grasp joint"
        check_health("completed replay")
        assert ticks >= 1000, f"Too few simulated ticks to validate the full replay: {ticks}"
        report.update(
            status="passed",
            physics_ticks=ticks,
            simulated_seconds=round(simulated_time, 3),
            max_root_translation_m=max_root_translation,
            max_root_rotation_rad=max_root_rotation,
            callback_errors=[],
            xr_input="synthetic controllers; real PhysX and production callbacks",
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    finally:
        # Restore the input boundary even if an assertion, timeout or physics read
        # fails. Pause first so no further callback can consume a replayed grip.
        timeline.pause()
        right.grip = right.trigger = left.grip = left.trigger = 0.0
        if original_reader is not None:
            ex._get_xr_input_device = original_reader
        if tracker is not None and original_update is not None:
            tracker.update = original_update
        if original_reader is not None:
            ex._drop_everything()
            for side in ("left", "right"):
                ex._deactivate_hand(side)
            tracker._update_highlight(None)


async def _run_humanoid_validation_with_timeout():
    # Kit's server drives top-level coroutines manually. wait_for needs a real
    # asyncio Task, so create one explicitly at the server's await boundary.
    await asyncio.wait_for(_validate_humanoid_live(EX, __RELOAD_EXAMPLE__), timeout=__EXECUTION_TIMEOUT__)


await asyncio.ensure_future(_run_humanoid_validation_with_timeout())
'''


def build_live_code(reload_example: bool, timeout: float) -> str:
    """Build Kit source while leaving time for cleanup before the host socket expires."""
    return LIVE_CODE.replace("__RELOAD_EXAMPLE__", repr(reload_example)).replace(
        "__EXECUTION_TIMEOUT__", repr(max(1.0, timeout - 5.0))
    )


def main() -> int:
    """Send the integration replay to the existing application's Python server."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=60.0, help="Host socket timeout in seconds (default: 60)")
    parser.add_argument(
        "--reload-example",
        action="store_true",
        help="Clear/rebuild the example for fresh package positions; discards current stage edits",
    )
    args = parser.parse_args()
    if args.timeout <= 5.0:
        parser.error("--timeout must be greater than five seconds")
    return execute(build_live_code(args.reload_example, args.timeout), args.host, args.port, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
