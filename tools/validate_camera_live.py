#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the robot-mounted camera in an already-running Isaac Sim.

Load the Humanoid example through tools/launch_isaac_vr.bat, then run this host CLI
with ordinary Python. It checks real body transforms, the real USD camera, and XR
camera scheduling during Play and Pause. Synthetic physical headset poses cover
translation, yaw, pitch, and roll; the script never changes the gaze tracker.
The mount is checked relative to the measured robot body, so physical torso
deflection under arm contact is distinguished from a headset-driven camera offset.

The XR proxy forwards every scheduled camera to the actual runtime. This validates
application camera control, not a physical headset's compositor or tracking quality.
The script restores its input overrides and leaves the timeline paused.
"""

from __future__ import annotations

import argparse

from kit_exec import DEFAULT_HOST, DEFAULT_PORT, execute

# These imports and the async replay belong to Kit's Python process.
LIVE_CODE = r"""
async def _validate_camera_live(ex):
    import json
    import math
    import types

    import numpy as np
    import omni.kit.app
    import omni.timeline
    from pxr import Gf

    assert ex is not None and ex.g1 is not None, "Load the Humanoid example first"
    assert ex._g1_locomotion == "stationary", "This replay requires stationary manipulation mode"
    assert ex._xr_camera_mode == "robot_head", "Load the updated robot-mounted camera implementation"
    assert ex._head_camera_update_sub is not None, "Missing application-frame camera subscription"
    assert ex._head_camera_mount_local is not None, "Camera mount was not prepared"
    assert ex._head_camera_mount_body_path, "Camera has no robot body mount"
    assert ex._xr_core is not None, "XRCore must be loaded to validate runtime camera scheduling"

    class XRForwarder:
        def __init__(self, runtime):
            self.runtime = runtime
            self.scheduled = []

        def __getattr__(self, name):
            return getattr(self.runtime, name)

        def schedule_set_camera(self, matrix):
            result = self.runtime.schedule_set_camera(matrix)
            self.scheduled.append(Gf.Matrix4d(matrix))
            return result

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    original_core = ex._xr_core
    original_reader = ex.__dict__.get("_read_physical_head_pose")
    reader_was_overridden = "_read_physical_head_pose" in ex.__dict__
    proxy = XRForwarder(original_core)
    tracker = ex._eye_gaze_tracker
    errors_before = set(ex._physics_step_error_logged)
    fake_head = Gf.Matrix4d(1.0)
    physical_reads = 0
    max_view_delta = 0.0
    max_body_delta = 0.0
    max_mount_error = 0.0
    max_mount_local_delta = 0.0
    max_root_translation = 0.0
    max_root_rotation = 0.0
    report = {"mount_body": ex._head_camera_mount_body_path, "phases": {}}

    def read_fake_head():
        nonlocal physical_reads
        physical_reads += 1
        return Gf.Matrix4d(fake_head)

    def root_pose():
        positions, orientations = ex.g1.robot.get_world_poses()
        position = np.asarray(ex._first_pose_value(positions), dtype=float)
        orientation = np.asarray(ex._first_pose_value(orientations), dtype=float)
        assert position.shape == (3,) and orientation.shape == (4,), "Invalid root pose"
        assert np.isfinite(position).all() and np.isfinite(orientation).all(), "Nonfinite root pose"
        orientation /= np.linalg.norm(orientation)
        return position, orientation

    async def frames(count):
        for _ in range(count):
            await app.next_update_async()

    def check_pose(baseline_view, baseline_body, baseline_mount, baseline_root):
        nonlocal max_view_delta, max_body_delta, max_mount_error, max_mount_local_delta
        nonlocal max_root_translation, max_root_rotation
        reads_before = physical_reads
        ex._on_robot_camera_update(types.SimpleNamespace(payload={"dt": 0.0}))
        assert physical_reads == reads_before, "Camera update consumed a physical headset pose"
        body = ex._read_camera_mount_body_pose()
        assert body is not None, "Robot mount lost its live body pose"
        view = ex._head_camera_transform_op.Get()
        expected = ex._head_camera_mount_local * body
        assert proxy.scheduled, "No camera pose was forwarded to XRCore"
        mount_error = float(np.max(np.abs(np.asarray(view) - np.asarray(expected))))
        schedule_error = float(np.max(np.abs(np.asarray(proxy.scheduled[-1]) - np.asarray(expected))))
        max_mount_error = max(max_mount_error, mount_error, schedule_error)
        assert mount_error < 1e-5 and schedule_error < 1e-5, "Viewport/XR camera detached from the robot body"
        # Contact forces can deflect the torso even with an anchored pelvis. A
        # rigid camera must follow that body motion while retaining its mount.
        camera_local = view * body.GetInverse()
        scheduled_local = proxy.scheduled[-1] * body.GetInverse()
        local_delta = float(np.max(np.abs(np.asarray(camera_local) - np.asarray(baseline_mount))))
        scheduled_local_delta = float(np.max(np.abs(np.asarray(scheduled_local) - np.asarray(baseline_mount))))
        max_mount_local_delta = max(max_mount_local_delta, local_delta, scheduled_local_delta)
        assert local_delta < 1e-5 and scheduled_local_delta < 1e-5, "Headset motion changed the camera/body offset"
        view_delta = float(np.max(np.abs(np.asarray(view) - np.asarray(baseline_view))))
        max_view_delta = max(max_view_delta, view_delta)
        body_delta = float(np.max(np.abs(np.asarray(body) - np.asarray(baseline_body))))
        max_body_delta = max(max_body_delta, body_delta)
        position, orientation = root_pose()
        translation = float(np.linalg.norm(position - baseline_root[0]))
        rotation = 2 * math.acos(min(1.0, abs(float(np.dot(orientation, baseline_root[1])))))
        max_root_translation = max(max_root_translation, translation)
        max_root_rotation = max(max_root_rotation, rotation)
        assert translation < 1e-4 and rotation < 1e-3, "Stationary robot root moved"
        new_errors = set(ex._physics_step_error_logged) - errors_before
        assert not new_errors, f"Runtime callback errors: {new_errors}"
        assert ex._eye_gaze_tracker is tracker, "Gaze tracker was replaced"

    head_poses = (
        ((0, 0, 1.7), 0, 0, 0),
        ((0.7, -0.5, 1.2), 0, 0, 0),
        ((-1.1, 0.9, 1.95), 90, 0, 0),
        ((0.2, -0.3, 1.4), -65, 45, 0),
        ((0.6, 0.8, 1.8), 145, -35, 40),
        ((0, 0, 1.7), -170, 60, -55),
    )
    try:
        ex._xr_core = proxy
        ex._read_physical_head_pose = read_fake_head
        timeline.play()
        await frames(30)
        assert ex._physics_ready and ex.g1.robot.is_physics_tensor_entity_valid(), "Physics failed to initialize"
        ex._on_robot_camera_update(types.SimpleNamespace(payload={"dt": 0.0}))
        baseline_view = Gf.Matrix4d(ex._head_camera_transform_op.Get())
        baseline_body = ex._read_camera_mount_body_pose()
        assert baseline_body is not None, "Camera mount has no initial live body pose"
        baseline_mount = Gf.Matrix4d(ex._head_camera_mount_local)
        baseline_root = root_pose()
        for phase in ("playing", "paused"):
            if phase == "paused":
                timeline.pause()
                await frames(3)
            first_schedule = len(proxy.scheduled)
            for index, (position, yaw, pitch, roll) in enumerate(head_poses):
                fake_head = (
                    Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), roll))
                    * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), pitch))
                    * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw))
                )
                fake_head.SetTranslateOnly(Gf.Vec3d(*position))
                previous_schedule = len(proxy.scheduled)
                await frames(4)
                assert len(proxy.scheduled) > previous_schedule, f"No automatic camera refresh while {phase}"
                check_pose(baseline_view, baseline_body, baseline_mount, baseline_root)
                if index % 2 == 0:
                    ex._request_xr_recenter()
                else:
                    ex._cycle_xr_camera_mode()
                assert ex._xr_camera_mode == "robot_head", "Controller button unlocked the mounted camera"
                check_pose(baseline_view, baseline_body, baseline_mount, baseline_root)
            report["phases"][phase] = {
                "head_poses": len(head_poses),
                "forwarded_xr_camera_updates": len(proxy.scheduled) - first_schedule,
            }
        report.update(
            max_camera_matrix_delta=max_view_delta,
            max_body_matrix_delta=max_body_delta,
            max_mount_matrix_error=max_mount_error,
            max_mount_local_matrix_delta=max_mount_local_delta,
            max_root_translation_m=max_root_translation,
            max_root_rotation_rad=max_root_rotation,
            physical_head_reads_outside_camera_update=physical_reads,
            gaze_tracker_preserved=True,
            callback_errors=[],
            hardware_tracking_tested=False,
        )
        print("CAMERA LIVE VALIDATION PASSED " + json.dumps(report, sort_keys=True))
    finally:
        timeline.pause()
        ex._xr_core = original_core
        if reader_was_overridden:
            ex._read_physical_head_pose = original_reader
        else:
            ex.__dict__.pop("_read_physical_head_pose", None)
        ex._update_head_camera_view(force=True, dt=0.0)


await _validate_camera_live(EX)
"""


def main() -> int:
    """Send the camera replay to the running Kit application's Python server."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    return execute(LIVE_CODE, args.host, args.port, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
