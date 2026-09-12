#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure teleoperation performance inside an already-running Isaac Sim session.

Enable Python Server, LOAD the Humanoid example, and press Play before running:

    python tools/profile_teleop_live.py --seconds 20 --timeout 45

This command observes real input and temporarily times selected Python methods.
It never changes gaze, camera, recording settings, input devices, or the timeline.
Wrappers are restored in finally, including on cancellation or a server timeout.
The JSON report measures instrumented app updates, not headset/compositor FPS.
Python timing adds overhead; compare runs with the same instrumentation and scene.
"""

from __future__ import annotations

import argparse
import math

from kit_exec import DEFAULT_HOST, DEFAULT_PORT, execute

LIVE_CODE = r"""
import asyncio
import json
import math
import time

import omni.kit.app
import omni.timeline


async def _profile_teleop_live(ex, seconds):
    timeline = omni.timeline.get_timeline_interface()
    started = time.perf_counter()
    start_sim = float(getattr(ex, "_headset_gait_time", 0.0))
    robot = getattr(ex, "g1", None)
    restored = []
    subscription = None
    updates = [0]
    physics_dts = []
    stats = {}
    report = {
        "status": "inconclusive",
        "requested_wall_seconds": seconds,
        "input": "unmodified live input",
        "playing_at_start": timeline.is_playing(),
        "timeline_changed_by_profiler": False,
        "gaze_camera_recording_settings_changed": False,
        "configured_physics_dt": getattr(ex, "_world_settings", {}).get("physics_dt"),
        "preexisting_callback_errors": sorted(str(key) for key in getattr(ex, "_physics_step_error_logged", ())),
        "instrumentation_note": (
            "Temporary Python timing wrappers add overhead. Durations include nested calls; "
            "do not sum them as disjoint frame costs. App-update FPS is not headset/compositor FPS. "
            "Gaze and camera methods are not instrumented."
        ),
    }

    def on_update(event):
        updates[0] += 1

    def wrap_method(owner, name, label):
        original = getattr(owner, name)
        if not callable(original):
            raise TypeError(f"{label} is not callable")
        own_attributes = vars(owner)
        had_own_attribute = name in own_attributes
        own_value = own_attributes.get(name)
        record = {"name": label, "calls": 0, "exceptions": 0, "total_seconds": 0.0, "max_seconds": 0.0}
        stats[label] = record

        def timed(*args, **kwargs):
            begin = time.perf_counter()
            try:
                return original(*args, **kwargs)
            except BaseException:
                record["exceptions"] += 1
                raise
            finally:
                elapsed = time.perf_counter() - begin
                record["calls"] += 1
                record["total_seconds"] += elapsed
                record["max_seconds"] = max(record["max_seconds"], elapsed)
                if label == "g1.forward":
                    value = args[0] if args else kwargs.get("dt")
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        value = 0.0
                    if math.isfinite(value) and value > 0:
                        physics_dts.append(value)

        setattr(owner, name, timed)
        # Preserve descriptors: inherited methods must not remain shadowed on the instance.
        restored.append((owner, name, had_own_attribute, own_value))

    try:
        if ex is None or robot is None or not getattr(ex, "_physics_ready", False):
            report["reason"] = "Load the Humanoid example, press Play, and wait for physics initialization."
            return
        if not timeline.is_playing():
            report["reason"] = "The timeline is paused or stopped; no automatic Play was performed."
            return

        for method in (
            "_update_controller_command",
            "_update_g1_fingers",
            "_update_g1_arms_from_hand_tracking",
            "_collect_all_behavioral_data",
        ):
            wrap_method(ex, method, method)
        wrap_method(robot, "forward", "g1.forward")
        subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            on_update, name="G1 temporary teleoperation performance observation"
        )
        # Exclude wrapper installation from the measured interval.
        started = time.perf_counter()
        start_sim = float(ex._headset_gait_time)
        while time.perf_counter() - started < seconds:
            await asyncio.sleep(min(0.1, max(0.0, seconds - (time.perf_counter() - started))))
            if not timeline.is_playing():
                report["reason"] = (
                    "The timeline paused or stopped during observation; the partial interval is reported."
                )
                return
            if ex.g1 is not robot or not ex._physics_ready:
                report["reason"] = "The robot or physics initialization changed during observation."
                return
        report["status"] = "measured"
    except asyncio.CancelledError:
        report["reason"] = "Observation cancelled or server execution deadline reached."
        raise
    except Exception as error:
        report["status"] = "error"
        report["reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        ended = time.perf_counter()
        # Releasing the carb subscription removes the update callback.
        subscription = None
        restore_errors = []
        for owner, name, had_own_attribute, own_value in reversed(restored):
            try:
                if had_own_attribute:
                    setattr(owner, name, own_value)
                else:
                    delattr(owner, name)
            except Exception as error:
                restore_errors.append(f"{name}: {type(error).__name__}: {error}")

        try:
            end_sim = float(getattr(ex, "_headset_gait_time", start_sim))
        except (TypeError, ValueError):
            end_sim = start_sim
        wall_seconds = ended - started
        simulated_seconds = end_sim - start_sim
        report.update(
            wall_seconds=round(wall_seconds, 6),
            simulated_seconds=round(simulated_seconds, 6),
            real_time_factor=simulated_seconds / wall_seconds if wall_seconds > 0 and simulated_seconds >= 0 else None,
            app_updates=updates[0],
            app_update_fps=updates[0] / wall_seconds if wall_seconds > 0 else None,
            physics_steps=len(physics_dts),
            observed_physics_dt_min=min(physics_dts) if physics_dts else None,
            observed_physics_dt_max=max(physics_dts) if physics_dts else None,
            observed_physics_dt_mean=sum(physics_dts) / len(physics_dts) if physics_dts else None,
            observed_physics_seconds=sum(physics_dts),
            playing_at_end=timeline.is_playing(),
            callback_errors=sorted(str(key) for key in getattr(ex, "_physics_step_error_logged", ())),
            wrappers_restored=not restore_errors,
            restore_errors=restore_errors,
        )
        if restore_errors:
            report["status"] = "error"
            report["reason"] = "One or more temporary wrappers could not be restored; inspect restore_errors."
        elif report["status"] == "measured" and (simulated_seconds <= 0 or not physics_dts or not updates[0]):
            report["status"] = "inconclusive"
            report["reason"] = "Physics or application updates did not advance, or the simulation clock reset."
        report["subsystems"] = [
            {
                "name": record["name"],
                "calls": record["calls"],
                "exceptions": record["exceptions"],
                "total_ms": record["total_seconds"] * 1000,
                "mean_ms": record["total_seconds"] * 1000 / record["calls"] if record["calls"] else None,
                "max_ms": record["max_seconds"] * 1000 if record["calls"] else None,
                "wall_percent": record["total_seconds"] / wall_seconds * 100 if wall_seconds > 0 else None,
            }
            for record in sorted(stats.values(), key=lambda item: item["total_seconds"], reverse=True)
        ]
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)


async def _run_profile_with_timeout():
    await asyncio.wait_for(_profile_teleop_live(EX, __SECONDS__), timeout=__EXECUTION_TIMEOUT__)


await asyncio.ensure_future(_run_profile_with_timeout())
"""


def build_live_code(seconds: float, timeout: float) -> str:
    """Build trusted remote source with a deadline before the host socket expires.

    Args:
        seconds: Requested wall-clock observation interval.
        timeout: Host socket timeout, including time to restore wrappers and return JSON.

    Returns:
        Python source for the existing Kit Python server.
    """
    return LIVE_CODE.replace("__SECONDS__", repr(seconds)).replace("__EXECUTION_TIMEOUT__", repr(timeout - 5.0))


def main() -> int:
    """Profile an active scene without supplying inputs or changing its settings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seconds", type=float, default=20.0, help="Wall-clock interval, between 0.1 and 120 seconds")
    parser.add_argument(
        "--timeout", type=float, default=45.0, help="Host socket timeout; server deadline is five seconds less"
    )
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0.1 <= args.seconds <= 120.0:
        parser.error("--seconds must be between 0.1 and 120")
    if not math.isfinite(args.timeout) or args.timeout <= 5.0:
        parser.error("--timeout must be finite and greater than five seconds")
    return execute(build_live_code(args.seconds, args.timeout), args.host, args.port, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
