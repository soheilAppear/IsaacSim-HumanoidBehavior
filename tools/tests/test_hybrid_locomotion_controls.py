# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise hybrid ownership and the real neutral/look-away interlocks without Kit."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType

MODULE_DIR = (
    Path(__file__).resolve().parents[2]
    / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"
)


def load_module(name: str) -> ModuleType:
    """Load a pure control module without importing the extension package."""
    spec = importlib.util.spec_from_file_location(f"hybrid_test_{name}", MODULE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HYBRID = load_module("hybrid_locomotion")
CONTROLLER = load_module("controller_locomotion")
GAZE = load_module("gaze_locomotion")


class TestHybridLocomotionControls(unittest.TestCase):
    def setUp(self) -> None:
        self.hybrid = HYBRID.HybridLocomotion()

    def select(self, **kwargs: object) -> HYBRID.HybridSelection:
        values = dict(
            controller_present=True,
            controller_valid=True,
            controller_state="idle",
            sticks_deflected=False,
        )
        values.update(kwargs)
        result = self.hybrid.select(**values)
        self.assertIs(result, self.hybrid.selection)
        return result

    def claim(self) -> HYBRID.HybridSelection:
        return self.select(controller_state="active", sticks_deflected=True)

    def test_absent_controllers_allow_gaze_at_startup(self) -> None:
        selection = self.select(controller_present=False, controller_valid=False, controller_state="unavailable")
        self.assertEqual(selection.source, "gaze")
        self.assertTrue(selection.reset_gaze)
        self.assertFalse(selection.stop_immediately)
        self.assertFalse(self.hybrid.status["controller_claimed"])

    def test_mixed_neutral_controller_and_optical_hand_allow_gaze(self) -> None:
        for state in ("unavailable", "awaiting_neutral"):
            with self.subTest(state=state):
                self.hybrid.reset()
                for _ in range(3):
                    selection = self.select(controller_valid=False, controller_state=state)
                    self.assertEqual(selection.source, "gaze")
                    self.assertFalse(selection.stop_immediately)

    def test_unarmed_neutral_controllers_do_not_claim_gaze(self) -> None:
        self.assertEqual(self.select(controller_state="awaiting_neutral").source, "gaze")
        self.assertFalse(self.select(controller_state="awaiting_neutral").reset_gaze)

    def test_armed_deflection_selects_only_controller_and_clears_old_filter_once(self) -> None:
        self.select()
        first = self.claim()
        self.assertEqual(first.source, "controller")
        self.assertTrue(first.reset_gaze)
        self.assertFalse(first.stop_immediately)
        for _ in range(3):
            held = self.claim()
            self.assertEqual(held.source, "controller")
            self.assertFalse(held.reset_gaze)
            self.assertFalse(held.stop_immediately)

    def test_deflected_unarmed_sticks_suppress_gaze_without_bypassing_neutral(self) -> None:
        for _ in range(3):
            selection = self.select(controller_state="awaiting_neutral", sticks_deflected=True)
            self.assertEqual(selection.source, "none")
            self.assertTrue(selection.stop_immediately)
        self.assertTrue(self.hybrid.status["controller_claimed"])
        self.assertTrue(self.select().stop_immediately)
        self.assertEqual(self.select().source, "gaze")

    def test_partial_tracking_with_one_valid_deflection_suppresses_gaze(self) -> None:
        selection = self.select(controller_valid=False, controller_state="unavailable", sticks_deflected=True)
        self.assertEqual(selection.source, "none")
        self.assertTrue(selection.reset_gaze)
        self.assertTrue(selection.stop_immediately)
        repeated = self.select(controller_valid=False, controller_state="unavailable", sticks_deflected=True)
        self.assertFalse(repeated.reset_gaze)
        self.assertEqual(repeated.source, "none")

    def test_neutral_release_inserts_zero_handoff_before_gaze(self) -> None:
        self.claim()
        release = self.select()
        self.assertEqual(release.source, "none")
        self.assertTrue(release.reset_gaze)
        self.assertTrue(release.stop_immediately)
        self.assertFalse(self.hybrid.status["controller_claimed"])
        resumed = self.select()
        self.assertEqual(resumed.source, "gaze")
        self.assertFalse(resumed.reset_gaze)

    def test_tracking_loss_after_claim_stops_with_or_without_controller_handle(self) -> None:
        for present in (False, True):
            with self.subTest(present=present):
                self.hybrid.reset()
                self.claim()
                values = dict(controller_present=present, controller_valid=False, controller_state="unavailable")
                lost = self.select(**values)
                self.assertEqual(lost.source, "none")
                self.assertTrue(lost.reset_gaze)
                self.assertTrue(lost.stop_immediately)
                self.assertEqual(self.select(**values).source, "gaze")

    def test_loss_with_remaining_deflected_axis_keeps_gaze_suspended(self) -> None:
        self.claim()
        values = dict(controller_valid=False, controller_state="unavailable", sticks_deflected=True)
        lost = self.select(**values)
        self.assertEqual(lost.source, "none")
        self.assertTrue(lost.reset_gaze)
        for _ in range(3):
            self.assertEqual(self.select(**values).source, "none")
        self.assertEqual(self.select(controller_valid=False, controller_state="unavailable").source, "none")
        self.assertEqual(self.select(controller_valid=False, controller_state="unavailable").source, "gaze")

    def test_recovered_active_sticks_clear_a_previous_source_filter(self) -> None:
        self.claim()
        self.select(controller_valid=False, controller_state="unavailable", sticks_deflected=True)
        recovered = self.claim()
        self.assertEqual(recovered.source, "controller")
        self.assertTrue(recovered.reset_gaze)
        self.assertFalse(recovered.stop_immediately)

    def test_real_controller_neutral_interlock_and_sample_gap_remain_effective(self) -> None:
        for axis in ("left_x", "left_y", "right_x"):
            with self.subTest(axis=axis):
                self.hybrid.reset()
                controller = CONTROLLER.ControllerLocomotion()

                def sample(now_s: float, value: float) -> HYBRID.HybridSelection:
                    axes = dict(left_x=0.0, left_y=0.0, right_x=0.0)
                    axes[axis] = value
                    command = controller.update(**axes, left_valid=True, right_valid=True, now_s=now_s)
                    return self.select(
                        controller_state=command.state, sticks_deflected=abs(value) > controller.deadzone
                    )

                self.assertEqual(sample(0.01, 0.8).source, "none")
                self.assertEqual(controller.command.velocity, (0, 0, 0))
                self.assertEqual(sample(0.02, 0).source, "none")
                self.assertEqual(sample(0.03, 0.8).source, "controller")
                self.assertTrue(any(controller.command.velocity))
                self.assertEqual(sample(0.50, 0.8).source, "none")
                self.assertEqual(controller.command.velocity, (0, 0, 0))
                self.assertEqual(sample(0.51, 0.8).source, "none")
                sample(0.52, 0)
                self.assertEqual(sample(0.53, 0.8).source, "controller")

    def test_real_gaze_requires_fresh_valid_look_away_after_controller_handoff(self) -> None:
        for control_mode in ("hold", "step"):
            with self.subTest(control_mode=control_mode):
                self.hybrid.reset()
                gaze = GAZE.GazeLocomotion(
                    GAZE.GazeLocomotionConfig(preview=False, control_mode=control_mode, dwell_s=0)
                )
                now_s = 0.0

                def tick(target: str | None, *, eye_valid: bool = True, **sticks: object) -> object:
                    nonlocal now_s
                    now_s += 0.01
                    selection = self.select(**sticks)
                    return gaze.update(
                        target=target,
                        eye_valid=eye_valid,
                        now_s=now_s,
                        dt=0.01,
                        pose=(0, 0, 0),
                        linear_speed_m_s=0.0,
                        yaw_speed_rad_s=0.0,
                        ready=selection.source == "gaze",
                    )

                tick(None)
                self.assertGreater(tick("forward").velocity[0], 0)
                self.assertEqual(tick("forward", controller_state="active", sticks_deflected=True).velocity, (0, 0, 0))
                # Looking away while sticks own control must not rearm gaze.
                tick(None, controller_state="active", sticks_deflected=True)
                self.assertEqual(tick(None).velocity, (0, 0, 0))
                self.assertEqual(tick("forward").state, "awaiting_release")
                tick(None, eye_valid=False)
                self.assertEqual(tick("forward").state, "awaiting_release")
                tick(None)
                self.assertGreater(tick("forward").velocity[0], 0)

    def test_bad_boolean_metadata_requests_zero_instead_of_coercing_truthiness(self) -> None:
        for name in ("controller_present", "controller_valid", "sticks_deflected"):
            for value in (None, 1, "true"):
                with self.subTest(name=name, value=value):
                    self.hybrid.reset()
                    selection = self.select(**{name: value})
                    self.assertEqual(selection.source, "none")
                    self.assertTrue(selection.reset_gaze)
                    self.assertTrue(selection.stop_immediately)

    def test_unknown_or_contradictory_controller_state_requests_zero(self) -> None:
        for state in (None, "unknown", ["active"], "active"):
            with self.subTest(state=state):
                self.hybrid.reset()
                selection = self.select(controller_state=state)
                self.assertEqual(selection.source, "none")
                self.assertTrue(selection.stop_immediately)
                self.assertTrue(selection.reset_gaze)

    def test_active_metadata_without_current_valid_devices_cannot_select_controller(self) -> None:
        for values in (
            dict(controller_present=False),
            dict(controller_valid=False),
            dict(controller_present=False, controller_valid=False),
        ):
            with self.subTest(values=values):
                self.hybrid.reset()
                self.assertEqual(self.select(controller_state="active", sticks_deflected=True, **values).source, "none")

    def test_reset_clears_ownership_and_requests_gaze_invalidation(self) -> None:
        self.claim()
        self.hybrid.reset()
        self.assertEqual(self.hybrid.selection.source, "gaze")
        self.assertTrue(self.hybrid.selection.reset_gaze)
        self.assertFalse(self.hybrid.status["controller_claimed"])
        self.assertTrue(self.select(controller_present=False, controller_valid=False).reset_gaze)
        self.assertFalse(self.select(controller_present=False, controller_valid=False).reset_gaze)

    def test_selection_is_immutable_and_status_is_an_independent_snapshot(self) -> None:
        selected = self.claim()
        with self.assertRaises(FrozenInstanceError):
            selected.source = "gaze"
        status = self.hybrid.status
        status["source"] = "gaze"
        status["controller_claimed"] = False
        self.assertEqual(self.hybrid.selection.source, "controller")
        self.assertTrue(self.hybrid.status["controller_claimed"])


if __name__ == "__main__":
    unittest.main()
