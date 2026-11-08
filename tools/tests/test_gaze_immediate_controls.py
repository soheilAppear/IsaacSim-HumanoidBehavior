# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check zero-dwell gaze activation, release and tracking-loss interlocks."""

from __future__ import annotations

import unittest

import test_gaze_locomotion_controls as fixture

GAZE = fixture.GAZE


class TestImmediateGazeControls(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = GAZE.GazeLocomotion(GAZE.GazeLocomotionConfig(control_mode="hold", dwell_s=0, preview=False))
        self.time = 0.0

    def tick(self, target=None, **overrides):
        """Deliver one current eye sample with finite native motion measurements."""
        self.time += 0.005
        values = dict(
            target=target,
            eye_valid=True,
            now_s=self.time,
            dt=0.005,
            pose=(0.0, 0.0, 0.0),
            linear_speed_m_s=0.0,
            yaw_speed_rad_s=0.0,
        )
        values.update(overrides)
        return self.controller.update(**values)

    def test_every_direction_starts_on_first_sample_without_repeating(self) -> None:
        for count, action in enumerate(GAZE.ACTIONS, 1):
            self.tick()
            result = self.tick(action)
            self.assertEqual(
                (result.state, result.activated_action, result.activation_count),
                ("active", action, count),
            )
            self.assertTrue(any(result.velocity))
            for _ in range(10):
                result = self.tick(action)
                self.assertEqual(result.activation_count, count)
                self.assertIsNone(result.activated_action)
                self.assertTrue(any(result.velocity))

    def test_tracking_loss_stops_and_cannot_rearm_until_valid_look_away(self) -> None:
        self.tick()
        self.tick("forward")
        result = self.tick("forward", eye_valid=False)
        self.assertTrue(result.stop_immediately)
        self.assertFalse(any(result.velocity))
        for _ in range(10):
            self.assertEqual(self.tick("forward").activation_count, 1)
        self.tick()
        self.assertEqual(self.tick("forward").activation_count, 2)

    def test_reset_and_occlusion_require_fresh_verified_outside_sample(self) -> None:
        self.assertEqual(self.tick("forward").activation_count, 0)
        self.tick()
        self.tick("forward")
        result = self.tick("occluded")
        self.assertTrue(result.stop_immediately)
        self.assertFalse(any(result.velocity))
        self.assertEqual(self.tick("turn_left").activation_count, 1)
        self.controller.reset()
        self.assertEqual(self.tick("forward").activation_count, 0)
        self.tick()
        self.assertEqual(self.tick("forward").activation_count, 1)

    def test_preview_selects_immediately_but_never_moves(self) -> None:
        self.controller = GAZE.GazeLocomotion(GAZE.GazeLocomotionConfig(control_mode="hold", dwell_s=0, preview=True))
        self.tick(pose=None)
        result = self.tick("forward", pose=None)
        self.assertEqual((result.state, result.activated_action), ("preview", "forward"))
        self.assertFalse(any(result.velocity))
        self.assertEqual(self.tick("forward", pose=None).activation_count, 1)


if __name__ == "__main__":
    unittest.main()
