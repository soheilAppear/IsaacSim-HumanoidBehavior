# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope XR orientation and near clipping to the active robot view."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

from humanoid_test_support import CARB, HUMANOID, USD
from pxr import Usd, UsdGeom


class TestRobotCameraXRSettings(unittest.TestCase):
    def setUp(self):
        self.values = {
            "/persistent/xr/overrideFloorAlignedAnchor": False,
            "/persistent/xr/render/nearPlane": 0.02,
        }
        self.settings = types.SimpleNamespace(
            get=Mock(side_effect=self.values.get),
            set=Mock(side_effect=self.values.__setitem__),
            set_bool=Mock(side_effect=self.values.__setitem__),
            destroy_item=Mock(side_effect=lambda key: self.values.pop(key, None)),
        )
        module = types.ModuleType("carb.settings")
        module.get_settings = lambda: self.settings
        for patcher in (
            patch.dict(sys.modules, {"carb.settings": module}),
            patch.object(CARB, "settings", module, create=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.example = HUMANOID.HumanoidExample()
        self.profile = types.SimpleNamespace(get_persistent_path=Mock(return_value="/persistent/xr/profile/vr/"))
        self.example._xr_core = types.SimpleNamespace(get_current_profile=lambda: self.profile)
        self.key = "/persistent/xr/profile/vr/overrideFloorAlignedAnchor"
        self.near_key = "/persistent/xr/profile/vr/render/nearPlane"

    def test_profile_override_preserves_global_and_absence_on_cleanup(self):
        original = dict(self.values)
        self.example._configure_robot_head_xr_orientation()
        self.assertIs(self.values[self.key], True)
        self.assertEqual(self.values[self.near_key], 0.05)
        self.assertIs(self.values["/persistent/xr/overrideFloorAlignedAnchor"], False)
        self.assertEqual(self.values["/persistent/xr/render/nearPlane"], 0.02)
        self.example._restore_robot_head_xr_orientation()
        self.assertEqual(self.values, original)

    def test_existing_preference_restored_exactly_once(self):
        for previous in (True, False):
            with self.subTest(previous=previous):
                self.values[self.key] = previous
                self.example._configure_robot_head_xr_orientation()
                for _ in range(20):
                    self.example._configure_robot_head_xr_orientation()
                self.example._restore_robot_head_xr_orientation()
                self.assertIs(self.values[self.key], previous)
                self.values[self.key] = "later owner's value"
                self.example._restore_robot_head_xr_orientation()
                self.assertEqual(self.values[self.key], "later owner's value")

    def test_near_plane_preference_is_restored_without_recapturing_our_override(self):
        for previous in (0.001, 0.025, 0.05, 0.1):
            with self.subTest(previous=previous):
                self.values[self.near_key] = previous
                original = dict(self.values)
                self.settings.set.reset_mock()
                self.example._configure_robot_head_xr_orientation()
                for _ in range(20):
                    self.example._configure_robot_head_xr_orientation()
                self.assertEqual(self.values[self.near_key], 0.05)
                self.settings.set.assert_called_once_with(self.near_key, 0.05)
                self.example._restore_robot_head_xr_orientation()
                self.assertEqual(self.values, original)
                self.values[self.near_key] = 0.075
                self.example._restore_robot_head_xr_orientation()
                self.assertEqual(self.values[self.near_key], 0.075)

    def test_switching_profile_restores_first_before_overriding_second(self):
        self.example._configure_robot_head_xr_orientation()
        self.profile.get_persistent_path.return_value = "/xr/profile/test/"
        second_key = "/xr/profile/test/overrideFloorAlignedAnchor"
        second_near_key = "/xr/profile/test/render/nearPlane"
        self.values[second_key] = False
        self.values[second_near_key] = 0.012
        self.example._configure_robot_head_xr_orientation()
        self.assertNotIn(self.key, self.values)
        self.assertNotIn(self.near_key, self.values)
        self.assertIs(self.values[second_key], True)
        self.assertEqual(self.values[second_near_key], 0.05)
        self.example._restore_robot_head_xr_orientation()
        self.assertIs(self.values[second_key], False)
        self.assertEqual(self.values[second_near_key], 0.012)

    def test_no_profile_does_not_change_app_settings(self):
        self.profile = None
        original = dict(self.values)
        self.example._configure_robot_head_xr_orientation()
        self.assertEqual(self.values, original)
        self.settings.set_bool.assert_not_called()
        self.settings.set.assert_not_called()

    def test_camera_creation_and_xr_use_the_same_fixed_near_plane(self):
        stage = Usd.Stage.CreateInMemory()
        with (
            patch.object(USD, "get_context", return_value=types.SimpleNamespace(get_stage=lambda: stage)),
            patch.object(self.example, "_prepare_robot_head_camera_mount"),
        ):
            self.example._create_head_camera()
            camera = UsdGeom.Camera.Get(stage, self.example._head_camera_path)
            near, far = camera.GetClippingRangeAttr().Get()
            self.assertAlmostEqual(near, 0.05, places=7)
            self.assertEqual(far, 1000.0)
            self.example._configure_robot_head_xr_orientation()
            self.assertAlmostEqual(near, self.values[self.near_key], places=7)
            self.example._restore_robot_head_xr_orientation()
            # Recreating the USD camera and reacquiring the profile must also
            # agree if the local configuration was deliberately adjusted.
            self.example._head_camera_near_clip_m = 0.025
            self.example._create_head_camera()
            self.example._configure_robot_head_xr_orientation()
            self.assertAlmostEqual(camera.GetClippingRangeAttr().Get()[0], 0.025, places=7)
            self.assertEqual(self.values[self.near_key], 0.025)
            self.example._restore_robot_head_xr_orientation()
