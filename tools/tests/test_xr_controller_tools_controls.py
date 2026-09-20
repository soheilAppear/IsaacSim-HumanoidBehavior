# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reserve controller tools while preserving raw XR input and user preferences."""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import Mock, patch

from humanoid_test_support import CARB, HUMANOID


class TestControllerXRTools(unittest.TestCase):
    def setUp(self):
        self.values = {
            "/xr/openxr/components/omni.kit.xr.openxr.ext.hand_tracking/active": True,
            "/xr/openxr/components/omni.kit.xr.openxr.ext.eye_gaze_interaction/active": True,
            "/xr/profile/vr/gui/layers": ["controllers", "hands"],
        }
        self.settings = types.SimpleNamespace(
            get=Mock(side_effect=self.values.get),
            set=Mock(side_effect=self.values.__setitem__),
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
        self.ex = HUMANOID.HumanoidExample()
        self.ex._controller_walking_enabled = True
        self.ex.g1 = object()
        self.profile = types.SimpleNamespace(
            get_persistent_path=Mock(return_value="/persistent/xr/profile/vr/"),
            get_name=Mock(return_value="vr"),
            is_enabled=Mock(return_value=True),
        )
        self.core = types.SimpleNamespace(
            get_current_profile=Mock(side_effect=lambda: self.profile),
            is_xr_enabled=Mock(return_value=True),
        )
        self.ex._xr_core = self.core
        self.key = "/persistent/xr/profile/vr/tools/layout"

    def test_only_high_level_layout_changes_and_absence_is_restored(self):
        original = dict(self.values)
        owned = self.ex._encode_controller_xr_tools_layout(None)
        self.ex._sync_controller_xr_tools()
        self.assertEqual(self.values, {**original, self.key: owned})
        for _ in range(20):
            self.ex._sync_controller_xr_tools()
        self.settings.set.assert_called_once_with(self.key, owned)
        self.ex._restore_controller_xr_tools()
        self.assertEqual(self.values, original)
        self.ex._restore_controller_xr_tools()
        self.settings.destroy_item.assert_called_once_with(self.key)

    def test_exact_existing_layout_restored_on_mode_exit(self):
        for previous in ("vr", "", "user_custom_layout"):
            with self.subTest(previous=previous):
                self.ex._controller_walking_enabled = True
                self.values[self.key] = previous
                self.ex._sync_controller_xr_tools()
                self.ex._controller_walking_enabled = False
                self.ex._sync_controller_xr_tools()
                self.assertEqual(self.values[self.key], previous)
                self.assertIsNone(self.ex._controller_xr_tools_setting)

    def test_profile_switch_restores_previous_then_owns_only_next_profile(self):
        self.values[self.key] = "vr"
        self.ex._sync_controller_xr_tools()
        self.profile.get_persistent_path.return_value = "/xr/profile/another/"
        second = "/xr/profile/another/tools/layout"
        self.values[second] = "custom_previous"
        self.ex._sync_controller_xr_tools()
        self.assertEqual(self.values[self.key], "vr")
        self.assertEqual(self.values[second], self.ex._encode_controller_xr_tools_layout("custom_previous"))
        self.ex._restore_controller_xr_tools()
        self.assertEqual(self.values[second], "custom_previous")

    def test_xr_exit_and_reentry_restore_and_reacquire_preference(self):
        self.values[self.key] = "vr"
        self.ex._sync_controller_xr_tools()
        self.core.is_xr_enabled.return_value = False
        self.ex._sync_controller_xr_tools()
        self.assertEqual(self.values[self.key], "vr")
        self.core.is_xr_enabled.return_value = True
        self.ex._sync_controller_xr_tools()
        self.assertEqual(self.values[self.key], self.ex._encode_controller_xr_tools_layout("vr"))
        self.profile = None
        self.ex._sync_controller_xr_tools()
        self.assertEqual(self.values[self.key], "vr")

    def test_later_preference_edit_is_preserved_then_becomes_next_restore_value(self):
        self.values[self.key] = "vr"
        self.ex._sync_controller_xr_tools()
        self.values[self.key] = "later_edit"
        self.ex._restore_controller_xr_tools()
        self.assertEqual(self.values[self.key], "later_edit")
        self.ex._sync_controller_xr_tools()
        self.values[self.key] = "newer_edit"
        self.ex._sync_controller_xr_tools()
        self.assertEqual(self.values[self.key], self.ex._encode_controller_xr_tools_layout("newer_edit"))
        self.ex._restore_controller_xr_tools()
        self.assertEqual(self.values[self.key], "newer_edit")

    def test_empty_profile_restores_owned_layout_without_acquiring_parent_namespace(self):
        self.values[self.key] = "custom_original"
        self.ex._sync_controller_xr_tools()
        self.profile.get_name.return_value = ""
        self.profile.get_persistent_path.return_value = "/persistent/xr/profile//"
        self.profile.get_persistent_path.reset_mock()
        self.settings.set.reset_mock()
        self.ex._sync_controller_xr_tools()
        self.assertTrue(self.core.is_xr_enabled())  # Native shutdown is still pending.
        self.assertEqual(self.values[self.key], "custom_original")
        self.assertIsNone(self.ex._controller_xr_tools_setting)
        self.profile.get_persistent_path.assert_not_called()
        self.settings.set.assert_called_once_with(self.key, "custom_original")
        self.assertNotIn("/persistent/xr/profile/tools/layout", self.values)

    def test_disabled_profile_preserves_stale_recovery_marker_until_active(self):
        marker = self.ex._encode_controller_xr_tools_layout("custom_previous")
        self.values[self.key] = marker
        self.profile.is_enabled.return_value = False
        self.ex._sync_controller_xr_tools()
        self.assertEqual(self.values[self.key], marker)
        self.assertIsNone(self.ex._controller_xr_tools_setting)
        self.profile.get_persistent_path.assert_not_called()
        self.settings.set.assert_not_called()
        # Recovery waits for the same real profile to be active; no restoration
        # data is lost or accidentally written under an empty profile namespace.
        self.profile.is_enabled.return_value = True
        self.ex._controller_walking_enabled = False
        self.ex._sync_controller_xr_tools()
        self.assertEqual(self.values[self.key], "custom_previous")

    def test_unloaded_robot_and_noncontroller_modes_cannot_claim_xr_settings(self):
        for enabled, robot in ((False, object()), (True, None)):
            self.ex._controller_walking_enabled = enabled
            self.ex.g1 = robot
            original = dict(self.values)
            self.ex._sync_controller_xr_tools()
            self.assertEqual(self.values, original)
        self.settings.set.assert_not_called()

    def test_app_updates_sync_even_outside_robot_head_view_and_stage_close_restores(self):
        self.ex._g1_locomotion = "policy"
        self.ex._xr_camera_mode = "head_compose"
        context = types.SimpleNamespace(get_stage=Mock(return_value=object()))
        with patch.object(HUMANOID.omni.usd, "get_context", return_value=context):
            self.ex._on_robot_camera_update(None)
            self.assertEqual(self.values[self.key], self.ex._encode_controller_xr_tools_layout(None))
            context.get_stage.return_value = None
            self.ex._on_robot_camera_update(None)
            self.assertNotIn(self.key, self.values)

    def test_both_example_cleanup_paths_restore_layout_before_other_teardown(self):
        # Abort immediately after our restore hook; the rest of real scene cleanup
        # requires Kit. This checks both lifecycle entry points retain ownership.
        for asynchronous in (False, True):
            self.values[self.key] = "vr"
            self.ex._sync_controller_xr_tools()
            with patch.object(
                self.ex, "_restore_robot_head_xr_orientation", side_effect=RuntimeError("teardown marker")
            ):
                with self.assertRaisesRegex(RuntimeError, "teardown marker"):
                    if asynchronous:
                        asyncio.run(self.ex.setup_post_clear())
                    else:
                        self.ex.physics_cleanup()
            self.assertEqual(self.values[self.key], "vr")

    def _new_session(self, controller):
        fresh = HUMANOID.HumanoidExample()
        fresh._controller_walking_enabled = controller
        fresh.g1 = object()
        fresh._xr_core = self.core
        return fresh

    def test_stationary_new_instance_recovers_crash_without_claiming_tools(self):
        for previous in (None, "vr", "", 'custom layout "with" unicode \u00e9'):
            with self.subTest(previous=previous):
                if previous is None:
                    self.values.pop(self.key, None)
                else:
                    self.values[self.key] = previous
                # Abandon the owning instance without invoking cleanup, just as
                # a native process crash can abandon its persistent preference.
                crashed = self._new_session(controller=True)
                crashed._sync_controller_xr_tools()
                persisted = self.values[self.key]
                self.assertNotEqual(persisted, previous)
                fresh = self._new_session(controller=False)
                fresh._sync_controller_xr_tools()
                self.assertEqual(self.values.get(self.key), previous)
                self.assertEqual(self.key in self.values, previous is not None)
                self.assertIsNone(fresh._controller_xr_tools_setting)

    def test_controller_new_instance_recovers_before_claiming_tools(self):
        for previous in (None, "custom_previous"):
            with self.subTest(previous=previous):
                self.values[self.key] = self.ex._encode_controller_xr_tools_layout(previous)
                fresh = self._new_session(controller=True)
                fresh._sync_controller_xr_tools()
                self.assertEqual(fresh._controller_xr_tools_setting[2], previous)
                self.assertEqual(self.values[self.key], fresh._encode_controller_xr_tools_layout(previous))
                fresh._restore_controller_xr_tools()
                self.assertEqual(self.values.get(self.key), previous)
                self.assertEqual(self.key in self.values, previous is not None)

    def test_invalid_private_markers_and_unrelated_layouts_are_not_recovered(self):
        prefix = "g1_controller_teleop__v1__"
        invalid = (
            "user_custom_layout",
            "g1_controller_teleop_lookalike",
            prefix,
            prefix + "nothex",
            prefix + "ff",  # Invalid UTF-8.
            prefix + "31",  # A JSON number is not a layout opinion.
            prefix + "7b7d",  # Nor is a JSON object.
            prefix + "20226d796c61796f757422",  # Noncanonical leading JSON whitespace.
            prefix + "6e756c6c".upper(),  # Noncanonical hexadecimal form.
        )
        for layout in invalid:
            with self.subTest(layout=layout):
                self.values[self.key] = layout
                fresh = self._new_session(controller=False)
                fresh._sync_controller_xr_tools()
                self.assertEqual(self.values[self.key], layout)
                self.assertIsNone(fresh._controller_xr_tools_setting)

    def test_exact_legacy_owned_marker_recovers_default_vr_layout(self):
        self.values[self.key] = "g1_controller_teleop"
        self._new_session(controller=False)._sync_controller_xr_tools()
        self.assertEqual(self.values[self.key], "vr")

    def test_bad_layout_type_fails_before_claiming_or_mutating_preferences(self):
        self.values[self.key] = 123
        with self.assertRaisesRegex(ValueError, "string or an absent"):
            self.ex._sync_controller_xr_tools()
        self.assertEqual(self.values[self.key], 123)
        self.assertIsNone(self.ex._controller_xr_tools_setting)


if __name__ == "__main__":
    unittest.main()
