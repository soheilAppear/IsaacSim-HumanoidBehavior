# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep fast pose output scoped to the humanoid scene's lifetime."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import Mock, patch

from humanoid_test_support import CARB, HUMANOID, USD, load_source

OUTPUT_PATHS = (
    "/physics/updateToUsd",
    "/physics/updateParticlesToUsd",
    "/physics/updateVelocitiesToUsd",
    "/physics/updateForceSensorsToUsd",
)


class TestPhysicsOutputLifecycle(unittest.TestCase):
    """Exercise production setup/restore with the SDK's coupled flag behavior."""

    def setUp(self) -> None:
        self.values = dict(zip(OUTPUT_PATHS, (True, False, True, False)))
        self.state = {"device": "cpu", "fabric": False}
        self.settings = types.SimpleNamespace(
            get=Mock(side_effect=self.values.get),
            set_bool=Mock(side_effect=self.values.__setitem__),
            destroy_item=Mock(side_effect=lambda path: self.values.pop(path, None)),
        )
        settings_module = types.ModuleType("carb.settings")
        settings_module.get_settings = lambda: self.settings
        self.manager = types.SimpleNamespace(
            get_physics_sim_device=Mock(side_effect=lambda: self.state["device"]),
            is_fabric_enabled=Mock(side_effect=lambda: self.state["fabric"]),
            set_physics_sim_device=Mock(side_effect=self.set_device),
            enable_fabric=Mock(side_effect=self.enable_fabric),
            set_backend=Mock(),
            get_available_physics_engines=Mock(),
        )
        helpers = load_source("interactive/utils.py", SimulationManager=self.manager)
        patches = (
            patch.dict(sys.modules, {"carb.settings": settings_module}),
            patch.object(CARB, "settings", settings_module, create=True),
            patch.object(HUMANOID, "SimulationManager", self.manager, create=True),
            patch.object(
                HUMANOID, "snapshot_physics_simulation_state", helpers.snapshot_physics_simulation_state, create=True
            ),
            patch.object(
                HUMANOID, "restore_physics_simulation_state", helpers.restore_physics_simulation_state, create=True
            ),
            patch.object(HUMANOID, "G1TeleopRobot", Mock(), create=True),
            patch.object(USD, "get_context", return_value=types.SimpleNamespace(get_stage=lambda: object())),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.ex = HUMANOID.HumanoidExample()
        self.environment_state = None
        for name in (
            "_create_environment",
            "_ensure_scene_lighting",
            "_apply_ground_material",
            "_create_work_surfaces",
            "_create_sample_boxes",
            "_prepare_physical_grasping",
            "_create_arm_control_rig",
            "_create_head_camera",
            "_create_xr_anchor",
        ):
            setattr(self.ex, name, Mock())
        self.ex._create_environment.side_effect = self.capture_environment_state
        self.ex._material_highlights.prepare = Mock()
        self.ex._navigation_scene.create = Mock()

    def set_device(self, device: str) -> None:
        """Model the SDK's implicit Fabric activation when selecting CUDA."""
        self.state["device"] = device
        if device.startswith("cuda"):
            self.enable_fabric(True)

    def enable_fabric(self, enabled: bool) -> None:
        """Model the SDK overwriting all four USD output settings."""
        self.state["fabric"] = enabled
        self.values.update({path: not enabled for path in OUTPUT_PATHS})

    def capture_environment_state(self) -> None:
        """Observe global state at the first simulation asset loading boundary."""
        self.environment_state = (dict(self.state), dict(self.values))

    def test_fabric_is_ready_before_assets_without_changing_cpu_or_timestep(self) -> None:
        before_world = dict(self.ex._world_settings)
        self.ex.setup_scene()
        self.assertEqual(
            self.environment_state, ({"device": "cpu", "fabric": True}, dict.fromkeys(OUTPUT_PATHS, False))
        )
        self.assertEqual(self.ex._world_settings, before_world)
        self.manager.set_physics_sim_device.assert_called_once_with("cpu")

    def test_restore_preserves_mixed_output_preferences_after_fabric_changes_them(self) -> None:
        for device, fabric in (("cpu", False), ("cpu", True), ("cuda:0", True)):
            with self.subTest(device=device, fabric=fabric):
                self.state.update(device=device, fabric=fabric)
                self.values.update(dict(zip(OUTPUT_PATHS, (False, True, False, True))))
                before_state, before_outputs = dict(self.state), dict(self.values)
                self.ex.setup_scene()
                self.ex._restore_physics_simulation_state()
                self.assertEqual(self.state, before_state)
                self.assertEqual(self.values, before_outputs)

    def test_absent_output_setting_is_absent_again_after_cleanup(self) -> None:
        missing = OUTPUT_PATHS[-1]
        del self.values[missing]
        before = dict(self.values)
        self.ex.setup_scene()
        self.assertIn(missing, self.values)
        self.ex._restore_physics_simulation_state()
        self.assertEqual(self.values, before)
        self.settings.destroy_item.assert_called_once_with(missing)

    def test_second_cleanup_does_not_overwrite_a_later_scene_setting(self) -> None:
        self.ex.setup_scene()
        self.ex._restore_physics_simulation_state()
        self.values[OUTPUT_PATHS[0]] = False
        self.state.update(device="cuda:0", fabric=True)
        before_state, before_outputs = dict(self.state), dict(self.values)
        self.manager.set_physics_sim_device.reset_mock()
        self.manager.enable_fabric.reset_mock()
        self.settings.set_bool.reset_mock()
        self.ex._restore_physics_simulation_state()
        self.assertEqual(self.state, before_state)
        self.assertEqual(self.values, before_outputs)
        self.manager.set_physics_sim_device.assert_not_called()
        self.manager.enable_fabric.assert_not_called()
        self.settings.set_bool.assert_not_called()

    def test_cleanup_before_setup_leaves_current_app_state_alone(self) -> None:
        self.ex._restore_physics_simulation_state()
        self.manager.set_physics_sim_device.assert_not_called()
        self.manager.enable_fabric.assert_not_called()
        self.settings.set_bool.assert_not_called()

    def test_asset_loading_failure_does_not_lose_the_original_settings(self) -> None:
        before_state, before_outputs = dict(self.state), dict(self.values)
        self.ex._create_environment.side_effect = RuntimeError("Asset load failed")
        with self.assertRaisesRegex(RuntimeError, "Asset load failed"):
            self.ex.setup_scene()
        self.ex._restore_physics_simulation_state()
        self.assertEqual(self.state, before_state)
        self.assertEqual(self.values, before_outputs)

    def test_failed_load_retry_keeps_original_app_preferences_until_cleanup(self) -> None:
        for device, fabric in (("cpu", False), ("cuda:0", True)):
            with self.subTest(device=device, fabric=fabric):
                self.state.update(device=device, fabric=fabric)
                self.values.pop(OUTPUT_PATHS[-1], None)
                before_state, before_outputs = dict(self.state), dict(self.values)
                self.ex._create_environment.side_effect = RuntimeError("Asset load failed")
                with self.assertRaisesRegex(RuntimeError, "Asset load failed"):
                    self.ex.setup_scene()

                # Retrying the same instance without an intervening clear must
                # not adopt the first attempt's output flags as user preferences.
                self.ex._create_environment.side_effect = self.capture_environment_state
                self.ex.setup_scene()
                self.assertEqual(self.state, {"device": "cpu", "fabric": True})
                self.ex._restore_physics_simulation_state()
                self.assertEqual(self.state, before_state)
                self.assertEqual(self.values, before_outputs)

    def test_failed_setting_restoration_can_retry_without_touching_other_settings(self) -> None:
        before = dict(self.values)
        self.ex.setup_scene()
        failed = OUTPUT_PATHS[0]

        def restore_value(path: str, value: bool) -> None:
            if path == failed:
                raise RuntimeError("Temporary settings failure")
            self.values[path] = value

        self.settings.set_bool.side_effect = restore_value
        self.ex._restore_physics_simulation_state()
        self.assertEqual(self.ex._prev_physics_usd_settings, {failed: before[failed]})
        self.manager.enable_fabric.reset_mock()
        self.settings.set_bool.reset_mock()
        self.settings.set_bool.side_effect = self.values.__setitem__
        self.ex._restore_physics_simulation_state()
        self.settings.set_bool.assert_called_once_with(failed, before[failed])
        self.manager.enable_fabric.assert_not_called()
        self.assertEqual(self.values, before)
        self.assertIsNone(self.ex._prev_physics_usd_settings)


if __name__ == "__main__":
    unittest.main()
