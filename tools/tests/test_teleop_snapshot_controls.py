# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Share synchronous hand reads without replaying input after loss, reset, or a new step."""

from __future__ import annotations

import types
import unittest
from unittest.mock import Mock

from humanoid_test_support import HUMANOID
from pxr import Gf
from test_finger_controls import SkeletonDevice, skeleton_positions


class TestTeleopSnapshotControls(unittest.TestCase):
    def setUp(self) -> None:
        self.ex = HUMANOID.HumanoidExample()
        self.device = SkeletonDevice(skeleton_positions(self.ex, {"index": 0.2}), flags=0xA)
        self.ex._xr_core = types.SimpleNamespace(get_input_device=Mock(return_value=self.device))

    def test_navigation_fingers_and_palm_share_current_sample_and_next_step_refreshes(self) -> None:
        ex = self.ex
        ex._measure_hand_tracking_finger_curls = Mock(wraps=ex._measure_hand_tracking_finger_curls)
        ex._read_hand_joint_position = Mock(wraps=ex._read_hand_joint_position)
        results = []

        def step(dt, context):
            navigation = ex._read_navigation_hand()[0]
            device = ex._get_xr_input_device("/user/hand/left")
            fingers = ex._get_hand_tracking_finger_curls(device)
            ex._get_optical_arm_pose("left", device)
            ex._get_optical_palm_frame("left", device)
            results.append((navigation, fingers))

        ex._advance_physics_step = step
        ex.on_physics_step(0.005, None)
        self.assertIsNone(ex._teleop_step_hand_samples)
        ex._xr_core.get_input_device.assert_called_once_with("/user/hand/left")
        ex._measure_hand_tracking_finger_curls.assert_called_once_with(self.device)
        landmarks = [call.args[1] for call in ex._read_hand_joint_position.call_args_list]
        self.assertEqual(len(landmarks), len(set(landmarks)))
        self.assertEqual(results[-1][0], results[-1][1])
        self.assertAlmostEqual(results[-1][1]["index"], 0.2)

        self.device.positions = skeleton_positions(ex, {"index": 0.8})
        ex.on_physics_step(0.005, None)
        self.assertAlmostEqual(results[-1][1]["index"], 0.8)
        self.assertEqual(ex._measure_hand_tracking_finger_curls.call_count, 2)
        self.device.flags = 0
        ex.on_physics_step(0.005, None)
        self.assertEqual(results[-1], (None, None))

    def test_cached_curls_and_positions_cannot_be_mutated_by_consumers(self) -> None:
        ex = self.ex

        def step(dt, context):
            first = ex._get_hand_tracking_finger_curls(self.device)
            first["index"] = 100.0
            position = ex._get_hand_joint_position(self.device, "wrist")
            position[0] = 100.0
            self.assertAlmostEqual(ex._get_hand_tracking_finger_curls(self.device)["index"], 0.2)
            self.assertEqual(ex._get_hand_joint_position(self.device, "wrist")[0], 0.0)

        ex._advance_physics_step = step
        ex.on_physics_step(0.005, None)

    def test_input_kind_is_classified_once_then_refreshed_after_source_loss(self) -> None:
        ex = self.ex
        self.device.get_hand_tracking_data_source = Mock(return_value="hand")
        self.device.get_pose_names = Mock(wraps=self.device.get_pose_names)
        observed = []

        def step(dt, context):
            observed.append(ex._get_hand_input_kind(self.device))
            self.assertEqual(ex._get_hand_input_kind(self.device), observed[-1])

        ex._advance_physics_step = step
        ex.on_physics_step(0.005, None)
        self.device.get_hand_tracking_data_source.assert_called_once_with()
        self.device.get_pose_names.assert_called_once_with()
        self.device.get_hand_tracking_data_source.return_value = ""
        self.device.flags = 0
        self.device.get_virtual_world_pose_desc = Mock(
            return_value=types.SimpleNamespace(validity_flags=0, pose_matrix=Gf.Matrix4d(1))
        )
        ex.on_physics_step(0.005, None)
        self.assertEqual(observed, ["hand_tracking", "none"])
        self.assertEqual(self.device.get_hand_tracking_data_source.call_count, 2)

    def test_action_values_share_one_read_and_refresh_even_after_invalid_input(self) -> None:
        ex = self.ex
        device = types.SimpleNamespace(
            has_input=Mock(return_value=True),
            has_input_gesture=Mock(return_value=True),
            get_input_gesture_value=Mock(return_value=0.75),
        )

        def step(dt, context):
            first = ex._get_xr_gesture_value(device, "trigger", "value")
            self.assertEqual(ex._get_xr_gesture_value(device, "trigger", "value"), first)

        ex._advance_physics_step = step
        for value in (0.75, 0.0, float("nan"), RuntimeError("input lost")):
            device.get_input_gesture_value.reset_mock()
            device.get_input_gesture_value.side_effect = value if isinstance(value, Exception) else None
            device.get_input_gesture_value.return_value = value
            ex.on_physics_step(0.005, None)
            device.get_input_gesture_value.assert_called_once_with("trigger", "value")
        device.get_input_gesture_value.side_effect = None
        device.get_input_gesture_value.return_value = 0.25
        self.assertEqual(ex._get_xr_gesture_value(device, "trigger", "value"), 0.25)
        device.get_input_gesture_value.return_value = 0.5
        self.assertEqual(ex._get_xr_gesture_value(device, "trigger", "value"), 0.5)

    def test_missing_actions_are_cached_per_key_and_device_only_until_next_step(self) -> None:
        ex = self.ex
        device = types.SimpleNamespace(has_input=Mock(return_value=False))

        def step(dt, context):
            for name, gesture in (("trigger", "value"), ("trigger", "value"), ("trigger", "click")):
                self.assertEqual(ex._get_xr_gesture_value(device, name, gesture), 0.0)
            self.assertEqual(ex._get_xr_gesture_value(self.device, "trigger", "value"), 0.0)

        ex._advance_physics_step = step
        ex.on_physics_step(0.005, None)
        self.assertEqual(device.has_input.call_count, 2)
        ex.on_physics_step(0.005, None)
        self.assertEqual(device.has_input.call_count, 4)

    def test_palm_frame_is_measured_once_per_side_and_cannot_be_mutated(self) -> None:
        ex = self.ex
        ex._measure_optical_palm_frame = Mock(wraps=ex._measure_optical_palm_frame)

        def step(dt, context):
            first = ex._get_optical_palm_frame("left", self.device)
            expected = Gf.Matrix4d(first)
            first.SetTranslateOnly(Gf.Vec3d(100))
            self.assertEqual(ex._get_optical_palm_frame("left", self.device), expected)
            self.assertIsNotNone(ex._get_optical_palm_frame("right", self.device))

        ex._advance_physics_step = step
        ex.on_physics_step(0.005, None)
        self.assertEqual(ex._measure_optical_palm_frame.call_count, 2)
        self.device.flags = 0

        def invalid_step(dt, context):
            self.assertIsNone(ex._get_optical_palm_frame("left", self.device))
            self.assertIsNone(ex._get_optical_palm_frame("left", self.device))

        ex._advance_physics_step = invalid_step
        ex.on_physics_step(0.005, None)
        self.assertEqual(ex._measure_optical_palm_frame.call_count, 3)

    def test_loss_and_callback_exception_discard_new_cached_fields(self) -> None:
        ex = self.ex
        ex._set_grab_candidate = Mock()
        ex._release_grabbed_object = Mock()
        ex._set_arm_rig_target_visible = Mock()
        self.device.has_input = Mock(return_value=True)
        self.device.has_input_gesture = Mock(return_value=True)
        self.device.get_input_gesture_value = Mock(return_value=1.0)

        def step(dt, context):
            device = ex._get_xr_input_device("/user/hand/left")
            self.assertEqual(ex._get_hand_input_kind(device), "hand_tracking")
            self.assertIsNotNone(ex._get_optical_palm_frame("left", device))
            self.assertEqual(ex._get_xr_gesture_value(device, "trigger", "value"), 1.0)
            self.device.flags = 0
            self.device.get_input_gesture_value.return_value = 0.0
            ex._deactivate_hand("left")
            self.assertIsNone(ex._get_optical_palm_frame("left", device))
            self.assertEqual(ex._get_xr_gesture_value(device, "trigger", "value"), 0.0)
            raise RuntimeError("interrupted")

        ex._advance_physics_step = step
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            ex.on_physics_step(0.005, None)
        self.assertIsNone(ex._teleop_step_hand_samples)
        self.device.get_input_gesture_value.return_value = 0.5
        self.assertEqual(ex._get_xr_gesture_value(self.device, "trigger", "value"), 0.5)

    def test_failure_early_return_and_standalone_calls_do_not_keep_a_sample(self) -> None:
        ex = self.ex

        def failing_step(dt, context):
            ex._get_hand_tracking_finger_curls(self.device)
            raise RuntimeError("interrupted callback")

        ex._advance_physics_step = failing_step
        with self.assertRaisesRegex(RuntimeError, "interrupted callback"):
            ex.on_physics_step(0.005, None)
        self.assertIsNone(ex._teleop_step_hand_samples)
        self.device.positions = skeleton_positions(ex, {"index": 0.7})
        self.assertAlmostEqual(ex._get_hand_tracking_finger_curls(self.device)["index"], 0.7)
        self.device.flags = 0
        self.assertIsNone(ex._get_hand_tracking_finger_curls(self.device))
        ex._advance_physics_step = Mock(return_value=None)
        ex.on_physics_step(0.0, None)
        self.assertIsNone(ex._teleop_step_hand_samples)

    def test_head_device_reads_are_not_cached(self) -> None:
        ex = self.ex

        def step(dt, context):
            ex._get_xr_input_device("/user/head")
            ex._get_xr_input_device("/user/head")

        ex._advance_physics_step = step
        ex.on_physics_step(0.005, None)
        self.assertEqual(ex._xr_core.get_input_device.call_count, 2)

    def test_loss_and_reset_discard_hand_sample_before_callback_returns(self) -> None:
        ex = self.ex
        ex._set_grab_candidate = Mock()
        ex._release_grabbed_object = Mock()
        ex._set_arm_rig_target_visible = Mock()

        def step(dt, context):
            device = ex._get_xr_input_device("/user/hand/left")
            self.assertIsNotNone(ex._get_hand_tracking_finger_curls(device))
            self.device.flags = 0
            ex._deactivate_hand("left")
            self.assertIsNone(ex._get_hand_tracking_finger_curls(device))
            ex._reset_teleoperation_state()
            self.assertIsNone(ex._teleop_step_hand_samples)

        ex._advance_physics_step = step
        ex.on_physics_step(0.005, None)


class TestBulkTeleopSnapshotControls(unittest.TestCase):
    def setUp(self) -> None:
        self.ex = HUMANOID.HumanoidExample()
        self.device = SkeletonDevice(skeleton_positions(self.ex, {"index": 0.2}), flags=0xA)
        self.device.get_all_virtual_world_poses = Mock(side_effect=self.bulk_poses)
        self.device.get_virtual_world_pose_desc = Mock(wraps=self.device.get_virtual_world_pose_desc)
        self.ex._xr_core = types.SimpleNamespace(get_input_device=Mock(return_value=self.device))

    def bulk_poses(self):
        return {name: SkeletonDevice.get_virtual_world_pose_desc(self.device, name) for name in self.device.positions}

    def test_controller_to_optical_handoff_uses_first_valid_callback_and_rejects_stale_sticks(self):
        # The runtime may reuse one device object, retaining controller actions
        # and even a controller-derived skeleton during the mode transition.
        device = self.device
        device.positions["grip"] = Gf.Vec3d(0.1, 0.2, 1.0)
        device.get_hand_tracking_data_source = Mock(return_value="controller")
        device.get_pose_desc = Mock(return_value=types.SimpleNamespace(pose_matrix=Gf.Matrix4d(1), validity_flags=0xF))
        device.has_input = Mock(return_value=True)
        device.has_input_gesture = Mock(return_value=True)
        device.get_input_gesture_value = Mock(side_effect=lambda name, axis: 0.4 if axis == "x" else 0.8)
        device.flags = 0xF
        observed = []

        def step(dt, context):
            current = self.ex._get_xr_input_device("/user/hand/left")
            observed.append(
                (
                    self.ex._get_hand_input_kind(current),
                    self.ex._get_optical_arm_pose("left", current),
                    self.ex._get_hand_tracking_finger_curls(current),
                    self.ex._read_tracked_controller_stick(current),
                )
            )

        self.ex._advance_physics_step = step
        self.ex.on_physics_step(0.005, None)
        self.assertEqual(observed[-1], ("controller", None, None, (True, 0.4, 0.8)))

        # A hand label alone does not fabricate usable finger/arm tracking.
        device.get_hand_tracking_data_source.return_value = "hand"
        device.flags = 0
        self.ex.on_physics_step(0.005, None)
        self.assertEqual(observed[-1], ("hand_tracking", None, None, (False, 0.0, 0.0)))

        device.positions = skeleton_positions(self.ex, {"index": 0.8})
        device.positions["grip"] = Gf.Vec3d(0.1, 0.2, 1.0)
        device.flags = 0x2  # Orientation/tracked bits are not needed by optical joints.
        self.ex.on_physics_step(0.005, None)
        kind, arm, curls, stick = observed[-1]
        self.assertEqual(kind, "hand_tracking")
        self.assertIsNotNone(arm)
        self.assertAlmostEqual(curls["index"], 0.8)
        self.assertEqual(stick, (False, 0.0, 0.0))
        self.assertEqual(device.get_all_virtual_world_poses.call_count, 2)
        self.assertIsNone(self.ex._teleop_step_hand_samples)

    def test_missing_device_is_replaced_by_new_optical_device_on_next_callback(self):
        observed = []

        def step(dt, context):
            current = self.ex._get_xr_input_device("/user/hand/left")
            observed.append(
                (
                    self.ex._get_optical_arm_pose("left", current),
                    self.ex._get_hand_tracking_finger_curls(current),
                )
            )

        self.ex._advance_physics_step = step
        self.ex._xr_core.get_input_device.return_value = None
        self.ex.on_physics_step(0.005, None)
        self.assertEqual(observed[-1], (None, None))
        self.ex._xr_core.get_input_device.return_value = self.device
        self.ex.on_physics_step(0.005, None)
        self.assertIsNotNone(observed[-1][0])
        self.assertAlmostEqual(observed[-1][1]["index"], 0.2)
        self.assertEqual(self.ex._xr_core.get_input_device.call_count, 2)
        self.device.get_all_virtual_world_poses.assert_called_once_with()

    def test_one_bulk_call_shares_navigation_and_fingers_but_next_step_refreshes_tracking_flags(self):
        results = []

        def step(dt, context):
            navigation = self.ex._read_navigation_hand()[0]
            curls = self.ex._get_hand_tracking_finger_curls(self.device)
            results.append((navigation, curls))

        self.ex._advance_physics_step = step
        self.ex.on_physics_step(0.005, None)
        self.assertEqual(results[-1][0], results[-1][1])
        self.assertAlmostEqual(results[-1][1]["index"], 0.2)
        self.device.get_all_virtual_world_poses.assert_called_once_with()
        self.device.get_virtual_world_pose_desc.assert_not_called()

        self.device.positions = skeleton_positions(self.ex, {"index": 0.8})
        self.device.flags = 0x2  # Position remains usable for fingers, but is no longer measured tracking.
        self.ex.on_physics_step(0.005, None)
        self.assertIsNone(results[-1][0])
        self.assertAlmostEqual(results[-1][1]["index"], 0.8)
        self.device.flags = 0
        self.ex.on_physics_step(0.005, None)
        self.assertEqual(results[-1], (None, None))
        self.assertEqual(self.device.get_all_virtual_world_poses.call_count, 3)
        self.assertIsNone(self.ex._teleop_step_hand_samples)

    def test_copied_flags_and_coordinates_do_not_alias_runtime_descriptors(self):
        provider = self.bulk_poses()
        provider["wrist"].validity_flags = 0x4A
        self.device.get_all_virtual_world_poses.side_effect = None
        self.device.get_all_virtual_world_poses.return_value = provider

        def step(dt, context):
            first = self.ex._get_hand_world_pose_sample(self.device, "wrist")
            self.assertEqual(first, (0x4A, (0.0, 0.0, 0.0)))
            provider["wrist"].pose_matrix.SetTranslateOnly(Gf.Vec3d(100, 0, 0))
            provider["wrist"].validity_flags = 0
            self.assertEqual(self.ex._get_hand_world_pose_sample(self.device, "wrist"), first)
            self.assertEqual(self.ex._get_hand_joint_position(self.device, "wrist"), Gf.Vec3d(0))

        self.ex._advance_physics_step = step
        self.ex.on_physics_step(0.005, None)
        # No callback-local sample may leak into standalone diagnostic reads.
        self.assertIsNone(self.ex._get_hand_joint_position(self.device, "wrist"))
        self.assertEqual(self.device.get_all_virtual_world_poses.call_count, 2)

    def test_missing_malformed_or_failed_bulk_result_never_retries_individual_poses(self):
        for result in (None, [], {}, {"wrist": object()}, RuntimeError("XR bulk unavailable")):
            with self.subTest(result=result):
                self.device.get_all_virtual_world_poses.reset_mock()
                self.device.get_all_virtual_world_poses.side_effect = result if isinstance(result, Exception) else None
                self.device.get_all_virtual_world_poses.return_value = result

                def step(dt, context):
                    for _ in range(2):
                        self.assertIsNone(self.ex._get_hand_joint_position(self.device, "wrist"))
                    self.assertIsNone(self.ex._read_navigation_hand()[0])

                self.ex._advance_physics_step = step
                self.ex.on_physics_step(0.005, None)
                self.device.get_all_virtual_world_poses.assert_called_once_with()
                self.device.get_virtual_world_pose_desc.assert_not_called()

    def test_invalid_landmark_stops_navigation_without_discarding_other_fingers(self):
        provider = self.bulk_poses()
        provider["index_tip"].pose_matrix.SetTranslateOnly(Gf.Vec3d(float("nan"), 0, 0))
        self.device.get_all_virtual_world_poses.side_effect = None
        self.device.get_all_virtual_world_poses.return_value = provider

        def step(dt, context):
            self.assertIsNone(self.ex._read_navigation_hand()[0])
            curls = self.ex._get_hand_tracking_finger_curls(self.device)
            self.assertNotIn("index", curls)
            self.assertIn("middle", curls)
            self.assertIsNone(self.ex._get_hand_joint_position(self.device, "index_tip"))

        self.ex._advance_physics_step = step
        self.ex.on_physics_step(0.005, None)

    def test_legacy_adapter_without_bulk_api_shares_individual_descriptor_reads(self):
        del self.device.get_all_virtual_world_poses

        def step(dt, context):
            self.assertIsNotNone(self.ex._read_navigation_hand()[0])
            self.assertIsNotNone(self.ex._get_hand_tracking_finger_curls(self.device))

        self.ex._advance_physics_step = step
        self.ex.on_physics_step(0.005, None)
        names = [call.args[0] for call in self.device.get_virtual_world_pose_desc.call_args_list]
        self.assertTrue(names)
        self.assertEqual(len(names), len(set(names)))

    def test_loss_reset_and_callback_exception_discard_bulk_sample(self):
        self.ex._set_grab_candidate = Mock()
        self.ex._release_grabbed_object = Mock()
        self.ex._set_arm_rig_target_visible = Mock()

        def step(dt, context):
            device = self.ex._get_xr_input_device("/user/hand/left")
            self.assertIsNotNone(self.ex._get_hand_tracking_finger_curls(device))
            self.device.flags = 0
            self.ex._deactivate_hand("left")
            self.assertIsNone(self.ex._get_hand_tracking_finger_curls(device))
            self.ex._reset_teleoperation_state()
            self.assertIsNone(self.ex._teleop_step_hand_samples)
            raise RuntimeError("callback ended")

        self.ex._advance_physics_step = step
        with self.assertRaisesRegex(RuntimeError, "callback ended"):
            self.ex.on_physics_step(0.005, None)
        self.assertIsNone(self.ex._teleop_step_hand_samples)
        self.device.get_all_virtual_world_poses.assert_called()
        self.assertEqual(self.device.get_all_virtual_world_poses.call_count, 2)


if __name__ == "__main__":
    unittest.main()
