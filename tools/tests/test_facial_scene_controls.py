# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check real USD facial-toggle geometry and stable headset feedback."""

from __future__ import annotations

import math
import types
import unittest

from humanoid_test_support import load_source
from pxr import Gf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade

GAZE = load_source("interactive/humanoid/gaze_controls_scene.py")
FACE = load_source("interactive/humanoid/facial_controls_scene.py", GazeControlsScene=GAZE.GazeControlsScene)


class TestFacialSceneControls(unittest.TestCase):
    """Verify facial feedback remains useful during a wink and leaves gaze geometry intact."""

    def setUp(self) -> None:
        self.stage = Usd.Stage.CreateInMemory()
        self.scene = FACE.FacialControlsScene()
        self.scene.create(self.stage)
        self.scene.set_camera_pose(Gf.Matrix4d(1))
        self._notice_callbacks = []
        self.addCleanup(self.scene.cleanup)

    def ray(self, *, eye_x: float = 0.0, pose: Gf.Matrix4d | None = None, button: str | None = None) -> tuple:
        """Aim an eye at a displayed face toggle or ordinary gaze-button center."""
        pose = Gf.Matrix4d(1) if pose is None else pose
        x, y = self.scene.FACE_CENTER if button is None else self.scene.BUTTON_CENTERS[button]
        origin = pose.Transform(Gf.Vec3d(eye_x, 0, 0))
        target = pose.Transform(Gf.Vec3d(x, y, -self.scene.DEPTH))
        return origin, target - origin

    def is_drawn(self, suffix: str) -> bool:
        """Read the actual bound shader and preauthored text-mask transform."""
        prim = self.stage.GetPrimAtPath(f"{self.scene.root_path}/{suffix}")
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        shader = UsdShade.Shader(self.stage.GetPrimAtPath(f"{material.GetPath()}/Shader"))
        visible = shader.GetInput("opacity").Get() >= shader.GetInput("opacityThreshold").Get()
        operations = UsdGeom.Xformable(prim).GetOrderedXformOps()
        if operations:
            self.assertEqual(operations[0].Get(), Gf.Vec3d(0, 0, 0 if visible else 2 * self.scene.DEPTH))
        return visible

    def status_label(self) -> str:
        """Read the visible status mesh and its text metadata together."""
        root = self.stage.GetPrimAtPath(f"{self.scene.root_path}/Status")
        visible = [
            prim.GetAttribute("gazeControls:label").Get()
            for prim in root.GetChildren()
            if prim.IsA(UsdGeom.Mesh) and self.is_drawn(f"Status/{prim.GetName()}")
        ]
        self.assertEqual(len(visible), 1)
        self.assertEqual(root.GetAttribute("gazeControls:label").Get(), visible[0])
        return visible[0]

    def command(self, state: str, action: str | None = None, progress: float = 0.0) -> object:
        """Provide the public facial-command display fields without simulating physics."""
        return types.SimpleNamespace(state=state, target=action, last_action=action, progress=progress)

    def observe_changes(self) -> list:
        """Collect notices to detect unnecessary stage writes and topology changes."""
        changes = []

        def changed(notice, sender) -> None:
            changes.append((tuple(notice.GetResyncedPaths()), tuple(notice.GetChangedInfoOnlyPaths())))

        # `Tf.Notice` keeps a weak callable reference. Retain the callback after
        # this helper returns so silence cannot masquerade as an idempotent HUD.
        self._notice_callbacks.append(changed)
        listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, changed, self.stage)
        self.addCleanup(listener.Revoke)
        return changes

    def test_create_is_idempotent_and_preserves_selected_mode(self) -> None:
        self.scene.update_face_toggle(True, 0.5)
        self.scene.update_facial(self.command("active", "forward", 1.0))
        before = self.stage.GetRootLayer().ExportToString()
        changes = self.observe_changes()
        self.scene.create(self.stage)
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), before)
        self.assertEqual(changes, [])
        self.assertTrue(self.is_drawn("face_on"))
        self.assertFalse(self.is_drawn("face_off"))

    def test_toggle_matches_stereo_rays_after_camera_translation_and_rotation(self) -> None:
        pose = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3), 73))
        pose.SetTranslateOnly(Gf.Vec3d(5, -2, 1.7))
        self.scene.set_camera_pose(pose)
        for eye_x in (-0.032, 0, 0.032):
            with self.subTest(eye_x=eye_x):
                origin, direction = self.ray(eye_x=eye_x, pose=pose)
                self.assertTrue(self.scene.hit_test_face_toggle(origin, direction * 9))
                self.assertIsNone(self.scene.hit_test(origin, direction))

    def test_toggle_and_ordinary_buttons_never_select_each_other(self) -> None:
        for button in self.scene.BUTTON_CENTERS:
            origin, direction = self.ray(button=button)
            self.assertFalse(self.scene.hit_test_face_toggle(origin, direction))
            self.assertEqual(self.scene.hit_test(origin, direction), button)
        self.assertFalse(self.scene.hit_test_face_toggle((0, 0, 0), (0, 0, -1)))
        self.assertTrue(self.scene.hit_test_face_toggle(*self.ray()))
        self.assertIsNone(self.scene.hit_test(*self.ray()))

    def test_occluder_distance_is_world_space_and_equal_depth_is_occluded(self) -> None:
        for eye_x in (-0.032, 0.032):
            origin, direction = self.ray(eye_x=eye_x)
            distance = direction.GetLength()
            self.assertFalse(self.scene.hit_test_face_toggle(origin, direction * 5, distance - 0.01))
            self.assertFalse(self.scene.hit_test_face_toggle(origin, direction, distance))
            self.assertTrue(self.scene.hit_test_face_toggle(origin, direction, distance + 0.01))

    def test_bad_rays_backward_rays_and_distances_fail_closed(self) -> None:
        for origin, direction in (
            (None, (0, 0, -1)),
            ((0, 0), (0, 0, -1)),
            ((0, 0, 0), (0, 0)),
            ((math.nan, 0, 0), (0, 0, -1)),
            ((0, 0, 0), (math.inf, 0, -1)),
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, 0), (0, 0, 1)),
            ((0.55, -0.49, -2), (0, 0, -1)),
        ):
            with self.subTest(origin=origin, direction=direction):
                self.assertFalse(self.scene.hit_test_face_toggle(origin, direction))
        for distance in (math.nan, math.inf, 0, -1):
            self.assertFalse(self.scene.hit_test_face_toggle(*self.ray(), max_distance=distance))

    def test_invalid_camera_prevents_toggle_and_recovery_restores_hit(self) -> None:
        with self.assertRaises(ValueError):
            self.scene.set_camera_pose(Gf.Matrix4d().SetScale(Gf.Vec3d(1, 2, 1)))
        self.assertFalse(self.scene.hit_test_face_toggle(*self.ray()))
        self.scene.set_camera_pose(Gf.Matrix4d(1))
        self.assertTrue(self.scene.hit_test_face_toggle(*self.ray()))

    def test_toggle_labels_mask_and_progress_updates_are_bounded(self) -> None:
        ring = UsdGeom.Mesh(self.stage.GetPrimAtPath(f"{self.scene.root_path}/FaceToggleProgress"))
        for enabled, progress, count in ((False, 0, 0), (True, 0.5, 32), (True, 3, 64), (False, math.nan, 0)):
            self.scene.update_face_toggle(enabled, progress)
            self.assertEqual(self.is_drawn("face_on"), enabled)
            self.assertEqual(self.is_drawn("face_off"), not enabled)
            self.assertEqual(len(ring.GetFaceVertexCountsAttr().Get()), count)
            self.assertEqual(len(ring.GetFaceVertexIndicesAttr().Get()), 4 * count)

    def test_facial_states_show_commands_without_an_eye_sample(self) -> None:
        # A wink can invalidate gaze. `update_facial` deliberately uses face
        # state instead of overwriting valid face feedback with NO EYE.
        self.scene.update({"state": "idle"}, eye_valid=False)
        self.assertEqual(self.status_label(), "NO EYE")
        for state, label in (
            ("idle", "FACE READY"),
            ("awaiting_neutral", "RELAX FACE"),
            ("awaiting_release", "RELAX FACE"),
            ("confirming", "HOLD L LIPS"),
            ("blink_pause", "OPEN EYES"),
            ("unavailable", "NO FACE DATA"),
            ("preview", "PREVIEW"),
            ("active", "RUNNING"),
            ("stopping", "STOPPING"),
            ("fault", "FAULT"),
        ):
            with self.subTest(state=state):
                self.scene.update_facial(self.command(state, "strafe_left", 0.5))
                self.assertEqual(self.status_label(), label)
                feedback = self.stage.GetPrimAtPath(f"{self.scene.root_path}/FeedbackAction")
                self.assertEqual(feedback.GetAttribute("gazeControls:label").Get(), "STEP LEFT")

    def test_confirmation_feedback_names_the_pending_expression(self) -> None:
        for action, label in (
            ("turn_left", "HOLD L WINK"),
            ("turn_right", "HOLD R WINK"),
            ("forward", "HOLD MOUTH"),
            ("back", "HOLD PUCKER"),
            ("strafe_right", "HOLD R LIPS"),
        ):
            self.scene.update_facial(self.command("confirming", action, 0.5))
            self.assertEqual(self.status_label(), label)
        for action, label in (("turn_left", "OPEN R EYE"), ("turn_right", "OPEN L EYE")):
            self.scene.update_facial(self.command("blink_pause", action, 0.0))
            self.assertEqual(self.status_label(), label)

    def test_facial_updates_preserve_meshes_and_bindings_without_resync(self) -> None:
        def geometry() -> dict:
            return {
                str(prim.GetPath()): (
                    tuple(UsdGeom.Mesh(prim).GetPointsAttr().Get()),
                    tuple(prim.GetRelationship("material:binding").GetTargets()),
                )
                for prim in self.stage.Traverse()
                if prim.IsA(UsdGeom.Mesh)
            }

        before = geometry()
        changes = self.observe_changes()
        for state in ("idle", "confirming", "preview", "active", "awaiting_neutral", "unavailable"):
            self.scene.update_face_toggle(True, 0.5)
            self.scene.update_facial(self.command(state, "strafe_right", 0.5))
        self.scene.update_face_toggle(False, 0)
        self.assertEqual(geometry(), before)
        self.assertTrue(changes)
        self.assertTrue(all(not resynced for resynced, _ in changes))
        changes.clear()
        self.scene.update_facial(self.command("idle"))
        changes.clear()
        for _ in range(5):
            self.scene.update_face_toggle(False, 0)
            self.scene.update_facial(self.command("idle"))
        self.assertEqual(changes, [])

    def test_face_mode_dims_only_gaze_buttons_and_restores_original_colors(self) -> None:
        def color(name: str) -> tuple:
            shader = UsdShade.Shader(self.stage.GetPrimAtPath(f"{self.scene.root_path}/Materials/{name}/Shader"))
            return tuple(shader.GetInput("emissiveColor").Get())

        names = (*self.scene.BUTTON_CENTERS, "face_gaze_text", "text", "face_on", "face_off")
        original = {name: color(name) for name in names}
        self.scene.update_face_toggle(True, 0)
        expected = tuple(Gf.Vec3f(*self.scene._STATE_COLORS["disabled"]))
        for name in (*self.scene.BUTTON_CENTERS, "face_gaze_text"):
            self.assertEqual(color(name), expected)
        for name in ("text", "face_on", "face_off"):
            self.assertEqual(color(name), original[name])
        self.scene.update_face_toggle(False, 0)
        self.assertEqual({name: color(name) for name in names}, original)

    def test_hud_remains_nonphysical_and_survives_recreate(self) -> None:
        for prim in self.stage.Traverse():
            self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))
            self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        self.scene.cleanup()
        self.assertFalse(self.scene.hit_test_face_toggle((0, 0, 0), (0.55, -0.49, -1)))
        self.scene.update_face_toggle(True, 1)
        self.scene.update_facial(self.command("idle"))
        self.scene.create(self.stage)
        self.scene.set_camera_pose(Gf.Matrix4d(1))
        self.assertTrue(self.scene.hit_test_face_toggle(*self.ray()))
        self.assertTrue(self.is_drawn("face_off"))


if __name__ == "__main__":
    unittest.main()
