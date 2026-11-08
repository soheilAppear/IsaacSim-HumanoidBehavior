# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check real USD gaze panels, stereo eye rays, occlusion, and progress states."""

from __future__ import annotations

import math
import types
import unittest

from humanoid_test_support import load_source
from pxr import Gf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade

SCENE = load_source("interactive/humanoid/gaze_controls_scene.py")


class TestGazeControlsScene(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = Usd.Stage.CreateInMemory()
        self.box = UsdGeom.Cube.Define(self.stage, "/World/Box")
        UsdPhysics.CollisionAPI.Apply(self.box.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(self.box.GetPrim())
        self.scene = SCENE.GazeControlsScene()
        self.scene.create(self.stage)
        self.scene.set_camera_pose(Gf.Matrix4d(1))
        self.addCleanup(self.scene.cleanup)

    def ray(self, button: str, pose: Gf.Matrix4d | None = None, eye_x: float = 0.0) -> tuple:
        pose = Gf.Matrix4d(1) if pose is None else pose
        x, y = self.scene.BUTTON_CENTERS[button]
        origin = pose.Transform(Gf.Vec3d(eye_x, 0, 0))
        target = pose.Transform(Gf.Vec3d(x, y, -self.scene.DEPTH))
        return origin, target - origin

    def label(self, suffix: str = "Status") -> str:
        prim = self.stage.GetPrimAtPath(self.scene.root_path + "/" + suffix)
        if suffix in ("Status", "FeedbackAction"):
            labels = self.visible_labels(suffix)
            self.assertLessEqual(len(labels), 1)
            visible_label = labels[0] if labels else ""
            self.assertEqual(prim.GetAttribute("gazeControls:label").Get(), visible_label)
            return visible_label
        if not self.is_drawn(prim):
            return ""
        return prim.GetAttribute("gazeControls:label").Get()

    def visible_labels(self, suffix: str) -> list[str]:
        parent = self.stage.GetPrimAtPath(self.scene.root_path + "/" + suffix)
        return [
            child.GetAttribute("gazeControls:label").Get()
            for child in parent.GetChildren()
            if child.IsA(UsdGeom.Mesh) and self.is_drawn(child)
        ]

    def is_drawn(self, prim: Usd.Prim) -> bool:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        shader = UsdShade.Shader(self.stage.GetPrimAtPath(str(material.GetPath()) + "/Shader"))
        return (
            UsdGeom.Imageable(prim).GetVisibilityAttr().Get() != UsdGeom.Tokens.invisible
            and shader.GetInput("opacity").Get() >= shader.GetInput("opacityThreshold").Get()
        )

    def assert_mask_translation(self, prim: Usd.Prim) -> None:
        operations = UsdGeom.Xformable(prim).GetOrderedXformOps()
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0].GetOpType(), UsdGeom.XformOp.TypeTranslate)
        expected_z = 0.0 if self.is_drawn(prim) else 2.0 * self.scene.DEPTH
        self.assertEqual(operations[0].Get(), Gf.Vec3d(0.0, 0.0, expected_z))

    def label_geometry(self) -> dict:
        paths = tuple(self.scene.root_path + suffix for suffix in ("/Status/", "/StatusCards/", "/FeedbackAction/"))
        result = {}
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if prim.IsA(UsdGeom.Mesh) and (path.startswith(paths) or path == self.scene.root_path + "/FeedbackPrefix"):
                mesh = UsdGeom.Mesh(prim)
                result[path] = (
                    tuple(tuple(point) for point in mesh.GetPointsAttr().Get()),
                    tuple(mesh.GetFaceVertexCountsAttr().Get()),
                    tuple(mesh.GetFaceVertexIndicesAttr().Get()),
                    tuple(prim.GetRelationship("material:binding").GetTargets()),
                    UsdGeom.Imageable(prim).GetVisibilityAttr().Get(),
                    tuple(str(op.GetOpName()) for op in UsdGeom.Xformable(prim).GetOrderedXformOps()),
                )
        return result

    def progress_faces(self, button: str) -> int:
        mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath(f"{self.scene.root_path}/{button}/Progress"))
        return len(mesh.GetFaceVertexCountsAttr().Get())

    def test_scene_is_nonphysical_emissive_double_sided_geometry(self) -> None:
        root = self.stage.GetPrimAtPath(self.scene.root_path)
        mesh_count = 0
        for prim in Usd.PrimRange(root):
            self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI), prim.GetPath())
            self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI), prim.GetPath())
            self.assertFalse(prim.IsA(UsdPhysics.Joint), prim.GetPath())
            if prim.IsA(UsdGeom.Mesh):
                mesh_count += 1
                self.assertTrue(UsdGeom.Mesh(prim).GetDoubleSidedAttr().Get())
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                self.assertTrue(material)
                shader = UsdShade.Shader(self.stage.GetPrimAtPath(str(material.GetPath()) + "/Shader"))
                self.assertIsNotNone(shader.GetInput("emissiveColor").Get())
        self.assertGreaterEqual(mesh_count, 22)
        self.assertTrue(self.box.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
        self.assertTrue(self.box.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI))

    def test_all_buttons_match_both_eye_origins_and_scaled_directions(self) -> None:
        for button in self.scene.BUTTON_CENTERS:
            for eye_x in (-0.032, 0, 0.032):
                with self.subTest(button=button, eye_x=eye_x):
                    origin, direction = self.ray(button, eye_x=eye_x)
                    self.assertEqual(self.scene.hit_test(origin, direction * 5), button)

    def test_translated_rotated_camera_matches_authored_transform(self) -> None:
        pose = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3), 73))
        pose.SetTranslateOnly(Gf.Vec3d(5, -2, 1.7))
        self.scene.set_camera_pose(pose)
        actual = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.scene.root_path))
        self.assertEqual(actual, pose)
        for button in self.scene.BUTTON_CENTERS:
            self.assertEqual(self.scene.hit_test(*self.ray(button, pose)), button)
        pose.SetTranslateOnly(Gf.Vec3d(100, 200, 300))
        self.assertEqual(
            UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.scene.root_path)), actual
        )

    def test_camera_pose_only_authors_changed_transform_or_visibility(self) -> None:
        self.scene.update({"state": "idle"})
        notices = []

        def on_changed(notice, sender):
            notices.append(
                (
                    tuple(str(path) for path in notice.GetResyncedPaths()),
                    tuple(str(path) for path in notice.GetChangedInfoOnlyPaths()),
                )
            )

        listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, on_changed, self.stage)
        self.addCleanup(listener.Revoke)
        for _ in range(3):
            self.scene.set_camera_pose(Gf.Matrix4d(1))
            self.scene.update({"state": "idle"})
        self.assertEqual(notices, [])

        pose = Gf.Matrix4d().SetTranslate(Gf.Vec3d(1, 2, 3))
        self.scene.set_camera_pose(pose)
        self.assertEqual(notices, [((), (self.scene.root_path + ".xformOp:transform",))])
        self.assertEqual(self.scene.hit_test(*self.ray("forward", pose)), "forward")
        notices.clear()

        invalid = Gf.Matrix4d().SetScale(Gf.Vec3d(1, 2, 1))
        for _ in range(2):
            with self.assertRaises(ValueError):
                self.scene.set_camera_pose(invalid)
            self.assertIsNone(self.scene.hit_test(*self.ray("forward", pose)))
        self.assertEqual(notices, [((), (self.scene.root_path + ".visibility",))])
        notices.clear()

        self.scene.set_camera_pose(pose)
        self.assertEqual(notices, [((), (self.scene.root_path + ".visibility",))])
        self.assertEqual(self.scene.hit_test(*self.ray("forward", pose)), "forward")
        notices.clear()
        for _ in range(3):
            self.scene.set_camera_pose(pose)
        self.assertEqual(notices, [])

    def test_center_button_gaps_and_backward_rays_do_not_select(self) -> None:
        for direction in ((0, 0, -1), (-0.55, 0, -1), (0.55, 0, -1), (0, 0, 1), (-0.55, 0.7, -1)):
            with self.subTest(direction=direction):
                self.assertIsNone(self.scene.hit_test((0, 0, 0), direction))
        self.assertIsNone(self.scene.hit_test((-0.55, 0.22, -2), (0, 0, -1)))

    def test_nearest_occluder_distance_is_in_world_meters(self) -> None:
        origin, direction = self.ray("forward")
        distance = direction.GetLength()
        self.assertIsNone(self.scene.hit_test(origin, direction * 9, distance - 0.01))
        self.assertIsNone(self.scene.hit_test(origin, direction, distance))
        self.assertEqual(self.scene.hit_test(origin, direction, distance + 0.01), "forward")

    def test_invalid_rays_and_occlusion_distances_fail_closed(self) -> None:
        for origin, direction in (
            ((math.nan, 0, 0), (0, 0, -1)),
            ((0, 0, 0), (math.inf, 0, -1)),
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0), (0, 0, -1)),
            (None, (0, 0, -1)),
        ):
            with self.subTest(origin=origin, direction=direction):
                self.assertIsNone(self.scene.hit_test(origin, direction))
        for distance in (math.nan, math.inf, -1, 0):
            self.assertIsNone(self.scene.hit_test(*self.ray("forward"), distance))

    def test_invalid_camera_clears_cached_hit_transform_and_hides_controls(self) -> None:
        for pose in (Gf.Matrix4d().SetScale(Gf.Vec3d(1, 2, 1)), Gf.Matrix4d().SetTranslate(Gf.Vec3d(math.nan, 0, 0))):
            with self.subTest(pose=pose):
                self.scene.set_camera_pose(Gf.Matrix4d(1))
                with self.assertRaises(ValueError):
                    self.scene.set_camera_pose(pose)
                self.assertIsNone(self.scene.hit_test(*self.ray("forward")))
                root = UsdGeom.Imageable(self.stage.GetPrimAtPath(self.scene.root_path))
                self.assertEqual(root.GetVisibilityAttr().Get(), UsdGeom.Tokens.invisible)

    def test_disabled_back_stays_visible_labeled_and_never_selects(self) -> None:
        self.scene.cleanup()
        self.scene = SCENE.GazeControlsScene(backwards_enabled=False)
        self.scene.create(self.stage)
        self.scene.set_camera_pose(Gf.Matrix4d(1))
        self.assertEqual(self.label("back/Label"), "BACK")
        self.assertEqual(self.label("back/Disabled"), "REVERSE OFF")
        self.assertIsNone(self.scene.hit_test(*self.ray("back")))
        self.assertEqual(self.scene.hit_test(*self.ray("forward")), "forward")
        self.scene.update({"state": "dwelling", "target": "back", "progress": 1})
        self.assertEqual(self.progress_faces("back"), 0)

    def test_progress_ring_and_status_follow_controller_without_new_prims(self) -> None:
        paths = [str(prim.GetPath()) for prim in self.stage.Traverse()]
        self.scene.update(types.SimpleNamespace(state="dwelling", target="forward", progress=0.5))
        self.assertEqual(self.label(), "DWELL")
        self.assertEqual(self.progress_faces("forward"), 32)
        self.scene.update({"state": "preview", "target": "turn_left", "progress": 1})
        self.assertEqual(self.label(), "PREVIEW")
        self.assertEqual(self.progress_faces("forward"), 0)
        self.assertEqual(self.progress_faces("turn_left"), 64)
        self.scene.update({"state": "active", "target": "forward", "progress": 0.25})
        self.assertEqual(self.label(), "RUNNING")
        self.assertEqual(self.progress_faces("forward"), 16)
        self.assertEqual(paths, [str(prim.GetPath()) for prim in self.stage.Traverse()])

    def test_lost_eye_and_disabled_states_clear_stale_progress(self) -> None:
        self.scene.update({"state": "dwelling", "target": "forward", "progress": 1})
        self.scene.update({"state": "dwelling", "target": "forward", "progress": 1}, eye_valid=False)
        self.assertEqual(self.label(), "NO EYE")
        self.assertTrue(all(self.progress_faces(button) == 0 for button in self.scene.BUTTON_CENTERS))
        self.scene.update({"state": "unavailable", "target": "forward", "progress": 1})
        self.assertEqual(self.label(), "DISABLED")
        self.assertEqual(self.progress_faces("forward"), 0)
        self.scene.update({"state": "dwelling", "target": "forward", "progress": math.nan})
        self.assertEqual(self.progress_faces("forward"), 0)

    def test_preview_heading_and_last_action_feedback_persist_until_controller_reset(self) -> None:
        self.assertEqual(self.label("Title"), "PREVIEW ONLY")
        self.scene.update({"state": "preview", "target": "turn_left", "progress": 1, "last_action": "turn_left"})
        self.assertEqual(self.label("FeedbackPrefix"), "WOULD")
        self.assertEqual(self.label("FeedbackAction"), "TURN LEFT")
        for state in ("awaiting_release", "idle", "unavailable"):
            with self.subTest(state=state):
                self.scene.update({"state": state, "target": None, "progress": 0, "last_action": "turn_left"})
                self.assertEqual(self.label("FeedbackPrefix"), "WOULD")
                self.assertEqual(self.label("FeedbackAction"), "TURN LEFT")
        self.scene.update({"state": "awaiting_release", "last_action": None})
        self.assertEqual(self.label("FeedbackPrefix"), "")
        self.assertEqual(self.label("FeedbackAction"), "")

    def test_text_backings_are_nonphysical_unselectable_and_footer_tracks_feedback(self) -> None:
        for suffix in ("TitleBacking", "StatusBacking", "FeedbackBacking"):
            prim = self.stage.GetPrimAtPath(self.scene.root_path + "/" + suffix)
            self.assertTrue(prim.IsA(UsdGeom.Mesh))
            self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))
            self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            expected = "feedback_backing" if suffix == "FeedbackBacking" else "panel"
            self.assertEqual(str(material.GetPath()), self.scene.root_path + "/Materials/" + expected)
            shader = UsdShade.Shader(self.stage.GetPrimAtPath(str(material.GetPath()) + "/Shader"))
            panel_shader = UsdShade.Shader(self.stage.GetPrimAtPath(self.scene.root_path + "/Materials/panel/Shader"))
            self.assertEqual(shader.GetInput("emissiveColor").Get(), panel_shader.GetInput("emissiveColor").Get())
        for direction in ((-0.55, 0.465, -1), (0.55, 0.465, -1), (-0.55, -0.49, -1)):
            self.assertIsNone(self.scene.hit_test((0, 0, 0), direction))
        footer = self.stage.GetPrimAtPath(self.scene.root_path + "/FeedbackBacking")
        self.assertFalse(self.is_drawn(footer))
        self.scene.update({"state": "preview", "last_action": "turn_right"})
        self.assertTrue(self.is_drawn(footer))
        self.scene.update({"state": "idle", "last_action": "turn_right"})
        self.assertTrue(self.is_drawn(footer))
        self.scene.update({"state": "awaiting_release", "last_action": None})
        self.assertFalse(self.is_drawn(footer))

    def test_live_heading_never_claims_preview(self) -> None:
        self.scene.cleanup()
        self.scene = SCENE.GazeControlsScene(preview=False)
        self.scene.create(self.stage)
        self.assertEqual(self.label("Title"), "LIVE WALK")
        self.scene.update({"state": "active", "target": "forward", "progress": 0.5, "last_action": "forward"})
        self.assertEqual(self.label("FeedbackPrefix"), "LAST")
        self.assertEqual(self.label("FeedbackAction"), "FORWARD")

    def test_eye_loss_does_not_hide_accepted_motion_or_fault(self) -> None:
        for state, expected in (("active", "RUNNING"), ("stopping", "STOPPING"), ("fault", "FAULT")):
            with self.subTest(state=state):
                self.scene.update({"state": state, "target": "forward", "progress": 0.5}, eye_valid=False)
                self.assertEqual(self.label(), expected)
                self.assertEqual(self.progress_faces("forward"), 32 if state != "fault" else 0)

    def test_controller_status_stays_visible_without_eye_tracking(self) -> None:
        before = self.label_geometry()
        for state, label in (
            ("controller", "STICKS"),
            ("controller_preview", "STICK PREVIEW"),
            ("controller_wait", "CENTER STICKS"),
        ):
            with self.subTest(state=state):
                self.scene.update({"state": state, "target": "forward", "progress": 1.0}, eye_valid=False)
                self.assertEqual(self.label(), label)
                self.assertEqual(self.progress_faces("forward"), 0)
        self.assertEqual(self.label_geometry(), before)

    def test_status_and_feedback_switch_opacity_without_rebuilding_text(self) -> None:
        before = self.label_geometry()
        self.assertEqual(len(before), 29)  # Twelve status/card pairs, four actions, and one prefix.
        cards = self.stage.GetPrimAtPath(self.scene.root_path + "/StatusCards").GetChildren()
        self.assertEqual(len(cards), 12)
        backing = UsdGeom.Mesh(self.stage.GetPrimAtPath(self.scene.root_path + "/StatusBacking"))
        for card in cards:
            for point, fixed in zip(UsdGeom.Mesh(card).GetPointsAttr().Get(), backing.GetPointsAttr().Get()):
                self.assertEqual((point[0], point[1]), (fixed[0], fixed[1]))
                self.assertAlmostEqual(point[2], -self.scene.DEPTH + 0.001, places=6)
        material_paths = set()
        for path in before:
            prim = self.stage.GetPrimAtPath(path)
            self.assertEqual(UsdGeom.Imageable(prim).GetVisibilityAttr().Get(), UsdGeom.Tokens.inherited)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            material_paths.add(str(material.GetPath()))
            shader = UsdShade.Shader(self.stage.GetPrimAtPath(str(material.GetPath()) + "/Shader"))
            self.assertEqual(shader.GetInput("opacityThreshold").Get(), 0.5)
            self.assertIn(shader.GetInput("opacity").Get(), (0.0, 1.0))
            self.assertEqual(shader.GetInput("emissiveColor").Get(), shader.GetInput("diffuseColor").Get())
            self.assertGreater(sum(shader.GetInput("emissiveColor").Get()), 0.0)
            self.assert_mask_translation(prim)
        self.assertEqual(len(material_paths), 29)
        self.assertEqual(self.visible_labels("Status"), ["NO EYE"])
        self.assertEqual([card.GetName() for card in cards if self.is_drawn(card)], ["no_eye"])
        self.assertEqual(self.visible_labels("FeedbackAction"), [])
        expected_statuses = {
            "idle": "READY",
            "dwelling": "DWELL",
            "preview": "PREVIEW",
            "active": "RUNNING",
            "stopping": "STOPPING",
            "awaiting_release": "LOOK AWAY",
            "unavailable": "DISABLED",
            "fault": "FAULT",
            "no_eye": "NO EYE",
            "controller": "STICKS",
            "controller_preview": "STICK PREVIEW",
            "controller_wait": "CENTER STICKS",
        }
        for state, label in expected_statuses.items():
            for action, action_label in (
                ("forward", "FORWARD"),
                ("back", "BACK"),
                ("turn_left", "TURN LEFT"),
                ("turn_right", "TURN RIGHT"),
            ):
                with self.subTest(state=state, action=action):
                    self.scene.update({"state": state, "last_action": action})
                    self.assertEqual(self.visible_labels("Status"), [label])
                    self.assertEqual(self.visible_labels("FeedbackAction"), [action_label])
                    self.assertEqual([card.GetName() for card in cards if self.is_drawn(card)], [state])
                    for path in (*before, self.scene.root_path + "/FeedbackBacking"):
                        self.assert_mask_translation(self.stage.GetPrimAtPath(path))
        # Actual RTX regression: the old orange status and the previous footer
        # must both be masked after a transition, independent of USD visibility.
        self.scene.update({"state": "stopping", "last_action": "turn_right"})
        self.scene.update({"state": "idle", "last_action": "forward"})
        self.assertEqual(self.visible_labels("Status"), ["READY"])
        self.assertEqual(self.visible_labels("FeedbackAction"), ["FORWARD"])
        self.assertEqual([card.GetName() for card in cards if self.is_drawn(card)], ["idle"])
        for old_path in ("Status/stopping", "FeedbackAction/turn_right"):
            self.assertFalse(self.is_drawn(self.stage.GetPrimAtPath(self.scene.root_path + "/" + old_path)))
        self.scene.update({"state": "idle", "last_action": None})
        self.assertEqual(self.visible_labels("Status"), ["READY"])
        self.assertEqual(self.visible_labels("FeedbackAction"), [])
        self.assertEqual([card.GetName() for card in cards if self.is_drawn(card)], ["idle"])
        self.assertEqual(self.label("FeedbackPrefix"), "")
        for path in (*before, self.scene.root_path + "/FeedbackBacking"):
            self.assert_mask_translation(self.stage.GetPrimAtPath(path))
        self.assertEqual(self.label_geometry(), before)

    def test_cleanup_and_external_root_deletion_fail_closed(self) -> None:
        self.stage.RemovePrim(self.scene.root_path)
        self.assertIsNone(self.scene.hit_test(*self.ray("forward")))
        self.scene.update({"state": "active", "target": "forward", "progress": 1})
        self.scene.cleanup()
        self.scene.cleanup()
        self.assertTrue(self.stage.GetPrimAtPath("/World/Box").IsValid())
        self.scene.create(self.stage)
        self.assertIsNone(self.scene.hit_test(*self.ray("forward")))
        self.scene.set_camera_pose(Gf.Matrix4d(1))
        self.assertEqual(self.scene.hit_test(*self.ray("forward")), "forward")


if __name__ == "__main__":
    unittest.main()
