# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive humanoid robot simulation example using H1 robot with GPU-accelerated physics and keyboard control."""

import csv
import json
import math
import random
import time
from pathlib import Path

import carb
import isaacsim.core.experimental.utils.stage as stage_utils
import omni
import omni.appwindow
from isaacsim.core.deprecation_manager import import_module
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.examples.base.base_sample_experimental import BaseSample
from isaacsim.robot.policy.examples.interactive.utils import (
    restore_physics_simulation_state,
    snapshot_physics_simulation_state,
)
from isaacsim.robot.policy.examples.robots import H1FlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


class HumanoidExample(BaseSample):
    """A humanoid robot simulation example using H1 robot with GPU-accelerated physics.

    This class demonstrates a complete humanoid robot simulation setup with real-time control capabilities.
    It configures a high-frequency physics simulation (200 Hz) with GPU acceleration and provides keyboard-based
    control for the H1 humanoid robot. The example includes proper scene setup, physics callbacks, and cleanup
    management.

    The simulation uses optimized settings with 200 Hz physics timestep and 25 Hz rendering to ensure smooth
    real-time performance. The H1 robot is controlled through a policy-based system that processes movement
    commands and maintains balance during locomotion.

    Keyboard controls:
        - NUMPAD_8 or UP: Move forward
        - NUMPAD_4 or LEFT: Turn left
        - NUMPAD_6 or RIGHT: Turn right

    VR extensions (this fork — see HUMANOID_VR_CONTROL.md for the full guide):
        - Headset gait: bob the HMD up/down (step in place) to walk forward; a
          horizontal-motion gate suppresses false triggers from nodding on the spot.
          Currently DISABLED by default (_headset_gait_enabled = False) while step
          detection is tuned; the HMD pose is still read every step for logging.
        - Quest Pro eye gaze (eye_gaze_tracker.py): the runtime's calibrated unified
          gaze (its own fusion of both eyes) drawn as a red ray with a large
          blood-red marker sphere at the gazed collider (boxes, ground), gazed box
          tinted yellow, and live "[EyeGaze] looking at ..." terminal events on
          every target change.
        - First-person eye camera: the viewport/XR camera follows the H1 head at eye height.
        - Hand-tracking / controller arm teleoperation, plus a grab system for sample boxes.
        - Behavioral session logging: each load creates a session folder under
          ~/BehavioralCollection/raw_sessions/ holding behavior.csv, hand_tracking.csv,
          gaze.csv, object_states.csv, frame_timestamps.csv + eye-camera PNG frames,
          and metadata.json — the raw input for the learning/ pipeline in this repo.

    The example automatically handles robot initialization after scene reset and manages GPU memory resources
    through proper cleanup routines. Physics tensors are validated each step to ensure robust simulation
    restart capabilities.
    """

    def __init__(self):
        super().__init__()
        # Configure simulation settings for GPU dynamics with high-frequency physics
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 200.0  # 200 Hz physics
        self._world_settings["rendering_dt"] = 1.0 / 90.0  # VR-friendly rendering cadence
        self._world_settings["device"] = "cuda"
        self._world_settings["backend"] = "torch"

        self._base_command = None
        self._physics_ready = False
        self.h1 = None
        self._physics_callback_id = None
        self._event_timer_callback = None
        self._sub_keyboard = None
        self._input = None
        self._keyboard = None
        self._gamepad = None
        self._xr_core = None
        self._xr_input_status_logged = False
        self._keyboard_command = None
        self._controller_command = None
        self._controller_deadzone = 0.15
        self._locomotion_trigger_threshold = 0.25
        self._arm_pose_enable_threshold = 0.25
        self._max_forward_speed = 1.0
        self._max_yaw_speed = 1.0
        self._command_response_time = 0.0
        self._headset_gait_enabled = False    # temporarily disabled: step detection not yet stable; flip to True to restore
        self._headset_gait_forward_intensity = 1.0    # full speed walk per detected step
        self._headset_gait_min_amplitude = 0.012      # lowered: detect smaller head bobs (~1.2 cm)
        self._headset_gait_min_step_interval = 0.18   # allow faster cadence
        self._headset_gait_max_extremum_gap = 0.95
        self._headset_gait_baseline_time = 1.4
        self._headset_gait_filter_time = 0.05         # faster low-pass response
        self._headset_gait_velocity_deadzone = 0.008  # more sensitive direction detection
        self._headset_gait_pulse_duration = 0.50      # longer walk burst so motion is visible
        self._headset_gait_attack_time = 0.040
        self._headset_gait_release_time = 0.28
        self._headset_gait_pose_candidates = ("", "head", "hmd")
        self._headset_gait_status_logged = False
        self._headset_gait_time = 0.0
        self._headset_gait_height_baseline = None
        self._headset_gait_filtered_height = None
        self._headset_gait_velocity_sign = 0
        self._headset_gait_last_peak_height = None
        self._headset_gait_last_peak_time = None
        self._headset_gait_last_trough_height = None
        self._headset_gait_last_trough_time = None
        self._headset_gait_last_extremum_time = -999.0
        self._headset_gait_last_step_time = -999.0
        self._headset_gait_pulse_time_remaining = 0.0
        self._headset_gait_output = 0.0
        self._head_camera_path = "/World/H1_HeadCamera"
        self._h1_head_prim_path = None
        self._h1_head_prim_lookup_complete = False
        self._head_camera_transform_op = None
        self._physics_step_error_logged = set()      # (subsystem, error) pairs already warned about
        self._first_person_head_forward_offset = 0.35   # well ahead of the head so the camera never meets
                                                         # the robot mesh and the view stays fully clear
        self._first_person_head_up_offset = 0.0         # extra fine-tune on top of the eye height below
        self._first_person_eye_height_above_base = 0.75  # m above the pelvis/base link: ~5 cm above the top of
                                                         # the H1 head — close to first-person but clear of the
                                                         # head mesh (0.45 = strict eye level inside the head)
        self._head_camera_yaw_sign = 1.0                # flip to -1.0 only if the DESKTOP view turns opposite
                                                        # to the robot; the in-VR reversal was caused by
                                                        # per-step camera forcing, fixed by the XR anchor below
        # XR camera modes (the reversal/frozen-camera saga, in order of discovery):
        #   "camera_lock"   - schedule_set_camera(robot head pose) every step. Follows
        #                     the robot but cancels the user's own head rotation.
        #   "custom_anchor" - XR custom-anchor prim. Natural head tracking, but this
        #                     Kit build does not track a MOVING anchor prim, so the
        #                     camera stays behind while the robot walks.
        #   "head_compose"  - schedule_set_camera(physical_head_pose · axis_fix · anchor).
        #                     The runtime internally subtracts the current head pose;
        #                     pre-multiplying it in makes that subtraction cancel
        #                     itself instead of the head motion. Follows the robot AND
        #                     keeps natural head tracking. Absolute, so no drift.
        self._xr_camera_mode = "head_compose"
        self._xr_anchor_path = "/World/H1_XRAnchor"
        self._xr_anchor_op = None
        self._xr_anchor_configured = False
        self._xr_anchor_forward_offset = 0.25           # m: anchor ahead of the base so the robot's head and
                                                        # shoulders stay out of the user's view
        self._xr_anchor_height_offset = 0.5             # m: lifts the whole VR rig; your real eye height adds
                                                        # on top, putting the view above the robot's head
        self._xr_anchor_yaw_offset_deg = -90.0          # aligns physical "room forward" with robot +X in
                                                        # head_compose mode; try 0/90/180 if you spawn rotated
        self._head_camera_last_base = None              # stashed by _get_head_camera_pose for the anchor
        self._head_camera_last_yaw = None
        self._first_person_head_target_distance = 1.8
        # Headset velocity and horizontal-motion tracking (gait gate)
        self._last_headset_raw_position = None       # Gf.Vec3d: position from pose reader
        self._last_headset_pose_matrix = None        # Gf.Matrix4d: full pose for orientation logging
        self._headset_prev_position = None           # position from previous step for velocity
        self._headset_velocity = Gf.Vec3d(0.0, 0.0, 0.0)
        self._headset_velocity_filter_time = 0.10    # low-pass time constant (s)
        self._headset_horiz_speed = 0.0              # floor-plane speed magnitude (m/s)
        self._headset_gait_min_horiz_speed = 0.025   # m/s threshold: suppress gait below this
        self._headset_gait_horiz_gate = 0.0          # 0-1 multiplier applied to gait output
        self._headset_gait_step_event = False        # True for one sample when step fires
        # Behavioral data collection for AI/RL training
        self._behavioral_data_enabled = True
        self._behavioral_data_records = []
        self._behavioral_data_step_counter = 0
        self._behavioral_data_log_every_n_steps = 2  # ~100 Hz (200 Hz / 2)
        self._behavioral_flush_every_n_steps = 500   # ~2.5 s at 200 Hz: short sessions keep their
                                                     # data even when the app closes without teardown
        self._behavioral_csv_fieldnames = {}         # filename -> header columns, fixed at first flush
        self._behavioral_data_output_dir = Path.home() / "BehavioralCollection"
        self._behavioral_sessions_root = self._behavioral_data_output_dir / "raw_sessions"
        self._behavioral_session_dir = None           # Path to the current session folder
        self._behavioral_session_id = None
        self._behavioral_dof_names = []              # populated on first sample
        # Session-relative logs added alongside behavior.csv
        self._hand_tracking_records = []
        self._gaze_records = []
        self._object_state_records = []
        self._object_state_prev_positions = {}        # object path -> Gf.Vec3d, for finite-diff velocity
        self._gaze_raycast_max_distance = 20.0        # m: range cap for the HMD-forward gaze raycast
        # Meta Quest Pro eye tracking (optional; gaze.csv falls back to HMD-forward without it)
        self._eye_gaze_enabled = True
        self._eye_gaze_ray_visual_enabled = True      # draw the red gaze ray + hit marker in the scene
        self._eye_gaze_tracker = None                 # EyeGazeTracker instance once XR is up
        # Eye-camera frame capture (~10 Hz PNG sequence)
        self._behavioral_frame_records = []
        self._behavioral_frame_dir = None
        self._behavioral_frame_camera = None          # None = not yet tried, False = failed, else Camera
        self._behavioral_frame_log_every_n_steps = 20  # ~10 Hz (200 Hz / 20)
        self._behavioral_frame_counter = 0            # persistent PNG index: the records buffer is
                                                      # cleared on every flush, so len() must not name files
        self._head_camera_update_counter = 0
        self._head_camera_update_interval = 1
        self._hand_tracking_arm_control_enabled = True
        self._controller_arm_control_enabled = True
        self._hand_tracking_status_logged = False
        self._h1_arm_dofs_configured = False
        self._h1_arm_dof_indices_by_side = {}
        self._h1_arm_joint_names_by_side = {}
        self._h1_arm_joint_defaults = {}
        self._h1_arm_joint_limits = {}
        self._hand_pose_candidates = ("palm", "wrist", "grip", "aim", "")
        self._controller_pose_candidates = ("grip", "aim", "")
        self._arm_smoothing = 0.34
        self._smoothed_arm_targets = {}
        self._arm_rig_smoothing = 0.38
        self._smoothed_arm_rig_targets = {}
        self._controller_arm_neutral_positions = {}
        self._grab_radius = 0.42
        self._grabbed_objects_by_side = {}
        self._grabbed_object_offsets = {}
        self._grabbed_object_was_kinematic = {}
        self._arm_rig_root_path = "/World/H1_ArmControlRig"
        self._arm_rig_target_paths = {
            "left": f"{self._arm_rig_root_path}/LeftHandTarget",
            "right": f"{self._arm_rig_root_path}/RightHandTarget",
        }
        self._arm_rig_target_ops = {}
        self._sample_box_root_path = "/World/H1_SampleBoxes"
        self._sample_box_count = 10
        self._sample_box_seed = 12
        self._sample_box_density = 5.0
        self._sample_box_min_mass = 0.45
        self._sample_box_max_mass = 4.5
        self._spawn_g1_hand_reference = False
        self._g1_reference_path = "/World/G1_HandReference"
        self._g1_reference_usd_path = (
            Path(__file__).resolve().parents[7] / "isaacsim.asset.transformer.rules" / "data" / "tests" / "G1" / "g1.usda"
        )
        self._h1_attached_hands_enabled = True
        self._h1_attached_hands_root_path = "/World/H1_AttachedHands"
        self._h1_hand_target_paths = {
            "left": f"{self._h1_attached_hands_root_path}/LeftHand",
            "right": f"{self._h1_attached_hands_root_path}/RightHand",
        }
        self._h1_hand_target_ops = {}
        self._h1_hand_attachment_paths = {}
        self._h1_hand_attachment_ops = {}
        self._h1_link_hands_created = False
        self._h1_terminal_arm_prim_paths = {}
        self._h1_terminal_arm_lookup_complete = False
        self._h1_hand_local_offsets = {
            "left": Gf.Vec3d(0.13, 0.035, 0.0),
            "right": Gf.Vec3d(0.13, -0.035, 0.0),
        }
        self._h1_wrist_bottom_hand_offset = Gf.Vec3d(0.24, 0.0, -0.035)
        self._active_h1_hand_target_matrices = {}
        self._preserve_existing_rig_calibration = True
        self._snap_h1_hands_to_wrist_connections = True
        self._manual_arm_rig_target_world_positions = {}
        self._arm_rig_world_offsets = {}
        self._prev_physics_sim_device: str | None = None
        self._prev_fabric_enabled: bool | None = None

        # Bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [self._max_forward_speed, 0.0, 0.0],
            "UP": [self._max_forward_speed, 0.0, 0.0],
            # yaw command (positive)
            "NUMPAD_4": [0.0, 0.0, self._max_yaw_speed],
            "LEFT": [0.0, 0.0, self._max_yaw_speed],
            # yaw command (negative)
            "NUMPAD_6": [0.0, 0.0, -self._max_yaw_speed],
            "RIGHT": [0.0, 0.0, -self._max_yaw_speed],
        }

    def _apply_ground_material(self, static_friction: float, dynamic_friction: float, restitution: float) -> None:
        """Apply physics material to the ground plane.

        Args:
            static_friction: Static friction coefficient.
            dynamic_friction: Dynamic friction coefficient.
            restitution: Restitution coefficient.
        """
        stage = omni.usd.get_context().get_stage()
        material_path = "/World/ground/Looks/PhysicsMaterial"

        material = UsdShade.Material.Define(stage, material_path)
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr().Set(static_friction)
        physics_material.CreateDynamicFrictionAttr().Set(dynamic_friction)
        physics_material.CreateRestitutionAttr().Set(restitution)

        ground_geom_path = "/World/ground/GroundPlane/CollisionPlane"
        ground_geom = stage.GetPrimAtPath(ground_geom_path)
        if ground_geom.IsValid():
            binding_api = UsdShade.MaterialBindingAPI.Apply(ground_geom)
            binding_api.Bind(material)

    def _create_sample_boxes(self) -> None:
        """Create physical boxes in front of H1 for controller/arm interaction testing."""
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, self._sample_box_root_path)
        rng = random.Random(self._sample_box_seed)

        for index in range(self._sample_box_count):
            size = rng.uniform(0.45, 0.90)
            x = rng.uniform(5.0, 12.0)
            y = rng.uniform(-3.0, 3.0)
            box_path = f"{self._sample_box_root_path}/Box_{index:02d}"
            cube = UsdGeom.Cube.Define(stage, box_path)
            cube.CreateSizeAttr(size)
            cube.CreateDisplayColorAttr().Set(
                [Gf.Vec3f(rng.uniform(0.25, 0.95), rng.uniform(0.25, 0.95), rng.uniform(0.25, 0.95))]
            )
            cube.ClearXformOpOrder()
            cube.AddTranslateOp().Set(Gf.Vec3d(x, y, size * 0.5 + 0.02))

            prim = cube.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            package_mass = self._clamp_value(
                self._sample_box_density * size * size * size,
                self._sample_box_min_mass,
                self._sample_box_max_mass,
            )
            mass_api.CreateMassAttr().Set(package_mass)
            prim.CreateAttribute("h1:packageSize", Sdf.ValueTypeNames.Float).Set(size)
            prim.CreateAttribute("h1:packageMass", Sdf.ValueTypeNames.Float).Set(package_mass)

        carb.log_info(
            f"HumanoidExample: created {self._sample_box_count} sample boxes under {self._sample_box_root_path}"
        )

    def _create_arm_control_rig(self) -> None:
        """Create visible controller target markers and hand meshes used as the H1 arm-control rig."""
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, self._arm_rig_root_path)
        colors = {"left": Gf.Vec3f(0.1, 0.55, 1.0), "right": Gf.Vec3f(1.0, 0.25, 0.15)}

        self._arm_rig_target_ops = {}
        for side, path in self._arm_rig_target_paths.items():
            target_exists = stage.GetPrimAtPath(path).IsValid()
            if self._preserve_existing_rig_calibration and target_exists:
                try:
                    existing_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
                    self._manual_arm_rig_target_world_positions[side] = existing_matrix.ExtractTranslation()
                    carb.log_info(
                        f"HumanoidExample: captured existing {side} hand rig calibration at "
                        f"{self._manual_arm_rig_target_world_positions[side]}"
                    )
                except Exception:
                    pass

            target = UsdGeom.Xform.Define(stage, path)
            target.ClearXformOpOrder()
            self._arm_rig_target_ops[side] = target.AddTransformOp()

            marker = UsdGeom.Sphere.Define(stage, f"{path}/TargetMarker")
            marker.CreateRadiusAttr(0.08)
            marker.CreateDisplayColorAttr().Set([colors[side]])
            UsdGeom.Imageable(marker.GetPrim()).MakeInvisible()

        carb.log_info(f"HumanoidExample: created H1 arm-control rig under {self._arm_rig_root_path}")

    def _create_h1_rig_hand_mesh(self, parent_path: str, side: str, color: Gf.Vec3f) -> None:
        """Create a simple hand mesh as a child of the arm rig target."""
        stage = omni.usd.get_context().get_stage()
        side_sign = 1.0 if side == "left" else -1.0
        hand_path = f"{parent_path}/HandMesh"
        if stage.GetPrimAtPath(hand_path).IsValid():
            carb.log_info(f"HumanoidExample: preserving existing calibrated hand mesh at {hand_path}")
            return

        UsdGeom.Xform.Define(stage, hand_path)

        palm = UsdGeom.Cube.Define(stage, f"{hand_path}/Palm")
        palm.CreateSizeAttr(1.0)
        palm.CreateDisplayColorAttr().Set([color])
        palm.ClearXformOpOrder()
        palm.AddTranslateOp().Set(Gf.Vec3d(0.035, 0.0, 0.0))
        palm.AddScaleOp().Set(Gf.Vec3f(0.14, 0.065, 0.035))

        finger_offsets = (-0.042, -0.014, 0.014, 0.042)
        for index, y_offset in enumerate(finger_offsets):
            finger = UsdGeom.Cube.Define(stage, f"{hand_path}/Finger_{index}")
            finger.CreateSizeAttr(1.0)
            finger.CreateDisplayColorAttr().Set([color])
            finger.ClearXformOpOrder()
            finger.AddTranslateOp().Set(Gf.Vec3d(0.15, y_offset, 0.018))
            finger.AddScaleOp().Set(Gf.Vec3f(0.09, 0.012, 0.014))

        thumb = UsdGeom.Cube.Define(stage, f"{hand_path}/Thumb")
        thumb.CreateSizeAttr(1.0)
        thumb.CreateDisplayColorAttr().Set([color])
        thumb.ClearXformOpOrder()
        thumb.AddTranslateOp().Set(Gf.Vec3d(0.06, 0.073 * side_sign, -0.005))
        thumb.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, -35.0 * side_sign))
        thumb.AddScaleOp().Set(Gf.Vec3f(0.075, 0.016, 0.014))

    def _ensure_scene_lighting(self) -> None:
        """Add explicit lights so referenced robot assets are visible in flat/new stages."""
        stage = omni.usd.get_context().get_stage()

        dome_path = "/World/H1_VR_DomeLight"
        if not stage.GetPrimAtPath(dome_path).IsValid():
            dome = UsdLux.DomeLight.Define(stage, dome_path)
            dome.CreateIntensityAttr().Set(1200.0)
            dome.CreateExposureAttr().Set(0.0)

        distant_path = "/World/H1_VR_DistantLight"
        if not stage.GetPrimAtPath(distant_path).IsValid():
            distant = UsdLux.DistantLight.Define(stage, distant_path)
            distant.CreateIntensityAttr().Set(2500.0)
            distant.CreateAngleAttr().Set(0.5)
            distant_xform = UsdGeom.Xformable(distant.GetPrim())
            distant_xform.ClearXformOpOrder()
            distant_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))

    def _create_g1_hand_reference(self) -> None:
        """Reference the local G1 humanoid-with-hands asset next to the H1 policy robot."""
        if not self._spawn_g1_hand_reference:
            return

        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(self._g1_reference_path).IsValid():
            return

        if not self._g1_reference_usd_path.exists():
            carb.log_warn(f"HumanoidExample: G1 hand asset not found: {self._g1_reference_usd_path}")
            return

        stage_utils.add_reference_to_stage(
            usd_path=str(self._g1_reference_usd_path).replace("\\", "/"),
            path=self._g1_reference_path,
        )

        g1_prim = stage.GetPrimAtPath(self._g1_reference_path)
        variant_choices = {
            "Physics": "None",
            "Sensor": "None",
            "left_hand": "Inspire",
            "right_hand": "Inspire",
            "Thor": "None",
        }
        for set_name, selection in variant_choices.items():
            variant_set = g1_prim.GetVariantSets().GetVariantSet(set_name)
            if variant_set and selection in variant_set.GetVariantNames():
                variant_set.SetVariantSelection(selection)

        g1_xform = UsdGeom.Xformable(g1_prim)
        g1_xform.ClearXformOpOrder()
        g1_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -2.75, 0.0))
        g1_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 90.0))

        carb.log_info(
            f"HumanoidExample: referenced G1 Inspire-hand humanoid at {self._g1_reference_path} from {self._g1_reference_usd_path}"
        )

    def _apply_hand_part_physics(self, prim) -> None:
        """Give H1 hand parts simple collision so they can touch scene objects."""
        UsdPhysics.CollisionAPI.Apply(prim)
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
        try:
            rigid_body.CreateKinematicEnabledAttr().Set(True)
        except Exception:
            pass

    def _create_h1_attached_hands(self) -> None:
        """Create simple hand-shaped attachments under H1's own arm end-link hierarchy."""
        if not self._h1_attached_hands_enabled:
            return

        stage = omni.usd.get_context().get_stage()
        self._find_h1_terminal_arm_prim_paths()
        if not self._h1_terminal_arm_prim_paths:
            return

        side_colors = {"left": Gf.Vec3f(0.15, 0.45, 1.0), "right": Gf.Vec3f(1.0, 0.35, 0.18)}
        xform_cache = UsdGeom.XformCache()
        created_sides = []
        for side, terminal_path in self._h1_terminal_arm_prim_paths.items():
            terminal_prim = stage.GetPrimAtPath(terminal_path)
            if not terminal_prim.IsValid():
                continue

            attachment_path = f"{terminal_path}/H1_{side.capitalize()}HandAttachment"
            attachment_exists = stage.GetPrimAtPath(attachment_path).IsValid()
            attachment = UsdGeom.Xform.Define(stage, attachment_path)
            if self._snap_h1_hands_to_wrist_connections:
                local_matrix = self._get_h1_wrist_connection_hand_local_matrix(side)
                carb.log_info(
                    f"HumanoidExample: snapped {side} hand attachment to wrist connection at {attachment_path}"
                )
            elif attachment_exists:
                terminal_world = xform_cache.GetLocalToWorldTransform(terminal_prim)
                attachment_world = xform_cache.GetLocalToWorldTransform(attachment.GetPrim())
                local_matrix = attachment_world * terminal_world.GetInverse()
                carb.log_info(
                    f"HumanoidExample: preserving existing H1-local {side} hand attachment transform at {attachment_path}"
                )
            else:
                local_matrix = self._get_saved_h1_hand_local_matrix(side, terminal_prim, xform_cache)

            attachment.ClearXformOpOrder()
            attachment_op = attachment.AddTransformOp()
            self._h1_hand_attachment_paths[side] = attachment_path
            self._h1_hand_attachment_ops[side] = attachment_op

            attachment_op.Set(local_matrix)
            self._create_h1_rig_hand_mesh(attachment_path, side, side_colors[side])
            self._remove_legacy_hand_duplicates(side)
            created_sides.append(side)

        self._h1_link_hands_created = bool(created_sides)
        carb.log_info(
            f"HumanoidExample: created H1 link-parented hand attachments for {created_sides}: "
            f"{self._h1_hand_attachment_paths}"
        )

    def _get_saved_h1_hand_local_matrix(self, side: str, terminal_prim, xform_cache: UsdGeom.XformCache) -> Gf.Matrix4d:
        """Convert an existing manually placed hand/rig transform into the H1 terminal link frame."""
        stage = omni.usd.get_context().get_stage()
        terminal_world = xform_cache.GetLocalToWorldTransform(terminal_prim)

        calibration_paths = (
            f"{self._arm_rig_target_paths[side]}/HandMesh",
            self._h1_hand_target_paths[side],
        )
        for path in calibration_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            try:
                hand_world = xform_cache.GetLocalToWorldTransform(prim)
                local_matrix = hand_world * terminal_world.GetInverse()
                carb.log_info(
                    f"HumanoidExample: saved {side} hand calibration from {path} into H1 link local transform"
                )
                return local_matrix
            except Exception:
                continue

        if side in self._manual_arm_rig_target_world_positions:
            hand_world = Gf.Matrix4d().SetTranslate(self._manual_arm_rig_target_world_positions[side])
            local_matrix = hand_world * terminal_world.GetInverse()
            carb.log_info(
                f"HumanoidExample: saved {side} hand calibration from current rig target into H1 link local transform"
            )
            return local_matrix

        side_sign = 1.0 if side == "left" else -1.0
        return Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.12, 0.02 * side_sign, 0.0))

    def _get_h1_wrist_connection_hand_local_matrix(self, side: str) -> Gf.Matrix4d:
        """Place the hand base below the H1 wrist/terminal arm link origin."""
        return Gf.Matrix4d().SetTranslate(self._h1_wrist_bottom_hand_offset)

    def _remove_legacy_hand_duplicates(self, side: str) -> None:
        """Remove old world/rig-parented hand meshes after their pose has been migrated under H1."""
        stage = omni.usd.get_context().get_stage()
        legacy_paths = (
            f"{self._arm_rig_target_paths[side]}/HandMesh",
            self._h1_hand_target_paths[side],
        )
        for path in legacy_paths:
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(path)
                carb.log_info(f"HumanoidExample: removed duplicate legacy hand prim {path}")

        root_prim = stage.GetPrimAtPath(self._h1_attached_hands_root_path)
        if root_prim.IsValid() and not any(root_prim.GetChildren()):
            stage.RemovePrim(self._h1_attached_hands_root_path)

    def _find_h1_terminal_arm_prim_paths(self) -> None:
        """Find the best terminal arm link prims for attaching visual hands."""
        if self._h1_terminal_arm_lookup_complete:
            return

        stage = omni.usd.get_context().get_stage()
        h1_root = stage.GetPrimAtPath("/World/H1")
        if not h1_root.IsValid():
            return
        self._h1_terminal_arm_lookup_complete = True

        for side in ("left", "right"):
            best_score = -999
            best_path = None
            for prim in Usd.PrimRange(h1_root):
                name = prim.GetName().lower()
                path = str(prim.GetPath())
                path_lower = path.lower()
                if side not in path_lower and side not in name:
                    continue
                if "h1_" in name or "handattachment" in path_lower or "handmesh" in path_lower:
                    continue
                score = 0
                if "wrist_yaw_link" in name or "wrist_yaw_link" in path_lower:
                    score += 80
                elif "wrist" in name or "wrist" in path_lower:
                    score += 70
                if "hand" in name or "hand" in path_lower:
                    score += 25
                if "forearm" in name or "lower_arm" in path_lower:
                    score += 20
                if "elbow" in name or "elbow" in path_lower:
                    score += 12
                if "link" in name:
                    score += 8
                if "collision" in path_lower or "collisions" in path_lower:
                    score -= 20
                if "visual" in path_lower or "mesh" in path_lower:
                    score -= 8
                if score > best_score:
                    best_score = score
                    best_path = path

            if best_path is not None and best_score > 0:
                self._h1_terminal_arm_prim_paths[side] = best_path

        carb.log_info(f"HumanoidExample H1 hand attachment links: {self._h1_terminal_arm_prim_paths}")

    def _update_h1_attached_hands(self) -> None:
        """Ensure H1 link-parented hands are created after the robot asset is available."""
        if self._h1_attached_hands_enabled and not self._h1_link_hands_created:
            self._create_h1_attached_hands()

    def setup_scene(self):
        """Set up the scene with robot and environment."""
        # Snapshot prior physics device/fabric state so cleanup can restore it.
        self._prev_physics_sim_device, self._prev_fabric_enabled = snapshot_physics_simulation_state()

        # Set device and backend BEFORE creating robot so it uses GPU
        SimulationManager.set_backend(self._world_settings["backend"])
        SimulationManager.set_physics_sim_device(self._world_settings["device"])
        SimulationManager.get_available_physics_engines(verbose=True)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        stage_utils.add_reference_to_stage(
            usd_path=assets_root_path + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )

        self._ensure_scene_lighting()

        # Apply physics material to ground to match training configuration
        self._apply_ground_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        self._create_sample_boxes()
        self._create_g1_hand_reference()

        # Create H1 robot (auto-detects active physics engine for policy selection)
        self.h1 = H1FlatTerrainPolicy(
            prim_path="/World/H1",
            position=[0, 0, 1.05],
        )
        self._create_arm_control_rig()
        self._create_h1_attached_hands()
        self._create_head_camera()
        self._create_xr_anchor()

    async def setup_post_load(self):
        """Setup keyboard input and physics callback after initial load."""
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._gamepad = self._appwindow.get_gamepad(0)
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        try:
            from omni.kit.xr.core import XRCore

            self._xr_core = XRCore.get_singleton()
        except Exception as e:
            self._xr_core = None
            carb.log_warn(f"HumanoidExample: XRCore unavailable for VR controller input: {e}")
        # Quest Pro eye-gaze tracker (optional). Imported lazily so a failure here can
        # never break example registration; gaze.csv falls back to HMD-forward gaze.
        self._eye_gaze_tracker = None
        if self._eye_gaze_enabled and self._xr_core is not None:
            try:
                from .eye_gaze_tracker import EyeGazeTracker

                self._eye_gaze_tracker = EyeGazeTracker(
                    self._xr_core, draw_ray=self._eye_gaze_ray_visual_enabled
                )
            except Exception as e:
                carb.log_warn(f"HumanoidExample: Quest Pro eye-gaze tracker unavailable: {e}")
        self._xr_input_status_logged = False
        self._hand_tracking_status_logged = False
        self._h1_arm_dofs_configured = False
        self._h1_head_prim_path = None
        self._h1_head_prim_lookup_complete = False
        self._h1_terminal_arm_prim_paths = {}
        self._h1_terminal_arm_lookup_complete = False
        self._h1_hand_attachment_paths = {}
        self._h1_hand_attachment_ops = {}
        self._h1_link_hands_created = False
        self._grabbed_objects_by_side = {}
        self._grabbed_object_offsets = {}
        self._controller_arm_neutral_positions = {}
        self._smoothed_arm_rig_targets = {}
        self._active_h1_hand_target_matrices = {}
        self._reset_headset_gait_state()
        # Flush any session left open by an unclean teardown (e.g. a reload that
        # skipped the scene clear) so its buffered rows are saved, not discarded.
        self._save_behavioral_data()
        self._behavioral_data_records = []
        self._behavioral_data_step_counter = 0
        self._behavioral_dof_names = []
        self._last_headset_raw_position = None
        self._last_headset_pose_matrix = None
        self._hand_tracking_records = []
        self._gaze_records = []
        self._object_state_records = []
        self._object_state_prev_positions = {}
        self._behavioral_frame_records = []
        self._behavioral_frame_camera = None
        self._physics_step_error_logged = set()
        self._xr_anchor_configured = False  # re-apply anchor settings on every load
        self._head_camera_last_base = None
        self._head_camera_last_yaw = None
        self._start_behavioral_session()

        torch = import_module("torch")
        self._base_command = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        self._keyboard_command = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        self._controller_command = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        self._physics_ready = False
        self._set_active_head_camera()

        # Register physics callback using SimulationManager
        if self._physics_callback_id is None:
            self._physics_callback_id = SimulationManager.register_callback(
                self.on_physics_step, IsaacEvents.POST_PHYSICS_STEP
            )

    async def setup_pre_reset(self):
        """Called before world reset."""
        # Reset physics ready flag before reset
        self._physics_ready = False

    async def setup_post_reset(self):
        """Called after world reset."""
        # Reset physics ready flag after reset so robot reinitializes on next play
        self._physics_ready = False

    async def setup_post_clear(self):
        """Called after clearing the scene."""
        # Deregister physics callback
        if self._physics_callback_id is not None:
            try:
                SimulationManager.deregister_callback(self._physics_callback_id)
            except Exception as e:
                carb.log_warn(f"Could not deregister callback {self._physics_callback_id}: {e}")
            self._physics_callback_id = None

        self._event_timer_callback = None
        self._unsubscribe_keyboard()
        self._save_behavioral_data()
        if self._eye_gaze_tracker is not None:
            self._eye_gaze_tracker.cleanup()
            self._eye_gaze_tracker = None
        self.h1 = None
        self._physics_ready = False
        self._head_camera_transform_op = None  # handles die with the stage; never reuse them
        self._xr_anchor_op = None
        self._restore_physics_simulation_state()

    def on_physics_step(self, dt: float, context: object) -> None:
        """Physics step callback - initialize on first step, then run policy.

        Args:
            dt: Delta time for the physics step.
            context: Physics step context.
        """
        if not self.h1:
            return

        # Check if physics tensors are valid, if not, reinitialize
        if not self.h1.robot.is_physics_tensor_entity_valid():
            self._physics_ready = False

        if self._physics_ready:
            # Robot is initialized, run the policy
            self._update_controller_command(dt)
            target_command = self._keyboard_command + self._controller_command
            target_command[0] = target_command[0].clamp(-self._max_forward_speed, self._max_forward_speed)
            target_command[2] = target_command[2].clamp(-self._max_yaw_speed, self._max_yaw_speed)
            self._smooth_base_command(target_command, dt)
            self.h1.forward(dt, self._base_command)
            # Stage edits (undo, prim deletion, clears) can invalidate prims any of
            # these subsystems hold handles to; isolate each one so a single failure
            # cannot abort the step and silently stop behavioral data collection.
            try:
                self._update_h1_arms_from_hand_tracking()
            except Exception as e:
                self._log_physics_step_error("arm teleoperation", e)
            try:
                self._update_h1_attached_hands()
            except Exception as e:
                self._log_physics_step_error("hand attachments", e)
            try:
                self._update_head_camera_view()
            except Exception as e:
                self._log_physics_step_error("head camera update", e)
            if self._eye_gaze_tracker is not None:
                try:
                    self._eye_gaze_tracker.update(dt)
                except Exception as e:
                    self._log_physics_step_error("eye gaze update", e)
            try:
                self._collect_all_behavioral_data()
            except Exception as e:
                self._log_physics_step_error("behavioral data collection", e)
        else:
            # First physics step after play - initialize the robot
            self._physics_ready = True
            self.h1.initialize()  # This already sets default state internally
            self.h1.post_reset()
            self._configure_h1_arm_dofs()
            self._update_h1_attached_hands()
            self._update_head_camera_view(force=True)

    def _log_physics_step_error(self, subsystem: str, error: Exception) -> None:
        """Warn once per distinct physics-step subsystem failure instead of spamming at 200 Hz."""
        key = (subsystem, type(error).__name__, str(error))
        if key in self._physics_step_error_logged:
            return
        self._physics_step_error_logged.add(key)
        carb.log_warn(
            f"HumanoidExample: {subsystem} failed with {type(error).__name__}: {error} "
            "(suppressing repeats; physics step continues)"
        )

    def _create_head_camera(self) -> None:
        """Create a USD camera used for robot-head first-person viewing."""
        stage = omni.usd.get_context().get_stage()
        camera = UsdGeom.Camera.Define(stage, self._head_camera_path)
        camera.CreateFocalLengthAttr().Set(16.0)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))

        xformable = UsdGeom.Xformable(camera.GetPrim())
        xformable.ClearXformOpOrder()
        self._head_camera_transform_op = xformable.AddTransformOp()
        carb.log_info(f"HumanoidExample: created H1 head camera at {self._head_camera_path}")

    def _create_xr_anchor(self) -> None:
        """Create the Xform prim the VR rig anchors to (XR custom-anchor mode).

        The rig origin corresponds to the physical floor of the playspace, so the
        anchor sits at ground level under the robot; the user's real standing
        height then puts their eyes near the robot's eye level, with full natural
        head tracking on top.
        """
        stage = omni.usd.get_context().get_stage()
        anchor = UsdGeom.Xform.Define(stage, self._xr_anchor_path)
        anchor.ClearXformOpOrder()
        self._xr_anchor_op = anchor.AddTransformOp()
        carb.log_info(f"HumanoidExample: created XR rig anchor at {self._xr_anchor_path}")

    def _configure_xr_custom_anchor(self) -> None:
        """Switch the VR profile to custom-anchor mode, pointed at the H1 anchor prim.

        Both the live and the persistent settings variants are written so the
        viewport XR controller picks the change up regardless of which one it
        watches in this Kit version.
        """
        if self._xr_anchor_configured:
            return
        self._xr_anchor_configured = True
        try:
            import carb.settings

            profile = "vr"
            try:
                name = str(self._xr_core.get_current_profile().get_name())
                if name:
                    profile = name
            except Exception:
                pass

            settings = carb.settings.get_settings()
            # The XR settings wrapper resolves "profile/persistent/anchorMode" to
            # "/xr/profile/<name>/persistent/anchorMode" — note the "persistent/"
            # SEGMENT inside the path (it is part of the setting name, not just a
            # settings hive). Every plausible variant is written; extras are inert.
            anchor_mode_paths = (
                f"/xr/profile/{profile}/persistent/anchorMode",
                f"/persistent/xr/profile/{profile}/persistent/anchorMode",
                f"/xr/profile/{profile}/anchorMode",
                f"/persistent/xr/profile/{profile}/anchorMode",
            )
            custom_anchor_paths = (
                f"/xr/profile/{profile}/stage/customAnchor",
                f"/persistent/xr/profile/{profile}/stage/customAnchor",
            )
            for path in anchor_mode_paths:
                settings.set(path, "custom anchor")
            for path in custom_anchor_paths:
                settings.set(path, self._xr_anchor_path)
            carb.log_info(
                f"HumanoidExample: XR profile '{profile}' anchor mode set to 'custom anchor' "
                f"-> {self._xr_anchor_path}"
            )
        except Exception as e:
            carb.log_warn(f"HumanoidExample: could not configure XR custom anchor: {e}")

    def _update_xr_anchor(self, base: Gf.Vec3d, yaw: float) -> None:
        """Move the XR anchor with the robot: ground position under the base + robot yaw.

        Assumes flat ground at z=0 (this scene). The user rides the anchor like a
        platform: walking/turning the robot carries them, while their own head
        motion stays fully tracked by the runtime.
        """
        if self._xr_anchor_op is None or not self._xr_anchor_op.GetAttr().IsValid():
            self._xr_anchor_op = None
            self._create_xr_anchor()
            if self._xr_anchor_op is None:
                return
        yaw_deg = math.degrees(yaw) + self._xr_anchor_yaw_offset_deg
        forward = Gf.Vec3d(math.cos(yaw), math.sin(yaw), 0.0)
        position = (
            Gf.Vec3d(float(base[0]), float(base[1]), self._xr_anchor_height_offset)
            + forward * self._xr_anchor_forward_offset
        )
        rot_m = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw_deg))
        trans_m = Gf.Matrix4d(1.0).SetTranslate(position)
        try:
            self._xr_anchor_op.Set(rot_m * trans_m)
        except Exception:
            self._xr_anchor_op = None  # stage changed under us; recreate next update

    def _schedule_composed_xr_camera(self) -> None:
        """Follow the robot AND keep natural head tracking with one schedule_set_camera call.

        schedule_set_camera(M) makes the final rendered view equal M by internally
        subtracting the user's current physical head pose. Passing

            M = physical_head_pose · (Y-up→Z-up axis fix) · robot_anchor

        means that internal subtraction cancels the head term we injected — the
        effective rig origin becomes the robot anchor, and the runtime keeps
        compositing the LIVE head pose on top. The user rides the robot with full
        natural head tracking. The math is absolute (recomputed from the current
        poses every step, never integrated), so timing mismatches between our
        read and the runtime's latch produce at most a transient, never drift.
        """
        if self._head_camera_last_base is None or self._head_camera_last_yaw is None:
            return
        headset = self._get_xr_input_device("/user/head")
        if headset is None:
            return

        head_pose = None
        for reader_name in ("get_pose", "get_raw_pose"):
            reader = getattr(headset, reader_name, None)
            if reader is None:
                continue
            try:
                head_pose = Gf.Matrix4d(reader())
                break
            except Exception:
                continue
        if head_pose is None:
            return

        base = self._head_camera_last_base
        yaw = self._head_camera_last_yaw
        yaw_deg = math.degrees(yaw) + self._xr_anchor_yaw_offset_deg
        forward = Gf.Vec3d(math.cos(yaw), math.sin(yaw), 0.0)
        position = (
            Gf.Vec3d(float(base[0]), float(base[1]), self._xr_anchor_height_offset)
            + forward * self._xr_anchor_forward_offset
        )

        anchor_m = Gf.Matrix4d(1.0).SetRotate(
            Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw_deg)
        ) * Gf.Matrix4d(1.0).SetTranslate(position)
        # Physical (OpenXR) space is Y-up; the stage is Z-up. Rotate +90° about X
        # so the head pose composes correctly under the Z-up anchor.
        yup_to_zup = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 90.0))
        try:
            self._xr_core.schedule_set_camera(head_pose * yup_to_zup * anchor_m)
        except Exception:
            pass

    def _set_active_head_camera(self) -> None:
        """Switch the active viewport to the H1 head camera when the viewport API is present."""
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = Sdf.Path(self._head_camera_path)
                carb.log_info(f"HumanoidExample: active viewport camera set to {self._head_camera_path}")
        except Exception as e:
            carb.log_warn(f"HumanoidExample: could not set active viewport camera: {e}")

    def _find_h1_head_prim_path(self) -> str | None:
        """Find a likely H1 head prim when the asset exposes one."""
        self._h1_head_prim_lookup_complete = True
        stage = omni.usd.get_context().get_stage()
        h1_root = stage.GetPrimAtPath("/World/H1")
        if not h1_root.IsValid():
            return None

        candidates = []
        for prim in Usd.PrimRange(h1_root):
            name = prim.GetName().lower()
            path = str(prim.GetPath())
            path_lower = path.lower()
            if "head" not in name and "head" not in path_lower:
                continue
            score = 0
            if name in ("head", "head_link"):
                score += 10
            if "collision" in path_lower:
                score -= 2
            if "visual" in path_lower:
                score -= 1
            candidates.append((score, path))

        if not candidates:
            carb.log_info("HumanoidExample: no explicit H1 head prim found; using base-offset head camera")
            return None
        candidates.sort(reverse=True)
        head_path = candidates[0][1]
        carb.log_info(f"HumanoidExample: using H1 head prim for camera: {head_path}")
        return head_path

    def _first_pose_value(self, values):
        """Convert a Warp/Torch pose array to the first [x, y, z] or [w, x, y, z] row."""
        try:
            return values.numpy()[0]
        except Exception:
            pass
        try:
            warp = import_module("warp")

            return warp.to_torch(values).detach().cpu().numpy()[0]
        except Exception:
            pass
        try:
            return values.detach().cpu().numpy()[0]
        except Exception:
            return None

    def _get_head_camera_pose(self):
        """Compute a first-person camera pose from the H1 head/eye position."""
        if not self.h1 or not self.h1.robot.is_physics_tensor_entity_valid():
            return None

        positions, orientations = self.h1.robot.get_world_poses()
        position = self._first_pose_value(positions)
        orientation = self._first_pose_value(orientations)
        if position is None or orientation is None or len(position) < 3 or len(orientation) < 4:
            return None

        qw, qx, qy, qz = (float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3]))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        yaw *= self._head_camera_yaw_sign
        forward = Gf.Vec3d(math.cos(yaw), math.sin(yaw), 0.0)
        base = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        # Stash for the XR anchor update, which needs the raw base/yaw rather
        # than the finished camera matrix.
        self._head_camera_last_base = base
        self._head_camera_last_yaw = yaw

        if not self._h1_head_prim_lookup_complete:
            self._h1_head_prim_path = self._find_h1_head_prim_path()

        # Horizontal anchor: the head prim when available (so the view tracks
        # torso lean); HEIGHT is always base + eye offset. Head-link origins sit
        # at the top of the skull and vary between assets, so deriving height
        # from them kept parking the camera above the robot's head.
        anchor = None
        if self._h1_head_prim_path is not None:
            stage = omni.usd.get_context().get_stage()
            head_prim = stage.GetPrimAtPath(self._h1_head_prim_path)
            if head_prim.IsValid():
                try:
                    anchor = UsdGeom.XformCache().GetLocalToWorldTransform(head_prim).ExtractTranslation()
                except Exception:
                    anchor = None
        if anchor is None:
            anchor = base

        eye = Gf.Vec3d(
            float(anchor[0]),
            float(anchor[1]),
            base[2] + self._first_person_eye_height_above_base + self._first_person_head_up_offset,
        )
        eye += forward * self._first_person_head_forward_offset
        target = eye + forward * self._first_person_head_target_distance
        return Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()

    def _update_head_camera_view(self, force: bool = False) -> None:
        """Move the viewport/XR camera to the H1 head pose."""
        if self._head_camera_transform_op is None:
            return
        self._head_camera_update_counter += 1
        if not force and self._head_camera_update_counter % self._head_camera_update_interval != 0:
            return

        # Undo or a stage clear can delete the camera prim while physics keeps
        # stepping; Set() on the stale handle then raises "Accessed schema on
        # invalid prim" at 200 Hz. GetAttr().IsValid() also catches the
        # delete-then-recreate case, where a prim exists at the path again but
        # the cached handle is still dead — so rebuild the camera either way.
        if not self._head_camera_transform_op.GetAttr().IsValid():
            carb.log_warn(
                f"HumanoidExample: head camera prim {self._head_camera_path} vanished; recreating it"
            )
            self._head_camera_transform_op = None
            self._create_head_camera()
            if self._head_camera_transform_op is None:
                return

        camera_pose = self._get_head_camera_pose()
        if camera_pose is None:
            return
        self._head_camera_transform_op.Set(camera_pose)
        if self._xr_core is not None:
            if self._xr_camera_mode == "head_compose":
                self._schedule_composed_xr_camera()
            elif self._xr_camera_mode == "custom_anchor":
                self._configure_xr_custom_anchor()
                if self._head_camera_last_base is not None and self._head_camera_last_yaw is not None:
                    self._update_xr_anchor(self._head_camera_last_base, self._head_camera_last_yaw)
            else:  # "camera_lock" — follows the robot but cancels user head rotation
                try:
                    self._xr_core.schedule_set_camera(camera_pose)
                except Exception:
                    pass

    def _get_gamepad_value(self, gamepad_input: carb.input.GamepadInput) -> float:
        """Return a gamepad/VR-controller input value if a controller is available."""
        if self._input is None or self._gamepad is None:
            return 0.0
        try:
            return float(self._input.get_gamepad_value(self._gamepad, gamepad_input))
        except Exception:
            return 0.0

    def _apply_deadzone(self, value: float) -> float:
        """Suppress small analog stick or trigger noise."""
        return value if abs(value) >= self._controller_deadzone else 0.0

    def _clamp_value(self, value: float, lower: float, upper: float) -> float:
        """Clamp a scalar value."""
        return max(lower, min(upper, value))

    def _smooth_base_command(self, target_command, dt: float) -> None:
        """Ramp locomotion commands so the H1 policy does not get abrupt step inputs."""
        if self._base_command is None:
            return
        if self._command_response_time <= 0.0:
            self._base_command[:] = target_command
            return
        alpha = self._clamp_value(float(dt) / self._command_response_time, 0.0, 1.0)
        self._base_command[:] = self._base_command + (target_command - self._base_command) * alpha

    def _get_xr_input_device(self, handle: str):
        """Return an OpenXR input device such as /user/hand/left, if XR is active."""
        if self._xr_core is None:
            return None
        try:
            return self._xr_core.get_input_device(handle)
        except Exception:
            return None

    def _get_xr_gesture_value(self, input_device, input_name: str, gesture_name: str) -> float:
        """Read an XR controller gesture value, returning 0.0 when unsupported/inactive."""
        if input_device is None:
            return 0.0
        try:
            if not input_device.has_input(input_name):
                return 0.0
            if not input_device.has_input_gesture(input_name, gesture_name):
                return 0.0
            return float(input_device.get_input_gesture_value(input_name, gesture_name))
        except Exception:
            return 0.0

    def _reset_headset_gait_state(self) -> None:
        """Reset the headset-height gait detector."""
        self._headset_gait_status_logged = False
        self._headset_gait_time = 0.0
        self._headset_gait_height_baseline = None
        self._headset_gait_filtered_height = None
        self._headset_gait_velocity_sign = 0
        self._headset_gait_last_peak_height = None
        self._headset_gait_last_peak_time = None
        self._headset_gait_last_trough_height = None
        self._headset_gait_last_trough_time = None
        self._headset_gait_last_extremum_time = -999.0
        self._headset_gait_last_step_time = -999.0
        self._headset_gait_pulse_time_remaining = 0.0
        self._headset_gait_output = 0.0
        self._headset_prev_position = None
        self._headset_velocity = Gf.Vec3d(0.0, 0.0, 0.0)
        self._headset_horiz_speed = 0.0
        self._headset_gait_horiz_gate = 0.0
        self._headset_gait_step_event = False

    def _get_xr_up_vector(self) -> Gf.Vec3d:
        """Return the XR coordinate system up vector, falling back to the USD Z axis."""
        if self._xr_core is not None:
            try:
                up_vector = Gf.Vec3d(self._xr_core.get_coordinate_system().get_up_vector())
                length = up_vector.GetLength()
                if length > 0.0:
                    return up_vector / length
            except Exception:
                pass
        return Gf.Vec3d(0.0, 0.0, 1.0)

    def _get_robot_head_world_height(self, up_vector: Gf.Vec3d) -> float | None:
        """Return H1 head position projected onto up_vector for virtual-world-pose gait correction."""
        if not self.h1 or not self.h1.robot.is_physics_tensor_entity_valid():
            return None
        positions, _ = self.h1.robot.get_world_poses()
        position = self._first_pose_value(positions)
        if position is None or len(position) < 3:
            return None
        head_pos = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        head_pos += Gf.Vec3d(0.0, 0.0, self._first_person_eye_height_above_base)
        return float(Gf.Dot(head_pos, up_vector))

    def _get_headset_tracking_height(self) -> float | None:
        """Return the physical HMD height used for step-in-place gait detection."""
        headset = self._get_xr_input_device("/user/head")
        if headset is None:
            return None

        up_vector = self._get_xr_up_vector()
        try:
            pose_names = {str(name) for name in headset.get_pose_names()}
        except Exception:
            pose_names = set()

        # Prefer physical poses so robot head camera motion is not fed back into the gait detector.
        for pose_reader_name in ("get_pose", "get_raw_pose"):
            pose_reader = getattr(headset, pose_reader_name, None)
            if pose_reader is None:
                continue
            for pose_name in self._headset_gait_pose_candidates:
                if pose_name and pose_names and pose_name not in pose_names:
                    continue
                try:
                    pose = pose_reader(pose_name) if pose_name else pose_reader()
                    position = pose.ExtractTranslation()
                    self._last_headset_raw_position = Gf.Vec3d(position)
                    self._last_headset_pose_matrix = Gf.Matrix4d(pose)
                    if not self._headset_gait_status_logged:
                        self._headset_gait_status_logged = True
                        carb.log_info(
                            f"HumanoidExample: headset gait using /user/head {pose_reader_name} pose"
                        )
                    return float(Gf.Dot(Gf.Vec3d(position), up_vector))
                except Exception:
                    continue

        # Fallback: virtual-world pose with robot-head height subtracted to break the feedback loop.
        # schedule_set_camera shifts the VR origin to the robot head each frame, so the virtual-world
        # reading contains the robot's own head bob unless we correct for it here.
        pose_reader = getattr(headset, "get_virtual_world_pose", None)
        if pose_reader is not None:
            for pose_name in self._headset_gait_pose_candidates:
                if pose_name and pose_names and pose_name not in pose_names:
                    continue
                try:
                    pose = pose_reader(pose_name) if pose_name else pose_reader()
                    position = pose.ExtractTranslation()
                    self._last_headset_raw_position = Gf.Vec3d(position)
                    self._last_headset_pose_matrix = Gf.Matrix4d(pose)
                    raw_height = float(Gf.Dot(Gf.Vec3d(position), up_vector))
                    robot_head_height = self._get_robot_head_world_height(up_vector)
                    corrected_height = raw_height - robot_head_height if robot_head_height is not None else raw_height
                    if not self._headset_gait_status_logged:
                        self._headset_gait_status_logged = True
                        carb.log_info(
                            "HumanoidExample: headset gait using /user/head get_virtual_world_pose "
                            "(robot-head correction applied)"
                        )
                    return corrected_height
                except Exception:
                    continue

        if not self._headset_gait_status_logged:
            self._headset_gait_status_logged = True
            carb.log_warn(
                "HumanoidExample: headset gait could not read /user/head pose "
                "(tried get_pose, get_raw_pose, get_virtual_world_pose) – gait detection disabled. "
                "Check that XRCore is initialised and the headset is tracked."
            )
        return None

    def _update_headset_velocity(self, dt: float) -> None:
        """Compute filtered headset 3-D velocity and horizontal speed from consecutive positions."""
        pos = self._last_headset_raw_position
        if pos is None or dt <= 0.0:
            return
        if self._headset_prev_position is None:
            self._headset_prev_position = Gf.Vec3d(pos)
            return
        raw_vel = (pos - self._headset_prev_position) * (1.0 / dt)
        self._headset_prev_position = Gf.Vec3d(pos)
        alpha = self._clamp_value(dt / self._headset_velocity_filter_time, 0.0, 1.0)
        self._headset_velocity += (raw_vel - self._headset_velocity) * alpha
        # Horizontal speed = velocity minus the up-axis component
        up = self._get_xr_up_vector()
        vel_up_component = Gf.Dot(self._headset_velocity, up)
        vel_horiz = self._headset_velocity - up * vel_up_component
        self._headset_horiz_speed = math.sqrt(
            vel_horiz[0] ** 2 + vel_horiz[1] ** 2 + vel_horiz[2] ** 2
        )
        # Gate: 0 when stationary, reaches 1 at 2× the minimum-speed threshold
        gate_denom = self._headset_gait_min_horiz_speed * 2.0
        self._headset_gait_horiz_gate = self._clamp_value(
            self._headset_horiz_speed / gate_denom if gate_denom > 0.0 else 1.0,
            0.0, 1.0,
        )

    def _start_behavioral_session(self) -> None:
        """Open a new recording session: create its folder tree and write metadata.json.

        Layout produced under ~/BehavioralCollection/raw_sessions/:

            session_YYYY-MM-DD_HH-MM-SS/
            ├── metadata.json            (written here, at session start)
            ├── frames/eye_camera/       (PNG frames appended during the run)
            ├── behavior.csv             ┐
            ├── frame_timestamps.csv     │ written by _save_behavioral_data()
            ├── hand_tracking.csv        │ when the session ends
            ├── gaze.csv                 │
            └── object_states.csv        ┘
        """
        if not self._behavioral_data_enabled:
            return

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        session_id = f"session_{timestamp}"
        session_dir = self._behavioral_sessions_root / session_id
        # Uniquify so two loads within the same second never merge into one folder.
        suffix = 2
        while session_dir.exists():
            session_id = f"session_{timestamp}_{suffix}"
            session_dir = self._behavioral_sessions_root / session_id
            suffix += 1
        frame_dir = session_dir / "frames" / "eye_camera"
        try:
            frame_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            carb.log_warn(f"HumanoidExample: could not create behavioral session folder: {e}")
            self._behavioral_session_dir = None
            self._behavioral_frame_dir = None
            return

        self._behavioral_session_dir = session_dir
        self._behavioral_frame_dir = frame_dir
        self._behavioral_session_id = session_id
        self._behavioral_csv_fieldnames = {}
        self._behavioral_frame_counter = 0

        physics_dt = self._world_settings.get("physics_dt", 1.0 / 200.0)
        try:
            from isaacsim.core.version import get_version

            isaac_sim_version = get_version()[0] or None
        except Exception:
            isaac_sim_version = None

        metadata = {
            "session_id": session_id,
            "start_unix_time": time.time(),
            "isaac_sim_version": isaac_sim_version,
            "physics_dt": physics_dt,
            "rendering_dt": self._world_settings.get("rendering_dt"),
            "robot_name": "H1",
            "headset_gait_enabled": self._headset_gait_enabled,
            "eye_gaze_enabled": self._eye_gaze_enabled,
            "behavioral_data_log_rate_hz": (1.0 / physics_dt) / self._behavioral_data_log_every_n_steps,
            "frame_log_rate_hz": (1.0 / physics_dt) / self._behavioral_frame_log_every_n_steps,
            "camera_names": ["eye_camera"],
            "notes": "",
        }
        try:
            with open(session_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            carb.log_warn(f"HumanoidExample: could not write session metadata.json: {e}")

        carb.log_info(f"HumanoidExample: started behavioral session {session_id} at {session_dir}")

    def _collect_behavioral_sample(self) -> None:
        """Record one rich behavior.csv row (HMD pose/velocity, gait state, robot pose,
        commands, and every joint's position/velocity), ~100 Hz.

        Rate gating and dispatch happen in _collect_all_behavioral_data().
        """
        unix_now = time.time()
        record = {
            "unix_time":          round(unix_now, 6),
            "sim_time":           round(self._headset_gait_time, 6),
            "step_index":         self._behavioral_data_step_counter,
            # --- headset position ---
            "hmd_pos_x": None, "hmd_pos_y": None, "hmd_pos_z": None,
            # --- headset orientation (quaternion from pose matrix) ---
            "hmd_qw": None, "hmd_qi": None, "hmd_qj": None, "hmd_qk": None,
            "hmd_yaw": None,
            # --- headset velocity ---
            "hmd_vel_x":          round(float(self._headset_velocity[0]), 6),
            "hmd_vel_y":          round(float(self._headset_velocity[1]), 6),
            "hmd_vel_z":          round(float(self._headset_velocity[2]), 6),
            "hmd_horiz_speed":    round(self._headset_horiz_speed, 6),
            # --- gait signal state ---
            "gait_filtered_h":    round(self._headset_gait_filtered_height or 0.0, 6),
            "gait_vel_sign":      int(self._headset_gait_velocity_sign),
            "gait_horiz_gate":    round(self._headset_gait_horiz_gate, 6),
            "gait_output":        round(self._headset_gait_output, 6),
            "gait_pulse_rem":     round(self._headset_gait_pulse_time_remaining, 6),
            "step_event":         int(self._headset_gait_step_event),
            # --- robot base pose ---
            "robot_pos_x": None, "robot_pos_y": None, "robot_pos_z": None,
            "robot_qw": None, "robot_qi": None, "robot_qj": None, "robot_qk": None,
            "robot_yaw": None,
            # --- locomotion commands ---
            "cmd_forward":        round(float(self._base_command[0]), 6) if self._base_command is not None else 0.0,
            "cmd_lateral":        round(float(self._base_command[1]), 6) if self._base_command is not None else 0.0,
            "cmd_yaw":            round(float(self._base_command[2]), 6) if self._base_command is not None else 0.0,
        }

        # Headset full pose
        pos = self._last_headset_raw_position
        if pos is not None:
            record["hmd_pos_x"] = round(float(pos[0]), 6)
            record["hmd_pos_y"] = round(float(pos[1]), 6)
            record["hmd_pos_z"] = round(float(pos[2]), 6)
        mat = self._last_headset_pose_matrix
        if mat is not None:
            try:
                q = mat.ExtractRotationQuat()
                imag = q.GetImaginary()
                record["hmd_qw"] = round(float(q.GetReal()), 6)
                record["hmd_qi"] = round(float(imag[0]), 6)
                record["hmd_qj"] = round(float(imag[1]), 6)
                record["hmd_qk"] = round(float(imag[2]), 6)
                qw, qi, qj, qk = float(q.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])
                record["hmd_yaw"] = round(math.atan2(2.0 * (qw * qk + qi * qj), 1.0 - 2.0 * (qj * qj + qk * qk)), 6)
            except Exception:
                pass

        # Robot base pose + joints
        if self.h1 and self.h1.robot.is_physics_tensor_entity_valid():
            positions, orientations = self.h1.robot.get_world_poses()
            rp = self._first_pose_value(positions)
            ro = self._first_pose_value(orientations)
            if rp is not None and len(rp) >= 3:
                record["robot_pos_x"] = round(float(rp[0]), 6)
                record["robot_pos_y"] = round(float(rp[1]), 6)
                record["robot_pos_z"] = round(float(rp[2]), 6)
            if ro is not None and len(ro) >= 4:
                qw, qi, qj, qk = float(ro[0]), float(ro[1]), float(ro[2]), float(ro[3])
                record["robot_qw"] = round(qw, 6)
                record["robot_qi"] = round(qi, 6)
                record["robot_qj"] = round(qj, 6)
                record["robot_qk"] = round(qk, 6)
                record["robot_yaw"] = round(math.atan2(2.0 * (qw * qk + qi * qj), 1.0 - 2.0 * (qj * qj + qk * qk)), 6)

            # Joint positions and velocities
            try:
                dof_names = list(getattr(self.h1.robot, "dof_names", []))
                if dof_names and not self._behavioral_dof_names:
                    self._behavioral_dof_names = dof_names
                joint_pos_raw = self.h1.robot.get_dof_positions()
                joint_vel_raw = self.h1.robot.get_dof_velocities()
                joint_pos = self._first_pose_value(joint_pos_raw)
                joint_vel = self._first_pose_value(joint_vel_raw)
                for i, name in enumerate(self._behavioral_dof_names):
                    safe = name.replace(" ", "_")
                    record[f"j_{safe}_pos"] = round(float(joint_pos[i]), 6) if joint_pos is not None and i < len(joint_pos) else None
                    record[f"j_{safe}_vel"] = round(float(joint_vel[i]), 6) if joint_vel is not None and i < len(joint_vel) else None
            except Exception:
                pass

        self._headset_gait_step_event = False   # consume the flag
        self._behavioral_data_records.append(record)

    def _append_csv(self, path: Path, records: list) -> None:
        """Append dict records to a session CSV, writing the header on first creation.

        The header is fixed at the first append (union of the buffered rows' keys in
        first-seen order) and reused for the rest of the session. This matters for
        behavior.csv: the j_* joint columns only appear on rows logged while the
        robot's physics tensors were valid. Rows missing a column write an empty
        cell (restval), and unexpected late-appearing keys are dropped rather than
        raising (extrasaction) — losing one exotic column beats losing the file.
        """
        if not records:
            return
        fieldnames = self._behavioral_csv_fieldnames.get(path.name)
        if fieldnames is None:
            seen: dict[str, None] = {}
            for record in records:
                for key in record:
                    seen.setdefault(key)
            fieldnames = list(seen)
            self._behavioral_csv_fieldnames[path.name] = fieldnames
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(records)

    def _flush_behavioral_csvs(self) -> None:
        """Append all buffered rows to the session CSVs and clear the buffers.

        Called every ~10 s during play and once at session close. Incremental
        appends mean a crash or force-quit loses at most the last few seconds.
        (Previously the CSVs were only written at scene clear — which never runs
        when the app window is simply closed, so whole sessions ended up with
        camera frames but no sensor logs at all.)
        """
        if self._behavioral_session_dir is None:
            return
        streams = (
            ("behavior.csv", self._behavioral_data_records),
            ("frame_timestamps.csv", self._behavioral_frame_records),
            ("hand_tracking.csv", self._hand_tracking_records),
            ("gaze.csv", self._gaze_records),
            ("object_states.csv", self._object_state_records),
        )
        for filename, records in streams:
            if not records:
                continue
            try:
                self._append_csv(self._behavioral_session_dir / filename, records)
            except Exception as e:
                carb.log_warn(f"HumanoidExample: failed to flush {filename}: {e}")
                continue
            # clear() keeps the same list object so every collector stays bound to it
            records.clear()

    def _save_behavioral_data(self) -> None:
        """Flush any remaining rows and close the current recording session.

        Runs when the scene is cleared or the example is torn down. Idempotent: the
        buffers and the session handle are reset afterwards, so a second call is a no-op.
        """
        if not self._behavioral_data_enabled or self._behavioral_session_dir is None:
            return

        session_dir = self._behavioral_session_dir
        self._flush_behavioral_csvs()
        carb.log_info(
            f"HumanoidExample: closed behavioral session {self._behavioral_session_id} at {session_dir}"
        )

        # Reset buffers and close the session.
        self._behavioral_data_records = []
        self._behavioral_frame_records = []
        self._hand_tracking_records = []
        self._gaze_records = []
        self._object_state_records = []
        self._object_state_prev_positions = {}
        self._behavioral_csv_fieldnames = {}
        self._behavioral_data_step_counter = 0
        self._behavioral_frame_counter = 0
        self._behavioral_session_dir = None
        self._behavioral_frame_dir = None
        self._behavioral_frame_camera = None

    def _parse_sample_box_index(self, prim_path) -> int | None:
        """Extract the numeric index from a sample-box prim path like '.../Box_07', else None."""
        if not prim_path:
            return None
        name = str(prim_path).rsplit("/", 1)[-1]
        if name.startswith("Box_"):
            try:
                return int(name.split("_")[-1])
            except ValueError:
                return None
        return None

    def _ensure_eye_camera_capture_initialized(self) -> None:
        """Lazily wrap the existing /World/H1_HeadCamera prim in a Camera sensor.

        Camera.initialize() attaches a render product + RGB annotator to the prim.
        Frame data refreshes at RENDER rate (rendering_dt, ~90 Hz here), not physics
        rate, so later get_rgba() calls are cheap: they return the most recently
        rendered frame, or None until the first frame is available. Initialization is
        deferred to the first capture attempt because the render pipeline is not ready
        during scene setup.
        """
        if self._behavioral_frame_camera is not None:
            return  # already initialized (Camera instance) or permanently failed (False)
        try:
            from isaacsim.sensors.camera import Camera

            camera = Camera(prim_path=self._head_camera_path, resolution=(256, 256))
            camera.initialize()
            self._behavioral_frame_camera = camera
            carb.log_info("HumanoidExample: initialized eye-camera frame capture")
        except Exception as e:
            carb.log_warn(f"HumanoidExample: could not initialize eye-camera capture, frames disabled: {e}")
            self._behavioral_frame_camera = False  # sentinel: tried once, don't retry every step

    def _capture_eye_camera_frame(self) -> None:
        """Save one PNG frame from the H1 eye camera (~10 Hz) and log it for frame_timestamps.csv.

        Deliberately slower than the 100 Hz sensor logs: PNG encoding is expensive, and
        the downstream sync tool aligns each frame to the nearest behavior.csv row by
        sim_time anyway, so a dense frame stream buys nothing.
        """
        if self._behavioral_session_dir is None or self._behavioral_frame_dir is None:
            return
        if self._behavioral_data_step_counter % self._behavioral_frame_log_every_n_steps != 0:
            return

        self._ensure_eye_camera_capture_initialized()
        if not self._behavioral_frame_camera:
            return

        try:
            rgba = self._behavioral_frame_camera.get_rgba()
        except Exception:
            rgba = None
        if rgba is None:
            # No rendered frame yet (rendering runs slower than physics); skip quietly.
            return

        frame_id = self._behavioral_frame_counter
        image_name = f"frame_{frame_id:06d}.png"
        try:
            from PIL import Image

            Image.fromarray(rgba[:, :, :3]).save(self._behavioral_frame_dir / image_name)
        except Exception as e:
            carb.log_warn(f"HumanoidExample: failed to save eye-camera frame: {e}")
            return

        self._behavioral_frame_records.append(
            {
                "frame_id": frame_id,
                "unix_time": round(time.time(), 6),
                "sim_time": round(self._headset_gait_time, 6),
                "camera_name": "eye_camera",
                "image_path": f"frames/eye_camera/{image_name}",
            }
        )
        self._behavioral_frame_counter += 1

    def _get_raw_hand_or_controller_pose(self, input_device):
        """Return an absolute hand-tracking or controller pose for logging (ignores grip gating)."""
        if input_device is None:
            return None
        hand_pose = self._get_hand_tracking_pose(input_device)
        if hand_pose is not None:
            return hand_pose
        try:
            pose_names = {str(name) for name in input_device.get_pose_names()}
        except Exception:
            pose_names = set()
        for pose_name in self._controller_pose_candidates:
            if pose_name and pose_names and pose_name not in pose_names:
                continue
            try:
                return input_device.get_virtual_world_pose(pose_name)
            except Exception:
                continue
        return None

    def _collect_hand_tracking_sample(self) -> None:
        """Record left/right hand (or controller) pose plus grip/trigger/button values, ~100 Hz.

        Unlike the arm-teleop path, poses here are logged unconditionally — not gated on
        grip being held — so the dataset always contains raw hand motion whenever
        tracking is available. valid=0 rows mean the device was absent that sample.
        """
        left_xr = self._get_xr_input_device("/user/hand/left")
        right_xr = self._get_xr_input_device("/user/hand/right")

        def _pose_fields(device):
            pose = self._get_raw_hand_or_controller_pose(device)
            if pose is None:
                return {"valid": 0, "pos_x": None, "pos_y": None, "pos_z": None, "qw": None, "qx": None, "qy": None, "qz": None}
            position = pose.ExtractTranslation()
            quat = pose.ExtractRotationQuat()
            imag = quat.GetImaginary()
            return {
                "valid": 1,
                "pos_x": round(float(position[0]), 6),
                "pos_y": round(float(position[1]), 6),
                "pos_z": round(float(position[2]), 6),
                "qw": round(float(quat.GetReal()), 6),
                "qx": round(float(imag[0]), 6),
                "qy": round(float(imag[1]), 6),
                "qz": round(float(imag[2]), 6),
            }

        left_fields = _pose_fields(left_xr)
        right_fields = _pose_fields(right_xr)

        left_grip = max(
            self._get_xr_gesture_value(left_xr, "squeeze", "value"),
            self._get_xr_gesture_value(left_xr, "squeeze", "click"),
            self._get_xr_gesture_value(left_xr, "grip", "value"),
        )
        right_grip = max(
            self._get_xr_gesture_value(right_xr, "squeeze", "value"),
            self._get_xr_gesture_value(right_xr, "squeeze", "click"),
            self._get_xr_gesture_value(right_xr, "grip", "value"),
        )
        left_trigger = max(
            self._get_xr_gesture_value(left_xr, "trigger", "value"),
            self._get_xr_gesture_value(left_xr, "trigger", "click"),
        )
        right_trigger = max(
            self._get_xr_gesture_value(right_xr, "trigger", "value"),
            self._get_xr_gesture_value(right_xr, "trigger", "click"),
        )
        left_button_x = max(
            self._get_xr_gesture_value(left_xr, "x", "value"),
            self._get_xr_gesture_value(left_xr, "x", "click"),
        )
        right_button_a = max(
            self._get_xr_gesture_value(right_xr, "a", "value"),
            self._get_xr_gesture_value(right_xr, "a", "click"),
        )

        self._hand_tracking_records.append(
            {
                "unix_time": round(time.time(), 6),
                "sim_time": round(self._headset_gait_time, 6),
                "step_index": self._behavioral_data_step_counter,
                "left_hand_valid": left_fields["valid"],
                "left_hand_pos_x": left_fields["pos_x"],
                "left_hand_pos_y": left_fields["pos_y"],
                "left_hand_pos_z": left_fields["pos_z"],
                "left_hand_qw": left_fields["qw"],
                "left_hand_qx": left_fields["qx"],
                "left_hand_qy": left_fields["qy"],
                "left_hand_qz": left_fields["qz"],
                "right_hand_valid": right_fields["valid"],
                "right_hand_pos_x": right_fields["pos_x"],
                "right_hand_pos_y": right_fields["pos_y"],
                "right_hand_pos_z": right_fields["pos_z"],
                "right_hand_qw": right_fields["qw"],
                "right_hand_qx": right_fields["qx"],
                "right_hand_qy": right_fields["qy"],
                "right_hand_qz": right_fields["qz"],
                "left_grip": round(left_grip, 6),
                "right_grip": round(right_grip, 6),
                "left_trigger": round(left_trigger, 6),
                "right_trigger": round(right_trigger, 6),
                "left_button_x": round(left_button_x, 6),
                "right_button_a": round(right_button_a, 6),
            }
        )

    def _collect_gaze_sample(self) -> None:
        """Record one gaze ray row (~100 Hz), preferring real eye tracking when present.

        Two sources, distinguished by the gaze_source column:
        - "eye_tracker": Quest Pro OpenXR eye gaze via EyeGazeTracker, including its
          raycast hit (already robot-body filtered).
        - "hmd_forward": fallback — HMD position + facing (-Z) direction as a weak
          intent proxy, with a local PhysX raycast capped at _gaze_raycast_max_distance.
        Sample-box hits also get their numeric object_id in both cases.
        """
        record = {
            "unix_time": round(time.time(), 6),
            "sim_time": round(self._headset_gait_time, 6),
            "step_index": self._behavioral_data_step_counter,
            "gaze_source": None,
            "gaze_valid": 0,
            "gaze_origin_x": None,
            "gaze_origin_y": None,
            "gaze_origin_z": None,
            "gaze_dir_x": None,
            "gaze_dir_y": None,
            "gaze_dir_z": None,
            "gaze_hit_valid": 0,
            "gaze_hit_x": None,
            "gaze_hit_y": None,
            "gaze_hit_z": None,
            "gaze_hit_distance": None,
            "gaze_hit_object_path": None,
            "gaze_hit_object_id": None,
        }

        # Preferred source: real Quest Pro eye tracking (raycast already done there).
        gaze = self._eye_gaze_tracker.latest if self._eye_gaze_tracker is not None else None
        if gaze is not None and gaze.valid and gaze.origin is not None and gaze.direction is not None:
            record["gaze_source"] = "eye_tracker"
            record["gaze_valid"] = 1
            record["gaze_origin_x"] = round(float(gaze.origin[0]), 6)
            record["gaze_origin_y"] = round(float(gaze.origin[1]), 6)
            record["gaze_origin_z"] = round(float(gaze.origin[2]), 6)
            record["gaze_dir_x"] = round(float(gaze.direction[0]), 6)
            record["gaze_dir_y"] = round(float(gaze.direction[1]), 6)
            record["gaze_dir_z"] = round(float(gaze.direction[2]), 6)
            if gaze.hit_valid and gaze.hit_position is not None:
                record["gaze_hit_valid"] = 1
                record["gaze_hit_x"] = round(float(gaze.hit_position[0]), 6)
                record["gaze_hit_y"] = round(float(gaze.hit_position[1]), 6)
                record["gaze_hit_z"] = round(float(gaze.hit_position[2]), 6)
                if gaze.hit_distance is not None:
                    record["gaze_hit_distance"] = round(float(gaze.hit_distance), 6)
                record["gaze_hit_object_path"] = gaze.hit_object_path
                record["gaze_hit_object_id"] = self._parse_sample_box_index(gaze.hit_object_path)
            self._gaze_records.append(record)
            return

        # Fallback source: HMD position + facing direction.
        origin = self._last_headset_raw_position
        mat = self._last_headset_pose_matrix
        if origin is not None and mat is not None:
            try:
                forward = mat.ExtractRotation().TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
                length = forward.GetLength()
            except Exception:
                length = 0.0
            if length > 0.0:
                forward = forward / length
                record["gaze_source"] = "hmd_forward"
                record["gaze_valid"] = 1
                record["gaze_origin_x"] = round(float(origin[0]), 6)
                record["gaze_origin_y"] = round(float(origin[1]), 6)
                record["gaze_origin_z"] = round(float(origin[2]), 6)
                record["gaze_dir_x"] = round(float(forward[0]), 6)
                record["gaze_dir_y"] = round(float(forward[1]), 6)
                record["gaze_dir_z"] = round(float(forward[2]), 6)

                try:
                    # Imported lazily: physics extension load order is not guaranteed at module import time.
                    import omni.physics.core

                    query = omni.physics.core.get_physics_scene_query_interface()
                    origin_t = (float(origin[0]), float(origin[1]), float(origin[2]))
                    forward_t = (float(forward[0]), float(forward[1]), float(forward[2]))
                    try:
                        # both_sides is required on this API; omitting it raised
                        # TypeError and silently blanked every gaze_hit_* column.
                        ret, hit = query.raycast_closest(
                            origin_t, forward_t, self._gaze_raycast_max_distance, False
                        )
                    except TypeError:
                        ret, hit = query.raycast_closest(origin_t, forward_t, self._gaze_raycast_max_distance)
                    if ret:
                        record["gaze_hit_valid"] = 1
                        hit_pos = getattr(hit, "position", None)
                        if hit_pos is not None:
                            record["gaze_hit_x"] = round(float(hit_pos[0]), 6)
                            record["gaze_hit_y"] = round(float(hit_pos[1]), 6)
                            record["gaze_hit_z"] = round(float(hit_pos[2]), 6)
                        record["gaze_hit_distance"] = round(float(hit.distance), 6)

                        # PhysX reports the hit body either as an Sdf path string or as an
                        # encoded int that PhysicsSchemaTools can decode back into a path.
                        # Imported lazily: pxr.PhysicsSchemaTools is registered by the PhysX
                        # schema extension, which this extension does not depend on at load
                        # time — a module-level import would break extension startup and
                        # remove the Humanoid example from the menu.
                        rigid_body = getattr(hit, "rigid_body", None)
                        hit_path = None
                        if isinstance(rigid_body, str):
                            hit_path = rigid_body
                        elif rigid_body is not None:
                            try:
                                from pxr import PhysicsSchemaTools

                                hit_path = str(PhysicsSchemaTools.intToSdfPath(rigid_body))
                            except Exception:
                                hit_path = None
                        record["gaze_hit_object_path"] = hit_path
                        record["gaze_hit_object_id"] = self._parse_sample_box_index(hit_path)
                except Exception:
                    pass

        self._gaze_records.append(record)

    def _collect_object_state_sample(self) -> None:
        """Record one row per sample box: world pose, velocity, and grab state, ~100 Hz.

        Velocity is a finite difference between consecutive logged positions (the boxes
        are plain USD rigid bodies, not tensor-API entities, so there is no cheap
        velocity getter). The first row for each box therefore reports zero velocity.
        """
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(self._sample_box_root_path)
        if not root.IsValid():
            return

        unix_now = round(time.time(), 6)
        sim_now = round(self._headset_gait_time, 6)
        dt = self._world_settings.get("physics_dt", 1.0 / 200.0) * self._behavioral_data_log_every_n_steps

        grabbed_by_path = {path: side for side, path in self._grabbed_objects_by_side.items()}
        xform_cache = UsdGeom.XformCache()
        for prim in root.GetChildren():
            path = str(prim.GetPath())
            try:
                world_matrix = xform_cache.GetLocalToWorldTransform(prim)
                position = world_matrix.ExtractTranslation()
                quat = world_matrix.ExtractRotationQuat()
            except Exception:
                continue
            imag = quat.GetImaginary()

            prev_position = self._object_state_prev_positions.get(path)
            if prev_position is not None and dt > 0.0:
                velocity = (position - prev_position) * (1.0 / dt)
            else:
                velocity = Gf.Vec3d(0.0, 0.0, 0.0)
            self._object_state_prev_positions[path] = Gf.Vec3d(position)

            self._object_state_records.append(
                {
                    "unix_time": unix_now,
                    "sim_time": sim_now,
                    "step_index": self._behavioral_data_step_counter,
                    "object_id": self._parse_sample_box_index(path),
                    "object_path": path,
                    "pos_x": round(float(position[0]), 6),
                    "pos_y": round(float(position[1]), 6),
                    "pos_z": round(float(position[2]), 6),
                    "qw": round(float(quat.GetReal()), 6),
                    "qx": round(float(imag[0]), 6),
                    "qy": round(float(imag[1]), 6),
                    "qz": round(float(imag[2]), 6),
                    "vel_x": round(float(velocity[0]), 6),
                    "vel_y": round(float(velocity[1]), 6),
                    "vel_z": round(float(velocity[2]), 6),
                    "is_grabbed": int(path in grabbed_by_path),
                    "grabbed_by": grabbed_by_path.get(path),
                }
            )

    def _collect_all_behavioral_data(self) -> None:
        """Advance the shared logging counter and dispatch every per-modality collector.

        Called once per physics step (200 Hz). One counter drives all modalities so
        their rows stay aligned by step_index:
        - eye-camera frames: every 20 steps (~10 Hz) — PNG encode is comparatively slow
        - behavior / hand / gaze / object rows: every 2 steps (~100 Hz)
        - CSV flush to disk: every 2000 steps (~10 s), so a crash loses seconds, not the session
        """
        if not self._behavioral_data_enabled:
            return
        self._behavioral_data_step_counter += 1
        self._capture_eye_camera_frame()
        if self._behavioral_data_step_counter % self._behavioral_flush_every_n_steps == 0:
            self._flush_behavioral_csvs()
        if self._behavioral_data_step_counter % self._behavioral_data_log_every_n_steps != 0:
            return
        self._collect_behavioral_sample()
        self._collect_hand_tracking_sample()
        self._collect_gaze_sample()
        self._collect_object_state_sample()

    def _smooth_headset_gait_output(self, target: float, dt: float) -> float:
        """Smooth the normalized headset-gait forward command."""
        target = self._clamp_value(target, 0.0, 1.0)
        if dt <= 0.0:
            self._headset_gait_output = target
            return self._headset_gait_output
        smoothing_time = self._headset_gait_attack_time if target > self._headset_gait_output else self._headset_gait_release_time
        alpha = 1.0 if smoothing_time <= 0.0 else self._clamp_value(dt / smoothing_time, 0.0, 1.0)
        self._headset_gait_output += (target - self._headset_gait_output) * alpha
        self._headset_gait_output = self._clamp_value(self._headset_gait_output, 0.0, 1.0)
        return self._headset_gait_output

    def _register_headset_gait_extremum(self, extremum_type: str, height: float, time_now: float) -> None:
        """Accept a local peak/trough and trigger one constant step pulse when it looks human-paced."""
        if time_now - self._headset_gait_last_extremum_time < 0.10:
            return
        self._headset_gait_last_extremum_time = time_now

        amplitude = 0.0
        opposite_gap = None
        if extremum_type == "peak":
            if self._headset_gait_last_trough_height is not None and self._headset_gait_last_trough_time is not None:
                amplitude = height - self._headset_gait_last_trough_height
                opposite_gap = time_now - self._headset_gait_last_trough_time
            self._headset_gait_last_peak_height = height
            self._headset_gait_last_peak_time = time_now
        else:
            if self._headset_gait_last_peak_height is not None and self._headset_gait_last_peak_time is not None:
                amplitude = self._headset_gait_last_peak_height - height
                opposite_gap = time_now - self._headset_gait_last_peak_time
            self._headset_gait_last_trough_height = height
            self._headset_gait_last_trough_time = time_now

        if amplitude < self._headset_gait_min_amplitude:
            return
        if opposite_gap is None or opposite_gap > self._headset_gait_max_extremum_gap:
            return
        if time_now - self._headset_gait_last_step_time < self._headset_gait_min_step_interval:
            return

        self._headset_gait_last_step_time = time_now
        self._headset_gait_step_event = True
        self._headset_gait_pulse_time_remaining = max(
            self._headset_gait_pulse_time_remaining,
            self._headset_gait_pulse_duration,
        )

    def _update_headset_gait_command(self, dt: float) -> float:
        """Convert headset vertical peaks/troughs into a normalized forward walking command."""
        self._headset_gait_time += max(float(dt), 0.0)
        self._headset_gait_pulse_time_remaining = max(0.0, self._headset_gait_pulse_time_remaining - max(dt, 0.0))

        # Read the HMD pose and velocity even when gait walking is disabled:
        # sim_time and the hmd_* columns in behavior.csv, plus the HMD-forward
        # fallback in gaze.csv, come from these reads — not from the detector.
        height = self._get_headset_tracking_height()
        self._update_headset_velocity(dt)
        if not self._headset_gait_enabled or height is None:
            return self._smooth_headset_gait_output(0.0, dt)

        if self._headset_gait_height_baseline is None:
            self._headset_gait_height_baseline = height
            self._headset_gait_filtered_height = 0.0
            return self._smooth_headset_gait_output(0.0, dt)

        baseline_alpha = self._clamp_value(dt / self._headset_gait_baseline_time, 0.0, 1.0)
        self._headset_gait_height_baseline += (height - self._headset_gait_height_baseline) * baseline_alpha
        centered_height = height - self._headset_gait_height_baseline

        previous_filtered = self._headset_gait_filtered_height
        filter_alpha = self._clamp_value(dt / self._headset_gait_filter_time, 0.0, 1.0)
        filtered_height = previous_filtered + (centered_height - previous_filtered) * filter_alpha
        self._headset_gait_filtered_height = filtered_height

        velocity = 0.0 if dt <= 0.0 else (filtered_height - previous_filtered) / dt
        if velocity > self._headset_gait_velocity_deadzone:
            velocity_sign = 1
        elif velocity < -self._headset_gait_velocity_deadzone:
            velocity_sign = -1
        else:
            velocity_sign = self._headset_gait_velocity_sign

        if self._headset_gait_velocity_sign > 0 and velocity_sign < 0:
            self._register_headset_gait_extremum("peak", previous_filtered, self._headset_gait_time)
        elif self._headset_gait_velocity_sign < 0 and velocity_sign > 0:
            self._register_headset_gait_extremum("trough", previous_filtered, self._headset_gait_time)

        if velocity_sign != 0:
            self._headset_gait_velocity_sign = velocity_sign

        target = self._headset_gait_forward_intensity if self._headset_gait_pulse_time_remaining > 0.0 else 0.0
        # Gate: suppress gait when headset is stationary (nodding in place) vs. actually moving
        target *= self._headset_gait_horiz_gate
        return self._smooth_headset_gait_output(target, dt)

    def _log_xr_input_status_once(self, left_xr, right_xr) -> None:
        """Log whether XR controllers are visible to the example."""
        if self._xr_input_status_logged:
            return
        self._xr_input_status_logged = True
        carb.log_info(
            f"HumanoidExample XR input devices: left={left_xr is not None}, right={right_xr is not None}"
        )
        for label, device in (("left", left_xr), ("right", right_xr)):
            if device is None:
                continue
            try:
                input_names = [str(name) for name in device.get_input_names()]
                carb.log_info(f"HumanoidExample XR {label} controller inputs: {input_names}")
            except Exception:
                pass

    def _log_hand_tracking_status_once(self, left_xr, right_xr) -> None:
        """Log available XR hand-tracking data once for tuning/debugging."""
        if self._hand_tracking_status_logged:
            return
        self._hand_tracking_status_logged = True
        for label, device in (("left", left_xr), ("right", right_xr)):
            if device is None:
                carb.log_info(f"HumanoidExample XR {label} hand tracking: no device")
                continue
            try:
                source = str(device.get_hand_tracking_data_source())
            except Exception:
                source = ""
            try:
                pose_names = [str(name) for name in device.get_pose_names()]
            except Exception:
                pose_names = []
            carb.log_info(
                f"HumanoidExample XR {label} hand tracking source='{source}', pose_names={pose_names}"
            )

    def _configure_h1_arm_dofs(self) -> None:
        """Find H1 arm DOFs so hand tracking can override only the arms."""
        if self._h1_arm_dofs_configured or not self.h1:
            return

        dof_names = list(getattr(self.h1.robot, "dof_names", []))
        if not dof_names:
            return

        arm_name_map = {
            "left": {
                "shoulder_pitch": ("left_shoulder_pitch_joint", "left_shoulder_pitch"),
                "shoulder_roll": ("left_shoulder_roll_joint", "left_shoulder_roll"),
                "shoulder_yaw": ("left_shoulder_yaw_joint", "left_shoulder_yaw"),
                "elbow": ("left_elbow_joint", "left_elbow"),
            },
            "right": {
                "shoulder_pitch": ("right_shoulder_pitch_joint", "right_shoulder_pitch"),
                "shoulder_roll": ("right_shoulder_roll_joint", "right_shoulder_roll"),
                "shoulder_yaw": ("right_shoulder_yaw_joint", "right_shoulder_yaw"),
                "elbow": ("right_elbow_joint", "right_elbow"),
            },
        }

        for side, joints in arm_name_map.items():
            self._h1_arm_dof_indices_by_side[side] = {}
            self._h1_arm_joint_names_by_side[side] = {}
            for joint_key, candidates in joints.items():
                match = next((name for name in candidates if name in dof_names), None)
                if match is None:
                    match = next((name for name in dof_names if any(candidate in name for candidate in candidates)), None)
                if match is None:
                    continue
                self._h1_arm_dof_indices_by_side[side][joint_key] = dof_names.index(match)
                self._h1_arm_joint_names_by_side[side][joint_key] = match

        all_indices = sorted(
            {
                index
                for side_indices in self._h1_arm_dof_indices_by_side.values()
                for index in side_indices.values()
            }
        )
        self._h1_arm_joint_defaults = {}
        self._h1_arm_joint_limits = {}
        if all_indices:
            try:
                default_pos = self.h1.default_pos.detach().cpu().numpy()
            except Exception:
                default_pos = None
            try:
                lower_limits, upper_limits = self.h1.robot.get_dof_limits(dof_indices=all_indices)
                lower_limits = lower_limits.numpy()[0]
                upper_limits = upper_limits.numpy()[0]
            except Exception:
                lower_limits = None
                upper_limits = None
            for i, dof_index in enumerate(all_indices):
                self._h1_arm_joint_defaults[dof_index] = float(default_pos[dof_index]) if default_pos is not None else 0.0
                if lower_limits is not None and upper_limits is not None:
                    self._h1_arm_joint_limits[dof_index] = (float(lower_limits[i]), float(upper_limits[i]))

        self._h1_arm_dofs_configured = True
        carb.log_info(f"HumanoidExample H1 DOFs: {dof_names}")
        carb.log_info(f"HumanoidExample H1 hand-tracked arm DOFs: {self._h1_arm_joint_names_by_side}")

    def _get_hand_tracking_pose(self, input_device):
        """Return a tracked hand pose matrix from the best available hand pose name."""
        if input_device is None:
            return None
        try:
            if str(input_device.get_hand_tracking_data_source()) != "hand":
                return None
        except Exception:
            return None

        try:
            pose_names = {str(name) for name in input_device.get_pose_names()}
        except Exception:
            pose_names = set()

        for pose_name in self._hand_pose_candidates:
            if pose_name and pose_name not in pose_names:
                continue
            try:
                return input_device.get_virtual_world_pose(pose_name)
            except Exception:
                continue
        return None

    def _get_controller_arm_pose(self, side: str, input_device):
        """Return a relative controller pose for controller-driven arm teleoperation."""
        if input_device is None:
            return None
        if not self._is_controller_arm_pose_enabled(input_device):
            self._controller_arm_neutral_positions.pop(side, None)
            return None
        try:
            pose_names = {str(name) for name in input_device.get_pose_names()}
        except Exception:
            pose_names = set()

        for pose_name in self._controller_pose_candidates:
            if pose_name and pose_name not in pose_names:
                continue
            try:
                pose = input_device.get_virtual_world_pose(pose_name)
                position = pose.ExtractTranslation()
                neutral_position = self._controller_arm_neutral_positions.get(side)
                if neutral_position is None:
                    self._controller_arm_neutral_positions[side] = Gf.Vec3d(position)
                    neutral_position = position
                relative_position = Gf.Vec3d(position) - neutral_position
                relative_pose = Gf.Matrix4d(pose)
                relative_pose.SetTranslateOnly(relative_position)
                return relative_pose
            except Exception:
                continue
        return None

    def _is_controller_arm_pose_enabled(self, input_device) -> bool:
        """Use controller pose for arm teleop only while the grip/squeeze is held."""
        squeeze = self._get_xr_gesture_value(input_device, "squeeze", "value")
        squeeze_click = self._get_xr_gesture_value(input_device, "squeeze", "click")
        grip = self._get_xr_gesture_value(input_device, "grip", "value")
        return max(squeeze, squeeze_click, grip) >= self._arm_pose_enable_threshold

    def _get_h1_base_pose_for_arms(self):
        """Return H1 base position and yaw for body-frame arm mapping."""
        if not self.h1 or not self.h1.robot.is_physics_tensor_entity_valid():
            return None
        positions, orientations = self.h1.robot.get_world_poses()
        position = self._first_pose_value(positions)
        orientation = self._first_pose_value(orientations)
        if position is None or orientation is None or len(position) < 3 or len(orientation) < 4:
            return None

        qw, qx, qy, qz = (float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3]))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])), yaw

    def _stage_vector_to_h1_body(self, vector: Gf.Vec3d, yaw: float) -> Gf.Vec3d:
        """Rotate a world-space vector into the H1 base frame using base yaw."""
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return Gf.Vec3d(
            cos_yaw * vector[0] + sin_yaw * vector[1],
            -sin_yaw * vector[0] + cos_yaw * vector[1],
            vector[2],
        )

    def _h1_body_point_to_stage(self, point: Gf.Vec3d, base_position: Gf.Vec3d, yaw: float) -> Gf.Vec3d:
        """Transform a body-frame point into stage coordinates."""
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return base_position + Gf.Vec3d(
            cos_yaw * point[0] - sin_yaw * point[1],
            sin_yaw * point[0] + cos_yaw * point[1],
            point[2],
        )

    def _compute_arm_target_body_position(
        self, side: str, hand_pose: Gf.Matrix4d, base_position: Gf.Vec3d, yaw: float, pose_is_relative: bool = False
    ) -> Gf.Vec3d:
        """Map a controller/hand pose into the visible arm-rig target point."""
        side_sign = 1.0 if side == "left" else -1.0
        hand_position = hand_pose.ExtractTranslation()
        if pose_is_relative:
            controller_delta = self._stage_vector_to_h1_body(hand_position, yaw)
            hand_body = Gf.Vec3d(0.38, 0.38 * side_sign, 0.18)
            hand_body += Gf.Vec3d(
                self._clamp_value(controller_delta[0] * 1.6, -0.45, 0.70),
                self._clamp_value(controller_delta[1] * 1.5, -0.60, 0.60),
                self._clamp_value(controller_delta[2] * 1.8, -0.65, 0.65),
            )
        else:
            hand_body = self._stage_vector_to_h1_body(hand_position - base_position, yaw)
        return hand_body

    def _set_arm_rig_target_visible(self, side: str, visible: bool) -> None:
        """Show or hide one arm-control rig marker without hiding the hand mesh."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(f"{self._arm_rig_target_paths[side]}/TargetMarker")
        if not prim.IsValid():
            return
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _smooth_arm_rig_target(self, side: str, target_body: Gf.Vec3d) -> Gf.Vec3d:
        """Smooth the visible arm-rig target in H1 body coordinates."""
        previous = self._smoothed_arm_rig_targets.get(side)
        if previous is None:
            self._smoothed_arm_rig_targets[side] = Gf.Vec3d(target_body)
            return target_body
        smoothed = previous + (target_body - previous) * self._arm_rig_smoothing
        self._smoothed_arm_rig_targets[side] = Gf.Vec3d(smoothed)
        return smoothed

    def _update_arm_rig_target(self, side: str, target_body: Gf.Vec3d, base_position: Gf.Vec3d, yaw: float) -> None:
        """Move the visible rig target marker to the desired hand target."""
        target_op = self._arm_rig_target_ops.get(side)
        if target_op is None:
            return
        target_world = self._h1_body_point_to_stage(target_body, base_position, yaw)
        if side not in self._arm_rig_world_offsets and side in self._manual_arm_rig_target_world_positions:
            self._arm_rig_world_offsets[side] = self._manual_arm_rig_target_world_positions[side] - target_world
            carb.log_info(
                f"HumanoidExample: locked {side} hand rig calibration offset {self._arm_rig_world_offsets[side]}"
            )
        target_world += self._arm_rig_world_offsets.get(side, Gf.Vec3d(0.0, 0.0, 0.0))
        target_matrix = Gf.Matrix4d().SetTranslate(target_world)
        target_op.Set(target_matrix)
        self._active_h1_hand_target_matrices[side] = target_matrix
        self._set_arm_rig_target_visible(side, True)

    def _get_active_hand_world_position(self, side: str):
        """Return the controller hand target position used for grabbing."""
        active_matrix = self._active_h1_hand_target_matrices.get(side)
        if active_matrix is not None:
            return active_matrix.ExtractTranslation()

        stage = omni.usd.get_context().get_stage()
        attachment_path = self._h1_hand_attachment_paths.get(side)
        if attachment_path is None:
            return None
        attachment_prim = stage.GetPrimAtPath(attachment_path)
        if not attachment_prim.IsValid():
            return None
        try:
            return UsdGeom.XformCache().GetLocalToWorldTransform(attachment_prim).ExtractTranslation()
        except Exception:
            return None

    def _set_rigid_body_kinematic(self, prim, enabled: bool) -> None:
        """Toggle kinematic mode for an object being carried by the hand."""
        try:
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
            attr = rigid_body.GetKinematicEnabledAttr()
            if not attr:
                attr = rigid_body.CreateKinematicEnabledAttr()
            attr.Set(enabled)
        except Exception as e:
            carb.log_warn(f"HumanoidExample: could not set kinematic={enabled} on {prim.GetPath()}: {e}")

    def _set_prim_world_translation(self, prim, position: Gf.Vec3d) -> None:
        """Move a simple world-parented sample object to a world position."""
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(position)

    def _get_grabbable_object_size(self, prim) -> float:
        """Return the authored sample-box size when available."""
        try:
            attr = prim.GetAttribute("h1:packageSize")
            if attr and attr.HasAuthoredValueOpinion():
                return float(attr.Get())
        except Exception:
            pass
        return 0.5

    def _find_nearest_grabbable_object(self, hand_position: Gf.Vec3d):
        """Find the nearest sample box close enough to the hand target."""
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(self._sample_box_root_path)
        if not root.IsValid():
            return None, None

        taken_paths = {path for path in self._grabbed_objects_by_side.values() if path is not None}
        xform_cache = UsdGeom.XformCache()
        best_path = None
        best_position = None
        best_distance = math.inf
        for prim in root.GetChildren():
            path = str(prim.GetPath())
            if path in taken_paths:
                continue
            try:
                object_position = xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            except Exception:
                continue
            delta = object_position - hand_position
            distance = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])
            size = self._get_grabbable_object_size(prim)
            grab_radius = max(self._grab_radius, size * 0.65)
            if distance <= grab_radius and distance < best_distance:
                best_path = path
                best_position = object_position
                best_distance = distance

        return best_path, best_position

    def _set_visual_grip_pose(self, side: str, closed: bool) -> None:
        """Curl or open the simple visual fingers on the H1 hand attachment."""
        stage = omni.usd.get_context().get_stage()
        attachment_path = self._h1_hand_attachment_paths.get(side)
        if attachment_path is None:
            return
        hand_path = f"{attachment_path}/HandMesh"
        side_sign = 1.0 if side == "left" else -1.0

        for index, y_offset in enumerate((-0.042, -0.014, 0.014, 0.042)):
            finger_prim = stage.GetPrimAtPath(f"{hand_path}/Finger_{index}")
            if not finger_prim.IsValid():
                continue
            finger = UsdGeom.Cube(finger_prim)
            finger.ClearXformOpOrder()
            if closed:
                finger.AddTranslateOp().Set(Gf.Vec3d(0.075, y_offset, -0.018))
                finger.AddRotateXYZOp().Set(Gf.Vec3f(0.0, -62.0, 0.0))
            else:
                finger.AddTranslateOp().Set(Gf.Vec3d(0.15, y_offset, 0.018))
            finger.AddScaleOp().Set(Gf.Vec3f(0.09, 0.012, 0.014))

        thumb_prim = stage.GetPrimAtPath(f"{hand_path}/Thumb")
        if thumb_prim.IsValid():
            thumb = UsdGeom.Cube(thumb_prim)
            thumb.ClearXformOpOrder()
            if closed:
                thumb.AddTranslateOp().Set(Gf.Vec3d(0.055, 0.052 * side_sign, -0.018))
                thumb.AddRotateXYZOp().Set(Gf.Vec3f(0.0, -45.0, -55.0 * side_sign))
            else:
                thumb.AddTranslateOp().Set(Gf.Vec3d(0.06, 0.073 * side_sign, -0.005))
                thumb.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, -35.0 * side_sign))
            thumb.AddScaleOp().Set(Gf.Vec3f(0.075, 0.016, 0.014))

    def _release_grabbed_object(self, side: str) -> None:
        """Release an object currently held by one hand."""
        object_path = self._grabbed_objects_by_side.pop(side, None)
        self._grabbed_object_offsets.pop(side, None)
        if object_path is None:
            self._set_visual_grip_pose(side, False)
            return
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(object_path)
        if prim.IsValid():
            self._set_rigid_body_kinematic(prim, False)
            carb.log_info(f"HumanoidExample: released {object_path} from {side} hand")
        self._set_visual_grip_pose(side, False)

    def _update_grabbed_object(self, side: str, grip_active: bool) -> None:
        """Attach a nearby sample object to the hand while grip is held."""
        if not grip_active:
            self._release_grabbed_object(side)
            return

        hand_position = self._get_active_hand_world_position(side)
        if hand_position is None:
            return

        stage = omni.usd.get_context().get_stage()
        object_path = self._grabbed_objects_by_side.get(side)
        if object_path is None:
            object_path, object_position = self._find_nearest_grabbable_object(hand_position)
            if object_path is None or object_position is None:
                self._set_visual_grip_pose(side, False)
                return
            self._grabbed_objects_by_side[side] = object_path
            self._grabbed_object_offsets[side] = object_position - hand_position
            prim = stage.GetPrimAtPath(object_path)
            if prim.IsValid():
                self._set_rigid_body_kinematic(prim, True)
                carb.log_info(f"HumanoidExample: grabbed {object_path} with {side} hand")

        prim = stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            self._release_grabbed_object(side)
            return

        hold_offset = self._grabbed_object_offsets.get(side, Gf.Vec3d(0.0, 0.0, 0.0))
        self._set_prim_world_translation(prim, hand_position + hold_offset)
        self._set_visual_grip_pose(side, True)

    def _compute_arm_targets_from_body_position(self, side: str, hand_body: Gf.Vec3d):
        """Map an arm-control rig target point into rough H1 shoulder/elbow joint targets."""
        side_sign = 1.0 if side == "left" else -1.0
        shoulder_body = Gf.Vec3d(0.05, 0.22 * side_sign, 0.35)
        arm_vector = hand_body - shoulder_body

        forward = self._clamp_value(float(arm_vector[0]), -0.25, 0.70)
        outward = self._clamp_value(float(arm_vector[1]) * side_sign, -0.15, 0.65)
        up = self._clamp_value(float(arm_vector[2]), -0.55, 0.55)
        reach = math.sqrt(forward * forward + outward * outward + up * up)

        indices = self._h1_arm_dof_indices_by_side.get(side, {})
        targets = {}
        if "shoulder_pitch" in indices:
            dof_index = indices["shoulder_pitch"]
            targets[dof_index] = self._h1_arm_joint_defaults.get(dof_index, 0.0) + self._clamp_value(
                -1.35 * forward + 0.65 * up, -1.2, 1.2
            )
        if "shoulder_roll" in indices:
            dof_index = indices["shoulder_roll"]
            targets[dof_index] = self._h1_arm_joint_defaults.get(dof_index, 0.0) + side_sign * self._clamp_value(
                1.45 * outward + 0.20 * up, -0.9, 0.9
            )
        if "shoulder_yaw" in indices:
            dof_index = indices["shoulder_yaw"]
            targets[dof_index] = self._h1_arm_joint_defaults.get(dof_index, 0.0) + side_sign * self._clamp_value(
                0.85 * outward + 0.35 * forward, -0.8, 0.8
            )
        if "elbow" in indices:
            dof_index = indices["elbow"]
            bend = self._clamp_value((0.75 - reach) * 2.0, 0.0, 1.25)
            targets[dof_index] = self._h1_arm_joint_defaults.get(dof_index, 0.0) + bend
        return targets

    def _smooth_and_clamp_arm_targets(self, raw_targets: dict[int, float]) -> dict[int, float]:
        """Smooth hand-tracking joint targets and clamp to joint limits."""
        targets = {}
        for dof_index, raw_target in raw_targets.items():
            previous = self._smoothed_arm_targets.get(dof_index, raw_target)
            smoothed = previous + self._arm_smoothing * (raw_target - previous)
            lower, upper = self._h1_arm_joint_limits.get(dof_index, (-math.inf, math.inf))
            smoothed = self._clamp_value(smoothed, lower, upper)
            self._smoothed_arm_targets[dof_index] = smoothed
            targets[dof_index] = smoothed
        return targets

    def _update_h1_arms_from_hand_tracking(self) -> None:
        """Override H1 arm DOF targets from Meta/OpenXR hand-tracking poses."""
        if not self._hand_tracking_arm_control_enabled or self._xr_core is None or not self.h1:
            return

        self._configure_h1_arm_dofs()
        if not self._h1_arm_dofs_configured:
            return

        left_xr = self._get_xr_input_device("/user/hand/left")
        right_xr = self._get_xr_input_device("/user/hand/right")
        self._log_hand_tracking_status_once(left_xr, right_xr)

        base_pose = self._get_h1_base_pose_for_arms()
        if base_pose is None:
            return
        base_position, yaw = base_pose

        raw_targets = {}
        active_sides = set()
        for side, device in (("left", left_xr), ("right", right_xr)):
            hand_pose = self._get_hand_tracking_pose(device)
            pose_is_relative = False
            if hand_pose is None and self._controller_arm_control_enabled:
                hand_pose = self._get_controller_arm_pose(side, device)
                pose_is_relative = hand_pose is not None
            if hand_pose is None:
                continue
            active_sides.add(side)
            target_body = self._compute_arm_target_body_position(side, hand_pose, base_position, yaw, pose_is_relative)
            target_body = self._smooth_arm_rig_target(side, target_body)
            self._update_arm_rig_target(side, target_body, base_position, yaw)
            self._update_grabbed_object(side, self._is_controller_arm_pose_enabled(device))
            raw_targets.update(self._compute_arm_targets_from_body_position(side, target_body))

        for side in ("left", "right"):
            if side not in active_sides:
                self._set_arm_rig_target_visible(side, False)
                self._smoothed_arm_rig_targets.pop(side, None)
                self._active_h1_hand_target_matrices.pop(side, None)
                self._release_grabbed_object(side)

        if not raw_targets:
            return
        targets = self._smooth_and_clamp_arm_targets(raw_targets)
        dof_indices = sorted(targets)
        self.h1.robot.set_dof_position_targets([targets[index] for index in dof_indices], dof_indices=dof_indices)

    def _read_xr_controller_axes(self) -> tuple[float, float]:
        """Return forward and yaw commands from XR controller buttons/sticks."""
        left_xr = self._get_xr_input_device("/user/hand/left")
        right_xr = self._get_xr_input_device("/user/hand/right")
        self._log_xr_input_status_once(left_xr, right_xr)

        right_trigger = self._get_xr_gesture_value(right_xr, "trigger", "value")
        right_trigger_click = self._get_xr_gesture_value(right_xr, "trigger", "click")
        forward = max(
            right_trigger if right_trigger >= self._locomotion_trigger_threshold else 0.0,
            right_trigger_click,
        )
        forward = self._apply_deadzone(forward)
        turn_left = max(
            self._get_xr_gesture_value(left_xr, "x", "click"),
            self._get_xr_gesture_value(left_xr, "x", "value"),
        )
        turn_right = max(
            self._get_xr_gesture_value(right_xr, "a", "click"),
            self._get_xr_gesture_value(right_xr, "a", "value"),
        )

        yaw = self._apply_deadzone(turn_left - turn_right)
        return forward, yaw

    def _read_gamepad_controller_axes(self) -> tuple[float, float]:
        """Return forward and yaw commands from a normal gamepad fallback."""
        left_stick_forward = self._get_gamepad_value(carb.input.GamepadInput.LEFT_STICK_UP)
        left_stick_backward = self._get_gamepad_value(carb.input.GamepadInput.LEFT_STICK_DOWN)
        right_stick_left = self._get_gamepad_value(carb.input.GamepadInput.RIGHT_STICK_LEFT)
        right_stick_right = self._get_gamepad_value(carb.input.GamepadInput.RIGHT_STICK_RIGHT)
        right_trigger = self._get_gamepad_value(carb.input.GamepadInput.RIGHT_TRIGGER)
        button_a = self._get_gamepad_value(carb.input.GamepadInput.A)
        button_x = self._get_gamepad_value(carb.input.GamepadInput.X)

        forward = self._apply_deadzone(max(left_stick_forward - left_stick_backward, right_trigger))
        stick_yaw = self._apply_deadzone(right_stick_left - right_stick_right)
        button_yaw = self._apply_deadzone(button_x - button_a)
        yaw = stick_yaw if abs(stick_yaw) > abs(button_yaw) else button_yaw
        return forward, yaw

    def _update_controller_command(self, dt: float) -> None:
        """Map VR/gamepad inputs to H1 policy command velocities.

        Suggested VR/gamepad mapping:
            - Right trigger: walk forward
            - Step in place: walk forward from headset vertical peak/trough detection
            - X button: turn left
            - A button: turn right
            - Left/right grip or squeeze: arm pose teleoperation
        """
        if self._controller_command is None:
            return

        xr_forward, xr_yaw = self._read_xr_controller_axes()
        gamepad_forward, gamepad_yaw = self._read_gamepad_controller_axes()
        headset_gait_forward = self._update_headset_gait_command(dt)
        forward = max(xr_forward, gamepad_forward, headset_gait_forward)
        yaw = xr_yaw if abs(xr_yaw) > abs(gamepad_yaw) else gamepad_yaw

        self._controller_command[0] = self._max_forward_speed * forward
        self._controller_command[1] = 0.0
        self._controller_command[2] = self._max_yaw_speed * yaw

    def _sub_keyboard_event(self, event: object, *args: object, **kwargs: object) -> bool:
        """Handle keyboard input for robot control.

        Args:
            event: The keyboard event.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True to indicate the event was handled.
        """
        torch = import_module("torch")
        if self._keyboard_command is None:
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # On pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._keyboard_command += torch.tensor(
                    self._input_keyboard_mapping[event.input.name], device=self._base_command.device
                )
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # On release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._keyboard_command -= torch.tensor(
                    self._input_keyboard_mapping[event.input.name], device=self._base_command.device
                )
        return True

    def _unsubscribe_keyboard(self):
        """Unsubscribe from keyboard events if currently subscribed."""
        if self._sub_keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None

    def physics_cleanup(self):
        """Clean up physics resources."""
        # Deregister physics callback
        if self._physics_callback_id is not None:
            try:
                SimulationManager.deregister_callback(self._physics_callback_id)
            except Exception as e:
                carb.log_warn(f"Could not deregister callback {self._physics_callback_id}: {e}")
            self._physics_callback_id = None

        self._event_timer_callback = None
        self._unsubscribe_keyboard()
        # Flush the recorded session even when the example is closed without a scene
        # clear (setup_post_clear also saves; the call is idempotent, so both are safe).
        self._save_behavioral_data()
        if self._eye_gaze_tracker is not None:
            self._eye_gaze_tracker.cleanup()
            self._eye_gaze_tracker = None
        self.h1 = None
        self._physics_ready = False
        self._head_camera_transform_op = None  # handles die with the stage; never reuse them
        self._xr_anchor_op = None
        self._restore_physics_simulation_state()

    def _restore_physics_simulation_state(self) -> None:
        """Restore the physics sim device and fabric state captured in ``setup_scene``."""
        restore_physics_simulation_state(self._prev_physics_sim_device, self._prev_fabric_enabled)
        self._prev_physics_sim_device = None
        self._prev_fabric_enabled = None
