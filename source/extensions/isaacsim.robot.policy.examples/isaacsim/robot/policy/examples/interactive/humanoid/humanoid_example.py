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

"""Interactive humanoid simulation example: Unitree G1 with dexterous hands, VR teleoperated."""

from __future__ import annotations

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
from isaacsim.robot.policy.examples.interactive.humanoid.material_highlights import MaterialHighlights
from isaacsim.robot.policy.examples.interactive.humanoid.xr_pose import read_world_pose, smoothing_alpha
from isaacsim.robot.policy.examples.interactive.utils import (
    restore_physics_simulation_state,
    snapshot_physics_simulation_state,
)
from isaacsim.robot.policy.examples.robots import G1TeleopRobot
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

#: Arm teleoperation inverse kinematics, measured rather than hand-tuned.
#:
#: The previous mapping was four hand-written gain rows, and measured against the real
#: hand link it was wrong in ways no amount of re-tuning would have found by feel:
#: commanded "up" moved the hand DOWN (correlation -0.92 over 24 cm of travel),
#: commanded "outward" saturated after 2 cm, and "forward" was doing most of the vertical
#: motion. That is what "the hand does not go where my hand goes" actually was.
#:
#: These come from a measured Jacobian of hand position against the four driven arm
#: joints, taken around ARM_IK_NEUTRAL on the right arm (condition number 7.4, so the
#: inverse is well behaved). ARM_IK_PINV is its damped least-squares pseudo-inverse:
#:
#:     dq = ARM_IK_PINV @ (target_body - ARM_IK_REFERENCE)
#:
#: Left-arm values mirror about the sagittal plane: negate the target's y, then negate
#: the resulting roll and yaw deltas.
#:
#: Re-measure with scratchpad/measure_arm_jacobian.py if the asset or the neutral pose
#: changes -- do not re-tune these by hand.
ARM_IK_JOINT_ORDER = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
#: The neutral pose is deliberately an ARM-RAISED one. A linear inverse is only good near
#: the pose it was measured at, and the first version was taken with the arm hanging (hand
#: at 0.79 m), so its +/-0.30 m band spanned 0.63-1.03 m -- while the work surfaces are at
#: 0.99 and 1.08 m and the packages sit at 0.99-1.15 m. The arm itself reaches 1.40 m, so
#: nothing was physically out of range; the linearisation was simply centred in the wrong
#: place, and raising ARM_IK_MAX_OFFSET did not help (measured: ceiling stayed at ~1.03 m
#: for caps of 0.30 through 0.60). Centred here the band is 0.74-1.34 m, which covers both
#: surfaces and every package, and the Jacobian is better conditioned too (5.4 vs 7.4).
ARM_IK_NEUTRAL = (-1.20, 0.0, 0.0, 0.60)  # rad, the pose the Jacobian was taken at
ARM_IK_REFERENCE = (0.4121, -0.0890, 0.2832)  # m, right hand vs pelvis at that pose
ARM_IK_PINV = (
    (-4.30655, -1.07936, -2.41711),  # shoulder_pitch
    (-0.89639, +2.09579, -0.38582),  # shoulder_roll
    (-0.77920, +1.23439, -0.23597),  # shoulder_yaw
    (+7.68972, +1.71222, -0.58731),  # elbow
)
#: m: the linear inverse is only valid near the neutral pose, so the requested offset is
#: capped to this radius before it is applied. Beyond it the hand stops tracking rather
#: than flinging the arm somewhere the linearisation cannot justify.
ARM_IK_MAX_OFFSET = 0.30


class HumanoidExample(BaseSample):
    """A humanoid robot simulation example using the Unitree G1 with dexterous hands.

    This class demonstrates a complete humanoid teleoperation setup with real-time control
    capabilities. The default 100 Hz simulation supports VR arm and finger control
    while the G1's pelvis remains anchored at its spawn pose.

    The robot is the 29-DOF G1 fitted with Inspire five-finger hands — the Unitree humanoid
    that Isaac Teleop drives for dexterous manipulation. Stationary mode disables
    walking/turning commands and the balance policy, uses a world-to-pelvis fixed
    articulation root, and holds the standing posture with gravity off on robot links.
    The arms and fingers remain articulated; scene objects retain their gravity.
    Optional policy/kinematic locomotion must be selected before loading a new scene.

    VR extensions (this fork — see HUMANOID_VR_CONTROL.md for the full guide):
        - Finger teleoperation: OpenXR hand-tracking joints give a per-finger curl that
          drives the G1's real finger joints. With controllers, grip enables arm
          movement/rotation and trigger curls all fingers for pickup.
        - Headset gait: bob the HMD up/down (step in place) to walk forward; a
          horizontal-motion gate suppresses false triggers from nodding on the spot.
          Currently DISABLED by default (_headset_gait_enabled = False) while step
          detection is tuned; stationary mode suppresses it regardless of the toggle.
          The HMD pose is still read every step for logging.
        - Quest Pro eye gaze (eye_gaze_tracker.py): the runtime's calibrated unified
          gaze (its own fusion of both eyes) drawn as a red ray with a large
          blood-red marker sphere at the gazed collider (boxes, ground), gazed box
          tinted yellow, and live "[EyeGaze] looking at ..." terminal events on
          every target change.
        - First-person eye camera: the viewport/XR camera follows the G1 head at eye height.
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
        # Physics runs on the CPU at 100 Hz, and both halves of that matter.
        #
        # CUDA was inherited from the H1 sample and is simply the wrong device here. GPU
        # PhysX is built to step thousands of environments in parallel; with a single
        # robot you pay the kernel-launch overhead and get none of the parallelism.
        # Measured on this scene: 23-28 ms per step on CUDA against 6.8 ms on CPU. The
        # sim was running at 0.22x real time, which is what "terrible movement",
        # "it never stops" and the sluggish turns actually were -- commands arriving
        # late into a world playing back five times too slowly.
        #
        # 100 Hz then buys the rest: the per-step cost barely moves but the budget
        # doubles to 10 ms, giving 1.39x real time with headroom. Control stays at 50 Hz
        # because the walking policy's decimation is derived from dt, not hard-coded.
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 100.0  # 100 Hz physics
        self._world_settings["rendering_dt"] = 1.0 / 90.0  # VR-friendly rendering cadence
        self._world_settings["device"] = "cpu"
        self._world_settings["backend"] = "torch"

        self._base_command = None
        self._physics_ready = False
        self.g1 = None
        self._physics_callback_id = None
        self._event_timer_callback = None
        self._sub_keyboard = None
        self._input = None
        self._keyboard = None
        self._gamepad = None
        self._xr_core = None
        self._xr_input_status_logged = False
        self._locomotion_input_logged = False
        self._locomotion_brake = False  # raised by the keyboard brake: full stop
        self._latest_stick_lateral = 0.0
        self._keyboard_brake = False
        self._keyboard_command = None
        self._controller_command = None
        self._controller_deadzone = 0.15
        self._locomotion_trigger_threshold = 0.25
        self._arm_pose_enable_threshold = 0.25
        self._max_forward_speed = 1.0
        #: m/s cap for strafing. The walking policy has always accepted a lateral
        #: velocity (WALK_MAX_COMMAND = (0.8, 0.5, 1.57)) but nothing ever sent one, so
        #: the left stick only ever drove one axis. Capped below forward because the
        #: gait is less assured sideways.
        self._max_lateral_speed = 0.5
        # Reverse is ON/OFF at this exact value, not proportional, and the value is not a
        # preference -- it is the only one that works. Measured achieved speed against
        # commanded, over 5 s each: -0.80 -> -0.119 m/s (reverses), -0.60 -> +0.215,
        # -0.40 -> +0.389, -0.20 -> +0.528. Every gentler reverse command walks the robot
        # FORWARDS, because the policy's forward steady-state error outweighs it. So the
        # trigger commands full reverse or nothing; there is no usable middle.
        self._max_backward_speed = 0.8  # the only commanded reverse the policy honours
        self._max_yaw_speed = 1.0
        # Command ramping, asymmetric on purpose (see _smooth_base_command). Easing INTO
        # a command stops the gait lurching; easing out of one just makes the robot feel
        # like it will not stop, so the release is nearly immediate.
        self._command_attack_time = 0.30  # s: ramp up — smooths starts and turns
        self._command_release_time = 0.06  # s: ramp down — release the stick, it stops
        self._headset_gait_enabled = (
            False  # temporarily disabled: step detection not yet stable; flip to True to restore
        )
        self._headset_gait_forward_intensity = 1.0  # full speed walk per detected step
        self._headset_gait_min_amplitude = 0.012  # lowered: detect smaller head bobs (~1.2 cm)
        self._headset_gait_min_step_interval = 0.18  # allow faster cadence
        self._headset_gait_max_extremum_gap = 0.95
        self._headset_gait_baseline_time = 1.4
        self._headset_gait_filter_time = 0.05  # faster low-pass response
        self._headset_gait_velocity_deadzone = 0.008  # more sensitive direction detection
        self._headset_gait_pulse_duration = 0.50  # longer walk burst so motion is visible
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
        self._head_camera_path = "/World/G1_HeadCamera"
        self._head_camera_transform_op = None
        self._head_camera_mount_body_path = None
        self._head_camera_mount_local = None
        self._head_camera_mount_prim = None
        self._head_camera_update_sub = None
        self._physics_step_error_logged = set()  # (subsystem, error) pairs already warned about
        # Camera geometry, re-tuned for the G1: it stands 1.32 m tall against the H1's
        # 1.80 m, so every offset that was measured against the H1 skull had to shrink.
        self._first_person_head_forward_offset = 0.26  # well ahead of the head so the camera never meets
        # the robot mesh and the view stays fully clear
        self._first_person_head_up_offset = 0.0  # extra fine-tune on top of the eye height below
        self._first_person_eye_height_above_base = 0.58  # m above the pelvis/base link: ~5 cm above the top of
        # the G1 head — close to first-person but clear of the
        # head mesh (0.46 = strict eye level inside the head)
        self._head_camera_yaw_sign = 1.0  # flip to -1.0 only if the DESKTOP view turns opposite
        # to the robot; the in-VR reversal was caused by
        # per-step camera forcing, fixed by the XR anchor below
        # The stationary manipulation view is a rigid robot-mounted camera. The
        # remaining modes are retained for later experiments with a moving robot.
        #   "robot_head"    - fixed camera-to-head mount; ignores physical HMD motion.
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
        #   "stage_anchor"  - XRCore.schedule_set_stage_anchor(anchor prim). The runtime
        #                     treats the anchor Xform as the physical space origin, so
        #                     nothing fights its reprojection. Smoothest when the build
        #                     re-reads a moving anchor prim; try it if head_compose
        #                     still feels detached.
        self._xr_camera_mode = "robot_head"
        self._xr_anchor_path = "/World/G1_XRAnchor"
        self._xr_anchor_op = None
        self._xr_anchor_configured = False
        self._xr_camera_states_logged = set()
        self._xr_anchor_forward_offset = 0.18  # m: anchor ahead of the base so the robot's head and
        # shoulders stay out of the user's view
        self._xr_anchor_height_offset = -0.30  # m: shifts the whole VR rig; your real standing eye
        # height adds on top. NEGATIVE for the G1: it is only
        # 1.32 m tall, so the rig must sink below the floor for
        # your eyes to land near the robot's (~1.25 m). Raise
        # towards 0 if you are shorter, lower if you are taller.
        self._xr_anchor_yaw_offset_deg = -90.0  # aligns physical "room forward" with robot +X in
        # head_compose mode. -90 is the exact value for a
        # user facing down physical -Z; auto-calibration
        # below replaces it with the measured one.
        # --- VR rig auto-calibration (the "camera is not on the robot" fix) ---
        # The two rig constants above are only correct for one body height and one
        # standing direction, and getting either wrong looks exactly like a detached
        # camera: you are either sunk into the floor, floating above the robot, or
        # facing 90 degrees off its walking direction. Both are measurable from the
        # headset itself, so they are measured instead of guessed -- once, latched, over
        # the first quarter-second of tracked head poses, and re-run on demand with B.
        self._xr_auto_calibrate = True
        self._xr_yaw_base_offset_deg = -90.0  # exact axis term; the measured user yaw adds to it
        self._xr_calibration_samples = []  # (physical head height, physical head yaw)
        # Latching after a quarter-second was too eager: it fires the moment the head
        # pose first reads, which can be while the headset is still being put on or is
        # resting on a desk. Capturing 1.27 m there and calling it your standing eye
        # height puts the rig 30-40 cm too low, and the robot then appears below you and
        # away -- indistinguishable from a camera that never attached. So: a longer
        # window, and it must be STILL before the value is trusted.
        #: Gf.Vec3d, the operator's calibrated head offset already rotated into the
        #: pre-anchor (Z-up) stage frame. None until the rig has been calibrated.
        #: Subtracting all THREE components is what puts them inside the robot; an
        #: earlier version subtracted only the height and left them standing however
        #: far from the robot they happened to be from their room's origin.
        self._xr_calibration_head_offset = None
        self._xr_calibration_window = 2.0  # s of head pose considered
        self._xr_calibration_max_spread = 0.15  # m: height range allowed within it
        self._xr_calibrated = False
        self._xr_min_head_height = 0.40  # m: below this the pose is not a real standing head
        self._xr_max_head_height = 2.40  # m: above this it is not one either
        self._xr_recenter_button_down = False  # B (right controller) edge detection
        self._xr_mode_button_down = False  # left thumbstick click edge detection
        #: Cycled in-headset by clicking the LEFT thumbstick. Which of these actually
        #: lands the view on the robot depends on how this runtime interprets
        #: schedule_set_camera, which cannot be established from outside a live headset
        #: -- so it is switchable from inside one rather than guessed at.
        self._xr_camera_mode_cycle = ("robot_head", "head_compose", "stage_anchor", "camera_lock")
        # --- automatic convention detection ---
        # schedule_set_camera(M) either makes the rendered VIEW equal M (the runtime
        # subtracting the live head pose internally) or sets the RIG ORIGIN to M and
        # composites the head on top. Which one decides whether the head term must be
        # pre-multiplied in or left out -- and getting it wrong throws the view a whole
        # head-pose away from the robot, which is exactly the 'camera is separate from
        # the robot' symptom. It cannot be settled from outside a live headset, and the
        # 'attached' log line only ever meant the call did not raise. So measure it:
        # try each candidate briefly and keep whichever lands the virtual head closest
        # to the robot's eye.
        self._xr_convention_autodetect = True
        self._xr_convention_candidates = ("compose", "anchor_only")
        self._xr_convention = None  # resolved candidate name
        self._xr_convention_index = 0  # which candidate is being tried
        self._xr_convention_errors = {}  # name -> list of eye-distance samples
        self._xr_convention_frames = 45  # samples per candidate (~0.45 s at 100 Hz)
        self._xr_convention_pending = None  # (name, desired eye position) awaiting readback
        self._xr_diagnostic_countdown = 400  # physics steps before the one-shot state dump
        self._xr_diagnostic_done = False
        self._drop_button_down = False  # Y (left controller) edge detection
        self._xr_head_device_handle = None  # resolved once; may not be /user/head on every runtime
        #: Last usable physical head pose, and how many steps ago it was read. The
        #: runtime hands back an identity pose for the occasional frame, and treating
        #: that as 'no headset' dropped the whole rig to camera-lock for as long as it
        #: lasted -- which is what 'the camera is not always attached' was. Riding out
        #: the gap on the last good pose is correct: the head has not teleported.
        self._xr_last_head_pose = None
        self._xr_head_pose_age = 0
        self._xr_head_pose_max_age = 30  # steps (~0.3 s at 100 Hz) before giving up
        self._head_camera_last_base = None  # stashed by _get_head_camera_pose for the anchor
        self._head_camera_last_yaw = None
        # --- Camera stabilization (the VR wobble fix) ---
        # A walking humanoid's pelvis bobs ~2-3 cm vertically, sways laterally and
        # wobbles in yaw once per step. At the 0.8 s gait period that is ~1.25 Hz, right
        # in the band that causes VR sickness, and the camera rides the pelvis. A real
        # neck does not pass that through: the head stays far steadier than the hips.
        # These first-order low-passes reproduce that. Height gets the strongest
        # filtering (nothing here intentionally changes the robot's height, so lag costs
        # nothing); x/y and yaw are filtered more gently because walking and turning are
        # intentional and should not feel sluggish.
        # Filtering x/y harder than it might seem safe to is fine: a first-order lag on
        # position does not slow the camera down, it just parks it a few centimetres
        # behind the robot at constant speed — a static offset nobody notices, unlike the
        # oscillation it removes. (Measured: travel 3.36 m -> 3.31 m over 6 s.)
        self._camera_stabilization_enabled = True
        self._camera_height_filter_time = 0.35  # s: removes the gait bob
        self._camera_lateral_filter_time = 0.30  # s: removes side-to-side sway
        self._camera_yaw_filter_time = 0.22  # s: removes per-step yaw wobble
        self._camera_filtered_base = None
        self._camera_filtered_yaw = None
        self._first_person_head_target_distance = 1.8
        #: m: how far BELOW the horizon the first-person view aims, at the target
        #: distance above. Rendering the view showed the operator looking at a brick
        #: wall: the camera aimed dead level while the work surface sits 0.5 m ahead
        #: and 0.5 m below, about 45 degrees down and completely out of frame. A
        #: person working at a bench does not stare at the horizon. In VR head
        #: tracking still lets you look anywhere; this only sets where you start.
        self._first_person_head_target_drop = 0.65
        # Headset velocity and horizontal-motion tracking (gait gate)
        self._last_headset_raw_position = None  # Gf.Vec3d: position from pose reader
        self._last_headset_pose_matrix = None  # Gf.Matrix4d: full pose for orientation logging
        self._headset_prev_position = None  # position from previous step for velocity
        self._headset_velocity = Gf.Vec3d(0.0, 0.0, 0.0)
        self._headset_velocity_filter_time = 0.10  # low-pass time constant (s)
        self._headset_horiz_speed = 0.0  # floor-plane speed magnitude (m/s)
        self._headset_gait_min_horiz_speed = 0.025  # m/s threshold: suppress gait below this
        self._headset_gait_horiz_gate = 0.0  # 0-1 multiplier applied to gait output
        self._headset_gait_step_event = False  # True for one sample when step fires
        # Behavioral data collection for AI/RL training
        self._behavioral_data_enabled = True
        self._behavioral_data_records = []
        self._behavioral_data_step_counter = 0
        # Logging intervals are derived from the physics rate, not hard-coded, so the
        # dataset keeps its ~100 Hz sampling whatever physics_dt is set to. They used to
        # be literals tuned for 200 Hz; dropping to 100 Hz would have silently halved
        # every rate in the recorded data.
        physics_hz = 1.0 / self._world_settings["physics_dt"]
        self._behavioral_data_log_every_n_steps = max(1, round(physics_hz / 100.0))  # ~100 Hz
        self._behavioral_flush_every_n_steps = max(1, round(physics_hz * 2.5))  # ~2.5 s of data
        self._behavioral_csv_fieldnames = {}  # filename -> header columns, fixed at first flush
        self._behavioral_data_output_dir = Path.home() / "BehavioralCollection"
        self._behavioral_sessions_root = self._behavioral_data_output_dir / "raw_sessions"
        self._behavioral_session_dir = None  # Path to the current session folder
        self._behavioral_session_id = None
        self._behavioral_dof_names = []  # populated on first sample
        # Session-relative logs added alongside behavior.csv
        self._hand_tracking_records = []
        self._gaze_records = []
        self._object_state_records = []
        self._object_state_prev_positions = {}  # object path -> Gf.Vec3d, for finite-diff velocity
        self._gaze_raycast_max_distance = 20.0  # m: range cap for the HMD-forward gaze raycast
        # Meta Quest Pro eye tracking (optional; gaze.csv falls back to HMD-forward without it)
        self._eye_gaze_enabled = True
        self._eye_gaze_ray_visual_enabled = True  # draw the red gaze ray + hit marker in the scene
        self._eye_gaze_tracker = None  # EyeGazeTracker instance once XR is up
        # Eye-camera frame capture (~10 Hz PNG sequence)
        self._behavioral_frame_records = []
        self._behavioral_frame_dir = None
        self._behavioral_frame_camera = None  # None = not yet tried, False = failed, else Camera
        self._behavioral_frame_log_every_n_steps = max(1, round(physics_hz / 10.0))  # ~10 Hz
        self._behavioral_frame_counter = 0  # persistent PNG index: the records buffer is
        # cleared on every flush, so len() must not name files
        self._head_camera_update_counter = 0
        self._head_camera_update_interval = 1
        self._last_physics_dt = self._world_settings["physics_dt"]  # for subsystems not passed dt
        self._teleop_input_dt = self._last_physics_dt
        self._last_teleop_wall_time = None
        self._hand_tracking_arm_control_enabled = True
        self._controller_arm_control_enabled = True
        self._hand_tracking_status_logged = False
        self._g1_arm_dofs_configured = False
        self._g1_arm_dof_indices_by_side = {}
        self._g1_arm_joint_names_by_side = {}
        self._g1_arm_joint_defaults = {}
        self._g1_arm_joint_limits = {}
        self._hand_pose_candidates = ("palm", "wrist", "grip", "aim", "")
        self._controller_pose_candidates = ("grip", "aim", "")
        # Names runtimes use for the thumbstick; the first that answers wins.
        self._xr_stick_input_candidates = ("thumbstick", "joystick", "trackpad")
        # --- real IK, on the live PhysX Jacobian ---
        # The ARM_IK_* constants below are a single Jacobian measured once at one pose;
        # they are only right near that pose, which is what makes reaching feel like the
        # arm is fighting you. PhysX recomputes the true Jacobian every step, so use it
        # and solve properly. Set False to fall back to the measured constants.
        self._arm_ik_use_jacobian = True
        self._arm_ik_damping = 0.08  # damped-least-squares lambda; higher = smoother, slower
        self._arm_ik_gain = 0.55  # fraction of the solved step taken per control tick
        self._arm_ik_max_step = 0.25  # rad: cap per joint per tick, so it never snaps
        self._arm_ik_max_error = 0.30  # m: cap the position error supplied to each solve
        self._arm_max_joint_speed = 2.5  # rad/s: applies to IK, fallback, and reacquisition
        self._arm_max_tracking_error = 0.15  # rad: bounded drive error even if an arm is blocked
        self._arm_track_orientation = True
        self._arm_orientation_weight = 0.2  # metres per radian in the weighted IK residual
        self._arm_posture_gain = 0.1  # resolve redundant motion toward the comfortable starting posture
        self._arm_orientation_anchors = {}
        self._robot_palm_local_frames = {}
        self._robot_palm_local_centers = {}
        self._arm_input_sources = {}
        self._arm_ik_link_index = {}  # side -> articulation link index of the hand
        self._arm_ik_logged = False
        self._arm_smoothing = 0.34
        self._smoothed_arm_targets = {}
        self._arm_rig_smoothing = 0.38
        self._smoothed_arm_rig_targets = {}
        self._controller_arm_neutral_positions = {}
        self._controller_arm_neutral_targets = {}
        # All distances use live rigid-body positions, not desired XR targets.
        self._grab_radius = 0.18  # m: object-centre search around the measured palm/fingers
        # This centre-distance gate is assistance, not a collision/contact test.
        self._grasp_contact_distance = 0.09  # m: nearest palm/finger link centre to object centre
        self._grasp_link_prims = {}  # side -> [RigidPrim] palm and finger links
        self._grasp_joint_root = "/World/G1_GraspJoints"
        self._grasp_joints_by_side = {}  # side -> fixed joint holding an object
        self._grab_assist_radius = 0.30  # m: candidate highlight only; never pulls an object
        self._material_highlights = MaterialHighlights()
        self._grab_candidate_highlight = True  # tint the package the trigger would take
        self._grab_candidate_color = Gf.Vec3f(0.15, 1.0, 0.3)  # green = squeeze to take this
        self._grab_candidate_material_path = "/World/G1_GrabCandidateMaterial"
        self._grab_candidate_by_side = {}  # side -> currently tinted package path
        self._grab_miss_report_time = {}  # side -> sim time of the last 'nothing in range'
        self._grab_miss_report_interval = 2.0  # s between repeats of that message
        self._grab_time = 0.0  # accumulated physics time, for the above
        self._grabbed_objects_by_side = {}
        self._hand_link_prims = {}  # side -> RigidPrim on the real hand link
        self._object_rigid_prims = {}  # package path -> live physics pose reader
        self._grab_requires_release = {}  # inhibit re-grasp after drop or tracking loss
        self._hand_closed_by_side = {}
        self._arm_rig_root_path = "/World/G1_ArmControlRig"
        self._arm_rig_target_paths = {
            "left": f"{self._arm_rig_root_path}/LeftHandTarget",
            "right": f"{self._arm_rig_root_path}/RightHandTarget",
        }
        self._arm_rig_target_ops = {}
        self._sample_box_root_path = "/World/G1_SampleBoxes"
        self._sample_box_count = 10
        self._sample_box_seed = 12
        self._sample_box_density = 5.0
        self._sample_box_min_mass = 0.45
        self._sample_box_max_mass = 4.5
        # --- Warehouse environment ---
        # Isaac ships no "factory" environment (Props/Factory is bolts and nuts for the
        # assembly tasks); the warehouses are its industrial scenes. full_warehouse comes
        # already dressed with racking, pallets, stacked boxes and forklifts.
        self._environment_usd_path = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
        self._environment_prim_path = "/World/ground"
        self._environment_is_warehouse = True
        self._robot_spawn_xy = (0.0, 0.0)  # clear floor in full_warehouse's central aisle
        # --- Pickable packages ---
        # Real warehouse crates rather than procedural cubes. Sized for a facility, not for
        # a G1 hand, which is why the grab system (close hand near it -> attaches) carries
        # the interaction rather than real finger friction.
        # The YCB set, which is the standard benchmark for grasping, and the only thing
        # here a G1 hand can physically close on. Measured widths against a ~0.12 m hand:
        # the warehouse crates that used to be here are 0.60 x 0.40 m and the small KLT
        # bin is 0.20 x 0.30 m -- all far too big, which is why the grab had to be a
        # magnet and why no amount of teleop tuning made picking feel real.
        self._package_usd_paths = (
            "/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",  # 0.068 x 0.102
            "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",  # 0.102 x 0.084
            "/Isaac/Props/YCB/Axis_Aligned/061_foam_brick.usd",  # 0.078 x 0.051
            "/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",  # 0.093 x 0.176
            "/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",  # 0.096 x 0.191
        )
        self._package_spawn_radius = (1.2, 3.0)  # m from the robot: within a short walk
        self._package_use_props = True  # False falls back to procedural cubes
        # --- Work surfaces ---
        # Packages go on tables, not the floor, and the reason is measured: standing, the
        # G1's hand bottoms out at 0.778 m while a crate on the floor tops out at 0.23 m.
        # It is 0.55 m short, and reaching harder topples it (the walking policy was
        # trained with fixed arms and cannot compensate for the shift). Picking off the
        # ground needs a robot that can squat; these tables put the work in the hand's
        # real 0.78-1.33 m band instead.
        self._work_surface_root_path = "/World/G1_WorkSurfaces"
        # Both surfaces are the LOWER of the two packing tables on purpose. The taller one
        # (packing_table.usd, top 1.083 m) put the biggest crate's centre at 1.147 m, which
        # is above where the teleoperated hand can comfortably get even after the arm IK
        # was re-centred -- measured reach ceiling 1.097 m in the operator's commanded
        # range. A table you cannot reach the far half of is not a work surface.
        # One entry per placement below. The FIRST is a small side table, not a packing
        # bench: a 2.47 m bench in front of the spawn walls the robot in (measured -- it
        # could only walk 0.19 m before hitting it), whereas this one is 0.92 x 0.42 m
        # and can be stepped around. Its top is 0.812 m, inside the hand's reachable
        # 0.80-1.30 m band, so objects on it are grabbable without moving at all.
        self._work_surface_usd_paths = (
            "/Isaac/Environments/Hospital/Props/SM_SideTable_02a.usd",  # top 0.812 m
            "/Isaac/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/"
            "SM_HeavyDutyPackingTable_C02_01_physics.usd",  # top 0.994 m
            "/Isaac/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/"
            "SM_HeavyDutyPackingTable_C02_01_physics.usd",
        )
        #: (x, y, yaw_deg) per surface. Placed to the robot's left and right rather than
        #: straight ahead: with a table dead ahead at 2.1 m the robot walked into it after
        #: one metre (measured 1.01 m travelled, versus 3.30 m down a clear aisle), which
        #: is a poor first thirty seconds in a headset. Yaw 0 puts their 2.47 m length
        #: across the approach, so a quarter turn faces the long working edge.
        # (x, y, yaw_deg). The FIRST one is directly in front of the spawn and is the
        # one that matters: measured, the hand reaches 0.444 m in front of the pelvis,
        # so a bench whose near edge sits at x ~ 0.30 puts its working edge inside the
        # reachable shell and the operator can pick things up without moving at all.
        # The table is 0.762 m deep, so a centre at x = 0.68 puts that edge at 0.30.
        # The two side benches stay as somewhere to walk to.
        self._work_surface_placements = (
            # yaw 90 so the LONG axis runs across the robot's view and the shallow 0.42 m
            # depth points at it. With yaw 0 the long axis ran front-to-back, and the
            # 'line the packages up with where the robot stands' projection then placed
            # them beside the robot instead of in front -- 14 cm from the pelvis, inside
            # the arm's inner limit, where it physically cannot reach.
            (0.50, -0.28, 90.0),  # small side table, front-right, clear of the walking line
            (0.0, 2.0, 0.0),
            (0.0, -2.0, 0.0),
        )
        self._package_floor_count = 2  # left on the floor on purpose: unreachable
        # attempts are real signal for intent models
        #: m: clearance between a package's measured BASE and the surface it rests on.
        #: This used to be 0.12 applied to the package ORIGIN, which is not its base --
        #: these crate assets carry their origin near the centre, so a 0.5 m crate was
        #: spawned a quarter of a metre INSIDE the table and PhysX fired it across the
        #: room on the first step. Every package then ended up on the floor metres away,
        #: which is what 'nothing within reach' really was. The base is now measured.
        self._package_drop_height = 0.01
        #: m: how far IN from the near lip a package sits. A fixed inset, not a fraction
        #: of the depth -- 0.42 of a 0.42 m deep side table leaves 3 cm of lip and the
        #: objects simply fell off, while the same fraction on a 0.76 m bench is fine.
        #: A margin works for any surface depth.
        self._package_edge_margin = 0.12
        #: m: spacing between packages along a bench. Random placement put three of them
        #: within 0.26 m of each other while they were 0.4-0.6 m wide, so they spawned
        #: interpenetrating and PhysX fired them across the room -- which read as
        #: 'the packages are never where the robot can reach'. Deterministic now.
        self._package_spacing = 0.16
        self._active_g1_hand_target_matrices = {}
        self._preserve_existing_rig_calibration = True
        self._manual_arm_rig_target_world_positions = {}
        self._arm_rig_world_offsets = {}
        # G1 robot configuration. Unlike the H1 this asset carries real articulated hands,
        # so the fake box-hand attachments the H1 needed are gone entirely.
        self._g1_prim_path = "/World/G1"
        self._g1_hand_variant = "Inspire"  # "Inspire" (5 fingers) or "ThreeFinger" (Dex3)
        self._g1_spawn_position = [0.0, 0.0, 0.80]  # pelvis height of the standing posture:
        # the lowest spawn whose feet rest on the
        # ground instead of penetrating it
        # Locomotion mode:
        #   "stationary" - fixed world anchor, no walking/turning; arms and hands stay active.
        #   "policy"    - Unitree's pretrained G1 walking policy drives the 12 leg joints
        #                 with real physics-based gait; waist and arms stay free for
        #                 teleoperation. Needs robots/data/g1_unitree_motion.pt.
        #   "kinematic" - no policy: the posture is held and the base glides on command.
        # Choose a mode before loading the scene; the fixed anchor changes its topology.
        self._g1_locomotion = "stationary"
        # Finger teleoperation
        self._finger_control_enabled = True
        self._finger_smoothing = 0.35  # low-pass on curl, per physics step
        self._finger_curl_deadzone = 0.04  # ignore controller/tracking noise near open
        self._smoothed_finger_curls = {}  # side -> {finger role: curl}
        self._latest_finger_curls = {}  # side -> {finger role: curl}, for hand_tracking.csv
        self._finger_curl_source = {}  # side -> "hand_tracking" | "controller" | "none"
        self._finger_grab_threshold = 0.55  # mean curl at which a nearby box is grabbed
        self._finger_release_threshold = 0.35  # hysteresis avoids repeated attach/detach near the threshold
        self._grab_trigger_release_threshold = 0.35
        self._grab_trigger_threshold = 0.6  # trigger pull that counts as 'take this'.
        # Deliberately high: a light touch curls the
        # index finger without grabbing anything.
        self._finger_roles = ("thumb", "index", "middle", "ring", "little")
        # Sum adjacent bone bends, including the knuckle, instead of comparing only
        # the first and last bone. A tightly folded finger can turn past 180 degrees;
        # the endpoint angle would decrease again and incorrectly reopen the robot.
        self._finger_curl_joint_chains = {
            "thumb": ("thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip"),
            "index": ("index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip"),
            "middle": ("middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip"),
            "ring": ("ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip"),
            "little": ("little_metacarpal", "little_proximal", "little_intermediate", "little_distal", "little_tip"),
        }
        # Bone angle counted as a fully closed finger. The thumb only folds about halfway
        # as far as the fingers do, so sharing one threshold would leave it half open in
        # a fist. Each value is the sum of bends from the metacarpal to the fingertip.
        self._finger_curl_full_flexion_rad = {
            "thumb": math.radians(95.0),
            "index": math.radians(150.0),
            "middle": math.radians(150.0),
            "ring": math.radians(150.0),
            "little": math.radians(150.0),
        }
        #: Thumb metacarpal angle in the palm plane, from the thumb-side direction.
        #: Opposing the thumb across the palm is separate from bending its two joints.
        self._thumb_opposition_open_angle = math.radians(45.0)
        self._thumb_opposition_closed_angle = math.radians(135.0)
        self._finger_tracking_status_logged = False
        self._prev_physics_sim_device: str | None = None
        self._prev_fabric_enabled: bool | None = None
        self._pressed_keys = set()

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
        """Bind a physics material to whatever the robot walks on.

        The grid environment exposes a single ``GroundPlane/CollisionPlane``; the
        warehouse has no such prim, so the material is bound to the environment root
        instead and inherits down to the floor meshes. Friction matters more here than it
        did with the kinematic base: the walking policy pushes against the floor, and on
        a slippery one the gait skates and falls.

        Args:
            static_friction: Static friction coefficient.
            dynamic_friction: Dynamic friction coefficient.
            restitution: Restitution coefficient.
        """
        stage = omni.usd.get_context().get_stage()
        material_path = f"{self._environment_prim_path}/Looks/PhysicsMaterial"

        material = UsdShade.Material.Define(stage, material_path)
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr().Set(static_friction)
        physics_material.CreateDynamicFrictionAttr().Set(dynamic_friction)
        physics_material.CreateRestitutionAttr().Set(restitution)

        targets = [f"{self._environment_prim_path}/GroundPlane/CollisionPlane", self._environment_prim_path]
        for path in targets:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            try:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
                )
                carb.log_info(f"HumanoidExample: bound physics material to {path}")
                return
            except Exception as e:
                carb.log_warn(f"HumanoidExample: could not bind physics material to {path}: {e}")

    def _create_environment(self) -> None:
        """Reference the environment the robot works in, and make sure it has a floor.

        The warehouse asset carries its own collision geometry. A ground plane is added
        underneath regardless as a safety net: without a collider anywhere, a
        physics-walking robot falls out of the world on the first step.
        """
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        stage_utils.add_reference_to_stage(
            usd_path=assets_root_path + self._environment_usd_path,
            path=self._environment_prim_path,
        )
        carb.log_info(f"HumanoidExample: loaded environment {self._environment_usd_path}")

        if self._environment_is_warehouse:
            stage = omni.usd.get_context().get_stage()
            backstop_path = f"{self._environment_prim_path}/G1_FloorBackstop"
            if not stage.GetPrimAtPath(backstop_path).IsValid():
                plane = UsdGeom.Plane.Define(stage, backstop_path)
                plane.CreateAxisAttr().Set("Z")
                plane.CreateWidthAttr(200.0)
                plane.CreateLengthAttr(200.0)
                UsdGeom.Imageable(plane.GetPrim()).MakeInvisible()
                UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

    def _create_sample_boxes(self) -> None:
        """Scatter pickable packages around the robot for VR grasping.

        Packages keep the ``Box_<nn>`` naming and the ``/G1_SampleBoxes`` root whatever
        their geometry, because three other systems key off them: the grab search, the
        ``object_states.csv`` logger, and the eye tracker's highlight filter.
        """
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, self._sample_box_root_path)
        rng = random.Random(self._sample_box_seed)
        assets_root_path = get_assets_root_path()
        use_props = self._package_use_props and assets_root_path is not None

        slots = self._build_package_slots(rng)
        for index in range(self._sample_box_count):
            box_path = f"{self._sample_box_root_path}/Box_{index:02d}"
            x, y, z, on_surface = slots[index]

            if use_props:
                size = self._create_package_from_prop(box_path, rng, x, y, z)
            else:
                size = self._create_package_cube(box_path, rng, x, y, z)

            prim = stage.GetPrimAtPath(box_path)
            if not prim.IsValid():
                continue
            package_mass = self._clamp_value(
                self._sample_box_density * size * size * size,
                self._sample_box_min_mass,
                self._sample_box_max_mass,
            )
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr().Set(package_mass)
            prim.CreateAttribute("g1:packageSize", Sdf.ValueTypeNames.Float).Set(size)
            prim.CreateAttribute("g1:packageMass", Sdf.ValueTypeNames.Float).Set(package_mass)
            # Logged so the dataset distinguishes a package the robot could actually
            # reach from one deliberately left out of reach on the floor.
            prim.CreateAttribute("g1:packageOnSurface", Sdf.ValueTypeNames.Bool).Set(bool(on_surface))

        on_surface_count = sum(1 for slot in slots if slot[3])
        carb.log_info(
            f"HumanoidExample: created {self._sample_box_count} packages under {self._sample_box_root_path} "
            f"({'warehouse props' if use_props else 'procedural cubes'}); "
            f"{on_surface_count} within reach on work surfaces, "
            f"{self._sample_box_count - on_surface_count} on the floor (unreachable by design)"
        )

    def _build_package_slots(self, rng: random.Random) -> list[tuple[float, float, float, bool]]:
        """Decide where every package spawns: ``(x, y, z, on_surface)`` per package.

        Most go on the work surfaces, inside the hand's measured 0.78-1.33 m band. A
        couple are left on the floor on purpose — the robot cannot reach those, and
        recording the operator trying anyway is useful supervision rather than a bug.
        """
        surfaces = self._get_work_surface_tops()
        floor_count = self._package_floor_count if surfaces else self._sample_box_count
        floor_count = min(floor_count, self._sample_box_count)

        slots: list[tuple[float, float, float, bool]] = []
        # How many packages each surface will hold, so each row can be centred.
        surface_total = self._sample_box_count - floor_count
        per_surface: dict[int, int] = {}
        if surfaces:
            for index in range(surface_total):
                key = index % len(surfaces)
                per_surface[key] = per_surface.get(key, 0) + 1
        surface_counts: dict[int, int] = {}
        for index in range(self._sample_box_count):
            if index >= self._sample_box_count - floor_count or not surfaces:
                # Floor package: ring placement around the robot.
                angle = rng.uniform(-math.pi, math.pi)
                distance = rng.uniform(*self._package_spawn_radius)
                slots.append(
                    (
                        self._robot_spawn_xy[0] + math.cos(angle) * distance,
                        self._robot_spawn_xy[1] + math.sin(angle) * distance,
                        0.06,
                        False,
                    )
                )
                continue

            surface_index = index % len(surfaces)
            centre_x, centre_y, yaw_deg, top_z, length, width = surfaces[surface_index]
            yaw = math.radians(yaw_deg)
            # Put packages on the edge of the surface FACING THE ROBOT, lined up with
            # where it stands -- not scattered over the middle. Mid-surface placement is
            # what made every package unreachable: the hand only reaches 0.44 m forward,
            # and half a table is deeper than that.
            along_axis = (math.cos(yaw), math.sin(yaw))
            across_axis = (-math.sin(yaw), math.cos(yaw))
            to_robot = (
                self._robot_spawn_xy[0] - centre_x,
                self._robot_spawn_xy[1] - centre_y,
            )
            side = to_robot[0] * across_axis[0] + to_robot[1] * across_axis[1]
            side = 1.0 if side >= 0.0 else -1.0
            across = side * max(0.0, width * 0.5 - self._package_edge_margin)
            # Lay them out in a row along the bench, centred on the robot's own line and
            # spaced so they cannot overlap. Random placement collided them together.
            robot_along = to_robot[0] * along_axis[0] + to_robot[1] * along_axis[1]
            seat = surface_counts.get(surface_index, 0)
            surface_counts[surface_index] = seat + 1
            offset = (seat - (per_surface.get(surface_index, 1) - 1) * 0.5) * self._package_spacing
            along = self._clamp_value(robot_along + offset, -0.42 * length, 0.42 * length)
            slots.append(
                (
                    centre_x + math.cos(yaw) * along - math.sin(yaw) * across,
                    centre_y + math.sin(yaw) * along + math.cos(yaw) * across,
                    top_z + self._package_drop_height,
                    True,
                )
            )
        return slots

    def _create_work_surfaces(self) -> None:
        """Place the packing tables the packages sit on."""
        assets_root_path = get_assets_root_path()
        if assets_root_path is None or not self._work_surface_placements:
            return
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, self._work_surface_root_path)

        for index, (x, y, yaw_deg) in enumerate(self._work_surface_placements):
            surface_path = f"{self._work_surface_root_path}/Surface_{index:02d}"
            wrapper = UsdGeom.Xform.Define(stage, surface_path)
            wrapper.ClearXformOpOrder()
            wrapper.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
            wrapper.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, yaw_deg))
            usd_path = assets_root_path + self._work_surface_usd_paths[index % len(self._work_surface_usd_paths)]
            stage_utils.add_reference_to_stage(usd_path=usd_path, path=f"{surface_path}/Asset")

            # Static scenery: collision so packages rest on it and the robot cannot walk
            # through it, but no rigid body, so it never gets shoved around.
            asset_prim = stage.GetPrimAtPath(f"{surface_path}/Asset")
            if asset_prim.IsValid() and not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(asset_prim)):
                UsdPhysics.CollisionAPI.Apply(asset_prim)

        carb.log_info(
            f"HumanoidExample: created {len(self._work_surface_placements)} work surfaces "
            f"under {self._work_surface_root_path}"
        )

    def _get_work_surface_tops(self) -> list[tuple[float, float, float, float, float]]:
        """Measure each work surface: ``(x, y, yaw_deg, top_z, length, width)``.

        The top height is measured rather than assumed — the two table assets differ by
        9 cm, and a pallet (0.14 m) would have put every package back out of reach.
        """
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(self._work_surface_root_path)
        if not root.IsValid():
            return []

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        surfaces = []
        for index, (x, y, yaw_deg) in enumerate(self._work_surface_placements):
            prim = stage.GetPrimAtPath(f"{self._work_surface_root_path}/Surface_{index:02d}")
            if not prim.IsValid():
                continue
            try:
                box_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if box_range.IsEmpty():
                    continue
                extent = box_range.GetSize()
                top_z = float(box_range.GetMax()[2])
            except Exception:
                continue
            # After a 90 degree yaw the asset's long axis lies along world Y, so take the
            # larger horizontal extent as the length whichever way it was placed.
            length, width = (max(extent[0], extent[1]), min(extent[0], extent[1]))
            surfaces.append((x, y, yaw_deg, top_z, float(length), float(width)))
        return surfaces

    def _create_package_from_prop(self, box_path: str, rng: random.Random, x: float, y: float, z: float) -> float:
        """Reference a warehouse crate/bin asset as one pickable package.

        The asset is referenced onto a *child* prim and the package's own pose is authored
        on the wrapper. Setting the pose directly on the reference would mean clearing its
        xform op order, and these crates carry a centimetre-to-metre scale op there: drop
        it and a 0.7 m crate renders 100x too big, which is exactly what happened the
        first time (70 m crates, and a grab radius to match).

        Returns:
            The package's largest horizontal extent in metres, used for grab radius and mass.
        """
        stage = omni.usd.get_context().get_stage()
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + rng.choice(self._package_usd_paths)

        wrapper = UsdGeom.Xform.Define(stage, box_path)
        wrapper.ClearXformOpOrder()  # safe: this prim is ours, it has no asset ops
        translate_op = wrapper.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(x, y, z))
        wrapper.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, rng.uniform(-180.0, 180.0)))

        stage_utils.add_reference_to_stage(usd_path=usd_path, path=f"{box_path}/Asset")
        prim = wrapper.GetPrim()

        # Drop it so its measured BASE rests on the surface. `z` is where the base should
        # be; the asset's origin is somewhere else entirely, and assuming otherwise spawns
        # the crate inside the table.
        self._rest_package_base_at(prim, translate_op, x, y, z)

        # Exactly one rigid body per package, and it has to be the wrapper: that is the
        # prim the grab system moves and object_states.csv records. Some of these assets
        # ship their own rigid body further down, and nesting them is invalid in PhysX.
        asset_prim = stage.GetPrimAtPath(f"{box_path}/Asset")
        if asset_prim.IsValid():
            for descendant in Usd.PrimRange(asset_prim):
                if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
                    descendant.RemoveAPI(UsdPhysics.RigidBodyAPI)
                if descendant.IsA(UsdGeom.Mesh):
                    # Pickable objects are dynamic; triangle-mesh collision is not
                    # supported for them. Author the approximation before play so
                    # PhysX does not replace collision shapes during initialization.
                    UsdPhysics.MeshCollisionAPI.Apply(descendant).CreateApproximationAttr().Set("convexHull")
            if not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(asset_prim)):
                # Put fallback colliders on geometry, alongside the mesh approximation
                # above. A collider on the Asset Xform instead cooks an aggregate
                # triangle mesh and ignores its children's convexHull settings.
                for descendant in Usd.PrimRange(asset_prim):
                    if descendant.IsA(UsdGeom.Gprim):
                        UsdPhysics.CollisionAPI.Apply(descendant)
        UsdPhysics.RigidBodyAPI.Apply(prim)

        return self._measure_prim_size(prim)

    def _rest_package_base_at(self, prim, translate_op, x: float, y: float, base_z: float) -> None:
        """Shift a referenced package so the bottom of its bounding box sits at ``base_z``.

        Args:
            prim: The package wrapper prim.
            translate_op: Its translate op, already set to ``(x, y, base_z)``.
            x, y: Horizontal position, unchanged.
            base_z: World height the package's underside should rest at.
        """
        try:
            bounds = (
                UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                .ComputeWorldBound(prim)
                .ComputeAlignedRange()
            )
            if bounds.IsEmpty():
                return
            measured_base = float(bounds.GetMin()[2])
            translate_op.Set(Gf.Vec3d(x, y, base_z + (base_z - measured_base)))
        except Exception as e:
            carb.log_warn(f"HumanoidExample: could not rest {prim.GetPath()} on its surface: {e}")

    def _create_package_cube(self, box_path: str, rng: random.Random, x: float, y: float, z: float) -> float:
        """Create one procedural package cube (the fallback when props are unavailable)."""
        stage = omni.usd.get_context().get_stage()
        size = rng.uniform(0.30, 0.55)
        cube = UsdGeom.Cube.Define(stage, box_path)
        cube.CreateSizeAttr(size)
        cube.CreateDisplayColorAttr().Set(
            [Gf.Vec3f(rng.uniform(0.35, 0.75), rng.uniform(0.25, 0.55), rng.uniform(0.15, 0.35))]
        )
        cube.ClearXformOpOrder()
        cube.AddTranslateOp().Set(Gf.Vec3d(x, y, z + size * 0.5))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        return size

    def _measure_prim_size(self, prim) -> float:
        """Return the largest horizontal extent of a prim's bounding box, in metres."""
        try:
            bounds = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
            ).ComputeWorldBound(prim)
            box_range = bounds.ComputeAlignedRange()
            if box_range.IsEmpty():
                return 0.4
            extent = box_range.GetSize()
            return float(max(extent[0], extent[1]))
        except Exception:
            return 0.4

    def _create_arm_control_rig(self) -> None:
        """Create visible controller target markers and hand meshes used as the G1 arm-control rig."""
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

        carb.log_info(f"HumanoidExample: created G1 arm-control rig under {self._arm_rig_root_path}")

    def _ensure_scene_lighting(self) -> None:
        """Add fill lighting, scaled to how much the environment already provides.

        The grid is an empty stage that needs everything. The warehouse ships its own
        interior lighting, so the same values would blow the exposure out; it gets a
        gentle fill only, enough to keep the robot and packages readable.
        """
        stage = omni.usd.get_context().get_stage()
        dome_intensity = 200.0 if self._environment_is_warehouse else 1200.0
        distant_intensity = 400.0 if self._environment_is_warehouse else 2500.0

        dome_path = "/World/G1_VR_DomeLight"
        if not stage.GetPrimAtPath(dome_path).IsValid():
            dome = UsdLux.DomeLight.Define(stage, dome_path)
            dome.CreateIntensityAttr().Set(dome_intensity)
            dome.CreateExposureAttr().Set(0.0)

        distant_path = "/World/G1_VR_DistantLight"
        if not stage.GetPrimAtPath(distant_path).IsValid():
            distant = UsdLux.DistantLight.Define(stage, distant_path)
            distant.CreateIntensityAttr().Set(distant_intensity)
            distant.CreateAngleAttr().Set(0.5)
            distant_xform = UsdGeom.Xformable(distant.GetPrim())
            distant_xform.ClearXformOpOrder()
            distant_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))

    def setup_scene(self):
        """Set up the scene with robot and environment."""
        # Snapshot prior physics device/fabric state so cleanup can restore it.
        self._prev_physics_sim_device, self._prev_fabric_enabled = snapshot_physics_simulation_state()

        # Set device and backend BEFORE creating robot so it uses GPU
        SimulationManager.set_backend(self._world_settings["backend"])
        SimulationManager.set_physics_sim_device(self._world_settings["device"])
        SimulationManager.get_available_physics_engines(verbose=True)

        self._create_environment()
        self._ensure_scene_lighting()

        # Friction the walking policy can push against.
        self._apply_ground_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        self._create_work_surfaces()  # before the packages: they measure its top height
        self._create_sample_boxes()
        # Material-binding schemas and override prims must exist before physics
        # builds tensor views. Runtime highlighting changes only relationships.
        self._material_highlights.prepare(
            omni.usd.get_context().get_stage(),
            [f"{self._sample_box_root_path}/Box_{index:02d}" for index in range(self._sample_box_count)],
        )

        # Create the Unitree G1 with dexterous hands. Isaac Sim ships no G1 locomotion
        # policy, so walking comes from Unitree's own pretrained one, which drives the
        # legs and leaves the arms to the operator (see G1TeleopRobot).
        spawn_position = [self._robot_spawn_xy[0], self._robot_spawn_xy[1], self._g1_spawn_position[2]]
        self.g1 = G1TeleopRobot(
            prim_path=self._g1_prim_path,
            position=spawn_position,
            hand_variant=self._g1_hand_variant,
            locomotion=self._g1_locomotion,
        )
        self._create_arm_control_rig()
        self._create_head_camera()
        self._create_xr_anchor()

    async def setup_post_load(self):
        """Setup keyboard input and physics callback after initial load."""
        self._unsubscribe_keyboard()
        self._reset_teleoperation_state()
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
                    self._xr_core, draw_ray=self._eye_gaze_ray_visual_enabled, highlights=self._material_highlights
                )
            except Exception as e:
                carb.log_warn(f"HumanoidExample: Quest Pro eye-gaze tracker unavailable: {e}")
        self._xr_input_status_logged = False
        self._locomotion_input_logged = False
        self._locomotion_brake = False
        self._keyboard_brake = False
        self._xr_calibration_samples = []
        self._xr_calibrated = False
        self._xr_calibration_head_offset = None
        self._xr_head_device_handle = None
        self._xr_last_head_pose = None
        self._xr_head_pose_age = 0
        self._xr_convention = None
        self._xr_convention_index = 0
        self._xr_convention_errors = {}
        self._xr_convention_pending = None
        self._xr_diagnostic_countdown = 400
        self._xr_diagnostic_done = False
        self._xr_recenter_button_down = False
        self._drop_button_down = False
        self._grab_time = 0.0
        self._grab_miss_report_time = {}
        self._hand_tracking_status_logged = False
        self._g1_arm_dofs_configured = False
        self._grabbed_objects_by_side = {}
        self._controller_arm_neutral_positions = {}
        self._smoothed_arm_rig_targets = {}
        self._active_g1_hand_target_matrices = {}
        self._hand_link_prims = {}
        self._smoothed_finger_curls = {}
        self._latest_finger_curls = {}
        self._finger_curl_source = {}
        self._finger_tracking_status_logged = False
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
        self._xr_camera_states_logged = set()  # re-report camera state each load
        self._head_camera_last_base = None
        self._head_camera_last_yaw = None
        self._camera_filtered_base = None
        self._camera_filtered_yaw = None
        self._start_behavioral_session()

        torch = import_module("torch")
        device = self._world_settings["device"]
        self._base_command = torch.zeros(3, device=device)
        self._keyboard_command = torch.zeros(3, device=device)
        self._controller_command = torch.zeros(3, device=device)
        self._physics_ready = False
        self._set_active_head_camera()
        # Keep the XR view attached between physics steps and while paused. The
        # camera callback does not read or modify eye-gaze devices or their settings.
        self._head_camera_update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_robot_camera_update, name="G1 robot-mounted camera")
        )

        # Register physics callback using SimulationManager
        if self._physics_callback_id is None:
            self._physics_callback_id = SimulationManager.register_callback(
                self.on_physics_step, IsaacEvents.POST_PHYSICS_STEP
            )

    async def setup_pre_reset(self):
        """Called before world reset."""
        # Remove live constraints before physics destroys their body handles.
        self._reset_teleoperation_state()
        if self._eye_gaze_tracker is not None:
            self._eye_gaze_tracker.cleanup()
        self._physics_ready = False

    async def setup_post_reset(self):
        """Called after world reset."""
        # Reset physics ready flag after reset so robot reinitializes on next play
        self._physics_ready = False

    async def setup_post_clear(self):
        """Called after clearing the scene."""
        self._head_camera_update_sub = None
        self._reset_teleoperation_state()
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
        self.g1 = None
        self._physics_ready = False
        self._head_camera_transform_op = None  # handles die with the stage; never reuse them
        self._head_camera_mount_body_path = None
        self._head_camera_mount_local = None
        self._xr_anchor_op = None
        self._restore_physics_simulation_state()

    def on_physics_step(self, dt: float, context: object) -> None:
        """Physics step callback - initialize on first step, then drive the robot.

        Args:
            dt: Delta time for the physics step.
            context: Physics step context.
        """
        if not self.g1:
            return
        if not math.isfinite(dt) or dt <= 0.0:
            return

        # Check if physics tensors are valid, if not, reinitialize
        if not self.g1.robot.is_physics_tensor_entity_valid():
            self._physics_ready = False

        if self._physics_ready:
            # Robot is initialized: advance the base and hold the posture, then let the
            # teleoperation layer below override the arm and finger DOFs it owns.
            self._last_physics_dt = float(dt)
            self._update_teleop_input_clock(float(dt))
            self._grab_time += float(dt)
            self._update_controller_command(dt)
            target_command = self._keyboard_command + self._controller_command
            if self._locomotion_brake:
                # Translation only; the yaw the operator asked for survives so they can
                # still turn on the spot while braked.
                target_command[0] = target_command[0] * 0.0
                target_command[1] = target_command[1] * 0.0
            # Asymmetric on purpose: reverse is capped lower than forward.
            target_command[0] = target_command[0].clamp(-self._max_backward_speed, self._max_forward_speed)
            target_command[1] = target_command[1].clamp(-self._max_lateral_speed, self._max_lateral_speed)
            target_command[2] = target_command[2].clamp(-self._max_yaw_speed, self._max_yaw_speed)
            self._smooth_base_command(target_command, dt)
            self.g1.forward(dt, self._base_command)
            # Stage edits (undo, prim deletion, clears) can invalidate prims any of
            # these subsystems hold handles to; isolate each one so a single failure
            # cannot abort the step and silently stop behavioral data collection.
            # Grab decisions consume this tick's finger sample, never the previous one.
            try:
                self._update_g1_fingers()
            except Exception as e:
                self._log_physics_step_error("finger teleoperation", e)
            try:
                self._update_g1_arms_from_hand_tracking()
            except Exception as e:
                self._log_physics_step_error("arm teleoperation", e)
                for side in ("left", "right"):
                    self._deactivate_hand(side)
            try:
                self._update_head_camera_view(dt=dt)
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
            self._reset_teleoperation_state()
            self.g1.initialize()  # This already sets default state internally
            if not self.g1._initialized:
                return
            self.g1.post_reset()
            self._update_recorded_locomotion_mode()
            self._configure_g1_arm_dofs()
            self._update_head_camera_view(force=True, dt=dt)
            self._physics_ready = True

    def _reset_teleoperation_state(self) -> None:
        """Release constraints and discard input, calibration, and cached physics handles."""
        for side in ("left", "right"):
            self._release_grabbed_object(side)
            self._set_grab_candidate(side, None)
        for command in (self._base_command, self._keyboard_command, self._controller_command):
            if command is not None:
                command[:] = 0.0
        self._pressed_keys.clear()
        self._keyboard_brake = False
        self._locomotion_brake = False
        self._latest_stick_lateral = 0.0
        for mapping in (
            self._hand_link_prims,
            self._grasp_link_prims,
            self._object_rigid_prims,
            self._arm_ik_link_index,
            self._controller_arm_neutral_positions,
            self._controller_arm_neutral_targets,
            self._arm_orientation_anchors,
            self._robot_palm_local_frames,
            self._robot_palm_local_centers,
            self._arm_input_sources,
            self._smoothed_arm_targets,
            self._smoothed_arm_rig_targets,
            self._active_g1_hand_target_matrices,
            self._smoothed_finger_curls,
            self._latest_finger_curls,
            self._finger_curl_source,
            self._hand_closed_by_side,
            self._grab_requires_release,
        ):
            mapping.clear()
        self._g1_arm_dofs_configured = False
        self._last_teleop_wall_time = None
        self._teleop_input_dt = self._last_physics_dt
        self._g1_arm_dof_indices_by_side.clear()
        self._g1_arm_joint_names_by_side.clear()
        self._g1_arm_joint_limits.clear()
        self._object_state_prev_positions.clear()
        self._camera_filtered_base = None
        self._camera_filtered_yaw = None
        self._head_camera_mount_prim = None
        self._xr_recenter_button_down = False
        self._xr_mode_button_down = False
        self._drop_button_down = False
        self._last_headset_raw_position = None
        self._last_headset_pose_matrix = None
        self._xr_last_head_pose = None
        self._xr_head_pose_age = 0
        self._reset_headset_gait_state(reset_clock=False)

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
        self._prepare_robot_head_camera_mount(stage)
        carb.log_info(f"HumanoidExample: created G1 head camera at {self._head_camera_path}")

    def _prepare_robot_head_camera_mount(self, stage: Usd.Stage) -> None:
        """Record a fixed camera-to-body transform before the physics scene starts.

        G1's visual head is a fixed child of the torso, not a separate articulation
        link in the Inspire asset. Find its rigid ancestor and follow that body's
        live physics pose. The existing camera path stays stable for recording.
        """
        self._head_camera_mount_prim = None
        self._head_camera_mount_local = None
        self._head_camera_mount_body_path = None
        for path in (
            f"{self._g1_prim_path}/torso_link/head_link",
            f"{self._g1_prim_path}/head_link",
            f"{self._g1_prim_path}/torso_link",
        ):
            body = stage.GetPrimAtPath(path)
            while body.IsValid() and str(body.GetPath()).startswith(self._g1_prim_path + "/"):
                if body.HasAPI(UsdPhysics.RigidBodyAPI):
                    break
                body = body.GetParent()
            if body.IsValid() and body.HasAPI(UsdPhysics.RigidBodyAPI):
                self._head_camera_mount_body_path = str(body.GetPath())
                break
        if self._head_camera_mount_body_path is None:
            carb.log_warn("HumanoidExample: no rigid head/torso body for the robot camera")
            return
        cache = UsdGeom.XformCache()
        root_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(self._g1_prim_path))
        # Preserve the established eye position and viewing direction at spawn, then
        # keep this mount rigid through all six axes of the robot body's motion.
        eye = root_world.Transform(
            Gf.Vec3d(
                self._first_person_head_forward_offset,
                0.0,
                self._first_person_eye_height_above_base + self._first_person_head_up_offset,
            )
        )
        direction = root_world.TransformDir(
            Gf.Vec3d(self._first_person_head_target_distance, 0.0, -self._first_person_head_target_drop)
        )
        up = root_world.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
        camera_world = Gf.Matrix4d().SetLookAt(eye, eye + direction, up).GetInverse()
        body_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(self._head_camera_mount_body_path))
        self._head_camera_mount_local = camera_world * body_world.GetInverse()

    def _read_camera_mount_body_pose(self) -> Gf.Matrix4d | None:
        """Read the real head/torso body, including when Fabric leaves USD stale."""
        if self._head_camera_mount_body_path is None:
            return None
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        body = stage.GetPrimAtPath(self._head_camera_mount_body_path)
        if not body.IsValid():
            return None
        if not self.g1 or not self.g1.robot.is_physics_tensor_entity_valid():
            # Before Play/after Stop there is no live tensor pose; use the authored
            # mount. Never substitute authored transforms for a running rigid body.
            return UsdGeom.XformCache().GetLocalToWorldTransform(body)
        try:
            if self._head_camera_mount_prim is None:
                from isaacsim.core.experimental.prims import RigidPrim

                self._head_camera_mount_prim = RigidPrim(paths=self._head_camera_mount_body_path)
            positions, orientations = self._head_camera_mount_prim.get_world_poses()
            position = self._first_pose_value(positions)
            quaternion = self._first_pose_value(orientations)
            if position is None or quaternion is None:
                return None
            values = [float(v) for v in position[:3]] + [float(v) for v in quaternion[:4]]
            if len(values) != 7 or not all(math.isfinite(v) for v in values):
                return None
            rotation = Gf.Quatd(values[3], Gf.Vec3d(*values[4:]))
            if rotation.GetLength() < 1e-8:
                return None
            matrix = Gf.Matrix4d().SetRotate(rotation.GetNormalized())
            matrix.SetTranslateOnly(Gf.Vec3d(*values[:3]))
            return matrix
        except Exception as error:
            self._head_camera_mount_prim = None
            self._log_physics_step_error("robot camera mount", error)
            return None

    def _get_robot_head_camera_pose(self) -> Gf.Matrix4d | None:
        """Return the rigid robot-mounted view without consuming a headset pose."""
        if self._head_camera_mount_local is None:
            return None
        body_world = self._read_camera_mount_body_pose()
        return self._head_camera_mount_local * body_world if body_world is not None else None

    def _on_robot_camera_update(self, event: object) -> None:
        """Reassert the mounted XR view every application frame, including Pause."""
        if self._g1_locomotion != "stationary" and self._xr_camera_mode != "robot_head":
            return
        if omni.usd.get_context().get_stage() is None:
            return  # A stage close can precede the example's cleanup callback.
        try:
            self._update_head_camera_view(force=True, dt=0.0)
        except Exception as error:
            self._log_physics_step_error("robot camera frame update", error)

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
        """Switch the VR profile to custom-anchor mode, pointed at the G1 anchor prim.

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
            self._log_xr_camera_state("no robot base pose yet (physics not running?)")
            return
        head_pose = self._read_physical_head_pose()
        if head_pose is None:
            # Without the headset pose the compose trick cannot work, but the view can
            # still be locked to the robot -- worse (your head rotation stops working)
            # yet far better than a camera left at the world origin.
            self._log_xr_camera_state(
                "no readable head pose (tried "
                f"{self._xr_head_device_handle or '/user/head'} + a scan of every XR device); "
                "falling back to camera-lock so the view at least rides the robot. "
                "Is the VR session actually started?"
            )
            self._schedule_locked_xr_camera()
            return

        base = self._head_camera_last_base
        yaw = self._head_camera_last_yaw
        self._update_xr_calibration(head_pose, base)
        yaw_deg = math.degrees(yaw) + self._xr_anchor_yaw_offset_deg
        forward = Gf.Vec3d(math.cos(yaw), math.sin(yaw), 0.0)
        rotation = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw_deg))

        robot_eye_z = float(base[2]) + self._first_person_eye_height_above_base + self._first_person_head_up_offset
        if self._xr_calibration_head_offset is not None:
            # Place the anchor so the operator's CALIBRATED head pose maps exactly onto
            # the robot's eye. The offset is rotated by the robot's live yaw, so they stay
            # locked inside it as it turns, and any movement away from the calibrated spot
            # still moves them relative to the robot -- room-scale is preserved, only the
            # constant displacement is removed.
            calibrated = rotation.TransformDir(self._xr_calibration_head_offset)
            position = (
                Gf.Vec3d(float(base[0]), float(base[1]), robot_eye_z)
                + forward * self._xr_anchor_forward_offset
                - Gf.Vec3d(calibrated[0], calibrated[1], calibrated[2])
            )
        else:
            # Not calibrated yet (first couple of seconds): height only, as before.
            position = (
                Gf.Vec3d(float(base[0]), float(base[1]), self._xr_anchor_height_offset)
                + forward * self._xr_anchor_forward_offset
            )

        anchor_m = rotation * Gf.Matrix4d(1.0).SetTranslate(position)
        # Physical (OpenXR) space is Y-up; the stage is Z-up. Rotate +90 deg about X so the
        # head pose composes correctly under the Z-up anchor.
        yup_to_zup = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 90.0))
        rig = yup_to_zup * anchor_m  # the rig origin this frame, in stage space

        # Where the operator's eye SHOULD end up: the robot's eye.
        desired_eye = Gf.Vec3d(float(base[0]), float(base[1]), robot_eye_z)
        self._score_xr_convention(desired_eye)

        candidate = self._current_xr_convention()
        matrix = (head_pose * rig) if candidate == "compose" else rig
        try:
            self._xr_core.schedule_set_camera(matrix)
        except Exception as e:
            self._log_xr_camera_state(f"schedule_set_camera failed: {e}; using camera-lock")
            self._schedule_locked_xr_camera()
            return
        self._xr_convention_pending = (candidate, desired_eye)
        self._dump_xr_diagnostics_once(head_pose, rig, matrix, desired_eye, candidate)
        if self._xr_convention is not None:
            self._log_xr_camera_state(
                f"attached (head_compose/{self._xr_convention}: rides the robot, head tracking live)"
            )

    def _dump_xr_diagnostics_once(self, head_pose, rig_matrix, scheduled, desired_eye, candidate) -> None:
        """Print the complete XR rig state once, a few seconds in.

        Everything needed to derive the correct camera formula from data: the physical head
        pose, the rig transform the runtime actually applied, where that puts the head, what
        we scheduled, and where the robot's eye is. One run of this settles what three
        rounds of reasoning about `schedule_set_camera` could not.
        """
        if self._xr_diagnostic_done:
            return
        self._xr_diagnostic_countdown -= 1
        if self._xr_diagnostic_countdown > 0:
            return
        self._xr_diagnostic_done = True

        def describe(label, matrix):
            if matrix is None:
                print(f"[G1-XR]   {label:22s} None", flush=True)
                return
            t = matrix.ExtractTranslation()
            f = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
            print(
                f"[G1-XR]   {label:22s} pos=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})  "
                f"fwd=({f[0]:+.2f}, {f[1]:+.2f}, {f[2]:+.2f})",
                flush=True,
            )

        print("[G1-XR] ---- one-shot XR rig state ----", flush=True)
        print(f"[G1-XR]   candidate in use: {candidate}", flush=True)
        describe("physical head", head_pose)
        describe("rig (phys->stage)", self._read_rig_transform())
        describe("scheduled matrix", scheduled)
        if head_pose is not None:
            live_rig = self._read_rig_transform()
            if live_rig is not None:
                describe("head * rig", head_pose * live_rig)
        print(
            f"[G1-XR]   robot eye should be: ({desired_eye[0]:+.3f}, {desired_eye[1]:+.3f}, " f"{desired_eye[2]:+.3f})",
            flush=True,
        )
        virtual = self._read_virtual_head_position()
        if virtual is not None:
            error = float((virtual - desired_eye).GetLength())
            print(
                f"[G1-XR]   head actually at:    ({virtual[0]:+.3f}, {virtual[1]:+.3f}, "
                f"{virtual[2]:+.3f})  -> {error:.3f} m from the robot's eye",
                flush=True,
            )
        print(
            f"[G1-XR]   anchor yaw offset {self._xr_anchor_yaw_offset_deg:+.1f} deg, "
            f"calibrated={self._xr_calibrated}",
            flush=True,
        )
        print("[G1-XR] ---- end ----", flush=True)

    def _current_xr_convention(self) -> str:
        """The convention in use, or the one currently being trialled."""
        if self._xr_convention is not None:
            return self._xr_convention
        if not self._xr_convention_autodetect:
            return "compose"
        candidates = self._xr_convention_candidates
        return candidates[min(self._xr_convention_index, len(candidates) - 1)]

    def _read_rig_transform(self):
        """The runtime's own physical-to-stage rig transform, or None."""
        try:
            return Gf.Matrix4d(self._xr_core.get_physical_to_virtual_world_transform())
        except Exception:
            return None

    def _read_virtual_head_position(self):
        """Where the operator's head actually lands in the stage.

        Derived from the rig transform rather than read from the head device's
        virtual-world pose: that pose demonstrably does not track what we schedule (see
        the module notes), so scoring against it was measuring nothing.
        """
        rig = self._read_rig_transform()
        head_pose = self._read_physical_head_pose()
        if rig is not None and head_pose is not None:
            return Gf.Vec3d((head_pose * rig).ExtractTranslation())

        headset = self._find_xr_head_device()
        reader = getattr(headset, "get_virtual_world_pose", None) if headset else None
        if reader is None:
            return None
        for pose_name in ("", "head"):
            try:
                pose = Gf.Matrix4d(reader(pose_name) if pose_name else reader())
            except Exception:
                continue
            if pose == Gf.Matrix4d(1.0):
                continue
            return Gf.Vec3d(pose.ExtractTranslation())
        return None

    def _score_xr_convention(self, desired_eye: Gf.Vec3d) -> None:
        """Grade the candidate scheduled last frame by how close it put the head to the robot.

        The metric is the operator's complaint, stated numerically: distance from where the
        runtime thinks their head is to where the robot's eye is. A correct convention keeps
        that within head-movement range; a wrong one leaves it a whole head-pose away.
        """
        if self._xr_convention is not None or not self._xr_convention_autodetect:
            return
        pending = self._xr_convention_pending
        self._xr_convention_pending = None
        if pending is None:
            return
        name, expected_eye = pending
        actual = self._read_virtual_head_position()
        if actual is None:
            # No readback on this runtime: nothing can be measured, so keep the
            # theoretically-derived candidate rather than guessing from noise.
            self._xr_convention = self._xr_convention_candidates[0]
            self._log_xr_camera_state(
                "no virtual-world head pose to measure against; keeping the " f"{self._xr_convention} convention"
            )
            return

        error = float((actual - expected_eye).GetLength())
        samples = self._xr_convention_errors.setdefault(name, [])
        samples.append(error)
        if len(samples) < self._xr_convention_frames:
            return

        self._xr_convention_index += 1
        if self._xr_convention_index < len(self._xr_convention_candidates):
            return  # try the next candidate

        medians = {
            key: sorted(values)[len(values) // 2] for key, values in self._xr_convention_errors.items() if values
        }
        if not medians:
            self._xr_convention = self._xr_convention_candidates[0]
            return
        self._xr_convention = min(medians, key=medians.get)
        detail = ", ".join(f"{key} {value:.2f} m" for key, value in sorted(medians.items()))
        print(
            f"[G1] VR camera convention measured: using '{self._xr_convention}' "
            f"(head-to-robot-eye distance -- {detail})",
            flush=True,
        )
        if medians[self._xr_convention] > 1.0:
            print(
                "[G1] WARNING: even the best option leaves you "
                f"{medians[self._xr_convention]:.1f} m from the robot's head. Click the LEFT "
                "thumbstick to try the other camera modes.",
                flush=True,
            )

    # ------------------------------------------------------------ XR head pose

    def _find_xr_head_device(self):
        """Resolve the headset input device, remembering which handle answered.

        ``/user/head`` is the handle this Kit build hardcodes, but a single failed lookup
        used to disable the whole first-person rig with no way to tell whether the handle
        was wrong or the VR session simply was not running. Falling back to a scan of the
        live device list distinguishes the two and survives a runtime that names the
        headset something else.
        """
        if self._xr_core is None:
            return None
        if self._xr_head_device_handle is not None:
            device = self._get_xr_input_device(self._xr_head_device_handle)
            if device is not None:
                return device
            self._xr_head_device_handle = None  # went away; re-resolve below

        device = self._get_xr_input_device("/user/head")
        if device is not None:
            self._xr_head_device_handle = "/user/head"
            return device

        try:
            devices = self._xr_core.get_all_input_devices()
        except Exception:
            return None
        for candidate in devices or ():
            try:
                name = str(candidate.get_name())
            except Exception:
                continue
            if "head" in name.lower() or "hmd" in name.lower():
                self._xr_head_device_handle = name
                self._log_xr_camera_state(f"headset resolved by device scan as {name}")
                return candidate
        return None

    def _read_physical_head_pose(self):
        """Return the headset pose in PHYSICAL (playspace) coordinates, or None.

        Physical, not virtual-world: the compose path re-injects this pose so the runtime's
        own head subtraction cancels it. A virtual-world pose already has the rig transform
        folded in and would make that cancellation feed back on itself.
        """
        headset = self._find_xr_head_device()
        pose = None
        if headset is not None:
            for reader_name in ("get_pose", "get_raw_pose"):
                reader = getattr(headset, reader_name, None)
                if reader is None:
                    continue
                try:
                    candidate = Gf.Matrix4d(reader())
                except Exception:
                    continue
                if candidate == Gf.Matrix4d(1.0):
                    continue  # identity = not tracked this frame
                pose = candidate
                break

        if pose is not None:
            self._xr_last_head_pose = pose
            self._xr_head_pose_age = 0
            return pose

        # Ride out a short dropout on the last good pose rather than detaching the rig.
        if self._xr_last_head_pose is not None:
            self._xr_head_pose_age += 1
            if self._xr_head_pose_age <= self._xr_head_pose_max_age:
                return self._xr_last_head_pose
            self._xr_last_head_pose = None
        return None

    @staticmethod
    def _physical_head_frame(head_pose: Gf.Matrix4d) -> tuple[Gf.Vec3d, float, float]:
        """Split a Y-up OpenXR head pose into (offset in the Z-up stage frame, height, yaw).

        The offset is the head's full playspace translation mapped through the same
        Y-up -> Z-up rotation the rig uses, so it can be subtracted from the anchor
        directly. All three components matter: the height decides whether the operator
        floats above or sinks below the robot, and the two horizontal components decide
        whether they stand INSIDE it or several metres to one side.

        The yaw is measured so a user facing physical -Z (the OpenXR "forward" a playspace
        is set up around) reads 0. It is the term that has to come back out of the anchor
        rotation for "straight ahead in the headset" to mean "the way the robot walks",
        whichever way the user happens to be standing when the session starts.
        """
        translation = head_pose.ExtractTranslation()
        height = float(translation[1])  # Y is up in OpenXR physical space
        # RotX(+90) in Gf's row-vector convention maps (x, y, z) -> (x, -z, y).
        offset = Gf.Vec3d(float(translation[0]), -float(translation[2]), height)
        forward = head_pose.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))  # OpenXR looks down -Z
        # After the Y-up -> Z-up fix, (fx, fz) maps to the stage-plane direction (fx, -fz);
        # expressed as a rotation off stage +Y that is atan2(-fx, -fz).
        yaw = math.atan2(-float(forward[0]), -float(forward[2]))
        return offset, height, yaw

    def _update_xr_calibration(self, head_pose: Gf.Matrix4d, robot_base: Gf.Vec3d) -> None:
        """Measure the rig height and yaw offsets from the headset, once, then latch them.

        Both offsets used to be hand-tuned constants, and both are wrong for anyone whose
        height or starting orientation differs from whoever tuned them -- which presents as
        the camera being "not attached": eyes sunk into the floor or floating above the
        robot, facing away from the direction it walks.

        The height term is exact: the rig origin is the playspace floor, the user's eyes sit
        ``head height`` above it, so putting their eyes at the robot's eye level means
        ``anchor_z = robot_eye_z - head_height``. The yaw term is the axis constant minus
        however far the user happens to be turned right now.
        """
        if not self._xr_auto_calibrate or self._xr_calibrated:
            return
        offset, height, user_yaw = self._physical_head_frame(head_pose)
        if not (self._xr_min_head_height <= height <= self._xr_max_head_height):
            return  # not a plausible standing/seated head pose; wait for a real one

        window = max(10, int(self._xr_calibration_window / max(self._last_physics_dt, 1e-3)))
        self._xr_calibration_samples.append((height, user_yaw, offset))
        if len(self._xr_calibration_samples) > window:
            self._xr_calibration_samples.pop(0)
        if len(self._xr_calibration_samples) < window:
            return

        heights = [sample[0] for sample in self._xr_calibration_samples]
        spread = max(heights) - min(heights)
        if spread > self._xr_calibration_max_spread:
            # Still moving -- putting the headset on, standing up, walking to the desk.
            # Keep sliding the window rather than freezing a value from mid-motion.
            return

        # Median, not mean: one bad frame cannot drag it.
        mean_height = sorted(heights)[len(heights) // 2]
        count = len(self._xr_calibration_samples)
        mean_x = sum(math.cos(sample[1]) for sample in self._xr_calibration_samples) / count
        mean_y = sum(math.sin(sample[1]) for sample in self._xr_calibration_samples) / count
        mean_yaw = math.atan2(mean_y, mean_x)

        robot_eye_z = (
            float(robot_base[2]) + self._first_person_eye_height_above_base + self._first_person_head_up_offset
        )
        count_half = len(self._xr_calibration_samples) // 2
        self._xr_calibration_head_offset = Gf.Vec3d(
            sorted(float(sample[2][0]) for sample in self._xr_calibration_samples)[count_half],
            sorted(float(sample[2][1]) for sample in self._xr_calibration_samples)[count_half],
            mean_height,
        )
        self._xr_anchor_height_offset = robot_eye_z - mean_height
        self._xr_anchor_yaw_offset_deg = self._xr_yaw_base_offset_deg - math.degrees(mean_yaw)
        self._xr_calibrated = True
        self._xr_calibration_samples = []
        print(
            f"[G1] VR rig calibrated: your eyes are {mean_height:.2f} m above your floor "
            f"(steady to {spread * 100:.0f} cm over {self._xr_calibration_window:.0f} s), "
            f"the robot's are at {robot_eye_z:.2f} m -> rig floor "
            f"{self._xr_anchor_height_offset:+.2f} m, yaw offset "
            f"{self._xr_anchor_yaw_offset_deg:+.1f} deg, standing "
            f"{math.hypot(self._xr_calibration_head_offset[0], self._xr_calibration_head_offset[1]):.2f} m "
            "from your room origin (that offset is now removed).",
            flush=True,
        )
        if mean_height < 1.35:
            print(
                f"[G1] NOTE: {mean_height:.2f} m is low for a standing adult. If you were "
                "seated or still putting the headset on, stand up facing the way you want "
                "to walk and press B to redo this.",
                flush=True,
            )
        else:
            print("[G1] Face the way you want to walk and press B to redo this.", flush=True)

    def _request_xr_recenter(self) -> None:
        """Restore the fixed view, or recalibrate a legacy moving mode, on B."""
        if self._g1_locomotion == "stationary" or self._xr_camera_mode == "robot_head":
            self._update_head_camera_view(force=True, dt=0.0)
            self._log_xr_camera_state("view restored to the fixed robot head mount")
            return
        self._xr_calibrated = False
        self._xr_calibration_samples = []
        print("[G1] recentering the VR view on your current head pose and facing...", flush=True)

    def describe_xr(self) -> str:
        """One-line XR status for the live Kit console. See tools/kit_exec.py."""
        head_pose = self._read_physical_head_pose()
        if head_pose is None:
            head = "none"
        else:
            offset, height, user_yaw = self._physical_head_frame(head_pose)
            head = (
                f"height={height:.2f}m yaw={math.degrees(user_yaw):+.1f}deg "
                f"horiz={math.hypot(offset[0], offset[1]):.2f}m"
            )
        try:
            devices = ", ".join(str(d.get_name()) for d in self._xr_core.get_all_input_devices())
        except Exception as e:
            devices = f"<unavailable: {e}>"
        return (
            f"mode={self._xr_camera_mode} convention={self._xr_convention} "
            f"handle={self._xr_head_device_handle} head={head} "
            f"calibrated={self._xr_calibrated} anchor_z={self._xr_anchor_height_offset:+.3f} "
            f"yaw_off={self._xr_anchor_yaw_offset_deg:+.1f} "
            f"base={self._head_camera_last_base} devices=[{devices}]"
        )

    def _schedule_stage_anchor_xr_camera(self) -> None:
        """Attach the VR rig to the anchor prim through XRCore's own anchoring API.

        An alternative to head_compose that asks the runtime to treat the anchor Xform as
        the physical space origin, rather than re-deriving the view matrix ourselves every
        frame. Smoother when it works -- nothing fights the runtime's reprojection -- but it
        depends on the build re-reading a MOVING anchor prim, which an earlier Kit did not.
        """
        if self._head_camera_last_base is None or self._head_camera_last_yaw is None:
            self._log_xr_camera_state("no robot base pose yet (physics not running?)")
            return
        self._update_xr_anchor(self._head_camera_last_base, self._head_camera_last_yaw)
        try:
            current = str(self._xr_core.get_stage_anchor_prim_path())
        except Exception:
            current = None
        if current != self._xr_anchor_path:
            try:
                self._xr_core.schedule_set_stage_anchor(self._xr_anchor_path)
                self._log_xr_camera_state(f"attached (stage_anchor: rig anchored to {self._xr_anchor_path})")
            except Exception as e:
                self._log_xr_camera_state(f"schedule_set_stage_anchor failed: {e}; using camera-lock")
                self._schedule_locked_xr_camera()

    def _schedule_locked_xr_camera(self) -> None:
        """Put the VR view on the robot's eye pose directly.

        The blunt fallback: it follows the robot but cancels your own head rotation,
        because the runtime subtracts the live head pose from whatever is scheduled.
        Uncomfortable to wear for long, but it proves the rig is attached and is far
        better than a view stranded at the world origin.
        """
        try:
            pose = self._get_head_camera_pose()
            if pose is not None:
                self._xr_core.schedule_set_camera(pose)
        except Exception as e:
            self._log_xr_camera_state(f"camera-lock fallback also failed: {e}")

    def _log_xr_camera_state(self, message: str) -> None:
        """Report the XR camera state once per distinct message.

        This path used to fail at four different points with a bare ``return``, so a
        detached VR view looked identical to a working one that simply had not moved.
        """
        if message in self._xr_camera_states_logged:
            return
        self._xr_camera_states_logged.add(message)
        print(f"[G1] XR camera: {message}", flush=True)
        carb.log_info(f"HumanoidExample XR camera: {message}")

    def _set_active_head_camera(self) -> None:
        """Switch the active viewport to the G1 head camera when the viewport API is present."""
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = Sdf.Path(self._head_camera_path)
                carb.log_info(f"HumanoidExample: active viewport camera set to {self._head_camera_path}")
        except Exception as e:
            carb.log_warn(f"HumanoidExample: could not set active viewport camera: {e}")

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

    def _get_head_camera_pose(self, dt: float | None = None):
        """Compute a first-person camera pose from the G1 head/eye position."""
        if self._g1_locomotion == "stationary" or self._xr_camera_mode == "robot_head":
            return self._get_robot_head_camera_pose()
        if not self.g1 or not self.g1.robot.is_physics_tensor_entity_valid():
            return None

        positions, orientations = self.g1.robot.get_world_poses()
        position = self._first_pose_value(positions)
        orientation = self._first_pose_value(orientations)
        if position is None or orientation is None or len(position) < 3 or len(orientation) < 4:
            return None

        qw, qx, qy, qz = (float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3]))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        yaw *= self._head_camera_yaw_sign
        base = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))

        # Damp the gait oscillation out before anything downstream sees it.
        if dt is None:
            dt = float(self._world_settings.get("physics_dt", 1.0 / 200.0))
        base, yaw = self._stabilize_camera_pose(base, yaw, dt)

        forward = Gf.Vec3d(math.cos(yaw), math.sin(yaw), 0.0)
        # Stash for the XR anchor update, which needs the raw base/yaw rather
        # than the finished camera matrix. These are the *stabilized* values, so the VR
        # rig rides the same steady frame the desktop camera does.
        self._head_camera_last_base = base
        self._head_camera_last_yaw = yaw

        # The whole pose is derived from the articulation root read through the physics
        # tensor API. The G1 does expose a head_link, but reading its transform through
        # UsdGeom.XformCache would return the authored pose, not the simulated one: this
        # example runs on the GPU pipeline with fabric enabled, which keeps live
        # transforms out of USD. The base is exact and the posture is held, so
        # base + eye offset is both correct and cheaper.
        eye = Gf.Vec3d(
            float(base[0]),
            float(base[1]),
            base[2] + self._first_person_eye_height_above_base + self._first_person_head_up_offset,
        )
        eye += forward * self._first_person_head_forward_offset
        target = (
            eye
            + forward * self._first_person_head_target_distance
            - Gf.Vec3d(0.0, 0.0, self._first_person_head_target_drop)
        )
        return Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()

    def _stabilize_camera_pose(self, base: Gf.Vec3d, yaw: float, dt: float) -> tuple[Gf.Vec3d, float]:
        """Low-pass the robot's base pose so the gait does not shake the camera.

        Three separate time constants because the three axes have different needs:
        height only ever oscillates (never intentionally changes here) so it is filtered
        hard; x/y and yaw carry the operator's intent and are filtered just enough to
        remove the per-step wobble without feeling laggy.

        Args:
            base: Raw base position from the articulation root.
            yaw: Raw base yaw in radians.
            dt: Physics timestep in seconds.

        Returns:
            The smoothed ``(base, yaw)`` to build the camera from.
        """
        if not self._camera_stabilization_enabled or dt <= 0.0:
            return base, yaw

        if self._camera_filtered_base is None or self._camera_filtered_yaw is None:
            self._camera_filtered_base = Gf.Vec3d(base)
            self._camera_filtered_yaw = yaw
            return base, yaw

        previous = self._camera_filtered_base
        lateral_alpha = self._clamp_value(dt / self._camera_lateral_filter_time, 0.0, 1.0)
        height_alpha = self._clamp_value(dt / self._camera_height_filter_time, 0.0, 1.0)
        self._camera_filtered_base = Gf.Vec3d(
            previous[0] + (base[0] - previous[0]) * lateral_alpha,
            previous[1] + (base[1] - previous[1]) * lateral_alpha,
            previous[2] + (base[2] - previous[2]) * height_alpha,
        )

        # Yaw is filtered through the shortest angular difference, so crossing +/-pi
        # does not spin the view the long way round.
        yaw_alpha = self._clamp_value(dt / self._camera_yaw_filter_time, 0.0, 1.0)
        delta = self._wrap_angle(yaw - self._camera_filtered_yaw)
        self._camera_filtered_yaw = self._wrap_angle(self._camera_filtered_yaw + delta * yaw_alpha)

        return self._camera_filtered_base, self._camera_filtered_yaw

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap an angle into [-pi, pi]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _update_head_camera_view(self, force: bool = False, dt: float | None = None) -> None:
        """Move the viewport/XR camera to the G1 head pose."""
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
            carb.log_warn(f"HumanoidExample: head camera prim {self._head_camera_path} vanished; recreating it")
            self._head_camera_transform_op = None
            self._create_head_camera()
            if self._head_camera_transform_op is None:
                return

        camera_pose = self._get_head_camera_pose(dt)
        if camera_pose is None:
            return
        self._head_camera_transform_op.Set(camera_pose)
        if self._xr_core is not None:
            if self._g1_locomotion == "stationary" or self._xr_camera_mode == "robot_head":
                try:
                    self._xr_core.schedule_set_camera(camera_pose)
                except Exception as error:
                    self._log_physics_step_error("robot-mounted XR camera", error)
            elif self._xr_camera_mode == "head_compose":
                self._schedule_composed_xr_camera()
            elif self._xr_camera_mode == "stage_anchor":
                self._schedule_stage_anchor_xr_camera()
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
        """Ramp locomotion commands, asymmetrically: ease in, stop hard.

        A single time constant for both directions was a mistake. Smoothing the *start*
        of a command is what stops the gait lurching, but applying the same lag to the
        *end* means releasing the stick leaves the robot walking for about a second —
        "it never stops", plus mushy turns. Speeding up and slowing down are simply not
        the same problem: only one of them needs protecting.

        So accelerating uses the slow attack, and anything that reduces the command —
        releasing, reversing a turn, braking — uses a much faster release. The brake
        bypasses this entirely (see ``_locomotion_brake``) and zeroes the command outright.
        """
        if self._base_command is None:
            return
        if self._g1_locomotion == "stationary":
            # Clear even an old/stale command immediately; no ramp-down movement.
            self._base_command[:] = 0.0
            return
        attack = max(self._command_attack_time, 1e-6)
        release = max(self._command_release_time, 1e-6)
        current = self._base_command
        for axis in range(3):
            if self._locomotion_brake and axis < 2:
                current[axis] = 0.0
                continue
            target_axis = target_command[axis]
            # Toward zero, or reversing sign: that is a release, and it should be quick.
            decelerating = abs(float(target_axis)) < abs(float(current[axis])) or (
                float(target_axis) * float(current[axis]) < 0.0
            )
            alpha = 1.0 - math.exp(-max(float(dt), 0.0) / (release if decelerating else attack))
            current[axis] = current[axis] + (target_axis - current[axis]) * alpha

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
            value = float(input_device.get_input_gesture_value(input_name, gesture_name))
            return self._clamp_value(value, -1.0, 1.0) if math.isfinite(value) else 0.0
        except Exception:
            return 0.0

    def _reset_headset_gait_state(self, reset_clock: bool = True) -> None:
        """Reset gait filters; preserve recording time across a world reset."""
        self._headset_gait_status_logged = False
        if reset_clock:
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
        """Return G1 head position projected onto up_vector for virtual-world-pose gait correction."""
        if not self.g1 or not self.g1.robot.is_physics_tensor_entity_valid():
            return None
        positions, _ = self.g1.robot.get_world_poses()
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
                        carb.log_info(f"HumanoidExample: headset gait using /user/head {pose_reader_name} pose")
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
        self._headset_horiz_speed = math.sqrt(vel_horiz[0] ** 2 + vel_horiz[1] ** 2 + vel_horiz[2] ** 2)
        # Gate: 0 when stationary, reaches 1 at 2× the minimum-speed threshold
        gate_denom = self._headset_gait_min_horiz_speed * 2.0
        self._headset_gait_horiz_gate = self._clamp_value(
            self._headset_horiz_speed / gate_denom if gate_denom > 0.0 else 1.0,
            0.0,
            1.0,
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
            "robot_name": "G1",
            "robot_hand_variant": self._g1_hand_variant,
            "robot_locomotion_requested": self._g1_locomotion,
            "robot_locomotion": getattr(self.g1, "_locomotion", self._g1_locomotion),
            "grasp_mode": "distance_gated_fixed_joint",
            "base_fixed_to_world": self._g1_locomotion == "stationary",
            "station_hold_enabled": bool(self._g1_locomotion == "policy" and self.g1 and self.g1.STATION_HOLD_ENABLED),
            "turn_hold_enabled": bool(self._g1_locomotion == "policy" and self.g1 and self.g1.TURN_HOLD_ENABLED),
            "arm_orientation_tracking": self._arm_track_orientation,
            "finger_control_enabled": self._finger_control_enabled,
            "finger_roles": list(self._finger_roles),
            "finger_actuator_roles": [*self._finger_roles, "thumb_yaw"],
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

    def _update_recorded_locomotion_mode(self) -> None:
        """Record the resolved mode after checkpoint loading may have selected fallback."""
        if self._behavioral_session_dir is None:
            return
        path = self._behavioral_session_dir / "metadata.json"
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["robot_locomotion"] = self.g1._locomotion
            path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except (OSError, ValueError) as error:
            carb.log_warn(f"HumanoidExample: could not update locomotion metadata: {error}")

    def _collect_behavioral_sample(self) -> None:
        """Record one rich behavior.csv row (HMD pose/velocity, gait state, robot pose,
        commands, and every joint's position/velocity), ~100 Hz.

        Rate gating and dispatch happen in _collect_all_behavioral_data().
        """
        unix_now = time.time()
        record = {
            "unix_time": round(unix_now, 6),
            "sim_time": round(self._headset_gait_time, 6),
            "step_index": self._behavioral_data_step_counter,
            # --- headset position ---
            "hmd_pos_x": None,
            "hmd_pos_y": None,
            "hmd_pos_z": None,
            # --- headset orientation (quaternion from pose matrix) ---
            "hmd_qw": None,
            "hmd_qi": None,
            "hmd_qj": None,
            "hmd_qk": None,
            "hmd_yaw": None,
            # --- headset velocity ---
            "hmd_vel_x": round(float(self._headset_velocity[0]), 6),
            "hmd_vel_y": round(float(self._headset_velocity[1]), 6),
            "hmd_vel_z": round(float(self._headset_velocity[2]), 6),
            "hmd_horiz_speed": round(self._headset_horiz_speed, 6),
            # --- gait signal state ---
            "gait_filtered_h": round(self._headset_gait_filtered_height or 0.0, 6),
            "gait_vel_sign": int(self._headset_gait_velocity_sign),
            "gait_horiz_gate": round(self._headset_gait_horiz_gate, 6),
            "gait_output": round(self._headset_gait_output, 6),
            "gait_pulse_rem": round(self._headset_gait_pulse_time_remaining, 6),
            "step_event": int(self._headset_gait_step_event),
            # --- robot base pose ---
            "robot_pos_x": None,
            "robot_pos_y": None,
            "robot_pos_z": None,
            "robot_qw": None,
            "robot_qi": None,
            "robot_qj": None,
            "robot_qk": None,
            "robot_yaw": None,
            # --- locomotion commands ---
            "cmd_forward": round(float(self._base_command[0]), 6) if self._base_command is not None else 0.0,
            "cmd_lateral": round(float(self._base_command[1]), 6) if self._base_command is not None else 0.0,
            "cmd_yaw": round(float(self._base_command[2]), 6) if self._base_command is not None else 0.0,
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
        if self.g1 and self.g1.robot.is_physics_tensor_entity_valid():
            positions, orientations = self.g1.robot.get_world_poses()
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
                dof_names = list(getattr(self.g1.robot, "dof_names", []))
                if dof_names and not self._behavioral_dof_names:
                    self._behavioral_dof_names = dof_names
                joint_pos_raw = self.g1.robot.get_dof_positions()
                joint_vel_raw = self.g1.robot.get_dof_velocities()
                joint_pos = self._first_pose_value(joint_pos_raw)
                joint_vel = self._first_pose_value(joint_vel_raw)
                for i, name in enumerate(self._behavioral_dof_names):
                    safe = name.replace(" ", "_")
                    record[f"j_{safe}_pos"] = (
                        round(float(joint_pos[i]), 6) if joint_pos is not None and i < len(joint_pos) else None
                    )
                    record[f"j_{safe}_vel"] = (
                        round(float(joint_vel[i]), 6) if joint_vel is not None and i < len(joint_vel) else None
                    )
            except Exception:
                pass

        self._headset_gait_step_event = False  # consume the flag
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
        carb.log_info(f"HumanoidExample: closed behavioral session {self._behavioral_session_id} at {session_dir}")

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
        """Lazily wrap the existing /World/G1_HeadCamera prim in a Camera sensor.

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
        """Save one PNG frame from the G1 eye camera (~10 Hz) and log it for frame_timestamps.csv.

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
            pose = read_world_pose(input_device, pose_name)
            if pose is not None:
                return pose
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
                return {
                    "valid": 0,
                    "pos_x": None,
                    "pos_y": None,
                    "pos_z": None,
                    "qw": None,
                    "qx": None,
                    "qy": None,
                    "qz": None,
                }
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

        record = {
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

        # Per-finger curl actually sent to the robot's hands, 0 = open, 1 = closed, plus
        # the source that produced it ("hand_tracking", "controller" or "none"). These
        # are the finger columns the manipulation half of the dataset needs.
        for side in ("left", "right"):
            curls = self._latest_finger_curls.get(side, {})
            record[f"{side}_finger_source"] = self._finger_curl_source.get(side, "none")
            for role in (*self._finger_roles, "thumb_yaw"):
                value = curls.get(role)
                record[f"{side}_finger_{role}"] = round(float(value), 6) if value is not None else None
            record[f"{side}_hand_closure"] = round(self._get_hand_closure(side), 6)

        self._hand_tracking_records.append(record)

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
            record["gaze_source"] = gaze.source
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

        # A tracker that reports loss must stay invalid; do not resurrect its last
        # headset pose from the gait logger. If the optional tracker could not load,
        # record a fresh validated world-space head direction without claiming an eye hit.
        if self._eye_gaze_tracker is None:
            pose = read_world_pose(self._get_xr_input_device("/user/head"))
            if pose is not None:
                origin = pose.ExtractTranslation()
                forward = pose.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
                record["gaze_source"] = "hmd_forward"
                record["gaze_valid"] = 1
                for axis, value in zip("xyz", origin):
                    record[f"gaze_origin_{axis}"] = round(float(value), 6)
                for axis, value in zip("xyz", forward):
                    record[f"gaze_dir_{axis}"] = round(float(value), 6)

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
        for prim in root.GetChildren():
            path = str(prim.GetPath())
            pose = self._get_object_world_pose(path)
            if pose is None:
                continue
            position, quat = pose
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
        smoothing_time = (
            self._headset_gait_attack_time if target > self._headset_gait_output else self._headset_gait_release_time
        )
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
        carb.log_info(f"HumanoidExample XR input devices: left={left_xr is not None}, right={right_xr is not None}")
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
            carb.log_info(f"HumanoidExample XR {label} hand tracking source='{source}', pose_names={pose_names}")

    def _configure_g1_arm_dofs(self) -> None:
        """Find G1 arm DOFs so hand tracking can override only the arms."""
        if self._g1_arm_dofs_configured or not self.g1:
            return

        dof_names = list(getattr(self.g1.robot, "dof_names", []))
        if not dof_names:
            return

        arm_name_map = {
            "left": {
                "shoulder_pitch": ("left_shoulder_pitch_joint", "left_shoulder_pitch"),
                "shoulder_roll": ("left_shoulder_roll_joint", "left_shoulder_roll"),
                "shoulder_yaw": ("left_shoulder_yaw_joint", "left_shoulder_yaw"),
                "elbow": ("left_elbow_joint", "left_elbow"),
                "wrist_roll": ("left_wrist_roll_joint",),
                "wrist_pitch": ("left_wrist_pitch_joint",),
                "wrist_yaw": ("left_wrist_yaw_joint",),
            },
            "right": {
                "shoulder_pitch": ("right_shoulder_pitch_joint", "right_shoulder_pitch"),
                "shoulder_roll": ("right_shoulder_roll_joint", "right_shoulder_roll"),
                "shoulder_yaw": ("right_shoulder_yaw_joint", "right_shoulder_yaw"),
                "elbow": ("right_elbow_joint", "right_elbow"),
                "wrist_roll": ("right_wrist_roll_joint",),
                "wrist_pitch": ("right_wrist_pitch_joint",),
                "wrist_yaw": ("right_wrist_yaw_joint",),
            },
        }

        for side, joints in arm_name_map.items():
            self._g1_arm_dof_indices_by_side[side] = {}
            self._g1_arm_joint_names_by_side[side] = {}
            for joint_key, candidates in joints.items():
                match = next((name for name in candidates if name in dof_names), None)
                if match is None:
                    match = next(
                        (name for name in dof_names if any(candidate in name for candidate in candidates)), None
                    )
                if match is None:
                    continue
                self._g1_arm_dof_indices_by_side[side][joint_key] = dof_names.index(match)
                self._g1_arm_joint_names_by_side[side][joint_key] = match

        all_indices = sorted(
            {index for side_indices in self._g1_arm_dof_indices_by_side.values() for index in side_indices.values()}
        )
        self._g1_arm_joint_defaults = {}
        self._g1_arm_joint_limits = {}
        if all_indices:
            try:
                default_pos = self.g1.default_pos.detach().cpu().numpy()
            except Exception:
                default_pos = None
            try:
                lower_limits, upper_limits = self.g1.robot.get_dof_limits(dof_indices=all_indices)
                lower_limits = lower_limits.numpy()[0]
                upper_limits = upper_limits.numpy()[0]
            except Exception:
                lower_limits = None
                upper_limits = None
            for i, dof_index in enumerate(all_indices):
                self._g1_arm_joint_defaults[dof_index] = (
                    float(default_pos[dof_index]) if default_pos is not None else 0.0
                )
                if lower_limits is not None and upper_limits is not None:
                    self._g1_arm_joint_limits[dof_index] = (float(lower_limits[i]), float(upper_limits[i]))

        self._g1_arm_dofs_configured = True
        carb.log_info(f"HumanoidExample G1 DOFs: {dof_names}")
        carb.log_info(f"HumanoidExample G1 hand-tracked arm DOFs: {self._g1_arm_joint_names_by_side}")

    def _get_hand_input_kind(self, input_device) -> str:
        """Distinguish optical landmarks from a live controller despite source metadata.

        Kit can report `hand` immediately after its hand component starts, while the
        device still contains only the six Touch interaction poses. Generic `palm`,
        `pinch`, and `poke` poses are not a skeleton. Keep partially present skeletons
        optical, so an occlusion cannot turn stale buttons into a controller grab.
        """
        if input_device is None:
            return "none"
        try:
            source = str(input_device.get_hand_tracking_data_source())
        except (AttributeError, RuntimeError):
            source = ""
        try:
            pose_names = {str(name) for name in input_device.get_pose_names()}
        except (AttributeError, RuntimeError):
            pose_names = set()
        if source == "hand":
            skeletal_names = {"wrist"}
            skeletal_names.update(name for chain in self._finger_curl_joint_chains.values() for name in chain)
            if pose_names.intersection(skeletal_names):
                return "hand_tracking"
            # Accept mislabeled Touch input only with its real grip pose and action
            # components. Hand-interaction pinch/poke poses alone cannot enable this.
            if "grip" not in pose_names or read_world_pose(input_device, "grip") is None:
                return "none"
            try:
                inputs = {str(name) for name in input_device.get_input_names()}
            except (AttributeError, RuntimeError):
                return "none"
            return "controller" if {"trigger", "thumbstick"}.issubset(inputs) else "none"
        # A disconnected controller may keep stale analog values. It cannot keep
        # the fingers closed without any valid controller pose in this frame.
        for name in self._controller_pose_candidates:
            if name and pose_names and name not in pose_names:
                continue
            if read_world_pose(input_device, name) is not None:
                return "controller"
        return "none"

    def _get_hand_tracking_pose(self, input_device):
        """Return a tracked hand pose matrix from the best available hand pose name."""
        if self._get_hand_input_kind(input_device) != "hand_tracking":
            return None

        try:
            pose_names = {str(name) for name in input_device.get_pose_names()}
        except Exception:
            pose_names = set()

        for pose_name in self._hand_pose_candidates:
            if pose_name and pose_names and pose_name not in pose_names:
                continue
            pose = read_world_pose(input_device, pose_name)
            if pose is not None:
                return pose
        return None

    def _get_controller_arm_pose(self, side: str, input_device):
        """Return the valid world controller pose while the grip clutch is held.

        Calibration is done in the robot body frame by
        `_compute_arm_target_body_position`, so moving or turning the robot cannot
        masquerade as operator hand motion.
        """
        if self._get_hand_input_kind(input_device) != "controller":
            return None
        if not self._is_controller_arm_pose_enabled(input_device):
            self._controller_arm_neutral_positions.pop(side, None)
            self._controller_arm_neutral_targets.pop(side, None)
            return None
        try:
            pose_names = {str(name) for name in input_device.get_pose_names()}
        except Exception:
            pose_names = set()

        for pose_name in self._controller_pose_candidates:
            if pose_name and pose_names and pose_name not in pose_names:
                continue
            pose = read_world_pose(input_device, pose_name)
            if pose is not None:
                return pose
        return None

    def _is_controller_arm_pose_enabled(self, input_device) -> bool:
        """Use controller pose for arm teleop only while the grip/squeeze is held."""
        squeeze = self._get_xr_gesture_value(input_device, "squeeze", "value")
        squeeze_click = self._get_xr_gesture_value(input_device, "squeeze", "click")
        grip = self._get_xr_gesture_value(input_device, "grip", "value")
        return max(squeeze, squeeze_click, grip) >= self._arm_pose_enable_threshold

    def _get_g1_base_pose_for_arms(self):
        """Return G1 base position and yaw for body-frame arm mapping."""
        if not self.g1 or not self.g1.robot.is_physics_tensor_entity_valid():
            return None
        positions, orientations = self.g1.robot.get_world_poses()
        position = self._first_pose_value(positions)
        orientation = self._first_pose_value(orientations)
        if position is None or orientation is None or len(position) < 3 or len(orientation) < 4:
            return None

        qw, qx, qy, qz = (float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3]))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])), yaw

    def _stage_vector_to_g1_body(self, vector: Gf.Vec3d, yaw: float) -> Gf.Vec3d:
        """Rotate a world-space vector into the G1 base frame using base yaw."""
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return Gf.Vec3d(
            cos_yaw * vector[0] + sin_yaw * vector[1],
            -sin_yaw * vector[0] + cos_yaw * vector[1],
            vector[2],
        )

    def _g1_body_point_to_stage(self, point: Gf.Vec3d, base_position: Gf.Vec3d, yaw: float) -> Gf.Vec3d:
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
        hand_body = self._stage_vector_to_g1_body(hand_position - base_position, yaw)
        if pose_is_relative:
            neutral = self._controller_arm_neutral_positions.get(side)
            if neutral is None:
                self._controller_arm_neutral_positions[side] = Gf.Vec3d(hand_body)
                neutral = hand_body
                # Engage the clutch where the actual palm is, avoiding a jump to a
                # hard-coded pose on every squeeze or tracking reacquisition.
                target = Gf.Vec3d(0.24, 0.24 * side_sign, 0.08)
                hand_prim = self._get_hand_link_prim(side)
                if hand_prim is not None:
                    positions, _ = hand_prim.get_world_poses()
                    value = self._first_pose_value(positions)
                    palm_world = Gf.Vec3d(*[float(v) for v in value[:3]])
                    palm_world += self._palm_offset_world(side, palm_world)
                    target = self._stage_vector_to_g1_body(palm_world - base_position, yaw)
                self._controller_arm_neutral_targets[side] = target
            controller_delta = hand_body - neutral
            hand_body = Gf.Vec3d(self._controller_arm_neutral_targets[side])
            hand_body += Gf.Vec3d(
                self._clamp_value(controller_delta[0] * 1.3, -0.28, 0.42),
                self._clamp_value(controller_delta[1] * 1.2, -0.36, 0.36),
                self._clamp_value(controller_delta[2] * 1.5, -0.40, 0.40),
            )
        return hand_body

    @staticmethod
    def _build_palm_frame(
        side: str, wrist: Gf.Vec3d, middle: Gf.Vec3d, index: Gf.Vec3d, little: Gf.Vec3d
    ) -> Gf.Matrix4d | None:
        """Build a rotation from rigid palm landmarks, independent of finger curl.

        The rows are the distal direction, a side-adjusted transverse direction,
        and the outward palm normal. Side adjustment keeps a proper right-handed
        frame for both hands, with the same physical meaning for the palm normal.
        Collinear or missing landmarks cannot establish a palm orientation.
        """
        if side not in ("left", "right") or any(point is None for point in (wrist, middle, index, little)):
            return None
        if not all(math.isfinite(float(value)) for point in (wrist, middle, index, little) for value in point):
            return None
        forward = middle - wrist
        if forward.GetLength() <= 1e-6:
            return None
        forward.Normalize()
        radial = index - little
        width = radial.GetLength()
        radial -= forward * Gf.Dot(radial, forward)
        if radial.GetLength() <= max(1e-6, width * 0.05):
            return None
        radial.Normalize()
        transverse = radial if side == "left" else -radial
        normal = Gf.Cross(forward, transverse).GetNormalized()
        transverse = Gf.Cross(normal, forward).GetNormalized()
        return Gf.Matrix4d(
            forward[0],
            forward[1],
            forward[2],
            0.0,
            transverse[0],
            transverse[1],
            transverse[2],
            0.0,
            normal[0],
            normal[1],
            normal[2],
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    def _get_robot_palm_local_frame(self, side: str) -> Gf.Matrix4d | None:
        """Read the Inspire palm frame from fixed joint anchors in the hand link.

        Proximal joint anchors on the hand base remain fixed when fingers curl.
        Read their USD joint schemas once because no prim wrapper exposes these
        local anchors. Do not calibrate against the robot's initial wrist pose:
        that would preserve an arbitrary flat-palm/vertical-palm mismatch.
        """
        if side in self._robot_palm_local_frames:
            return self._robot_palm_local_frames[side]
        stage = omni.usd.get_context().get_stage()
        hand_path = self._get_hand_link_path(side)
        if stage is None or hand_path is None:
            return None
        letter = "L" if side == "left" else "R"
        points = {}
        for role, joint_role in (("middle", "middle"), ("index", "index"), ("little", "pinky")):
            joint_path = f"{self._g1_prim_path}/{side}_hand/joints/{letter}_{joint_role}_proximal_joint"
            joint = UsdPhysics.Joint.Get(stage, joint_path)
            if not joint or joint.GetBody0Rel().GetTargets() != [Sdf.Path(hand_path)]:
                return None
            anchor = joint.GetLocalPos0Attr().Get()
            if anchor is None:
                return None
            points[role] = Gf.Vec3d(anchor)
        frame = self._build_palm_frame(side, Gf.Vec3d(0.0), points["middle"], points["index"], points["little"])
        if frame is not None:
            self._robot_palm_local_frames[side] = frame
            # Match the middle-metacarpal centre rather than a moving fingertip
            # centroid. This point is rigidly attached to the hand base.
            self._robot_palm_local_centers[side] = points["middle"] * 0.5
        return frame

    def _get_robot_palm_local_center(self, side: str) -> Gf.Vec3d | None:
        """Return the fixed midpoint between the wrist and middle knuckle."""
        if self._get_robot_palm_local_frame(side) is None:
            return None
        return self._robot_palm_local_centers.get(side)

    def _get_optical_palm_frame(self, side: str, input_device: object) -> Gf.Matrix4d | None:
        """Measure the operator's anatomical palm frame from world joint positions."""
        names = ("wrist", "middle_proximal", "index_proximal", "little_proximal")
        try:
            available = {str(name) for name in input_device.get_pose_names()}
        except (AttributeError, TypeError, RuntimeError):
            return None
        if not all(name in available for name in names):
            return None
        points = [self._get_hand_joint_position(input_device, name) for name in names]
        return self._build_palm_frame(side, *points)

    def _get_optical_arm_pose(self, side: str, input_device: object) -> Gf.Matrix4d | None:
        """Get a stable palm-centre target from position-valid optical landmarks.

        Runtime palm and wrist pose origins differ. Always use the wrist-to-middle
        midpoint, and reject missing or degenerate endpoints rather than switching
        origins during occlusion. Missing transverse landmarks disable orientation
        tracking separately; they do not discard a usable position target. Raw
        recording poses retain their existing runtime frame and validity semantics.
        """
        if self._get_hand_input_kind(input_device) != "hand_tracking":
            return None
        wrist = self._get_hand_joint_position(input_device, "wrist")
        middle = self._get_hand_joint_position(input_device, "middle_proximal")
        if wrist is None or middle is None or (middle - wrist).GetLength() <= 1e-6:
            return None
        frame = self._get_optical_palm_frame(side, input_device)
        result = Gf.Matrix4d(frame) if frame is not None else Gf.Matrix4d(1.0)
        result.SetTranslateOnly((wrist + middle) * 0.5)
        return result

    def _compute_arm_target_orientation(
        self, side: str, hand_pose: Gf.Matrix4d, yaw: float, input_device: object | None = None
    ) -> Gf.Matrix4d | None:
        """Align optical palms anatomically; retain relative rotation for controllers.

        Optical landmarks give an absolute palm orientation, so an operator's flat
        palm immediately requests a flat robot palm regardless of the robot's
        starting posture. Controller grip axes have no anatomical landmarks; their
        existing clutch calibration continues to preserve the initial wrist pose.
        """
        if not self._arm_track_orientation:
            return None
        if input_device is not None and self._get_hand_input_kind(input_device) == "hand_tracking":
            operator_frame = self._get_optical_palm_frame(side, input_device)
            robot_local_frame = self._get_robot_palm_local_frame(side)
            if operator_frame is None or robot_local_frame is None:
                return None
            # Gf uses row vectors: local anatomy * desired wrist = world anatomy.
            # World landmarks already include the XR rig and robot heading; there
            # is no additional yaw correction or initial operator-pose offset.
            return robot_local_frame.GetInverse() * operator_frame
        body_to_world = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw)))
        hand_rotation = Gf.Matrix4d().SetRotate(hand_pose.ExtractRotationQuat()) * body_to_world.GetInverse()
        anchor = self._arm_orientation_anchors.get(side)
        if anchor is None:
            prim = self._get_hand_link_prim(side)
            if prim is None:
                return None
            _, orientations = prim.get_world_poses()
            quaternion = self._first_pose_value(orientations)
            if quaternion is None:
                return None
            robot_rotation = Gf.Matrix4d().SetRotate(
                Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*[float(v) for v in quaternion[1:4]]))
            )
            anchor = (hand_rotation, robot_rotation * body_to_world.GetInverse())
            self._arm_orientation_anchors[side] = anchor
        return anchor[1] * anchor[0].GetInverse() * hand_rotation * body_to_world

    def _set_arm_rig_target_visible(self, side: str, visible: bool) -> None:
        """Show or hide one arm-control rig marker without hiding the hand mesh."""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(f"{self._arm_rig_target_paths[side]}/TargetMarker")
        if not prim.IsValid():
            return
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _smooth_arm_rig_target(self, side: str, target_body: Gf.Vec3d) -> Gf.Vec3d:
        """Smooth the visible arm-rig target in G1 body coordinates."""
        previous = self._smoothed_arm_rig_targets.get(side)
        if previous is None:
            self._smoothed_arm_rig_targets[side] = Gf.Vec3d(target_body)
            return target_body
        alpha = smoothing_alpha(self._arm_rig_smoothing, self._teleop_input_dt)
        smoothed = previous + (target_body - previous) * alpha
        self._smoothed_arm_rig_targets[side] = Gf.Vec3d(smoothed)
        return smoothed

    def _update_arm_rig_target(self, side: str, target_body: Gf.Vec3d, base_position: Gf.Vec3d, yaw: float) -> None:
        """Move the visible rig target marker to the desired hand target."""
        target_op = self._arm_rig_target_ops.get(side)
        if target_op is None:
            return
        target_world = self._g1_body_point_to_stage(target_body, base_position, yaw)
        if side not in self._arm_rig_world_offsets and side in self._manual_arm_rig_target_world_positions:
            self._arm_rig_world_offsets[side] = self._manual_arm_rig_target_world_positions[side] - target_world
            carb.log_info(
                f"HumanoidExample: locked {side} hand rig calibration offset {self._arm_rig_world_offsets[side]}"
            )
        target_world += self._arm_rig_world_offsets.get(side, Gf.Vec3d(0.0, 0.0, 0.0))
        target_matrix = Gf.Matrix4d().SetTranslate(target_world)
        target_op.Set(target_matrix)
        self._active_g1_hand_target_matrices[side] = target_matrix
        self._set_arm_rig_target_visible(side, True)

    def _get_hand_link_prim(self, side: str):
        """Lazily wrap the robot's actual hand link so its live pose can be read.

        Read through the physics tensor API: when Fabric is enabled, USD may retain
        the authored pose rather than the current simulated pose.
        """
        if side in self._hand_link_prims:
            return self._hand_link_prims[side]

        prim = None
        letter = "L" if side == "left" else "R"
        candidates = (
            f"{self._g1_prim_path}/{side}_hand/{letter}_hand_base_link",  # Inspire
            f"{self._g1_prim_path}/{side}_hand_palm_link",  # Dex3
            f"{self._g1_prim_path}/{side}_wrist_yaw_link",  # bare wrist
        )
        for path in candidates:
            try:
                from isaacsim.core.experimental.prims import RigidPrim

                candidate = RigidPrim(paths=path)
                candidate.get_world_poses()  # probe: raises if the path is wrong
                prim = candidate
                carb.log_info(f"HumanoidExample: grabbing uses the real hand link {path}")
                break
            except Exception:
                continue
        self._hand_link_prims[side] = prim
        if prim is None:
            carb.log_warn(
                f"HumanoidExample: could not resolve the {side} hand link; "
                "grabbing is disabled for that hand until the example is reset."
            )
        return prim

    def _get_active_hand_world_position(self, side: str):
        """Center the grab search on the measured palm/fingers of an active hand.

        The wrist is behind the grasp surface and can be outside the search radius
        even when the fingers touch an object. Use the same live palm point as IK;
        neither the XR pose nor the desired arm target proves physical proximity.
        """
        if self._active_g1_hand_target_matrices.get(side) is None:
            return None  # that hand is not being teleoperated this step

        hand_prim = self._get_hand_link_prim(side)
        if hand_prim is not None:
            try:
                positions, _ = hand_prim.get_world_poses()
                position = self._first_pose_value(positions)
                if position is not None and len(position) >= 3:
                    wrist = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
                    return wrist + self._palm_offset_world(side, wrist)
            except Exception:
                pass
        # A desired target is not a measured hand position and cannot authorize a grasp.
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
            attr = prim.GetAttribute("g1:packageSize")
            if attr and attr.HasAuthoredValueOpinion():
                return float(attr.Get())
        except Exception:
            pass
        return 0.5

    def _find_nearest_grabbable_object(self, hand_position: Gf.Vec3d, radius: float | None = None):
        """Find the nearest free package within ``radius`` of the hand.

        Args:
            hand_position: World position of the robot's hand link.
            radius: Search radius in metres. Defaults to the firm grasp radius; pass
                ``_grab_assist_radius`` to include packages the operator is clearly
                reaching for but has not landed the hand on.

        Returns:
            ``(prim path, world position)`` of the nearest candidate, or ``(None, None)``.
        """
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None, None
        root = stage.GetPrimAtPath(self._sample_box_root_path)
        if not root.IsValid():
            return None, None
        search_radius = self._grab_radius if radius is None else radius

        taken_paths = {path for path in self._grabbed_objects_by_side.values() if path is not None}
        best_path = None
        best_position = None
        best_distance = math.inf
        for prim in root.GetChildren():
            path = str(prim.GetPath())
            if path in taken_paths:
                continue
            pose = self._get_object_world_pose(path)
            if pose is None:
                continue
            object_position, _ = pose
            delta = object_position - hand_position
            distance = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])
            size = self._get_grabbable_object_size(prim)
            grab_radius = max(search_radius, size * 0.65)
            if distance <= grab_radius and distance < best_distance:
                best_path = path
                best_position = object_position
                best_distance = distance

        return best_path, best_position

    def _get_object_world_pose(self, object_path: str) -> tuple[Gf.Vec3d, Gf.Quatd] | None:
        """Read a package through physics, including when Fabric leaves USD poses stale."""
        try:
            from isaacsim.core.experimental.prims import RigidPrim

            rigid = self._object_rigid_prims.get(object_path)
            if rigid is None:
                rigid = RigidPrim(paths=object_path)
                self._object_rigid_prims[object_path] = rigid
            positions, orientations = rigid.get_world_poses()
            position = self._first_pose_value(positions)
            orientation = self._first_pose_value(orientations)
            if position is None or orientation is None:
                return None
            values = [float(v) for v in position[:3]] + [float(v) for v in orientation[:4]]
            if len(values) != 7 or not all(math.isfinite(v) for v in values):
                return None
            return Gf.Vec3d(*values[:3]), Gf.Quatd(values[3], Gf.Vec3d(*values[4:])).GetNormalized()
        except Exception as error:
            self._object_rigid_prims.pop(object_path, None)
            self._log_physics_step_error("package pose read", error)
            return None

    def _release_grabbed_object(self, side: str) -> None:
        """Release an object currently held by one hand."""
        object_path = self._grabbed_objects_by_side.pop(side, None)
        self._destroy_grasp_joint(side)
        if object_path is None:
            return
        carb.log_info(f"HumanoidExample: released {object_path} from {side} hand")

    def _update_grabbed_object(self, side: str, grip_active: bool) -> None:
        """Attach a nearby sample object to the hand while the hand is closed.

        Args:
            side: ``"left"`` or ``"right"``.
            grip_active: Whether that hand is currently closed enough to hold an object —
                a squeezed controller grip, or fingers curled past
                ``_finger_grab_threshold`` when hand tracking is driving them.
        """
        hand_position = self._get_active_hand_world_position(side)

        # Y/drop and tracking loss require a fresh open-then-close gesture. Otherwise
        # a still-held trigger recreates the joint in the very same physics step.
        if self._grab_requires_release.get(side, False):
            if not grip_active:
                self._grab_requires_release[side] = False
            self._set_grab_candidate(side, None)
            return

        if not grip_active:
            self._release_grabbed_object(side)
            # Nothing is being taken, so show what WOULD be taken. Without this the
            # operator is guessing where the robot's hand is relative to a crate, through
            # a heuristic arm map, in a headset -- which is the whole reason grabbing felt
            # impossible even once the radius was generous.
            if hand_position is not None:
                candidate, _ = self._find_nearest_grabbable_object(hand_position, self._grab_assist_radius)
                self._set_grab_candidate(side, candidate)
            else:
                self._set_grab_candidate(side, None)
            return

        self._set_grab_candidate(side, None)
        if hand_position is None:
            self._release_grabbed_object(side)
            self._grab_requires_release[side] = True
            return

        stage = omni.usd.get_context().get_stage()
        object_path = self._grabbed_objects_by_side.get(side)
        if object_path is not None:
            joint_path = self._grasp_joints_by_side.get(side)
            if (
                stage is None
                or not stage.GetPrimAtPath(object_path).IsValid()
                or not joint_path
                or not stage.GetPrimAtPath(joint_path).IsValid()
            ):
                self._release_grabbed_object(side)
                self._grab_requires_release[side] = True
            return
        if object_path is None:
            # The assist radius is for SHOWING what you are near; the grasp itself needs
            # the hand actually at the object, because it creates a physical joint.
            object_path, object_position = self._find_nearest_grabbable_object(hand_position, self._grab_radius)
            if object_path is None or object_position is None:
                self._report_grab_miss(side, hand_position)
                return
            gap = self._grasp_contact_gap(side, object_position)
            if gap > self._grasp_contact_distance:
                # Near enough to aim at, not near enough to hold. Welding here would
                # leave the object floating beside the hand.
                self._report_grasp_gap(side, object_path, gap)
                return
            if not self._create_grasp_joint(side, object_path):
                return
            self._grabbed_objects_by_side[side] = object_path
            print(
                f"[G1] {side} hand grasped {self._friendly_package_name(object_path)} "
                f"(fingers {gap * 100:.1f} cm from it)",
                flush=True,
            )
            carb.log_info(f"HumanoidExample: grasped {object_path} with {side} hand")

        # Nothing else to do while held: the joint carries the object, so it stays a
        # normal dynamic body -- it can knock things over, rest on the table, and be put
        # down rather than only dropped.

    #: Link-name fragments that make up the grasping surface of a hand.
    GRASP_LINK_FRAGMENTS = ("palm", "index", "middle", "thumb", "ring", "little", "hand_base")

    def _get_grasp_link_prims(self, side: str) -> list:
        """Live rigid prims for that hand's palm and finger links, resolved once."""
        if side in self._grasp_link_prims:
            return self._grasp_link_prims[side]
        prims = []
        try:
            from isaacsim.core.experimental.prims import RigidPrim

            names = [str(name) for name in self.g1.robot.link_names]
            paths = self.g1.robot.link_paths[0]
            for index, name in enumerate(names):
                lowered = name.lower()
                if side[0] not in ("l", "r"):
                    continue
                # G1 link names are either "<side>_..." or "L_/R_..." depending on the hand.
                belongs = lowered.startswith(f"{side}_") or lowered.startswith("l_" if side == "left" else "r_")
                if not belongs:
                    continue
                if not any(fragment in lowered for fragment in self.GRASP_LINK_FRAGMENTS):
                    continue
                try:
                    prim = RigidPrim(paths=str(paths[index]))
                    prim.get_world_poses()
                    prims.append(prim)
                except Exception:
                    continue
        except Exception as e:
            carb.log_warn(f"HumanoidExample: could not resolve {side} grasp links: {e}")
        if prims:
            print(f"[G1] {side} grasp surface: {len(prims)} palm/finger links", flush=True)
        self._grasp_link_prims[side] = prims
        return prims

    def _palm_offset_world(self, side: str, wrist_world) -> Gf.Vec3d:
        """Live vector from the hand base link to the centre of its grasping surface.

        Prefer a rigid palm point from the asset's knuckle anchors. Averaging moving
        fingertips changes the arm's target as fingers close and also requires many
        individual physics reads. The fallback supports assets without those anchors.
        """
        local_center = self._get_robot_palm_local_center(side)
        if local_center is not None:
            hand_prim = self._get_hand_link_prim(side)
            if hand_prim is not None:
                _, orientations = hand_prim.get_world_poses()
                quaternion = self._first_pose_value(orientations)
                if quaternion is not None:
                    rotation = Gf.Matrix4d().SetRotate(
                        Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*[float(v) for v in quaternion[1:4]]))
                    )
                    return rotation.TransformDir(local_center)
        prims = self._get_grasp_link_prims(side)
        if not prims:
            return Gf.Vec3d(0.0, 0.0, 0.0)
        total = Gf.Vec3d(0.0, 0.0, 0.0)
        count = 0
        for prim in prims:
            try:
                positions, _ = prim.get_world_poses()
                value = self._first_pose_value(positions)
                if value is None or len(value) < 3:
                    continue
                total += Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))
                count += 1
            except Exception:
                continue
        if not count:
            return Gf.Vec3d(0.0, 0.0, 0.0)
        centre = total / float(count)
        return Gf.Vec3d(
            float(centre[0]) - float(wrist_world[0]),
            float(centre[1]) - float(wrist_world[1]),
            float(centre[2]) - float(wrist_world[2]),
        )

    def _grasp_contact_gap(self, side: str, object_position: Gf.Vec3d) -> float:
        """Nearest palm/finger link-centre distance to the object centre, in metres.

        This proximity heuristic does not measure mesh separation, friction, or contact.
        """
        best = float("inf")
        for prim in self._get_grasp_link_prims(side):
            try:
                positions, _ = prim.get_world_poses()
                value = self._first_pose_value(positions)
                if value is None or len(value) < 3:
                    continue
                delta = Gf.Vec3d(
                    float(value[0]) - float(object_position[0]),
                    float(value[1]) - float(object_position[1]),
                    float(value[2]) - float(object_position[2]),
                )
                best = min(best, float(delta.GetLength()))
            except Exception:
                continue
        return best

    def _create_grasp_joint(self, side: str, object_path: str) -> bool:
        """Create an assisted grasp while preserving the live relative body pose.

        Both joint frames must coincide in world space at creation. Identity local
        frames would constrain the two body origins to coincide and yank the object
        into the wrist. This distance-gated fixed joint is grasp assistance, not a
        friction/contact-validated grasp, and recordings must be interpreted that way.
        """
        stage = omni.usd.get_context().get_stage()
        hand_path = self._get_hand_link_path(side)
        if stage is None or hand_path is None:
            carb.log_warn(f"HumanoidExample: no hand link for {side}; cannot create a grasp joint")
            return False

        object_pose = self._get_object_world_pose(object_path)
        hand_prim = self._get_hand_link_prim(side)
        if object_pose is None or hand_prim is None:
            return False
        positions, orientations = hand_prim.get_world_poses()
        position = self._first_pose_value(positions)
        orientation = self._first_pose_value(orientations)
        if position is None or orientation is None:
            return False
        hand_world = Gf.Matrix4d().SetRotate(
            Gf.Quatd(float(orientation[0]), Gf.Vec3d(*[float(v) for v in orientation[1:4]]))
        )
        hand_world.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in position[:3]]))
        object_world = Gf.Matrix4d().SetRotate(object_pose[1])
        object_world.SetTranslateOnly(object_pose[0])
        # Gf uses row vectors: object-local -> world -> hand-local.
        local_hand_frame = object_world * hand_world.GetInverse()

        self._destroy_grasp_joint(side)
        UsdGeom.Xform.Define(stage, self._grasp_joint_root)
        joint_path = f"{self._grasp_joint_root}/{side}_grasp"
        try:
            joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([hand_path])
            joint.CreateBody1Rel().SetTargets([object_path])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(local_hand_frame.ExtractTranslation()))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(local_hand_frame.ExtractRotationQuat().GetNormalized()))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
            joint.CreateCollisionEnabledAttr().Set(False)
            joint.CreateExcludeFromArticulationAttr().Set(True)
            self._grasp_joints_by_side[side] = joint_path
            return True
        except Exception as e:
            if stage.GetPrimAtPath(joint_path).IsValid():
                stage.RemovePrim(joint_path)
            carb.log_warn(f"HumanoidExample: could not create the {side} grasp joint: {e}")
            return False

    def _destroy_grasp_joint(self, side: str) -> None:
        """Remove that hand's grasp joint, letting the object go under gravity."""
        joint_path = self._grasp_joints_by_side.pop(side, None)
        if joint_path is None:
            return
        try:
            stage = omni.usd.get_context().get_stage()
            if stage is not None and stage.GetPrimAtPath(joint_path).IsValid():
                stage.RemovePrim(joint_path)
        except Exception as e:
            carb.log_warn(f"HumanoidExample: could not remove the {side} grasp joint: {e}")

    def _get_hand_link_path(self, side: str) -> str | None:
        """USD path of the hand link the grasp joint attaches to."""
        letter = "L" if side == "left" else "R"
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        for path in (
            f"{self._g1_prim_path}/{side}_hand/{letter}_hand_base_link",
            f"{self._g1_prim_path}/{side}_hand_palm_link",
            f"{self._g1_prim_path}/{side}_wrist_yaw_link",
        ):
            if stage.GetPrimAtPath(path).IsValid():
                return path
        return None

    def _friendly_package_name(self, path: str) -> str:
        """Short, readable name for a package prim path."""
        return str(path).rsplit("/", 1)[-1] if path else "nothing"

    def _report_grasp_gap(self, side: str, object_path: str, gap: float) -> None:
        """Say that the hand is close but not touching, at most every few seconds."""
        last = self._grab_miss_report_time.get(side)
        if last is not None and (self._grab_time - last) < self._grab_miss_report_interval:
            return
        self._grab_miss_report_time[side] = self._grab_time
        print(
            f"[G1] {side} hand: {self._friendly_package_name(object_path)} is "
            f"{gap * 100:.0f} cm from your fingers -- needs "
            f"{self._grasp_contact_distance * 100:.0f} cm to close on it. Reach a little further.",
            flush=True,
        )

    def _report_grab_miss(self, side: str, hand_position: Gf.Vec3d) -> None:
        """Say why a trigger pull took nothing, at most every few seconds.

        A grab that silently does nothing is indistinguishable from a broken grab, which
        is exactly how this felt to use.
        """
        last = self._grab_miss_report_time.get(side)
        if last is not None and (self._grab_time - last) < self._grab_miss_report_interval:
            return
        self._grab_miss_report_time[side] = self._grab_time

        nearest_path, nearest_position = self._find_nearest_grabbable_object(hand_position, 1e9)
        if nearest_path is None or nearest_position is None:
            print(f"[G1] {side} hand: no free packages left to grab", flush=True)
            return
        distance = (nearest_position - hand_position).GetLength()
        print(
            f"[G1] {side} hand: nothing within reach — nearest is "
            f"{self._friendly_package_name(nearest_path)} at {distance:.2f} m "
            f"(pickup search is {self._grab_radius:.2f} m). "
            "Move the robot's palm/fingers closer to an object on the near table.",
            flush=True,
        )

    # ------------------------------------------------------- grab candidate tint

    def _ensure_grab_candidate_material(self):
        """Create (once) the green material that marks the package the trigger would take.

        Bound as a material rather than set as ``displayColor`` for the same reason the
        gaze highlight is: the warehouse crates ship bound PBR materials, and in the RTX
        renderer a bound material wins over displayColor, so the tint would simply not
        appear.
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._grab_candidate_material_path)
        if prim.IsValid():
            return UsdShade.Material(prim)
        try:
            material = UsdShade.Material.Define(stage, self._grab_candidate_material_path)
            shader = UsdShade.Shader.Define(stage, f"{self._grab_candidate_material_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(self._grab_candidate_color)
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(self._grab_candidate_color)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return material
        except Exception as e:
            carb.log_warn(f"HumanoidExample: could not create the grab-candidate material: {e}")
            return None

    def _set_grab_candidate(self, side: str, path: str | None) -> None:
        """Show the candidate without overwriting gaze or the other hand's selection."""
        if not self._grab_candidate_highlight:
            path = None
        stage = omni.usd.get_context().get_stage()
        material = self._ensure_grab_candidate_material() if path is not None and stage is not None else None
        self._material_highlights.set(stage, f"grab:{side}", path, material, priority=20)
        if path is None:
            self._grab_candidate_by_side.pop(side, None)
        else:
            self._grab_candidate_by_side[side] = path

    def _drop_everything(self) -> None:
        """Release both hands (Y on the left controller). The panic button."""
        held = [path for path in self._grabbed_objects_by_side.values() if path]
        for side in ("left", "right"):
            self._release_grabbed_object(side)
            self._grab_requires_release[side] = True
        if held:
            print(f"[G1] dropped {len(held)} held package(s)", flush=True)
        else:
            print("[G1] nothing was being held", flush=True)

    def _get_arm_link_index(self, side: str) -> int | None:
        """Articulation link index of that hand, matched by name against the link list."""
        if side in self._arm_ik_link_index:
            return self._arm_ik_link_index[side]
        index = None
        try:
            names = [str(name) for name in self.g1.robot.link_names]
            letter = "L" if side == "left" else "R"
            wanted = (
                f"{letter}_hand_base_link",
                f"{side}_hand_palm_link",
                f"{side}_wrist_yaw_link",
            )
            for candidate in wanted:
                if candidate in names:
                    index = names.index(candidate)
                    break
        except Exception as e:
            carb.log_warn(f"HumanoidExample: could not resolve the {side} hand link index: {e}")
        self._arm_ik_link_index[side] = index
        return index

    def _solve_arm_ik_jacobian(
        self,
        side: str,
        hand_body: Gf.Vec3d,
        target_orientation: Gf.Matrix4d | None = None,
        *,
        physics_snapshot: dict | None = None,
    ) -> dict[int, float] | None:
        """One damped-least-squares IK step for that arm, on the live Jacobian.

        Returns ``{dof_index: target_angle}``, or None when the live solve is unavailable
        (physics not up, link not found, degenerate Jacobian) so the caller can fall back.

        Closed loop: it reads where the hand actually is and steps the joints toward the
        target, rather than mapping a target through a fixed linear model. That is the
        difference between the arm tracking your hand and merely correlating with it.
        Both arms may share ``physics_snapshot`` within one synchronous physics
        callback. Omitting it always reads fresh tensors; never retain it across steps.
        """
        indices = self._g1_arm_dof_indices_by_side.get(side, {})
        if not indices or not self.g1 or not self.g1.robot.is_physics_tensor_entity_valid():
            return None
        link_index = self._get_arm_link_index(side)
        if link_index is None:
            return None

        base_pose = self._get_g1_base_pose_for_arms()
        hand_prim = self._get_hand_link_prim(side)
        if base_pose is None or hand_prim is None:
            return None
        base_position, yaw = base_pose

        try:
            import warp as wp

            positions, orientations = hand_prim.get_world_poses()
            hand_world = self._first_pose_value(positions)
            if hand_world is None or len(hand_world) < 3:
                return None
            # The Jacobian is expressed in WORLD axes, so the error has to be too --
            # mixing a body-frame error with a world-frame Jacobian steers the arm sideways
            # the moment the robot is not facing +x.
            target_world = self._g1_body_point_to_stage(hand_body, base_position, yaw)
            # Aim the PALM at the target, not the wrist. The IK controls the hand base
            # link, which sits ~8 cm behind the grasping surface, so driving the wrist to
            # the object leaves the fingers permanently short of it -- measured, they
            # settled 8.6 cm away, which is why a grasp there welded the object into mid
            # air and why lifting it had so little authority. Subtracting the live
            # wrist-to-palm vector makes the operator's hand position mean the robot's
            # palm, which is what they are actually aiming with.
            palm_offset = self._palm_offset_world(side, hand_world)
            error = [
                float(target_world[axis]) - float(hand_world[axis]) - float(palm_offset[axis]) for axis in range(3)
            ]
            length = math.sqrt(sum(component * component for component in error))
            if length > self._arm_ik_max_error > 0.0:
                scale = self._arm_ik_max_error / length
                error = [component * scale for component in error]

            snapshot = {} if physics_snapshot is None else physics_snapshot
            if "arm_jacobians" not in snapshot or "arm_dof_positions" not in snapshot:
                jacobians = wp.to_torch(self.g1.robot.get_jacobian_matrices()).detach().cpu().numpy()
                current = wp.to_torch(self.g1.robot.get_dof_positions()).reshape(-1).detach().cpu().numpy()
                # Publish both reads together. A failed read must not leave a partial
                # snapshot that a second arm could mistake for a complete observation.
                snapshot.update(arm_jacobians=jacobians, arm_dof_positions=current)
            jacobians = snapshot["arm_jacobians"]
            current = snapshot["arm_dof_positions"]
            joint_order = [
                name for name in (*ARM_IK_JOINT_ORDER, "wrist_roll", "wrist_pitch", "wrist_yaw") if name in indices
            ]
            # Floating base: the first SIX columns are the free root's linear and angular
            # DOFs and the joint columns follow, so joint j lives at column j + 6. Indexing
            # as if the joints came first reads six unrelated columns and the arm creeps in
            # the wrong direction.
            num_dofs = int(self.g1.robot.num_dofs)
            column_offset = 6 if int(jacobians.shape[-1]) == num_dofs + 6 else 0
            # Fixed-base Jacobians omit the root link; floating-base ones include it.
            row_index = link_index - (int(jacobians.shape[1]) == int(self.g1.robot.num_links) - 1)
            if row_index < 0 or int(jacobians.shape[-1]) not in (num_dofs, num_dofs + 6):
                return None
            columns = [indices[name] + column_offset for name in joint_order]
            rows = jacobians[0, row_index, :, :][:, columns]
            full_matrix = rows.astype(float)
        except Exception as e:
            self._log_physics_step_error("arm IK jacobian read", e)
            return None

        try:
            import numpy as np

            # Shift the wrist Jacobian to the palm point used in the residual:
            # v_palm = v_wrist + omega cross r. Without this, wrist rotation moves
            # the controlled point in a direction that the solver does not predict.
            matrix = full_matrix[:3] + np.cross(full_matrix[3:6].T, np.asarray(palm_offset)).T
            arm_positions = np.asarray([current[indices[name]] for name in joint_order], dtype=float)
            if (
                not np.all(np.isfinite(full_matrix))
                or not np.all(np.isfinite(error))
                or not np.all(np.isfinite(arm_positions))
            ):
                return None
            damping = self._arm_ik_damping
            gain = self._arm_ik_gain
            step_cap = self._arm_ik_max_step
            if not all(math.isfinite(value) for value in (damping, gain, step_cap)) or step_cap <= 0:
                return None

            def damped_step(jacobian, residual):
                return jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping * damping * np.eye(jacobian.shape[0]), residual
                )

            # Keep each command within the available joint travel as well as the IK
            # correction cap. If contact pushed a measured joint beyond a limit, do
            # not ask it to move farther outward; the final command limiter restores
            # the authored limit. The zero correction remains feasible in this solve.
            lower_step = np.full(len(joint_order), -step_cap)
            upper_step = np.full(len(joint_order), step_cap)
            for index, name in enumerate(joint_order):
                lower, upper = self._g1_arm_joint_limits.get(indices[name], (-math.inf, math.inf))
                if math.isnan(lower) or math.isnan(upper) or lower > upper:
                    return None
                lower_step[index] = max(-step_cap, min(0.0, lower - arm_positions[index]))
                upper_step[index] = min(step_cap, max(0.0, upper - arm_positions[index]))

            def add_within_budget(base, correction):
                """Fit a secondary direction without shrinking the primary step."""
                fraction = 1.0
                for index, value in enumerate(correction):
                    if value > 1e-12:
                        fraction = min(fraction, (upper_step[index] - base[index]) / value)
                    elif value < -1e-12:
                        fraction = min(fraction, (lower_step[index] - base[index]) / value)
                # A common scale preserves the correction's nullspace direction.
                return base + max(0.0, min(1.0, fraction)) * correction

            # Reaching is the primary task. A blended position/orientation solve can
            # sacrifice the reach to preserve an infeasible wrist angle, eventually
            # twisting the shoulder against its limits during a sustained grip.
            # Re-solve without joints already at a limit if their correction points
            # outward. Otherwise one saturated joint can stop the entire reach.
            free = np.ones(len(joint_order), dtype=bool)
            step = np.zeros(len(joint_order))
            for _ in range(len(joint_order) + 1):
                step[:] = 0.0
                step[free] = damped_step(matrix[:, free], np.asarray(error, dtype=float) * gain)
                blocked = free & (((lower_step >= -1e-8) & (step < 0.0)) | ((upper_step <= 1e-8) & (step > 0.0)))
                if not np.any(blocked):
                    break
                free[blocked] = False
            step = add_within_budget(np.zeros(len(joint_order)), step)

            def task_nullspace(tasks):
                """Project only through joints available to the primary task."""
                projector = np.zeros((len(joint_order), len(joint_order)))
                free_indices = np.flatnonzero(free)
                restricted = tasks[:, free]
                projector[np.ix_(free_indices, free_indices)] = (
                    np.eye(len(free_indices)) - np.linalg.pinv(restricted, rcond=1e-4) @ restricted
                )
                return projector

            position_null = task_nullspace(matrix)
            posture_tasks = matrix
            if target_orientation is not None:
                quaternion = self._first_pose_value(orientations)
                current_rotation = Gf.Matrix4d().SetRotate(
                    Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*[float(v) for v in quaternion[1:4]]))
                )
                rotation_error = (current_rotation.GetInverse() * target_orientation).ExtractRotation()
                angle = self._wrap_angle(math.radians(rotation_error.GetAngle()))
                rotation_vector = np.asarray(rotation_error.GetAxis()) * angle
                weight = self._arm_orientation_weight
                angular = full_matrix[3:6]
                # Use only motion that preserves the primary palm task to first order.
                # The exact SVD projector is intentional: a damped projector leaks the
                # secondary task into the reach when the arm is near a singular pose.
                orientation_matrix = angular @ position_null * weight
                orientation_residual = (rotation_vector * gain - angular @ step) * weight
                orientation_step = position_null @ damped_step(orientation_matrix, orientation_residual)
                step = add_within_budget(step, orientation_step)
                posture_tasks = np.vstack((matrix, angular))

            # The seven-joint arm still has redundant freedom. Resolve it toward the
            # default posture instead of allowing shoulder/wrist branches to drift.
            posture_null = task_nullspace(posture_tasks)
            rest_error = np.asarray(
                [
                    self._g1_arm_joint_defaults.get(indices[name], float(current[indices[name]]))
                    - float(current[indices[name]])
                    for name in joint_order
                ]
            )
            posture_step = self._arm_posture_gain * gain * (posture_null @ rest_error)
            step = add_within_budget(step, posture_step)
            if not np.all(np.isfinite(step)):
                return None
        except Exception:
            return None

        targets = {}
        for offset, name in enumerate(joint_order):
            dof_index = indices[name]
            targets[dof_index] = float(current[dof_index]) + float(step[offset])

        if not self._arm_ik_logged:
            self._arm_ik_logged = True
            print(
                f"[G1] arm IK: live PhysX Jacobian ({side} hand = link {link_index}, " f"joints {joint_order})",
                flush=True,
            )
        return targets

    def _compute_arm_targets_from_body_position(
        self,
        side: str,
        hand_body: Gf.Vec3d,
        target_orientation: Gf.Matrix4d | None = None,
        *,
        physics_snapshot: dict | None = None,
    ) -> dict[int, float]:
        """Turn a desired hand position (body frame, relative to the pelvis) into arm joint targets.

        A linear inverse of the arm's MEASURED Jacobian, not a hand-tuned gain table. See
        ARM_IK_PINV above for why: the old gains inverted the vertical axis and flattened
        the lateral one, so the robot's hand moved down when the operator's went up and
        barely moved sideways at all.

        Args:
            side: ``"left"`` or ``"right"``.
            hand_body: Target hand position in the robot's body frame, relative to the
                pelvis, in metres.
            target_orientation: Desired wrist rotation in world space for the live IK solver.
            physics_snapshot: Optional tensor reads shared by both arms in this physics
                callback only. Use a fresh dictionary for every callback.

        Returns:
            ``{dof_index: target_angle}`` for the seven live arm joints, or the
            four-joint degraded fallback when a live Jacobian is unavailable.
        """
        if self._arm_ik_use_jacobian:
            solved = self._solve_arm_ik_jacobian(side, hand_body, target_orientation, physics_snapshot=physics_snapshot)
            if solved:
                return solved

        indices = self._g1_arm_dof_indices_by_side.get(side, {})
        if not indices:
            return {}

        # Work in right-arm space; the left arm is its mirror image about y.
        mirror = 1.0 if side == "right" else -1.0
        offset = [
            float(hand_body[0]) - ARM_IK_REFERENCE[0],
            float(hand_body[1]) * mirror - ARM_IK_REFERENCE[1],
            float(hand_body[2]) - ARM_IK_REFERENCE[2],
        ]

        # Cap the offset to the radius the linearisation is good for, keeping its
        # direction so the hand still points where the operator is reaching.
        length = math.sqrt(sum(component * component for component in offset))
        if length > ARM_IK_MAX_OFFSET > 0.0:
            scale = ARM_IK_MAX_OFFSET / length
            offset = [component * scale for component in offset]

        targets = {}
        for row, joint in enumerate(ARM_IK_JOINT_ORDER):
            dof_index = indices.get(joint)
            if dof_index is None:
                continue
            delta = sum(ARM_IK_PINV[row][axis] * offset[axis] for axis in range(3))
            angle = ARM_IK_NEUTRAL[row] + delta
            if joint in ("shoulder_roll", "shoulder_yaw"):
                angle *= mirror  # mirrored joints for the left arm
            targets[dof_index] = angle
        return targets

    def _update_teleop_input_clock(self, dt: float, now: float | None = None) -> None:
        """Keep input filtering responsive when rendering slows simulation progress.

        Hand samples arrive in real time. Filtering them using only simulated time
        stretches the filter delay whenever the simulation runs below real time.
        Use bounded wall time for input filters, while physics and joint-speed
        limits continue using the actual physics timestep. A pause or long stall
        starts a fresh filter interval rather than permitting a large catch-up jump.

        Args:
            dt: Positive physics timestep in seconds.
            now: Optional monotonic timestamp for deterministic validation.
        """
        now = time.perf_counter() if now is None else float(now)
        previous = self._last_teleop_wall_time
        self._last_teleop_wall_time = now if math.isfinite(now) else None
        elapsed = 0.0 if previous is None else now - previous
        if not math.isfinite(elapsed) or elapsed <= 0.0 or elapsed > 0.25:
            self._teleop_input_dt = dt
        else:
            self._teleop_input_dt = max(dt, min(elapsed, 0.05))

    def _smooth_and_clamp_arm_targets(self, raw_targets: dict[int, float]) -> dict[int, float]:
        """Smooth hand-tracking joint targets and clamp to joint limits."""
        targets = {}
        positions = self._first_pose_value(self.g1.robot.get_dof_positions())
        alpha = smoothing_alpha(self._arm_smoothing, self._teleop_input_dt)
        max_step = self._arm_max_joint_speed * self._last_physics_dt
        for dof_index, raw_target in raw_targets.items():
            if not math.isfinite(raw_target):
                continue
            current = float(positions[dof_index])
            previous = self._smoothed_arm_targets.get(dof_index, current)
            smoothed = previous + alpha * (raw_target - previous)
            lower, upper = self._g1_arm_joint_limits.get(dof_index, (-math.inf, math.inf))
            # Slew-limit the COMMAND, then bound its lead over the measured joint.
            # Limiting target-current to speed*dt instead starves a position drive:
            # at 100 Hz its persistent 0.025 rad error produces very slow arm motion.
            # A bounded tracking allowance lets the PD drive catch up while preventing
            # a stalled arm from accumulating a distant target behind an obstacle.
            smoothed = self._clamp_value(smoothed, previous - max_step, previous + max_step)
            max_error = self._arm_max_tracking_error
            smoothed = self._clamp_value(smoothed, current - max_error, current + max_error)
            smoothed = self._clamp_value(smoothed, lower, upper)
            self._smoothed_arm_targets[dof_index] = smoothed
            targets[dof_index] = smoothed
        return targets

    def _update_g1_arms_from_hand_tracking(self) -> None:
        """Override G1 arm DOF targets from Meta/OpenXR hand-tracking poses."""
        if not self._hand_tracking_arm_control_enabled or self._xr_core is None or not self.g1:
            for side in ("left", "right"):
                self._deactivate_hand(side)
            return

        self._configure_g1_arm_dofs()
        if not self._g1_arm_dofs_configured:
            return

        left_xr = self._get_xr_input_device("/user/hand/left")
        right_xr = self._get_xr_input_device("/user/hand/right")
        self._log_hand_tracking_status_once(left_xr, right_xr)

        base_pose = self._get_g1_base_pose_for_arms()
        if base_pose is None:
            for side in ("left", "right"):
                self._deactivate_hand(side)
            return
        base_position, yaw = base_pose

        raw_targets = {}
        # Both arms observe the same completed physics step. Reuse its full
        # Jacobian/joint snapshot, and discard it before the next callback.
        physics_snapshot = {}
        active_sides = set()
        for side, device in (("left", left_xr), ("right", right_xr)):
            hand_pose = self._get_optical_arm_pose(side, device)
            source = "hand_tracking" if hand_pose is not None else "controller"
            pose_is_relative = False
            if hand_pose is None and self._controller_arm_control_enabled:
                hand_pose = self._get_controller_arm_pose(side, device)
                pose_is_relative = hand_pose is not None
            if hand_pose is None:
                continue
            if self._arm_input_sources.get(side) != source:
                for index in self._g1_arm_dof_indices_by_side.get(side, {}).values():
                    self._smoothed_arm_targets.pop(index, None)
                self._controller_arm_neutral_positions.pop(side, None)
                self._controller_arm_neutral_targets.pop(side, None)
                self._arm_orientation_anchors.pop(side, None)
                self._smoothed_arm_rig_targets.pop(side, None)
                self._arm_input_sources[side] = source
            active_sides.add(side)
            target_body = self._compute_arm_target_body_position(side, hand_pose, base_position, yaw, pose_is_relative)
            target_body = self._smooth_arm_rig_target(side, target_body)
            self._update_arm_rig_target(side, target_body, base_position, yaw)
            self._update_grabbed_object(side, self._is_hand_closed(side, device))
            orientation = self._compute_arm_target_orientation(
                side, hand_pose, yaw, input_device=device if source == "hand_tracking" else None
            )
            raw_targets.update(
                self._compute_arm_targets_from_body_position(
                    side, target_body, orientation, physics_snapshot=physics_snapshot
                )
            )

        for side in ("left", "right"):
            if side not in active_sides:
                self._deactivate_hand(side)
                for index in self._g1_arm_dof_indices_by_side.get(side, {}).values():
                    raw_targets[index] = self._g1_arm_joint_defaults.get(index, 0.0)

        if not raw_targets:
            return
        targets = self._smooth_and_clamp_arm_targets(raw_targets)
        dof_indices = sorted(targets)
        self.g1.robot.set_dof_position_targets([targets[index] for index in dof_indices], dof_indices=dof_indices)

    def _deactivate_hand(self, side: str) -> None:
        """Release a lost input source and invalidate its clutch calibration."""
        # Clear the active command once on tracking loss. Repeating this every
        # inactive tick restarts the return-to-rest slew at the measured joint and
        # prevents a position drive from building enough error to move promptly.
        if side in self._arm_input_sources:
            for index in self._g1_arm_dof_indices_by_side.get(side, {}).values():
                self._smoothed_arm_targets.pop(index, None)
        if self._grabbed_objects_by_side.get(side) is not None:
            self._grab_requires_release[side] = True
        self._release_grabbed_object(side)
        self._set_grab_candidate(side, None)
        self._active_g1_hand_target_matrices.pop(side, None)
        self._smoothed_arm_rig_targets.pop(side, None)
        self._controller_arm_neutral_positions.pop(side, None)
        self._controller_arm_neutral_targets.pop(side, None)
        self._arm_orientation_anchors.pop(side, None)
        self._arm_input_sources.pop(side, None)
        self._hand_closed_by_side[side] = False
        self._set_arm_rig_target_visible(side, False)

    def _update_g1_fingers(self) -> None:
        """Drive the G1's real finger joints from hand tracking or the controllers.

        Runs every physics step, before arm/grasp decisions. Each hand independently
        prefers OpenXR hand tracking (per-finger flexion measured from the tracked hand
        skeleton) and falls back to the controller's trigger. Thumb opposition has
        its own target. Curls are smoothed, sent to the robot, and recorded.
        """
        if not self._finger_control_enabled or not self.g1 or not self.g1.has_finger_control():
            return
        if self._xr_core is None:
            # Also discard previous samples if a runtime disappears during play.
            self._latest_finger_curls.clear()
            self._finger_curl_source.clear()
            self._smoothed_finger_curls.clear()
            for side in ("left", "right"):
                self.g1.set_finger_curls(side, {role: 0.0 for role in self._finger_roles})
            return

        for side in ("left", "right"):
            device = self._get_xr_input_device(f"/user/hand/{side}")
            curls = self._get_hand_tracking_finger_curls(device)
            source = "hand_tracking"
            if curls is None:
                curls = self._get_controller_finger_curls(device)
                source = "controller"
            if curls is None:
                # No device on this side: let the hand relax open rather than freezing
                # it in whatever pose it held when tracking was lost.
                curls = {role: 0.0 for role in self._finger_roles}
                source = "none"

            curls = self._smooth_finger_curls(side, curls)
            self._latest_finger_curls[side] = curls
            self._finger_curl_source[side] = source
            self.g1.set_finger_curls(side, curls)

    def _smooth_finger_curls(self, side: str, curls: dict[str, float]) -> dict[str, float]:
        """Low-pass the per-finger curls so tracking jitter does not buzz the joints."""
        previous = self._smoothed_finger_curls.get(side, {})
        alpha = smoothing_alpha(self._finger_smoothing, self._teleop_input_dt)
        smoothed = {}
        for role in (*self._finger_roles, "thumb_yaw"):
            # Controllers close the whole hand, including opposition. Optical input
            # supplies opposition explicitly, so flexing the thumb does not rotate it.
            fallback = curls.get("thumb", 0.0) if role == "thumb_yaw" else 0.0
            target = curls.get(role, fallback)
            if not math.isfinite(float(target)):
                target = 0.0
            target = self._clamp_value(float(target), 0.0, 1.0)
            if target < self._finger_curl_deadzone:
                target = 0.0
            last = previous.get(role, target)
            smoothed[role] = last + (target - last) * alpha
        self._smoothed_finger_curls[side] = smoothed
        return smoothed

    def _get_hand_tracking_finger_curls(self, input_device) -> dict[str, float] | None:
        """Measure each finger's curl from the OpenXR tracked hand skeleton.

        Read each digit independently; a missing joint relaxes only that digit.
        Summing adjacent bone bends is independent of hand size, room position, and
        wrist orientation, and remains closed when a fist folds beyond 180 degrees.
        The six-actuator Inspire hand couples each finger's distal joints mechanically;
        retarget its total bend, with a separate palm-plane thumb opposition target.
        """
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
            return None

        positions_by_name = {}

        def position(name):
            if name not in positions_by_name:
                positions_by_name[name] = (
                    self._get_hand_joint_position(input_device, name) if name in pose_names else None
                )
            return positions_by_name[name]

        curls = {}
        for role, chain in self._finger_curl_joint_chains.items():
            positions = [position(name) for name in chain]
            if any(position is None for position in positions):
                continue
            bones = [second - first for first, second in zip(positions, positions[1:])]
            angles = [self._angle_between(first, second) for first, second in zip(bones, bones[1:])]
            if any(angle is None for angle in angles):
                continue
            full_flexion = self._finger_curl_full_flexion_rad.get(role, math.radians(150.0))
            curls[role] = self._clamp_value(sum(angles) / full_flexion, 0.0, 1.0)

        if not curls:
            return None
        # Missing opposition landmarks must not substitute thumb curl: that would
        # rotate the thumb merely because its tip flexed during an occlusion.
        curls["thumb_yaw"] = self._get_hand_tracking_thumb_opposition(position)
        if not self._finger_tracking_status_logged:
            self._finger_tracking_status_logged = True
            carb.log_info(f"HumanoidExample: finger teleoperation using hand-tracking joints {sorted(curls)}")
        return curls

    def _get_hand_joint_position(self, input_device, pose_name: str):
        """Read a finite world joint position without requiring tracked orientation.

        Finger bend uses translation only. OpenXR may mark a joint's position valid
        while its orientation is unavailable; keep that usable skeletal landmark.
        Invalid descriptors never fall back to a stale matrix or physical-room pose.
        """
        try:
            reader = getattr(input_device, "get_virtual_world_pose_desc", None)
            if not callable(reader):
                pose = read_world_pose(input_device, pose_name)
                return Gf.Vec3d(pose.ExtractTranslation()) if pose is not None else None
            descriptor = reader(pose_name)
            if int(descriptor.validity_flags) & 0x2 == 0:
                return None
            matrix = descriptor.pose_matrix
            if matrix is None or len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                return None
            coordinates = tuple(float(matrix[3][axis]) for axis in range(3))
            if not all(math.isfinite(value) for value in coordinates):
                return None
            return Gf.Vec3d(*coordinates)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None

    def _get_hand_tracking_thumb_opposition(self, position) -> float:
        """Measure thumb opposition in the palm plane, independently of tip flexion."""
        landmarks = [
            position(name)
            for name in (
                "wrist",
                "middle_proximal",
                "index_proximal",
                "little_proximal",
                "thumb_metacarpal",
                "thumb_proximal",
            )
        ]
        if any(landmark is None for landmark in landmarks):
            return 0.0
        wrist, middle, index, little, thumb_base, thumb_knuckle = landmarks
        forward = middle - wrist
        outward = index - little
        if forward.GetLength() <= 1e-6:
            return 0.0
        forward.Normalize()
        outward -= forward * Gf.Dot(outward, forward)
        if outward.GetLength() <= 1e-6:
            return 0.0
        outward.Normalize()
        thumb = thumb_knuckle - thumb_base
        thumb_in_plane = forward * Gf.Dot(thumb, forward) + outward * Gf.Dot(thumb, outward)
        angle = self._angle_between(outward, thumb_in_plane)
        if angle is None:
            return 0.0
        travel = self._thumb_opposition_closed_angle - self._thumb_opposition_open_angle
        return self._clamp_value((angle - self._thumb_opposition_open_angle) / travel, 0.0, 1.0)

    def _angle_between(self, first: Gf.Vec3d, second: Gf.Vec3d) -> float | None:
        """Return the unsigned angle in radians between two vectors, or None if degenerate."""
        first_length = first.GetLength()
        second_length = second.GetLength()
        if not math.isfinite(first_length + second_length) or first_length <= 1e-6 or second_length <= 1e-6:
            return None
        cosine = self._clamp_value(Gf.Dot(first, second) / (first_length * second_length), -1.0, 1.0)
        return math.acos(cosine)

    def _get_controller_finger_curls(self, input_device) -> dict[str, float] | None:
        """Close the controller-driven hand with the trigger; grip is the arm clutch.

        Keeping the clutch separate lets the operator reach with an open hand, then
        curl around the target without changing the arm's calibration.
        """
        if self._get_hand_input_kind(input_device) != "controller":
            return None
        trigger = max(
            self._get_xr_gesture_value(input_device, "trigger", "value"),
            self._get_xr_gesture_value(input_device, "trigger", "click"),
        )
        return {role: trigger for role in self._finger_roles}

    def _get_hand_closure(self, side: str) -> float:
        """Return how closed one hand is, as the mean curl across its fingers."""
        curls = self._latest_finger_curls.get(side)
        if not curls:
            return 0.0
        return sum(curls.get(role, 0.0) for role in self._finger_roles) / len(self._finger_roles)

    def _is_hand_closed(self, side: str, input_device) -> bool:
        """Return whether that hand is deliberately closing on something.

        Grabbing is on the **trigger**, not the grip. It used to be the grip -- the same
        button that enables arm teleoperation -- which meant you could not reach an arm
        toward a table without picking up whatever was within the grab radius. Holding
        grip is how you *move*; pulling the trigger is how you *take*.

        Under hand tracking there are no buttons, so actually closing the fingers is the
        gesture.
        """
        was_closed = self._hand_closed_by_side.get(side, False)
        if self._finger_curl_source.get(side) == "hand_tracking":
            value = self._get_hand_closure(side)
            threshold = self._finger_release_threshold if was_closed else self._finger_grab_threshold
        else:
            value = max(
                self._get_xr_gesture_value(input_device, "trigger", "value"),
                self._get_xr_gesture_value(input_device, "trigger", "click"),
            )
            threshold = self._grab_trigger_release_threshold if was_closed else self._grab_trigger_threshold
        closed = value >= threshold
        self._hand_closed_by_side[side] = closed
        return closed

    def _get_xr_stick_axis(self, input_device, axis: str) -> float:
        """Read a thumbstick axis, trying the input names different runtimes use.

        Returns 0.0 both when the stick is centred and when the runtime does not expose
        one, which is exactly the fallback behaviour the caller wants.
        """
        for input_name in self._xr_stick_input_candidates:
            value = self._get_xr_gesture_value(input_device, input_name, axis)
            if value != 0.0:
                return value
        return 0.0

    def _read_xr_controller_axes(self) -> tuple[float, float]:
        """Return forward and yaw commands from the XR controllers.

        Every source is read and then combined, largest magnitude wins per axis. It used
        to return early as soon as a thumbstick reported anything, which meant the A/X
        buttons were dead the whole time you were pushing the stick — you could walk or
        turn, never both, and rotation simply stopped responding mid-walk.

        Mapping:
            left stick Y / right trigger  -> forward
            left stick back / left trigger -> brake
            right stick X                 -> turn (analog)
            X button (left controller)    -> turn LEFT
            A button (right controller)   -> turn RIGHT
        """
        left_xr = self._get_xr_input_device("/user/hand/left")
        right_xr = self._get_xr_input_device("/user/hand/right")
        self._log_xr_input_status_once(left_xr, right_xr)

        # --- thumbsticks: the left one is now fully omnidirectional ---
        stick_forward = self._apply_deadzone(self._get_xr_stick_axis(left_xr, "y"))
        # OpenXR +X is stick-right; the robot's +Y velocity is body-left.
        stick_lateral = -self._apply_deadzone(self._get_xr_stick_axis(left_xr, "x"))
        stick_yaw = -self._apply_deadzone(self._get_xr_stick_axis(right_xr, "x"))
        if stick_forward < 0.0:
            # Reverse is on/off at the policy's command limit -- see _max_backward_speed for
            # the sweep. Anything gentler measurably walks the robot FORWARDS.
            stick_forward = -1.0

        # The triggers no longer drive locomotion at all. They were doing double duty as
        # walk-forward AND the grasp, which meant reaching for something and moving were
        # the same button; the stick covers every direction now, so they are free.
        # --- buttons: X turns left, A turns right ---
        turn_left = max(
            self._get_xr_gesture_value(left_xr, "x", "click"),
            self._get_xr_gesture_value(left_xr, "x", "value"),
        )
        turn_right = max(
            self._get_xr_gesture_value(right_xr, "a", "click"),
            self._get_xr_gesture_value(right_xr, "a", "value"),
        )
        button_yaw = self._apply_deadzone(turn_left - turn_right)

        # --- B recenters the VR rig, Y drops whatever is held ---
        # Both are edge-triggered: these fire an action, not a continuous command, and
        # a held button at 100 Hz would fire it a hundred times a second.
        recenter_down = self._get_xr_gesture_value(right_xr, "b", "click") >= 0.5
        if recenter_down and not self._xr_recenter_button_down:
            self._request_xr_recenter()
        self._xr_recenter_button_down = recenter_down

        drop_down = self._get_xr_gesture_value(left_xr, "y", "click") >= 0.5
        if drop_down and not self._drop_button_down:
            self._drop_everything()
        self._drop_button_down = drop_down

        mode_down = self._get_xr_stick_click(left_xr) >= 0.5
        if mode_down and not self._xr_mode_button_down:
            self._cycle_xr_camera_mode()
        self._xr_mode_button_down = mode_down

        if self._g1_locomotion == "stationary":
            # B/Y/camera actions above still run. Sticks and A/X cannot move the base.
            self._latest_stick_lateral = 0.0
            return 0.0, 0.0

        forward = stick_forward
        yaw = stick_yaw if abs(stick_yaw) > abs(button_yaw) else button_yaw

        self._latest_stick_lateral = stick_lateral
        if (forward != 0.0 or yaw != 0.0 or stick_lateral != 0.0) and not self._locomotion_input_logged:
            self._locomotion_input_logged = True
            using = []
            if stick_forward != 0.0 or stick_yaw != 0.0 or stick_lateral != 0.0:
                using.append("thumbstick")
            if stick_lateral != 0.0:
                using.append("strafe")
            if button_yaw != 0.0:
                using.append("A/X buttons")
            print(f"[G1] locomotion input: {' + '.join(using) or 'none'}", flush=True)
        return forward, yaw

    def _get_xr_stick_click(self, input_device) -> float:
        """Read the thumbstick click, whichever name the runtime gives the stick."""
        for input_name in self._xr_stick_input_candidates:
            value = self._get_xr_gesture_value(input_device, input_name, "click")
            if value != 0.0:
                return value
        return 0.0

    def _cycle_xr_camera_mode(self) -> None:
        """Switch to the next first-person camera mode, from inside the headset.

        Which mode actually works depends on how this Kit build interprets
        ``schedule_set_camera``, and that cannot be settled from outside a live VR
        session -- the "attached" log line only means the call did not raise. Rather than
        guess, the operator cycles the modes while wearing the headset and keeps the one
        that puts them in the robot.
        """
        if self._g1_locomotion == "stationary":
            self._xr_camera_mode = "robot_head"
            self._log_xr_camera_state("stationary mode keeps the camera fixed to the robot head")
            return
        cycle = self._xr_camera_mode_cycle
        try:
            index = cycle.index(self._xr_camera_mode)
        except ValueError:
            index = -1
        self._xr_camera_mode = cycle[(index + 1) % len(cycle)]
        self._xr_camera_states_logged.clear()  # let the new mode report itself
        self._xr_calibrated = False  # and re-measure the rig for it
        self._xr_calibration_samples = []
        print(
            f"[G1] VR camera mode -> {self._xr_camera_mode} "
            f"({cycle.index(self._xr_camera_mode) + 1} of {len(cycle)}; "
            "click the left thumbstick again to try the next one)",
            flush=True,
        )

    def _read_gamepad_controller_axes(self) -> tuple[float, float]:
        """Return forward and yaw commands from a normal gamepad fallback."""
        left_stick_forward = self._get_gamepad_value(carb.input.GamepadInput.LEFT_STICK_UP)
        left_stick_backward = self._get_gamepad_value(carb.input.GamepadInput.LEFT_STICK_DOWN)
        right_stick_left = self._get_gamepad_value(carb.input.GamepadInput.RIGHT_STICK_LEFT)
        right_stick_right = self._get_gamepad_value(carb.input.GamepadInput.RIGHT_STICK_RIGHT)
        right_trigger = self._get_gamepad_value(carb.input.GamepadInput.RIGHT_TRIGGER)
        button_a = self._get_gamepad_value(carb.input.GamepadInput.A)
        button_x = self._get_gamepad_value(carb.input.GamepadInput.X)

        stick_forward = left_stick_forward - left_stick_backward
        forward = self._apply_deadzone(stick_forward if abs(stick_forward) >= right_trigger else right_trigger)
        stick_yaw = self._apply_deadzone(right_stick_left - right_stick_right)
        button_yaw = self._apply_deadzone(button_x - button_a)
        yaw = stick_yaw if abs(stick_yaw) > abs(button_yaw) else button_yaw
        return forward, yaw

    def _update_controller_command(self, dt: float) -> None:
        """Poll XR actions and recording inputs, optionally allowing locomotion.

        Stationary mode still handles drop/recenter/camera buttons and updates the
        HMD recording clock, but suppresses every base command. Arm and finger input
        is read independently by the teleoperation methods later in the same tick.
        """
        if self._controller_command is None:
            return

        # Cleared before the inputs are polled; any of them may raise it this step.
        self._locomotion_brake = self._keyboard_brake
        xr_forward, xr_yaw = self._read_xr_controller_axes()
        headset_gait_forward = self._update_headset_gait_command(dt)
        if self._g1_locomotion == "stationary":
            self._controller_command[:] = 0.0
            self._latest_stick_lateral = 0.0
            return
        gamepad_forward, gamepad_yaw = self._read_gamepad_controller_axes()

        forward = xr_forward if abs(xr_forward) > abs(gamepad_forward) else gamepad_forward
        if self._headset_gait_enabled:
            forward = max(forward, headset_gait_forward)
        yaw = xr_yaw if abs(xr_yaw) > abs(gamepad_yaw) else gamepad_yaw

        if self._locomotion_brake:
            # Brake stops TRANSLATION, not rotation. It used to zero the yaw too, which
            # made the most natural way to line up on a package -- hold the stick back to
            # stay put, turn to face it -- do nothing at all. Turning in place is safe now
            # that a yaw-only command holds station (see g1.TURN_HOLD_ENABLED).
            forward = 0.0
        # Reverse is capped separately: the walking policy tracks backward commands at
        # about 40% and is less stable doing it, so full stick back is not full speed back.
        speed = self._max_forward_speed if forward >= 0.0 else self._max_backward_speed
        self._controller_command[0] = speed * forward
        # Strafe. The policy has always taken this; nothing ever sent it.
        self._controller_command[1] = self._max_lateral_speed * float(getattr(self, "_latest_stick_lateral", 0.0))
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
        if self._g1_locomotion == "stationary":
            self._keyboard_command[:] = 0.0
            self._pressed_keys.clear()
            self._keyboard_brake = False
            return True
        key = self._keyboard_event_key(event)
        if key is None:
            return True
        # Rebuild from the pressed-key set. Increment/decrement accounting can leave a
        # nonzero command after repeated presses or an unmatched release event.
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed_keys.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed_keys.discard(key)
        else:
            return True  # key repeat must not release the brake
        self._keyboard_brake = bool(self._pressed_keys.intersection({"DOWN", "NUMPAD_2"}))
        values = [0.0, 0.0, 0.0]
        for pressed in self._pressed_keys:
            for axis, value in enumerate(self._input_keyboard_mapping.get(pressed, (0.0, 0.0, 0.0))):
                values[axis] += value
        self._keyboard_command[:] = torch.tensor(values, device=self._keyboard_command.device)
        return True

    @staticmethod
    def _keyboard_event_key(event: object) -> str | None:
        """Return the key name for a keyboard event, whatever type carb hands us.

        ``event.input`` is documented as a ``carb.input.KeyboardInput`` enum, but in this
        Kit build it arrives as a plain string, so ``event.input.name`` raised
        ``AttributeError: 'str' object has no attribute 'name'`` on EVERY keypress. The
        handler died before it did anything, which killed the whole keyboard control path
        silently -- the traceback goes to the log, not the screen, and the VR controls kept
        working, so nothing pointed at it. Accept both shapes.
        """
        key = getattr(event, "input", None)
        if key is None:
            return None
        return str(getattr(key, "name", key))

    def _unsubscribe_keyboard(self):
        """Unsubscribe from keyboard events if currently subscribed."""
        if self._sub_keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None

    def physics_cleanup(self):
        """Clean up physics resources."""
        self._head_camera_update_sub = None
        self._reset_teleoperation_state()
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
        self.g1 = None
        self._physics_ready = False
        self._head_camera_transform_op = None  # handles die with the stage; never reuse them
        self._head_camera_mount_body_path = None
        self._head_camera_mount_local = None
        self._xr_anchor_op = None
        self._restore_physics_simulation_state()

    def _restore_physics_simulation_state(self) -> None:
        """Restore the physics sim device and fabric state captured in ``setup_scene``."""
        restore_physics_simulation_state(self._prev_physics_sim_device, self._prev_fabric_enabled)
        self._prev_physics_sim_device = None
        self._prev_fabric_enabled = None
