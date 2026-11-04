# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unitree G1 teleoperation with stationary, policy, and kinematic base modes.

Stationary mode anchors the pelvis to the world before physics starts. It holds
the standing posture, ignores base commands, and leaves arms/fingers articulated.

Legacy policy mode runs the bundled recurrent actor for twelve leg joints.
``physical_locomotion`` runs the pinned GR00T balance/walk pair at 50 Hz, with
its own initial pose, observation history and actuator settings. The actors own
only legs and waist and observe independently driven arms. Their 200 Hz explicit,
torque-limited body PD matches the released deployment. This mode disables root
assists and kinematic fallback; faults require the caller to pause physics.
An optional project feature adds model gravity compensation only to the fourteen
externally controlled arm joints; the learned leg/waist PD stays unchanged.
Fingers remain independently controlled.

Kinematic mode integrates the commanded base velocity with gravity disabled.
Only the legacy policy path can fall back to it when walking assets are missing
or incompatible, before joint defaults are created. Physical locomotion faults
instead remain latched until reset or initialization.

Construction authors USD variants, root topology and initial joint state before
creating an articulation view. ``initialize`` requires running physics;
``invalidate_physics_state`` discards episode state without reading destroyed
views. ``post_reset`` restores the registered spawn state and controller history.
See ``docs/humanoid-control.md`` for frames, validation, and limitations.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import carb
import isaacsim.core.experimental.utils.transform as transform_utils
import numpy as np
import omni
from isaacsim.core.deprecation_manager import import_module
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.utils.prim import get_prim_at_path
from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.storage.native import get_assets_root_path
from omni.physics.core import get_physics_simulation_interface
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

from .g1_decoupled_policy import G1DecoupledPolicy
from .g1_foot_contacts import configure_decoupled_foot_contacts


class G1TeleopRobot:
    """Unitree G1 29-DOF humanoid with dexterous hands, posed by teleoperation.

    Commands use body-frame forward/lateral velocity in m/s and yaw rate in
    rad/s as ``(v_x, v_y, w_z)``. Stage positions are in meters, joint angles
    are radians, and public base orientations use scalar-first ``wxyz`` quaternions.
    DOF arrays follow live articulation order unless explicitly mapped by name.

    Stationary mode leaves the fixed root to the solver. Legacy policy mode owns
    twelve leg targets and supplies default upper-body targets for teleoperation
    to override. Physical decoupled mode owns fifteen leg/waist targets, observes
    fourteen independently controlled arm joints, and applies explicit PD to all
    twenty-nine body motors. Fingers retain separate drives in every mode.
    Kinematic mode writes the root pose and reported velocity directly.

    The interface mirrors the policy robots in this package (``robot``, ``default_pos``,
    ``initialize()``, ``post_reset()``, ``forward(dt, command)``) so the interactive
    examples can drive it exactly like ``H1FlatTerrainPolicy``.

    Args:
        prim_path: The prim path of the robot on the stage.
        usd_path: Robot USD filepath, or None to resolve the packaged Isaac Sim asset.
        position: Initial stage position in meters, or None for the mode's spawn height.
        orientation: The initial orientation of the robot as a ``wxyz`` quaternion.
        hand_variant: ``"Inspire"`` (five-finger), ``"ThreeFinger"`` (Dex3) or ``"None"``.
        physics_variant: Physics variant to select on the asset.
        sensor_variant: ``"None"`` skips the onboard RealSense/Livox prims.
        locomotion: ``"stationary"`` for anchored manipulation, ``"policy"`` for walking,
            or ``"kinematic"`` for a movable avatar.
        walk_policy_path: Optional legacy TorchScript path. Physical mode uses
            its paired, hash-validated packaged ONNX checkpoints exclusively.
        physical_locomotion: Require unassisted, gravity-driven policy locomotion.
            Valid only with ``locomotion="policy"``. The caller must pause physics
            when initialization or ``forward`` raises a locomotion fault.
            ``G1_WALK_POLICY`` may be unset or ``decoupled``; other values fail.
        arm_gravity_compensation: Add model-based holding torques to the fourteen
            external arm joints in physical decoupled mode. This project option
            preserves the policy's lower-body PD and all motor effort limits.
            It compensates robot-link gravity, not contact or payload forces.

    Raises:
        ValueError: If physical locomotion, checkpoint selection, or arm gravity
            compensation is requested with an incompatible mode.
    """

    #: Stationary/avatar body posture in radians. Unlisted body joints use zero;
    #: finger defaults remain separate so mimic couplings retain their authored pose.
    DEFAULT_STANDING_POSE = {
        "left_shoulder_pitch_joint": 0.20,
        "right_shoulder_pitch_joint": 0.20,
        "left_shoulder_roll_joint": 0.18,
        "right_shoulder_roll_joint": -0.18,
        "left_elbow_joint": 0.60,
        "right_elbow_joint": 0.60,
    }

    #: Stationary/avatar pelvis height in metres for this asset and standing pose.
    #: Initial foot clearance avoids ground penetration pushing against a fixed root.
    STANDING_BASE_HEIGHT = 0.80

    #: Stationary/avatar implicit PD gains by joint group. Walking modes use
    #: their own controller gains; physical decoupled mode disables implicit body PD.
    JOINT_GAINS = (
        (("hip_yaw", "hip_roll", "hip_pitch", "knee"), 150.0, 5.0),
        (("ankle",), 40.0, 2.0),
        (("waist",), 88.0, 5.0),
        (("shoulder", "elbow", "wrist"), 150.0, 10.0),
    )

    #: Simulation finger-drive stiffness in N*m/rad, damping in N*m*s/rad,
    #: and effort limit in N*m.
    #: Only existing driven joints receive these gains. Passive mimic joints
    #: retain their coupling; adding drives would introduce competing constraints.
    #: The physical-grasp scene configures that coupling separately before Play.
    FINGER_STIFFNESS = 20.0
    FINGER_DAMPING = 0.6
    FINGER_MAX_EFFORT = 10.0

    #: Simulation finger speed cap in rad/s. This override reduces actuator lag
    #: during teleoperation; it is not a measured physical-hand speed specification.
    FINGER_MAX_VELOCITY = 8.0

    #: Inspire hand: the joints that carry a drive. Their ``intermediate``/``distal``
    #: partners are PhysX mimic joints and follow automatically, so writing these six
    #: targets curls all twelve joints of the hand.
    INSPIRE_DRIVEN_FINGER_JOINTS = {
        "left": {
            "index": "L_index_proximal_joint",
            "middle": "L_middle_proximal_joint",
            "ring": "L_ring_proximal_joint",
            "little": "L_pinky_proximal_joint",
            "thumb": "L_thumb_proximal_pitch_joint",
            "thumb_yaw": "L_thumb_proximal_yaw_joint",
        },
        "right": {
            "index": "R_index_proximal_joint",
            "middle": "R_middle_proximal_joint",
            "ring": "R_ring_proximal_joint",
            "little": "R_pinky_proximal_joint",
            "thumb": "R_thumb_proximal_pitch_joint",
            "thumb_yaw": "R_thumb_proximal_yaw_joint",
        },
    }

    #: Dex3 three-finger hand equivalent. Its index/middle joints close towards their
    #: *lower* limit, which ``get_finger_joint_range`` resolves from the live limits.
    THREE_FINGER_DRIVEN_FINGER_JOINTS = {
        "left": {
            "index": "left_hand_index_0_joint",
            "middle": "left_hand_middle_0_joint",
            "thumb": "left_hand_thumb_1_joint",
            "thumb_yaw": "left_hand_thumb_0_joint",
        },
        "right": {
            "index": "right_hand_index_0_joint",
            "middle": "right_hand_middle_0_joint",
            "thumb": "right_hand_thumb_1_joint",
            "thumb_yaw": "right_hand_thumb_0_joint",
        },
    }

    # ------------------------------------------------------------------------------
    # Legacy Unitree walking policy (`locomotion="policy"`, nonphysical profile).
    #
    # The bundled `unitree_rl_gym` recurrent actor consumes 47 observations and
    # produces twelve leg actions. Its observation/action scales and joint order
    # below form a checkpoint contract; they do not configure the GR00T actors.
    # Waist and arms receive posture targets that the teleoperator can override.
    # ------------------------------------------------------------------------------

    #: The 12 leg joints in the order the policy expects (the training URDF's joint
    #: order). This is NOT the order PhysX reports in ``dof_names`` — that interleaves
    #: left/right and the waist — so every read and write is remapped through the names.
    WALK_LEG_JOINT_ORDER = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )
    #: Stance the policy's actions are offsets from — a slight crouch.
    WALK_LEG_DEFAULT_ANGLES = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2
    WALK_LEG_STIFFNESS = (100.0, 100.0, 100.0, 150.0, 40.0, 40.0) * 2
    WALK_LEG_DAMPING = (2.0, 2.0, 2.0, 4.0, 2.0, 2.0) * 2
    #: Gains for the joints the policy does not drive, held at their posture targets.
    WALK_WAIST_STIFFNESS, WALK_WAIST_DAMPING = 300.0, 3.0
    WALK_ARM_STIFFNESS = (100.0, 100.0, 50.0, 50.0, 20.0, 20.0, 20.0)
    WALK_ARM_DAMPING = (2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0)
    WALK_ARM_JOINT_ORDER = (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )

    #: Observation scales, action scale and gait clock, all from the deployment config.
    WALK_ANG_VEL_SCALE = 0.25
    WALK_DOF_POS_SCALE = 1.0
    WALK_DOF_VEL_SCALE = 0.05
    WALK_ACTION_SCALE = 0.25
    WALK_CMD_SCALE = (2.0, 2.0, 0.25)
    WALK_GAIT_PERIOD = 0.8
    WALK_OBS_DIM = 47
    WALK_NUM_ACTIONS = 12
    #: Legacy command caps in m/s, m/s, and rad/s, applied before observation scaling.
    WALK_MAX_COMMAND = (0.8, 0.5, 1.57)
    #: Legacy actor rate in simulation Hz. The scheduler accumulates `dt` so
    #: noninteger physics/control ratios preserve phase without replaying one
    #: observation through the stateful actor multiple times after a long step.
    WALK_CONTROL_HZ = 50.0
    #: Pelvis spawn height used for policy mode — the training config's `init_state.pos`.
    WALK_BASE_HEIGHT = 0.80

    # Legacy station keeping corrects measured drift when translation is idle.
    # These assists are bypassed entirely by physical decoupled locomotion, where
    # zero command still runs the balance actor and never pins or teleports the root.
    STATION_KEEP_ENABLED = True
    STATION_KEEP_DEADBAND = 0.05  # command magnitude below which the operator counts as idle
    STATION_KEEP_GAIN = 1.2  # how hard to cancel measured linear drift
    STATION_KEEP_YAW_GAIN = 0.8  # ... and measured yaw drift
    STATION_KEEP_MAX_LINEAR = 0.35  # m/s: cap, so a correction never becomes a lurch
    STATION_KEEP_MAX_YAW = 0.5  # rad/s

    # Legacy idle hold captures and rewrites the base pose and leg targets after
    # the delay. This is explicit assistance, not learned balance or a contact hold.
    STATION_HOLD_ENABLED = True
    STATION_HOLD_DELAY = 0.4  # s of idle before the hold engages

    # Legacy turn assistance cancels horizontal drift and corrects root XY toward
    # the turn's starting point. It preserves the current root yaw and height in
    # those writes, but remains an assisted trajectory rather than free locomotion.
    TURN_HOLD_ENABLED = True
    TURN_HOLD_DEADBAND = 0.05  # translation command below this = 'I only asked to turn'
    TURN_HOLD_SETTLE_TIME = 0.20  # s: time constant pulling the base back to the anchor
    TURN_HOLD_MAX_CORRECTION = 1.5  # m/s: cap, so a large excursion is eased back, not yanked

    #: Fraction of the joint's travel used at full curl, per finger role. The thumb stops
    #: short of its hard stop so a closed fist does not drive the thumb into the fingers.
    FINGER_CLOSE_FRACTION = {
        "index": 0.95,
        "middle": 0.95,
        "ring": 0.95,
        "little": 0.95,
        "thumb": 0.90,
        "thumb_yaw": 0.75,
    }

    def __init__(
        self,
        prim_path: str,
        usd_path: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        hand_variant: str = "Inspire",
        physics_variant: str = "SimplifiedPhysX",
        sensor_variant: str = "None",
        locomotion: str = "kinematic",
        walk_policy_path: str | None = None,
        physical_locomotion: bool = False,
        arm_gravity_compensation: bool = False,
    ):
        if physical_locomotion and locomotion != "policy":
            raise ValueError("physical_locomotion requires locomotion='policy'")
        if arm_gravity_compensation and not physical_locomotion:
            raise ValueError("arm_gravity_compensation requires physical_locomotion=True")
        self._physical_locomotion = bool(physical_locomotion)
        self._arm_gravity_compensation = bool(arm_gravity_compensation)
        self._locomotion_fault = None
        self._velocity_policy = None
        self._walk_policy_kind = locomotion
        if physical_locomotion:
            selection = os.environ.get("G1_WALK_POLICY", "decoupled").strip().lower()
            self._walk_policy_kind = selection
            if selection != "decoupled":
                raise ValueError(f"Unknown physical walking policy {selection!r}; expected decoupled")
            if walk_policy_path is not None:
                raise ValueError("Decoupled WBC requires its paired packaged checkpoints, not walk_policy_path")
            self._velocity_policy = G1DecoupledPolicy()
        assets_root_path = get_assets_root_path()
        if usd_path is None:
            usd_path = assets_root_path + "/Isaac/Robots/Unitree/G1/g1.usd"

        self._prim_path = prim_path
        self._hand_variant = hand_variant
        self._locomotion = locomotion if locomotion in ("stationary", "kinematic", "policy") else "kinematic"
        if self._locomotion != locomotion:
            carb.log_warn(f"G1TeleopRobot: unknown locomotion mode {locomotion!r}; falling back to kinematic.")
        self._walk_policy_path = walk_policy_path
        self._walk_policy = None

        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            prim.GetReferences().AddReference(usd_path)

        # Variants must be selected before the Articulation is constructed: they decide
        # which prims carry the articulation root and the finger joints.
        self._select_variants(physics_variant, hand_variant, sensor_variant)
        self._ensure_physx_articulation_api()
        # The assisted stationary/avatar modes disable robot-link gravity so arms
        # can follow tracking without balancing the torso. Objects keep their gravity.
        self._disable_gravity = self._locomotion != "policy"
        self._author_disabled_gravity(self._disable_gravity)

        if position is None:
            height = self.WALK_BASE_HEIGHT if self._locomotion == "policy" else self.STANDING_BASE_HEIGHT
            position = [0.0, 0.0, height]
        if orientation is None:
            orientation = [1.0, 0.0, 0.0, 0.0]

        # The spawn pose is authored on the wrapper Xform rather than passed to the
        # Articulation. On this asset the articulation root is the *pelvis*, a prim
        # inside the reference: positioning it directly would author transform ops into
        # the referenced robot instead of onto the stage prim this example owns.
        self._set_spawn_transform(position, orientation)
        self._configure_stationary_anchor()
        self._author_stationary_arm_pose()
        self._author_physical_walk_pose()
        if self.uses_explicit_body_efforts:
            configure_decoupled_foot_contacts(omni.usd.get_context().get_stage(), prim_path)
        self.robot = Articulation(paths=prim_path)

        self.default_pos = None
        self.default_vel = None
        self._default_dof_positions = None
        self._body_dof_indices = []
        self._finger_dof_indices = {}
        self._finger_open_closed = {}
        self._spawn_position = list(position)
        self._spawn_orientation = list(orientation)
        self._spawn_yaw = self._yaw_from_quaternion(orientation)
        self._base_position = list(position)
        self._base_yaw = self._spawn_yaw
        self._initialized = False
        # Walking-policy state
        self._leg_dof_indices = []
        self._posture_dof_indices = []
        self._walk_previous_action = None
        self._walk_leg_defaults_tensor = None
        self._walk_step_counter = 0
        self._walk_time = 0.0
        self._walk_next_control_time = None
        self._walk_step_error_logged = False
        self._idle_time = 0.0  # seconds the operator has asked for nothing
        self._hold_pose = None  # (position, orientation) captured when the hold engaged
        self._hold_leg_targets = None
        self._turn_anchor_xy = None  # world XY captured when a yaw-only command started

    """
    Asset setup.
    """

    def _select_variants(self, physics_variant: str, hand_variant: str, sensor_variant: str) -> None:
        """Select the Physics, Sensor and per-hand variants on the G1 asset."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._prim_path)
        if not prim.IsValid():
            return

        # "Physics" first: its variant also drives the inner Physics variant of the two
        # hand prims, so selecting a hand before it would be overwritten.
        for set_name, selection in (
            ("Physics", physics_variant),
            ("Sensor", sensor_variant),
            ("left_hand", hand_variant),
            ("right_hand", hand_variant),
            ("Thor", "None"),
        ):
            variant_set = prim.GetVariantSets().GetVariantSet(set_name)
            if not variant_set:
                continue
            available = list(variant_set.GetVariantNames())
            match = next((name for name in available if name.lower() == selection.lower()), None)
            if match is None:
                carb.log_warn(
                    f"G1TeleopRobot: variant {selection!r} not available in set {set_name!r} "
                    f"on {self._prim_path} (available: {available}); left unchanged."
                )
                continue
            variant_set.SetVariantSelection(match)
        carb.log_info(
            f"G1TeleopRobot: {self._prim_path} configured with Physics={physics_variant}, "
            f"hands={hand_variant}, Sensor={sensor_variant}"
        )

    def _set_spawn_transform(self, position: list[float], orientation: list[float]) -> None:
        """Author the spawn pose on the wrapper Xform prim holding the asset reference."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._prim_path)
        if not prim.IsValid():
            return
        # One double-precision transform op rather than translate + orient: the G1 asset
        # already authors a `quatd` xformOp:orient on its root, and AddOrientOp() would
        # ask for float precision on that existing attribute and raise.
        rotation = Gf.Matrix4d(1.0).SetRotate(
            Gf.Quatd(float(orientation[0]), Gf.Vec3d(*[float(value) for value in orientation[1:]]))
        )
        translation = Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*[float(value) for value in position]))

        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(rotation * translation)

    def _ensure_physx_articulation_api(self) -> None:
        """Apply ``PhysxArticulationAPI`` to whichever prim carries the articulation root."""
        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(self._prim_path) if stage is not None else None
        if not (root_prim and root_prim.IsValid()):
            return
        for prim in Usd.PrimRange(root_prim):
            if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                continue
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                continue
            PhysxSchema.PhysxArticulationAPI.Apply(prim)

    @staticmethod
    def _move_articulation_root(source, destination) -> None:
        """Move root schemas while preserving authored articulation solver settings."""
        UsdPhysics.ArticulationRootAPI.Apply(destination)
        PhysxSchema.PhysxArticulationAPI.Apply(destination)
        for attribute in source.GetAttributes():
            if attribute.GetName().startswith("physxArticulation:") and attribute.HasAuthoredValueOpinion():
                destination.GetAttribute(attribute.GetName()).Set(attribute.Get())
        source.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        source.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

    def _configure_stationary_anchor(self) -> None:
        """Configure a fixed-base articulation before creating its tensor view.

        A zero walking command still runs the balance actor and permits root drift.
        Instead, the fixed joint connects the pelvis to the world and carries the
        articulation root API. It is part of the articulation, not a separately
        solved grasp constraint. This prevents base motion from arm/object forces
        without teleporting the base on every step or freezing the arm joints.
        """
        stage = omni.usd.get_context().get_stage()
        wrapper = stage.GetPrimAtPath(self._prim_path)
        anchor_path = f"{self._prim_path}/G1_StandingAnchor"
        existing = stage.GetPrimAtPath(anchor_path)
        if existing.IsValid():
            # Persist the source root path so recreating the wrapper or loading a
            # saved stage can restore a non-stationary mode without two roots.
            source_path = existing.GetCustomDataByKey("g1StandingOriginalRoot")
            if not source_path:
                raise ValueError(f"G1TeleopRobot: refusing to replace an unrecognized anchor at {anchor_path}")
            if self._locomotion == "stationary":
                return
            original = stage.GetPrimAtPath(source_path)
            if not original.IsValid():
                raise ValueError(f"G1TeleopRobot: original articulation root is missing: {source_path}")
            self._move_articulation_root(existing, original)
            stage.RemovePrim(anchor_path)
            return
        if self._locomotion != "stationary":
            return

        roots = [prim for prim in Usd.PrimRange(wrapper) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1:
            raise ValueError(f"G1TeleopRobot: stationary mode needs exactly one articulation root, found {len(roots)}")
        original = roots[0]
        pelvis = (
            original if original.HasAPI(UsdPhysics.RigidBodyAPI) else stage.GetPrimAtPath(f"{self._prim_path}/pelvis")
        )
        if not pelvis.IsValid() or not pelvis.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("G1TeleopRobot: cannot anchor stationary mode without a rigid pelvis link")

        # Author coincident frames for inspectability; fixed articulation roots use
        # the root link's initial world transform as their actual anchor pose.
        world = UsdGeom.XformCache().GetLocalToWorldTransform(pelvis)
        anchor = UsdPhysics.FixedJoint.Define(stage, anchor_path)
        anchor.CreateBody0Rel().SetTargets([])  # empty body relationship means world
        anchor.CreateBody1Rel().SetTargets([pelvis.GetPath()])
        anchor.CreateLocalPos0Attr().Set(Gf.Vec3f(world.ExtractTranslation()))
        anchor.CreateLocalRot0Attr().Set(Gf.Quatf(world.ExtractRotationQuat().GetNormalized()))
        anchor.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        anchor.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        anchor.CreateJointEnabledAttr().Set(True)
        anchor.CreateExcludeFromArticulationAttr().Set(False)
        anchor.GetPrim().SetCustomDataByKey("g1StandingOriginalRoot", str(original.GetPath()))
        self._move_articulation_root(original, anchor.GetPrim())
        carb.log_info("G1TeleopRobot: stationary base anchored; walking and turning disabled, arms/hands active")

    def _author_disabled_gravity(self, disabled: bool = True) -> None:
        """Set ``physxRigidBody:disableGravity`` on every link for the selected mode.

        Authored in USD before the simulation starts rather than set through the tensor
        API afterwards, because the runtime call does not take on this articulation: with
        it alone the torso and legs still sagged, drifting ~0.18 rad into the roll joints
        over a few seconds as the drives balanced against a gravity that was still on.
        """
        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(self._prim_path) if stage is not None else None
        if not (root_prim and root_prim.IsValid()):
            return
        count = 0
        for prim in Usd.PrimRange(root_prim):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            attr = physx_body.GetDisableGravityAttr()
            if not attr:
                attr = physx_body.CreateDisableGravityAttr()
            attr.Set(disabled)
            count += 1
        carb.log_info(f"G1TeleopRobot: gravity disabled={disabled} on {count} links of {self._prim_path}")

    def _author_stationary_arm_pose(self) -> None:
        """Place stationary arms in their ready pose before the first solver step.

        The referenced G1 can contain an extended arm pose. Waiting until
        ``initialize()`` sets tensor positions is too late: it runs after the
        first physics step, when those hands can already strike tabletop props.
        PhysX's articulation-only JointStateAPI supplies the initial state in
        degrees; this agrees with the radian defaults used by ``initialize()``.

        Author only the fourteen named arm joints in stationary mode. Finger
        states and mimic constraints retain their asset configuration. Existing
        arm drives receive the matching starting targets without changing gains,
        limits, or drive types. Never call this as part of per-frame control.
        """
        if self._locomotion != "stationary":
            return
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(self._prim_path) if stage is not None else None
        if not (root and root.IsValid()):
            return
        names = {f"{side}_{role}_joint" for side in ("left", "right") for role in self.WALK_ARM_JOINT_ORDER}
        count = 0
        for prim in Usd.PrimRange(root):
            name = prim.GetName()
            if name not in names or not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
            position_deg = math.degrees(self.DEFAULT_STANDING_POSE.get(name, 0.0))
            state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
            state.CreatePositionAttr().Set(position_deg)
            state.CreateVelocityAttr().Set(0.0)
            drive = UsdPhysics.DriveAPI(prim, "angular")
            if drive:
                drive.CreateTargetPositionAttr().Set(position_deg)
                drive.CreateTargetVelocityAttr().Set(0.0)
            count += 1
        carb.log_info(f"G1TeleopRobot: authored the stationary starting pose for {count} arm joints before physics")

    def _author_physical_walk_pose(self) -> None:
        """Author the physical policy's initial joint state before the first solve.

        All body joints use the checkpoint's initial pose and actuator settings
        so loading cannot sweep an unrelated authored pose across nearby props.
        Finger and mimic states remain the hand asset's authored configuration.
        JointStateAPI uses degrees, while the policy and tensor API use radians.
        """
        if not getattr(self, "_physical_locomotion", False):
            return
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(self._prim_path) if stage is not None else None
        if not (root and root.IsValid()):
            raise ValueError("G1TeleopRobot: physical walking requires a valid robot prim")
        policy = self._velocity_policy
        defaults = dict(zip(policy.joint_names, getattr(policy, "initial_positions", policy.default_positions)))
        for prim in Usd.PrimRange(root):
            name = prim.GetName()
            if name not in defaults or not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
            position_deg = math.degrees(defaults[name])
            state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
            state.CreatePositionAttr().Set(position_deg)
            state.CreateVelocityAttr().Set(0.0)
            drive = UsdPhysics.DriveAPI(prim, "angular")
            if drive:
                drive.CreateTargetPositionAttr().Set(position_deg)
                drive.CreateTargetVelocityAttr().Set(0.0)
                index = policy.joint_names.index(name)
                # USD angular drive gains use degrees; tensor gains use radians.
                drive.CreateStiffnessAttr().Set(float(policy.stiffness[index]) * math.pi / 180.0)
                drive.CreateDampingAttr().Set(float(policy.damping[index]) * math.pi / 180.0)
                if policy.effort_limits is not None:
                    drive.CreateMaxForceAttr().Set(float(policy.effort_limits[index]))
                drive.CreateTypeAttr().Set("force")
            index = policy.joint_names.index(name)
            joint_api = PhysxSchema.PhysxJointAPI.Apply(prim)
            if policy.armature is not None:
                joint_api.CreateArmatureAttr().Set(float(policy.armature[index]))
            if policy.velocity_limits is not None:
                joint_api.CreateMaxJointVelocityAttr().Set(math.degrees(float(policy.velocity_limits[index])))

    """
    Lifecycle.
    """

    def initialize(self) -> None:
        """Configure the selected controller and register a complete spawn/reset state.

        Call only after physics has created a valid articulation view. Missing
        views or DOF names leave the robot uninitialized without accessing drives.
        Policy selection precedes joint defaults so legacy fallback receives the
        correct avatar posture and gravity. Finger poses remain asset-derived.
        Physical mode finally switches body motors to explicit effort control;
        its position targets are a command buffer for that motor loop.

        Raises:
            RuntimeError: If physical controller loading or actuator setup fails.
        """
        self._initialized = False
        # Stop destroys tensor views before Play recreates them. A callback that
        # arrives during that transition must not access the preceding episode.
        if not self.robot.is_physics_tensor_entity_valid():
            return
        dof_names = list(self.robot.dof_names)
        if not dof_names:
            carb.log_warn("G1TeleopRobot: articulation reports no DOFs; initialization skipped.")
            return

        finger_names = self._resolve_finger_joint_names(dof_names)
        self._body_dof_indices = [i for i, name in enumerate(dof_names) if name not in finger_names]
        # Resolve the actual mode before choosing posture and gravity. A missing or
        # incompatible checkpoint must not leave a gravity-driven crouch in avatar mode.
        if self._locomotion == "policy":
            self._configure_walk_policy(dof_names)

        # Establish body force drives before mode-specific gains/control settings.
        # Physical decoupled mode switches these to effort control below. The
        # `set_dof_drive_types` USD edit must be flushed before tensor-backed calls.
        self.robot.set_dof_drive_types("force", dof_indices=self._body_dof_indices)
        get_physics_simulation_interface().flush_changes()
        self.robot.switch_dof_control_mode("position", dof_indices=self._body_dof_indices)

        # Finger joints keep whatever pose the hand loaded in. Forcing them to zero along
        # with the body would snap the mimic-coupled joints to a position their coupling
        # disagrees with, and the constraint then fights the drives: the two hands ended
        # up resting in visibly different poses before a single command was sent.
        try:
            default_positions = np.asarray(self.robot.get_dof_positions().numpy()[0], dtype=np.float32).copy()
        except Exception:
            default_positions = np.zeros(len(dof_names), dtype=np.float32)
        for index in self._body_dof_indices:
            default_positions[index] = 0.0
        if getattr(self, "_physical_locomotion", False):
            policy = self._velocity_policy
            initial_positions = getattr(policy, "initial_positions", policy.default_positions)
            for name, angle in zip(policy.joint_names, initial_positions):
                default_positions[dof_names.index(name)] = angle
        elif self._locomotion == "policy":
            # Match the pose the policy was trained around: the crouched leg stance its
            # actions are offsets from, and waist/arms at zero as its deployment config
            # holds them. The teleoperation layer takes the arms over from there.
            for name, angle in zip(self.WALK_LEG_JOINT_ORDER, self.WALK_LEG_DEFAULT_ANGLES):
                if name in dof_names:
                    default_positions[dof_names.index(name)] = angle
        else:
            for name, value in self.DEFAULT_STANDING_POSE.items():
                if name in dof_names:
                    default_positions[dof_names.index(name)] = value
        default_velocities = np.zeros(len(dof_names), dtype=np.float32)

        self.robot.set_dof_positions([default_positions[self._body_dof_indices]], dof_indices=self._body_dof_indices)
        self.robot.set_dof_velocities([default_velocities])
        self.robot.set_default_state(
            positions=[self._spawn_position],
            orientations=[self._spawn_orientation],
            linear_velocities=[[0.0, 0.0, 0.0]],
            angular_velocities=[[0.0, 0.0, 0.0]],
            dof_positions=[default_positions],
            dof_velocities=[default_velocities],
        )

        torch = import_module("torch")
        device = torch.device(str(self.robot._device))
        self._default_dof_positions = default_positions
        self.default_pos = torch.tensor(default_positions, device=device)
        self.default_vel = torch.tensor(default_velocities, device=device)

        self._apply_body_joint_gains(dof_names)
        if self.uses_explicit_body_efforts:
            # Position targets remain the shared command buffer: the actor owns
            # legs/waist and teleoperation owns arms. Only explicit PD actuates it.
            self.robot.set_dof_position_targets(
                [default_positions[self._velocity_dof_indices]], dof_indices=self._velocity_dof_indices
            )
        self._configure_finger_dofs(dof_names)
        self._apply_articulation_properties()

        if self._disable_gravity:
            # Stationary and kinematic avatars disable robot-link gravity. The
            # former has a fixed root and the latter writes its root pose; objects
            # retain gravity in both cases. Physical policy mode never takes this path.
            try:
                self.robot.set_link_enabled_gravities([[False] * self.robot.num_links])
            except Exception as e:
                carb.log_warn(f"G1TeleopRobot: could not disable link gravity: {e}")

        self._base_position = list(self._spawn_position)
        self._base_yaw = self._spawn_yaw
        self._initialized = True
        carb.log_info(f"G1TeleopRobot: initialized with {len(dof_names)} DOFs: {dof_names}")

    def invalidate_physics_state(self) -> None:
        """Discard episode control state after Stop without touching dead tensors.

        The owning example calls this on timeline Stop and when a tensor view
        becomes unavailable. A subsequent Play must initialize the new physics
        episode before driving joints. Faults remain latched until explicit reset.
        """
        self._initialized = False
        self._reset_walk_policy_state(clear_fault=False)

    def post_reset(self) -> None:
        """Restore the registered spawn state, controller history, and zero body efforts.

        Initialization must already have resolved the current articulation.
        An invalid view preserves the fault latch and requires fresh initialization.

        Raises:
            RuntimeError: If the articulation is uninitialized or its view is invalid.
        """
        if not self._initialized or not self.robot.is_physics_tensor_entity_valid():
            self.invalidate_physics_state()
            raise RuntimeError("G1 reset requires an initialized, valid physics articulation")
        self.robot.reset_to_default_state()
        self._base_position = list(self._spawn_position)
        self._base_yaw = self._spawn_yaw
        self._reset_walk_policy_state()
        if self.uses_explicit_body_efforts:
            self.robot.set_dof_efforts(np.zeros((1, 29), dtype=np.float32), dof_indices=self._velocity_dof_indices)

    def forward(self, dt: float, command: object, *, defer_body_efforts: bool = False) -> None:
        """Advance the base and hold the posture the operator is not driving.

        Args:
            dt: Physics timestep in seconds.
            command: Base command velocities ``(v_x, v_y, w_z)`` in m/s and rad/s.
            defer_body_efforts: Let an integrated teleoperator update arm targets
                before calling ``apply_body_efforts()`` once in the same physics
                callback. The caller must perform that final update before the
                next simulation step. Standalone callers keep the default.

        Raises:
            RuntimeError: If physical locomotion has a latched fault, no active
                actor, invalid actor output, or a failed motor-effort update.
        """
        physical = getattr(self, "_physical_locomotion", False)
        if physical and self.locomotion_fault:
            raise RuntimeError(self.locomotion_fault)
        if not self._initialized or not math.isfinite(dt) or dt <= 0.0:
            return
        if not self.robot.is_physics_tensor_entity_valid():
            self.invalidate_physics_state()
            return
        if physical and (self._locomotion != "policy" or self._walk_policy is None):
            self._fail_physical_locomotion("Physical walking has no active policy")
        if self._locomotion == "stationary":
            # The solver owns the world anchor. Never integrate/teleport the base,
            # even if a caller accidentally supplies a nonzero velocity command.
            posture_indices = self._body_dof_indices
        elif self._locomotion == "policy" and self._walk_policy is not None:
            if physical:
                # Zero command still needs the balance actor. No root transforms,
                # velocity overrides, or idle constraints are written in this mode.
                self._step_velocity_policy(dt, command)
            elif not self._update_idle_hold(dt, command):
                self._step_walk_policy(dt, command)
                self._update_turn_in_place(dt, command)
            posture_indices = self._posture_dof_indices
        else:
            self._integrate_base_command(dt, command)
            self._write_base_pose(command)
            posture_indices = self._body_dof_indices
        if self._default_dof_positions is not None and posture_indices:
            # Assisted modes rewrite their body/posture defaults before the caller
            # overrides arm targets. Legacy policy legs are excluded. Physical
            # decoupled mode leaves this list empty, preserving independent arm targets.
            self.robot.set_dof_position_targets(
                [self._default_dof_positions[posture_indices]], dof_indices=posture_indices
            )
        if not defer_body_efforts:
            self.apply_body_efforts()

    def apply_body_efforts(self) -> None:
        """Refresh decoupled-WBC motor torques from the latest body targets.

        Call once per 200 Hz physics step, after updating any arm targets. Forward
        also calls this for standalone users unless ``defer_body_efforts=True``.
        A later call simply replaces the same effort buffer and does not integrate
        or apply an additional impulse.
        Fingers retain their independent drives. No root force is ever written.
        Optional arm gravity feedforward is added before the same motor clipping;
        it never changes the fifteen actor-owned leg/waist efforts.
        """
        if not self.uses_explicit_body_efforts:
            return
        if self.locomotion_fault:
            raise RuntimeError(self.locomotion_fault)
        if not self._initialized:
            return
        if not self.robot.is_physics_tensor_entity_valid():
            self.invalidate_physics_state()
            return
        indices = getattr(self, "_velocity_dof_indices", [])
        try:
            if len(indices) != 29 or len(set(indices)) != 29:
                raise ValueError("Explicit body PD requires 29 uniquely mapped motor joints")
            policy = self._velocity_policy
            stiffness = np.asarray(policy.stiffness, dtype=np.float64)
            damping = np.asarray(policy.damping, dtype=np.float64)
            limits = np.asarray(policy.effort_limits, dtype=np.float64)
            for label, values in (("stiffness", stiffness), ("damping", damping), ("effort limits", limits)):
                if values.shape != (29,) or not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError(f"Explicit body PD requires 29 finite nonnegative {label}")
            measurements = (
                self.robot.get_dof_position_targets(dof_indices=indices),
                self.robot.get_dof_positions(dof_indices=indices),
                self.robot.get_dof_velocities(dof_indices=indices),
            )
            targets, positions, velocities = (np.asarray(value.numpy(), dtype=np.float64) for value in measurements)
            for values in (targets, positions, velocities):
                if values.shape != (1, 29) or not np.isfinite(values).all():
                    raise ValueError("Explicit body PD received invalid targets, positions, or velocities")
            # Match the released MuJoCo motor loop: desired velocity is zero,
            # with one combined position/damping effort clipped at each motor.
            raw = stiffness * (targets - positions) - damping * velocities
            if self.arm_gravity_compensation_enabled:
                raw[:, 15:] += self._read_arm_gravity_compensation()
            if not np.isfinite(raw).all():
                raise ValueError("Explicit body PD produced nonfinite torques")
            efforts = np.clip(raw, -limits, limits).astype(np.float32)
            self.robot.set_dof_efforts(efforts, dof_indices=indices)
        except Exception as error:
            # Do not leave the preceding valid step's torque latched on a bad
            # input. The caller must still pause physics until an explicit reset.
            try:
                self.robot.set_dof_efforts(np.zeros((1, len(indices)), dtype=np.float32), dof_indices=indices)
            except Exception:
                pass
            self._fail_physical_locomotion(f"Physical walking body effort update failed: {error}")

    def _read_arm_gravity_compensation(self) -> np.ndarray:
        """Read fresh joint-space gravity torques for the fourteen external arms.

        Native PhysX orders floating-base generalized forces as six root force/
        torque entries followed by joint DOFs. The installed 6.1 experimental
        DOF helper does not offset its slice, so read the complete native tensor
        and validate its fixed/floating shape before selecting named arm joints.
        See PhysX 5.6.1 Articulations, ``Gravity Compensation Force``.

        Values already counter gravity and are in each revolute joint's positive
        direction, in N*m. Do not negate them, rotate them, or multiply by gravity
        again. Root and finger entries are never returned. Model gravity excludes
        contacts/payload loads and is not force-sensor or pressure feedback.

        Raises:
            ValueError: The native layout, policy mapping, or forces are invalid.
        """
        policy = self._velocity_policy
        native_names = tuple(self.robot.dof_names)
        count = int(self.robot.num_dofs)
        indices = tuple(self._velocity_dof_indices)
        expected_arm_names = tuple(
            f"{side}_{joint}_joint" for side in ("left", "right") for joint in self.WALK_ARM_JOINT_ORDER
        )
        if (
            len(native_names) != count
            or len(set(native_names)) != count
            or len(indices) != 29
            or len(set(indices)) != 29
            or any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or not 0 <= index < count
                for index in indices
            )
            or tuple(policy.joint_names[15:]) != expected_arm_names
            or tuple(policy.action_joint_names) != tuple(policy.joint_names[:15])
            or tuple(native_names[index] for index in indices) != tuple(policy.joint_names)
        ):
            raise ValueError("Arm gravity compensation requires unique named native/policy DOF mappings")
        view = self.robot._physics_articulation_view
        fixed_base = view.shared_metatype.fixed_base
        if not isinstance(fixed_base, (bool, np.bool_)):
            raise ValueError("Arm gravity compensation requires a known fixed/floating base layout")
        offset = 0 if fixed_base else 6
        forces = np.asarray(view.get_gravity_compensation_forces().numpy(), dtype=np.float64)
        if forces.shape != (1, count + offset) or not np.isfinite(forces).all():
            raise ValueError("Arm gravity compensation received an invalid native gravity tensor")
        return forces[:, [index + offset for index in indices[15:]]]

    """
    Walking policy (locomotion="policy").
    """

    def _configure_walk_policy(self, dof_names: list[str]) -> None:
        """Load the selected actor and map its joints onto the live articulation."""
        try:
            if getattr(self, "_physical_locomotion", False):
                self._configure_velocity_policy(dof_names)
            else:
                self._load_walk_policy(dof_names)
        except Exception as error:
            if getattr(self, "_physical_locomotion", False):
                self._fail_physical_locomotion(f"Physical walking initialization failed: {error}")
            raise

    def _configure_velocity_policy(self, dof_names: list[str]) -> None:
        """Map observed and driven joints separately; fingers never belong to an actor."""
        policy = self._velocity_policy
        self._velocity_dof_indices = policy.resolve_joint_indices(dof_names)
        action_names = getattr(policy, "action_joint_names", policy.joint_names)
        self._velocity_action_dof_indices = [dof_names.index(name) for name in action_names]
        self._leg_dof_indices = [dof_names.index(name) for name in self.WALK_LEG_JOINT_ORDER]
        self._posture_dof_indices = []  # The teleop layer independently owns any non-actor arms.
        lower, upper = self.robot.get_dof_limits(dof_indices=self._velocity_action_dof_indices)
        self._velocity_lower_limits = np.asarray(lower.numpy()[0], dtype=np.float32)
        self._velocity_upper_limits = np.asarray(upper.numpy()[0], dtype=np.float32)
        if (
            self._velocity_lower_limits.shape != (len(action_names),)
            or self._velocity_upper_limits.shape != (len(action_names),)
            or not np.isfinite(self._velocity_lower_limits).all()
            or not np.isfinite(self._velocity_upper_limits).all()
            or np.any(self._velocity_lower_limits > self._velocity_upper_limits)
        ):
            raise ValueError("Physical walking requires finite ordered joint limits for every action")
        policy.load()
        self._walk_policy = policy
        self._reset_walk_policy_state()
        carb.log_info(
            f"G1TeleopRobot: loaded {type(policy).__name__}, {len(action_names)} actions; CPU ONNX, no root assists"
        )

    def _step_velocity_policy(self, dt: float, command: object) -> None:
        """Run the physical actor at its simulation-time period using live measurements.

        At most one inference runs per callback. A late callback skips overdue
        periods instead of feeding repeated observations into controller history.
        World angular velocity is rotated into body axes; normalized scalar-first
        base orientation also supplies projected gravity. Only actor-owned target
        indices are written here; the caller updates arm targets and motor efforts.
        """
        if self.locomotion_fault:
            raise RuntimeError(self.locomotion_fault)
        self._walk_time += dt
        self._walk_step_counter += 1
        period = self._velocity_policy.control_dt
        if self._walk_next_control_time is not None and self._walk_time + 1e-9 < self._walk_next_control_time:
            return
        if self._walk_next_control_time is None:
            self._walk_next_control_time = self._walk_time + period
        else:
            skipped = math.floor((self._walk_time - self._walk_next_control_time + 1e-9) / period) + 1
            self._walk_next_control_time += skipped * period
        try:
            _, orientation = self.robot.get_world_poses()
            quat = np.asarray(orientation.numpy()[0], dtype=np.float64)
            if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
                raise ValueError("Physical walking base orientation is invalid")
            qw, qx, qy, qz = quat / np.linalg.norm(quat)
            rotation = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ]
            )
            _, angular_world = self.robot.get_velocities()
            angular_body = rotation.T @ np.asarray(angular_world.numpy()[0], dtype=np.float32)
            gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
            positions = self.robot.get_dof_positions(dof_indices=self._velocity_dof_indices).numpy()[0]
            velocities = self.robot.get_dof_velocities(dof_indices=self._velocity_dof_indices).numpy()[0]
            targets = self._velocity_policy.infer(
                positions, velocities, angular_body, gravity_body, self._command_to_floats(command)
            )
            targets = np.clip(targets, self._velocity_lower_limits, self._velocity_upper_limits)
            self.robot.set_dof_position_targets(
                [targets],
                dof_indices=getattr(self, "_velocity_action_dof_indices", self._velocity_dof_indices),
            )
        except Exception as error:
            self._fail_physical_locomotion(f"Physical walking policy step failed: {error}")

    def _load_walk_policy(self, dof_names: list[str]) -> None:
        """Load the legacy recurrent actor, falling back to avatar mode when unavailable."""
        torch = import_module("torch")

        missing = [name for name in self.WALK_LEG_JOINT_ORDER if name not in dof_names]
        if missing:
            carb.log_error(f"G1TeleopRobot: leg joints missing from the articulation: {missing}")
            self._activate_kinematic_fallback(f"Required walking leg joints are missing: {missing}")
            return
        # The remap that makes the whole thing work: policy vector position -> PhysX DOF
        # index, resolved by joint name. PhysX orders DOFs by tree depth
        # (left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, ...) while the
        # policy expects the training URDF's order (all left leg, then all right).
        self._leg_dof_indices = [dof_names.index(name) for name in self.WALK_LEG_JOINT_ORDER]
        leg_index_set = set(self._leg_dof_indices)
        self._posture_dof_indices = [i for i in self._body_dof_indices if i not in leg_index_set]

        policy_path = self._resolve_walk_policy_path()
        if policy_path is None:
            next_step = (
                "Physical walking cannot start."
                if getattr(self, "_physical_locomotion", False)
                else "Falling back to kinematic locomotion."
            )
            carb.log_error(
                "G1TeleopRobot: no walking policy found. Download Unitree's pretrained G1 policy "
                "(BSD-3-Clause) with:\n"
                "  curl -L -o <path>/g1_unitree_motion.pt https://raw.githubusercontent.com/"
                "unitreerobotics/unitree_rl_gym/main/deploy/pre_train/g1/motion.pt\n"
                f"{next_step}"
            )
            self._activate_kinematic_fallback("The walking checkpoint was not found")
            return

        try:
            device = torch.device(str(self.robot._device))
            self._walk_policy = torch.jit.load(str(policy_path), map_location=device)
            self._walk_policy.eval()
            # Without this cuDNN recompacts the LSTM weights on every single call and
            # warns about it 50 times a second.
            try:
                self._walk_policy.memory.flatten_parameters()
            except Exception:
                pass
        except Exception as e:
            carb.log_error(f"G1TeleopRobot: could not load walking policy {policy_path}: {e}")
            self._activate_kinematic_fallback(f"The walking checkpoint could not load: {e}")
            return

        self._walk_leg_defaults_tensor = torch.tensor(self.WALK_LEG_DEFAULT_ANGLES, dtype=torch.float32, device=device)
        self._walk_previous_action = torch.zeros(self.WALK_NUM_ACTIONS, dtype=torch.float32, device=device)
        lower, upper = self.robot.get_dof_limits(dof_indices=self._leg_dof_indices)
        self._walk_leg_lower_limits = torch.tensor(lower.numpy()[0], dtype=torch.float32, device=device)
        self._walk_leg_upper_limits = torch.tensor(upper.numpy()[0], dtype=torch.float32, device=device)
        if getattr(self, "_physical_locomotion", False):
            if (
                self._walk_leg_lower_limits.numel() != self.WALK_NUM_ACTIONS
                or self._walk_leg_upper_limits.numel() != self.WALK_NUM_ACTIONS
                or not torch.isfinite(self._walk_leg_lower_limits).all()
                or not torch.isfinite(self._walk_leg_upper_limits).all()
                or not (self._walk_leg_lower_limits <= self._walk_leg_upper_limits).all()
            ):
                raise ValueError("Physical walking requires twelve finite, ordered leg joint limits")
        self._reset_walk_policy_state()
        carb.log_info(
            f"G1TeleopRobot: loaded Unitree G1 walking policy from {policy_path}; "
            f"leg DOF indices {self._leg_dof_indices}"
        )

    @property
    def walking_policy_name(self) -> str:
        """Identify the selected controller without implying physical validation."""
        return getattr(self, "_walk_policy_kind", self._locomotion)

    @property
    def supports_arm_teleoperation(self) -> bool:
        """Return whether this mode allows teleoperation to own arm targets."""
        return not getattr(self, "_physical_locomotion", False) or isinstance(
            getattr(self, "_velocity_policy", None), G1DecoupledPolicy
        )

    @property
    def uses_explicit_body_efforts(self) -> bool:
        """Return whether this actor requires explicit clipped motor PD every physics step."""
        return getattr(self, "_physical_locomotion", False) and isinstance(
            getattr(self, "_velocity_policy", None), G1DecoupledPolicy
        )

    @property
    def arm_gravity_compensation_enabled(self) -> bool:
        """Return whether the physical arm-only model-gravity option is enabled."""
        return bool(getattr(self, "_arm_gravity_compensation", False)) and self.uses_explicit_body_efforts

    @property
    def locomotion_fault(self) -> str | None:
        """Return the latched physical policy failure, cleared only by reset/load."""
        return getattr(self, "_locomotion_fault", None)

    def _fail_physical_locomotion(self, reason: str) -> None:
        """Latch a fault without teleporting or disabling gravity; caller must pause."""
        if self.locomotion_fault is None:
            self._locomotion_fault = str(reason)
            carb.log_error(f"G1TeleopRobot: {reason}; pause physics and Reset before resuming")
        raise RuntimeError(self.locomotion_fault)

    def _activate_kinematic_fallback(self, reason: str = "Walking policy is unavailable") -> None:
        """Select avatar posture/gravity for legacy failure; fault in physical mode.

        Policy-owned joint mappings must be cleared before defaults are selected.
        Physical mode raises through its fault latch before changing root behavior.
        """
        if getattr(self, "_physical_locomotion", False):
            self._fail_physical_locomotion(reason)
        self._walk_policy = None
        self._locomotion = "kinematic"
        self._disable_gravity = True
        self._author_disabled_gravity()
        get_physics_simulation_interface().flush_changes()
        self._leg_dof_indices = []
        self._posture_dof_indices = list(self._body_dof_indices)

    def _resolve_walk_policy_path(self):
        """Return the first existing candidate path for the walking policy weights."""
        candidates = []
        if self._walk_policy_path:
            candidates.append(Path(self._walk_policy_path))
        here = Path(__file__).resolve().parent
        candidates += [
            here / "data" / "g1_unitree_motion.pt",
            Path.home() / "BehavioralCollection" / "policies" / "g1_unitree_motion.pt",
        ]
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate
            except Exception:
                continue
        return None

    def _reset_walk_policy_state(self, *, clear_fault: bool = True) -> None:
        """Reset the selected actor's history or legacy LSTM memory.

        Physical mode delegates to its actor's history reset. The legacy scripted
        ``PolicyExporterLSTM`` instead stores hidden/cell state inside the module;
        clear both so a new episode or assisted hold cannot resume an old stride.
        This helper resets Python/controller state without reading physics tensors.

        Args:
            clear_fault: Clear the fault only for an explicit reset or initialization.
                Timeline Stop invalidation discards history while retaining faults.
        """
        self._walk_step_counter = 0
        self._walk_time = 0.0
        self._walk_next_control_time = None
        self._walk_step_error_logged = False
        if clear_fault:
            self._locomotion_fault = None
        self._idle_time = 0.0
        self._hold_pose = None
        self._hold_leg_targets = None
        self._turn_anchor_xy = None
        if getattr(self, "_velocity_policy", None) is not None:
            self._velocity_policy.reset()
            return
        if self._walk_previous_action is not None:
            self._walk_previous_action.zero_()
        if self._walk_policy is None:
            return
        try:
            self._walk_policy.hidden_state[:] = 0.0
            self._walk_policy.cell_state[:] = 0.0
        except Exception as e:
            carb.log_warn(f"G1TeleopRobot: could not reset the walking policy's LSTM state: {e}")

    def _compute_walk_observation(self, command: object):
        """Build the legacy actor's 47-value observation in checkpoint order and units."""
        torch = import_module("torch")
        import warp as wp

        device = torch.device(str(self.robot._device))
        _, ang_vel_world = self.robot.get_velocities()
        _, orientations = self.robot.get_world_poses()
        quat = wp.to_torch(orientations).reshape(-1)[:4].to(torch.float32)
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]

        # Body-frame angular velocity: rotate the world reading by the base's inverse.
        R_IB = wp.to_torch(transform_utils.quaternion_to_rotation_matrix(orientations)).reshape(3, 3)
        ang_vel_body = R_IB.t() @ wp.to_torch(ang_vel_world).reshape(3).to(torch.float32)

        # Gravity direction in the body frame, using the exact expression from Unitree's
        # deployment script so the sign convention matches what the policy was trained on.
        gravity_body = torch.stack(
            [
                2.0 * (-qz * qx + qw * qy),
                -2.0 * (qz * qy + qw * qx),
                1.0 - 2.0 * (qw * qw + qz * qz),
            ]
        )

        leg_pos = wp.to_torch(self.robot.get_dof_positions()).reshape(-1)[self._leg_dof_indices].to(torch.float32)
        leg_vel = wp.to_torch(self.robot.get_dof_velocities()).reshape(-1)[self._leg_dof_indices].to(torch.float32)

        forward_speed, lateral_speed, yaw_rate = self._command_to_floats(command)
        limits = self.WALK_MAX_COMMAND
        commands = torch.tensor(
            [
                self._clamp(forward_speed, -limits[0], limits[0]) * self.WALK_CMD_SCALE[0],
                self._clamp(lateral_speed, -limits[1], limits[1]) * self.WALK_CMD_SCALE[1],
                self._clamp(yaw_rate, -limits[2], limits[2]) * self.WALK_CMD_SCALE[2],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Gait clock: the policy was trained with a periodic phase signal, which is what
        # gives it a stepping rhythm rather than a shuffle.
        phase = (self._walk_time % self.WALK_GAIT_PERIOD) / self.WALK_GAIT_PERIOD
        phase_signal = torch.tensor(
            [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
            dtype=torch.float32,
            device=device,
        )

        obs = torch.zeros(self.WALK_OBS_DIM, dtype=torch.float32, device=device)
        obs[0:3] = ang_vel_body * self.WALK_ANG_VEL_SCALE
        obs[3:6] = gravity_body
        obs[6:9] = commands
        obs[9:21] = (leg_pos - self._walk_leg_defaults_tensor) * self.WALK_DOF_POS_SCALE
        obs[21:33] = leg_vel * self.WALK_DOF_VEL_SCALE
        obs[33:45] = self._walk_previous_action
        obs[45:47] = phase_signal
        return obs

    def _update_idle_hold(self, dt: float, command: object) -> bool:
        """Apply the legacy assisted base/leg hold after sustained idle input.

        The captured root pose and zero velocities are rewritten while holding.
        Physical decoupled locomotion bypasses this helper entirely.

        Args:
            dt: Simulation timestep in seconds.
            command: Body-frame forward/lateral speed and yaw rate before assistance.

        Returns:
            True when the hold is active and has driven the robot this step, in which
            case the caller should not run the policy.

        """
        if not (self.STATION_HOLD_ENABLED and self.STATION_KEEP_ENABLED):
            return False

        magnitude = max(abs(value) for value in self._command_to_floats(command))
        if magnitude > self.STATION_KEEP_DEADBAND:
            # The operator is driving again: release, and clear the policy's memory so it
            # does not resume from a stride it took before the hold.
            self._idle_time = 0.0
            if self._hold_pose is not None:
                self._hold_pose = None
                self._hold_leg_targets = None
                self._reset_walk_policy_state()
            return False

        self._idle_time += max(float(dt), 0.0)
        if self._idle_time < self.STATION_HOLD_DELAY:
            return False

        import warp as wp

        try:
            if self._hold_pose is None:
                positions, orientations = self.robot.get_world_poses()
                self._hold_pose = (
                    np.asarray(positions.numpy()[0], dtype=np.float64).copy(),
                    np.asarray(orientations.numpy()[0], dtype=np.float64).copy(),
                )
                legs = wp.to_torch(self.robot.get_dof_positions()).reshape(-1)[self._leg_dof_indices]
                self._hold_leg_targets = legs.detach().clone()

            position, orientation = self._hold_pose
            self.robot.set_world_poses(positions=[position.tolist()], orientations=[orientation.tolist()])
            self.robot.set_velocities(linear_velocities=[[0.0, 0.0, 0.0]], angular_velocities=[[0.0, 0.0, 0.0]])
            self.robot.set_dof_position_targets(
                wp.from_torch(self._hold_leg_targets), dof_indices=self._leg_dof_indices
            )
        except Exception as e:
            carb.log_warn(f"G1TeleopRobot: idle hold failed: {e}")
            return False
        return True

    def _update_turn_in_place(self, dt: float, command: object) -> None:
        """Correct legacy root XY during yaw-only input without replacing leg targets.

        This assisted pose/velocity write follows the legacy actor and preserves
        current root height and orientation. Physical decoupled mode bypasses it.

        Args:
            dt: Physics timestep in seconds.
            command: The operator's ``(v_x, v_y, w_z)`` command, before station keeping.
        """
        if not self.TURN_HOLD_ENABLED:
            return

        forward_speed, lateral_speed, yaw_rate = self._command_to_floats(command)
        translating = max(abs(forward_speed), abs(lateral_speed)) > self.TURN_HOLD_DEADBAND
        turning = abs(yaw_rate) > self.TURN_HOLD_DEADBAND
        if translating or not turning:
            self._turn_anchor_xy = None
            return

        try:
            positions, orientations = self.robot.get_world_poses()
            position = np.asarray(positions.numpy()[0], dtype=np.float64).copy()
            orientation = np.asarray(orientations.numpy()[0], dtype=np.float64).copy()
        except Exception as e:
            carb.log_warn(f"G1TeleopRobot: turn-in-place hold could not read the base pose: {e}")
            self._turn_anchor_xy = None
            return

        if self._turn_anchor_xy is None:
            self._turn_anchor_xy = position[:2].copy()
            return

        # First-order pull back to where the turn started, capped so a large excursion
        # (say, the operator turned straight after walking) is eased in rather than
        # snapped, which would knock the gait over.
        alpha = self._clamp(float(dt) / self.TURN_HOLD_SETTLE_TIME, 0.0, 1.0)
        error_x = float(self._turn_anchor_xy[0] - position[0])
        error_y = float(self._turn_anchor_xy[1] - position[1])
        step_x, step_y = error_x * alpha, error_y * alpha
        max_step = self.TURN_HOLD_MAX_CORRECTION * max(float(dt), 0.0)
        step_length = math.hypot(step_x, step_y)
        if step_length > max_step > 0.0:
            scale = max_step / step_length
            step_x, step_y = step_x * scale, step_y * scale
        position[0] += step_x
        position[1] += step_y

        try:
            import warp as wp

            linear_world, angular_world = self.robot.get_velocities()
            linear = wp.to_torch(linear_world).reshape(-1)[:3]
            angular = wp.to_torch(angular_world).reshape(-1)[:3]
            vertical = float(linear[2])
            angular_velocity = [float(angular[0]), float(angular[1]), float(angular[2])]
        except Exception:
            vertical = 0.0
            angular_velocity = [0.0, 0.0, float(yaw_rate)]

        try:
            self.robot.set_world_poses(positions=[position.tolist()], orientations=[orientation.tolist()])
            # Zero the HORIZONTAL velocity only. Left alone it re-accumulates the creep the
            # position correction just removed; zeroing all of it would also cancel the
            # gait's vertical motion and the turn itself.
            self.robot.set_velocities(
                linear_velocities=[[0.0, 0.0, vertical]],
                angular_velocities=[angular_velocity],
            )
        except Exception as e:
            carb.log_warn(f"G1TeleopRobot: turn-in-place hold failed: {e}")

    def _station_keeping_command(self, command: object) -> object:
        """Add legacy drift correction when requested translation is near zero.

        A translating command passes through. During yaw-only input, preserve
        requested yaw while correcting linear drift; at full idle, correct yaw
        drift too. Corrections use body-frame measured velocity and bounded gains.
        """
        if not self.STATION_KEEP_ENABLED:
            return command
        forward_speed, lateral_speed, yaw_rate = self._command_to_floats(command)
        # Translation idle is the condition, not 'everything idle'. Requiring yaw to be
        # idle as well meant that the instant the operator asked to turn, nothing was
        # cancelling the policy's forward creep any more -- so every turn became an arc.
        if max(abs(forward_speed), abs(lateral_speed)) > self.STATION_KEEP_DEADBAND:
            return command
        turning = abs(yaw_rate) > self.STATION_KEEP_DEADBAND
        if turning and not (self.TURN_HOLD_ENABLED or getattr(self, "_physical_locomotion", False)):
            return command

        try:
            import warp as wp

            linear_world, angular_world = self.robot.get_velocities()
            linear = wp.to_torch(linear_world).reshape(-1)[:3]
            angular = wp.to_torch(angular_world).reshape(-1)[:3]
            _, orientations = self.robot.get_world_poses()
            quat = wp.to_torch(orientations).reshape(-1)[:4]
            qw, qx, qy, qz = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            vx, vy = float(linear[0]), float(linear[1])
            measured_yaw_rate = float(angular[2])
        except Exception:
            return command

        # World velocity into the body frame, so the correction is expressed the way the
        # policy's command is.
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        body_forward = cos_yaw * vx + sin_yaw * vy
        body_lateral = -sin_yaw * vx + cos_yaw * vy

        return [
            self._clamp(
                -self.STATION_KEEP_GAIN * body_forward, -self.STATION_KEEP_MAX_LINEAR, self.STATION_KEEP_MAX_LINEAR
            ),
            self._clamp(
                -self.STATION_KEEP_GAIN * body_lateral, -self.STATION_KEEP_MAX_LINEAR, self.STATION_KEEP_MAX_LINEAR
            ),
            # While turning, the operator's yaw is theirs -- only the translation is
            # corrected. Cancelling their yaw here would fight the stick.
            (
                yaw_rate
                if turning
                else self._clamp(
                    -self.STATION_KEEP_YAW_GAIN * measured_yaw_rate,
                    -self.STATION_KEEP_MAX_YAW,
                    self.STATION_KEEP_MAX_YAW,
                )
            ),
        ]

    def _step_walk_policy(self, dt: float, command: object) -> None:
        """Step the legacy recurrent actor at 50 simulation Hz and write leg targets."""
        torch = import_module("torch")
        import warp as wp

        if getattr(self, "_physical_locomotion", False) and self.locomotion_fault:
            raise RuntimeError(self.locomotion_fault)
        if not math.isfinite(dt) or dt <= 0.0:
            return
        self._walk_time += max(float(dt), 0.0)
        self._walk_step_counter += 1
        period = 1.0 / self.WALK_CONTROL_HZ
        if self._walk_next_control_time is None or self._walk_time + 1e-9 >= self._walk_next_control_time:
            # Accumulate seconds rather than rounding a decimation (120 / 50 is not
            # an integer). Never replay the stateful LSTM multiple times on one pose
            # after a long frame; skip overdue ticks and preserve the fractional phase.
            if self._walk_next_control_time is None:
                self._walk_next_control_time = self._walk_time + period
            else:
                skipped = math.floor((self._walk_time - self._walk_next_control_time + 1e-9) / period) + 1
                self._walk_next_control_time += skipped * period
            try:
                command = self._station_keeping_command(command)
                obs = self._compute_walk_observation(command)
                if obs.numel() != self.WALK_OBS_DIM or not torch.isfinite(obs).all():
                    raise ValueError("Walking observation must contain 47 finite values")
                with torch.no_grad():
                    # The LSTM is stateful, so this must be called exactly once per
                    # control tick, in order — never speculatively or twice per step.
                    action = self._walk_policy(obs.unsqueeze(0)).detach().reshape(-1)
                if action.numel() != self.WALK_NUM_ACTIONS or not torch.isfinite(action).all():
                    raise ValueError("Walking policy must return 12 finite joint actions")
                self._walk_previous_action = action.clone()
                targets = action * self.WALK_ACTION_SCALE + self._walk_leg_defaults_tensor
                targets = torch.clamp(targets, min=self._walk_leg_lower_limits, max=self._walk_leg_upper_limits)
                self.robot.set_dof_position_targets(wp.from_torch(targets), dof_indices=self._leg_dof_indices)
            except Exception as e:
                if getattr(self, "_physical_locomotion", False):
                    self._fail_physical_locomotion(f"Physical walking policy step failed: {e}")
                if not self._walk_step_error_logged:
                    carb.log_warn(f"G1TeleopRobot: walking policy step failed; retaining last leg targets: {e}")
                    self._walk_step_error_logged = True

    """
    Kinematic base.
    """

    def _integrate_base_command(self, dt: float, command: object) -> None:
        """Integrate ``(v_x, v_y, w_z)`` into the base position and yaw."""
        forward_speed, lateral_speed, yaw_rate = self._command_to_floats(command)
        self._base_yaw = self._wrap_angle(self._base_yaw + yaw_rate * dt)
        cos_yaw = math.cos(self._base_yaw)
        sin_yaw = math.sin(self._base_yaw)
        self._base_position[0] += (cos_yaw * forward_speed - sin_yaw * lateral_speed) * dt
        self._base_position[1] += (sin_yaw * forward_speed + cos_yaw * lateral_speed) * dt
        self._base_position[2] = self._spawn_position[2]

    def _write_base_pose(self, command: object) -> None:
        """Teleport the articulation root to the integrated pose and match its velocity."""
        half_yaw = self._base_yaw * 0.5
        orientation = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
        self.robot.set_world_poses(positions=[self._base_position], orientations=[orientation])

        # Reporting the commanded velocity (rather than leaving whatever the solver
        # produced) keeps get_velocities() meaningful for the behavioural logs and stops
        # the limbs from reacting to a root velocity the teleport already consumed.
        forward_speed, lateral_speed, yaw_rate = self._command_to_floats(command)
        cos_yaw = math.cos(self._base_yaw)
        sin_yaw = math.sin(self._base_yaw)
        linear = [
            cos_yaw * forward_speed - sin_yaw * lateral_speed,
            sin_yaw * forward_speed + cos_yaw * lateral_speed,
            0.0,
        ]
        try:
            self.robot.set_velocities(linear_velocities=[linear], angular_velocities=[[0.0, 0.0, yaw_rate]])
        except Exception:
            pass

    """
    Finger control.
    """

    def _resolve_finger_joint_names(self, dof_names: list[str]) -> set[str]:
        """Return every hand DOF name present on the articulation, driven or mimic."""
        hand_markers = (
            "_hand_",
            "L_thumb",
            "R_thumb",
            "L_index",
            "R_index",
            "L_middle",
            "R_middle",
            "L_ring",
            "R_ring",
            "L_pinky",
            "R_pinky",
        )
        return {name for name in dof_names if any(marker in name for marker in hand_markers)}

    def _driven_finger_joint_map(self) -> dict[str, dict[str, str]]:
        """Return the per-side finger-role to joint-name map for the selected hand."""
        if self._hand_variant.lower() == "inspire":
            return self.INSPIRE_DRIVEN_FINGER_JOINTS
        if self._hand_variant.lower() == "threefinger":
            return self.THREE_FINGER_DRIVEN_FINGER_JOINTS
        return {}

    def _configure_finger_dofs(self, dof_names: list[str]) -> None:
        """Resolve finger DOF indices and their open/closed target angles from joint limits."""
        self._finger_dof_indices = {}
        self._finger_open_closed = {}

        joint_map = self._driven_finger_joint_map()
        indices_by_side = {}
        for side, roles in joint_map.items():
            resolved = {role: dof_names.index(name) for role, name in roles.items() if name in dof_names}
            if resolved:
                indices_by_side[side] = resolved
        self._finger_dof_indices = indices_by_side
        if not indices_by_side:
            carb.log_warn(
                f"G1TeleopRobot: no {self._hand_variant} finger DOFs found on the articulation; "
                "finger teleoperation will be inactive."
            )
            return

        all_indices = sorted({index for roles in indices_by_side.values() for index in roles.values()})
        try:
            lower_limits, upper_limits = self.robot.get_dof_limits(dof_indices=all_indices)
            lower_limits = lower_limits.numpy()[0]
            upper_limits = upper_limits.numpy()[0]
        except Exception as e:
            carb.log_warn(f"G1TeleopRobot: could not read finger joint limits: {e}")
            return

        limits_by_index = {
            dof_index: (float(lower_limits[i]), float(upper_limits[i])) for i, dof_index in enumerate(all_indices)
        }
        try:
            self.robot.set_dof_drive_types("force", dof_indices=all_indices)
            get_physics_simulation_interface().flush_changes()
            self.robot.switch_dof_control_mode("position", dof_indices=all_indices)
            self.robot.set_dof_gains(
                stiffnesses=[[self.FINGER_STIFFNESS] * len(all_indices)],
                dampings=[[self.FINGER_DAMPING] * len(all_indices)],
                dof_indices=all_indices,
            )
            self.robot.set_dof_max_efforts([[self.FINGER_MAX_EFFORT] * len(all_indices)], dof_indices=all_indices)
            get_physics_simulation_interface().flush_changes()
            self.robot.set_dof_max_velocities([[self.FINGER_MAX_VELOCITY] * len(all_indices)], dof_indices=all_indices)
        except Exception as e:
            carb.log_warn(f"G1TeleopRobot: could not stiffen the finger drives: {e}")

        for side, roles in indices_by_side.items():
            self._finger_open_closed[side] = {}
            for role, dof_index in roles.items():
                lower, upper = limits_by_index[dof_index]
                # Whichever end of the travel is further from zero is the flexed end: the
                # Inspire joints close towards their positive limit, the Dex3 index and
                # middle joints towards their negative one.
                if abs(upper) >= abs(lower):
                    open_angle, closed_end = lower, upper
                else:
                    open_angle, closed_end = upper, lower
                fraction = self.FINGER_CLOSE_FRACTION.get(role, 0.95)
                closed_angle = open_angle + (closed_end - open_angle) * fraction
                self._finger_open_closed[side][role] = (open_angle, closed_angle)

        carb.log_info(f"G1TeleopRobot: {self._hand_variant} finger DOFs: {indices_by_side}")

    def get_finger_roles(self, side: str) -> list[str]:
        """Return the sorted driven-finger roles resolved for one hand.

        Args:
            side: ``"left"`` or ``"right"``.

        Returns:
            Available actuator-role names, or an empty list if none were resolved.
        """
        return sorted(self._finger_dof_indices.get(side, {}))

    def has_finger_control(self) -> bool:
        """Return whether initialization resolved at least one driven finger joint.

        This describes the configured mapping, not current tracking or contact health.
        """
        return any(self._finger_dof_indices.values())

    def get_finger_joint_range(self, side: str, role: str) -> tuple[float, float] | None:
        """Return the configured open/closed actuator angles in radians.

        These are the same endpoints used by :meth:`set_finger_curls`, including
        its margin before the hard stop. The closed angle can be lower than the
        open angle for a Dex3 finger; callers must preserve that direction.

        Args:
            side: ``"left"`` or ``"right"``.
            role: A driven actuator role, including ``thumb_yaw`` for opposition.

        Returns:
            Open and closed endpoint angles, or None for an unresolved, nonfinite,
            or effectively zero-width actuator range.
        """
        if role not in self._finger_dof_indices.get(side, {}):
            return None
        try:
            open_angle, closed_angle = self._finger_open_closed.get(side, {})[role]
            open_angle, closed_angle = float(open_angle), float(closed_angle)
            if not all(math.isfinite(value) for value in (open_angle, closed_angle)):
                return None
            if abs(closed_angle - open_angle) <= 1e-8:
                return None
            return open_angle, closed_angle
        except (KeyError, TypeError, ValueError):
            return None

    def get_finger_curls(self, side: str) -> dict[str, float]:
        """Measure normalized finger actuator positions from one live joint read.

        Values use the command ranges and are clamped to ``[0, 1]`` if contact
        pushes a joint outside that range. Missing/invalid measurements are omitted;
        unavailable physics returns an empty dictionary. Commanded targets are never
        substituted for actual motion or inferred for mimic joints.

        Args:
            side: ``"left"`` or ``"right"``.

        Returns:
            Measured actuator curls keyed by the roles with usable joint measurements.
        """
        roles = self._finger_dof_indices.get(side, {})
        ranges = {role: value for role in roles if (value := self.get_finger_joint_range(side, role)) is not None}
        if not ranges:
            return {}
        try:
            if not self.robot.is_physics_tensor_entity_valid():
                return {}
            # The articulation returns a Warp array; `numpy()` exposes the values
            # on the host regardless of the configured simulation device.
            positions = np.asarray(self.robot.get_dof_positions().numpy(), dtype=float)
            if positions.ndim != 2 or positions.shape[0] != 1:
                return {}
            positions = positions[0]
        except Exception:
            return {}
        measured = {}
        for role, (open_angle, closed_angle) in ranges.items():
            index = roles[role]
            if not isinstance(index, (int, np.integer)) or index < 0 or index >= len(positions):
                continue
            angle = float(positions[index])
            if math.isfinite(angle):
                measured[role] = min(1.0, max(0.0, (angle - open_angle) / (closed_angle - open_angle)))
        return measured

    def set_finger_curls(self, side: str, curls: dict[str, float]) -> None:
        """Drive one hand's fingers from normalized curl values.

        Only resolved driven joints receive targets. Missing roles keep their
        existing targets; nonfinite values are skipped and finite values are
        clamped to the normalized range. Passive mimic joints follow their coupling.

        Args:
            side: ``"left"`` or ``"right"``.
            curls: Finger role (``index``, ``middle``, ``ring``, ``little``, ``thumb``) to
                curl in ``[0, 1]``, where 0 is fully open and 1 is fully closed. The thumb
                opposition (``thumb_yaw``) follows the thumb curl unless given explicitly.
        """
        roles = self._finger_dof_indices.get(side)
        ranges = self._finger_open_closed.get(side)
        if not roles or not ranges:
            return

        targets = {}
        for role, dof_index in roles.items():
            curl = curls.get(role)
            if curl is None and role == "thumb_yaw":
                curl = curls.get("thumb")
            if curl is None:
                continue
            open_angle, closed_angle = ranges[role]
            curl = float(curl)
            if not math.isfinite(curl):
                continue
            curl = min(1.0, max(0.0, curl))
            targets[dof_index] = open_angle + (closed_angle - open_angle) * curl

        if not targets:
            return
        dof_indices = sorted(targets)
        self.robot.set_dof_position_targets([targets[index] for index in dof_indices], dof_indices=dof_indices)

    """
    Helpers.
    """

    def _apply_body_joint_gains(self, dof_names: list[str]) -> None:
        """Configure body actuators by mode without modifying finger drives.

        Physical decoupled mode installs deployment limits and armature, then
        disables implicit PD by switching to effort control. Its explicit loop
        reads position targets later; keeping implicit gains active would apply
        two controllers to the same body motors.
        """
        if getattr(self, "_physical_locomotion", False):
            policy = self._velocity_policy
            indices = self._velocity_dof_indices
            try:
                self.robot.set_dof_gains(stiffnesses=[policy.stiffness], dampings=[policy.damping], dof_indices=indices)
                if policy.armature is not None:
                    self.robot.set_dof_armatures([policy.armature], dof_indices=indices)
                if policy.effort_limits is not None:
                    self.robot.set_dof_max_efforts([policy.effort_limits], dof_indices=indices)
                if policy.velocity_limits is not None:
                    self.robot.set_dof_max_velocities([policy.velocity_limits], dof_indices=indices)
                if self.uses_explicit_body_efforts:
                    if policy.effort_limits is None:
                        raise ValueError("Explicit body PD requires the deployment motor effort limits")
                    # Effort mode zeros implicit stiffness/damping, avoiding two
                    # competing PD loops. Targets still feed apply_body_efforts.
                    self.robot.switch_dof_control_mode("effort", dof_indices=indices)
                    self.robot.set_dof_efforts(np.zeros((1, len(indices)), dtype=np.float32), dof_indices=indices)
            except Exception as error:
                self._fail_physical_locomotion(f"Could not configure physical policy actuators: {error}")
            return
        if self._locomotion == "policy":
            self._apply_walk_policy_gains(dof_names)
            return
        for keywords, stiffness, damping in self.JOINT_GAINS:
            indices = [
                index for index in self._body_dof_indices if any(keyword in dof_names[index] for keyword in keywords)
            ]
            if not indices:
                continue
            try:
                self.robot.set_dof_gains(
                    stiffnesses=[[stiffness] * len(indices)],
                    dampings=[[damping] * len(indices)],
                    dof_indices=indices,
                )
            except Exception as e:
                carb.log_warn(f"G1TeleopRobot: could not set gains for {keywords}: {e}")

    def _apply_walk_policy_gains(self, dof_names: list[str]) -> None:
        """Apply the legacy actor's leg gains and upper-body posture gains.

        These implicit drives are part of that checkpoint's deployment contract.
        They do not configure the physical decoupled actor's explicit motor loop.
        """
        groups: list[tuple[list[int], list[float], list[float]]] = []
        if self._leg_dof_indices:
            groups.append((list(self._leg_dof_indices), list(self.WALK_LEG_STIFFNESS), list(self.WALK_LEG_DAMPING)))

        waist = [i for i in self._posture_dof_indices if "waist" in dof_names[i]]
        if waist:
            groups.append((waist, [self.WALK_WAIST_STIFFNESS] * len(waist), [self.WALK_WAIST_DAMPING] * len(waist)))

        for index in self._posture_dof_indices:
            name = dof_names[index]
            for keyword, stiffness, damping in zip(
                self.WALK_ARM_JOINT_ORDER, self.WALK_ARM_STIFFNESS, self.WALK_ARM_DAMPING
            ):
                if keyword in name:
                    groups.append(([index], [stiffness], [damping]))
                    break

        for indices, stiffnesses, dampings in groups:
            try:
                self.robot.set_dof_gains(stiffnesses=[stiffnesses], dampings=[dampings], dof_indices=indices)
            except Exception as e:
                carb.log_warn(f"G1TeleopRobot: could not set walking gains for {indices}: {e}")

    def _apply_articulation_properties(self) -> None:
        """Apply the robot's solver and collision settings without changing the timestep."""
        try:
            self.robot.set_solver_iteration_counts(position_counts=[16], velocity_counts=[1])
            self.robot.set_enabled_self_collisions([False])
            self.robot.set_sleep_thresholds([0.0])
        except Exception as e:
            carb.log_warn(f"G1TeleopRobot: could not set articulation properties: {e}")

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        """Clamp a scalar."""
        return max(lower, min(upper, float(value)))

    @staticmethod
    def _command_to_floats(command: object) -> tuple[float, float, float]:
        """Convert the first three command components, returning zero if any is invalid."""
        try:
            values = float(command[0]), float(command[1]), float(command[2])
            return values if all(math.isfinite(value) for value in values) else (0.0, 0.0, 0.0)
        except Exception:
            return 0.0, 0.0, 0.0

    @staticmethod
    def _yaw_from_quaternion(orientation: list[float]) -> float:
        """Extract the yaw angle from a ``wxyz`` quaternion."""
        qw, qx, qy, qz = (float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3]))
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap a radian angle into ``[-pi, pi)`` for bounded yaw state."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
