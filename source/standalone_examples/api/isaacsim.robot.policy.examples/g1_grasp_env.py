"""Vectorised G1 grasping environment for reinforcement learning, without Isaac Lab.

Isaac Lab is not installed on this machine, and it is a large dependency to add for one
task, so this builds the same thing directly on Isaac Sim's cloner and physics tensor API:
N copies of the G1's upper body, each with a table and one YCB object, stepped together on
GPU PhysX.

Scope is deliberately narrow, and the narrowness is what makes it trainable in hours
rather than days:

* **The base is fixed.** Grasping and walking are separate problems, and the Unitree
  locomotion policy already handles the second one. Pinning the pelvis removes the legs,
  removes falling over, and removes the need for the policy to learn balance before it can
  learn to reach.
* **Four arm joints plus one finger scalar.** The same DOFs teleoperation drives, so a
  policy trained here can be dropped into the same control path.
* **YCB objects.** Hand-sized (0.05-0.19 m); the warehouse crates this project started with
  are 0.60 x 0.40 m against a ~0.12 m hand and cannot be grasped at all.

Run directly to measure environment throughput, which is what sets the training time:

    python.bat source/standalone_examples/api/isaacsim.robot.policy.examples/g1_grasp_env.py --envs 512
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--envs", type=int, default=256, help="number of parallel environments")
parser.add_argument("--device", default="cuda", help="physics device: cuda or cpu")
parser.add_argument("--steps", type=int, default=300, help="steps to time when run directly")
parser.add_argument("--spacing", type=float, default=2.5, help="metres between environments")
parser.add_argument("--headless", action="store_true", default=True)
ARGS, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": ARGS.headless})

import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.cloner import GridCloner  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.experimental.utils import stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

#: The arm joints the policy commands -- the same four teleoperation drives.
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
#: Physics rate. 100 Hz matches the interactive example, so a trained policy transfers
#: into it without a timestep mismatch.
PHYSICS_DT = 1.0 / 100.0
#: Control decimation: the policy acts at 50 Hz, as the locomotion policy does.
DECIMATION = 2

G1_USD = "/Isaac/Robots/Unitree/G1/g1.usd"
TABLE_USD = "/Isaac/Environments/Hospital/Props/SM_SideTable_02a.usd"
OBJECT_USDS = (
    "/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned/061_foam_brick.usd",
)
TABLE_TOP = 0.812          # m, measured
PELVIS_HEIGHT = 0.80       # m, the spawn height the G1 asset expects


class G1GraspEnv:
    """N parallel G1 upper bodies, each reaching for one object on a table."""

    def __init__(self, num_envs: int, device: str = "cuda", spacing: float = 2.5):
        self.num_envs = num_envs
        self.device = device
        self._assets_root = get_assets_root_path()
        if self._assets_root is None:
            raise RuntimeError("Isaac assets root not reachable")

        SimulationManager.set_backend("torch")
        SimulationManager.set_physics_sim_device(device)
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=PHYSICS_DT,
            rendering_dt=PHYSICS_DT * 4.0,
            backend="torch",
            device=device,
        )
        self.world.scene.add_default_ground_plane()
        self._enlarge_gpu_buffers()

        self._build_template()
        self._clone(spacing)

    def _enlarge_gpu_buffers(self) -> None:
        """Raise the GPU PhysX capacities well above the defaults sized for one robot.

        These live on the PhysicsScene PRIM, not in carb settings -- setting the carb keys
        does nothing and PhysX keeps demanding more capacity while silently missing
        interactions, which both wrecks the contacts a grasp depends on and collapses the
        step rate. With N cloned humanoids the aggregate-pair count scales with N, so the
        default (sized for a single robot) is short by orders of magnitude.
        """
        stage = omni.usd.get_context().get_stage()
        scene_prim = None
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene_prim = prim
                break
        if scene_prim is None:
            print("[grasp-env] WARNING: no PhysicsScene prim; GPU capacities left at defaults", flush=True)
            return

        api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        # Scaled off the environment count, with generous headroom: undersizing costs far
        # more (missed contacts) than the memory these buffers take on a 32 GB card.
        envs = max(self.num_envs, 1)
        for setter, value in (
            (api.CreateGpuFoundLostAggregatePairsCapacityAttr, max(1 << 16, envs * 2048)),
            (api.CreateGpuTotalAggregatePairsCapacityAttr, max(1 << 16, envs * 2048)),
            (api.CreateGpuFoundLostPairsCapacityAttr, max(1 << 20, envs * 8192)),
            (api.CreateGpuCollisionStackSizeAttr, max(1 << 26, envs * 1 << 20)),
            (api.CreateGpuMaxRigidContactCountAttr, max(1 << 21, envs * 8192)),
            (api.CreateGpuMaxRigidPatchCountAttr, max(1 << 20, envs * 2048)),
        ):
            try:
                setter(int(value))
            except Exception as error:
                print(f"[grasp-env] could not set a GPU capacity: {error}", flush=True)
        print(f"[grasp-env] GPU capacities sized for {envs} envs on {scene_prim.GetPath()}", flush=True)

    # ------------------------------------------------------------------ scene

    def _build_template(self) -> None:
        """Author one environment that the cloner replicates."""
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/envs")
        UsdGeom.Xform.Define(stage, "/World/envs/env_0")

        # --- robot -------------------------------------------------------
        robot_root = "/World/envs/env_0/G1"
        wrapper = UsdGeom.Xform.Define(stage, robot_root)
        wrapper.ClearXformOpOrder()
        wrapper.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, PELVIS_HEIGHT))
        stage_utils.add_reference_to_stage(
            usd_path=self._assets_root + G1_USD, path=f"{robot_root}/Asset"
        )
        self._select_g1_variants(stage.GetPrimAtPath(f"{robot_root}/Asset"))
        self._fix_base(stage, robot_root)
        self._strip_unused_physics(stage, robot_root)

        # --- table -------------------------------------------------------
        table_root = "/World/envs/env_0/Table"
        table = UsdGeom.Xform.Define(stage, table_root)
        table.ClearXformOpOrder()
        table.AddTranslateOp().Set(Gf.Vec3d(0.52, -0.28, 0.0))
        stage_utils.add_reference_to_stage(
            usd_path=self._assets_root + TABLE_USD, path=f"{table_root}/Asset"
        )
        for prim in Usd.PrimRange(stage.GetPrimAtPath(table_root)):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)

        # --- object ------------------------------------------------------
        object_root = "/World/envs/env_0/Object"
        obj = UsdGeom.Xform.Define(stage, object_root)
        obj.ClearXformOpOrder()
        obj.AddTranslateOp().Set(Gf.Vec3d(0.40, -0.20, TABLE_TOP + 0.05))
        stage_utils.add_reference_to_stage(
            usd_path=self._assets_root + OBJECT_USDS[0], path=f"{object_root}/Asset"
        )
        UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
        for prim in Usd.PrimRange(stage.GetPrimAtPath(object_root)):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)

    def _fix_base(self, stage, robot_root: str) -> None:
        """Weld the pelvis to the world.

        Grasping and locomotion are separate problems and the Unitree policy already solves
        the second. Pinning the base removes the legs' dynamics, removes falling over, and
        removes the need for the policy to learn balance before it can learn to reach --
        and it is most of the simulation cost, since a free 53-DOF humanoid in contact with
        the ground is far more solver work than an arm.
        """
        pelvis = None
        for candidate in (f"{robot_root}/Asset/pelvis", f"{robot_root}/Asset/G1/pelvis"):
            if stage.GetPrimAtPath(candidate).IsValid():
                pelvis = candidate
                break
        if pelvis is None:
            for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root)):
                if prim.GetName() == "pelvis":
                    pelvis = str(prim.GetPath())
                    break
        if pelvis is None:
            print("[grasp-env] WARNING: pelvis not found; base left free", flush=True)
            return
        # Body0 must be a prim INSIDE the environment, not the world. A world-anchored
        # joint (no Body0) cannot be remapped by the cloner -- it warns "localPose wont be
        # updated" -- so every clone stays welded to env_0's origin and all N robots get
        # dragged onto the same spot, interpenetrating. Measured, that alone was most of
        # the simulation cost.
        joint = UsdPhysics.FixedJoint.Define(stage, f"{robot_root}/BaseWeld")
        joint.CreateBody0Rel().SetTargets(["/World/envs/env_0"])
        joint.CreateBody1Rel().SetTargets([pelvis])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, PELVIS_HEIGHT))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        print(f"[grasp-env] base welded at {pelvis} (anchored inside the env)", flush=True)

    #: Link name fragments whose collision geometry the grasp task can never use. The
    #: pelvis is welded, so nothing below the waist moves or touches anything.
    DEAD_COLLISION = ("hip", "knee", "ankle", "foot", "leg")

    def _strip_unused_physics(self, stage, robot_root: str) -> int:
        """Turn off self-collision and the collision geometry of the locked lower body.

        Contact pairs are what the GPU broadphase spends its time on, and a welded humanoid
        generates a great many that can never produce a useful contact. Returns the number
        of colliders disabled.
        """
        disabled = 0
        root = stage.GetPrimAtPath(robot_root)
        for prim in Usd.PrimRange(root):
            name = prim.GetName().lower()
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                try:
                    PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(False)
                except Exception:
                    pass
            if not any(fragment in name for fragment in self.DEAD_COLLISION):
                continue
            for descendant in Usd.PrimRange(prim):
                if descendant.HasAPI(UsdPhysics.CollisionAPI):
                    try:
                        UsdPhysics.CollisionAPI(descendant).CreateCollisionEnabledAttr(False)
                        disabled += 1
                    except Exception:
                        pass
        print(f"[grasp-env] self-collision off, {disabled} lower-body colliders disabled", flush=True)
        return disabled

    def _select_g1_variants(self, prim) -> None:
        """Pick the simplified-physics G1 with Inspire hands, as the example does."""
        if not prim.IsValid():
            return
        wanted = {"Physics": "SimplifiedPhysX", "left_hand": "Inspire", "right_hand": "Inspire"}
        try:
            variants = prim.GetVariantSets()
            for name, value in wanted.items():
                if name in variants.GetNames():
                    variants.GetVariantSet(name).SetVariantSelection(value)
        except Exception as error:
            print(f"[grasp-env] variant selection skipped: {error}", flush=True)

    def _clone(self, spacing: float) -> None:
        cloner = GridCloner(spacing=spacing)
        targets = cloner.generate_paths("/World/envs/env", self.num_envs)
        self.env_origins = cloner.clone(
            source_prim_path="/World/envs/env_0", prim_paths=targets, replicate_physics=True,
            base_env_path="/World/envs",
        )
        self.world.reset()

        self.robots = Articulation(paths="/World/envs/env_.*/G1")
        self.objects = RigidPrim(paths="/World/envs/env_.*/Object")

        dof_names = [str(name) for name in self.robots.dof_names]
        self.arm_dofs = self._resolve_arm_dofs(dof_names)
        self.finger_dofs = [i for i, n in enumerate(dof_names)
                            if "index" in n or "middle" in n or "thumb" in n
                            or "ring" in n or "little" in n]
        self.locked_dofs = [
            index for index, name in enumerate(dof_names)
            if index not in self.arm_dofs and index not in self.finger_dofs
        ]
        self._lock_idle_joints()
        print(f"[grasp-env] {self.num_envs} envs | {len(dof_names)} dofs | "
              f"arm dofs {self.arm_dofs} | {len(self.finger_dofs)} finger dofs | "
              f"{len(self.locked_dofs)} locked", flush=True)

    @staticmethod
    def _resolve_arm_dofs(dof_names: list[str]) -> list[int]:
        """Right-arm joint indices, matched by name."""
        found = []
        for joint in ARM_JOINTS:
            for index, name in enumerate(dof_names):
                if name == f"right_{joint}_joint":
                    found.append(index)
                    break
        return found

    def _lock_idle_joints(self) -> None:
        """Hold every non-arm, non-finger joint at its default with a stiff drive.

        The legs and waist are not part of this task and the pelvis is welded, so letting
        the solver integrate them is work spent on joints that are supposed to stay still.
        """
        if not self.locked_dofs:
            return
        try:
            import warp as wp

            count = len(self.locked_dofs)
            stiffness = torch.full((self.num_envs, count), 2000.0, device=self.device)
            damping = torch.full((self.num_envs, count), 100.0, device=self.device)
            self.robots.set_dof_gains(
                stiffnesses=wp.from_torch(stiffness),
                dampings=wp.from_torch(damping),
                dof_indices=self.locked_dofs,
            )
            targets = torch.zeros((self.num_envs, count), device=self.device)
            self.robots.set_dof_position_targets(wp.from_torch(targets), dof_indices=self.locked_dofs)
        except Exception as error:
            print(f"[grasp-env] could not lock idle joints: {error}", flush=True)

    # ------------------------------------------------------------------- step

    def step_physics(self, count: int = 1) -> None:
        for _ in range(count):
            self.world.step(render=False)


def main() -> int:
    env = G1GraspEnv(ARGS.envs, ARGS.device, ARGS.spacing)
    env.step_physics(20)

    start = time.time()
    env.step_physics(ARGS.steps)
    elapsed = time.time() - start

    steps_per_second = ARGS.steps / elapsed
    env_steps = steps_per_second * ARGS.envs
    print("\n--- throughput ---", flush=True)
    print(f"  {ARGS.envs} envs on {ARGS.device}: {steps_per_second:,.0f} sim steps/s "
          f"-> {env_steps:,.0f} env-steps/s", flush=True)
    print(f"  policy steps/s (decimation {DECIMATION}): {env_steps / DECIMATION:,.0f}", flush=True)
    for budget in (20e6, 50e6):
        hours = budget / max(env_steps / DECIMATION, 1.0) / 3600.0
        print(f"  {budget / 1e6:.0f}M policy steps would take {hours:.2f} h", flush=True)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
