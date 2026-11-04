"""Benchmark a retained prototype of cloned, fixed-base G1 grasp scenes.

This standalone experiment uses Isaac Sim's cloner and physics APIs without Isaac
Lab. It builds one G1/table/YCB-object scene, clones it, and times physics stepping.
It does not implement a training loop, observations/actions/rewards interface,
episode reset, or a trained grasping policy. The reported policy-step budgets are
throughput extrapolations that exclude learning and data-processing costs.

The pelvis is anchored inside each clone, selected lower-body collisions are
disabled, and idle joints receive stiff drives. These simplifications differ from
the active VR example's physical manipulation and walking. The four right-arm
indices retained here also differ from the current VR example's seven-joint IK;
no direct policy-transfer compatibility is established.

Run this file as a separate Isaac Sim process. Argument parsing and SimulationApp
startup occur at module scope, so importing it is not a passive utility import.
The initial object is the first entry in `OBJECT_USDS`; no object randomization is
implemented. Use Isaac Sim's Python wrapper to measure the configured scene:

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

#: Right-arm joint roles resolved by this prototype; no action API uses them yet.
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
#: Physics timestep in seconds for this standalone benchmark, not every VR profile.
PHYSICS_DT = 1.0 / 100.0
#: Divide measured physics throughput by this factor for the printed budget estimate.
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
    """Build cloned fixed-base scenes and expose physics stepping for timing.

    Construction creates a World, authors the template, clones it, resets physics,
    and resolves articulation views. Reaching and grasping actions are not executed.

    Args:
        num_envs: Number of scene copies requested from the grid cloner.
        device: Physics/tensor device passed to Isaac Sim.
        spacing: Grid separation in meters.

    Raises:
        RuntimeError: The Isaac Sim assets root cannot be resolved.
    """

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
        """Request GPU PhysX capacities scaled to the number of cloned scenes.

        These requests are authored on the PhysicsScene prim. Missing scenes or
        unsupported attributes are reported and leave the corresponding settings
        unchanged. The chosen capacities do not establish that a particular GPU
        can run every requested environment count.
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
        # Retain the prototype's capacity floors and per-environment scaling.
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
        """Anchor the pelvis relative to its environment so the cloner can remap it.

        This removes free-base motion from the benchmark; it does not remove the
        leg joints from the articulation. If no pelvis is found, the function
        reports a warning and retains the unanchored scene.
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
        # dragged onto the same spot, interpenetrating.
        joint = UsdPhysics.FixedJoint.Define(stage, f"{robot_root}/BaseWeld")
        joint.CreateBody0Rel().SetTargets(["/World/envs/env_0"])
        joint.CreateBody1Rel().SetTargets([pelvis])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, PELVIS_HEIGHT))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        print(f"[grasp-env] base welded at {pelvis} (anchored inside the env)", flush=True)

    #: Lower-body link-name fragments whose colliders this benchmark disables.
    DEAD_COLLISION = ("hip", "knee", "ankle", "foot", "leg")

    def _strip_unused_physics(self, stage, robot_root: str) -> int:
        """Disable articulation self-collision and selected lower-body colliders.

        This is a benchmark simplification, not the active VR contact configuration.
        Individual authoring failures are skipped by the retained implementation.

        Returns:
            Number of successful collider-disable operations.
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
        """Request this benchmark's simplified-physics and Inspire-hand variants."""
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
        """Clone the template, reset physics, and bind batched articulation views.

        Joint indices are resolved after reset because the composed asset defines
        the actual DOF ordering. Scene/view errors propagate to the caller.
        """
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
        """Return matching right-arm indices in the prototype's requested order.

        Missing names are omitted; this helper does not validate a complete arm.
        """
        found = []
        for joint in ARM_JOINTS:
            for index, name in enumerate(dof_names):
                if name == f"right_{joint}_joint":
                    found.append(index)
                    break
        return found

    def _lock_idle_joints(self) -> None:
        """Request zero position targets and stiff drives for the remaining joints.

        These are drive requests, not removal of DOFs or a hard joint lock. Failed
        drive setup is reported without aborting the retained benchmark.
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
        """Advance the benchmark World without rendering or supplying new actions.

        Args:
            count: Number of physics steps to request.

        Example:

            .. code-block:: python

                env.step_physics(20)  # Warm up a constructed benchmark scene.
        """
        for _ in range(count):
            self.world.step(render=False)


def main() -> int:
    """Time warmed-up physics stepping, print extrapolations, and close the app.

    Returns:
        Zero after the benchmark completes normally.
    """
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
