# Third-party policy weights

## NVIDIA GR00T decoupled balance and walking actors

`g1_decoupled_balance.onnx` and `g1_decoupled_walk.onnx` are unchanged NVIDIA
GR00T-WholeBodyControl checkpoints, pinned to
[`7f151314d4d1606544bf249d2a7a1cb754c64582`](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/7f151314d4d1606544bf249d2a7a1cb754c64582/decoupled_wbc/sim2mujoco/resources/robots/g1/policy).
They observe all 29 body joints and command the 15 leg/waist joints. Arm targets
are supplied separately by the physical walking adapter, retaining arm
teleoperation. Loading or passing an inference test alone does not establish
physical balance, grasping, or payload capability with this project's robot.

These are **GR00T Whole-Body Control locomotion actors**, not the GR00T
foundation/VLA model. The walking launcher loads these two local ONNX files via
[`g1_decoupled_policy.py`](../g1_decoupled_policy.py), validating both SHA-256
hashes before starting inference. Their full repository directory is
`source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/data/`.

| Local file | Bytes | SHA-256 |
|---|---:|---|
| [g1_decoupled_balance.onnx](g1_decoupled_balance.onnx) | 1,886,682 | `f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18` |
| [g1_decoupled_walk.onnx](g1_decoupled_walk.onnx) | 1,886,682 | `7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407` |

The model weights use the **NVIDIA Open Model License**, reproduced unchanged in
[nvidia_groot_OPEN_MODEL_LICENSE.txt](nvidia_groot_OPEN_MODEL_LICENSE.txt).
The repository's source/configuration uses Apache-2.0; its original dual-license
notice and source license are in [nvidia_groot_LICENSE.txt](nvidia_groot_LICENSE.txt).
These model weights do not inherit the source code's Apache license.

[g1_decoupled_policy.json](g1_decoupled_policy.json) records the source URLs,
checksums, exact joint order, observation layout, startup positions, gains, and
actuator parameters used by the adapter. The original
[policy YAML](g1_decoupled_policy_deploy.yaml) and
[launch YAML](g1_decoupled_launch.yaml) are preserved unchanged. The policy YAML
contains historical relative model filenames; the JSON selects the released
Balance/Walk binaries from the launch configuration explicitly.

Both models take `input`, float32 `[batch_size,516]`, and return `output`, float32
`[batch_size,15]`. The adapter uses batch size one and the CPU execution provider
from Isaac Sim's bundled `isaacsim.pip.onnx`. Each 86-value observation frame is:
scaled velocity command (3), height command (1), torso RPY command (3), body-frame
angular velocity (3), body-frame gravity (3), relative joint positions (29),
joint velocities (29), and previous raw leg/waist action (15). Six **complete
frames** are concatenated oldest first; missing initial history frames are zero.

Inference runs at 50 Hz with 200 Hz physics. Targets are the 15 default positions
plus `0.25 * action`; the model observes measured arm motion without replacing
arm targets. Balance is selected for unscaled command norm below `0.05`; walking
is selected otherwise, preserving history across switches. The height input is
`0.74 m` and torso RPY is zero. These are policy inputs, not root constraints.

The active launch `MOTOR_KP`/`MOTOR_KD` arrays supply all 29 body gains. Effort
limits are mapped by joint name from the paired MuJoCo XML, whose default joint
armature is `0.01`. The launch file's effort list also includes finger joints and
must not be sliced as a 29-joint array. Velocity limits use its explicit 29-entry
list. Observation defaults contain zero arm angles; startup arms use the
upstream LOW-elbow command with shoulder roll `+0.2/-0.2 rad`. The initial 15
leg/waist positions are zero, matching the simulation XMLs' default `qpos`; the
nonzero policy observation offsets are not the physical spawn pose. The standalone
reference applies default-angle PD targets for four physics ticks before its first
actor update. The source XML places the pelvis at `0.793 m`; this project spawns
at `0.8 m` and lets physics settle it. Walking mode uses explicit, clipped body
PD and replaces only the foot collision points with the paired XML geometry.
The contact override preserves authored mass, COM and inertia. The project's
validation record separates the passed unloaded physical replays from the
remaining headset, grasping and payload checks.

The controller-stick profile additionally enables this project's optional gravity
feedforward for the fourteen externally controlled arm joints. It adds native PhysX
model gravity before the same effort clipping, without changing the fifteen actor
motor calculations, source metadata, gains, checkpoints or observations. This is a
project adapter feature, not a setting claimed to be present in the pinned deployment.
Other profiles retain the original default with that option disabled.

To reproduce the weights from PowerShell, run from this directory:

```powershell
$pinnedModels = 'https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/7f151314d4d1606544bf249d2a7a1cb754c64582/decoupled_wbc/sim2mujoco/resources/robots/g1/policy'
Invoke-WebRequest -Uri "$pinnedModels/GR00T-WholeBodyControl-Balance.onnx" -OutFile '.\g1_decoupled_balance.onnx'
Invoke-WebRequest -Uri "$pinnedModels/GR00T-WholeBodyControl-Walk.onnx" -OutFile '.\g1_decoupled_walk.onnx'
Get-FileHash '.\g1_decoupled_balance.onnx', '.\g1_decoupled_walk.onnx' -Algorithm SHA256
```

Use the `media.githubusercontent.com` URLs for the payloads: the corresponding
raw GitHub URLs contain Git LFS pointers rather than ONNX binaries.

## `g1_unitree_motion.pt`

Unitree's older 12-joint flat-terrain walking policy is retained for legacy
assisted policy mode. It is distinct from the decoupled balance/walk actors above.

- **Source:** [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym),
  `deploy/pre_train/g1/motion.pt`
- **License:** BSD 3-Clause — © 2016-2023 HangZhou YuShu TECHNOLOGY CO., LTD ("Unitree
  Robotics"). See the upstream `LICENSE`.
- **Architecture:** TorchScript `PolicyExporterLSTM` — `LSTM(47 -> 64)` into an
  `MLP(64 -> 32 -> 12)`, ELU. Stateful: it keeps its hidden and cell state internally.
- **Interface:** 47-dim observation in, 12 leg-joint actions out, 50 Hz.

To re-download:

```bash
curl -L -o g1_unitree_motion.pt \
  https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/deploy/pre_train/g1/motion.pt
```

The legacy loader also looks in `~/BehavioralCollection/policies/g1_unitree_motion.pt`,
or accepts an explicit path through `walk_policy_path=`. Legacy assisted mode can
fall back to kinematic motion when its weights are unavailable. Physical walking
must fail explicitly instead of substituting kinematic motion.
