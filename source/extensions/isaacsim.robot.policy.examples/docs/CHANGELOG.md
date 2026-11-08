# Changelog

## HumanoidBehavior faster continuous gaze movement — 2026-11-08

- Increase requested hold-mode defaults to 0.40 m/s forward, 0.25 m/s backward and
  0.40 rad/s yaw. Retain step-mode defaults of 0.20/0.15 m/s and 0.25 rad/s, and
  preserve explicit speed overrides. Actual movement per wall second still depends
  on simulation performance and the walking controller's physical response.
- Pass 1,001 offline regressions, including launcher mode/default/override coverage,
  and apply the faster configuration to the live instance without an observed
  health fault or callback error. Dedicated higher-speed physical and carrying
  validation remains pending; earlier physical results use the prior speed profile.

## HumanoidBehavior gaze translation heading feedback — 2026-11-08

- Capture robot yaw at the start of Forward/Back in both gaze modes and request
  bounded corrective yaw through the existing walking controller while active.
  Keep explicit turns unchanged; hold release/eye loss ends the correction with
  the movement request. Do not pin/teleport the robot or change camera pose.
- Default heading feedback to enabled, with gain 3.0/s and a 0.15 rad/s correction
  cap; preserve explicit environment overrides. Filter heading error over 0.15
  simulation seconds and apply a 0.5-degree deadband.
- Paired physical checks pass all four actions with feedback off and gain 3.0/s,
  observing each release for five simulation seconds without faults. Forward
  heading error decreases from 3.20° to 1.31° right; reverse sideways drift remains
  about 3.6 cm despite improved heading. Inputs are synthetic, arms neutral and
  no payload is carried. Pass 999 offline regressions with the final gain-3.0
  default, including launcher default/override coverage. Earlier results predate feedback.

## HumanoidBehavior immediate gaze activation — 2026-11-08

- Default hold mode to zero activation delay: the first valid tracked button hit
  starts the request, including a quick glance, with no dwell ring. Preserve an
  explicit dwell override; a positive value restores the delay and progress ring.
  Step mode keeps its 0.7-second default and bounded movement behavior.
- Remove the duplicate pelvis-settling gate and its four-second auto-pause from
  hold mode. A valid look-away requests zero and rearms; invalid tracking requires
  a subsequent valid look-away. Keep the existing release smoothing and independent
  support-aware unintended-motion, balance/fall and runaway guards. Step mode
  retains measured settling and its timeout.
- Pass 990 offline regressions and actual-scene preview/physical checks for
  immediate activation of all four controls with synthetic eyes. Three-second
  holds continue past the old bounded goals; every release is observed for five
  simulation seconds without a fault. Forward eye loss immediately clears the
  filtered command; arm/finger updates continue. Hardware gaze and payload
  acceptance remain separate; earlier results below apply to delayed activation.

## HumanoidBehavior hold-to-move gaze controls — 2026-11-08

- Default the gaze profile to `G1_GAZE_MODE=hold`: after the 0.7-second dwell,
  continue translation or turning only while the same button remains fixated.
  Looking away, changing buttons, or losing valid eye tracking requests a stop;
  actual braking remains physical. Both optical hands retain manipulation control.
- Preserve anchored preview on every launcher invocation, with physical movement
  enabled only by a first `--move`. Preserve an explicit `G1_GAZE_MODE=step` override
  for the previous measured 0.20 m steps and 10-degree turns. Distance and angle
  settings apply only to step mode; continuous turns use the configured yaw speed.
- Permit bounded balance sway with low measured drift as an alternative settling
  criterion, retaining the existing deadline and independent physics guards.
- Pass 981 offline regressions and actual-scene hold preview/movement checks.
  All four physical actions continue beyond the old 20 cm/10-degree limits;
  Back and both turns settle after look-away, and Forward settles after simulated
  eye loss with an immediate zero command. The run has no latched fault and retains
  arm/finger updates. Inputs are synthetic; hardware gaze and payload acceptance
  remain separate. Earlier results below describe step mode.

## HumanoidBehavior eye-gaze walking — 2026-11-08

- Add four stereo scene buttons with a configurable 0.7-second dwell and progress
  rings. Selection requires tracked combined-eye input; head-forward visualization
  cannot activate controls. Each selection fires once and requires look-away to rearm.
- Add anchored preview and explicit physical movement profiles through
  `tools/launch_gaze_walking.bat`. Short measured steps/turns use the existing
  GR00T controller, preserve independent arm/finger updates, and stop automatically.
- Retain busy-command rejection, bounded motion/settling, and existing physics
  health guards. XR shutdown cancels motion; eye loss cancels only unfinished dwell.
  Occluded buttons cannot rearm, and UI failures cannot prevent stopping.
- Add deterministic control, real-USD panel, input ownership, and launcher tests,
  plus an isolated actual-scene validator with synthetic eye descriptors. Hardware
  gaze usability and loaded two-hand carrying remain separate acceptance checks.
- Pass 965 standalone humanoid regressions, including both optical hands changing
  arm targets and finger curls during an active gaze command. Actual-scene preview
  holds the base fixed; all four unloaded physical actions reach their measured
  targets and stop. Side-panel rendering is checked from the robot camera.

## HumanoidBehavior documentation cleanup — 2026-11-04

This fork-only documentation update does not change the extension version,
controller behavior, physics settings, policy weights, recording formats, or
historical validation results.

- Explain the custom G1 lifecycle, control ownership, coordinate/time conventions,
  and retained fallback paths in source comments and docstrings.
- Correct the extension overview and on-screen descriptions of physical grasping,
  the robot-head view, and experimental pointing destinations.
- Clarify launch profiles, local recording semantics, and the boundary between
  implemented data collection and the planned learning pipeline.
- Validate the cleanup with 891 standalone humanoid regressions before and after
  the edits. Executable Python syntax trees and launcher commands match the prior
  checkpoint, except for four corrected UI instruction strings; comments and
  docstrings are the only other source changes.

## HumanoidBehavior fork changes — 2026-09-07

This section records changes in this repository fork. It does not introduce an
upstream extension release or change the upstream package version below.

### Added

- Stationary Unitree G1 teleoperation with Inspire hands, a fixed pelvis, arm IK,
  and controller grip/trigger controls for assisted object pickup.
- Optical finger curl control and separate thumb opposition, including partial
  tracking loss handling and controller fallback when skeletal joints are absent.
- A robot-mounted head-height camera that follows the body in Play and Pause,
  independent of physical headset movement in stationary mode.
- Session recording and repeatable offline and live validation tools for humanoid
  control, finger mapping, camera attachment, and tracking observations.

### Fixed

- Grip-related physics failures caused by structural material highlight updates.
- Unstable arm commands through bounded targets and position-priority IK.
- Pickup joint placement using live palm and object poses with coincident joint frames.
- Misleading hand-source metadata suppressing valid controller input before actual
  skeletal landmarks arrive.

### Documentation

- Updated setup, controls, recording, CPU physics, performance troubleshooting, and
  validation guidance for the current stationary G1 configuration.
- Distinguished implemented recording from the planned learning pipeline. No trained
  world model or learned planner is supplied by `learning/`.
- Recorded the remaining real Quest tracking limitation: valid optical landmark
  delivery and the reported open-hand/fist behavior require further hardware validation.
  Existing gaze behavior is preserved.

## [5.2.11] - 2026-05-28
### Fixed
- Conditional use of asset until menagerie assets are tested with Isaac lab

## [5.2.10] - 2026-05-21
### Fixed
- Restore the prior physics sim device and fabric state in the cleanup paths of the interactive Quadruped, Go2, and Humanoid examples so the PhysX direct-GPU API flag is not left enabled, which previously caused `PxArticulationJointReducedCoordinate::setDriveTarget` errors when modifying USD in subsequent sessions

### Added
- Unit tests covering the snapshot/restore helpers in `isaacsim.robot.policy.examples.interactive.utils`, plus per-example roundtrip tests that verify the Quadruped, Go2, and Humanoid examples leave the physics sim device and fabric flag unchanged after cleanup

### Changed
- Extract the snapshot/restore physics-state logic shared by the interactive Quadruped, Go2, and Humanoid examples into `isaacsim.robot.policy.examples.interactive.utils`


## [5.2.9] - 2026-05-21
### Fixed
- Spot fall over issue in example by reverting the physics dt to 500hz

## [5.2.8] - 2026-05-20
### Fixed
- Select the asset's `Physics` variant before constructing the `Articulation` in `PolicyController.__init__`, so `UsdPhysics.ArticulationRootAPI` is authored on a descendant prim before `Articulation.fetch_articulation_root_api_prim_paths` resolves the root (prevents `Path.IsValidPathString(NoneType)` crashes)
- Apply `PhysxArticulationAPI` to the articulation root prim in `PolicyController._set_articulation_props` if it is missing, avoiding `Empty typeName` USD errors when the asset's Physics variant does not author the API
- Update default USD paths for Go2 and Spot policy controllers to the nested `Mujoco_Menagerie/<robot>/<robot>/<robot>.usda` layout

## [5.2.7] - 2026-05-20
### Fixed
- Missing _timeline error on policy reset

## [5.2.6] - 2026-05-05
### Added
- Standing test for the Go2 policy that holds a zero command and asserts the robot remains upright

### Fixed
- Select the USD `Physics` variant from `SimulationManager.get_active_physics_engine()` so the Newton-compatible variant is chosen when Newton is the active engine
- Log a warning when the requested USD `Physics` variant is not declared on the robot prim instead of silently selecting a non-existent variant
- Remove the `_set_physics_variant` override in `SpotFlatTerrainPolicy` so it inherits the engine-to-variant mapping from `PolicyController`

## [5.2.5] - 2026-04-23
### Changed
- Decreased test tolerance to pass with newton backend.

## [5.2.4] - 2026-04-23
### Fixed
- Added isaacsim.physics.newton.tensors as extension dep

## [5.2.3] - 2026-04-23
### Changed
- Defer torch import to avoid loading it at startup

## [5.2.2] - 2026-04-08
### Fixed
- Fix physics variant selection to match USD variant names case-insensitively, resolving H1 robot loading failure when variant set uses `Physx` instead of `physx`
- Register `isaacsim.robot.policy.examples.robots` as a public module in the extension manifest

## [5.2.1] - 2026-04-06
### Changed
- Set `reset_xform_op_properties` to True when instantiating the Articulation

## [5.2.0] - 2026-03-17
### Added
- Newton can be used as a physics backend

## [5.1.1] - 2026-03-04
### Changed
- Fix api errors

## [5.1.0] - 2026-03-04
### Changed
- Added Overview.md, python_api.md and updated docstrings

## [5.0.4] - 2026-02-04
### Changed
- Update physics rate for drawer opening test for CPU test

## [5.0.3] - 2025-12-08
### Changed
- Removed rendering manager test time dependency (moved to base sample)

## [5.0.2] - 2025-12-03
### Changed
- Remove TODOs.

## [5.0.1] - 2025-12-02
### Changed
- Removed unecessary dependencies
- Removed remaining experimental api references

## [5.0.0] - 2025-12-01
### Changed
- Changed the backend to experimental API using warp and torch
- Enabled GPU physics to inference policies
- Moved policy based interactive examples to the isaacsim.robot.policy.examples folder

## [4.3.0] - 2025-10-27
### Changed
- Replace import statements with the deprecation function when importing PyTorch
- Make omni.isaac.ml_archive an explicit test dependency

## [4.2.0] - 2025-10-17
### Changed
- Migrate PhysX subscription and simulation control interfaces to Omni Physics

## [4.1.11] - 2025-07-07
### Fixed
- Correctly enable omni.kit.loop-isaac in test dependency (fixes issue from 4.1.10)

## [4.1.10] - 2025-07-03
### Changed
- Make omni.kit.loop-isaac an explicit test dependency

## [4.1.9] - 2025-06-25
### Changed
- Add --reset-user to test args

## [4.1.8] - 2025-06-11
### Changed
- Update Franka Open Drawer Policy example
- Simplified the observation computation

## [4.1.7] - 2025-05-31
### Changed
- Use default nucleus server for all tests

## [4.1.6] - 2025-05-19
### Changed
- Update copyright and license to apache v2.0

## [4.1.5] - 2025-05-16
### Changed
- Make extension target a specific kit version

## [4.1.4] - 2025-05-10
### Changed
- Enable FSD in test settings
- Add get_physx_simulation_interface().flush_changes() to policy_controller to fix FSD issues

## [4.1.3] - 2025-05-09
### Fixed
- Buf with setting max effort with FSD enabled

## [4.1.2] - 2025-05-02
### Changed
- Update Franka and Anymal USD paths

## [4.1.1] - 2025-04-14
### Changed
- Update Isaac Sim robot asset path

## [4.1.0] - 2025-04-13
### Added
- Franka open drawer policy class

## [4.0.8] - 2025-04-09
### Changed
- Update all test args to be consistent

## [4.0.7] - 2025-04-04
### Changed
- Version bump to fix extension publishing issues

## [4.0.6] - 2025-03-26
### Changed
- Cleanup and standardize extension.toml, update code formatting for all code

## [4.0.5] - 2025-03-11
### Changed
- Switch asset root for tests to internal nucleus

## [4.0.4] - 2025-03-11
### Changed
- Reduce time to complete tests

## [4.0.3] - 2025-01-26
### Changed
- Update test settings

## [4.0.2] - 2025-01-21
### Changed
- Update extension description and add extension specific test settings

## [4.0.1] - 2024-12-22
### Added
- Rootpath input for when the articulation root is not the same as the root prim path

## [4.0.0] - 2024-11-01
### Removed
- Unitree quadruped optimized controller class
- Optimized controller based standalone and ROS examples

### Added
- Policy Controller and config loader helpers for Isaac Lab based env config

## [3.0.2] - 2024-10-28
### Changed
- Remove test imports from runtime

## [3.0.1] - 2024-10-24
### Changed
- Updated dependencies and imports after renaming

## [3.0.0] - 2024-10-07
### Changed
- Extension renamed to isaacsim.robot.policy.example.
- Optimization control based robot example removed.
- Moved Humanoid template to robot policy examples

## [2.0.1] - 2024-08-28
### Fixed
- Spot unit test

## [2.0.0] - 2024-08-01
### Added
- RL policy based robot simulation for anymal and spot quadruped

## [1.4.5] - 2024-04-30
### Changed
- Updated unitree folder structure and opt to use unitree models with sensors attached

## [1.4.4] - 2024-03-07
### Changed
- Removed the usage of the deprecated dynamic_control extension

## [1.4.3] - 2024-02-12
### Changed
- Removed IMU sensor from unitree.py (since the controller uses ground truth data)
- Reduced the frequency of osqp solver from every physics step to every 5 physics steps
- Contact sensor now uses the interface instead directly of the python wrapper

## [1.4.2] - 2024-02-02
### Changed
- Updated path to the nucleus extension

## [1.4.1] - 2023-12-11
### Fixed
- Add the missing import statement for the IMUSensor

## [1.4.0] - 2023-12-06
### Changed
- Added torque clamp to the quadruped control in response to physics change
- Behavior changed, investigating performance issue

## [1.3.2] - 2023-08-22
### Fixed
- Fixed robot articulation bug after stop or reset

## [1.3.1] - 2023-06-07
### Changed
- Eliminated dependency on "bezier" python package. Behavior is unchanged.

## [1.3.0] - 2023-02-01
### Removed
- Removed Quadruped class
- Removed dynamic control extension dependency
- Used omni.isaac.sensor classes for Contact and IMU sensors

## [1.2.2] - 2022-12-10
### Fixed
- Updated camera pipeline with writers

## [1.2.1] - 2022-11-03
### Fixed
- Incorrect viewport name issue
- Viewports not docking correctly

## [1.2.0] - 2022-08-30
### Changed
- Remove direct legacy viewport calls

## [1.1.2] - 2022-05-19
### Changed
- Updated unitree vision class to use OG ROS nodes
- Updated ROS1/ROS2 quadruped standalone samples to use OG ROS nodes

## [1.1.1] - 2022-05-15
### Fixed
- DC joint order change related fixes.

## [1.1.0] - 2022-05-05
### Added
- Added the ANYmal robot

## [1.0.2] - 2022-04-21
### Changed
- Decoupled sensor testing from A1 and Go1 unit test
- Fixed contact sensor bug in example and standalone

## [1.0.1] - 2022-04-20
### Changed
- Replaced find_nucleus_server() with get_assets_root_path()

## [1.0.0] - 2022-04-13
### Added
- Quadruped class, unitree class (support both a1, go1), unitree vision class (unitree class with stereo cameras), and unitree direct class (unitree class that subscribe to external controllers)
- Quadruped controllers
- Documentations and unit tests
- Quadruped standalone with ros 1 and ros 2 vio examples
