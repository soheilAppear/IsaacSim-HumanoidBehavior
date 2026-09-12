# VR Humanoid Behavior Lab

Control a **Unitree G1 with Inspire five-finger hands** in Isaac Sim using Quest Pro
controllers or optical hand tracking. The project records hand input, robot joints,
gaze, object states, and images for later analysis and learning experiments.

[![Isaac Sim 6.1 default](https://img.shields.io/badge/Isaac%20Sim-6.1.0%20default-76b900.svg)](https://github.com/isaac-sim/IsaacSim)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](tools/run_humanoid_tests.py)
[![License](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)

[Setup and controls](HUMANOID_VR_CONTROL.md) ·
[Recorded demos](#recorded-g1-demos) ·
[Developer guide and validation](docs/humanoid-control.md) ·
[Performance](docs/humanoid-control.md#performance-and-cpu-physics) ·
[Learning pipeline](learning/README.md)

| G1 bottle lift | G1 package lift |
|:---:|:---:|
| ![White Inspire fingers lifting a yellow bottle](docs/readme/g1-bottle-lift.png) | ![Right Inspire hand lifting a highlighted package](docs/readme/g1-package-lift.png) |

Robot-camera screenshots from the **7 September 2026** G1 recording, at their native
256×256 resolution. Colored targets and gaze highlights are part of the recorded scene.

## Current behavior

The default is **stationary manipulation**. A world constraint anchors the robot's
pelvis while its arms and fingers remain controllable. Walking, turning, keyboard
locomotion, and step-in-place inputs cannot move the base in this mode.

| Capability | Current implementation |
|---|---|
| Arm control | Controller grip clutches that arm; valid optical wrist/middle-knuckle positions drive an anatomical palm target without a grip button or fist; measured scenery contacts limit inward arm motion |
| Finger control | Five independent curl targets and separate thumb opposition from the optical skeleton; controller trigger closes the whole hand |
| Pickup | Experimental physical contact grasping with bounded finger targets; objects must be held by fingers and friction, with no hand-to-object fixed joint |
| Camera | Head turns rotate the view left/right, up/down, and in roll; its position stays mounted to the robot head/torso, including during Pause |
| Gaze | Existing eye-tracking and highlighting behavior is retained; recordings label real eye input separately from HMD-forward fallback |
| Physics | CPU, with a configured 100 Hz physics timestep; rendering requests 90 Hz |
| Recording | Enabled by default: sensor CSVs, metadata, and a PNG image sequence |
| Walking and learned control | Optional walking modes and learning scaffolding remain available for experiments; they are outside the stationary-mode acceptance results |

Inspire has six independent hand actuators. Distal knuckles are coupled, so it cannot
reproduce every human knuckle angle or finger-splay movement separately.

The first valid headset orientation sets forward; **B** resets it. Walking, leaning,
or crouching in the room does not translate the camera. This turns the view only:
the G1's visible head remains fixed to its torso, with no actuated neck joint.
Eye-gaze tracking is unchanged.

The front-right table now offers **six lightweight physical objects**, with varied
shapes, heights, and thicknesses:

| Color | Object | Dimensions | Mass |
|---|---|---|---:|
| Blue | Cube | 6 × 6 × 6 cm | 60 g |
| Orange | Cylinder | 6 cm diameter × 10 cm high | 80 g |
| Green | Sphere | 7 cm diameter | 50 g |
| Purple | Tall thin cylinder | 4 cm diameter × 14 cm high | 60 g |
| Yellow | Flat block | 9 × 6 × 3 cm | 70 g |
| Red | Cone | 7 cm diameter × 10 cm high | 50 g |

All use gravity and real collisions. The original cube and cylinder retain their
sizes and masses; earlier historical pickup results used smaller 4 cm objects.
Grasping the four new shapes has not been validated.
The sphere, cube, and orange cylinder now sit nearer the robot, with the other
three shapes staggered behind them. An open-hand replay confirmed that both arms
can nearly straighten fully and retract without a fault. Reach still follows the
G1's actual geometry and joint limits; this layout change adds no arm length or
grasp guarantee. See [arm reach](HUMANOID_VR_CONTROL.md#palm-angle-arm-bending-and-latency).
Physical mode holds the arm pose on tracking loss or grip release.

**The reproduced sudden-stop case now passes.** Closing and opening the fingers
against the table previously caused unstable joint speeds and an automatic pause.
Adding a small joint-inertia floor to the hands passed all 20 close/open cycles,
with no fault or base movement and a clean withdrawal. A second test reconstructed
from recorded robot motion also passed. The guard remains enabled.

The enlarged cylinder passed a physical lift, hold, and release trial. **Cube pickup
remains unresolved:** both tested approaches failed to establish an opposing pinch,
although the robot remained stable. Earlier smaller-cube successes were also inconsistent.
Real headset handling remains unverified for this fix. See the
[validation record](docs/validation/isaac-sim-6.1.md).
The earlier fixed-joint assisted mode remains selectable for comparison; the recorded
demos below show that older assisted mode.
Each actuator retains its first-contact closing cap, with a **0.18 rad** lead,
preventing further closure from following a finger as it rolls over the contacted
body. A contact-report gap of up to **40 ms of simulation time** retains the command
cap only; support evidence clears immediately. Opening always wins; fully opening
the hand, input loss, and reset clear the retained caps.
Physical mode prepares finite, critically damped finger couplings before Play,
replacing the rigid settings that could destabilize the articulation at the table.
After arm smoothing, measured scenery-contact normals restrict motion into the
surface, including motion caused by wrist rotation. Withdrawal and sliding along
the surface remain available. Graspable props are excluded from this arm limit.
Loaded predictive contacts also limit arm motion before visible surface intersection.
The six actuators still drive the hand; passive knuckles receive no extra drives.
An invalid joint state pauses the simulation and requires **Reset or a scene reload**;
pressing Play alone cannot clear the fault. Gaze and hidden hand target spheres
are unchanged; head-view rotation is described above.

## Recorded G1 demos

These excerpts show **assisted pickup, lifting, and release** in a recorded stationary
G1 session. Click a preview to open its video file, or use the MP4 links below.

| Bottle pickup and transfer | Package pickup and release |
|:---:|:---:|
| [![Animated bottle pickup preview](docs/readme/g1-bottle-preview.gif)](docs/readme/g1-bottle-pickup.mp4) | [![Animated package pickup preview](docs/readme/g1-package-preview.gif)](docs/readme/g1-package-pickup.mp4) |
| [Open / download MP4 · 4.1 s](docs/readme/g1-bottle-pickup.mp4?raw=true) | [Open / download MP4 · 3.1 s](docs/readme/g1-package-pickup.mp4?raw=true) |

Videos are silent H.264 at 256×256 and 10 frames per simulated second. Playback follows
**simulation time**, not the slower wall-clock capture rate. These are visual examples;
they do not establish real Quest finger-tracking quality or camera-lock validation.
See [media provenance and export commands](HUMANOID_VR_CONTROL.md#screenshots-and-videos)
and the [validation results](#validation-and-known-limitations) for those distinctions.

## Quick start

The Windows launcher defaults to Isaac Sim **6.1.0** and loads the complete modified
`isaacsim.robot.policy.examples` extension directly from this checkout. The
[6.1 validation record](docs/validation/isaac-sim-6.1.md) covers software replay results
and the later live tracking recovery. A full
Isaac Sim source build is optional. For Quest Pro, keep the working OpenXR runtime and gaze setup;
the documented working connection uses SteamVR with Steam Link.

1. Clone this fork's teleoperation branch and fetch its Git LFS assets:

   ```powershell
   git clone --branch TeleopG1Sim https://github.com/soheilAppear/IsaacSim-HumanoidBehavior.git
   cd IsaacSim-HumanoidBehavior
   git lfs install
   git lfs pull
   ```

2. Extract Isaac Sim 6.1.0. The launcher uses
   `C:\Users\Soheil\Downloads\isaac-sim-standalone-6.1.0-windows-x86_64` by default.
   For another location, set `$env:ISAAC_DIR = "C:\path\to\isaac-sim"` in PowerShell.
   See [Installation](HUMANOID_VR_CONTROL.md#installation) for manual launch options.
3. Save any stage changes, close the previous Isaac session, connect the headset,
   and run:

   ```powershell
   .\tools\launch_isaac_vr.bat
   ```

   The launcher selects this checkout's complete policy extension, enables the Python
   server, and requests OpenXR skeletal hand tracking before the XR session starts.
   No copying into the standalone installation is needed.
4. Open **Window → Examples → Robotics Examples → Policy → Humanoid**
   (panel title **Humanoid: Unitree G1**), click **LOAD**, then **Play**.
5. Begin with the small cube and cylinder on the near edge of the front-right table.

See [Running](HUMANOID_VR_CONTROL.md#running) for the detailed startup procedure.
After changing Python source, restart Isaac Sim and reload the example so it uses
the updated classes.

## Controls

| Input | Action |
|---|---|
| Hold left/right **grip** | Move and rotate that arm; grip alone leaves the fingers open |
| **Trigger** | Close that hand around the object; physical pickup depends on contact and friction |
| Release **grip** | Hold that arm's pose in physical mode; valid trigger input still controls the fingers |
| Controller **Y** | Force both hands open; release the trigger or open your tracked fingers to rearm |
| **B** | Make the current headset orientation the forward view |
| Turn / tilt the headset | Rotate the view while its position stays mounted to the robot |
| Left stick click | Retain the mounted camera mode |
| Sticks, X/A, locomotion keys, head movement | No base movement in stationary mode |
| Open optical hand | Valid wrist/middle-knuckle positions activate the arm without a fist; fingers update independently from valid digit landmarks |
| Bend an optical finger | Move the corresponding robot finger |
| Close/open optical hand | Pinch or wrap around the object with thumb opposition; open to release |

Y is ignored while using bare-hand tracking. If switching from a controller after
a drop, open all four tracked fingers once; the thumb can remain naturally relaxed.

For controller pickup, **hold grip**, reach with the trigger released, then pull the
trigger when the actual robot fingers are close to the object. Keep grip and trigger
held to carry; release the trigger to open. A highlight does not attach an object.
Use optical thumb opposition and a pinch or wrap that actually encloses the object;
for the cube, bring the thumb pad across to face the index pad on opposite sides.
Making a fist nearby is insufficient. See the [full control guide](HUMANOID_VR_CONTROL.md#controls)
for input mapping, tracking loss, and release behavior.

If optical arm tracking disappears, physical mode holds the measured arm pose until
tracking returns. Missing finger input still opens the fingers. This prevents an
automatic return to rest from dragging a hand through the table.

## Validation and known limitations

Historical checks on **7 September 2026**, using Isaac Sim **6.0.0**:

| Check | Result and scope |
|---|---|
| Offline regressions | **82 passed**, including open-hand acquisition without buttons, independent fingers, grip/pickup, camera, lifecycle, and gaze regressions |
| Live grip and pickup replay | Passed with real PhysX: attached, lifted, and released a package; stationary base unchanged |
| Live finger replay | All ten fingers and both thumb-opposition joints moved independently; controller fallback and tracking-loss checks passed |
| Live camera replay | Fixed camera-to-body mount held during Play and Pause; tested with the Quest session connected |
| Real eye input | `eye_tracker` was observed with zero failed updates during the connected checks; gaze implementation/settings were preserved |
| Real Quest finger input | Skeletons arrived intermittently in this historical check; see the 12 September recovery below |

These camera results concern the former fully fixed view. The current rotation-only
view has separate checks; they do not retroactively validate its headset behavior.

The live replays inject input at the XR boundary and measure real simulation behavior.
They establish control and physics behavior; they do not establish headset tracking
quality. Generic grip/aim/pinch/poke interaction poses are insufficient for
individual fingers. A detected controller model also does not establish that a real
hand skeleton is reaching Isaac. See [Troubleshooting](HUMANOID_VR_CONTROL.md#troubleshooting).

The **11 September 2026 / Isaac Sim 6.1.0** checks passed NVIDIA's hardware checker,
all **83 offline tests**, actual PhysX pickup/release, camera mounting during Play
and Pause, and a full finger replay in an initialized fresh scene. A finger replay
run immediately after pickup failed on the right little finger; its cause remains
unresolved. The fresh-scene pass does not establish that all checks pass in every
manipulation state.

During a separate 60-second real-input observation, the connected headset used
Oculus/Meta transport through SteamVR and supplied **zero optical hand samples**;
unified eye gaze was unavailable. Gaze code and settings were preserved. On **12
September**, the user confirmed that starting SteamVR independently restored both.
The subsequent recording contains 1,627 optical-hand samples per side and 1,627
measured-eye samples during its last uninterrupted minute.
See the [6.1 validation record](docs/validation/isaac-sim-6.1.md) for measurements,
the corrected package colliders, and exact reproduction steps.

The current update includes anatomical palm alignment, stable palm-centre targets,
bounded IK task priorities, wall-time input filtering, and an experimental physical
grasp mode. The user accepted the arm changes. The table-contact repair combines
finite compliant finger couplings, limits on inward arm commands, and a joint-health
guard. Three consecutive fixed-pose table-contact replays passed, but the later
dynamic close/open stress test exposed a further fault. The selected hand-joint
inertia floor passed that same 20-cycle test. Larger objects and motion derived from
the recorded failure were checked separately: the derived path and larger-cylinder
pickup passed; both larger-cube grasp trials failed without destabilizing the robot. See
[palm angle, arm bending, and latency](HUMANOID_VR_CONTROL.md#palm-angle-arm-bending-and-latency).

The current suite passed **242 offline tests**, including head-turn direction, release-latch recovery, retained contact caps,
arm-contact limits, articulation health, and hand-joint stabilization. See the
[validation record](docs/validation/isaac-sim-6.1.md) for configurations and results.
Historical starter-object passes apply only to their recorded settings; no replay
establishes arbitrary-object pickup or headset tracking accuracy.
The earlier fresh-scene combined pickup replay failed because the smaller cube
slipped with the exploratory grasp profile; it does not validate the enlarged objects.

At the stop-fix handoff, a fresh production scene was paused with live input restored,
hidden hand markers, and no physics fault or callback error. Isaac reported no XR
devices, so these checks do not establish real Quest handling. The ready-state record
is `_compat61/stop-fix-ready-state.json`; see the
[validation record](docs/validation/isaac-sim-6.1.md) for its timestamp and configuration.
That snapshot predates the four additional practice shapes.

Run offline tests with Isaac Sim's Python wrapper:

```powershell
& "C:/path/to/isaac-sim-standalone-6.1.0-windows-x86_64/python.bat" tools/run_humanoid_tests.py
```

With the updated example loaded and the Python server enabled, run live checks
**one at a time** using ordinary host Python:

```powershell
python tools/validate_camera_live.py
```

For a fresh, initialized physical-mode scene, check light contact with the table:

```powershell
python tools/validate_table_contact_live.py --timeout 240
```

This synthetic right-hand replay approaches an empty table area, commands a small
downward motion, holds, then withdraws. It records contacts, joint state, and root
stability in `_compat61/table-contact-live.json` and leaves live input restored with
the timeline paused. Its report determines whether that run passed.

In a fresh physical-mode scene with the original cube and cylinder on the front table, recheck
the synthetic palm-down grasps after changes to hand physics. The replay selects
an object-specific thumb pose and approach:

```powershell
python tools/validate_contact_grasp_live.py --object both --timeout 300
```

It records contact/lift/hold/release results in `_compat61/contact-grasp-live.json`,
restores input, and leaves the scene paused. Open your input before resuming after
its forced-drop cleanup. This trial uses real PhysX with synthetic optical landmarks;
its report states whether that run passed. It does not validate headset tracking.
`--object both` exercises the original cube and cylinder only; it does not test the
sphere, tall cylinder, flat block, or cone.
When a recording session exists, `validation_<run_id>.json` marks this synthetic
interval by time and physics step; other samples in the session may be real user input.

Before a full finger replay, restart Isaac Sim, LOAD a fresh example, press Play,
and wait for initialized finger joints. Then run:

```powershell
python tools/validate_fingers_live.py --timeout 420
```

These replays restore live input and leave the timeline paused. To observe real
hands, press Play, switch the headset to bare hands, and bend individual fingers:

```powershell
python tools/observe_hand_tracking_live.py --seconds 20
```

The observer changes no input, timeline, or gaze settings. Commands for the separate
legacy assisted pickup/highlight replay, which temporarily substitutes gaze input, and detailed
results are in [Automated validation](docs/humanoid-control.md#automated-validation).

## Performance

**100 Hz physics and 90 Hz rendering are configured rates, not measured performance
guarantees.** The last active minute of the 12 September recording advanced 16.26
simulated seconds in 59.992 wall seconds: **0.271× real time**, or roughly 3.7 times
slower. This is a baseline from before the arm-response changes.

The code performs synchronous PNG encoding and CSV writes inside physics callbacks,
creates an additional recording-camera render product, and repeats some robot-pose
and IK reads. These are candidates for measurement; their individual contribution
has not yet been isolated. CPU physics was chosen using older scene measurements,
which do not establish current performance.

The next useful comparison is the same scene with recording enabled and disabled
**before loading**, with only one Isaac session open. The
[performance guide](docs/humanoid-control.md#performance-and-cpu-physics) explains the
settings, measurement procedure, and limitations. Gaze and camera behavior should be
held constant during that comparison.

## Data and learning

Sessions are written under `~/BehavioralCollection/raw_sessions/` with sensor CSVs,
`metadata.json`, frame timestamps, and eye-camera PNGs. Rates are defined in simulation
time; gaze queries run at 50 Hz and their latest result is reused by 100 Hz log rows.
Periodic flushing limits buffered data, but an interrupted process can still lose
unflushed rows.

Gaze hits record the collider intersected by the supplied ray; they are observations
of the gaze signal, not a guarantee of human intent. Grasp mode must accompany each
recording; old fixed-joint demonstrations are assisted grasping, not physical contact
successes. See [Behavioral data collection](HUMANOID_VR_CONTROL.md#behavioral-data-collection)
for files and columns.

The [learning directory](learning/README.md) contains the planned dataset and
world-model workflow. A trained planner is not integrated into the current robot
controller. After contact grasping passes live checks, the
[next research phase](HUMANOID_VR_CONTROL.md#next-research-phase) can add a webcam's
shoulder/elbow posture alongside Quest hands and eyes, then investigate gaze and
hand/finger velocity as intent cues. Velocity is not measured finger pressure.
Neither webcam fusion nor a new intent-training pipeline is implemented yet.

## Project layout

| Path | Purpose |
|---|---|
| [interactive/humanoid/](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/) | Scene, input, IK, pickup, camera, gaze, and recording |
| [robots/g1.py](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/g1.py) | Stationary anchor, joint control, optional locomotion, and reset |
| [tools/](tools/) | VR launcher, live Python helper, observers, and validation replays |
| [tools/tests/](tools/tests/) | Offline control regression suite |
| [HUMANOID_VR_CONTROL.md](HUMANOID_VR_CONTROL.md) | Installation, operation, recording, and troubleshooting |
| [docs/humanoid-control.md](docs/humanoid-control.md) | Implementation details, performance, and validation evidence |
| [learning/](learning/) | Learning workflow plan and directory scaffolding |

## Earlier captures

![Earlier H1 gaze and hand-control capture](docs/readme/vr_gaze_showcase.gif)

This clip was recorded with the earlier H1 robot. It illustrates gaze highlighting
and recording; the current G1's stationary behavior and finger validation are
described above.

## Isaac Sim source and licensing

This fork includes the Isaac Sim source tree. For full simulator development, see
the [upstream Isaac Sim repository](https://github.com/isaac-sim/IsaacSim),
[Windows developer setup](docs/readme/windows_developer_configuration.md), and
[container build guide](tools/docker/README.md). `build.bat --help` or
`./build.sh --help` lists this checkout's source-build options.

Repository licensing is described in [LICENSE](LICENSE). Bundled policies and assets
retain their own notices; see the [setup guide's license references](HUMANOID_VR_CONTROL.md#license).
Fork-specific issues belong in [this repository's issue tracker](https://github.com/soheilAppear/IsaacSim-HumanoidBehavior/issues).
