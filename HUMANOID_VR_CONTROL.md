# Humanoid VR Control — Unitree G1

This fork of [NVIDIA Isaac Sim](https://github.com/isaac-sim/IsaacSim) provides stationary
VR manipulation with a Unitree G1, Inspire five-finger hands, Quest Pro eye tracking,
and synchronized behavioral recordings.

**Current default:** the pelvis is fixed to the world. Arms and fingers remain articulated,
and the camera's position stays mounted to the robot's head/torso. Turning or tilting the
headset rotates the view; walking, leaning, and crouching do not translate it.
Optical hands control individual fingers and thumb opposition; Touch controllers
use grip for arm movement and the index trigger for whole-hand closure. The default
grasp mode is now experimental physical contact, without a hand-to-object fixed joint.
Gaze retains the working implementation.

**An open hand should be tracked immediately when valid optical data arrives.** A fist is
not an activation requirement or a guarantee of pickup. On 12 September the user confirmed
that starting SteamVR independently restored eye and bare-hand tracking. The subsequent
working recording contains continuous optical-hand and eye-tracker samples over its last
recorded minute. The user subsequently accepted the palm/arm response changes.
The table-contact repair uses finite compliant finger couplings, limits inward arm
commands against external obstacles, and holds the arm pose on input loss or grip
release. Right-open, right-curled, and left-open fixed-pose contact/withdrawal tests
passed in the same scene, including the corrected handoff. Later finger close/open
motion against the table exposed another instability. A small artificial joint-inertia
floor now stabilizes that reproduced case: all 20 close/open cycles passed, followed
by clean withdrawal, with no joint-health fault or base movement. The guard stays enabled.
A separate path reconstructed from recorded robot motion also passed; this was not
an exact replay of the original optical hand landmarks.

The current table contains six objects: the **6 cm cube** and **6 cm diameter,
10 cm cylinder**, plus a green sphere, purple tall thin cylinder, yellow flat block,
and red cone. See [Scene and pickup behavior](#scene-and-pickup-behavior) for dimensions
and masses. The four new shapes have no validated pickup results. Earlier pickup
results used a 4 cm cube and 4 cm diameter, 8 cm cylinder; those remain historical.
The enlarged cylinder has now passed lift, hold, and release with an explicit higher
approach. Both the high and lower enlarged-cube pinches failed to produce opposing
support; the robot stayed stable during these failures. Cube pickup remains unresolved.

**Cube pickup remains unreliable.** The revised thumb pose passed one retry but
slipped again in the final fresh-scene test. The cylinder passed, but the combined
pickup check failed.
These earlier smaller-object results show remaining contact-path/pose sensitivity.
See [the validation record](docs/validation/isaac-sim-6.1.md) for the passing and
failing configurations. Live headset handling remains an operator check.

[Screenshots and videos](#screenshots-and-videos) ·
[Installation](#installation) · [Running](#running) · [Controls](#controls) ·
[Validation](#validation) · [Recordings](#behavioral-data-collection) ·
[Quest Pro gaze](#quest-pro-eye-tracking-optional) · [Troubleshooting](#troubleshooting)

For implementation details, coordinate frames, lifecycle behavior, and test coverage, see
[the developer guide](docs/humanoid-control.md).

## Screenshots and videos

![Both G1 Inspire hands with cyan and orange hand-target markers](docs/readme/g1-inspire-hands.png)

Both Inspire hands in the historical robot-camera recording. The colored spheres
are debug target markers; the current version hides them by default so the fingers
remain visible. This still does not demonstrate independent finger motion.

| Recorded action | Video | Preview |
|---|---|---|
| Bottle pickup, lift, hand transfer, and release | [Silent MP4, 4.1 s](docs/readme/g1-bottle-pickup.mp4?raw=true) | [Animated GIF](docs/readme/g1-bottle-preview.gif) |
| Package pickup, lift, and release | [Silent MP4, 3.1 s](docs/readme/g1-package-pickup.mp4?raw=true) | [Animated GIF](docs/readme/g1-package-preview.gif) |

The [README gallery](README.md#recorded-g1-demos) displays both animated previews and
the bottle/package screenshots. All new media comes from local recording
`session_2026-09-07_02-40-23`, captured on **7 September 2026**. Its metadata identifies
Isaac Sim 6.0.0, G1, Inspire hands, stationary locomotion, a world-fixed base, and
`distance_gated_fixed_joint` grasp assistance. Raw session data is not included here.

| Published asset | Original `eye_camera` frames | Simulation time |
|---|---|---|
| `g1-bottle-pickup.mp4` / `g1-bottle-preview.gif` | 150–190 inclusive | 15.2–19.2 s |
| `g1-package-pickup.mp4` / `g1-package-preview.gif` | 265–295 inclusive | 26.7–29.7 s |
| `g1-bottle-lift.png` | 180 | 18.2 s |
| `g1-package-lift.png` | 280 | 28.2 s |
| `g1-inspire-hands.png` | 292 | 29.4 s |

Screenshots are unmodified PNG frames. Videos preserve the native **256×256** frame
size, use silent H.264 with browser-compatible pixel format, and play at **10 fps in
simulation time**. The final frame is displayed for another 0.1 s, giving durations
of 4.1 s and 3.1 s. The original first-to-last wall-clock spans were approximately
17.46 s and 13.79 s; these clips are not real-time performance measurements.

The session metadata does not distinguish operator input from injected replay input.
These clips demonstrate visible manipulation, not verified Quest tracking accuracy,
per-finger independence, or camera invariance. Refer to [Validation](#validation) for
measured results and outstanding hardware checks. Existing gaze markers and highlights
are preserved in the source images; exporting this media does not change gaze code or
settings. The July H1 clip in the README remains labeled as historical footage.

To export the same excerpts from a local copy of this recording, install FFmpeg and
run from the repository root. Set `$sessionDir` to the folder containing `metadata.json`:

```powershell
$sessionDir = "C:/path/to/raw_sessions/session_2026-09-07_02-40-23"
$frames = Join-Path $sessionDir "frames/eye_camera/frame_%06d.png"
ffmpeg -y -framerate 10 -start_number 150 -i $frames -frames:v 41 -an -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart docs/readme/g1-bottle-pickup.mp4
ffmpeg -y -framerate 10 -start_number 265 -i $frames -frames:v 31 -an -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart docs/readme/g1-package-pickup.mp4

foreach ($clip in "bottle", "package") {
    ffmpeg -y -i "docs/readme/g1-$clip-pickup.mp4" -filter_complex '[0:v]split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=sierra2_4a' -loop 0 "docs/readme/g1-$clip-preview.gif"
}
```

For other sessions, check `frame_timestamps.csv` before choosing a playback rate;
missing frames or changed recording settings can invalidate a fixed 10 fps export.
PNGs and MP4s use this repository's Git LFS rules, so contributors need `git lfs pull`.
The small GIF previews are stored directly in Git and linked to the MP4 files for
GitHub readers.

## Installation

### Prerequisites

- Isaac Sim Standalone **6.1.0** and a compatible NVIDIA GPU/driver. The Windows launcher
  now defaults to 6.1. The recorded 7 September live validation used 6.0.0; see the
  [11 September 6.1 results](docs/validation/isaac-sim-6.1.md) for current software
  checks and the later tracking recovery. Linux launch syntax is included below but has
  not been validated for this project's Quest setup.
- Isaac Sim's bundled Python 3.12 for the offline test runner and standalone simulation.
  The TCP utility commands use ordinary host Python.
- For VR, a working OpenXR headset connection. Keep an already-working gaze runtime and
  streaming configuration; the [verified Quest Pro setup](#quest-pro-eye-tracking-optional)
  uses SteamVR and Steam Link.
- Access to the NVIDIA robot, warehouse, and YCB assets used by the example.

Load the **complete** `isaacsim.robot.policy.examples` extension from this checkout.
Copying only `humanoid_example.py` omits the robot wrapper, helpers, registration, and
startup settings. Isaac Sim 6.1 bundles a newer policy-extension API, so merging individual
project files into that installed package can also cause import errors.

### Windows launcher

The repository launcher loads this extension directly with `--ext-path` and selects
its version, `isaacsim.robot.policy.examples-5.2.11`. The standalone installation's
bundled files remain intact; no copy or junction is needed.

Its default installation directory is:

```text
C:\Users\Soheil\Downloads\isaac-sim-standalone-6.1.0-windows-x86_64
```

For another installation, set an environment override before launching from the
repository root. It applies to this PowerShell session and does not edit the launcher:

```powershell
$env:ISAAC_DIR = "C:\path\to\isaac-sim-standalone-6.1.0-windows-x86_64"
.\tools\launch_isaac_vr.bat
```

Restart Isaac Sim after Python changes; rebuilding the scene alone does not reload
imported modules. Save any stage changes and close the previous session first.

### Manual extension selection

For manual launch, use the extension's exact directory and version. From the Isaac Sim
installation directory, this desktop example loads the same custom package:

```powershell
.\isaac-sim.bat `
  --ext-path "C:/path/to/IsaacSim-HumanoidBehavior/source/extensions/isaacsim.robot.policy.examples" `
  --enable isaacsim.robot.policy.examples-5.2.11
```

Linux desktop syntax:

```bash
./isaac-sim.sh \
  --ext-path /path/to/IsaacSim-HumanoidBehavior/source/extensions/isaacsim.robot.policy.examples \
  --enable isaacsim.robot.policy.examples-5.2.11
```

Adding the entire source extension search folder without selecting a version can choose
the newer bundled policy extension. Confirm that the Humanoid panel says **Unitree G1**.
The VR commands below add the Python server and the hand-tracking startup request.

## Running

### Full stationary example

Run [`tools/launch_isaac_vr.bat`](tools/launch_isaac_vr.bat) from the repository root.
It defaults to the extracted 6.1 installation above; use the `ISAAC_DIR` environment
override only if your installation is elsewhere:

```powershell
.\tools\launch_isaac_vr.bat
```

Start the headset's existing streaming connection first. This launcher opens Isaac Sim XR VR
with the complete repository policy extension, enables the Python server at `127.0.0.1:8226`,
and requests the native OpenXR hand-tracking component before XR starts. It preserves the
selected runtime and gaze configuration.
Do not launch a second Kit process when one is already open.

Manual Windows launch from the installation directory:

```powershell
.\isaac-sim.xr.vr.bat `
  --ext-path "C:/path/to/IsaacSim-HumanoidBehavior/source/extensions/isaacsim.robot.policy.examples" `
  --enable isaacsim.robot.policy.examples-5.2.11 `
  --enable isaacsim.code_editor.python_server `
  --/xr/openxr/components/omni.kit.xr.openxr.ext.hand_tracking/enabled=true
```

Linux equivalent:

```bash
./isaac-sim.xr.vr.sh \
  --ext-path /path/to/IsaacSim-HumanoidBehavior/source/extensions/isaacsim.robot.policy.examples \
  --enable isaacsim.robot.policy.examples-5.2.11 \
  --enable isaacsim.code_editor.python_server \
  --/xr/openxr/components/omni.kit.xr.openxr.ext.hand_tracking/enabled=true
```

In the application:

1. Open **Window → Examples → Robotics Examples**, then **Policy → Humanoid** (Unitree G1).
2. Click **LOAD**, then **Play**.
3. Start with objects on the small table in front of the robot, slightly to its right.

The full example provides the camera, manipulation, gaze visualization, and session recorder.
For desktop inspection, launch `isaac-sim.bat` instead; keyboard locomotion is still disabled
in the stationary example.

### Apply an update to an existing session

Save your stage, restart Isaac Sim to reload Python, and LOAD a fresh example. A previous
walking scene must be rebuilt to create the fixed pelvis anchor. The extension and launcher
now enable Kit's native hand component; if it is enabled after an XR instance already exists,
restart the **XR session** as well so that instance requests the skeletal extensions.

The standalone `g1_standalone.py` is a separate walking/hand demo, not this stationary VR
workflow. Its commands are listed under [optional experiments](#optional-moving-mode-experiments).

## Controls

### Stationary controls at a glance

| Input | Current behavior |
|---|---|
| Valid optical wrist and middle-knuckle positions | Move the matching arm automatically; no grip, pinch, or fist required |
| Rotate a tracked open palm | Align the robot palm anatomically, independent of its starting wrist angle |
| Bend one tracked finger | Bend that robot finger independently |
| Move thumb across the palm | Control thumb opposition separately from thumb flexion |
| Controller side grip | Hold to move/rotate that arm; release holds its pose in physical mode while valid finger input continues |
| Controller index trigger | Curl all fingers and oppose the thumb; grip alone still leaves fingers open |
| B | Set the current headset orientation as the forward view |
| Y | Force both hands open; reopen/release your input before closing is rearmed |
| Headset rotation | Turn the view left/right, up/down, and in roll at the mounted eye position |
| Left stick click | Keep the robot-mounted camera mode |
| Sticks, X/A, keyboard movement keys, gamepad locomotion, headset gait | No base movement in stationary mode |

The camera position follows the live robot head/torso body. Headset rotation is applied
relative to the first valid orientation, so face forward when tracking starts; **B**
sets forward again. Headset translation is ignored, including while physics is paused.
If head tracking disappears, the view holds its last relative rotation until tracking
returns. The G1 asset has no movable neck joint: this rotates the camera view, while
the visible head mesh remains fixed to the torso. Eye tracking is unchanged. The
retained moving-camera modes are inactive in this default mode.

### Pickup with Touch controllers

1. Release trigger and **hold grip** to acquire the arm at its current position.
2. Move and rotate the controller to place the **actual robot palm/fingers** around the
   small cube or cylinder on the nearest front-right table. Grip alone keeps fingers open.
3. Keep grip held and pull the **index trigger** gradually. It curls every digit; thumb
   opposition follows the thumb curl. Contact must enclose and support the object.
4. Keep grip and trigger held while moving the controller to lift/carry.
5. Release the trigger to open. Releasing grip holds the arm's measured pose; the
   trigger still controls the fingers, so grip release alone does not command a drop.

After **Y**, release the trigger before another pickup. Y forces the hands open even
while a trigger remains held, until an open input rearms closure. Finger-input loss
opens the hand and clears contact state; valid input can resume without an extra
open-then-close requirement unless Y is still latched.
A target marker or highlight alone does not mean the actual robot hand has reached the object.
These instructions describe physical mode. The legacy assisted mode uses its documented
0.60/0.35 trigger thresholds to attach and detach a fixed joint.

### Open hands and individual fingers

Put down the controllers and use the headset's bare-hand mode. Keep hands visible to the
tracking cameras. The runtime must supply wrist/palm and named finger landmarks, such as
`index_proximal`, `index_intermediate`, `index_distal`, and `index_tip`.

**Tracking starts from valid data, regardless of whether the hand is open or closed.**
Present an open hand, move it to guide the arm, and bend each finger separately. No activation
button or fist is required. Move the thumb across the palm to oppose the other fingers,
then **pinch or wrap around the object itself**. Physical pickup requires opposing
contacts and friction, not a whole-hand curl threshold or a fist held nearby. Open
your fingers to release. Start with the small cube and cylinder, not the larger props.

For the cube, oppose the thumb farther across the palm so its pad faces the index
pad on the other side of the object. A shallow edge pinch can slip despite closing
the fingers; adjust the actual opposing contact before lifting.

Each hand selects its input independently. Individual optical fingers use their own joint
chains. Missing data relaxes affected fingers. Loss of the arm pose stops tracking
and holds the measured arm pose in physical mode, preventing an automatic return
through the table. Valid finger input remains independent. Complete finger-input
loss commands open.
Y is accepted only from a physical controller, so a stale or emulated button on a
bare-hand device cannot force both hands open. After a controller drop, release
the trigger or open all four tracked fingers to rearm; the thumb can remain
naturally relaxed. Missing finger landmarks cannot count as that release gesture.
Tracking loss alone does not latch this requirement; valid finger input resumes
when it returns.

The Inspire hand has six driven actuators per side: four finger proximals, thumb pitch,
and thumb yaw. Distal joints follow mimic couplings. It reproduces independent digit curl
and thumb opposition, not every human knuckle angle or finger-splay movement separately.
Controllers retain trigger-based whole-hand closure and coupled thumb opposition;
that is not individual optical tracking.

### What constitutes skeletal tracking

OpenXR interaction poses named `grip`, `aim`, `palm`, `pinch`, or `poke` are not a finger
skeleton. A hand-shaped avatar, controller emulation, or `source="hand"` label alone is
also insufficient. Kit's native hand component can set that label before publishing joints.
The controller fallback therefore checks valid Touch grip poses/actions, while partially
available skeletal hands remain optical.

On the tested SteamVR connection, the runtime advertised `XR_EXT_hand_tracking`, but an
earlier Kit instance had not requested it. The new hand-component setting fixes that
application setup omission without changing the working gaze runtime. Earlier observations
were intermittent. After the user started SteamVR independently, the 12 September recording
provided 1,627 optical samples on each hand and 1,627 measured-eye samples during its last
recorded minute. This confirms working transport in that interval, not anatomical accuracy
or acceptance of later arm-control changes.

### Palm angle, arm bending, and latency

Bare hands now use an anatomical frame built from the wrist and index, middle, and little
knuckles. The robot frame comes from its fixed knuckle anchors. A flat hand therefore asks
for a flat robot palm immediately; it no longer inherits an initial 90-degree offset.
The position target is the wrist-to-middle-knuckle midpoint on both operator and robot.
Missing endpoints end arm tracking; missing transverse landmarks leave position tracking
available while withholding orientation. Finger tracking remains independent.

The arm already uses seven-joint IK. Reach has first priority, wrist alignment second,
and a comfortable default posture resolves redundant motion. A large wrist correction
cannot consume the reach's step budget. Joint limits remain enforced. Actual human elbow
swivel is not measured by hand tracking, so the robot's elbow pose is inferred.

A free-space extension replay confirmed that both arms already approach full extension:
the palm reached **48.87 cm forward of the robot base**, with about **0.28° / 0.34°**
remaining elbow bend,
no joint-health fault, and clean retraction. The report is
`_compat61/arm-extension-baseline.json`. No IK, gain, robot-dimension, or joint-limit
change was needed. Objects were moved closer instead; see the current coordinates
under [Scene and pickup behavior](#scene-and-pickup-behavior).
Targets beyond the real arm workspace cannot be reached, and full extension leaves
less freedom to align the wrist. This reach check does not establish a successful grasp.

The last uninterrupted minute of `session_2026-09-12_02-02-51` advanced 16.26 simulated
seconds in 59.992 real seconds: real-time factor **0.271**. Input filters now use bounded
wall-clock intervals to avoid multiplying filter delay by that slowdown. Joint-speed
limits still use physical simulation time. These changes do not establish real-time
performance; the full simulator and recording overhead must be measured separately.

Enable **Window → Extensions → Python Server**, keep the scene playing, and run:

```powershell
python tools/profile_teleop_live.py --seconds 20 --timeout 45
```

The profiler reports simulation speed, application updates, and control/recording callback
times. Its temporary wrappers are restored afterwards; gaze and camera are not instrumented.
Restart Isaac and LOAD a fresh example to apply the source changes before judging them.

### Next research phase

First validate light table contact and withdrawal, bilateral palm alignment, reachable
arm poses, tracking loss, repeated physical pinch/wrap/lift/release on the small objects,
and wall-clock latency. The following milestones are designs, not enabled features
or a new training pipeline.

**Webcam body posture.** A separate local helper could run pretrained MediaPipe Pose
Landmarker and supply shoulder/elbow posture preferences. Keep Quest palms, fingers,
and eyes authoritative and the robot base stationary. Pose Landmarker supports Windows
and estimates 33 landmarks; its hip-relative 3D output is not a measured transform into
the Isaac stage. Calibrate handedness, camera/XR axes, body proportions, and timestamps;
use only fresh, confident elbow cues in the IK freedom remaining after the hand task.
Fade back to the existing posture preference on occlusion or stale input.
[Pose guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python),
[Windows support](https://developers.google.com/edge/mediapipe/solutions/setup_python)

Test webcam tracking while wearing the actual headset before enabling fusion: the
official model card warns about face occlusion and does not cover accurate metric depth
or an unavailable head. An RGB webcam is a first experiment; depth hardware can add
measured distance but cannot recover a limb hidden behind another surface. No webcam
capture or fusion is included in the current grasp milestone.
[BlazePose model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf),
[Depth projection](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)

**Velocity and gaze intent.** Later, combine the existing gaze observations with hand
approach direction, finger closing speed, closure, and object context to infer a
gentle-to-firm grip request. These are intent cues, **not measured finger pressure**.
Keep the gaze reader unchanged. Use real timestamps, explicit tracking-gap handling,
and calibrated contact/slip feedback for any force-adaptation experiment. Compare
fixed-grip, speed-only, and gaze-plus-speed baselines on success, slip/drop rate,
contact force, and latency before training on held-out objects and sessions. No new
intent model or training pipeline is implemented yet.

## Validation

Run one live test at a time against the already-loaded application. Do not run a synthetic
replay at the same time as real-hand observation.

| Check | What it verifies | What it does not establish |
|---|---|---|
| Offline regression suite | **242 tests passed**, including head rotation, release-latch recovery, contact limits, input-loss arm holding, joint armature, and articulation health | Headset transport, rendered behavior, or live grasp success |
| Table contact live replay | The reproduced 20-cycle close/open failure passes with a 1e-4 kg·m² hand-joint armature floor, including clean withdrawal and zero faults/base motion | Every contact trajectory, hardware tracking, or all hand orientations |
| Contact grasp live replay | Earlier smaller-object combined replay failed because the cube slipped; isolated successes retain their original configurations | Pickup of the enlarged objects, reliable cube pickup, or hardware tracking |
| Humanoid live replay | Actual IK, rigid-body physics, grip, assisted pickup/lift/release, and root stability | Default physical grasping or hardware tracking accuracy |
| Finger live replay | Measured movement of all ten digits and both thumb-opposition joints, source transitions, tracking loss | Real headset camera recognition |
| Historical fixed-camera replay | Former constant camera-to-body transform during Play/Pause and simulated head motion | Current rotation-only camera behavior or detailed compositor behavior for every runtime |
| Real-hand observer | Actual XR landmarks, optical commands, and measured robot-joint variation | Guaranteed tracking accuracy or detection from an unseen hand |

From the repository root, use Isaac Sim's bundled Python for the offline suite:

```powershell
& "C:/path/to/isaac-sim/python.bat" tools/run_humanoid_tests.py
```

```bash
/path/to/isaac-sim/python.sh tools/run_humanoid_tests.py --isaac-sim-dir /path/to/isaac-sim
```

For the experimental physical mode, LOAD a fresh scene, initialize physics, and enable
the Python server. Test light table contact with ordinary host Python:

```powershell
python tools/validate_table_contact_live.py --timeout 240
```

The default right-hand replay uses synthetic optical input to approach an empty area
of the front table, lower the hand, command a further 1 cm downward motion, hold for
1.5 simulated seconds, and withdraw. It records contact forces, measured joints,
waist motion, and root stability in `_compat61/table-contact-live.json`. It restores
input and leaves the scene paused. A pass covers that trajectory; it does not establish
headset tracking quality or every finger/table orientation. Run live scripts one at
a time, and inspect each report's result.

Earlier open and curled right-hand runs at **1000 rad/s coupling and 0.12 rad lead**
passed with actual table contact and a
1 cm inward command. Peak waist deflection was 0.00781 rad (0.45°) with the open
hand and 0.004744 rad (0.27°) with curled fingers. Both had zero base movement and
zero callback errors. Reports are `_compat61/table-contact-production-open.json`
and `_compat61/table-contact-production-curled.json`. These results do not validate
the current **500 rad/s, 0.18 rad** defaults; their fresh-scene recheck is recorded
separately below. Real headset handling remains an operator check.

Later fresh-default right-hand open and curled trials also passed at **500 rad/s,
damping ratio 1, and 0.18 rad lead**, recorded in `_compat61/table-final-right-open.json`
and `_compat61/table-final-right-curled.json`. The transition to a left-hand trial
then exposed an automatic right-arm return through the table on input loss; the
joint-health guard paused that transition. Those isolated passes do not validate the
full sequence. The corrected handoff now passes with arm-pose holding on input loss
and predictive contact constraints enabled.

**Earlier fixed-pose table-contact checks passed sequentially in one scene**:
right open, right curled, then left open. Each ran for 12.9 simulated seconds and
withdrew cleanly. All three had zero base movement, articulation faults, and callback
errors. Maximum waist deflection across the sequence was 0.008452 rad (0.484°).
Reports: `_compat61/table-stable-right-open.json`,
`_compat61/table-stable-right-curled.json`, and `_compat61/table-stable-left-open.json`.
The following same-scene pickup run passed the cylinder but the cube slipped with
the shallow thumb profile. The cube retry passed after increasing thumb opposition;
the final fresh-scene pickup recheck nevertheless slipped the cube again. This is a
failed combined pickup validation, not a repeatable grasp result.

These table trials held each selected finger pose; they did not repeatedly close and
open the hand while loaded by the table. A later 20-cycle dynamic stress test failed
on its first closing phase at 9.8 simulated seconds, with a measured joint speed of
100.868 rad/s (`_compat61/dynamic-table-before.json`). The sudden pause therefore
reflects a real unstable joint state. The selected fix changes joint inertia while
the guard threshold remains unchanged.

The selected **1e-4 kg·m² armature floor** passed the same 20-cycle test, including
withdrawal: 29.36 simulated seconds, peak joint speed 8.294 rad/s, maximum waist
deflection 0.015026 rad (0.861°), zero root motion, and no articulation fault or callback
error. Evidence: `_compat61/dynamic-table-armature1e4.json`. Run this dynamic check
against a freshly initialized scene with the Python server enabled:

```powershell
python tools/validate_table_contact_live.py --close-open-cycles 20 --timeout 300
```

A separate test used recorded **measured arm joint positions** as smoothed,
contact-limited targets and passed 27.4 simulated seconds, with peak joint speed
15.243 rad/s, all 2,740 contact reads healthy, zero root movement, and no fault or
callback error (`_compat61/stop-fix-recorded-path.json`). This is a reconstruction
from robot motion, not an exact replay of raw optical landmarks.

The enlarged **6 cm diameter, 10 cm cylinder** passed with palm offset
`(0.003, 0.015, 0.070)` world metres: minimum lift 0.110497 m, opposing support
throughout its 2.01 simulated-second hold, and a 0.130688 m fall after release.
Evidence: `_compat61/larger-cylinder-pickup.json`. These are new-size results,
separate from the earlier 4 cm cylinder trials. The enlarged cube's first high
approach touched the top with its thumb and the side with its index; those normals
were not opposing, and the pickup failed. The lower cube approach at world offset
`(0.008, 0.013, 0.035)` also failed to establish opposing support, without instability
(`_compat61/larger-cube-side-pinch.json`). The higher `(0.008, 0.013, 0.050)` and lower
approaches both remain recorded as failed grip trials.

The stop-fix handoff at **12 September 2026, 09:58:36 UTC** restored a fresh production
scene, initialized and paused, with live input/contact readers restored. All 24 hand
armatures read approximately `1e-4 kg·m²`; the enlarged dimensions were confirmed,
markers were hidden, and no articulation fault or callback error was present.
Isaac reported an empty XR-device list. This records a clean simulation state,
not verified real Quest input. Evidence: `_compat61/stop-fix-ready-state.json`.
That snapshot predates the four additional practice shapes.

In a fresh scene with the original cube and cylinder on the front table, recheck
their pickup using:

```powershell
python tools/validate_contact_grasp_live.py --object both --timeout 300
```

This replay moves the simulated arm and fingers using synthetic optical landmarks;
production IK, contacts, and PhysX remain active. It records each phase, contact forces,
object motion, and pass/fail checks in `_compat61/contact-grasp-live.json`. It does not
reload the scene, move objects directly, or create grasp joints. On exit it restores
live input, commands a drop, and leaves the timeline paused. Open your input before
resuming after this forced-drop cleanup. Each run reports its own pass/fail status;
headset hardware is outside this synthetic test's scope.
`--object both` selects the original cube and cylinder, not the four added shapes.
A combined trial using **500 rad/s coupling, damping ratio 1, and 0.18 rad contact
lead**, with the earlier **4 cm cube and 4 cm diameter, 8 cm cylinder**, lifted the
cube at least 11.535 cm and cylinder at least 11.463 cm,
held both for two simulated seconds with opposing contact throughout, and released
them freely. This was an intermediate success: final fresh-scene reproduction failed
on the cube and does not establish reliable gripping.
Earlier rigid-coupling passes predate the table-contact repair. See configurations
and current results in
[the validation record](docs/validation/isaac-sim-6.1.md).

The later same-scene run exposed a shallow cube pinch: thumb curl `0.35` and
opposition `0.30` slipped, while the cylinder passed with an 11.4844 cm minimum lift.
The cube retry passed at thumb curl `0.45` and opposition `0.40`, lifting at least
11.3552 cm, holding with opposing contact throughout, and falling 11.3632 cm after
opening. This more opposed thumb/index pad contact remains the **exploratory cube replay
default**, but a later fresh-scene retry slipped again. It does not change human-hand
retargeting or establish a robust grasp. Cylinder defaults remain thumb curl `0.55`
and opposition `0.50`. Coupling `500/1` and the `0.18 rad` lead were unchanged.
The final fresh-scene report `_compat61/contact-stable-default-profiles.json` records
the combined failure. Its cylinder trial passed with a minimum lift of 11.4477 cm,
opposing support throughout a 2.01 simulated-second hold, and an 11.4704 cm fall after
release. All 2,943 contact reads were healthy, with no articulation faults, callback
errors, or root movement. This separates the remaining cube-grasp limitation from
the tested table-stability repair. Keep this failure alongside isolated successes
when assessing grasp quality; see [the validation record](docs/validation/isaac-sim-6.1.md).

The existing humanoid replay requires **legacy assisted mode**; set
`EX._grasp_mode = "assisted"` before loading/rebuilding its example. It expects an
attachment and does not validate default physical grasping. With the example loaded
and the Python server enabled, use ordinary Python for live replays:

```powershell
python tools/validate_humanoid_live.py --reload-example --timeout 300
python tools/validate_camera_live.py --timeout 180
```

For the full finger replay, restart Isaac Sim, LOAD a fresh example, press Play,
and wait until physics has initialized all six finger roles per hand. This avoids
reusing the post-pickup scene configuration:

```powershell
python tools/validate_fingers_live.py --timeout 420
# Shorter verification of controller transitions only:
python tools/validate_fingers_live.py --controllers-only --timeout 120
```

These replay inputs and leave the timeline paused afterward. The humanoid replay also
exercises highlight transitions. Actual robot joints, collisions, and pickup constraints
remain live. Save stage edits before using `validate_humanoid_live.py --reload-example`:
that option clears and rebuilds the example for fresh package positions.

The recorded stationary checks on 7 September 2026 included a 1,286-step humanoid replay
with a **0.1292 m lift**, unchanged root pose, and no callback errors. The full finger replay
passed **19.73 simulated seconds**, including independent 0.7 normalized targets on all ten
fingers and both thumb-opposition joints. These are historical assisted simulation
results, not acceptance of the new physical grasp mode or later hardware changes. See the
[validation details](docs/humanoid-control.md#automated-validation) for scope and measurements.

The [11 September Isaac Sim 6.1 validation](docs/validation/isaac-sim-6.1.md) passed
pickup/release, the historical fully fixed camera, and the full finger replay in an
initialized fresh scene. The finger replay failed on the right little finger when run after pickup;
the cause of that condition-dependent failure remains unresolved. An earlier
headset observation supplied no optical skeletons or unified eye gaze; the later
12 September recording confirms recovery after the user started SteamVR
independently. New palm/arm changes still await live acceptance. Gaze code and
settings were not changed.

### Real-hand observation

Press Play, wear the headset, put down the controllers, and move individual bare fingers
while running:

```powershell
python tools/observe_hand_tracking_live.py --seconds 20
```

The observer supplies no inputs and changes no timeline, gaze, or tracking settings. It
uses **wall time**, reports real pose names and validity, and compares optical targets
with measured robot-joint ranges. Motion flags indicate observed optical commands and
corresponding robot movement; they do not measure anatomical accuracy. Missing movement
can mean occlusion, a still hand, missing transport, or too few physics updates.

For the fist-only report, compare an interval with fingers open/moving separately against
one with a closed hand. If landmarks disappear when the hand opens, the input is missing
before retargeting. Increasing grasp thresholds or forcing a fake skeleton will not correct
that. If landmarks remain valid but the robot stops following them, preserve the observer
report and Kit log to investigate the control path.

CPU physics defaults to **100 simulated steps/second**, with a **90 Hz requested rendering
cadence**. These settings do not guarantee wall-clock FPS. Slow VR rendering stretches
simulation-time replays and motion response. See
[performance and CPU physics](docs/humanoid-control.md#performance-and-cpu-physics).

## Scene and pickup behavior

The scene uses NVIDIA's `full_warehouse.usd`. The small table at approximately
`(0.50, -0.28)` is the initial stationary workspace. The anchored robot cannot walk
to the side benches or squat to the floor.

The current physical-grasp scene spaces six distinct shapes across the nearest
right-front table. The sphere, cube, and orange cylinder form the nearer row; the
tall cylinder, flat block, and cone sit in a staggered second row. They are lightweight
dynamic bodies with real collision and gravity:

| Color | Shape | Dimensions | Mass | Center X, Y (m) |
|---|---|---|---:|---|
| Blue | Cube | 6 × 6 × 6 cm | 60 g | 0.345, -0.140 |
| Orange | Cylinder | 6 cm diameter × 10 cm high | 80 g | 0.345, -0.360 |
| Green | Sphere | 7 cm diameter | 50 g | 0.345, 0.100 |
| Purple | Tall thin cylinder | 4 cm diameter × 14 cm high | 60 g | 0.465, 0.100 |
| Yellow | Flat rectangular block | 9 × 6 × 3 cm | 70 g | 0.465, -0.065 |
| Red | Cone | 7 cm diameter × 10 cm high | 50 g | 0.455, -0.245 |

The original cube/cylinder retain their dimensions and masses. The four additions
provide different surfaces and thicknesses for practice; their pickup behavior has
not been validated. Existing cylinder success and cube limitations apply only to
their recorded trajectories, before the current position refresh. Moving objects
closer does not establish new pickup success or increase the robot's physical reach.
The varied YCB package/floor layout remains in legacy
assisted mode.

**Physical contact is the experimental default.** The earlier fixed-pose table sequence
passed after the input-loss arm-hold fix, but repeated close/open contact later failed.
The selected hand-joint armature floor now passes that same 20-cycle dynamic test;
the reconstructed recorded-motion path and enlarged-cylinder pickup also pass.
Both tested enlarged-cube pinches failed to establish opposing support without
destabilizing the robot; cube-grasp reliability remains unresolved. Earlier smaller-cube
pickup was pose/contact-path-sensitive; successful lifts did not establish repeatability.
Real headset handling is a separate check. Bounded finger drive targets close the
actual articulated digits against the object. The
first contact stores a separate closing cap for each actuator/contacted body at the
measured curl plus **0.18 rad**, converted to that actuator's range. That ceiling
stays fixed while the contact persists. Each command is separately bounded by the
request, current measured curl plus the lead, and the ceiling. This preserves
preload during gradual closing without following a finger rolling over the object.
A gap of up to **40 ms of simulation time** retains
only this command cap; missing opposing contacts clears support evidence immediately.
Opening always takes priority; fully open finger and thumb-opposition input, tracking
loss, or reset clears the caps. Thumb opposition is contact-limited even with straight fingers.
The closing limit also responds to measured contacts with scenery and other robot links,
including predictive contacts with a positive gap. Only eligible-object contacts
within 2 mm can contribute to opposing pinch/wrap support evidence. The
object remains dynamic and must be supported by contacts and friction; no hand-to-object
fixed joint, object teleport, or kinematic attachment supplies support. A candidate
highlight is guidance only. Open to release; Y forces open until your input opens and
rearms closure. Contact-based status is recorded separately from legacy attachment state.

Physical mode prepares the hand's existing mimic relationships with finite,
critically damped compliance (`naturalFrequency = 500` rad/s, `dampingRatio = 1`)
before Play. The asset's original soft settings allowed excessive thumb deflection;
the subsequent rigid `0/0` setting passed simple pickups but could destabilize the
whole articulation against the table. The current fix preserves gearing, offsets,
limits, references, and all drive settings; passive knuckles remain undriven.
Assisted mode retains the asset's original coupling. These are simulation settings,
not calibrated tendon properties or a finger-pressure controller.
The instrumented physical props stay awake so static contact forces remain observable.

Physical mode also applies an artificial armature floor of **1e-4 kg·m²** to all 24
hand revolute joints before Play. This adds joint-space inertia to reduce abrupt
acceleration of light finger links. Higher authored values and existing per-axis
overrides are respected. Passive joints remain undriven and receive no new independent
velocity caps. This is numerical stabilization, not a calibrated motor/rotor model.

Measured hand contacts against non-graspable external colliders also limit the final
arm command after smoothing. The limit blocks motion into each contact normal,
including fingertip motion caused by wrist rotation, while allowing withdrawal and
tangential motion. All six graspable practice objects are excluded so the arm can carry them.
Other external colliders are treated as obstacles even if they are dynamic; the
robot's own links are excluded from this arm limit.
Loaded predictive contacts are included even at positive separation; the 2 mm
support-diagnostic threshold does not discard an obstacle that already exerts force.
This reacts to observed contact; it is not a collision-free path planner. Invalid
contact reads hold the arm at measured joints and command fingers open until healthy
readings return. Gaze, camera mounting, stationary anchoring, and hidden target spheres
retain their existing behavior.

For historical comparisons, set `EX._grasp_mode = "assisted"` on the example **before
loading/rebuilding the scene**. There is no UI mode selector. Assisted mode retains
distance-gated fixed-joint pickup and the settings below; they are not physical-grasp
success criteria. In that mode coincident attachment frames preserve relative pose,
and release/drop/tracking loss/reset remove the constraint.

| Setting | Default | Purpose |
|---|---|---|
| `_grab_assist_radius` | 0.30 m | Green candidate highlight |
| `_grab_radius` | 0.18 m | Search around the measured palm/finger center |
| `_grasp_contact_distance` | 0.09 m | Object center to nearest palm/finger link center |
| Controller close/release | 0.60 / 0.35 | Trigger hysteresis |
| Optical close/release | 0.55 / 0.35 | Mean five-finger curl hysteresis |

A green candidate can still be too far away to grasp. Gaze alone does not pick up an
object. An assisted fixed joint can support an object an unassisted hand could not;
keep mode labels when comparing recordings and do not treat the published assisted
videos as validation of the new physical mode.

Yellow gaze and green candidate highlights share pre-authored session-layer overrides.
Runtime selection changes only material properties; it does not recreate collision prims.
This preserves the physics tensor views that the earlier highlight/grip freeze invalidated.

## Behavioral Data Collection

Every example load creates a session under `~/BehavioralCollection/raw_sessions/`:

```text
session_YYYY-MM-DD_HH-MM-SS/
├── metadata.json
├── behavior.csv
├── hand_tracking.csv
├── gaze.csv
├── object_states.csv
├── frame_timestamps.csv
└── frames/eye_camera/*.png
```

`metadata.json` records robot/mode settings, time steps, log rates, fixed-base and grasp
configuration, and finger roles. Hand mechanics are labeled by `hand_mimic_coupling`,
`hand_mimic_natural_frequency_rad_s`, `hand_mimic_damping_ratio`, and
`hand_mimic_joint_paths`. Current physical recordings use `compliant`; earlier `hard`
recordings describe the superseded rigid setting. Physical `contact_supported` and
`contact_hand` status is separate from `is_grabbed`, which remains a legacy fixed-joint attachment field and
is false for unassisted physical objects. Sensor CSV files share `unix_time`, `sim_time`, and
`step_index` for alignment. `frame_timestamps.csv` instead contains `frame_id`, `unix_time`,
`sim_time`, `camera_name`, and `image_path`; align images by timestamps. The default
requests one sensor row per physics step, nominally
100 Hz in **simulation time**, with 256×256 eye-camera PNGs approximately every tenth step.
Frame timestamps identify captured images; wall-clock capture rates depend on runtime performance.

Hand inertia is recorded in `hand_joint_armature_floor_kg_m2`, the actual authored
`hand_joint_armature_kg_m2_by_path` map, and `hand_joint_armature_meaning`. Use the
actual values when comparing trials; the floor is numerical stabilization, not
measured hardware inertia. Earlier recordings without these fields predate this fix.

CSV buffers flush at a nominal 2.5-second simulation interval and on orderly cleanup.
An abrupt shutdown can lose buffered rows. Save/review complete sessions before changing
recording schemas or training on them.

The contact-grasp replay writes `validation_<run_id>.json` into the active session
when one exists. It labels `source` as `synthetic_hand_replay` and records start/end
Unix time, simulation time, and physics step. Use these interval boundaries when
filtering training or analysis data: the rest of that same recording may contain
real user input. The replay report includes the marker path in `recording_provenance`.

| File | Contents |
|---|---|
| `behavior.csv` | HMD motion, robot pose, base commands, gait state, and joint states |
| `hand_tracking.csv` | Hand/controller poses, buttons, source labels, finger targets, and hand closure |
| `gaze.csv` | Gaze origin/direction, hit point/object, and explicit source |
| `object_states.csv` | Sample-object poses, velocities, grasp mode, contact evidence/health, and legacy attachment state |
| `frame_timestamps.csv` | Alignment between sensor rows and eye-camera frames |

### Finger columns in `hand_tracking.csv`

| Column | Meaning |
|---|---|
| `<side>_finger_thumb/index/middle/ring/little` | Smoothed operator curl request, 0=open and 1=closed |
| `<side>_finger_thumb_yaw` | Smoothed thumb-opposition request |
| `<side>_finger_target_<role>` | Applied curl target after physical contact limits, including `thumb_yaw` |
| `<side>_contact_supported/object/mode` | Opposing-contact support evidence, object path, and `pinch`/`wrap` classification |
| `<side>_hand_closure` | Mean curl across the five fingers, excluding thumb opposition |
| `<side>_finger_source` | `hand_tracking`, `controller`, or `none` |

Optical curl sums adjacent bone bends from the metacarpal to the tip. Full-flexion references
are 150° for fingers and 95° for the thumb; angles beyond 180° do not reverse the resulting
curl. Thumb opposition uses a palm-local frame. Missing fingers relax independently.
The request columns describe smoothed operator input, and target columns describe
commands sent to the robot after contact limits; `behavior.csv` joint
states describe its measured response.

### `behavior.csv` schema

| Column group | Columns | Description |
|---|---|---|
| **Timestamps** | `unix_time`, `sim_time`, `step_index` | Wall-clock (Unix epoch, float), simulation time (s), sample counter |
| **HMD position** | `hmd_pos_x`, `hmd_pos_y`, `hmd_pos_z` | Headset position (m), normally in the runtime's physical tracking frame; see coordinate-frame caveat below |
| **HMD orientation** | `hmd_qw`, `hmd_qi`, `hmd_qj`, `hmd_qk`, `hmd_yaw` | Quaternion + extracted yaw (rad) |
| **HMD velocity** | `hmd_vel_x`, `hmd_vel_y`, `hmd_vel_z`, `hmd_horiz_speed` | Filtered velocity (m/s) and horizontal speed magnitude |
| **Gait signal** | `gait_filtered_h`, `gait_vel_sign`, `gait_horiz_gate`, `gait_output`, `gait_pulse_rem` | Internal gait detector state |
| **Step event** | `step_event` | `1` on the exact sample when a footfall is detected, `0` otherwise |
| **Robot pose** | `robot_pos_x/y/z`, `robot_qw/qi/qj/qk`, `robot_yaw` | G1 base world pose |
| **Commands** | `cmd_forward`, `cmd_lateral`, `cmd_yaw` | Locomotion commands driving the base (m/s, rad/s) |
| **Joint states** | `j_<joint_name>_pos`, `j_<joint_name>_vel` | Position (rad) and velocity (rad/s) for every one of the G1's 53 DOFs (29 body + 24 finger) |

The HMD reader prefers the runtime's physical `get_pose()`/`get_raw_pose()` data and falls
back to a virtual-world pose when necessary. The cached HMD sample has no frame/source
column, so recorded HMD coordinates are not guaranteed to share the robot's USD Z-up
world frame. Confirm the actual pose source and transform before combining these columns
with robot or object coordinates; derived HMD velocity/yaw require the same care.

Stationary sessions have zero base locomotion commands. HMD features and the retained gait
columns must not be interpreted as evidence that the robot walked. For manipulation learning,
align hand inputs and targets with measured joints, object states, gaze source, and frames.
The [`learning/`](learning/) scaffold provides the downstream data/model workflow; it does
not replace inspecting the session metadata or verifying hardware input quality.

### Gaze source and camera frames

Each gaze row identifies `eye_tracker`, `hmd_forward`, or an unavailable source. HMD-forward
is a fallback intent direction, not measured eye gaze. Raycasts run at 50 Hz in simulation
time; recorded rows may reuse the latest hit between raycast ticks. Tracking loss clears
cached hits. Camera frames use the robot-mounted view in the current stationary mode.

## Quest Pro Eye Tracking (optional)

`eye_gaze_tracker.py` reads the runtime's combined ("unified") OpenXR eye-gaze
pose (`XR_EXT_eye_gaze_interaction`) — already the runtime's calibrated fusion
of both eyes — raycasts it into the PhysX scene, and draws a **thin red ray
from your eyes to the gazed point** with a **large blood-red marker sphere** at
the collision. The per-eye devices are deliberately not mixed in: on
SteamVR + Steam Link their poses carry a head-like orientation that biased the
ray toward the view center when gazing near straight ahead. The sample boxes
and the ground plane are physics colliders, so both are valid gaze targets. On
top of the visuals:

- The looked-at sample box is tinted yellow.
- Every change of gaze target prints a live terminal line, e.g.
  `[EyeGaze] looking at sample box Box_03 @ (5.21, -0.44, 0.31) m, 3.80 m away`
  (fires on transitions only, so the terminal stays readable).
- The latest raycast sample feeds `gaze.csv` at ~100 Hz; raycasts run at 50 Hz.
  `gaze_source` is `eye_tracker`, `hmd_forward`, or blank when unavailable.
  Robot hits are excluded and tracking loss clears the cached hit immediately.

### Verified working setup (SteamVR + Steam Link)

> ⚠️ **The verified eye-gaze setup uses Steam Link, connected and left connected.**
> The gaze has to reach the PC as `XR_EXT_eye_gaze_interaction`, and on this rig that has
> only ever happened while the Steam Link driver was the one carrying the headset —
> its log line `Client HMD type (3) supports eye tracking, creating input component`
> is present in every working session and absent from every broken one. Start Steam Link
> **before** Isaac Sim and leave it up: in the broken sessions Steam Link had already
> disconnected minutes before the VR session began.
>
> An earlier version of this document claimed Meta's PC runtime "never" exposes the
> extension over Link. The logs on this machine do not support that: the extension is
> enabled and `/user/eyes_ext` is an active OpenXR device in the failing sessions too.
> An active device alone does not establish that usable gaze poses reached Kit; the
> streaming path and action bindings must also work.

1. **Headset:** Settings → Movement tracking → **Eye tracking: ON** (grant the
   permission and run the eye calibration once).
2. **Headset:** Settings → Privacy & Safety → App permissions → **Eye tracking →
   allow Steam Link**.
3. **Steam Link app settings** (on the headset): **"Share eye tracking data to
   other apps on this PC": ON** — off by default, and the switch most people miss.
4. **PC:** set SteamVR as the active OpenXR runtime (SteamVR → Settings →
   OpenXR → *Set SteamVR as OpenXR runtime*), connect via Steam Link, then launch
   `isaac-sim.xr.vr.bat`.

On success the console logs `[EyeGaze] ray source: real eye tracking (unified gaze
device)` — and it logs it whenever gaze comes good, including mid-session, so you can put
the headset on, fix the toggle, and watch the line change without restarting Isaac Sim.
Kit registers the eye device as `/user/eye/unified`; the pose name in this build's
manifest is **`gaze_ext`** (`gaze` is kept as a fallback for other runtimes).

### Eye-tracking checks

- `xrCreateInstance ... XR_ERROR_API_VERSION_UNSUPPORTED` for API 1.1 at startup
  is **harmless** — Kit retries and creates the session with OpenXR 1.0.
- **`python tools/check_eye_tracking.py`** answers "is eye tracking reaching Isaac Sim?"
  in a second, without launching anything: it reads the OpenXR runtime from the registry,
  the SteamVR share setting, the Steam Link driver log, and the newest Kit session log, and
  tells you which link is broken. Watch the *date* on the Steam Link line — a stale "ok"
  from a previous day is the usual trap.
- If no gaze arrives for ~3 s of play, the tracker prints the numbered fix list above
  plus **the full list of XR devices the session can see**, on stdout where you will
  actually see it. If `/user/eye/unified` is missing, unified eye gaze is unavailable
  to Kit. Recheck steps 1–3 and the runtime/action bindings; the missing device alone
  does not identify which part failed. When a valid headset pose is available,
  `gaze.csv` falls back to `hmd_forward` rows, and `gaze_source` records the source.
- A visible ray can come from either real eye gaze or the HMD-forward fallback.
  Check the source log or `gaze_source` column to distinguish them; early-session
  poses may be unavailable until tracking becomes valid.

Toggles in `HumanoidExample.__init__()`:

```python
self._eye_gaze_enabled = True             # master switch for the tracker
self._eye_gaze_ray_visual_enabled = True  # red ray + hit marker in the scene
```

## Troubleshooting

### Hand detection starts only when making a fist

This remains a reported hardware/input issue, not a resolved feature. The application reads
valid optical skeletons with an open hand and does not gate finger updates on closure. A fist
only requests pickup. Run the [real-hand observer](#real-hand-observation) during open-hand
and closed-hand movement. A raw `hand` source with only interaction poses, or poses whose
validity flags are zero, is not usable optical finger data.

Verify that the headset is worn, bare hands are visible, and its hand-tracking/automatic
hand-controller switching mode is active. Keep the working SteamVR/Steam Link eye-gaze
configuration while diagnosing missing landmarks. The confirmed Kit startup setting is:

```text
/xr/openxr/components/omni.kit.xr.openxr.ext.hand_tracking/enabled = true
```

The launcher/extension already supply it. An existing XR instance needs an XR-session restart
to request the extension; this does not by itself guarantee the streaming app supplies hands.
A valid source plus wrist/finger landmarks must still appear.

### Grip/trigger stopped working after enabling hand tracking

Kit can temporarily label Touch devices as `hand` while exposing only generic interaction
poses. The updated classifier retains controller fallback only when a valid world grip pose
and trigger/thumbstick components are present. Partial real skeletons remain optical so
occlusion cannot convert stale button data into a grab. Restart after source changes and run
the short controller replay to check this path.

### The robot or camera still moves like the old version

Confirm that the loaded extension comes from the updated checkout. Save the scene, restart
Isaac Sim, and LOAD a fresh example. Stationary mode authors a fixed world-to-pelvis anchor;
an existing moving scene is not converted by changing a Python string in a live instance.
The default `robot_head` view keeps its position on the live head/torso mount while
following relative headset rotation. **B** resets the forward orientation; it does not
change eye-gaze calibration. The left stick cannot enable room-scale camera motion.

### Touching the table bends the robot or pauses physics

The latest sudden stops correspond to measured joint-speed faults, including a real
ring-joint observation above 108 rad/s. They are not merely missing hand input or a
false pause flag. Earlier fixed-pose table passes did not exercise repeated close/open
motion. The selected fix adds a **1e-4 kg·m²** armature floor before Play and passed
the reproduced 20-cycle close/open stress test. Passive finger joints remain undriven,
and gaze, camera mounting, and the pause threshold are preserved. This pass covers
that trajectory; real headset handling and other contact paths are outside its scope.

Restart Isaac after source changes and LOAD a fresh example. The updated physical
mode uses compliant hand couplings and removes inward arm commands at measured
scenery contacts. Test with an open hand over an empty table area: lower it gently
until the fingertips touch, slide a short distance along the surface, then lift away.
Repeat with each hand. The arm may stop short of your real hand's target while the
surface blocks it; withdrawal should remain possible. Do not judge table behavior
from a previously loaded rigid-coupling scene.

Release the controller grip or briefly hide the tracked hand near the table. Physical
mode should hold that arm's measured pose instead of returning to rest. Missing
finger input opens the hand; a still-valid trigger continues to control finger curl.
Reacquire tracking or hold grip again to resume arm movement.

An articulation-health fault automatically pauses the timeline before further
teleoperation updates. **Use Reset or reload the scene before continuing**; pressing
Play alone does not clear the fault. Keep the console's joint/fault message if it
recurs. The health guard detects invalid/nonfinite state, absolute joint angles over
20 rad, joint speeds over 100 rad/s, or joint-limit overshoot over 0.35 rad. These
bounds detect solver failure; they are not normal movement targets or hardware limits.

### A highlighted object will not lift

Physical mode does not attach highlighted objects. Use the small cube/cylinder and place
the thumb opposite the contacting fingers; pinch or wrap around the object before lifting.
Hold grip and use the trigger for controller closure. Open/release after Y to rearm;
tracking loss alone opens and clears contact state without latching a rearm requirement.
Check actual contacts and finger travel; a marker cannot
pull an object to the hand. In legacy assisted mode only, compare the actual hand/object
distance with the attachment thresholds above.

### Movement is slow or a replay times out

A fixed 0.01-second physics step is simulation time, not proof of 100 wall-clock updates each
second. VR rendering, recording, and hardware load can slow the loop substantially. Compare
`wall_seconds` and `simulated_seconds` in observer/replay reports and see
[performance and CPU physics](docs/humanoid-control.md#performance-and-cpu-physics).
Use a longer replay timeout or the controller-only check; a timeout without callback errors
is not evidence of a failed joint drive.

### Inspect an already-running session

The project launcher enables the Python server. In an existing app, it can also be enabled
from **Window → Extensions → Python Server**. Run these read-only checks from the repository root:

```powershell
python tools/kit_exec.py "print(EX.describe_xr())"
python tools/kit_exec.py "print(EX._eye_gaze_tracker.describe())"
python tools/check_eye_tracking.py
python tools/observe_hand_tracking_live.py --seconds 20
```

`EX` resolves to the loaded `HumanoidExample`. Logs are under
`~/.nvidia-omniverse/logs/Kit/Isaac-Sim XR VR/6.1/` for the new default installation
(`6.0/` contains earlier sessions). Read the newest log and preserve the first
error, rather than only the repeated downstream exceptions:

```powershell
rg -n 'EyeGaze|\[G1\]|hand_tracking|py stderr|Error' "C:/path/to/kit_current.log"
```

Use `tools/launch_isaac_vr.bat --xr-verbose` on the next launch for OpenXR initialization
information. Check log dates; a successful old Steam Link connection does not establish
that the current session receives gaze or finger data.

## Optional moving-mode experiments

These paths are retained for later work. They are **not** the current stationary
manipulation workflow or a validation of balance with moving arms/payloads.

### Locomotion choices

The interactive default in `HumanoidExample.__init__()` is:

```python
self._g1_locomotion = "stationary"
```

Explicit alternatives are `policy` and `kinematic`; rebuilding the scene is required to
change the fixed-base configuration. `policy` runs Unitree's recurrent 12-leg-joint actor
at 50 Hz; `kinematic` integrates a commanded base pose while holding posture. The latter
is also the policy-load fallback and provides no dynamic balance controller.

Policy integration depends on joint-name mapping, trained observation scales/gains,
command limits, and resetting recurrent state. The actor's training joint order is not
PhysX's articulation order. Its command limits are 0.8 m/s forward, 0.5 m/s lateral, and
1.57 rad/s yaw. Historical travel/turning measurements in earlier revisions do not qualify
these modes for current manipulation loads. See
[locomotion implementation](docs/humanoid-control.md#locomotion-and-lifecycle).

In moving modes only, keyboard Up requests forward, Left/Right yaw, and Down brakes
translation. XR left-stick axes request forward/reverse/strafe; right stick and X/A request
yaw. Headset gait remains disabled by default. Enabling `_headset_gait_enabled` cannot move
the robot while locomotion remains `stationary`.

### Standalone walking/hand demo

From the Isaac Sim install directory:

```powershell
$demo = "C:/path/to/IsaacSim-HumanoidBehavior/source/standalone_examples/api/isaacsim.robot.policy.examples/g1_standalone.py"
.\python.bat $demo --device cpu
.\python.bat $demo --device cpu --headless --seconds 20
.\python.bat $demo --device cpu --locomotion kinematic
.\python.bat $demo --device cpu --hand ThreeFinger
```

Its flags are `--headless`, `--seconds N`, `--locomotion policy|kinematic`,
`--hand Inspire|ThreeFinger|None`, `--device cuda|cpu`, and `--test`. The standalone script
still defaults to `policy` locomotion and CUDA if not overridden. It is a separate demo;
it does not provide the full stationary VR scene and behavioral collection flow above.

### Retained moving-camera modes

`head_compose`, `stage_anchor`, `custom_anchor`, and `camera_lock` are historical experiments
for a moving avatar. Physical-head calibration, alternate camera conventions, headset-height
offsets, and gait stabilization belong to those paths. They are bypassed by stationary mode,
which uses a fixed mounted eye position plus relative headset rotation. Its **B**
action only resets forward orientation. Do not apply old room-scale calibration or
camera-cycle advice to the current default.

## Source reference and tuning

| File | Responsibility |
|---|---|
| [`humanoid_example.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/humanoid_example.py) | Scene, hand/controller input, IK, pickup, camera mount, recording |
| [`g1.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/g1.py) | Standing anchor, robot joint drives, optional locomotion |
| [`xr_pose.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/xr_pose.py) | Shared world-pose validation and time-based smoothing |
| [`material_highlights.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/material_highlights.py) | Stable overlapping gaze/candidate material overrides |
| [`grasp_controller.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/grasp_controller.py) | Contact-limited closing and eligible-object pinch/wrap support diagnostics |
| [`grasp_contacts.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/grasp_contacts.py) | Prepared PhysX contact reader for hands, object/scenery identities, and reader health |
| [`arm_contact.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/arm_contact.py) | Constrains final arm commands against measured scenery-contact normals |
| [`articulation_health.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/articulation_health.py) | Detects unusable joint states and supplies a reason for latched simulation pause |
| [`eye_gaze_tracker.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/eye_gaze_tracker.py) | Existing unified-eye gaze, raycasts, and visualization |
| [`tools/`](tools/) | Offline tests, live replays, read-only inspection utilities |

The principal finger defaults are `_finger_smoothing=0.35`, `_finger_control_enabled=True`,
and the open/close thresholds listed above. Robot-side finger drive defaults are
`FINGER_STIFFNESS=20.0`, `FINGER_DAMPING=0.6`, `FINGER_MAX_EFFORT=10.0`, and
`FINGER_MAX_VELOCITY=8.0` rad/s. Only the six driven joints per hand receive position targets;
driving mimic joints separately fights their coupling. These simulation settings do not
specify hardware actuator limits.

Recording intervals are derived from physics frequency. Review metadata and timestamps
after changing them. Keep tuning localized to the subsystem under investigation; missing
skeletal input does not call for changing gaze or widening pickup distance.

## License

This file is a modification of NVIDIA Isaac Sim source code and is licensed under the **Apache License 2.0**, the same license as the upstream repository.  
See [LICENSE](LICENSE) for the full text.

Original copyright: © 2020–2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  
Modifications: © 2026 Soheil Sepahyar.

The optional bundled Unitree walking checkpoint has its own **BSD 3-Clause** provenance; see
[robots/data/README.md](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/data/README.md)
and the linked Unitree source/license. Robot and environment assets retain their respective
NVIDIA/asset-provider terms.
