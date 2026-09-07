# Humanoid VR Control — Unitree G1

This fork of [NVIDIA Isaac Sim](https://github.com/isaac-sim/IsaacSim) provides stationary
VR manipulation with a Unitree G1, Inspire five-finger hands, Quest Pro eye tracking,
and synchronized behavioral recordings.

**Current default:** the pelvis is fixed to the world. Arms and fingers remain articulated,
and the camera stays mounted to the robot's head/torso when the operator moves their real
head. Optical hands control individual fingers and thumb opposition; Touch controllers
use grip for arm movement and trigger for pickup. Gaze retains the working implementation.

**An open hand should be tracked immediately when valid optical data arrives.** A fist is
only a pickup gesture, never an activation gesture. The reported behavior where real hand
tracking appears only with a fist has **not been verified as resolved**. The simulator's
finger control passes replay tests, but the headset must supply valid skeletal landmarks.
Use the [real-hand observer](#real-hand-observation) to distinguish missing input from a
robot-control problem.

[Installation](#installation) · [Running](#running) · [Controls](#controls) ·
[Validation](#validation) · [Recordings](#behavioral-data-collection) ·
[Quest Pro gaze](#quest-pro-eye-tracking-optional) · [Troubleshooting](#troubleshooting)

For implementation details, coordinate frames, lifecycle behavior, and test coverage, see
[the developer guide](docs/humanoid-control.md).

## Installation

### Prerequisites

- Isaac Sim Standalone 6.0.0 and a compatible NVIDIA GPU/driver. Live validation for this
  project used the Windows installation; Linux commands are included below.
- Isaac Sim's bundled Python 3.12 for the offline test runner and standalone simulation.
  The TCP utility commands use ordinary host Python.
- For VR, a working OpenXR headset connection. Keep an already-working gaze runtime and
  streaming configuration; the [verified Quest Pro setup](#quest-pro-eye-tracking-optional)
  uses SteamVR and Steam Link.
- Access to the NVIDIA robot, warehouse, and YCB assets used by the example.

Install the **complete** `isaacsim.robot.policy.examples` extension. Copying only
`humanoid_example.py` omits the robot wrapper, helpers, registration, and startup settings.
Save any open scene and close Isaac Sim before replacing an installed extension.

### Option A — Copy the extension

Keep a backup of the installed extension. Run from this repository's root.

```powershell
$isaacRoot = "C:/path/to/isaac-sim-standalone-6.0.0-windows-x86_64"
$sourceExt = "source/extensions/isaacsim.robot.policy.examples"
$installedExt = Join-Path $isaacRoot "exts/isaacsim.robot.policy.examples"
Copy-Item -Path "$sourceExt/*" -Destination $installedExt -Recurse -Force
```

```bash
ISAAC_ROOT="/path/to/isaac-sim-standalone-6.0.0-linux-x86_64"
cp -a source/extensions/isaacsim.robot.policy.examples/. "$ISAAC_ROOT/exts/isaacsim.robot.policy.examples/"
```

Repeat the copy after updating the checkout, then restart Isaac Sim.

### Option B — Link the extension for development

A junction/symlink makes the installation load the checkout directly. Restart Isaac Sim
after Python changes; rebuilding the scene alone does not reload imported modules.

Windows, with permission to create a junction:

```powershell
$standaloneExt = "C:/path/to/isaac-sim-standalone-6.0.0-windows-x86_64/exts/isaacsim.robot.policy.examples"
$githubExt = "C:/path/to/IsaacSim-HumanoidBehavior/source/extensions/isaacsim.robot.policy.examples"
Rename-Item -LiteralPath $standaloneExt -NewName "isaacsim.robot.policy.examples.orig"
New-Item -ItemType Junction -Path $standaloneExt -Target $githubExt
```

To restore the backup, first verify that `$standaloneExt` is the junction you created:

```powershell
Get-Item -LiteralPath $standaloneExt | Select-Object FullName, LinkType, Target
# Only after verifying the junction: remove the link, without -Recurse.
Remove-Item -LiteralPath $standaloneExt
Rename-Item -LiteralPath "${standaloneExt}.orig" -NewName "isaacsim.robot.policy.examples"
```

Linux:

```bash
STANDALONE_EXT="/path/to/isaac-sim/exts/isaacsim.robot.policy.examples"
GITHUB_EXT="/path/to/IsaacSim-HumanoidBehavior/source/extensions/isaacsim.robot.policy.examples"
mv "$STANDALONE_EXT" "${STANDALONE_EXT}.orig"
ln -s "$GITHUB_EXT" "$STANDALONE_EXT"
```

To undo the Linux link, verify it with `ls -ld "$STANDALONE_EXT"`, remove only the
symlink with `unlink "$STANDALONE_EXT"`, and restore the `.orig` directory.

### Option C — Add an extension search folder

From the Isaac Sim installation directory:

```powershell
.\isaac-sim.xr.vr.bat --ext-folder "C:/path/to/IsaacSim-HumanoidBehavior/source/extensions"
# Desktop alternative:
.\isaac-sim.bat --ext-folder "C:/path/to/IsaacSim-HumanoidBehavior/source/extensions"
```

```bash
./isaac-sim.xr.vr.sh --ext-folder /path/to/IsaacSim-HumanoidBehavior/source/extensions
```

Kit resolves extensions by its search paths and versions. Confirm that the loaded extension
comes from this checkout; equal-version copies can make selection ambiguous. A junction is
more predictable for repeated local development.

## Running

### Full stationary example

For Windows development, edit `ISAAC_DIR` in
[`tools/launch_isaac_vr.bat`](tools/launch_isaac_vr.bat) if necessary, then run from the repository root:

```powershell
.\tools\launch_isaac_vr.bat
```

Start the headset's existing streaming connection first. This launcher opens Isaac Sim XR VR,
enables the Python server at `127.0.0.1:8226`, and requests the native OpenXR hand-tracking
component before XR starts. It preserves the selected runtime and gaze configuration.
Do not launch a second Kit process when one is already open.

Manual Windows launch from the installation directory:

```powershell
.\isaac-sim.xr.vr.bat --enable isaacsim.code_editor.python_server --/xr/openxr/components/omni.kit.xr.openxr.ext.hand_tracking/enabled=true
```

Linux equivalent:

```bash
./isaac-sim.xr.vr.sh --enable isaacsim.code_editor.python_server \
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
| Valid optical wrist/palm pose | Move the matching arm automatically; no grip, pinch, or fist required |
| Bend one tracked finger | Bend that robot finger independently |
| Move thumb across the palm | Control thumb opposition separately from thumb flexion |
| Controller side grip | Hold to move/rotate that arm; release to end tracking and drop its object |
| Controller trigger | Curl all fingers and request pickup near an object |
| B | Restore the fixed robot-head view |
| Y | Drop both objects; open/release before grabbing again |
| Left stick click | Keep the camera locked to the robot head |
| Sticks, X/A, keyboard movement keys, gamepad locomotion, headset gait | No base movement in stationary mode |

The camera follows the live robot head/torso body through translation and rotation. Moving
or turning the physical headset cannot detach it, including while physics is paused.
The retained moving-camera modes are inactive in this default mode.

### Pickup with Touch controllers

1. Release trigger and **hold grip** to acquire the arm at its current position.
2. Move and rotate the controller to bring the **actual robot palm/fingers** close to a
   near-edge object on the small front-right table. Grip alone keeps the fingers open.
3. Keep grip held and pull trigger to at least **0.60** to close and request attachment.
4. Keep grip and trigger held while moving the controller to lift/carry.
5. Release trigger below **0.35** to drop. Releasing grip also drops the object.

After **Y** or tracking loss while holding, release trigger before another pickup.
A target marker or highlight alone does not mean the actual robot hand has reached the object.

### Open hands and individual fingers

Put down the controllers and use the headset's bare-hand mode. Keep hands visible to the
tracking cameras. The runtime must supply wrist/palm and named finger landmarks, such as
`index_proximal`, `index_intermediate`, `index_distal`, and `index_tip`.

**Tracking starts from valid data, regardless of whether the hand is open or closed.**
Present an open hand, move it to guide the arm, and bend each finger separately. No activation
button or fist is required. Close the fingers around a nearby object to request pickup:
mean curl across five fingers must reach **0.55**; opening below **0.35** releases it.

Each hand selects its input independently. Individual optical fingers use their own joint
chains. Missing data relaxes affected fingers; loss of the arm pose ends tracking and drops
its object. Reopen before grabbing again after a tracking-loss drop.

The Inspire hand has six driven actuators per side: four finger proximals, thumb pitch,
and thumb yaw. Distal joints follow mimic couplings. It reproduces independent digit curl
and thumb opposition, not every human knuckle angle or finger-splay movement separately.
Controllers retain trigger-based whole-hand closure; that is not individual optical tracking.

### What constitutes skeletal tracking

OpenXR interaction poses named `grip`, `aim`, `palm`, `pinch`, or `poke` are not a finger
skeleton. A hand-shaped avatar, controller emulation, or `source="hand"` label alone is
also insufficient. Kit's native hand component can set that label before publishing joints.
The controller fallback therefore checks valid Touch grip poses/actions, while partially
available skeletal hands remain optical.

On the tested SteamVR connection, the runtime advertised `XR_EXT_hand_tracking`, but an
earlier Kit instance had not requested it. The new hand-component setting fixes that
application setup omission without changing the working gaze runtime. Real skeletal poses
subsequently appeared intermittently. Continuous open-hand transport and the reported
fist-only detection remain to be confirmed on the headset; do not treat the setting or
successful synthetic replay as proof that every physical finger is arriving.

## Validation

Run one live test at a time against the already-loaded application. Do not run a synthetic
replay at the same time as real-hand observation.

| Check | What it verifies | What it does not establish |
|---|---|---|
| Offline regression suite | Production control logic with real USD/matrix libraries; 82 tests passed | Headset transport, rendered behavior, or live physics |
| Humanoid live replay | Actual IK, rigid-body physics, grip, assisted pickup/lift/release, and root stability | Physical controller or optical tracking accuracy |
| Finger live replay | Measured movement of all ten digits and both thumb-opposition joints, source transitions, tracking loss | Real headset camera recognition |
| Camera live replay | Constant camera-to-body mount during Play/Pause and simulated head motion | Detailed compositor behavior for every runtime |
| Real-hand observer | Actual XR landmarks, optical commands, and measured robot-joint variation | Guaranteed tracking accuracy or detection from an unseen hand |

From the repository root, use Isaac Sim's bundled Python for the offline suite:

```powershell
& "C:/path/to/isaac-sim/python.bat" tools/run_humanoid_tests.py
```

```bash
/path/to/isaac-sim/python.sh tools/run_humanoid_tests.py --isaac-sim-dir /path/to/isaac-sim
```

With the updated example loaded and the Python server enabled, use ordinary Python for
live replays:

```powershell
python tools/validate_humanoid_live.py
python tools/validate_camera_live.py
python tools/validate_fingers_live.py --timeout 300
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
fingers and both thumb-opposition joints. These are measured simulation results, not an
assertion that the unresolved headset hand-detection issue is fixed. See the
[validation details](docs/humanoid-control.md#automated-validation) for scope and measurements.

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
`(0.50, -0.28)` is the initial stationary workspace. Side benches and floor props remain
visible, but the anchored robot cannot walk to them or squat to the floor.

Pickable YCB objects include a soup can, potted-meat can, foam brick, sugar box, and mustard
bottle. Their scene names remain `/World/G1_SampleBoxes/Box_<nn>` for selection and recording.

Pickup uses **distance-gated fixed-joint assistance**. The object remains dynamic, and a
successful grasp creates a fixed joint between its live rigid body and the real hand.
Coincident attachment frames preserve its relative pose instead of snapping its origin
into the wrist. Release, drop, tracking loss, and reset remove the constraint.

| Setting | Default | Purpose |
|---|---|---|
| `_grab_assist_radius` | 0.30 m | Green candidate highlight |
| `_grab_radius` | 0.18 m | Search around the measured palm/finger center |
| `_grasp_contact_distance` | 0.09 m | Object center to nearest palm/finger link center |
| Controller close/release | 0.60 / 0.35 | Trigger hysteresis |
| Optical close/release | 0.55 / 0.35 | Mean five-finger curl hysteresis |

A green candidate can still be too far away to attach. Gaze alone does not pick up an
object. The distance gates are not a test of friction closure or force balance: an assisted
fixed joint can support an object an unassisted hand could not. Session metadata records
the assistance mode.

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

`metadata.json` records robot/mode settings, time steps, log rates, fixed-base and assisted
grasp configuration, and finger roles. Sensor CSV files share `unix_time`, `sim_time`, and
`step_index` for alignment. `frame_timestamps.csv` instead contains `frame_id`, `unix_time`,
`sim_time`, `camera_name`, and `image_path`; align images by timestamps. The default
requests one sensor row per physics step, nominally
100 Hz in **simulation time**, with 256×256 eye-camera PNGs approximately every tenth step.
Frame timestamps identify captured images; wall-clock capture rates depend on runtime performance.

CSV buffers flush at a nominal 2.5-second simulation interval and on orderly cleanup.
An abrupt shutdown can lose buffered rows. Save/review complete sessions before changing
recording schemas or training on them.

| File | Contents |
|---|---|
| `behavior.csv` | HMD motion, robot pose, base commands, gait state, and joint states |
| `hand_tracking.csv` | Hand/controller poses, buttons, source labels, finger targets, and hand closure |
| `gaze.csv` | Gaze origin/direction, hit point/object, and explicit source |
| `object_states.csv` | Sample-object poses, velocities, and held/released state |
| `frame_timestamps.csv` | Alignment between sensor rows and eye-camera frames |

### Finger columns in `hand_tracking.csv`

| Column | Meaning |
|---|---|
| `<side>_finger_thumb/index/middle/ring/little` | Commanded curl, 0=open and 1=closed |
| `<side>_finger_thumb_yaw` | Independent thumb-opposition target |
| `<side>_hand_closure` | Mean curl across the five fingers, excluding thumb opposition |
| `<side>_finger_source` | `hand_tracking`, `controller`, or `none` |

Optical curl sums adjacent bone bends from the metacarpal to the tip. Full-flexion references
are 150° for fingers and 95° for the thumb; angles beyond 180° do not reverse the resulting
curl. Thumb opposition uses a palm-local frame. Missing fingers relax independently.
The source and target columns describe commands sent to the robot; `behavior.csv` joint
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
The default `robot_head` view follows the real head/torso link. B restores this view rather
than calibrating the physical headset, and the left stick cannot cycle out of it.

### A highlighted object will not attach

Bring the actual palm/finger links closer, using near-edge objects on the front-right table.
The green radius is larger than the attachment radius. Hold grip and trigger for controller
pickup; use a valid tracked hand with sufficient curl for optical pickup. After a forced drop
or tracking loss, open/release before trying again. A marker outside the robot's reach cannot
pull an object to the hand.

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
`~/.nvidia-omniverse/logs/Kit/Isaac-Sim XR VR/6.0/`. Read the newest log and preserve the first
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
which always uses the fixed robot-head mount. Do not apply old recenter/camera-cycle advice
to the current default.

## Source reference and tuning

| File | Responsibility |
|---|---|
| [`humanoid_example.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/humanoid_example.py) | Scene, hand/controller input, IK, pickup, camera mount, recording |
| [`g1.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/g1.py) | Standing anchor, robot joint drives, optional locomotion |
| [`xr_pose.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/xr_pose.py) | Shared world-pose validation and time-based smoothing |
| [`material_highlights.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/material_highlights.py) | Stable overlapping gaze/candidate material overrides |
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
