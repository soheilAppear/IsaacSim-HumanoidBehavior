# Humanoid VR Control — Unitree G1

This fork of [NVIDIA Isaac Sim](https://github.com/isaac-sim/IsaacSim) provides stationary
VR manipulation with a Unitree G1, Inspire five-finger hands, Quest Pro eye tracking,
and timestamped behavioral recordings.

This is the operator guide: installation, launch profiles, controls, recording files,
and troubleshooting. The [developer guide](docs/humanoid-control.md) explains the
implementation and test entry points; the [validation record](docs/validation/isaac-sim-6.1.md)
preserves dated results and their limitations. The custom application runs inside the
modified policy-examples extension, using the installed Isaac Sim runtime.

**Current default:** the pelvis is fixed to the world. Arms and fingers remain articulated.
The stabilized robot view uses the G1's optical-sensor height, smooths robot travel
and heading, and removes torso roll/pitch and gait bob. Turning or tilting the headset
rotates the view and its eye offset around the measured neck in the stabilized robot
frame. Walking, leaning, and crouching in the room do not translate it. The cosmetic
head transform still turns; physical neck joints and collision shapes remain unchanged.
Optical hands control individual fingers and thumb opposition; Touch controllers
use grip for arm movement and the index trigger for whole-hand closure. The default
grasp mode is now experimental physical contact, without a hand-to-object fixed joint.
Gaze retains the working implementation.

The current [walking profile](#controller-walking-and-optical-manipulation) uses
the left controller stick for forward/backward and sideways movement, and the
right stick for turning. Controller grips move the arms and index triggers close
the fingers; optical tracking takes over when valid bare-hand input returns.
Controller face buttons and stick clicks remain unused in this profile. NVIDIA decoupled policies
control the legs and waist. Twelve handle-free shelf cartons join the six table
objects and four floor cartons. Blue pointing navigation remains implemented but
disabled. Earlier pointing and physical-walking results are historical evidence,
not acceptance of the new controller layout, shelf pickups or bimanual transport.
Eye tracking and its red ray remain unchanged.

The separate [gaze-button profile](#gaze-buttons-with-both-hands-free) keeps both
optical hands free. Its default preview anchors the base; a fresh `--move` launch
enables physical walking/turning after a deliberate eye dwell, continuing while
the same button remains fixated and stopping when the operator looks away.

The following September 2026 observations describe specific historical tests and
operator reports. They do not establish acceptance of every subsequent revision.

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

### 20 September — webcam and robot demonstration

The [current README gallery](README.md#20-september--robot-and-operator-views) uses
selected excerpts from the operator's **20 September 2026** OBS recording,
`2026-09-20 19-40-28.mp4`. The source is **2048×864 at 60 fps**, with a total duration
of **743.667 seconds**. The original recording is not included in the repository.
The [43.5-second highlights reel](docs/readme/g1-quest-webcam-highlights.mp4?raw=true)
joins the four excerpts in the table's order. Each excerpt preserves **1× wall-clock
playback**; the cuts do not depict one continuous carry-and-place task.

All listed files are under `docs/readme/`. Each clip has a matching PNG poster basename.
Times are offsets into the original recording, not simulation timestamps; clip end
times are exclusive.

| Clip | Source start → end | Duration | Poster source time |
|---|---|---:|---:|
| [`g1-carton-lift-webcam.mp4`](docs/readme/g1-carton-lift-webcam.mp4?raw=true) | 141.0 → 150.0 s (02:21 → 02:30) | 9 s | 147 s |
| [`g1-hand-tracking-webcam.mp4`](docs/readme/g1-hand-tracking-webcam.mp4?raw=true) | 60.0 → 72.0 s (01:00 → 01:12) | 12 s | 68 s |
| [`g1-factory-walking-webcam.mp4`](docs/readme/g1-factory-walking-webcam.mp4?raw=true) | 440.0 → 449.5 s (07:20 → 07:29.5) | 9.5 s | 445 s |
| [`g1-small-object-webcam.mp4`](docs/readme/g1-small-object-webcam.mp4?raw=true) | 679.0 → 692.0 s (11:19 → 11:32) | 13 s | 686 s |

Videos use **1600×900, silent H.264, 30 fps**. The composed layout takes the recorded
right VR-eye crop at **x=734, y=68, width=464, height=454** and webcam crop at
**x=1468, y=118, width=512, height=384**. Both views are scaled with their aspect ratios
preserved and placed beside captions; the PNG posters use the same layout. These are
cropped and composed views of recorded frames, not unmodified full-screen screenshots.
No source audio, generated imagery or interpolated motion frames are included.
The export changes neither the robot, camera controls nor gaze implementation.

The timing recipe for each clip is `duration = end - start` and
`output_time = source_time - start`. In FFmpeg terms, trim to the source interval,
reset timestamps with `setpts=PTS-STARTPTS`, select recorded frames at `fps=30`, crop
the two views, scale with aspect ratio preserved and composite the captions. There is
no speed multiplier. The reel concatenates clips at offsets **0, 9, 21 and 30.5 s**,
ending at **43.5 s**. `g1-quest-webcam-preview.gif` selects source **144–149 s**
(carton-clip offsets **3–8 s**), resized to **640×360 at 8 fps** for a five-second,
40-frame loop.

The clips show selected visible actions, not a measured success rate or a complete
successful placement. Webcam hand landmarks are visible, but footage alone does not
identify which input source drove the robot at every moment or establish tracking
accuracy, grip pressure or autonomous operation. The longer recording contains other
attempts; these excerpts should not be presented as an uninterrupted pickup/carry/place
run. Implementation and validation limits remain in the
[validation record](docs/validation/isaac-sim-6.1.md).

### 7 September — historical assisted grasping

![Both G1 Inspire hands with cyan and orange hand-target markers](docs/readme/g1-inspire-hands.png)

Both Inspire hands in the historical robot-camera recording. The colored spheres
are debug target markers; the current version hides them by default so the fingers
remain visible. This still does not demonstrate independent finger motion.

| Recorded action | Video | Preview |
|---|---|---|
| Bottle pickup, lift, hand transfer, and release | [Silent MP4, 4.1 s](docs/readme/g1-bottle-pickup.mp4?raw=true) | [Animated GIF](docs/readme/g1-bottle-preview.gif) |
| Package pickup, lift, and release | [Silent MP4, 3.1 s](docs/readme/g1-package-pickup.mp4?raw=true) | [Animated GIF](docs/readme/g1-package-preview.gif) |

The [historical README gallery](README.md#7-september--historical-assisted-grasping)
preserves both animated previews and the bottle/package screenshots. These older assets come from local recording
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

The launcher also sets `--/metricsAssembler/changeListenerEnabled=false` after a
native Clear/Load crash entered NVIDIA's stage-edit listener. Initial asset
conversion stays enabled; in-place asset variant/reference changes require a
fresh load. A fresh load and subsequent Clear/Load passed with the workaround,
followed by successful XR startup. This validates that reproduced reload path.

The Windows VR launchers select **Direct3D12** with `--/app/vulkan=false` for this
process, overriding the installed app's Vulkan default without changing installed
files or global settings. This local workaround passed two native XR off/on cycles
that crashed under Vulkan, plus camera-pose and backward/settle checks. Restart Isaac
through the launcher to apply it. For a deliberate Vulkan diagnostic only, append
`--/app/vulkan=true`; forwarded arguments take precedence. Renderer performance and
live hand/eye tracking quality were not established by those checks.

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
3. Start with objects on the centered practice table in front of the robot.

The full example provides the camera, manipulation, gaze visualization, and session recorder.
For desktop inspection, launch `isaac-sim.bat` instead; keyboard locomotion is still disabled
in the stationary example.

### Controller walking and optical manipulation

Save the stage, close Isaac, keep SteamVR running and launch from this checkout:

```powershell
Set-Location 'C:\Users\Soheil\Documents\GitHub\IsaacSim-HumanoidBehavior'
.\tools\launch_controller_walking.bat
```

Open **Policy → Humanoid**, click **LOAD**, then **Play**. The former
`launch_gesture_walking.bat` is now a compatibility alias for this same profile.
The normal `launch_isaac_vr.bat` still selects stationary manipulation unless
walking is explicitly enabled in its environment. Restart and reload when
changing profiles; an already loaded articulation cannot switch anchors safely.

| Input | Walking profile behavior |
|---|---|
| Center both sticks | Enable walking after startup, Stop/Pause/Reset, controller loss, or a long input interruption |
| Left stick forward/back | Request forward/reverse walking |
| Left stick left/right | Request sidestepping |
| Right stick left/right | Request turning |
| Release sticks | Clear movement requests; allow the physical robot to brake |
| Controller loss | Stop and require centered sticks on reconnect |
| Tracked optical palm/fingers | Move the corresponding arm and individual fingers without a clutch |
| Each controller grip | Hold to move/rotate that arm; release holds its measured pose |
| Each controller index trigger | Proportionally close that hand's fingers; release opens, independently of grip |
| A/B/X/Y and stick clicks | Unused in this walking profile |
| Keyboard B | Recenter the horizontal view heading |

Small stick movements request slower travel; larger movements request faster travel.
The first **15%** is a noise deadzone, and the remaining travel scales linearly to
**0.65 m/s forward, 0.35 m/s reverse, 0.30 m/s sideways and 0.40 rad/s turning**.
At 57.5% total deflection the request is half the corresponding cap. Diagonal
translation is bounded. Neutral clears the request immediately; physical braking
still takes time. The learned actor's configured limits have not been increased.

The blue ray and selector are disabled, not deleted. Gaze/red-ray behavior is
unchanged. While this profile owns XR, Kit's built-in teleport/fly/rotate and
object grab/move/select/menu tools are unbound, preventing duplicate actions from
the same controls. Raw tracking and GUI layers remain enabled. Leaving XR, changing
its profile, or cleaning up the example restores the prior tools layout.
The controller layout uses the existing learned gait and health/contact
guards. All six directions, neutral rearming, release and tracking loss passed
physical input replays; these do not establish live hardware quality or loaded walking.
The residual-motion guard checks foot-centroid travel and mean sole heading, allowing
small stance corrections. After braking, matching root/support motion must also
exceed 15 cm or 0.35 rad from a fixed stationary reference before this guard trips.
Those are minimum excursions with speed/dwell prerequisites, not hard travel limits;
slow or intermittent drift can remain below the prerequisites. Fast-runaway and fall
guards remain active. The current controller/camera revision passed **888 offline
tests**. The preceding stance/contact revision also passed a fresh physical approach
plus carton press/withdrawal check without a fault.
The contact replay used synthetic optical input; complete real-hand package placement
remains an operator acceptance test. See
the [validation record](docs/validation/isaac-sim-6.1.md#balance-sway-and-remaining-pause-limitation).

**VR restart validation:** native XR off/on reproduced a Vulkan/OpenXR rendering crash
despite camera/profile and viewport-readiness guards. The default Direct3D12 launch
then passed that cycle twice. Brief native transition errors still occurred, so this
is a bounded local workaround rather than an extended-session reliability guarantee.
The user reported **Stop → Humanoid Reset**, without requesting an XR restart.
A controlled call to the registered Reset handler and a subsequent test of the existing
built Reset button both passed under Direct3D12 while XR stayed enabled. The latter
invoked `Button.call_clicked_fn()` with the normal UI subscriptions active; it was
not a physical mouse click. The precise cause of the earlier reported crash remains
unresolved. See the
[renderer validation](docs/validation/isaac-sim-6.1.md#direct3d12-process-workaround).

**The user's current Quest/SteamVR runtime switches between controllers and
optical tracking instead of providing both together.** Use controller grip for arm
movement and trigger for closure while the controllers are active. Releasing grip
holds the arm pose; a held trigger retains closure while the sticks remain available.
Stop before switching to bare hands: valid optical palms and fingers then take over,
and retained stick values are invalidated. Center both sticks again when returning
to controllers. Trigger closure is a control command, not optical finger measurement.
The webcam is display-only and never replaces missing Quest/controller input in the
current setup. Only Quest optical hands or controllers control robot arms and fingers.

The walking launcher defaults `G1_OPTICAL_HANDS_ONLY=0`. An explicit environment
override is preserved: set it to `1` before launch to disable controller manipulation
and retain the former optical-only profile. This does not change stick locomotion.

The fourteen arm joints receive a model-based gravity compensation term in this
controller walking profile. It helps support the robot's arm weight while following
or holding a pose, using the existing motor effort limits. It does not compensate
for a carried object's weight, infer finger pressure, or alter the learned leg/waist
controller. Stationary manipulation and the retained pointing profile leave this
option off. Physical contacts, command limits and health guards remain active;
successful shelf pickup and carrying still require separate validation.

Three **1.60 m wide** open racks hold **12 handle-free boxes**: two decks with two boxes per rack.
The deck tops are **0.68/0.94 m**, **0.74/1.02 m**, and **0.78/1.06 m** high.
Cartons range from **16–24 cm deep, 20–28 cm wide and 12–22 cm tall**, at
**100–250 g**. The existing six table objects and four floor cartons remain.
These represent light empty practice cartons with real box colliders and gravity,
not loaded warehouse packages.

For a two-hand attempt, stop close enough to reach without fully locking the elbows.
Raise the open hands to the box's height **in front of the shelf**, then move around
its opposite side faces. Reaching diagonally upward beneath a shelf makes the
contact guard stop against its underside. Bring the open palms together gently
against opposite side faces, then curl the fingers while holding that position.
Lift only after both sides maintain contact. Open and separate both hands to release.
The diagnostic recognizes sustained
opposing contacts **across hands**, even if neither hand independently forms a
pinch. It reports normal forces and an estimated vertical friction capacity using
the material coefficient. That estimate is not fingertip pressure, measured slip,
successful pickup or a load-carrying guarantee. No attachment joint, adhesive
force or automatic closing is added.
Shelf pickup remains experimental: the latest synthetic replay cleared the open
approach but its fingers caught the rear face and failed to hold the box. The new
palm-first validator checks opposing palm contact before curl and opposing side
contact before lift. A live grip-only trial of that sequence failed the palm-contact
gate and stopped before finger closure or lift; successful pickup remains unverified.

### Gaze buttons with both hands free

This experimental profile uses genuine tracked eye gaze for four controls:
**Forward**, **Back**, **Turn left**, and **Turn right**. The panels follow the
camera on either side, leaving the central box/manipulation view free. Optical
hands continue to control arms and fingers independently; no hand gesture or
controller button activates walking. Head direction can still serve the existing
gaze visualization/recording fallback, but it cannot select a locomotion button.

1. Enable and calibrate eye tracking in the headset, allow the streaming/runtime
   eye-tracking permissions, and connect the supported XR runtime. Extension
   availability or a visible red ray alone does not establish actual eye input.
2. Launch the anchored preview:

   ```powershell
   .\tools\launch_gaze_walking.bat
   ```

3. Open **Window → Examples → Robotics Examples → Policy → Humanoid** (panel title
   **Humanoid: Unitree G1**), click **LOAD**, then **Play**.
   If the headset view is not already active, open **Window → Rendering → XR**.
   Under **Experience → Select Experience Type**, use **Virtual Reality Experience**;
   under **General → Selected Output Plugin**, use **OpenXR**, if these are not
   already selected. Preserve the working OpenXR runtime settings and click
   **Start XR**.
4. Put down the controllers and show both hands to the headset. Look between the
   buttons first to arm selection. Look at one button for **0.7 seconds** while its
   ring fills, then keep looking to sustain the request. Look away to end it.
   Preview reports the interaction while the base remains anchored.
   Look away from every button before trying another selection.
   **NO EYE** means Isaac is not receiving a currently tracked eye pose. Check
   headset calibration, eye-tracking permission and runtime transport before
   continuing; a visible head-direction ray cannot substitute for eye input.
5. After verifying tracking and comfortable panel placement, exit Isaac and run:

   ```powershell
   .\tools\launch_gaze_walking.bat --move
   ```

   Repeat LOAD, Play, and Start XR if needed. The new scene uses the physical GR00T
   balance/walk actor. Try a brief **Back** hold first, then look away to stop and
   make room for turning beside the front shelf. Check the surrounding space;
   gaze controls do not automatically reject turns near obstacles.
   Preview and movement share the shelf/carton layout, but the preview's fixed
   anchor changes the robot topology: do not try to turn preview into movement
   by toggling an attribute in a running scene.

The default **hold** mode starts after a completed dwell and continues only while
you keep looking at the same button. **Look away to stop**, including when looking
back at the box. Changing buttons or lost/invalid eye tracking also requests a
stop. Requests use **0.20 m/s forward**, **0.15 m/s backward**, or **0.25 rad/s
yaw**; turning continues while the turn button is held. Physical braking and
contacts determine actual travel and stopping distance. A stop request is not an
instant physical stop. Look away from all buttons and wait for stopping to finish
before beginning another dwell. XR shutdown, Pause/Stop, Reset and recentering
cancel selection/movement and require rearming. Use the example's **Reset** after
a latched simulation fault; Play alone does not clear it.

To retain the previous bounded interaction, set `$env:G1_GAZE_MODE='step'` before
launching. Each completed dwell then requests one **0.20 m** forward/backward move
or **10°** left/right turn. That accepted step continues to its measured stop after
looking away or losing eye tracking; tracking loss still cancels unfinished dwell.
Holding a button cannot repeat a step. New selections are ignored while busy, and
look-away is required before another dwell. Set `$env:G1_GAZE_MODE='hold'` to return
to continuous movement in the next launch.

Every launcher invocation resets preview to on, including when an old terminal
contains `G1_GAZE_PREVIEW=0`. Only a **first** `--move` enables motion; a first
`--preview` explicitly selects the default. Remaining arguments pass to the base
Isaac launcher, for example `--move --xr-verbose`. The launcher fixes
`G1_LOCOMOTION_INPUT=gaze`, `G1_POINTING_NAVIGATION=0`, and
`G1_OPTICAL_HANDS_ONLY=1`; controller manipulation and webcam robot input stay off.
The launcher defaults `G1_GAZE_MODE` to `hold` only when it is unset; it preserves
an explicit `step` override. Supply settings in the environment before starting Isaac:

| Environment variable | Default | Meaning |
|---|---:|---|
| `G1_GAZE_MODE` | `hold` | Continue while looking; `step` selects one bounded movement |
| `G1_GAZE_DWELL_S` | `0.7` | Initial continuous button dwell, in wall seconds |
| `G1_GAZE_DISTANCE_M` | `0.20` | Step mode only: requested forward/backward distance, in meters |
| `G1_GAZE_TURN_DEG` | `10` | Step mode only: requested turn angle, in degrees |
| `G1_GAZE_FORWARD_SPEED` | `0.20` | Forward command speed, in m/s |
| `G1_GAZE_BACKWARD_SPEED` | `0.15` | Backward command speed, in m/s |
| `G1_GAZE_YAW_SPEED` | `0.25` | Turn command speed, in rad/s |

Set `G1_GAZE_BACKWARD_SPEED=0` to disable backward selection. The default enables
it. Dwell uses wall time; requested speeds and settling use simulation time, so
slow simulation changes actual movement per wall second.

The directional speed settings remain bounded by the existing physical profile's
0.65 m/s forward, 0.35 m/s backward and 0.40 rad/s yaw caps. The source supports
backward walking. **981 offline regressions passed**, followed by actual-scene
synthetic-eye hold checks: anchored preview stayed fixed, and all four physical
directions continued beyond the old 20 cm/10-degree limits. Back and both turns
settled after look-away; Forward settled after simulated eye-tracking loss, with
an immediate zero command. The physical run completed without a fault and retained
arm/finger updates. Preview checks selection and hand availability, not balance,
gait, braking, shelf pickup or loaded transport. A displayed selection is an
explicit interface action, not a claim that gaze reveals general mental intent.
Live eye-tracker calibration, in-headset readability, and picking up/carrying a
box while walking remain operator acceptance checks. See the
[validation details](docs/humanoid-control.md#gaze-buttons-and-optical-hands).

### Experimental gesture walking

This is the retained **pointing** implementation, disabled in both walking
launchers. The following controls and prior measurements apply only when it is
explicitly selected for development:

**This opt-in scene combines physical walking with arm and finger teleoperation.**
NVIDIA's pretrained G1 decoupled policies control the legs and waist. Both arms
and fingers continue following their ordinary inputs and contact limits.
**The blue left-index ray alone selects a destination** in this launcher.
The normal `tools\launch_isaac_vr.bat` workflow remains stationary.
Save the current stage, close Isaac, keep the working SteamVR/Quest connection,
and launch from the repository directory rather than the Isaac installation:

```powershell
Set-Location 'C:\Users\Soheil\Documents\GitHub\IsaacSim-HumanoidBehavior'
$env:G1_GESTURE_WALKING = '1'
$env:G1_LOCOMOTION_INPUT = 'pointing'
$env:G1_POINTING_NAVIGATION = '1'
.\tools\launch_isaac_vr.bat
```

These variables select pointing for the normal VR launcher. Open **Policy → Humanoid**,
click **LOAD**, then **Play**. A previously
loaded stationary scene must be rebuilt; changing an environment variable during
Play cannot remove its world anchor. To return to stationary mode, close Isaac and
launch `tools\launch_isaac_vr.bat` normally, with `G1_GESTURE_WALKING` unset or `0`.

Navigation uses **Quest optical left-hand tracking**. Webcam fallback and controller
poses cannot start or sustain walking. Eye tracking and the red ray keep their
existing behavior independently: gaze direction, loss, and `hmd_forward` fallback
do not gate selection or walking. Webcam robot input is now disabled by the standard
launchers; its former arm/finger fallback remains a separate, disabled experiment.

| Action | Navigation result |
|---|---|
| Open the left hand after startup, Stop, Pause, or Reset | Clear the release requirement before choosing a destination |
| Extend the left index; curl middle, ring, and little | Show a blue pointing ray; thumb position is ignored |
| Hold the blue ray steadily on visible floor or a registered practice object for 0.6 s | Acquire a destination; cyan shows acquisition, green shows the locked target |
| Keep pointing after selection | Plan a detour if needed, turn toward the route, then walk; changing gaze does not redirect it |
| Open the left hand or lose valid hand tracking | Clear forward and turn commands; release before choosing again |
| Reach within 30 cm of the target, or encounter a blocked route | Stop and require release before a new destination |

Destinations must be **0.5–50 m** from the robot, on static floor near ground
height. Selection works in all directions around the robot; it physically turns
before moving forward when the heading difference is large. Pointing directly at a registered
practice object selects a checked floor approach in front of it. The blue ray ends
at the object; the separate floor marker is the stopping destination. Tables and
walls cannot authorize an approach. Keep the selected floor or object steady
throughout the dwell. Both
arms and fingers remain tracked during selection
and walking; the former navigation arm hold is removed.

An adaptive angular filter steadies the blue ray while responding more quickly
to deliberate direction changes. The visible ray and collision query use the same
filtered direction. The 0.6-second dwell accepts **18–30 cm** of drift, depending
on the initial target distance; its anchor and tolerance stay fixed during that
attempt. On a miss the hit marker disappears. The ray still stops at the nearest
non-robot collider, so pointing through a wall, box, or table cannot select floor.
The collision ray extends up to **60 m**, but walking destinations remain limited to
**50 m** from the robot; a visible hit can therefore be outside the permitted range.

Cruise requests are capped at **0.65 m/s**, with slowing beginning at **1.60 m**.
The previous 0.45 m/s profile and this profile request the same straight-line
speeds within 1.20 m of the target; arrival remains 0.30 m. Curved forward steering
is capped at **0.20 rad/s**. Large heading changes use a separate **0.40 rad/s
stepping-turn request**, with zero forward/lateral command. These are command caps,
not guaranteed real-world speeds. The 19 September physical baseline passed
forward walking, both turns, and stopping, before the current stepping-turn change.
Its straight phase averaged **0.488 m/s in simulation**; world-forward progress
per wall second was **0.1889 m/s**, versus **0.0825 m/s** in the earlier 0.45 m/s
run. These bounded replays do not establish operator pointing accuracy.

The robot uses forward travel and physical stepping turns; no reverse or strafe is
requested. Feet can shift the body during a turn: its position is not anchored.
A volume query checks each route segment with **0.76 m width**,
**1.5 m height**, and endpoint clearance. Meaningful yaw requests also require a
wider turning-footprint check, refreshed from the measured position. Heading must
improve by **0.05 rad within 15 wall seconds**, or navigation stops and requires
release. The watchdog resets on a new best heading improvement, turn exit,
release, reset, and route-corner changes; oscillating jitter cannot keep it active.
The current profile passed a synthetic 180° turn-and-approach replay, with **0.493 m**
maximum drift during its initial stepping turn. This does not establish every
turn, route, or loaded-transport scenario.
If a direct route is blocked, a bounded planner searches for a detour while the
robot stands still. Keep pointing during planning, or open the hand to cancel.
Only real floor with sufficient body and turning clearance is accepted. Walls,
narrow aisles, unsupported floor, and search limits can leave a destination unreachable.
New obstacles on an active segment stop the robot and require a fresh selection.
Planned detours use approximately **0.687–0.910 m half-width**, depending on the
route angle. This projects the entire world-axis turning square into the corridor
before adding a 0.15 m tracking reserve. At the start and actual route corners,
the planner adds a **0.50 m stepping-drift reserve**, producing approximately
**1.037 m half-width**. Its search tracks incoming heading so straight passages
need not satisfy the larger corner reserve at every point. These margins can
reject narrow routes; direct close-object approaches keep their earlier
braking-aware query. Earlier narrow and unprojected diagonal routes stopped
safely at a table. The corrected planner then passed a physical detour around
that table to a destination 8.53 m away, stopping 0.299 m from its target.
This replay used synthetic pointing input; it does not establish every factory
route or walking while carrying a package.
Keep the left hand in the headset's tracking area. Clearing commands requests
physical braking; it does not teleport the body or guarantee an instantaneous stop.

The start ramp uses bounded wall time from the existing teleoperation clock;
the physical gait still advances at simulation speed. Live Quest acceptance of
point recognition, stable floor selection, arrival, release, and occlusion is pending.
Walking does not consume gaze samples or change the eye tracker or its red ray.

Stop-on-tracking-loss depends on the validity reported by the XR runtime. Some
runtimes may retain a cached wrist pose or skeleton marked valid while a hand is
occluded. That can conceal the loss from the software; synthetic invalid-input
tests do not establish an immediate stop in that situation. Real Quest tests must
check hands leaving the camera field of view and temporary occlusion before this
behavior can be accepted.

For diagnostics, call `gesture_locomotion_status()` on the running Humanoid example.
The mode is `point_only`. `target`, `candidate`, and `acquisition_progress`
describe selection; `detail` explains readiness, missing input, rejected surfaces,
out-of-range destinations, blocked routes, or a latched fault. `ray_hit_path`,
`ray_hit_distance_m`, `ray_walkable`, `ray_error`, and `corridor_error` separate a
ray miss from an ineligible hit or an obstructed route.
`requested_velocity` and `filtered_velocity` contain `[forward, lateral, yaw]`
in m/s, m/s, and rad/s; `policy_mode` reports `balance` or `walk`.
`route_state`, `route_waypoints`, `route_queries`, and `route_expanded`
describe a pending or completed detour search. `hand_reserved` is false:
navigation does not freeze an arm. Invalid input zeroes
commands and requires a measured release before a fresh selection. Physics faults
pause the timeline; inspect the fault and use Reset or LOAD before continuing.

Stop invalidates the cached physics state so Play reinitializes the controller.
Pause and Stop also discard navigation intent. XR camera scheduling waits for
readiness and valid finite poses, preserving the mounted-camera mapping and gaze.
**Three Stop/Play and two Pause/Play cycles passed with real XR active and no
fault.** The reported native restart crash did not recur in those checks. This
validates that earlier restart sequence, before the point-only change. The full
point-only physical replay passed with synthetic hand input; actual Quest pointing
accuracy, occlusion, and extended-session reliability remain unverified.

**Historical V-controller result:** the 18 September pre-adjustment Quest capture kept a valid optical hand source,
but about 97% of active samples stayed below the walk actor's command threshold.
An earlier zero-command drift at about 0.197 m/s had also triggered a physics pause;
after Reset, the subsequent idle check was stable. These were separate symptoms,
not evidence that the headset source had disconnected. This capture predates the
destination selectors and does not validate the current point-only implementation.

**Stop before precision pickup.** The physical replay reached while walking,
closed and reopened the right fingers, and stopped with the arm extended. Its
extended palm remained about 9–10 cm from the requested target, so the current
controller is not accurate enough to promise precision grasping while walking.
Finger movement does not validate pickup or payload carrying. A simulation fault
pauses the timeline; use Reset or reload instead of repeatedly pressing Play.

The walking scene moves the six-object practice table to world `(2, -2)` m, retaining
its 0.92 m height, and adds four dynamic shipping cartons:

| Asset | Horizontal bounding-box size | Mass | Floor position X, Y |
|---|---:|---:|---|
| CardBox A_01 | 22 cm | 250 g | `(1.5, 1.0)` m |
| CardBox B_01 | 28 cm | 350 g | `(1.5, -1.0)` m |
| CardBox C_01 | 34 cm | 500 g | `(2.6, 1.15)` m |
| CardBox D_04 | 40 cm | 700 g | `(2.6, -1.15)` m |

These are referenced warehouse cardboard assets with gravity and collision geometry,
placed beside the central aisle. The sizes describe their largest horizontal bounds,
not cube side lengths. The scene also adds three tall handled cartons:

| Object | Body dimensions W × D × H | Total mass | Floor position X, Y | Authored handle-center height |
|---|---|---:|---|---:|
| `Box_10` | 32 × 32 × 76 cm | 250 g | `(3.5, 2.8)` m | 68.5 cm |
| `Box_11` | 40 × 32 × 88 cm | 350 g | `(3.5, -2.8)` m | 80.5 cm |
| `Box_12` | 30 × 38 × 96 cm | 450 g | `(1.35, 0)` m | 94.0 cm |

`Box_12` is the initial pickup station directly ahead of the starting robot.
Stop near its handle before reaching; longer walking routes must go around it.
All three props settle through a 5 mm initial floor gap, leaving handle centers
at **68.0, 80.0, and 93.5 cm**. The higher `Box_12` grip reduces downward arm reach
while keeping the existing body clearance and braking reserve.

Each handle has a **3.5 cm diameter**, **16 cm grip length**, and **7 cm opening**
behind it. Handle and carton are one dynamic compound body; the opening remains
open in the collision geometry. These are deliberately lightweight practice props,
not realistic loaded-carton masses. Stop walking, place an open hand around the
handle, oppose the thumb and close the fingers, then lift. Open to release.
Pickup still requires contact and friction; no fixed joint attaches the carton.
Tall-carton grasping and loaded walking need their own live validation.
Floor crouching, bimanual carrying, and reliable floor pickup are not implemented.

This mode requests **60 Hz rendering**, uses **200 Hz CPU physics**, and runs pretrained NVlabs GR00T-WholeBodyControl G1 **balance/walk
actors at 50 Hz**, running through Isaac's bundled ONNX Runtime. The actors
observe all 29 body joints and command 15 leg/waist joints; arm IK supplies the
remaining body targets. Torque-limited body PD and foot contacts matched to the
released deployment are applied only in walking mode. The referenced robot asset
and its mass properties are unchanged. Like the H1 sample, movement comes from
learned joint actuation. The current G1 weights are retained; H1's network is not
transplanted onto a robot with different joints and morphology. The balance/walk
models were trained upstream; no training pipeline is implemented in this repository.

The decoupled walking mode also applies the source controller's shoulder-roll
limits: left **0.19 to 2.2515 rad**, right **−2.2515 to −0.19 rad**, intersected with
the asset's joint limits. They apply while balancing as well as walking. Both arms
remain tracked, but the requested pose keeps the upper arms outward of these inner
limits. A target already outside the range returns gradually under the existing
speed/lead limits; contact protection may hold it rather than force it through a
surface. This affects walking-mode IK and subsequent arm-command processing only;
stationary reach, fingers, and gaze are unchanged. Physical checks of this
configuration passed destination approach and stopping phases plus a separate
final balance hold. Reaching throughout this range while walking remains unverified.

Gravity stays enabled, with no world anchor, root teleporting, station-hold or
turn-hold assistance, or kinematic fallback. Missing policy assets or incompatible
joint data fail explicitly. The weights use the **NVIDIA Open Model License**;
source/configuration uses Apache-2.0. Pinned checkpoints and settings are documented
in [policy provenance](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/data/README.md).

**19 September pointing/throughput update:** **568 offline checks passed**.
Bulk hand reads, bounded 100 Hz raw arm IK, one final body PD update per physics
step, faster lossless PNG compression, and a 60 Hz walking render configuration
reduced repeated work. Input validity, contact feedback, and body torques still
update at 200 Hz. Short live observations changed from **0.255 to 0.364 real-time
factor**, with application updates changing from **23.00 to 21.77/second**.
This is about 42% more simulation progress, not real-time simulation or headset
FPS. The **0.65 m/s physical baseline passed** balance, forward motion, both turns,
and all stopping phases. Straight-release travel was **0.188 m**, ending at
**0.0138 m/s** planar speed. A separate **0.65 m/s** arm/finger replay passed with
maximum sampled palm error **9.23 cm**. The revised 3 m destination replay passed,
traveling **2.702 m** before requesting arrival stop at **0.298 m** remaining.
Three Stop/Play and two Pause/Play cycles passed with XR active. Real operator
pointing accuracy, perceived jitter, and occlusion acceptance remain unverified.
The final 15-second unmodified Quest observation found optical input and the
`hand_tracking` control source in all **134 samples per hand**, with no read or
callback errors. No fingers crossed the observer's motion thresholds during that
window, so it confirms input availability rather than moving-hand accuracy.
Earlier results below retain their original scope.

**Earlier 19 September speed/hand update:** **523 offline checks passed**. The 0.45 m/s
command replay passed straight walking, both turns, and stopping; both hands
passed a physical curl response replay. Three Stop/Play and two Pause/Play cycles
passed with VR active. A short final live sample measured 23.37 app updates/second
and 0.259 real-time factor, up from 19.08 and 0.212. Gaze behavior is unchanged;
end-to-end headset latency and loaded grasp improvement are not established.
See [the measured update](docs/humanoid-control.md#measured-update--19-september-2026).

**18 September point-only validation:** **504 offline checks passed in 9.017 seconds**.
The complete physical replay passed destination approach, arrival stopping,
hand release, explicit tracking loss, and final balance. It moved **0.6047 m**
forward and reached **0.29987 m** from its target, without a health fault or callback
error. Only navigation hand measurements were substituted; production floor and
clearance queries, blue visualization, and learned physical control remained active.
The hand reader was restored, the gaze tracker was preserved without substituting
eye input, and the timeline was paused after cleanup. The updated mode also passed
**three Stop/Play and two Pause/Play cycles in 6.291 seconds**, with three physics
initializations, fresh-release requirements, and no faults or callback errors.
XR was disabled with no connected devices in that run; this is desktop physics
lifecycle acceptance. Point-only Quest hardware, pointing accuracy, and tracking
occlusion remain operator checks.

**Historical point-and-gaze validation:** **505 offline checks passed.** A synthetic
navigation replay using the real scene queries, blue visuals, and physical policy
advanced **0.603 m** and arrived **0.29985 m** from the destination. Arrival stopping,
release, and explicit tracking loss passed. The 300-second wall-time limit expired
during the final balance hold; a separate **3.01 simulation-second** continuation
passed in **84.41 wall seconds**, ending near **0.0001 m/s** with no health fault.
The original timed-out report is retained. This does not establish a complete
uninterrupted replay pass or real Quest pointing/gaze/occlusion acceptance.
Navigation measurement overrides were restored and the existing gaze tracker was
preserved. These results predate the point-only change and do not validate it.
Actual Quest pointing acceptance, payload carrying, and arbitrary simultaneous reaching
remain unverified.

To replay the current selector against an initialized walking scene:

```powershell
python .\tools\validate_point_gaze_walking_live.py --output .\.webcam\point-only-walking-live.json
```

The validator retains its historical filename; its default output is
`.webcam/point-only-walking-live.json`. Its default wall-time budget is **600 seconds** for slow VR sessions; simulation
duration, progress, stopping, and loss checks are unchanged. The tool prints a job
UUID, saves partial progress, restores temporary navigation input overrides, and
pauses afterward. `--resume UUID` retrieves an existing job. See the
[validation record](docs/validation/isaac-sim-6.1.md) for the tested scope and artifacts.

**Historical V-controller-era validation:** the PhysX baseline passed zero-command balance, forward motion, both forward arcs,
and stopping. A separate right-arm replay passed reaching, finger closure/release,
stopping with the arm extended, and synthetic tracking loss without health faults
or new callback errors. These are simulated inputs, not physical Quest tracking,
point-only destination selection, or loaded-object tests. To reproduce the physical
controller replays against an already-loaded walking scene:

```powershell
python .\tools\validate_gesture_walking_live.py --suite all --reset --output .\.webcam\gesture-walking-validation.json
```

Run from the repository root. `--suite baseline` or `--suite reach` selects one
replay; `--port 8227` targets a nondefault Python-server port. `--reset` resets
before the first replay; `all` also resets between replays. The tool restores input
overrides and leaves the simulation paused. It prints a job UUID and polls short
requests, so the replay survives a host connection closing. To retrieve that same
job later, use `--resume UUID --output .\.webcam\gesture-walking-validation.json`
with the same `--port`; do not add `--reset`. Only one validation job can run at a
time. See the
[validation record](docs/validation/isaac-sim-6.1.md) for measurements and remaining
hardware checks.

### Apply an update to an existing session

Save your stage, restart Isaac Sim to reload Python, and LOAD a fresh example in the
selected profile. Switching between stationary and walking modes requires rebuilding
the scene because the fixed pelvis anchor changes the articulation topology. The
extension and launcher enable Kit's native hand component; if it is enabled after an XR instance already exists,
restart the **XR session** as well so that instance requests the skeletal extensions.

The standalone `g1_standalone.py` is a separate walking/hand demo, not this stationary VR
workflow. Its commands are listed under [optional experiments](#optional-moving-mode-experiments).

## Webcam alongside Quest

The webcam now **only displays the operator and body/hand landmarks** for observation
or recording. It does not control the robot, including when Quest tracking is lost.
Robot arms/fingers use Quest optical tracking or VR controllers. Gaze and camera
control are unchanged. This mode applies to current launches; it does not establish
the input configuration used in earlier published recordings.

1. Open PowerShell in the repository folder and prepare the separate environment
   and pretrained models once:

   ```powershell
   Set-Location "C:\Users\Soheil\Documents\GitHub\IsaacSim-HumanoidBehavior"
   python tools/webcam_setup.py
   ```

2. Start the display-only preview in its own terminal:

   ```powershell
   Set-Location "C:\Users\Soheil\Documents\GitHub\IsaacSim-HumanoidBehavior"
   .\tools\launch_webcam_tracking.bat --hands
   ```

   If PowerShell reports that the launcher is not recognized, check the current
   folder with `Get-Location` and run the `Set-Location` command above. Both
   launchers resolve their own files after they start.

   Omit `--hands` for body landmarks only, which costs less CPU. Use `--camera 1`
   to select a second camera. Close other applications using that camera if it
   cannot be opened. No webcam image files are recorded by this tool.

3. Position the camera to show your hands, with good lighting. In OBS, capture the
   **Humanoid webcam tracking** window instead of opening the same webcam again.
   The overlay reads **DISPLAY ONLY - robot uses Quest/controllers**. Press **Q** or
   **Esc** to close it. **C is ignored; calibration is disabled.**

`launch_webcam_tracking.bat` always adds `--preview-only`. In that mode the process
creates no UDP socket and never encodes or sends robot-input packets. It keeps local
MediaPipe inference and visible landmarks; `--headless` is incompatible with this
display-only launcher. Standard Isaac launchers force `G1_WEBCAM_ENABLED=0`, including
the controller walking profile, even if an older terminal has it set to `1`.

The old UDP sender/receiver and calibration code remain for a future manually enabled
experiment, not the current workflow. Running `tools/webcam_tracking.py` directly
without `--preview-only` retains that sender behavior; use the `.bat` preview launcher
for recording. A camera image cannot control the robot through this preview mode.

The environment and model cache live in ignored `.webcam/`. Model setup downloads
Google's pretrained task files. Inference is local; this preview sends no landmark
JSON to Isaac. MediaPipe's own package privacy terms apply to its
usage/performance metrics. See the official
[Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
and [MediaPipe package notice](https://pypi.org/project/mediapipe/1.0.1/).

## Controls

### Stationary controls at a glance

Grip/trigger manipulation and optical hand input also apply in the controller walking
profile. Its sticks control locomotion, while A/B/X/Y and stick clicks remain unused;
use keyboard B to recenter there. The face-button actions below are stationary controls.

| Input | Current behavior |
|---|---|
| Valid optical wrist and middle-knuckle positions | Move the matching arm automatically; no grip, pinch, or fist required |
| Rotate a tracked open palm | Align the robot palm anatomically, independent of its starting wrist angle |
| Bend one tracked finger | Bend that robot finger independently |
| Move thumb across the palm | Control thumb opposition separately from thumb flexion |
| Controller side grip | Hold to move/rotate that arm; release holds its pose in physical mode while valid finger input continues |
| Controller index trigger | Curl all fingers and oppose the thumb; grip alone still leaves fingers open |
| B | Recenter forward heading without saving head tilt as a camera offset |
| Y | Force both hands open; reopen/release your input before closing is rearmed |
| Headset rotation | Turn the view and visible head; rotate the eye offset around the measured neck in the stabilized robot frame |
| Left stick click | Keep the robot-mounted camera mode |
| Sticks, X/A, keyboard movement keys, gamepad locomotion, headset gait | No base movement in stationary mode |

The view follows a level, stabilized robot frame. Its neutral height comes from the authored
`torso_link/d435_link` optical marker, about **4.6 cm above** the former render-head
bounding-box midpoint. A one-time calibration after the walking actor settles adjusts
for standing height; subsequent gait bob and torso roll/pitch do not shake the view.
Robot XY travel and yaw are filtered, while the current measured HMD rotation is
applied without that filter. Head turning also rotates the neck-to-eye offset, so
looking down follows the head's eye arc instead of looking inside its mesh.
The camera checks physical-head descriptor orientation and position validity bits.
This installed Kit's raw wrapper lacks its native method, so the validated processed
descriptor is used; supported raw descriptors are preferred on other runtimes.
An invalid descriptor never falls back to an unvalidated
matrix-only read; the existing bounded dropout hold still applies.
"Unfiltered" describes this input/control path; XR runtime and rendering latency remain.

Face forward when tracking starts; **B** sets forward again. Calibration uses only
horizontal heading, so putting on a headset that started tilted cannot leave the
world permanently slanted. Actual head pitch and roll still rotate the view.
If the view faces sideways, face your intended forward direction and press **B**
on the keyboard with the Isaac window focused. Headset translation is
ignored, including during Pause. Tracking loss holds the last relative orientation.
The visible head now rotates with HMD orientation around a fixed neck pivot on the
torso. It follows the torso's position instead of the filtered camera position,
so camera smoothing cannot separate the head from the body. The comfort view keeps
its existing robot-motion stabilization and applies the HMD-driven eye arc in that
stable frame. The eye arc does not copy physical torso bob, pitch or roll.
G1 has no movable neck joint; head collision shapes stay fixed.
This mode assumes a level factory floor and does not follow ramps or stairs.
Eye tracking is unchanged. Native XR floor leveling is overridden only for the
owned robot-view profile; cleanup restores its previous setting. The earlier
rigid-mount validation does not establish comfort of the current implementation.

The current helper keeps the render-only head visible and anchored to the torso
in both camera modes. Earlier camera-following and reversible hiding helpers remain
for legacy diagnostics; production teleoperation uses neither. Physical colliders
are unchanged. A **5 cm baseline near plane** applies to the robot first-person USD
camera and native XR profile. Measured head intrusion during sway/travel increases
only that clip to clear the visible head bounds plus **15 mm**, quantized in **5 mm**
steps; it returns toward baseline when clear. The **30 cm maximum** has an explicit
diagnostic for larger intrusion and is not protection against arbitrary falls.
Very near hands or packages can disappear temporarily. Camera/HMD/gaze matrices,
head attachment/visibility, external-camera clipping and physics are unchanged.
Mode exit and cleanup restore the prior XR near-plane setting. The adaptive change
awaits live visual acceptance. Earlier fixed-5-cm validation passed 25 native XR poses and ten rendered
desktop first-person views, including 85° down and 180° yaw. The head stayed visible
and attached, with gaze and collision opinions preserved. Native per-eye visibility
and perceived headset comfort still require an operator check.

### Pickup with Touch controllers

Grip/trigger pickup also applies in controller walking unless an explicit optical-only
override is selected. In walking, controller face buttons remain unused: use keyboard
B to recenter and release the trigger to open rather than using controller Y.

1. Release trigger and **hold grip** to acquire the arm at its current position.
2. Move and rotate the controller to place the **actual robot palm/fingers** around the
   cube or cylinder on the centered practice table. Grip alone keeps fingers open.
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
cannot consume the reach's step budget. When a wrist reaches its joint limit, the
solver uses the other available joints to continue alignment without changing the
accepted palm reach. This applies to the live seven-joint solve in both stationary
and walking modes; a physically unreachable angle can still remain misaligned.
Joint limits remain enforced. Actual human elbow
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

**Possible later webcam-input study.** The sender and calibrated arm/finger fallback
are retained but disabled; the current [webcam preview](#webcam-alongside-quest) is
display-only. If webcam robot input is deliberately revisited, measure field-of-view
handover, tracking confidence, latency,
and calibration repeatability while wearing the actual headset. The hip-relative
landmarks are estimates rather than measured room depth, and neither camera can
recover a hand hidden from both views.

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
| Offline regression suite | Head rotation, release-latch recovery, contact limits, input-loss arm holding, joint armature, articulation health, and Quest/webcam priority | Headset transport, rendered behavior, or live grasp success |
| Table contact live replay | The reproduced 20-cycle close/open failure passes with a 1e-4 kg·m² hand-joint armature floor, including clean withdrawal and zero faults/base motion | Every contact trajectory, hardware tracking, or all hand orientations |
| Contact grasp live replay | Earlier smaller-object combined replay failed because the cube slipped; isolated successes retain their original configurations | Pickup of the enlarged objects, reliable cube pickup, or hardware tracking |
| Humanoid live replay | Actual IK, rigid-body physics, grip, assisted pickup/lift/release, and root stability | Default physical grasping or hardware tracking accuracy |
| Finger live replay | Measured movement of all ten digits and both thumb-opposition joints, source transitions, tracking loss | Real headset camera recognition |
| Historical fixed-camera replay | Former constant camera-to-body transform during Play/Pause and simulated head motion | Current orientation-driven eye arc or detailed compositor behavior for every runtime |
| Real-hand observer | Actual XR landmarks, optical commands, and measured robot-joint variation | Guaranteed tracking accuracy or detection from an unseen hand |

See the [dated validation record](docs/validation/isaac-sim-6.1.md) for test counts
and live results. From the repository root, use Isaac Sim's bundled Python for the offline suite:

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

Stationary CPU physics defaults to **100 simulated steps/second**, with a **90 Hz
requested rendering cadence**. Walking uses **200 Hz physics / 60 Hz rendering**;
its policy remains 50 Hz. These settings do not guarantee wall-clock FPS. Slow VR rendering stretches
simulation-time replays and motion response. See
[performance and CPU physics](docs/humanoid-control.md#performance-and-cpu-physics).

## Scene and pickup behavior

The scene uses NVIDIA's `full_warehouse.usd`. The small table is centered at
`(0.46, 0.0)`, with its top **0.92 m above the floor**. Its visual and collision
geometry are resized vertically so the feet remain grounded. The anchored robot
cannot walk to the side benches or squat to the floor.

The current physical-grasp scene spaces six distinct shapes across this centered
table. The sphere, cube, and orange cylinder form the nearer row; the
tall cylinder, flat block, and cone sit in the second row. They are lightweight
dynamic bodies with real collision and gravity:

| Color | Shape | Dimensions | Mass | Center X, Y (m) |
|---|---|---|---:|---|
| Blue | Cube | 6 × 6 × 6 cm | 60 g | 0.320, -0.160 |
| Orange | Cylinder | 6 cm diameter × 10 cm high | 80 g | 0.320, 0.000 |
| Green | Sphere | 7 cm diameter | 50 g | 0.320, 0.160 |
| Purple | Tall thin cylinder | 4 cm diameter × 14 cm high | 60 g | 0.440, 0.160 |
| Yellow | Flat rectangular block | 9 × 6 × 3 cm | 70 g | 0.440, 0.000 |
| Red | Cone | 7 cm diameter × 10 cm high | 50 g | 0.440, -0.160 |

Stationary mode sets the starting arm pose before the first physics step. This
prevents the asset's extended starting hands from striking the side objects as
the scene starts or resets. Restart Isaac and LOAD a fresh example to apply
the updated table and starting pose.

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
tangential motion. Graspable props normally remain outside this arm limit. During
physical decoupled walking, a measured upward normal on the hand (`normal.z >= 0.5`)
with at least **0.02 N** also limits inward arm motion into a graspable package.
Horizontal side contacts remain available for squeezing; tangential motion and upward
withdrawal remain possible. This is directional yielding, not a force controller or
proof that a shelf supports the box: top contacts with a free or tilted carton can
also qualify. Finger commands, gains, the walking policy and gaze are unchanged.
The final native carton press/withdrawal check passed, including **1,024 actual inward
requests limited by the projection** and clear withdrawal. This checks loaded contact
response, not a complete real-hand lower-and-rotate placement.
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

Yellow gaze and green candidate highlights share prebuilt visual-only shells. Each
shell permanently inherits its carton's motion and has no collision or mass schema.
Selection changes only color/opacity inputs on a material outside the rigid body's
subtree. Original materials and all material bindings remain untouched during Play.
If ownership changes while the winning color stays the same, no USD write occurs.
Clearing the last owner makes the shell transparent, revealing the original object.
This addresses cartons returning to stale authored poses when bindings changed with
Fabric enabled; even a `full` render-purpose binding failed the physical check.
The shell implementation passed a fresh physical reset test with a 20.30 cm
authored/live pose difference and at most 1.23e-7 m motion during set/clear.
Fresh rendered captures also passed: original carton, green selection, then the
restored original appearance. See the
[validation record](docs/validation/isaac-sim-6.1.md#carton-reset-and-explicit-reset-investigation).
Gaze input, ray direction, hit selection, and highlight colors/priorities are unchanged.

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
requests behavior, hand, gaze, and object rows at nominally **100 Hz in simulation
time in both modes**. Derived recording intervals account for the 200 Hz walking
physics timestep, so walking does not double these log rates. The 256×256 eye-camera PNG rate
remains approximately 10 Hz; gaze raycasts remain 50 Hz. PNG compression uses
lossless level 1 to reduce encoding time; files can be larger than earlier captures.
Frame timestamps identify saved images; wall-clock capture rates depend on runtime performance.
`get_rgba()` returns the most recently rendered image, and its timestamp is written
after PNG saving. The recorder does not store that render's original timestamp or
a per-frame camera transform, so CSV/image alignment is approximate rather than
hardware synchronization.

Hand inertia is recorded in `hand_joint_armature_floor_kg_m2`, the actual authored
`hand_joint_armature_kg_m2_by_path` map, and `hand_joint_armature_meaning`. Use the
actual values when comparing trials; the floor is numerical stabilization, not
measured hardware inertia. Earlier recordings without these fields predate this fix.

CSV buffers flush at a nominal 2.5-second simulation interval and on orderly cleanup.
One background writer handles up to eight pending batches, preserving file schemas
and row order. A full queue keeps new rows in their original buffers; orderly cleanup
drains accepted writes before starting a new session. No simulator APIs run on this
worker. Failed batches remain in `_behavioral_csv_failed_batches` with their original
paths and generate warnings; later rows for a failed file are retained rather than
appended out of order. This recovery buffer is memory, not durable storage.
An abrupt shutdown can lose queued or buffered rows. Save/review complete sessions
before changing recording schemas or training on them. PNG encoding remains synchronous;
frame timestamp rows are created only after an image saves successfully.

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
states describe its measured response. Final arm joint targets, walking-policy joint
actions, and body motor efforts are not in the current CSV schema. Hand poses alone
therefore do not specify the full action applied to the robot.

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
- The latest raycast sample feeds `gaze.csv` at nominally 100 Hz in both stationary
  and gesture-walking modes; raycasts remain 50 Hz.
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
The default `robot_head` view uses a level, stabilized robot frame at fixed optical
height while following headset pitch/roll and relative heading. **B** resets heading;
it never stores headset tilt as a horizon offset and does not
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

The supported entry point for the new physical experiment is
[Experimental gesture walking](#experimental-gesture-walking). The choices below
describe the retained **legacy** paths; they are separate from strict physical
gesture walking and do not validate balance with moving arms or payloads.

### Locomotion choices

The interactive default in `HumanoidExample.__init__()` is:

```python
self._g1_locomotion = "stationary"
```

Legacy alternatives are `policy` and `kinematic`; rebuilding the scene is required to
change the fixed-base configuration. Without strict physical locomotion, `policy`
runs Unitree's recurrent twelve-leg-joint actor at 50 Hz; `kinematic` integrates a
commanded base pose while holding posture. Kinematic fallback belongs only to this
legacy assisted path and provides no dynamic balance controller. The gesture launcher
instead selects the decoupled balance/walk policies and fails explicitly on policy errors.

Policy integration depends on joint-name mapping, trained observation scales/gains,
command limits, and resetting recurrent state. The actor's training joint order is not
PhysX's articulation order. Its command limits are 0.8 m/s forward, 0.5 m/s lateral, and
1.57 rad/s yaw. Historical travel/turning measurements in earlier revisions do not qualify
these modes for current manipulation loads. See
[locomotion implementation](docs/humanoid-control.md#locomotion-and-lifecycle).

In these legacy moving modes only, keyboard Up requests forward, Left/Right yaw, and Down brakes
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
which uses a stabilized robot mount and an eye arc driven by headset orientation.
Its **B** action only resets horizontal heading. Do not apply old room-scale calibration or
camera-cycle advice to the current default.

## Source reference and tuning

| File | Responsibility |
|---|---|
| [`humanoid_example_extension.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/humanoid_example_extension.py) | Registers the Humanoid example and builds its profile-specific instructions |
| [`humanoid_example.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/humanoid_example.py) | Scene, hand/controller input, IK, pickup, camera mount, recording |
| [`g1.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/g1.py) | Standing anchor, robot joint drives, optional locomotion |
| [`g1_decoupled_policy.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/g1_decoupled_policy.py) | Paired balance/walk actors, 29-joint observations, 15 leg/waist actions, and history |
| [`g1_foot_contacts.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/g1_foot_contacts.py) | Walking-only foot contact compatibility; preserves referenced asset mass properties |
| [`validate_gesture_walking_live.py`](tools/validate_gesture_walking_live.py) | Reproducible PhysX walking and right-arm acceptance replay |
| [`point_gaze_navigation.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/point_gaze_navigation.py) | `PointNavigation`: left-index floor dwell, destination locking, release, and bounded velocity commands; filename retained for compatibility |
| [`point_navigation_scene.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/point_navigation_scene.py) | Blue visuals, real floor/object approaches, segment clearance, and turning footprints |
| [`factory_route.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/factory_route.py) | Bounded incremental detour planning over verified floor |
| [`comfort_camera.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/comfort_camera.py) | Robot XY/yaw stabilization and fixed neutral optical height; no HMD filtering |
| [`tall_packages.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/tall_packages.py) | Lightweight tall cartons with physical handles |
| [`xr_pose.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/xr_pose.py) | Shared world-pose validation and time-based smoothing |
| [`material_highlights.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/material_highlights.py) | Preauthored visual shells for gaze/hand colors; avoids binding edits on running rigid bodies |
| [`grasp_controller.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/grasp_controller.py) | Contact-limited closing and eligible-object pinch/wrap support diagnostics |
| [`grasp_contacts.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/grasp_contacts.py) | Prepared PhysX contact reader for hands, object/scenery identities, and reader health |
| [`arm_contact.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/arm_contact.py) | Constrains final arm commands against measured scenery-contact normals |
| [`articulation_health.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/articulation_health.py) | Detects unusable joint states and supplies a reason for latched simulation pause |
| [`eye_gaze_tracker.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/eye_gaze_tracker.py) | Existing unified-eye gaze, raycasts, and visualization |
| [`behavioral_csv_writer.py`](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/behavioral_csv_writer.py) | Bounded background CSV writes; no simulator API access from the worker |
| [`tools/`](tools/) | Host-side offline tests, live replays and read-only inspection utilities |

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

The NVIDIA GR00T balance/walk weights use the **NVIDIA Open Model License**; their
source/configuration uses **Apache-2.0**. The older twelve-leg-joint Unitree RL Gym
checkpoint retains its **BSD 3-Clause** provenance; see
[robots/data/README.md](source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/data/README.md)
and the linked upstream sources/licenses. Robot and environment assets retain their respective
NVIDIA/asset-provider terms.
