# Humanoid control developer guide

The interactive G1 example now defaults to **stationary manipulation**: its pelvis is
fixed to the world so it stays at its spawn position and orientation while you control
the arms and fingers. Walking, turning, and headset gait cannot move it in this mode.
Gaze, camera controls, and recording remain available.

This guide describes the G1 controller implementation and how to validate changes.
The longer [setup guide](../HUMANOID_VR_CONTROL.md) contains installation instructions
and historical experiments. Its recorded performance numbers are not acceptance results
for the current implementation.

Optical control accepts a valid **open hand** immediately. Finger updates do not
require a fist, a grip/pinch gesture, or an active arm target. Closing the hand is
only needed to request pickup. The reported real-headset symptom of detection only
while making a fist remains unresolved: recent hardware observations did not provide
valid finger skeletons, so continuous open-hand tracking has not been verified there.

## Controls

| Input | Result |
|---|---|
| Left/right XR stick axes; X/A buttons | No robot movement in the default stationary mode |
| Grip held | Move/rotate that arm relative to its clutch calibration |
| Trigger | Close all fingers; request nearby assisted pickup at 0.60, release below 0.35 |
| B | Restore the fixed robot-head view |
| Y | Drop both objects; release/open before picking up again |
| Left stick click | Keep the robot-head camera locked in stationary mode |
| Optical hand tracking | Wrist/palm target, five independent finger curls, and separate thumb opposition; no controller clutch |
| Keyboard arrows/numpad movement keys; gamepad locomotion inputs | No robot movement in the default stationary mode |
| Head motion/step-in-place | No robot or camera movement; existing gaze and recording remain available |

Triggers on XR controllers never command walking. Grip alone leaves fingers open.
Stationary mode also prevents stick clicks from selecting a room-scale camera mode.

### Pick up with VR controllers

1. Restore the robot-head view with **B**. Keep the trigger
   released, then **hold the grip** on the hand you want to move.
2. Move that controller forward, upward, or sideways and rotate it to guide the robot
   palm and wrist. Start with the objects on the near edge of the **small front-right
   table**. Watch the actual robot fingers approach a reachable object; the
   target marker alone does not establish pickup range.
3. Keep holding grip and **squeeze the trigger** to close the fingers. At 60% trigger
   travel or more, the controller requests pickup if the simulated hand is close enough.
4. **Keep grip and trigger held** while moving the controller to lift or carry.
5. Ease the trigger below 35% to release. **Y** drops objects from both hands. After Y,
   release the trigger before squeezing again. Letting go of grip also releases the
   object and ends that arm's tracking.

### Pick up with optical hand tracking

1. Use your runtime's hand-tracking mode so it supplies wrist/palm and finger skeleton
   poses. Present an **open hand** in the tracking cameras' view.
   The launcher and extension enable Kit's OpenXR Hand Tracking component; restart
   an already-running XR session once after updating. Keep the existing OpenXR runtime.
   No fist or pinch is needed to begin tracking. A valid wrist/palm drives the arm;
   valid finger landmarks independently drive the fingers.
2. Reach and rotate your hand to guide the corresponding robot palm toward an object
   on the near edge of the small front-right table. Optical tracking does not require
   a grip button or controller clutch.
3. **Close your fingers into a fist** near the object to request pickup, then move your
   closed hand to lift. The average measured finger curl must reach 0.55 to engage.
4. **Open your hand** to release; average curl below 0.35 releases the attachment.
   If tracking is lost, the object is released. Open the hand before closing again
   after a tracking-loss drop.

Each hand works independently. Pickup remains assisted by a fixed joint; neither a
highlight nor a closed hand alone guarantees attachment when the object is out of reach.
Move one finger at a time to control the corresponding robot finger; thumb opposition
is independent of thumb flexion. Inspire has six actuators: each finger's distal joints
are mechanically coupled, so individual human knuckles and finger splay cannot all be
reproduced independently. Controllers retain whole-hand trigger closure; individual
finger tracking requires the optical skeleton from the headset.
The distant side benches and floor objects remain in the scene; the stationary robot
cannot walk over to them or lower its base to reach the floor.

## Source layout

Paths in this table are relative to
`source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/`.

| File | Responsibility |
|---|---|
| `interactive/humanoid/humanoid_example.py` | Scene, input arbitration, arm/finger retargeting, pickup, camera, recorder, lifecycle |
| `interactive/humanoid/xr_pose.py` | Validated virtual-world poses and timestep-independent smoothing |
| `interactive/humanoid/eye_gaze_tracker.py` | Eye/head provenance, raycasts, latest sample, gaze visuals |
| `interactive/humanoid/material_highlights.py` | Shared reversible gaze and hand material overrides |
| `robots/g1.py` | Stationary world anchor, joint ownership, posture, optional walking actor/kinematic mode, reset |
| `robots/data/g1_unitree_motion.pt` | Bundled Unitree TorchScript locomotion checkpoint |

Keep these files together when installing. Copying only the example file omits its
helper imports and robot changes. The complete extension can be copied or linked as
described in the setup guide. Save the stage and reload the app/example after edits;
an already-created Python instance retains its earlier class implementation.

## Frames and pose validity

The stage is Z-up. Base commands are `(forward, left, yaw)` in body coordinates
(m/s, m/s, rad/s); positive yaw turns left. XR stick-right therefore maps to negative
body Y. Quaternions passed to Isaac pose APIs use `w, x, y, z`. `Gf.Matrix4d` uses
row-vector composition: a local frame transforms to world as `local * body_world`.

Arm and gaze readers only consume virtual-world poses. Physical room poses cannot
be mixed with stage coordinates after recentering or robot movement. The experimental
head-bob gait detector intentionally uses physical HMD height; it remains off by default.
`xr_pose.read_world_pose()` checks both position-valid and orientation-valid bits on
Kit's pose descriptor, validates matrix dimensions, finite values, and a nonsingular
positive rotation determinant. A valid descriptor may contain an identity pose.
The legacy matrix-only API has no validity flags, so an identity matrix is treated
as unavailable. Missing or invalid input never falls back to a stale matrix.

## Arm and finger retargeting

On controller grip engagement, the example saves the tracked hand position and the
robot palm position in the robot's yaw-aligned body frame. Subsequent hand deltas
are scaled and bounded from those anchors. Translating or turning the robot therefore
does not itself move the target within the body frame. Releasing grip or switching
input source clears calibration. Optical hands use their current body-frame position.

Wrist orientation is calibrated relatively so the robot keeps its starting wrist
axes while following operator rotation. With row rotations, the desired rotation is:

```text
H = tracked_world_rotation * inverse(base_yaw_rotation)
desired_world = robot_hand_body_at_clutch * inverse(H_at_clutch) * H * base_yaw_rotation
```

The live solve uses all seven arm joints and gives palm position first priority.
A damped-least-squares position step is followed by a wrist-orientation correction
in the remaining position nullspace, computed with an SVD projector. An unreachable
wrist orientation therefore cannot reverse the primary reach step. A small default
posture gain (`_arm_posture_gain = 0.1`) returns redundant joints toward their standing
defaults within the combined position/orientation nullspace. If the IK correction
exceeds the per-joint step cap, one common factor scales the whole correction before
command limiting, preserving its direction.

The solver handles the root row/column differences between fixed- and floating-base
PhysX Jacobians. Because those Jacobians are world-space, position
residuals are rotated into world space. The palm offset contributes angular motion:
`J_palm = J_linear + J_angular × palm_offset`, column by column. The hand pose and
Jacobian must refer to the same link; an elbow is not substituted for a wrist.

Final targets obey joint limits and a 2.5 rad/s command slew limit relative to the
*previous target*. A separate 0.15 rad bound limits the target's lead or lag relative
to the current physical joint. This lets the position drive develop enough tracking
error to move the arm, while bounding stored error if the arm meets an obstacle.
Limiting that error to `speed * dt` every tick had made the arm respond too slowly.
External contacts can make the two bounds conflict; the measured-position bound takes
priority in that case. These are target limits, not a guarantee about measured velocity
under contact. Source changes and tracking loss clear joint and hand-target smoothing;
reacquisition starts from the measured joints. Filters account for `dt`.
If the live Jacobian is unavailable, the existing measured four-joint local map remains
a degraded position fallback; it cannot reproduce the full wrist-orientation solve.

Optical finger curls sum adjacent bone bends along each complete digit chain. This
avoids a deep fist wrapping past 180 degrees and reopening the robot finger. Joint
positions need the OpenXR position-valid flag; missing orientation is acceptable for
this position-only calculation. Thumb opposition is measured in the hand's own palm
plane and has a separate target and recorded `thumb_yaw` column.
An open digit has zero curl and is still a valid measurement. Finger updates run
independently of the arm's active target and the pickup closure threshold.
Missing finger roles relax open and remain in the whole-hand averaging denominator, so one tracked thumb cannot
look like a fully closed hand. Controller trigger pressure drives all roles. Only
independent finger driver joints are commanded; mimic joints follow their coupling.
Curls update before the arm/pickup decision in each physics tick. Kit can label a
Touch device `hand` before a skeleton arrives. The input classifier requires skeletal
landmark names for optical control; a valid grip pose plus real trigger/thumbstick
actions preserves controller control when only interaction poses exist. A partially
present skeleton stays optical, so missing joints cannot activate stale trigger input.

## Robot-mounted camera

The default `robot_head` mode fixes the camera pose relative to the robot's head body.
The Inspire asset's visual head is fixed under `torso_link`, so the mount follows that
body's full live position and rotation. It uses physics tensors while running, including
with Fabric enabled. The camera keeps its existing `/World/G1_HeadCamera` path and eye
offset so the viewport and recorder share the same view.

Physical headset translation and rotation do not enter the mounted camera transform.
An application-frame subscription reasserts this view through XRCore during Play and
Pause, in addition to the physics update. **B** restores this fixed view; the left stick
click cannot unlock it in stationary mode. Older room-scale modes remain available
only for explicitly configured moving-robot experiments. Cleanup releases the frame
subscription, and reset discards stale body-pose handles. Gaze code and its settings
are unchanged by these camera and finger additions.

For Steam Link, enable hand tracking on the headset and switch from controllers to
bare hands. Valve documents that Steam Link forwards the skeleton through
`XR_EXT_hand_tracking`. Controller aim/grip poses alone are insufficient for per-finger
control. [Valve's hand-tracking announcement](https://store.steampowered.com/news/posts/?appgroupname=SteamVR&appids=250820&enddate=1728412551&feed=steam_community_announcements)

## Pickup and tracking loss

Pickup is distance-gated **fixed-joint assistance**. It is not a friction/contact
grasp controller. The object stays dynamic, but the constraint may hold objects that
the fingers alone could not. The recorder identifies this assisted mode in metadata.

An active hand searches from the measured palm/finger centre: the live wrist-link pose
plus its rotated palm offset. The candidate search therefore uses the same working
point as arm IK, rather than the wrist origin or the desired XR marker.
Object poses also come from physics tensors; USD transforms may be stale with Fabric.
The default 0.30 m radius highlights a candidate, 0.18 m permits candidate selection
for pickup, and 0.09 m gates the nearest palm/finger link-centre distance to the object
centre. That final test does not establish surface contact.

For a successful request, both fixed-joint frames coincide at creation:

```text
local_hand_frame = object_world * inverse(hand_world)
local_object_frame = identity
local_hand_frame * hand_world == local_object_frame * object_world
```

This retains the live relative pose instead of pulling the object into the wrist.
The joint is excluded from the articulation and collisions between the joined pair
are disabled. No ease-in, object teleport, or kinematic conversion is performed.

Controller pickup uses 0.60/0.35 close/open hysteresis; optical hands use mean curl
0.55/0.35. Y/drop and tracking loss while holding set a release requirement: an unchanged
closed hand/held trigger cannot reattach immediately. Releasing grip, disabling tracking,
losing a usable hand pose, clearing, or resetting removes the constraint and clears
targets and cached physics handles. Finger tracking loss opens the driven fingers.
Reconnection starts with fresh calibration and a target slew starting at measured joints.

Gaze and hand highlights share an anonymous session sublayer prepared during scene
setup, before physics creates tensor views. Grab selection has priority over gaze on
the same object. Strong parent bindings also tint referenced child meshes. Runtime
selection changes update only material relationship targets and binding strength.
Releasing the last selection clears these property opinions, restoring the original
authored/inherited materials. The layer, prim overrides, and binding API schemas stay
in place for the stage's lifetime. Cleanup also tolerates an already-closed Kit stage.

The grip-related freeze was traced to the previous highlight implementation: changing
from one highlighted package to another removed and recreated a USD prim override.
During a physics callback this made PhysX reparse the referenced collision mesh and
invalidate its simulation tensor view. Subsequent robot pose/target calls then failed
with `Failed to get DOF position targets from backend`. The failure was reproduced
inside the running VR application by switching gaze highlights from Box_03 to Box_00.
Do not use `RemovePrim`, change applied schemas, or detach highlight layers to clear
selection while physics is running.

## Locomotion and lifecycle

The interactive example selects `self._g1_locomotion = "stationary"`. In this mode a
fixed world-to-pelvis joint anchors the base at its spawn pose, the robot holds its
standing posture, robot-link gravity is disabled, and no walking checkpoint is run.
Scene objects retain their normal gravity. Arms and fingers keep their
separate control paths. Base commands are suppressed at both the example and robot
wrapper, including keyboard, XR/gamepad sticks and buttons, and headset gait. The
world constraint also prevents arm motion and carried-object reactions from tipping
the free base. This is a supported manipulation setup, not a demonstration of balance.

The following modes remain available for explicitly configured experiments. They are
**not the interactive default** and their physical stability is not established by
the stationary-mode tests.

`policy` mode runs the bundled 47-observation recurrent actor and writes twelve leg
targets at 50 Hz. Arms and fingers remain separately owned. Scheduling uses elapsed
time, including at 60, 90, 100, 120, and 200 Hz physics. A long frame evaluates one fresh
observation rather than advancing recurrent state repeatedly on identical input.
Nonfinite observations/actions are rejected, the last valid targets are retained,
and leg targets are clipped to available joint limits.

`kinematic` mode integrates the root pose with gravity disabled. Missing joints or an
unloadable policy choose this fallback before selecting posture and reset defaults.
Reset restores the full spawn position/orientation and zero root/joint velocities,
resets the actor, and clears input/filter state. Keyboard presses are tracked as a set
so repeat events and unmatched releases cannot accumulate velocity.

The existing policy station-hold and turn-hold options directly assist the base. This is
not a newly trained whole-body balance controller. The actor was not retrained for
arbitrary teleoperated arm poses, payloads, or uneven terrain. Physical stability and
reverse response remain live acceptance items, even when the checkpoint test passes.

## Gaze and recordings

The tracker uses combined eye-gaze devices; per-eye view transforms are not assumed
to measure gaze. If real eye input is absent, the optional HMD-forward ray is labeled
`hmd_forward`. A missing ray uses `None` (blank in CSV); real combined gaze is `eye_tracker`. These
labels reach the CSV unchanged. Runtime support, permissions, and calibration must
be checked separately from whether an OpenXR extension is advertised.

Pose validity is checked every tick; raycasts run at 50 Hz. Source changes and loss
update immediately. An all-hit scene query selects the nearest non-robot collider,
including nearby objects and static collider paths. It does not assume returned hits
are sorted or skip the first 35 cm of the ray.

Default physics is 100 Hz. Sensor rows are written each physics tick and reuse the
latest 50 Hz gaze result; 100 logged rows do not mean 100 independent gaze raycasts.
`GazeSample.sample_time` records query time for in-process consumers. Camera capture
is requested at 10 Hz and CSV buffers flush about every 2.5 seconds. `metadata.json`
records requested/resolved locomotion mode, grasp mode, and assistance settings.
The session's simulation clock stays monotonic across world resets; gait and velocity
filters clear so reset discontinuities do not become motion estimates.

## Performance and CPU physics

The defaults request **100 Hz CPU physics** and **90 Hz rendering**, using the Torch
backend. These are configured simulation intervals, not a guarantee of 90 displayed
frames per second or real-time simulation. During a previous live VR run, the
simulation advanced about **0.3 simulated seconds per wall-clock second**. XR load
varied between runs; that observation is not a benchmark for every scene or headset.
When the simulation falls behind, robot movement and tests that wait for simulated
time also take longer in the real world.

CPU physics was chosen after an earlier scene comparison reported approximately
6.8 ms per step on CPU versus 23–28 ms on CUDA. Those historical measurements are
not a current benchmark of this stationary scene. Switching to GPU physics is not
an established fix for the present slowdown.

The code performs the following work while the example is loaded and playing.
These are plausible contributors; their individual costs have **not been measured**
in the current setup.

| Work | Current behavior |
|---|---|
| Automatic recording | `_behavioral_data_enabled = True`; behavior, hand, gaze, and object rows are collected at about 100 Hz |
| Recorded camera | An additional 256×256 camera render product and RGB annotator refresh at render cadence, even though PNGs are saved at about 10 Hz |
| File writes | PNG encoding and saving happen synchronously inside the physics callback; buffered CSVs are also written there about every 2.5 seconds |
| Camera attachment | The body pose is read and the mounted view scheduled every physics tick and every application frame, including while paused |
| Arm and pickup poses | Palm/finger poses are read repeatedly for IK, candidate selection, and grasp distance; each active arm separately fetches the full articulation Jacobian and joint positions |

Finger landmarks are cached within each hand's curl calculation. Joint command
limits and smoothing also affect movement response, independently of frame rate;
their current values and purpose are described under arm and finger retargeting.
A paused, empty session is not executing this example's physics, IK, or recording
callbacks. Check which application and scene are active before attributing a slowdown
to those paths. Run only one Kit application during comparisons.

### Compare recording on and off

1. Use the same scene, headset connection, renderer settings, physics settings, and
   repeatable hand movements for both runs. Leave gaze behavior and settings unchanged.
   Wait for scene loading and shader compilation to settle before measuring.
2. Start with the default `_behavioral_data_enabled = True` in
   `HumanoidExample.__init__()`. Restart Isaac Sim, load the example, and press Play.
3. Measure wall time and simulated progress with the read-only probe below. Repeat
   it three times and retain the results rather than choosing only the fastest run.
4. Set `_behavioral_data_enabled = False` **before restarting and loading the example**,
   then repeat the same measurements. Disabling it only after the recording camera
   has initialized does not establish that its render product was removed. The fresh
   run is needed to compare against a scene where that product was never created.
5. Compare simulated-seconds/wall-seconds and the viewport's measured frame rate.
   Restore recording to `True` and reload when returning to data collection.

Run this PowerShell probe from the repository root with the example playing and the
Python server enabled:

```powershell
@'
import asyncio
import time
import omni.timeline

assert EX is not None and EX._physics_ready, "Load the Humanoid example and press Play"
assert omni.timeline.get_timeline_interface().is_playing(), "Keep the timeline playing"
wall_start = time.perf_counter()
sim_start = EX._headset_gait_time
await asyncio.sleep(20.0)
wall_seconds = time.perf_counter() - wall_start
sim_seconds = EX._headset_gait_time - sim_start
print({"wall_seconds": wall_seconds, "sim_seconds": sim_seconds,
       "real_time_factor": sim_seconds / wall_seconds})
'@ | python tools/kit_exec.py --timeout 45 -
```

The simulation clock advances independently of recording, so the probe works in
both runs. A real-time factor of 1.0 means one simulated second per wall-clock second;
0.3 means roughly six simulated seconds during this 20-second observation. This
comparison measures the combined recording cost, not separate PNG, CSV, rendering,
or controller costs. Lowering `_behavioral_frame_log_every_n_steps` increases the PNG
rate; raising it reduces saves but does not by itself reduce the render product's
refresh rate. `_behavioral_data_log_every_n_steps` controls the sensor-row interval.

## Automated validation

Run from the repository root with an Isaac Sim standalone Python wrapper:

```powershell
& "C:/path/to/isaac-sim-standalone-6.0.0-windows-x86_64/python.bat" tools/run_humanoid_tests.py
```

```bash
/path/to/isaac-sim/python.sh tools/run_humanoid_tests.py --isaac-sim-dir /path/to/isaac-sim
```

The runner locates bundled NumPy, PyTorch, and USD libraries. It does not start Kit,
connect XR, or change an open stage. Tests use the actual production class/function
definitions with module imports replaced by controlled dependencies: XR devices,
Kit services, and articulation tensor views are doubles; matrix math, USD stages,
materials, joints, and the bundled TorchScript checkpoint are real.

Coverage includes invalid poses; source transitions; moving/turning clutch frames;
wrist orientation and Jacobian indexing; target speed limits; finger loss; keyboard
repeat; lateral direction; trigger separation; hysteresis; drop rearming; reset;
coincident grasp frames; material restoration; policy timing; invalid actions; joint
limits; fallback posture; and deterministic checkpoint reset. Highlight regressions
also observe real USD `ObjectsChanged` notices: repeated gaze/grip overlap, target
changes, release, and reset must not resync any collision prim. A separate teardown
check covers an invalid stage wrapper after the application closes its stage. IK
regressions also cover conflicting wrist/position requests, redundant-joint posture
recovery without changing the hand task, and world-axis rotation from a nontrivial pose.
Open-hand regressions verify acquisition without grip, pinch, or fist and independent
finger updates without a closed hand or active arm target.

These tests do not validate extension startup/import ordering, live XR bindings, rendered
visuals, collision/contact dynamics, or balance. Run Kit integration checks separately
and sequentially. Do not start a second Kit application alongside an existing session.

### Replay against the running VR application

With the updated code loaded in Isaac Sim XR VR and the Python server enabled, run
this command with ordinary host Python:

```powershell
python tools/validate_humanoid_live.py
```

It exercises grip hold/movement/release, switches gaze and hand highlights inside the
actual physics callback, and reaches for Box_00, attaches it, lifts it, and releases
the trigger. It asserts root position/orientation stability, simulated tick progress,
empty callback-error records, and successful pickup/release. The default timeout is
60 seconds. Both input devices and the gaze callback are restored on exit, held objects
are released, and the timeline is left paused.

For repeatable initial package positions, save any stage edits and explicitly rebuild
the example first:

```powershell
python tools/validate_humanoid_live.py --reload-example
```

This option clears the current stage and reloads the example. It does not reload Python
modules; restart Isaac Sim after source changes so the running classes match the files.
The replay supplies synthetic controller poses/buttons and highlight targets at the
XR boundary. Joint drives, IK, collisions, pickup constraints, and physics callbacks
are real. It does not verify headset/controller hardware or optical finger tracking.

Live checks on 7 September 2026 reproduced the old highlight-switch freeze and passed
after the fix. The complete saved replay, with wrist tracking enabled and a fresh
scene, passed 1,286 physics ticks (12.86 simulated seconds). It attached Box_00 at a
0.0781 m nearest-finger gap, lifted it 0.1292 m, and released it when the trigger opened.
Root translation and rotation stayed unchanged, and no callback errors occurred.
A separate six-cycle replay of both grips and repeated gaze changes passed 155
physics steps with unchanged root position and orientation. A live 20-degree wrist
command reduced angular error from 19.5 to 4.4 degrees with 0.008 m palm displacement.
These are stationary simulation results; they do not establish walking balance or
hardware tracking quality.

### Camera and finger validation

Run these checks sequentially against the already-loaded stationary example:

```powershell
python tools/validate_camera_live.py
python tools/validate_fingers_live.py --timeout 300
```

The camera replay checks a constant camera-to-body mount during Play and Pause with
six simulated physical-head poses, B, and left stick click. It forwards camera updates
to the real XRCore and verifies that gaze's tracker instance is retained. World camera
motion caused by actual torso deflection is allowed; movement relative to the torso
is not. The finger replay drives each of ten fingers and both thumb-opposition joints,
then checks partial loss, controller triggers, misleading source metadata, and complete
loss against measured PhysX joints. These scripts restore input and leave the timeline
paused. They do not validate the headset's cameras or finger transport. A low VR frame
rate stretches wall time because the finger replay waits for simulated time.

For a shorter controller-transition check, use
`python tools/validate_fingers_live.py --controllers-only --timeout 120`; its report
explicitly marks the optical checks as skipped.

To observe actual hardware, press Play, put down the controllers, switch the headset
to bare hands, and bend individual fingers while running:

```powershell
python tools/observe_hand_tracking_live.py --seconds 20
```

This observer supplies no input and changes no timeline, gaze, or tracking settings.
It reports real skeletal landmarks, optical targets, and measured robot-joint ranges.
Run it separately from synthetic replays. Motion flags show real optical input and
corresponding joint variation; they do not measure detailed tracking accuracy. No motion
can mean absent/occluded tracking, a still hand, or too few physics updates.

The current suite contains **82 offline regression tests**, including two open-hand
acquisition and independent-finger regressions. On 7 September 2026, the full live finger
replay passed **19.73 simulated seconds** with all ten fingers and both thumb-opposition
joints reaching their independent 0.7 normalized targets. Partial/full tracking loss,
normal controller input, and Touch input mislabeled as `hand` passed for both hands;
root translation was zero and no callback errors occurred.

The live camera replay passed both Play and Pause with the Quest session connected.
Maximum camera-to-body matrix change was `3.33e-16`; the authored camera matched the
computed mount exactly. A separate read of the real headset's virtual eye position
matched the mounted camera within `4.25e-8 m` after the physical headset had moved
over a metre. Gaze still reported `eye_tracker` with zero failed updates.

Actual skeletal poses were seen intermittently after enabling Kit's hand component,
but subsequent 20- and 45-second real-input observations contained no valid finger landmarks.
The synthetic replay therefore establishes retargeting and joint-drive behavior;
it does not establish continuous tracking from the Quest Pro's cameras. The reported
fist-only detection symptom remains unresolved. There is no application requirement
to make a fist before acquiring an optical hand, and an open finger's zero curl is
valid. Without valid incoming skeleton data, the hardware behavior cannot be confirmed
or described as fixed. Repeat the observer with open hands and individual finger
movements to separate missing/occluded input from a frozen simulation, without
altering gaze or the active OpenXR runtime.

## Live acceptance procedure

1. Save your current stage and reload the changed extension/application. Open
   **Window → Examples → Robotics Examples → Policy → Humanoid: Unitree G1**, LOAD,
   then Play. Check the console for import, articulation, policy, and IK errors.
2. Confirm the resolved locomotion mode is `stationary`. Tilt both sticks, press X/A,
   try keyboard movement keys and gamepad inputs, and move your head. The robot's base
   position and orientation must stay fixed. Repeat while reaching with both arms.
3. Move your physical head sideways, vertically, and rotate it. The camera must retain
   its fixed transform relative to the robot head. Repeat while paused and press B or
   the left stick: neither should enable room-scale camera movement.
   Hold grip with trigger released: the hand should stay open. Reach forward/up/out,
   rotate the wrist, then release/reengage grip. Verify the actual palm follows smoothly,
   does not jump when reacquired, and does not move or tip the robot's base.
4. Reach the actual palm/fingers close to an object, pull trigger, and verify the object
   does not snap to the wrist on attachment. Lift, release, and observe it fall. Press Y
   while trigger remains held; it must stay released until you open and close again.
   Keep grip held while carrying; verify releasing grip also drops the object.
5. Present open hands first, without making a fist or pinching, and verify tracking
   starts. Bend each optical finger separately and move the thumb across the palm without
   curling its tip; verify independent robot-finger and thumb-opposition movement.
   Repeat pickup with optical hand tracking, partial occlusion, and device loss/reconnection.
   Verify objects release and stale hand targets are not used. Reset while holding an
   object; verify no joint remains and the robot returns to its spawn pose.
6. Hold the head still and move only the eyes. Confirm the runtime reports real
   `eye_tracker` and the ray changes. Disable eye input and verify `hmd_forward` or
   an unavailable source, with no stale object hit. Check a nearby object and a target behind robot
   self-colliders. Overlap gaze/hand selection and verify the original material returns.
7. Inspect the new session CSVs and metadata for source labels, resolved locomotion,
   moving object poses, monotonic times, and expected capture rates.

If deliberately testing a non-default moving mode, run its forward/reverse/strafe/turn
and stopping checks separately; a successful stationary session does not validate
walking or balance with teleoperated arms and payloads.

Optional read-only diagnostics use the repository's TCP helper after enabling
`isaacsim.code_editor.python_server` in **Window → Extensions**:

```powershell
python tools/kit_exec.py "print(EX.describe_xr())"
python tools/kit_exec.py "print(EX._eye_gaze_tracker.describe())"
```

`tools/launch_isaac_vr.bat` enables the server when launching a new session. A refused
connection on port 8226 means the server is unavailable; it does not diagnose the
robot or headset. The stationary manipulation path has also been exercised in the
running VR application with synthetic XR inputs and real PhysX, as described above.
Headset hardware tracking and optional walking modes require separate live checks.

`g1_grasp_env.py` is an experimental benchmark scaffold. It does not provide a complete
training environment with step/reward/reset semantics, and this repair does not claim
to train or integrate a new grasp or whole-body policy.

## API references

- [PhysX tensor Jacobians](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html): world-frame Jacobians and fixed/floating root layout.
- [USD joint frames](https://openusd.org/release/api/class_usd_physics_joint.html): local position/rotation for each connected body.
- [Physics scene queries](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/physics_umbrella/physics_umbrella_runtime.html): all-hit raycast callback and query lifecycle.
