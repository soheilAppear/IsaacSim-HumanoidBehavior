# Humanoid control developer guide

The interactive G1 example now defaults to **stationary manipulation**: its pelvis is
fixed to the world so it stays at its spawn position and orientation while you control
the arms and fingers. Walking, turning, and headset gait cannot move it in this mode.
Gaze, camera controls, and recording remain available.

This guide describes the G1 controller implementation and how to validate changes.
The longer [setup guide](../HUMANOID_VR_CONTROL.md) contains installation instructions
and historical experiments. Its recorded performance numbers are not acceptance results
for the current implementation.

## Controls

| Input | Result |
|---|---|
| Left/right XR stick axes; X/A buttons | No robot movement in the default stationary mode |
| Grip held | Move/rotate that arm relative to its clutch calibration |
| Trigger | Close all fingers; request nearby assisted pickup at 0.60, release at 0.35 |
| B | Recenter view |
| Y | Drop both objects; release/open before picking up again |
| Left stick click | Cycle camera mode |
| Optical hand tracking | Wrist/palm target plus individual finger curls; no controller clutch |
| Keyboard arrows/numpad movement keys; gamepad locomotion inputs | No robot movement in the default stationary mode |
| Head motion/step-in-place | No robot movement; tracking remains available for view, gaze, and recording |

Triggers on XR controllers never command walking. Grip alone leaves fingers open.
Stick clicks are distinct from stick axes: the left stick click still changes camera
mode even though tilting either stick does not move the robot.

### Pick up with VR controllers

1. Recenter with **B** while facing the robot's forward direction. Keep the trigger
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
2. Reach and rotate your hand to guide the corresponding robot palm toward an object
   on the near edge of the small front-right table. Optical tracking does not require
   a grip button or controller clutch.
3. **Close your fingers into a fist** near the object to request pickup, then move your
   closed hand to lift. The average measured finger curl must reach 0.55 to engage.
4. **Open your hand** to release; average curl at or below 0.35 releases the attachment.
   If tracking is lost, the object is released. Open the hand before closing again
   after a tracking-loss drop.

Each hand works independently. Pickup remains assisted by a fixed joint; neither a
highlight nor a closed hand alone guarantees attachment when the object is out of reach.
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

Optical finger curls come from available skeleton joints. Missing finger roles relax
open and remain in the whole-hand averaging denominator, so one tracked thumb cannot
look like a fully closed hand. Controller trigger pressure drives all roles. Only
independent finger driver joints are commanded; mimic joints follow their coupling.
Curls update before the arm/pickup decision in each physics tick.

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

## Live acceptance procedure

1. Save your current stage and reload the changed extension/application. Open
   **Window → Examples → Robotics Examples → Policy → Humanoid: Unitree G1**, LOAD,
   then Play. Check the console for import, articulation, policy, and IK errors.
2. Confirm the resolved locomotion mode is `stationary`. Tilt both sticks, press X/A,
   try keyboard movement keys and gamepad inputs, and move your head. The robot's base
   position and orientation must stay fixed. Repeat while reaching with both arms.
3. Hold grip with trigger released: the hand should stay open. Reach forward/up/out,
   rotate the wrist, then release/reengage grip. Verify the actual palm follows smoothly,
   does not jump when reacquired, and does not move or tip the robot's base.
4. Reach the actual palm/fingers close to an object, pull trigger, and verify the object
   does not snap to the wrist on attachment. Lift, release, and observe it fall. Press Y
   while trigger remains held; it must stay released until you open and close again.
   Keep grip held while carrying; verify releasing grip also drops the object.
5. Repeat with optical hand tracking, partial occlusion, and device loss/reconnection.
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
