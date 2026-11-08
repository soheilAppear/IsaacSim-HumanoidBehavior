# Humanoid control developer guide

The interactive G1 example now defaults to **stationary manipulation**: its pelvis is
fixed to the world so it stays at its spawn position and orientation while you control
the arms and fingers. Walking, turning, and headset gait cannot move it in this mode.
Gaze, camera controls, and recording remain available. The default grasp mode is
experimental **physical contact**: bounded finger drives and friction must support
the object, without a hand-to-object fixed joint. Finite compliant couplings and
arm-contact limits address the table-touch instability. Physical mode holds the arm
pose when its input/clutch is lost. Right-open, right-curled, and left-open table
tests passed sequentially in the same scene, including the corrected handoff. Those
tests held each finger pose. A later dynamic close/open test reproduced another
joint-speed fault. The selected **1e-4 kg·m²** armature floor passed the same 20-cycle
test with clean withdrawal, no faults, and zero base motion. A path reconstructed
from recorded measured arm joints also passed. The enlarged cylinder passed its
pickup trial; both enlarged-cube pinch trials failed to establish opposing support
while the articulation remained stable. Cube reliability remains unresolved.

**Cube pickup remains pose/contact-path-sensitive.** An opposed-thumb retry passed,
but the final fresh-scene combined replay slipped the cube again. The cylinder passed,
but the combined validation failed; earlier successful lifts do not establish a robust
cube grasp. These results used the earlier 4 cm objects; the enlarged shapes were
evaluated separately, with the cylinder passing and both cube pinches failing.
Current results belong in the
[validation record](validation/isaac-sim-6.1.md), with headset handling
still an operator check.

This guide describes the G1 controller implementation and how to validate changes.
The longer [setup guide](../HUMANOID_VR_CONTROL.md) contains installation instructions
and historical experiments. Its recorded performance numbers are not acceptance results
for the current implementation.

The current opt-in [walking profile](#controller-sticks-and-optical-hands) uses
controller sticks for base movement and optical hands for arms/fingers. NVIDIA's
decoupled actors observe all 29 body joints and command legs/waist; no new gait is
generated here. Blue pointing navigation remains in the repository but is disabled
in both walking launchers. Twelve shelf cartons join the existing table/floor props.
Earlier pointing, walking and lifecycle results do not validate this new profile,
bimanual shelf pickup or loaded transport. Stationary defaults and gaze are unchanged.

Optical control accepts a valid **open hand** immediately. Finger updates do not
require a fist, a grip/pinch gesture, or an active arm target. Pickup requires actually
closing the robot fingers around the object. The user confirmed that eye and finger tracking
returned after starting SteamVR independently. The 12 September recording also contains
both optical hands and real eye-gaze samples. This confirms input recovery; the new
palm alignment and response changes still require a live hardware acceptance check.

## Optional webcam input

**Current mode is display-only.** `tools/launch_webcam_tracking.bat` always selects
`--preview-only`: MediaPipe body/optional hand inference and the preview stay active,
but no UDP socket is created, no packet is encoded/sent, and C cannot increment a
calibration request. `--headless` is rejected with this mode. The overlay identifies
it as **DISPLAY ONLY - robot uses Quest/controllers**. Standard Isaac launchers force
`G1_WEBCAM_ENABLED=0`, including walking, overriding a stale `1` environment value.
Quest optical tracking or controllers are the robot's arm/finger sources; the webcam
is only an operator/landmark view suitable for OBS capture.

`tools/webcam_tracking.py` runs MediaPipe Tasks on CPU in a separate environment
and process. A capture thread retains only its newest frame. Pose inference uses
VIDEO mode and shared `perf_counter_ns()` timestamps; optional `--hands` adds detailed hand
inference. The default lite model targets 24 updates/s. Actual rate and frame age
depend on the camera and PC load. Frame timestamps mark availability after
OpenCV `read()`, so reported age excludes unknown exposure/driver buffering.

### Retained sender/receiver experiment — disabled

The architecture below describes the earlier robot-input experiment, not the current
preview workflow. It remains for future manual development. Directly running
`tools/webcam_tracking.py` without `--preview-only` retains sender behavior; the
standard `.bat` preview launcher offers no sender override. Its historical validation
does not imply that current previews influence the robot.

The humanoid-side `webcam_tracking.py` has no OpenCV or MediaPipe dependency.
It receives bounded JSON datagrams on loopback UDP port 8765, rejects malformed,
nonfinite, stale, future, and out-of-order samples, and requires explicit calibration.
Only one receiver should bind the selected port. No UDP packet can command gaze,
the camera, the base, or joints directly.

| Stage | Responsibility |
|---|---|
| Webcam process | Unmirrored pose and optional hand landmarks; preview mirroring is display-only; C increments a calibration-request counter |
| Receiver | Validate protocol version, session, sequence, timestamp, landmark count/range, visibility and presence; drain bounded pending input once per control tick |
| Calibration | Require both fresh Quest optical palm targets and visible webcam arms; establish anatomical +X forward, +Y left, +Z up; scale and align shoulder-relative arm motion |
| Input selection | Quest optical first; valid controller grip/trigger fallback in stationary and controller walking profiles unless explicitly optical-only; calibrated webcam fallback when primary input is unavailable |
| Arm controller | Quest controls its full arm while valid; only webcam arm fallback enables webcam elbow guidance in remaining IK freedom; existing joint/contact limits |
| Fingers | Optical curls first; controller trigger mapping only outside optical-only profiles; optional complete webcam landmarks provide fallback curls |

World pose landmarks are monocular estimates centered at the hips, not camera
extrinsics or measured room positions. Human arm span sets scale against the
measured G1 shoulder-to-wrist link span (0.37 m fallback if unavailable); a paired
Quest palm target supplies alignment.
This prototype assumes a single operator and requires visible shoulders and hips.
No network training or full-body locomotion is introduced.

Outside the standard launchers, `G1_WEBCAM_ENABLED=1` can configure a manually created
experimental instance; standard launches always override it to `0`. Retained helpers are
`enable_webcam_tracking(port=8765)`, `disable_webcam_tracking()`,
`request_webcam_calibration()`, and `webcam_tracking_status()`.
Reset/reload, sender restart, and headset recenter invalidate calibration.
Calibration requests expire after ten seconds if paired input is unavailable.
Confidence loss does not extrapolate hidden limbs indefinitely. Source transitions
retain smoothing; physical contact and articulation-health checks remain authoritative.
Selection is independent per arm and per finger input. A valid Quest palm is never
discarded because its `POSITION_TRACKED` bit clears, and webcam visibility is not
required for Quest control. Webcam elbow guidance runs only when the selected
arm source is `webcam_body`. This preserves the established position-valid Quest
mapping, including runtimes that temporarily infer poses. If a runtime retains
such valid inferred poses outside its camera view, webcam takeover waits until
the Quest input actually becomes unavailable. No source estimates indefinitely
hidden hand motion after both inputs become unavailable.
In controller walking, the user-confirmed Quest/SteamVR setup switches between
controller and optical-hand input. Controller grips/triggers now control arms/fingers
in the walking profile while controllers are active. Optical tracking takes over when
it returns; no optical skeleton is fabricated from trigger closure. A released
controller grip means hold, not automatic webcam takeover. The explicit
`G1_OPTICAL_HANDS_ONLY=1` option disables controller manipulation and permits the
existing calibrated webcam fallback while a device remains controller-classified.
These fallback capabilities require deliberate manual experimental activation; the
current launchers disable webcam input even when optical-only manipulation is selected.
See [operator setup](../HUMANOID_VR_CONTROL.md#webcam-alongside-quest) and the
[validation record](validation/isaac-sim-6.1.md) for tested scope.

Protocol v1 fields are `schema_version`, `sender_session`, `seq`,
`capture_monotonic_ns`, `calibration_request`, `pose_world`, `pose_image`,
`hand_world`, `hand_image`, and `hand_confidence`. Pose rows have 33 entries of
`[x,y,z,visibility,presence]`; hand rows have 21 entries of `[x,y,z]` keyed by
anatomical `left`/`right`. Detailed hands are associated with pose wrists to avoid
mirror-based left/right swaps. The hand model's handedness score is not a direct
measurement of tracking accuracy. Camera images are not transmitted by this tool.
MediaPipe omits separate metacarpal bases for the four fingers. The adapter derives
a virtual reference bone in the measured palm plane, so lateral spread of a flat
open hand is not counted as knuckle flexion. Distal bends and thumb landmarks
retain their existing angle mapping.
All timestamps assume the sender and receiver run on the same computer. Both use
`perf_counter_ns()` (Windows QueryPerformanceCounter), since `monotonic_ns()` can
use different clock origins across the sender and Isaac Python versions.

## Controls

The table below describes stationary/legacy controls. Grip/trigger manipulation and
optical hands also work in controller walking, where left-stick forward/backward/strafe
and right-stick turning replace stationary locomotion suppression. Face buttons and
stick clicks remain unused in walking; keyboard B still recenters heading.

| Input | Result |
|---|---|
| Left/right XR stick axes; X/A buttons | No robot movement in the default stationary mode |
| Grip held | Move/rotate that arm relative to its clutch calibration |
| Trigger | Close all fingers and oppose the thumb; open to release physical contact |
| B (keyboard or right controller) | Recenter horizontal heading without changing physical pitch/roll |
| Y | Force both hands open; open/release the input before closure is rearmed |
| Left stick click | Retain the robot-mounted camera mode |
| Optical hand tracking | Anatomical palm position/orientation, five independent finger curls, and separate thumb opposition; no controller clutch |
| Keyboard arrows/numpad movement keys; gamepad locomotion inputs | No robot movement in the default stationary mode |
| Head rotation | Rotate the mounted view left/right, in pitch, and in roll; no base movement |
| Head translation/step-in-place | No camera translation or base movement; existing gaze and recording remain available |

Triggers on XR controllers never command walking. Grip alone leaves fingers open.
Releasing grip ends tracking and holds that arm's measured pose in physical mode;
valid trigger input continues to control the fingers independently, so grip release
alone does not command a physical drop.
Stationary mode also prevents stick clicks from selecting a room-scale camera mode.

Hand target spheres are hidden by default so the robot fingers and object contacts
remain visible. Target transforms continue updating for arm control. For debugging,
set `EX._arm_rig_markers_enabled = True` through `tools/kit_exec.py`; active hand
targets then show a small sphere with a 1 cm radius. This setting
only affects hand target visualization, not the gaze ray or gaze hit marker.

### Pick up with VR controllers

These grip/trigger steps also apply in controller walking unless
`G1_OPTICAL_HANDS_ONLY=1` is explicitly selected. In walking, use **keyboard B** for
heading recentering and release the trigger to open; controller B/Y remain unused.

1. Set your current head heading as forward with **B**. Keep the trigger
   released, then **hold the grip** on the hand you want to move.
2. Move that controller forward, upward, or sideways and rotate it to guide the robot
   palm and wrist. Start with the 6 cm cube or cylinder on the near edge of the
   **centered practice table**. Watch the actual robot fingers enclose the object;
   neither the target marker nor a highlight establishes contact.
3. Keep holding grip and gradually **squeeze the index trigger** to close all fingers.
   The controller fallback couples thumb opposition to thumb curl. Opposing contacts
   must support the object; crossing a trigger threshold does not attach it.
4. **Keep grip and trigger held** while moving the controller to lift or carry.
5. Release the trigger to open. **Y** forces both hands open until you release the
   trigger and then squeeze again. Letting go of grip ends arm tracking while valid
   trigger input continues to control closure.

### Pick up with optical hand tracking

1. Use your runtime's hand-tracking mode so it supplies wrist/palm and finger skeleton
   poses. Present an **open hand** in the tracking cameras' view.
   The launcher and extension enable Kit's OpenXR Hand Tracking component; restart
   an already-running XR session once after updating. Keep the existing OpenXR runtime.
   No fist or pinch is needed to begin tracking. Position-valid wrist and middle
   knuckle landmarks drive the arm; valid finger landmarks independently drive the
   fingers. Hand-joint orientation flags are not required for this anatomical mapping.
2. Reach and rotate your hand to guide the corresponding robot palm toward an object
   on the near edge of the centered practice table. Optical tracking does not require
   a grip button or controller clutch.
3. **Oppose your thumb and pinch or wrap around the object**. For the cube, bring
   the thumb pad across to face the index pad on the object's opposite side; a
   shallow edge pinch can slip. Watch the robot's actual thumb and finger contacts,
   then lift slowly. A fist beside the object does not
   pick it up, and physical mode has no mean-curl attachment threshold.
4. **Open your hand** to release. Missing finger input commands the fingers open.
   After Y, open before closing again. Tracking loss alone clears contact state and
   commands open; valid input can resume without a separate rearm gesture.

Each hand works independently. Physical pickup depends on contact and friction and
can slip even when the contact diagnostic reports support.
Move one finger at a time to control the corresponding robot finger; thumb opposition
is independent of thumb flexion. Inspire has six actuators: each finger's distal joints
are mechanically coupled, so individual human knuckles and finger splay cannot all be
reproduced independently. Controllers retain whole-hand trigger closure; individual
finger tracking uses the optical skeleton from the headset. Webcam landmarks are
display-only in current launches; its old finger fallback is disabled.
The stationary robot cannot walk to the distant side benches or lower its base to reach
the floor. Experimental gesture walking adds floor cartons and tall lightweight cartons
with handles at standing hand height. Stop before grasping a handle; oppose the thumb,
close the fingers around it, and lift through physical contact. No crouching or
floor-pickup controller is added. Legacy assisted mode retains its separate package layout.

## Source layout

Paths in this table are relative to
`source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/`.

| File | Responsibility |
|---|---|
| `interactive/humanoid/humanoid_example_extension.py` | Examples-browser registration and profile-specific UI instructions |
| `interactive/humanoid/humanoid_example.py` | Scene, input arbitration, arm/finger retargeting, pickup, camera, recorder, lifecycle |
| `interactive/humanoid/behavioral_csv_writer.py` | Bounded, ordered disk writes with immutable CSV batches and explicit failure retention |
| `interactive/humanoid/xr_pose.py` | Validated virtual-world poses and timestep-independent smoothing |
| `interactive/humanoid/eye_gaze_tracker.py` | Eye/head provenance, raycasts, latest sample, gaze visuals |
| `interactive/humanoid/material_highlights.py` | Preauthored nonphysical visual shells and shared gaze/hand color ownership; no live rigid-body material rebinding |
| `interactive/humanoid/grasp_controller.py` | Contact-limited finger targets and single-hand opposing-contact diagnostics |
| `interactive/humanoid/bimanual_grasp.py` | Cross-hand measured contacts, dwell and estimated friction capacity; no control forces |
| `interactive/humanoid/grasp_contacts.py` | Prepared PhysX hand-contact view, object identities, and contact health |
| `interactive/humanoid/arm_contact.py` | Projects arm commands against observed scenery-contact normals after smoothing |
| `interactive/humanoid/articulation_health.py` | Detects unusable measured joints before teleoperation and latches a pause |
| `interactive/humanoid/point_gaze_navigation.py` | `PointNavigation`: blue-ray dwell, locked destinations, route following, physical turns, and release; filename retained for compatibility |
| `interactive/humanoid/point_navigation_scene.py` | Blue visuals, real floor/object approaches, segment clearance, and turning-footprint queries |
| `interactive/humanoid/factory_route.py` | Incremental bounded floor-route search and collision-checked corner reduction |
| `interactive/humanoid/controller_locomotion.py` | Joystick deadzones, signs, diagonal cap and neutral/reconnect interlock |
| `interactive/humanoid/gaze_locomotion.py` | Immediate or delayed eye selection, hold-to-move and optional bounded steps/turns, preview, step settling and look-away interlock |
| `interactive/humanoid/gaze_controls_scene.py` | Camera-following button panels, dwell/command feedback and eye-ray hit geometry |
| `interactive/humanoid/comfort_camera.py` | Wall-time filtering of robot XY/yaw, idle hold, and fixed neutral eye-height calibration; no HMD filtering |
| `interactive/humanoid/tall_packages.py` | Three lightweight compound cartons with collision-open handles |
| `interactive/humanoid/package_shelves.py` | Twelve free handle-free cartons on three static two-level racks |
| `interactive/humanoid/operator_head.py` | Reversible render-head rotation and measured eye/neck anchors; no physics edits |
| `robots/g1.py` | Stationary world anchor, joint ownership, posture, optional walking actor/kinematic mode, reset |
| `robots/g1_decoupled_policy.py` | Paired balance/walk actors, 29-joint observation, 15 leg/waist actions, and 50 Hz inference |
| `robots/g1_foot_contacts.py` | Walking-only deployment foot contacts in a session layer, preserving mass properties |
| `robots/data/g1_decoupled_*.onnx` and `g1_decoupled_policy.json` | Pinned NVIDIA actors and deployment contract; provenance and model/source licenses are in the adjacent README |
| `robots/data/g1_unitree_motion.pt` | Older twelve-leg-joint TorchScript actor retained for legacy assisted policy mode |

Keep these files together when loading the extension. The project launcher selects
this checkout's complete package with `--ext-path` and its exact extension version;
copying individual files into the installed package can mix incompatible APIs.
Save the stage, restart the application, and load the example after Python edits;
rebuilding a scene alone does not replace already-imported classes.

The following entry points sit outside that extension package. Their paths are
relative to the repository root:

| File | Scope |
| --- | --- |
| [`tools/validate_gesture_walking_live.py`](../tools/validate_gesture_walking_live.py) | Host CLI for reproducible real-PhysX baseline and right-arm replays |
| [`tools/validate_controller_walking_live.py`](../tools/validate_controller_walking_live.py) | Bounded synthetic-stick physical replay: directions, release, loss, and neutral rearm |
| [`tools/validate_gaze_walking.py`](../tools/validate_gaze_walking.py) | Separate headless preview/movement checks in the real warehouse with synthetic eyes and camera captures |
| [`source/standalone_examples/api/isaacsim.robot.policy.examples/g1_grasp_env.py`](../source/standalone_examples/api/isaacsim.robot.policy.examples/g1_grasp_env.py) | Retained cloned-scene throughput benchmark; no action/reward interface, episode reset, or training loop |
| [`source/standalone_examples/api/isaacsim.robot.policy.examples/g1_standalone.py`](../source/standalone_examples/api/isaacsim.robot.policy.examples/g1_standalone.py) | Retained keyboard/scripted assisted walking and hand demo; separate from the current strict physical VR profile and recorder |

Both standalone files start their own Isaac Sim application at module scope;
importing them is not a passive way to reuse a helper.

## Runtime and data flow

The repository includes the upstream Isaac Sim platform as well as this custom
application. The main path is the project launcher, the modified
`isaacsim.robot.policy.examples` package, `HumanoidExampleExtension` registration,
and the `HumanoidExample` scene/controller. The standard launcher uses an installed
Isaac Sim runtime; it does not build or replace that runtime. NVIDIA's pretrained
balance/walk actors remain upstream assets executed by the G1 wrapper.

For an initialized robot, `on_physics_step()` creates a hand-input snapshot for one
callback, then `_advance_physics_step()` runs this sequence:

1. Check physics-view validity, measured articulation health, and walking balance.
2. Read the selected locomotion source, bound its velocity request, check unintended
   motion, and update the filtered base command.
3. Call `g1.forward()` to update the selected posture/locomotion controller. In
   physical decoupled mode, the learned actors own legs and waist while teleoperation
   owns the arms; finger drives are separate.
4. Update contact-limited fingers and arm IK targets. With explicit body PD, call
   `g1.apply_body_efforts()` after arm targets are final so the body receives one
   combined effort update for the next simulation step.
5. Update the mounted view and gaze, then collect the scheduled recording rows and
   image. The callback releases its hand snapshot before the next update.

Input, controller targets, and measured state are different data. A controller
stick or hand pose expresses operator input; clamping, IK, contact constraints,
and the policy determine the command sent to the articulation. Joint/object state
is the measured response. Recording currently captures only part of that action
path, as described below and in the [learning guide](../learning/README.md).

There are also three distinct clocks: physics time for dynamics, bounded wall time
for operator-response filters, and render updates for camera images. The nominal
100 Hz CSV and 10 Hz image schedules use simulation time; they do not promise those
rates per real second or a newly rendered image at every capture request.

Retained code is not necessarily an active input. `PointNavigation` runs only in
its explicit pointing profile; the older `LocomotionGesture` V-sign helper is not
wired into this callback. Head-bob gait is disabled. Standard launchers disable
webcam robot input, and the optional EEG host application records/previews signals
without sending anything into the robot's control loop.

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
Optical arm and finger geometry uses `_get_hand_joint_position()` instead: it accepts
position-valid landmarks even when the runtime cannot supply their orientation.
Raw recording poses retain their existing runtime frame and validity semantics.

## Arm and finger retargeting

On controller grip engagement, the example saves the tracked hand position and the
robot palm position in the robot's yaw-aligned body frame. Subsequent hand deltas
are scaled and bounded from those anchors. Translating or turning the robot therefore
does not itself move the target within the body frame. Releasing grip or switching
input source clears calibration. Optical hands use their current body-frame position.

`_get_optical_arm_pose()` places the optical target at the midpoint of `wrist` and
`middle_proximal`. The robot's corresponding point is the midpoint between its hand
base origin and middle proximal joint anchor. These stable palm points do not move
when fingertips curl. Missing or coincident wrist/middle landmarks count as arm
tracking loss; the controller never switches to a raw wrist/palm origin during an
occlusion. Missing transverse landmarks leave position tracking active but disable
the orientation request.

Optical orientation is **absolute anatomical palm alignment**. The wrist-to-middle
direction and index-to-little knuckle span define a proper frame for each hand, with
a side-adjusted outward palm normal. The robot frame comes from the Inspire proximal
joints' fixed `body0` anchors in the rigid hand base, rather than its initial pose or
moving fingertips. With row rotations:

```text
desired_robot_wrist_world = inverse(robot_palm_local_frame) * operator_palm_world_frame
```

A flat operator palm therefore requests a flat robot palm even if the robot starts
with its palm vertical. Reacquiring optical tracking does not preserve that initial
misalignment. Degenerate or unavailable landmarks never invent an orientation; an
asset without the required anatomical anchors receives no absolute orientation target.

Controller wrist orientation retains relative clutch calibration so the robot keeps
its starting wrist axes while following controller rotation:

```text
H = tracked_world_rotation * inverse(base_yaw_rotation)
desired_world = robot_hand_body_at_clutch * inverse(H_at_clutch) * H * base_yaw_rotation
```

The live solve uses all seven arm joints and gives palm position first priority.
A damped-least-squares position step is followed by a wrist-orientation correction
in the remaining position nullspace, computed with an SVD projector. An unreachable
wrist orientation therefore cannot reverse the primary reach step. A small default
posture gain (`_arm_posture_gain = 0.1`) returns redundant joints toward their standing
defaults within the combined position/orientation nullspace. The primary reach step
consumes the available travel within each joint's correction cap, then solves the
remaining palm residual through joints whose travel is not exhausted. This bounded
loop uses at most eight solves and also handles joints merely close to a limit;
a tiny remaining budget no longer stalls the entire arm. Wrist and posture corrections use only the remaining
budget, each with a common scale that preserves its nullspace direction. They cannot
shrink the accepted primary step by forcing a rescale of the combined correction.

Orientation uses a bounded active-set solve when that common scale reaches a joint
bound. It retains the accepted position/orientation step, removes the saturated
joints from the next orientation correction, recomputes the exact position-nullspace
projector over available joints, and solves the remaining angular residual. This
prevents one blocked wrist from cancelling all otherwise feasible alignment. At most
eight secondary solves run for the seven-joint arm; an unconstrained correction still
uses one. Every added correction preserves palm motion to first order and fits the
remaining joint travel and step cap. Posture, measured-pose lead, slew and contact
processing retain their existing behavior. This applies wherever the live arm solver
runs; the degraded position fallback has no equivalent orientation solve. It remains
local IK: an infeasible angle, singular pose or different elbow branch can leave error.

The existing solver already permits nearly full free-space extension. In
`_compat61/arm-extension-baseline.json`, the palm reached **0.4887 m forward of the
robot base** and the two elbows retained only about **0.28° / 0.34°** bend. Both arms retracted without
a joint-health fault. No IK gains, limb dimensions, or joint limits were changed for
the current reach request; the tabletop objects were repositioned closer instead.
These results are bounded by the actual G1 workspace and the tested trajectory.
They do not establish object pickup or make arbitrary wrist orientations reachable
at full extension.

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
under contact. Source changes and tracking loss clear hand-target calibration;
reacquisition starts from measured joints. In physical mode, deactivation latches the
measured arm joint pose and holds it until valid arm input returns or the scene is
reset. Controller grip release uses the same hold. This replaces an automatic return
to rest that could pull the hand through the table between trials. Valid finger input
remains independent, while missing finger input commands open. Legacy assisted mode
retains its existing return-to-rest behavior.

In physical mode, `_limit_arm_targets_at_contacts()` runs **after** smoothing. It
uses measured hand contacts against non-graspable external colliders with at least
0.02 N normal force, including predictive contacts at positive separation. The arm
limit has no 2 mm separation cutoff: a loaded predictive contact already constrains
the hand before visible intersection. The separate grasp-support diagnostic retains
its 2 mm threshold. For each contact point the arm limit forms
`n · J_point`, including the wrist rotation's contribution at that point, and
projects the requested joint displacement onto non-inward half-spaces and joint
bounds. This prevents a smoothed target behind the table from continuing to load
the hand. Withdrawal and motion tangent to the active surfaces remain possible;
the projected command also replaces the smoothing cache. The robot's own links are
excluded. Graspable props normally remain excluded, with one directional exception
when the physical decoupled walking policy uses explicit body efforts: eligible
contacts with force **on the hand** having world-normal Z **>= 0.5** and normal force
**>= 0.02 N** enter the same existing contact-point Jacobian projection. This prevents
continued inward arm commands when pushing a package downward, including into a shelf.
Horizontal side contacts remain free for palm squeezing; tangential motion and upward
withdrawal remain available. This adds no force controller, shelf-support classifier,
attachment or force target. It can also yield at qualifying top contacts with a free
or tilted carton. Finger limits/gains, policy weights and gaze are unchanged.
Other external colliders
are treated as obstacles even if dynamic; the filter does not require a static body.
Finger closure still handles its own contact limits.

The pure `project_contact_step()` helper uses bounded Dykstra projections. If its
32-iteration budget does not produce a converged feasible result, it returns zero
motion. Missing contact data holds commanded arms at their measured joints while
finger control commands open; invalid measured state latches a physics fault.
This is a local first-order response to measured contact, not a collision-free
planner or a guarantee against every collision between physics ticks. It leaves
the accepted anatomical palm mapping, free-space IK priorities, gaze, camera mount,
and hidden hand target spheres unchanged.

Hand-target, joint-target, and finger filters use `_teleop_input_dt`, derived from
elapsed wall time because XR input arrives in real time. At the default physics rate,
the filter interval is bounded between one physics step and 50 ms. A nonpositive,
nonfinite, or greater-than-250-ms gap starts a fresh interval using the physics step;
it does not permit a large catch-up jump. The 2.5 rad/s joint command slew limit still
uses **physics time**, with the separate 0.15 rad tracking-error bound unchanged.
Responsive input filtering does not make an overloaded simulation run in real time.
If the live Jacobian is unavailable, the existing measured four-joint local map remains
a degraded position fallback; it cannot reproduce the full wrist-orientation solve.

Optical finger curls sum adjacent bone bends along each complete digit chain. This
avoids a deep fist wrapping past 180 degrees and reopening the robot finger. Joint
positions need the OpenXR position-valid flag; missing orientation is acceptable for
this position-only calculation. Thumb opposition is measured in the hand's own palm
plane and has a separate target and recorded `thumb_yaw` column.
An open digit has zero curl and is still a valid measurement. Finger updates run
independently of the arm's active target.
Missing finger roles relax open and remain in the whole-hand averaging denominator, so one tracked thumb cannot
look like a fully closed hand in legacy assisted mode. Controller trigger travel
drives all roles, with thumb-opposition fallback from thumb curl. Only
independent finger driver joints are commanded; mimic joints follow their coupling.
Curls update before the arm/pickup decision in each physics tick. Kit can label a
Touch device `hand` before a skeleton arrives. The input classifier requires skeletal
landmark names for optical control; a valid grip pose plus real trigger/thumbstick
actions preserves controller control when only interaction poses exist. A partially
present skeleton stays optical, so missing joints cannot activate stale trigger input.

## Robot-mounted camera

The default `robot_head` mode enables `_head_camera_comfort_enabled`. The view
follows a level robot frame with a calibrated neutral optical height. HMD left/right
turns, pitch and roll rotate both view orientation and the neck-to-eye offset around
the measured neck anchor in that stabilized frame.
Physical HMD translation is discarded: walking, leaning, or crouching in the room
does not translate the robot view. Torso roll/pitch and gait bob are excluded.

G1 has no actuated neck joint. `operator_head.py` edits only
`torso_link/visuals/head_link`; it refuses a render subtree containing physics.
The torso, head collision shapes, masses, policy joints, and gaze implementation
remain unchanged. The visual edit and render-instance setting are restored on
cleanup. This is a visual head feature, not a physical neck controller.

Both comfort and rigid viewing now call `RobotHeadVisual.apply()` to rotate the
visible head around its measured neck pivot, whose position stays fixed relative
to the torso. The helper converts the existing camera-local HMD orientation delta
through the mount rotation; it does not read XR or gaze itself. Camera filtering
no longer translates the head away from the body. The comfort camera retains its
robot-motion stabilization and applies the HMD-driven neck-to-eye rotation in that
stable frame, without inheriting physical torso bob, pitch or roll. Head colliders,
masses and policy joints remain
unchanged. `apply_camera_attached()` and the reversible `set_first_person_hidden()`
helper remain for legacy diagnostics; production teleoperation uses neither and
keeps the head visible. The intermediate fixed-eye version passed native neck-pivot
checks but looked inside the head when looking down. It is superseded by the eye
arc. The earlier fixed-5-cm combination passed a fresh 25-pose native XR
replay and visual inspection of ten desktop first-person views, including 85° down
and 180° yaw. The visible neck remained fixed, gaze and all 145 inspected collision
opinions were preserved, and no callback or health fault occurred. Native per-eye
visibility and perceived headset comfort were not tested.

The neutral eye sits 1.5 cm beyond the render head's front bound, centered laterally, at
the authored `torso_link/d435_link` sensor origin's height. Its torso-local Z is
0.429870 m, about **4.6 cm above** the previous render-bounds midpoint. The sensor's
fixed downward pitch is not copied. Missing optical markers retain the measured
render-bound height; the diagnostic `_head_camera_eye_height_source` identifies
the source. Assets without a safe render subtree retain the original mount fallback.

`_get_robot_head_camera_pose()` still reads the real torso tensor, including with
Fabric enabled, for measured height and cosmetic head motion.
`_get_comfort_camera_mount()` uses the root's XY and yaw. Its
`ComfortCameraFilter` uses monotonic wall time, with exponential time constants
of **0.18 s** for XY, **0.25 s** for yaw, and a **0.10 s** maximum update interval.
It follows stopping coast for one second, then holds idle changes within **4 cm**
and **0.10 rad** independently. Larger measured movement remains filtered;
an active movement command immediately removes the idle hold. Repeated or backward
timestamps return the previous filtered frame, never a raw pose.

After at least 0.5 simulated seconds of walking-actor settling, a one-time optical
height calibration transitions with a **0.30 s** time constant. Later body-Z changes
are ignored; HMD-driven eye rotation may still change the resulting camera height.
Stationary mode calibrates when physics becomes ready. Reset and mount
recreation discard the filter and calibration. This assumes the level factory floor;
ramps, stairs, and intentional squat-height changes are unsupported.
`/World/G1_HeadCamera` remains the viewport and recording camera.

`_head_camera_near_clip_m = 0.05` sets the baseline near plane for the robot first-person
USD camera and the active native XR profile's `render/nearPlane` setting. Native XR
does not inherit USD clipping, so both are configured. Measured visible-head intrusion
increases only that near plane to the head bounds plus a 15 mm margin, quantized in
5 mm steps, and it returns toward baseline when clear. The maximum is 0.30 m, with an
explicit diagnostic if the required clearance exceeds it; arbitrary falls are not
guaranteed clear. Very near hands/objects can clip during sway or travel. Camera/HMD/
gaze matrices, head attachment and visibility, external-camera clipping and physics
remain unchanged. This is adaptive clipping, not a camera-position adjustment.
Live visual acceptance of this addition is pending; the earlier pose/render passes
above concern fixed clipping.
Mode exit and cleanup restore the exact prior XR setting, including an absent
opinion, using the active profile's resolved persistent path.

The first valid physical HMD heading defines neutral forward. The reference is
yaw-only: project local -Z forward onto the physical XZ floor plane and preserve
physical +Y as up. Startup pitch/roll must never become part of the reference;
otherwise a headset initialized while tilted renders a slanted world after it is
put on, and horizontal turns acquire false pitch/roll. **B** resets heading only,
including when pressed while looking up/down or tilting the head. A nearly vertical
startup view (horizontal forward length below 0.1) defers heading calibration;
after calibration, looking vertically remains fully tracked. Recenter requests
with an invalid or nearly vertical pose retain the current view; look ahead and
press **B** again. Keyboard **B** is available during bare-hand stationary and
walking. The right controller's B button applies only outside the joystick-only
walking profile, where all controller buttons are ignored. OpenXR and USD
cameras both use local -Z forward and +Y up. Missing or invalid head rotation holds
the last valid orientation while the stabilized body frame continues updating.
`_head_camera_operator_rotation_enabled` defaults to `True`; disabling it keeps
the neutral view. The camera reader prefers `get_raw_pose_desc`, requiring OpenXR
orientation and position VALID bits (`0x3`); TRACKED bits are not required. If that
API is absent it uses `get_pose_desc`, which the runtime may smooth. The installed
6.1 wrapper exposes `get_raw_pose_desc` but its native implementation lacks it;
an `AttributeError` naming that exact missing method also selects the validated
processed descriptor. Other failures and invalid descriptors never fall back to
stale matrix-only input. Older adapters
without either descriptor retain their nonidentity-matrix compatibility path.
Fresh HMD orientation is composed after robot filtering, without our own HMD
smoothing stage. This is a control-path property, not a measured headset
motion-to-photon latency claim. The visible head rotates around its fixed torso-local
neck pivot. The camera applies the corresponding eye arc in its stabilized frame;
HMD orientation, not room-scale translation or physical torso sway, drives this arc.

An application-frame subscription owns camera scheduling during Play and Pause;
ordinary physics callbacks do not submit competing comfort-view updates. The active
XR profile enables `overrideFloorAlignedAnchor` while this view is owned, so the
native headset display preserves the requested camera pitch and roll instead of
silently leveling them. Cleanup restores the exact previous setting, including an
absent profile override. **B** resets the operator-view reference; it does not
change eye-gaze calibration. Reset/clear/load discard the reference and stale body
handles. The left stick cannot enable room-scale motion in stationary mode.
Setting `_head_camera_comfort_enabled = False` explicitly retains the historical
rigid view for comparison. Its earlier live tests do not establish current comfort.

For Steam Link, enable hand tracking on the headset and switch from controllers to
bare hands. Valve documents that Steam Link forwards the skeleton through
`XR_EXT_hand_tracking`. Controller aim/grip poses alone are insufficient for per-finger
control. [Valve's hand-tracking announcement](https://store.steampowered.com/news/posts/?appgroupname=SteamVR&appids=250820&enddate=1728412551&feed=steam_community_announcements)

## Pickup and tracking loss

### Experimental physical contact (default)

The current physical scene contains six lightweight objects on the centered
practice table at `(0.46, 0.0)`. Its measured top is 0.92 m above the floor;
`_fit_work_surface_height` scales only Z and grounds the asset's measured bottom,
preserving the horizontal size and transforming collisions with the visual geometry.
Paths below are under `/World/G1_SampleBoxes/`:

| Object root | Color / shape | World dimensions | Mass | World center X, Y (m) |
|---|---|---|---:|---|
| `Box_00` | Blue cube | 6 × 6 × 6 cm | 60 g | 0.320, -0.160 |
| `Box_01` | Orange cylinder | 6 cm diameter × 10 cm high | 80 g | 0.320, 0.000 |
| `Box_02` | Green sphere | 7 cm diameter | 50 g | 0.320, 0.160 |
| `Box_03` | Purple tall thin cylinder | 4 cm diameter × 14 cm high | 60 g | 0.440, 0.160 |
| `Box_04` | Yellow flat rectangular block | 9 × 6 × 3 cm | 70 g | 0.440, 0.000 |
| `Box_05` | Red cone | 7 cm diameter × 10 cm high | 50 g | 0.440, -0.160 |

`G1TeleopRobot._author_stationary_arm_pose()` authors the fourteen arm joint
positions and zero velocities before creating the articulation. PhysX joint state
angles are degrees, converted from `DEFAULT_STANDING_POSE` radians. Existing arm
drive targets match those values. This prevents the asset's extended startup
hands from striking props before `initialize()` runs. It applies only to
stationary mode and preserves finger states, mimic constraints, gains, and limits.

All are dynamic bodies with collision and gravity. The original `Box_00`/`Box_01`
sizes and masses remain unchanged. Shape dimensions, extents, recording metadata,
and table placement must agree. The sphere, cube, and orange cylinder now form a
nearer row; the remaining shapes sit in a second row. Center height is still
derived from table height, half the shape height, and the existing spawn clearance.
This position refresh retains the actual robot geometry and joint limits, and has
no new grasp acceptance result.
Their grip behavior is not validated, and existing cube/cylinder results do not
establish it. The larger package layout remains available in assisted mode.
Earlier trials passed right-hand lift, two-second hold, and release, but these
successes were not consistently reproduced. With current 500 rad/s coupling, damping
ratio 1, and 0.18 rad lead, the final fresh-scene combined replay failed because the
cube slipped. The more opposed thumb retry is an isolated success, not robust
validation. Hardware handling, left-hand pickup, and robustness across other grasp
poses remain separate checks; assisted demonstrations do not establish physical
contact performance. These historical pickup trials used the earlier **4 cm cube**
and **4 cm diameter, 8 cm cylinder**, not the enlarged current objects.

`ContactReader` prepares existing hand rigid bodies before physics initialization and
reads their actual PhysX contacts once per physics callback. It preserves the identity
of each contacted object and distinguishes a valid empty read from unavailable or
invalid contact data. Contact forces are solver measurements; distance to a candidate
and a closed hand are not substituted for contact.

`ContactGraspController` allows free fingers to follow the input. Measured contacts
with scenery or other robot links also limit closure, as do predictive solver contacts
at positive separation; they are not silently discarded because they are unsuitable
for a grasp. At first contact, it stores a cap separately for each actuator and
contacted body: measured curl plus a **0.18 rad** drive lead converted using that
actuator's actual range. The stored ceiling is independent of the still-ramping
request. While that contact persists,
the command is bounded by the request, current measured curl plus lead, and the
retained ceiling. The ceiling stays fixed: a temporary measured retreat or partial
opening cannot permanently ratchet it down, while further closing cannot move it
up. This preserves bounded preload without chasing a rolling contact.
Thumb contact also limits thumb yaw, and non-target/table contacts also create caps.

A contact-report gap retains command caps for up to **0.04 simulated seconds**.
Missing opposing contacts clears `contact_supported` immediately; the gap allowance
does not retain or invent support evidence. A new contacted body establishes its own
cap. Opening always wins. Fully open input (all six actuator requests, including
thumb opposition, below 0.08), input loss, or reset clears retained caps. Straight
fingers with a closing thumb opposition still receive contact limiting. This works
through bounded position drives; it is not
a commanded contact force or measured operator pressure. Missing finger input,
unavailable contact data, or unavailable measured finger positions commands open.

Before Play, `_prepare_physical_hand_couplings` sets the existing hand mimic
constraints' `naturalFrequency` to **500 rad/s** and `dampingRatio` to **1**. This
provides finite, critically damped compliance. The asset's original soft values
allowed loaded thumb knuckles to bend backward while the proximal actuator tracked
its target. A subsequent hard `0/0` setting passed simple pickups but caused
whole-articulation instability at table contact, so it is no longer the default.
Gearing, offsets, reference joints, limits, and all drive settings stay unchanged;
passive joints remain undriven. The override is restricted to the Inspire hand
subtrees in physical mode. Assisted mode retains the asset-authored mechanism.
These simulation settings do not describe calibrated real-world tendon compliance.
The physical practice props also disable sleeping, which otherwise suppresses static
contact reports. Session metadata records `hand_mimic_coupling: compliant`,
`hand_mimic_natural_frequency_rad_s`, `hand_mimic_damping_ratio`, and the exact
`hand_mimic_joint_paths`. Earlier `hard` metadata identifies a different mechanism
and must remain distinct in validation or training data.

Support diagnostics separately require eligible-object contacts with separation at
most **0.002 m**, in addition to valid force/contact data. They require sustained
opposing contacts on the **same object**: thumb
against index/middle for a pinch, or thumb/palm against at least two distinct digits
for a wrap. Repeated points on one finger do not count as multiple fingers. The default
dwell is 0.06 simulated seconds. `contact_supported` records this evidence, not a
guarantee against slip; `contact_hand` identifies the supporting hand(s), and
`contact_health` records the reader status. Objects remain dynamic, with no attachment,
teleport, or kinematic conversion. The legacy `is_grabbed` attachment field stays false
in physical mode, and session metadata identifies the grasp mode and configuration.
`hand_tracking.csv` separates smoothed operator requests (`<side>_finger_<role>`)
from contact-limited commands (`<side>_finger_target_<role>`). Actual robot joint
positions remain in `behavior.csv`. Per-hand contact object, mode, and support fields
describe the measured contact evidence.

The controller **grip remains an arm clutch only**. Valid trigger input controls all
fingers, even with grip released. Optical input separately controls each finger and
thumb opposition. Open to release. **Y** is accepted only from a physical controller;
optical-device button aliases cannot trigger a drop. Y commands both hands open and
inhibits closure until the controller trigger is below 0.15, or all four tracked
fingers are below the optical release threshold of 0.35. Thumb flexion and opposition
do not veto an otherwise open optical hand. This uses complete, finite raw input
before smoothing; missing digits cannot masquerade as open fingers. The release
tick stays open and discards old smoothing history before control resumes.
A still-held trigger or closed optical hand cannot immediately close again.
Finger-input loss alone commands open
and resets contact state, without latching the Y release requirement. Valid input
can resume when it returns; an already-latched Y still requires opening. Reset discards contact dwell, cached
physics handles, targets, and release state. Arm reacquisition starts from measured
joints, with a fresh controller clutch or absolute optical anatomical alignment.

### Articulation-health pause and recovery

The latest sudden stops are genuine measured-state failures. The real session
reported a ring-joint speed of 108.058 rad/s, following an earlier thumb fault above
318 rad/s. A 20-cycle dynamic table test reproduced the issue on its first closing
phase at 9.8 simulated seconds and 100.868 rad/s (`_compat61/dynamic-table-before.json`).
Earlier fixed-pose table passes did not cover this closing/opening motion.

Physical mode now applies a **1e-4 kg·m² armature floor** before Play across all 24
Inspire hand revolute DOFs. It preserves larger authored values and respects existing
per-axis armature attributes, which take precedence in Isaac 6.1. No new per-axis
schema is introduced merely to set armature, and passive joints receive neither
independent drives nor new velocity caps. This artificial joint-space inertia reduces
abrupt acceleration of the light finger links; it is not calibrated motor/rotor data.
Contact constraints and the health guard remain active.

The same 20-cycle test passed with this floor: **29.36 simulated seconds**, peak
absolute joint speed **8.294 rad/s**, maximum waist deflection **0.015026 rad (0.861°)**,
zero root movement, no articulation fault/callback error, and clean withdrawal.
Evidence: `_compat61/dynamic-table-armature1e4.json`. Fresh production-scene readback
confirmed all 24 hand-joint armatures. A separate recorded-motion-derived path check
also passed: **27.4 simulated seconds**, peak joint speed **15.243 rad/s**, all **2,740**
contact reads healthy, no articulation fault/callback error, and zero root movement
(`_compat61/stop-fix-recorded-path.json`). That test feeds recorded measured arm-joint
positions through command smoothing and contact limits. It is not an exact replay
of the original optical landmarks or the full XR-to-IK path.

Metadata records `hand_joint_armature_floor_kg_m2`, actual effective authored values
in `hand_joint_armature_kg_m2_by_path`, and `hand_joint_armature_meaning`. Record both
the selected floor and per-joint values when comparing runs; existing larger values
are not lowered. Earlier recordings without these fields predate this stabilization.

Each initialized physics callback validates measured joint positions, velocities,
and authored limits before feeding the state to teleoperation, camera, or recording
updates. Nonfinite or unreadable state, an absolute position above 20 rad, speed
above 100 rad/s, or a limit overshoot above 0.35 rad latches an articulation fault
and pauses the timeline. The log identifies the joint or failed state read. These
are divergence bounds, not normal control limits or real robot specifications.

The walking motion guard checks measured base velocity and foot support motion.
After 1.5 simulated seconds of zero-command braking, it requires 0.4 s of continuous
root scalar overspeed (XY above 0.10 m/s or absolute yaw above 0.25 rad/s), plus one
matching residual-motion channel:

| Channel | Root net motion over 0.4 s | Support motion over 0.4 s | Support excursion from stationary reference |
|---|---|---|---|
| Translation | Time-weighted XY speed > 0.10 m/s | Foot-centroid speed > 0.10 m/s | > 0.15 m |
| Turning | Time-weighted signed yaw magnitude > 0.25 rad/s | Mean sole-heading rate > 0.25 rad/s | > 0.35 rad |

Support heading averages the soles' horizontal forward directions from their measured
orientations. It does not use the line between the ankles, which can rotate during
a single corrective step without the robot turning. The stationary reference is
captured at the end of braking and retained through settling and hand-source changes;
new movement commands or episode/lifecycle invalidation clear it. These excursions
are **minimum evidence requirements, not hard travel bounds**. Slow or intermittent
motion can remain below the speed/dwell prerequisites despite travelling farther.

The reader uses native read-only views of existing ankle bodies, without authoring
USD or physics state. Invalid support measurements fail closed. No grasp or
input-source state disables the guard. Reversing sway cancels in the net calculation,
and shifting the pelvis over planted feet does not produce support travel.

The separate fast guard remains unconditional: root speed above 0.8 m/s or yaw above
1.2 rad/s for 0.25 s pauses the simulation, even if the support reader fails. Fall and
joint-state guards are unchanged. The preceding stance/contact source passed **881 tests
in 12.301 seconds**. The earlier 871-test suite and physical backward/settle passes
cover the intermediate foot-support guard. Fresh native acceptance of the final
contact-yield change passed after a physical shelf approach: 1,024 actual inward
requests were projected during a carton press/withdrawal replay, with no health fault
or callback error and clear withdrawal. This used synthetic optical landmarks and
does not qualify a complete real-hand lower-and-rotate placement.
The historical incident replay reconstructed feet and velocities from recorded poses;
it was not a live manipulation test. See the
[validation record](validation/isaac-sim-6.1.md#balance-sway-and-remaining-pause-limitation).

**Reset or reload the scene after a fault.** Pressing Play alone pauses again because
the fault latch remains set. Reset clears the latch, cached limits, contact handles,
and input targets while restoring the robot; a scene reload is appropriate when
changing coupling settings. Explicit Reset also clears warning deduplication, so
the same fault in a new attempt is reported again. Pause/Play retains that history
and does not bypass a latched fault. The guard contains a failure rather than repairing
already divergent state in place. Reload cleanup drops stale camera/body handles
before the stage is replaced; it does not change the camera mount or gaze behavior.

### Legacy assisted mode

For comparison with older recordings and attachment replays, set
`EX._grasp_mode = "assisted"` **before loading/rebuilding the example**. There is no UI
selector. Do not change this field during Play: physical contact preparation belongs
to scene setup. This mode uses distance-gated fixed-joint assistance, which can hold
objects the fingers alone could not. Keep its mode label in recordings and comparisons.

In assisted mode, an active hand searches from the measured rigid palm centre: the
live wrist-link pose plus its rotated, asset-derived palm offset. Candidate selection uses the
same working point as arm IK, rather than the wrist origin or the desired XR marker.
Assets without the Inspire anchors retain the earlier geometric offset fallback.
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

Assisted controller pickup uses 0.60/0.35 close/open hysteresis; optical hands use mean curl
0.55/0.35. Y/drop and tracking loss while holding set a release requirement: an unchanged
closed hand/held trigger cannot reattach immediately. Releasing grip, disabling tracking,
losing a usable hand pose, clearing, or resetting removes the constraint and clears
targets and cached physics handles. Finger tracking loss opens the driven fingers.
Reconnection starts with a target slew from measured joints. Controller reconnection
starts fresh clutch calibration; optical orientation resumes absolute anatomical alignment.

### Shared highlights

Gaze and hand highlights share an anonymous session sublayer prepared during scene
setup, before physics creates tensor views. Grab selection has priority over gaze on
the same object. Each package gets visual-only copies of its visible render geometry,
expanded 0.5% about each shape's bounds center to avoid depth flicker. Copies retain
geometry attributes and baked body-relative transforms; they copy no physics APIs,
mass, source references or source relationships. The shells inherit the body's
Fabric motion without runtime transform updates.

Each shell has a permanently bound material outside the rigid body's subtree under
`/World/G1_HighlightMaterials`. Runtime selection changes only that shader's diffuse
color, emissive color, and opacity. Opacity 0/1 with threshold 0.5 hides/reveals the
shell; removing selection reveals the untouched original geometry and materials.
No binding or other property below a live rigid body is edited. A cache of the
effective winner avoids all USD writes when only ownership changes, such as gaze
leaving a carton that both hands still highlight green. Cleanup masks the shells
through the independent Sdf layer and tolerates an already-closed Kit stage.

The grip-related freeze was traced to the previous highlight implementation: changing
from one highlighted package to another removed and recreated a USD prim override.
During a physics callback this made PhysX reparse the referenced collision mesh and
invalidate its simulation tensor view. Subsequent robot pose/target calls then failed
with `Failed to get DOF position targets from backend`. The failure was reproduced
inside the running VR application by switching gaze highlights from Box_03 to Box_00.
Do not use `RemovePrim`, change applied schemas, or detach highlight layers to clear
selection while physics is running.

A separate 20 September failure remained even without prim resyncs: changing an
all-purpose binding reset a displaced carton to its stale authored USD transform
while Fabric held the simulated pose. The user's recording contains a 25.6 cm snap
on the same sample that gaze left the carton, immediately after opposed two-hand
contacts had lifted it approximately 13.6 cm above its settled height. The shared
helper's first repair used a `full` render-purpose binding, but a fresh physical
probe still reproduced a 20.05 cm snap. The current prebuilt-shell implementation
removes runtime body-subtree edits entirely. Gaze acquisition, ray direction, hit
selection, colors and priority rules remain unchanged. The fresh physical probe
passed: set/clear caused at most 1.23e-7 m movement while the body's authored pose
was 20.30 cm from its settled pose. Fresh rendered captures confirmed the original
carton, green selection, and restored original appearance; see the
[validation record](validation/isaac-sim-6.1.md#carton-reset-and-explicit-reset-investigation).

## Locomotion and lifecycle

The interactive example selects `self._g1_locomotion = "stationary"`. In this mode a
fixed world-to-pelvis joint anchors the base at its spawn pose, the robot holds its
standing posture, robot-link gravity is disabled, and no walking checkpoint is run.
Scene objects retain their normal gravity. Arms and fingers keep their
separate control paths. Base commands are suppressed at both the example and robot
wrapper, including keyboard, XR/gamepad sticks and buttons, and headset gait. The
world constraint also prevents arm motion and carried-object reactions from tipping
the free base. This is a supported manipulation setup, not a demonstration of balance.

### Gaze buttons and optical hands

`tools/launch_gaze_walking.bat` selects `G1_LOCOMOTION_INPUT=gaze`, disables
pointing navigation, and sets `G1_OPTICAL_HANDS_ONLY=1` so both hands remain
available for manipulation. It defaults `G1_GAZE_MODE=hold` only when unset,
preserving an explicit `step` override. Its default `G1_GAZE_PREVIEW=1` is reset
on every invocation; only a first `--move` changes it to `0`. A first `--preview` explicitly
retains preview. Remaining raw arguments pass through to `launch_isaac_vr.bat`.
The base launcher still disables webcam robot input and loads this checkout's
complete extension. See the [operator procedure](../HUMANOID_VR_CONTROL.md#gaze-buttons-with-both-hands-free).

Preview uses the stationary world anchor with the same shelf/carton layout as
movement. It exercises gaze selection without any base movement. Movement uses
the existing physical GR00T scene, 200 Hz physics and 50 Hz actor schedule.
Changing between them requires exit/relaunch and LOAD because the base topology
differs. Preview does not validate dynamic balance, braking or carried loads.

Camera-following side panels keep the middle of the view available for the box.
Look off all buttons to arm, then look at Forward, Back, Turn left or Turn right.
In the default **hold** mode, the first valid button hit activates immediately,
including a quick glance. There is no dwell timer or ring at the default zero
delay. Movement continues while the
same valid, unoccluded button remains fixated. Looking away or at another button
ramps the requested velocity toward zero through the existing 0.06-second release
filter; eye loss, button occlusion, a callback gap, or session failure requests
zero immediately. Movement uses the existing 0.30-second attack filter; both
filter constants use wall time. Physical braking still takes time. A valid look
outside all buttons rearms the next selection without a separate gaze settling
wait. Selection does not transfer directly to a different button: a fresh
off-all-buttons look-away must rearm it. Invalid tracking requires a subsequent
valid look-away before activation. There is no fixed travel distance or turn angle in hold
mode. Both hands continue updating during fixation.

Optional **step** mode starts one measured movement after its default 0.7-second
dwell. A positive delay displays a progress ring. Its
accepted command continues after looking away or losing eye tracking, allowing
the operator to look back at the box; eye loss cancels unfinished dwell. Holding
the button does not repeat a step, and busy selection is ignored. Both modes
require look-away to rearm. XR profile/display shutdown and lifecycle/recenter
transitions cancel and disarm. Eye-driven commands require the combined gaze
device's current tracked pose; the visual/recording HMD-forward fallback never authorizes motion. Runtime
permissions and calibration still determine whether the device supplies usable
eye data. Application polling times are not hardware capture timestamps.

Defaults are `G1_GAZE_MODE=hold`, `G1_GAZE_DISTANCE_M=0.20`, `G1_GAZE_TURN_DEG=10`,
`G1_GAZE_FORWARD_SPEED=0.20`, `G1_GAZE_BACKWARD_SPEED=0.15`,
`G1_GAZE_YAW_SPEED=0.25`, and `G1_GAZE_DWELL_S=0` in hold mode. Step mode defaults
to `G1_GAZE_DWELL_S=0.7`; an explicit override is preserved in either mode. A
positive override restores delayed activation and the ring. Distance is in meters,
turn angle in degrees, translational speed in m/s, and yaw speed in rad/s.
Distance and turn angle apply only to `G1_GAZE_MODE=step`; hold mode continues
at the requested translation or yaw speed while the same button remains fixated.
Forward/reverse/yaw requests remain bounded by the direct-control caps
0.65/0.35 m/s and 0.40 rad/s. These are requested targets, not guaranteed physical
travel or stopping distances. The existing actor supports reverse movement;
loaded transport still requires a separate physical acceptance check.

Forward/Back translations in both gaze modes capture measured robot yaw at
activation and request corrective yaw toward that reference while active. The
default `G1_GAZE_HEADING_HOLD=1` enables this feedback; `0` disables it. The
proportional gain `G1_GAZE_HEADING_GAIN=3.0` has units of 1/s (rad/s per radian of
heading error), with a fixed 0.5-degree deadband and a 0.15-second simulation-time
error filter. `G1_GAZE_HEADING_MAX_YAW_SPEED=0.15` bounds the correction in rad/s.
The resulting yaw request passes through the existing base-command filter and
walking controller. Explicit Turn left/right commands keep their existing speed
and behavior. Ending a hold request, including eye loss, stops generating heading
correction; an accepted step retains feedback until that step ends. Reset and a
new activation discard the old heading reference.

This is yaw feedback, not a world-pose constraint: it does not pin or teleport the
base or change the camera to hide veering. It does not regulate cross-track
position. Heading error and sideways displacement must be measured separately;
heading stability alone does not establish straight-line travel.

Setting `G1_GAZE_BACKWARD_SPEED=0` disables reverse selection. Dwell uses wall time;
step-mode progress and settling use simulation time, with measured planar pose and
speed determining completion rather than elapsed wall time alone. Hold mode does
not enter the additional pelvis-settling gate or its four-second timeout when
looking away. It delegates physical stopping faults to the unchanged support-aware
unintended-motion, balance/fall and runaway guards.

The reported automatic pause was the duplicate four-second pelvis-settling fault
during active arm movement: the movement command was near zero and the robot
remained upright. The existing support-aware motion guard remains responsible for persistent
unintended movement in hold mode.

In step mode, stopping ends
with exactly zero requested velocity, then waits for measured translation and
rotation to remain within the settling limits. Settling accepts either the
existing low instantaneous speeds or a one-second rolling pose window with
bounded sway, low fitted and endpoint drift, and bounded speed peaks; the
criterion must hold for 0.20 simulation seconds. Failure to settle within four
simulation seconds latches a fault. This accommodates limited balance sway
without treating continued travel as a stop. Independent balance and runaway
guards remain active. Gaze movement does not perform the route/turn-clearance
checks used by pointing navigation; the operator must leave room for the robot.

Inspect `EX.gaze_locomotion_status()` through `tools/kit_exec.py` for the current
selection/movement state. Gaze controls supply only base intent: the per-step
health/balance/unintended-motion checks, independent finger and arm updates, and
final explicit body-effort update retain their ownership. Any UI/log timestamp
describes application observation rather than proven device freshness. The
existing `gaze.csv` distinguishes eye tracking from HMD fallback; a ray hit alone
does not establish selection or human intent.

**Heading-feedback validation on 8 November 2026:** paired actual-scene runs with
feedback disabled and with gain 3.0/s passed all four actions. Each used a
three-second hold and five simulation seconds of observation after release,
with the physical guards active and no latched fault. Measurements below are at
the end of the translation hold, relative to its captured starting heading:

| Action | Heading error, feedback off | Heading error, gain 3.0/s | Sideways drift, feedback off | Sideways drift, gain 3.0/s |
| --- | --- | --- | --- | --- |
| Forward | 3.20° right | 1.31° right | 1.33 cm right | 0.73 cm right |
| Back | 2.31° right | 0.47° right | 3.63 cm right | 3.62 cm right |

Forward's largest sampled heading error with feedback was 2.30°. Heading drift
decreased in these runs; reverse sideways drift was essentially unchanged. The
gain-3.0 run retained 6,974 arm updates and 6,974 finger updates. These checks used
synthetic eyes and neutral arms, not hardware tracking, active-arm manipulation or
a carried payload. The final gain-3.0 default passed **999 offline regressions in
12.405 seconds**.

**Immediate-hold validation before heading feedback on 8 November 2026:** all **990 offline regressions
passed**, and the actual-scene anchored preview passed immediate activation for
all four controls with synthetic eyes. The physical run also passed all four
directions, holding each for three seconds and continuing beyond the former
0.20 m/10-degree goals. Back and both turns used normal look-away release;
Forward used simulated eye-tracking loss and immediately cleared the filtered
command. Every release was observed for five simulation seconds, beyond the old
four-second gate, with no latched fault and the independent guards still active.
The run retained 6,974 arm updates and 6,974 finger updates. These are real-actor
checks with synthetic eyes; they do not validate hardware gaze or payload grasping.

**Earlier delayed-hold validation on 8 November 2026:** all **981 offline regressions passed**.
The actual-scene anchored preview passed, and all four physical actions stayed
active under sustained synthetic fixation beyond the former 0.20 m/10-degree
goals. Back and both turns requested stopping on look-away; Forward used simulated
eye-tracking loss, immediately zeroed its filtered command, and then settled.
All four returned to idle with zero commands and no latched fault. The run
retained 4,194 arm updates and 4,194 finger updates.
That revision used a 0.7-second dwell and the additional hold-settling gate;
these results do not validate immediate activation or removal of that gate.

| Held action | Measured final travel after stopping |
| --- | --- |
| Back | 0.318 m backward |
| Turn left | 24.81 degrees left; 0.031 m planar drift |
| Turn right | 31.55 degrees right; 0.025 m planar drift |
| Forward, then eye loss | 0.258 m forward |

These are unloaded measurements from the real actor with synthetic tracked eye
descriptors and a simulated selection clock. The validator checks floor/corridor
or turning clearance before each action; that fixture check is not a gaze-mode
obstacle-avoidance feature. Measured settling permits bounded balance sway, not
an exactly motionless pelvis. Live headset usability, real optical grasping and
loaded transport remain separate acceptance checks.

**Earlier step-mode validation on 8 November 2026:** actual-scene anchored preview
passed all four selections with zero sampled base displacement. Ordinary 0.3-second glances did
not activate; full dwell fired once; sustained fixation did not repeat. The
physical sequence also passed all four actions and measured stopping, with
3,084 arm updates and 3,084 finger updates during the same run. It first backed
away from the front shelf to satisfy the conservative turn-clearance preflight.
No root/object teleports, policy changes, or physics-guard bypasses were used.
These results predate hold-to-move and do not validate that interaction.

| Request | Measured final result in that unloaded run |
| --- | --- |
| Back 0.20 m | 0.218 m backward |
| Turn left 10 degrees | 10.93 degrees left; 0.015 m planar drift |
| Turn right 10 degrees | 11.45 degrees right; 0.018 m planar drift |
| Forward 0.20 m | 0.202 m forward |

The validator uses synthetic tracked eye descriptors and an explicitly simulated
selection clock; it does not test live XR delivery, wall-time fixation quality,
headset readability, optical hand tracking, or frictional carrying. The real
warehouse camera renders were inspected separately. In that earlier revision,
all **965 offline regressions passed**, including production optical curl extraction
and two-arm IK updates during active gaze locomotion with controlled contact/actuator fixtures. That test
explicitly does not establish a physically supported grasp. Tests also cover strict
source/validity, XR exit, occlusion, one-shot/rearm behavior, UI failure stopping,
geometry, input ownership, and launcher argument handling.

To reproduce the bounded step checks, set `ISAAC_DIR` to the installed Isaac Sim 6.1
directory and run from this checkout with no other Isaac process open:

```powershell
$gazeCheck = @(
  'tools\validate_gaze_walking.py', '--output-dir', "$env:TEMP\gaze-validation",
  '--ext-path', "$PWD\source\extensions\isaacsim.robot.policy.examples",
  '--enable', 'isaacsim.robot.policy.examples-5.2.11',
  '--/app/vulkan=false', '--/metricsAssembler/changeListenerEnabled=false',
  '--/app/extensions/fsWatcherEnabled=0'
)
& "$env:ISAAC_DIR\python.bat" @gazeCheck --mode preview --control-mode step
& "$env:ISAAC_DIR\python.bat" @gazeCheck --mode move --control-mode step
```

Use `--control-mode hold` in the two commands to run the sustained-fixation,
look-away and simulated eye-loss checks instead.

Each invocation owns and closes its own headless process, disables behavioral
recording, and retains a unique JSON report and PNGs in the requested output
directory, including failed runs. A cold shader/asset cache may need a larger
`--load-timeout`; that deadline cannot interrupt a native blocking startup call.

### Controller sticks and optical hands

`tools/launch_controller_walking.bat` is the controller walking entry point.
`tools/launch_gesture_walking.bat` delegates to it as a compatibility alias.
Both set `G1_GESTURE_WALKING=1` to select the existing physical GR00T scene,
`G1_LOCOMOTION_INPUT=controller` and `G1_POINTING_NAVIGATION=0`. The launcher defaults
`G1_OPTICAL_HANDS_ONLY=0` only when unset, preserving an explicit `1` override for the
former optical-only manipulation behavior. The base VR launcher remains stationary by default.
These settings are read before scene construction; restart/reload to change them.

The controller-manipulation and adaptive-clipping update passed **888 offline tests
in 10.540 seconds** (`.webcam/controller-camera-final-tests.log`). This is application
regression coverage, not live controller/hand transport or headset visual acceptance.

`ControllerLocomotion` accepts only left X/Y and right X. Left Y becomes forward,
left X is negated for body-left, and right X is negated for yaw-left. A 0.15 axis
deadzone is removed and the remaining range rescaled; diagonal translation is
normalized to magnitude at most one. The physical command branch then applies
the actor's configured directional speed caps: 0.65 m/s forward, 0.35 m/s reverse,
0.30 m/s sideways and 0.40 rad/s yaw. At 0.575 axis deflection the corresponding
request is half its cap; full deflection reaches the cap. The mapping and safe limits
were already proportional and remain unchanged. Neutral returns exact zero requests;
physical braking and achieved speed depend on the actor and contacts.
Grips/triggers control arms/fingers independently; face buttons, stick clicks,
keyboard movement and pointing gestures cannot generate walking commands.
Keyboard B remains a separate camera-heading action. Blue visuals are hidden and
the pointing planner remains implemented but inactive; eye gaze is unchanged.

`_sync_controller_xr_tools()` temporarily sets the active XR profile's
`tools/layout` to a private `g1_controller_teleop__v1__…` marker encoding the prior
layout string or absence for restart recovery. Kit's default high-level teleport, fly,
rotate, grab, move, select and menu tools then have no bindings, so they cannot also
consume these controls. Raw OpenXR actions, HMD/eye/optical-hand components and GUI
layers remain enabled. The override belongs only to controller walking and stays
active during Pause. XR/profile exit and example cleanup restore the previous
layout opinion, including absence; a later unrelated edit is preserved. Reentry
captures and applies the override again. This does not change gaze logic or settings.

Both sticks must be neutral before enabling movement after startup, reset or
controller loss. A callback gap over 0.25 wall seconds also disarms input. Neutral
and invalid input return exact zero requests; actual stopping still requires
physical braking. `_read_tracked_controller_stick()` requires complete tracked
physical grip descriptors, actual axis components and finite values. On the current
mutually exclusive transport, optical-hand classification invalidates retained
controller actions. The installed XR
wrapper exposes no per-action activity timestamp: pose/action checks and callback
gap protection do not establish hardware freshness beyond what that runtime reports.

Arm/finger selection now also accepts valid controller poses/grips and index triggers.
Grip held moves the corresponding arm; release holds its measured posture. Trigger
closure remains independent, so it can stay held while the arm holds and sticks move
the robot. Valid optical input takes priority when the device returns to hand mode.
The current runtime switches between controller and optical input rather than supplying
both together. Trigger-derived curls remain identified as controller input, not measured
optical fingers. The webcam is now display-only and cannot supply arm/finger fallback;
a released controller grip still means hold. Explicit optical-only mode disables
controller manipulation without enabling webcam input. Webcam input cannot authorize
walking. Hardware concurrency and loaded movement remain
unvalidated.

XR hand devices, input classification and landmark snapshots are shared only within
one synchronous physics callback, then read again on the next callback. Losing a
source clears its cached sample and calibration. This lets the first valid optical
sample replace a controller or missing device without retaining an earlier negative
result. Regression tests cover this handoff, rejecting stale sticks after optical
tracking returns, and sharing a single bulk landmark read within each callback.
These tests validate application behavior; live Quest controller-to-hand recognition
latency still requires measurement with the headset.

This profile enables `G1TeleopRobot(arm_gravity_compensation=True)`. The fourteen
external arm joints use `clip(Kp*(target-q) - Kd*qdot + G_arm(q), -limit, limit)`.
The fresh PhysX model term supports the robot's arm weight while following or
holding a target; it is model-based feedforward, not measured contact-force feedback.
It does not estimate payload weight or grip pressure. The fifteen leg/waist motor
calculations, actor checkpoints, gains and effort limits are unchanged. The option defaults off
outside this profile, including stationary and retained pointing launches, and is
recorded in session metadata. The constructor requires physical locomotion to enable
it; application additionally requires the decoupled explicit-effort motor path.
Finger drives and robot gravity remain unchanged. Better arm support and orientation
feasibility do not establish a successful shelf lift or loaded transport.

`G1TeleopRobot._read_arm_gravity_compensation()` validates native DOF names and the
complete gravity tensor. A floating base has six leading root force/torque entries;
only the following named arm-joint entries are added. The installed 6.1 experimental
selected-DOF wrapper omits that offset, so the adapter reads the complete native
tensor directly. Invalid layout or nonfinite data uses the existing zero-effort,
latched-fault path. The indexing and sign follow the
[PhysX gravity-compensation contract](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html#gravity-compensation-force).

The controller scene retains six table props and four floor cartons and adds
`Box_10`–`Box_21` on three open racks at `(3.5, 2.8)`, `(3.5, -2.8)`, and
`(1.35, 0)` m. Deck-top pairs are `(0.68, 0.94)`, `(0.74, 1.02)`, and
`(0.78, 1.06)` m, with 1.60 m wide racks for outward hand clearance. Each carton has one free dynamic body and box collider, no handle
or attachment; tape, labels and barcodes are visual only. Masses span 0.10–0.25 kg.
Static/dynamic friction is 0.8/0.6, with zero restitution. Dimensions and loading
clearances are authored in `package_shelves.py`; shelf pickup remains a live test.

`BimanualGraspMonitor` recognizes opposing contacts across both hands on the same
registered object. Both valid hands need at least one finger curl of 0.08, normals
must oppose below a -0.5 dot product, points must be at least 8 mm apart, and support
requires 0.06 simulation seconds of current contact evidence. Invalid input or
contact health immediately resets dwell. Each hand can have no independent pinch
while the aggregate bimanual contact evidence is valid; the logs distinguish these.

For measured normal force `N` and its unit normal `n` pointing **onto the hand**,
the object receives vertical normal force `-N*n_z`. The estimated maximum upward
friction is `mu*N*sqrt(1-n_z²)`. Opposite vertical box faces reduce this to
`mu*(N_left+N_right) >= mass*g`; the default estimate uses the configured dynamic
coefficient 0.6 and the cached authored object mass. Unknown mass leaves comparison
unavailable. Normal forces are newtons, not pressure. Capacity exceeding weight
ignores torque balance, accelerations and slip and does not prove a successful lift.
The helper never closes fingers, changes effort, adds force or creates adhesion.

`tools/validate_controller_walking_live.py` tests the production input/actor path
with synthetic sticks, retaining physics guards, real body motion and original
hand readers. It includes signed forward/reverse/strafe/yaw progress, neutral start,
release, controller loss and rearming, then pauses and restores overrides. It does
not test real controller/optical concurrency or establish loaded transport. Record
actual results separately; no live pass is implied by adding this validator.

`tools/validate_bimanual_box_live.py` replays two optical skeletons through the real
IK, finger drives and PhysX contacts. It requires a reachable initial scene spawn.
The guarded approach first lifts the open hands slightly, spreads them before
withdrawing, then raises and reaches forward at carton height while still spread.
The palms narrow around the box only after the forward reach has settled.
Collision/position/orientation checks abort a blocked approach before closing.
It never attaches or teleports the carton or robot. The latest spread-before-withdraw
replay completed its open approach without recorded external hand contact, but
curling while moving inward caught both fingers on the rear face and failed the
lift. The validator now keeps fingers open through palm placement and squeeze,
then requires 0.06 simulation seconds of current opposing **palm** contact before
closure. With requested palm positions held fixed, it curls the fingers, settles,
and requires another 0.06 seconds of current opposing **side** contact before lift.
The contacts must align with the measured carton's opposite lateral faces;
same-direction rear hooks and expired earlier dwell do not qualify. Default
finger curl is 0.5, thumb curl 0, and opposition 0. A live grip-only trial of this
palm-first revision failed the current palm-contact gate and stopped before
curling or lifting; the initial shelf approach was excluded from that trial.
Thumb-only contact at a top edge also failed an
earlier lift despite estimated friction capacity exceeding weight; inspect contact
roles and sustained box motion, not just a support label.
`--clearance-only` stops after a free-space arm diagnostic and explicitly does not
validate a grasp. Each phase records arm targets, measured joints, contact forces,
box motion and gravity diagnostics; input ownership is restored and physics paused.
The wall-time budget accepts 30–900 seconds (default 300, maximum `--timeout 900`);
this does not lengthen the
required simulation dwell or relax any lift/contact criterion. Synthetic skeleton
generation and diagnostics add overhead, so its wall rate is not headset tracking FPS.

### Experimental Quest gesture walking

The following is the retained pointing implementation and its historical evidence.
It is disabled by the current walking launchers. Explicit development activation:

```powershell
$env:G1_GESTURE_WALKING = '1'
$env:G1_LOCOMOTION_INPUT = 'pointing'
$env:G1_POINTING_NAVIGATION = '1'
.\tools\launch_isaac_vr.bat
```

Run the command above from the repository root, then LOAD a fresh
Humanoid scene. `G1_GESTURE_WALKING=1` is read before example construction
and inherits the normal Quest/gaze settings and disabled webcam receiver. It selects
policy locomotion with strict physical behavior. Default launches remain stationary.
Changing the environment variable after loading cannot safely change articulation
anchoring; restart and rebuild the scene when changing modes.

This explicitly selected profile uses `PointNavigation` in the compatibility-named
`point_gaze_navigation.py` module. Only a complete left **Quest optical** skeleton
can authorize navigation:
index extended, middle/ring/little curled, thumb ignored. Open the hand before the
first selection and after Stop, Pause, Reset, arrival, blocking, or tracking loss.
Missing digit observations do not count as release. Both arms and fingers continue
through ordinary tracking, smoothing, and contact limits; navigation never reserves
the left arm.

Navigation requires `POSITION_VALID | POSITION_TRACKED` for the wrist and all
four gesture-finger chains, rejecting inferred poses after occlusion. Its allowed
callback interval adapts to previously observed render cadence, with a 0.75 s
absolute cap. An unexpectedly long interval still stops first; a valid interval
below that cap can inform the next attempt only after a measured hand release.
This supports slow render frames without resuming a held point after interruption.

The blue ray uses the virtual-world index-tip position and an adaptive angular
filter on the index direction. Deliberate angular motion increases responsiveness;
small changes receive stronger smoothing. Drawing and collision queries use the
same filtered ray, whose nearest non-robot collider is queried independently of
gaze. No synthetic floor intersection replaces a miss or bypasses an obstruction.
The hit marker is hidden when the ray misses. Walking uses only
the left-hand observation and the blue-ray destination. The existing eye tracker, red
ray, calibration, recording, and fallback remain unchanged; gaze direction and
availability do not gate selection or walking. Controller and webcam poses
cannot authorize navigation.

Acquisition requires the blue ray to hold a valid destination steadily for
**0.6 s**. Empty-floor hits select that floor point. A direct hit on a registered
practice object or floor carton searches for real, clear floor in front of it; the
blue ray still ends at the object, while a separate floor marker shows the approach
destination. Walls and tables themselves cannot become object-approach targets.
No red/blue agreement is required. The candidate must lie **0.5–50 m** from the
robot; the launcher enables selection in every heading and physical turning.
The collision ray extends **60 m** and stops at the nearest non-robot collider.
Elevated surfaces cannot become walking destinations. Blue shows pointing,
cyan shows acquisition, and green shows the locked
destination. Once acquired, that world-space destination stays fixed while the
left pointing pose remains held. Gaze is independent throughout selection and walking.

The dwell drift allowance is `clamp(0.04 * initial_distance_m, 0.18, 0.30)` meters.
Its anchor is the first valid hit and both anchor and allowance remain fixed for
that attempt. A drifting ray cannot drag the anchor or enlarge its own tolerance;
leaving the allowance restarts the dwell. The selected destination is the current
validated floor hit or object-approach point when acquisition completes.

The robot receives bounded forward/yaw commands toward the target, without reverse
or strafe. Heading error above **0.55 rad** starts a physical stepping turn;
forward motion resumes below **0.25 rad**, preventing rapid switching between
walking and turning. This phase requests a constant signed **0.40 rad/s** with
zero forward/lateral command. The learned actor can drift while stepping; no
root pose is pinned. Curved forward steering retains its **0.20 rad/s** cap.
`max_requested_yaw_speed` exposes the larger stepping-turn limit to the downstream
clamp without raising the curved-steering limit. Every meaningful yaw request
requires fresh full-turn clearance from the measured robot position.
A progress watchdog requires **0.05 rad** of new best heading improvement within
**15 wall seconds**. It stops a nonresponsive turn, and clears on turn exit,
release, reset, route-corner change, or a latched stop. Repeated oscillation around
the same heading cannot renew it. These timings accommodate slow simulation;
the new stepping-turn profile passed a synthetic 180° approach replay, with
0.493 m maximum initial-turn drift. This is not a fixed-position pivot result.
Navigation requests a stop within **0.30 m** of the final destination.
Opening the pointing hand, invalid hand observations, or a blocked route immediately
clears commands and requires release before reacquisition. Actual body braking is
a physical response that must be measured separately.

`PointNavigationScene` performs a conservative direct-route `overlap_box` query:
half-width **0.38 m**, height **1.5 m**, bottom **0.06 m** above ground, with endpoint
padding. The queried center route ends 0.20 m before the target: the 0.30 m
arrival radius retains 0.10 m of braking reserve, plus the existing 0.38 m body
clearance. Object approach tries seven standoffs from 0.30 to 1.20 m, downward
raycasts each candidate onto real floor, and checks the full corridor without
ignoring the clicked object. With routing enabled, a candidate with sufficient
turning clearance can instead be handed to the detour planner when the direct
corridor is blocked. Candidate searches may be reused for at most 50 ms
only within 2 cm of the same hit and base position; the final corridor is queried
fresh every navigation callback. Intermediate route corners receive no stopping
shortfall. Turning clearance encloses every yaw angle of the body footprint, with
half-width `0.38 * sqrt(2)` m. Robot colliders are excluded; other or undecodable
colliders block motion. Invalid coordinates and failed queries also block motion.

Detour planning uses a wider `is_route_segment_clear` query throughout each edge:
`0.38 * sqrt(2) * (abs(cos(yaw)) + abs(sin(yaw))) + 0.15`, approximately
**0.687–0.910 m half-width**. Runtime turning checks use a world-axis square;
projecting that square into the rotated corridor encloses its corners even on
diagonal routes. The additional 0.15 m allows tracking error. `is_route_corner_clear` reserves
`0.38 * sqrt(2) + 0.50`, approximately **1.037 m half-width**, at the start and
actual intermediate corners, based on the observed 0.493 m stepping-turn drift.
The search includes incoming direction in each state; changing direction needs
the larger corner reserve, while straight passage does not. Final goals do not
require an unused extra stepping turn. String pulling rechecks both resulting
segments and corners. Direct close-object approaches and all current-pose runtime
collision/health guards retain their existing behavior. These wider planning
margins reject some narrow routes and are not a guarantee against larger drift.
The narrower route and a later diagonal route without square projection both
stopped safely at a table. The projected planner subsequently passed a physical
detour to a destination 8.53 m away: two waypoints, 2076 scene queries, 45 expanded
nodes, and 0.299 m remaining at arrival. This synthetic-input replay validates
that route, not every factory aisle or loaded transport; see the
[20 September validation](validation/isaac-sim-6.1.md#comfort-camera-factory-routes-and-tall-cartons--20-september-2026).

When a locked destination has no clear direct route, `FactoryRoutePlanner` searches
while commanding zero motion. It uses a **0.6 m grid**, up to **6,000 expanded nodes**,
and a **12 m margin** outside the start/goal rectangle. Production advances at most
**24 scene-query callbacks per 0.1 wall seconds**. Every edge and shortened route
segment requires collision clearance and real floor samples at intervals of at most
0.3 m, with support checks around the centerline; route corners also require
turning clearance. Search limits, missing floor, blocked starts/goals, and narrow
passages can fail the route. This does not guarantee access to every factory location.
Keep pointing while planning; release or tracking loss cancels it. Active route
segments are checked again against current obstacles. A new blockage stops and
requires a fresh selection; it does not silently continue along the cached route.
Left-hand grasp contacts prevent that hand from authorizing navigation.

The independent noncolliding blue cylinder, marker, and target disk are prepared
before physics starts. Runtime drawing updates attributes without recreating prims;
reset/clear drops cached handles.

The pretrained NVlabs GR00T-WholeBodyControl G1 Balance/Walk models output 15
leg/waist targets and observe measured positions and velocities for all 29 body
joints. Existing arm IK supplies the remaining body
targets from Quest optical tracking or controllers; webcam robot input is disabled
in standard launches. The current G1 weights remain unchanged. Like the H1 sample, locomotion is
learned joint actuation, but H1's network is not transplanted onto the G1 morphology.
These are GR00T **Whole-Body Control** locomotion checkpoints, not the GR00T
foundation/VLA model. The adapter loads `robots/data/g1_decoupled_walk.onnx` and
`robots/data/g1_decoupled_balance.onnx`; `g1_decoupled_policy.json` pins their
hashes and deployment contract. The checkpoints were trained upstream and are
unchanged; this repository runs inference and does not implement their training
pipeline. Walking is driven by
joint torques and physical contacts, not root-translation animation.
Default stationary manipulation remains available via `tools/launch_isaac_vr.bat`.

The decoupled actor's upstream arm posture contract is also applied to arm IK and
its smoothed/contact-projected commands. With `walking_policy_name == "decoupled"`
and explicit body efforts enabled, shoulder-roll limits are the intersection of
authored limits with left `[0.19, 2.2515]` and right `[-2.2515, -0.19]` radians.
This remains active in the actor's zero-command balance mode. Existing targets
outside the range return gradually under the 2.5 rad/s command slew and 0.15 rad
measured-joint lead; contact constraints can hold recovery instead of pushing
through geometry. Both arms remain tracked within this range. Stationary mode,
finger control, and gaze are unchanged. This configuration was enabled during the
earlier physical destination and stopping phases and separate final balance check;
arbitrary simultaneous reaching and walking remains unvalidated.

Command smoothing uses bounded wall-clock `_teleop_input_dt`; the physical gait still
advances in simulation time. Release clears the base command immediately. Stop
invalidates controller/physics caches and requires initialization before driving the
robot again; Pause and reset also discard navigation intent. XR camera scheduling
checks readiness and finite valid poses before submitting a transform. These are
lifecycle checks, preserving the accepted camera mapping and gaze behavior.
Explicit Reset invalidates the episode before Play. The base sample then awaits
an application update, which may already initialize the robot. The post-reset hook
preserves that result instead of forcing a second initialization on the following
physics tick. If no physics callback has arrived, initialization remains pending.

A controlled Stop followed by the registered Humanoid Reset handler passed with XR
enabled throughout, valid physics tensors and zero pending UI tasks. The test invoked
the real registered callback with its UI unbuilt; it did not click a displayed button.
This does not reproduce or resolve the user's reported Stop → Humanoid Reset crash.
Separately, native XR off/on still crashed under Vulkan after the camera
viewport-readiness guard; that guard is defensive, not a native-reentry fix.
The Direct3D12 workaround below passed that controlled cycle twice. No user-requested
XR toggle is inferred from the reported Reset action. See the
[lifecycle evidence](validation/isaac-sim-6.1.md#stop-reset-and-native-xr-reentry).

The repository's Windows VR launcher now appends `--/app/vulkan=false` before user
arguments, selecting Direct3D12 for the process without changing installed app files
or global renderer settings. The controller launcher and gesture alias inherit it.
An explicit trailing `--/app/vulkan=true` remains available for a Vulkan diagnostic.
Twenty-seven stub-launch checks cover defaults, aliases, verbose mode, quoted
arguments and override ordering; they do not initialize a renderer. A real launch
separately confirmed Direct3D12, passed two native XR reentries, 25 camera poses,
backward/release/settle and the registered Reset handler. The Reset check had no built
button. A later fresh launch also passed Stop → Reset through the existing built
button's `Button.call_clicked_fn()`, with normal UI subscriptions active, exactly one
reset and XR preserved. This exercises the built callback, not a physical mouse click.
Hardware hand/gaze quality, payload and long-session reliability remain unverified.
Brief native errors during XR transitions remain. See the
[Direct3D12 evidence](validation/isaac-sim-6.1.md#direct3d12-process-workaround).

A separate native crash during Clear/Load entered NVIDIA's
`metricsAssemblerChangeListener.py` while closing the stage. The launcher disables
that edit listener with `--/metricsAssembler/changeListenerEnabled=false` without
changing installed files or disabling initial asset conversion. In-place asset
variant/reference edits therefore require a fresh load. A fresh load and subsequent
Clear/Load passed with 22 cartons, valid physics and no health fault; XR then enabled
successfully. This verifies the reproduced reload sequence, not arbitrary extended
reload use. The separate Vulkan XR off/on crash has the bounded Direct3D12 workaround
described above; the Metrics Assembler setting alone does not address it.

**Three Stop/Play and two Pause/Play cycles passed with real XR active and no
fault.** The native restart crash did not recur during that sequence, which
predates the point-only change. The full point-only physical replay passed with
synthetic hand input. Actual Quest pointing accuracy, occlusion, and extended-session
reliability remain unverified; earlier restart acceptance does not establish those results.

`gesture_locomotion_status()` reports `mode="point_only"`, `state`, `target`,
`candidate`, `acquisition_progress`, `detail`, requested/filtered velocities, and
`policy_mode`. `route_state`, `route_waypoints`, `route_queries`, and
`route_expanded` expose incremental planning progress. Velocity arrays are
`[forward, lateral, yaw]` in m/s, m/s, and rad/s;
policy mode is `balance` or `walk`. `hand_reserved` is false. Missing observations
clear navigation without latching a physics fault. Balance/unintended-motion faults
pause physics; inspect the reported fault and use Reset or LOAD before continuing.
Ray diagnostics expose hit path and distance, while rejection details distinguish
missing colliders, ineligible surfaces, target range/heading, and blocked routes.

**Historical V-controller result:** in the 18 September pre-adjustment live capture,
the optical source remained valid while about 97% of active samples stayed below
the actor's walk threshold. An earlier zero-command drift of about 0.197 m/s caused
a separate motion-guard pause; the idle test after Reset was stable. Those findings
predate the destination selectors and are not acceptance results for point-only navigation.

Loss detection is limited by XR's reported validity. A runtime that repeatedly
returns a cached wrist pose or skeleton with VALID flags can conceal real-world
occlusion. Repeated polling does not prove that those measurements are new. The
software loss tests invalidate their inputs explicitly; hardware acceptance must
separately establish stopping when the Quest cameras lose sight of the hand.
This cached-valid-input case remains an explicit limitation.

The scene retains six practice objects, moves their 0.92 m table to `(2, -2)` m,
and adds dynamic warehouse cartons `Box_06` through `Box_09`. Their largest
horizontal bounds are 0.22, 0.28, 0.34, and 0.40 m; masses are 0.25, 0.35, 0.50, and
0.70 kg. Floor centers are `(1.5, 1.0)`, `(1.5, -1.0)`, `(2.6, 1.15)`, and
`(2.6, -1.15)` m. Asset centimetre conversion stays on the reference child, while
the wrapper supplies scene sizing, centering, and one dynamic rigid body. Mesh
collision uses convex hulls.

`tall_packages.py` adds `Box_10`–`Box_12` at `(3.5, 2.8)`,
`(3.5, -2.8)`, and `(1.35, 0)` m. The first two stations are separated from
the original low cartons. The tallest carton is directly ahead of spawn for an
initial pickup attempt and remains a real obstacle for longer routes.
Body dimensions are **32 × 32 × 76 cm**,
**40 × 32 × 88 cm**, and **30 × 38 × 96 cm**; total masses are **0.25, 0.35,
and 0.45 kg**. Their 3.5 cm diameter, 16 cm long handles have 7 cm collision-open
clearance behind them. Authored handle centers are **0.685, 0.805, and
0.940 m** above the floor; after the 5 mm initial floor gap settles, they are
**0.680, 0.800, and 0.935 m**. `Box_12` alone places its handle 2.5 cm below the
carton top, reducing downward arm reach without changing body clearance or
braking guards. Each carton and handle share one dynamic compound body;
packing tape is visual only. These deliberately light props are not loaded-carton
mass models, and add no attachment joints or scripted motion.
All thirteen roots are registered for contacts, object approach, highlighting,
and recording. Stop before reaching an open hand around a handle, close with
opposed thumb contact, then lift; opening releases it. Tall-carton pickup and
payload carrying need separate live checks. No floor crouch or bimanual-carry
controller is provided.

The point-navigation cruise cap is **0.65 m/s**, shared with the downstream
physics command clamp. The slow radius is **1.60 m** and the arrival radius is
**0.30 m**. Keeping the approach slope at 0.5/s preserves the prior profile
inside 1.20 m while raising far-target cruise from 0.45 m/s. The 0.08 m/s
minimum, 0.20 rad/s curved-steering cap, heading attenuation, and all motion guards
remain. The new zero-forward stepping-turn phase has its own 0.40 rad/s request
and progress watchdog, as described above. The 19 September physical baseline
passed forward walking, both turns, and stopping before this stepping-turn change;
the straight phase averaged 0.4884 m/s in simulation. Revised synthetic destination
and 0.65 m/s reaching replays also passed; operator accuracy remains unverified.
Earlier 0.45 m/s replay and throughput
measurements below retain their original scope.

The physical path uses **200 Hz CPU physics** and **50 Hz** NVIDIA GR00T decoupled
balance/walk inference through `isaacsim.pip.onnx` with its CPU execution provider.
Gravity stays enabled. There is no fixed world anchor, station-hold/turn-hold root
assistance, per-step root teleport, or kinematic fallback. Missing or incompatible
policy data fails explicitly; locomotion or articulation-health faults pause
physics. Reset restores spawn state and clears history; it cannot repair a failed
balance controller during Play.

Each float32 input frame has 86 values: velocity command scaled by `[2,2,0.5]`,
height command `0.74`, torso RPY `[0,0,0]`, body angular velocity ×0.5, projected
gravity, 29 relative joint positions, 29 velocities ×0.05, and 15 previous raw
actions. Six complete frames are concatenated oldest first into `[1,516]`, with
zero past frames on reset. The actor returns 15 leg/waist actions, mapped by joint
name, with targets `default_positions + 0.25 * actions`. Arm observation offsets
are zero. Balance is selected for unscaled velocity-command norm below `0.05`;
otherwise Walk is selected, preserving history and previous action across switches.
There is no gait clock, base-linear-velocity observation, or finger output.

All 29 body motors use the paired deployment gains with explicit
`clip(Kp * (target - q) - Kd * qd, -limit, limit)` torques each 200 Hz physics tick.
Position drives are disabled for those joints to avoid competing control loops.
The example defers the robot's initial effort update and refreshes efforts once
after arm IK, so the next step uses the latest arm targets without duplicate PD
work. Standalone callers retain the default update inside `forward()`. Finger
drives retain their existing bounded control. Initial leg/waist
positions are zero, distinct from the actor's nonzero observation offsets; initial
shoulder roll is `+0.2/-0.2 rad`. The pelvis spawns at `0.8 m` and settles physically.

Walking-only foot compatibility replaces the asset's shortened contact pattern
with the released XML's four 5 mm spheres per foot: heel `x=-0.05, y=±0.025` and
toe `x=0.12, y=±0.03`, all at ankle-local `z=-0.03 m`. This is an anonymous session
layer change; the referenced asset and authored rigid-body mass, COM, and inertia
remain unchanged. Missing or changed collider structure fails explicitly. These
contact and actuator settings are part of the validated configuration, not a
guarantee for other assets, materials, slopes, or payloads.

The immutable checkpoints, original YAMLs, exact metadata, and model/source
licenses are recorded in
[policy provenance](../source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/data/README.md).
The weights use the NVIDIA Open Model License; source/configuration uses Apache-2.0.

**Historical 18 September point-only checks:** **504 offline tests passed in 9.017 seconds**.
The complete physical replay passed all five phases: destination approach, arrival
stopping, hand release, explicit tracking loss, and final balance. Forward travel
was **0.604704 m**, with **0.299868 m** remaining at arrival. There were no health
faults or callback errors. Only the navigation hand reader was substituted;
production scene queries, blue visuals, and learned physical control remained
active. The reader was restored, gaze identity was preserved without eye-input
substitution, and cleanup paused the timeline. The updated mode then passed
**three Stop/Play and two Pause/Play cycles in 6.291 seconds**: three physics
initializations, fresh-release requirements, and no faults or callback errors.
That lifecycle run had XR disabled and no connected devices; it validates desktop
physics lifecycle behavior only. Point-only Quest hardware, pointing accuracy,
and occlusion behavior remain operator checks. The earlier XR-enabled sequence
above is historical evidence and is not a hardware test of this selector.

**Historical point-and-gaze checks:** **505 offline tests passed.** The physical replay
substituted only the navigation hand/gaze readers, retaining actual floor raycasts,
carton clearance, blue visualization, the policy, and health guards. It advanced
**0.60276 m** and reached **0.29985 m** from the destination. Arrival stopping,
release, and explicit tracking-loss phases passed. The 300-second wall timeout
interrupted the final balance phase; the original failure report remains retained.
A separate **3.01 simulation-second** balance continuation passed in **84.41 wall
seconds**, ending near **0.0001 m/s** without a health fault. Readers were restored
and gaze identity preserved. These phase results do not constitute a complete
uninterrupted replay pass or validate real operator pointing/occlusion behavior.
They predate the point-only change; current synthetic replay results are listed above.

Run the current selector replay against an initialized walking scene:

```powershell
python .\tools\validate_point_gaze_walking_live.py --output .\.webcam\point-only-walking-live.json
```

The tool retains its historical filename; its default output is
`.webcam/point-only-walking-live.json`. The default wall timeout is **600 seconds**
to accommodate slow VR rendering.
Its simulation-time approach budget derives from distance and minimum command
speed; a progress check rejects an approach that stops getting closer. Acceptance
thresholds are unchanged. The task saves partial reports, supports `--resume UUID`,
restores temporary navigation input overrides, and acknowledges Pause before completing cleanup.
See the [validation record](validation/isaac-sim-6.1.md) for artifacts and limitations.

**Historical V-controller-era PhysX baseline and right-arm replays passed.** Baseline phases cover six
seconds at zero command, forward walking, left/right forward arcs, and stopping.
The right-arm replay uses production IK and finger control to reach while walking,
close/reopen the hand, stop with the arm extended, and lose synthetic input. In the
measured run, the extended palm error was `0.09854 m` while walking and `0.09167 m`
while stopped; stopped planar speed was about `0.0039 m/s`, index curl followed
`0.8`, and reopening returned it near zero. No health fault or new callback error
was recorded. These are synthetic inputs in real physics, not Quest hardware tests.

Stop before precision pickup. The roughly 9–10 cm extended-palm error remains a
control limitation even at a stop. Floor pickup, crouching, loaded-object carrying,
and arbitrary simultaneous reaching/walking are not validated by these replays.
Real Quest stable pointing, destination acquisition, steering, release,
arrival, obstacle rejection, and cached-valid occlusion tests remain required.
Three Stop/Play and two Pause/Play cycles separately passed with real XR active
and no fault. The old replays do not validate the new selector; they preserved their
then-active gaze and webcam arbitration. Current launches disable webcam robot input.

The historical controller replay CLI runs against an already-loaded walking scene:

```powershell
python .\tools\validate_gesture_walking_live.py --suite all --reset --output .\.webcam\gesture-walking-validation.json
```

Run from the repository root. `--suite baseline` or `--suite reach` runs one replay;
`--port` selects its Python server. `--reset` awaits a scene reset before the first
replay; `all` resets again between suites. The tool restores input overrides,
diffs new callback errors against existing history, and pauses in `finally`.
The host starts a bounded background Task and polls short Python-server requests.
A job UUID identifies its retained report; `--resume UUID` reconnects without
starting a second replay or resetting the scene. Concurrent validation jobs are
rejected. Timeout cancellation still runs cleanup, and reports/errors remain in
the live Kit session for recovery (the eight most recent jobs are retained).
It verifies gaze identity within each replay without replacing the tracker;
intentional reset lifecycle changes are outside that identity assertion. See the
[validation record](validation/isaac-sim-6.1.md) for final results and test counts.

### Legacy moving modes

The older assisted `policy` path retains its 47-observation recurrent actor and
twelve leg outputs at 50 Hz. Its station-hold and turn-hold options directly assist
the base. Its `kinematic` fallback integrates the root with gravity disabled and
does not demonstrate balance. These behaviors belong to the legacy path and are
excluded by strict physical gesture walking. Reset restores spawn state and input
filters; keyboard presses use a set so repeat events cannot accumulate velocity.

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

Default stationary physics is 100 Hz; gesture walking requests 200 Hz. Behavior,
hand, gaze, and object recording intervals are derived from that timestep to retain
**100 Hz logging in both modes**. At 200 Hz physics, those records run every second
physics tick. Gaze rows reuse the latest 50 Hz raycast result; two rows do not mean
two independent gaze queries.
`GazeSample.sample_time` records query time for in-process consumers. Camera capture
is requested at 10 Hz and CSV buffers flush about every 2.5 seconds. `metadata.json`
records requested/resolved locomotion mode, grasp mode, and grasp configuration.
The session's simulation clock stays monotonic across world resets; gait and velocity
filters clear so reset discontinuities do not become motion estimates. A future
episode converter must still mark reset boundaries rather than interpreting a
continuous timestamp as a continuous physical trajectory.

CSV rows share a physics-step index, but the image logger reads the camera's most
recent render and stamps the save with the current clocks. It does not record the
original render timestamp, render frame ID, or per-frame camera extrinsics. The
recorder writes RGB only; depth, segmentation, full contact arrays, final arm joint
targets, and body motor efforts are not part of this dataset schema. Applied finger
curls and filtered base commands are recorded separately from operator requests.

## Performance and CPU physics

The stationary defaults request **100 Hz CPU physics** and **90 Hz rendering**, using the Torch
backend. Gesture walking requests **200 Hz CPU physics** to match the actor's source
environment, evaluates its ONNX actor at 50 Hz, and requests **60 Hz rendering**.
Render cadence is configured before loading the scene. These are simulation intervals,
not displayed-frame or real-time guarantees. An earlier 12 September recorded interval
advanced **16.26 simulated seconds in 59.992 wall seconds**, a real-time factor of
**0.271**. This is evidence from CSV timestamps, not a measurement by the new live
profiler and not a benchmark of the revised controllers. XR load varies between runs.
When the simulation falls behind, robot movement and tests that wait for simulated
time also take longer in the real world.

CPU physics was chosen after an earlier scene comparison reported approximately
6.8 ms per step on CPU versus 23–28 ms on CUDA. Those historical measurements are
not a current benchmark of this stationary scene. Switching to GPU physics is not
an established fix for the present slowdown.

The code performs the following work while the example is loaded and playing.
The 19 September live observations below measure selected Python paths; they do
not isolate every native solver, rendering, or recording cost.

| Work | Current behavior |
|---|---|
| Automatic recording | `_behavioral_data_enabled = True`; derived intervals retain nominally 100 Hz behavior, hand, gaze, and object rows in both modes, including with 200 Hz walking physics |
| Recorded camera | An additional 256×256 camera render product and RGB annotator refresh at render cadence, even though PNGs are saved at about 10 Hz |
| File writes | CSV batches queue about every 2.5 simulation seconds to one background disk writer, bounded to eight outstanding batches; PNG encoding/saving remain synchronous with lossless compression level 1 |
| Camera attachment | The body pose is read and the mounted view scheduled every physics tick and every application frame, including while paused |
| Arm and pickup poses | Walking raw IK goals can be reused for one intervening 200 Hz step, giving a nominal 100 Hz solve rate; validity, measured-joint limits, contact constraints, and final body PD remain 200 Hz. Both new arm solves share that tick's measured Jacobian and joints |

The CSV worker receives copied, read-only dictionaries from the five telemetry
collectors, whose cells are native immutable scalars. All streams in a flush are
detached before their worker jobs begin; field selection and text formatting run
on that worker. The generic writer API freezes other values before submission.
These copies contain no simulator views or camera objects. Python thread
contention and synchronous PNG work can still affect timing, so moving disk I/O
does not by itself establish smooth headset rendering.

Hand devices, validated landmarks, and optical curls are shared only within one
physics callback. The installed bulk XR API reads each hand skeleton once per
callback; older adapters retain individual descriptor reads. A `finally` block discards the sample on every exit, including
errors; reset and hand deactivation also invalidate it. No sample is retained for
the next physics tick, and head/eye input is not cached by this mechanism. Both
new arm IK solves, smoothing, and contact holding share that tick's measured joints.
The walking raw-goal cache retains no physics tensors or final motor commands;
tracking loss, source changes, large target changes, and invalid clocks end reuse.
Stationary IK remains at its existing 100 Hz physics cadence. Body efforts are
written once after arm updates, avoiding the earlier duplicated PD calculation.
Hidden target markers retain their control pose in memory without USD transform
writes; visibility edits occur only when visibility changes. Joint command
limits and smoothing also affect movement response, independently of frame rate;
their current values and purpose are described under arm and finger retargeting.
A paused, empty session is not executing this example's physics, IK, or recording
callbacks. Check which application and scene are active before attributing a slowdown
to those paths. Run only one Kit application during comparisons.

### Profile the active teleoperation session

With the updated example loaded and playing, enable its Python server and use
ordinary host Python from the repository root:

```powershell
python tools/profile_teleop_live.py --seconds 20 --timeout 45
```

The profiler observes real input and temporarily times input arbitration, finger
updates, arm/pickup updates, recording, and `g1.forward`. It reports wall/simulation
progress, real-time factor, application-update rate, observed physics intervals,
per-method timings, and callback errors. It does not alter gaze, camera, recording
settings, input devices, or the timeline. Temporary wrappers are restored on exit.
The application-update rate is not headset/compositor FPS; nested method timings
overlap, and instrumentation adds overhead. Compare repeated runs using the same
scene and instrumentation.

### Measured update — 19 September 2026

The first update's short, instrumented observations used the same walking scene and 100 application
updates per window, Quest optical hands, recording enabled, and RTX rendering.
Physics remained CPU at 200 Hz, policy inference at 50 Hz, and requested rendering
at 90 Hz. One Isaac process was open on the Ryzen 9 9900X / RTX 5090 machine.

| Measurement | Before | Updated |
|---|---:|---:|
| Application updates/second | 19.08 | 23.37 |
| Simulation seconds/wall second | 0.212 | 0.259 |
| Arm update, mean per physics callback | 3.54 ms | 2.63 ms |
| Optical curl reader, mean per call including cache hits | 0.682 ms | 0.508 ms |
| Existing GPU frame statistic, mean | 8.80 ms | 8.82 ms |

This is about **23% more simulation progress per wall second**, not headset FPS
or a motion-to-photon latency measurement. The short windows are observations,
not repeated controlled benchmarks; input and renderer workload can vary. Native
profiler history may include earlier samples, so it is not used to attribute the
entire improvement to the solver. Both final input sources were optical hands,
no physics callback errors occurred, and all timing wrappers were restored.

The example now enables **Fabric pose output before loading scene assets**, while
retaining CPU physics. This avoids per-step USD pose/velocity authoring. Renderer
poses, gaze raycasts, arm IK, contacts, camera mounting, and recorded object poses
use their existing live physics paths. Cleanup restores the prior device, Fabric,
and exact four USD-output preferences; failed-load retries retain the original
snapshot. Dynamic USD transforms can remain at authored values during Play;
inspect physics tensors for current robot/object poses.

A wall-clock catch-up experiment raised simulation throughput but reduced app
updates to about 12/second, so that pacing change was rejected. The one-thread
experiment was also not retained; the final comparison restored eight PhysX
threads. No solver iterations, drive limits, contact safeguards, camera mapping,
or gaze algorithm were changed. Higher CPU/GPU utilization is not itself a
performance target: saturating the hardware can increase input/render queues.
Rendering uses the GPU; this single-robot physics and policy path remains on CPU.

Simulation is **still below real time**. A synthetic free-hand curl replay reached
within 0.15 normalized curl of each step request in about 0.155 simulated seconds
(0.54–0.55 wall seconds), with both hands opening/closing and all six actuators
following. These timestamps exclude headset sensing, transport, and display
latency. Loaded grasps, arbitrary reaches while walking, and an operator-rated
end-to-end delay improvement were not established by that replay.

Local evidence: `.webcam/speed-latency-baseline.json`,
`.webcam/speed-latency-final.json`, and `.webcam/faster-hands-response.json`.
The [validation record](validation/isaac-sim-6.1.md#faster-walking-and-hand-processing--19-september-2026)
contains the physical walking and lifecycle results.

### Pointing and throughput follow-up — 19 September 2026

The subsequent update batches XR hand descriptors per physics callback, reuses
bounded raw IK goals at nominally 100 Hz while retaining 200 Hz feedback, removes
duplicate PD calculation, and uses lossless PNG compression level 1. A fresh
walking scene configures rendering at 60 Hz; stationary rendering stays 90 Hz.
CPU physics remains 200 Hz and policy inference remains 50 Hz.

| Measurement | Before follow-up | Updated fresh startup |
|---|---:|---:|
| Application updates/second | 23.00 | 21.77 |
| Simulation seconds/wall second | 0.2553 | 0.3636 |
| Physics steps over 100 application updates | 222 | 334 |

These short windows show **42.4% more simulation progress per wall second** and
**5.3% fewer application updates**. Both observed optical hand sources and no
callback errors; timing wrappers were restored. They do not establish headset FPS,
end-to-end hand latency, or real-time simulation. Physics and policy still advance
at their original simulated rates. The changes do not justify forcing hardware
utilization to 100% or scaling physics time by measured frame duration.

The final comparison uses `.webcam/point-speed-before.json` and
`.webcam/point-speed-final.json`. An intermediate live `set_dt()` timing change
produced an inconclusive report; it is not acceptance evidence. The retained
configuration is applied at fresh startup. **568 offline tests passed**. The
0.65 m/s physical baseline passed balance, forward motion, both turning arcs, and
all three stopping phases without faults or callback errors. Forward X travel
was 1.9119 m over 4.015 simulated / 10.122 wall seconds; the earlier 0.45 m/s run
covered 1.2829 m over 4.015 simulated / 15.552 wall seconds. World-forward progress
per wall second was thus about 2.29 times higher in these runs. This combines the
changed command and runtime performance; it is not a controlled policy benchmark.
The initial reach replay passed at its existing 0.25 m/s fixture speed. A separate
0.65 m/s reach replay passed with maximum sampled palm error 0.0923 m, including
finger closing/reopening, stopping with the arm extended, and synthetic tracking
loss. The revised 3 m destination replay passed approach, arrival, release, loss,
and final balance, progressing 2.7023 m before requesting stop at 0.2984 m remaining.
All four carton corridor queries rejected passage. Three Stop/Play and two
Pause/Play cycles passed with XR active and gaze preserved. These fixtures do not
establish operator pointing accuracy or perceived jitter. A final 15-second real
Quest observation reported optical input and `hand_tracking` control in all 134
samples per hand, without read or callback errors. No fingers exceeded its motion
thresholds, so moving-hand accuracy remains unverified. See the
[current validation record](validation/isaac-sim-6.1.md#pointing-stability-and-walking-throughput--19-september-2026).

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
& "C:/path/to/isaac-sim-standalone-6.1.0-windows-x86_64/python.bat" tools/run_humanoid_tests.py
```

```bash
/path/to/isaac-sim/python.sh tools/run_humanoid_tests.py --isaac-sim-dir /path/to/isaac-sim
```

The runner locates bundled NumPy, PyTorch, and USD libraries. It does not start Kit,
connect XR, or change an open stage. Tests use the actual production class/function
definitions with module imports replaced by controlled dependencies: XR devices,
Kit services, and articulation tensor views are doubles; matrix math, USD stages,
materials, joints, and the bundled TorchScript checkpoint are real.

The **12 September head/reach update** passed **242 offline tests** in 6.625 seconds
using Isaac Sim 6.1 Python. The log is `_compat61/head-reach-tests.log`.
See the [dated validation record](validation/isaac-sim-6.1.md) for subsequent
webcam and Quest-priority results. Coverage includes first-contact
caps, per-body isolation, short gaps, opening/reset, final arm-contact projection,
input-loss arm holding, finite-coupling preparation, and articulation-health checks.
Coverage also includes the armature floor and preserved authored/per-axis values. Offline tests do not
establish collision dynamics or headset accuracy, and earlier rigid-coupling pickup
passes do not validate the new mechanism. See the
[validation record](validation/isaac-sim-6.1.md) for live configurations and results.

Coverage includes invalid poses; source transitions; moving/turning clutch frames;
wrist orientation and Jacobian indexing; target speed limits; finger loss; keyboard
repeat; lateral direction; trigger separation; hysteresis; drop rearming; reset;
coincident grasp frames; material restoration; policy timing; invalid actions; joint
limits; fallback posture; and deterministic checkpoint reset. Highlight regressions
also observe real USD `ObjectsChanged` notices: repeated gaze/grip overlap, target
changes, release, and reset must not resync any collision prim. Additional checks
verify runtime notices are confined to external shader inputs, no object-subtree
notice occurs, physics and original visual materials stay unchanged, and owner
changes with the same visible winner emit no notice. Geometry-copy tests cover
nested rotation/scale, matching expanded bounds, retained mesh topology, and the
absence of copied physics APIs/references. Reset regressions cover the post-reset hook
both before and after the first Play physics tick, plus warning deduplication
clearing only on explicit Reset. A separate teardown
check covers an invalid stage wrapper after the application closes its stage. IK
regressions also cover conflicting wrist/position requests, redundant-joint posture
recovery without changing the hand task, and world-axis rotation from a nontrivial pose.
The orientation-limit regression exercises both bound directions with zero, tiny and
partial remaining wrist travel, coupled translation/rotation axes, and exhaustion of
all useful secondary motion. Gravity-feedforward tests cover reordered native DOFs,
fixed/floating root offsets, arm-only ownership, effort clipping and invalid-data faults.
Open-hand regressions verify acquisition without grip, pinch, or fist and independent
finger updates without a closed hand or active arm target.
Palm regressions cover the initial flat-hand/vertical-robot mismatch on both sides,
world rotation and scale, missing/degenerate landmarks, reacquisition, stable rigid
palm centres, and preservation of controller clutch behavior. Further regressions
cover position-valid-only optical poses, partial orientation loss, bounded wall-time
filtering, IK secondary-task budgets and joint limits, physical arm holding on input
loss, and the retained assisted-mode return-to-rest slew after deactivation.

These tests do not validate extension startup/import ordering, live XR bindings, rendered
visuals, collision/contact dynamics, or balance. Run Kit integration checks separately
and sequentially. Do not start a second Kit application alongside an existing session.

### Table contact replay against the running VR application

LOAD a fresh stationary physical-mode example, initialize physics, and enable
`isaacsim.code_editor.python_server`. From the repository root, use ordinary host Python:

```powershell
python tools/validate_table_contact_live.py --timeout 240
```

The default right-hand replay generates optical landmarks and uses production
retargeting, IK, finger drives, contact reporting, and PhysX. It approaches an empty
part of the front table palm-down, lowers to contact, requests 1 cm further downward
motion, holds for 1.5 simulated seconds, then withdraws. Its report records actual
contacts and forces, all joint positions/velocities, waist deflection, root stability,
callback faults, and cleanup. Unsafe state stops the replay. This is a bounded
contact test; a pass is not general collision avoidance or headset validation.

The default artifact is `_compat61/table-contact-live.json`; `--output` changes the
path. The script does not move objects, modify physics settings, reload the scene,
or change gaze. It restores live inputs and pauses on completion, failure, or
cancellation. With recording active, `validation_<run_id>.json` labels its interval
as `synthetic_table_contact_replay`. Run this separately from pickup replays and
hardware observations. Current results are in the
[validation record](validation/isaac-sim-6.1.md); older rigid-coupling passes apply
only to their recorded configurations.

An earlier **1000 rad/s coupling, damping ratio 1, and 0.12 rad contact lead**
configuration passed both open and curled **right-hand**
replays with actual table contact, a 1 cm inward command, hold, and withdrawal.
The open-hand run's maximum waist deflection was 0.00781 rad (0.45°); the curled
run reached 0.004744 rad (0.27°). Both reported zero root movement and zero callback
errors. Evidence: `_compat61/table-contact-production-open.json` and
`_compat61/table-contact-production-curled.json`. Those passes do not validate the
current **500 rad/s, damping ratio 1, and 0.18 rad lead** defaults. Final table checks
with those settings are below; pickup and real headset handling are separate checks.

Later fresh-default right open and curled trials passed at 500 rad/s, damping ratio
1, and 0.18 rad lead (`_compat61/table-final-right-open.json` and
`_compat61/table-final-right-curled.json`). Switching to a left-hand trial then lost
right-hand input, triggering an automatic return through the table. The guard paused
at a measured 105.689 rad/s joint speed. The isolated right-hand passes are retained
as evidence, but do not establish that the complete sequence passed. Current code
holds the measured arm pose on input loss/clutch release and includes loaded predictive
contacts in the arm constraints. The corrected table-test handoff now passes.

**Earlier fixed-pose production table sequence: passed** at 500 rad/s coupling, damping ratio 1,
and 0.18 rad contact lead. These three runs completed sequentially in the same scene:

| Hand pose | Contact duration | Maximum waist deflection | Peak absolute joint speed | Report under `_compat61/` |
|---|---:|---:|---:|---|
| Right open | 2.04 s | 0.007803 rad | 8.529 rad/s | `table-stable-right-open.json` |
| Right curled | 1.82 s | 0.001424 rad | 4.235 rad/s | `table-stable-right-curled.json` |
| Left open | 2.94 s | 0.008452 rad | 2.889 rad/s | `table-stable-left-open.json` |

Durations are simulation time. Each run lasted 12.9 simulated seconds, withdrew
cleanly, and recorded zero root movement, articulation faults, and callback errors.
Maximum waist deflection across the sequence was 0.484°. These results cover the
specified approaches and their handoffs; they do not establish every table-contact
orientation, left-curled behavior, or headset tracking. The following same-scene
pickup run passed the cylinder but exposed a shallow cube pinch; the revised cube
profile passed a retry, as described below.

The table passes hold one finger pose per trial and do not validate repeated
closing/opening against the table. The subsequent dynamic stress failure described
under articulation health was reproduced and then passed with the selected armature
floor. To repeat that dynamic test on a fresh initialized scene:

```powershell
python tools/validate_table_contact_live.py --close-open-cycles 20 --timeout 300
```

This adds 20 close/open cycles at the loaded contact pose before hold and withdrawal.
The separate recorded-motion-derived path also passed; enlarged-object pickup
results have their own geometry and trajectory scope below.

### Physical contact replay against the running VR application

LOAD a fresh stationary example in default `physical` mode, with the cube and cylinder
on the front table. Enable `isaacsim.code_editor.python_server` and run this trial
trajectory from the repository root using ordinary host Python:

```powershell
python tools/validate_contact_grasp_live.py --object both --timeout 300
```

The replay supplies synthetic optical landmarks to production IK and finger control.
The command above uses a downward palm normal and separate approaches for the original
cube and cylinder. `--object both` does not test `Box_02` through `Box_05`; the four
added shapes have no validated grasp profiles. The CLI also exposes finger direction, offset, finger curl, thumb
curl, and opposition for explicit trajectory experiments. Left-hand runs require
explicit `--offset`, `--thumb-curl`, and `--opposition`; no left-hand grasp defaults
have been validated. Real PhysX contacts
and object motion determine the outcome. The script verifies that the objects remain
dynamic and free of joints, and records approach, close, lift, hold, and release phases.
Checks include actual contact, opposing support during the hold, sustained measured
lift, falling after opening, and cleared support after release. A failed trajectory is
reported as a failure, without substituting assisted attachment.

The default report is `_compat61/contact-grasp-live.json`; `--output` selects another
path. It includes trajectory parameters, actual contact lead, contact forces/health,
object motion, root stability, and cleanup results. The script does not reload the
scene or directly reposition objects. It restores live input and the contact reader,
commands a drop, and leaves the timeline paused. Open your input before resuming after
this forced-drop cleanup. Run it separately from hardware observation and other
replays. Right-hand starter-object results and their coupling configurations are
recorded in [the validation report](validation/isaac-sim-6.1.md). Earlier passes used
the superseded rigid setting. Assess each new report independently; a pass validates
that synthetic trajectory, not headset tracking, left-hand pickup, or general grasp reliability.

A combined tuning trial with 500 rad/s coupling, damping ratio 1, and 0.18 rad contact
lead used the earlier **4 cm cube and 4 cm diameter, 8 cm cylinder**. Both passed in
sequence: minimum cube lift 11.535 cm,
minimum cylinder lift 11.463 cm, two simulated seconds of hold each with opposing
contact throughout, then free release. This was an intermediate success; final
fresh-scene reproduction failed on the cube. In the later run following all three
table-contact tests, the original cube profile (thumb curl `0.35`, opposition `0.30`)
slipped. The cylinder passed with
11.4844 cm minimum lift, opposing support throughout its hold, and an 11.5079 cm fall
after release. No articulation faults or callback errors occurred.

A same-scene cube retry at thumb curl `0.45` and opposition `0.40` passed: minimum
lift 11.3552 cm, opposing contact throughout the hold, and 11.3632 cm fall after
opening. This profile provides more opposed thumb/index pad contact and remains the
exploratory cube replay default. Coupling `500/1`, the `0.18 rad` lead, and production
human-hand retargeting were unchanged. The final fresh-scene run still slipped the
cube, so the isolated retry does not establish repeatability. The combined failure
is recorded in `_compat61/contact-stable-default-profiles.json`; keep it alongside
the earlier successes in validation data.

The final report's cylinder trial passed with minimum lift **0.114477 m**, opposing
support during every observation of its **2.01 simulated-second** hold, and a
**0.114704 m** fall after opening. Across the combined replay, all **2,943** contact
reads were healthy, with zero articulation faults, callback errors, and root movement.
The overall result remains **failed because the cube slipped**. The fixed-pose
table sequence and smaller-cylinder trajectory passed. The new armature floor also
passes the reproduced dynamic table test and reconstructed recorded-motion path;
other contact trajectories need separate validation.

For the enlarged **6 cm diameter, 10 cm cylinder**, an explicit palm offset of
`(0.003, 0.015, 0.070)` passed (`_compat61/larger-cylinder-pickup.json`): minimum lift
**0.110497 m**, opposing support during its entire **2.01 simulated-second** hold,
and a **0.130688 m** fall after opening. Finger/hand physics settings were unchanged.
This validates the selected larger-cylinder trajectory, not the earlier smaller-object
defaults for every shape.

The first enlarged **6 cm cube** trial at offset `(0.008, 0.013, 0.050)` failed. Its
thumb normal was nearly +Z while its index normal was nearly +X: top/side contact
did not produce an opposing pinch. The lower offset `(0.008, 0.013, 0.035)` also
failed grip while the robot stayed stable (`_compat61/larger-cube-side-pinch.json`).
No gains, forces, or hand physics changed between these geometry trials. Neither
trial established opposing support, and enlarged-cube pickup remains unresolved.

The clean stop-fix production scene was restored and paused at **2026-09-12 09:58:36 UTC**
(Kit PID 41076). `_compat61/stop-fix-ready-state.json` records all 24 actual armatures
at approximately `1e-4 kg·m²`, cube dimensions `(0.06, 0.06, 0.06)` m and cylinder
dimensions `(0.06, 0.06, 0.10)` m, healthy contacts, no articulation fault or callback
errors, and no input/contact-reader overrides. Both hand markers were invisible.
The fresh recording session had no synthetic-validation interval files. XR devices
were absent, so this handoff does not establish real optical/controller transport.
This snapshot predates the four additional practice shapes.

| Exploratory replay object | Palm offset from measured open-hand reference (world metres) | Finger / thumb curl | Thumb opposition |
|---|---|---|---|
| Cube | `(0.008, 0.013, 0.040)` | `0.75 / 0.45` | `0.40` |
| Cylinder | `(0.003, 0.015, 0.060)` | `0.75 / 0.55` | `0.50` |

These are exploratory replay poses from the smaller objects, not automatic grasp
planning or changes to human hand retargeting. They are not validated defaults for the
enlarged shapes; the larger-object trials above use explicit offsets. The cube profile
is not robustly validated. The cylinder approach
clears its taller top; both profiles aim to oppose thumb and index contacts.
Before overriding input, every replay checks all 12
actual hand couplings against the loaded instance's finite compliant configuration
and preparation paths. Stale hard `0/0` couplings fail preflight.

If a behavioral recording session exists, the replay also writes
`validation_<run_id>.json` there. It records `source: synthetic_hand_replay`, start/end
Unix time, simulation time, physics steps, and the final status. The report's
`recording_provenance` field points to this marker. Apply its interval boundaries when
selecting data for analysis or training: other intervals in the same recording may
contain real operator input. An incomplete marker without an end boundary must not be
treated as a completed validation run.

### Legacy assisted replay against the running VR application

The existing `validate_humanoid_live.py` replay checks **legacy assisted attachment**.
It does not validate the new default physical grasp. Configure assisted mode before
loading its scene. With that example loaded in Isaac Sim XR VR and the Python server
enabled, run this command with ordinary host Python:

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
These are historical assisted stationary simulation results; they do not establish
physical contact grasping, walking balance, or hardware tracking quality.

### Camera and finger validation

The historical `validate_camera_live.py` checks an older rigid-mount contract.
`validate_robot_head_camera_live.py` now checks the comfort camera. Run current
camera and finger checks sequentially against the already-loaded stationary example:

```powershell
python tools/validate_robot_head_camera_live.py
python tools/validate_fingers_live.py --timeout 300
```

The current camera contract holds a calibrated neutral optical height, rejects robot
bob and roll/pitch, smooths robot XY/yaw, and applies fresh HMD rotation to orientation
and the neck-to-eye offset. Check standing and walking camera motion separately, then HMD left/right,
pitch/roll, physical-translation rejection, Pause, B recentering, and Stop/Play.
Compare actual native XR view and authored camera poses during moving-head input;
static pose equality alone does not establish dynamic tracking or headset comfort.
Verify gaze remains aligned, and look far left/right/up/down for unwanted head-mesh
occlusion. Verify the visible neck pivot stays fixed in torso coordinates during
HMD rotation and body motion; a fixed camera-to-render-head transform is no longer
the production contract. This preserves the visible body's attachment while the
comfort camera remains independent.
Current results belong in the [validation record](validation/isaac-sim-6.1.md).
The finger replay drives each of ten fingers and both thumb-opposition joints,
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

The offline suite includes open-hand acquisition and independent-finger regressions.
On 7 September 2026, the full live finger replay passed **19.73 simulated seconds**
with all ten fingers and both thumb-opposition
joints reaching their independent 0.7 normalized targets. Partial/full tracking loss,
normal controller input, and Touch input mislabeled as `hand` passed for both hands;
root translation was zero and no callback errors occurred.

The **historical fully fixed-camera replay** passed both Play and Pause with the Quest session connected.
Maximum camera-to-body matrix change was `3.33e-16`; the authored camera matched the
computed mount exactly. A separate read of the real headset's virtual eye position
matched the mounted camera within `4.25e-8 m` after the physical headset had moved
over a metre. Gaze still reported `eye_tracker` with zero failed updates.

Earlier 20- and 45-second real-input observations contained no valid finger landmarks.
The user subsequently confirmed that eyes and fingers worked again after starting
SteamVR independently. In the 12 September session ending `02-02-51`, the last
complete minute contains **1,627 rows with both optical hands and `eye_tracker`**.
This supersedes the earlier missing-skeleton observation and confirms recovered input
transport. It does not validate the new absolute palm alignment, revised IK response,
or tracking accuracy on hardware. Repeat the real-input observer and the acceptance
checks below after loading the changed code; keep the working gaze/runtime settings.

## Live acceptance procedure

1. Save your current stage and reload the changed extension/application. Open
   **Window → Examples → Robotics Examples → Policy → Humanoid: Unitree G1**, LOAD,
   then Play. Check the console for import, articulation, policy, and IK errors.
2. Confirm the resolved locomotion mode is `stationary`. Tilt both sticks, press X/A,
   try keyboard movement keys and gamepad inputs, and move your head. The robot's base
   position and orientation must stay fixed. Repeat while reaching with both arms.
3. Move your physical head sideways, vertically, and forward/backward: the camera
   position must ignore that room-scale translation. Turn left/right and tilt up/down
   or in roll: the view must rotate in the same direction and the eye must follow
   its neck arc within the stabilized robot frame. Repeat while paused. Press **B** to
   make your current heading forward; the left stick must not enable room-scale
   camera motion. The visible head follows rotation while the physical head
   collision shapes stay fixed to the torso. Check steep downward/backward views for
   head-mesh occlusion and verify balance corrections do not shake or tilt the horizon.
   Hold grip with trigger released: the hand should stay open. Reach forward/up/out,
   rotate the wrist, then release/reengage grip. Verify the actual palm follows smoothly,
   does not jump when reacquired, and does not move or tip the robot's base.
4. Confirm grasp mode is `physical`. With an open hand, gently touch an empty table
   area, slide a short distance along its surface, then withdraw upward. Repeat on
   both sides. Inward targets should stop at contact without deforming the robot;
   withdrawal must remain available. If a health fault pauses physics, save its
   console message and Reset or reload before continuing; Play alone is insufficient.
   Near the table, release grip or briefly hide the optical hand: the arm should
   hold its measured pose until reacquired, without automatically returning to rest.
   Missing finger input should still open the hand; valid trigger input remains active.
   Then start with the small cube and cylinder, place
   opposing robot fingers around the object, and pull the trigger gradually. Lift
   slowly and hold; observe contact, slip, and drop behavior. Verify no hand-to-object
   joint is created and the object is never snapped or teleported to the wrist.
   Release the trigger and observe the fingers open and the unsupported object fall.
   Press Y while trigger remains held: fingers must stay open until you release and
   squeeze again. Releasing grip must end arm tracking while the valid trigger
   continues to control the fingers independently.
5. Present open hands first, without making a fist or pinching, and verify tracking
   starts. Begin with flat palms while the robot hands are vertical: both robot palms
   should align with yours. Rotate palm-up, palm-down, and sideways, then briefly hide
   and reacquire each hand; the initial orientation mismatch must not return. Bend
   each optical finger separately and move the thumb across the palm without
   curling its tip; verify independent robot-finger and thumb-opposition movement.
   Repeat a thumb-opposed pinch and multi-finger wrap with optical tracking. Test
   partial occlusion and device loss/reconnection separately from a successful hold.
   Missing wrist/middle landmarks must end arm tracking without a raw-origin jump;
   losing only transverse landmarks should preserve position tracking while omitting
   the orientation target.
   Verify missing finger input commands open and stale hand targets are not used.
   Reset while holding an object; verify contact support state clears, no grasp joint
   exists, and the robot returns to its spawn pose.
6. Hold the head still and move only the eyes. Confirm the runtime reports real
   `eye_tracker` and the ray changes. Disable eye input and verify `hmd_forward` or
   an unavailable source, with no stale object hit. Check a nearby object and a target behind robot
   self-colliders. Overlap gaze/hand selection and verify the original material returns.
7. Inspect the new session CSVs and metadata for source labels, resolved locomotion,
   grasp mode, contact health/support, moving object poses, monotonic times, and
   expected capture rates. Physical `contact_supported` is distinct from the legacy
   `is_grabbed` attachment flag and must not be treated as proof of a successful lift.

For point-only walking, test blue-ray floor selection, dwell, target locking,
forward motion, turns on both sides, point-release stopping, arrival, obstacle rejection, and
Quest tracking loss separately. Gaze direction and eye-tracking loss must not
block selection or redirect the robot; the existing eye tracker and red ray must
continue independently. Controllers and webcam input must not authorize selection.
After Stop, Pause, or Reset, a held point must not resume walking until
a valid release and new acquisition. Test distant targets, a target behind the robot,
a blocked direct route with a clear detour, an impossible route, and cancellation
during planning. Turning must use the physical actor and require clearance.
Reverse and strafe remain unavailable. Stop before trying each tall carton handle;
check lift, slip, opening release, and object collision independently of walking.
The legacy moving modes have separate control tests.
A successful stationary session does not validate walking with teleoperated arms
and payloads.

Optional read-only diagnostics use the repository's TCP helper after enabling
`isaacsim.code_editor.python_server` in **Window → Extensions**:

```powershell
python tools/kit_exec.py "print(EX.describe_xr())"
python tools/kit_exec.py "print(EX._eye_gaze_tracker.describe())"
```

`tools/launch_isaac_vr.bat` enables the server when launching a new session. A refused
connection on port 8226 means the server is unavailable; it does not diagnose the
robot or headset. The earlier assisted stationary path was exercised in the running
VR application with synthetic XR inputs and real PhysX, as described above.
Headset hardware tracking and optional walking modes require separate live checks.

## Later milestones

The RGB-webcam helper currently provides an operator/landmark preview only; its
[earlier input architecture](#optional-webcam-input) is retained but disabled.
If webcam control is deliberately revisited later, hardware checks would include
headset-worn recognition, field-of-view handover and end-to-end latency. Hip-relative estimated 3D coordinates
are not measured room depth, and the current calibration does not establish metric
camera extrinsics. [MediaPipe Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)

First verify that a headset-wearing operator can be tracked: the official model card
lists a head outside the image and metric-accurate depth outside its intended scope.
Headset occlusion is therefore a feasibility risk. An ordinary RGB webcam is a low-cost
prototype; a depth camera adds measured visible-surface depth but still needs calibration
and cannot observe a hidden elbow. Neither camera can recover motion that both views obscure.
[BlazePose GHUM model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf),
[RealSense projection/deprojection](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)

A later study could use hand-closing velocity and gaze at the same object to infer a
grasp-intent or gentle-to-firm request. These cues do not measure operator pressure.
Evaluate them against fixed-grip and speed-only baselines using actual contact, slip,
drop, and release outcomes before adding adaptation. No gaze/velocity intent model or
training pipeline is implemented; the current contact diagnostic does not alter gaze.

[`g1_grasp_env.py`](../source/standalone_examples/api/isaacsim.robot.policy.examples/g1_grasp_env.py)
is a retained throughput benchmark. It does not provide a training environment with
actions, rewards, or episode resets; its printed policy-step budgets extrapolate
physics throughput without training costs. The [learning guide](../learning/README.md)
separates the recording data and planned learning stages from these prototypes.

## API references

- [PhysX tensor Jacobians](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html): world-frame Jacobians and fixed/floating root layout.
- [USD joint frames](https://openusd.org/release/api/class_usd_physics_joint.html): local position/rotation for each connected body.
- [Physics scene queries](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/physics_umbrella/physics_umbrella_runtime.html): all-hit raycast callback and query lifecycle.
