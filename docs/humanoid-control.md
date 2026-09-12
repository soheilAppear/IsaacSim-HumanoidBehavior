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

Optical control accepts a valid **open hand** immediately. Finger updates do not
require a fist, a grip/pinch gesture, or an active arm target. Pickup requires actually
closing the robot fingers around the object. The user confirmed that eye and finger tracking
returned after starting SteamVR independently. The latest recording also contains
both optical hands and real eye-gaze samples. This confirms input recovery; the new
palm alignment and response changes still require a live hardware acceptance check.

## Controls

| Input | Result |
|---|---|
| Left/right XR stick axes; X/A buttons | No robot movement in the default stationary mode |
| Grip held | Move/rotate that arm relative to its clutch calibration |
| Trigger | Close all fingers and oppose the thumb; open to release physical contact |
| B | Set the current headset orientation as forward |
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

1. Set your current head orientation as forward with **B**. Keep the trigger
   released, then **hold the grip** on the hand you want to move.
2. Move that controller forward, upward, or sideways and rotate it to guide the robot
   palm and wrist. Start with the 6 cm cube or cylinder on the near edge of the
   **small front-right table**. Watch the actual robot fingers enclose the object;
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
   on the near edge of the small front-right table. Optical tracking does not require
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
finger tracking requires the optical skeleton from the headset.
The stationary robot cannot walk to the distant side benches or lower its base to reach
the floor. The larger package/floor-object layout is retained in legacy assisted mode.

## Source layout

Paths in this table are relative to
`source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/`.

| File | Responsibility |
|---|---|
| `interactive/humanoid/humanoid_example.py` | Scene, input arbitration, arm/finger retargeting, pickup, camera, recorder, lifecycle |
| `interactive/humanoid/xr_pose.py` | Validated virtual-world poses and timestep-independent smoothing |
| `interactive/humanoid/eye_gaze_tracker.py` | Eye/head provenance, raycasts, latest sample, gaze visuals |
| `interactive/humanoid/material_highlights.py` | Shared reversible gaze and hand material overrides |
| `interactive/humanoid/grasp_controller.py` | Contact-limited finger targets and opposing-contact diagnostics |
| `interactive/humanoid/grasp_contacts.py` | Prepared PhysX hand-contact view, object identities, and contact health |
| `interactive/humanoid/arm_contact.py` | Projects arm commands against observed scenery-contact normals after smoothing |
| `interactive/humanoid/articulation_health.py` | Detects unusable measured joints before teleoperation and latches a pause |
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
is fitted to the per-joint correction cap and available joint travel first. Joints
already at a limit are excluded and the reach is solved again if their correction
would point farther outward. Wrist and posture corrections then use only the remaining
budget, each with a common scale that preserves its nullspace direction. They cannot
shrink the accepted primary step by forcing a rescale of the combined correction.

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
the projected command also replaces the smoothing cache. Graspable props and the
robot's own links are excluded from this arm constraint. Other external colliders
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

The default `robot_head` mode fixes the camera's **position** on the robot and applies
relative physical-headset rotation to the view. Left/right turns, pitch, and roll follow
the HMD; physical X/Y/Z translation is discarded. The G1 has no actuated neck joint:
its visual `torso_link/head_link` is fixed to the torso. This feature turns the camera,
not the head mesh or a physical neck.

`_get_robot_head_camera_pose()` remains the pure mount reference. It follows the
torso's full live position and rotation through physics tensors, including with Fabric
enabled. `_get_head_camera_pose()` applies `_apply_operator_head_rotation()` to that
reference. The existing `/World/G1_HeadCamera` path and eye offset are retained, so
the viewport and recorder share the same view. With Gf row-vector matrices:

```text
mounted = mount_local * body_world
relative_rotation = head_rotation * inverse(initial_head_rotation)
camera = relative_rotation * mounted
camera.translation = mounted.translation
```

The first valid physical HMD orientation defines neutral forward. OpenXR and USD
cameras both use local -Z forward and +Y up, so the mount already maps the relative
rotation into stage coordinates; no additional Y-up/Z-up conversion is applied.
Missing or invalid head rotation holds the last valid relative orientation while the
body mount continues updating. `_head_camera_operator_rotation_enabled` defaults to
`True`; disabling it retains the historical rigid view without sampling the HMD.

An application-frame subscription updates XRCore during Play and Pause, in addition
to the physics callback. **B** clears only the operator-view orientation reference;
the next valid pose becomes forward. It does not change eye-gaze calibration or the
legacy moving-mode calibration state. Reset/clear/load discard the new orientation
state along with stale camera handles. The left stick cannot enable room-scale motion
in stationary mode. Eye-gaze code and settings are unchanged.

For Steam Link, enable hand tracking on the headset and switch from controllers to
bare hands. Valve documents that Steam Link forwards the skeleton through
`XR_EXT_hand_tracking`. Controller aim/grip poses alone are insufficient for per-finger
control. [Valve's hand-tracking announcement](https://store.steampowered.com/news/posts/?appgroupname=SteamVR&appids=250820&enddate=1728412551&feed=steam_community_announcements)

## Pickup and tracking loss

### Experimental physical contact (default)

The current physical scene contains six lightweight objects on the nearest
front-right table. Paths below are under `/World/G1_SampleBoxes/`:

| Object root | Color / shape | World dimensions | Mass | World center X, Y (m) |
|---|---|---|---:|---|
| `Box_00` | Blue cube | 6 × 6 × 6 cm | 60 g | 0.345, -0.140 |
| `Box_01` | Orange cylinder | 6 cm diameter × 10 cm high | 80 g | 0.345, -0.360 |
| `Box_02` | Green sphere | 7 cm diameter | 50 g | 0.345, 0.100 |
| `Box_03` | Purple tall thin cylinder | 4 cm diameter × 14 cm high | 60 g | 0.465, 0.100 |
| `Box_04` | Yellow flat rectangular block | 9 × 6 × 3 cm | 70 g | 0.465, -0.065 |
| `Box_05` | Red cone | 7 cm diameter × 10 cm high | 50 g | 0.455, -0.245 |

All are dynamic bodies with collision and gravity. The original `Box_00`/`Box_01`
sizes and masses remain unchanged. Shape dimensions, extents, recording metadata,
and table placement must agree. The sphere, cube, and orange cylinder now form a
nearer row; the remaining shapes are staggered behind them. Center height is still
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

**Reset or reload the scene after a fault.** Pressing Play alone pauses again because
the fault latch remains set. Reset clears the latch, cached limits, contact handles,
and input targets while restoring the robot; a scene reload is appropriate when
changing coupling settings. The guard contains a failure rather than repairing
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
records requested/resolved locomotion mode, grasp mode, and grasp configuration.
The session's simulation clock stays monotonic across world resets; gait and velocity
filters clear so reset discontinuities do not become motion estimates.

## Performance and CPU physics

The defaults request **100 Hz CPU physics** and **90 Hz rendering**, using the Torch
backend. These are configured simulation intervals, not a guarantee of 90 displayed
frames per second or real-time simulation. The latest uninterrupted recorded interval
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
These are plausible contributors; their individual costs have **not been measured**
in the current setup.

| Work | Current behavior |
|---|---|
| Automatic recording | `_behavioral_data_enabled = True`; behavior, hand, gaze, and object rows are collected at about 100 Hz |
| Recorded camera | An additional 256×256 camera render product and RGB annotator refresh at render cadence, even though PNGs are saved at about 10 Hz |
| File writes | PNG encoding and saving happen synchronously inside the physics callback; buffered CSVs are also written there about every 2.5 seconds |
| Camera attachment | The body pose is read and the mounted view scheduled every physics tick and every application frame, including while paused |
| Arm and pickup poses | Inspire arm IK and candidate selection use a cached rigid palm offset; detailed grasp-distance checks still read finger links. Both arm solves share Jacobian and joint-position reads within one physics tick |

Finger landmarks are cached within each hand's curl calculation. Joint command
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

The current live profiling attempt was blocked because the Python server was
unavailable. No measured subsystem timings or post-change speedup are claimed.
A refused connection on port 8226 indicates server availability, not a robot or
headset diagnosis. The recorded 0.271 real-time factor above establishes slow
simulation progress but does not isolate its cause.

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

The current suite passed **242 offline tests** in 6.625 seconds using Isaac Sim 6.1
Python. The log is `_compat61/stop-fix-tests.log`. Coverage includes first-contact
caps, per-body isolation, short gaps, opening/reset, final arm-contact projection,
input-loss arm holding, finite-coupling preparation, and articulation-health checks.
Black, isort, and pyflakes also passed on the 23 changed Python files. Coverage includes
the armature floor and preserved authored/per-axis values. Offline tests do not
establish collision dynamics or headset accuracy, and earlier rigid-coupling pickup
passes do not validate the new mechanism. See the
[validation record](validation/isaac-sim-6.1.md) for live configurations and results.

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

Run these checks sequentially against the already-loaded stationary example:

```powershell
python tools/validate_camera_live.py
python tools/validate_fingers_live.py --timeout 300
```

The current camera contract permits relative HMD rotation while its position remains
at the robot mount during Play and Pause. Check left/right direction, pitch/roll,
physical-translation rejection, B recentering, and preservation of the gaze tracker.
The earlier constant-full-transform replay described below tested the historical
fully fixed mode; those results do not validate the current rotation-only behavior.
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
SteamVR independently. In the latest recorded session ending `02-02-51`, the last
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
   position must stay at the robot mount. Turn left/right and tilt up/down or in roll:
   the view must rotate in the same direction. Repeat while paused. Press **B** to
   make your current orientation forward; the left stick must not enable room-scale
   camera motion. The visible G1 head remains fixed to the torso.
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
robot or headset. The earlier assisted stationary path was exercised in the running
VR application with synthetic XR inputs and real PhysX, as described above.
Headset hardware tracking and optional walking modes require separate live checks.

## Later milestones

After physical pickup and input latency are validated, a separate RGB-webcam helper
could estimate shoulders and elbows with pretrained MediaPipe Pose Landmarker. Keep
Quest fingers, hand targets, and eyes authoritative; use webcam posture only as a
secondary arm objective under the hand task, with the base stationary. Its 33-landmark
output includes hip-relative estimated 3D coordinates, not a calibrated camera-to-XR
transform. Calibrate frames and arm proportions, align timestamps, reject low-confidence
or stale observations, and fade back to the current posture prior when unavailable.
The live-stream API can skip incoming frames while busy, so measure end-to-end latency
and consume the latest result. [MediaPipe Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)

First verify that a headset-wearing operator can be tracked: the official model card
lists a head outside the image and metric-accurate depth outside its intended scope.
Headset occlusion is therefore a feasibility risk. An ordinary RGB webcam is a low-cost
prototype; a depth camera adds measured visible-surface depth but still needs calibration
and cannot observe a hidden elbow. No webcam capture or fusion is implemented here.
[BlazePose GHUM model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf),
[RealSense projection/deprojection](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)

A later study could use hand-closing velocity and gaze at the same object to infer a
grasp-intent or gentle-to-firm request. These cues do not measure operator pressure.
Evaluate them against fixed-grip and speed-only baselines using actual contact, slip,
drop, and release outcomes before adding adaptation. No gaze/velocity intent model or
training pipeline is implemented; the current contact diagnostic does not alter gaze.

`g1_grasp_env.py` is an experimental benchmark scaffold. It does not provide a complete
training environment with step/reward/reset semantics, and this repair does not claim
to train or integrate a new grasp or whole-body policy.

## API references

- [PhysX tensor Jacobians](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html): world-frame Jacobians and fixed/floating root layout.
- [USD joint frames](https://openusd.org/release/api/class_usd_physics_joint.html): local position/rotation for each connected body.
- [Physics scene queries](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/physics_umbrella/physics_umbrella_runtime.html): all-hit raycast callback and query lifecycle.
