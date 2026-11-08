# Overview

The isaacsim.robot.policy.examples extension provides interactive robot demonstrations
and reusable policy controllers for Isaac Sim. This fork extends the Humanoid example
with Unitree G1 VR teleoperation, physical manipulation, optional walking, and local
behavioral recording. The Franka, quadruped, and standalone policy examples remain
available as separate demonstrations.

## Key Components

### Humanoid Teleoperation

**{class}`HumanoidExample <isaacsim.robot.policy.examples.interactive.humanoid.HumanoidExample>`
coordinates the G1 scene, input ownership, arm/hand control, view, and recording.**
Launch configuration selects the control profile before the scene is built:

- Stationary teleoperation anchors the pelvis and suppresses base movement inputs.
- Controller walking uses physical balance/walking policies, with stick input for
  translation and turning and controller or optical input for manipulation.
- Experimental pointing navigation uses an optical left-index ray to select a
  checked destination and requires the pointing gesture to remain active.
- Eye-gaze buttons use tracked combined-eye input. The default `G1_GAZE_MODE=hold`
  activates immediately on a button hit, including a quick glance, and continues while the same button is
  fixated; looking away or losing valid eye tracking requests a stop. Preview
  anchors the base; explicit movement mode uses the existing walking policy.
  Optional `step` mode retains a 0.7-second dwell and bounded steps/turns that finish
  without fixation. An explicit positive `G1_GAZE_DWELL_S` adds a delay/progress ring
  to either mode; the default zero-delay hold mode has no dwell ring.

These profiles share the arm, finger, contact, and recording pipeline; switching
between anchored and walking configurations requires rebuilding the scene. They
use different physics configurations, so a requested sensor or render rate is not
a promise of real-time performance.

The example accepts Quest Pro controller poses and OpenXR optical hand landmarks.
Controller grip acts as an arm clutch and trigger controls finger closure. Valid
optical landmarks drive the corresponding arm without a clutch; finger curls and
thumb opposition drive the six Inspire actuators per hand. Current physical
grasping uses opposing contacts and friction, with contact-limited finger targets
and arm obstruction guards. The retained assisted fixed-joint pickup path is not
the current default and is not evidence that a physical grasp succeeds.

The robot-mounted view follows tracked head orientation through a neck/eye arc
and comfort stabilization. Room-scale headset translation does not reposition
the robot-head view. The visible operator-head proxy and near-camera visibility
handling are visual features, not extra physical robot joints. The eye-gaze
visualization retains its source-labelled head-forward fallback, but only the
currently tracked combined-eye source can activate the gaze walking controls.
Side-panel meshes appear in the headset through ordinary stereo scene rendering.
The gaze profile widens the desktop camera lens to fit them; native headset
projection remains owned by the runtime.

The recorder writes robot state/commands, hand inputs, gaze, object states, and
RGB images. CSV output uses an asynchronous writer with bounded queues; RGB image
capture/saving remains in the sample's collection path. Simulation timestamps
support alignment but do not establish hardware-synchronized sensor capture or
the original rendering time of the latest image. Source validity, requested
controls, applied finger targets, and contact evidence must be interpreted using
their recorded fields rather than treating every row as a successful action.

The OpenXR component setting
`/xr/openxr/components/omni.kit.xr.openxr.ext.hand_tracking/enabled` requests skeletal
hand tracking when the XR instance starts. The headset and active runtime must also
provide valid landmarks. A connected headset or controller poses alone do not prove
that optical tracking is available. Generated-input replay checks do not verify
headset tracking quality, occlusion recovery, or walking with a carried payload.

**{class}`G1TeleopRobot <isaacsim.robot.policy.examples.robots.G1TeleopRobot>`
provides the robot articulation and joint-control interface used by the example.**
It owns articulation initialization, joint-name mappings, drives, limits, and
profile-specific body control. The default stationary mode does not run a walking
policy. Physical walking uses the pinned decoupled balance/walk actors through
`G1DecoupledPolicy`: those actors command the legs and waist while the teleoperator
retains arm targets. Strict physical walking fails on unavailable policy/physics
requirements instead of substituting the retained legacy kinematic motion path.

The sample applies input validity checks and command limits before robot output.
An articulation fault clears motion requests and pauses the timeline; resuming
Play alone does not clear a latched fault. Scene reset is the recovery boundary.

### Custom Humanoid Modules

The `interactive.humanoid` package contains this fork's implementation boundaries:

- Input and view helpers normalize XR poses, interpret controller/pointing/gaze inputs,
  and compute the robot-head camera and visual head transforms.
- Contact and grasp helpers interpret PhysX evidence and constrain requested
  movement; they do not replace physical contact with a successful-grasp label.
- Scene helpers author practice objects, shelf cartons, navigation queries, and
  preauthored highlight visuals.
- Recording helpers write local CSV streams without controlling the robot.

The main sample integrates these helpers with Kit lifecycle callbacks. Retained
gesture and assisted-control experiments are documented separately from the
profiles selected by the current launchers. The standalone EEG experiment under
`tools/eeg` does not send movement commands to this extension.

### Other Interactive Examples

**{class}`FrankaExample <isaacsim.robot.policy.examples.interactive.franka.FrankaExample>`
demonstrates learned drawer opening with a Franka Emika Panda.**
The example repeatedly executes and resets the manipulation task.

**{class}`QuadrupedExample <isaacsim.robot.policy.examples.interactive.quadruped.QuadrupedExample>`
demonstrates flat-terrain locomotion with Boston Dynamics Spot.**
Keyboard input supplies forward, lateral, and yaw commands.

**{class}`Go2Example <isaacsim.robot.policy.examples.interactive.go2.Go2Example>`
demonstrates flat-terrain locomotion with a Unitree Go2.**
Keyboard input controls the commanded velocity and direction. These examples retain
their own physics and rendering configurations; the G1 CPU settings do not describe
every example in this extension.

### Policy Controllers and Configuration

**{class}`PolicyController <isaacsim.robot.policy.examples.controllers.PolicyController>`
provides the shared framework for loading and executing robot policies.**
It manages robot initialization and policy-based joint control. Standalone controllers
remain available for other tasks:

- {class}`AnymalFlatTerrainPolicy <isaacsim.robot.policy.examples.robots.AnymalFlatTerrainPolicy>`
- {class}`Go2FlatTerrainPolicy <isaacsim.robot.policy.examples.robots.Go2FlatTerrainPolicy>`
- {class}`SpotFlatTerrainPolicy <isaacsim.robot.policy.examples.robots.SpotFlatTerrainPolicy>`
- {class}`H1FlatTerrainPolicy <isaacsim.robot.policy.examples.robots.H1FlatTerrainPolicy>`
- {class}`FrankaOpenDrawerPolicy <isaacsim.robot.policy.examples.robots.FrankaOpenDrawerPolicy>`

**{func}`parse_env_config <isaacsim.robot.policy.examples.controllers.parse_env_config>`
loads YAML environment configuration for policy deployment.**
The associated configuration helpers interpret joint properties, articulation and
physics settings, observations, and actions.

## Integration

The extension registers interactive demonstrations in the Isaac Sim examples browser
under **Policy**. Select **Humanoid**, load the scene, and start simulation for G1
teleoperation. XR must be active for controller or optical hand input.

The extension uses the Isaac Sim sample framework, experimental articulation APIs,
asset storage, and the examples browser. VR operation additionally depends on the
XR experience and a working OpenXR runtime. The repository README and humanoid guides
provide installation, control, recording, troubleshooting, and validation instructions.
