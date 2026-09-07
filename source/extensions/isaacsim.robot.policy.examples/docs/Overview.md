# Overview

The isaacsim.robot.policy.examples extension provides interactive robot demonstrations
and reusable policy controllers for Isaac Sim. This fork adds stationary Unitree G1
VR teleoperation to the Humanoid example while retaining the Franka, quadruped, and
standalone locomotion policy examples.

## Key Components

### Humanoid Teleoperation

**{class}`HumanoidExample <isaacsim.robot.policy.examples.interactive.humanoid.HumanoidExample>`
provides stationary arm and hand teleoperation for a Unitree G1 with Inspire hands.**
The pelvis is fixed to the world, and walking and turning inputs are suppressed.
The current configuration uses CPU physics with a 100 Hz simulation step and a
requested 90 Hz rendering cadence. These configured rates do not guarantee real-time
performance; rendering, recording, and control work also consume frame time.

The example supports Quest Pro controller poses and OpenXR optical hand landmarks.
Controller grip acts as an arm clutch, and trigger controls finger closure and
assisted pickup. Valid optical hand poses drive the arm without requiring a fist;
individual finger curls and thumb opposition drive the six Inspire actuators per
hand. Closing and opening the hand requests pickup and release. Pickup attaches a
nearby object with a fixed joint rather than relying solely on finger contact.

The camera maintains a fixed transform relative to the robot body at head height,
including while paused. Physical headset translation and rotation do not move the
camera in stationary mode. Existing gaze tracking and its recording remain active.
The session recorder writes robot, hand, gaze, object, and camera data for later
analysis.

The OpenXR component setting
`/xr/openxr/components/omni.kit.xr.openxr.ext.hand_tracking/enabled` requests skeletal
hand tracking when the XR instance starts. The headset and active runtime must also
provide valid landmarks. A connected headset or controller poses alone do not prove
that optical tracking is available. Real Quest landmark delivery remains an active
validation issue; generated-input replay checks do not verify headset tracking quality.

**{class}`G1TeleopRobot <isaacsim.robot.policy.examples.robots.G1TeleopRobot>`
provides the robot articulation and joint-control interface used by the example.**
It configures the standing posture, arm and hand drives, joint limits, and stationary
base. The Humanoid example does not run a walking policy in its default mode.

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
