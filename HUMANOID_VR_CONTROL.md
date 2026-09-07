# Humanoid VR Control — Unitree G1 with Dexterous Hands, Quest Pro Eye Tracking & Behavioral Data Recording

> **Fork of:** [NVIDIA IsaacSim](https://github.com/isaac-sim/IsaacSim) · Isaac Sim 6.0.0 GA  
> **Files modified/added:** `.../interactive/humanoid/humanoid_example.py` · `eye_gaze_tracker.py` (same folder) · `.../robots/g1.py` (new)  
> **Author:** [@soheilAppear](https://github.com/soheilAppear)

---

For the current implementation, controls, tests, and live acceptance checks, see the
[Humanoid control developer guide](docs/humanoid-control.md). Historical measurements
and headset captures below describe earlier sessions; they have not been revalidated
against these changes. Offline tests cover control logic and USD construction, not
headset accuracy, dynamic balance, or contact-based grasp success.

**Current default: stationary manipulation.** The robot's pelvis is fixed to the world
at its spawn pose. It does not walk, turn, or respond to locomotion inputs, including
headset gait. Arm/finger tracking, pickup, and gaze remain active. The camera is rigidly
mounted to the robot's head: physical headset translation and rotation cannot move it.
B restores that fixed view, and the left-stick click cannot unlock it in stationary mode.
Optical tracking independently drives each digit's curl and thumb opposition; controllers
retain grip for arm tracking and trigger for whole-hand closure. The launcher requests
the native OpenXR hand-joint component without changing gaze settings. Restart an
already-running XR session once after updating to activate it.
Save your stage and restart/reload the example to replace an existing walking instance.
The walking measurements and tuning sections below are retained as historical,
non-default experiments; they do not describe the current stationary setup.

## What This Is

This fork replaces the stock Isaac Sim H1 humanoid example with the **Unitree G1 fitted with
Inspire five-finger hands** — the Unitree humanoid that [Isaac Teleop](https://github.com/NVIDIA/IsaacTeleop)
drives for dexterous manipulation — and wraps it in a full **VR-driven teleoperation and
data-collection pipeline**:

| Feature | Description |
|---------|-------------|
| **Stationary manipulation** *(default)* | A fixed world-to-pelvis joint supports the standing robot; all base movement commands are suppressed while arms, hands, and gaze remain active |
| **Headset gait detection** *(optional moving modes only)* | Experimental peak/trough processing on HMD height; disabled by default and cannot move the robot in stationary mode |
| **Horizontal motion gate** | Prevents false walking from pure head nodding — gait is suppressed unless the headset also moves in the floor plane |
| **Eye-level first-person camera** | Fixed to the live robot head/torso body; physical headset movement is ignored in stationary mode |
| **VR hand tracking → arm control** | OpenXR hand/controller poses drive the G1 arm joints |
| **Finger teleoperation** | Independent per-finger bend and thumb opposition from the OpenXR skeleton drive the Inspire hand's six actuators; with controllers, grip clutches the arm and trigger curls all fingers |
| **Grab system** | Nearby objects use distance-gated fixed-joint grasp assistance: hold grip and pull trigger, or close tracked fingers |
| **Quest Pro eye tracking** | Real OpenXR eye gaze (the runtime's calibrated fusion of both eyes) drawn as a **red ray** from your eyes to the gazed collider (sample boxes, ground) with a **large blood-red marker sphere** at the collision point; the looked-at box is tinted yellow; every gaze-target change is printed live to the terminal as `[EyeGaze] looking at ...`; robot self-hits filtered out of the raycast (`eye_gaze_tracker.py`) |
| **Behavioral session recorder** | Every run creates a session folder under `~/BehavioralCollection/raw_sessions/` with `metadata.json` + five time-aligned ~100 Hz CSVs (behavior, hand tracking, gaze, object states, frame timestamps), flushed to disk every ~2.5 s during play |
| **Eye-camera frame capture** | First-person 256×256 PNG frames at ~10 Hz, timestamped for video–sensor sync |
| **Gaze logging with fallback** | `gaze.csv` uses real eye tracking when available, else HMD-forward direction — tagged per row via `gaze_source` |
| **Learning pipeline scaffold** | `learning/` folder for the V-JEPA world-model pipeline (data sync → baselines → multimodal predictor → planner) |

### Stationary mode and optional walking experiments

The interactive example uses this setting in `humanoid_example.py`:

```python
self._g1_locomotion = "stationary"  # world-fixed pelvis; arms and fingers remain controllable
```

This mode does not run the locomotion actor. Robot-link gravity is disabled; scene
objects retain gravity. A physics constraint holds the base in
place even while the arms move or hold an object, so manipulation does not depend on
the walking policy balancing a changed arm pose or payload. Keyboard movement keys,
stick axes, X/A turning buttons, gamepad locomotion, and headset gait are ignored for
base motion. **B**, **Y**, grip/trigger input, and the left stick camera-mode click
remain available.

`policy` and `kinematic` are explicit alternatives for later experiments. The policy
description and measurements below refer to earlier moving-mode sessions, not a
validation of the current stationary setup.

Isaac Sim ships a trained flat-terrain locomotion policy for the H1, but **none for the
G1** (`/Isaac/Samples/Policies/` contains `h1`, `go2`, `spot`, `anymal` and Franka only,
in 4.5, 5.0 and 6.0 alike). Unitree, however, publishes one for their own robot, and this
fork uses it: [`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym)'s
`deploy/pre_train/g1/motion.pt` (BSD-3-Clause), vendored here as
`robots/data/g1_unitree_motion.pt`.

It is an **LSTM actor** — `LSTM(47 → 64)` into an `MLP(64 → 32 → 12)` — trained in Isaac
Gym and running here at 50 Hz. Two properties make it a good fit:

- **It drives the legs only.** The waist and both arms stay free for teleoperation,
  instead of being owned by the balance controller the way the H1's whole-body policy
  owned them. Your arms and fingers do not fight the gait.
- **It transfers.** Measured in Isaac Sim on the 53-DOF G1 with Inspire hands: 2.69 m
  travelled on a 0.5 m/s command over 5 s, pelvis height 0.75–0.78 m, tilt never above
  8.5°, and left/right ankle correlation of **−0.62** — the feet genuinely alternate
  rather than shuffle. It turns (+113° on a 0.6 rad/s command over 4 s) and stands still
  without falling.

Optional moving modes in `humanoid_example.py`:

```python
# self._g1_locomotion = "policy"     # optional Unitree walking policy
# self._g1_locomotion = "kinematic"  # optional commanded base glide
```

`kinematic` is the policy-load fallback: the robot holds its standing
posture, gravity is disabled on its links, and the base is integrated straight from the
`(vx, vy, wz)` command. It permits motion and does not provide the default stationary
mode's fixed world anchor.

**Things that will bite you if you change this code:**

- **The joint order is not the obvious one.** The policy expects the training URDF's
  order (all of the left leg, then all of the right); PhysX reports DOFs by tree depth
  (`left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, …`). The mapping comes out
  as `[0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18]` — resolved by *name* in
  `_configure_walk_policy`. Index it positionally and the robot falls instantly.
- **The policy is stateful.** Its LSTM memory lives inside the scripted module, so the
  same observation twice gives two different actions. It must be called exactly once per
  50 Hz tick, in order, and its hidden and cell state zeroed on reset (`post_reset`).
- **The gains are part of the policy.** It learned against Unitree's PD values
  (hip 100/2, knee 150/4, ankle 40/2), so those are applied in policy mode rather than
  the Isaac Lab ones used for the kinematic posture hold.
- Commands are clamped to the range it was trained within — [0.8, 0.5, 1.57] in m/s, m/s
  and rad/s. Asking for more is a reliable way to make it fall.

If you would rather train your own — for rough terrain, or to include the arms — the task
is `Isaac-Velocity-Flat-G1-v0` in Isaac Lab; budget roughly 150M steps (~1500 iterations
× 4096 envs × 24), which is on the order of half an hour on an RTX 5090, plus however
many runs of reward tuning it takes.

---

## Prerequisites

| Requirement | Version / Notes |
|-------------|-----------------|
| **NVIDIA Isaac Sim Standalone** | 6.0.0 GA (Windows x64 or Linux x64) |
| **GPU** | RTX 4080 minimum, RTX 5080+ recommended |
| **VR headset** *(optional)* | Any OpenXR-compatible HMD (Meta Quest via Link, SteamVR, etc.) |
| **CUDA** | 12.x (bundled with Isaac Sim) |
| **Python** | 3.12 (bundled with Isaac Sim) |

---

## Repository Structure

```
IsaacSim-HumanoidBehavior/
├── source/
│   └── extensions/
│       └── isaacsim.robot.policy.examples/
│           └── isaacsim/robot/policy/examples/
│               └── interactive/humanoid/
│                   ├── humanoid_example.py   ← the modified file
│                   └── eye_gaze_tracker.py   ← Quest Pro eye-gaze module (new)
├── learning/                                 ← V-JEPA / world-model learning pipeline
├── README.md                                 ← NVIDIA's original readme
└── HUMANOID_VR_CONTROL.md                    ← this file
```

---

## Installation — How to Apply

You have three options. Pick the one that fits your workflow.

---

### Option A — Copy the extension

Copy the complete extension so the robot wrapper, bundled policy, registration, and
shared XR/highlight helpers remain compatible. Keep a backup of the installed
extension before replacing its files. Run from the repository root.

**Windows:**

```powershell
$isaacRoot = "C:/path/to/isaac-sim-standalone-6.0.0-windows-x86_64"
$sourceExt = "source/extensions/isaacsim.robot.policy.examples"
$installedExt = Join-Path $isaacRoot "exts/isaacsim.robot.policy.examples"
Copy-Item -Path "$sourceExt/*" -Destination $installedExt -Recurse -Force
```

**Linux:**

```bash
ISAAC_ROOT="/path/to/isaac-sim-standalone-6.0.0-linux-x86_64"
cp -a source/extensions/isaacsim.robot.policy.examples/. "$ISAAC_ROOT/exts/isaacsim.robot.policy.examples/"
```

Restart Isaac Sim after saving your session to load the changed Python modules.

---

### Option B — Symlink (best for active development)

The standalone automatically uses this repo's file — no copy step needed after edits.

**Windows (run as Administrator):**
```powershell
$standaloneExt = "C:\path\to\isaac-sim-standalone-6.0.0-windows-x86_64\exts\isaacsim.robot.policy.examples"
$githubExt     = "C:\path\to\IsaacSim-HumanoidBehavior\source\extensions\isaacsim.robot.policy.examples"

Rename-Item $standaloneExt "${standaloneExt}.orig"          # keeps original as backup
New-Item -ItemType Junction -Path $standaloneExt -Target $githubExt
```

To undo:
```powershell
Remove-Item $standaloneExt
Rename-Item "${standaloneExt}.orig" $standaloneExt
```

**Linux:**
```bash
STANDALONE_EXT="/path/to/isaac-sim-standalone-6.0.0-linux-x86_64/exts/isaacsim.robot.policy.examples"
GITHUB_EXT="/path/to/IsaacSim-HumanoidBehavior/source/extensions/isaacsim.robot.policy.examples"

mv "$STANDALONE_EXT" "${STANDALONE_EXT}.orig"
ln -s "$GITHUB_EXT" "$STANDALONE_EXT"
```

---

### Option C — Extension folder override (no changes to standalone)

```powershell
# Windows — VR mode
cd "C:\path\to\isaac-sim-standalone-6.0.0-windows-x86_64"
.\isaac-sim.xr.vr.bat --ext-folder "C:\path\to\IsaacSim-HumanoidBehavior\source\extensions"

# Windows — desktop mode
.\isaac-sim.bat --ext-folder "C:\path\to\IsaacSim-HumanoidBehavior\source\extensions"
```

```bash
# Linux — VR mode
cd /path/to/isaac-sim-standalone-6.0.0-linux-x86_64
./isaac-sim.xr.vr.sh --ext-folder /path/to/IsaacSim-HumanoidBehavior/source/extensions
```

> **Note:** Kit loads the extension it finds first. If versions match, results may be inconsistent. Prefer Option B for development.

---

## Running

Every command below is run from the Isaac Sim install directory:

```powershell
cd "C:\path\to\isaac-sim-standalone-6.0.0-windows-x86_64"
```

### Option 1 — the full example, with UI (keyboard, VR, gaze, data recording)

```powershell
.\isaac-sim.bat            # desktop / keyboard
.\isaac-sim.xr.vr.bat      # VR headset (start SteamVR first)
```

Once Isaac Sim is open:
1. **Window -> Examples -> Robotics Examples**, then **Policy -> Humanoid: Unitree G1**
2. Click **LOAD**, then **Play**

This is the path that records behavioural sessions, draws the gaze ray, and runs the
first-person camera. Everything else in this guide applies to it.

### Option 2 — straight from the terminal, no clicking

`g1_standalone.py` spawns the G1, walks it and cycles its hands, with no UI to navigate:

```powershell
# with a window; arrow keys / numpad walk, SPACE toggles the fists
.\python.bat "C:\path\to\IsaacSim-HumanoidBehavior\source\standalone_examples\api\isaacsim.robot.policy.examples\g1_standalone.py"

# no window at all - scripted walk + turn + fists; good for a quick check or over SSH
.\python.bat "...\g1_standalone.py" --headless --seconds 20

# the can't-fall glide mode instead of the walking policy
.\python.bat "...\g1_standalone.py" --locomotion kinematic

# the three-finger Dex3 hand instead of the Inspire five-finger
.\python.bat "...\g1_standalone.py" --hand ThreeFinger
```

Flags: `--headless`, `--seconds N`, `--locomotion policy|kinematic`,
`--hand Inspire|ThreeFinger|None`, `--device cuda|cpu`, `--test`.

> Installed by junction (Option B above)? Both paths pick up your repo edits directly -
> only an Isaac Sim restart is needed, never a reinstall.

---

## Controls

These controls describe the default **stationary** interactive example. Its base stays
fixed; only the arms/fingers and view respond to the manipulation controls below.

### Keyboard
| Key | Action |
|-----|--------|
| `↑` or `Numpad 8` | No base movement in stationary mode |
| `↓` or `Numpad 2` | No base movement in stationary mode |
| `←` or `Numpad 4` | No base movement in stationary mode |
| `→` or `Numpad 6` | No base movement in stationary mode |

> **This was broken and nothing said so.** `event.input` is documented as a
> `carb.input.KeyboardInput` enum, but this Kit build delivers a plain string, so
> `event.input.name` raised `AttributeError: 'str' object has no attribute 'name'` on
> **every keypress** — the handler died before doing anything, and the whole keyboard
> path was dead. It looked like nothing at all: the traceback goes to the log rather than
> the screen, and the VR controls kept working. Found in the session log, not by testing.
> `_keyboard_event_key()` now accepts both shapes.

### VR Controller

Hold a grip to move and rotate that arm. Keep the trigger released while reaching,
then pull it to close the fingers and request pickup. XR triggers are reserved for manipulation.

| Input | Action |
|-------|--------|
| Left stick forward / back | No base movement in stationary mode |
| Left stick left / right | No base movement in stationary mode |
| Right stick left / right | No base movement in stationary mode |
| X / A | No base movement in stationary mode |
| Left / right grip | Clutch that arm; releasing cancels tracking and releases its object |
| Trigger pressure | Curl all fingers on that controller's hand |
| Trigger at or above 0.60 | Request pickup if the active robot hand is close enough |
| Trigger at or below 0.35 | Release pickup and rearm the next grab |
| B | Restore the fixed robot-head view |
| Y | Drop both objects; open/release before grabbing again |
| Left stick click | Keep the camera locked to the robot head in stationary mode |

**Controller pickup sequence:** press **B** to restore the robot-head view.
With trigger released, **hold grip**, then move and rotate the controller
to bring the actual robot palm/fingers close to an object on the near edge of the
**small front-right table**. **Keep grip held and squeeze
trigger** past 60% to close and request attachment. Move the controller while keeping
both grip and trigger held to lift/carry. Ease trigger below 35% to drop. **Y** drops
both hands; release trigger before the next squeeze. Releasing grip also ends that
arm's tracking and drops its object. Grip alone leaves the fingers open.

### Optical hand tracking

Use the runtime's hand-tracking mode with your hands visible to its tracking cameras.
No controller grip button is needed: valid wrist/palm poses drive the arm and the
finger skeleton drives curls.

1. Start with an **open hand**, then reach and rotate it to guide the robot's actual
   palm toward a near-edge object on the small front-right table.
2. **Close your fingers into a fist** near the object. Mean curl at or above 0.55
   requests assisted pickup.
3. Move your closed hand to lift/carry. **Open your hand** to release; mean curl at or
   below 0.35 releases the object.
4. If tracking is lost, targets clear, objects release, and driven fingers open. Open
   your hand before closing again after a tracking-loss drop. Reacquisition is limited
   from the robot's measured joint positions.

Both hands work independently. Objects outside the robot's physical reach cannot be
picked up from a distant target marker or by gaze alone.

The runtime must expose the relevant bindings. Use `EX.describe_xr()` and its device
inventory to diagnose missing inputs. A visible target marker is a requested pose;
pickup distance is always measured from the actual simulated hand.

### Gamepad
| Input | Action |
|-------|--------|
| Left stick up / down | No base movement in stationary mode |
| Right stick left / right | No base movement in stationary mode |
| Right trigger | No base movement in stationary mode |
| X / A buttons | No base movement in stationary mode |

### Optional moving modes: how a keypress becomes a step

The following applies only after explicitly choosing `policy` or `kinematic`.
Stationary mode suppresses the base command regardless of these input mappings.
In moving modes, keyboard Up requests forward, Left/Right request yaw, and Down
brakes translation. XR left-stick axes command forward/reverse and strafe; the right
stick and X/A command yaw. Gamepad fallback also accepts a right-trigger forward input.

Locomotion input lands in the same place: a single base velocity command
`(v_x, v_y, w_z)` in m/s and rad/s. Keyboard, VR controller and gamepad all just write
into it, and it is smoothed and clamped before use.

In `policy` locomotion that command becomes `obs[6:9]` of the walking policy's
observation (scaled by `[2.0, 2.0, 0.25]`). The policy answers with 12 leg-joint
position targets at 50 Hz, and the PhysX PD drives turn those into joint torques. So
"forward" is not a translation applied to the robot — the robot is asked to *walk* at
that speed and the gait emerges from the policy. In `kinematic` mode the same command is
integrated straight into the base pose instead.

Command limits: forward 1.0 m/s and yaw 1.0 rad/s as requested, then clamped by the
robot to the range the policy was trained within — 0.8 m/s forward, 0.5 m/s lateral,
1.57 rad/s yaw. Lateral strafing (`v_y`) is supported by the policy but is not bound
to any key.

XR reverse retains the project's capped on/off policy command because earlier
sessions showed poor low-speed reversing. These measurements do not establish safe
balance, reverse, or stopping behavior under new loads or arm poses. Keyboard Down
brakes translation while permitting yaw. XR triggers remain dedicated to manipulation.

> **Head motion does not move the robot.** Headset gait is off
> (`_headset_gait_enabled = False`) and, while disabled, its output is excluded from the
> command mix entirely rather than merely being zero. Your HMD pose is still read every
> step, but only for `sim_time`, the `hmd_*` columns in `behavior.csv`, and the
> HMD-forward gaze fallback.

### Optional moving modes: VR Headset Gait (step-in-place walking)

> **Disabled by default** since 2026-07 while step detection is stabilized. Re-enable with
> `self._headset_gait_enabled = True` in `HumanoidExample.__init__()` only after deliberately
> choosing a moving locomotion mode. Enabling this flag cannot move stationary mode. The HMD pose is
> still read every step regardless, so `behavior.csv` and the gaze fallback keep working.

Bob your head up and down rhythmically at ~1 Hz (like walking in place). The system:
1. Low-pass filters the headset height signal
2. Detects peaks and troughs (amplitude ≥ 1.2 cm)
3. Validates that the headset is also moving in the floor plane (≥ 2.5 cm/s horizontal speed) — this prevents accidental triggering from pure nodding or breathing
4. Fires a 0.5 s forward-walking pulse on each validated step

**Tips:**
- Exaggerate the head bob slightly on first use until the baseline calibrates (~1–2 s)
- Turning while gait-walking: use A/X buttons for yaw simultaneously
- The robot matches your cadence — faster bobbing = more frequent step pulses

---

## The Warehouse Scene

The robot works in `/Isaac/Environments/Simple_Warehouse/full_warehouse.usd` — NVIDIA's
dressed industrial scene, with racking, pallets, stacked boxes and forklifts already in
place. Isaac ships **no "factory" environment**; `Props/Factory/` is bolts and nuts for
the assembly tasks, so the warehouse is the closest thing to a real facility.

```python
self._environment_usd_path  = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
self._environment_is_warehouse = True     # also dims the fill lighting, see below
self._robot_spawn_xy = (0.0, 0.0)         # clear floor in the central aisle
```

### Pickable packages

Start stationary pickup practice at the **small front-right table**, positioned at
`(x=0.50, y=-0.28, yaw=90°)` relative to the scene origin. Its near-edge objects are
intended for reaching from the spawn pose. The distant side benches and floor objects
remain in the scene; stationary mode cannot walk to them or squat to reach the floor.

Current props use the smaller YCB objects:

```python
self._package_usd_paths = (
    ".../YCB/Axis_Aligned/005_tomato_soup_can.usd",
    ".../YCB/Axis_Aligned/010_potted_meat_can.usd",
    ".../YCB/Axis_Aligned/061_foam_brick.usd",
    ".../YCB/Axis_Aligned/004_sugar_box.usd",
    ".../YCB/Axis_Aligned/006_mustard_bottle.usd",
)
self._package_spawn_radius = (1.2, 3.0)  # floor/fallback placements, not tabletop reach
self._package_use_props = True    # False -> procedural cubes
```

They keep the `Box_<nn>` names under `/World/G1_SampleBoxes` whatever their geometry,
because three separate systems key off that: the grab search, `object_states.csv`, and
the eye tracker's highlight filter.

**How picking works.** Close the tracked hand near a package, or hold controller grip
and squeeze trigger, to request a fixed-joint attachment. Open the hand/release trigger
to drop it. The fingers articulate, but the attachment supports the object rather than
finger friction alone. A highlight can appear before the actual hand is close enough
to attach. See the distance gates below.

### Historical grasp and reach investigations

The following measurement narratives describe earlier scene/controller revisions.
Use the controls above and the current fixed-joint distance table below for the
stationary setup; the earlier walking results are not current acceptance evidence.

#### Grabbing: why it used to miss

The grab search was centred on the **arm-rig target** — where teleoperation *asked* the
hand to go — rather than on the hand itself. Because the arm is posed by a heuristic and
not an IK solve, those two drift apart. Measured across four reach poses:

| | Before | After |
|---|---|---|
| Grab point vs actual hand | 0.10-0.27 m (mean ~0.19) | **0.000 m** |
| Grab radius | 0.30 m | 0.22 m |

The targeting error was almost as large as the search radius, so which crate you got was
decided more by error than by aim. The grab now reads the real hand link through the
physics tensor API (`_get_hand_link_prim`), and the radius is tightened to 0.22 m since it
no longer has to absorb that error.

### The arm mapping was inverted, and that is why reaching felt impossible

The measurements in this section describe the earlier four-joint mapping. The current
default uses a live seven-joint PhysX Jacobian with a palm offset. Palm position is the
primary IK task; wrist orientation uses only its remaining nullspace, and a small
posture term keeps redundant joints near their standing defaults. This prevents an
unreachable wrist orientation from steering the palm away from its reach target.
Joint commands slew at up to 2.5 rad/s relative to the previous target, with a separate
0.15 rad bound relative to the measured joint position. This gives the position drive
room to follow a moving hand while bounding error against an obstacle. The measured
bound takes priority if external contact makes the limits conflict. Switching input
source or losing tracking clears smoothing; reacquisition starts from measured joints.
See [the current retargeting guide](docs/humanoid-control.md#arm-and-finger-retargeting).

The note above used to end by saying a real IK solver was "the next thing worth doing".
It was, and measuring it first showed the mapping was not merely approximate — it was
**backwards**. Driving the real teleop path and tracking the actual hand link:

| Commanded axis | Before | After |
|---|---|---|
| Up → hand height | **−0.920 (inverted)**, 24 cm travel | **+0.967**, 34 cm |
| Forward → hand forward | +0.779, 17 cm (and 44 cm of *vertical* cross-talk) | **+0.950**, 28 cm |
| Lateral → hand lateral | saturated after 2 cm — effectively dead | **+0.950**, 23 cm |

Raise your hand and the robot's hand went **down**. Push forward and it mostly went *up*.
Reach sideways and nothing happened. No amount of practice makes that feel smooth, and no
amount of re-tuning four hand-written gain rows was going to find it by feel.

The gains are gone. `ARM_IK_PINV` is the damped least-squares pseudo-inverse of the arm's
**measured** Jacobian — d(hand position)/d(joint) sampled around a neutral reach pose on
the real articulation (condition number 5.4, so the inverse is well behaved):

```python
dq = ARM_IK_PINV @ (target_body - ARM_IK_REFERENCE)
q  = ARM_IK_NEUTRAL + dq          # roll and yaw mirrored for the left arm
```

The offset is capped to `ARM_IK_MAX_OFFSET` (0.30 m) because a linear inverse is only
valid near the pose it was measured at; past that the hand stops tracking rather than
flinging the arm somewhere the linearisation cannot justify.

**Where that neutral pose sits matters more than the cap does.** Taken with the arm
hanging, the band was 0.63–1.03 m — and the work surfaces are at **0.994 m and 1.083 m**,
with the packages at 0.99–1.15 m. That is the "objects are above the robot, it can't touch
them" complaint, exactly: four of the six table packages sat above the ceiling. Raising the
cap does not help (measured: the ceiling stayed at ~1.03 m for caps of 0.30, 0.40, 0.50 and
0.60), because the limit was never the cap — it was that a linearisation centred on a
hanging arm cannot describe a raised one. The arm's own limit is **1.40 m** at
`shoulder_pitch = -2.6` rad, nowhere near its -177° stop.

So the neutral pose is now a *raised* one (`shoulder_pitch = -1.20`, hand at 1.044 m),
which centres the band on **0.74–1.34 m** — covering both surfaces and every package.

Re-measure with `scratchpad/measure_arm_jacobian.py` if the asset or the neutral pose
changes. **Do not re-tune these by hand** — that is what produced the inverted mapping.

> A measurement trap worth remembering: the first version of this sweep set the arm
> targets and *then* called `g1.forward()`, which rewrites the posture target for every
> non-leg DOF. It reported a 1 mm reach envelope across 91 poses — "the arm never moves".
> The real physics step writes the arm command *after* `forward()`. Order matters.

### The objects were five times too big, and nowhere near the robot

The operator's session log was a wall of `nothing within reach -- nearest is Box_04 at
1.52 m (reach is 0.55 m)`. That is a task-layout problem, not a tuning one, and measuring
it settled three separate causes at once:

| | measured |
|---|---|
| Hand's reachable shell | ~30 x 50 x 54 cm, reaching **0.444 m** in front of the pelvis |
| Packing crates | **0.60 x 0.40 m** — against a ~0.12 m Inspire hand |
| Work surfaces | 2 m to either side, packages scattered across the **middle** of them |
| Packages grabbable without walking | **0 of 10** |

Three fixes, each measured:

1. **YCB objects instead of warehouse crates.** Isaac ships the YCB set — the standard
   grasping benchmark — at 0.05–0.19 m. A hand cannot close on something five times its
   size, so the grab could only ever have been a magnet, and no amount of policy training
   would have changed that.
2. **A small side table in front of the spawn** (`SM_SideTable_02a`, 0.92 x 0.42 m, top at
   0.812 m — inside the reachable band). A 2.47 m packing bench there walled the robot in:
   it could only walk **0.19 m** before hitting it. This one can be stepped around.
3. **Packages on the near edge, in a spaced row.** They were placed mid-surface, which is
   deeper than the arm reaches, and randomly — which put three of them within 0.26 m of
   each other while they were 0.4–0.6 m wide, so they spawned interpenetrating and PhysX
   fired them across the room. They are now laid out in a row, inset a fixed 0.12 m from
   the lip, and rested on their **measured base** rather than their origin (these assets
   carry the origin near the centre, so the old 0.12 m "drop height" spawned a 0.5 m crate
   a quarter of a metre inside the table).

Result: **2 objects sit 0.12–0.15 m from the hand at spawn**, grabbable without moving,
and the robot can still walk (1.36 m in 5 s past the table).

### Distance-gated fixed-joint grasp assistance

Objects remain dynamic. A successful request creates a `UsdPhysics.FixedJoint`
between the real hand link and the object. Both local joint frames are authored
from live physics poses so they coincide in world space at creation. This preserves
the initial relative pose instead of snapping the object origin into the wrist.
Releasing, dropping, losing tracking, or resetting removes the joint.

This is assisted pickup: the gate measures body-centre distances and does not prove
finger contact, friction closure, or force balance. A fixed joint can hold an object
that an unassisted hand could not. Recordings identify this mode in metadata.

| Setting | Default | Meaning |
|---|---|---|
| `_grab_assist_radius` | 0.30 m | Green candidate highlight only |
| `_grab_radius` | 0.18 m | Search radius around the measured palm/finger centre (wrist pose plus rotated palm offset) |
| `_grasp_contact_distance` | 0.09 m | Object centre to nearest palm/finger link centre |
| `_grab_trigger_threshold` / `_grab_trigger_release_threshold` | 0.60 / 0.35 | Controller close/open hysteresis |
| `_finger_grab_threshold` / `_finger_release_threshold` | 0.55 / 0.35 | Tracked-hand mean curl hysteresis |

A green candidate can still be too far away to attach. Move the actual hand closer;
the highlight radius does not pull objects. Yellow gaze and green hand highlights
share session-layer overrides prepared before physics starts. Selection changes only
material properties; releasing restores the original materials, including those on
referenced child meshes. The layer and prim overrides remain stable while physics runs.

After Y/drop or tracking loss while holding, release the trigger/open the hand before
the next grab. An unchanged held trigger cannot immediately pick the object up again.

### Why the packages are on tables, not the floor

Because the robot physically cannot reach the floor. Measured, standing, with the walking
policy holding it up:

| shoulder pitch | hand height | result |
|---|---|---|
| 0° | 0.825 m | stood |
| +45° | **0.778 m** | stood — lowest reach |
| +75° | 0.250 m | **fell over** |

The hand bottoms out at **0.778 m**. A crate resting on the floor tops out at **0.23 m** —
a **0.55 m shortfall** — and reaching harder topples the robot, because the walking policy
was trained on a fixed-arm G1 and cannot compensate for the centre-of-mass shift.

So packages spawn on packing tables, whose tops were measured rather than assumed:

| Asset | Top surface | Usable? |
|---|---|---|
| `packing_table.usd` | 1.083 m | yes |
| `SM_HeavyDutyPackingTable_C02_01` | 0.994 m | yes |
| `pallet.usd` | 0.143 m | no — below reach |
| `pallet_holder.usd` | 0.660 m | no — and no collider |

```python
self._work_surface_placements = ((0.0, 2.0, 0.0), (0.0, -2.0, 0.0))     # x, y, yaw
self._package_floor_count = 2        # left unreachable on purpose
self._package_drop_height = 0.12     # spawned above the top, settles onto it
```

The tables sit to the robot's **left and right**, not straight ahead. With one dead ahead
at 2.1 m the robot walked into it after a single metre (measured: 1.01 m travelled, versus
3.30 m down a clear aisle) — a poor first thirty seconds in a headset. A quarter turn now
faces either table's long working edge, and the forward aisle stays open for walking.

**Two packages stay on the floor deliberately.** The robot cannot pick them up, and that
is the point: an operator reaching for something unreachable is a labelled failed attempt,
which is supervision for intent prediction rather than a bug. `g1:packageOnSurface` marks
which is which, so the dataset can separate them. Set `_package_floor_count = 0` to drop
the idea.

**Floor picking needs a different robot controller**, not a tuned one — something that can
squat while staying balanced (a loco-manipulation policy, or NVIDIA's GEAR-SONIC whole-body
controller). No amount of training the *locomotion* policy changes the length of the arm.

### Two things the swap broke, and how they are fixed

- **The floor.** The grid exposed one `GroundPlane/CollisionPlane` to bind friction to;
  the warehouse has no such prim. Friction is now bound to the environment root and
  inherits down. An invisible collision plane is also added underneath as a backstop —
  with no collider anywhere, a physics-walking robot falls out of the world on step one.
- **The gaze highlight.** Tinting used `displayColor`, which bound PBR materials override
  — real crates would have silently stopped turning yellow. The tracker now binds an
  emissive highlight material and restores the original binding afterwards, so the tint
  works regardless of what the asset ships with.

### Lighting

The warehouse brings its own interior lighting, so the fill lights are dimmed
(dome 1200 → 200, distant 2500 → 400) to avoid blowing out the exposure. Switching
`_environment_is_warehouse` back to `False` restores the bright values the empty grid
needs.

---

## Behavioral Data Collection

Every load of the example creates a **session folder**:

```
~/BehavioralCollection/raw_sessions/session_YYYY-MM-DD_HH-MM-SS/
├── metadata.json           session config: physics/rendering dt, log rates, robot name…
├── behavior.csv            main ~100 Hz log (schema below)
├── hand_tracking.csv       left/right hand or controller pose + grip/trigger/buttons,
│                           plus per-finger curl actually sent to the robot (~100 Hz)
├── gaze.csv                gaze ray + raycast hit point/object (~100 Hz); real Quest Pro
│                           eye tracking when available, else HMD-forward (see gaze_source)
├── object_states.csv       sample-box poses, velocities, grab state (~100 Hz)
├── frame_timestamps.csv    one row per captured camera frame
└── frames/eye_camera/      PNG frames from the first-person camera (~10 Hz, 256×256)
```

All logs share `unix_time`, `sim_time`, and `step_index` columns, so any row in any
file can be time-aligned with any other. CSVs are **appended to disk every ~2.5 s
during play** and finalized when the simulation is cleared or the example closes,
so a crash or force-quit loses at most the last few seconds; `metadata.json` is
written at session start.

> Each gaze row's `gaze_source` column says where it came from: `eye_tracker` (real
> Quest Pro eye tracking, see below) or `hmd_forward` (HMD position + facing direction —
> a weaker but still useful intent signal).

### Finger columns in `hand_tracking.csv`

| Column | Description |
|---|---|
| `<side>_finger_thumb/index/middle/ring/little` | Curl actually commanded to that finger, `0.0` = open, `1.0` = fully closed |
| `<side>_finger_thumb_yaw` | Separate thumb-opposition target, `0.0` = open, `1.0` = opposed across the palm |
| `<side>_hand_closure` | Mean curl across the five fingers — a single "how closed is this hand" scalar |
| `<side>_finger_source` | `hand_tracking` (measured from the tracked hand skeleton), `controller` (trigger), or `none` (that hand was not tracked this sample) |

Curl under hand tracking sums the bends between adjacent bones from metacarpal to
tip, normalised by a full-flexion reference (150° for the fingers, 95° for the thumb).
This keeps tightly folded fingers closed even past 180° of total bend. Thumb opposition
uses the thumb metacarpal direction in the palm plane. Both measurements are independent
of hand size, room position, and wrist rotation. Inspire's six actuators couple distal
knuckles; they cannot reproduce every human knuckle or finger-splay movement separately.

### Seeing through the robot's head

The current default is **`robot_head`**: a fixed camera-to-body mount follows the live
torso/head transform. Physical headset translation and rotation do not move the view,
including while paused. **B** restores this view, and left stick click cannot unlock it
in stationary mode. Gaze code and settings are unchanged. See the
[current camera implementation and validation](docs/humanoid-control.md#robot-mounted-camera).

### Historical camera experiments for moving modes

The following calibration and camera-mode experiments describe the retained optional
moving-robot paths. Their headset composition and mode-switch instructions do not
apply to the default stationary `robot_head` mode.

The rig rides the robot **and** keeps live head tracking (`_xr_camera_mode =
"head_compose"`). Two numbers decide whether that actually feels like being inside the
robot: how high the VR floor sits, and which way "forward" points. Both used to be
hand-tuned constants, and both are wrong for anyone whose height or starting orientation
differs from whoever tuned them — which is indistinguishable, in the headset, from a
camera that never attached at all. Sunk into the floor, floating above the robot, or
facing ninety degrees away from the way it walks.

Both are now **measured from the headset itself**, once, over the first quarter-second of
tracked head poses:

```
[G1] VR rig calibrated: your eyes are 1.71 m above your floor, the robot's are at
     1.37 m -> rig floor -0.34 m, yaw offset -90.0 deg. Face the way you want to walk
     and press B to redo this.
```

The height term is exact rather than approximate: the rig origin *is* the playspace
floor, your eyes sit your own height above it, so putting them at the robot's eye level
means `anchor_z = robot_eye_z - your_head_height`. The yaw term is the axis constant minus
however far you happen to be turned when it calibrates. Verified as a matrix identity for
body heights from 1.15 m to 1.85 m and every standing direction: eye position exact to
0.0000 mm, heading exact to 1e-6 degrees, and head tracking still 1:1 (a 30° head turn
sweeps the view 30°). The old fixed `-0.30 m` put a 1.85 m user 18 cm too high and a
seated one 52 cm too low.

**Press B (right controller) to recenter** — stand facing the way you want to walk, press
it, and the rig re-measures from where you are now.

**The measurement waits until you are still.** The first version latched after a quarter
second, which fires the instant the head pose first reads — quite possibly while the
headset is still being put on, or resting on a desk. A real session captured **1.27 m**
that way and called it the operator's standing eye height, which drops the rig ~35 cm and
puts the robot below you and away: indistinguishable, in the headset, from a camera that
never attached. It now takes the **median over a 2 s window** and refuses to latch until
the height has been steady within 15 cm, and it says so:

```
[G1] VR rig calibrated: your eyes are 1.68 m above your floor (steady to 3 cm over 2 s),
     the robot's are at 1.35 m -> rig floor -0.33 m, yaw offset -90.0 deg.
[G1] NOTE: 1.27 m is low for a standing adult. If you were seated or still putting the
     headset on, stand up facing the way you want to walk and press B to redo this.
```

### The rig has to cancel WHERE YOU STAND, not just how tall you are

The convention auto-detection below reported this on a real run:

```
[G1] VR camera convention measured: using 'compose'
     (head-to-robot-eye distance -- anchor_only 1.66 m, compose 1.37 m)
[G1] WARNING: even the best option leaves you 1.4 m from the robot's head.
```

Both candidates ~1.4 m out, when the correct one should measure ~0. That ruled the
convention out as the cause and pointed at the calibration: it captured the head's
**height** and **facing** and threw away `translation[0]` and `translation[2]` — where you
are standing in your room. The compose math passes that straight into the stage, so you end
up displaced from the robot by exactly how far you stand from your playspace origin. Which
is the "the camera is separate from the robot's body" symptom, and 1.4 m is a very ordinary
distance to stand from a room's centre.

All three components are now measured and subtracted, rotated by the robot's live yaw so
you stay locked inside it as it turns:

```
[G1] VR rig calibrated: your eyes are 1.52 m above your floor (steady to 3 cm over 2 s),
     the robot's are at 1.35 m -> rig floor -0.17 m, yaw offset -90.0 deg, standing
     1.37 m from your room origin (that offset is now removed).
```

Moving away from the calibrated spot still moves you relative to the robot — room-scale is
preserved. Only the constant displacement goes.

### The camera convention is measured, not assumed

`schedule_set_camera(M)` does one of two things, and the documentation does not say which:
either it makes the rendered **view** equal `M` (the runtime subtracting the live head pose
internally), or it sets the **rig origin** to `M` and composites the head on top. The
difference decides whether the head term must be pre-multiplied in or left out — and
getting it wrong throws the view a whole head-pose away from the robot, which is exactly
the "the camera is separate from the robot's body" symptom.

It cannot be settled from outside a live headset. Kit's own SimulatedXR display was tried
for this and does not come up in a standalone app (the XR viewport asks the `vr` profile
for OpenXR, which is not loaded, and forcing it crashes Kit on startup). And the `attached`
log line only ever meant `schedule_set_camera` did not raise — it printed "attached" in the
very sessions that were reported as detached, which is how this went unnoticed.

So the example now **measures it, in your headset, automatically**. It tries each candidate
for about half a second and scores it by the operator's own complaint stated numerically —
the distance from where the runtime believes your head is to where the robot's eye is:

```
[G1] VR camera convention measured: using 'anchor_only'
     (head-to-robot-eye distance -- anchor_only 0.11 m, compose 1.68 m)
```

Expect the view to shift once or twice in the first second while this runs. If even the
winner is more than a metre out it says so and tells you to try the other camera modes.

Set `_xr_convention_autodetect = False` to pin the theoretical choice instead.

**Click the LEFT thumbstick to cycle camera modes** — `head_compose → stage_anchor →
camera_lock` — if the automatic choice still does not put you in the robot:

```
[G1] VR camera mode -> stage_anchor (2 of 3; click the left thumbstick again ...)
```

Set `_xr_auto_calibrate = False` to go back to the hand-tuned constants.

If something upstream fails, the path says so once, on the terminal:

```
[G1] XR camera: attached (head_compose: rides the robot, head tracking live)
[G1] XR camera: no readable head pose (tried /user/head + a scan of every XR device) ...
[G1] XR camera: schedule_set_camera failed: ... ; using camera-lock
```

`camera-lock` is the fallback: the view follows the robot but your own head rotation stops
working, because the runtime subtracts the live head pose from whatever is scheduled. It
is uncomfortable to wear for long, but it proves the rig is attached rather than stranded
at the world origin. If you see it, the message says which step failed.

`_xr_camera_mode = "stage_anchor"` is a third option that hands the job to XRCore's own
`schedule_set_stage_anchor()` instead of composing the view matrix ourselves. It is
smoother when it works — nothing fights the runtime's reprojection — but it depends on the
build re-reading a *moving* anchor prim, which an earlier Kit did not do.

### If the red ball follows your head instead of your eyes

The ray **always draws**, and now prints its source on every **change** of source:

```
[EyeGaze] ray source: real eye tracking (unified gaze device)
[EyeGaze] ray source: HMD-forward direction (no eye-gaze device found -- the ray
          follows your head, not your eyes)
```

The change matters. The report used to latch on the first line printed, which hid the one
case that matters most on this rig: eye tracking is not up when the session starts, the
tracker says "HMD-forward", and then it silently upgrades to real eye tracking half a
second later — or silently drops back. The terminal then disagreed with what the ball was
actually doing. It is now truthful at every transition, in both directions, so you can
watch it come good without restarting.

**What the session logs show.** Comparing a session where gaze worked with one where it
did not (`~/.nvidia-omniverse/logs/Kit/Isaac-Sim XR VR/6.0/`), everything on the PC side
is identical in both: `XR_EXT_eye_gaze_interaction` is enabled, the
`/interaction_profiles/ext/eye_gaze_interaction` profile is bound, and `/user/eyes_ext` is
an active OpenXR device. The single difference is XRCore's own device list:

```
working:  input devices /user/head /user/hand/left /user/hand/right /user/eye/left
                        /user/eye/right /user/eye/unified          <-- appears ~0.5 s in
broken:   input devices /user/head /user/hand/left /user/hand/right /user/eye/left
                        /user/eye/right
```

`/user/eye/unified` is created **when the headset starts delivering gaze**, about half a
second after the session begins — not when the session begins. And in the Steam Link
driver log (`C:\Program Files (x86)\Steam\logs\driver_vrlink.txt`), the working runs
have a line the broken ones do not:

```
Client HMD type (3) supports eye tracking, creating input component
```

So when the ball follows your head, the gap is on the **headset / streaming side** every
time, never in Kit's configuration. The fix list is in the setup section below, and the
tracker prints it too.

`/user/eye/left` and `/user/eye/right` are **not** used as a fallback even though they are
always present. XRCore creates them as the stereo *display* view poses — the log lists
them as the input device handles of `hmdLeft` and `hmdRight` — and captured together they
carry the **same rotation**, differing only by your interpupillary distance. Using them
would report head direction while claiming to be eye tracking. The head fallback at least
says what it is. `HMD_FALLBACK_ENABLED = False` restores the old draw-nothing behaviour.

**The ball is always visible now.** It used to be hidden whenever the ray hit nothing, so
looking at open space made it vanish and the tracker looked broken; with nothing to land
on it parks at `_no_hit_marker_distance` (4 m) along the gaze instead of running the full
20 m to a dot on the horizon. Gaze also updates at ~50 Hz rather than ~25.

**And the ray is now actually visible.** "I don't see any red raycast" turned out not to
be a logic failure at all — the session log showed the ray resolving targets continuously
the whole time (`[EyeGaze] looking at ... PackingTable ...`). It was drawn and it could
not be seen, for two compounding reasons:

- **4 mm thick.** A hairline cylinder viewed from the robot's own head, down its own
  length, is close to nothing. Now 12 mm.
- **Not emissive.** The prims carried `displayColor` and no bound material, so the RTX
  renderer gave them a default *lit* material and the warehouse lighting shaded a
  0.45-red cylinder down to near-black. They are now bound to an emissive material
  (`_ensure_glow_material`) and the marker colour is a bright red instead of a dark one.

This is the same lesson the gaze *highlight* already had to learn — a bound material beats
`displayColor` in RTX — which had simply never been applied to the ray itself.

### Quest Pro Eye Tracking (optional)

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

#### Verified working setup (SteamVR + Steam Link)

> ⚠️ **Eye gaze needs Steam Link (or Virtual Desktop), connected and left connected.**
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
> What is missing is the headset actually delivering gaze through the streaming app.

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

#### Troubleshooting

- `xrCreateInstance ... XR_ERROR_API_VERSION_UNSUPPORTED` for API 1.1 at startup
  is **harmless** — Kit retries and creates the session with OpenXR 1.0.
- **`python tools/check_eye_tracking.py`** answers "is eye tracking reaching Isaac Sim?"
  in a second, without launching anything: it reads the OpenXR runtime from the registry,
  the SteamVR share setting, the Steam Link driver log, and the newest Kit session log, and
  tells you which link is broken. Watch the *date* on the Steam Link line — a stale "ok"
  from a previous day is the usual trap.
- If no gaze arrives for ~3 s of play, the tracker prints the numbered fix list above
  plus **the full list of XR devices the session can see**, on stdout where you will
  actually see it. If `/user/eye/unified` is missing from that list, the headset is not
  delivering gaze (recheck steps 1–3). `gaze.csv` then falls back to `hmd_forward` rows,
  and `gaze_source` records which one every row came from.
- The ray only appears once the eye tracker locks on (the pose reads as identity
  for the first seconds of a session).

Toggles in `HumanoidExample.__init__()`:

```python
self._eye_gaze_enabled = True             # master switch for the tracker
self._eye_gaze_ray_visual_enabled = True  # red ray + hit marker in the scene
```

### behavior.csv Schema

| Column group | Columns | Description |
|---|---|---|
| **Timestamps** | `unix_time`, `sim_time`, `step_index` | Wall-clock (Unix epoch, float), simulation time (s), sample counter |
| **HMD position** | `hmd_pos_x`, `hmd_pos_y`, `hmd_pos_z` | Headset world position (m) |
| **HMD orientation** | `hmd_qw`, `hmd_qi`, `hmd_qj`, `hmd_qk`, `hmd_yaw` | Quaternion + extracted yaw (rad) |
| **HMD velocity** | `hmd_vel_x`, `hmd_vel_y`, `hmd_vel_z`, `hmd_horiz_speed` | Filtered velocity (m/s) and horizontal speed magnitude |
| **Gait signal** | `gait_filtered_h`, `gait_vel_sign`, `gait_horiz_gate`, `gait_output`, `gait_pulse_rem` | Internal gait detector state |
| **Step event** | `step_event` | `1` on the exact sample when a footfall is detected, `0` otherwise |
| **Robot pose** | `robot_pos_x/y/z`, `robot_qw/qi/qj/qk`, `robot_yaw` | G1 base world pose |
| **Commands** | `cmd_forward`, `cmd_lateral`, `cmd_yaw` | Locomotion commands driving the base (m/s, rad/s) |
| **Joint states** | `j_<joint_name>_pos`, `j_<joint_name>_vel` | Position (rad) and velocity (rad/s) for every one of the G1's 53 DOFs (29 body + 24 finger) |

**Sample rate:** ~100 Hz (every physics step at the default 100 Hz physics rate).

### Using the Data for AI/RL Training

The dataset captures aligned time-series of:
- **User intent signal** — headset motion, gait events, controller inputs
- **Robot response** — base trajectory + full joint state over time
- **Command mapping** — what policy velocity was sent at each timestep

Suggested uses:

```python
import pandas as pd

df = pd.read_csv("~/BehavioralCollection/raw_sessions/session_2026-06-07_22-30-00/behavior.csv")

# Headset gait features (input to model)
hmd_features = df[["hmd_pos_x","hmd_pos_y","hmd_pos_z",
                    "hmd_vel_x","hmd_vel_y","hmd_vel_z",
                    "gait_filtered_h","gait_horiz_gate","step_event"]]

# Robot state (output / ground truth)
robot_state = df[[c for c in df.columns if c.startswith("j_") or c.startswith("robot_")]]

# Locomotion commands (supervision signal)
commands = df[["cmd_forward","cmd_lateral","cmd_yaw"]]
```

Suitable for:
- **Imitation learning** — learn a policy that maps HMD motion → robot joint targets
- **Inverse kinematics from intent** — map user walking pattern to whole-body motion
- **LLM fine-tuning** — structured time-series annotations with step events as semantic markers
- **Gait analysis** — study the coupling between headset dynamics and robot locomotion

---

## Tuning Parameters

All parameters are set in `HumanoidExample.__init__()`. Key ones to adjust:

### Gait sensitivity
```python
self._headset_gait_min_amplitude      = 0.012   # minimum peak-trough height (m) to count as a step
self._headset_gait_min_horiz_speed    = 0.025   # minimum horizontal speed (m/s) to validate gait
self._headset_gait_forward_intensity  = 1.0     # forward speed fraction (0–1) on each step pulse
self._headset_gait_pulse_duration     = 0.50    # seconds of forward command per detected step
self._headset_gait_min_step_interval  = 0.18    # minimum seconds between consecutive step pulses
self._headset_gait_max_extremum_gap   = 0.95    # max seconds between peak and next trough
```

### Camera
```python
self._first_person_eye_height_above_base = 0.58  # eye height above the G1 pelvis/base link (m)
self._first_person_head_forward_offset   = 0.26  # forward from the head, out of the skull mesh (m)
self._first_person_head_up_offset        = 0.0   # extra fine-tune on top of the eye height (m)
self._head_camera_yaw_sign               = 1.0   # legacy moving-camera modes only
```
These offsets establish the camera pose at scene creation. In `robot_head` mode that
pose becomes a fixed local mount on the head's rigid ancestor (`torso_link` in the
Inspire asset), then follows its full live position and rotation. Adjust the offsets
and reload the example to change the mount. `0.58` places the view above the skull;
the forward offset keeps the camera outside its mesh.

Note these numbers shrank when the robot changed: the G1 stands 1.32 m tall against
the H1's 1.80 m, so every offset measured against the old skull had to come down.

### Optional policy mode: turning experiments

The same steady-state error that causes the idle drift shows up as a *curve* the moment
you ask for yaw alone: you press A to look right, and the robot walks a wide arc while it
turns. In a VR teleoperation rig that is worse than the idle drift, because turning is how
you aim before you reach for something — so every turn moved the thing you were aiming at.

Measured, on a yaw-only command held for 4 s:

| | Before | After |
|---|---|---|
| Ground travelled, turning left | 1.853 m | **0.007 m** |
| Ground travelled, turning right | 3.139 m | **0.004 m** |
| Rotation achieved, left | +120.8° | **+159.1°** |
| Rotation achieved, right | −75.8° | **−200.9°** |

The robot also *turns more* now, because none of the command is being spent curving
forward. Two layers do it, and neither is enough alone (`TURN_HOLD_ENABLED` in `g1.py`):

1. **The policy is asked to cancel its own translation.** Station keeping used to require
   *everything* to be idle, so the instant you asked to turn, nothing was cancelling the
   forward creep any more. The condition is now "translation idle", not "everything idle".
   Your yaw is never touched — cancelling that would fight the stick.
2. **The base is pulled back to where the turn started**, with its horizontal velocity
   zeroed so the residual cannot re-accumulate. A first-order pull with a 0.20 s time
   constant and a 1.5 m/s cap, so a turn straight out of a walk is eased back rather than
   snapped, which would knock the gait over.

Yaw and height are left completely alone: the turn itself, and the gait bob under it, are
real. Walking while turning still curves normally (3.65 m and +77° over 4 s) — the hold
only engages when you asked for **no** translation at all.

### Optional policy mode: drift and stopping experiments

Commanded zero, the walking policy does not hold position — it walks off. Measured, held
at zero command for 10 s:

| Configuration | Drift | Real time |
|---|---|---|
| 100 Hz, nothing | 0.622 m/s, 5.2 deg/s yaw | 1.39x |
| 100 Hz + velocity station keeping | 0.497 m/s | 1.39x |
| 150 Hz + station keeping | 0.232 m/s | 0.80x |
| 200 Hz + station keeping | 0.150 m/s | 0.62x |
| **100 Hz + idle hold (current)** | **0.000 m/s** | **1.36x** |

Two things were going on. The policy has a genuine steady-state error at zero command
(0.376 m/s even at the 200 Hz it was trained for), and running it at 100 Hz makes that
worse. Velocity feedback — measure the drift, command the opposite — only recovered a
fifth of it, because cancelling forward drift means commanding *reverse* and this policy
tracks reverse at about 40%.

So after `STATION_HOLD_DELAY` (0.4 s) of an idle command the base is simply **pinned**
and the legs hold their current pose. Letting go of the stick means the robot stays where
you left it, exactly. Moving the stick releases the hold and resets the policy's LSTM
memory, so it does not resume mid-stride from before the pause.

```python
STATION_HOLD_ENABLED = True
STATION_HOLD_DELAY = 0.4        # s of idle before pinning
STATION_KEEP_ENABLED = True     # velocity feedback, still runs in the 0.4 s before the pin
```

This is a teleoperation convenience, and it is worth being clear that it is not free
physics: during a hold the base is kinematically fixed rather than balanced. If a session
needs uninterrupted physical realism at rest in `policy` mode, set `STATION_HOLD_ENABLED = False`
and accept ~0.5 m/s of creep, or run at 200 Hz for 0.15 m/s at 0.62x real time.

**Solver iterations are not a knob worth turning here.** Dropping the articulation from 16
position iterations to 8 made everything worse (step 7.9 ms, drift 0.43 m/s, pelvis
sagging to 0.69 m) and 4 collapsed the robot outright (pelvis 0.08 m). Fewer iterations
destabilised it, which generated more contact work than it saved.

### The single biggest thing: physics device and rate

If the robot ever feels laggy, unresponsive, or like it "never stops", check this before
touching anything else. Measured on this scene with one G1:

| Configuration | Step time | Real time |
|---|---|---|
| CUDA @ 200 Hz (the old default) | 23–28 ms | **0.22x** |
| CPU @ 200 Hz | 6.83 ms | 0.73x |
| **CPU @ 100 Hz (current)** | **7.17 ms** | **1.39x** |

The example used to run physics on **CUDA**, inherited from the H1 sample. That is the
wrong device for this workload: GPU PhysX exists to step thousands of environments in
parallel, and with a single robot you pay all of the kernel-launch overhead and get none
of the parallelism. The sim ran at a fifth of real time, which is what "terrible
movement", "it never stops" and sluggish rotation actually were — commands arriving late
into a world playing back five times too slowly, rendering at ~17 FPS.

Dropping to 100 Hz physics then buys the rest: the per-step cost barely changes but the
budget doubles. Control stays at 50 Hz because the policy's decimation is derived from
`dt`, and the behavioural logging intervals are derived from the physics rate too, so the
dataset keeps its ~100 Hz sampling.

**What was *not* the problem**, each ruled out by measurement:

| Suspect | Cost | Verdict |
|---|---|---|
| The 26k-prim warehouse | grid 22.98 ms vs warehouse 23.41 ms | irrelevant |
| The whole control loop | 0.99 ms | ~3% of the frame |
| Policy inference | 0.34 ms | negligible |
| Inspire hand colliders | 2.65 ms | real, but worth keeping |

Training a new locomotion policy would not have moved any of these numbers.

### Optional policy mode: gait experiments

Two causes, and only one of them is fixable from here.

**Fixed:** locomotion commands were stepped, not ramped, so every stick flick slammed the
command from 0 to full in a single physics step and the gait answered with a lurch.
Ramping cut peak body tilt from 12.2° to 10.8° (−11%).

A single ramp time for both directions was itself a mistake, though: it made releasing the
stick take about a second, so the robot felt like it would never stop, and turns went
mushy. Smoothing the *start* of a command is what protects the gait; smoothing the *end*
just adds lag. The ramp is now asymmetric:

```python
self._command_attack_time  = 0.30   # ease in  - smooths starts and turns
self._command_release_time = 0.06   # ease out - release the stick and it stops
```

Measured: stop in **0.16 s**, reverse a turn in **0.04 s**, reach 90% speed in 0.69 s, and
the brake (stick back) bypasses the ramp entirely and zeroes the command in one step.

**Not fixable without retraining:** the policy was trained on `g1_12dof.urdf` — a G1 whose
arms are *fixed*. It has never seen arm motion and has no way to compensate for the
centre-of-mass shift when you move your hands in VR. That is the same reason a +75°
shoulder reach topples it. Improving this means training a policy that has the arms in its
observation, in Isaac Lab (`Isaac-Velocity-Flat-G1-v0`): roughly 30–60 minutes per run on
an RTX 5090, times however many reward-tuning runs it takes.

Note that Unitree's checkpoint cannot be fine-tuned in place on modern hardware — it was
trained in Isaac Gym / legged_gym, a deprecated stack that predates Blackwell GPUs.
Retraining in Isaac Lab is the practical route.

### Optional policy mode: camera stabilization experiments

In `policy` locomotion the camera rides the pelvis, and a walking humanoid's pelvis is
not a steady platform. Measured on this robot while walking at 0.5 m/s, the raw camera
motion was:

| | Raw (unstabilized) | Stabilized |
|---|---|---|
| Height bob | 3.06 cm peak-to-peak | see below |
| Lateral sway | 2.47 cm peak-to-peak | |
| Yaw wobble | 5.00° peak-to-peak | |

All of it at the 0.8 s gait period — about **1.25 Hz**, squarely in the band that causes
VR sickness. A real neck does not pass that through; the head stays far steadier than the
hips. `_stabilize_camera_pose` reproduces that with three first-order low-passes:

```python
self._camera_stabilization_enabled = True
self._camera_height_filter_time  = 0.35   # s: strongest — nothing here intentionally
                                          #    changes height, so lag costs nothing
self._camera_lateral_filter_time = 0.18   # s: removes sway, keeps walking responsive
self._camera_yaw_filter_time     = 0.22   # s: removes per-step yaw wobble
```

The filtered pose feeds both the desktop camera *and* the XR rig anchor, so the headset
gets the same steady frame. Camera roll and pitch never enter at all — the pose is built
from yaw with world-up — so bob, sway and yaw were the whole problem.

Raise the time constants if any wobble remains; lower them if walking feels laggy. Set
`_camera_stabilization_enabled = False` to feel the raw pelvis motion again.
The same applies to the VR rig anchor, which now sits *below* the floor:

```python
self._xr_anchor_height_offset = -0.30  # your real standing eye height adds on top
```
The G1's eyes are at ~1.25 m. If you stand 1.7 m tall, your eyes need the rig to
drop ~0.3 m for them to land at the robot's. Raise towards 0 if you are shorter.

### Robot and fingers
```python
self._g1_hand_variant     = "Inspire"   # "Inspire" (5 fingers) or "ThreeFinger" (Dex3)
self._g1_spawn_position   = [0.0, 0.0, 0.80]   # pelvis height of the standing posture
self._finger_control_enabled = True
self._finger_smoothing       = 0.35     # low-pass on curl, per physics step
self._finger_grab_threshold  = 0.55     # mean curl at which a nearby box is grabbed
self._finger_curl_full_flexion_rad = {"thumb": 95°, others: 150°}  # bone angle = fully closed
```

The robot-side finger constants live in `robots/g1.py` (`G1TeleopRobot`):

```python
FINGER_STIFFNESS    = 20.0   # the hand asset's own ~0.05-0.19 stalls the fingers part-closed
FINGER_DAMPING      = 0.6
FINGER_MAX_EFFORT   = 10.0
FINGER_MAX_VELOCITY = 8.0    # rad/s; the asset caps at 0.5 rad/s (the real hand's actuator
                             # speed), which takes 3.2 s to close a fist. 8 rad/s takes ~0.2 s
```

Only six joints per hand are driven (four finger proximals + thumb pitch and yaw); the
other six are PhysX **mimic** joints that follow their driver automatically. Never write
positions to the mimic joints — snapping them against their coupling leaves the two hands
resting in different poses and the fingers stalling short of the commanded pose.

### Data collection
```python
self._behavioral_data_log_every_n_steps = 1      # default 100Hz physics: 1=100Hz, 2=50Hz
self._behavioral_frame_log_every_n_steps = 10    # default 100Hz physics: 10=10Hz
self._behavioral_data_output_dir = Path.home() / "BehavioralCollection"
self._gaze_raycast_max_distance = 20.0           # m: range cap for the gaze raycast
```

---

## Architecture Overview

```text
on_physics_step (default 100 Hz)
├── validated input → deadzone/clamp → time-based command smoothing
├── G1.forward(dt, body command)
│   ├── stationary (default): suppress base command, world-fixed pelvis, hold posture
│   ├── policy (optional): 47 observations → recurrent actor → 12 leg targets at 50 Hz
│   └── kinematic (optional): integrate base pose, hold default posture
├── fingers: optical skeleton or controller trigger → drive targets
├── arms: world pose → body-frame calibration → 7-joint palm/wrist IK
│   └── grasp: live hand/object distances → coincident fixed-joint frames
├── camera follows/stabilizes the robot view
├── gaze: validate every tick → closest non-robot raycast hit at 50 Hz
└── record behavior/hand/gaze/object rows (~100 Hz), images (~10 Hz)
    └── append buffered CSV rows every ~2.5 s and on stop
```

Policy and gaze schedules use elapsed time so non-divisor physics rates do not
silently change their frequency. Finger curls are updated before pickup decisions.
See the [developer guide](docs/humanoid-control.md) for frames, reset behavior,
regression coverage, and the remaining live acceptance checks.

---

## Debugging a live session

Isaac Sim ships `isaacsim.code_editor.python_server`, a TCP server that executes Python
inside the running Kit process. It is off by default. With it on, the questions that
otherwise need a headset on someone's face — which XR devices exist, what the head pose
is, whether the camera is attached, where the gaze ray is pointing — become one command,
and values can be retuned without restarting a 7 GB application.

```bat
tools\launch_isaac_vr.bat              REM same VR app, plus the Python server on :8226
tools\launch_isaac_vr.bat --xr-verbose REM ... and the OpenXR runtime diagnostics
```

Already have a session open? Turn it on from **Window → Extensions**, search
"Python Server", toggle it — no restart needed.

```bash
# Full first-person rig status: head pose, calibration, anchor offsets, device list.
python tools/kit_exec.py "print(EX.describe_xr())"

# Gaze status: source, ray origin and direction, what it is hitting.
python tools/kit_exec.py "print(EX._eye_gaze_tracker.describe())"

# Which XR devices does the runtime actually expose right now?
python tools/kit_exec.py "from omni.kit.xr.core import XRCore; \
print([d.get_name() for d in XRCore.get_singleton().get_all_input_devices()])"

# Retune without restarting - e.g. widen the grab assist and re-run the calibration.
python tools/kit_exec.py "EX._grab_assist_radius = 0.7; EX._request_xr_recenter()"
```

`EX` is bound automatically to the running `HumanoidExample`.

### The logs are worth reading before guessing

`~/.nvidia-omniverse/logs/Kit/Isaac-Sim XR VR/6.0/` keeps a full log per session, and it
records everything the example prints plus everything XRCore does. Three greps answer most
VR questions without a headset:

```bash
grep -a "XR camera\|EyeGaze\|\[G1\]" kit_*.log      # what the example decided
grep -a "xr.core.plugin.*input devices"   kit_*.log      # what devices XRCore created
grep -a "py stderr"                        kit_*.log      # tracebacks nobody saw on screen
```

The dead keyboard handler and the missing eye-gaze device were both found this way, in
logs that already existed.

---

## License

This file is a modification of NVIDIA Isaac Sim source code and is licensed under the **Apache License 2.0**, the same license as the upstream repository.  
See [LICENSE](LICENSE) for the full text.

Original copyright: © 2020–2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  
Modifications: © 2026 Soheil Sepahyar.
