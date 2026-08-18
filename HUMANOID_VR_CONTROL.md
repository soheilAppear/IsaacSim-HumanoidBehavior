# Humanoid VR Control — Unitree G1 with Dexterous Hands, Quest Pro Eye Tracking & Behavioral Data Recording

> **Fork of:** [NVIDIA IsaacSim](https://github.com/isaac-sim/IsaacSim) · Isaac Sim 6.0.0 GA  
> **Files modified/added:** `.../interactive/humanoid/humanoid_example.py` · `eye_gaze_tracker.py` (same folder) · `.../robots/g1.py` (new)  
> **Author:** [@soheilAppear](https://github.com/soheilAppear)

---

## What This Is

This fork replaces the stock Isaac Sim H1 humanoid example with the **Unitree G1 fitted with
Inspire five-finger hands** — the Unitree humanoid that [Isaac Teleop](https://github.com/NVIDIA/IsaacTeleop)
drives for dexterous manipulation — and wraps it in a full **VR-driven teleoperation and
data-collection pipeline**:

| Feature | Description |
|---------|-------------|
| **Headset gait detection** *(disabled by default)* | Bob your head up/down in VR to move the robot forward — real-time peak/trough signal processing on the HMD height signal. Currently off while step detection is tuned; set `_headset_gait_enabled = True` to re-enable |
| **Horizontal motion gate** | Prevents false walking from pure head nodding — gait is suppressed unless the headset also moves in the floor plane |
| **Eye-level first-person camera** | Viewport/XR camera placed at robot eye height (not top of skull) |
| **VR hand tracking → arm control** | OpenXR hand/controller poses drive the G1 arm joints |
| **Finger teleoperation** | Real articulated fingers: per-finger curl measured from the OpenXR hand skeleton drives the G1's Inspire hand joints; with controllers the trigger curls the index and the grip closes the rest |
| **Grab system** | Physical sample boxes can be grabbed by squeezing the controller grip — or, under hand tracking, simply by closing your fingers |
| **Quest Pro eye tracking** | Real OpenXR eye gaze (the runtime's calibrated fusion of both eyes) drawn as a **red ray** from your eyes to the gazed collider (sample boxes, ground) with a **large blood-red marker sphere** at the collision point; the looked-at box is tinted yellow; every gaze-target change is printed live to the terminal as `[EyeGaze] looking at ...`; robot self-hits filtered out of the raycast (`eye_gaze_tracker.py`) |
| **Behavioral session recorder** | Every run creates a session folder under `~/BehavioralCollection/raw_sessions/` with `metadata.json` + five time-aligned ~100 Hz CSVs (behavior, hand tracking, gaze, object states, frame timestamps), flushed to disk every ~10 s during play |
| **Eye-camera frame capture** | First-person 256×256 PNG frames at ~10 Hz, timestamped for video–sensor sync |
| **Gaze logging with fallback** | `gaze.csv` uses real eye tracking when available, else HMD-forward direction — tagged per row via `gaze_source` |
| **Learning pipeline scaffold** | `learning/` folder for the V-JEPA world-model pipeline (data sync → baselines → multimodal predictor → planner) |

### How the G1 walks — read this first

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

Set the mode in `humanoid_example.py`:

```python
self._g1_locomotion = "policy"      # Unitree walking policy — real physics gait
#                     "kinematic"   # no policy: posture held, base glides on command
```

`kinematic` is the fallback and is worth knowing about: the robot holds its standing
posture, gravity is disabled on its links, and the base is integrated straight from the
`(vx, vy, wz)` command. It cannot fall over, which makes it useful for recording clean
manipulation sessions where a stumble would ruin the take.

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

### Option A — Direct file replacement (simplest)

Copy the modified file over the installed one in your Isaac Sim standalone package.

**Windows:**
```powershell
$isaacRoot = "C:\path\to\isaac-sim-standalone-6.0.0-windows-x86_64"
$extPath   = "exts\isaacsim.robot.policy.examples\isaacsim\robot\policy\examples\interactive\humanoid"

Copy-Item `
  "source\extensions\isaacsim.robot.policy.examples\isaacsim\robot\policy\examples\interactive\humanoid\humanoid_example.py", `
  "source\extensions\isaacsim.robot.policy.examples\isaacsim\robot\policy\examples\interactive\humanoid\eye_gaze_tracker.py" `
  "$isaacRoot\$extPath\"
```

**Linux:**
```bash
ISAAC_ROOT="/path/to/isaac-sim-standalone-6.0.0-linux-x86_64"
EXT_PATH="exts/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"

cp source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/humanoid_example.py \
   source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/eye_gaze_tracker.py \
   "$ISAAC_ROOT/$EXT_PATH/"
```

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

### Keyboard
| Key | Action |
|-----|--------|
| `↑` or `Numpad 8` | Walk forward |
| `↓` or `Numpad 2` | Walk backward (slower — the gait is less stable in reverse) |
| `←` or `Numpad 4` | Turn left |
| `→` or `Numpad 6` | Turn right |

### VR Controller

Thumbsticks are used when the runtime exposes them, because the triggers do double duty —
the trigger that would drive locomotion is also that hand's index-finger curl, so walking
on it would clench the robot's hand at the same time.

| Input | Action |
|-------|--------|
| **Left stick** | Walk forward / backward (preferred) |
| **Right stick** | Turn left / right (preferred) |
| Right trigger (hold) | Walk forward *(fallback if no stick)* |
| Left trigger (hold) | Walk backward *(fallback if no stick)* |
| X button (left controller) | Turn left *(fallback)* |
| A button (right controller) | Turn right *(fallback)* |
| Left grip (hold) | Arm teleoperation — left arm |
| Right grip (hold) | Arm teleoperation — right arm |
| Trigger pressure | Index-finger curl on that hand |
| Grip pressure | Middle / ring / little / thumb curl on that hand |

The example tells you which one it picked, once, the first time you move:

```
[G1] locomotion input: THUMBSTICK
[G1] locomotion input: TRIGGERS/BUTTONS (no thumbstick seen). ...
```

If you get the second line but your controllers do have sticks, the log also prints
`HumanoidExample XR left controller inputs: [...]` — find the stick's name there and add
it to `_xr_stick_input_candidates`.

The names used are Kit's own (`XRInputTokens.thumbstick`, `XRGestureTokens.x` / `.y`), and
the mapping is verified: stick up walks forward, stick down reverses at the capped speed,
right stick right turns right, small drift is deadzoned away, and a centred stick still
falls through to the trigger.

### Gamepad
| Input | Action |
|-------|--------|
| Left stick up / down | Walk forward / backward |
| Right stick left / right | Turn |
| Right trigger | Walk forward |
| X / A buttons | Turn left / right |

### How a keypress becomes a step

Every input above lands in the same place: a single base velocity command
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
1.57 rad/s yaw. Backward is deliberately capped lower (`_max_backward_speed = 0.4`).
Lateral strafing (`v_y`) is supported by the policy but is not bound to any key.

> **Head motion does not move the robot.** Headset gait is off
> (`_headset_gait_enabled = False`) and, while disabled, its output is excluded from the
> command mix entirely rather than merely being zero. Your HMD pose is still read every
> step, but only for `sim_time`, the `hmd_*` columns in `behavior.csv`, and the
> HMD-forward gaze fallback.

### VR Headset Gait (step-in-place walking)

> **Disabled by default** since 2026-07 while step detection is stabilized. Re-enable with
> `self._headset_gait_enabled = True` in `HumanoidExample.__init__()`. The HMD pose is
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
file can be time-aligned with any other. CSVs are **appended to disk every ~10 s
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
| `<side>_hand_closure` | Mean curl across the five fingers — a single "how closed is this hand" scalar |
| `<side>_finger_source` | `hand_tracking` (measured from the tracked hand skeleton), `controller` (trigger/grip), or `none` (that hand was not tracked this sample) |

Curl under hand tracking is the angle between each finger's metacarpal bone and its
distal phalanx, normalised by a full-flexion reference (150° for the fingers, 95° for
the thumb). Measuring an *angle between bones* rather than a fingertip-to-palm distance
makes the signal independent of hand size and of where your hand is in the room.

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
- The same data feeds `gaze.csv` at ~100 Hz with `gaze_source=eye_tracker`
  (robot self-hits filtered out of the raycast).

#### Verified working setup (SteamVR + Steam Link)

> ⚠️ **Quest Link / Air Link cannot deliver eye gaze.** Meta's PC OpenXR runtime
> never exposes `XR_EXT_eye_gaze_interaction` over Link (only a filtered avatar
> extension). The working path is SteamVR with the eye data carried by **Steam
> Link** (free) or **Virtual Desktop** ("Forward tracking data" enabled).

1. **Headset:** Settings → Movement tracking → **Eye tracking: ON** (grant the
   permission and run the eye calibration once).
2. **Headset:** Settings → Privacy & Safety → App permissions → **Eye tracking →
   allow Steam Link**.
3. **Steam Link app settings** (on the headset): **"Share eye tracking data to
   other apps on this PC": ON** — off by default, and the switch most people miss.
4. **PC:** set SteamVR as the active OpenXR runtime (SteamVR → Settings →
   OpenXR → *Set SteamVR as OpenXR runtime*), connect via Steam Link, then launch
   `isaac-sim.xr.vr.bat`.

On success the console logs `EyeGazeTracker: using unified eye gaze ...`.
Kit registers the eye device as `/user/eye/unified` (pose `gaze`).

#### Troubleshooting

- `xrCreateInstance ... XR_ERROR_API_VERSION_UNSUPPORTED` for API 1.1 at startup
  is **harmless** — Kit retries and creates the session with OpenXR 1.0.
- If no eye data arrives for ~10 s of play, the tracker logs one warning that
  **includes the full list of XR devices the session can see** — if
  `/user/eye/unified` is missing from that list, the streaming app is not
  forwarding eye tracking (recheck steps 2–3). `gaze.csv` then silently falls
  back to `hmd_forward` rows.
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

**Sample rate:** ~100 Hz (every 2 physics steps at 200 Hz physics)

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
self._head_camera_yaw_sign               = 1.0   # set -1.0 if the view turns opposite to the robot
```
The camera height is always `base_z + eye_height (+ up_offset)` — deterministic
regardless of how the asset's head-link origin is placed. Adjust `eye_height` in
±0.02 steps until the view matches the robot's eyes; `0.46` is strict eye level
inside the G1's head, `0.58` clears the top of the skull.

Note these numbers shrank when the robot changed: the G1 stands 1.32 m tall against
the H1's 1.80 m, so every offset measured against the old skull had to come down.

### Camera stabilization — the VR wobble fix

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
self._behavioral_data_log_every_n_steps = 2      # 1=200Hz, 2=100Hz, 4=50Hz
self._behavioral_frame_log_every_n_steps = 20    # camera frames: 20=10Hz, 10=20Hz
self._behavioral_data_output_dir = Path.home() / "BehavioralCollection"
self._gaze_raycast_max_distance = 20.0           # m: range cap for the gaze raycast
```

---

## Architecture Overview

```
on_physics_step (200 Hz)
│
├── _update_controller_command(dt)
│   ├── _read_xr_controller_axes()        ← VR trigger / A / X buttons
│   ├── _read_gamepad_controller_axes()   ← gamepad fallback
│   └── _update_headset_gait_command(dt)   ← pose reads for logging; output NOT used
│       ├── _get_headset_tracking_height() ← reads /user/head XR pose
│       ├── _update_headset_velocity(dt)   ← 3D velocity + horizontal gate
│       └── (step detection, only when _headset_gait_enabled)
│
├── g1.forward(dt, base_command)          ← G1TeleopRobot
│   ├── policy mode:    47-dim obs → LSTM → 12 leg targets @ 50 Hz
│   └── kinematic mode: integrate the base pose, hold the posture
├── _update_g1_arms_from_hand_tracking()  ← OpenXR hand → arm DOFs
├── _update_g1_fingers()                  ← hand skeleton / trigger+grip → finger DOFs
├── _update_head_camera_view(dt)          ← first-person camera
│   └── _stabilize_camera_pose()          ← low-pass out the gait wobble (VR comfort)
└── _collect_all_behavioral_data()        ← behavior/hand/gaze/object rows (~100 Hz)
                                             + eye-camera PNG frame (~10 Hz)
                                             (CSVs appended every ~10 s + on stop)
```

---

## License

This file is a modification of NVIDIA Isaac Sim source code and is licensed under the **Apache License 2.0**, the same license as the upstream repository.  
See [LICENSE](LICENSE) for the full text.

Original copyright: © 2020–2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  
Modifications: © 2026 Soheil Sepahyar.
