# Humanoid VR Control — H1 Robot with VR Gait, Quest Pro Eye Tracking & Behavioral Data Recording

> **Fork of:** [NVIDIA IsaacSim](https://github.com/isaac-sim/IsaacSim) · Isaac Sim 6.0.0 GA  
> **Files modified/added:** `source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/humanoid_example.py` · `eye_gaze_tracker.py` (same folder)  
> **Author:** [@soheilAppear](https://github.com/soheilAppear)

---

## What This Is

This fork extends the stock Isaac Sim H1 humanoid interactive example with a full **VR-driven locomotion and data-collection pipeline**:

| Feature | Description |
|---------|-------------|
| **Headset gait detection** *(disabled by default)* | Bob your head up/down in VR to make the robot walk — real-time peak/trough signal processing on the HMD height signal. Currently off while step detection is tuned; set `_headset_gait_enabled = True` to re-enable |
| **Horizontal motion gate** | Prevents false walking from pure head nodding — gait is suppressed unless the headset also moves in the floor plane |
| **Eye-level first-person camera** | Viewport/XR camera placed at robot eye height (not top of skull) |
| **VR hand tracking → arm control** | OpenXR hand/controller poses drive H1 arm joints |
| **Grab system** | Physical sample boxes in the scene can be grabbed with controllers |
| **Quest Pro eye tracking** | Real OpenXR eye gaze (the runtime's calibrated fusion of both eyes) drawn as a **red ray** from your eyes to the gazed collider (sample boxes, ground) with a **large blood-red marker sphere** at the collision point; the looked-at box is tinted yellow; every gaze-target change is printed live to the terminal as `[EyeGaze] looking at ...`; robot self-hits filtered out of the raycast (`eye_gaze_tracker.py`) |
| **Behavioral session recorder** | Every run creates a session folder under `~/BehavioralCollection/raw_sessions/` with `metadata.json` + five time-aligned ~100 Hz CSVs (behavior, hand tracking, gaze, object states, frame timestamps), flushed to disk every ~10 s during play |
| **Eye-camera frame capture** | First-person 256×256 PNG frames at ~10 Hz, timestamped for video–sensor sync |
| **Gaze logging with fallback** | `gaze.csv` uses real eye tracking when available, else HMD-forward direction — tagged per row via `gaze_source` |
| **Learning pipeline scaffold** | `learning/` folder for the V-JEPA world-model pipeline (data sync → baselines → multimodal predictor → planner) |

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

### Without VR (keyboard only)
```powershell
cd "C:\path\to\isaac-sim-standalone-6.0.0-windows-x86_64"
.\isaac-sim.bat
```

### With VR headset
```powershell
.\isaac-sim.xr.vr.bat
```

Once Isaac Sim is open:
1. Go to **Isaac Examples → Robot Policy → Humanoid**
2. Click **Load**
3. Click **Play**

---

## Controls

### Keyboard
| Key | Action |
|-----|--------|
| `↑` or `Numpad 8` | Walk forward |
| `←` or `Numpad 4` | Turn left |
| `→` or `Numpad 6` | Turn right |

### VR Controller
| Input | Action |
|-------|--------|
| Right trigger (hold) | Walk forward |
| X button (left controller) | Turn left |
| A button (right controller) | Turn right |
| Left grip (hold) | Arm teleoperation — left arm |
| Right grip (hold) | Arm teleoperation — right arm |

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
├── hand_tracking.csv       left/right hand or controller pose + grip/trigger/buttons (~100 Hz)
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
| **Robot pose** | `robot_pos_x/y/z`, `robot_qw/qi/qj/qk`, `robot_yaw` | H1 base world pose |
| **Commands** | `cmd_forward`, `cmd_lateral`, `cmd_yaw` | Locomotion commands sent to policy (m/s, rad/s) |
| **Joint states** | `j_<joint_name>_pos`, `j_<joint_name>_vel` | Position (rad) and velocity (rad/s) for every H1 DOF |

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
self._first_person_head_up_offset     = -0.18   # camera height offset from head prim (m, negative = lower)
self._first_person_head_fallback_height = 1.50  # eye height above robot base when no head prim found (m)
self._first_person_head_forward_offset = 0.14   # how far forward the camera sits from head origin (m)
```
More negative `up_offset` = lower on the face; larger `forward_offset` = further out
of the head mesh. Adjust in ±0.02 steps until the view matches the robot's eyes.

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
│   └── _update_headset_gait_command(dt)
│       ├── _get_headset_tracking_height() ← reads /user/head XR pose
│       ├── _update_headset_velocity(dt)   ← 3D velocity + horizontal gate
│       ├── low-pass filter on height
│       ├── peak/trough detection
│       └── pulse output × horiz_gate
│
├── h1.forward(dt, base_command)          ← H1FlatTerrainPolicy step
├── _update_h1_arms_from_hand_tracking()  ← OpenXR hand → arm DOFs
├── _update_head_camera_view()            ← move first-person camera
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
