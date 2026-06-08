# Humanoid VR Control — H1 Robot with Gait Detection & Behavioral Data Logger

> **Fork of:** [NVIDIA IsaacSim](https://github.com/isaac-sim/IsaacSim) · Isaac Sim 6.0.0 GA  
> **File modified:** `source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/humanoid_example.py`  
> **Author:** [@soheilAppear](https://github.com/soheilAppear)

---

## What This Is

This fork extends the stock Isaac Sim H1 humanoid interactive example with a full **VR-driven locomotion and data-collection pipeline**:

| Feature | Description |
|---------|-------------|
| **Headset gait detection** | Bob your head up/down in VR to make the robot walk — uses real-time peak/trough signal processing on the HMD height signal |
| **Horizontal motion gate** | Prevents false walking from pure head nodding — gait is suppressed unless the headset also moves in the floor plane |
| **Eye-level first-person camera** | Viewport camera placed at robot eye height (not top of skull) |
| **VR hand tracking → arm control** | OpenXR hand/controller poses drive H1 arm joints |
| **Grab system** | Physical sample boxes in the scene can be grabbed with controllers |
| **Behavioral data logger** | Every session auto-saves a rich ~100 Hz CSV to `~/BehavioralCollection/` — suitable for AI/RL training |

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
│                   └── humanoid_example.py   ← the modified file
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
  "source\extensions\isaacsim.robot.policy.examples\isaacsim\robot\policy\examples\interactive\humanoid\humanoid_example.py" `
  "$isaacRoot\$extPath\humanoid_example.py"
```

**Linux:**
```bash
ISAAC_ROOT="/path/to/isaac-sim-standalone-6.0.0-linux-x86_64"
EXT_PATH="exts/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid"

cp source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/humanoid_example.py \
   "$ISAAC_ROOT/$EXT_PATH/humanoid_example.py"
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

Every simulation session automatically saves a CSV to:

```
~/BehavioralCollection/behavior_YYYY-MM-DD_HH-MM-SS_<unix_timestamp>.csv
```

The file is written when the simulation stops (clicking Stop or closing Isaac Sim).

### CSV Schema

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

df = pd.read_csv("~/BehavioralCollection/behavior_2026-06-07_22-30-00_1749340200.csv")

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
self._first_person_head_up_offset     = -0.10   # camera height offset from head prim (m, negative = lower)
self._first_person_head_fallback_height = 1.55  # eye height above robot base when no head prim found (m)
self._first_person_head_forward_offset = 0.10   # how far forward the camera sits from head origin (m)
```

### Data collection
```python
self._behavioral_data_log_every_n_steps = 2     # 1=200Hz, 2=100Hz, 4=50Hz
self._behavioral_data_output_dir = Path.home() / "BehavioralCollection"
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
└── _collect_behavioral_sample()          ← write row to in-memory buffer
                                             (flushed to CSV on stop)
```

---

## License

This file is a modification of NVIDIA Isaac Sim source code and is licensed under the **Apache License 2.0**, the same license as the upstream repository.  
See [LICENSE](LICENSE) for the full text.

Original copyright: © 2020–2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  
Modifications: © 2026 Soheil Sepahyar.
