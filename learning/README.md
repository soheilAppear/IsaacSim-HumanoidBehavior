# Learning Pipeline — Humanoid Teleoperation Data

This directory reserves space for a learning pipeline built from the Unitree G1
teleoperation recordings. The Isaac Sim extension already records sessions, but
`learning/` currently contains directory placeholders and a dependency list only.
There is no dataset converter, training entry point, trained world model, or live
learned planner in this directory.

For running the robot and collecting data, use the
[project README](../README.md), [VR guide](../HUMANOID_VR_CONTROL.md), and
[control and recording reference](../docs/humanoid-control.md).

## Available recording data

The recorder lives in
[`humanoid_example.py`](../source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/humanoid_example.py).
When recording is enabled, loading the example creates a session under
`~/BehavioralCollection/raw_sessions/` (`~` is the user's home directory):

```text
session_YYYY-MM-DD_HH-MM-SS/
├── metadata.json
├── behavior.csv
├── hand_tracking.csv
├── gaze.csv
├── object_states.csv
├── frame_timestamps.csv
└── frames/eye_camera/
    └── *.png
```

The CSV streams contain robot and HMD state, hand poses and finger commands, gaze,
object state, and camera-frame timestamps. Sensor sampling is requested at 100 Hz
of simulation time, camera capture at 10 Hz, and buffered CSV writes every
2.5 seconds of simulation time. CSV buffers also flush when the session closes.
These are requested sampling rates, not measured wall-clock throughput; a slow
simulation records fewer simulated seconds per real second.

`metadata.json` identifies the robot, hand variant, simulation timing, locomotion
mode, finger actuator roles, and grasp mode. Use it with the timestamps and
`step_index` fields to interpret a session. Camera frames and sensor rows have
different sampling rates; a future converter must align them explicitly.

## Limits of the current data

- The default robot is the G1 with Inspire hands, fixed in place for arm and hand
  teleoperation. Current default sessions do not demonstrate walking or balance.
- Each hand exposes five finger curls and separate thumb opposition. Coupled
  robot knuckles do not reproduce every human joint independently.
- Valid optical landmarks drive open hands as well as closed hands. Real Quest
  landmark delivery has been intermittent; controller-only input and missing
  skeletons must not be labeled as valid optical finger demonstrations.
- Pickup uses a distance-gated fixed joint. The recorded grasp is assisted and
  does not establish that an unassisted physical grasp would hold the object.
- The camera is rigidly mounted to the robot body at head height. The HMD reader
  prefers physical-space poses but can fall back to virtual-world poses; the CSV
  does not identify that choice per row. Check the session log before treating HMD
  coordinates as physical motion or combining them with stage-space signals.
- Gaze rows identify their source through `gaze_source`. Preserve this field when
  distinguishing eye tracking from HMD-forward fallback.

Inspect tracking availability, source labels, timestamps, and successful object
movement before selecting sessions for a future training set. A live replay test
with generated inputs validates the controller; it does not establish the quality
of a real headset recording.

## Planned pipeline

The research direction is an action-conditioned world model, potentially using
frozen V-JEPA video features:

```text
recorded observations + human intent + candidate robot actions
  → predicted future robot/world state
  → evaluate candidate actions
  → execute a selected action in Isaac Sim
```

Only the recording stage is implemented. The remaining stages below are planned:

| Stage | Status |
| --- | --- |
| Session recording: metadata, PNG frames, and sensor CSVs | Implemented in the Isaac Sim extension |
| Timestamp alignment and windowed dataset conversion | Planned |
| CSV-only prediction baseline | Planned |
| Frozen video-embedding extraction | Planned |
| Multimodal and action-conditioned prediction | Planned |
| Offline planning and live learned control | Planned |
| Prediction, ablation, and task-success evaluation | Planned |

## Directory layout and environment

```text
learning/
├── configs/       # Placeholder for dataset and training configuration
├── data_tools/    # Placeholder for alignment and dataset conversion
├── models/        # Placeholder for model definitions
├── train/         # Placeholder for training entry points
├── eval/          # Placeholder for evaluation tools
├── inference/     # Placeholder for inference and planning
└── requirements.txt
```

The dependency list is a starting point for future development, not a tested
training environment or a requirement for running teleoperation. Keep future
learning dependencies in a separate Python virtual environment rather than
installing them into the Isaac Sim bundled interpreter.

Store datasets, embeddings, checkpoints, and reports outside the repository under
`~/BehavioralCollection/`. The existing [`.gitignore`](../.gitignore) also excludes
common generated media and model artifacts within `learning/`.
