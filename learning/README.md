# Learning Pipeline — Humanoid Teleoperation Data

This directory reserves space for a learning pipeline built from the Unitree G1
teleoperation recordings. The Isaac Sim extension already records sessions, but
`learning/` currently contains directory placeholders and a dependency list only.
There is no dataset converter, training entry point, trained world model, or live
learned planner in this directory.

For running the robot and collecting data, use the
[project README](../README.md), [VR guide](../HUMANOID_VR_CONTROL.md), and
[control and recording reference](../docs/humanoid-control.md).

The retained standalone examples are separate from this planned pipeline:

- [`g1_grasp_env.py`](../source/standalone_examples/api/isaacsim.robot.policy.examples/g1_grasp_env.py)
  builds cloned, fixed-base scenes and measures physics throughput. It has no
  action/reward interface, episode reset, training loop, or trained grasp policy;
  its printed policy-step budgets are extrapolations.
- [`g1_standalone.py`](../source/standalone_examples/api/isaacsim.robot.policy.examples/g1_standalone.py)
  demonstrates keyboard or scripted movement and finger closure using legacy
  assisted policy/kinematic modes. It does not collect VR demonstrations or run
  the current launcher's strict physical balance/walk profile.

Both files start an Isaac Sim application at module scope. Run them as separate
entry points; do not import them as dataset or training utilities.

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

The current files distinguish some, but not all, operator inputs from applied
robot actions:

| Role | Recorded today | Remaining gap |
| --- | --- | --- |
| Observations | RGB, robot base pose, joint positions/velocities, hand poses and validity, gaze source/hits, sample-object poses/contact status | Exact camera render timing and camera calibration per frame |
| Operator input | Controller buttons/grip/trigger, hand poses, smoothed finger requests | A complete calibrated task-intent label |
| Applied action | Filtered base velocity commands and contact-limited finger curl targets | Final arm joint targets, policy joint actions and body motor efforts |
| Outcomes | Subsequent measured joint/base/object states and contact/support diagnostics | Explicit episode boundaries, failure reasons and verified task success |

For action-conditioned learning, use commands actually sent to the robot rather
than treating a raw hand pose as its full motor action. Object linear velocities
are finite differences between logged positions; full contact forces and object
angular velocities are not in the object CSV. The recorder provides data for a
future pipeline, not a ready-made observation/action/next-state training set.

## Limits of the current data

- The default profile fixes the G1 pelvis for arm and hand teleoperation. The
  opt-in walking profile runs pretrained balance/walk policies. Select sessions
  using `robot_locomotion`, `base_fixed_to_world`, and the other metadata fields;
  a stationary session does not demonstrate walking or balance.
- Each hand exposes five finger curls and separate thumb opposition. Coupled
  robot knuckles do not reproduce every human joint independently.
- Valid optical landmarks drive open hands as well as closed hands. Real Quest
  landmark delivery has been intermittent; controller-only input and missing
  skeletons must not be labeled as valid optical finger demonstrations.
- Current physical grasping uses contact and friction, with no hand-to-object
  fixed joint. Earlier assisted sessions use a distance-gated attachment. Preserve
  `grasp_mode`: `is_grabbed` describes legacy attachment, while `contact_supported`
  describes opposing-contact evidence and does not prove a stable lift or hold.
- The current camera stabilizes robot travel/heading and uses HMD orientation to
  rotate the view and eye offset around the measured neck. Physical room translation
  does not translate the view. Earlier fixed-camera sessions use a different mapping.
  The raw HMD recorder can fall back from physical-space to virtual-world poses
  without a per-row source field; do not assume those values share the robot's
  world frame.
- Gaze rows identify their source through `gaze_source`. Preserve this field when
  distinguishing eye tracking from HMD-forward fallback. Gaze hits are observations,
  not verified task intent.
- Images are RGB only. Capture reads the most recent render and stamps the save;
  original render timestamps, camera extrinsics, depth and segmentation are absent.
  A future converter must handle image/state alignment and possible repeated renders.
- Reset, pause, tracking loss, synthetic replay intervals and grasp-mode changes
  must remain visible when making episodes. A continuous session clock does not
  imply an uninterrupted physical trajectory.

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
