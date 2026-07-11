# Learning Pipeline — V-JEPA-Based World Model for H1 VR Control

This folder is a separate learning pipeline that consumes data recorded by the
Isaac Sim H1 humanoid VR extension (see [`HUMANOID_VR_CONTROL.md`](../HUMANOID_VR_CONTROL.md))
and turns it into training data for embodied world models.

It is independent of the Isaac Sim / Kit Python environment — install its
dependencies (`requirements.txt`) into a normal Python virtual environment,
not the Isaac Sim bundled interpreter.

## Goal

Move from VR teleoperation of the H1 robot toward a learned world model:

```
current observation + human intent signals + candidate robot action
  → predicted future robot/world state
  → choose the best action
  → execute in Isaac Sim
```

## Phased plan

| Phase | Description |
|---|---|
| 0 | Repo scaffolding (this folder) — no data yet |
| 1 | Extend the Isaac Sim logger: per-session folders with synchronized video, HMD, hand, gaze, and object-state logs |
| 2 | Build a dataset converter that time-aligns video frames with sensor CSVs into windowed training samples |
| 3 | Train a CSV-only baseline (Transformer/GRU) predicting future commands and base motion — prove the sensor data is useful before touching video |
| 4 | Extract frozen pretrained V-JEPA 2 video embeddings per session clip |
| 5 | Train a multimodal predictor (V-JEPA embedding + sensor sequence → future state) and compare against the Phase 3 baseline |
| 6 | Train an action-conditioned latent world model: `z_t + action_sequence → z_t+H` |
| 7 | Offline latent-space MPC planner over candidate action sequences |
| 8 | Wire a trained policy/planner back into the live Isaac Sim callback, with safety fallbacks to manual VR control |
| 9 | Evaluation: prediction error, ablations (HMD-only vs. +hands vs. +gaze vs. +video), planning success rate |

## Folder layout

```
learning/
├── configs/       # dataset.yaml, train_*.yaml
├── data_tools/     # session -> synchronized, windowed dataset conversion
├── models/         # model definitions (sensor encoder, transformer, V-JEPA wrapper, predictors)
├── train/          # training entry points
├── eval/           # evaluation / plotting scripts
└── inference/      # live policy stub, MPC planner
```

Datasets, embeddings, checkpoints, and reports live outside the repo under
`~/BehavioralCollection/` (`raw_sessions/`, `processed_sessions/`, `embeddings/`,
`models/`, `reports/`) and are never committed — see the root `.gitignore`.

## Status

Phase 0 only: folder scaffolding. No session-folder logging, sync tooling, or
models exist yet. `humanoid_example.py` is unmodified by this phase.
