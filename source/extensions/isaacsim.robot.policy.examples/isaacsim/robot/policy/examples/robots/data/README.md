# Third-party policy weights

## `g1_unitree_motion.pt`

Unitree's pretrained G1 flat-terrain walking policy, used by
`G1TeleopRobot(locomotion="policy")`.

- **Source:** [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym),
  `deploy/pre_train/g1/motion.pt`
- **License:** BSD 3-Clause — © 2016-2023 HangZhou YuShu TECHNOLOGY CO., LTD ("Unitree
  Robotics"). See the upstream `LICENSE`.
- **Architecture:** TorchScript `PolicyExporterLSTM` — `LSTM(47 -> 64)` into an
  `MLP(64 -> 32 -> 12)`, ELU. Stateful: it keeps its hidden and cell state internally.
- **Interface:** 47-dim observation in, 12 leg-joint actions out, 50 Hz.

To re-download:

```bash
curl -L -o g1_unitree_motion.pt \
  https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/deploy/pre_train/g1/motion.pt
```

`G1TeleopRobot` also looks in `~/BehavioralCollection/policies/g1_unitree_motion.pt`, or
you can pass an explicit path via `walk_policy_path=`. If no weights are found it logs
the download command and falls back to kinematic locomotion rather than failing to load.
