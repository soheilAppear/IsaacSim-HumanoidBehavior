# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned GR00T Decoupled WBC inference with independently controlled arms.

The two actors command legs and waist, observing all 29 body joints. Unlike the
full-body velocity actor, they never overwrite a teleoperator's arm targets.
Observation histories follow the deployed six-frame, oldest-first contract.
No simulator, root-motion, external-force, or tracking-device APIs live here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class G1DecoupledPolicy:
    """Load a paired balance/walk actor and its checkpoint-specific contract."""

    def __init__(self, metadata_path: str | Path | None = None):
        data = Path(__file__).resolve().parent / "data"
        self.metadata_path = Path(metadata_path) if metadata_path is not None else data / "g1_decoupled_policy.json"
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != 1:
            raise ValueError("Unsupported decoupled-policy metadata version")
        self.joint_names = tuple(self.metadata["joint_names"])
        self.action_joint_names = tuple(self.metadata["action_joint_names"])
        if len(self.joint_names) != 29 or len(set(self.joint_names)) != 29:
            raise ValueError("Decoupled WBC requires 29 unique observed body joints")
        if self.action_joint_names != self.joint_names[:15] or any(
            "shoulder" in name or "elbow" in name or "wrist" in name for name in self.action_joint_names
        ):
            raise ValueError("Decoupled WBC actions must contain the first 15 leg/waist joints only")
        self.default_positions = self._vector(self.metadata["default_positions"], 29, "default positions")
        self.initial_positions = self._vector(self.metadata["initial_positions"], 29, "initial positions")
        self.stiffness = self._vector(self.metadata["stiffness"], 29, "stiffness")
        self.damping = self._vector(self.metadata["damping"], 29, "damping")
        for value in (self.stiffness, self.damping):
            if np.any(value < 0):
                raise ValueError("Decoupled actuator gains must be nonnegative")
        # A missing deployment override means retaining the robot asset's value.
        for attribute, key in (
            ("armature", "armature"),
            ("effort_limits", "effort_limit_sim"),
            ("velocity_limits", "velocity_limit_sim"),
        ):
            value = self.metadata.get(key)
            value = None if value is None else self._vector(value, 29, key, broadcast=True)
            if value is not None and np.any(value < 0):
                raise ValueError(f"{key} must be nonnegative")
            setattr(self, attribute, value)
        self.action_scale = self._vector(self.metadata["action_scale"], 15, "action scale", broadcast=True)
        self.command_scale = self._vector(self.metadata["command_scale"], 3, "command scale")
        self.angular_scale = float(self.metadata["angular_scale"])
        self.joint_velocity_scale = float(self.metadata["joint_vel_scale"])
        self.control_dt = float(self.metadata["control_dt"])
        self.physics_dt = float(self.metadata["physics_dt"])
        self.balance_threshold = float(self.metadata["balance_threshold"])
        self.height_command = float(self.metadata["height_cmd"])
        self.torso_rpy = self._vector(self.metadata["torso_rpy"], 3, "torso command")
        if (
            self.control_dt != 0.02
            or self.physics_dt != 0.005
            or self.metadata["history_length"] != 6
            or self.metadata["frame_dimension"] != 86
            or not np.isfinite(
                [self.angular_scale, self.joint_velocity_scale, self.balance_threshold, self.height_command]
            ).all()
            or self.balance_threshold <= 0
        ):
            raise ValueError("Decoupled WBC requires finite deployment scales and its 50/200 Hz, 6x86 contract")
        self._sessions = {}
        self.reset()

    @staticmethod
    def _vector(value: object, size: int, label: str, *, broadcast: bool = False) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if broadcast and array.ndim == 0:
            array = np.full(size, array, dtype=np.float32)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(f"{label} must contain {size} finite values")
        return array.copy()

    def resolve_joint_indices(self, dof_names: list[str]) -> list[int]:
        """Resolve all observed joints by name, independent of tensor/SDK ordering."""
        if len(dof_names) != len(set(dof_names)):
            raise ValueError("Articulation contains ambiguous duplicate DOF names")
        missing = [name for name in self.joint_names if name not in dof_names]
        if missing:
            raise ValueError(f"Decoupled WBC joints missing from articulation: {missing}")
        return [dof_names.index(name) for name in self.joint_names]

    def load(self) -> None:
        """Validate both pinned binaries before publishing either runtime session."""
        import onnxruntime as ort

        sessions = {}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        for mode in ("balance", "walk"):
            spec = self.metadata["models"][mode]
            path = self.metadata_path.parent / spec["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
                raise ValueError(f"Decoupled {mode} checkpoint checksum does not match metadata")
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            inputs, outputs = session.get_inputs(), session.get_outputs()
            # The released models name a dynamic batch dimension; this adapter
            # always supplies one robot and validates the returned batch below.
            input_shape = inputs[0].shape if len(inputs) == 1 else []
            output_shape = outputs[0].shape if len(outputs) == 1 else []
            if (
                len(inputs) != 1
                or inputs[0].name != spec["input_name"]
                or len(input_shape) != 2
                or input_shape[1] != 516
                or not (input_shape[0] == 1 or isinstance(input_shape[0], str))
                or inputs[0].type != "tensor(float)"
                or len(outputs) != 1
                or outputs[0].name != spec["output_name"]
                or len(output_shape) != 2
                or output_shape[1] != 15
                or not (output_shape[0] == 1 or isinstance(output_shape[0], str))
                or outputs[0].type != "tensor(float)"
            ):
                raise ValueError(f"Decoupled {mode} ONNX interface must be float32 [1,516] -> [1,15]")
            sessions[mode] = session
        self._sessions = sessions
        self.reset()

    def reset(self) -> None:
        """Start with five zero past frames, as in the reference deployment."""
        self.last_action = np.zeros(15, dtype=np.float32)
        self.last_observation = None
        self.active_mode = None
        self._history = np.zeros((6, 86), dtype=np.float32)

    def infer(
        self,
        positions: object,
        velocities: object,
        angular_velocity: object,
        projected_gravity: object,
        command: object,
    ) -> np.ndarray:
        """Return 15 leg/waist targets from measured body-frame state.

        ``height_command`` and ``torso_rpy`` default to the released deployment
        values. Arms affect balance through measured q/qd, never fabricated past
        actions: the previous-action term contains only the 15 policy outputs.
        """
        if set(self._sessions) != {"balance", "walk"}:
            raise RuntimeError("Decoupled WBC checkpoints have not been loaded")
        command = self._vector(command, 3, "command")
        height = self._vector([self.height_command], 1, "height command")
        frame = np.concatenate(
            (
                command * self.command_scale,
                height,
                self._vector(self.torso_rpy, 3, "torso command"),
                self._vector(angular_velocity, 3, "angular velocity") * self.angular_scale,
                self._vector(projected_gravity, 3, "projected gravity"),
                self._vector(positions, 29, "joint positions") - self.default_positions,
                self._vector(velocities, 29, "joint velocities") * self.joint_velocity_scale,
                self._vector(self.last_action, 15, "last action"),
            )
        )
        history = np.concatenate((self._history[1:], frame[None, :]), axis=0)
        observation = history.reshape(1, 516)
        if not np.isfinite(observation).all():
            raise ValueError("Decoupled WBC observation must contain 516 finite values")
        mode = "balance" if np.linalg.norm(command) < self.balance_threshold else "walk"
        spec = self.metadata["models"][mode]
        result = self._sessions[mode].run([spec["output_name"]], {spec["input_name"]: observation})
        if len(result) != 1 or np.asarray(result[0]).shape != (1, 15):
            raise ValueError("Decoupled WBC must return [1,15] actions")
        action = self._vector(result[0][0], 15, "policy action")
        targets = action * self.action_scale + self.default_positions[:15]
        if not np.isfinite(targets).all():
            raise ValueError("Decoupled WBC position targets are nonfinite")
        self._history = history
        self.last_action = action
        self.last_observation = observation.copy()
        self.active_mode = mode
        return targets
