# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connect local Steam Link expressions to the humanoid's existing body command path.

Face and gaze are explicitly selected modes. This avoids interpreting a wink's
temporary gaze shift as another walking command. Arm and finger input ownership
is unchanged. The host retains its native balance, stopping, and motion guards.
"""

from __future__ import annotations

import math
import os
import time

import carb
from isaacsim.robot.policy.examples.interactive.humanoid.face_tracking_input import SteamLinkFaceInput
from isaacsim.robot.policy.examples.interactive.humanoid.facial_locomotion import (
    FacialLocomotion,
    FacialLocomotionConfig,
)
from isaacsim.robot.policy.examples.interactive.humanoid.gaze_locomotion import GazeLocomotion, GazeLocomotionConfig


class FacialLocomotionIntegration:
    """Own the optional face transport, mode switch, and locomotion state machine.

    Args:
        host: Loaded humanoid example providing pose, body buffers, and headset HUD.
    """

    def __init__(self, host) -> None:
        self.host = host
        preview = os.environ.get("G1_FACE_PREVIEW", "1").strip().lower()
        if preview not in ("0", "1", "false", "true", "no", "yes", "off", "on"):
            raise ValueError("G1_FACE_PREVIEW must be a boolean")
        config = FacialLocomotionConfig(
            preview=host._gaze_locomotion.config.preview or preview in ("1", "true", "yes", "on"),
            control_mode=os.environ.get("G1_FACE_MODE", "hold").strip().lower(),
            distance_m=float(os.environ.get("G1_FACE_DISTANCE_M", "0.15")),
            turn_degrees=float(os.environ.get("G1_FACE_TURN_DEG", "10")),
            forward_speed=float(os.environ.get("G1_FACE_FORWARD_SPEED", "0.20")),
            backward_speed=float(os.environ.get("G1_FACE_BACKWARD_SPEED", "0.15")),
            lateral_speed=float(os.environ.get("G1_FACE_LATERAL_SPEED", "0.15")),
            yaw_speed=float(os.environ.get("G1_FACE_YAW_SPEED", "0.25")),
            activation_threshold=float(os.environ.get("G1_FACE_ACTIVATION_THRESHOLD", "0.65")),
            wink_threshold=(
                float(os.environ["G1_FACE_WINK_THRESHOLD"]) if "G1_FACE_WINK_THRESHOLD" in os.environ else None
            ),
            jaw_threshold=float(os.environ["G1_FACE_JAW_THRESHOLD"]) if "G1_FACE_JAW_THRESHOLD" in os.environ else None,
            pucker_threshold=(
                float(os.environ["G1_FACE_PUCKER_THRESHOLD"]) if "G1_FACE_PUCKER_THRESHOLD" in os.environ else None
            ),
            mouth_side_threshold=(
                float(os.environ["G1_FACE_MOUTH_SIDE_THRESHOLD"])
                if "G1_FACE_MOUTH_SIDE_THRESHOLD" in os.environ
                else None
            ),
            release_threshold=float(os.environ.get("G1_FACE_RELEASE_THRESHOLD", "0.25")),
            eye_open_threshold=float(os.environ.get("G1_FACE_EYE_OPEN_THRESHOLD", "0.15")),
            confirm_s=float(os.environ.get("G1_FACE_CONFIRM_S", "0.18")),
            post_blink_wink_confirm_s=float(os.environ.get("G1_FACE_POST_BLINK_WINK_CONFIRM_S", "0.50")),
            neutral_s=float(os.environ.get("G1_FACE_NEUTRAL_S", "0.25")),
        )
        self.controller = FacialLocomotion(config)
        self.receiver = SteamLinkFaceInput(port=int(os.environ.get("G1_FACE_OSC_PORT", "9015")))
        self.toggle = GazeLocomotion(GazeLocomotionConfig(preview=True, dwell_s=0.7))
        self.enabled = False
        self.error = None
        self.sample = None

    def start(self) -> None:
        """Open the exclusive loopback receiver without preventing normal gaze use on failure."""
        self.close()
        self.error = None
        try:
            self.receiver.start()
        except OSError as error:
            self.error = str(error)
            carb.log_warn(f"Face tracking unavailable: {error}. Close any other receiver on the OSC port and reload.")

    def close(self) -> None:
        """Release the socket and require explicit facial rearming after scene reload."""
        self.receiver.close()
        self.reset()
        self.sample = None

    def reset(self) -> None:
        """Disarm on timeline/reset/recenter events and discard incomplete gestures."""
        self.enabled = False
        self.controller.reset()
        self.toggle.reset()
        self.host._gaze_controls.update_face_toggle(False, 0.0)

    def set_enabled(self, enabled: bool) -> dict:
        """Switch body ownership and zero old intent before requiring a neutral face."""
        if type(enabled) is not bool:
            raise TypeError("Facial control enabled must be a boolean")
        if enabled and self.error is not None:
            raise RuntimeError(f"Facial tracking receiver unavailable: {self.error}")
        self.enabled = enabled
        self.controller.reset()
        self.host._gaze_locomotion.reset()
        self.host._hybrid_locomotion.reset()
        self.host._controller_locomotion.reset()
        self.host._hybrid_controller_velocity = (0.0, 0.0, 0.0)
        self.host._hybrid_display_status = None
        # Keep measured stop/runaway supervision across mode changes. A new input
        # owner must not erase evidence that the physical body is still moving.
        for name in ("_base_command", "_keyboard_command", "_controller_command"):
            command = getattr(self.host, name, None)
            if command is not None:
                command[:] = 0.0
        self.host._gaze_controls.update_face_toggle(enabled, 0.0)
        return self.status()

    def _session_ready(self) -> bool:
        """Check XR display/session state without requiring an open-eye gaze ray."""
        tracker = self.host._eye_gaze_tracker
        return bool(
            self.host._physics_ready
            and not self.host._timeline_stopped
            and self.host._articulation_health_fault is None
            and tracker is not None
            and tracker.control_session_ready()
        )

    def update(self, dt: float) -> bool:
        """Poll and optionally own this body's input tick, leaving hands untouched.

        Returns:
            True when facial control or a mode handoff supplied the body command.
        """
        now = time.monotonic()
        host = self.host
        self.sample = self.receiver.poll(now_s=now)
        ready = self._session_ready()
        tracker = host._eye_gaze_tracker
        ray = tracker.read_control_ray() if tracker is not None else None
        toggle_hit = False
        toggle_valid = ray is not None
        if ray is not None:
            toggle_hit = host._gaze_controls.hit_test_face_toggle(*ray)
            if toggle_hit and not host._gaze_controls.hit_test_face_toggle(
                *ray, max_distance=tracker.control_occlusion_distance(*ray)
            ):
                toggle_valid = False
        toggle = self.toggle.update(
            target="forward" if toggle_hit else None,
            eye_valid=toggle_valid,
            ready=ready,
            now_s=now,
            dt=dt,
            pose=None,
            linear_speed_m_s=0.0,
            yaw_speed_rad_s=0.0,
        )
        if toggle.activated_action is not None:
            if not self.enabled and self.error is not None:
                carb.log_warn(f"Cannot enable face controls: {self.error}")
            else:
                self.set_enabled(not self.enabled)
                # Do not let a different mode publish in the same switch tick.
                return True
        host._gaze_controls.update_face_toggle(self.enabled, toggle.progress if toggle_hit else 0.0)
        if not self.enabled:
            return False

        if host._gaze_controller_override_enabled:
            devices = tuple(host._get_xr_input_device(f"/user/hand/{side}") for side in ("left", "right"))
            left, right = (host._read_tracked_controller_stick(device) for device in devices)
            deadzone = host._controller_locomotion.deadzone
            if (left[0] and max(abs(left[1]), abs(left[2])) > deadzone) or (right[0] and abs(right[1]) > deadzone):
                self.set_enabled(False)
                return True

        pose = None
        linear_speed = yaw_speed = 0.0
        if not self.controller.config.preview and ready:
            base = host._get_g1_base_pose_for_arms()
            if base is not None:
                pose = (float(base[0][0]), float(base[0][1]), float(base[1]))
            linear, angular = host.g1.robot.get_velocities()
            linear = tuple(float(value) for value in host._first_pose_value(linear))
            angular = tuple(float(value) for value in host._first_pose_value(angular))
            if len(linear) != 3 or len(angular) != 3:
                raise ValueError("Invalid facial walking velocity dimensions")
            linear_speed, yaw_speed = math.hypot(linear[0], linear[1]), abs(angular[2])
        command = self.controller.update(
            sample=self.sample,
            now_s=now,
            dt=dt,
            pose=pose,
            linear_speed_m_s=linear_speed,
            yaw_speed_rad_s=yaw_speed,
            ready=ready and not host._keyboard_brake,
            enabled=True,
        )
        host._keyboard_command[:] = 0.0
        host._locomotion_brake = host._keyboard_brake
        for axis, value in enumerate(command.velocity):
            host._controller_command[axis] = value
        if command.stop_immediately or self.controller.config.preview:
            host._base_command[:] = 0.0
        host._navigation_scene.hide()
        host._gaze_controls.update_facial(command)
        if command.activated_action is not None:
            carb.log_info(f"Facial walking: {command.detail} (selection {command.activation_count})")
        if command.state == "fault":
            host._articulation_health_fault = f"facial walking: {command.detail}"
            host._pause_faulted_articulation()
        return True

    def status(self) -> dict:
        """Report input validity and control state without consuming another packet."""
        command = self.controller.command
        return {
            "available": True,
            "enabled": self.enabled,
            "preview": self.controller.config.preview,
            "control_mode": self.controller.config.control_mode,
            "state": command.state,
            "detail": command.detail,
            "target": command.target,
            "last_action": command.last_action,
            "activation_count": command.activation_count,
            "velocity": command.velocity,
            "sample_valid": bool(self.sample is not None and self.sample.valid),
            "sample_reason": self.sample.reason if self.sample is not None else "No sample polled",
            "confidence_upper": self.sample.confidence_upper if self.sample is not None else 0.0,
            "confidence_lower": self.sample.confidence_lower if self.sample is not None else 0.0,
            "weights": dict(self.sample.weights) if self.sample is not None else {},
            "thresholds": {
                "wink": self.controller._threshold_for("turn_left"),
                "jaw": self.controller._threshold_for("forward"),
                "pucker": self.controller._threshold_for("back"),
                "mouth_side": self.controller._threshold_for("strafe_left"),
                "release": self.controller.config.release_threshold,
                "eye_open": self.controller.config.eye_open_threshold,
                "confirm_s": self.controller.config.confirm_s,
                "post_blink_wink_confirm_s": self.controller.config.post_blink_wink_confirm_s,
                "neutral_s": self.controller.config.neutral_s,
            },
            "transport": self.receiver.diagnostics(),
            "error": self.error,
        }
