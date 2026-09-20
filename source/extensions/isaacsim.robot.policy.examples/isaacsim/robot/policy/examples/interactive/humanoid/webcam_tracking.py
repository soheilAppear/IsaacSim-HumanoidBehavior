# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receive optional, local webcam landmarks without importing camera or ML libraries.

The camera process sends unmirrored MediaPipe world landmarks, in estimated
metres, over loopback UDP. Timestamps use ``time.perf_counter_ns()`` in both
processes (Windows QueryPerformanceCounter), not ``time.monotonic_ns()`` whose
implementation/origin differs between the Isaac and standalone Python versions.
These timestamps are comparable only on this PC; this is deliberately not a
network/remote-camera protocol. Body reconstruction
uses anatomical shoulders and hips to produce +X forward, +Y left, +Z up.
Coordinates are shoulder-relative, not measured room positions. No root, gaze,
orientation, finger, or articulation commands are issued by this module.
"""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BodyArm",
    "BodyPose",
    "WebcamArmTarget",
    "WebcamCalibration",
    "WebcamFrame",
    "WebcamReceiver",
    "decode_frame",
    "derive_body_pose",
]

SCHEMA_VERSION = 1
MAX_DATAGRAM_BYTES = 32768
MAX_FRAME_AGE_S = 0.25
MAX_CALIBRATION_AGE_S = 0.10
MIN_LANDMARK_CONFIDENCE = 0.65
SIDES = ("left", "right")
Vec3 = tuple[float, float, float]
PoseLandmark = tuple[float, float, float, float, float]


def _number(value: Any, *, minimum: float, maximum: float) -> float:
    """Validate a finite JSON number without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Landmarks and confidence must be numbers")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("Landmark or confidence number exceeds finite bounds") from error
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError("Landmark or confidence is outside its finite bounds")
    return number


def _integer(value: Any, name: str) -> int:
    """Validate a bounded, nonnegative JSON integer."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be a nonnegative 63-bit integer")
    return value


def _vec3(value: Any, bound: float = 5.0) -> Vec3:
    """Validate a three-component finite coordinate."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Coordinates must have three components")
    return tuple(_number(component, minimum=-bound, maximum=bound) for component in value)


@dataclass(frozen=True)
class WebcamFrame:
    """A validated webcam sample; missing pose and hands represent tracking loss."""

    sender_session: str
    seq: int
    capture_monotonic_ns: int
    pose_world: tuple[PoseLandmark, ...] | None
    hand_world: dict[str, tuple[Vec3, ...]]
    hand_confidence: dict[str, float]
    calibration_request: int = 0

    def age_s(self, now_ns: int | None = None) -> float:
        """Return capture age on this PC's monotonic clock, including inference."""
        now = time.perf_counter_ns() if now_ns is None else now_ns
        return (now - self.capture_monotonic_ns) / 1e9


def decode_frame(payload: bytes, *, now_ns: int | None = None, max_age_s: float = MAX_FRAME_AGE_S) -> WebcamFrame:
    """Validate one protocol-v1 packet and reject stale or future captures.

    Args:
        payload: UTF-8 JSON datagram from the local sender.
        now_ns: Current local monotonic time, injectable for deterministic tests.
        max_age_s: Maximum capture age, including inference and transport.

    Returns:
        A validated frame containing only supported control fields.

    Raises:
        ValueError: If the packet is malformed, oversized, old, or from the future.
    """
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_DATAGRAM_BYTES:
        raise ValueError("Webcam packet is empty or exceeds 32 KiB")
    try:
        data = json.loads(payload)
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise ValueError("Webcam packet is not valid UTF-8 JSON") from error
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported webcam protocol version")
    session = data.get("sender_session")
    if (
        not isinstance(session, str)
        or not 1 <= len(session) <= 64
        or not session.isascii()
        or not session.isprintable()
    ):
        raise ValueError("Sender session must be a short printable ASCII identifier")
    seq = _integer(data.get("seq"), "Sequence")
    captured = _integer(data.get("capture_monotonic_ns"), "Capture timestamp")
    now = time.perf_counter_ns() if now_ns is None else now_ns
    age_s = (now - captured) / 1e9
    if age_s < -0.005 or age_s > max_age_s:
        raise ValueError("Webcam frame is stale or has a future capture timestamp")
    pose = data.get("pose_world")
    if pose is not None:
        if not isinstance(pose, list) or len(pose) != 33:
            raise ValueError("Pose must contain exactly 33 world landmarks")
        validated = []
        for landmark in pose:
            if not isinstance(landmark, list) or len(landmark) != 5:
                raise ValueError("Pose landmark must contain xyz, visibility, and presence")
            xyz = _vec3(landmark[:3])
            confidence = tuple(_number(value, minimum=0.0, maximum=1.0) for value in landmark[3:])
            validated.append((*xyz, *confidence))
        pose = tuple(validated)
    hands = data.get("hand_world", {})
    scores = data.get("hand_confidence", {})
    if (
        not isinstance(hands, dict)
        or not isinstance(scores, dict)
        or set(hands) - set(SIDES)
        or set(scores) - set(SIDES)
    ):
        raise ValueError("Hand maps must use anatomical left/right keys")
    hand_world = {}
    hand_confidence = {}
    for side, landmarks in hands.items():
        if not isinstance(landmarks, list) or len(landmarks) != 21:
            raise ValueError("Hand must contain exactly 21 world landmarks")
        hand_world[side] = tuple(_vec3(value, bound=1.0) for value in landmarks)
        hand_confidence[side] = _number(scores.get(side), minimum=0.0, maximum=1.0)
    request = _integer(data.get("calibration_request", 0), "Calibration request")
    return WebcamFrame(session, seq, captured, pose, hand_world, hand_confidence, request)


class WebcamReceiver:
    """Drain bounded loopback UDP input without waiting inside a physics callback.

    Args:
        port: Local UDP port; zero selects a free port for tests.
        max_age_s: Maximum accepted capture age.

    Raises:
        OSError: If the local port cannot be exclusively bound.
        ValueError: If the port or age limit is invalid.

    Example:

    .. code-block:: python

        receiver = WebcamReceiver()
        try:
            frame = receiver.poll()  # None until a fresh local sample arrives.
        finally:
            receiver.close()
    """

    def __init__(self, port: int = 8765, max_age_s: float = MAX_FRAME_AGE_S) -> None:
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("Port must be an integer between zero and 65535")
        if not math.isfinite(max_age_s) or not 0.01 <= max_age_s <= 1.0:
            raise ValueError("Frame age limit must be between 0.01 and 1 second")
        self.max_age_s = max_age_s
        self.latest: WebcamFrame | None = None
        self.accepted_packets = 0
        self.rejected_packets = 0
        self.last_error: str | None = None
        self._retired_sessions: set[str] = set()
        self._socket: socket.socket | None = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            self._socket.bind(("127.0.0.1", port))
            self._socket.setblocking(False)
            self.port = self._socket.getsockname()[1]
        except OSError:
            self.close()
            raise

    @property
    def is_open(self) -> bool:
        """Return whether the local UDP socket remains available for reception."""
        return self._socket is not None

    def close(self) -> None:
        """Release the UDP port and invalidate the cached frame; safe to repeat."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self.latest = None

    def accept(self, payload: bytes, *, now_ns: int | None = None) -> bool:
        """Validate and order a packet, returning whether it replaced the cache.

        A sender restart requires a new session identifier and a newer capture.
        Retired sessions cannot take control again via delayed datagrams.
        """
        try:
            frame = decode_frame(payload, now_ns=now_ns, max_age_s=self.max_age_s)
            if frame.sender_session in self._retired_sessions:
                raise ValueError("Frame belongs to a retired sender session")
            previous = self.latest
            if previous is not None:
                if frame.capture_monotonic_ns <= previous.capture_monotonic_ns:
                    raise ValueError("Capture timestamp did not advance")
                if frame.sender_session == previous.sender_session:
                    if frame.seq <= previous.seq:
                        raise ValueError("Frame sequence did not advance")
                else:
                    if len(self._retired_sessions) >= 32:
                        raise ValueError("Too many sender restarts; re-enable webcam input")
                    self._retired_sessions.add(previous.sender_session)
            self.latest = frame
            self.accepted_packets += 1
            self.last_error = None
            return True
        except ValueError as error:
            self.rejected_packets += 1
            self.last_error = str(error)
            return False

    def poll(self, *, now_ns: int | None = None) -> WebcamFrame | None:
        """Return the latest fresh frame, or None on loss, closure, or backlog.

        At most 128 datagrams are consumed per call. A full drain budget yields
        no control sample until the queue catches up, keeping callback time
        bounded without knowingly using a sample behind a backlog.
        """
        if self._socket is None:
            return None
        now = time.perf_counter_ns() if now_ns is None else now_ns
        for _ in range(128):
            try:
                payload, address = self._socket.recvfrom(MAX_DATAGRAM_BYTES + 1)
            except BlockingIOError:
                break
            except OSError as error:
                self.last_error = str(error)
                self.close()
                return None
            if address[0] != "127.0.0.1":
                self.rejected_packets += 1
                continue
            self.accept(payload, now_ns=now)
        else:
            self.last_error = "Webcam input backlog; waiting for latest sample"
            return None
        if self.latest is not None and 0 <= (now - self.latest.capture_monotonic_ns) / 1e9 <= self.max_age_s:
            return self.latest
        return None


def _subtract(left: Vec3, right: Vec3) -> Vec3:
    """Subtract two vectors."""
    return tuple(a - b for a, b in zip(left, right))


def _dot(left: Vec3, right: Vec3) -> float:
    """Compute the vector dot product."""
    return sum(a * b for a, b in zip(left, right))


def _length(vector: Vec3) -> float:
    """Compute a vector's Euclidean length."""
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vec3) -> Vec3:
    """Normalize a nondegenerate vector."""
    length = _length(vector)
    if length < 1e-6:
        raise ValueError("Body landmarks do not define a stable coordinate frame")
    return tuple(component / length for component in vector)


def _cross(left: Vec3, right: Vec3) -> Vec3:
    """Compute the vector cross product."""
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


@dataclass(frozen=True)
class BodyArm:
    """Anatomical body-frame shoulder-relative arm landmarks in estimated metres."""

    wrist: Vec3
    elbow: Vec3
    length: float
    confidence: float


@dataclass(frozen=True)
class BodyPose:
    """Both visible arms and the capture identity used for explicit calibration."""

    sender_session: str
    capture_monotonic_ns: int
    arms: dict[str, BodyArm]


def derive_body_pose(frame: WebcamFrame, min_confidence: float = MIN_LANDMARK_CONFIDENCE) -> BodyPose | None:
    """Derive shoulder-relative arms from a confident, nondegenerate torso.

    Args:
        frame: Validated webcam frame.
        min_confidence: Minimum visibility and presence for every used landmark.

    Returns:
        A body pose, or None when hips/shoulders cannot define a stable frame.
        Individually obscured arms are omitted rather than extrapolated.
    """
    pose = frame.pose_world
    if pose is None:
        return None
    confident = lambda indices: min(min(pose[index][3:]) for index in indices) >= min_confidence
    torso = (11, 12, 23, 24)
    if not confident(torso):
        return None
    shoulder_mid = tuple((pose[11][axis] + pose[12][axis]) * 0.5 for axis in range(3))
    hip_mid = tuple((pose[23][axis] + pose[24][axis]) * 0.5 for axis in range(3))
    lateral = _subtract(pose[11][:3], pose[12][:3])
    vertical = _subtract(shoulder_mid, hip_mid)
    if not 0.15 <= _length(lateral) <= 0.8 or not 0.15 <= _length(vertical) <= 1.0:
        return None
    try:
        left = _normalize(lateral)
        forward = _normalize(_cross(left, vertical))
        up = _normalize(_cross(forward, left))
    except ValueError:
        return None
    # Near-collinear torso axes magnify monocular depth noise.
    if abs(_dot(left, _normalize(vertical))) > 0.8:
        return None
    arms = {}
    for side, indices in (("left", (11, 13, 15)), ("right", (12, 14, 16))):
        if not confident((*torso, *indices)):
            continue
        shoulder, elbow, wrist = (pose[index][:3] for index in indices)
        upper = _length(_subtract(elbow, shoulder))
        lower = _length(_subtract(wrist, elbow))
        if not 0.10 <= upper <= 0.60 or not 0.10 <= lower <= 0.60:
            continue
        wrist_relative = _subtract(wrist, shoulder)
        elbow_relative = _subtract(elbow, shoulder)
        axes = (forward, left, up)
        arms[side] = BodyArm(
            tuple(_dot(wrist_relative, axis) for axis in axes),
            tuple(_dot(elbow_relative, axis) for axis in axes),
            upper + lower,
            min(min(pose[index][3:]) for index in (*torso, *indices)),
        )
    return BodyPose(frame.sender_session, frame.capture_monotonic_ns, arms)


@dataclass(frozen=True)
class WebcamArmTarget:
    """A calibrated wrist fallback and elbow guide in the robot body frame."""

    wrist_body: Vec3
    elbow_body: Vec3
    confidence: float


class WebcamCalibration:
    """Map shoulder-relative webcam motion onto explicitly paired Quest targets.

    Calibration is deliberately an explicit operation. A user must face the
    webcam, keep both hands visible to both trackers, and request calibration.
    Estimated human arm length supplies scale; each simultaneous Quest palm
    target supplies translation and the wrist-to-palm offset. This is anatomical
    retargeting, not camera extrinsic calibration or room-position measurement.
    A sender restart invalidates the calibration until the user repeats it.

    Example:

    .. code-block:: python

        alignment = WebcamCalibration()
        # Call only after an explicit user request, with fresh optical targets.
        alignment.calibrate(frame, quest_targets_body, robot_arm_lengths=arm_lengths)
        print(alignment.map(frame))
    """

    def __init__(self) -> None:
        self.sender_session: str | None = None
        self.scales: dict[str, float] = {}
        self.offsets: dict[str, Vec3] = {}
        self.calibrated_at_ns: int | None = None

    @property
    def is_calibrated(self) -> bool:
        """Return whether an explicit calibration has supplied both arm mappings."""
        return self.sender_session is not None and all(side in self.offsets for side in SIDES)

    def clear(self) -> None:
        """Invalidate alignment when the user, camera setup, or sender changes."""
        self.sender_session = None
        self.scales.clear()
        self.offsets.clear()
        self.calibrated_at_ns = None

    def calibrate(
        self,
        frame: WebcamFrame,
        quest_targets_body: dict[str, Vec3],
        *,
        robot_arm_lengths: dict[str, float] | None = None,
        now_ns: int | None = None,
    ) -> None:
        """Pair a fresh body sample with simultaneous, valid Quest palm targets.

        Args:
            frame: Fresh webcam frame with both arms confidently visible.
            quest_targets_body: Current optical Quest palm targets for both arms,
                before command smoothing, expressed in the robot body frame.
            robot_arm_lengths: Shoulder-to-wrist chain length for each robot arm.
                If omitted, use an approximate 37 cm G1 shoulder-to-wrist span.
            now_ns: Local monotonic clock value for frame-age validation.

        Raises:
            ValueError: If either source is incomplete or the alignment is implausible.
        """
        now = time.perf_counter_ns() if now_ns is None else now_ns
        if not 0 <= frame.age_s(now) <= MAX_CALIBRATION_AGE_S:
            raise ValueError("Calibration requires a fresh webcam frame")
        body = derive_body_pose(frame)
        if body is None or any(side not in body.arms for side in SIDES):
            raise ValueError("Calibration requires visible shoulders, hips, elbows, and both wrists")
        if any(side not in quest_targets_body for side in SIDES):
            raise ValueError("Calibration requires both current optical Quest palm targets")
        lengths = robot_arm_lengths if robot_arm_lengths is not None else {side: 0.37 for side in SIDES}
        scales, offsets = {}, {}
        for side in SIDES:
            target = _vec3(quest_targets_body[side], bound=2.0)
            span = _number(lengths.get(side), minimum=0.2, maximum=1.0)
            scale = span / body.arms[side].length
            if not 0.4 <= scale <= 1.8:
                raise ValueError("Estimated human/robot arm scale is implausible")
            offset = tuple(target[axis] - body.arms[side].wrist[axis] * scale for axis in range(3))
            scales[side] = scale
            offsets[side] = offset
        # Commit only after both arms validate, preserving a prior valid mapping
        # when an attempted recalibration has incomplete input.
        self.sender_session = frame.sender_session
        self.scales = scales
        self.offsets = offsets
        self.calibrated_at_ns = now

    def map(self, frame: WebcamFrame, *, now_ns: int | None = None) -> dict[str, WebcamArmTarget]:
        """Map fresh, confident arms or return no targets when alignment is absent.

        A changed sender invalidates prior calibration permanently, so delayed
        packets cannot revive an alignment after a camera process restart.
        The caller retains primary Quest input and bounds all resulting robot
        commands with the existing IK, joint-speed, and contact controllers.
        """
        if self.sender_session is not None and frame.sender_session != self.sender_session:
            self.clear()
        if not self.is_calibrated:
            return {}
        now = time.perf_counter_ns() if now_ns is None else now_ns
        if not 0 <= (now - frame.capture_monotonic_ns) / 1e9 <= MAX_FRAME_AGE_S:
            return {}
        body = derive_body_pose(frame)
        if body is None:
            return {}
        result = {}
        for side, arm in body.arms.items():
            scale, offset = self.scales[side], self.offsets[side]
            wrist = tuple(offset[axis] + scale * arm.wrist[axis] for axis in range(3))
            elbow = tuple(offset[axis] + scale * arm.elbow[axis] for axis in range(3))
            result[side] = WebcamArmTarget(wrist, elbow, arm.confidence)
        return result
