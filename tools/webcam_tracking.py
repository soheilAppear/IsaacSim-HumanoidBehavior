# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run pretrained webcam body tracking in a process separate from Isaac Sim.

The recording launcher uses ``--preview-only``: landmarks are drawn locally,
without a UDP socket, robot input, or calibration requests. Direct Python use
without this flag retains the experimental localhost landmark sender. Camera
images are never recorded by this tool.
Inference uses MediaPipe Tasks VIDEO mode on the newest available camera frame.
The capture thread overwrites its one-frame slot, so slow inference cannot build a
queue of old poses. Optional detailed hand inference adds CPU cost.

Protocol v1 uses meters for estimated world landmarks, normalized image landmarks,
and the same-computer ``time.perf_counter_ns()`` clock (Windows QPC). Python
versions before 3.13 can give ``monotonic_ns()`` a different epoch, so using it
across the camera and Isaac interpreters would reject otherwise fresh frames.
The wire field names retain ``monotonic_ns`` because QPC is also monotonic.
Pose world coordinates are hip-centered, NOT
room coordinates; the Isaac receiver must calibrate before moving robot arms.
Camera timestamps mark frame availability after ``read()``, not sensor exposure.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import threading
import time
import urllib.request
import uuid
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / ".webcam" / "models"
MAX_DATAGRAM_BYTES = 32768
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}
POSE_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
)
HAND_CONNECTIONS = tuple(
    edge
    for chain in (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (5, 9, 10, 11, 12),
        (9, 13, 14, 15, 16),
        (13, 17, 18, 19, 20),
        (0, 17),
    )
    for edge in zip(chain, chain[1:])
)
LOG = logging.getLogger("webcam_tracking")


def _serialize_landmarks(landmarks: Any, count: int, confidence: bool = False) -> list[list[float]] | None:
    """Convert one complete finite MediaPipe landmark list without changing axes."""
    if landmarks is None or len(landmarks) != count:
        return None
    result = []
    for landmark in landmarks:
        row = [float(landmark.x), float(landmark.y), float(landmark.z)]
        if confidence:
            # A missing confidence is unknown, not a confident detection.
            row.extend(float(getattr(landmark, name, 0.0) or 0.0) for name in ("visibility", "presence"))
        if not all(math.isfinite(value) for value in row):
            return None
        result.append(row)
    return result


def _associate_hands(pose_image: list | None, hand_result: Any) -> tuple[dict, dict, dict]:
    """Assign hands to anatomical pose wrists; omit ambiguous or distant matches.

    Model handedness conventions can differ for mirrored inputs. Matching image
    wrist locations to pose wrists avoids silently swapping robot hands. Detailed
    hand world coordinates remain local to each hand and are used only for curls.
    """
    world_by_side, image_by_side, confidence_by_side = {}, {}, {}
    if not pose_image or hand_result is None:
        return world_by_side, image_by_side, confidence_by_side
    candidates = []
    for index, image_landmarks in enumerate(hand_result.hand_landmarks):
        image = _serialize_landmarks(image_landmarks, 21)
        world = _serialize_landmarks(hand_result.hand_world_landmarks[index], 21)
        categories = hand_result.handedness[index]
        if image is None or world is None or not categories:
            continue
        score = float(categories[0].score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            continue
        distances = []
        for side, wrist_index in (("left", 15), ("right", 16)):
            wrist = pose_image[wrist_index]
            if min(wrist[3:5]) < 0.6:
                continue
            distance = math.hypot(image[0][0] - wrist[0], image[0][1] - wrist[1])
            distances.append((distance, side, min(score, wrist[3], wrist[4])))
        distances.sort()
        if not distances or distances[0][0] > 0.18:
            continue
        # Overlapping wrists are not a reliable identity observation.
        if len(distances) == 2 and distances[1][0] - distances[0][0] < 0.025:
            continue
        distance, side, confidence = distances[0]
        candidates.append((distance, side, confidence, world, image))
    for _, side, confidence, world, image in sorted(candidates, key=lambda item: item[0]):
        if side not in world_by_side:
            world_by_side[side], image_by_side[side], confidence_by_side[side] = world, image, confidence
    return world_by_side, image_by_side, confidence_by_side


def build_packet(
    pose_result: Any,
    hand_result: Any,
    *,
    sender_session: str,
    seq: int,
    capture_monotonic_ns: int,
    calibration_request: int = 0,
) -> dict[str, Any]:
    """Build a versioned packet, including explicit no-detection observations.

    Empty frames must be sent too, so the receiver cannot reuse an old visible
    pose when the person leaves the image. ``calibration_request`` is a persistent
    counter; the receiver detects changes even if an individual UDP packet drops.
    """
    pose_world = pose_image = None
    if pose_result is not None and pose_result.pose_world_landmarks and pose_result.pose_landmarks:
        pose_world = _serialize_landmarks(pose_result.pose_world_landmarks[0], 33, confidence=True)
        pose_image = _serialize_landmarks(pose_result.pose_landmarks[0], 33, confidence=True)
    hand_world, hand_image, hand_confidence = _associate_hands(pose_image, hand_result)
    return {
        "schema_version": 1,
        "sender_session": sender_session,
        "seq": seq,
        "capture_monotonic_ns": capture_monotonic_ns,
        "sent_monotonic_ns": time.perf_counter_ns(),
        "pose_world": pose_world,
        "pose_image": pose_image,
        "hand_world": hand_world,
        "hand_image": hand_image,
        "hand_confidence": hand_confidence,
        "calibration_request": calibration_request,
    }


def encode_packet(packet: dict[str, Any]) -> bytes:
    """Encode finite JSON below the receiver's single-datagram size limit."""
    payload = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_DATAGRAM_BYTES:
        raise ValueError("Webcam packet exceeds the 32 KiB limit")
    return payload


def prepare_models(model_dir: Path, pose_model: str, hands: bool, download: bool) -> dict[str, Path]:
    """Find cached Tasks bundles or explicitly download official Google models.

    Downloads use a bounded size, timeout, a temporary sibling, and ZIP validation
    before replacement. Existing valid bundles are reused without network access.
    """
    paths = {}
    for name in (pose_model, "hand") if hands else (pose_model,):
        path = model_dir / MODEL_URLS[name].rsplit("/", 1)[1]
        if not (path.is_file() and zipfile.is_zipfile(path)):
            if not download:
                raise FileNotFoundError(f"Missing model {path}. Run tools/webcam_setup.py or add --download-models.")
            model_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".download-{uuid.uuid4().hex}")
            LOG.info("Downloading official %s model to %s", name, path)
            try:
                total = 0
                with urllib.request.urlopen(MODEL_URLS[name], timeout=30) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > 64 * 1024 * 1024:
                            raise ValueError("Model download exceeds 64 MiB")
                        output.write(chunk)
                if not zipfile.is_zipfile(temporary):
                    raise ValueError("Downloaded model is not a MediaPipe Tasks ZIP bundle")
                with zipfile.ZipFile(temporary) as bundle:
                    if not bundle.namelist() or bundle.testzip() is not None:
                        raise ValueError("Downloaded model bundle is corrupt")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        paths[name] = path
    return paths


class LatestFrameCapture:
    """Read a camera continuously and retain only its newest frame.

    Args:
        capture: An opened OpenCV VideoCapture, exclusively owned by this instance.
    """

    def __init__(self, capture: Any) -> None:
        self._capture = capture
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._sequence = 0
        self._error = None
        self._thread = threading.Thread(target=self._read_loop, name="webcam-capture", daemon=True)
        self._thread.start()

    def next_frame(self, after: int, timeout: float = 0.1) -> tuple | None:
        """Return the latest frame newer than ``after``, never an inference queue."""
        with self._condition:
            self._condition.wait_for(lambda: self._sequence > after or self._error is not None, timeout)
            if self._error is not None:
                raise RuntimeError(self._error)
            return self._latest if self._sequence > after else None

    def close(self) -> None:
        """Stop capture and release the camera on its owning thread."""
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            LOG.warning("Camera driver is still blocked in read; it will close when this process exits")

    def _read_loop(self) -> None:
        """Replace the single frame slot and expose camera failures to the owner."""
        failed_since = None
        try:
            while not self._stop.is_set():
                ok, frame = self._capture.read()
                timestamp = time.perf_counter_ns()
                if not ok or frame is None:
                    failed_since = failed_since or time.monotonic()
                    if time.monotonic() - failed_since > 3.0:
                        raise RuntimeError("Camera stopped delivering frames; check its cable and privacy settings")
                    self._stop.wait(0.02)
                    continue
                failed_since = None
                with self._condition:
                    self._sequence += 1
                    self._latest = (self._sequence, timestamp, frame)
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = f"Camera capture failed: {error}"
                self._condition.notify_all()
        finally:
            self._capture.release()


def _draw_preview(
    cv2: Any, frame: Any, packet: dict, mirror: bool, elapsed_ms: float, *, preview_only: bool = False
) -> Any:
    """Draw landmarks on a display copy without mirroring tracking coordinates."""
    height, width = frame.shape[:2]
    preview = frame.copy()
    for landmarks, connections, color in (
        (packet["pose_image"], POSE_CONNECTIONS, (80, 230, 80)),
        *((hand, HAND_CONNECTIONS, (20, 200, 255)) for hand in packet["hand_image"].values()),
    ):
        if not landmarks:
            continue
        points = [(int(row[0] * width), int(row[1] * height)) for row in landmarks]
        visible = [len(row) == 3 or min(row[3:5]) >= 0.6 for row in landmarks]
        for first, second in connections:
            if visible[first] and visible[second]:
                cv2.line(preview, points[first], points[second], color, 2)
        for point, show in zip(points, visible):
            if show:
                cv2.circle(preview, point, 3, color, -1)
    if mirror:
        preview = cv2.flip(preview, 1)
    status = "Body visible" if packet["pose_world"] else "No body detected - step back into view"
    mode_lines = (
        ("DISPLAY ONLY - robot uses Quest/controllers", "Q/Esc: quit | Calibration disabled")
        if preview_only
        else (
            "Experimental UDP sender. C: request calibration | Q/Esc: quit",
            f"Calibration request: {packet['calibration_request']} (check Isaac status for result)",
        )
    )
    lines = (f"{status} | {elapsed_ms:.0f} ms | Hands: {len(packet['hand_world'])}", *mode_lines)
    for index, line in enumerate(lines):
        location = (12, 26 + index * 26)
        cv2.putText(preview, line, location, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(preview, line, location, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return preview


def create_parser() -> argparse.ArgumentParser:
    """Build the CLI without importing OpenCV or MediaPipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index, usually 0 or 1")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=24, help="Maximum inference/send rate; camera may differ")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto")
    parser.add_argument("--port", type=int, default=8765, help="Localhost UDP receiver port")
    parser.add_argument("--pose-model", choices=("lite", "full"), default="lite")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--hands", action="store_true", help="Also estimate 21 landmarks per hand; increases CPU use")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Display landmarks only; no robot UDP socket or calibration requests",
    )
    parser.add_argument("--headless", action="store_true", help="No preview; stop with Ctrl+C")
    parser.add_argument("--unmirrored-preview", action="store_true", help="Display camera view instead of a mirror")
    parser.add_argument("--download-models", action="store_true", help="Allow missing official model downloads")
    parser.add_argument("--download-only", action="store_true", help="Download models and exit without using a camera")
    parser.add_argument("--duration", type=float, default=0, help="Stop after this many seconds; 0 runs until quit")
    return parser


def run(args: argparse.Namespace) -> int:
    """Run local inference, optionally sending landmarks in experimental sender mode."""
    paths = prepare_models(args.model_dir, args.pose_model, args.hands, args.download_models or args.download_only)
    if args.download_only:
        LOG.info("Models ready: %s", ", ".join(str(path) for path in paths.values()))
        return 0
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as error:
        raise RuntimeError(
            "Webcam dependencies are missing. Run tools/webcam_setup.py, then the webcam launcher."
        ) from error

    # Do not consume all application CPU threads for color conversion/preview.
    cv2.setNumThreads(1)
    delegate = python.BaseOptions.Delegate.CPU
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(paths[args.pose_model]), delegate=delegate),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_segmentation_masks=False,
    )
    backend = {"auto": cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[
        args.backend
    ]
    with ExitStack() as cleanup:
        pose = cleanup.enter_context(vision.PoseLandmarker.create_from_options(pose_options))
        hand = None
        if args.hands:
            hand = cleanup.enter_context(
                vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(
                        base_options=python.BaseOptions(model_asset_path=str(paths["hand"]), delegate=delegate),
                        running_mode=vision.RunningMode.VIDEO,
                        num_hands=2,
                        min_hand_detection_confidence=0.6,
                        min_hand_presence_confidence=0.6,
                        min_tracking_confidence=0.6,
                    )
                )
            )
        capture = cv2.VideoCapture(args.camera, backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Cannot open webcam {args.camera}; try --camera 1 or --backend msmf and close other camera apps"
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frames = LatestFrameCapture(capture)
        cleanup.callback(frames.close)
        cleanup.callback(cv2.destroyAllWindows)
        sender = None
        if not args.preview_only:
            sender = cleanup.enter_context(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
            sender.setblocking(False)
        session = str(uuid.uuid4())
        seq = frame_sequence = calibration_request = 0
        last_timestamp_ms = -1
        started = last_frame_time = last_status = time.monotonic()
        next_inference = started
        if args.preview_only:
            LOG.info("DISPLAY ONLY: webcam %s, pose %s, detailed hands %s", args.camera, args.pose_model, args.hands)
            LOG.info("No robot control, UDP socket, calibration, or camera recording. Q/Esc exits.")
        else:
            LOG.info(
                "Experimental webcam sender %s -> 127.0.0.1:%s, pose %s, detailed hands %s",
                args.camera,
                args.port,
                args.pose_model,
                args.hands,
            )
            LOG.info("No camera recording. C requests calibration; Q/Esc exits.")
        while not args.duration or time.monotonic() - started < args.duration:
            snapshot = frames.next_frame(frame_sequence)
            if snapshot is None:
                if time.monotonic() - last_frame_time > 5.0:
                    raise RuntimeError("No camera frame arrived within five seconds")
                continue
            if time.monotonic() < next_inference:
                time.sleep(max(0.0, min(next_inference - time.monotonic(), 0.02)))
                continue
            frame_sequence, captured_ns, frame = snapshot
            last_frame_time = time.monotonic()
            next_inference = last_frame_time + 1.0 / args.fps
            timestamp_ms = max(last_timestamp_ms + 1, captured_ns // 1_000_000)
            last_timestamp_ms = timestamp_ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pose_result = pose.detect_for_video(image, timestamp_ms)
            hand_result = hand.detect_for_video(image, timestamp_ms) if hand else None
            packet = build_packet(
                pose_result,
                hand_result,
                sender_session=session,
                seq=seq,
                capture_monotonic_ns=captured_ns,
                calibration_request=calibration_request,
            )
            seq += 1
            if sender is not None:
                try:
                    sender.sendto(encode_packet(packet), ("127.0.0.1", args.port))
                except BlockingIOError:
                    # Losing one observation is preferable to delaying future poses.
                    pass
            latency_ms = (time.perf_counter_ns() - captured_ns) / 1_000_000
            if not args.headless:
                cv2.imshow(
                    "Humanoid webcam tracking",
                    _draw_preview(
                        cv2, frame, packet, not args.unmirrored_preview, latency_ms, preview_only=args.preview_only
                    ),
                )
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if cv2.getWindowProperty("Humanoid webcam tracking", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("c") and sender is not None:
                    calibration_request += 1
                    LOG.info(
                        "Calibration requested; hold both wrists visible to Quest and webcam and check Isaac status"
                    )
            if time.monotonic() - last_status >= 5.0:
                LOG.info(
                    "Body %s, hands %s, frame age %.0f ms, processed %s frames",
                    bool(packet["pose_world"]),
                    len(packet["hand_world"]),
                    latency_ms,
                    seq,
                )
                last_status = time.monotonic()
        LOG.info("Webcam tracking stopped after %s frames", seq)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Validate options and report actionable setup/camera errors without a traceback."""
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.preview_only and args.headless:
        parser.error("Preview-only mode needs a display; remove --headless")
    if args.camera < 0 or not 1 <= args.port <= 65535 or not 1 <= args.fps <= 120:
        parser.error("Camera must be nonnegative, port 1..65535, and FPS 1..120")
    if not 160 <= args.width <= 4096 or not 120 <= args.height <= 2160 or not 0 <= args.duration < math.inf:
        parser.error("Use width 160..4096, height 120..2160, and a finite nonnegative duration")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return run(args)
    except KeyboardInterrupt:
        LOG.info("Stopped by user")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        LOG.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
