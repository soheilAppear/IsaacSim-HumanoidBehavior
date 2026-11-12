# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receive Quest Pro face weights from Steam Link's local, custom OSC output.

The documented Steam Link protocol uses port 9015 and ``/sl/xrfb/facew/``
expression names, plus ``/sl/xrfb/facec/`` regional confidences. The schema is
documented by its example implementation at https://github.com/danwillm/VRCFT-SteamLink.
This module consumes that protocol directly, without VRCFaceTracking or VRChat.

Steam Link's documented OSC fields do not include OpenXR's sensor capture time
or ``isValid`` status. Validity here means all required fields arrived recently
with adequate regional confidence. It cannot detect a sender repeatedly sending
frozen sensor values. OSC bundle timetags are scheduling metadata, not verified
sensor timestamps, and are deliberately not presented as capture timestamps.
The controller must additionally require an enabled, healthy XR session.
"""

from __future__ import annotations

import math
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any

__all__ = ["EXPRESSION_NAMES", "FaceTrackingSample", "SteamLinkFaceInput", "decode_osc_packet"]

EXPRESSION_NAMES = (
    "EyesClosedL",
    "EyesClosedR",
    "JawDrop",
    "MouthLeft",
    "MouthRight",
    "LipPuckerL",
    "LipPuckerR",
)
_CONFIDENCE_NAMES = ("UpperFace", "LowerFace")
_ADDRESSES = {f"/sl/xrfb/facew/{name}": name for name in EXPRESSION_NAMES}
_ADDRESSES.update({f"/sl/xrfb/facec/{name}": name for name in _CONFIDENCE_NAMES})
_REQUIRED_NAMES = EXPRESSION_NAMES + _CONFIDENCE_NAMES
_MAX_DATAGRAM_BYTES = 65507
_MAX_MESSAGES = 256
_MAX_BUNDLE_DEPTH = 4


def _read_string(payload: bytes, offset: int) -> tuple[str, int]:
    """Read a bounded, zero-padded OSC string."""
    end = payload.find(b"\0", offset)
    if end < 0:
        raise ValueError("Unterminated OSC string")
    next_offset = (end + 4) & ~3
    if next_offset > len(payload) or any(payload[end:next_offset]):
        raise ValueError("Invalid OSC string padding")
    try:
        return payload[offset:end].decode("utf-8"), next_offset
    except UnicodeDecodeError as error:
        raise ValueError("Invalid OSC string encoding") from error


def decode_osc_packet(payload: bytes) -> list[tuple[str, tuple[Any, ...]]]:
    """Decode bounded OSC messages or nested bundles without evaluating input.

    Args:
        payload: One UDP datagram using standard OSC 1.0 encoding.

    Returns:
        Address and argument pairs. Unsupported OSC types reject the packet.

    Raises:
        ValueError: If lengths, strings, types, or bundle limits are invalid.
    """
    if not isinstance(payload, bytes) or not 0 < len(payload) <= _MAX_DATAGRAM_BYTES:
        raise ValueError("OSC datagram is empty or oversized")
    messages: list[tuple[str, tuple[Any, ...]]] = []

    def parse(data: bytes, depth: int) -> None:
        """Parse one element after checking its enclosing length."""
        if depth > _MAX_BUNDLE_DEPTH:
            raise ValueError("OSC bundle nesting exceeds limit")
        if data.startswith(b"#bundle\0"):
            if len(data) < 16:
                raise ValueError("Truncated OSC bundle timetag")
            offset = 16
            while offset < len(data):
                if offset + 4 > len(data):
                    raise ValueError("Truncated OSC bundle element size")
                size = struct.unpack_from(">i", data, offset)[0]
                offset += 4
                if size <= 0 or size % 4 or offset + size > len(data):
                    raise ValueError("Invalid OSC bundle element size")
                parse(data[offset : offset + size], depth + 1)
                offset += size
            return
        if len(messages) >= _MAX_MESSAGES:
            raise ValueError("OSC message count exceeds limit")
        address, offset = _read_string(data, 0)
        if not address.startswith("/"):
            raise ValueError("Invalid OSC address")
        tags, offset = _read_string(data, offset)
        if not tags.startswith(","):
            raise ValueError("OSC message lacks type tags")
        args: list[Any] = []
        for tag in tags[1:]:
            if tag in "ifdh":
                size = 4 if tag in "if" else 8
                if offset + size > len(data):
                    raise ValueError("Truncated OSC numeric argument")
                args.append(struct.unpack_from(">" + ("q" if tag == "h" else tag), data, offset)[0])
                offset += size
            elif tag in "TFNI":
                args.append({"T": True, "F": False, "N": None, "I": math.inf}[tag])
            elif tag == "s":
                value, offset = _read_string(data, offset)
                args.append(value)
            elif tag == "b":
                if offset + 4 > len(data):
                    raise ValueError("Truncated OSC blob size")
                size = struct.unpack_from(">i", data, offset)[0]
                offset += 4
                end = offset + size
                padded_end = (end + 3) & ~3
                if size < 0 or padded_end > len(data) or any(data[end:padded_end]):
                    raise ValueError("Invalid OSC blob length or padding")
                args.append(data[offset:end])
                offset = padded_end
            else:
                raise ValueError("Unsupported OSC argument type")
        if offset != len(data):
            raise ValueError("OSC message contains trailing bytes")
        messages.append((address, tuple(args)))

    parse(payload, 0)
    return messages


@dataclass(frozen=True)
class FaceTrackingSample:
    """A conservative snapshot of independently refreshed facial weights."""

    weights: dict[str, float]
    #: Meta expression names with weights in the unit interval.
    valid: bool
    #: Whether all required fields are fresh and both confidence regions qualify.
    reason: str
    #: Human-readable status suitable for a preview panel.
    received_at: float
    #: Oldest required field's local monotonic receipt time, never sensor time.
    age_s: float
    #: Age of the oldest required field at snapshot creation.
    confidence_upper: float
    #: Upper-face tracking confidence supplied by Steam Link.
    confidence_lower: float
    #: Lower-face tracking confidence supplied by Steam Link.
    sequence: int
    #: Complete refresh cycles of every required field, including confidences.


class SteamLinkFaceInput:
    """Read an exclusively bound loopback UDP socket without blocking Kit.

    Args:
        port: Steam Link custom OSC output port. Zero chooses a free test port.
        stale_after_s: Maximum local receipt age for every required field.
        min_confidence: Minimum confidence for both face regions.
        max_packets_per_poll: Maximum datagrams drained during one callback.

    Raises:
        ValueError: If configuration values are outside their valid ranges.
    """

    def __init__(
        self,
        port: int = 9015,
        *,
        stale_after_s: float = 0.25,
        min_confidence: float = 0.5,
        max_packets_per_poll: int = 128,
    ) -> None:
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("Face OSC port must be an integer from 0 to 65535")
        if not math.isfinite(stale_after_s) or not 0.01 <= stale_after_s <= 2.0:
            raise ValueError("Face timeout must be between 0.01 and 2 seconds")
        if not math.isfinite(min_confidence) or not 0.0 < min_confidence <= 1.0:
            raise ValueError("Face confidence must be greater than zero and at most one")
        if type(max_packets_per_poll) is not int or not 1 <= max_packets_per_poll <= 1024:
            raise ValueError("Face packet limit must be an integer from 1 to 1024")
        self.port = port
        self.stale_after_s = float(stale_after_s)
        self.min_confidence = float(min_confidence)
        self.max_packets_per_poll = max_packets_per_poll
        self._socket: socket.socket | None = None
        self._values: dict[str, tuple[float, float]] = {}
        self._pending_fields: set[str] = set()
        self._last_poll_s: float | None = None
        self._drain_only = False
        self._sequence = 0
        self._packet_count = 0
        self._accepted_packet_count = 0
        self._malformed_count = 0
        self._ignored_count = 0
        self._backlog_count = 0
        self._reason = "Face OSC receiver has not started"

    def start(self) -> None:
        """Bind localhost exclusively; raise ``OSError`` for an occupied port."""
        if self._socket is not None:
            return
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                receiver.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            receiver.bind(("127.0.0.1", self.port))
            receiver.setblocking(False)
        except OSError:
            receiver.close()
            raise
        self._socket = receiver
        self.port = receiver.getsockname()[1]
        self._values.clear()
        self._pending_fields.clear()
        self._last_poll_s = time.monotonic()
        self._reason = "Waiting for Steam Link face OSC"

    def close(self) -> None:
        """Release the receiver and forget cached expressions."""
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        self._values.clear()
        self._pending_fields.clear()
        self._last_poll_s = None
        self._drain_only = False
        self._reason = "Face OSC receiver is closed"

    def ingest_packet(self, payload: bytes, *, now_s: float | None = None) -> bool:
        """Validate and merge one datagram, also usable for deterministic tests.

        A malformed datagram clears the cached frame instead of silently keeping
        its previous command alive. Unrelated OSC addresses cannot refresh it.
        """
        now = time.monotonic() if now_s is None else float(now_s)
        if not math.isfinite(now):
            raise ValueError("Face receipt time must be finite")
        self._packet_count += 1
        try:
            updates = {}
            for address, args in decode_osc_packet(payload):
                name = _ADDRESSES.get(address)
                if name is None:
                    continue
                if (
                    len(args) != 1
                    or isinstance(args[0], bool)
                    or not isinstance(args[0], (float, int))
                    or not math.isfinite(args[0])
                    or not 0.0 <= args[0] <= 1.0
                ):
                    raise ValueError("Face OSC weights and confidence require one finite value from zero to one")
                updates[name] = (float(args[0]), now)
        except ValueError:
            self._malformed_count += 1
            self._values.clear()
            self._pending_fields.clear()
            self._reason = "Malformed face OSC packet"
            return False
        if not updates:
            self._ignored_count += 1
            return False
        self._values.update(updates)
        self._accepted_packet_count += 1
        self._pending_fields.update(updates)
        # Steam Link splits a face over several OSC bundles. One old high
        # expression must not accumulate confirmation while unrelated channels
        # keep arriving. Only a refresh of the entire required set is a new
        # recognition sample; per-field expiry remains independently enforced.
        if self._pending_fields.issuperset(_REQUIRED_NAMES):
            self._sequence += 1
            self._pending_fields.clear()
        self._reason = "Waiting for all face fields"
        return True

    def snapshot(self, *, now_s: float | None = None) -> FaceTrackingSample:
        """Return validity based on each field's independent receipt and confidence."""
        now = time.monotonic() if now_s is None else float(now_s)
        if not math.isfinite(now):
            raise ValueError("Face snapshot time must be finite")
        weights = {name: self._values[name][0] for name in EXPRESSION_NAMES if name in self._values}
        upper = self._values.get("UpperFace", (0.0, now))[0]
        lower = self._values.get("LowerFace", (0.0, now))[0]
        oldest = min((value[1] for value in self._values.values()), default=now)
        age = max(0.0, now - oldest)
        reason = self._reason
        valid = False
        if all(name in self._values for name in _REQUIRED_NAMES):
            ages = [now - self._values[name][1] for name in _REQUIRED_NAMES]
            if min(ages) < -0.001 or max(ages) > self.stale_after_s:
                reason = "Face tracking is stale"
            elif min(upper, lower) < self.min_confidence:
                reason = "Face tracking confidence is low"
            else:
                reason = "Face tracking ready"
                valid = True
        return FaceTrackingSample(weights, valid, reason, oldest, age, upper, lower, self._sequence)

    def poll(self, *, now_s: float | None = None) -> FaceTrackingSample:
        """Drain a bounded packet batch and return the most recent complete sample.

        After a callback stall or UDP backlog, discard queued data until the
        socket is empty. This prevents old queued gestures being treated as new.
        Fresh datagrams from a subsequent callback are required before rearming.
        """
        now = time.monotonic() if now_s is None else float(now_s)
        if not math.isfinite(now):
            raise ValueError("Face polling time must be finite")
        if self._socket is None:
            return self.snapshot(now_s=now)
        if self._last_poll_s is not None and not 0 <= now - self._last_poll_s <= self.stale_after_s:
            self._drain_only = True
            self._values.clear()
            self._pending_fields.clear()
            self._reason = "Discarding face data after callback interruption"
        self._last_poll_s = now
        discard_this_poll = self._drain_only
        for _ in range(self.max_packets_per_poll):
            try:
                payload, peer = self._socket.recvfrom(_MAX_DATAGRAM_BYTES + 1)
            except BlockingIOError:
                self._drain_only = False
                break
            except OSError as error:
                self.close()
                self._reason = f"Face OSC socket error ({error.errno})"
                break
            if peer[0] != "127.0.0.1":
                continue
            if not discard_this_poll:
                self.ingest_packet(payload, now_s=now)
        else:
            # Even exactly a full batch is conservatively invalidated. The
            # default budget covers short Steam Link bursts inside the accepted
            # callback window; longer interruptions still discard before parsing.
            self._drain_only = True
            self._backlog_count += 1
            self._values.clear()
            self._pending_fields.clear()
            self._reason = "Discarding face OSC backlog"
        return self.snapshot(now_s=now)

    def diagnostics(self) -> dict[str, Any]:
        """Return bounded counters and protocol limitations without raw captures."""
        return {
            "host": "127.0.0.1",
            "port": self.port,
            "started": self._socket is not None,
            "packets": self._packet_count,
            "accepted_packets": self._accepted_packet_count,
            "complete_samples": self._sequence,
            "malformed_packets": self._malformed_count,
            "unrelated_packets": self._ignored_count,
            "backlog_events": self._backlog_count,
            "received_fields": sorted(self._values),
            "sensor_capture_timestamp_available": False,
            "freshness_basis": "local monotonic receipt of every required field",
        }
