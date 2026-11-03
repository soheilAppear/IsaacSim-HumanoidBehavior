# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect four-channel EEG and optionally preview experimental SSVEP matches.

Input order is O1, O2, T3, T4 and input units are volts. Signal-quality checks use
all four channels; frequency-template scores use only O1/O2. Scores describe a
spectral fit, not a probability, calibrated intent, or hardware validation.
Configured targets require an independently timed external SSVEP stimulus. This
module provides no stimulus, network transport, robot commands, or actuation.
"""

from __future__ import annotations

import math
from collections import deque
from numbers import Integral, Real

import numpy as np


class PreviewEngine:
    """Preview signal quality and explicitly configured frequency templates.

    With no targets, commands always remain ``none``. A configured result must
    win on a full window and keep winning for at least 0.6 seconds of both new
    samples and elapsed receiver time. Repeated polls cannot establish dwell.
    Transport loss, invalid input, poor quality, or an ambiguous winner clears
    qualification. These conservative heuristics are not a medical EEG quality
    assessment or an independently validated brain-intent classifier.

    Args:
        sample_rate: Positive nominal samples per second, at most 2000.
        targets: Optional mapping of forward/left/right/stop to stimulus Hz.
            At least two targets are required when configured. Fundamental and
            second-harmonic templates must be separated by one window bin.
        window_seconds: Analysis duration, from 1 through 10 seconds.
        packet_modulus: Sequence-counter wrap value, from 2 through 2**32.
            Use 2048 for original BrainBit and 2**32 for BrainBit2.

    Raises:
        ValueError: If sampling or target configuration is malformed or ambiguous.
    """

    CHANNELS = ("O1", "O2", "T3", "T4")
    MAX_AGE_SECONDS = 0.5
    DWELL_SECONDS = 0.6
    MIN_SCORE = 0.55
    MIN_MARGIN = 0.15
    MIN_WINNER_RATIO = 1.5
    ALLOWED_COMMANDS = frozenset(("forward", "left", "right", "stop"))

    def __init__(
        self,
        *,
        sample_rate: float = 250.0,
        targets: dict[str, float] | None = None,
        window_seconds: float = 2.0,
        packet_modulus: int = 2048,
    ) -> None:
        if not self._finite(sample_rate) or not 0 < sample_rate <= 2000:
            raise ValueError("Sample rate must be finite, positive, and at most 2000 Hz")
        if not self._finite(window_seconds) or not 1 <= window_seconds <= 10:
            raise ValueError("Window duration must be finite and between 1 and 10 seconds")
        if (
            not isinstance(packet_modulus, Integral)
            or isinstance(packet_modulus, (bool, np.bool_))
            or not 2 <= packet_modulus <= 2**32
        ):
            raise ValueError("Packet modulus must be an integer between 2 and 2**32")
        self.sample_rate = float(sample_rate)
        self.window_seconds = float(window_seconds)
        self.packet_modulus = int(packet_modulus)
        self.window_samples = int(math.ceil(self.sample_rate * self.window_seconds))
        if self.window_samples < 16:
            raise ValueError("The analysis window must contain at least 16 samples")
        self.targets = self._validate_targets(targets)
        self._templates = self._build_templates()
        self._samples: deque[np.ndarray] = deque(maxlen=self.window_samples)
        self._last_packet: int | None = None
        self._last_received: float | None = None
        self._last_evaluated_time: float | None = None
        self._revision = 0
        self._evaluated_revision = -1
        self._sample_total = 0
        self._candidate: str | None = None
        self._candidate_since: float | None = None
        self._candidate_sample_start = 0
        self._fault_quality: str | None = None
        self._fault_reason = ""
        self._cached_result: dict | None = None

    @staticmethod
    def _finite(value: object) -> bool:
        return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)

    def _validate_targets(self, targets: dict[str, float] | None) -> dict[str, float]:
        if targets is None:
            return {}
        if not isinstance(targets, dict) or not 2 <= len(targets) <= 4:
            raise ValueError("Configure two to four explicit command/frequency targets, or omit targets")
        result = {}
        for name, frequency in targets.items():
            if name not in self.ALLOWED_COMMANDS:
                raise ValueError("Target commands must be forward, left, right, or stop")
            if not self._finite(frequency) or not 6 <= frequency <= 40 or frequency >= self.sample_rate / 2:
                raise ValueError("Target frequencies must be finite, between 6 and 40 Hz, and below Nyquist")
            result[name] = float(frequency)
        resolution = self.sample_rate / self.window_samples
        names = list(result)
        for index, name in enumerate(names):
            for other in names[index + 1 :]:
                for first in self._harmonics(result[name]):
                    for second in self._harmonics(result[other]):
                        if abs(first - second) < resolution - 1e-9:
                            raise ValueError(
                                "Target fundamentals/harmonics overlap or are not resolvable in this window"
                            )
        return result

    def _harmonics(self, frequency: float) -> list[float]:
        return [frequency * harmonic for harmonic in (1, 2) if frequency * harmonic < self.sample_rate / 2]

    def _build_templates(self) -> dict[str, np.ndarray]:
        time_axis = np.arange(self.window_samples, dtype=float) / self.sample_rate
        self._trend_basis = np.linalg.qr(np.column_stack((np.ones(self.window_samples), time_axis)))[0]
        templates = {}
        for name, frequency in self.targets.items():
            columns = []
            for harmonic in self._harmonics(frequency):
                phase = 2 * np.pi * harmonic * time_axis
                columns.extend((np.sin(phase), np.cos(phase)))
            basis = np.column_stack(columns)
            basis -= self._trend_basis @ (self._trend_basis.T @ basis)
            templates[name] = np.linalg.qr(basis)[0]
        return templates

    def _clear_candidate(self) -> None:
        self._candidate = None
        self._candidate_since = None
        self._candidate_sample_start = self._sample_total

    def _invalidate(self, quality: str, reason: str) -> None:
        self._samples.clear()
        self._last_packet = None
        self._last_received = None
        self._last_evaluated_time = None
        self._sample_total = 0
        self._revision += 1
        self._cached_result = None
        self._fault_quality = quality
        self._fault_reason = reason
        self._clear_candidate()

    def add_samples(
        self,
        samples_v: list[list[float]],
        packet_numbers: list[int],
        received_monotonic: float,
    ) -> None:
        """Accept a contiguous batch of O1/O2/T3/T4 samples in volts.

        Invalid batches are discarded and clear all history/qualification. Packet
        numbers increment once per sample modulo ``packet_modulus``. Empty batches
        never refresh freshness. Receiver timestamps must use the same monotonic clock as
        ``evaluate``; they measure arrival age, not acquisition or inference age.

        Args:
            samples_v: Rows containing exactly four finite channel values.
            packet_numbers: Matching integer sequence numbers in [0, packet_modulus).
            received_monotonic: Local monotonic receipt time in seconds.
        """
        if not self._finite(received_monotonic) or received_monotonic < 0:
            self._invalidate("gap", "Invalid receiver clock; waiting for a fresh contiguous window")
            return
        try:
            samples = np.asarray(samples_v)
            if samples.dtype.kind not in "iuf":
                raise ValueError("Samples must be real numeric values")
            samples = samples.astype(float, copy=False)
            packets = list(packet_numbers)
        except (TypeError, ValueError, OverflowError):
            self._invalidate("artifact", "Malformed sample batch; expected four numeric channels")
            return
        if samples.size == 0 and samples.shape in ((0,), (0, 4)) and not packets:
            return
        if samples.ndim != 2 or samples.shape[1] != 4 or len(samples) != len(packets) or not len(samples):
            self._invalidate("artifact", "Batch must contain matching four-channel samples and packet numbers")
            return
        if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 500e-6:
            self._invalidate("artifact", "Nonfinite or gross-amplitude signal; waiting for clean samples")
            return
        if any(
            not isinstance(packet, Integral)
            or isinstance(packet, (bool, np.bool_))
            or not 0 <= packet < self.packet_modulus
            for packet in packets
        ):
            self._invalidate("gap", f"Invalid packet counter; expected integers modulo {self.packet_modulus}")
            return
        if self._last_received is not None and (
            received_monotonic < self._last_received or received_monotonic - self._last_received > self.MAX_AGE_SECONDS
        ):
            self._invalidate("gap", "Input interrupted or receiver clock reversed; a new full window is required")
            return
        previous = self._last_packet
        for packet in packets:
            if previous is not None and packet != (previous + 1) % self.packet_modulus:
                self._invalidate("gap", "Missing, duplicated, or out-of-order packet; a new full window is required")
                return
            previous = int(packet)
        self._samples.extend(row.copy() for row in samples)
        self._last_packet = previous
        self._last_received = float(received_monotonic)
        self._sample_total += len(samples)
        self._revision += 1
        self._fault_quality = None
        self._fault_reason = ""

    def _result(
        self,
        quality: str,
        reason: str,
        *,
        rms: list[float] | None = None,
        candidate: str = "none",
        command: str = "none",
        score: float = 0.0,
    ) -> dict:
        return {
            "command": command,
            "candidate": candidate,
            "reason": reason,
            "quality": quality,
            "score": float(score),
            "channel_rms_uv": list(rms) if rms is not None else [],
            "sample_count": len(self._samples),
        }

    def evaluate(self, now: float) -> dict:
        """Return quality and an optional qualified shadow-preview command.

        RMS is AC RMS in microvolts. Score is mean explained spectral energy in
        O1/O2, from 0 to 1; it is not an intent probability. Artifact thresholds
        are fixed conservative heuristics requiring hardware/session validation.

        Args:
            now: Current time on the receiver's monotonic clock, in seconds.

        Returns:
            Dictionary with command, candidate, reason, quality, score,
            channel_rms_uv, and the current analysis-window sample_count.
        """
        if not self._finite(now) or now < 0:
            self._invalidate("stale", "Invalid evaluation clock; preview disarmed")
        elif self._last_evaluated_time is not None and now < self._last_evaluated_time:
            self._invalidate("stale", "Evaluation clock reversed; preview disarmed")
        elif self._last_received is not None and not 0 <= now - self._last_received <= self.MAX_AGE_SECONDS:
            self._invalidate("stale", "No fresh EEG within 0.5 seconds; preview disarmed")
        if self._fault_quality is not None:
            return self._result(self._fault_quality, self._fault_reason)
        self._last_evaluated_time = float(now)
        if self._cached_result is not None and self._evaluated_revision == self._revision:
            return {**self._cached_result, "channel_rms_uv": list(self._cached_result["channel_rms_uv"])}
        if not self._samples:
            return self._result("waiting", "Waiting for four-channel EEG samples")
        values = np.asarray(self._samples) * 1e6
        centered = values - np.mean(values, axis=0)
        rms_values = np.sqrt(np.mean(centered**2, axis=0))
        rms = rms_values.tolist()
        if (
            np.any(rms_values > 100)
            or np.any(np.ptp(values, axis=0) > 400)
            or (len(values) > 1 and np.any(np.abs(np.diff(values, axis=0)) > 100))
        ):
            self._invalidate("artifact", "Gross amplitude or abrupt signal movement; preview disarmed")
            return self._result("artifact", self._fault_reason, rms=rms)
        if len(values) < self.window_samples:
            return self._result("waiting", "Collecting a full contiguous analysis window", rms=rms)
        if np.any(rms_values < 0.5):
            self._invalidate("artifact", "Flat or disconnected channel; preview disarmed")
            return self._result("artifact", self._fault_reason, rms=rms)
        if not self.targets:
            result = self._result("good", "Unconfigured: signal preview only; no frequency targets", rms=rms)
        else:
            occipital = centered[:, :2]
            detrended = occipital - self._trend_basis @ (self._trend_basis.T @ occipital)
            power = np.sum(detrended**2, axis=0)
            if np.any(power < self.window_samples * 0.5**2):
                self._invalidate("artifact", "Occipital signal is flat after detrending; preview disarmed")
                return self._result("artifact", self._fault_reason, rms=rms)
            scores = {
                name: float(np.clip(np.mean(np.sum((basis.T @ detrended) ** 2, axis=0) / power), 0.0, 1.0))
                for name, basis in self._templates.items()
            }
            ranked = sorted(scores, key=scores.get, reverse=True)
            winner, runner_up = ranked[:2]
            score, second = scores[winner], scores[runner_up]
            if score < self.MIN_SCORE or score - second < self.MIN_MARGIN or score < second * self.MIN_WINNER_RATIO:
                self._clear_candidate()
                result = self._result("good", "No clear frequency-template winner", rms=rms, score=score)
            else:
                if winner != self._candidate:
                    self._candidate = winner
                    self._candidate_since = self._last_received
                    self._candidate_sample_start = self._sample_total
                qualified = (
                    self._last_received - self._candidate_since >= self.DWELL_SECONDS - 1e-9
                    and self._sample_total - self._candidate_sample_start
                    >= math.ceil(self.DWELL_SECONDS * self.sample_rate)
                )
                result = self._result(
                    "good",
                    (
                        "Qualified experimental spectral match; shadow preview only"
                        if qualified
                        else "Frequency candidate needs more consecutive fresh evidence"
                    ),
                    rms=rms,
                    candidate=winner,
                    command=winner if qualified else "none",
                    score=score,
                )
        self._evaluated_revision = self._revision
        self._cached_result = result
        return {**result, "channel_rms_uv": list(result["channel_rms_uv"])}
