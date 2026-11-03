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
from copy import deepcopy
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
    Full windows are linearly detrended and optionally projected away from a
    configured mains-frequency sine/cosine pair. Large fitted line components
    remain an explicit interference result even when the residual is small.
    Changing baseline trends retain the existing 100 microvolt RMS limit.
    Passing these prototype checks does not establish clean EEG or readiness
    for control.

    Args:
        sample_rate: Positive nominal samples per second, at most 2000.
        targets: Optional mapping of forward/left/right/stop to stimulus Hz.
            At least two targets are required when configured. Fundamental and
            second-harmonic templates must be separated by one window bin.
        window_seconds: Analysis duration, from 1 through 10 seconds.
        packet_modulus: Sequence-counter wrap value, from 2 through 2**32.
            Use 2048 for original BrainBit and 2**32 for BrainBit2.
        line_frequency_hz: Optional fitted mains frequency, exactly 50 or 60 Hz.
            None disables it. Frequencies at or above Nyquist are not represented;
            the resolved value is then None in the engine and its results.

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
        line_frequency_hz: float | None = 60.0,
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
        if line_frequency_hz is not None and (not self._finite(line_frequency_hz) or line_frequency_hz not in (50, 60)):
            raise ValueError("Line frequency must be exactly 50 Hz, 60 Hz, or None")
        self.line_frequency_hz = (
            float(line_frequency_hz)
            if line_frequency_hz is not None and line_frequency_hz < self.sample_rate / 2
            else None
        )
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
        self._fault_metrics: dict[str, list[float]] = {}
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
            if self._overlaps_line(float(frequency)):
                raise ValueError("Target fundamental overlaps the removed line-frequency bin")
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

    def _overlaps_line(self, frequency: float) -> bool:
        return (
            self.line_frequency_hz is not None
            and abs(frequency - self.line_frequency_hz) <= self.sample_rate / self.window_samples + 1e-9
        )

    def _harmonics(self, frequency: float) -> list[float]:
        return [
            frequency * harmonic
            for harmonic in (1, 2)
            if frequency * harmonic < self.sample_rate / 2 and not self._overlaps_line(frequency * harmonic)
        ]

    def _build_templates(self) -> dict[str, np.ndarray]:
        time_axis = np.arange(self.window_samples, dtype=float) / self.sample_rate
        self._trend_basis = np.linalg.qr(np.column_stack((np.ones(self.window_samples), time_axis)))[0]
        self._line_basis = np.empty((self.window_samples, 0))
        if self.line_frequency_hz is not None:
            phase = 2 * np.pi * self.line_frequency_hz * time_axis
            line_columns = np.column_stack((np.sin(phase), np.cos(phase)))
            # Orthogonalize against the trend so the combined projection is the
            # joint least-squares fit, rather than two interfering regressions.
            line_columns -= self._trend_basis @ (self._trend_basis.T @ line_columns)
            self._line_basis = np.linalg.qr(line_columns)[0]
        self._nuisance_basis = np.column_stack((self._trend_basis, self._line_basis))
        templates = {}
        for name, frequency in self.targets.items():
            columns = []
            for harmonic in self._harmonics(frequency):
                phase = 2 * np.pi * harmonic * time_axis
                columns.extend((np.sin(phase), np.cos(phase)))
            basis = np.column_stack(columns)
            basis -= self._nuisance_basis @ (self._nuisance_basis.T @ basis)
            templates[name] = np.linalg.qr(basis)[0]
        return templates

    def _clear_candidate(self) -> None:
        self._candidate = None
        self._candidate_since = None
        self._candidate_sample_start = self._sample_total

    def _invalidate(self, quality: str, reason: str, *, metrics: dict[str, list[float]] | None = None) -> None:
        self._samples.clear()
        self._last_packet = None
        self._last_received = None
        self._last_evaluated_time = None
        self._sample_total = 0
        self._revision += 1
        self._cached_result = None
        self._fault_quality = quality
        self._fault_reason = reason
        self._fault_metrics = deepcopy(metrics) if metrics is not None else {}
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
        Absolute values above 1 V are rejected as a broad numeric sanity check,
        not a measured hardware clipping limit. DC offsets alone are not artifacts.

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
        if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1.0:
            self._invalidate("artifact", "Nonfinite signal or value outside the broad 1 V numeric sanity bound")
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
        self._fault_metrics = {}

    def _result(
        self,
        quality: str,
        reason: str,
        *,
        metrics: dict[str, list[float]] | None = None,
        candidate: str = "none",
        command: str = "none",
        score: float = 0.0,
    ) -> dict:
        result = {
            "command": command,
            "candidate": candidate,
            "reason": reason,
            "quality": quality,
            "score": float(score),
            "sample_count": len(self._samples),
            "line_frequency_hz": self.line_frequency_hz,
        }
        for name in (
            "channel_rms_uv",
            "channel_offset_mv",
            "channel_trend_rms_uv",
            "channel_detrended_rms_uv",
            "channel_line_rms_uv",
            "channel_conditioned_rms_uv",
        ):
            result[name] = list(metrics.get(name, [])) if metrics is not None else []
        return result

    def _cache_result(self, result: dict) -> dict:
        self._evaluated_revision = self._revision
        self._cached_result = deepcopy(result)
        return deepcopy(result)

    @staticmethod
    def _has_identical_run(values: np.ndarray) -> bool:
        # Four successive equal differences mean at least five identical raw
        # samples on one channel. This catches plateaus without claiming an ADC rail.
        equal = np.diff(values, axis=0) == 0.0
        return bool(np.any(equal[:-3] & equal[1:-2] & equal[2:-1] & equal[3:]))

    def evaluate(self, now: float) -> dict:
        """Return quality and an optional qualified shadow-preview command.

        ``channel_rms_uv`` remains raw mean-removed AC RMS. Full-window diagnostics
        additionally expose mean-removed linear-trend RMS, linear-detrended RMS,
        fitted line-component RMS and conditioned residual RMS. The line metric is a sinusoidal fit, not Welch
        band power. Score is the mean fraction of conditioned O1/O2 energy
        explained by a target template, not an intent probability. All thresholds
        are prototype heuristics requiring hardware/session validation.

        Args:
            now: Current time on the receiver's monotonic clock, in seconds.

        Returns:
            Dictionary with command, candidate, reason, quality, score, channel
            RMS/offset diagnostics, resolved line_frequency_hz, and buffered
            sample_count. Conditioning diagnostics are empty before a full window.
        """
        if not self._finite(now) or now < 0:
            self._invalidate("stale", "Invalid evaluation clock; preview disarmed")
        elif self._last_evaluated_time is not None and now < self._last_evaluated_time:
            self._invalidate("stale", "Evaluation clock reversed; preview disarmed")
        elif self._last_received is not None and not 0 <= now - self._last_received <= self.MAX_AGE_SECONDS:
            self._invalidate("stale", "No fresh EEG within 0.5 seconds; preview disarmed")
        if self._fault_quality is not None:
            return self._result(self._fault_quality, self._fault_reason, metrics=self._fault_metrics)
        self._last_evaluated_time = float(now)
        if self._cached_result is not None and self._evaluated_revision == self._revision:
            return deepcopy(self._cached_result)
        if not self._samples:
            return self._result("waiting", "Waiting for four-channel EEG samples")
        values = np.asarray(self._samples) * 1e6
        mean = np.mean(values, axis=0)
        centered = values - mean
        metrics = {
            "channel_rms_uv": np.sqrt(np.mean(centered**2, axis=0)).tolist(),
            "channel_offset_mv": (mean / 1000.0).tolist(),
        }
        if len(values) < self.window_samples:
            return self._result("waiting", "Collecting a full contiguous analysis window", metrics=metrics)
        trend_component = self._trend_basis @ (self._trend_basis.T @ centered)
        trend_component -= np.mean(trend_component, axis=0)
        trend_rms = np.sqrt(np.mean(trend_component**2, axis=0))
        detrended = centered - trend_component
        line_component = self._line_basis @ (self._line_basis.T @ detrended)
        conditioned = detrended - line_component
        line_rms = np.sqrt(np.mean(line_component**2, axis=0))
        conditioned_rms = np.sqrt(np.mean(conditioned**2, axis=0))
        metrics.update(
            {
                "channel_trend_rms_uv": trend_rms.tolist(),
                "channel_detrended_rms_uv": np.sqrt(np.mean(detrended**2, axis=0)).tolist(),
                "channel_line_rms_uv": line_rms.tolist(),
                "channel_conditioned_rms_uv": conditioned_rms.tolist(),
            }
        )
        if np.any(trend_rms > 100):
            self._invalidate("artifact", "Changing baseline exceeds prototype limit", metrics=metrics)
            return self._result("artifact", self._fault_reason, metrics=self._fault_metrics)
        if self._has_identical_run(values):
            self._invalidate(
                "artifact", "Five or more identical raw samples on a channel; preview disarmed", metrics=metrics
            )
            return self._result("artifact", self._fault_reason, metrics=self._fault_metrics)
        if (
            np.any(conditioned_rms > 100)
            or np.any(np.ptp(conditioned, axis=0) > 400)
            or np.any(np.abs(np.diff(conditioned, axis=0)) > 100)
        ):
            self._invalidate(
                "artifact", "Gross conditioned amplitude or abrupt residual movement; preview disarmed", metrics=metrics
            )
            return self._result("artifact", self._fault_reason, metrics=self._fault_metrics)
        if np.any(line_rms > 100) or np.any(line_rms > 2 * conditioned_rms):
            self._clear_candidate()
            return self._cache_result(
                self._result(
                    "interference",
                    "Fitted mains-frequency component exceeds prototype limits; preview disarmed",
                    metrics=metrics,
                )
            )
        if np.any(conditioned_rms < 0.5):
            self._invalidate("artifact", "Flat conditioned channel; preview disarmed", metrics=metrics)
            return self._result("artifact", self._fault_reason, metrics=self._fault_metrics)
        if not self.targets:
            result = self._result(
                "good",
                "Unconfigured: prototype residual checks passed; signal preview only, not verified clean EEG",
                metrics=metrics,
            )
        else:
            occipital = conditioned[:, :2]
            power = np.sum(occipital**2, axis=0)
            scores = {
                name: float(np.clip(np.mean(np.sum((basis.T @ occipital) ** 2, axis=0) / power), 0.0, 1.0))
                for name, basis in self._templates.items()
            }
            ranked = sorted(scores, key=scores.get, reverse=True)
            winner, runner_up = ranked[:2]
            score, second = scores[winner], scores[runner_up]
            if score < self.MIN_SCORE or score - second < self.MIN_MARGIN or score < second * self.MIN_WINNER_RATIO:
                self._clear_candidate()
                result = self._result(
                    "good",
                    "Prototype residual checks passed; no clear frequency-template winner",
                    metrics=metrics,
                    score=score,
                )
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
                        "Qualified prototype spectral match; shadow preview only"
                        if qualified
                        else "Prototype frequency candidate needs more consecutive fresh evidence"
                    ),
                    metrics=metrics,
                    candidate=winner,
                    command=winner if qualified else "none",
                    score=score,
                )
        return self._cache_result(result)
