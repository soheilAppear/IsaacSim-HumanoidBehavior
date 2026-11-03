"""Exercise shadow EEG preview with generated signals; no hardware validation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "eeg" / "preview.py"
SPEC = importlib.util.spec_from_file_location("eeg_preview_under_test", MODULE_PATH)
PREVIEW = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREVIEW
SPEC.loader.exec_module(PREVIEW)


class TestEegPreview(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PREVIEW.PreviewEngine(targets={"forward": 12.0, "left": 15.0, "right": 20.0})
        self.position = 0
        self.now = 1.0
        self.random = np.random.default_rng(701)

    def signal(self, count: int, frequency: float | None = 12.0, *, amplitude_uv: float = 25.0) -> np.ndarray:
        time_axis = np.arange(self.position, self.position + count) / 250.0
        samples = self.random.normal(0.0, 2e-6, (count, 4))
        if frequency is not None:
            samples[:, 0] += amplitude_uv * 1e-6 * np.sin(2 * np.pi * frequency * time_axis)
            samples[:, 1] += amplitude_uv * 1e-6 * np.sin(2 * np.pi * frequency * time_axis + 0.6)
        return samples

    def feed(self, count: int = 25, frequency: float | None = 12.0, **kwargs) -> dict:
        samples = self.signal(count, frequency, **kwargs)
        packets = [(self.position + index) % 2048 for index in range(count)]
        self.position += count
        self.now += count / 250.0
        self.engine.add_samples(samples.tolist(), packets, self.now)
        return self.engine.evaluate(self.now)

    def qualify(self, frequency: float = 12.0) -> dict:
        result = None
        for _ in range(28):
            result = self.feed(frequency=frequency)
        return result

    def test_unconfigured_default_reports_quality_without_command(self) -> None:
        self.engine = PREVIEW.PreviewEngine()
        result = self.qualify()
        self.assertEqual(result["quality"], "good")
        self.assertEqual(result["command"], "none")
        self.assertEqual(result["candidate"], "none")
        self.assertIn("Unconfigured", result["reason"])
        self.assertEqual(
            set(result), {"command", "candidate", "reason", "quality", "score", "channel_rms_uv", "sample_count"}
        )

    def test_each_synthetic_frequency_selects_its_target(self) -> None:
        for frequency, command in ((12, "forward"), (15, "left"), (20, "right")):
            with self.subTest(frequency=frequency):
                self.setUp()
                result = self.qualify(frequency)
                self.assertEqual(result["quality"], "good")
                self.assertEqual(result["command"], command)
                self.assertGreater(result["score"], 0.9)
                self.assertEqual(len(result["channel_rms_uv"]), 4)

    def test_full_window_requires_additional_fresh_evidence(self) -> None:
        for _ in range(20):
            result = self.feed()
        self.assertEqual(result["sample_count"], 500)
        self.assertEqual(result["candidate"], "forward")
        self.assertEqual(result["command"], "none")
        for _ in range(5):
            self.assertEqual(self.feed()["command"], "none")
        self.assertEqual(self.feed()["command"], "forward")

    def test_repeated_evaluation_cannot_arm_or_extend_freshness(self) -> None:
        for _ in range(20):
            self.feed()
        for offset in (0.1, 0.2, 0.3, 0.4, 0.5):
            result = self.engine.evaluate(self.now + offset)
            self.assertEqual(result["command"], "none")
        result = self.engine.evaluate(self.now + 0.501)
        self.assertEqual(result["quality"], "stale")
        self.assertEqual(result["sample_count"], 0)

    def test_burst_samples_cannot_substitute_for_elapsed_dwell(self) -> None:
        self.engine.add_samples(self.signal(500).tolist(), list(range(500)), self.now)
        self.engine.evaluate(self.now)
        self.position = 500
        self.engine.add_samples(self.signal(200).tolist(), list(range(500, 700)), self.now + 0.01)
        result = self.engine.evaluate(self.now + 0.01)
        self.assertEqual(result["candidate"], "forward")
        self.assertEqual(result["command"], "none")

    def test_noise_has_no_winner(self) -> None:
        for _ in range(40):
            result = self.feed(frequency=None)
            self.assertEqual(result["command"], "none")
        self.assertEqual(result["quality"], "good")
        self.assertEqual(result["candidate"], "none")
        self.assertLess(result["score"], 0.1)

    def test_equal_target_mixture_has_no_winner(self) -> None:
        for _ in range(30):
            samples = self.signal(25)
            time_axis = np.arange(self.position, self.position + 25) / 250
            samples[:, 0] += 25e-6 * np.sin(2 * np.pi * 15 * time_axis)
            samples[:, 1] += 25e-6 * np.sin(2 * np.pi * 15 * time_axis + 0.6)
            packets = [(self.position + index) % 2048 for index in range(25)]
            self.position += 25
            self.now += 0.1
            self.engine.add_samples(samples.tolist(), packets, self.now)
            result = self.engine.evaluate(self.now)
        self.assertEqual(result["quality"], "good")
        self.assertEqual(result["command"], "none")
        self.assertEqual(result["candidate"], "none")

    def test_temporal_tones_do_not_vote_for_occipital_targets(self) -> None:
        for _ in range(30):
            samples = self.signal(25, None)
            time_axis = np.arange(self.position, self.position + 25) / 250
            samples[:, 2:] += (50e-6 * np.sin(2 * np.pi * 12 * time_axis))[:, None]
            packets = [(self.position + index) % 2048 for index in range(25)]
            self.position += 25
            self.now += 0.1
            self.engine.add_samples(samples.tolist(), packets, self.now)
            result = self.engine.evaluate(self.now)
        self.assertEqual(result["command"], "none")
        self.assertLess(result["score"], 0.1)

    def test_flat_temporal_channel_rejects_otherwise_clear_target(self) -> None:
        samples = self.signal(500)
        samples[:, 3] = 0
        self.engine.add_samples(samples.tolist(), list(range(500)), self.now)
        result = self.engine.evaluate(self.now)
        self.assertEqual(result["quality"], "artifact")
        self.assertEqual(result["command"], "none")
        self.assertEqual(result["sample_count"], 0)

    def test_gross_artifact_immediately_clears_qualified_command(self) -> None:
        self.assertEqual(self.qualify()["command"], "forward")
        result = self.feed(amplitude_uv=1500)
        self.assertEqual(result["quality"], "artifact")
        self.assertEqual(result["command"], "none")
        self.assertEqual(self.feed()["quality"], "waiting")

    def test_abrupt_artifact_below_absolute_amplitude_cap_is_rejected(self) -> None:
        samples = self.signal(500)
        samples[250, 2] = 300e-6
        self.engine.add_samples(samples.tolist(), list(range(500)), self.now)
        result = self.engine.evaluate(self.now)
        self.assertEqual(result["quality"], "artifact")
        self.assertEqual(result["command"], "none")

    def test_nonfinite_or_malformed_samples_clear_history(self) -> None:
        for samples, packets in (
            ([[float("nan"), 0, 0, 0]], [0]),
            ([[0, float("inf"), 0, 0]], [0]),
            ([[1e-6, 2e-6, 3e-6]], [0]),
            ([[1e-6] * 4], []),
            ([["bad"] * 4], [0]),
            ([["0.000001"] * 4], [0]),
            ([[True] * 4], [0]),
            ([[1e-6 + 1j] * 4], [0]),
        ):
            with self.subTest(samples=samples):
                self.engine.add_samples(samples, packets, self.now)
                result = self.engine.evaluate(self.now)
                self.assertEqual(result["quality"], "artifact")
                self.assertEqual(result["command"], "none")
                self.assertEqual(result["sample_count"], 0)

    def test_missing_duplicate_and_reverse_packets_clear_history(self) -> None:
        for packet in (26, 24, 20):
            with self.subTest(packet=packet):
                self.setUp()
                self.feed()
                self.engine.add_samples(self.signal(1).tolist(), [packet], self.now + 0.01)
                result = self.engine.evaluate(self.now + 0.01)
                self.assertEqual(result["quality"], "gap")
                self.assertEqual(result["sample_count"], 0)

    def test_gap_within_batch_is_rejected(self) -> None:
        self.engine.add_samples(self.signal(3).tolist(), [10, 11, 13], self.now)
        self.assertEqual(self.engine.evaluate(self.now)["quality"], "gap")

    def test_invalid_packet_types_and_ranges_are_rejected(self) -> None:
        for packet in (-1, 2048, True, 0.5, "0"):
            with self.subTest(packet=packet):
                self.engine.add_samples(self.signal(1).tolist(), [packet], self.now)
                self.assertEqual(self.engine.evaluate(self.now)["quality"], "gap")

    def test_sequence_wrap_is_contiguous(self) -> None:
        self.position = 1990
        result = self.qualify()
        self.assertEqual(result["quality"], "good")
        self.assertEqual(result["command"], "forward")

    def test_brainbit2_counter_accepts_values_above_original_modulus(self) -> None:
        self.engine = PREVIEW.PreviewEngine(packet_modulus=2**32)
        self.engine.add_samples(self.signal(500).tolist(), list(range(5000, 5500)), self.now)
        result = self.engine.evaluate(self.now)
        self.assertEqual(result["quality"], "good")
        self.assertEqual(result["sample_count"], 500)

    def test_brainbit2_uint32_wrap_is_contiguous_across_batches(self) -> None:
        self.engine = PREVIEW.PreviewEngine(packet_modulus=2**32)
        self.engine.add_samples(self.signal(250).tolist(), list(range(2**32 - 250, 2**32)), self.now)
        self.position += 250
        self.engine.add_samples(self.signal(250).tolist(), list(range(250)), self.now + 0.1)
        result = self.engine.evaluate(self.now + 0.1)
        self.assertEqual(result["quality"], "good")
        self.assertEqual(result["sample_count"], 500)

    def test_brainbit2_out_of_range_counter_is_rejected(self) -> None:
        self.engine = PREVIEW.PreviewEngine(packet_modulus=2**32)
        self.engine.add_samples(self.signal(1).tolist(), [2**32], self.now)
        result = self.engine.evaluate(self.now)
        self.assertEqual(result["quality"], "gap")
        self.assertIn(str(2**32), result["reason"])

    def test_invalid_packet_modulus_fails_explicitly(self) -> None:
        for modulus in (0, 1, -1, True, np.bool_(True), 2048.0, 2**32 + 1, float("nan"), "2048"):
            with self.subTest(modulus=modulus):
                with self.assertRaises(ValueError):
                    PREVIEW.PreviewEngine(packet_modulus=modulus)

    def test_stale_qualified_result_clears_history_and_requires_full_window(self) -> None:
        self.assertEqual(self.qualify()["command"], "forward")
        self.now += 0.6
        result = self.engine.evaluate(self.now)
        self.assertEqual(result["quality"], "stale")
        self.assertEqual(result["command"], "none")
        self.assertEqual(self.feed()["quality"], "waiting")

    def test_receive_gap_is_detected_without_intermediate_poll(self) -> None:
        self.feed()
        self.now += 0.6
        result = self.feed()
        self.assertEqual(result["quality"], "gap")
        self.assertEqual(result["sample_count"], 0)

    def test_empty_batch_does_not_keep_old_sample_fresh(self) -> None:
        self.qualify()
        self.engine.add_samples([], [], self.now + 0.4)
        self.assertEqual(self.engine.evaluate(self.now + 0.6)["quality"], "stale")

    def test_reversed_or_invalid_clocks_disarm(self) -> None:
        for clock in (float("nan"), float("inf"), -1, True):
            with self.subTest(clock=clock):
                self.setUp()
                self.feed()
                self.assertEqual(self.engine.evaluate(clock)["command"], "none")
                self.assertEqual(self.engine.evaluate(self.now)["quality"], "stale")
        self.setUp()
        self.feed()
        self.assertEqual(self.engine.evaluate(self.now - 0.01)["quality"], "stale")

    def test_target_change_clears_old_command_until_new_dwell(self) -> None:
        self.assertEqual(self.qualify()["command"], "forward")
        seen_none = False
        for _ in range(35):
            result = self.feed(frequency=15)
            seen_none |= result["command"] == "none"
        self.assertTrue(seen_none)
        self.assertEqual(result["command"], "left")

    def test_ambiguous_window_resets_dwell(self) -> None:
        self.assertEqual(self.qualify()["command"], "forward")
        for _ in range(22):
            result = self.feed(frequency=None)
        self.assertEqual(result["command"], "none")
        self.assertEqual(result["candidate"], "none")

    def test_harmonic_only_signal_can_match_the_declared_template(self) -> None:
        result = self.qualify(frequency=24)
        self.assertEqual(result["command"], "forward")

    def test_returned_rms_mutation_does_not_corrupt_cached_result(self) -> None:
        result = self.qualify()
        result["channel_rms_uv"][0] = 100000
        self.assertLess(self.engine.evaluate(self.now)["channel_rms_uv"][0], 100)

    def test_invalid_target_configurations_fail_explicitly(self) -> None:
        for targets in (
            {},
            {"forward": 12},
            {"up": 12, "left": 15},
            {"forward": 12, "left": 12},
            {"forward": 12, "left": 12.2},
            {"forward": 12, "left": 24},
            {"forward": float("nan"), "left": 15},
            {"forward": True, "left": 15},
            {"forward": 5, "left": 15},
            {"forward": 41, "left": 15},
            [12, 15],
        ):
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    PREVIEW.PreviewEngine(targets=targets)

    def test_invalid_sample_rate_and_duration_fail_explicitly(self) -> None:
        for sample_rate in (0, -1, 1, 2001, float("nan"), True):
            with self.subTest(sample_rate=sample_rate):
                with self.assertRaises(ValueError):
                    PREVIEW.PreviewEngine(sample_rate=sample_rate)
        for duration in (0, 0.5, 11, float("inf"), True):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    PREVIEW.PreviewEngine(window_seconds=duration)

    def test_nyquist_limit_rejects_invalid_target_at_lower_sample_rate(self) -> None:
        PREVIEW.PreviewEngine(sample_rate=50, targets={"forward": 12, "left": 15})
        with self.assertRaises(ValueError):
            PREVIEW.PreviewEngine(sample_rate=50, targets={"forward": 12, "left": 25})


if __name__ == "__main__":
    unittest.main()
