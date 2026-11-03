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
            set(result),
            {
                "command",
                "candidate",
                "reason",
                "quality",
                "score",
                "channel_rms_uv",
                "sample_count",
                "channel_offset_mv",
                "channel_detrended_rms_uv",
                "channel_trend_rms_uv",
                "channel_line_rms_uv",
                "channel_conditioned_rms_uv",
                "line_frequency_hz",
            },
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
        engine = PREVIEW.PreviewEngine(sample_rate=50, targets={"forward": 12, "left": 15})
        self.assertIsNone(engine.line_frequency_hz)
        self.assertIsNone(engine.evaluate(1.0)["line_frequency_hz"])
        with self.assertRaises(ValueError):
            PREVIEW.PreviewEngine(sample_rate=50, targets={"forward": 12, "left": 25})


class TestPreviewConditioning(unittest.TestCase):
    """Separate DC and line interference from meaningful residual-signal quality."""

    TARGETS = {"forward": 12.0, "left": 15.0}
    OFFSETS_V = np.array([0.020, 0.075, -0.125, 0.200])
    DIAGNOSTIC_LISTS = (
        "channel_rms_uv",
        "channel_offset_mv",
        "channel_detrended_rms_uv",
        "channel_trend_rms_uv",
        "channel_line_rms_uv",
        "channel_conditioned_rms_uv",
    )

    def signal(self, count: int = 750, *, frequency: float = 12.0) -> np.ndarray:
        times = np.arange(count) / 250.0
        random = np.random.default_rng(908)
        return random.normal(0.0, 1e-6, (count, 4)) + np.sin(2 * np.pi * frequency * times)[:, None] * np.array(
            [25e-6, 22e-6, 8e-6, 7e-6]
        )

    def line(self, count: int, amplitude_uv: float, *, frequency: float = 60.0) -> np.ndarray:
        times = np.arange(count) / 250.0
        return (amplitude_uv * 1e-6 * np.sin(2 * np.pi * frequency * times))[:, None]

    def stream(self, engine: object, samples: np.ndarray) -> list[dict]:
        results = []
        for start in range(0, len(samples), 25):
            end = min(start + 25, len(samples))
            now = 1.0 + end / 250.0
            engine.add_samples(samples[start:end], [index % 2048 for index in range(start, end)], now)
            results.append(engine.evaluate(now))
        return results

    def test_realistic_channel_offsets_preserve_quality_and_target_match(self) -> None:
        samples = self.signal()
        clean = self.stream(PREVIEW.PreviewEngine(targets=self.TARGETS), samples)[-1]
        shifted = self.stream(PREVIEW.PreviewEngine(targets=self.TARGETS), samples + self.OFFSETS_V)[-1]
        self.assertEqual(clean["command"], "forward")
        self.assertEqual(shifted["quality"], "good")
        self.assertEqual(shifted["command"], clean["command"])
        self.assertAlmostEqual(shifted["score"], clean["score"], places=9)
        np.testing.assert_allclose(
            np.array(shifted["channel_offset_mv"]) - clean["channel_offset_mv"], self.OFFSETS_V * 1000, atol=1e-8
        )
        for key in ("channel_rms_uv", "channel_detrended_rms_uv", "channel_conditioned_rms_uv"):
            np.testing.assert_allclose(shifted[key], clean[key], atol=1e-7)

    def test_diagnostics_distinguish_drift_line_and_residual(self) -> None:
        samples = self.signal(500)
        drift = (40e-6 * np.arange(500) / 250.0)[:, None]
        engine = PREVIEW.PreviewEngine()
        result = self.stream(engine, samples + drift + self.line(500, 3.0) + self.OFFSETS_V)[-1]
        self.assertEqual(result["quality"], "good")
        self.assertEqual(result["line_frequency_hz"], 60.0)
        self.assertTrue(np.all(np.array(result["channel_rms_uv"]) > result["channel_detrended_rms_uv"]))
        self.assertTrue(np.all(np.array(result["channel_trend_rms_uv"]) > 10.0))
        self.assertTrue(np.all(np.array(result["channel_trend_rms_uv"]) < 100.0))
        np.testing.assert_allclose(result["channel_line_rms_uv"], [3 / np.sqrt(2)] * 4, atol=0.15)
        self.assertTrue(np.all(np.array(result["channel_detrended_rms_uv"]) > result["channel_conditioned_rms_uv"]))

    def test_strong_sixty_hz_blocks_even_a_real_twelve_hz_template(self) -> None:
        engine = PREVIEW.PreviewEngine(targets=self.TARGETS)
        results = self.stream(engine, self.signal() + self.line(750, 200.0) + self.OFFSETS_V)
        self.assertTrue(all(result["command"] == "none" for result in results))
        interference = [result for result in results if result["quality"] == "interference"]
        self.assertTrue(interference)
        self.assertGreater(max(interference[-1]["channel_line_rms_uv"]), 100.0)
        self.assertEqual(interference[-1]["candidate"], "none")

    def test_line_only_input_never_qualifies_a_command(self) -> None:
        for amplitude_uv in (10.0, 200.0):
            with self.subTest(amplitude_uv=amplitude_uv):
                engine = PREVIEW.PreviewEngine(targets=self.TARGETS)
                results = self.stream(engine, np.repeat(self.line(750, amplitude_uv), 4, axis=1) + self.OFFSETS_V)
                self.assertTrue(all(result["command"] == "none" for result in results))
                self.assertEqual(results[-1]["quality"], "interference")

    def test_small_sixty_hz_component_is_removed_without_changing_match(self) -> None:
        samples = self.signal()
        clean = self.stream(PREVIEW.PreviewEngine(targets=self.TARGETS), samples)[-1]
        mixed = self.stream(PREVIEW.PreviewEngine(targets=self.TARGETS), samples + self.line(750, 3.0))[-1]
        self.assertEqual(mixed["quality"], "good")
        self.assertEqual(mixed["command"], "forward")
        self.assertAlmostEqual(mixed["score"], clean["score"], places=9)
        np.testing.assert_allclose(mixed["channel_conditioned_rms_uv"], clean["channel_conditioned_rms_uv"], atol=1e-7)

    def test_large_off_frequency_and_modulated_line_cannot_look_clean(self) -> None:
        times = np.arange(750) / 250.0
        interference_signals = (
            self.line(750, 200.0, frequency=59.0),
            self.line(750, 300.0) * np.sin(2 * np.pi * 0.7 * times)[:, None],
        )
        for interference in interference_signals:
            with self.subTest(peak=float(np.max(interference))):
                engine = PREVIEW.PreviewEngine(targets=self.TARGETS)
                results = self.stream(engine, self.signal() + interference + self.OFFSETS_V)
                self.assertTrue(all(result["command"] == "none" for result in results))
                self.assertFalse(any(result["quality"] == "good" for result in results))
                self.assertTrue(any(result["quality"] in ("artifact", "interference") for result in results))

    def test_spikes_and_steps_remain_artifacts_under_large_dc_offsets(self) -> None:
        for shape in ("spike", "step"):
            with self.subTest(shape=shape):
                samples = self.signal(500) + self.OFFSETS_V
                if shape == "spike":
                    samples[250, 2] += 300e-6
                else:
                    samples[250:, 2] += 300e-6
                engine = PREVIEW.PreviewEngine(targets=self.TARGETS)
                engine.add_samples(samples, list(range(500)), 3.0)
                result = engine.evaluate(3.0)
                self.assertEqual(result["quality"], "artifact")
                self.assertEqual(result["command"], "none")

    def test_five_identical_samples_on_one_channel_flag_flat_topped_artifact(self) -> None:
        samples = self.signal(500) + self.OFFSETS_V
        samples[200:205, 0] = samples[200, 0]
        engine = PREVIEW.PreviewEngine(targets=self.TARGETS)
        engine.add_samples(samples, list(range(500)), 3.0)
        result = engine.evaluate(3.0)
        self.assertEqual(result["quality"], "artifact")
        self.assertEqual(result["command"], "none")

    def test_conditioning_does_not_modify_input_or_share_its_storage(self) -> None:
        samples = self.signal(500) + self.OFFSETS_V
        original = samples.copy()
        engine = PREVIEW.PreviewEngine(targets=self.TARGETS)
        engine.add_samples(samples, list(range(500)), 3.0)
        np.testing.assert_array_equal(samples, original)
        samples[:] = 0.0
        result = engine.evaluate(3.0)
        self.assertEqual(result["quality"], "good")
        self.assertEqual(result["candidate"], "forward")
        np.testing.assert_allclose(result["channel_offset_mv"], np.mean(original, axis=0) * 1000, atol=1e-8)

    def test_all_diagnostic_lists_are_copied_from_cached_results(self) -> None:
        engine = PREVIEW.PreviewEngine(targets=self.TARGETS)
        result = self.stream(engine, self.signal() + self.OFFSETS_V)[-1]
        expected = {key: list(result[key]) for key in self.DIAGNOSTIC_LISTS}
        for key in self.DIAGNOSTIC_LISTS:
            result[key][:] = [-123456.0]
        cached = engine.evaluate(4.0)
        for key in self.DIAGNOSTIC_LISTS:
            self.assertEqual(cached[key], expected[key])

    def test_fifty_hz_and_disabled_options_use_the_requested_conditioning(self) -> None:
        samples = self.signal() + self.line(750, 3.0, frequency=50.0)
        enabled = self.stream(PREVIEW.PreviewEngine(targets=self.TARGETS, line_frequency_hz=50.0), samples)[-1]
        disabled = self.stream(PREVIEW.PreviewEngine(targets=self.TARGETS, line_frequency_hz=None), samples)[-1]
        self.assertEqual(enabled["command"], "forward")
        self.assertEqual(enabled["line_frequency_hz"], 50.0)
        np.testing.assert_allclose(enabled["channel_line_rms_uv"], [3 / np.sqrt(2)] * 4, atol=0.15)
        self.assertIsNone(disabled["line_frequency_hz"])
        np.testing.assert_allclose(
            disabled["channel_conditioned_rms_uv"], disabled["channel_detrended_rms_uv"], atol=1e-8
        )
        np.testing.assert_allclose(disabled["channel_line_rms_uv"], [0.0] * 4, atol=1e-8)

    def test_invalid_line_options_fail_explicitly(self) -> None:
        for frequency in (0, -50, 12, 59.0, 61, True, np.bool_(False), float("nan"), float("inf"), "60"):
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                PREVIEW.PreviewEngine(line_frequency_hz=frequency)

    def test_large_numeric_input_is_still_rejected_above_one_volt(self) -> None:
        engine = PREVIEW.PreviewEngine(targets=self.TARGETS)
        engine.add_samples([[1.01, 0.020, 0.125, 0.200]], [0], 1.0)
        result = engine.evaluate(1.0)
        self.assertEqual(result["quality"], "artifact")
        self.assertEqual(result["command"], "none")
        self.assertEqual(result["sample_count"], 0)

    def test_ten_millivolt_ramp_cannot_be_hidden_by_detrending(self) -> None:
        samples = self.signal(500) + self.OFFSETS_V
        samples += np.linspace(0.0, 0.010, 500)[:, None]
        engine = PREVIEW.PreviewEngine(targets=self.TARGETS)
        engine.add_samples(samples, list(range(500)), 3.0)
        result = engine.evaluate(3.0)
        self.assertEqual(result["quality"], "artifact")
        self.assertEqual(result["command"], "none")
        self.assertTrue(np.all(np.array(result["channel_trend_rms_uv"]) > 100.0))
        expected = list(result["channel_trend_rms_uv"])
        result["channel_trend_rms_uv"][0] = -1.0
        repeated = engine.evaluate(3.1)
        self.assertEqual(repeated["quality"], "artifact")
        self.assertEqual(repeated["channel_trend_rms_uv"], expected)

    def test_removed_second_harmonic_cannot_vote_for_its_target(self) -> None:
        for line_frequency, target_frequency in ((60.0, 30.0), (50.0, 25.0)):
            targets = {"forward": target_frequency, "left": 12.0}
            with self.subTest(line=line_frequency):
                engine = PREVIEW.PreviewEngine(targets=targets, line_frequency_hz=line_frequency)
                clean = self.stream(engine, self.signal(frequency=target_frequency))[-1]
                self.assertEqual(clean["command"], "forward")
                line_only = np.repeat(self.line(750, 30.0, frequency=line_frequency), 4, axis=1)
                engine = PREVIEW.PreviewEngine(targets=targets, line_frequency_hz=line_frequency)
                results = self.stream(engine, line_only + self.OFFSETS_V)
                self.assertTrue(all(result["command"] == "none" for result in results))
                self.assertEqual(results[-1]["quality"], "interference")


if __name__ == "__main__":
    unittest.main()
