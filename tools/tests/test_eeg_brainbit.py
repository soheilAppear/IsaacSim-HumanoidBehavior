# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the BrainBit worker against a fake NeuroSDK without Bluetooth hardware."""

from __future__ import annotations

import importlib.util
import json
import math
import queue
import threading
import types
import unittest
from enum import Enum
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("brainbit_under_test", Path(__file__).parents[1] / "eeg" / "brainbit.py")
BRAINBIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRAINBIT)


class _Command(Enum):
    StartResist = 0
    StopResist = 1
    StartSignal = 2
    StopSignal = 3


class _Family(Enum):
    LEBrainBit = 0
    LEBrainBitBlack = 1
    LEBrainBit2 = 2


class _State(Enum):
    StateInRange = 0
    StateOutOfRange = 1


class _Frequency(Enum):
    FrequencyHz250 = 4
    FrequencyHz500 = 5


def _device(address: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(Name="BrainBit", Address=address, SerialNumber="1234", SensFamily=_Family.LEBrainBit)


def _sample(packet: int = 9, first: float = 1e-6) -> types.SimpleNamespace:
    return types.SimpleNamespace(PackNum=packet, Marker=0, O1=first, O2=2e-6, T3=3e-6, T4=4e-6)


def _channel(name: str, number: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(Id=types.SimpleNamespace(name=f"EEGChId{name}"), Name=name, Num=number)


class _FakeSdk:
    """Model the SDK's commands, callbacks, retained references and native destruction."""

    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop
        self.calls = []
        self.devices = [_device("AA:BB")]
        self.on_start = None
        self.on_command = None
        self.create_error = None
        self.frequency = _Frequency.FrequencyHz250
        self.channels_count = 4
        self.channels = [_channel("O1", 3), _channel("O2", 0), _channel("T3", 1), _channel("T4", 2)]
        self.clear_on_stop = False
        self.sensor = None
        owner = self

        class Sensor:
            def __init__(self) -> None:
                self.signalDataReceived = None
                self.resistDataReceived = None
                self.state = _State.StateInRange
                self.sampling_frequency = owner.frequency
                self.channels_count = owner.channels_count
                self.supported_channels = owner.channels
                self.closed = False

            def exec_command(self, command: _Command) -> None:
                owner.calls.append(command.name)
                if owner.on_command is not None:
                    owner.on_command(self, command)
                elif command == _Command.StartResist:
                    self.resistDataReceived(self, types.SimpleNamespace(O1=1000.0, O2=2000.0, T3=3000.0, T4=4000.0))
                elif command == _Command.StartSignal:
                    self.signalDataReceived(self, [_sample(), _sample(10, 5e-6)])
                    owner.stop.set()

            def disconnect(self) -> None:
                owner.calls.append("disconnect")
                self.state = _State.StateOutOfRange

            def __del__(self) -> None:
                if not self.closed:
                    owner.calls.append("destroy_sensor")
                    self.closed = True

        class Scanner:
            def __init__(self, families: list) -> None:
                owner.calls.append(("scanner", families))
                self.closed = False

            def start(self) -> None:
                owner.calls.append("scan_start")
                if owner.on_start is not None:
                    owner.on_start()

            def stop(self) -> None:
                owner.calls.append("scan_stop")
                if owner.clear_on_stop:
                    owner.devices.clear()

            def sensors(self) -> list:
                owner.calls.append("sensors")
                return owner.devices

            def create_sensor(self, device: types.SimpleNamespace) -> Sensor:
                owner.calls.append(("create_sensor", device.Address))
                if owner.create_error is not None:
                    raise owner.create_error
                owner.sensor = Sensor()
                return owner.sensor

            def __del__(self) -> None:
                if not self.closed:
                    owner.calls.append("destroy_scanner")
                    self.closed = True

        self.modules = {
            "neurosdk": types.ModuleType("neurosdk"),
            "neurosdk.cmn_types": types.SimpleNamespace(
                SensorCommand=_Command, SensorFamily=_Family, SensorState=_State
            ),
            "neurosdk.scanner": types.SimpleNamespace(Scanner=Scanner),
        }


class TestBrainBitAcquisition(unittest.TestCase):
    def setUp(self) -> None:
        self.stop = threading.Event()
        self.events = queue.Queue()
        self.sdk = _FakeSdk(self.stop)

    def run_worker(self, **kwargs) -> list[dict]:
        with patch.dict("sys.modules", self.sdk.modules):
            BRAINBIT.run_brainbit(self.events, self.stop, scan_seconds=0.001, resistance_seconds=0.001, **kwargs)
        result = []
        while not self.events.empty():
            result.append(self.events.get_nowait())
            self.events.task_done()
        return result

    def test_modes_callbacks_units_and_explicit_cleanup_order(self) -> None:
        events = self.run_worker()
        self.assertEqual(
            self.sdk.calls,
            [
                ("scanner", [_Family.LEBrainBit, _Family.LEBrainBitBlack, _Family.LEBrainBit2]),
                "scan_start",
                "sensors",
                "scan_stop",
                ("create_sensor", "AA:BB"),
                "StartResist",
                "StopResist",
                "StartSignal",
                "StopSignal",
                "disconnect",
                "destroy_sensor",
                "destroy_scanner",
            ],
        )
        self.assertEqual(
            [item["state"] for item in events if item["type"] == "status"],
            ["scanning", "connecting", "contact_check", "streaming", "closed"],
        )
        info = next(item["info"] for item in events if item["type"] == "device")
        self.assertEqual(info["sample_rate_hz"], 250)
        self.assertEqual(info["channels"], ["O1", "O2", "T3", "T4"])
        resistance = next(item for item in events if item["type"] == "resistance")
        self.assertEqual(resistance["values_ohm"], [1000.0, 2000.0, 3000.0, 4000.0])
        signal = next(item for item in events if item["type"] == "samples")
        self.assertEqual([sample["packet"] for sample in signal["samples"]], [9, 10])
        self.assertEqual(signal["samples"][0]["values_v"], [1e-6, 2e-6, 3e-6, 4e-6])
        self.assertGreater(signal["received_monotonic"], 0)
        self.assertGreater(signal["received_unix_ns"], 0)
        self.assertIsNone(self.sdk.sensor.signalDataReceived)
        self.assertIsNone(self.sdk.sensor.resistDataReceived)

    def test_brainbit_black_uses_named_channels(self) -> None:
        self.sdk.devices[0].SensFamily = _Family.LEBrainBitBlack
        events = self.run_worker()
        info = next(event["info"] for event in events if event["type"] == "device")
        self.assertEqual(info["family"], "LEBrainBitBlack")
        self.assertIsNone(info["native_sample_indices"])
        self.assertEqual(events[-1]["state"], "closed")

    def test_brainbit2_maps_native_indices_and_retains_full_packet_counter(self) -> None:
        self.sdk.devices[0].SensFamily = _Family.LEBrainBit2

        def arrays(sensor, command) -> None:
            if command == _Command.StartResist:
                sensor.resistDataReceived(
                    sensor,
                    [
                        types.SimpleNamespace(PackNum=88, Samples=[2000.0, 3000.0, 4000.0, 1000.0], Referents=[]),
                        types.SimpleNamespace(PackNum=89, Samples=[math.inf, 3000.0, 4000.0, 1000.0], Referents=[]),
                    ],
                )
            elif command == _Command.StartSignal:
                sensor.signalDataReceived(
                    sensor, [types.SimpleNamespace(PackNum=123456, Marker=7, Samples=[2e-6, 3e-6, 4e-6, 1e-6])]
                )
                self.stop.set()

        self.sdk.on_command = arrays
        events = self.run_worker()
        info = next(event["info"] for event in events if event["type"] == "device")
        self.assertEqual(info["family"], "LEBrainBit2")
        self.assertEqual(info["native_sample_indices"], [3, 0, 1, 2])
        self.assertEqual(info["packet_modulus"], 2**32)
        self.assertEqual([channel["number"] for channel in info["native_channels"]], [3, 0, 1, 2])
        signal = next(event for event in events if event["type"] == "samples")
        self.assertEqual(signal["samples"], [{"packet": 123456, "marker": 7, "values_v": [1e-6, 2e-6, 3e-6, 4e-6]}])
        resistance = [event for event in events if event["type"] == "resistance"]
        self.assertEqual([event["packet"] for event in resistance], [88, 89])
        self.assertEqual(resistance[0]["values_ohm"], [1000.0, 2000.0, 3000.0, 4000.0])
        self.assertEqual(resistance[1]["values_ohm"], [1000.0, None, 3000.0, 4000.0])
        self.assertEqual(resistance[1]["unavailable_channels"], ["O2"])
        json.dumps(events, allow_nan=False)
        self.assertLess(self.sdk.calls.index("StopResist"), self.sdk.calls.index("StartSignal"))
        self.assertEqual(events[-1]["state"], "closed")

    def test_brainbit2_rejects_ambiguous_channel_descriptors(self) -> None:
        cases = [
            [_channel("O1", 0), _channel("O2", 0), _channel("T3", 2), _channel("T4", 3)],
            [_channel("O1", 0), _channel("O2", 1), _channel("T3", 2), _channel("T4", 4)],
            [_channel("O1", 0), _channel("O2", 1), _channel("T3", 2), _channel("P4", 3)],
            [_channel("O1", 0), _channel("O2", 1), _channel("T3", 2), _channel("O1", 3)],
            [_channel("Unknown", 0), _channel("O2", 1), _channel("T3", 2), _channel("T4", 3)],
            [_channel("O1", 0), _channel("O2", 1)],
        ]
        for descriptors in cases:
            with self.subTest(descriptors=descriptors):
                self.setUp()
                self.sdk.devices[0].SensFamily = _Family.LEBrainBit2
                self.sdk.channels = descriptors
                events = self.run_worker()
                self.assertEqual(events[-1]["state"], "error")
                self.assertNotIn("StartSignal", self.sdk.calls)
                self.assertNotIn("StartResist", self.sdk.calls)

    def test_brainbit2_accepts_explicit_name_for_unknown_id_but_rejects_conflict(self) -> None:
        sensor = types.SimpleNamespace(channels_count=4, supported_channels=self.sdk.channels)
        sensor.supported_channels[0].Id.name = "EEGChIdUnknown"
        indices, _ = BRAINBIT._read_channel_layout(sensor, True)
        self.assertEqual(indices, (3, 0, 1, 2))
        sensor.supported_channels[0].Id.name = "EEGChIdO2"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            BRAINBIT._read_channel_layout(sensor, True)

    def test_eight_channel_device_is_rejected(self) -> None:
        self.sdk.devices[0].SensFamily = _Family.LEBrainBit2
        self.sdk.channels_count = 8
        events = self.run_worker()
        self.assertEqual(events[-1]["state"], "error")
        self.assertIn("exactly four", events[-1]["message"])
        self.assertNotIn("StartResist", self.sdk.calls)

    def test_brainbit2_rejects_mismatched_sample_payload_size(self) -> None:
        self.sdk.devices[0].SensFamily = _Family.LEBrainBit2

        def short_packet(sensor, command) -> None:
            if command == _Command.StartSignal:
                sensor.signalDataReceived(sensor, [types.SimpleNamespace(PackNum=3, Marker=0, Samples=[1e-6, 2e-6])])

        self.sdk.on_command = short_packet
        events = self.run_worker()
        self.assertEqual(events[-1]["state"], "error")
        self.assertIn("Expected four", events[-1]["message"])
        self.assertIn("StopSignal", self.sdk.calls)

    def test_scanner_result_is_copied_before_stopping_discovery(self) -> None:
        self.sdk.clear_on_stop = True
        events = self.run_worker()
        self.assertIn(("create_sensor", "AA:BB"), self.sdk.calls)
        self.assertEqual(events[-1]["state"], "closed")

    def test_multiple_devices_require_explicit_address(self) -> None:
        self.sdk.devices.append(_device("CC:DD"))
        events = self.run_worker()
        self.assertIn("--address", events[-1]["message"])
        self.assertEqual(events[-1]["state"], "error")
        self.assertIsNone(self.sdk.sensor)
        self.assertEqual(self.sdk.calls[-1], "destroy_scanner")

    def test_address_selects_case_insensitively_without_connecting_other_devices(self) -> None:
        self.sdk.devices.append(_device("CC:DD"))
        self.run_worker(address="cc:dd")
        self.assertIn(("create_sensor", "CC:DD"), self.sdk.calls)
        self.assertNotIn(("create_sensor", "AA:BB"), self.sdk.calls)

    def test_no_matching_sensor_reports_error(self) -> None:
        events = self.run_worker(address="missing")
        self.assertEqual(events[-1]["state"], "error")
        self.assertIn("No matching BrainBit", events[-1]["message"])

    def test_scan_cancellation_stops_and_destroys_scanner(self) -> None:
        self.sdk.on_start = self.stop.set
        events = self.run_worker()
        self.assertEqual(self.sdk.calls[-2:], ["scan_stop", "destroy_scanner"])
        self.assertEqual(events[-1]["state"], "closed")
        self.assertIsNone(self.sdk.sensor)

    def test_contact_check_cancellation_never_starts_eeg(self) -> None:
        self.sdk.on_command = lambda sensor, command: self.stop.set() if command == _Command.StartResist else None
        self.run_worker()
        self.assertIn("StopResist", self.sdk.calls)
        self.assertNotIn("StartSignal", self.sdk.calls)
        self.assertEqual(self.sdk.calls[-3:], ["disconnect", "destroy_sensor", "destroy_scanner"])

    def test_disconnect_in_either_mode_ends_without_reconnect(self) -> None:
        for mode in (_Command.StartResist, _Command.StartSignal):
            with self.subTest(mode=mode):
                self.setUp()

                def disconnect(sensor, command) -> None:
                    if command == mode:
                        sensor.state = _State.StateOutOfRange

                self.sdk.on_command = disconnect
                events = self.run_worker()
                self.assertIn("disconnected", [item.get("state") for item in events])
                self.assertEqual(
                    sum(isinstance(call, tuple) and call[0] == "create_sensor" for call in self.sdk.calls), 1
                )
                self.assertEqual(self.sdk.calls[-3:], ["disconnect", "destroy_sensor", "destroy_scanner"])

    def test_plain_sdk_exception_is_contained_and_scanner_destroyed(self) -> None:
        self.sdk.create_error = Exception("Code 5. Message: BLE unavailable")
        events = self.run_worker()
        self.assertIn("BLE unavailable", events[-1]["message"])
        self.assertEqual(events[-1]["state"], "error")
        self.assertEqual(self.sdk.calls[-1], "destroy_scanner")

    def test_signal_queue_overflow_is_visible_and_stops_acquisition(self) -> None:
        self.events = queue.Queue(maxsize=8)

        def burst(sensor, command) -> None:
            if command == _Command.StartSignal:
                for packet in range(20):
                    sensor.signalDataReceived(sensor, [_sample(packet)])

        self.sdk.on_command = burst
        events = self.run_worker()
        error = events[-1]
        self.assertEqual(error["state"], "error")
        self.assertIn("overflow", error["message"])
        self.assertGreaterEqual(error["discarded_samples"], 1)
        self.assertGreaterEqual(error["discarded_events"], 1)
        self.assertTrue(self.stop.is_set())
        self.assertIn("StopSignal", self.sdk.calls)
        self.assertEqual(self.events.unfinished_tasks, 0)

    def test_invalid_callback_is_reported_before_native_callback_returns(self) -> None:
        def invalid(sensor, command) -> None:
            if command == _Command.StartSignal:
                sensor.signalDataReceived(sensor, [_sample(first=math.nan)])

        self.sdk.on_command = invalid
        events = self.run_worker()
        self.assertEqual(events[-1]["state"], "error")
        self.assertIn("Non-finite EEG sample", events[-1]["message"])
        self.assertIn("StopSignal", self.sdk.calls)

    def test_stop_command_failure_still_disconnects_and_frees_resources(self) -> None:
        def fail_on_stop(sensor, command) -> None:
            if command == _Command.StartSignal:
                self.stop.set()
            elif command == _Command.StopSignal:
                raise Exception("Code 5. Message: stop failed")

        self.sdk.on_command = fail_on_stop
        events = self.run_worker()
        self.assertIn("stop failed", events[-1]["message"])
        self.assertEqual(self.sdk.calls[-3:], ["disconnect", "destroy_sensor", "destroy_scanner"])

    def test_missing_optional_sdk_is_actionable(self) -> None:
        with patch.dict("sys.modules", {"neurosdk.cmn_types": None}):
            BRAINBIT.run_brainbit(self.events, self.stop)
        event = self.events.get_nowait()
        self.assertEqual(event["state"], "error")
        self.assertIn("EEG requirements", event["message"])

    def test_unexpected_sampling_rate_fails_before_contact_or_eeg_commands(self) -> None:
        self.sdk.frequency = _Frequency.FrequencyHz500
        events = self.run_worker()
        self.assertEqual(events[-1]["state"], "error")
        self.assertIn("500 Hz", events[-1]["message"])
        self.assertIn("requires 250 Hz", events[-1]["message"])
        self.assertNotIn("StartResist", self.sdk.calls)
        self.assertNotIn("StartSignal", self.sdk.calls)
        self.assertFalse(any(event["type"] == "device" for event in events))
        self.assertEqual(self.sdk.calls[-3:], ["disconnect", "destroy_sensor", "destroy_scanner"])

    def test_unavailable_resistance_is_json_safe_and_marks_channel_names(self) -> None:
        for invalid in (math.inf, -math.inf, math.nan, 0.0, -1.0):
            with self.subTest(invalid=invalid):
                self.setUp()

                def resistance(sensor, command) -> None:
                    if command == _Command.StartResist:
                        sensor.resistDataReceived(
                            sensor, types.SimpleNamespace(O1=invalid, O2=2000.0, T3=invalid, T4=4000.0)
                        )
                    elif command == _Command.StartSignal:
                        self.stop.set()

                self.sdk.on_command = resistance
                events = self.run_worker()
                contact = next(event for event in events if event["type"] == "resistance")
                self.assertEqual(contact["values_ohm"], [None, 2000.0, None, 4000.0])
                self.assertEqual(contact["unavailable_channels"], ["O1", "T3"])
                json.dumps(events, allow_nan=False)
                self.assertEqual(events[-1]["state"], "closed")

    def test_invalid_durations_fail_before_scanning(self) -> None:
        for value in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(value=value):
                events = queue.Queue()
                BRAINBIT.run_brainbit(events, threading.Event(), scan_seconds=value)
                self.assertEqual(events.get_nowait()["state"], "error")


if __name__ == "__main__":
    unittest.main()
