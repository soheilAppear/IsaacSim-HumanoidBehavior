# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acquire four-channel BrainBit-family data without importing its SDK at startup."""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Any

__all__ = ["run_brainbit"]

_CHANNELS = ("O1", "O2", "T3", "T4")
_POLL_SECONDS = 0.1


class _EventWriter:
    """Serialize SDK callbacks and make bounded-queue loss a terminal visible error."""

    def __init__(self, events: queue.Queue, stop: threading.Event) -> None:
        self.events = events
        self.stop = stop
        self.failed = False
        self.lock = threading.Lock()

    def emit(self, event: dict) -> bool:
        """Enqueue one event, terminating acquisition if the consumer falls behind."""
        with self.lock:
            if self.failed:
                return False
            try:
                self.events.put_nowait(event)
            except queue.Full:
                self._fail_locked("Event queue overflow; acquisition stopped", rejected=event)
                return False
            return True

    def status(self, state: str, message: str) -> bool:
        """Publish an acquisition lifecycle update."""
        return self.emit({"type": "status", "state": state, "message": message})

    def fail(self, message: str) -> None:
        """Publish the first terminal error even when the event queue is full."""
        with self.lock:
            if not self.failed:
                self._fail_locked(message)

    def _fail_locked(self, message: str, rejected: dict | None = None) -> None:
        """Reserve space for the terminal error and disclose every displaced event."""
        self.failed = True
        self.stop.set()
        lost_events = int(rejected is not None)
        lost_samples = len(rejected.get("samples", [])) if rejected else 0
        while True:
            event = {
                "type": "status",
                "state": "error",
                "message": f"{message}. Discarded events: {lost_events}; EEG samples: {lost_samples}.",
                "discarded_events": lost_events,
                "discarded_samples": lost_samples,
            }
            try:
                self.events.put_nowait(event)
                return
            except queue.Full:
                try:
                    discarded = self.events.get_nowait()
                except queue.Empty:
                    continue
                self.events.task_done()
                lost_events += 1
                lost_samples += len(discarded.get("samples", []))


def _read_values(data: Any, sample_indices: tuple[int, ...] | None = None) -> list[float]:
    """Copy named BrainBit or mapped BrainBit2 samples into O1/O2/T3/T4 order."""
    if sample_indices is None:
        return [float(getattr(data, channel)) for channel in _CHANNELS]
    if len(data.Samples) != len(_CHANNELS):
        raise ValueError(f"Expected four BrainBit2 samples, received {len(data.Samples)}")
    return [float(data.Samples[index]) for index in sample_indices]


def _read_channel_layout(sensor: Any, array_samples: bool) -> tuple[tuple[int, ...] | None, list[dict]]:
    """Validate channel count and resolve BrainBit2's explicit array indices."""
    if sensor.channels_count != len(_CHANNELS):
        raise ValueError(f"Unsupported BrainBit channel count: {sensor.channels_count}; exactly four are required")
    if not array_samples:
        return None, []
    descriptors = sensor.supported_channels
    if len(descriptors) != len(_CHANNELS):
        raise ValueError("BrainBit2 must describe exactly four EEG channels")
    channel_numbers = {}
    metadata = []
    for channel in descriptors:
        identifier = channel.Id.name.removeprefix("EEGChId")
        name = channel.Name.strip().upper()
        if identifier == "Unknown":
            identifier = name
        elif name in _CHANNELS and name != identifier:
            raise ValueError(f"Conflicting BrainBit2 channel ID/name: {identifier}/{name}")
        if identifier not in _CHANNELS or identifier in channel_numbers:
            raise ValueError(f"Unsupported or duplicated BrainBit2 EEG channel: {identifier}")
        if not isinstance(channel.Num, int) or not 0 <= channel.Num < len(_CHANNELS):
            raise ValueError(f"Invalid BrainBit2 sample index: {channel.Num}")
        channel_numbers[identifier] = channel.Num
        metadata.append({"id": channel.Id.name, "name": channel.Name, "number": channel.Num})
    if set(channel_numbers.values()) != set(range(len(_CHANNELS))):
        raise ValueError("BrainBit2 channel sample indices must uniquely cover 0 through 3")
    return tuple(channel_numbers[channel] for channel in _CHANNELS), metadata


def _wait_connected(sensor: Any, connected_state: Any, stop: threading.Event, seconds: float | None) -> bool:
    """Wait interruptibly, returning false when the sensor loses its BLE connection."""
    deadline = None if seconds is None else time.monotonic() + seconds
    while not stop.is_set():
        if sensor.state != connected_state:
            return False
        delay = _POLL_SECONDS if deadline is None else min(_POLL_SECONDS, deadline - time.monotonic())
        if delay <= 0:
            break
        stop.wait(delay)
    return True


def run_brainbit(
    events: queue.Queue,
    stop: threading.Event,
    *,
    address: str | None = None,
    scan_seconds: float = 8.0,
    resistance_seconds: float = 3.0,
) -> None:
    """Scan, connect, check electrode contact, and stream BrainBit samples in a worker.

    Signal values are volts in O1/O2/T3/T4 order. Receipt timestamps describe the
    callback batch, not the device's sampling clock. The original BrainBit cannot
    measure resistance and signal simultaneously, so the contact check finishes
    before EEG starts. Connection loss ends this run without reconnecting.

    Args:
        events: Queue consumed by the application; bounded queues are supported.
        stop: Event set by the application to cancel, or by this worker on error.
        address: Exact Bluetooth address to select when multiple devices are nearby.
        scan_seconds: Finite positive discovery duration.
        resistance_seconds: Finite positive duration of the initial contact check.

    Example:

        .. code-block:: python

            events, stop = queue.Queue(maxsize=256), threading.Event()
            worker = threading.Thread(target=run_brainbit, args=(events, stop))
            worker.start()
            print(events.get())
            stop.set()
            worker.join()
    """
    writer = _EventWriter(events, stop)
    scanner = sensor = None
    scanning = False
    active_stop_command = None
    try:
        if not math.isfinite(scan_seconds) or scan_seconds <= 0:
            raise ValueError("Scan duration must be finite and positive")
        if not math.isfinite(resistance_seconds) or resistance_seconds <= 0:
            raise ValueError("Resistance duration must be finite and positive")
        if address is not None and not address.strip():
            raise ValueError("BrainBit address cannot be empty")
        if stop.is_set():
            return

        from neurosdk.cmn_types import SensorCommand, SensorFamily, SensorState
        from neurosdk.scanner import Scanner

        if not writer.status("scanning", "Scanning for four-channel BrainBit, BrainBitBlack and BrainBit2 headbands"):
            return
        scanner = Scanner([SensorFamily.LEBrainBit, SensorFamily.LEBrainBitBlack, SensorFamily.LEBrainBit2])
        scanning = True
        scanner.start()
        if stop.wait(scan_seconds):
            return
        devices = list(scanner.sensors())
        scanner.stop()
        scanning = False
        if address is not None:
            devices = [device for device in devices if device.Address.casefold() == address.strip().casefold()]
        if not devices:
            raise RuntimeError("No matching BrainBit found; check power, Bluetooth and the selected address")
        if len(devices) != 1:
            addresses = ", ".join(device.Address for device in devices)
            raise RuntimeError(f"Multiple BrainBit devices found ({addresses}); select one with --address")
        selected = devices[0]
        if not writer.status("connecting", f"Connecting to {selected.Name} ({selected.Address})"):
            return
        sensor = scanner.create_sensor(selected)
        if stop.is_set():
            return
        if sensor.state != SensorState.StateInRange:
            writer.status("disconnected", "BrainBit disconnected while connecting; start a new session to retry")
            return

        frequency_name = sensor.sampling_frequency.name
        if not frequency_name.startswith("FrequencyHz"):
            raise RuntimeError(f"Unsupported BrainBit sampling frequency: {frequency_name}")
        sample_rate_hz = int(frequency_name.removeprefix("FrequencyHz"))
        if sample_rate_hz != 250:
            raise RuntimeError(
                f"Unsupported BrainBit sampling frequency: {sample_rate_hz} Hz; this recorder requires 250 Hz"
            )
        array_samples = selected.SensFamily == SensorFamily.LEBrainBit2
        sample_indices, native_channels = _read_channel_layout(sensor, array_samples)
        if not writer.emit(
            {
                "type": "device",
                "info": {
                    "name": selected.Name,
                    "address": selected.Address,
                    "serial_number": selected.SerialNumber,
                    "family": selected.SensFamily.name,
                    "sample_rate_hz": sample_rate_hz,
                    "channels": list(_CHANNELS),
                    "native_channels": native_channels,
                    "native_sample_indices": list(sample_indices) if sample_indices is not None else None,
                    "packet_modulus": 2**32 if array_samples else 2048,
                    "signal_units": "V",
                    "resistance_units": "ohm",
                },
            }
        ):
            return

        def on_signal(_sensor: Any, data: Any) -> None:
            """Copy one SDK signal batch before returning from its native callback."""
            if stop.is_set():
                return
            received_monotonic = time.monotonic()
            received_unix_ns = time.time_ns()
            try:
                samples = [
                    {
                        "packet": int(sample.PackNum),
                        "marker": int(sample.Marker),
                        "values_v": _read_values(sample, sample_indices),
                    }
                    for sample in data
                ]
                if any(not math.isfinite(value) for sample in samples for value in sample["values_v"]):
                    raise ValueError("Non-finite EEG sample")
                if samples:
                    writer.emit(
                        {
                            "type": "samples",
                            "received_monotonic": received_monotonic,
                            "received_unix_ns": received_unix_ns,
                            "samples": samples,
                        }
                    )
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                writer.fail(f"Invalid BrainBit signal callback: {error}")

        def on_resistance(_sensor: Any, data: Any) -> None:
            """Copy resistance values, marking unmeasurable electrodes explicitly."""
            if stop.is_set():
                return
            try:
                received_monotonic = time.monotonic()
                for packet in data if array_samples else [data]:
                    measured = _read_values(packet, sample_indices)
                    values = [value if math.isfinite(value) and value > 0 else None for value in measured]
                    event = {
                        "type": "resistance",
                        "received_monotonic": received_monotonic,
                        "values_ohm": values,
                        "unavailable_channels": [channel for channel, value in zip(_CHANNELS, values) if value is None],
                    }
                    if array_samples:
                        event["packet"] = int(packet.PackNum)
                    if not writer.emit(event):
                        break
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                writer.fail(f"Invalid BrainBit resistance callback: {error}")

        sensor.signalDataReceived = on_signal
        sensor.resistDataReceived = on_resistance
        if not writer.status("contact_check", "Measuring initial electrode resistance before EEG acquisition"):
            return
        active_stop_command = SensorCommand.StopResist
        sensor.exec_command(SensorCommand.StartResist)
        connected = _wait_connected(sensor, SensorState.StateInRange, stop, resistance_seconds)
        if stop.is_set() or not connected:
            if not connected:
                writer.status("disconnected", "BrainBit disconnected during contact check; no reconnect attempted")
            return
        sensor.exec_command(SensorCommand.StopResist)
        active_stop_command = None
        sensor.resistDataReceived = None
        if stop.is_set() or not writer.status("streaming", "Streaming four EEG channels in volts"):
            return
        active_stop_command = SensorCommand.StopSignal
        sensor.exec_command(SensorCommand.StartSignal)
        if not _wait_connected(sensor, SensorState.StateInRange, stop, None):
            writer.status("disconnected", "BrainBit disconnected; start a new session to reconnect")
    except ImportError as error:
        writer.fail(f"BrainBit SDK unavailable; install the project EEG requirements in its environment: {error}")
    except Exception as error:
        # NeuroSDK 2's `raise_exception_if()` raises plain `Exception`, with no
        # vendor-specific subclass. Contain that native boundary in this worker.
        writer.fail(f"BrainBit acquisition failed: {error}")
    finally:
        if sensor is not None:
            if active_stop_command is not None:
                try:
                    sensor.exec_command(active_stop_command)
                except Exception as error:
                    writer.fail(f"BrainBit stop command failed during cleanup: {error}")
            sensor.signalDataReceived = None
            sensor.resistDataReceived = None
            try:
                sensor.disconnect()
            except Exception as error:
                writer.fail(f"BrainBit disconnect failed during cleanup: {error}")
            # Native callback registrations retain Python objects. NeuroSDK 2
            # documents explicit destruction; its destructors are idempotent.
            try:
                sensor.__del__()
            except Exception as error:
                writer.fail(f"BrainBit resource release failed: {error}")
            del sensor
        if scanner is not None:
            if scanning:
                try:
                    scanner.stop()
                except Exception as error:
                    writer.fail(f"BrainBit scanner stop failed during cleanup: {error}")
            try:
                scanner.__del__()
            except Exception as error:
                writer.fail(f"BrainBit scanner release failed: {error}")
            del scanner
        writer.status("closed", "BrainBit acquisition closed")
