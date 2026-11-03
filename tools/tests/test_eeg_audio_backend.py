# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the audio subprocess boundary without playing speech or changing devices."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "eeg" / "audio.py"
SPEC = importlib.util.spec_from_file_location("eeg_audio_backend_under_test", MODULE_PATH)
AUDIO = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIO
SPEC.loader.exec_module(AUDIO)


class TestEegAudioBackend(unittest.TestCase):
    def setUp(self) -> None:
        self.executable = patch.object(AUDIO, "_powershell_executable", return_value=Path("powershell.exe"))
        self.executable.start()
        self.addCleanup(self.executable.stop)
        self.process = patch.object(AUDIO.subprocess, "run")
        self.run = self.process.start()
        self.addCleanup(self.process.stop)

    def respond(self, body: object, code: int = 0) -> None:
        self.run.return_value = subprocess.CompletedProcess([], code, json.dumps(body).encode("utf-8"), b"")

    def speech_response(self, **overrides) -> dict:
        response = {
            "ok": True,
            "selected_device": {"id": "token-1", "name": "VR speakers"},
            "volume": 60,
            "host_play_requested_utc": "2026-11-03T04:00:00.1234567+00:00",
            "host_play_completed_utc": "2026-11-03T04:00:01.2234567+00:00",
        }
        response.update(overrides)
        return response

    def test_list_preserves_exact_device_ids_and_unicode_names(self) -> None:
        devices = [{"id": r"HKEY_LOCAL_MACHINE\audio\one", "name": "耳機"}]
        self.respond({"ok": True, "outputs": devices})
        self.assertEqual(AUDIO.list_audio_outputs(), devices)
        self.assertEqual(json.loads(self.run.call_args.kwargs["input"]), {"action": "list"})

    def test_empty_device_list_is_valid(self) -> None:
        self.respond({"ok": True, "outputs": []})
        self.assertEqual(AUDIO.list_audio_outputs(), [])

    def test_text_and_device_are_json_data_and_never_shell_arguments(self) -> None:
        text = '<speak>Quotes " and `backticks`; $(Write-Output danger) & |\nשלום</speak>'
        device = "token-1; $(Write-Output danger)"
        self.respond(self.speech_response(selected_device={"id": device, "name": "VR speakers"}))
        result = AUDIO.speak(text, device)
        arguments = self.run.call_args.args[0]
        options = self.run.call_args.kwargs
        self.assertNotIn(text, arguments)
        self.assertNotIn(device, arguments)
        self.assertNotIn("-Command", arguments)
        self.assertEqual(arguments[-2], "-File")
        self.assertEqual(Path(arguments[-1]).name, "speak.ps1")
        self.assertIs(options["shell"], False)
        self.assertEqual(options["timeout"], 90.0)
        self.assertEqual(
            json.loads(options["input"]),
            {
                "action": "speak",
                "text": text,
                "device_id": device,
                "volume": 60,
            },
        )
        self.assertEqual(result["selected_device"]["id"], device)
        self.assertIn("no measured hardware onset", result["timing_basis"])

    def test_default_output_is_requested_explicitly_as_null(self) -> None:
        self.respond(self.speech_response())
        AUDIO.speak("Please stay still")
        self.assertIsNone(json.loads(self.run.call_args.kwargs["input"])["device_id"])

    def test_custom_volume_and_timeout_are_sent_without_global_settings(self) -> None:
        self.respond(self.speech_response(volume=40))
        result = AUDIO.speak("Start", volume=40, timeout=15)
        self.assertEqual(json.loads(self.run.call_args.kwargs["input"])["volume"], 40)
        self.assertEqual(self.run.call_args.kwargs["timeout"], 15.0)
        self.assertEqual(result["volume"], 40)

    def test_backend_device_failure_is_explicit_with_no_retry(self) -> None:
        self.respond({"ok": False, "error": "Requested output is unavailable"}, code=1)
        with self.assertRaisesRegex(AUDIO.AudioBackendError, "unavailable"):
            AUDIO.speak("Start", "missing-token")
        self.run.assert_called_once()

    def test_wrong_device_cannot_be_reported_as_success(self) -> None:
        self.respond(self.speech_response())
        with self.assertRaisesRegex(AUDIO.AudioBackendError, "exact requested"):
            AUDIO.speak("Start", "different-token")

    def test_wrong_volume_cannot_be_reported_as_success(self) -> None:
        self.respond(self.speech_response(volume=100))
        with self.assertRaisesRegex(AUDIO.AudioBackendError, "voice volume"):
            AUDIO.speak("Start")

    def test_timeout_is_not_reported_as_playback_completion(self) -> None:
        self.run.side_effect = subprocess.TimeoutExpired(["powershell.exe"], 10)
        with self.assertRaisesRegex(AUDIO.AudioBackendError, "completion is unconfirmed"):
            AUDIO.speak("Start", timeout=10)

    def test_launch_error_becomes_backend_error(self) -> None:
        self.run.side_effect = OSError("Access denied")
        with self.assertRaisesRegex(AUDIO.AudioBackendError, "Could not start"):
            AUDIO.list_audio_outputs()

    def test_invalid_json_and_utf8_are_explicit_errors(self) -> None:
        for payload in (b"", b"not json", b"\xff"):
            with self.subTest(payload=payload):
                self.run.return_value = subprocess.CompletedProcess([], 1, payload, b"backend failed")
                with self.assertRaisesRegex(AUDIO.AudioBackendError, "Invalid audio backend response"):
                    AUDIO.list_audio_outputs()

    def test_malformed_output_lists_are_rejected(self) -> None:
        for outputs in (
            None,
            {},
            [{"id": "a"}],
            [{"id": 1, "name": "bad"}],
            [{"id": "a", "name": "one"}, {"id": "a", "name": "two"}],
        ):
            with self.subTest(outputs=outputs):
                self.respond({"ok": True, "outputs": outputs})
                with self.assertRaises(AUDIO.AudioBackendError):
                    AUDIO.list_audio_outputs()

    def test_bad_success_envelopes_and_nonzero_status_are_rejected(self) -> None:
        for body, code in (([], 0), ({"ok": "true", "outputs": []}, 0), ({"ok": True, "outputs": []}, 1)):
            with self.subTest(body=body, code=code):
                self.respond(body, code)
                with self.assertRaises(AUDIO.AudioBackendError):
                    AUDIO.list_audio_outputs()

    def test_invalid_or_reversed_timestamps_are_not_completed_cues(self) -> None:
        for overrides in (
            {"host_play_completed_utc": None},
            {"host_play_completed_utc": "not a timestamp"},
            {"host_play_completed_utc": "2026-11-03T04:00:01"},
            {"host_play_completed_utc": "2026-11-03T04:00:01-05:00"},
            {"host_play_completed_utc": "2026-11-03T03:59:59Z"},
        ):
            with self.subTest(overrides=overrides):
                self.respond(self.speech_response(**overrides))
                with self.assertRaises(AUDIO.AudioBackendError):
                    AUDIO.speak("Start")

    def test_invalid_input_never_starts_backend(self) -> None:
        for text in (None, "", " \n", "a\0b", "a" * 4001):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    AUDIO.speak(text)
        for device in ("", 1, "a\0b"):
            with self.subTest(device=device):
                with self.assertRaises(ValueError):
                    AUDIO.speak("Start", device)
        for volume in (-1, 101, True, 50.0):
            with self.subTest(volume=volume):
                with self.assertRaises(ValueError):
                    AUDIO.speak("Start", volume=volume)
        for timeout in (0, -1, 601, float("nan"), True):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    AUDIO.list_audio_outputs(timeout=timeout)
        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
