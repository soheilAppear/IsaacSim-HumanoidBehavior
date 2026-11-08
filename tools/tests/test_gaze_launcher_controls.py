# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Windows gaze launcher against a stub, without starting Isaac."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "launch_gaze_walking.bat"


@unittest.skipUnless(os.name == "nt", "The launcher requires Windows cmd.exe")
class TestGazeLauncherControls(unittest.TestCase):
    """Check mode opt-in, argument forwarding, environment isolation and exit status."""

    def _run_launcher(
        self, *args: str, overrides: dict[str, str] | None = None, exit_code: int = 0
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, str], str]:
        """Copy only the wrapper beside a reporting stub in a path with spaces."""
        env = {key: value for key, value in os.environ.items() if not key.upper().startswith("G1_")}
        env.update(overrides or {})
        env["GAZE_STUB_EXIT_CODE"] = str(exit_code)
        with tempfile.TemporaryDirectory(prefix="gaze launcher ") as directory:
            launcher = Path(directory) / LAUNCHER.name
            shutil.copyfile(LAUNCHER, launcher)
            (Path(directory) / "launch_isaac_vr.bat").write_text(
                "@echo off\nset G1_\necho GAZE_STUB_ARGS=%*\nexit /b %GAZE_STUB_EXIT_CODE%\n",
                encoding="ascii",
            )
            # These arguments are fixed test fixtures, never external/user input.
            # Outer /s quotes enclose the full command; inner quotes protect its path.
            tail = " " + subprocess.list2cmdline(args) if args else ""
            command = f'"{env.get("COMSPEC", "cmd.exe")}" /d /s /c ""{launcher}"{tail}"'
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10, shell=False)
        values = {}
        forwarded = ""
        for line in result.stdout.splitlines():
            name, separator, value = line.partition("=")
            if separator and name.startswith("G1_"):
                values[name] = value
            elif name == "GAZE_STUB_ARGS":
                forwarded = value
        return result, values, forwarded

    def test_default_overrides_inherited_motion_and_hand_modes(self) -> None:
        result, values, forwarded = self._run_launcher(
            overrides={
                "G1_GAZE_PREVIEW": "0",
                "G1_LOCOMOTION_INPUT": "controller",
                "G1_OPTICAL_HANDS_ONLY": "0",
                "G1_POINTING_NAVIGATION": "1",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(forwarded, "")
        self.assertEqual(values["G1_GAZE_PREVIEW"], "1")
        self.assertEqual(values["G1_GESTURE_WALKING"], "1")
        self.assertEqual(values["G1_LOCOMOTION_INPUT"], "gaze")
        self.assertEqual(values["G1_OPTICAL_HANDS_ONLY"], "1")
        self.assertEqual(values["G1_POINTING_NAVIGATION"], "0")

    def test_first_move_enables_motion_and_preserves_quoted_remaining_arguments(self) -> None:
        result, values, forwarded = self._run_launcher("--move", "--xr-verbose", "--/app/title=two words")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["G1_GAZE_PREVIEW"], "0")
        self.assertEqual(forwarded, '--xr-verbose "--/app/title=two words"')

    def test_explicit_preview_is_case_insensitive_and_only_first_flag_is_consumed(self) -> None:
        result, values, forwarded = self._run_launcher("--PREVIEW", "--move", overrides={"G1_GAZE_PREVIEW": "0"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["G1_GAZE_PREVIEW"], "1")
        self.assertEqual(forwarded, "--move")

    def test_later_move_flag_cannot_enable_motion(self) -> None:
        result, values, forwarded = self._run_launcher("--xr-verbose", "--move")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["G1_GAZE_PREVIEW"], "1")
        self.assertEqual(forwarded, "--xr-verbose --move")

    def test_standalone_move_and_preview_flags_leave_no_forwarded_arguments(self) -> None:
        for flag, preview in (("--MOVE", "0"), ("--preview", "1")):
            with self.subTest(flag=flag):
                result, values, forwarded = self._run_launcher(flag)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(values["G1_GAZE_PREVIEW"], preview)
                self.assertEqual(forwarded, "")

    def test_numerical_defaults_and_explicit_override_reach_base_launcher(self) -> None:
        result, values, _ = self._run_launcher(overrides={"G1_GAZE_BACKWARD_SPEED": "0", "G1_GAZE_DWELL_S": "1.2"})
        self.assertEqual(result.returncode, 0, result.stderr)
        for key, value in {
            "G1_GAZE_DWELL_S": "1.2",
            "G1_GAZE_DISTANCE_M": "0.20",
            "G1_GAZE_TURN_DEG": "10",
            "G1_GAZE_FORWARD_SPEED": "0.20",
            "G1_GAZE_BACKWARD_SPEED": "0",
            "G1_GAZE_YAW_SPEED": "0.25",
        }.items():
            self.assertEqual(values[key], value)

    def test_base_launcher_failure_is_returned(self) -> None:
        result, _, _ = self._run_launcher("--move", exit_code=23)
        self.assertEqual(result.returncode, 23, result.stderr)


if __name__ == "__main__":
    unittest.main()
