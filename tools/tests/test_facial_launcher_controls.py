# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise both facial/gaze Windows wrappers without starting Isaac Sim."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "nt", "The launcher requires Windows cmd.exe")
class TestFacialLauncherControls(unittest.TestCase):
    """Verify motion opt-in across wrappers and preservation of existing controls."""

    def _run_launcher(
        self, *args: str, overrides: dict[str, str] | None = None, exit_code: int = 0
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, str], str]:
        """Run the actual wrapper chain against a base-launcher reporting stub."""
        env = {key: value for key, value in os.environ.items() if not key.upper().startswith("G1_")}
        env.update(overrides or {})
        env["FACE_STUB_EXIT_CODE"] = str(exit_code)
        with tempfile.TemporaryDirectory(prefix="facial launcher ") as directory:
            root = Path(directory)
            for name in ("launch_facial_walking.bat", "launch_gaze_walking.bat"):
                shutil.copyfile(TOOLS / name, root / name)
            (root / "launch_isaac_vr.bat").write_text(
                "@echo off\nset G1_\necho FACE_STUB_ARGS=%*\nexit /b %FACE_STUB_EXIT_CODE%\n", encoding="ascii"
            )
            launcher = root / "launch_facial_walking.bat"
            # Only fixed test fixtures enter this command. /s encloses the full
            # invocation, with inner quotes protecting the launcher path.
            tail = " " + subprocess.list2cmdline(args) if args else ""
            command = f'"{env.get("COMSPEC", "cmd.exe")}" /d /s /c ""{launcher}"{tail}"'
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10, shell=False)
        values = {}
        forwarded = ""
        for line in result.stdout.splitlines():
            name, separator, value = line.partition("=")
            if separator and name.startswith("G1_"):
                values[name] = value
            elif name == "FACE_STUB_ARGS":
                forwarded = value
        return result, values, forwarded

    def test_default_preview_overrides_inherited_motion(self) -> None:
        result, values, forwarded = self._run_launcher(
            overrides={"G1_GAZE_PREVIEW": "0", "G1_FACE_PREVIEW": "0", "G1_FACE_ENABLED": "0"}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(forwarded, "")
        self.assertEqual(values["G1_FACE_ENABLED"], "1")
        self.assertEqual(values["G1_FACE_MODE"], "hold")
        self.assertEqual(values["G1_FACE_PREVIEW"], "1")
        self.assertEqual(values["G1_GAZE_PREVIEW"], "1")
        self.assertEqual(values["G1_GAZE_MODE"], "hold")
        self.assertEqual(values["G1_GAZE_DWELL_S"], "0")
        self.assertEqual(values["G1_LOCOMOTION_INPUT"], "gaze")
        self.assertEqual(values["G1_OPTICAL_HANDS_ONLY"], "0")
        self.assertEqual(values["G1_GAZE_CONTROLLER_OVERRIDE"], "1")

    def test_first_move_enables_both_profiles_and_preserves_quoted_arguments(self) -> None:
        result, values, forwarded = self._run_launcher("--MOVE", "--xr-verbose", "--/app/title=two words")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["G1_FACE_PREVIEW"], "0")
        self.assertEqual(values["G1_GAZE_PREVIEW"], "0")
        self.assertEqual(forwarded, '--xr-verbose "--/app/title=two words"')

    def test_later_move_does_not_enable_either_profile(self) -> None:
        for args, expected in (
            (("--preview", "--move"), "--move"),
            (("--xr-verbose", "--move"), "--xr-verbose --move"),
        ):
            with self.subTest(args=args):
                result, values, forwarded = self._run_launcher(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(values["G1_FACE_PREVIEW"], "1")
                self.assertEqual(values["G1_GAZE_PREVIEW"], "1")
                self.assertEqual(forwarded, expected)

    def test_face_defaults_and_existing_gaze_speeds_are_independent(self) -> None:
        result, values, _ = self._run_launcher("--move")
        self.assertEqual(result.returncode, 0, result.stderr)
        for key, value in {
            "G1_FACE_OSC_PORT": "9015",
            "G1_FACE_DISTANCE_M": "0.15",
            "G1_FACE_TURN_DEG": "10",
            "G1_FACE_FORWARD_SPEED": "0.20",
            "G1_FACE_BACKWARD_SPEED": "0.15",
            "G1_FACE_LATERAL_SPEED": "0.15",
            "G1_FACE_YAW_SPEED": "0.25",
            "G1_FACE_WINK_THRESHOLD": "0.55",
            "G1_FACE_JAW_THRESHOLD": "0.45",
            "G1_FACE_PUCKER_THRESHOLD": "0.40",
            "G1_FACE_MOUTH_SIDE_THRESHOLD": "0.65",
            "G1_FACE_RELEASE_THRESHOLD": "0.35",
            "G1_GAZE_FORWARD_SPEED": "0.60",
            "G1_GAZE_BACKWARD_SPEED": "0.35",
            "G1_GAZE_YAW_SPEED": "0.40",
        }.items():
            self.assertEqual(values[key], value)

    def test_explicit_tuning_and_hand_settings_survive(self) -> None:
        overrides = {
            "G1_FACE_OSC_PORT": "9020",
            "G1_FACE_MODE": "step",
            "G1_FACE_DISTANCE_M": "0.10",
            "G1_FACE_TURN_DEG": "5",
            "G1_FACE_FORWARD_SPEED": "0.12",
            "G1_FACE_BACKWARD_SPEED": "0",
            "G1_FACE_LATERAL_SPEED": "0.10",
            "G1_FACE_YAW_SPEED": "0.15",
            "G1_FACE_WINK_THRESHOLD": "0.70",
            "G1_FACE_JAW_THRESHOLD": "0.60",
            "G1_FACE_PUCKER_THRESHOLD": "0.60",
            "G1_FACE_MOUTH_SIDE_THRESHOLD": "0.70",
            "G1_FACE_RELEASE_THRESHOLD": "0.25",
            "G1_GAZE_MODE": "step",
            "G1_OPTICAL_HANDS_ONLY": "1",
            "G1_GAZE_CONTROLLER_OVERRIDE": "0",
        }
        result, values, _ = self._run_launcher("--move", overrides=overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        for key, value in overrides.items():
            self.assertEqual(values[key], value)

    def test_base_exit_code_reaches_caller(self) -> None:
        result, _, _ = self._run_launcher("--move", exit_code=23)
        self.assertEqual(result.returncode, 23, result.stderr)


if __name__ == "__main__":
    unittest.main()
