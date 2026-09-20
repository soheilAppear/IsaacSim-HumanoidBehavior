# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a project-local webcam environment and fetch official pretrained models.

Run with a normal 64-bit Python installation, never Isaac Sim's bundled Python.
This explicit setup command downloads packages and model assets; normal tracking
reuses them. It does not start a camera, change Isaac packages, or record images.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Install the pinned tracker into its own venv and download Tasks bundles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-model", choices=("lite", "full"), default="lite")
    parser.add_argument("--skip-hands", action="store_true", help="Download only the pose model")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 12) or platform.architecture()[0] != "64bit":
        # Windows can associate `python`/`py` with a newer unsupported default.
        # Ask its launcher for a compatible installed interpreter before failing.
        launcher = shutil.which("py") if sys.platform == "win32" else None
        if launcher:
            for version in ("3.12", "3.11", "3.10"):
                probe = subprocess.run(
                    [launcher, f"-{version}", "-c", "import struct; print(struct.calcsize('P') * 8)"],
                    capture_output=True,
                    text=True,
                )
                if probe.returncode == 0 and probe.stdout.strip() == "64":
                    return subprocess.run(
                        [launcher, f"-{version}", str(Path(__file__).resolve()), *sys.argv[1:]]
                    ).returncode
        parser.error("Install a normal 64-bit Python 3.10, 3.11, or 3.12, then run this setup again")
    executable = str(Path(sys.executable).resolve()).lower()
    if "isaac-sim" in executable or "kit/python" in executable.replace("\\", "/"):
        parser.error("Use the Windows Python launcher (py tools/webcam_setup.py), not Isaac Sim's Python")
    environment = ROOT / ".webcam" / "venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    try:
        if not python.is_file():
            print(f"Creating isolated webcam Python at {environment}")
            venv.EnvBuilder(with_pip=True).create(environment)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(ROOT / "tools" / "webcam_requirements.txt"),
            ],
            check=True,
        )
        command = [
            str(python),
            str(ROOT / "tools" / "webcam_tracking.py"),
            "--download-only",
            "--pose-model",
            args.pose_model,
        ]
        if not args.skip_hands:
            command.append("--hands")
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Webcam setup failed: {error}", file=sys.stderr)
        return 1
    print("Webcam setup complete. Run tools\\launch_webcam_tracking.bat to open the preview.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
