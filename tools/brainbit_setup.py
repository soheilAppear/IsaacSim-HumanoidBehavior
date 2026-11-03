# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install the BrainBit preview dependencies in a project-local environment."""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Create the isolated environment without opening Bluetooth or Isaac Sim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12)) or struct.calcsize("P") != 8:
        parser.error("Use normal 64-bit Python 3.10–3.12, for example: py -3.11 tools/brainbit_setup.py")
    executable = str(Path(sys.executable).resolve()).lower().replace("\\", "/")
    if "isaac-sim" in executable or "/kit/python" in executable:
        parser.error("Use a normal Python installation, not Isaac Sim's bundled Python")
    environment = ROOT / ".eeg" / "venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    try:
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(environment)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(ROOT / "tools" / "brainbit_requirements.txt"),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"BrainBit setup failed: {error}", file=sys.stderr)
        return 1
    print("Setup complete. Demo: tools\\launch_brainbit_teleop.bat --source synthetic")
    print("Live headset: tools\\launch_brainbit_teleop.bat --source brainbit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
