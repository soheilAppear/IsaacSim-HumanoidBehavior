# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run humanoid control regressions without starting Kit or connecting a headset.

Use Isaac Sim's python.bat/python.sh. The runner imports that installation's USD,
NumPy, and PyTorch libraries; simulator services and XR inputs are replaced by test
doubles. This validates controller behavior and real USD joint/material authoring,
but does not validate hardware tracking quality or a dynamically balanced gait.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import unittest
from pathlib import Path


def main() -> int:
    """Configure the bundled dependencies and run the standalone regression suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-sim-dir", type=Path, default=os.environ.get("ISAAC_SIM_DIR"))
    args = parser.parse_args()
    install = args.isaac_sim_dir or Path(sys.executable).resolve().parents[2]
    dll_handles = []
    for usd in sorted((install / "extscache").glob("omni.usd.libs-*"), reverse=True):
        if (usd / "pxr").is_dir():
            sys.path.insert(0, str(usd))
            if os.name == "nt":
                dll_handles.append(os.add_dll_directory(str(usd / "bin")))
            break
    schema_plugins = []
    for schema in sorted((install / "extscache").glob("omni.usd.schema.physx-*"), reverse=True):
        if (schema / "pxr/PhysxSchema").is_dir():
            sys.path.insert(0, str(schema))
            if os.name == "nt":
                dll_handles.append(os.add_dll_directory(str(schema / "bin")))
            schema_plugins = list((schema / "plugins").glob("*/resources/plugInfo.json"))
            break
    for bundle in (
        install / "extsDeprecated/omni.isaac.ml_archive/pip_prebundle",
        install / "exts/omni.isaac.ml_archive/pip_prebundle",
    ):
        if bundle.is_dir():
            sys.path.append(str(bundle))
    try:
        for dependency in ("numpy", "torch", "pxr.Gf", "pxr.Usd", "pxr.UsdPhysics", "pxr.PhysxSchema"):
            importlib.import_module(dependency)
        registry = importlib.import_module("pxr.Plug").Registry()
        for plugin in schema_plugins:
            registry.RegisterPlugins(str(plugin))
    except ImportError as error:
        parser.exit(
            2, f"Missing Isaac Sim test dependency: {error}\nRun this script with Isaac Sim's Python wrapper.\n"
        )
    tests = Path(__file__).resolve().parent / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests), pattern="test_*control*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
