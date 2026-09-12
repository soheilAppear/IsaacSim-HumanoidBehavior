# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load production controllers with Kit startup imports isolated for offline tests.

Only module-level import nodes are omitted. All class bodies, helpers, and method
implementations come unchanged from the checkout. USD/Gf/NumPy/PyTorch are real;
the app, XR devices, and physics tensor views are supplied by individual tests.
"""

from __future__ import annotations

import ast
import importlib
import math
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples"


class Array:
    def __init__(self, data):
        self.tensor = torch.as_tensor(data)

    def numpy(self):
        return self.tensor.numpy()


def install_module(name):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    sys.modules[name] = module
    if "." in name:
        parent, attribute = name.rsplit(".", 1)
        setattr(install_module(parent), attribute, module)
    return module


CARB = install_module("carb")
CARB.log_info = Mock()
CARB.log_warn = Mock()
CARB.log_error = Mock()
CARB.input = types.SimpleNamespace(KeyboardEventType=types.SimpleNamespace(KEY_PRESS=1, KEY_RELEASE=2, KEY_REPEAT=3))
OMNI = install_module("omni")
USD = install_module("omni.usd")
USD.get_context = Mock()
PHYSICS = install_module("omni.physics.core")
PHYSICS.get_physics_scene_query_interface = Mock()
WARP = install_module("warp")
WARP.to_torch = lambda value: value.tensor if isinstance(value, Array) else value
WARP.from_torch = lambda value: Array(value)


class BaseSample:
    def __init__(self):
        self._world_settings = {}


def load_source(relative_path, **extra):
    source = PACKAGE / relative_path
    module = types.ModuleType("humanoid_test_" + source.stem)
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    module.__dict__.update(
        Gf=Gf,
        Sdf=Sdf,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdLux=UsdLux,
        UsdPhysics=UsdPhysics,
        PhysxSchema=PhysxSchema,
        UsdShade=UsdShade,
        np=np,
        math=math,
        random=random,
        time=time,
        Path=Path,
        carb=CARB,
        omni=OMNI,
        BaseSample=BaseSample,
        import_module=importlib.import_module,
        dataclass=dataclass,
        **extra,
    )
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)
    return module


POSE = load_source("interactive/humanoid/xr_pose.py")
HIGHLIGHTS = load_source("interactive/humanoid/material_highlights.py")
GRASP = load_source("interactive/humanoid/grasp_controller.py")
HEALTH = load_source("interactive/humanoid/articulation_health.py")
ARM_CONTACT = load_source("interactive/humanoid/arm_contact.py")
GAZE = load_source(
    "interactive/humanoid/eye_gaze_tracker.py",
    read_world_pose=POSE.read_world_pose,
    MaterialHighlights=HIGHLIGHTS.MaterialHighlights,
)
HUMANOID = load_source(
    "interactive/humanoid/humanoid_example.py",
    read_world_pose=POSE.read_world_pose,
    smoothing_alpha=POSE.smoothing_alpha,
    MaterialHighlights=HIGHLIGHTS.MaterialHighlights,
    ContactGraspController=GRASP.ContactGraspController,
    find_articulation_fault=HEALTH.find_articulation_fault,
    project_contact_step=ARM_CONTACT.project_contact_step,
)
G1 = load_source("robots/g1.py", get_physics_simulation_interface=Mock())


def pose(position=(0, 0, 0), yaw=0):
    matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw))
    matrix.SetTranslateOnly(Gf.Vec3d(*position))
    return matrix


class Device:
    def __init__(self, matrix, flags=3, source="controller"):
        self.matrix = matrix
        self.flags = flags
        self.source = source

    def get_virtual_world_pose_desc(self, name=""):
        return types.SimpleNamespace(pose_matrix=self.matrix, validity_flags=self.flags)

    def get_pose_names(self):
        return ["grip", "palm", "gaze_ext"]

    def get_hand_tracking_data_source(self):
        return self.source
