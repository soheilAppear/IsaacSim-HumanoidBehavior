# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load production controllers with Kit startup imports isolated for offline tests.

Only module-level import nodes are omitted. All class bodies, helpers, and method
implementations come unchanged from the checkout. USD/Gf/NumPy/PyTorch are real;
the app, XR devices, and physics tensor views are supplied by individual tests.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
import types
from collections import deque
from dataclasses import dataclass
from numbers import Real
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
        os=os,
        time=time,
        Path=Path,
        carb=CARB,
        omni=OMNI,
        BaseSample=BaseSample,
        import_module=importlib.import_module,
        dataclass=dataclass,
        deque=deque,
        SimpleNamespace=types.SimpleNamespace,
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
# Pure Python transport imports normally: unlike Kit imports these need no
# replacement, and their dataclasses retain the module's actual globals.
_webcam_spec = importlib.util.spec_from_file_location(
    "humanoid_test_webcam_tracking", PACKAGE / "interactive/humanoid/webcam_tracking.py"
)
WEBCAM = importlib.util.module_from_spec(_webcam_spec)
sys.modules[_webcam_spec.name] = WEBCAM
_webcam_spec.loader.exec_module(WEBCAM)
_gesture_spec = importlib.util.spec_from_file_location(
    "humanoid_test_locomotion_gesture", PACKAGE / "interactive/humanoid/locomotion_gesture.py"
)
GESTURE = importlib.util.module_from_spec(_gesture_spec)
sys.modules[_gesture_spec.name] = GESTURE
_gesture_spec.loader.exec_module(GESTURE)
_navigation_spec = importlib.util.spec_from_file_location(
    "humanoid_test_point_gaze_navigation", PACKAGE / "interactive/humanoid/point_gaze_navigation.py"
)
NAVIGATION = importlib.util.module_from_spec(_navigation_spec)
sys.modules[_navigation_spec.name] = NAVIGATION
_navigation_spec.loader.exec_module(NAVIGATION)
NAVIGATION_SCENE = load_source("interactive/humanoid/point_navigation_scene.py")
_factory_spec = importlib.util.spec_from_file_location(
    "humanoid_test_factory_route", PACKAGE / "interactive/humanoid/factory_route.py"
)
FACTORY_ROUTE = importlib.util.module_from_spec(_factory_spec)
sys.modules[_factory_spec.name] = FACTORY_ROUTE
_factory_spec.loader.exec_module(FACTORY_ROUTE)
POINT_RAY = load_source("interactive/humanoid/point_ray.py", Real=Real)
OPERATOR_HEAD = load_source("interactive/humanoid/operator_head.py")
COMFORT_CAMERA = load_source("interactive/humanoid/comfort_camera.py", Real=Real)
CONTROLLER_LOCOMOTION = load_source("interactive/humanoid/controller_locomotion.py", Real=Real)
GAZE_LOCOMOTION = load_source("interactive/humanoid/gaze_locomotion.py", Real=Real)
GAZE_CONTROLS_SCENE = load_source("interactive/humanoid/gaze_controls_scene.py")
_bimanual_spec = importlib.util.spec_from_file_location(
    "humanoid_test_bimanual_grasp", PACKAGE / "interactive/humanoid/bimanual_grasp.py"
)
BIMANUAL_GRASP = importlib.util.module_from_spec(_bimanual_spec)
sys.modules[_bimanual_spec.name] = BIMANUAL_GRASP
_bimanual_spec.loader.exec_module(BIMANUAL_GRASP)
TALL_PACKAGES = load_source("interactive/humanoid/tall_packages.py")
PACKAGE_SHELVES = load_source("interactive/humanoid/package_shelves.py")
_csv_writer_spec = importlib.util.spec_from_file_location(
    "humanoid_test_behavioral_csv_writer", PACKAGE / "interactive/humanoid/behavioral_csv_writer.py"
)
CSV_WRITER = importlib.util.module_from_spec(_csv_writer_spec)
sys.modules[_csv_writer_spec.name] = CSV_WRITER
_csv_writer_spec.loader.exec_module(CSV_WRITER)
GAZE = load_source(
    "interactive/humanoid/eye_gaze_tracker.py",
    read_world_pose=POSE.read_world_pose,
    MaterialHighlights=HIGHLIGHTS.MaterialHighlights,
)
HUMANOID = load_source(
    "interactive/humanoid/humanoid_example.py",
    PointRayFilter=POINT_RAY.PointRayFilter,
    RobotHeadVisual=OPERATOR_HEAD.RobotHeadVisual,
    compute_level_head_reference=OPERATOR_HEAD.compute_level_head_reference,
    ControllerLocomotion=CONTROLLER_LOCOMOTION.ControllerLocomotion,
    GazeLocomotion=GAZE_LOCOMOTION.GazeLocomotion,
    GazeLocomotionConfig=GAZE_LOCOMOTION.GazeLocomotionConfig,
    GazeControlsScene=GAZE_CONTROLS_SCENE.GazeControlsScene,
    BimanualGraspMonitor=BIMANUAL_GRASP.BimanualGraspMonitor,
    ComfortCameraFilter=COMFORT_CAMERA.ComfortCameraFilter,
    create_tall_packages=TALL_PACKAGES.create_tall_packages,
    create_shelf_packages=PACKAGE_SHELVES.create_shelf_packages,
    BehavioralCsvWriter=CSV_WRITER.BehavioralCsvWriter,
    CsvBatch=CSV_WRITER.CsvBatch,
    CsvWriteResult=CSV_WRITER.CsvWriteResult,
    json=json,
    read_world_pose=POSE.read_world_pose,
    smoothing_alpha=POSE.smoothing_alpha,
    MaterialHighlights=HIGHLIGHTS.MaterialHighlights,
    ContactGraspController=GRASP.ContactGraspController,
    find_articulation_fault=HEALTH.find_articulation_fault,
    project_contact_step=ARM_CONTACT.project_contact_step,
    WebcamCalibration=WEBCAM.WebcamCalibration,
    WebcamReceiver=WEBCAM.WebcamReceiver,
    LocomotionGesture=GESTURE.LocomotionGesture,
    map_gesture_to_velocity=GESTURE.map_gesture_to_velocity,
    PointNavigation=NAVIGATION.PointNavigation,
    FactoryRoutePlanner=FACTORY_ROUTE.FactoryRoutePlanner,
    is_pointing=NAVIGATION.is_pointing,
    PointNavigationScene=NAVIGATION_SCENE.PointNavigationScene,
)
DECOUPLED = load_source("robots/g1_decoupled_policy.py", hashlib=hashlib, json=json)
FOOT_CONTACTS = load_source("robots/g1_foot_contacts.py")
G1 = load_source(
    "robots/g1.py",
    get_physics_simulation_interface=Mock(),
    G1DecoupledPolicy=DECOUPLED.G1DecoupledPolicy,
    configure_decoupled_foot_contacts=FOOT_CONTACTS.configure_decoupled_foot_contacts,
)


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
