# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate XR poses at the boundary between tracking space and simulation space."""

from __future__ import annotations

import math

from pxr import Gf


def read_world_pose(device: object, pose_name: str = "") -> Gf.Matrix4d | None:
    """Read a finite, valid pose in the virtual stage frame.

    Use the runtime validity flags when available. Older matrix-only wrappers use
    identity as their untracked sentinel. An explicitly valid identity pose is legal.
    Physical-room poses must never be substituted for a missing world pose; the XR rig
    may have translated, rotated, or converted the room's up axis.

    Args:
        device: XR input device, or None if the runtime did not expose one.
        pose_name: Named pose on that device.

    Returns:
        World pose, or None when missing, invalid, or malformed.

    Example:

    .. code-block:: python

        >>> read_world_pose(None) is None
        True
    """
    if device is None:
        return None
    try:
        descriptor_reader = getattr(device, "get_virtual_world_pose_desc", None)
        if callable(descriptor_reader):
            descriptor = descriptor_reader(pose_name)
            # Kit's XRPoseValidityFlags use the OpenXR VALID bits (orientation=1,
            # position=2). TRACKED bits are not required: inferred valid poses are usable.
            if int(descriptor.validity_flags) & 0x3 != 0x3:
                return None
            value = descriptor.pose_matrix
        else:
            value = device.get_virtual_world_pose(pose_name)
        # Gf accepts undersized nested lists and fills omitted entries with identity;
        # reject that coercion at the tracking boundary instead of inventing a pose.
        if value is None or len(value) != 4 or any(len(row) != 4 for row in value):
            return None
        matrix = Gf.Matrix4d(value)
        if not callable(descriptor_reader) and matrix == Gf.Matrix4d(1.0):
            return None
        if not all(math.isfinite(float(matrix[row][column])) for row in range(4) for column in range(4)):
            return None
        if float(matrix.GetDeterminant()) < 1e-8:
            return None
        return matrix
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


def smoothing_alpha(reference_alpha: float, dt: float, reference_hz: float = 100.0) -> float:
    """Convert a per-tick smoothing weight to the actual physics timestep.

    Args:
        reference_alpha: Weight calibrated at the reference frequency, in [0, 1].
        dt: Elapsed simulation time in seconds.
        reference_hz: Frequency at which the weight was calibrated.

    Returns:
        Equivalent exponential filter weight for this timestep.

    Example:

    .. code-block:: python

        >>> round(smoothing_alpha(0.5, 0.02), 2)
        0.75
    """
    if not math.isfinite(dt) or dt <= 0.0:
        return 0.0
    alpha = max(0.0, min(1.0, reference_alpha))
    return 1.0 - (1.0 - alpha) ** (dt * reference_hz)
