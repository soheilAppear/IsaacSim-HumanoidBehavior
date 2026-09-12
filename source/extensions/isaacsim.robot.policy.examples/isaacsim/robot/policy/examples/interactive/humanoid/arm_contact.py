# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bound arm commands at contact without modifying articulation or object state."""

from __future__ import annotations

import numpy as np


def project_contact_step(
    step: np.ndarray,
    contact_rows: np.ndarray,
    lower_step: np.ndarray,
    upper_step: np.ndarray,
) -> np.ndarray:
    """Find the nearest joint step that does not move into any contact normal.

    Each row is the force-on-hand normal multiplied by that contact point's
    Jacobian for the controlled arm joints. Its dot product with the result must
    be nonnegative. This includes contact-point motion caused by wrist rotation,
    not just palm translation. The caller supplies bounds relative to measured
    joints and applies this projection after command filtering.

    Dykstra projections retain separate corrections for the joint bounds and each
    halfspace, minimizing Euclidean change when converged. Stop after 32 cycles;
    an unconverged solve commands zero motion, which is always feasible. This is
    a first-order nonpenetration constraint, not a force or collision solver.

    Args:
        step: Requested one-dimensional joint displacement in radians.
        contact_rows: Contact Jacobian rows, one row per constraint.
        lower_step: Per-joint lower displacement bounds, at most zero.
        upper_step: Per-joint upper displacement bounds, at least zero.

    Returns:
        A new displacement array; input arrays are never modified.

    Raises:
        ValueError: If inputs are malformed, nonfinite, or zero is outside the
            displacement bounds. Callers must stop motion if validation fails.
    """
    requested = np.asarray(step, dtype=float)
    lower = np.asarray(lower_step, dtype=float)
    upper = np.asarray(upper_step, dtype=float)
    rows = np.asarray(contact_rows, dtype=float)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError("Arm step must be a nonempty one-dimensional array")
    if lower.shape != requested.shape or upper.shape != requested.shape:
        raise ValueError("Arm step bounds must match the joint displacement shape")
    if rows.ndim == 1 and rows.size == 0:
        rows = rows.reshape(0, requested.size)
    if rows.ndim != 2 or rows.shape[1] != requested.size:
        raise ValueError("Contact rows must have one column per controlled joint")
    if any(not np.isfinite(values).all() for values in (requested, lower, upper, rows)):
        raise ValueError("Arm contact projection inputs must be finite")
    if np.any(lower > 0.0) or np.any(upper < 0.0):
        raise ValueError("Arm displacement bounds must contain zero")

    # Scale each row first to avoid overflow/underflow when normalizing. A truly
    # zero row places no restriction on this arm, so discard only exact zeros.
    magnitudes = np.max(np.abs(rows), axis=1)
    rows = rows[magnitudes > 0.0] / magnitudes[magnitudes > 0.0, None]
    rows = rows / np.linalg.norm(rows, axis=1)[:, None]
    if not len(rows):
        return np.clip(requested, lower, upper)

    tolerance = 1e-10
    current = requested.copy()
    corrections = np.zeros((len(rows) + 1, requested.size))

    def feasible(value: np.ndarray) -> bool:
        return (
            np.all(value >= lower - tolerance)
            and np.all(value <= upper + tolerance)
            and np.all(rows @ value >= -tolerance)
        )

    for _ in range(32):
        previous = current.copy()
        candidate = current + corrections[0]
        current = np.clip(candidate, lower, upper)
        corrections[0] = candidate - current
        for index, row in enumerate(rows, start=1):
            candidate = current + corrections[index]
            inward = min(0.0, float(row @ candidate))
            current = candidate - inward * row
            corrections[index] = candidate - current
        if not np.isfinite(current).all():
            break
        if np.max(np.abs(current - previous)) <= tolerance and feasible(current):
            # Every feasible set contains zero, so its nearest projection cannot
            # increase the norm. Reject a numerical failure of that property.
            if np.linalg.norm(current) <= np.linalg.norm(requested) + tolerance:
                return current
            break
    return np.zeros_like(requested)
