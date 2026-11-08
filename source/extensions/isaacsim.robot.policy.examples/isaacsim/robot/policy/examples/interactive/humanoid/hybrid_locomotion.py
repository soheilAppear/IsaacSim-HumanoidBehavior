# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose controller or gaze intent without combining their velocity commands.

The caller first updates its controller neutral interlock using current samples.
Left-stick X/Y and right-stick X participate in hybrid control; right-stick Y
is ignored by the caller. This module owns no XR sampling, clocks, robot state,
or physical stop guards. Preview inhibition remains the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HybridLocomotion", "HybridSelection"]


@dataclass(frozen=True)
class HybridSelection:
    """Select exactly one source, or request zero for a blocked/handoff update.

    ``reset_gaze`` requests cancellation of gaze selection and clearing of the
    previous source's command filter. Whenever ``source`` is not ``gaze``, the
    caller must keep gaze suspended with ``ready=False`` so it cannot consume
    look-away samples during controller ownership. The gaze controller's normal
    look-away interlock then governs resumption. Do not reset physical stop
    supervision when changing input ownership.
    """

    source: str = "gaze"
    reset_gaze: bool = True
    stop_immediately: bool = False
    detail: str = "Gaze available; look outside the buttons to enable selection"


class HybridLocomotion:
    """Give deliberate stick input priority while preserving both input interlocks.

    A deflected valid axis claims control even when its partner is unavailable or
    the controller neutral interlock is unarmed. Such attempts request zero.
    Neutral or unreadable controllers that have never claimed control leave gaze
    available, including mixed controller/optical-hand use. Release or loss of an
    established claim inserts one zero-command handoff update before gaze can
    become available again.
    """

    def __init__(self) -> None:
        self.reset()

    @property
    def status(self) -> dict[str, object]:
        """Return an independent snapshot for recording and UI diagnostics."""
        return {
            "source": self.selection.source,
            "reset_gaze": self.selection.reset_gaze,
            "stop_immediately": self.selection.stop_immediately,
            "detail": self.selection.detail,
            "controller_claimed": self._controller_claimed,
        }

    def reset(self) -> None:
        """Forget controller ownership and invalidate any previous gaze selection."""
        self._controller_claimed = False
        self._reset_pending = True
        self.selection = HybridSelection()

    def select(
        self,
        *,
        controller_present: bool,
        controller_valid: bool,
        controller_state: str,
        sticks_deflected: bool,
    ) -> HybridSelection:
        """Arbitrate current input metadata after updating controller locomotion.

        Args:
            controller_present: At least one current input device is a controller.
            controller_valid: Both required stick samples are currently valid.
            controller_state: Current controller interlock state: ``active``,
                ``idle``, ``awaiting_neutral`` or ``unavailable``.
            sticks_deflected: A valid left-X, left-Y or right-X sample exceeds its deadzone.
                Missing/unreadable axes do not themselves imply deflection.

        Returns:
            The immutable selection also assigned to ``selection``. Invalid or
            contradictory metadata requests zero and invalidates gaze.
        """
        valid_flags = all(type(value) is bool for value in (controller_present, controller_valid, sticks_deflected))
        valid_state = isinstance(controller_state, str) and controller_state in (
            "active",
            "idle",
            "awaiting_neutral",
            "unavailable",
        )
        if not valid_flags or not valid_state:
            self._controller_claimed = True
            return self._publish("none", True, "Invalid controller arbitration metadata; walking stopped")

        # An active command without a usable deflection hint indicates stale or
        # inconsistent caller metadata, rather than a verified neutral release.
        if controller_state == "active" and not sticks_deflected:
            self._controller_claimed = True
            return self._publish("none", True, "Controller command and stick samples disagree; walking stopped")

        if sticks_deflected:
            new_claim = not self._controller_claimed
            self._controller_claimed = True
            if controller_present and controller_valid and controller_state == "active":
                return self._publish(
                    "controller",
                    new_claim or self.selection.source != "controller",
                    "Controller sticks override gaze; release sticks to return to gaze",
                )
            return self._publish(
                "none",
                new_claim or self.selection.source == "controller",
                "Stick input is not armed or fully tracked; center both sticks before walking",
            )

        if self._controller_claimed:
            self._controller_claimed = False
            return self._publish(
                "none", True, "Controller control ended; look outside the gaze buttons before selecting again"
            )

        return self._publish("gaze", False, "Gaze available; controller sticks are not requesting movement")

    def _publish(self, source: str, reset_gaze: bool, detail: str) -> HybridSelection:
        """Keep the returned command and the integration accessor synchronized."""
        self.selection = HybridSelection(
            source=source,
            reset_gaze=reset_gaze or self._reset_pending,
            stop_immediately=source == "none",
            detail=detail,
        )
        self._reset_pending = False
        return self.selection
