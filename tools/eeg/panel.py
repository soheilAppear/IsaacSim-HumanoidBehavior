# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tk monitor for raw EEG and explicitly non-actuating command previews."""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

from eeg.app import PreviewApp
from eeg.session import CHANNELS

COLORS = ("#65b7ff", "#8bd6b0", "#e9be6c", "#cf9cf2")


def run_panel(app: PreviewApp, duration: float) -> None:
    """Display the monitor until closed, disconnected, or its duration expires."""
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(f"Cannot open the desktop panel; use --headless: {error}") from error
    root.title("BrainBit · EEG teleoperation lab")
    root.geometry("1040x790")
    root.minsize(850, 720)
    root.configure(bg="#111923")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#111923")
    style.configure("TLabel", background="#111923", foreground="#e8eef4", font=("Segoe UI", 11))
    style.configure("Title.TLabel", font=("Segoe UI", 23, "bold"))
    style.configure("Accent.TLabel", foreground="#8bd6b0", font=("Segoe UI", 11, "bold"))
    style.configure("Command.TLabel", font=("Segoe UI", 25, "bold"), foreground="#65b7ff")
    style.configure("TButton", font=("Segoe UI", 10), padding=(12, 7))
    body = ttk.Frame(root, padding=24)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="EEG teleoperation lab", style="Title.TLabel").pack(anchor="w")
    badge = "SYNTHETIC DEMO  ·  Artificial signals" if app.source == "synthetic" else "BRAINBIT  ·  Live acquisition"
    ttk.Label(body, text=f"{badge}  ·  Recording + preview only", style="Accent.TLabel").pack(anchor="w", pady=(5, 12))
    status = ttk.Label(body, text="Connecting…")
    status.pack(anchor="w")
    canvas = tk.Canvas(body, height=220, background="#172330", highlightthickness=0)
    canvas.pack(fill="x", pady=(14, 7))
    ttk.Label(body, text="O1 / O2 / T3 / T4 · Last 2 seconds · Display is centered per channel").pack(anchor="w")
    contact = ttk.Label(body, text="Contact resistance: waiting for preflight check")
    contact.pack(anchor="w", pady=(7, 12))
    command = ttk.Label(body, text="NO COMMAND", style="Command.TLabel")
    command.pack(anchor="w")
    reason = ttk.Label(body, text="Waiting for samples", wraplength=930)
    reason.pack(anchor="w", pady=(5, 5))
    ttk.Label(body, text="Experimental frequency matching. Scores are not probabilities of intent.").pack(anchor="w")
    ttk.Label(body, text="Robot output: disconnected · This panel cannot move the robot.", style="Accent.TLabel").pack(
        anchor="w", pady=(5, 15)
    )
    ttk.Label(body, text="Mark your current task for later analysis (does not change the prediction):").pack(anchor="w")
    labels = ttk.Frame(body)
    labels.pack(anchor="w", pady=8)
    annotation = ttk.Label(body, text="Task label: unlabeled")

    def mark(label: str) -> None:
        try:
            app.recorder.set_label(label)
            annotation.configure(text=f"Task label: {label}")
        except OSError as error:
            app.error = f"Cannot save task label: {error}"
            close()

    for label in ("rest", "forward", "left", "right", "stop", "unlabeled"):
        ttk.Button(labels, text=label.capitalize(), command=lambda value=label: mark(value)).pack(
            side="left", padx=(0, 6)
        )
    annotation.pack(anchor="w")
    footer = ttk.Frame(body)
    footer.pack(fill="x", side="bottom", pady=(12, 0))
    ttk.Label(footer, text=f"Session: {app.recorder.path.name}").pack(side="left")
    started = time.monotonic()
    closing = False

    def close() -> None:
        nonlocal closing
        if closing:
            return
        closing = True
        app.stop.set()
        root.destroy()

    ttk.Button(footer, text="Stop recording", command=close).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", close)

    def refresh() -> None:
        if closing:
            return
        try:
            app.pump()
            state = app.snapshot
            status.configure(
                text=f"{app.message}  |  {app.recorder.sample_count:,} samples  |  "
                f"{app.recorder.missing_samples} missing  |  Quality: {state['quality']}"
            )
            command.configure(text=state["command"].upper() if state["command"] != "none" else "NO COMMAND")
            reason.configure(text=f"{state['reason']}  ·  Match score: {state['score']:.3f}")
            if app.resistance is not None:
                values = "   ".join(
                    f"{ch}: {value / 1000:.0f} kΩ" if value is not None else f"{ch}: unavailable"
                    for ch, value in zip(CHANNELS, app.resistance)
                )
                age = time.monotonic() - app.resistance_time
                contact.configure(text=f"Contact preflight: {values}  ·  {age:.0f}s ago (not continuously measured)")
            elif app.source == "synthetic":
                contact.configure(text="Contact resistance: not measured in synthetic mode")
            canvas.delete("all")
            width = max(canvas.winfo_width(), 800)
            points = list(app.trace)
            for channel, (name, color) in enumerate(zip(CHANNELS, COLORS)):
                center_y = 27 + channel * 55
                canvas.create_line(65, center_y, width - 12, center_y, fill="#304052")
                canvas.create_text(12, center_y, text=name, fill=color, anchor="w", font=("Segoe UI", 11, "bold"))
                if len(points) > 1:
                    mean = sum(point[channel] for point in points) / len(points)
                    peak = max(abs(point[channel] - mean) for point in points)
                    scale = max(peak, 25e-6)
                    coords = []
                    for index, point in enumerate(points):
                        coords.extend(
                            (
                                65 + index * (width - 80) / (len(points) - 1),
                                center_y - 25 * (point[channel] - mean) / scale,
                            )
                        )
                    canvas.create_line(*coords, fill=color, width=1)
                    canvas.create_text(width - 16, center_y - 20, text=f"±{scale * 1e6:.0f} µV", fill=color, anchor="e")
            if duration > 0 and time.monotonic() - started >= duration:
                close()
                return
            root.after(100, refresh)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            app.error = f"Preview failed: {error}"
            close()

    root.after(50, refresh)
    try:
        root.mainloop()
    except tk.TclError as error:
        raise RuntimeError(f"Desktop panel failed: {error}") from error
    finally:
        if not closing:
            close()
