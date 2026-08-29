"""Tkinter GUI for building, inspecting and exporting nanocarbon structures.

Launch with ``nanocarbon-gui`` (installed entry point) or
``python -m nanocarbon_lab.gui``.

Three columns: parameters on the left, a live 3D preview in the middle,
and structure readout, history and export/render actions on the right.

Four things about the design are deliberate and worth knowing before
changing them.

**Every number is typeable.** A slider is convenient and imprecise; the
entry box beside it takes an exact value and accepts figures outside the
slider's comfortable range, up to the builder's real limit. A coil radius
of 37.5 Å is a perfectly good request that no slider detent will land on.

**Builds run in a killable subprocess**, not a thread -- see
:mod:`nanocarbon_lab.gui.worker`. A coil spends minutes inside numpy with
nothing checking a cancel flag, and Python cannot safely interrupt a
thread, so Cancel means terminating a process.

**What a build will cost is shown before you start it.** The same button
produces either a 60-atom cage in a tenth of a second or a 3000-atom coil
in six minutes, and :func:`nanocarbon_lab.jobs.estimate_cost` can tell
the difference without building anything.

**Failures never open a modal dialog.** Errors go to a panel that can be
read, copied and ignored. A modal blocks the event loop, which on a
headless run wedges the app entirely, and even for a human it throws away
the parameters they were about to fix.

Only the standard library's :mod:`tkinter` plus :mod:`matplotlib` are
used. ``tkinter`` ships with the python.org installers on Windows and
macOS; on Linux it is usually a separate package (``apt install
python3-tk``, ``dnf install python3-tkinter``).
"""

from __future__ import annotations

import glob
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "The nanocarbon_lab GUI needs tkinter, which is not available in this "
        "Python installation.\n"
        "  Debian/Ubuntu : sudo apt install python3-tk\n"
        "  Fedora/RHEL   : sudo dnf install python3-tkinter\n"
        "  macOS/Windows : use the python.org installer (tkinter is included)\n"
        "You can still build structures without a GUI via the 'nanocarbon' "
        "command."
    ) from exc

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

from ..builders import fullerene_mesh as fm
from ..exports.xyz import write_render_bundle
from ..jobs import MODES, Job, estimate_cost, to_cli
from ..validation.quality import sp2_quality
from .worker import WORKER_DIED, BuildWorker

# Ring-type colours, shared with the Blender presets' intent: hexagons are
# the neutral body, everything else marks curvature or a defect.
RING_COLOURS = {5: "#e4572e", 6: "#5b6472", 7: "#2e86ab", 8: "#f2c14e"}
RING_LABELS = {
    5: "pentagon (convex / cap)",
    6: "hexagon (body)",
    7: "heptagon (concave / saddle)",
    8: "octagon (divacancy)",
}

SHAPES = ["straight", "arc", "s_curve", "helix", "random"]
DOPANTS = ["none", "N", "B", "S", "P"]
JUNCTION_KINDS = ["L", "T", "Y", "X", "cross3d"]
SCHWARZITE_KINDS = ["primitive", "diamond", "gyroid"]
CAGE_FAMILIES = ["C60", "C20"]

BLENDER_STYLES = [
    "nature_dark",
    "acs_nano_vivid",
    "small_minimal",
    "blueprint_technical",
    "gold_nanotech",
]

OK_GREEN = "#2e7d32"
WARN_AMBER = "#b26a00"
BAD_RED = "#b3261e"
MUTED = "#777777"

# Above this many atoms the preview subsamples bonds. Matplotlib draws a
# Line3DCollection as one artist, but assembling a quarter of a million
# segments still costs seconds per redraw, and no one can see individual
# bonds at that density anyway.
PREVIEW_BOND_LIMIT = 20000

# Structures worth having one click away. Keys are parameter names as
# registered with `_var`, so applying a preset is a plain loop and the
# same format serves the save/load file.
PRESETS: dict[str, dict[str, object]] = {
    "C60 buckyball": {
        "mode_kind": "fullerene", "cage_family": "C60", "cage_freq": 1},
    "C540 giant cage": {
        "mode_kind": "fullerene", "cage_family": "C60", "cage_freq": 3},
    "Nano-onion C60@C240@C540": {
        "mode_kind": "nano-onion", "cage_family": "C60", "cage_freq": 1,
        "onion_shells": 3},
    "Capped nanotube": {
        "mode_kind": "capped tube", "rings": 10, "freq": 3, "shape": "straight",
        "roughness": 0.0, "n_sw": 0, "n_dv": 0},
    "CVD-rough nanotube": {
        "mode_kind": "capped tube", "rings": 10, "freq": 3, "shape": "straight",
        "roughness": 0.25, "anneal": 0, "n_sw": 2, "n_dv": 1},
    "N-doped nanotube": {
        "mode_kind": "capped tube", "rings": 10, "freq": 3,
        "dopant": "N", "dopant_conc": 0.03},
    "Nanocoil (swept, fast)": {
        "mode_kind": "capped tube", "shape": "helix", "freq": 2,
        "coil_radius": 90.0, "coil_pitch": 30.0, "coil_turns": 1.5},
    "Nanocoil (relaxed topology)": {
        "mode_kind": "coil (relaxed)", "coil_radius": 34.0, "coil_pitch": 20.0,
        "coil_turns": 1.25, "coil_tube_radius": 4.5, "anneal": 40},
    "Y junction": {
        "mode_kind": "junction", "j_kind": "Y", "j_radius": 6.0,
        "j_arm": 22.0, "j_blend": 4.0, "anneal": 80},
    "Gyroid schwarzite": {
        "mode_kind": "schwarzite", "s_kind": "gyroid", "s_cell": 36.0,
        "anneal": 0},
    "Double-wall nanotube": {
        "mode_kind": "multi-wall", "mw_shells": 2, "mw_inner": 3, "rings": 10},
    "Seven-tube rope": {
        "mode_kind": "bundle", "bundle_shells": 1, "freq": 3, "rings": 10},
}


def find_blender() -> str | None:
    """Locate a Blender executable across platforms.

    ``PATH`` alone is not enough: the Windows installer does not add
    Blender to ``PATH``, and the macOS build lives inside an ``.app``
    bundle, so on both platforms ``shutil.which`` finds nothing even
    though Blender is installed. Standard install locations are therefore
    searched too, newest version first.

    Set the ``BLENDER`` environment variable to override the search
    entirely (useful for portable, Steam or Flatpak installations).

    Returns
    -------
    str or None
        Path to the executable, or ``None`` if nothing was found -- in
        which case the GUI offers a file picker instead.
    """
    override = os.environ.get("BLENDER")
    if override and os.path.exists(override):
        return override

    for name in ("blender", "blender.exe"):
        found = shutil.which(name)
        if found:
            return found

    # Built with os.path.join, not pathlib: these are glob patterns (plain
    # strings), and pathlib would refuse to model a Windows path on POSIX,
    # which would also make this function untestable off-Windows.
    patterns: list[str] = []
    if os.name == "nt":
        for root in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ):
            patterns.append(os.path.join(root, "Blender Foundation", "Blender*", "blender.exe"))
            patterns.append(os.path.join(root, "Blender*", "blender.exe"))
    elif sys.platform == "darwin":
        patterns += [
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender/Blender.app/Contents/MacOS/Blender",
            os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"),
        ]
    else:
        patterns += [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/snap/bin/blender",
            "/var/lib/flatpak/exports/bin/org.blender.Blender",
            os.path.expanduser("~/.local/share/flatpak/exports/bin/org.blender.Blender"),
        ]

    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    # Newest-looking install first ("Blender 4.2" sorts above "Blender 3.6").
    candidates.sort(reverse=True)
    return candidates[0]


class ScrollableColumn(ttk.Frame):
    """A vertically scrollable container for the parameter panels.

    With eight structure types the parameter column is taller than a
    laptop screen, and without scrolling the Build button simply falls off
    the bottom where it cannot be reached -- the panel is packed, so
    nothing clips it or tells you it is there.
    """

    def __init__(self, parent: tk.Widget, width: int = 268) -> None:
        super().__init__(parent)
        self._canvas = tk.Canvas(self, width=width, highlightthickness=0,
                                 background=parent.winfo_toplevel().cget("background"))
        bar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self.interior = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window(
            (0, 0), window=self.interior, anchor="nw", width=width
        )
        self.interior.bind("<Configure>", self._on_interior_resize)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        # Wheel events go to the widget under the pointer, so bind on enter
        # and release on leave rather than grabbing them globally -- a
        # global binding would scroll this column while the pointer is over
        # the 3D preview, where the wheel means zoom.
        self.interior.bind("<Enter>", lambda _e: self._bind_wheel())
        self.interior.bind("<Leave>", lambda _e: self._unbind_wheel())

    def _on_interior_resize(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)

    def _bind_wheel(self) -> None:
        self._canvas.bind_all("<MouseWheel>", self._on_wheel)
        self._canvas.bind_all("<Button-4>", self._on_wheel)
        self._canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self) -> None:
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._canvas.unbind_all(sequence)

    def _on_wheel(self, event: tk.Event) -> None:
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:  # Windows and macOS report a signed delta instead
            delta = -1 if event.delta > 0 else 1
        self._canvas.yview_scroll(delta, "units")


class NanocarbonGUI:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("nanocarbon_lab — carbon nanostructure builder")
        self.root.geometry("1400x860")
        self.root.minsize(1120, 700)

        self.atoms = None
        self.last_saved_stem: Path | None = None
        self.blender_exe: str | None = None
        self._busy = False
        self._estimate_job: str | None = None
        # Registry of every parameter variable, keyed by a short name.
        # Presets, the save/load file and the estimate traces all iterate
        # this rather than naming each variable three times over.
        self._params: dict[str, tk.Variable] = {}
        # Built structures, most recent last, so a promising result is not
        # lost the moment the next parameter is nudged.
        self._history: list[tuple[str, object]] = []

        self.worker = BuildWorker()

        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll_worker()
        self.on_build()  # start with something on screen

    # ---------------------------------------------------------------- layout
    def _build_widgets(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        left = ScrollableColumn(outer)
        left.pack(side="left", fill="y", padx=(0, 8))
        centre = ttk.Frame(outer)
        centre.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(outer, width=300)
        right.pack(side="left", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        self._build_params(left.interior)
        self._build_preview(centre)
        self._build_actions(right)

    # ------------------------------------------------------------ parameters
    def _var(self, name: str, var: tk.Variable) -> tk.Variable:
        """Register a parameter variable under a short, stable name."""
        self._params[name] = var
        return var

    def _param(self, parent, label, var, lo, hi, row, *, integer=False,
               resolution=None, command=None, hard_lo=None, hard_hi=None):
        """A labelled parameter: exact entry box plus a slider.

        The slider covers the range that is comfortable to explore; the
        entry accepts anything between ``hard_lo`` and ``hard_hi``, which
        default to a wider window. That split is the point -- a slider
        cannot express 37.5 when its detent is 5, and clamping typed input
        to the slider's range would make the box pointless.

        Bad input is reverted rather than raising: a half-typed number is
        a normal intermediate state, not an error worth a dialog.
        """
        step = resolution or (1 if integer else 0.01)
        hard_lo = lo if hard_lo is None else hard_lo
        hard_hi = hi if hard_hi is None else hard_hi

        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        text = tk.StringVar()
        entry = ttk.Entry(parent, textvariable=text, width=8, justify="right")
        entry.grid(row=row, column=1, sticky="e")

        def render() -> None:
            text.set(str(var.get()) if integer else f"{float(var.get()):g}")

        def quantise(value: float) -> float:
            value = max(hard_lo, min(hard_hi, value))
            if integer:
                return round(value)
            return round(round(value / step) * step, 6)

        def commit(_event=None) -> None:
            try:
                wanted = quantise(float(text.get()))
            except (TypeError, ValueError):
                render()  # unparseable: put the live value back
                return
            if wanted != var.get():
                var.set(wanted)
            render()
            if command:
                command()

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        # Keep the box in step with the variable however it changed --
        # slider, preset, loaded file, or a mode switch forcing a value.
        # Without this the entry silently shows a stale number, which is
        # worse than showing none: it is the field you would trust.
        var.trace_add("write", lambda *_: render())

        def on_slide(_value=None) -> None:
            wanted = quantise(var.get())
            if wanted != var.get():
                var.set(wanted)
            render()
            if command:
                command()

        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var,
                          orient="horizontal", command=lambda _v: on_slide())
        scale.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        parent.columnconfigure(0, weight=1)
        render()

    def _build_params(self, parent: ttk.Frame) -> None:
        self.var_mode_kind = self._var("mode_kind", tk.StringVar(value=MODES[0]))
        mode_box = ttk.LabelFrame(parent, text="Structure type", padding=8)
        mode_box.pack(fill="x", pady=(0, 8))
        ttk.Combobox(mode_box, textvariable=self.var_mode_kind, values=list(MODES),
                     state="readonly").pack(fill="x")
        self.var_mode_kind.trace_add("write", lambda *_: self._on_mode_change())

        preset_row = ttk.Frame(mode_box)
        preset_row.pack(fill="x", pady=(6, 0))
        ttk.Label(preset_row, text="Preset").pack(side="left")
        self.var_preset = tk.StringVar(value="")
        preset_box = ttk.Combobox(preset_row, textvariable=self.var_preset,
                                  values=sorted(PRESETS), state="readonly", width=22)
        preset_box.pack(side="right")
        preset_box.bind("<<ComboboxSelected>>",
                        lambda _e: self.apply_preset(self.var_preset.get()))

        # --- tube geometry
        box = ttk.LabelFrame(parent, text="Geometry", padding=8)
        box.pack(fill="x")
        self.frame_tube = box

        self.var_rings = self._var("rings", tk.IntVar(value=8))
        self.var_freq = self._var("freq", tk.IntVar(value=3))
        self.var_bond = self._var("bond", tk.DoubleVar(value=1.42))
        self.var_bend = self._var("bend", tk.DoubleVar(value=0.0))
        self.var_seed = self._var("seed", tk.IntVar(value=0))
        self.var_shape = self._var("shape", tk.StringVar(value="straight"))
        self.var_waviness = self._var("waviness", tk.DoubleVar(value=0.7))
        self.var_max_strain = self._var("max_strain", tk.DoubleVar(value=0.08))
        self.var_shape_points = self._var("shape_points", tk.IntVar(value=9))
        self.var_coil_radius = self._var("coil_radius", tk.DoubleVar(value=60.0))
        self.var_coil_pitch = self._var("coil_pitch", tk.DoubleVar(value=25.0))
        self.var_coil_turns = self._var("coil_turns", tk.DoubleVar(value=1.5))
        self.var_coil_hand = self._var("coil_hand", tk.StringVar(value="right"))
        self.var_coil_taper = self._var("coil_taper", tk.DoubleVar(value=1.0))
        self.var_coil_tube_radius = self._var(
            "coil_tube_radius", tk.DoubleVar(value=6.0))
        self.var_pin_ends = self._var("pin_ends", tk.BooleanVar(value=False))
        self.var_anneal = self._var("anneal", tk.IntVar(value=80))
        self.var_roughness = self._var("roughness", tk.DoubleVar(value=0.0))
        self.var_dopant = self._var("dopant", tk.StringVar(value="none"))
        self.var_dopant_conc = self._var("dopant_conc", tk.DoubleVar(value=0.03))
        self.var_n_sw = self._var("n_sw", tk.IntVar(value=0))
        self.var_n_dv = self._var("n_dv", tk.IntVar(value=0))
        self.var_mw_shells = self._var("mw_shells", tk.IntVar(value=2))
        self.var_mw_inner = self._var("mw_inner", tk.IntVar(value=3))
        self.var_mw_step = self._var("mw_step", tk.IntVar(value=2))
        self.var_bundle_shells = self._var("bundle_shells", tk.IntVar(value=1))
        self.var_bundle_gap = self._var("bundle_gap", tk.DoubleVar(value=3.4))
        self.var_cage_family = self._var("cage_family", tk.StringVar(value="C60"))
        self.var_cage_freq = self._var("cage_freq", tk.IntVar(value=1))
        self.var_onion_shells = self._var("onion_shells", tk.IntVar(value=3))
        self.var_j_kind = self._var("j_kind", tk.StringVar(value="Y"))
        self.var_j_radius = self._var("j_radius", tk.DoubleVar(value=6.0))
        self.var_j_arm = self._var("j_arm", tk.DoubleVar(value=22.0))
        self.var_j_blend = self._var("j_blend", tk.DoubleVar(value=4.0))
        self.var_s_kind = self._var("s_kind", tk.StringVar(value="primitive"))
        self.var_s_cell = self._var("s_cell", tk.DoubleVar(value=36.0))
        self.var_s_thickness = self._var("s_thickness", tk.DoubleVar(value=0.0))

        self.lbl_radius = ttk.Label(box, text="", foreground="#2e86ab")

        self._param(box, "Body rings (length)", self.var_rings, 2, 30, 0,
                    integer=True, hard_hi=200)
        self._param(box, "Subdivision freq (diameter)", self.var_freq, 1, 8, 2,
                    integer=True, hard_hi=20, command=self._update_radius_hint)
        self.lbl_radius.grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self._param(box, "Bend angle (rad)", self.var_bend, 0.0, 1.0, 5,
                    resolution=0.01)
        self._param(box, "C–C bond (Å)", self.var_bond, 1.30, 1.55, 7,
                    resolution=0.005, hard_lo=1.20, hard_hi=1.80,
                    command=self._update_radius_hint)

        seed_row = ttk.Frame(box)
        seed_row.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(seed_row, text="Random seed").pack(side="left")
        ttk.Button(seed_row, text="🎲", width=3,
                   command=self.on_roll_seed).pack(side="right", padx=(4, 0))
        ttk.Spinbox(seed_row, from_=0, to=999999, textvariable=self.var_seed,
                    width=8).pack(side="right")

        # --- centreline shape
        sbox = ttk.LabelFrame(parent, text="Centreline", padding=8)
        sbox.pack(fill="x", pady=(8, 0))
        self.frame_centreline = sbox
        # Laid out with grid throughout: _param grids, and tkinter forbids
        # mixing grid and pack in one container.
        sbox.columnconfigure(0, weight=1)
        ttk.Label(sbox, text="Shape").grid(row=0, column=0, sticky="w")
        ttk.Combobox(sbox, textvariable=self.var_shape, values=SHAPES,
                     state="readonly", width=11).grid(row=0, column=1, sticky="e",
                                                      pady=(0, 6))
        self._param(sbox, "Waviness", self.var_waviness, 0.0, 1.0, 1,
                    resolution=0.05)
        self._param(sbox, "Control points", self.var_shape_points, 4, 20, 3,
                    integer=True, hard_hi=60)
        self._param(sbox, "Strain budget", self.var_max_strain, 0.02, 0.25, 5,
                    resolution=0.01, hard_hi=0.5, command=self._update_strain_hint)
        self.lbl_strain = ttk.Label(sbox, text="", foreground=MUTED,
                                    font=("TkDefaultFont", 8), wraplength=230,
                                    justify="left")
        self.lbl_strain.grid(row=7, column=0, columnspan=2, sticky="w")
        self.var_shape.trace_add("write", lambda *_: self._on_shape_change())
        ttk.Label(sbox, text="Thinner + longer tubes curve more at the same "
                             "strain — lower the frequency and raise the rings.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=8, column=0, columnspan=2, sticky="w",
                                       pady=(4, 0))

        # --- coil dimensions, in real Å. Shared by the swept helix (where
        # they size the tube) and the relaxed coil (where they size the
        # implicit surface), hence a top-level panel.
        self.frame_coil = ttk.LabelFrame(parent, text="Coil", padding=8)
        self.frame_coil.pack(fill="x", pady=(8, 0))
        self.frame_coil.columnconfigure(0, weight=1)
        self._param(self.frame_coil, "Coil radius (Å)", self.var_coil_radius,
                    15.0, 200.0, 0, resolution=0.5, hard_hi=2000.0,
                    command=self._update_coil_hint)
        self._param(self.frame_coil, "Coil pitch (Å)", self.var_coil_pitch,
                    5.0, 100.0, 2, resolution=0.5, hard_hi=1000.0,
                    command=self._update_coil_hint)
        self._param(self.frame_coil, "Turns", self.var_coil_turns,
                    0.5, 5.0, 4, resolution=0.05, hard_hi=40.0,
                    command=self._update_coil_hint)
        self._param(self.frame_coil, "Taper (end/start R)", self.var_coil_taper,
                    0.3, 2.0, 6, resolution=0.05, hard_lo=0.05, hard_hi=10.0,
                    command=self._update_coil_hint)
        hand = ttk.Frame(self.frame_coil)
        hand.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(2, 2))
        ttk.Label(hand, text="Handedness").pack(side="left")
        ttk.Combobox(hand, textvariable=self.var_coil_hand,
                     values=["right", "left"], state="readonly",
                     width=7).pack(side="right")
        self.lbl_coil = ttk.Label(self.frame_coil, text="", foreground=MUTED,
                                  font=("TkDefaultFont", 8), wraplength=225,
                                  justify="left")
        self.lbl_coil.grid(row=9, column=0, columnspan=2, sticky="w")
        # Only the relaxed coil sets its tube radius freely; the swept
        # helix takes it from the lattice-quantised frequency.
        self.frame_coil_tube = ttk.Frame(self.frame_coil)
        self.frame_coil_tube.grid(row=10, column=0, columnspan=2, sticky="ew")
        self.frame_coil_tube.columnconfigure(0, weight=1)
        self._param(self.frame_coil_tube, "Tube radius (Å)",
                    self.var_coil_tube_radius, 4.0, 12.0, 0, resolution=0.1,
                    hard_lo=2.0, hard_hi=30.0, command=self._update_coil_hint)
        ttk.Checkbutton(self.frame_coil_tube, text="Pin ends (hold the pitch)",
                        variable=self.var_pin_ends).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 2))
        ttk.Label(self.frame_coil_tube,
                  text="Rings follow the curvature here: pentagons inside, "
                       "heptagons outside, so bonds stay graphitic instead of "
                       "stretching. Slower to build.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=225,
                  justify="left").grid(row=3, column=0, columnspan=2, sticky="w")

        # --- defects
        dbox = ttk.LabelFrame(parent, text="Defects", padding=8)
        dbox.pack(fill="x", pady=(8, 0))
        self.frame_defects = dbox
        dbox.columnconfigure(0, weight=1)
        self._param(dbox, "Stone–Wales (5-7-7-5)", self.var_n_sw, 0, 12, 0,
                    integer=True, hard_hi=200)
        self._param(dbox, "Divacancy (5-8-5)", self.var_n_dv, 0, 12, 2,
                    integer=True, hard_hi=200)
        ttk.Label(dbox, text="Both are Euler-neutral: they change ring types, "
                             "never the pentagon budget.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=4, column=0, columnspan=2, sticky="w")

        # --- junction
        self.frame_junction = ttk.LabelFrame(parent, text="Junction", padding=8)
        self.frame_junction.columnconfigure(0, weight=1)
        ttk.Label(self.frame_junction, text="Kind").grid(row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_junction, textvariable=self.var_j_kind,
                     values=JUNCTION_KINDS, state="readonly", width=8).grid(
            row=0, column=1, sticky="e", pady=(0, 6))
        self._param(self.frame_junction, "Arm radius (Å)", self.var_j_radius,
                    4.0, 14.0, 1, resolution=0.1, hard_lo=2.0, hard_hi=40.0)
        self._param(self.frame_junction, "Arm length (Å)", self.var_j_arm,
                    10.0, 60.0, 3, resolution=0.5, hard_hi=400.0)
        self._param(self.frame_junction, "Neck blend (Å)", self.var_j_blend,
                    1.0, 12.0, 5, resolution=0.1, hard_hi=40.0)
        ttk.Label(self.frame_junction,
                  text="The branch is a saddle, so the remesher tiles it with "
                       "heptagons — nothing prescribes them.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=7, column=0, columnspan=2, sticky="w")

        # --- schwarzite
        self.frame_schwarzite = ttk.LabelFrame(parent, text="Schwarzite", padding=8)
        self.frame_schwarzite.columnconfigure(0, weight=1)
        ttk.Label(self.frame_schwarzite, text="Surface").grid(row=0, column=0,
                                                              sticky="w")
        ttk.Combobox(self.frame_schwarzite, textvariable=self.var_s_kind,
                     values=SCHWARZITE_KINDS, state="readonly", width=10).grid(
            row=0, column=1, sticky="e", pady=(0, 6))
        self._param(self.frame_schwarzite, "Cell length (Å)", self.var_s_cell,
                    30.0, 56.0, 1, resolution=0.5, hard_lo=20.0, hard_hi=120.0)
        self._param(self.frame_schwarzite, "Thickness offset", self.var_s_thickness,
                    -0.4, 0.4, 3, resolution=0.02, hard_lo=-1.0, hard_hi=1.0)
        ttk.Label(self.frame_schwarzite,
                  text="A periodic unit cell: tubes leave one face and return "
                       "through the opposite one. Bigger cells curve more "
                       "gently and relax cleaner — minimum 30 Å primitive, "
                       "36 gyroid and diamond.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=5, column=0, columnspan=2, sticky="w")

        # --- fullerene cage / nano-onion
        self.frame_cage = ttk.LabelFrame(parent, text="Cage", padding=8)
        self.frame_cage.columnconfigure(0, weight=1)
        ttk.Label(self.frame_cage, text="Family").grid(row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_cage, textvariable=self.var_cage_family,
                     values=CAGE_FAMILIES, state="readonly", width=7).grid(
            row=0, column=1, sticky="e", pady=(0, 6))
        self.var_cage_family.trace_add("write", lambda *_: self._update_cage_hint())
        self._param(self.frame_cage, "Frequency (size)", self.var_cage_freq,
                    1, 6, 1, integer=True, hard_hi=20,
                    command=self._update_cage_hint)
        self.frame_onion = ttk.Frame(self.frame_cage)
        self.frame_onion.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.frame_onion.columnconfigure(0, weight=1)
        self._param(self.frame_onion, "Shells", self.var_onion_shells, 1, 5, 0,
                    integer=True, hard_hi=12, command=self._update_cage_hint)
        self.lbl_cage = ttk.Label(self.frame_cage, text="", foreground=MUTED,
                                  font=("TkDefaultFont", 8), wraplength=230,
                                  justify="left")
        self.lbl_cage.grid(row=5, column=0, columnspan=2, sticky="w")

        # --- multi-wall
        self.frame_mw = ttk.LabelFrame(parent, text="Multi-wall", padding=8)
        self.frame_mw.columnconfigure(0, weight=1)
        self._param(self.frame_mw, "Shells", self.var_mw_shells, 1, 6, 0,
                    integer=True, hard_hi=20)
        self._param(self.frame_mw, "Inner freq", self.var_mw_inner, 1, 6, 2,
                    integer=True, hard_hi=20)
        self._param(self.frame_mw, "Freq step", self.var_mw_step, 1, 4, 4,
                    integer=True, hard_hi=10)
        ttk.Label(self.frame_mw,
                  text="Walls land ~3.9 Å apart at step 2: the lattice quantises "
                       "radius in ~1.96 Å steps, so it cannot hit graphite's "
                       "3.4 Å exactly. A nano-onion can.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=6, column=0, columnspan=2, sticky="w")

        # --- bundle
        self.frame_bundle = ttk.LabelFrame(parent, text="Bundle", padding=8)
        self.frame_bundle.columnconfigure(0, weight=1)
        self._param(self.frame_bundle, "Hex shells", self.var_bundle_shells,
                    0, 3, 0, integer=True, hard_hi=8)
        self._param(self.frame_bundle, "Wall gap (Å)", self.var_bundle_gap,
                    2.8, 6.0, 2, resolution=0.1, hard_lo=1.0, hard_hi=20.0)
        ttk.Label(self.frame_bundle,
                  text="0 / 1 / 2 / 3 shells give 1 / 7 / 19 / 37 tubes on a "
                       "triangular lattice at the van der Waals gap.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=4, column=0, columnspan=2, sticky="w")

        # --- surface finish
        self.frame_surface = ttk.LabelFrame(parent, text="Surface finish", padding=8)
        self.frame_surface.columnconfigure(0, weight=1)
        self._param(self.frame_surface, "Smoothing (anneal)", self.var_anneal,
                    0, 200, 0, integer=True, hard_hi=2000,
                    command=self._update_surface_hint)
        self._param(self.frame_surface, "Roughness (Å)", self.var_roughness,
                    0.0, 0.6, 2, resolution=0.01, hard_hi=2.0,
                    command=self._update_surface_hint)
        self.lbl_surface = ttk.Label(self.frame_surface, text="", foreground=MUTED,
                                     font=("TkDefaultFont", 8), wraplength=230,
                                     justify="left")
        self.lbl_surface.grid(row=4, column=0, columnspan=2, sticky="w")

        # --- chemistry
        self.frame_chem = ttk.LabelFrame(parent, text="Chemistry", padding=8)
        self.frame_chem.columnconfigure(0, weight=1)
        ttk.Label(self.frame_chem, text="Dopant").grid(row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_chem, textvariable=self.var_dopant, values=DOPANTS,
                     state="readonly", width=7).grid(row=0, column=1, sticky="e",
                                                     pady=(0, 6))
        self._param(self.frame_chem, "Concentration", self.var_dopant_conc,
                    0.0, 0.15, 1, resolution=0.005, hard_hi=0.5)

        # --- build / cancel and the cost estimate
        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=(12, 0))
        self.btn_build = ttk.Button(actions, text="Build structure",
                                    command=self.on_build)
        self.btn_build.pack(fill="x", ipady=4)
        self.btn_cancel = ttk.Button(actions, text="Cancel", state="disabled",
                                     command=self.on_cancel)
        self.btn_cancel.pack(fill="x", pady=(4, 0))
        self.progress = ttk.Progressbar(actions, mode="indeterminate")
        self.progress.pack(fill="x", pady=(6, 0))
        self.lbl_estimate = ttk.Label(actions, text="", foreground=MUTED,
                                      font=("TkDefaultFont", 8), wraplength=240,
                                      justify="left")
        self.lbl_estimate.pack(anchor="w", pady=(4, 0))

        # Any parameter change re-costs the build. Debounced, because
        # dragging a slider fires this on every pixel.
        for var in self._params.values():
            var.trace_add("write", lambda *_: self._schedule_estimate())

        self._update_radius_hint()
        self._update_strain_hint()
        self._update_coil_hint()
        self._update_surface_hint()
        self._update_cage_hint()
        self._on_mode_change()

    # ------------------------------------------------------------ visibility
    def _on_mode_change(self) -> None:
        """Show only the panels that apply to the selected structure type."""
        mode = self.var_mode_kind.get()
        for frame in (self.frame_tube, self.frame_centreline, self.frame_defects,
                      self.frame_coil, self.frame_junction, self.frame_schwarzite,
                      self.frame_cage, self.frame_mw, self.frame_bundle,
                      self.frame_surface, self.frame_chem):
            frame.pack_forget()

        if mode == "junction":
            self.frame_junction.pack(fill="x")
        elif mode == "schwarzite":
            self.frame_schwarzite.pack(fill="x")
            # Annealing is counterproductive on a minimal surface (it
            # stretches the bonds the 5-7 pairs were relieving), so
            # entering this mode turns the shared slider off rather than
            # letting its 80-sweep default quietly degrade the cell.
            self.var_anneal.set(0)
        elif mode == "coil (relaxed)":
            self.frame_coil.pack(fill="x")
        elif mode in ("fullerene", "nano-onion"):
            self.frame_cage.pack(fill="x")
            if mode == "nano-onion":
                self.frame_onion.grid()
            else:
                self.frame_onion.grid_remove()
            self._update_cage_hint()
        elif mode == "multi-wall":
            self.frame_tube.pack(fill="x")
            self.frame_mw.pack(fill="x", pady=(8, 0))
        elif mode == "bundle":
            self.frame_tube.pack(fill="x")
            self.frame_bundle.pack(fill="x", pady=(8, 0))
        else:
            self.frame_tube.pack(fill="x")
            self.frame_centreline.pack(fill="x", pady=(8, 0))
            self.frame_defects.pack(fill="x", pady=(8, 0))

        # The cages come from an exact seed polyhedron, so there is no
        # remeshing for flip annealing to clean up; showing the control
        # would imply an effect it cannot have.
        if mode not in ("fullerene", "nano-onion"):
            self.frame_surface.pack(fill="x", pady=(8, 0))
        self.frame_chem.pack(fill="x", pady=(8, 0))
        self._on_shape_change()
        self._schedule_estimate()

    def _on_shape_change(self) -> None:
        """Show the coil panel where coil dimensions actually apply."""
        if self.var_mode_kind.get() == "coil (relaxed)":
            self.frame_coil_tube.grid()
            return
        self.frame_coil_tube.grid_remove()
        if (self.var_shape.get() == "helix"
                and self.var_mode_kind.get() == "capped tube"):
            self.frame_coil.pack(fill="x", pady=(8, 0))
        else:
            self.frame_coil.pack_forget()

    # ----------------------------------------------------------------- hints
    def _update_radius_hint(self) -> None:
        radius = fm.radius_for_freq(int(self.var_freq.get()), self.var_bond.get())
        self.lbl_radius.config(
            text=f"→ tube radius ≈ {radius:.2f} Å  (lattice-quantised)"
        )

    def _update_strain_hint(self) -> None:
        value = float(self.var_max_strain.get())
        if value <= 0.10:
            text, colour = "physical sp2 regime", OK_GREEN
        elif value <= 0.15:
            text, colour = "strained but intact — fine for artwork", WARN_AMBER
        else:
            text, colour = "bonds stretch out of the sp2 range", BAD_RED
        self.lbl_strain.config(text=f"{value:.0%}: {text}", foreground=colour)

    def _update_cage_hint(self) -> None:
        """Say which named cage the current family/frequency gives."""
        from ..builders.fullerene import FAMILY_BASE_ATOMS

        family = self.var_cage_family.get()
        freq = int(self.var_cage_freq.get())
        base = FAMILY_BASE_ATOMS.get(family, 60)
        # Radius scales with frequency: ~3.52 Å per step for the C60
        # family, ~2.02 Å for C20 (measured on the relaxed cages).
        step = 3.52 if family == "C60" else 2.02
        text = f"C{base * freq**2}, radius ≈ {step * freq:.1f} Å."
        if self.var_mode_kind.get() == "nano-onion":
            shells = int(self.var_onion_shells.get())
            names = [f"C{base * (freq + k) ** 2}" for k in range(shells)]
            spacing = ("≈3.5 Å apart — graphitic" if family == "C60"
                       else "≈2.0 Å apart — too close to be physical; use C60")
            text = "@".join(names) + f", shells {spacing}."
        self.lbl_cage.config(text=text)

    def _update_surface_hint(self) -> None:
        anneal = int(self.var_anneal.get())
        rough = float(self.var_roughness.get())
        if anneal == 0:
            topo = "as-grown: keeps the stray 5-7 pairs the remesh leaves"
        elif anneal < 40:
            topo = "partly annealed"
        else:
            topo = "annealed: strays removed, only curvature-required rings"
        geom = ("ideally smooth" if rough <= 0
                else f"{rough:.2f} Å corrugation — CVD-like")
        colour = MUTED
        # On a minimal surface the 5-7 pairs are not disorder; they are how
        # the net covers the saddle. Annealing them away measurably
        # stretches the remaining bonds.
        if self.var_mode_kind.get() == "schwarzite" and anneal > 0:
            topo = (f"annealing hurts here — {anneal} sweeps stretches bonds; "
                    "the 5-7 pairs are how the net covers the saddle")
            colour = BAD_RED
        self.lbl_surface.config(text=f"{topo}; {geom}.", foreground=colour)

    def _update_coil_hint(self) -> None:
        from ..builders import centerline as cl

        radius = float(self.var_coil_radius.get())
        pitch = float(self.var_coil_pitch.get())
        turns = float(self.var_coil_turns.get())
        taper = float(self.var_coil_taper.get())
        arc = cl.helix_arc_length(radius, pitch, turns, taper=taper)

        if self.var_mode_kind.get() == "coil (relaxed)":
            # No strain budget applies: curvature is paid for in ring
            # topology, not bond stretch. What fails instead is the coil
            # closing on itself, so that is what the hint reports.
            tube_radius = float(self.var_coil_tube_radius.get())
            clearance = 2.0 * tube_radius + 3.4
            ok = pitch >= clearance
            self.lbl_coil.config(
                text=f"{arc:.0f} Å of tube; turns "
                     f"{pitch - 2 * tube_radius:.0f} Å apart"
                     + ("" if ok else
                        f" — needs ≥{clearance:.0f} Å pitch or the walls merge"),
                foreground=OK_GREEN if ok else BAD_RED,
            )
            return

        tube_radius = fm.radius_for_freq(int(self.var_freq.get()),
                                         float(self.var_bond.get()))
        # Taper-aware: a conical spring is judged at its tightest end.
        strain = tube_radius * cl.helix_curvature(radius, pitch, taper=taper)
        colour = (OK_GREEN if strain <= 0.10
                  else WARN_AMBER if strain <= cl.ARTISTIC_STRAIN_LIMIT
                  else BAD_RED)
        self.lbl_coil.config(
            text=f"{arc:.0f} Å of tube, wall strain {strain:.0%}"
                 + ("" if strain <= cl.ARTISTIC_STRAIN_LIMIT
                    else " — real nanocoils use much wider coils"),
            foreground=colour,
        )

    def _schedule_estimate(self) -> None:
        """Re-cost the build shortly, collapsing bursts of changes.

        A slider drag writes its variable on every pixel of travel; each
        write would otherwise rebuild a Job and re-run the estimate.
        """
        if self._estimate_job is not None:
            try:
                self.root.after_cancel(self._estimate_job)
            except (tk.TclError, ValueError):
                pass
        self._estimate_job = self.root.after(120, self._update_estimate)

    def _update_estimate(self) -> None:
        self._estimate_job = None
        try:
            severity, text = estimate_cost(self.current_job())
        except (tk.TclError, ValueError, KeyError) as exc:
            # A half-typed entry can make a job momentarily invalid; that
            # is not worth shouting about, only worth not crashing on.
            self.lbl_estimate.config(text=f"estimate unavailable ({exc})",
                                     foreground=MUTED)
            return
        colour = {"fast": OK_GREEN, "slow": WARN_AMBER,
                  "very slow": BAD_RED}.get(severity, MUTED)
        self.lbl_estimate.config(text=f"Estimate: {text}", foreground=colour)

    # ------------------------------------------------------------------ job
    def _current_defects(self) -> list[dict]:
        specs = []
        if self.var_n_sw.get():
            specs.append({"type": "stone_wales", "count": int(self.var_n_sw.get())})
        if self.var_n_dv.get():
            specs.append({"type": "divacancy", "count": int(self.var_n_dv.get())})
        return specs

    def current_job(self) -> Job:
        """The :class:`~nanocarbon_lab.jobs.Job` the controls describe.

        One place turns widgets into builder arguments. Building, costing
        and the copy-as-command-line button all go through here, so they
        cannot drift apart -- and because a Job is plain data, it is what
        gets handed to the worker process.
        """
        mode = self.var_mode_kind.get()
        dopant = self.var_dopant.get()
        common = dict(
            dopant=None if dopant == "none" else dopant,
            dopant_conc=float(self.var_dopant_conc.get()),
            seed=int(self.var_seed.get()),
        )

        if mode == "junction":
            params = dict(
                kind=self.var_j_kind.get(),
                tube_radius=float(self.var_j_radius.get()),
                arm_length=float(self.var_j_arm.get()),
                blend=float(self.var_j_blend.get()),
                anneal_sweeps=int(self.var_anneal.get()),
                roughness=float(self.var_roughness.get()),
            )
        elif mode == "schwarzite":
            params = dict(
                kind=self.var_s_kind.get(),
                cell=float(self.var_s_cell.get()),
                thickness=float(self.var_s_thickness.get()),
                anneal_sweeps=int(self.var_anneal.get()),
                roughness=float(self.var_roughness.get()),
            )
        elif mode == "coil (relaxed)":
            params = dict(
                coil_radius=float(self.var_coil_radius.get()),
                pitch=float(self.var_coil_pitch.get()),
                turns=float(self.var_coil_turns.get()),
                tube_radius=float(self.var_coil_tube_radius.get()),
                handedness=1 if self.var_coil_hand.get() == "right" else -1,
                taper=float(self.var_coil_taper.get()),
                bond=float(self.var_bond.get()),
                pin_ends=bool(self.var_pin_ends.get()),
                anneal_sweeps=int(self.var_anneal.get()),
                roughness=float(self.var_roughness.get()),
            )
        elif mode == "fullerene":
            params = dict(
                freq=int(self.var_cage_freq.get()),
                family=self.var_cage_family.get(),
                bond=float(self.var_bond.get()),
                roughness=float(self.var_roughness.get()),
            )
        elif mode == "nano-onion":
            params = dict(
                n_shells=int(self.var_onion_shells.get()),
                inner_freq=int(self.var_cage_freq.get()),
                family=self.var_cage_family.get(),
                bond=float(self.var_bond.get()),
                roughness=float(self.var_roughness.get()),
            )
        elif mode == "multi-wall":
            params = dict(
                n_shells=int(self.var_mw_shells.get()),
                inner_freq=int(self.var_mw_inner.get()),
                freq_step=int(self.var_mw_step.get()),
                n_body_rings=int(self.var_rings.get()),
                bond=float(self.var_bond.get()),
                roughness=float(self.var_roughness.get()),
            )
        elif mode == "bundle":
            params = dict(
                n_rings_across=int(self.var_bundle_shells.get()),
                freq=int(self.var_freq.get()),
                n_body_rings=int(self.var_rings.get()),
                gap=float(self.var_bundle_gap.get()),
                bond=float(self.var_bond.get()),
                roughness=float(self.var_roughness.get()),
            )
        else:
            shape = self.var_shape.get()
            params = dict(
                n_body_rings=int(self.var_rings.get()),
                freq=int(self.var_freq.get()),
                bond=float(self.var_bond.get()),
                bend_angle=float(self.var_bend.get()),
                shape=shape,
                helix_radius=(float(self.var_coil_radius.get())
                              if shape == "helix" else None),
                helix_pitch=float(self.var_coil_pitch.get()),
                helix_turns=float(self.var_coil_turns.get()),
                helix_handedness=1 if self.var_coil_hand.get() == "right" else -1,
                helix_taper=float(self.var_coil_taper.get()),
                roughness=float(self.var_roughness.get()),
                waviness=float(self.var_waviness.get()),
                max_strain=float(self.var_max_strain.get()),
                shape_points=int(self.var_shape_points.get()),
                defects=self._current_defects(),
            )
            # A bend and a swept shape are mutually exclusive in the
            # builder; sending both would raise where the user expects a
            # structure, so the shape wins and the bend is dropped.
            if shape != "straight":
                params["bend_angle"] = 0.0

        return Job(mode=mode, params=params, **common)

    # ---------------------------------------------------------------- build
    def on_build(self) -> None:
        if self._busy:
            return
        try:
            job = self.current_job()
        except (tk.TclError, ValueError) as exc:
            self._show_error("Invalid parameters", str(exc))
            return

        self._busy = True
        self.btn_build.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.progress.start(12)
        self._clear_error()
        _severity, cost = estimate_cost(job)
        self._set_status(f"Building {job.mode} ({cost})…")
        try:
            self.worker.submit(job)
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self._finish_build()
            self._show_error("Could not start the build process",
                             f"{exc}\n\n{traceback.format_exc()}")

    def on_cancel(self) -> None:
        """Stop the running build by killing the worker process."""
        if not self._busy:
            return
        degraded = self.worker.degraded
        self.worker.cancel()
        self._finish_build()
        self._set_status(
            "Build detached — the controls are usable again, but this "
            "environment could not start a separate process, so the work "
            "finishes in the background."
            if degraded else "Build cancelled."
        )

    def _finish_build(self) -> None:
        self._busy = False
        self.progress.stop()
        self.btn_build.config(state="normal")
        self.btn_cancel.config(state="disabled")

    def _poll_worker(self) -> None:
        """Collect a finished build, then always re-arm the timer.

        The re-arm is in a ``finally`` and the drawing is wrapped
        separately, because this callback is the only thing keeping the
        window responsive: an exception escaping it skips ``root.after``
        and polling stops for good, leaving the app permanently
        "building" with no way back. A single missing ``info`` key did
        exactly that once.
        """
        try:
            result = self.worker.poll()
            if result is not None:
                _job_id, kind, payload = result
                self._finish_build()
                if kind == "done":
                    self.atoms = payload
                    self.last_saved_stem = None
                    self._remember(payload)
                    try:
                        self._redraw()
                        self._update_info()
                        self._set_status("Build complete.")
                    except Exception:  # noqa: BLE001 - shown, never fatal
                        self._set_status("Built, but the display failed.")
                        self._show_error("Display failed",
                                         traceback.format_exc())
                else:
                    text, tb = payload
                    if text == WORKER_DIED:
                        self._set_status("Build process died.")
                        self._show_error(
                            "Build process died",
                            "The worker exited without returning a structure. "
                            "If you did not cancel it, this usually means it "
                            "ran out of memory — try fewer atoms.",
                        )
                    else:
                        self._set_status("Build failed — see the message below.")
                        self._show_error(text, tb)
        except Exception:  # noqa: BLE001 - the poll must never die
            self._set_status("Internal error while collecting the build.")
            self._show_error("Internal error", traceback.format_exc())
        finally:
            self.root.after(100, self._poll_worker)

    def on_roll_seed(self) -> None:
        """Pick a fresh random seed and rebuild.

        Defect placement, roughness and the random meander are all seeded,
        so this is the way to see a different sample of the same
        structure rather than a different structure.
        """
        import secrets

        self.var_seed.set(secrets.randbelow(1_000_000))
        self.on_build()

    # -------------------------------------------------------------- history
    def _remember(self, atoms) -> None:
        """Keep the most recent builds so a good one is not lost."""
        label = f"{len(self._history) + 1}. {self.var_mode_kind.get()} " \
                f"({len(atoms)} atoms)"
        self._history.append((label, atoms))
        del self._history[:-12]
        self.cmb_history["values"] = [name for name, _ in self._history]
        self.var_history.set(label)

    def on_restore_history(self) -> None:
        wanted = self.var_history.get()
        for label, atoms in self._history:
            if label == wanted:
                self.atoms = atoms
                self.last_saved_stem = None
                self._redraw()
                self._update_info()
                self._set_status(f"Restored {label}.")
                return

    # -------------------------------------------------------------- presets
    def apply_preset(self, name: str) -> None:
        """Set every parameter a preset names, then rebuild."""
        preset = PRESETS.get(name)
        if not preset:
            return
        self._apply_values(preset)
        self._set_status(f"Applied preset “{name}”.")
        self.on_build()

    def _apply_values(self, values: dict) -> None:
        for key, value in values.items():
            var = self._params.get(key)
            if var is None:
                continue  # a setting from a newer/older version: ignore it
            try:
                var.set(value)
            except tk.TclError:
                pass
        self._on_mode_change()

    def on_save_settings(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save parameters", defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialfile="nanocarbon_params.json",
        )
        if not path:
            return
        data = {key: var.get() for key, var in self._params.items()}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._set_status(f"Saved parameters to {Path(path).name}")

    def on_load_settings(self) -> None:
        path = filedialog.askopenfilename(
            title="Load parameters",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._show_error("Could not read that file", str(exc))
            return
        self._apply_values(data)
        self._set_status(f"Loaded parameters from {Path(path).name}")

    def on_copy_cli(self) -> None:
        """Put the equivalent ``nanocarbon`` command on the clipboard.

        The bridge from exploring by hand to a reproducible run: the same
        structure, scriptable, and short enough to paste into a methods
        section.
        """
        try:
            command = to_cli(self.current_job())
        except (tk.TclError, ValueError, KeyError) as exc:
            self._show_error("Could not build the command", str(exc))
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(command)
        self._set_status("Command copied to the clipboard.")
        self._show_error("Equivalent command line", command, error=False)

    # -------------------------------------------------------------- preview
    def _build_preview(self, parent: ttk.Frame) -> None:
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, parent).update()

        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(6, 0))
        self.var_show_bonds = tk.BooleanVar(value=True)
        self.var_colour_rings = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Bonds", variable=self.var_show_bonds,
                        command=self._redraw).pack(side="left")
        ttk.Checkbutton(bar, text="Colour by ring", variable=self.var_colour_rings,
                        command=self._redraw).pack(side="left", padx=(8, 0))

        # Ring filters. Hiding the hexagons is the fastest way to see where
        # the curvature actually went -- on a junction or a coil the 5s and
        # 7s are a handful of atoms buried in thousands.
        ttk.Label(bar, text="  show:").pack(side="left")
        self.var_ring_filter = {}
        for size in (5, 6, 7, 8):
            var = tk.BooleanVar(value=True)
            self.var_ring_filter[size] = var
            ttk.Checkbutton(bar, text=str(size), variable=var,
                            command=self._redraw).pack(side="left")
        self.lbl_preview = ttk.Label(bar, text="", foreground=MUTED,
                                     font=("TkDefaultFont", 8))
        self.lbl_preview.pack(side="right")

    def _ring_of_atom(self) -> list[int]:
        """Most informative ring size each atom belongs to."""
        n = len(self.atoms)
        best = [6] * n
        priority = {5: 3, 7: 2, 8: 1, 6: 0}
        for ring in self.atoms.info.get("rings", []):
            size = len(ring)
            for a in ring:
                if priority.get(size, 0) > priority.get(best[a], 0):
                    best[a] = size
        return best

    def _atom_colours(self, ring_of: list[int]) -> list[str]:
        if not self.var_colour_rings.get():
            return [RING_COLOURS[6]] * len(ring_of)
        return [RING_COLOURS.get(s, RING_COLOURS[6]) for s in ring_of]

    def _redraw(self) -> None:
        if self.atoms is None:
            return
        pos = self.atoms.get_positions()
        ring_of = self._ring_of_atom()
        keep = np.array([bool(self.var_ring_filter[s].get())
                         if s in self.var_ring_filter else True
                         for s in ring_of])
        self.ax.clear()

        note = ""
        if self.var_show_bonds.get():
            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            bonds = [(a, b) for a, b in self.atoms.info.get("bonds", [])
                     if keep[a] and keep[b]]
            if len(bonds) > PREVIEW_BOND_LIMIT:
                stride = len(bonds) // PREVIEW_BOND_LIMIT + 1
                bonds = bonds[::stride]
                note = f"showing 1 bond in {stride}"
            segs = [(pos[a], pos[b]) for a, b in bonds]
            if segs:
                self.ax.add_collection3d(
                    Line3DCollection(segs, colors="#9aa3ad", linewidths=0.7)
                )

        shown = np.flatnonzero(keep)
        if shown.size:
            colours = self._atom_colours(ring_of)
            self.ax.scatter(
                pos[shown, 0], pos[shown, 1], pos[shown, 2],
                c=[colours[i] for i in shown], s=16, depthshade=True,
            )
        hidden = len(pos) - shown.size
        self.lbl_preview.config(
            text=", ".join(filter(None, [
                f"{shown.size} of {len(pos)} atoms" if hidden else f"{len(pos)} atoms",
                note,
            ]))
        )

        # Equal aspect: an unscaled 3D axis stretches a long tube into a
        # blob and makes a coil look elliptical.
        span = (pos.max(axis=0) - pos.min(axis=0)).max() / 2.0 or 1.0
        mid = (pos.max(axis=0) + pos.min(axis=0)) / 2.0
        self.ax.set_xlim(mid[0] - span, mid[0] + span)
        self.ax.set_ylim(mid[1] - span, mid[1] + span)
        self.ax.set_zlim(mid[2] - span, mid[2] + span)
        self.ax.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------- readouts
    def _build_actions(self, parent: ttk.Frame) -> None:
        info = ttk.LabelFrame(parent, text="Structure", padding=8)
        info.pack(fill="x")
        self.txt_info = tk.Text(info, height=15, width=34, wrap="word",
                                font=("TkFixedFont", 9), relief="flat",
                                background=self.root.cget("background"))
        self.txt_info.pack(fill="x")
        self.txt_info.configure(state="disabled")

        hist = ttk.LabelFrame(parent, text="Session history", padding=8)
        hist.pack(fill="x", pady=(8, 0))
        self.var_history = tk.StringVar()
        self.cmb_history = ttk.Combobox(hist, textvariable=self.var_history,
                                        values=[], state="readonly")
        self.cmb_history.pack(fill="x")
        self.cmb_history.bind("<<ComboboxSelected>>",
                              lambda _e: self.on_restore_history())
        ttk.Label(hist, text="Every build this session; pick one to bring it "
                             "back into the preview.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=260,
                  justify="left").pack(anchor="w", pady=(4, 0))

        exp = ttk.LabelFrame(parent, text="Export", padding=8)
        exp.pack(fill="x", pady=(8, 0))
        ttk.Button(exp, text="Save .xyz + .json bundle…",
                   command=self.on_export).pack(fill="x", ipady=3)
        ttk.Button(exp, text="Copy equivalent CLI command",
                   command=self.on_copy_cli).pack(fill="x", pady=(4, 0))
        row = ttk.Frame(exp)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Save params…",
                   command=self.on_save_settings).pack(side="left", expand=True,
                                                       fill="x")
        ttk.Button(row, text="Load params…",
                   command=self.on_load_settings).pack(side="left", expand=True,
                                                       fill="x", padx=(4, 0))
        ttk.Label(exp, text="XYZ for any viewer; JSON carries bonds and ring "
                            "types for the Blender pipeline, plus a CIF for "
                            "periodic cells.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=260,
                  justify="left").pack(anchor="w", pady=(4, 0))

        ren = ttk.LabelFrame(parent, text="Blender render", padding=8)
        ren.pack(fill="x", pady=(8, 0))
        ttk.Label(ren, text="Style").pack(anchor="w")
        self.var_style = tk.StringVar(value=BLENDER_STYLES[0])
        ttk.Combobox(ren, textvariable=self.var_style, values=BLENDER_STYLES,
                     state="readonly").pack(fill="x", pady=(0, 6))
        ttk.Label(ren, text="Representation").pack(anchor="w")
        self.var_mode = tk.StringVar(value="ballstick")
        ttk.Combobox(ren, textvariable=self.var_mode,
                     values=["ballstick", "surface", "both"],
                     state="readonly").pack(fill="x", pady=(0, 6))
        ttk.Button(ren, text="Render with Blender…",
                   command=self.on_render).pack(fill="x", ipady=3)
        ttk.Button(ren, text="Locate Blender…",
                   command=self.on_locate_blender).pack(fill="x", pady=(4, 0))
        self.lbl_blender = ttk.Label(ren, text="", foreground=MUTED,
                                     font=("TkDefaultFont", 8), wraplength=260,
                                     justify="left")
        self.lbl_blender.pack(anchor="w", pady=(4, 0))
        self._check_blender()

        self.status = ttk.Label(parent, text="Ready", foreground="#555",
                                wraplength=280, justify="left")
        self.status.pack(anchor="w", pady=(10, 0))

        # Errors land here rather than in a modal dialog: a modal blocks
        # the event loop, and it throws away whatever the user was about
        # to change to fix the problem.
        self.frame_error = ttk.LabelFrame(parent, text="Message", padding=6)
        self.txt_error = tk.Text(self.frame_error, height=7, width=34,
                                 wrap="word", font=("TkFixedFont", 8),
                                 relief="flat")
        self.txt_error.pack(fill="both", expand=True)
        ttk.Button(self.frame_error, text="Dismiss",
                   command=self._clear_error).pack(fill="x", pady=(4, 0))

    def _show_error(self, title: str, detail: str, error: bool = True) -> None:
        """Put a message in the panel. Never opens a dialog.

        The text is left editable so a traceback can be selected and
        copied -- the whole point of showing one is that it can be pasted
        into a bug report.
        """
        self.frame_error.configure(text=title[:60])
        self.txt_error.delete("1.0", "end")
        self.txt_error.insert("1.0", detail)
        self.txt_error.configure(
            foreground=BAD_RED if error else "#333333")
        self.frame_error.pack(fill="both", expand=True, pady=(8, 0))

    def _clear_error(self) -> None:
        self.frame_error.pack_forget()

    def _update_info(self) -> None:
        a = self.atoms
        g = a.info["geometry"]
        counts = a.info["ring_counts"]
        rings_txt = "\n".join(
            f"  {RING_LABELS[s].split(' (')[0]:<10s} {counts.get(s, 0):>5d}"
            for s in (5, 6, 7, 8) if counts.get(s)
        )
        deficit = sum((6 - s) * c for s, c in counts.items())
        # Euler's budget is 12 per closed sphere-like shell. A schwarzite
        # with handles is legitimately negative (genus g gives 12(1-g)),
        # and an assembly of n disjoint shells owes 12 per shell.
        components = int(a.info.get("n_shells", a.info.get("n_tubes", 1)))
        expected = components * (12 - 12 * int(a.info.get("genus", 0)))
        clash = g["n_close_contacts"]

        lines = [f"atoms        {len(a):>6d}"]
        if "formula" in a.info:
            lines.append(f"formula      {a.info['formula']:>9s}")
        # Each field is reported on its own presence. Keying a whole block
        # off one field assumed a radius meant a tube, and a fullerene has
        # a radius but no length, shape or path strain.
        if "radius" in a.info:
            lines.append(f"radius       {a.info['radius']:>6.2f} Å")
        if "length" in a.info:
            lines.append(f"length       {a.info['length']:>6.1f} Å")
        if "shape" in a.info:
            lines.append(f"shape        {a.info['shape']:>6s}")
        if "path_strain" in a.info:
            lines.append(f"path strain  {a.info['path_strain']:>5.1%}")
        if "junction_kind" in a.info:
            lines.append(f"junction     {a.info['junction_kind']:>6s}")
        if "schwarzite_kind" in a.info:
            lines.append(f"surface      {a.info['schwarzite_kind']:>9s}")
        if "genus" in a.info:
            lines.append(f"genus        {a.info['genus']:>6d}")
        if all(a.get_pbc()):
            lines.append(f"periodic     {a.cell[0][0]:>6.1f} Å cell")
        if "n_shells" in a.info:
            lines.append(f"shells       {a.info['n_shells']:>6d}")
        # A multi-wall tube calls it wall spacing, an onion shell spacing --
        # a cage has no walls. Report whichever the builder recorded.
        for key, label in (("wall_spacing", "wall spacing"),
                           ("shell_spacing", "shell spacing")):
            if key in a.info:
                lines.append(f"{label:<12s} {a.info[key]:>6.2f} Å")
        if "n_tubes" in a.info:
            lines.append(f"tubes        {a.info['n_tubes']:>6d}")
        if "achieved_coil_radius" in a.info:
            lines.append(f"coil radius  {a.info['achieved_coil_radius']:>6.1f} Å")
            pitch = a.info.get("achieved_pitch", float("nan"))
            lines.append(
                "coil pitch      n/a (needs >1 turn)" if np.isnan(pitch)
                else f"coil pitch   {pitch:>6.1f} Å"
            )
        sep = a.info.get("geometry", {}).get("min_wall_separation")
        if sep is not None and not np.isnan(sep):
            lines.append(f"wall gap     {sep:>6.2f} Å")
        symbols = a.get_chemical_symbols()
        if set(symbols) != {"C"}:
            from collections import Counter

            by_element = Counter(symbols)
            lines.append("composition  " + " ".join(
                f"{el}{by_element[el]}" for el in sorted(by_element)))

        lines += [
            "",
            "rings",
            rings_txt,
            (f"  Euler sum  {deficit:>5d}  "
             f"{'OK' if deficit == expected else 'BROKEN'}"),
            "",
            "geometry",
            f"  bond   {g['bond_min']:.3f}–{g['bond_max']:.3f} Å",
            f"  angle  {g['angle_min']:.1f}–{g['angle_max']:.1f}°",
            f"  contacts <2Å  {clash}  {'OK' if clash == 0 else 'CHECK'}",
        ]
        # Zero clashes is not the same as a physical structure: an
        # over-tight coil keeps its atoms apart while stretching its bonds
        # past any real C-C. Say so in words, next to the numbers.
        verdict, why = sp2_quality(g)
        lines += ["", f"sp2 verdict  {verdict.upper()}", f"  {why}"]

        self.txt_info.configure(state="normal")
        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("1.0", "\n".join(lines))
        self.txt_info.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status.config(text=text)

    # ---------------------------------------------------------------- export
    def on_export(self) -> Path | None:
        if self.atoms is None:
            self._show_error("Nothing to export", "Build a structure first.")
            return None
        path = filedialog.asksaveasfilename(
            title="Save render bundle",
            defaultextension=".xyz",
            filetypes=[("XYZ structure", "*.xyz"), ("All files", "*.*")],
            initialfile=f"{self.var_mode_kind.get().replace(' ', '_')}.xyz",
        )
        if not path:
            return None
        stem = Path(path).with_suffix("")
        xyz_path, json_path = write_render_bundle(self.atoms, stem)
        self.last_saved_stem = stem
        extra = " (+ .cif)" if all(self.atoms.get_pbc()) else ""
        self._set_status(f"Saved {xyz_path.name} and {json_path.name}{extra}")
        return stem

    # --------------------------------------------------------------- blender
    def _check_blender(self) -> None:
        if self.blender_exe and Path(self.blender_exe).exists():
            return  # a path the user picked by hand wins over auto-detection
        self.blender_exe = find_blender()
        if self.blender_exe:
            self.lbl_blender.config(text=f"Found: {self.blender_exe}")
        else:
            self.lbl_blender.config(
                text="Blender not found automatically — use “Locate Blender…” "
                     "or export the bundle and run blender/render_cnt.py yourself."
            )

    def on_locate_blender(self) -> None:
        """Let the user point at blender.exe / the Blender binary directly.

        Needed mainly on Windows, where the installer does not add Blender
        to ``PATH``, and for portable/Steam installations anywhere.
        """
        if os.name == "nt":
            types = [("Blender executable", "blender.exe"), ("All files", "*.*")]
        else:
            types = [("All files", "*.*")]
        path = filedialog.askopenfilename(title="Locate the Blender executable",
                                          filetypes=types)
        if not path:
            return
        self.blender_exe = path
        self.lbl_blender.config(text=f"Using: {path}")
        self._set_status("Blender location set for this session.")

    def on_render(self) -> None:
        if self.atoms is None:
            self._show_error("Nothing to render", "Build a structure first.")
            return
        self._check_blender()
        if not self.blender_exe:
            self._show_error(
                "Blender not found",
                "Could not find Blender automatically.\n\n"
                "Click “Locate Blender…” and point at the executable (on "
                "Windows, usually\n"
                r"C:\Program Files\Blender Foundation\Blender 4.x\blender.exe)"
                ",\n\nor export the bundle and run it yourself:\n"
                "  blender -b -P blender/render_cnt.py -- --xyz <file>.xyz "
                "--json <file>.json --style <style> --out <image>.png",
            )
            return

        stem = self.last_saved_stem or self.on_export()
        if stem is None:
            return
        out_png = filedialog.asksaveasfilename(
            title="Save rendered image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=f"{stem.name}_{self.var_style.get()}.png",
        )
        if not out_png:
            return

        script = Path(__file__).resolve().parents[2] / "blender" / "render_cnt.py"
        if not script.exists():
            self._show_error(
                "Render script missing",
                f"Could not find {script}. Run the GUI from a full checkout of "
                "the project, or invoke Blender manually.",
            )
            return

        cmd = [
            self.blender_exe, "-b", "-P", str(script), "--",
            "--xyz", str(stem.with_suffix(".xyz")),
            "--json", str(stem.with_suffix(".json")),
            "--style", self.var_style.get(),
            "--mode", self.var_mode.get(),
            "--out", out_png,
        ]
        self._set_status("Rendering in Blender (this can take a while)…")
        self.root.update_idletasks()

        def render_worker():
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=3600, check=False)
                ok = proc.returncode == 0 and Path(out_png).exists()
                self._queue_render_result(ok, proc.stderr or proc.stdout, out_png)
            except Exception as exc:  # noqa: BLE001 - reported to the user
                self._queue_render_result(False, str(exc), out_png)

        threading.Thread(target=render_worker, daemon=True).start()

    def _queue_render_result(self, ok: bool, log: str, out_png: str) -> None:
        def report():
            if ok:
                self._set_status(f"Rendered → {Path(out_png).name}")
                self._show_error("Render complete", f"Wrote {out_png}", error=False)
            else:
                self._set_status("Render failed — see the message.")
                self._show_error("Render failed", log[-3000:] or "Unknown error.")

        self.root.after(0, report)

    # ----------------------------------------------------------------- close
    def on_close(self) -> None:
        """Shut the worker down before the window goes away.

        Without this the build subprocess outlives the GUI: it is a daemon
        of the parent interpreter, but the parent does not exit until Tk's
        main loop returns, and a half-finished coil would keep a core busy
        with nothing to report to.
        """
        try:
            self.worker.shutdown()
        finally:
            self.root.destroy()


def main() -> int:
    """Entry point for ``nanocarbon-gui``."""
    # Required before any process is spawned when this is bundled into a
    # frozen executable (PyInstaller and friends); a no-op otherwise.
    multiprocessing.freeze_support()
    root = tk.Tk()
    NanocarbonGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
