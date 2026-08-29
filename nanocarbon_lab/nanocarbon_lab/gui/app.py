"""Tkinter GUI for building capped/defected nanotubes and exporting them.

Launch with ``nanocarbon-gui`` (installed entry point) or
``python -m nanocarbon_lab.gui``.

The window is three columns: build parameters and a defect list on the
left, a live 3D preview in the middle, and export / render actions plus a
geometry-quality readout on the right. Building runs on a worker thread so
the interface stays responsive on the larger structures (a 3500-atom shell
takes a few seconds to relax).

Only the standard library's :mod:`tkinter` plus :mod:`matplotlib` are
used, so there is no heavy GUI dependency; matplotlib's 3D axes provide
the preview. ``tkinter`` ships with the python.org installers on Windows
and macOS; on Linux distributions it is usually a separate package
(``apt install python3-tk``, ``dnf install python3-tkinter``).
"""

from __future__ import annotations

import glob
import os
import queue
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "The nanocarbon_lab GUI needs tkinter, which is not available in this "
        "Python installation.\n"
        "  Debian/Ubuntu : sudo apt install python3-tk\n"
        "  Fedora/RHEL   : sudo dnf install python3-tkinter\n"
        "  macOS/Windows : use the python.org installer (tkinter is included)\n"
        "You can still build structures without a GUI via the 'nanocarbon "
        "cnt-cap' command."
    ) from exc

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

from ..builders import (
    build_bundle,
    build_capped_cnt,
    build_coil,
    build_fullerene,
    build_junction,
    build_multiwall_cnt,
    build_nano_onion,
    build_schwarzite,
)
from ..dopants import dope_random
from ..builders import fullerene_mesh as fm
from ..exports.xyz import write_render_bundle
from ..validation.quality import sp2_quality

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
MODES = ["capped tube", "coil (relaxed)", "fullerene", "nano-onion",
         "junction", "schwarzite", "multi-wall", "bundle"]
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


class NanocarbonGUI:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("nanocarbon_lab — capped nanotube builder")
        self.root.geometry("1280x780")
        self.root.minsize(1050, 660)

        self.atoms = None
        self.last_saved_stem: Path | None = None
        self._queue: queue.Queue = queue.Queue()
        self._busy = False

        self._build_widgets()
        self._poll_queue()
        self.on_build()  # start with something on screen

    # ---------------------------------------------------------------- layout
    def _build_widgets(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="y", padx=(0, 8))
        centre = ttk.Frame(outer)
        centre.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(outer, width=290)
        right.pack(side="left", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        self._build_params(left)
        self._build_preview(centre)
        self._build_actions(right)

    def _build_params(self, parent: ttk.Frame) -> None:
        self.var_mode_kind = tk.StringVar(value=MODES[0])
        mode_box = ttk.LabelFrame(parent, text="Structure type", padding=8)
        mode_box.pack(fill="x", pady=(0, 8))
        ttk.Combobox(mode_box, textvariable=self.var_mode_kind, values=MODES,
                     state="readonly").pack(fill="x")
        self.var_mode_kind.trace_add("write", lambda *_: self._on_mode_change())

        box = ttk.LabelFrame(parent, text="Geometry", padding=8)
        box.pack(fill="x")
        self.frame_tube = box

        self.var_rings = tk.IntVar(value=8)
        self.var_freq = tk.IntVar(value=3)
        self.var_bond = tk.DoubleVar(value=1.42)
        self.var_bend = tk.DoubleVar(value=0.0)
        self.var_seed = tk.IntVar(value=0)
        self.var_shape = tk.StringVar(value="straight")
        self.var_waviness = tk.DoubleVar(value=0.7)
        self.var_max_strain = tk.DoubleVar(value=0.08)
        self.var_shape_points = tk.IntVar(value=9)
        self.var_coil_radius = tk.DoubleVar(value=60.0)
        self.var_coil_pitch = tk.DoubleVar(value=25.0)
        self.var_coil_turns = tk.DoubleVar(value=1.5)
        self.var_coil_hand = tk.StringVar(value="right")
        self.var_coil_taper = tk.DoubleVar(value=1.0)
        self.var_coil_tube_radius = tk.DoubleVar(value=6.0)
        self.var_pin_ends = tk.BooleanVar(value=False)
        self.var_anneal = tk.IntVar(value=80)
        self.var_roughness = tk.DoubleVar(value=0.0)
        self.var_dopant = tk.StringVar(value="none")
        self.var_dopant_conc = tk.DoubleVar(value=0.03)
        self.var_mw_shells = tk.IntVar(value=2)
        self.var_mw_inner = tk.IntVar(value=3)
        self.var_bundle_shells = tk.IntVar(value=1)
        self.var_cage_family = tk.StringVar(value="C60")
        self.var_cage_freq = tk.IntVar(value=1)
        self.var_onion_shells = tk.IntVar(value=3)

        self.lbl_radius = ttk.Label(box, text="", foreground="#2e86ab")

        self._slider(box, "Body rings (length)", self.var_rings, 2, 30, 0, integer=True)
        self._slider(box, "Subdivision freq (diameter)", self.var_freq, 1, 8, 2,
                     integer=True, command=self._update_radius_hint)
        self.lbl_radius.grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self._slider(box, "Bend angle (rad)", self.var_bend, 0.0, 1.0, 5, resolution=0.05)
        self._slider(box, "C–C bond (Å)", self.var_bond, 1.30, 1.55, 7, resolution=0.01)

        row = ttk.Frame(box)
        row.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(row, text="Random seed").pack(side="left")
        ttk.Spinbox(row, from_=0, to=9999, textvariable=self.var_seed, width=8).pack(
            side="right"
        )

        # --- centreline shape
        sbox = ttk.LabelFrame(parent, text="Centreline", padding=8)
        sbox.pack(fill="x", pady=(8, 0))
        self.frame_centreline = sbox
        # This frame is laid out with grid throughout: _slider grids, and
        # tkinter forbids mixing grid and pack in the same container.
        sbox.columnconfigure(0, weight=1)
        ttk.Label(sbox, text="Shape").grid(row=0, column=0, sticky="w")
        ttk.Combobox(sbox, textvariable=self.var_shape, values=SHAPES,
                     state="readonly", width=11).grid(row=0, column=1, sticky="e",
                                                      pady=(0, 6))
        self._slider(sbox, "Waviness", self.var_waviness, 0.0, 1.0, 1, resolution=0.05)
        self._slider(sbox, "Control points", self.var_shape_points, 4, 20, 3,
                     integer=True)
        self._slider(sbox, "Strain budget", self.var_max_strain, 0.02, 0.25, 5,
                     resolution=0.01, command=self._update_strain_hint)
        self.lbl_strain = ttk.Label(sbox, text="", foreground="#777",
                                    font=("TkDefaultFont", 8), wraplength=230,
                                    justify="left")
        self.lbl_strain.grid(row=7, column=0, columnspan=2, sticky="w")

        self.var_shape.trace_add("write", lambda *_: self._on_shape_change())
        ttk.Label(sbox, text="Thinner + longer tubes curve more at the same "
                             "strain — lower the frequency and raise the rings.",
                  foreground="#777", font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=8, column=0, columnspan=2, sticky="w",
                                       pady=(4, 0))

        # Coil dimensions, in real Å. Shared by the swept helix (shape=
        # "helix", where they size the tube) and the relaxed-coil mode
        # (where they size the implicit surface), so this lives at the top
        # level rather than inside the centreline frame.
        self.frame_coil = ttk.LabelFrame(parent, text="Coil", padding=8)
        self.frame_coil.pack(fill="x", pady=(8, 0))
        self.frame_coil.columnconfigure(0, weight=1)
        self._slider(self.frame_coil, "Coil radius (Å)", self.var_coil_radius,
                     15.0, 200.0, 0, resolution=5.0, command=self._update_coil_hint)
        self._slider(self.frame_coil, "Coil pitch (Å)", self.var_coil_pitch,
                     5.0, 100.0, 2, resolution=5.0, command=self._update_coil_hint)
        self._slider(self.frame_coil, "Turns", self.var_coil_turns,
                     0.5, 5.0, 4, resolution=0.5, command=self._update_coil_hint)
        self._slider(self.frame_coil, "Taper (end/start R)", self.var_coil_taper,
                     0.3, 2.0, 6, resolution=0.1)
        hand = ttk.Frame(self.frame_coil)
        hand.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(2, 2))
        ttk.Label(hand, text="Handedness").pack(side="left")
        ttk.Combobox(hand, textvariable=self.var_coil_hand,
                     values=["right", "left"], state="readonly",
                     width=7).pack(side="right")
        self.lbl_coil = ttk.Label(self.frame_coil, text="", foreground="#777",
                                  font=("TkDefaultFont", 8), wraplength=225,
                                  justify="left")
        self.lbl_coil.grid(row=9, column=0, columnspan=2, sticky="w")
        # Only the relaxed-coil mode sets the tube radius freely; the swept
        # helix takes its radius from the lattice-quantised frequency.
        self.frame_coil_tube = ttk.Frame(self.frame_coil)
        self.frame_coil_tube.grid(row=10, column=0, columnspan=2, sticky="ew")
        self.frame_coil_tube.columnconfigure(0, weight=1)
        self._slider(self.frame_coil_tube, "Tube radius (Å)",
                     self.var_coil_tube_radius, 4.0, 12.0, 0, resolution=0.5)
        ttk.Checkbutton(
            self.frame_coil_tube, text="Pin ends (hold the pitch)",
            variable=self.var_pin_ends,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 2))
        ttk.Label(self.frame_coil_tube,
                  text="Rings fix curvature but not torsion, so a free coil "
                       "keeps its radius and springs open in pitch. Pinning "
                       "holds the pitch at the cost of a strained network.",
                  foreground="#777", font=("TkDefaultFont", 8), wraplength=225,
                  justify="left").grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(self.frame_coil_tube,
                  text="Rings follow the curvature here: pentagons on the inner "
                       "wall, heptagons on the outer, so the bonds stay "
                       "graphitic instead of stretching. Slower to build.",
                  foreground="#777", font=("TkDefaultFont", 8), wraplength=225,
                  justify="left").grid(row=4, column=0, columnspan=2, sticky="w")

        # --- defects
        dbox = ttk.LabelFrame(parent, text="Defects", padding=8)
        dbox.pack(fill="x", pady=(8, 0))
        self.frame_defects = dbox

        self.var_n_sw = tk.IntVar(value=0)
        self.var_n_dv = tk.IntVar(value=0)
        for label, var, hint in (
            ("Stone–Wales (5-7-7-5)", self.var_n_sw, "2 pentagons + 2 heptagons each"),
            ("Divacancy (5-8-5)", self.var_n_dv, "1 octagon + 2 pentagons each"),
        ):
            r = ttk.Frame(dbox)
            r.pack(fill="x", pady=2)
            ttk.Label(r, text=label).pack(side="left")
            ttk.Spinbox(r, from_=0, to=12, textvariable=var, width=5).pack(side="right")
            ttk.Label(dbox, text=hint, foreground="#777", font=("TkDefaultFont", 8)).pack(
                anchor="w", pady=(0, 4)
            )

        # --- display
        vbox = ttk.LabelFrame(parent, text="Preview", padding=8)
        vbox.pack(fill="x", pady=(8, 0))
        self.var_show_bonds = tk.BooleanVar(value=True)
        self.var_colour_rings = tk.BooleanVar(value=True)
        ttk.Checkbutton(vbox, text="Draw bonds", variable=self.var_show_bonds,
                        command=self._redraw).pack(anchor="w")
        ttk.Checkbutton(vbox, text="Colour by ring type", variable=self.var_colour_rings,
                        command=self._redraw).pack(anchor="w")

        # --- junction panel (shown only in junction mode)
        self.frame_junction = ttk.LabelFrame(parent, text="Junction", padding=8)
        self.var_j_kind = tk.StringVar(value="Y")
        self.var_j_radius = tk.DoubleVar(value=6.0)
        self.var_j_arm = tk.DoubleVar(value=22.0)
        self.var_j_blend = tk.DoubleVar(value=4.0)
        self.frame_junction.columnconfigure(0, weight=1)
        ttk.Label(self.frame_junction, text="Kind").grid(row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_junction, textvariable=self.var_j_kind,
                     values=JUNCTION_KINDS, state="readonly", width=9).grid(
            row=0, column=1, sticky="e", pady=(0, 6))
        self._slider(self.frame_junction, "Arm radius (Å)", self.var_j_radius,
                     3.0, 10.0, 1, resolution=0.5)
        self._slider(self.frame_junction, "Arm length (Å)", self.var_j_arm,
                     12.0, 120.0, 3, resolution=2.0)
        self._slider(self.frame_junction, "Neck blend (Å)", self.var_j_blend,
                     1.0, 8.0, 5, resolution=0.5)
        ttk.Label(self.frame_junction,
                  text="Heptagons appear at the neck on their own: the branch is "
                       "a saddle, and saddles carry negative curvature.",
                  foreground="#777", font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=7, column=0, columnspan=2, sticky="w")

        # --- schwarzite panel
        self.frame_schwarzite = ttk.LabelFrame(parent, text="Schwarzite", padding=8)
        self.var_s_kind = tk.StringVar(value="primitive")
        self.var_s_cell = tk.DoubleVar(value=36.0)
        self.frame_schwarzite.columnconfigure(0, weight=1)
        ttk.Label(self.frame_schwarzite, text="Surface").grid(row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_schwarzite, textvariable=self.var_s_kind,
                     values=SCHWARZITE_KINDS, state="readonly", width=10).grid(
            row=0, column=1, sticky="e", pady=(0, 6))
        self._slider(self.frame_schwarzite, "Cell length (Å)", self.var_s_cell,
                     30.0, 56.0, 1, resolution=2.0)
        ttk.Label(self.frame_schwarzite,
                  text="A periodic unit cell: tubes run out of one face and "
                       "back in the opposite one. Saddles everywhere, so "
                       "heptagons outnumber pentagons. Bigger cells curve more "
                       "gently and relax cleaner — minimum 30 Å primitive, "
                       "36 gyroid and diamond.",
                  foreground="#777", font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=3, column=0, columnspan=2, sticky="w")

        # --- surface finish: applies to every structure type
        self.frame_surface = ttk.LabelFrame(parent, text="Surface finish", padding=8)
        self.frame_surface.columnconfigure(0, weight=1)
        self._slider(self.frame_surface, "Smoothing (anneal)", self.var_anneal,
                     0, 200, 0, integer=True, command=self._update_surface_hint)
        self._slider(self.frame_surface, "Roughness (Å)", self.var_roughness,
                     0.0, 0.6, 2, resolution=0.05, command=self._update_surface_hint)
        self.lbl_surface = ttk.Label(self.frame_surface, text="", foreground="#777",
                                     font=("TkDefaultFont", 8), wraplength=230,
                                     justify="left")
        self.lbl_surface.grid(row=4, column=0, columnspan=2, sticky="w")

        # --- chemistry
        self.frame_chem = ttk.LabelFrame(parent, text="Chemistry", padding=8)
        self.frame_chem.columnconfigure(0, weight=1)
        ttk.Label(self.frame_chem, text="Dopant").grid(row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_chem, textvariable=self.var_dopant, values=DOPANTS,
                     state="readonly", width=7).grid(row=0, column=1, sticky="e")
        self._slider(self.frame_chem, "Concentration", self.var_dopant_conc,
                     0.0, 0.15, 1, resolution=0.01)

        # --- multi-wall
        self.frame_mw = ttk.LabelFrame(parent, text="Multi-wall", padding=8)
        self.frame_mw.columnconfigure(0, weight=1)
        self._slider(self.frame_mw, "Shells", self.var_mw_shells, 1, 6, 0, integer=True)
        self._slider(self.frame_mw, "Inner freq", self.var_mw_inner, 1, 6, 2, integer=True)
        ttk.Label(self.frame_mw,
                  text="Walls land ~3.9 Å apart: the lattice quantises radius in "
                       "~1.96 Å steps, so it cannot hit graphite's 3.4 Å exactly.",
                  foreground="#777", font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=4, column=0, columnspan=2, sticky="w")

        # --- bundle
        self.frame_bundle = ttk.LabelFrame(parent, text="Bundle", padding=8)
        self.frame_bundle.columnconfigure(0, weight=1)
        self._slider(self.frame_bundle, "Hex shells", self.var_bundle_shells,
                     0, 3, 0, integer=True)
        ttk.Label(self.frame_bundle,
                  text="0 / 1 / 2 / 3 shells give 1 / 7 / 19 / 37 tubes, packed "
                       "on a triangular lattice at the 3.4 Å van der Waals gap.",
                  foreground="#777", font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=2, column=0, columnspan=2, sticky="w")

        # --- fullerene cage / nano-onion
        self.frame_cage = ttk.LabelFrame(parent, text="Cage", padding=8)
        self.frame_cage.columnconfigure(0, weight=1)
        ttk.Label(self.frame_cage, text="Family").grid(row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_cage, textvariable=self.var_cage_family,
                     values=CAGE_FAMILIES, state="readonly", width=7).grid(
            row=0, column=1, sticky="e", pady=(0, 6))
        self.var_cage_family.trace_add("write", lambda *_: self._update_cage_hint())
        self._slider(self.frame_cage, "Frequency (size)", self.var_cage_freq,
                     1, 6, 1, integer=True, command=self._update_cage_hint)
        self.frame_onion = ttk.Frame(self.frame_cage)
        self.frame_onion.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.frame_onion.columnconfigure(0, weight=1)
        self._slider(self.frame_onion, "Shells", self.var_onion_shells,
                     1, 5, 0, integer=True, command=self._update_cage_hint)
        self.lbl_cage = ttk.Label(self.frame_cage, text="", foreground="#777",
                                  font=("TkDefaultFont", 8), wraplength=230,
                                  justify="left")
        self.lbl_cage.grid(row=5, column=0, columnspan=2, sticky="w")

        self.btn_build = ttk.Button(parent, text="Build structure", command=self.on_build)
        self.btn_build.pack(fill="x", pady=(12, 0), ipady=4)
        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.pack(fill="x", pady=(6, 0))
        self._update_radius_hint()
        self._update_strain_hint()
        self._update_coil_hint()
        self._update_surface_hint()
        self._on_shape_change()
        self._on_mode_change()

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
            # stretches the bonds that the 5-7 pairs were relieving), so
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
        self.frame_surface.pack(fill="x", pady=(8, 0))
        self.frame_chem.pack(fill="x", pady=(8, 0))
        self._on_shape_change()

    def _update_cage_hint(self) -> None:
        """Say which cage the current family/frequency actually gives.

        The size is not a free number: subdividing the seed by ``f``
        multiplies the atom count by ``f**2``, so the reachable cages are
        a discrete series. Naming them beats leaving the user to guess
        what "frequency 3" means.
        """
        from ..builders.fullerene import FAMILY_BASE_ATOMS

        family = self.var_cage_family.get()
        freq = int(self.var_cage_freq.get())
        base = FAMILY_BASE_ATOMS.get(family, 60)
        atoms = base * freq**2
        # Radius scales with the frequency: ~3.52 Å per step for the C60
        # family, ~2.0 Å for C20 (measured on the relaxed cages).
        step = 3.52 if family == "C60" else 2.02
        text = f"C{atoms}, radius ≈ {step * freq:.1f} Å."
        if self.var_mode_kind.get() == "nano-onion":
            shells = int(self.var_onion_shells.get())
            names = [f"C{base * (freq + k) ** 2}" for k in range(shells)]
            spacing = "≈3.5 Å apart — graphitic" if family == "C60" else (
                "≈2.0 Å apart — too close to be physical; use C60"
            )
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
        geom = "ideally smooth" if rough <= 0 else (
            f"{rough:.2f} Å corrugation — CVD-like"
        )
        colour = "#777"
        # On a minimal surface the 5-7 pairs are not disorder; they are how
        # the net covers the saddle. Annealing them away measurably
        # stretches the remaining bonds, so warn rather than let the shared
        # slider quietly degrade the structure.
        if self.var_mode_kind.get() == "schwarzite" and anneal > 0:
            topo = (f"annealing hurts here — {anneal} sweeps stretches bonds; "
                    "the 5-7 pairs are how the net covers the saddle")
            colour = "#b3261e"
        self.lbl_surface.config(text=f"{topo}; {geom}.", foreground=colour)

    def _slider(self, parent, label, var, lo, hi, row, *, integer=False,
                resolution=None, command=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        value_lbl = ttk.Label(parent, width=6, anchor="e")
        value_lbl.grid(row=row, column=1, sticky="e")

        def on_move(_evt=None, notify: bool = True):
            if integer:
                var.set(int(round(var.get())))
                value_lbl.config(text=str(var.get()))
            else:
                step = resolution or 0.01
                var.set(round(round(var.get() / step) * step, 4))
                value_lbl.config(text=f"{var.get():.2f}")
            if command and notify:
                command()

        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var,
                          orient="horizontal", command=lambda _v: on_move())
        scale.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        parent.columnconfigure(0, weight=1)
        # Seed the value label without firing `command`: callbacks touch
        # widgets that may not exist yet while the panel is still being built.
        on_move(notify=False)

    def _on_shape_change(self) -> None:
        """Show the coil panel where coil dimensions actually apply.

        That is the swept helix and the relaxed-coil mode; the tube-radius
        slider inside it is for the relaxed coil alone, because the swept
        tube's radius is fixed by the lattice-quantised frequency.
        """
        relaxed = self.var_mode_kind.get() == "coil (relaxed)"
        if relaxed:
            self.frame_coil_tube.grid()
            return
        self.frame_coil_tube.grid_remove()
        if self.var_shape.get() == "helix" and self.var_mode_kind.get() == "capped tube":
            self.frame_coil.pack(fill="x", pady=(8, 0))
        else:
            self.frame_coil.pack_forget()

    def _update_coil_hint(self) -> None:
        from ..builders import centerline as cl

        radius = float(self.var_coil_radius.get())
        pitch = float(self.var_coil_pitch.get())
        turns = float(self.var_coil_turns.get())
        taper = float(self.var_coil_taper.get())
        arc = cl.helix_arc_length(radius, pitch, turns, taper=taper)

        if self.var_mode_kind.get() == "coil (relaxed)":
            # No strain budget applies: the curvature is paid for in ring
            # topology, not bond stretch. What can fail instead is the coil
            # closing on itself, so that is what the hint reports.
            tube_radius = float(self.var_coil_tube_radius.get())
            clearance = 2.0 * tube_radius + 3.4
            ok = pitch >= clearance
            self.lbl_coil.config(
                text=f"{arc:.0f} Å of tube; turns {pitch - 2 * tube_radius:.0f} Å apart"
                     + ("" if ok else
                        f" — needs at least {clearance:.0f} Å pitch or the walls merge"),
                foreground="#2e7d32" if ok else "#b3261e",
            )
            return

        tube_radius = fm.radius_for_freq(int(self.var_freq.get()),
                                         float(self.var_bond.get()))
        # Taper-aware: a conical spring is judged at its tightest end.
        strain = tube_radius * cl.helix_curvature(radius, pitch, taper=taper)
        colour = "#2e7d32" if strain <= 0.10 else (
            "#b26a00" if strain <= cl.ARTISTIC_STRAIN_LIMIT else "#b3261e")
        self.lbl_coil.config(
            text=f"{arc:.0f} Å of tube, wall strain {strain:.0%}"
                 + ("" if strain <= cl.ARTISTIC_STRAIN_LIMIT
                    else " — real nanocoils use much wider coils"),
            foreground=colour,
        )

    def _update_strain_hint(self) -> None:
        value = float(self.var_max_strain.get())
        if value <= 0.10:
            text, colour = "physical sp2 regime", "#2e7d32"
        elif value <= 0.15:
            text, colour = "strained but intact — fine for artwork", "#b26a00"
        else:
            text, colour = "bonds stretch out of the sp2 range", "#b3261e"
        self.lbl_strain.config(text=f"{value:.0%}: {text}", foreground=colour)

    def _build_preview(self, parent: ttk.Frame) -> None:
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, parent).update()

    def _build_actions(self, parent: ttk.Frame) -> None:
        info = ttk.LabelFrame(parent, text="Structure", padding=8)
        info.pack(fill="x")
        self.txt_info = tk.Text(info, height=13, width=34, wrap="word",
                                font=("TkFixedFont", 9), relief="flat",
                                background=self.root.cget("background"))
        self.txt_info.pack(fill="x")
        self.txt_info.configure(state="disabled")

        exp = ttk.LabelFrame(parent, text="Export", padding=8)
        exp.pack(fill="x", pady=(8, 0))
        ttk.Button(exp, text="Save .xyz + .json bundle…",
                   command=self.on_export).pack(fill="x", ipady=3)
        ttk.Label(exp, text="XYZ for any viewer; JSON carries bonds\n"
                            "and ring types for the Blender pipeline.",
                  foreground="#777", font=("TkDefaultFont", 8),
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
        self.lbl_blender = ttk.Label(ren, text="", foreground="#777",
                                     font=("TkDefaultFont", 8), wraplength=250,
                                     justify="left")
        self.lbl_blender.pack(anchor="w", pady=(4, 0))
        self._check_blender()

        self.status = ttk.Label(parent, text="Ready", foreground="#555",
                                wraplength=270, justify="left")
        self.status.pack(anchor="w", pady=(10, 0))

    # ----------------------------------------------------------------- build
    def _current_defects(self) -> list[dict]:
        specs = []
        if self.var_n_sw.get():
            specs.append({"type": "stone_wales", "count": int(self.var_n_sw.get())})
        if self.var_n_dv.get():
            specs.append({"type": "divacancy", "count": int(self.var_n_dv.get())})
        return specs

    def _update_radius_hint(self) -> None:
        radius = fm.radius_for_freq(int(self.var_freq.get()), self.var_bond.get())
        self.lbl_radius.config(
            text=f"→ tube radius ≈ {radius:.2f} Å  (lattice-quantised)"
        )

    def on_build(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.btn_build.config(state="disabled")
        self.progress.start(12)
        self._set_status("Building and relaxing…")

        mode = self.var_mode_kind.get()
        if mode == "junction":
            builder, kwargs = build_junction, dict(
                kind=self.var_j_kind.get(),
                tube_radius=float(self.var_j_radius.get()),
                arm_length=float(self.var_j_arm.get()),
                blend=float(self.var_j_blend.get()),
                anneal_sweeps=int(self.var_anneal.get()),
                roughness=float(self.var_roughness.get()),
                seed=int(self.var_seed.get()),
            )
        elif mode == "schwarzite":
            builder, kwargs = build_schwarzite, dict(
                kind=self.var_s_kind.get(),
                cell=float(self.var_s_cell.get()),
                anneal_sweeps=int(self.var_anneal.get()),
                roughness=float(self.var_roughness.get()),
                seed=int(self.var_seed.get()),
            )
        elif mode == "coil (relaxed)":
            builder, kwargs = build_coil, dict(
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
                seed=int(self.var_seed.get()),
            )
        elif mode == "fullerene":
            builder, kwargs = build_fullerene, dict(
                freq=int(self.var_cage_freq.get()),
                family=self.var_cage_family.get(),
                bond=float(self.var_bond.get()),
                roughness=float(self.var_roughness.get()),
                seed=int(self.var_seed.get()),
            )
        elif mode == "nano-onion":
            builder, kwargs = build_nano_onion, dict(
                n_shells=int(self.var_onion_shells.get()),
                inner_freq=int(self.var_cage_freq.get()),
                family=self.var_cage_family.get(),
                bond=float(self.var_bond.get()),
                roughness=float(self.var_roughness.get()),
                seed=int(self.var_seed.get()),
            )
        elif mode == "multi-wall":
            builder, kwargs = build_multiwall_cnt, dict(
                n_shells=int(self.var_mw_shells.get()),
                inner_freq=int(self.var_mw_inner.get()),
                n_body_rings=int(self.var_rings.get()),
                bond=float(self.var_bond.get()),
                roughness=float(self.var_roughness.get()),
                seed=int(self.var_seed.get()),
            )
        elif mode == "bundle":
            builder, kwargs = build_bundle, dict(
                n_rings_across=int(self.var_bundle_shells.get()),
                freq=int(self.var_freq.get()),
                n_body_rings=int(self.var_rings.get()),
                bond=float(self.var_bond.get()),
                roughness=float(self.var_roughness.get()),
                seed=int(self.var_seed.get()),
            )
        else:
            builder = build_capped_cnt
            kwargs = dict(
            n_body_rings=int(self.var_rings.get()),
            freq=int(self.var_freq.get()),
            bond=float(self.var_bond.get()),
            bend_angle=float(self.var_bend.get()),
            shape=self.var_shape.get(),
            helix_radius=(float(self.var_coil_radius.get())
                          if self.var_shape.get() == "helix" else None),
            helix_pitch=float(self.var_coil_pitch.get()),
            helix_turns=float(self.var_coil_turns.get()),
            helix_handedness=1 if self.var_coil_hand.get() == "right" else -1,
            helix_taper=float(self.var_coil_taper.get()),
            roughness=float(self.var_roughness.get()),
            waviness=float(self.var_waviness.get()),
            max_strain=float(self.var_max_strain.get()),
            shape_points=int(self.var_shape_points.get()),
            defects=self._current_defects(),
            seed=int(self.var_seed.get()),
            )

        # Read every Tk variable here, on the main thread. Tkinter variables
        # are not safe to touch from the worker.
        dopant = self.var_dopant.get()
        dopant_conc = float(self.var_dopant_conc.get())
        seed = int(self.var_seed.get())

        def worker():
            try:
                built = builder(**kwargs)
                if dopant != "none" and dopant_conc > 0:
                    built = dope_random(built, dopant, dopant_conc, seed=seed)
                self._queue.put(("done", built))
            except Exception as exc:  # surfaced in the UI, not the console
                self._queue.put(("error", (exc, traceback.format_exc())))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        """Drain finished builds, then always re-arm the timer.

        The re-arm is in a ``finally``, and drawing is wrapped separately,
        because this callback is the only thing keeping the window alive:
        an exception escaping it skips ``root.after`` and polling stops
        for good, leaving the app permanently "building" with no way back.
        A single missing ``info`` key did exactly that -- one unhandled
        display error should never cost the user the whole session.
        """
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                self.progress.stop()
                self.btn_build.config(state="normal")
                self._busy = False
                if kind == "done":
                    self.atoms = payload
                    self.last_saved_stem = None
                    try:
                        self._redraw()
                        self._update_info()
                        self._set_status("Build complete.")
                    except Exception as exc:  # surfaced, never fatal
                        self._set_status(f"Built, but display failed: {exc}")
                        messagebox.showerror(
                            "Display failed",
                            "The structure was built successfully but could "
                            f"not be displayed:\n\n{traceback.format_exc()}",
                        )
                else:
                    exc, tb = payload
                    self._set_status(f"Build failed: {exc}")
                    messagebox.showerror("Build failed", f"{exc}\n\n{tb}")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------ draw
    def _atom_colours(self) -> list[str]:
        n = len(self.atoms)
        if not self.var_colour_rings.get():
            return [RING_COLOURS[6]] * n
        # Colour each atom by the most informative ring it belongs to.
        best = [6] * n
        priority = {5: 3, 7: 2, 8: 1, 6: 0}
        for ring in self.atoms.info["rings"]:
            size = len(ring)
            for a in ring:
                if priority.get(size, 0) > priority.get(best[a], 0):
                    best[a] = size
        return [RING_COLOURS.get(s, RING_COLOURS[6]) for s in best]

    def _redraw(self) -> None:
        if self.atoms is None:
            return
        pos = self.atoms.get_positions()
        self.ax.clear()

        if self.var_show_bonds.get():
            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            bonds = self.atoms.info["bonds"]
            segs = [(pos[a], pos[b]) for a, b in bonds]
            self.ax.add_collection3d(
                Line3DCollection(segs, colors="#9aa3ad", linewidths=0.6, alpha=0.75)
            )

        # Marker size shrinks as the structure grows so big shells stay legible.
        size = float(np.clip(2200.0 / max(len(pos), 1), 1.5, 26.0))
        self.ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                        c=self._atom_colours(), s=size, depthshade=True,
                        edgecolors="none")

        # Undistorted *and* space-filling: set each axis to its own data
        # range, then make the drawing box proportional to those ranges.
        # Forcing a cube instead would keep the proportions but shrink a long
        # tube to a sliver in the middle of the canvas.
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        extent = np.maximum(hi - lo, 1e-6)
        pad = 0.04 * extent
        self.ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        self.ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        self.ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
        self.ax.set_box_aspect(tuple(extent))
        self.ax.set_axis_off()
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas.draw_idle()

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
        # Each field is reported on its own presence. Keying the whole
        # block off "radius" assumed a radius meant a tube, and a
        # fullerene has a radius but no length, shape or path strain.
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
            lines.append(f"wall spacing {a.info['wall_spacing']:>6.2f} Å")
        if "n_tubes" in a.info:
            lines.append(f"tubes        {a.info['n_tubes']:>6d}")
        sep = a.info.get("geometry", {}).get("min_wall_separation")
        if sep is not None and sep == sep:  # not NaN
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
            f"  Euler sum  {deficit:>5d}  "
            f"{'OK' if deficit == expected else 'BROKEN'}",
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
            messagebox.showinfo("Nothing to export", "Build a structure first.")
            return None
        path = filedialog.asksaveasfilename(
            title="Save render bundle",
            defaultextension=".xyz",
            filetypes=[("XYZ structure", "*.xyz"), ("All files", "*.*")],
            initialfile="capped_cnt.xyz",
        )
        if not path:
            return None
        stem = Path(path).with_suffix("")
        xyz_path, json_path = write_render_bundle(self.atoms, stem)
        self.last_saved_stem = stem
        self._set_status(f"Saved {xyz_path.name} and {json_path.name}")
        return stem

    def _check_blender(self) -> None:
        if getattr(self, "blender_exe", None) and Path(self.blender_exe).exists():
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
            messagebox.showinfo("Nothing to render", "Build a structure first.")
            return
        self._check_blender()
        if not self.blender_exe:
            messagebox.showwarning(
                "Blender not found",
                "Could not find Blender automatically.\n\n"
                "Click “Locate Blender…” and point at the executable "
                "(on Windows, usually\n"
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
            messagebox.showerror(
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

        def worker():
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                ok = proc.returncode == 0 and Path(out_png).exists()
                self._queue_render_result(ok, proc.stderr or proc.stdout, out_png)
            except Exception as exc:
                self._queue_render_result(False, str(exc), out_png)

        threading.Thread(target=worker, daemon=True).start()

    def _queue_render_result(self, ok: bool, log: str, out_png: str) -> None:
        def report():
            if ok:
                self._set_status(f"Rendered → {Path(out_png).name}")
                messagebox.showinfo("Render complete", f"Wrote {out_png}")
            else:
                self._set_status("Render failed — see message.")
                messagebox.showerror("Render failed", log[-3000:] or "Unknown error.")

        self.root.after(0, report)


def main() -> int:
    """Entry point for ``nanocarbon-gui``."""
    root = tk.Tk()
    NanocarbonGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
