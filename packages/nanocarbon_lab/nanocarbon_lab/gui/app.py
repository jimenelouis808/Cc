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
import importlib.util
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from . import TKINTER_HELP

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except ImportError as exc:  # pragma: no cover - environment dependent
    # ImportError and not SystemExit: this is a module import, and pytest
    # cannot catch SystemExit while collecting, so raising one here took
    # the whole suite down on a machine without python3-tk. The friendly
    # advice reaches the user from ``nanocarbon_lab.gui.main`` instead.
    raise ImportError(TKINTER_HELP) from exc

# Deliberately no matplotlib.use("TkAgg") here. The canvas below is built
# by wrapping a bare Figure in FigureCanvasTkAgg, which is the embedding
# pattern -- it needs the Tk canvas class, not the global backend. Calling
# use() would reconfigure matplotlib for the whole process as a side effect
# of importing this module, and any later headless savefig would then fail
# with "cannot load backend 'TkAgg' ... 'headless' is currently running".
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
)
from matplotlib.figure import Figure

from .. import __version__
from ..builders import fullerene_mesh as fm
from ..builders.capped_cnt import MIN_CAP_FREQ
from ..builders.haeckelite import CATALOGUE as haeckelite_catalogue
from ..builders.haeckelite import PATTERNS as haeckelite_patterns
from ..builders.supernetwork import CAGES as supercages
from ..builders.supernetwork import SUPERLATTICES as superlattices
from ..cell import (
    MIN_IMAGE_SEPARATION,
    cell_report,
    describe_periodicity,
    supercell,
    to_unit_cell,
)
from ..dopants import DOPANT_ELEMENTS, get_chemistry
from ..dopants.codoping import AFFINITIES
from ..exports.xyz import write_cif, write_render_bundle
from ..functionalize import (
    GROUPS,
    describe,
    get_group,
    substitute,
    viable_swaps,
)
from ..jobs import (
    DOPANT_SITES,
    FAMILIES,
    MODES,
    TMD_EDITS,
    Job,
    estimate_atoms,
    estimate_cost,
    parse_codope_spec,
    parse_swaps,
    to_cli,
)
from ..tmd import MATERIALS as TMD_MATERIALS
from ..tmd.materials import (
    available_chalcogens,
    available_metals,
    chalcogens_for,
    material_for,
)
from ..tmd.quality import geometry_report as tmd_geometry_report
from ..tmd.quality import tmd_quality
from ..utils.constants import MAX_DOPING_FRACTION, MIN_DOPING_FRACTION
from ..validation.quality import sp2_quality
from ..viz import ELEMENT_COLOURS
from .worker import WORKER_DIED, BuildWorker

# Ring-type colours, shared with the Blender presets' intent: hexagons are
# the neutral body, everything else marks curvature or a defect.
RING_COLOURS = {5: "#e4572e", 6: "#5b6472", 7: "#2e86ab", 8: "#f2c14e"}

#: Filled rings drawn before thinning. Faces are far heavier to render
#: than the bonds, so this sits below PREVIEW_BOND_LIMIT.
PREVIEW_RING_FACE_LIMIT = 4000


def _to_rgb(colour: str) -> tuple[float, float, float]:
    """`#rrggbb` to a 0-1 triple, without importing matplotlib here."""
    value = colour.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
#: For an element with no entry in the palette, and for the "plain" mode
#: where the point is to see shape rather than composition.
UNKNOWN_ELEMENT_COLOUR = "#888888"
PLAIN_ATOM_COLOUR = "#5b6472"
RING_LABELS = {
    5: "pentagon (convex / cap)",
    6: "hexagon (body)",
    7: "heptagon (concave / saddle)",
    8: "octagon (divacancy)",
}

#: What the chemistry controls go back to when the structure type
#: changes. Dopants, dichalcogenide edits and grafted groups are all
#: chosen *for a material*: 3% pyridinic nitrogen means nothing once the
#: structure is a MoS2 ribbon, and a hydroxyl chosen for a nanotube would
#: otherwise still be there, still applied, on a schwarzite three clicks
#: later. The panels were already hidden on a change of mode; the values
#: behind them were not, so they went on reaching the builder unseen.
NEUTRAL_CHEMISTRY: dict[str, object] = {
    "dopant": "none", "dopant_conc": 0.03, "dopant_site": "random",
    "codope": "", "codope_affinity": "random",
    "tmd_edit": "none", "tmd_edit_element": "Se", "tmd_edit_amount": 0.5,
    "graft": "none", "graft_swap": "", "graft_coverage": 0.10,
    "graft_where": "all", "graft_face": "outer",
}

def _clock(seconds: float) -> str:
    """Elapsed time as a reader of a stopwatch expects it.

    Seconds below a minute, m:ss above -- because "143 s" makes someone
    divide and "2:23" does not.
    """
    if seconds < 60.0:
        return f"{seconds:.0f} s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}:{rest:02d}"


SHAPES = ["straight", "arc", "s_curve", "helix", "random"]
#: "none" first and selected by default: the host is carbon, and a
#: structure is pure carbon unless the user says otherwise.
DOPANTS = ["none", *DOPANT_ELEMENTS]
JUNCTION_KINDS = ["L", "T", "Y", "X", "cross3d"]
#: Zigzag first: it is the cheaper of the two at the same width and the
#: one whose edge states people usually come to a ribbon for.
RIBBON_EDGES = ["zigzag", "armchair"]
#: What an open tube can be made into. Only two, because only two are
#: real: a straight tube is exact crystallography and a wound one bends
#: that same lattice along a helix. The arc, S-curve and meander of the
#: capped tube need caps to anchor their relaxation and have no
#: counterpart here.
TUBE_SHAPES = ["straight", "helix"]
#: Modes whose atoms go straight onto the graphene lattice rather than
#: onto a meshed surface. They take defects and corrugation like the
#: meshed ones, but there is no mesh for the annealing slider to act on.
LATTICE_MODES = ("nanotube (open)", "nanoribbon")
#: Modes built by meshing a curved surface, where the pentagon-heptagon
#: pairs are how a hexagonal net takes up Gaussian curvature. Annealing
#: them away leaves the survivors carrying all of it and buckles the
#: wall, so all of these ask for zero sweeps -- measured on four junction
#: kinds out of four, and long established for the schwarzite.
CURVED_SURFACE_MODES = ("junction", "schwarzite", "network",
                        "coil (relaxed)")
SCHWARZITE_KINDS = ["primitive", "diamond", "gyroid"]

#: The haeckelite design patterns and catalogue, from the builder rather
#: than retyped -- the hint quotes each entry's own block and note.
HAECKELITE_PATTERNS = haeckelite_patterns
HAECKELITE_CATALOGUE = haeckelite_catalogue

#: The supernetwork catalogue, from the builder rather than a
#: copy: a net the window offers and the builder does not know
#: is a dead menu entry.
SUPERLATTICES = superlattices

#: The finite cages, offered beside the periodic nets. A cage
#: reads `scale` as its strut length rather than a cell edge,
#: which the hint says before the build.
SUPERCAGES = supercages
NETWORK_KINDS = ["cubic", "diamond"]
CAGE_FAMILIES = ["C60", "C20"]

# Dichalcogenide choices. The phase list is short on purpose: 2H and 1T
# are the two that matter, and 1T' is the distorted variant. There is no
# tetragonal TMD -- all of them are hexagonal, and what differs is the
# coordination polyhedron around the metal.
TMD_PHASES = ["2H", "1T", "1T'"]
TMD_STACKINGS = ["2H", "3R", "AA"]
TMD_EDGES = ["zigzag", "armchair"]
TMD_TERMINATIONS = ["mixed", "metal", "chalcogen"]

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

# Starting widths of the two side columns, in pixels. Wide enough for the
# longest label at the default font -- the old 268 clipped "Subdivision
# freq (diameter)" and the radius hint beneath it. They are a starting
# point, not a constraint: the dividers drag.
PARAM_COLUMN_WIDTH = 310
ACTION_COLUMN_WIDTH = 330

# Structures worth having one click away. Keys are parameter names as
# registered with `_var`, so applying a preset is a plain loop and the
# same format serves the save/load file.
PRESETS: dict[str, dict[str, object]] = {
    # --- carbon cages
    "C60 buckyball": {
        "mode_kind": "fullerene", "cage_family": "C60", "cage_freq": 1},
    "C540 giant cage": {
        "mode_kind": "fullerene", "cage_family": "C60", "cage_freq": 3},
    "Nano-onion C60@C240@C540": {
        "mode_kind": "nano-onion", "cage_family": "C60", "cage_freq": 1,
        "onion_shells": 3},
    # --- carbon tubes
    "Capped nanotube": {
        "mode_kind": "capped tube", "rings": 10, "freq": 3, "shape": "straight",
        "roughness": 0.0, "n_sw": 0, "n_dv": 0},
    "N-doped nanotube": {
        "mode_kind": "capped tube", "rings": 10, "freq": 3,
        "dopant": "N", "dopant_conc": 0.03},
    "Double-wall nanotube": {
        "mode_kind": "multi-wall", "mw_shells": 2, "mw_inner": 3, "rings": 10},
    "Seven-tube rope": {
        "mode_kind": "bundle", "bundle_shells": 1, "freq": 3, "rings": 10},
    # The crystalline ring: no disclinations at all. 110 periods of a
    # (5,5) is the smallest that fits the 8% strain budget.
    "Carbon toroid (all hexagons)": {
        "mode_kind": "toroid (polyhex)", "tp_n": 5, "tp_m": 5,
        "tp_periods": 110},
    "Carbon toroid (R/r = 4)": {
        "mode_kind": "toroid", "tor_major": 20.0, "tor_minor": 5.0,
        "anneal": 0},
    # The only disclination the sector cut places cleanly: one apex
    # pentagon, 210 hexagons, bonds 1.391–1.420 Å.
    "Nanocone (112.9°, one pentagon)": {
        "mode_kind": "nanocone", "nc_pent": 1, "nc_radius": 22.0,
        "nc_strict": True},
    # --- 2D carbon allotropes
    # Pentagons and heptagons only, no hexagons: the lattice published as
    # pentaheptite. The catalogue entry is an exact cover, so its census
    # is exact rather than whatever the rules let through.
    "Haeckelite R5,7 sheet": {
        "mode_kind": "haeckelite", "hk_pattern": "r57", "hk_nx": 4,
        "hk_ny": 4},
    "Haeckelite R5,7 tube": {
        "mode_kind": "haeckelite tube", "ht_pattern": "r57", "ht_nx": 12,
        "ht_ny": 4, "ht_roll": "a"},
    # A trivalent net of nothing but heptagons: the one entry here whose
    # answer is a proof rather than a structure. `hp_strict` is False on
    # purpose. Applying a preset BUILDS immediately, and under strict the
    # builder refuses by design -- so selecting this preset fired its
    # refusal as an error dialog before the window had drawn anything,
    # which is the worst possible way to deliver a result. Unticked, it
    # returns the strained lattice to look at, and the panel's own text
    # plus `sp2 verdict` carry the finding. Tick the box to see the
    # refusal in full.
    "Heptanene (Klein quartic, genus 3)": {
        "mode_kind": "heptanene", "hp_strict": False},
    # --- coils. Two routes, and they are not interchangeable, so the
    # names say which. One preset each; the old menu carried three
    # meshed coils differing only in radius and turn count, which is a
    # parameter sweep rather than three structures.
    #
    # The **rolled lattice** winds a real (n, m) nanotube: every ring is
    # a hexagon and the wall is graphitic. Bending a finished lattice can
    # only stretch it, so the coil has to be wide -- which is why real
    # carbon nanocoils are tens to hundreds of Å across.
    "Nanocoil (graphitic, rolled lattice)": {
        "mode_kind": "nanocoil", "cnt_shape": "helix", "cnt_n": 5, "cnt_m": 5,
        "coil_radius": 45.0, "coil_pitch": 12.0, "coil_turns": 2.0,
        "n_sw": 0, "n_dv": 0, "roughness": 0.0},
    # The **meshed wall** route fits a surface to the helix and tiles it,
    # so it reaches any radius -- but what it tiles with is an amorphous
    # CVD-like network, not a rolled lattice. `anneal` is 0 and must
    # stay 0: on a curved surface the 5-7 pairs ARE how the net covers
    # its curvature, and annealing them away leaves the survivors to
    # carry all of it. Measured on this Y junction, 80 sweeps widen the
    # bond spread from 0.0136 to 0.0175 Å.
    "Nanocoil (meshed wall)": {
        "mode_kind": "coil (relaxed)", "coil_radius": 18.0, "coil_pitch": 13.0,
        "coil_turns": 2.0, "coil_tube_radius": 4.5, "anneal": 0,
        "roughness": 0.0, "n_sw": 0, "n_dv": 0},
    # The one that goes into a plane-wave code: one turn closed on the
    # z-torus, so it is a cell and not a fragment with two dangling ends.
    # Six sides, because seen down its axis a real single-wall coil is a
    # POLYGON -- Liu et al. show the (6,6) as a hexagonal torus -- and at
    # D/d = 3.92 the hexagon puts 85% of its disclinations on the correct
    # side against the smooth helix's 68%.
    "Nanocoil (periodic cell, DFT)": {
        "mode_kind": "coil (periodic, DFT)", "coil_radius": 8.75,
        "coil_pitch": 9.6, "coil_tube_radius": 3.0, "coil_sides": 6,
        "roughness": 0.0, "n_sw": 0, "n_dv": 0},
    # --- junctions and periodic 3D carbon
    "Y junction": {
        "mode_kind": "junction", "j_kind": "Y", "j_radius": 6.0,
        "j_arm": 22.0, "j_blend": 4.0, "anneal": 0},
    "Gyroid schwarzite": {
        "mode_kind": "schwarzite", "s_kind": "gyroid", "s_cell": 36.0,
        "anneal": 0},
    "Schwarz P schwarzite": {
        "mode_kind": "schwarzite", "s_kind": "primitive", "s_cell": 36.0,
        "anneal": 0},
    "Nanotube network (cubic)": {
        "mode_kind": "network", "net_kind": "cubic", "net_cell": 40.0,
        "net_radius": 6.0, "net_blend": 5.0, "anneal": 0},
    # --- superlattices of nanotubes
    "Super-graphene (tubes at 120°)": {
        "mode_kind": "supernetwork", "sn_graph": "super-graphene",
        "sn_scale": 34.0, "sn_radius": 5.0, "sn_blend": 4.0, "anneal": 0},
    "Super-diamond (tubes at 109.47°)": {
        "mode_kind": "supernetwork", "sn_graph": "super-diamond",
        "sn_scale": 60.0, "sn_radius": 5.0, "sn_blend": 4.0, "anneal": 0},
    # Each of the 4-cube's 32 edges a nanotube. The struts are
    # deliberately unequal -- that is what a 4D object looks like in 3D.
    "Hypercube of tubes (4-cube)": {
        "mode_kind": "supernetwork", "sn_graph": "super-hypercube",
        "sn_scale": 20.0, "sn_radius": 3.5, "sn_blend": 2.5, "anneal": 0},
    # Super-graphene rolled: a nanotube whose every bond is a nanotube.
    "Supertube (6,6) of super-graphene": {
        "mode_kind": "supernetwork", "sn_graph": "supertube-(6,6)",
        "sn_scale": 14.0, "sn_radius": 3.0, "sn_blend": 2.0, "anneal": 0},
    "Icosahedral cage of tubes": {
        "mode_kind": "supernetwork", "sn_graph": "super-icosahedron",
        "sn_scale": 24.0, "sn_radius": 4.0, "sn_blend": 3.0, "anneal": 0},
    "Superfullerene (C60 of tubes)": {
        "mode_kind": "supernetwork", "sn_graph": "superfullerene-C60",
        "sn_scale": 14.2, "sn_radius": 3.0, "sn_blend": 2.0, "anneal": 0},
    # The three periodic nets that had no preset. Without one the Net
    # box is the only way to reach them, and picking a net there leaves
    # the cell, radius and blend at whatever the LAST preset set -- which
    # is how super-fcc got built at super-diamond's scale=60, radius=5
    # and ran for ninety minutes. Every number below is measured: fcc is
    # sound at 334.0 deg in 413 s at 40/3.0/2.0, and at 60/5/4 it asks
    # for 32_000 A^2 of wall instead of 12_800.
    "Super-square (tubes at 90°)": {
        "mode_kind": "supernetwork", "sn_graph": "super-square",
        "sn_scale": 34.0, "sn_radius": 5.0, "sn_blend": 4.0, "anneal": 0},
    "Super-cubic (tubes along the axes)": {
        "mode_kind": "supernetwork", "sn_graph": "super-cubic",
        "sn_scale": 34.0, "sn_radius": 5.0, "sn_blend": 4.0, "anneal": 0},
    "Super-fcc (twelve tubes per node)": {
        "mode_kind": "supernetwork", "sn_graph": "super-fcc",
        "sn_scale": 40.0, "sn_radius": 3.0, "sn_blend": 2.0, "anneal": 0},
    # --- dichalcogenides
    "MoS2 monolayer (2H)": {
        "mode_kind": "TMD layers", "tmd_material": "MoS2", "tmd_phase": "2H",
        "tmd_layers": 1, "tmd_nx": 1, "tmd_ny": 1},
    "MoS2 bilayer (2H)": {
        "mode_kind": "TMD layers", "tmd_material": "MoS2", "tmd_phase": "2H",
        "tmd_layers": 2, "tmd_stacking": "2H"},
    "MoS2 monolayer (1T)": {
        "mode_kind": "TMD layers", "tmd_material": "MoS2", "tmd_phase": "1T",
        "tmd_layers": 1},
    "MoS2 bulk crystal": {
        "mode_kind": "TMD bulk", "tmd_material": "MoS2", "tmd_stacking": "2H"},
    "MoS2 zigzag ribbon": {
        "mode_kind": "TMD ribbon", "tmd_material": "MoS2", "tmd_width": 8,
        "tmd_length": 2, "tmd_edge": "zigzag", "tmd_termination": "mixed"},
    "MoS2 nanotube (40,0)": {
        "mode_kind": "TMD nanotube", "tmd_material": "MoS2", "tmd_n": 40,
        "tmd_m": 0},
    "WSe2 monolayer": {
        "mode_kind": "TMD layers", "tmd_material": "WSe2", "tmd_phase": "2H",
        "tmd_layers": 1},
    "MoS2 Y junction": {
        "mode_kind": "TMD junction", "tmd_material": "MoS2",
        "tmd_j_kind": "Y", "tmd_j_radius": 12.0, "tmd_j_arm": 26.0,
        "tmd_j_parity": "split"},
    "MoS2 schwarzite (Schwarz P)": {
        "mode_kind": "TMD schwarzite", "tmd_material": "MoS2",
        "tmd_sw_kind": "primitive", "tmd_sw_cell": 36.0,
        "tmd_sw_parity": "flip"},
    "MoS2 coil (quarter turn)": {
        "mode_kind": "TMD coil", "tmd_material": "MoS2", "tmd_n": 30,
        "tmd_m": 0, "tmd_coil_radius": 220.0, "tmd_coil_pitch": 90.0,
        "tmd_coil_turns": 0.25, "tmd_coil_hand": "right"},
    # --- heterostructures
    # A quarter turn keeps the MX2 coil near 11k atoms. A full turn at a
    # radius loose enough to be unstrained runs to six figures, which is
    # the physics rather than a timid default.
    "Twisted bilayer graphene 21.8°": {
        "mode_kind": "twisted bilayer", "het_bottom": "graphene",
        "het_top": "same", "het_angle": 21.79, "het_max_index": 40},
    "Magic-angle bilayer 1.08°": {
        "mode_kind": "twisted bilayer", "het_bottom": "graphene",
        "het_top": "same", "het_angle": 1.08, "het_max_index": 40},
    "Graphene on hBN": {
        "mode_kind": "twisted bilayer", "het_bottom": "graphene",
        "het_top": "hBN", "het_angle": 7.34, "het_max_index": 40},
    "MoS2/WS2 stack": {
        "mode_kind": "vdW stack", "het_bottom": "MoS2", "het_top": "WS2",
        "het_third": "none", "het_nx": 2, "het_ny": 2},
}


def has_bpy() -> bool:
    """Is Blender available as a Python module in this interpreter?

    ``bpy`` on PyPI is a full Blender build importable from an ordinary
    interpreter, so `pip install bpy` makes the render pipeline work with
    no Blender application installed at all. That removes the pipeline's
    single most common failure -- "Blender not found" on a machine where
    the user has no intention of installing a 3D suite by hand.

    Checked with ``find_spec`` rather than an import: importing Blender
    costs hundreds of megabytes of process memory, and this runs while
    merely drawing a label.
    """
    try:
        return importlib.util.find_spec("bpy") is not None
    except (ImportError, ValueError):
        return False


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

    def __init__(self, parent: tk.Widget,
                 width: int = PARAM_COLUMN_WIDTH) -> None:
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
        self._rewrap(event.width)

    def _rewrap(self, width: int) -> None:
        """Re-wrap the explanatory labels to the column's current width.

        Every hint in this app was written with a ``wraplength`` in
        pixels, and two dozen of them were tuned to a column that no
        longer has a fixed width. A hardcoded wrap is wrong twice over:
        it clips when the font is larger than the author's, and it keeps
        wrapping at the old width after the divider is dragged wider,
        leaving a ragged column beside empty space.

        A label that wraps has a non-zero ``wraplength`` and an ordinary
        one has zero, so that flag is the selector -- no registry to keep
        in step, and hints added later are picked up for free.
        """
        target = max(120, width - 22)
        stack = [self.interior]
        while stack:
            widget = stack.pop()
            stack.extend(widget.winfo_children())
            try:
                if int(widget.cget("wraplength")) > 0:
                    widget.configure(wraplength=target)
            except (tk.TclError, ValueError):
                continue  # not a label, or no such option

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
        # The version goes in the title deliberately. A report of "the
        # bug is still there" is impossible to act on without knowing
        # which build is running, and an editable install left pointing
        # at an older extraction looks exactly like a fix that did not
        # work.
        self.root.title(
            f"nanocarbon_lab {__version__} — carbon nanostructure builder")
        self.root.geometry("1400x860")
        self.root.minsize(1120, 700)

        self.atoms = None
        self.last_saved_stem: Path | None = None
        #: Which cell each bond's far atom sits in, and the structure it
        #: was computed for. See :meth:`_bond_shifts`.
        self._shifts = np.zeros((0, 3), dtype=int)
        self._shifts_for = None
        self.blender_exe: str | None = None
        self._busy = False
        self._estimate_job: str | None = None
        # Registry of every parameter variable, keyed by a short name.
        # Presets, the save/load file and the estimate traces all iterate
        # this rather than naming each variable three times over.
        self._params: dict[str, tk.Variable] = {}
        #: True while a preset or a settings file is being applied. Those
        #: write `mode_kind` like a user does, and must not have the
        #: chemistry they just asked for reset out from under them.
        self._applying_values = False
        #: The mode the chemistry currently belongs to.
        self._chemistry_mode: str | None = None
        #: When the running build started, or None when idle.
        self._build_started: float | None = None
        #: Pending `after` id for the clock, so it can be cancelled.
        self._clock_job: str | None = None
        #: mode -> (atoms, seconds) for the last build of that mode in
        #: this session. A measurement beats an estimate, and the
        #: estimator here is a hand-written bucket ("a minute or two"),
        #: so once a mode has actually been built its own timing is
        #: shown alongside.
        self._measured: dict[str, tuple[int, float]] = {}
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
        """Three columns with draggable dividers.

        Packed at fixed widths this clipped its own labels: "Subdivision
        freq (diameter)" and the derived-radius hint under it both ran off
        the end of a 268 px column, and there was no way to widen it. A
        fixed pixel width cannot be right anyway -- it depends on the
        font, the theme and the platform, none of which this knows.

        A PanedWindow costs nothing and fixes the class of bug rather
        than the instance: the columns start wide enough for the longest
        label at the default font, and anyone whose font is bigger, or
        who wants the preview larger, drags the divider.
        """
        outer = ttk.PanedWindow(self.root, orient="horizontal")
        outer.pack(fill="both", expand=True, padx=8, pady=8)

        left = ScrollableColumn(outer)
        centre = ttk.Frame(outer)
        right = ScrollableColumn(outer, width=ACTION_COLUMN_WIDTH)
        # weight=0 on the two side columns: growing the window makes the
        # 3D preview bigger, which is what the extra space is for, rather
        # than stretching two columns of labels.
        outer.add(left, weight=0)
        outer.add(centre, weight=1)
        outer.add(right, weight=0)

        self._build_params(left.interior)
        self._build_preview(centre)
        self._build_actions(right.interior)

    # ------------------------------------------------------------ parameters
    def _var(self, name: str, var: tk.Variable) -> tk.Variable:
        """Register a parameter variable under a short, stable name."""
        self._params[name] = var
        return var

    def _param(self, parent, label, var, lo, hi, row, *, integer=False,
               resolution=None, command=None, hard_lo=None, hard_hi=None
               ) -> tuple[ttk.Label, ttk.Entry, ttk.Scale]:
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

        caption = ttk.Label(parent, text=label)
        caption.grid(row=row, column=0, sticky="w")
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

        def follow(*_args) -> None:
            # Keep the box in step with the variable however it changed --
            # slider, preset, loaded file, or a mode switch forcing a
            # value. Without this the entry silently shows a stale number,
            # which is worse than showing none: it is the field you would
            # trust. `command` runs here too, not only on the two
            # interactive paths: it is what refreshes the derived hints,
            # and a preset that set the value without firing it left them
            # describing the previous structure.
            render()
            if command:
                command()

        var.trace_add("write", follow)

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
        # Returned so a mode that the parameter does not apply to can
        # `grid_remove` the three of them together, rather than leaving a
        # live-looking control that feeds nothing.
        return caption, entry, scale

    def _build_params(self, parent: ttk.Frame) -> None:
        self.var_mode_kind = self._var("mode_kind", tk.StringVar(value=MODES[0]))
        mode_box = ttk.LabelFrame(parent, text="Structure type", padding=8)
        mode_box.pack(fill="x", pady=(0, 8))

        # Material family first, structure type second. Carbon and the
        # dichalcogenides share no parameters at all -- there is no bond
        # length, ring count or chirality that means the same thing in
        # both -- so mixing them in one list would offer every user a
        # dropdown that is mostly irrelevant to them.
        ttk.Label(mode_box, text="Material").pack(anchor="w")
        self.var_family = tk.StringVar(value="carbon")
        ttk.Combobox(mode_box, textvariable=self.var_family,
                     values=list(FAMILIES), state="readonly").pack(fill="x",
                                                                   pady=(0, 6))
        self.var_family.trace_add("write", lambda *_: self._on_family_change())

        ttk.Label(mode_box, text="Structure").pack(anchor="w")
        self.cmb_mode = ttk.Combobox(mode_box, textvariable=self.var_mode_kind,
                                     values=list(FAMILIES["carbon"]),
                                     state="readonly")
        self.cmb_mode.pack(fill="x")
        self.var_mode_kind.trace_add("write", lambda *_: self._on_mode_written())

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

        # The open tube is indexed rather than subdivided: a (n, m) and a
        # length, which is how nanotubes are named in every paper. The
        # capped tube cannot use them -- a cap is built by subdividing a
        # capsule, so its size is rings and frequency -- which is why the
        # two modes need separate controls rather than a shared panel.
        self.var_cnt_n = self._var("cnt_n", tk.IntVar(value=6))
        self.var_cnt_m = self._var("cnt_m", tk.IntVar(value=6))
        self.var_cnt_length = self._var("cnt_length", tk.DoubleVar(value=20.0))
        self.var_cnt_vacuum = self._var("cnt_vacuum", tk.DoubleVar(value=12.0))
        # Straight or wound. The two are different builders -- a straight
        # tube is exact crystallography, a wound one bends that lattice
        # along a helix -- so picking "helix" switches the mode rather
        # than quietly changing what "nanotube (open)" means.
        self.var_cnt_shape = self._var("cnt_shape", tk.StringVar(value="straight"))
        # The ribbon is indexed the same way ASE indexes it: a width in
        # rows and a length in repeat units, both counts rather than Å,
        # because the lattice quantises them and a length in Å would be
        # rounded to one of these anyway.
        self.var_rib_width = self._var("rib_width", tk.IntVar(value=6))
        self.var_rib_length = self._var("rib_length", tk.IntVar(value=8))
        self.var_rib_edge = self._var("rib_edge", tk.StringVar(value="zigzag"))
        self.var_rib_passivate = self._var("rib_passivate", tk.BooleanVar(value=False))
        self.var_rib_vacuum = self._var("rib_vacuum", tk.DoubleVar(value=15.0))
        self.var_rings = self._var("rings", tk.IntVar(value=8))
        self.var_freq = self._var("freq", tk.IntVar(value=3))
        self.var_bond = self._var("bond", tk.DoubleVar(value=1.42))
        # Degrees, not radians. The key changed with the unit so a
        # parameter file written when this held radians is ignored rather
        # than read as 0.5 degrees where it meant 0.5 rad.
        self.var_bend = self._var("bend_deg", tk.DoubleVar(value=0.0))
        self.var_seed = self._var("seed", tk.IntVar(value=0))
        self.var_shape = self._var("shape", tk.StringVar(value="straight"))
        self.var_waviness = self._var("waviness", tk.DoubleVar(value=0.7))
        self.var_max_strain = self._var("max_strain", tk.DoubleVar(value=0.08))
        self.var_shape_points = self._var("shape_points", tk.IntVar(value=9))
        self.var_coil_radius = self._var("coil_radius", tk.DoubleVar(value=22.0))
        self.var_coil_pitch = self._var("coil_pitch", tk.DoubleVar(value=14.0))
        self.var_coil_turns = self._var("coil_turns", tk.DoubleVar(value=3.0))
        self.var_coil_hand = self._var("coil_hand", tk.StringVar(value="right"))
        self.var_coil_taper = self._var("coil_taper", tk.DoubleVar(value=1.0))
        # 0 = smooth helix. A real coil is a polygon seen down the axis:
        # Liu et al. show the (6,6) coil's top view as a hexagonal torus
        # and say it matches what is observed, and concentrating the
        # curvature at a few knees measurably improves where the
        # pentagons and heptagons land.
        self.var_coil_sides = self._var("coil_sides", tk.IntVar(value=0))
        self.var_coil_tube_radius = self._var(
            "coil_tube_radius", tk.DoubleVar(value=5.0))
        self.var_pin_ends = self._var("pin_ends", tk.BooleanVar(value=False))
        self.var_anneal = self._var("anneal", tk.IntVar(value=80))
        self.var_roughness = self._var("roughness", tk.DoubleVar(value=0.0))
        self.var_place = self._var("place", tk.BooleanVar(value=False))
        self.var_anchor = self._var("anchor", tk.BooleanVar(value=False))
        self.var_dopant = self._var("dopant", tk.StringVar(value="none"))
        self.var_dopant_conc = self._var("dopant_conc", tk.DoubleVar(value=0.03))
        self.var_dopant_site = self._var("dopant_site", tk.StringVar(value="random"))
        self.var_codope = self._var("codope", tk.StringVar(value=""))
        self.var_codope_affinity = self._var("codope_affinity",
                                             tk.StringVar(value="random"))
        # The MX2 counterpart of the carbon dopant vars. Separate because
        # the chemistry is: there is no "substitute a heteroatom for a
        # carbon" in a dichalcogenide, and no Janus face in graphene.
        self.var_tmd_edit = self._var("tmd_edit", tk.StringVar(value="none"))
        self.var_tmd_edit_element = self._var("tmd_edit_element",
                                              tk.StringVar(value="Se"))
        self.var_tmd_edit_amount = self._var("tmd_edit_amount",
                                             tk.DoubleVar(value=0.5))
        # Grafting is a third chemistry axis, not an alternative to the
        # other two: a dopant *replaces* a host atom and so belongs to one
        # family, while a group is *added on top of* a surface, which
        # carbon, MX2 and a vdW stack all have.
        self.var_graft = self._var("graft", tk.StringVar(value="none"))
        self.var_graft_swap = self._var("graft_swap", tk.StringVar(value=""))
        self.var_graft_coverage = self._var("graft_coverage",
                                            tk.DoubleVar(value=0.10))
        self.var_graft_where = self._var("graft_where",
                                         tk.StringVar(value="all"))
        self.var_graft_face = self._var("graft_face",
                                        tk.StringVar(value="outer"))
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
        self.var_hk_pattern = self._var("hk_pattern",
                                        tk.StringVar(value="sparse"))
        self.var_hk_nx = self._var("hk_nx", tk.IntVar(value=4))
        self.var_hk_ny = self._var("hk_ny", tk.IntVar(value=4))
        self.var_hk_period = self._var("hk_period", tk.IntVar(value=2))
        self.var_hk_density = self._var("hk_density", tk.DoubleVar(value=0.15))
        self.var_s_kind = self._var("s_kind", tk.StringVar(value="primitive"))
        self.var_s_cell = self._var("s_cell", tk.DoubleVar(value=36.0))
        self.var_s_thickness = self._var("s_thickness", tk.DoubleVar(value=0.0))
        self.var_nc_pent = self._var("nc_pent", tk.IntVar(value=1))
        self.var_nc_radius = self._var("nc_radius",
                                       tk.DoubleVar(value=22.0))
        self.var_nc_strict = self._var("nc_strict",
                                       tk.BooleanVar(value=True))
        self.var_tp_n = self._var("tp_n", tk.IntVar(value=5))
        self.var_tp_m = self._var("tp_m", tk.IntVar(value=5))
        self.var_tp_periods = self._var("tp_periods",
                                        tk.IntVar(value=110))
        self.var_tor_major = self._var("tor_major",
                                       tk.DoubleVar(value=20.0))
        self.var_tor_minor = self._var("tor_minor",
                                       tk.DoubleVar(value=5.0))
        self.var_hp_strict = self._var("hp_strict",
                                       tk.BooleanVar(value=True))
        self.var_ht_pattern = self._var("ht_pattern",
                                        tk.StringVar(value="r57"))
        self.var_ht_nx = self._var("ht_nx", tk.IntVar(value=12))
        self.var_ht_ny = self._var("ht_ny", tk.IntVar(value=4))
        self.var_ht_roll = self._var("ht_roll", tk.StringVar(value="a"))
        self.var_sn_graph = self._var("sn_graph",
                                      tk.StringVar(value="super-graphene"))
        self.var_sn_scale = self._var("sn_scale", tk.DoubleVar(value=40.0))
        self.var_sn_radius = self._var("sn_radius", tk.DoubleVar(value=5.0))
        self.var_sn_blend = self._var("sn_blend", tk.DoubleVar(value=4.0))
        self.var_net_kind = self._var("net_kind", tk.StringVar(value="cubic"))
        self.var_net_cell = self._var("net_cell", tk.DoubleVar(value=40.0))
        self.var_net_radius = self._var("net_radius", tk.DoubleVar(value=6.0))
        self.var_net_blend = self._var("net_blend", tk.DoubleVar(value=5.0))
        # The formula stays the single value the job is built from; the
        # metal and chalcogen pickers below drive it. Keeping the formula
        # authoritative means the presets, which name a compound, need no
        # special case.
        self.var_tmd_material = self._var("tmd_material",
                                          tk.StringVar(value="MoS2"))
        self.var_tmd_metal = tk.StringVar(value="Mo")
        self.var_tmd_chalcogen = tk.StringVar(value="S")
        self._syncing_material = False
        self.var_tmd_phase = self._var("tmd_phase", tk.StringVar(value="2H"))
        self.var_tmd_stacking = self._var("tmd_stacking", tk.StringVar(value="2H"))
        self.var_tmd_layers = self._var("tmd_layers", tk.IntVar(value=1))
        self.var_tmd_nx = self._var("tmd_nx", tk.IntVar(value=1))
        self.var_tmd_ny = self._var("tmd_ny", tk.IntVar(value=1))
        self.var_tmd_width = self._var("tmd_width", tk.IntVar(value=8))
        self.var_tmd_length = self._var("tmd_length", tk.IntVar(value=2))
        self.var_tmd_edge = self._var("tmd_edge", tk.StringVar(value="zigzag"))
        self.var_tmd_termination = self._var("tmd_termination",
                                             tk.StringVar(value="mixed"))
        self.var_tmd_n = self._var("tmd_n", tk.IntVar(value=40))
        self.var_tmd_m = self._var("tmd_m", tk.IntVar(value=0))
        self.var_tmd_coil_radius = self._var("tmd_coil_radius",
                                             tk.DoubleVar(value=220.0))
        self.var_tmd_coil_pitch = self._var("tmd_coil_pitch",
                                            tk.DoubleVar(value=90.0))
        self.var_tmd_coil_turns = self._var("tmd_coil_turns",
                                            tk.DoubleVar(value=0.25))
        self.var_tmd_coil_hand = self._var("tmd_coil_hand",
                                           tk.StringVar(value="right"))
        self.var_tmd_sw_kind = self._var("tmd_sw_kind",
                                         tk.StringVar(value="primitive"))
        self.var_tmd_sw_cell = self._var("tmd_sw_cell",
                                         tk.DoubleVar(value=36.0))
        self.var_tmd_sw_parity = self._var("tmd_sw_parity",
                                           tk.StringVar(value="flip"))
        self.var_tmd_j_kind = self._var("tmd_j_kind", tk.StringVar(value="Y"))
        self.var_tmd_j_radius = self._var("tmd_j_radius",
                                          tk.DoubleVar(value=12.0))
        self.var_tmd_j_arm = self._var("tmd_j_arm", tk.DoubleVar(value=26.0))
        self.var_tmd_j_blend = self._var("tmd_j_blend", tk.DoubleVar(value=5.0))
        self.var_tmd_j_parity = self._var("tmd_j_parity",
                                          tk.StringVar(value="split"))
        self.var_het_bottom = self._var("het_bottom",
                                        tk.StringVar(value="graphene"))
        self.var_het_top = self._var("het_top", tk.StringVar(value="same"))
        self.var_het_angle = self._var("het_angle", tk.DoubleVar(value=7.34))
        self.var_het_max_index = self._var("het_max_index", tk.IntVar(value=40))
        self.var_het_gap = self._var("het_gap", tk.DoubleVar(value=3.35))
        self.var_het_third = self._var("het_third", tk.StringVar(value="none"))
        self.var_het_nx = self._var("het_nx", tk.IntVar(value=1))
        self.var_het_ny = self._var("het_ny", tk.IntVar(value=1))

        # A wraplength even though this is one short line: it is what marks
        # a label as an explanatory hint for ScrollableColumn._rewrap, and
        # without it this one alone kept clipping while the rest reflowed.
        self.lbl_radius = ttk.Label(box, text="", foreground="#2e86ab",
                                    wraplength=PARAM_COLUMN_WIDTH - 22,
                                    justify="left")

        self._param(box, "Body rings (length)", self.var_rings, 2, 30, 0,
                    integer=True, hard_hi=200, command=self._update_bend_hint)
        # Low end 2, not 1: a capped tube at freq=1 collapses (see
        # builders.capped_cnt.MIN_CAP_FREQ), and the slider offering it
        # meant a minute and a half of building before a message that
        # blamed a sweep. The bundle shares this control and shares the
        # floor for the same reason.
        self._param(box, "Subdivision freq (diameter)", self.var_freq,
                    MIN_CAP_FREQ, 8, 2, integer=True, hard_lo=MIN_CAP_FREQ,
                    hard_hi=20, command=self._update_radius_hint)
        self.lbl_radius.grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self._param(box, "Bend angle (°)", self.var_bend, 0.0, 180.0, 5,
                    resolution=1.0, hard_hi=359.0,
                    command=self._update_bend_hint)
        self.lbl_bend = ttk.Label(box, text="", foreground=MUTED,
                                  font=("TkDefaultFont", 8),
                                  wraplength=PARAM_COLUMN_WIDTH - 22,
                                  justify="left")
        self.lbl_bend.grid(row=7, column=0, columnspan=2, sticky="w")
        self._param(box, "C–C bond (Å)", self.var_bond, 1.30, 1.55, 8,
                    resolution=0.005, hard_lo=1.20, hard_hi=1.80,
                    command=self._update_radius_hint)

        seed_row = ttk.Frame(box)
        seed_row.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(seed_row, text="Random seed").pack(side="left")
        ttk.Button(seed_row, text="🎲", width=3,
                   command=self.on_roll_seed).pack(side="right", padx=(4, 0))
        ttk.Spinbox(seed_row, from_=0, to=999999, textvariable=self.var_seed,
                    width=8).pack(side="right")

        # --- centreline shape
        obox = ttk.LabelFrame(parent, text="Chiral indices", padding=8)
        self.frame_open_tube = obox
        self._param(obox, "n", self.var_cnt_n, 1, 30, 0, integer=True,
                    command=lambda *_: self._update_open_tube_hint())
        self._param(obox, "m", self.var_cnt_m, 0, 30, 2, integer=True,
                    command=lambda *_: self._update_open_tube_hint())
        self._cnt_length = self._param(
            obox, "Length (Å)", self.var_cnt_length, 5.0, 200.0, 4,
            command=lambda *_: self._update_open_tube_hint())
        self._param(obox, "Vacuum (Å)", self.var_cnt_vacuum, 8.0, 30.0, 6)
        shape_row = ttk.Frame(obox)
        shape_row.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Label(shape_row, text="Shape").pack(side="left")
        ttk.Combobox(shape_row, textvariable=self.var_cnt_shape,
                     values=TUBE_SHAPES, state="readonly",
                     width=9).pack(side="right")
        self.var_cnt_shape.trace_add("write", lambda *_: self._on_tube_shape())
        self.lbl_open_tube = ttk.Label(obox, text="", foreground=MUTED,
                                       wraplength=330, justify="left")
        self.lbl_open_tube.grid(row=10, column=0, columnspan=3, sticky="w",
                                pady=(6, 0))
        obox.pack_forget()

        # --- nanoribbon. The builder has been in the package all along;
        #     the window simply never offered it.
        rbox = ttk.LabelFrame(parent, text="Ribbon", padding=8)
        self.frame_ribbon = rbox
        rbox.columnconfigure(0, weight=1)
        ttk.Label(rbox, text="Edge").grid(row=0, column=0, sticky="w")
        ttk.Combobox(rbox, textvariable=self.var_rib_edge,
                     values=RIBBON_EDGES, state="readonly", width=9).grid(
            row=0, column=1, sticky="e", pady=(0, 6))
        self.var_rib_edge.trace_add("write",
                                    lambda *_: self._update_ribbon_hint())
        self._param(rbox, "Width (rows)", self.var_rib_width, 2, 20, 1,
                    integer=True, hard_hi=120,
                    command=lambda *_: self._update_ribbon_hint())
        self._param(rbox, "Length (units)", self.var_rib_length, 2, 30, 3,
                    integer=True, hard_hi=200,
                    command=lambda *_: self._update_ribbon_hint())
        self._param(rbox, "Vacuum (Å)", self.var_rib_vacuum, 8.0, 30.0, 5)
        ttk.Checkbutton(rbox, text="Passivate edges with H",
                        variable=self.var_rib_passivate,
                        command=self._update_ribbon_hint).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.lbl_ribbon = ttk.Label(rbox, text="", foreground=MUTED,
                                    wraplength=PARAM_COLUMN_WIDTH - 22,
                                    justify="left")
        self.lbl_ribbon.grid(row=8, column=0, columnspan=2, sticky="w",
                             pady=(6, 0))
        rbox.pack_forget()

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
        # Rows 6 and 7 -- the taper's own label/entry and its slider. The
        # legend below used to be gridded at row 7 as well, on top of the
        # slider, so the taper looked like a number box with no slider at
        # all. It now has its own row.
        self._coil_taper = self._param(
            self.frame_coil, "Taper (end/start R)", self.var_coil_taper,
            0.3, 2.0, 6, resolution=0.05, hard_lo=0.05, hard_hi=10.0,
            command=self._update_coil_hint)
        self.frame_coil_hand = ttk.Frame(self.frame_coil)
        self.frame_coil_hand.grid(row=8, column=0, columnspan=2, sticky="ew",
                                  pady=(2, 2))
        ttk.Label(self.frame_coil_hand, text="Handedness").pack(side="left")
        ttk.Combobox(self.frame_coil_hand, textvariable=self.var_coil_hand,
                     values=["right", "left"], state="readonly",
                     width=7).pack(side="right")
        self.lbl_coil_legend = ttk.Label(
            self.frame_coil, foreground=MUTED, font=("TkDefaultFont", 8),
            wraplength=225, justify="left",
            text="Coil radius: how far the tube's centre sits from the helix "
                 "axis — the spring's own radius, not the tube's.\n"
                 "Coil pitch: how far one full turn rises. Close to the tube "
                 "width gives a tight spring; several times it gives an open "
                 "spiral.\n"
                 "Turns: complete revolutions. Under two does not read as a "
                 "coil.\n"
                 "Taper: end radius over start radius. 1 is a cylinder, "
                 "below 1 a cone.",
        )
        self.lbl_coil_legend.grid(row=9, column=0, columnspan=2, sticky="w",
                                  pady=(4, 2))
        self.lbl_coil = ttk.Label(self.frame_coil, text="", foreground=MUTED,
                                  font=("TkDefaultFont", 8), wraplength=225,
                                  justify="left")
        self.lbl_coil.grid(row=10, column=0, columnspan=2, sticky="w")
        # Only the relaxed coil sets its tube radius freely; the swept
        # helix takes it from the lattice-quantised frequency.
        self.frame_coil_tube = ttk.Frame(self.frame_coil)
        self.frame_coil_tube.grid(row=11, column=0, columnspan=2, sticky="ew")
        self.frame_coil_tube.columnconfigure(0, weight=1)
        self._param(self.frame_coil_tube, "Tube radius (Å)",
                    self.var_coil_tube_radius, 4.0, 12.0, 0, resolution=0.1,
                    hard_lo=2.0, hard_hi=30.0, command=self._update_coil_hint)
        # Seen down the axis a real coil is a polygon, not a circle. Liu
        # et al. show the (6,6) coil's top view as a hexagonal torus and
        # say it matches what is observed; measured here, concentrating
        # the curvature at six knees puts every pentagon on the outer
        # wall, where the smooth helix leaves two of eleven inside.
        self._coil_sides = self._param(
            self.frame_coil_tube, "Sides per turn (0 = smooth)",
            self.var_coil_sides, 0, 12, 2, integer=True, hard_hi=24,
            command=self._update_coil_hint)
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

        # --- haeckelite
        self.frame_haeckelite = ttk.LabelFrame(
            parent, text="Haeckelite (patterned Stone-Wales)", padding=8)
        self.frame_haeckelite.columnconfigure(0, weight=1)
        ttk.Label(self.frame_haeckelite, text="Pattern").grid(
            row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_haeckelite, textvariable=self.var_hk_pattern,
                     values=list(HAECKELITE_PATTERNS), state="readonly",
                     width=10).grid(row=0, column=1, sticky="e", pady=(0, 6))
        # Each `_param` takes two grid rows -- its label and entry, then its
        # slider -- so these run 1, 3, 5, 7 and the hint lands on 9.
        self._param(self.frame_haeckelite, "Cells along x", self.var_hk_nx,
                    3, 10, 1, integer=True, hard_lo=3, hard_hi=20,
                    command=self._update_haeckelite_hint)
        self._param(self.frame_haeckelite, "Cells along y", self.var_hk_ny,
                    2, 10, 3, integer=True, hard_lo=2, hard_hi=20,
                    command=self._update_haeckelite_hint)
        self._param(self.frame_haeckelite, "Period (stripes, sparse)",
                    self.var_hk_period, 1, 6, 5, integer=True,
                    hard_lo=1, hard_hi=12)
        self._param(self.frame_haeckelite, "Density (random)",
                    self.var_hk_density, 0.05, 1.0, 7, resolution=0.05,
                    hard_lo=0.01, hard_hi=1.0)
        self.lbl_haeckelite = ttk.Label(
            self.frame_haeckelite, text="", foreground=MUTED,
            font=("TkDefaultFont", 8), wraplength=230, justify="left")
        self.lbl_haeckelite.grid(row=9, column=0, columnspan=2, sticky="w")
        # The combobox has no `_param` trace of its own, and the hint's text
        # depends on the pattern ("none" returns graphene), so without this
        # it would describe the previously selected one.
        self.var_hk_pattern.trace_add(
            "write", lambda *_: self._update_haeckelite_hint())

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

        # --- 3D interconnected nanotube network
        # --- nanocone
        self.frame_nc = ttk.LabelFrame(parent, text="Nanocone", padding=8)
        self.frame_nc.columnconfigure(0, weight=1)
        self._param(self.frame_nc, "Pentagons at apex", self.var_nc_pent,
                    1, 3, 0, integer=True, command=self._update_nc_hint)
        self._param(self.frame_nc, "Slant radius (Å)", self.var_nc_radius,
                    8.0, 60.0, 2, resolution=1.0,
                    command=self._update_nc_hint)
        ttk.Checkbutton(
            self.frame_nc,
            text="Only the clean cone (1 pentagon)",
            variable=self.var_nc_strict,
            command=self._update_nc_hint).grid(row=4, column=0, columnspan=2,
                                               sticky="w")
        self.lbl_nc = ttk.Label(self.frame_nc, text="", foreground=MUTED,
                                wraplength=260, justify="left")
        self.lbl_nc.grid(row=5, column=0, columnspan=2, sticky="w")

        # --- toroid, polyhex route
        self.frame_tp = ttk.LabelFrame(parent, text="Toroid (all hexagons)",
                                       padding=8)
        self.frame_tp.columnconfigure(0, weight=1)
        self._param(self.frame_tp, "Chiral index n", self.var_tp_n,
                    2, 20, 0, integer=True, command=self._update_tp_hint)
        self._param(self.frame_tp, "Chiral index m", self.var_tp_m,
                    0, 20, 2, integer=True, command=self._update_tp_hint)
        self._param(self.frame_tp, "Periods round the ring",
                    self.var_tp_periods, 10, 400, 4, integer=True,
                    command=self._update_tp_hint)
        self.lbl_tp = ttk.Label(self.frame_tp, text="", foreground=MUTED,
                                wraplength=260, justify="left")
        self.lbl_tp.grid(row=6, column=0, columnspan=2, sticky="w")

        # --- toroid
        self.frame_tor = ttk.LabelFrame(parent, text="Carbon toroid",
                                        padding=8)
        self.frame_tor.columnconfigure(0, weight=1)
        self._param(self.frame_tor, "Ring radius R (Å)", self.var_tor_major,
                    8.0, 80.0, 0, resolution=1.0,
                    command=self._update_tor_hint)
        self._param(self.frame_tor, "Tube radius r (Å)", self.var_tor_minor,
                    2.0, 15.0, 2, resolution=0.5,
                    command=self._update_tor_hint)
        self.lbl_tor = ttk.Label(self.frame_tor, text="", foreground=MUTED,
                                 wraplength=260, justify="left")
        self.lbl_tor.grid(row=4, column=0, columnspan=2, sticky="w")

        # --- heptanene
        self.frame_hp = ttk.LabelFrame(parent, text="Heptanene", padding=8)
        self.frame_hp.columnconfigure(0, weight=1)
        ttk.Checkbutton(
            self.frame_hp,
            text="Refuse if it is not carbon (recommended)",
            variable=self.var_hp_strict).grid(row=0, column=0, columnspan=2,
                                              sticky="w")
        self.lbl_hp = ttk.Label(
            self.frame_hp,
            text=("A trivalent net of nothing but heptagons. The Euler "
                  "budget rules out a sphere (it would need −12 heptagons) "
                  "and a flat periodic sheet (0), so it exists only in "
                  "hyperbolic geometry — and Hilbert's theorem then says "
                  "there is no flat sheet to build at any size. The "
                  "smallest orientable one is the Klein quartic: 24 "
                  "heptagons, 56 atoms, genus 3. Its geometry does not "
                  "reach carbon (bonds 1.11–1.70 Å), and the refusal is "
                  "the result. Untick to look at it anyway."),
            foreground=MUTED, wraplength=260, justify="left")
        self.lbl_hp.grid(row=1, column=0, columnspan=2, sticky="w")

        # --- haeckelite tube
        self.frame_ht = ttk.LabelFrame(parent, text="Haeckelite tube",
                                       padding=8)
        self.frame_ht.columnconfigure(0, weight=1)
        ttk.Label(self.frame_ht, text="Pattern").grid(
            row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_ht, textvariable=self.var_ht_pattern,
                     values=list(HAECKELITE_PATTERNS), state="readonly",
                     width=12).grid(row=0, column=1, sticky="ew")
        ttk.Label(self.frame_ht, text="Roll along").grid(
            row=1, column=0, sticky="w")
        ttk.Combobox(self.frame_ht, textvariable=self.var_ht_roll,
                     values=["a", "b"], state="readonly",
                     width=12).grid(row=1, column=1, sticky="ew")
        self._param(self.frame_ht, "Cells along x", self.var_ht_nx,
                    4, 40, 2, integer=True, command=self._update_ht_hint)
        self._param(self.frame_ht, "Cells along y", self.var_ht_ny,
                    2, 40, 4, integer=True, command=self._update_ht_hint)
        self.lbl_ht = ttk.Label(self.frame_ht, text="", foreground=MUTED,
                                wraplength=260, justify="left")
        self.lbl_ht.grid(row=6, column=0, columnspan=2, sticky="w")
        for _var in (self.var_ht_pattern, self.var_ht_roll):
            _var.trace_add("write", lambda *_: self._update_ht_hint())

        # --- supernetwork
        self.frame_sn = ttk.LabelFrame(parent, text="Supernetwork of tubes",
                                       padding=8)
        self.frame_sn.columnconfigure(0, weight=1)
        ttk.Label(self.frame_sn, text="Net").grid(row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_sn, textvariable=self.var_sn_graph,
                     values=list(SUPERLATTICES) + list(SUPERCAGES),
                     state="readonly",
                     width=18).grid(row=0, column=1, sticky="ew")
        self._param(self.frame_sn, "Cell length (Å)", self.var_sn_scale,
                    20.0, 120.0, 1, resolution=1.0,
                    command=self._update_sn_hint)
        self._param(self.frame_sn, "Tube radius (Å)", self.var_sn_radius,
                    2.0, 15.0, 3, resolution=0.5,
                    command=self._update_sn_hint)
        self._param(self.frame_sn, "Node blend (Å)", self.var_sn_blend,
                    1.0, 12.0, 5, resolution=0.5,
                    command=self._update_sn_hint)
        self.lbl_sn = ttk.Label(self.frame_sn, text="", foreground=MUTED,
                                wraplength=260, justify="left")
        self.lbl_sn.grid(row=7, column=0, columnspan=2, sticky="w")
        self.var_sn_graph.trace_add("write", lambda *_: self._update_sn_hint())

        self.frame_network = ttk.LabelFrame(parent, text="Nanotube network",
                                            padding=8)
        self.frame_network.columnconfigure(0, weight=1)
        ttk.Label(self.frame_network, text="Net").grid(row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_network, textvariable=self.var_net_kind,
                     values=NETWORK_KINDS, state="readonly", width=10).grid(
            row=0, column=1, sticky="e", pady=(0, 6))
        self.var_net_kind.trace_add("write", lambda *_: self._update_network_hint())
        self._param(self.frame_network, "Cell length (Å)", self.var_net_cell,
                    28.0, 90.0, 1, resolution=1.0, hard_lo=20.0, hard_hi=160.0,
                    command=self._update_network_hint)
        self._param(self.frame_network, "Tube radius (Å)", self.var_net_radius,
                    3.0, 12.0, 3, resolution=0.25, hard_lo=2.0, hard_hi=25.0,
                    command=self._update_network_hint)
        self._param(self.frame_network, "Node blend (Å)", self.var_net_blend,
                    2.0, 10.0, 5, resolution=0.5, hard_lo=1.0, hard_hi=20.0,
                    command=self._update_network_hint)
        self.lbl_network = ttk.Label(self.frame_network, text="",
                                     foreground=MUTED,
                                     font=("TkDefaultFont", 8), wraplength=230,
                                     justify="left")
        self.lbl_network.grid(row=7, column=0, columnspan=2, sticky="w",
                              pady=(4, 0))

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
        ttk.Checkbutton(
            self.frame_surface,
            text="Put the 5s and 7s where the curvature wants them",
            variable=self.var_place, command=self._update_surface_hint,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            self.frame_surface,
            text="Hold the wall on its own surface",
            variable=self.var_anchor, command=self._update_surface_hint,
        ).grid(row=4, column=0, columnspan=2, sticky="w")
        self.lbl_surface = ttk.Label(self.frame_surface, text="", foreground=MUTED,
                                     font=("TkDefaultFont", 8), wraplength=230,
                                     justify="left")
        self.lbl_surface.grid(row=5, column=0, columnspan=2, sticky="w")

        # --- chemistry
        self.frame_chem = ttk.LabelFrame(parent, text="Chemistry", padding=8)
        self.frame_chem.columnconfigure(0, weight=1)
        ttk.Label(self.frame_chem, text="Dopant").grid(row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_chem, textvariable=self.var_dopant, values=DOPANTS,
                     state="readonly", width=7).grid(row=0, column=1, sticky="e",
                                                     pady=(0, 4))
        self.var_dopant.trace_add("write", lambda *_: self._update_dopant_hint())
        ttk.Label(self.frame_chem, text="Site").grid(row=1, column=0, sticky="w")
        ttk.Combobox(self.frame_chem, textvariable=self.var_dopant_site,
                     values=list(DOPANT_SITES), state="readonly", width=9).grid(
            row=1, column=1, sticky="e", pady=(0, 6))
        self.var_dopant_site.trace_add("write", lambda *_: self._update_dopant_hint())
        self._param(self.frame_chem, "Concentration", self.var_dopant_conc,
                    MIN_DOPING_FRACTION, MAX_DOPING_FRACTION, 2,
                    resolution=0.005, hard_hi=0.5,
                    command=self._update_dopant_hint)
        # The dopant table's own words, not a paraphrase: what an element
        # does and how much of it is real are exactly what a user picking
        # from a fifteen-item dropdown cannot be expected to know.
        ttk.Label(self.frame_chem, text="Co-dope").grid(row=4, column=0,
                                                        sticky="w")
        ttk.Entry(self.frame_chem, textvariable=self.var_codope,
                  width=12, justify="right").grid(row=4, column=1, sticky="e")
        self.var_codope.trace_add("write", lambda *_: self._update_dopant_hint())
        ttk.Label(self.frame_chem, text="Affinity").grid(row=5, column=0,
                                                        sticky="w")
        ttk.Combobox(self.frame_chem, textvariable=self.var_codope_affinity,
                     values=list(AFFINITIES), state="readonly", width=9).grid(
            row=5, column=1, sticky="e", pady=(0, 6))
        self.var_codope_affinity.trace_add(
            "write", lambda *_: self._update_dopant_hint())
        # The dopant table's own words, not a paraphrase: what an element
        # does and how much of it is real are exactly what a user picking
        # from a fifteen-item dropdown cannot be expected to know.
        self.lbl_dopant = ttk.Label(self.frame_chem, text="", foreground=MUTED,
                                    font=("TkDefaultFont", 8), wraplength=230,
                                    justify="left")
        self.lbl_dopant.grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # --- surface functionalisation, shown for every family
        self.frame_graft = ttk.LabelFrame(parent, text="Surface groups",
                                          padding=8)
        self.frame_graft.columnconfigure(0, weight=1)
        ttk.Label(self.frame_graft, text="Group").grid(row=0, column=0,
                                                       sticky="w")
        ttk.Combobox(self.frame_graft, textvariable=self.var_graft,
                     values=["none", *sorted(GROUPS)], state="readonly",
                     width=10).grid(row=0, column=1, sticky="e", pady=(0, 4))
        self.var_graft.trace_add("write", lambda *_: self._update_graft_hint())
        ttk.Label(self.frame_graft, text="Sites").grid(row=1, column=0,
                                                       sticky="w")
        ttk.Combobox(self.frame_graft, textvariable=self.var_graft_where,
                     values=["all", "edge", "defect", "ring:5", "ring:7"],
                     state="readonly", width=10).grid(row=1, column=1,
                                                      sticky="e", pady=(0, 4))
        self.var_graft_where.trace_add("write",
                                       lambda *_: self._update_graft_hint())
        ttk.Label(self.frame_graft, text="Face").grid(row=2, column=0,
                                                      sticky="w")
        ttk.Combobox(self.frame_graft, textvariable=self.var_graft_face,
                     values=["outer", "inner", "both"], state="readonly",
                     width=10).grid(row=2, column=1, sticky="e", pady=(0, 6))
        self.var_graft_face.trace_add("write",
                                      lambda *_: self._update_graft_hint())
        self._param(self.frame_graft, "Coverage", self.var_graft_coverage,
                    0.01, 1.0, 2, resolution=0.01,
                    command=self._update_graft_hint)
        # Free text rather than a dropdown: the swap is a mapping, and
        # which elements are offered depends on the group chosen -- an
        # -OH reaches S/Se/Te, an -NH2 reaches P/As/B. The hint below
        # lists what the current group actually accepts.
        ttk.Label(self.frame_graft, text="Swap").grid(row=5, column=0,
                                                      sticky="w")
        ttk.Entry(self.frame_graft, textvariable=self.var_graft_swap,
                  width=12).grid(row=5, column=1, sticky="e", pady=(4, 0))
        self.var_graft_swap.trace_add("write",
                                      lambda *_: self._update_graft_hint())
        self.lbl_graft = ttk.Label(self.frame_graft, text="", foreground=MUTED,
                                   font=("TkDefaultFont", 8), wraplength=230,
                                   justify="left")
        self.lbl_graft.grid(row=6, column=0, columnspan=2, sticky="w",
                            pady=(4, 0))

        # --- dichalcogenide: material and phase, shared by all TMD modes
        self.frame_tmd = ttk.LabelFrame(parent, text="Dichalcogenide", padding=8)
        self.frame_tmd.columnconfigure(0, weight=1)
        # Metal and chalcogen rather than one formula list: that is how the
        # choice is actually made, and it keeps a 27-entry dropdown from
        # being the only way to find WSe2. The chalcogen list is narrowed
        # to what the chosen metal actually forms a layered MX2 with, so
        # an impossible pair cannot be selected at all.
        ttk.Label(self.frame_tmd, text="Metal").grid(row=0, column=0, sticky="w")
        self.cmb_tmd_metal = ttk.Combobox(
            self.frame_tmd, textvariable=self.var_tmd_metal,
            values=list(available_metals()), state="readonly", width=8)
        self.cmb_tmd_metal.grid(row=0, column=1, sticky="e", pady=(0, 4))
        ttk.Label(self.frame_tmd, text="Chalcogen").grid(row=1, column=0,
                                                         sticky="w")
        self.cmb_tmd_chalcogen = ttk.Combobox(
            self.frame_tmd, textvariable=self.var_tmd_chalcogen,
            values=list(chalcogens_for("Mo")), state="readonly", width=8)
        self.cmb_tmd_chalcogen.grid(row=1, column=1, sticky="e", pady=(0, 4))
        self.var_tmd_metal.trace_add("write", lambda *_: self._on_metal_change())
        self.var_tmd_chalcogen.trace_add(
            "write", lambda *_: self._sync_material_from_elements())
        self.var_tmd_material.trace_add(
            "write", lambda *_: self._sync_elements_from_material())
        ttk.Label(self.frame_tmd, text="Phase").grid(row=2, column=0, sticky="w")
        ttk.Combobox(self.frame_tmd, textvariable=self.var_tmd_phase,
                     values=TMD_PHASES, state="readonly", width=8).grid(
            row=2, column=1, sticky="e", pady=(0, 4))
        self.var_tmd_phase.trace_add("write", lambda *_: self._update_tmd_hint())
        self.lbl_tmd = ttk.Label(self.frame_tmd, text="", foreground=MUTED,
                                 font=("TkDefaultFont", 8), wraplength=230,
                                 justify="left")
        self.lbl_tmd.grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # --- dichalcogenide chemistry: the MX2 counterpart of doping
        self.frame_tmd_chem = ttk.LabelFrame(parent, text="MX2 chemistry",
                                             padding=8)
        self.frame_tmd_chem.columnconfigure(0, weight=1)
        ttk.Label(self.frame_tmd_chem, text="Edit").grid(row=0, column=0,
                                                         sticky="w")
        ttk.Combobox(self.frame_tmd_chem, textvariable=self.var_tmd_edit,
                     values=["none", *TMD_EDITS], state="readonly",
                     width=10).grid(row=0, column=1, sticky="e", pady=(0, 4))
        self.var_tmd_edit.trace_add("write", lambda *_: self._on_tmd_edit_change())
        ttk.Label(self.frame_tmd_chem, text="Element").grid(row=1, column=0,
                                                            sticky="w")
        self.cmb_tmd_edit_element = ttk.Combobox(
            self.frame_tmd_chem, textvariable=self.var_tmd_edit_element,
            values=list(available_chalcogens()), state="readonly", width=10)
        self.cmb_tmd_edit_element.grid(row=1, column=1, sticky="e", pady=(0, 4))
        self.var_tmd_edit_element.trace_add(
            "write", lambda *_: self._update_tmd_edit_hint())
        self._param(self.frame_tmd_chem, "Amount", self.var_tmd_edit_amount,
                    0.0, 1.0, 2, resolution=0.05, hard_hi=200.0,
                    command=self._update_tmd_edit_hint)
        self.lbl_tmd_chem = ttk.Label(self.frame_tmd_chem, text="",
                                      foreground=MUTED,
                                      font=("TkDefaultFont", 8), wraplength=230,
                                      justify="left")
        self.lbl_tmd_chem.grid(row=4, column=0, columnspan=2, sticky="w",
                               pady=(4, 0))

        # --- layers / bulk
        self.frame_tmd_layers = ttk.LabelFrame(parent, text="Layers", padding=8)
        self.frame_tmd_layers.columnconfigure(0, weight=1)
        self._param(self.frame_tmd_layers, "Layers", self.var_tmd_layers,
                    1, 8, 0, integer=True, hard_hi=60)
        ttk.Label(self.frame_tmd_layers, text="Stacking").grid(row=2, column=0,
                                                               sticky="w")
        ttk.Combobox(self.frame_tmd_layers, textvariable=self.var_tmd_stacking,
                     values=TMD_STACKINGS, state="readonly", width=8).grid(
            row=2, column=1, sticky="e", pady=(0, 4))
        self._param(self.frame_tmd_layers, "Supercell nx", self.var_tmd_nx,
                    1, 8, 3, integer=True, hard_hi=40)
        self._param(self.frame_tmd_layers, "Supercell ny", self.var_tmd_ny,
                    1, 8, 5, integer=True, hard_hi=40)
        ttk.Label(self.frame_tmd_layers,
                  text="2H alternates a 180° rotation (bulk MoS2); 3R shifts "
                       "each layer without rotating; AA is eclipsed. Both 2H "
                       "and 3R put the metal over the chalcogen below — they "
                       "differ at the third layer.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=7, column=0, columnspan=2, sticky="w")

        # --- ribbon
        self.frame_tmd_ribbon = ttk.LabelFrame(parent, text="Ribbon", padding=8)
        self.frame_tmd_ribbon.columnconfigure(0, weight=1)
        self._param(self.frame_tmd_ribbon, "Width (rows)", self.var_tmd_width,
                    2, 20, 0, integer=True, hard_hi=200)
        self._param(self.frame_tmd_ribbon, "Length (cells)", self.var_tmd_length,
                    1, 10, 2, integer=True, hard_hi=100)
        ttk.Label(self.frame_tmd_ribbon, text="Edge").grid(row=4, column=0,
                                                           sticky="w")
        ttk.Combobox(self.frame_tmd_ribbon, textvariable=self.var_tmd_edge,
                     values=TMD_EDGES, state="readonly", width=9).grid(
            row=4, column=1, sticky="e", pady=(0, 4))
        ttk.Label(self.frame_tmd_ribbon, text="Termination").grid(row=5, column=0,
                                                                  sticky="w")
        ttk.Combobox(self.frame_tmd_ribbon, textvariable=self.var_tmd_termination,
                     values=TMD_TERMINATIONS, state="readonly", width=9).grid(
            row=5, column=1, sticky="e", pady=(0, 4))
        ttk.Label(self.frame_tmd_ribbon,
                  text="MX2's two zigzag edges are chemically different: the "
                       "metal-terminated one is metallic and magnetic and "
                       "shapes CVD-grown triangles. Terminating both alike "
                       "leaves the ribbon deliberately off-stoichiometry.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=6, column=0, columnspan=2, sticky="w")

        # --- nanotube
        self.frame_tmd_tube = ttk.LabelFrame(parent, text="Nanotube", padding=8)
        self.frame_tmd_tube.columnconfigure(0, weight=1)
        self._param(self.frame_tmd_tube, "Chiral n", self.var_tmd_n,
                    10, 80, 0, integer=True, hard_hi=400,
                    command=self._update_tmd_hint)
        self._param(self.frame_tmd_tube, "Chiral m", self.var_tmd_m,
                    0, 80, 2, integer=True, hard_hi=400,
                    command=self._update_tmd_hint)
        self._param(self.frame_tmd_tube, "Length (cells)", self.var_tmd_length,
                    1, 10, 4, integer=True, hard_hi=100)
        self.lbl_tmd_tube = ttk.Label(self.frame_tmd_tube, text="",
                                      foreground=MUTED,
                                      font=("TkDefaultFont", 8), wraplength=230,
                                      justify="left")
        self.lbl_tmd_tube.grid(row=6, column=0, columnspan=2, sticky="w")

        # --- coil (a swept nanotube, so it reuses the chiral indices above)
        self.frame_tmd_coil = ttk.LabelFrame(parent, text="Coil", padding=8)
        self.frame_tmd_coil.columnconfigure(0, weight=1)
        self._param(self.frame_tmd_coil, "Coil radius (Å)",
                    self.var_tmd_coil_radius, 60.0, 1200.0, 0,
                    resolution=5.0, hard_hi=20000.0,
                    command=self._update_tmd_hint)
        self._param(self.frame_tmd_coil, "Pitch (Å)", self.var_tmd_coil_pitch,
                    20.0, 400.0, 2, resolution=5.0, hard_hi=5000.0,
                    command=self._update_tmd_hint)
        self._param(self.frame_tmd_coil, "Turns", self.var_tmd_coil_turns,
                    0.1, 3.0, 4, resolution=0.05, hard_hi=20.0,
                    command=self._update_tmd_hint)
        ttk.Label(self.frame_tmd_coil, text="Handedness").grid(
            row=6, column=0, sticky="w")
        ttk.Combobox(self.frame_tmd_coil, textvariable=self.var_tmd_coil_hand,
                     values=["right", "left"], state="readonly",
                     width=12).grid(row=6, column=1, sticky="ew")
        self.lbl_tmd_coil = ttk.Label(self.frame_tmd_coil, text="",
                                      foreground=MUTED,
                                      font=("TkDefaultFont", 8), wraplength=230,
                                      justify="left")
        self.lbl_tmd_coil.grid(row=7, column=0, columnspan=2, sticky="w")

        # --- schwarzite
        self.frame_tmd_sw = ttk.LabelFrame(parent, text="Schwarzite", padding=8)
        self.frame_tmd_sw.columnconfigure(0, weight=1)
        ttk.Label(self.frame_tmd_sw, text="Surface").grid(row=0, column=0,
                                                          sticky="w")
        ttk.Combobox(self.frame_tmd_sw, textvariable=self.var_tmd_sw_kind,
                     values=["primitive", "diamond", "gyroid"],
                     state="readonly", width=10).grid(row=0, column=1,
                                                      sticky="e", pady=(0, 4))
        self._param(self.frame_tmd_sw, "Cell (Å)", self.var_tmd_sw_cell,
                    30.0, 80.0, 1, resolution=1.0, hard_hi=200.0,
                    command=self._update_tmd_hint)
        ttk.Label(self.frame_tmd_sw, text="M/X parity").grid(row=3, column=0,
                                                             sticky="w")
        ttk.Combobox(self.frame_tmd_sw, textvariable=self.var_tmd_sw_parity,
                     values=["none", "flip", "split"], state="readonly",
                     width=10).grid(row=3, column=1, sticky="e", pady=(0, 4))
        self.var_tmd_sw_parity.trace_add(
            "write", lambda *_: (self._update_tmd_hint(),
                                 self._schedule_estimate()))
        self.lbl_tmd_sw = ttk.Label(self.frame_tmd_sw, text="",
                                    foreground=MUTED,
                                    font=("TkDefaultFont", 8), wraplength=230,
                                    justify="left")
        self.lbl_tmd_sw.grid(row=4, column=0, columnspan=2, sticky="w")

        # --- junction
        self.frame_tmd_j = ttk.LabelFrame(parent, text="Junction", padding=8)
        self.frame_tmd_j.columnconfigure(0, weight=1)
        ttk.Label(self.frame_tmd_j, text="Kind").grid(row=0, column=0,
                                                      sticky="w")
        ttk.Combobox(self.frame_tmd_j, textvariable=self.var_tmd_j_kind,
                     values=["L", "T", "Y", "X"], state="readonly",
                     width=6).grid(row=0, column=1, sticky="e", pady=(0, 4))
        self._param(self.frame_tmd_j, "Tube radius (Å)", self.var_tmd_j_radius,
                    8.0, 30.0, 1, resolution=0.5, hard_hi=200.0,
                    command=self._update_tmd_hint)
        self._param(self.frame_tmd_j, "Arm length (Å)", self.var_tmd_j_arm,
                    15.0, 60.0, 3, resolution=1.0, hard_hi=400.0,
                    command=self._update_tmd_hint)
        self._param(self.frame_tmd_j, "Blend (Å)", self.var_tmd_j_blend,
                    1.0, 12.0, 5, resolution=0.5, hard_hi=50.0)
        ttk.Label(self.frame_tmd_j, text="M/X parity").grid(row=7, column=0,
                                                            sticky="w")
        ttk.Combobox(self.frame_tmd_j, textvariable=self.var_tmd_j_parity,
                     values=["none", "flip", "split"], state="readonly",
                     width=8).grid(row=7, column=1, sticky="e", pady=(0, 4))
        self.var_tmd_j_parity.trace_add(
            "write", lambda *_: (self._update_tmd_hint(),
                                 self._schedule_estimate()))
        self.lbl_tmd_j = ttk.Label(self.frame_tmd_j, text="", foreground=MUTED,
                                   font=("TkDefaultFont", 8), wraplength=230,
                                   justify="left")
        self.lbl_tmd_j.grid(row=8, column=0, columnspan=2, sticky="w")

        # --- heterostructures
        from ..hetero import available_layers

        layer_names = list(available_layers())
        self.frame_het = ttk.LabelFrame(parent, text="Layers", padding=8)
        self.frame_het.columnconfigure(0, weight=1)
        ttk.Label(self.frame_het, text="Bottom").grid(row=0, column=0, sticky="w")
        ttk.Combobox(self.frame_het, textvariable=self.var_het_bottom,
                     values=layer_names, state="readonly", width=10).grid(
            row=0, column=1, sticky="e", pady=(0, 4))
        ttk.Label(self.frame_het, text="Top").grid(row=1, column=0, sticky="w")
        ttk.Combobox(self.frame_het, textvariable=self.var_het_top,
                     values=["same", *layer_names], state="readonly",
                     width=10).grid(row=1, column=1, sticky="e", pady=(0, 4))
        ttk.Label(self.frame_het, text="Third layer").grid(row=2, column=0,
                                                           sticky="w")
        ttk.Combobox(self.frame_het, textvariable=self.var_het_third,
                     values=["none", *layer_names], state="readonly",
                     width=10).grid(row=2, column=1, sticky="e", pady=(0, 4))
        self._param(self.frame_het, "Gap (Å)", self.var_het_gap,
                    2.5, 6.0, 3, resolution=0.05, hard_hi=20.0)
        self.lbl_het = ttk.Label(self.frame_het, text="", foreground=MUTED,
                                 font=("TkDefaultFont", 8), wraplength=230,
                                 justify="left")
        self.lbl_het.grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))
        for var in (self.var_het_bottom, self.var_het_top, self.var_het_third):
            var.trace_add("write", lambda *_: (self._update_het_hint(),
                                               self._schedule_estimate()))

        self.frame_twist = ttk.LabelFrame(parent, text="Twist", padding=8)
        self.frame_twist.columnconfigure(0, weight=1)
        self._param(self.frame_twist, "Angle (deg)", self.var_het_angle,
                    0.5, 30.0, 0, resolution=0.01, hard_hi=60.0,
                    command=self._update_het_hint)
        self._param(self.frame_twist, "Max index m", self.var_het_max_index,
                    5, 60, 2, integer=True, hard_hi=200,
                    command=self._update_het_hint)
        self.lbl_twist = ttk.Label(self.frame_twist, text="", foreground=MUTED,
                                   font=("TkDefaultFont", 8), wraplength=230,
                                   justify="left")
        self.lbl_twist.grid(row=4, column=0, columnspan=2, sticky="w")

        self.frame_stack = ttk.LabelFrame(parent, text="Supercell", padding=8)
        self.frame_stack.columnconfigure(0, weight=1)
        self._param(self.frame_stack, "nx", self.var_het_nx, 1, 8, 0,
                    integer=True, hard_hi=40)
        self._param(self.frame_stack, "ny", self.var_het_ny, 1, 8, 2,
                    integer=True, hard_hi=40)

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
        # A bouncing bar says "something is happening" and nothing else,
        # which is exactly the doubt a six-minute coil creates. The clock
        # says how long it has been happening for, and confirms on every
        # tick that the worker process is still alive -- so a build that
        # has genuinely died stops looking like one that is merely slow.
        self.lbl_elapsed = ttk.Label(actions, text="", foreground=MUTED,
                                     font=("TkDefaultFont", 8))
        self.lbl_elapsed.pack(anchor="w")
        self.lbl_estimate = ttk.Label(actions, text="", foreground=MUTED,
                                      font=("TkDefaultFont", 8), wraplength=240,
                                      justify="left")
        self.lbl_estimate.pack(anchor="w", pady=(4, 0))

        # Any parameter change re-costs the build. Debounced, because
        # dragging a slider fires this on every pixel.
        for var in self._params.values():
            var.trace_add("write", lambda *_: self._schedule_estimate())

        self._update_radius_hint()
        self._update_bend_hint()
        self._update_strain_hint()
        self._update_coil_hint()
        self._update_surface_hint()
        self._update_cage_hint()
        self._on_mode_change()

    # ------------------------------------------------------------ visibility
    def _update_open_tube_hint(self) -> None:
        """Diameter, chirality and the true length, from (n, m).

        A tube is an integer number of translational periods, so the
        length that comes out is almost never the length asked for. Saying
        so here is cheaper than letting it surface as a surprise in the
        exported cell.
        """
        n, m = int(self.var_cnt_n.get()), int(self.var_cnt_m.get())
        if m > n:
            self.lbl_open_tube.config(
                text=f"({n},{m}) is the same tube as ({m},{n}); the "
                     "convention is m <= n, so swap them.", foreground=BAD_RED)
            return
        if n < 1:
            self.lbl_open_tube.config(text="n must be at least 1.",
                                      foreground=BAD_RED)
            return

        bond = float(self.var_bond.get())
        q = n * n + n * m + m * m
        diameter = math.sqrt(3.0) * bond * math.sqrt(q) / math.pi
        divisor = math.gcd(2 * m + n, 2 * n + m)
        period = 3.0 * bond * math.sqrt(q) / divisor
        cells = max(1, math.ceil(float(self.var_cnt_length.get()) / period))
        family = ("armchair" if n == m else
                  "zigzag" if m == 0 else "chiral")
        # The classic (n-m) mod 3 rule: metallic when it divides, and for
        # a chiral tube that is the only thing that decides it.
        character = "metallic" if (n - m) % 3 == 0 else "semiconducting"
        self.lbl_open_tube.config(
            text=f"{family}, {character} by the (n−m) mod 3 rule. "
                 f"Diameter {diameter:.2f} Å. {cells} period"
                 f"{'s' if cells != 1 else ''} of {period:.2f} Å "
                 f"= {cells * period:.1f} Å, {cells * 4 * q // divisor} atoms.",
            foreground=MUTED)

    def _update_ribbon_hint(self) -> None:
        """Width in Å, the periodic axis, and what the edge decides.

        The two edges are not two looks of the same thing: at the same
        width in rows an armchair ribbon has twice the atoms of a zigzag
        one and a gap that closes and reopens in a period-three pattern,
        while every zigzag ribbon carries the flat edge band. Saying which
        is which here is cheaper than finding out from a band structure.
        """
        edge = self.var_rib_edge.get()
        width = int(self.var_rib_width.get())
        length = int(self.var_rib_length.get())
        bond = float(self.var_bond.get())
        per_row = 2 if edge == "zigzag" else 4
        atoms = per_row * width * length
        if self.var_rib_passivate.get():
            atoms += per_row * length
        # Measured off real builds rather than guessed from the lattice:
        # a row adds 3/2 bonds across a zigzag ribbon and sqrt(3) across an
        # armchair one, but both series start half a row short of the
        # obvious multiple, and the obvious multiple was over by that much
        # at every width.
        across = (bond * (1.5 * width - 1.0) if edge == "zigzag"
                  else math.sqrt(3.0) * bond * (width - 0.5))
        along = (math.sqrt(3.0) * bond if edge == "zigzag"
                 else 3.0 * bond) * length
        if edge == "zigzag":
            physics = ("flat edge band at the Fermi level — the magnetism "
                       "people build these for")
        else:
            # Delta(3p+1) > Delta(3p) > Delta(3p+2): the width counts dimer
            # lines here, which is what ASE's own convention means by it
            # (an armchair cell holds 4 atoms per line).
            gaps = {1: "widest gap", 0: "middling gap", 2: "narrowest gap"}
            physics = (f"{gaps[width % 3]} of the three armchair families "
                       f"(Δ is largest at 3p+1, smallest at 3p+2; this is "
                       f"{width})")
        self.lbl_ribbon.config(
            text=f"{across:.1f} Å across × {along:.1f} Å along, {atoms} atoms. "
                 f"Periodic along z only, so it runs on for ever along its "
                 f"length and ends in two edges across it. {physics}.",
            foreground=MUTED)

    def _on_tube_shape(self) -> None:
        """Switch between the straight tube and the wound one.

        They are different builders, and the difference is visible in the
        result: a straight tube is periodic along its axis and exact, a
        wound one is a finite object whose lattice has been bent. Picking
        the shape therefore moves the mode dropdown rather than leaving it
        saying "nanotube (open)" while something else is built.
        """
        if self._applying_values:
            return
        wanted = ("nanocoil" if self.var_cnt_shape.get() == "helix"
                  else "nanotube (open)")
        if self.var_mode_kind.get() != wanted:
            self.var_mode_kind.set(wanted)

    def _update_nanocoil_hint(self) -> None:
        """What this (n, m) tube needs before its wall survives winding.

        This route bends a finished hexagonal lattice, so the wall can
        only stretch -- nothing here relieves curvature the way a meshed
        coil does with 5-7 pairs. Two separate things break it, and the
        second is the one nobody expects: turns colliding does not depend
        on the coil radius at all.
        """
        from ..builders.nanocoil import TURN_CLEARANCE, _clean_coil_radius

        n, m = int(self.var_cnt_n.get()), int(self.var_cnt_m.get())
        if m > n or n < 1:
            self.lbl_open_tube.config(
                text="Need n ≥ 1 and 0 ≤ m ≤ n.", foreground=BAD_RED)
            return
        bond = float(self.var_bond.get())
        radius = math.sqrt(3.0) * bond * math.sqrt(n * n + n * m + m * m) / (2 * math.pi)
        coil_radius = float(self.var_coil_radius.get())
        pitch = float(self.var_coil_pitch.get())
        clearance = 2.0 * radius + TURN_CLEARANCE
        strain = radius * coil_radius / (coil_radius ** 2 + (pitch / (2 * math.pi)) ** 2)
        clean_from = _clean_coil_radius(radius, pitch)

        if pitch < clearance:
            self.lbl_open_tube.config(
                text=f"Pitch {pitch:.1f} Å is less than this tube's "
                     f"{2 * radius:.1f} Å width plus a graphite gap, so the "
                     f"turns would pass through each other whatever the coil "
                     f"radius. Raise the pitch above {clearance:.1f} Å.",
                foreground=BAD_RED)
            return
        atoms = self._scaled_estimate()
        if strain <= 0.08:
            self.lbl_open_tube.config(
                text=f"wall strain {strain:.1%} — graphitic. Tube ⌀"
                     f"{2 * radius:.1f} Å on a {coil_radius:.0f} Å coil, "
                     f"{atoms}. Every ring a hexagon: this winds the real "
                     f"(n,m) lattice instead of meshing a surface.",
                foreground=OK_GREEN)
        elif strain <= 0.15:
            self.lbl_open_tube.config(
                text=f"wall strain {strain:.1%} — intact but visibly "
                     f"stretched. Clean from about {clean_from:.0f} Å of coil "
                     f"radius at this pitch. {atoms}.",
                foreground=WARN_AMBER)
        else:
            self.lbl_open_tube.config(
                text=f"wall strain {strain:.1%} — this tears rather than "
                     f"winds, and the builder will refuse it. A "
                     f"{2 * radius:.1f} Å tube needs about {clean_from:.0f} Å "
                     f"of coil radius at this pitch; small coils cannot be "
                     f"graphitic, which is why real ones are hundreds of Å "
                     f"across.",
                foreground=BAD_RED)

    def _scaled_estimate(self) -> str:
        """The atom count for the current controls, as a phrase."""
        try:
            return f"{estimate_atoms(self.current_job())} atoms"
        except (tk.TclError, ValueError, KeyError):
            return "size unknown"

    def _on_mode_written(self) -> None:
        """React to `mode_kind` changing, whoever changed it.

        A preset writes the same variable a user does, so the reset is
        gated on who is writing rather than on the write itself -- and on
        the value actually differing, since Tk fires a trace on every set
        including one that changes nothing.
        """
        mode = self.var_mode_kind.get()
        if not self._applying_values and mode != self._chemistry_mode:
            self._reset_chemistry(mode)
        self._on_mode_change()

    def _reset_chemistry(self, mode: str) -> None:
        """Put the dopant, edit and grafting controls back to neutral."""
        changed = [key for key, value in NEUTRAL_CHEMISTRY.items()
                   if (var := self._params.get(key)) is not None
                   and var.get() != value]
        for key in changed:
            self._params[key].set(NEUTRAL_CHEMISTRY[key])
        self._chemistry_mode = mode
        if changed:
            self._set_status(
                f"Chemistry reset for “{mode}”: "
                + ", ".join(sorted(changed)) + " back to default."
            )

    def _on_mode_change(self) -> None:
        """Show only the panels that apply to the selected structure type."""
        mode = self.var_mode_kind.get()
        # Undo what the nanocoil branch hides, before any branch runs:
        # leaving a row hidden after leaving that mode is the same fault
        # as showing a dead one, in the other direction.
        if mode != "nanocoil":
            for widget in (*self._coil_taper, *self._cnt_length):
                widget.grid()
            self.frame_coil_hand.grid()
        for frame in (self.frame_tube, self.frame_open_tube,
                      self.frame_ribbon,
                      self.frame_centreline, self.frame_defects,
                      self.frame_coil, self.frame_junction, self.frame_schwarzite,
                      self.frame_haeckelite,
                      self.frame_cage, self.frame_mw, self.frame_bundle,
                      self.frame_network,
                      self.frame_ht, self.frame_sn, self.frame_hp,
                      self.frame_tor, self.frame_tp, self.frame_nc,
                      self.frame_tmd, self.frame_tmd_layers,
                      self.frame_tmd_ribbon, self.frame_tmd_tube,
                      self.frame_tmd_coil, self.frame_tmd_sw,
                      self.frame_het, self.frame_twist, self.frame_stack,
                      self.frame_tmd_j, self.frame_tmd_chem,
                      self.frame_surface, self.frame_chem, self.frame_graft):
            frame.pack_forget()

        if mode in ("twisted bilayer", "vdW stack"):
            # A stack is neither carbon nor dichalcogenide: none of the
            # sp2 controls (annealing, roughness, doping) nor the TMD
            # phase apply, so only the layer panels show.
            self.frame_het.pack(fill="x")
            if mode == "twisted bilayer":
                self.frame_twist.pack(fill="x", pady=(8, 0))
            else:
                self.frame_stack.pack(fill="x", pady=(8, 0))
            self._update_het_hint()
            self.frame_graft.pack(fill="x", pady=(8, 0))
            self._update_graft_hint()
            self._schedule_estimate()
            return

        if mode.startswith("TMD"):
            # Every dichalcogenide needs the material and phase; the rest
            # depends on which structure. None of the carbon panels apply:
            # annealing, roughness and doping are all sp2-specific.
            self.frame_tmd.pack(fill="x")
            if mode in ("TMD layers", "TMD bulk"):
                self.frame_tmd_layers.pack(fill="x", pady=(8, 0))
            elif mode == "TMD ribbon":
                self.frame_tmd_ribbon.pack(fill="x", pady=(8, 0))
            elif mode == "TMD nanotube":
                self.frame_tmd_tube.pack(fill="x", pady=(8, 0))
            elif mode == "TMD schwarzite":
                self.frame_tmd_sw.pack(fill="x", pady=(8, 0))
            elif mode == "TMD junction":
                self.frame_tmd_j.pack(fill="x", pady=(8, 0))
            elif mode == "TMD coil":
                # The coil is a swept nanotube, so it needs the chiral
                # indices as well as the helix panel.
                self.frame_tmd_tube.pack(fill="x", pady=(8, 0))
                self.frame_tmd_coil.pack(fill="x", pady=(8, 0))
            self.frame_tmd_chem.pack(fill="x", pady=(8, 0))
            self._update_tmd_hint()
            self._update_tmd_edit_hint()
            self.frame_graft.pack(fill="x", pady=(8, 0))
            self._update_graft_hint()
            self._schedule_estimate()
            return

        if mode == "junction":
            self.frame_junction.pack(fill="x")
            # Same reason as the schwarzite and the network, and measured
            # on all four kinds: the 5-7 pairs spread over the surface are
            # how the net takes up its curvature, and annealing most of
            # them away leaves the survivors to carry all of it -- which
            # buckles the arms. The as-grown wall is the smoother one.
            self.var_anneal.set(0)
        elif mode == "network":
            self.frame_network.pack(fill="x")
            # Same reason as the schwarzite: at a node the 5-7 pairs are
            # how a hexagonal net covers the curvature, so annealing them
            # away only makes the remaining bonds stretch.
            self.var_anneal.set(0)
            self._update_network_hint()
        elif mode == "haeckelite":
            self.frame_haeckelite.pack(fill="x")
            self._update_haeckelite_hint()
        elif mode == "haeckelite tube":
            self.frame_ht.pack(fill="x")
            self._update_ht_hint()
        elif mode == "toroid":
            self.frame_tor.pack(fill="x")
            # Same reason as every other curved surface: the 5-7 pairs
            # are how the net covers the curvature.
            self.var_anneal.set(0)
            self._update_tor_hint()
        elif mode == "nanocone":
            self.frame_nc.pack(fill="x")
            self._update_nc_hint()
        elif mode == "toroid (polyhex)":
            self.frame_tp.pack(fill="x")
            self._update_tp_hint()
        elif mode == "heptanene":
            self.frame_hp.pack(fill="x")
        elif mode == "supernetwork":
            self.frame_sn.pack(fill="x")
            # Same reason as the network and the schwarzite: at a vertex
            # the 5-7 pairs are how a hexagonal net covers the curvature,
            # so annealing them away only stretches the bonds that are
            # left.
            self.var_anneal.set(0)
            self._update_sn_hint()
        elif mode == "schwarzite":
            self.frame_schwarzite.pack(fill="x")
            # Annealing is counterproductive on a minimal surface (it
            # stretches the bonds the 5-7 pairs were relieving), so
            # entering this mode turns the shared slider off rather than
            # letting its 80-sweep default quietly degrade the cell.
            self.var_anneal.set(0)
        elif mode == "coil (periodic, DFT)":
            # One turn by definition, so the turns slider does not apply:
            # the cell *is* one period. Radius, pitch and tube radius do.
            self.frame_coil.pack(fill="x")
            self._update_coil_hint()
        elif mode == "coil (relaxed)":
            self.frame_coil.pack(fill="x")
            # Same rule as the junction and the schwarzite, and the most
            # visible of the three: annealed, this coil's wall comes out
            # 6.96 Å across for a 4.5 Å tube and the helix springs open,
            # which is what stops it reading as a spiral.
            self.var_anneal.set(0)
            # Without this the label keeps whatever the previous mode
            # computed, so entering the relaxed coil showed the swept
            # tube's strain warning -- a number that does not apply here,
            # under a panel where it looked like it did.
            self._update_coil_hint()
        elif mode in ("fullerene", "nano-onion"):
            self.frame_cage.pack(fill="x")
            if mode == "nano-onion":
                self.frame_onion.grid()
            else:
                self.frame_onion.grid_remove()
            self._update_cage_hint()
        elif mode == "nanotube (open)":
            # Only the indices and the length: no cap, so no subdivision
            # frequency, and no centreline, since a periodic tube is
            # straight by definition -- bending it would break the
            # periodicity that makes it usable in a plane-wave code.
            # The defects panel does apply, though: the wall is an sp2
            # sheet like any other, and the edits are made on it and
            # relaxed rather than meshed in.
            self.frame_open_tube.pack(fill="x")
            self.frame_defects.pack(fill="x", pady=(8, 0))
            self._applying_values = True
            self.var_cnt_shape.set("straight")
            self._applying_values = False
            self._cnt_length[0].grid()
            self._cnt_length[1].grid()
            self._cnt_length[2].grid()
            self._update_open_tube_hint()
        elif mode == "nanocoil":
            # The same (n, m) panel, because the tube is the same tube --
            # but its length comes from the helix now, so the length
            # control goes away rather than sitting there ignored.
            self.frame_open_tube.pack(fill="x")
            self.frame_coil.pack(fill="x", pady=(8, 0))
            self.frame_defects.pack(fill="x", pady=(8, 0))
            self._applying_values = True
            self.var_cnt_shape.set("helix")
            self._applying_values = False
            for widget in self._cnt_length:
                widget.grid_remove()
            # Taper and handedness belong to the swept and meshed coils;
            # this builder winds a right-handed cylindrical helix and has
            # no argument for either.
            for widget in self._coil_taper:
                widget.grid_remove()
            self.frame_coil_hand.grid_remove()
            self.frame_coil_tube.grid_remove()
            self._update_nanocoil_hint()
        elif mode == "nanoribbon":
            self.frame_ribbon.pack(fill="x")
            self.frame_defects.pack(fill="x", pady=(8, 0))
            self._update_ribbon_hint()
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
        self.frame_graft.pack(fill="x", pady=(8, 0))
        self._update_dopant_hint()
        self._update_graft_hint()
        self._on_shape_change()
        self._schedule_estimate()

    def _on_shape_change(self) -> None:
        """Show the coil panel where coil dimensions actually apply."""
        if self.var_mode_kind.get() in ("coil (relaxed)",
                                        "coil (periodic, DFT)"):
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
        # The bend's limit is r_tube * angle / length, so both ends of
        # that ratio change here.
        self._update_bend_hint()

    def _update_bend_hint(self) -> None:
        """How much wall this bend stretches, and where this tube's limit is.

        The limit is not a property of the angle. The bend is imposed as
        an arc whose length is the tube's own axial span, so the outer
        wall stretches by ``r_tube * angle / span``: a 30-ring thin tube
        takes 150° and stays intact, while the default 8-ring one tears
        before 57°. Saying which is cheaper than finding out from a
        refusal.
        """
        from ..builders.capped_cnt import RING_RISE
        from ..builders.centerline import DEFAULT_MAX_STRAIN

        degrees = float(self.var_bend.get())
        radius = fm.radius_for_freq(int(self.var_freq.get()),
                                    float(self.var_bond.get()))
        span = RING_RISE * radius * int(self.var_rings.get())
        limit = math.degrees(DEFAULT_MAX_STRAIN * span / radius)
        if degrees <= 0.0:
            self.lbl_bend.config(
                text=f"straight. This tube bends to about {limit:.0f}° before "
                     f"its wall stretches past {DEFAULT_MAX_STRAIN:.0%}; more "
                     "body rings or a lower frequency raises that.",
                foreground=MUTED)
            return
        strain = radius * math.radians(degrees) / span
        colour = (OK_GREEN if strain <= DEFAULT_MAX_STRAIN
                  else WARN_AMBER if strain <= 0.09 else BAD_RED)
        tail = ("" if strain <= DEFAULT_MAX_STRAIN else
                f" Clean to about {limit:.0f}° on this tube — lengthen it "
                "(more rings) or thin it (lower frequency) to bend further.")
        self.lbl_bend.config(
            text=f"outer wall stretched {strain:.1%} over {span:.0f} Å of "
                 f"tube.{tail}",
            foreground=colour)

    def _update_strain_hint(self) -> None:
        value = float(self.var_max_strain.get())
        if value <= 0.10:
            text, colour = "physical sp2 regime", OK_GREEN
        elif value <= 0.15:
            text, colour = "strained but intact — fine for artwork", WARN_AMBER
        else:
            text, colour = "bonds stretch out of the sp2 range", BAD_RED
        self.lbl_strain.config(text=f"{value:.0%}: {text}", foreground=colour)

    def _on_family_change(self) -> None:
        """Repopulate the structure list when the material family changes."""
        family = self.var_family.get()
        modes = list(FAMILIES.get(family, FAMILIES["carbon"]))
        self.cmb_mode.configure(values=modes)
        if self.var_mode_kind.get() not in modes:
            self.var_mode_kind.set(modes[0])
        else:
            self._on_mode_change()

    # The metal/chalcogen pickers and the formula are two views of one
    # choice, so each has to update the other -- and a naive pair of
    # traces would then feed each other forever. `_syncing_material` is
    # the reentrancy guard: whichever side the user touched wins, and the
    # write it makes to the other side does not bounce back.
    def _on_metal_change(self) -> None:
        """Renarrow the chalcogen list, then rebuild the formula.

        Not every metal forms a layered MX2 with every chalcogen -- Sn
        has no telluride, Nb no tabulated one -- so the second dropdown
        is repopulated rather than left offering a pair that would raise
        at build time. If the current chalcogen is not available for the
        new metal, the lightest one that is takes its place.
        """
        if self._syncing_material:
            return
        options = chalcogens_for(self.var_tmd_metal.get())
        if not options:
            return
        self.cmb_tmd_chalcogen.config(values=list(options))
        if self.var_tmd_chalcogen.get() not in options:
            # Writing this fires the chalcogen trace, which rebuilds the
            # formula -- so there is nothing more to do here.
            self.var_tmd_chalcogen.set(options[0])
            return
        self._sync_material_from_elements()

    def _sync_material_from_elements(self) -> None:
        """Formula follows the two pickers."""
        if self._syncing_material:
            return
        try:
            material = material_for(self.var_tmd_metal.get(),
                                    self.var_tmd_chalcogen.get())
        except KeyError:
            return
        self._syncing_material = True
        try:
            self.var_tmd_material.set(material.formula)
        finally:
            self._syncing_material = False
        self._update_tmd_hint()
        self._schedule_estimate()

    def _sync_elements_from_material(self) -> None:
        """Pickers follow the formula, so a preset naming a compound works.

        Presets set ``tmd_material`` because a preset names a compound,
        not a pair of elements. Without this the two dropdowns would go
        on showing the previous material after loading one.
        """
        if self._syncing_material:
            return
        material = TMD_MATERIALS.get(self.var_tmd_material.get())
        if material is None:
            return
        self._syncing_material = True
        try:
            self.cmb_tmd_chalcogen.config(
                values=list(chalcogens_for(material.metal)))
            self.var_tmd_metal.set(material.metal)
            self.var_tmd_chalcogen.set(material.chalcogen)
        finally:
            self._syncing_material = False
        self._update_tmd_hint()

    def _on_tmd_edit_change(self) -> None:
        """Repoint the element list and the amount at the chosen edit.

        One "Amount" field serves all four edits because it means a
        different thing in each -- a fraction, a count, a face -- and four
        fields of which three are always irrelevant would be worse. What
        makes that workable is the hint below saying which it is right
        now, so the number on screen is never ambiguous.
        """
        edit = self.var_tmd_edit.get()
        if edit == "janus":
            self.cmb_tmd_edit_element.config(values=list(available_chalcogens()))
        elif edit == "alloy":
            # Either sublattice: a chalcogen alloys the chalcogens, a
            # metal the metals, and which one follows from the element.
            self.cmb_tmd_edit_element.config(
                values=list(available_metals()) + list(available_chalcogens()))
        # Defect edits introduce no species, so the element box is left
        # showing whatever it had; the hint says it is unused.
        self._update_tmd_edit_hint()
        self._schedule_estimate()

    def _update_tmd_edit_hint(self) -> None:
        """Say what the Amount field means for the current edit."""
        edit = self.var_tmd_edit.get()
        amount = float(self.var_tmd_edit_amount.get())
        element = self.var_tmd_edit_element.get()
        if edit == "none":
            text = ("Pristine MX2. The edits here are the chemistry a "
                    "dichalcogenide actually undergoes — there is no "
                    "substituting a heteroatom for a carbon in one.")
        elif edit == "janus":
            face = "outward" if amount >= 0 else "inward"
            text = (f"Amount is the face: ≥0 puts {element} on the {face} "
                    "one (the outer wall of a tube, the top of a layer), "
                    "negative on the other. Breaks the mirror symmetry, "
                    "switching on an out-of-plane dipole neither parent has.")
        elif edit == "alloy":
            site = ("chalcogen" if element in available_chalcogens() else "metal")
            text = (f"Amount is the fraction of the {site} sublattice "
                    f"replaced by {element} ({amount:.0%}). The achieved "
                    "fraction is reported, since nine sites cannot be split "
                    "in half.")
        elif edit == "vacancies":
            text = (f"Amount is a count: {int(amount)} chalcogen atoms "
                    "removed. The element box is unused. This is the "
                    "commonest point defect in grown MoS2.")
        else:
            text = (f"Amount is a count: {int(amount)} metal atoms put on "
                    "chalcogen sites, as in sulphur-poor growth. The "
                    "element box is unused.")
        self.lbl_tmd_chem.config(text=text)

    def _update_haeckelite_hint(self) -> None:
        """Say what the pattern will reach, and whether this cell can hold it.

        A catalogue entry needs its block to divide the supercell and
        refuses otherwise, so the hint has to say that *before* the build
        runs -- a refusal after the fact is the worst way to learn a number
        had to be a multiple of four. A generative pattern is a request and
        the census is the honest account, so the hint says that too.
        """
        nx = int(self.var_hk_nx.get())
        ny = int(self.var_hk_ny.get())
        pattern = self.var_hk_pattern.get()
        atoms = 4 * nx * ny
        if pattern in HAECKELITE_CATALOGUE:
            block_m, block_n = HAECKELITE_CATALOGUE[pattern]["block"]
            note = HAECKELITE_CATALOGUE[pattern]["note"]
            if nx % block_m or ny % block_n:
                detail = (f"✗ the {pattern} lattice tiles a {block_m}×"
                          f"{block_n} block, so x must be a multiple of "
                          f"{block_m} and y of {block_n}. This cell would be "
                          "refused: a partial block leaves part of the sheet "
                          "hexagonal, which for this lattice is a different "
                          "material rather than an approximation.")
            else:
                detail = (f"{note}. Every rotation is applied, so the census "
                          "is exact. The cell it relaxes to is this "
                          "program's own number, not a published lattice "
                          "constant — re-relax before quoting it.")
        elif pattern == "none":
            detail = ("returns graphene exactly — the baseline every check "
                      "here is calibrated against.")
        else:
            detail = ("a rotation moves bonds, never atoms, so the count is "
                      "the same whatever the pattern. This is a generative "
                      "rule, and most candidates of a dense one are turned "
                      "down, so read the census rather than the name — "
                      "'dense' reaches about 67% non-hexagonal. For a "
                      "lattice with no hexagons at all, pick 'r57'.")
        self.lbl_haeckelite.configure(
            text=f"{atoms} atoms in a {nx}×{ny} supercell; {detail}")

    def _update_network_hint(self) -> None:
        """Say whether the cell actually leaves a tube between the nodes.

        The failure mode here is not obvious from the numbers: shrink the
        cell and the nodes grow into each other until there is no tube
        left, and what comes out is a sponge rather than a network of
        nanotubes. The builder refuses below the floor, so showing that
        floor -- and the free tube length above it -- is the difference
        between a build that fails after two minutes and one that never
        started.
        """
        from ..builders.network import STRUT_FRACTION, minimum_cell

        kind = self.var_net_kind.get()
        cell = float(self.var_net_cell.get())
        radius = float(self.var_net_radius.get())
        blend = float(self.var_net_blend.get())
        try:
            floor = minimum_cell(kind, radius, blend)
        except ValueError:
            return
        strut = STRUT_FRACTION[kind] * cell
        free = strut - 2.0 * (radius + blend)
        coordination = 6 if kind == "cubic" else 4
        angle = "90°" if kind == "cubic" else "109.47°"

        if cell < floor:
            self.lbl_network.config(
                text=(f"Too small: {strut:.0f} Å struts, and each node eats "
                      f"about {radius + blend:.0f} Å of either end, so no tube "
                      f"is left. Needs at least {floor:.0f} Å."),
                foreground=BAD_RED)
            return
        nodes = 1 if kind == "cubic" else 8
        self.lbl_network.config(
            text=(f"{nodes} node(s) per cell, {coordination}-coordinate at "
                  f"{angle}. Struts {strut:.0f} Å, of which {free:.0f} Å is "
                  f"free tube between the nodes. A periodic cell — the tubes "
                  f"leave one face and return through the opposite one, so "
                  f"this is ready for a DFT code as it stands."),
            foreground=MUTED)

    def _update_nc_hint(self) -> None:
        """Say the angle the disclination forces, before the build.

        The apex angle is not a parameter here -- it follows from the
        pentagon count as ``sin(theta/2) = 1 - N/6`` -- so the useful
        thing to show is which of the five possible cones this is, and
        that only the first is good chemistry.
        """
        from ..builders.nanocone import CLEAN, apex_angle

        pentagons = int(self.var_nc_pent.get())
        radius = float(self.var_nc_radius.get())
        angle = apex_angle(pentagons)
        ring = 6 - pentagons
        names = {5: "pentagon", 4: "square", 3: "triangle"}
        if pentagons != CLEAN and bool(self.var_nc_strict.get()):
            self.lbl_nc.config(
                text=(f"✗ {pentagons}×60° puts a {names.get(ring, '?')} at "
                      f"the apex — sound topology, poor chemistry ("
                      f"{'1.339' if pentagons == 2 else '1.230'} Å bonds, "
                      f"{'90' if pentagons == 2 else '60'}° angles). Nature "
                      f"uses {pentagons} separate pentagons instead. Untick "
                      "the box to build it anyway."))
            return
        note = ("" if pentagons == CLEAN else
                f" A {names.get(ring, '?')} apex is strained — this is the "
                "single-disclination form, not what nature does.")
        self.lbl_nc.config(
            text=(f"Apex angle {angle:.1f}° — quantised, not chosen: "
                  "sin(θ/2) = 1 − N/6, so the only cones are 112.9 / 83.6 / "
                  "60.0 / 38.9 / 19.2°, which are the five Krishnan et al. "
                  f"observed. A cone is developable, so the roll is an "
                  f"isometry and the wall stays all-hexagon around a single "
                  f"{names.get(ring, '?')}. Slant {radius:.0f} Å; the base "
                  f"rim is open, as a nanoribbon's edges are.{note}"))

    def _update_tp_hint(self) -> None:
        """Say the radius and the strain before the build.

        This route has no disclinations to relieve curvature with, so
        the outer wall simply stretches by r/R and no relaxation will
        shorten those bonds. That makes the strain the whole decision,
        and it is worth showing while the slider moves rather than as a
        warning afterwards.
        """
        from ..builders.toroid import POLYHEX_MAX_STRAIN, POLYHEX_TEAR_STRAIN

        n = int(self.var_tp_n.get())
        m = int(self.var_tp_m.get())
        periods = int(self.var_tp_periods.get())
        if m > n or n < 2 or periods < 3:
            self.lbl_tp.config(
                text="✗ needs n ≥ 2, 0 ≤ m ≤ n and at least 3 periods.")
            return
        # a = sqrt(3) * bond; the tube's circumference is a*sqrt(n^2+nm+m^2).
        bond = float(self.var_bond.get())
        lattice = math.sqrt(3.0) * bond
        radius = lattice * math.sqrt(n * n + n * m + m * m) / (2 * math.pi)
        period = (lattice * math.sqrt(3.0 * (n * n + n * m + m * m))
                  / math.gcd(2 * n + m, 2 * m + n))
        major = period * periods / (2 * math.pi)
        strain = radius / major if major else 1.0
        if strain > POLYHEX_TEAR_STRAIN:
            self.lbl_tp.config(
                text=(f"✗ r/R = {100 * strain:.1f}%, past the "
                      f"{100 * POLYHEX_TEAR_STRAIN:.0f}% where the wall "
                      "tears rather than loads. Refused — raise the period "
                      "count or narrow the tube."))
            return
        note = ("" if strain <= POLYHEX_MAX_STRAIN else
                f" Past the {100 * POLYHEX_MAX_STRAIN:.0f}% budget: real, "
                "but strained, and no relaxation will fix it — the stretch "
                "is geometric.")
        self.lbl_tp.config(
            text=(f"({n},{m}) tube, r ≈ {radius:.2f} Å, ring R ≈ "
                  f"{major:.1f} Å, outer wall stretched {100 * strain:.1f}% "
                  "(r/R). The wall stays ALL HEXAGONS — no disclinations at "
                  "all, unlike the meshed route, which pays for the same "
                  f"curvature in 5–7 pairs.{note}"))

    def _update_tor_hint(self) -> None:
        """Say whether the hole survives, and whether the ratio is one
        anybody has relaxed.

        A torus is genus 1, so its budget is exactly zero and its
        pentagons and heptagons must come in equal numbers -- worth
        saying before the build, because it is the one census here that
        can be predicted rather than measured.
        """
        from ..builders.toroid import LITERATURE_ASPECT, MIN_ASPECT

        major = float(self.var_tor_major.get())
        minor = float(self.var_tor_minor.get())
        if minor <= 0:
            return
        aspect = major / minor
        if aspect < MIN_ASPECT:
            self.lbl_tor.config(
                text=(f"✗ R/r = {aspect:.2f}, under {MIN_ASPECT}: the hole "
                      "is smaller than the tube is thick, so this would be "
                      "a dimpled sphere rather than a torus. Refused."))
            return
        low, high = LITERATURE_ASPECT
        band = ("" if low <= aspect <= high else
                f" Outside the published {low}–{high} band — not wrong, "
                "but not the geometry those calculations relaxed to.")
        self.lbl_tor.config(
            text=(f"R/r = {aspect:.2f}, inner-equator bend "
                  f"{100 * minor / major:.1f}%. Genus 1, so sum(6−n) is "
                  "exactly 0 and the pentagons and heptagons must come out "
                  "in equal numbers — the pentagons on the outer equator, "
                  f"the heptagons on the inner.{band}"))

    def _update_ht_hint(self) -> None:
        """Say what the roll will give before it runs.

        Two things about a rolled haeckelite are not obvious from the
        boxes and are both refusals rather than warnings: a catalogue
        pattern needs its block to divide the sheet, and the wrapped edge
        fixes the radius, which has a floor. Saying so here is the
        difference between learning it now and learning it after the
        relaxation.
        """
        from ..builders.haeckelite_tube import MIN_TUBE_RADIUS, WARN_TUBE_RADIUS

        nx = int(self.var_ht_nx.get())
        ny = int(self.var_ht_ny.get())
        pattern = self.var_ht_pattern.get()
        roll = self.var_ht_roll.get()
        # Graphene's rectangular cell, which is what the sheet is counted
        # in; the wrapped edge is a whole number of them either way.
        cells = nx if roll == "a" else ny
        edge = math.sqrt(3.0) * 1.42 if roll == "a" else 3.0 * 1.42
        radius = cells * edge / (2.0 * math.pi)

        if pattern in HAECKELITE_CATALOGUE:
            block_m, block_n = HAECKELITE_CATALOGUE[pattern]["block"]
            if nx % block_m or ny % block_n:
                self.lbl_ht.config(
                    text=(f"✗ the {pattern} lattice tiles a {block_m}×"
                          f"{block_n} block, so x must be a multiple of "
                          f"{block_m} and y of {block_n}. This sheet would "
                          "be refused before it was ever rolled."))
                return
        if radius < MIN_TUBE_RADIUS:
            self.lbl_ht.config(
                text=(f"✗ wrapping {cells} cells gives a {radius:.1f} Å "
                      f"radius, below the {MIN_TUBE_RADIUS} Å of the "
                      "narrowest nanotube ever observed — and that one was "
                      "grown inside a template holding it open. Raise the "
                      f"{'x' if roll == 'a' else 'y'} count."))
            return

        note = (" Narrower than a (5,5): the geometry will pass every check "
                "here, but those checks are a valence force field, which "
                "barely sees curvature — the real cost of a narrow tube is "
                "rehybridisation and nothing here prices it."
                if radius < WARN_TUBE_RADIUS else "")
        self.lbl_ht.config(
            text=(f"{4 * nx * ny} atoms, radius about {radius:.1f} Å before "
                  "relaxation. A cylinder is developable, so the roll is an "
                  "isometry: the flat lattice's pentagons and heptagons "
                  "carry over unchanged and sum(6−n) stays 0. The radius "
                  "and the axial period are outputs — both are re-fitted "
                  f"and reported, not held.{note}"))

    def _update_sn_hint(self) -> None:
        """Say whether a tube survives between two vertices, and what the
        ring budget will be.

        The same failure the plain network has -- shrink the cell and the
        vertices grow into each other until what is left is a sponge
        rather than tubes -- plus one this builder can state that the
        others cannot: the skeleton fixes ``sum(6-n)`` at ``12*(V-E)``
        before anything is meshed, so the hint can show the answer the
        build will be checked against.
        """
        from ..builders.supernetwork import build_cost_note, named_graph

        name = self.var_sn_graph.get()
        scale = float(self.var_sn_scale.get())
        try:
            graph = named_graph(name, scale)
        except ValueError:
            return
        radius = float(self.var_sn_radius.get())
        blend = float(self.var_sn_blend.get())
        shortest = float(graph.strut_lengths(scale).min())
        free = shortest - 2.0 * (radius + blend)

        if free <= 0:
            self.lbl_sn.config(
                text=(f"✗ {shortest:.0f} Å struts, and each vertex eats "
                      f"about {radius + blend:.0f} Å of either end, so "
                      "nothing recognisable as a tube is left between them. "
                      "This would be refused: a larger cell, a narrower "
                      "tube or a smaller blend."))
            return
        # The geometry check above passes on a net that is perfectly
        # sound and simply enormous, which is how a super-fcc left at
        # super-diamond's scale ran for ninety minutes without a word of
        # warning. Say the size as well as the shape.
        area = graph.wall_area(scale, radius)
        self.lbl_sn.config(
            text=(f"{graph.coordination} tubes per vertex, "
                  f"{len(graph.nodes)} vertex/vertices and "
                  f"{len(graph.edges)} strut(s) per cell, leaving "
                  f"{free:.0f} Å of free tube between them. That is "
                  f"{area:,.0f} Å² of wall, "
                  f"{build_cost_note(area, graph.periodic)}. "
                  f"The skeleton fixes sum(6−n) at {graph.ring_budget:+d} "
                  "before anything is meshed, and the build is checked "
                  f"against it. {graph.note}"
                  + ("" if name in SUPERLATTICES else
                     " This is a finite cage, so the length above is the "
                     "strut, not a cell edge.")))

    def _graft_fields(self) -> dict:
        """The grafting half of a Job, shared by all three families.

        Written once because a group applies to every mode: leaving it
        out of one branch would make the panel visibly present and
        silently ignored for that family.
        """
        name = self.var_graft.get()
        return dict(
            graft=None if name == "none" else name,
            graft_swap=self.var_graft_swap.get().strip(),
            graft_coverage=float(self.var_graft_coverage.get()),
            graft_where=self.var_graft_where.get(),
            graft_face=self.var_graft_face.get(),
        )

    def _update_graft_hint(self) -> None:
        """Say what the chosen group is and what it can be rebuilt as.

        The swap field is free text because which elements are offered
        depends on the group, so the panel has to say which ones -- a
        user picking from a ten-item dropdown cannot be expected to know
        that an -OH reaches S/Se/Te and an -NH2 reaches P/As/B.
        """
        name = self.var_graft.get()
        if name == "none":
            self.lbl_graft.config(
                text="No surface groups. A group is added on top of the "
                     "surface rather than substituted into it, so it works "
                     "on carbon and on a dichalcogenide alike.",
                foreground=MUTED)
            return

        group = get_group(name)
        lines = [describe(group, "C")]

        swaps = self.var_graft_swap.get().strip()
        if swaps:
            try:
                group = substitute(group, parse_swaps(swaps))
            except ValueError as exc:
                self.lbl_graft.config(text=str(exc), foreground=WARN_AMBER)
                return
            lines.append(f"Rebuilt as {group.formula}; every bond length "
                         "recomputed for the new elements.")
        else:
            offers = ", ".join(
                f"{element}:{'/'.join(options)}"
                for element, options in
                ((e, viable_swaps(e)) for e in group.elements()) if options)
            if offers:
                lines.append(f"Swap accepts {offers}.")

        if self.var_graft_where.get() in ("defect", "ring:5", "ring:7"):
            lines.append("Ring selection needs a builder that records its "
                         "rings: capped tube, fullerene, nano-onion, "
                         "junction, schwarzite, network, multi-wall, bundle.")
        if self.var_graft_face.get() == "both":
            lines.append("'both' alternates by sublattice — the chair "
                         "conformation. On an MX2 each chalcogen has only "
                         "one exposed face, so it is ignored there.")

        coverage = float(self.var_graft_coverage.get())
        lines.append(f"Asking for {coverage:.0%} of the selected sites; what "
                     "fits is measured and reported after the build.")
        self.lbl_graft.config(text=" ".join(lines), foreground=MUTED)

    def _update_dopant_hint(self) -> None:
        """Say what the chosen dopant is and whether this much is real.

        Fifteen elements in a dropdown is fifteen different chemistries,
        and the difference between 10% N and 10% Fe is the difference
        between a common material and one that does not exist. The
        warning fires at build time either way; showing it here means the
        user does not have to build to find out.
        """
        spec = self.var_codope.get().strip()
        element = self.var_dopant.get()
        if spec:
            # Co-doping replaces the single-element fields rather than
            # adding to them -- both substitute carbons, and `Job` refuses
            # the pair, so saying so here beats a build-time error.
            affinity = self.var_codope_affinity.get()
            meaning = {
                "seek": "placed as bonded pairs of unlike species — the B-N "
                        "domain case",
                "avoid": "spread so no two dopants are bonded",
                "random": "placed independently of one another",
            }[affinity]
            try:
                parsed = parse_codope_spec(spec)
            except ValueError as error:
                self.lbl_dopant.config(text=str(error), foreground=WARN_AMBER)
                return
            total = sum(fraction for _, fraction in parsed)
            names = ", ".join(f"{e} {f:.1%}" for e, f in parsed)
            text = (f"Co-doping {names} — each of the original carbon count, "
                    f"{total:.1%} in all, {meaning}. The single Dopant and "
                    "Site boxes above are ignored while this is set.")
            colour = WARN_AMBER if total > 0.3 else MUTED
            self.lbl_dopant.config(text=text, foreground=colour)
            return
        if element == "none":
            self.lbl_dopant.config(
                text="Pure carbon. Pick an element to substitute into the "
                     "lattice, or type a co-doping spec such as "
                     "'N:0.05,B:0.05'.", foreground=MUTED)
            return
        chem = get_chemistry(element)
        fraction = float(self.var_dopant_conc.get())
        site = self.var_dopant_site.get()

        text = (f"{chem.site}, r = {chem.radius:.2f} Å "
                f"({chem.size_mismatch:+.0%} vs C). {chem.note}")
        colour = MUTED
        if fraction > chem.max_fraction:
            colour = WARN_AMBER
            text = (f"{fraction:.1%} is past the ~{chem.max_fraction:.0%} that "
                    f"is physically meaningful for {element}. " + text)
        if site == "pentagon":
            # The fraction means something different here, and silently
            # is exactly how it would be misread.
            text += (" Pentagon placement counts the fraction against the "
                     "pentagon sites, not the whole structure, and needs a "
                     "builder that records rings.")
        self.lbl_dopant.config(text=text, foreground=colour)

    def _update_tmd_hint(self) -> None:
        """Name the material's real geometry, and cost a tube's curvature.

        The tube hint is the one that earns its place: rolling a sandwich
        of thickness h onto radius R strains the outer plane by h/2R, so a
        tube that would be unremarkable in carbon is badly strained in
        MoS2. Saying so before the build saves a wasted one.
        """
        from .. import tmd as tmd_module

        try:
            material = tmd_module.get_material(self.var_tmd_material.get())
        except KeyError:
            return
        phase = self.var_tmd_phase.get()
        coordination = ("trigonal prismatic" if phase == "2H" else "octahedral")
        note = ""
        if phase != material.natural_phase:
            note = f" — note {material.formula} is naturally {material.natural_phase}"
        self.lbl_tmd.config(
            text=f"a = {material.a:.3f} Å, M–X = {material.bond_length:.3f} Å, "
                 f"layer thickness {material.h:.2f} Å, van der Waals gap "
                 f"{material.vdw_gap:.2f} Å. {phase} is {coordination}{note}."
        )

        if not hasattr(self, "lbl_tmd_tube"):
            return
        n, m = int(self.var_tmd_n.get()), int(self.var_tmd_m.get())
        if n < 1 or m < 0 or m > n:
            self.lbl_tmd_tube.config(
                text="Need n ≥ 1 and 0 ≤ m ≤ n.", foreground=BAD_RED)
            return
        radius = tmd_module.tube_radius(material, n, m)
        strain = material.h / (2.0 * radius)
        family = ("zigzag" if m == 0 else "armchair" if n == m else "chiral")
        colour = (OK_GREEN if strain <= 0.05
                  else WARN_AMBER if strain <= 0.10 else BAD_RED)
        self.lbl_tmd_tube.config(
            text=f"({n},{m}) {family}: R = {radius:.1f} Å, diameter "
                 f"{2 * (radius + material.h / 2):.1f} Å, outer-plane strain "
                 f"{strain:.1%}"
                 + ("" if strain <= 0.10 else
                    " — real MX2 tubes are tens of nm across; raise n"),
            foreground=colour,
        )

        if hasattr(self, "lbl_tmd_j"):
            try:
                radius = float(self.var_tmd_j_radius.get())
            except (tk.TclError, ValueError):
                radius = 0.0
            floor = 2.0 * material.h
            if radius < floor:
                self.lbl_tmd_j.config(
                    text=f"Needs at least {floor:.1f} Å: the sandwich is "
                         f"{material.h:.1f} Å thick, so a narrower tube would "
                         "collapse its inner wall through the axis.",
                    foreground=BAD_RED)
            else:
                parity = self.var_tmd_j_parity.get()
                note = ("A capped junction is sphere-like, so sum(6−n) = +12 — "
                        "positive, unlike a schwarzite. With no pentagons "
                        "allowed it is paid in squares.")
                if parity == "split":
                    note += (" Genus 0 leaves no homology, so every ring even "
                             "means exactly zero M–M and X–X bonds.")
                else:
                    note += " Leaving rings odd costs ~14% homoelemental bonds."
                self.lbl_tmd_j.config(
                    text=note,
                    foreground=OK_GREEN if parity == "split" else WARN_AMBER)

        if hasattr(self, "lbl_tmd_sw"):
            from ..tmd.curved import MIN_SCHWARZITE_CELL

            kind = self.var_tmd_sw_kind.get()
            try:
                sw_cell = float(self.var_tmd_sw_cell.get())
            except (tk.TclError, ValueError):
                sw_cell = 0.0
            floor = MIN_SCHWARZITE_CELL.get(kind, 30.0)
            if sw_cell < floor:
                self.lbl_tmd_sw.config(
                    text=f"{kind} needs at least {floor:.0f} Å: the sandwich is "
                         f"{material.h:.1f} Å thick, so below that the channels "
                         "are narrower than the layer.",
                    foreground=BAD_RED)
            else:
                # Measured on Schwarz P at 36 Å; the ordering holds
                # everywhere, the exact numbers move with the surface.
                trade = {
                    "none": ("~11% of bonds M–M or X–X", "best geometry",
                             OK_GREEN),
                    "flip": ("~8% of bonds M–M or X–X",
                             "adds no vertices", WARN_AMBER),
                    "split": ("~2% of bonds M–M or X–X",
                              ("every ring even, but ~25% worst-case bond "
                               "strain"), WARN_AMBER),
                }[self.var_tmd_sw_parity.get()]
                self.lbl_tmd_sw.config(
                    text=f"{trade[0]} — {trade[1]}. Even rings alone cannot "
                         "remove them: genus > 0 leaves 2g more parity classes, "
                         "so what is left is a real inversion-domain boundary.",
                    foreground=MUTED)

        if not hasattr(self, "lbl_tmd_coil"):
            return
        # The coil pays a second strain on top of the roll, and the two
        # pull opposite ways: widening the tube cuts h/2R and raises
        # R_outer*kappa. Showing the sum is the only way to see that.
        from ..tmd.coil import helix_curvature

        try:
            coil_radius = float(self.var_tmd_coil_radius.get())
            pitch = float(self.var_tmd_coil_pitch.get())
        except (tk.TclError, ValueError):
            return
        if coil_radius <= 0 or pitch <= 0:
            self.lbl_tmd_coil.config(text="Coil radius and pitch must be "
                                          "positive.", foreground=BAD_RED)
            return
        outer = radius + material.h / 2.0
        bend = outer * helix_curvature(coil_radius, pitch)
        total = strain + bend
        colour = (OK_GREEN if total <= 0.06
                  else WARN_AMBER if total <= 0.15 else BAD_RED)
        self.lbl_tmd_coil.config(
            text=f"bend strain {bend:.1%} (R_outer × κ) on top of the "
                 f"{strain:.1%} roll — {total:.1%} total on the outer "
                 f"{material.chalcogen} plane."
                 + ("" if total <= 0.15 else
                    " Widening the tube cuts the roll and raises the bend, so "
                    "both have to grow together."),
            foreground=colour,
        )

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

    def _place_hint(self) -> str:
        """What the placement switch does, in the window's own words.

        It is the one control here that changes WHERE the defects are
        rather than how many, and that distinction is the whole reason
        it exists -- so it is stated rather than left to the name.
        """
        if not self.var_place.get():
            return ("Off: the remesher's own scatter of 5s and 7s is kept, "
                    "wherever it happened to put them.")
        return (
            "On: pentagons are moved toward the positively curved parts "
            "(caps, outer bends) and heptagons toward the negatively "
            "curved ones (necks, inner bends), by Stone-Wales flips. It "
            "cannot change HOW MANY there are — a flip that would add a "
            "5-7 pair is refused — only where they sit. Measured on the "
            "junctions: 94→98% of them on the correct side of the "
            "curvature on a Y, 90→94% on an L, 96→98% on an X. Costs a "
            "few seconds and is safe to leave on."
        )

    def _anchor_hint(self) -> str:
        """What holding the wall buys, and what it costs.

        The two switches pull in different directions and the window
        should say so: annealing places the rings better and leaves the
        wall rougher, and this is what pays that back.
        """
        if not self.var_anchor.get():
            return ("Wall free: relaxing only equalises bond lengths, and a "
                    "corrugated or collapsed wall does that just as well as "
                    "a round one.")
        return (
            "Wall held: each atom is restrained along its own surface "
            "normal, so it still slides within the wall but cannot leave "
            "it. Measured off-surface deviation on the coils: 0.80→0.39, "
            "1.01→0.33, 2.38→0.36 Å. It clears a collapsed wall — the "
            "super-cubic and the superfullerene both need it — and costs "
            "some bond spread, so it is worth leaving off where the "
            "Structure panel already reads sound."
        )

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
        if self.var_mode_kind.get() in CURVED_SURFACE_MODES and anneal > 0:
            topo = (f"annealing hurts here — {anneal} sweeps buckles the wall; "
                    "the 5-7 pairs are how the net covers the curvature")
            colour = BAD_RED
        # Annealing flips mesh edges, and a tube or ribbon has no mesh to
        # flip: its atoms go straight onto the graphene lattice. Saying so
        # is better than leaving a live-looking slider that does nothing.
        elif self.var_mode_kind.get() in LATTICE_MODES:
            topo = ("no annealing here — the atoms are placed on the lattice, "
                    "not meshed, so there are no stray rings to flip away")
            if anneal > 0:
                colour = WARN_AMBER
        self.lbl_surface.config(
            text=(f"{topo}; {geom}.\n\n{self._place_hint()}"
                  f"\n\n{self._anchor_hint()}"),
            foreground=colour)

    def _update_coil_hint(self) -> None:
        # The wound (n, m) tube shares the radius/pitch/turns controls but
        # not the reasoning behind them: its limit is its own lattice, not
        # a meshed surface, so it writes its own hint.
        if self.var_mode_kind.get() == "nanocoil":
            self._update_nanocoil_hint()
            return
        from ..builders import centerline as cl

        radius = float(self.var_coil_radius.get())
        pitch = float(self.var_coil_pitch.get())
        turns = float(self.var_coil_turns.get())
        taper = float(self.var_coil_taper.get())
        arc = cl.helix_arc_length(radius, pitch, turns, taper=taper)

        if self.var_mode_kind.get() == "coil (periodic, DFT)":
            tube_radius = float(self.var_coil_tube_radius.get())
            one_turn = cl.helix_arc_length(radius, pitch, 1.0)
            atoms = int(2.4 * tube_radius * one_turn)
            # Two different things, and the builder no longer conflates
            # them: turns that INTERSECT are impossible, turns merely
            # closer than a graphitic gap are tight -- and the relaxed
            # single-wall coils in the literature are tight. Liu et al.'s
            # (7,7) has a 12.11 Å pitch around a 9.52 Å tube.
            if pitch <= 2.0 * tube_radius:
                self.lbl_coil.config(
                    text=f"pitch must exceed the {2.0 * tube_radius:.1f} Å the "
                         "tube itself occupies, or one turn passes through "
                         "the next.", foreground=BAD_RED)
                return
            gap = pitch - 2.0 * tube_radius
            tight = ("" if gap >= 3.4 else
                     f"\nTight: {gap:.1f} Å wall to wall, inside the 3.4 Å "
                     "graphitic gap, so the walls touch. The published "
                     "single-wall coils are like this.")
            self.lbl_coil.config(
                text=f"One turn, {atoms} atoms, cell {2 * (radius + tube_radius + 10):.0f}"
                     f" × {2 * (radius + tube_radius + 10):.0f} × {pitch:.1f} Å, "
                     "periodic along z only. Turns and taper do not apply: the "
                     "cell is one period, and you extend it with the ×z box "
                     "under the viewer or with nz in your DFT input." + tight
                     + "\n"
                     "The wall carries pentagons and heptagons on purpose — "
                     "that is how a real coil relieves curvature, and a "
                     "pure-hexagon coil is stretched instead.",
                foreground=OK_GREEN)
            return

        if self.var_mode_kind.get() == "coil (relaxed)":
            # No strain budget applies: curvature is paid for in ring
            # topology, not bond stretch. What fails instead is the coil
            # closing on itself, so that is what the hint reports.
            tube_radius = float(self.var_coil_tube_radius.get())
            clearance = 2.0 * tube_radius + 3.4
            gap = pitch - 2.0 * tube_radius
            # Two ways to get a coil that is not worth building, and only
            # one of them used to be reported. Too tight and the walls
            # merge, which is a hard failure. Too loose and it builds
            # perfectly well and does not read as a coil at all -- a
            # gently curving tube -- which is the one people actually hit,
            # because the failure is visual rather than geometric.
            atoms = int(arc * 1.57 * tube_radius)
            if pitch < clearance:
                verdict, colour = (
                    f"walls merge — pitch needs ≥{clearance:.0f} Å", BAD_RED)
            elif turns < 2.0:
                verdict, colour = (
                    "under two turns: this reads as a bent tube, not a coil",
                    WARN_AMBER)
            elif gap > 2.0 * tube_radius:
                verdict, colour = (
                    f"turns {gap:.0f} Å apart, wider than the {2 * tube_radius:.0f} Å "
                    "tube: open, so the helix is hard to see", WARN_AMBER)
            else:
                verdict, colour = (f"turns {gap:.0f} Å apart — reads as a coil",
                                   OK_GREEN)
            self.lbl_coil.config(
                text=f"coil ⌀{2 * radius:.0f} Å, tube ⌀{2 * tube_radius:.0f} Å, "
                     f"{arc:.0f} Å of tube ≈ {atoms} atoms. {verdict}\n"
                     "This mode meshes a surface, so the wall is an "
                     "amorphous CVD-like network, not a rolled lattice — "
                     "that is what it is for. For a clean graphitic coil "
                     "use «capped tube» with shape «helix», which is also "
                     "about ten times faster.",
                foreground=colour,
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
        # A measurement beats a guess. The estimator's wording is a
        # hand-written bucket -- "a minute or two" -- so once this mode
        # has actually been built in this session, say what it took and
        # for how many atoms, and let the user scale it themselves. The
        # estimate stays alongside rather than being replaced: the last
        # build may have been a very different size.
        measured = self._measured.get(self.var_mode_kind.get())
        if measured is not None:
            text += " — " + self._scaled_prediction(measured)
        self.lbl_estimate.config(text=f"Estimate: {text}", foreground=colour)

    def _scaled_prediction(self, measured: tuple[int, float]) -> str:
        """Turn one timing on this machine into a prediction for this job.

        Profiling a coil puts 95% of the time in the shell relaxation and
        essentially all of that in 40 000 energy-and-gradient evaluations,
        whose cost is dominated by the non-bonded pair term -- so the work
        goes as roughly the square of the atom count. Two builds three
        times apart in size differed by a factor of eleven, which is that
        square and not a linear cost.

        Scaling the user's own measurement beats any constant baked in
        here, because it is their processor that the number has to be
        true of. Below a factor of two in size the scaling is not worth
        claiming, so the measurement is simply reported.
        """
        was, seconds = measured
        try:
            now = estimate_atoms(self.current_job())
        except (tk.TclError, ValueError, KeyError):
            now = was
        base = f"last here: {was} atoms in {_clock(seconds)}"
        if was <= 0 or now <= 0:
            return base
        ratio = now / was
        if 0.5 < ratio < 2.0:
            return base
        return f"{base}, so ≈{_clock(seconds * ratio ** 2)} for {now}"

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
        spec = self.var_codope.get().strip()
        common = dict(
            # `Job` refuses a dopant and a co-doping spec together, since
            # both substitute carbons. The spec wins here, which matches
            # what the hint tells the user while they type it.
            dopant=None if (spec or dopant == "none") else dopant,
            dopant_conc=float(self.var_dopant_conc.get()),
            dopant_site=self.var_dopant_site.get(),
            codope=spec,
            codope_affinity=self.var_codope_affinity.get(),
            seed=int(self.var_seed.get()),
            **self._graft_fields(),
        )
        edit = self.var_tmd_edit.get()
        tmd_chemistry = dict(
            tmd_edit=None if edit == "none" else edit,
            tmd_edit_element=self.var_tmd_edit_element.get(),
            tmd_edit_amount=float(self.var_tmd_edit_amount.get()),
        )

        if mode.startswith("TMD"):
            # The dichalcogenide builders take no seed and no dopant: the
            # placement is exact crystallography with nothing random in
            # it, so `common` would only carry arguments they reject.
            # ...but the MX2 chemistry that follows the build *is*
            # random, so it brings the seed with it.
            return Job(mode=mode, params=self._tmd_params(mode),
                       seed=int(self.var_seed.get()), **tmd_chemistry,
                       **self._graft_fields())

        if mode in ("twisted bilayer", "vdW stack"):
            # Commensurate stacking is exact too, for the same reason.
            # Commensurate stacking is exact, so the seed is 0 -- but a
            # graft is random placement and needs the real one.
            return Job(mode=mode, params=self._hetero_params(mode),
                       seed=int(self.var_seed.get()) if self.var_graft.get()
                       != "none" else 0,
                       **self._graft_fields())

        if mode == "junction":
            params = dict(
                kind=self.var_j_kind.get(),
                tube_radius=float(self.var_j_radius.get()),
                arm_length=float(self.var_j_arm.get()),
                blend=float(self.var_j_blend.get()),
                anneal_sweeps=int(self.var_anneal.get()),
                place_curvature=bool(self.var_place.get()),
                wall_anchor=1.0 if self.var_anchor.get() else 0.0,
                roughness=float(self.var_roughness.get()),
            )
        elif mode == "network":
            params = dict(
                kind=self.var_net_kind.get(),
                cell=float(self.var_net_cell.get()),
                tube_radius=float(self.var_net_radius.get()),
                blend=float(self.var_net_blend.get()),
                anneal_sweeps=int(self.var_anneal.get()),
                place_curvature=bool(self.var_place.get()),
                wall_anchor=1.0 if self.var_anchor.get() else 0.0,
                roughness=float(self.var_roughness.get()),
            )
        elif mode == "haeckelite":
            params = dict(
                nx=int(self.var_hk_nx.get()),
                ny=int(self.var_hk_ny.get()),
                pattern=self.var_hk_pattern.get(),
                period=int(self.var_hk_period.get()),
                density=float(self.var_hk_density.get()),
            )
        elif mode == "haeckelite tube":
            params = dict(
                nx=int(self.var_ht_nx.get()),
                ny=int(self.var_ht_ny.get()),
                pattern=self.var_ht_pattern.get(),
                roll=self.var_ht_roll.get(),
            )
        elif mode == "toroid":
            params = dict(
                major_radius=float(self.var_tor_major.get()),
                minor_radius=float(self.var_tor_minor.get()),
                anneal_sweeps=int(self.var_anneal.get()),
                place_curvature=bool(self.var_place.get()),
                wall_anchor=1.0 if self.var_anchor.get() else 0.0,
                roughness=float(self.var_roughness.get()),
            )
        elif mode == "nanocone":
            params = dict(
                n_pentagons=int(self.var_nc_pent.get()),
                radius=float(self.var_nc_radius.get()),
                strict=bool(self.var_nc_strict.get()),
            )
        elif mode == "toroid (polyhex)":
            params = dict(
                n=int(self.var_tp_n.get()),
                m=int(self.var_tp_m.get()),
                periods=int(self.var_tp_periods.get()),
            )
        elif mode == "heptanene":
            params = dict(strict=bool(self.var_hp_strict.get()))
        elif mode == "supernetwork":
            params = dict(
                graph=self.var_sn_graph.get(),
                scale=float(self.var_sn_scale.get()),
                tube_radius=float(self.var_sn_radius.get()),
                blend=float(self.var_sn_blend.get()),
                anneal_sweeps=int(self.var_anneal.get()),
                place_curvature=bool(self.var_place.get()),
                wall_anchor=1.0 if self.var_anchor.get() else 0.0,
                roughness=float(self.var_roughness.get()),
            )
        elif mode == "schwarzite":
            params = dict(
                kind=self.var_s_kind.get(),
                cell=float(self.var_s_cell.get()),
                thickness=float(self.var_s_thickness.get()),
                anneal_sweeps=int(self.var_anneal.get()),
                place_curvature=bool(self.var_place.get()),
                wall_anchor=1.0 if self.var_anchor.get() else 0.0,
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
                place_curvature=bool(self.var_place.get()),
                wall_anchor=1.0 if self.var_anchor.get() else 0.0,
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
        elif mode == "coil (periodic, DFT)":
            sides = int(self.var_coil_sides.get())
            params = dict(
                coil_radius=float(self.var_coil_radius.get()),
                pitch=float(self.var_coil_pitch.get()),
                tube_radius=float(self.var_coil_tube_radius.get()),
                sides=sides if sides >= 3 else None,
                bond=float(self.var_bond.get()),
                handedness=1 if self.var_coil_hand.get() == "right" else -1,
                place_curvature=bool(self.var_place.get()),
                wall_anchor=1.0 if self.var_anchor.get() else 0.0,
            )
        elif mode == "nanotube (open)":
            params = dict(
                n=int(self.var_cnt_n.get()),
                m=int(self.var_cnt_m.get()),
                length=float(self.var_cnt_length.get()),
                bond=float(self.var_bond.get()),
                vacuum=float(self.var_cnt_vacuum.get()),
                roughness=float(self.var_roughness.get()),
                defects=self._current_defects(),
            )
        elif mode == "nanocoil":
            # No length: the helix's arc length sets it, so asking for one
            # as well would be two answers to the same question.
            params = dict(
                n=int(self.var_cnt_n.get()),
                m=int(self.var_cnt_m.get()),
                coil_radius=float(self.var_coil_radius.get()),
                pitch=float(self.var_coil_pitch.get()),
                n_turns=float(self.var_coil_turns.get()),
                bond=float(self.var_bond.get()),
                vacuum=float(self.var_cnt_vacuum.get()),
            )
        elif mode == "nanoribbon":
            params = dict(
                width=int(self.var_rib_width.get()),
                length=int(self.var_rib_length.get()),
                edge=self.var_rib_edge.get(),
                bond=float(self.var_bond.get()),
                vacuum=float(self.var_rib_vacuum.get()),
                passivate=bool(self.var_rib_passivate.get()),
                roughness=float(self.var_roughness.get()),
                defects=self._current_defects(),
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
                bend_angle=math.radians(float(self.var_bend.get())),
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

    def _hetero_params(self, mode: str) -> dict:
        """Builder arguments for one stacking mode."""
        bottom = self.var_het_bottom.get()
        top = self.var_het_top.get()
        third = self.var_het_third.get()
        if mode == "twisted bilayer":
            return dict(
                layer=bottom,
                top_layer=None if top == "same" else top,
                target_angle=float(self.var_het_angle.get()),
                max_index=int(self.var_het_max_index.get()),
                gap=float(self.var_het_gap.get()),
            )
        layers = [bottom, bottom if top == "same" else top]
        if third != "none":
            layers.append(third)
        return dict(
            layers=layers,
            gap=float(self.var_het_gap.get()),
            nx=int(self.var_het_nx.get()),
            ny=int(self.var_het_ny.get()),
        )

    def _update_het_hint(self) -> None:
        """Show the achieved twist and the mismatch before building.

        Both are things the user cannot choose freely and would otherwise
        discover only from the finished structure: the angle is snapped
        to the commensurate series, and stacking unlike layers strains
        one of them.
        """
        from ..hetero.moire import MAX_MISMATCH, get_layer, nearest_commensurate

        try:
            bottom = get_layer(self.var_het_bottom.get())
            chosen = self.var_het_top.get()
            top = bottom if chosen == "same" else get_layer(chosen)
        except KeyError:
            return
        mismatch = (top.a - bottom.a) / bottom.a
        colour = BAD_RED if abs(mismatch) > MAX_MISMATCH else (
            OK_GREEN if abs(mismatch) < 0.005 else WARN_AMBER)
        note = (f"{bottom.name} a = {bottom.a:.3f} Å, {top.name} "
                f"a = {top.a:.3f} Å → {mismatch:+.2%} mismatch")
        if abs(mismatch) > MAX_MISMATCH:
            note += f". Over the {MAX_MISMATCH:.0%} limit; this will be refused."
        elif mismatch:
            note += ", imposed on the top layer as strain."
        self.lbl_het.config(text=note, foreground=colour)

        if not hasattr(self, "lbl_twist"):
            return
        try:
            wanted = float(self.var_het_angle.get())
            max_index = int(self.var_het_max_index.get())
        except (tk.TclError, ValueError):
            return
        m, n, angle, cells = nearest_commensurate(wanted, max_index)
        atoms = cells * (bottom.n_sites + top.n_sites)
        self.lbl_twist.config(
            text=f"nearest commensurate: {angle:.4f}° at (m,n) = ({m},{n}), "
                 f"{cells} cells per layer → {atoms} atoms. No periodic cell "
                 "exists between these, so the angle is snapped.",
            foreground=WARN_AMBER if atoms > 8000 else MUTED,
        )

    def _tmd_params(self, mode: str) -> dict:
        """Builder arguments for one dichalcogenide mode."""
        material = self.var_tmd_material.get()
        phase = self.var_tmd_phase.get()
        if mode == "TMD layers":
            return dict(
                material=material, phase=phase,
                n_layers=int(self.var_tmd_layers.get()),
                stacking=self.var_tmd_stacking.get(),
                nx=int(self.var_tmd_nx.get()), ny=int(self.var_tmd_ny.get()),
            )
        if mode == "TMD bulk":
            return dict(
                material=material, phase=phase,
                stacking=self.var_tmd_stacking.get(),
                nx=int(self.var_tmd_nx.get()), ny=int(self.var_tmd_ny.get()),
            )
        if mode == "TMD ribbon":
            return dict(
                material=material, phase=phase,
                width=int(self.var_tmd_width.get()),
                length=int(self.var_tmd_length.get()),
                edge=self.var_tmd_edge.get(),
                termination=self.var_tmd_termination.get(),
            )
        if mode == "TMD junction":
            return dict(
                material=material, phase=phase,
                kind=self.var_tmd_j_kind.get(),
                tube_radius=float(self.var_tmd_j_radius.get()),
                arm_length=float(self.var_tmd_j_arm.get()),
                blend=float(self.var_tmd_j_blend.get()),
                parity=self.var_tmd_j_parity.get(),
            )
        if mode == "TMD schwarzite":
            return dict(
                material=material, phase=phase,
                kind=self.var_tmd_sw_kind.get(),
                cell=float(self.var_tmd_sw_cell.get()),
                parity=self.var_tmd_sw_parity.get(),
            )
        if mode == "TMD coil":
            return dict(
                material=material, phase=phase,
                n=int(self.var_tmd_n.get()), m=int(self.var_tmd_m.get()),
                coil_radius=float(self.var_tmd_coil_radius.get()),
                pitch=float(self.var_tmd_coil_pitch.get()),
                turns=float(self.var_tmd_coil_turns.get()),
                handedness=1 if self.var_tmd_coil_hand.get() == "right" else -1,
            )
        return dict(
            material=material, phase=phase,
            n=int(self.var_tmd_n.get()), m=int(self.var_tmd_m.get()),
            length=int(self.var_tmd_length.get()),
        )

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
        self._build_started = time.monotonic()
        self._tick_clock()
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

    def _finish_build(self, atoms: int | None = None, mode: str = "",
                      failed: bool = False) -> None:
        self._busy = False
        self.progress.stop()
        self.btn_build.config(state="normal")
        self.btn_cancel.config(state="disabled")
        if self._clock_job is not None:
            self.root.after_cancel(self._clock_job)
            self._clock_job = None
        elapsed = (time.monotonic() - self._build_started
                   if self._build_started is not None else None)
        self._build_started = None
        if elapsed is None:
            self.lbl_elapsed.config(text="")
            return
        if atoms is not None and mode:
            self._measured[mode] = (atoms, elapsed)
        # "took 1:31" says the same thing whether the structure arrived or
        # the builder refused it, and the elapsed line is what people
        # actually look at when the progress bar stops. A build that
        # failed says so, in red, or the previous structure sitting in the
        # preview reads as the new one.
        self.lbl_elapsed.config(
            text=(f"FAILED after {_clock(elapsed)}" if failed else
                  f"took {_clock(elapsed)}"
                  + (f" for {atoms} atoms" if atoms is not None else "")),
            foreground=BAD_RED if failed else MUTED)

    def _tick_clock(self) -> None:
        """Count up while a build runs, and check it is still running.

        Re-arms itself rather than riding the poll loop, so the reading
        stays smooth even when a poll is busy drawing a finished
        structure.
        """
        if self._build_started is None:
            return
        elapsed = time.monotonic() - self._build_started
        alive = self.worker.is_alive()
        self.lbl_elapsed.config(
            text=f"building… {_clock(elapsed)}"
                 + ("" if alive else "  — the worker process is gone"),
            foreground=MUTED if alive else BAD_RED)
        self._clock_job = self.root.after(250, self._tick_clock)

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
                self._finish_build(
                    atoms=len(payload) if kind == "done" else None,
                    mode=self.var_mode_kind.get() if kind == "done" else "",
                    failed=kind != "done")
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
                    self._mark_preview_stale()
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
        self._applying_values = True
        for key, value in values.items():
            var = self._params.get(key)
            if var is None:
                continue  # a setting from a newer/older version: ignore it
            try:
                var.set(value)
            except tk.TclError:
                pass
        self._applying_values = False
        # The values just applied *are* the chemistry for this mode, so
        # record it as such rather than letting the next repaint treat
        # them as leftovers from the mode before.
        self._chemistry_mode = self.var_mode_kind.get()
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
    #: Named viewpoints as (elevation, azimuth). Down an axis is how a
    #: tube's cross-section, a sheet's lattice and a coil's pitch are
    #: each read; the isometric one is the default matplotlib opens on.
    VIEWPOINTS: tuple[tuple[str, float, float], ...] = (
        ("+x", 0.0, 0.0), ("+y", 0.0, 90.0), ("+z", 90.0, -90.0),
        ("iso", 22.0, -60.0),
    )

    def _build_view_bar(self, parent: ttk.Frame) -> None:
        """Viewpoints, zoom and the cell, in one row under the canvas."""
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(4, 0))

        ttk.Label(bar, text="view:").pack(side="left")
        for label, elev, azim in self.VIEWPOINTS:
            ttk.Button(bar, text=label, width=4,
                       command=lambda e=elev, a=azim: self._set_view(e, a)
                       ).pack(side="left", padx=1)

        ttk.Label(bar, text="  zoom:").pack(side="left")
        ttk.Button(bar, text="+", width=3,
                   command=lambda: self._zoom(1 / 1.25)).pack(side="left", padx=1)
        ttk.Button(bar, text="−", width=3,
                   command=lambda: self._zoom(1.25)).pack(side="left", padx=1)
        ttk.Button(bar, text="fit", width=4,
                   command=self._zoom_fit).pack(side="left", padx=1)

        # Repetition, GDIS-style: see the structure as the periodic thing
        # it is rather than as one cell floating in space. Only the
        # directions that are actually periodic are offered -- repeating
        # a vacuum direction would stack copies through each other.
        ttk.Label(bar, text="  ×").pack(side="left")
        self.var_repeat = {}
        for axis in "xyz":
            var = tk.IntVar(value=1)
            self.var_repeat[axis] = var
            ttk.Spinbox(bar, from_=1, to=6, width=2, textvariable=var,
                        command=self._redraw).pack(side="left")
        ttk.Button(bar, text="apply ×", width=8,
                   command=self.on_apply_repeat).pack(side="left", padx=(2, 0))

        self.var_show_cell = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="cell", variable=self.var_show_cell,
                        command=self._redraw).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="to unit cell", width=11,
                   command=self.on_view_unit_cell).pack(side="left", padx=(2, 0))
        # Vacuum per non-periodic side. It has a default and people still
        # need to change it: 10 Å is fine for a tube and thin for a
        # charged or strongly polar slab, and the number is what decides
        # whether the cell converges.
        self.var_cell_vacuum = tk.DoubleVar(value=10.0)
        ttk.Spinbox(bar, from_=4.0, to=30.0, increment=1.0, width=4,
                    textvariable=self.var_cell_vacuum).pack(side="left")
        ttk.Label(bar, text="Å vac").pack(side="left")
        # What comes back flagged periodic. "true" keeps a tube periodic
        # along its axis and a sheet in its plane, which is what they are
        # and what the QE writer reads to pick the k-mesh. "3D" flags all
        # three, for formats with no way to say anything else.
        self.var_cell_mark = tk.StringVar(value="true")
        ttk.Combobox(bar, textvariable=self.var_cell_mark, width=5,
                     state="readonly", values=["true", "3D"]
                     ).pack(side="left", padx=(2, 0))

        ttk.Button(bar, text="save image…", width=11,
                   command=self.on_save_image).pack(side="right")
        ttk.Label(bar, text="drag to rotate", foreground=MUTED,
                  font=("TkDefaultFont", 8)).pack(side="right", padx=(0, 8))

    def _set_view(self, elevation: float, azimuth: float) -> None:
        self.ax.view_init(elev=elevation, azim=azimuth)
        self.canvas.draw_idle()

    def _zoom(self, factor: float) -> None:
        """Scale the three axis ranges about their common centre.

        All three together, and by the same factor: scaling them apart
        would shear the structure, and a 3D axes has no rubber-band box
        to drag anyway.
        """
        self._zoom_scale = max(0.05, min(20.0,
                                         getattr(self, "_zoom_scale", 1.0) * factor))
        self._apply_limits()

    def _zoom_fit(self) -> None:
        self._zoom_scale = 1.0
        self._apply_limits()

    def _apply_limits(self) -> None:
        """Equal-aspect limits about the structure, at the current zoom.

        An unscaled 3D axes stretches a long tube into a blob and makes a
        coil look elliptical, so the span is one number for all three.
        """
        if self.atoms is None:
            return
        pos = self.atoms.get_positions()
        # Frame every drawn copy, not just the first: repeating along z
        # and then seeing only the original cell is the same bug as
        # drawing them and calling it a picture of the structure.
        corners = np.vstack([pos + shift for shift in self._offsets()])
        # Include the drawn edge-bond stubs. They reach up to a cell
        # beyond the atoms, and framing only the atoms cropped them off
        # the view on any structure where they are a large share.
        extra = getattr(self, "_drawn_extent", None)
        if extra is not None and len(extra):
            corners = np.vstack([corners, extra])
        scale = getattr(self, "_zoom_scale", 1.0)
        span = (float((corners.max(axis=0) - corners.min(axis=0)).max()) / 2.0
                or 1.0) * scale
        mid = (corners.max(axis=0) + corners.min(axis=0)) / 2.0
        self.ax.set_xlim(mid[0] - span, mid[0] + span)
        self.ax.set_ylim(mid[1] - span, mid[1] + span)
        self.ax.set_zlim(mid[2] - span, mid[2] + span)
        # Equal LIMITS are not equal AXES. Matplotlib's 3D box defaults
        # to (4, 4, 3), so a sphere with identical x, y and z ranges
        # still renders flattened along z -- a C60 came out visibly
        # squashed while its bonds measured 1.420-1.420 Å and its angles
        # 108.0-120.0, which is a perfect truncated icosahedron. The
        # structure was right and the picture was lying.
        self.ax.set_box_aspect((1.0, 1.0, 1.0))
        self.canvas.draw_idle()

    def _draw_cell(self) -> None:
        """The periodic box as twelve edges.

        Only the periodic directions are drawn. A 2D sheet's vacuum
        direction has a cell vector because a plane-wave code needs one,
        but drawing it would put a lid on a structure that has none.
        """
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        cell = np.asarray(self.atoms.cell)
        if not np.any(cell):
            return
        corners = np.array([i * cell[0] + j * cell[1] + k * cell[2]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                 if bin(a ^ b).count("1") == 1]
        self.ax.add_collection3d(Line3DCollection(
            [(corners[a] + shift, corners[b] + shift)
             for shift in self._offsets() for a, b in edges],
            colors="#b0b6bd", linewidths=0.8, linestyles=":"))

    def on_view_unit_cell(self) -> None:
        """Make the structure a periodic cell a DFT code will accept.

        This works on whatever is on screen, not on one favoured mode. A
        cage becomes a molecule in a box, a tube keeps its own period
        along the axis and gets vacuum across it, a sheet gets vacuum
        along z, and something already periodic in three directions is
        left alone rather than padded into nonsense.

        What the button adds over the conversion it calls is the verdict:
        a cell whose images are too close is the commonest way a
        first-principles run is quietly wrong, and it is one number.
        """
        if self.atoms is None:
            return
        before = describe_periodicity(self.atoms)
        if before == "3D":
            report = cell_report(self.atoms)
            self.var_show_cell.set(True)
            self._redraw()
            self._set_status(
                f"Already a 3D periodic cell ({report['lengths'][0]:.1f} × "
                f"{report['lengths'][1]:.1f} × {report['lengths'][2]:.1f} Å); "
                "nothing to convert."
            )
            return
        try:
            converted = to_unit_cell(self.atoms,
                                     vacuum=float(self.var_cell_vacuum.get()),
                                     mark=self.var_cell_mark.get())
        except (ValueError, RuntimeError, tk.TclError) as exc:
            self._show_error("Could not convert to a unit cell", str(exc))
            return

        self.atoms = converted
        self.last_saved_stem = None
        report = cell_report(converted)
        self.var_show_cell.set(True)
        self._zoom_fit()
        self._redraw()
        self._update_info()
        lengths = report["lengths"]
        separation = report["image_separation"]
        verdict = (
            "" if separation is None else
            f" Nearest image {separation:.1f} Å — "
            + ("converged." if report["converged"] else
               f"under {MIN_IMAGE_SEPARATION:.0f} Å, so add vacuum before "
               "trusting an energy.")
        )
        # Flagging vacuum as periodic is not a labelling choice: the QE
        # writer picks the k-mesh from pbc, so a cage marked 3D is sampled
        # 2x2x2 across its own vacuum instead of 1x1x1, and a sheet 5x5x2
        # instead of 5x5x1 -- work spent on nothing, and assume_isolated
        # dropped along with it.
        after = describe_periodicity(converted)
        axes = "".join(name for name, on
                       in zip("xyz", converted.get_pbc(), strict=True) if on)
        along = f" along {axes}" if axes and after != "3D" else ""
        warning = ""
        if self.var_cell_mark.get() == "3D" and before != "3D":
            warning = (" Marked 3D although it is not: a k-mesh will be spent "
                       "sampling the vacuum, and assume_isolated is dropped. "
                       "Use «true» unless your format cannot express it.")
        self._set_status(
            f"{before} → {after}{along}: {lengths[0]:.1f} × {lengths[1]:.1f} × "
            f"{lengths[2]:.1f} Å, {len(converted)} atoms.{verdict}{warning}"
        )

    def on_save_image(self) -> None:
        """Save the preview as it stands, viewpoint and all.

        Replacing matplotlib's toolbar took its save button with it, and
        that button was the one part of it that did work on a 3D axes.
        At 200 dpi rather than the screen's 100: this is a working
        snapshot for a notebook or a message, and a figure for a paper
        comes from the export bundle instead.
        """
        if self.atoms is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save the preview", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile=f"{self.last_saved_stem or 'structure'}.png",
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=200, bbox_inches="tight")
        except (OSError, ValueError) as exc:
            self._show_error("Could not save the image", str(exc))
            return
        self._set_status(f"Saved {Path(path).name}")

    def _build_preview(self, parent: ttk.Frame) -> None:
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # Matplotlib's own toolbar is built for 2D. On a 3D axes its
        # magnifier and its pan cross do not mean what their icons say,
        # and it has no rotate button at all -- because rotation is the
        # drag gesture, which is exactly what makes the toolbar look
        # like it is missing one. So: the canonical viewpoints, a zoom
        # that changes the limits rather than rubber-banding a box, and
        # the drag gesture written down where it can be read.
        self._build_view_bar(parent)

        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(6, 0))
        self.var_show_bonds = tk.BooleanVar(value=True)
        # Element, not ring, by default. Ring colouring is the more
        # informative view once there is curvature to look at, but a
        # pristine tube is all hexagons, so it painted every atom the one
        # hexagon colour and the viewer looked broken -- and a doped tube
        # gave no hint that the dopant was there at all.
        self.var_colour_by = tk.StringVar(value="element")
        ttk.Checkbutton(bar, text="Bonds", variable=self.var_show_bonds,
                        command=self._redraw).pack(side="left")
        # A bond that leaves through one face and returns through the
        # opposite one is 1.42 Å long and is DRAWN, unless this is off, as
        # the short stub it is -- see _redraw. The switch is here because
        # even a stub is clutter on a structure whose surface is mostly
        # boundary: a gyroid cell has 131 of them.
        self.var_show_wrapped = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="edge bonds", variable=self.var_show_wrapped,
                        command=self._redraw).pack(side="left")
        # Colouring ATOMS by ring cannot answer "where are the pentagons"
        # on a closed cage, and C60 is the proof: every one of its 60
        # atoms belongs to exactly one pentagon, so every atom takes the
        # pentagon colour and the picture says "all pentagons" about a
        # structure that is 12 pentagons and 20 hexagons. Filling the
        # RINGS is the only view that separates them.
        self.var_ring_faces = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="ring faces", variable=self.var_ring_faces,
                        command=self._redraw).pack(side="left")
        ttk.Label(bar, text="  colour:").pack(side="left")
        ttk.Combobox(bar, textvariable=self.var_colour_by, width=8,
                     state="readonly", values=["element", "ring", "plain"]
                     ).pack(side="left", padx=(2, 0))
        self.var_colour_by.trace_add("write", lambda *_: self._redraw())

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
        """One colour per atom, by whichever scheme is selected.

        ``element`` answers "what is this made of", which is the question
        a doped or grafted structure raises; ``ring`` answers "where did
        the curvature go", which is the question a coil or a junction
        raises. Neither subsumes the other, so both are offered rather
        than one being picked for the user.
        """
        mode = self.var_colour_by.get()
        if mode == "ring":
            return [RING_COLOURS.get(s, RING_COLOURS[6]) for s in ring_of]
        if mode == "element" and self.atoms is not None:
            return [ELEMENT_COLOURS.get(symbol, UNKNOWN_ELEMENT_COLOUR)
                    for symbol in self.atoms.get_chemical_symbols()]
        return [PLAIN_ATOM_COLOUR] * len(ring_of)

    def _bond_shifts(self) -> np.ndarray:
        """Which cell each bond's far atom is in, cached per structure.

        Three milliseconds for the 2253 bonds of a gyroid cell, which is
        cheap once and wasteful on every drag of the view, so it is kept
        until the structure itself is replaced.
        """
        from ..utils.geometry import bond_shifts

        bonds = self.atoms.info.get("bonds", [])
        if self._shifts_for is not self.atoms or len(self._shifts) != len(bonds):
            self._shifts = bond_shifts(self.atoms, bonds)
            self._shifts_for = self.atoms
        return self._shifts

    def _repeat_counts(self) -> tuple[int, int, int]:
        """How many copies along each axis, ignoring aperiodic ones.

        Repeating a direction that carries vacuum rather than a lattice
        vector stacks copies straight through the structure, so the
        request is silently clamped to 1 there rather than obeyed.
        """
        pbc = self.atoms.get_pbc() if self.atoms is not None else (False,) * 3
        cell = np.asarray(self.atoms.cell) if self.atoms is not None else np.zeros((3, 3))
        counts = []
        for index, axis in enumerate("xyz"):
            wanted = max(1, int(self.var_repeat[axis].get()))
            usable = bool(pbc[index]) and float(np.linalg.norm(cell[index])) > 1e-6
            counts.append(wanted if usable else 1)
        return tuple(counts)

    def _offsets(self) -> list[np.ndarray]:
        """Translation of each drawn copy, the identity one first."""
        cell = np.asarray(self.atoms.cell)
        nx, ny, nz = self._repeat_counts()
        return [i * cell[0] + j * cell[1] + k * cell[2]
                for i in range(nx) for j in range(ny) for k in range(nz)]

    def on_apply_repeat(self) -> None:
        """Make the repetition real, so it can be exported and analysed.

        Drawing copies is a picture; this replaces the structure with the
        supercell, which is what a calculation would need.

        Through :func:`~nanocarbon_lab.cell.supercell` rather than ASE's
        ``repeat``, because ``repeat`` copies ``info`` verbatim: the
        supercell kept the bond list, ring list and ring census of one
        cell. The preview then drew bonds on the first copy only, the
        JSON bundle exported a connectivity covering an eighth of the
        atoms, and the Euler check called a sound 2×2×2 gyroid broken.
        The atom count looked right throughout, which is what made it
        worth doing properly.
        """
        if self.atoms is None:
            return
        counts = self._repeat_counts()
        if counts == (1, 1, 1):
            self._set_status("Nothing to repeat: no periodic direction is set "
                             "above one.")
            return
        try:
            self.atoms = supercell(self.atoms, counts)
        except ValueError as exc:
            self._show_error("Could not build the supercell", str(exc))
            return
        self.last_saved_stem = None
        for axis in "xyz":
            self.var_repeat[axis].set(1)
        self._history.append(
            (f"{len(self._history) + 1}. {counts[0]}×{counts[1]}×{counts[2]} "
             f"supercell ({len(self.atoms)} atoms)", self.atoms))
        del self._history[:-12]
        self.cmb_history["values"] = [name for name, _ in self._history]
        self.var_history.set(self.cmb_history["values"][-1])
        self._zoom_fit()
        self._redraw()
        self._update_info()
        self._set_status(
            f"Supercell {counts[0]}×{counts[1]}×{counts[2]}: "
            f"{len(self.atoms)} atoms. Every export button now writes this."
        )

    #: Rings wider than this are not drawn as faces. A ring whose atoms
    #: straddle the periodic boundary comes back from the ring finder as
    #: a correct list of atoms with coordinates on opposite sides of the
    #: cell, and joining those in order draws a polygon across the whole
    #: structure. A real ring of at most eight sp2 carbons never spans
    #: more than about 5 Å.
    MAX_RING_SPAN = 6.0

    def _draw_ring_faces(self, pos, offsets, keep) -> str:
        """Fill each ring, coloured by its size.

        This is the view that answers the question the ring colours were
        meant to answer and cannot: which rings are the pentagons. The
        atom colouring assigns each ATOM one ring size, and on a closed
        cage every atom is in a pentagon, so it says nothing.

        Hexagons are drawn faintly and everything else strongly, because
        the hexagons are the background and the 5s, 7s and 8s are the
        subject -- that is also why the ring filters exist.
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        rings = self.atoms.info.get("rings", [])
        if not rings:
            return "no ring list — nothing to fill"

        polys, colours, dropped = [], [], 0
        for ring in rings:
            size = len(ring)
            if size in self.var_ring_filter and not self.var_ring_filter[size].get():
                continue
            if not all(keep[a] for a in ring):
                continue
            corners = pos[list(ring)]
            if float(np.ptp(corners, axis=0).max()) > self.MAX_RING_SPAN:
                dropped += 1
                continue
            for shift in offsets:
                polys.append(corners + shift)
                colours.append(RING_COLOURS.get(size, RING_COLOURS[6]))
        if not polys:
            return "no rings to fill at this filter"
        if len(polys) > PREVIEW_RING_FACE_LIMIT:
            stride = len(polys) // PREVIEW_RING_FACE_LIMIT + 1
            polys, colours = polys[::stride], colours[::stride]
        alphas = [0.18 if c == RING_COLOURS[6] else 0.55 for c in colours]
        collection = Poly3DCollection(
            polys, facecolors=colours, edgecolors="none")
        collection.set_alpha(None)
        collection.set_facecolor([
            (*_to_rgb(c), a) for c, a in zip(colours, alphas, strict=True)])
        self.ax.add_collection3d(collection)
        return (f"{dropped} rings cross the cell and are not filled"
                if dropped else "")

    def _redraw(self) -> None:
        if self.atoms is None:
            return
        pos = self.atoms.get_positions()
        offsets = self._offsets()
        ring_of = self._ring_of_atom()
        keep = np.array([bool(self.var_ring_filter[s].get())
                         if s in self.var_ring_filter else True
                         for s in ring_of])
        self.ax.clear()

        note = ""
        if self.var_ring_faces.get():
            note = self._draw_ring_faces(pos, offsets, keep) or note

        if self.var_show_bonds.get():
            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            # Every bond as the SHORT vector it is. A bond list built
            # under the minimum image convention names the pair and not
            # the cell the far atom is in, so drawing it from raw
            # coordinates gives a line straight across the structure --
            # 46 Å on the 1502-atom gyroid, 131 times over. Those lines
            # were the noise; they were never the bonds.
            cell = np.asarray(self.atoms.cell)
            shifts = self._bond_shifts()
            show_wrapped = self.var_show_wrapped.get()
            bonds, hidden = [], 0
            for index, (a, b) in enumerate(self.atoms.info.get("bonds", [])):
                if not (keep[a] and keep[b]):
                    continue
                crosses = bool(shifts[index].any())
                if crosses and not show_wrapped:
                    hidden += 1
                    continue
                bonds.append((pos[a], pos[b] - shifts[index] @ cell
                              if crosses else pos[b]))
            if len(bonds) > PREVIEW_BOND_LIMIT:
                stride = len(bonds) // PREVIEW_BOND_LIMIT + 1
                bonds = bonds[::stride]
                note = f"showing 1 bond in {stride}"
            if hidden:
                note = ", ".join(filter(None, [note, f"{hidden} edge bonds hidden"]))
            # A structure whose cell is barely bigger than its bonds has
            # a large share of them crossing, and each is drawn as a stub
            # reaching a full cell OUTSIDE the cluster. Heptanene is the
            # extreme: 29 of its 84 bonds cross a 6.2 Å cell, so the
            # stubs reach twice as far as the structure and the picture
            # is mostly stubs. Saying the share is the difference
            # between a confusing image and an explained one.
            elif show_wrapped:
                crossing = int(sum(1 for row in shifts if row.any()))
                total = max(1, len(shifts))
                if crossing / total > 0.2:
                    note = ", ".join(filter(None, [
                        note,
                        f"{100 * crossing / total:.0f}% of bonds cross the "
                        "cell — untick 'edge bonds' for a clearer view"]))
            segs = [(start + shift, end + shift)
                    for shift in offsets for start, end in bonds]
            self._drawn_extent = (
                np.array([p for seg in segs for p in seg])
                if segs else None)
            if segs:
                self.ax.add_collection3d(
                    Line3DCollection(segs, colors="#9aa3ad", linewidths=0.7)
                )

        shown = np.flatnonzero(keep)
        if shown.size:
            colours = self._atom_colours(ring_of)
            drawn = [colours[i] for i in shown]
            for shift in offsets:
                self.ax.scatter(
                    pos[shown, 0] + shift[0], pos[shown, 1] + shift[1],
                    pos[shown, 2] + shift[2],
                    c=drawn, s=16, depthshade=True,
                )
        hidden = len(pos) - shown.size
        self.lbl_preview.config(
            text=", ".join(filter(None, [
                f"{shown.size} of {len(pos)} atoms" if hidden else f"{len(pos)} atoms",
                note,
            ]))
        )

        if self.var_show_cell.get():
            self._draw_cell()

        self._apply_limits()
        self.ax.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------- readouts
    def on_analyse_file(self):
        """Read a structure someone else made, and say what it is.

        The report goes where a build's readout goes, because it answers
        the same question -- and the structure is loaded as the current
        one, so every other button (export, unit cell, render, grafting)
        applies to it as it would to something built here. A file from
        another code is a first-class structure, not a read-only guest.
        """
        from ..analyse import analyse, format_report, read_structure

        path = filedialog.askopenfilename(
            title="Open a structure file",
            filetypes=[("Structures", "*.xyz *.extxyz *.cif *.pdb *.vasp "
                                      "*.traj *.cube *.gen *.json"),
                       ("All files", "*.*")])
        if not path:
            return None

        try:
            atoms = read_structure(path)
            result = analyse(atoms)
        except ValueError as exc:
            self._show_error("Could not read that file", str(exc))
            self.lbl_analyse.config(text=str(exc), foreground=BAD_RED)
            return None

        self.atoms = atoms
        self.last_saved_stem = None
        self._history.append((f"{len(self._history) + 1}. {Path(path).name} "
                              f"({len(atoms)} atoms)", atoms))
        del self._history[:-12]
        self.cmb_history["values"] = [name for name, _ in self._history]
        self.var_history.set(self.cmb_history["values"][-1])
        self._redraw()

        self.txt_info.configure(state="normal")
        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("1.0", format_report(result, Path(path).name))
        self.txt_info.configure(state="disabled")

        shape = result["inferred"]["shape"]
        verdict = result["verdict"]
        colour = OK_GREEN if result["validation"]["ok"] else BAD_RED
        self.lbl_analyse.config(
            text=f"{shape['dimensionality']}D {shape['shape']}, "
                 f"{len(atoms)} atoms — {str(verdict['verdict']).upper()}. "
                 + ("Rings read from the file."
                    if result["inferred"]["rings_are_recorded"]
                    else f"Rings inferred by "
                         f"{result['inferred']['rings']['method']}."),
            foreground=colour)
        self._set_status(f"Analysed {Path(path).name}.")
        return result

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

        # The supercell control, duplicated from the bar under the canvas
        # and sharing its variables -- two views of one setting, not two
        # settings. It is here because this is the panel people are
        # looking at when they want a supercell to export, and an eight
        # character button in the view bar is not where they look.
        sup = ttk.LabelFrame(parent, text="Supercell", padding=8)
        sup.pack(fill="x", pady=(8, 0))
        row = ttk.Frame(sup)
        row.pack(fill="x")
        ttk.Label(row, text="copies:").pack(side="left")
        for axis in "xyz":
            ttk.Label(row, text=f"  {axis}").pack(side="left")
            ttk.Spinbox(row, from_=1, to=6, width=2,
                        textvariable=self.var_repeat[axis],
                        command=self._redraw).pack(side="left")
        ttk.Button(sup, text="Make supercell",
                   command=self.on_apply_repeat).pack(fill="x", pady=(6, 0),
                                                      ipady=3)
        ttk.Label(sup, text="Only directions that actually repeat are used; "
                            "a count on a vacuum direction is ignored. The "
                            "preview shows the copies as you change these; "
                            "the button makes them real, with the bonds and "
                            "the ring census carried over, so every export "
                            "below then writes the supercell.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=260,
                  justify="left").pack(anchor="w", pady=(4, 0))

        exp = ttk.LabelFrame(parent, text="Export", padding=8)
        exp.pack(fill="x", pady=(8, 0))
        ttk.Button(exp, text="Save .xyz + .json bundle…",
                   command=self.on_export).pack(fill="x", ipady=3)
        ttk.Button(exp, text="Save unit cell (.cif) for DFT…",
                   command=self.on_export_cell).pack(fill="x", pady=(4, 0),
                                                     ipady=3)
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
        self.lbl_cell = ttk.Label(exp, text="", foreground=MUTED,
                                  font=("TkDefaultFont", 8), wraplength=260,
                                  justify="left")
        self.lbl_cell.pack(anchor="w", pady=(4, 0))
        ttk.Label(exp, text="XYZ for any viewer; JSON carries bonds and ring "
                            "types for the Blender pipeline, plus a CIF for "
                            "periodic cells.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=260,
                  justify="left").pack(anchor="w", pady=(4, 0))

        anl = ttk.LabelFrame(parent, text="Analyse a file", padding=8)
        anl.pack(fill="x", pady=(8, 0))
        ttk.Button(anl, text="Open structure file…",
                   command=self.on_analyse_file).pack(fill="x", ipady=3)
        ttk.Label(anl, text="Any file ASE reads — xyz, cif, POSCAR, pdb, a "
                            "relaxation output. Describes what it is and "
                            "loads it into the preview, so it can then be "
                            "exported, converted to a unit cell or "
                            "functionalised like anything else.",
                  foreground=MUTED, font=("TkDefaultFont", 8), wraplength=260,
                  justify="left").pack(anchor="w", pady=(4, 0))
        self.lbl_analyse = ttk.Label(anl, text="", foreground=MUTED,
                                     font=("TkDefaultFont", 8), wraplength=260,
                                     justify="left")
        self.lbl_analyse.pack(anchor="w", pady=(4, 0))

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

    def _mark_preview_stale(self) -> None:
        """Say that what is on screen is NOT what was just asked for.

        When a build fails, the preview, the readout and the structure
        panel all keep showing the previous structure, unchanged and
        unmarked -- and the user reads that as "it built and nothing
        updated". It is the opposite: nothing was built, so nothing
        updated. The one thing that must not happen is the old numbers
        passing for the new ones, so they are labelled where they are
        read.
        """
        if self.atoms is None:
            return
        self.lbl_preview.config(
            text=self.lbl_preview.cget("text").split("  —  ")[0]
            + "  —  previous structure; the last build failed")
        self.txt_info.configure(state="normal")
        self.txt_info.insert(
            "1.0", "THE LAST BUILD FAILED — everything below describes the\n"
                   "previous structure, not the one you just asked for.\n\n")
        self.txt_info.configure(state="disabled")

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
        if str(a.info.get("structure_type", "")).startswith("tmd"):
            self._update_tmd_info(a)
            return
        if a.info.get("structure_type") in ("twisted_bilayer", "vdw_stack"):
            self._update_stack_info(a)
            return
        # Not every structure carries these: a file read from another
        # code has no ring census, and a builder that places atoms on an
        # exact lattice has no measured geometry to report. Missing
        # metadata must read as "not available" rather than as a
        # traceback, because the window offers to open foreign files.
        g = a.info.get("geometry")
        counts = a.info.get("ring_counts", {})
        rings_txt = "\n".join(
            f"  {RING_LABELS[s].split(' (')[0]:<10s} {counts.get(s, 0):>5d}"
            for s in (5, 6, 7, 8) if counts.get(s)
        )
        deficit = sum((6 - s) * c for s, c in counts.items())
        # Euler's budget is 12 per closed sphere-like shell. A schwarzite
        # with handles is legitimately negative (genus g gives 12(1-g)),
        # and an assembly of n disjoint shells owes 12 per shell.
        components = int(a.info.get("n_shells", a.info.get("n_tubes", 1)))
        # A builder may state its own budget. The genus formula below
        # assumes a closed shell, which a tube periodic along its axis and
        # a ribbon with two free edges are not: both owe nothing, and
        # judging them by the closed-shell rule reported every pristine
        # one as BROKEN.
        expected = int(a.info.get(
            "euler_expected",
            components * (12 - 12 * int(a.info.get("genus", 0))),
        ))
        # `.get`, not `[...]`: this method runs after the redraw, so a
        # missing key here leaves the new structure on screen beside the
        # PREVIOUS structure's numbers. Heptanene shipped without
        # `n_close_contacts` and did exactly that.
        clash = g.get("n_close_contacts", 0) if g else 0

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
        # Which axes repeat, not just whether all three do: a tube is
        # periodic along one, a sheet along two, and saying "periodic" of
        # neither -- which is what the all() test did -- hid exactly the
        # fact a plane-wave calculation turns on.
        pbc = list(a.get_pbc())
        if any(pbc):
            lengths = a.cell.lengths()
            spans = ", ".join(f"{name} {lengths[index]:.1f} Å"
                              for index, name in enumerate("xyz") if pbc[index])
            lines.append(f"periodic     {sum(pbc)}D — {spans}")
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

        # sp2/sp3 is the question asked of every real sample in this
        # field, and here it can be measured rather than inferred from a
        # D/G ratio. It is NOT a restatement of the ring census: a flat
        # haeckelite is full of pentagons and heptagons and reads 360.0
        # deg, exactly sp2.
        hyb = ""
        if a.info.get("bonds") is not None and len(a) >= 4:
            try:
                from ..analyse.hybridisation import hybridisation_report

                report = hybridisation_report(
                    a.positions, a.info.get("bonds", []), np.asarray(a.cell))
                if report["n_measured"]:
                    hyb = (f"  sp3        {100 * report['sp3_fraction']:>5.1f}% "
                           f"of {report['n_measured']} C\n"
                           f"  angle sum  {report['angle_sum_min']:.1f}"
                           f"-{report['angle_sum_max']:.1f}° "
                           f"(360 flat, 328 sp3)")
                    # A collapse is invisible to the bond and contact checks,
                    # which are local. This is the only line that would have
                    # caught a coil whose tube pinched shut while every bond
                    # stayed 1.28-1.54 A and no contact fired.
                    from ..analyse.hybridisation import collapsed_wall

                    if collapsed_wall(report):
                        hyb += ("\n  WALL COLLAPSED - past tetrahedral, "
                                "which no carbon reaches")
            except Exception:  # noqa: BLE001 - a reading, never fatal
                hyb = ""

        lines += [
            "",
            "rings",
            rings_txt,
            (f"  Euler sum  {deficit:>5d}  "
             f"{'OK' if deficit == expected else 'BROKEN'}"),
            *(["", "hybridisation", hyb] if hyb else []),
            "",
            "geometry",
        ]
        if g:
            lines += [
                f"  bond   {g['bond_min']:.3f}–{g['bond_max']:.3f} Å",
                f"  angle  {g['angle_min']:.1f}–{g['angle_max']:.1f}°",
                f"  contacts <2Å  {clash}  {'OK' if clash == 0 else 'CHECK'}",
            ]
            # Zero clashes is not the same as a physical structure: an
            # over-tight coil keeps its atoms apart while stretching its
            # bonds past any real C-C. Say so in words, next to the
            # numbers.
            # Judged against the window the builder nominates: a lattice
            # whose non-hexagons are deliberate (a haeckelite, or anything
            # carrying a requested defect) has angles the pristine sp2
            # window calls broken and a specialist would call correct.
            verdict, why = sp2_quality(g, a.info.get("quality_family", "sp2"))
            lines += ["", f"sp2 verdict  {verdict.upper()}", f"  {why}"]
        # Whether the disclinations sit where the curvature puts them: a
        # pentagon is a +60 deg disclination and a heptagon a -60 deg one,
        # so pentagons belong in positive Gaussian curvature and
        # heptagons in negative. This was computed on every meshed build
        # and shown nowhere, which left the census on screen with no way
        # to tell an ordered wall from a disordered one.
        check = a.info.get("disclination_check")
        if check and check.get("agreement") is not None:
            lines += ["", "disclinations",
                      f"  {100 * check['agreement']:.0f}% on the curvature "
                      f"side they belong on",
                      f"  ({check['n_correct']}/{check['n_scored']} "
                      "non-hexagonal rings)"]
            for size, row in sorted(check["sizes"].items()):
                want = ("wants +" if size < 6
                        else "wants −" if size > 6 else "flat    ")
                lines.append(f"  {size}-ring  n={row['count']:<4d} "
                             f"mean sign {row['mean_sign']:+.2f}  {want}")
        else:
            # An exact-lattice builder has nothing to measure against: the
            # geometry is ideal by construction, which is a different
            # statement from "measured and found to be ideal" and is said
            # differently here rather than being faked into the same shape.
            lines.append("  ideal by construction — placed on the lattice, "
                         "not relaxed, so there is nothing measured to report")

        self.txt_info.configure(state="normal")
        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("1.0", "\n".join(lines))
        self.txt_info.configure(state="disabled")

    def _update_stack_info(self, a) -> None:
        """Readout for a stack. No rings and no Euler budget -- what
        matters is which layers, how far apart, at what twist, and how
        much strain the common cell cost."""
        from ..validation.checks import run_basic_checks

        info = a.info
        report = run_basic_checks(a)
        lines = [
            f"atoms        {len(a):>6d}",
            f"formula      {a.get_chemical_formula():>9s}",
        ]
        if info["structure_type"] == "twisted_bilayer":
            m, n = info["commensurate_index"]
            lines += [
                f"stack        {info['bottom_layer']} / {info['top_layer']}",
                f"twist        {info['twist_angle']:>7.4f}°  (asked "
                f"{info['requested_angle']:.2f}°)",
                f"index        (m,n) = ({m},{n})",
                f"moire        {info['moire_period']:>6.1f} Å period",
                f"cells/layer  {info['cells_per_layer']:>6d}",
            ]
        else:
            lines += [
                "stack        " + " / ".join(info["layers"]),
                f"supercell    {info['supercell'][0]} x {info['supercell'][1]}",
            ]
        lines.append(f"gap          {info['interlayer_gap']:>6.2f} Å")
        strain = info.get("imposed_strain", 0.0)
        worst = (max(abs(v) for v in strain) if isinstance(strain, list)
                 else abs(strain))
        lines.append(f"strain       {worst:>6.2%} imposed to share one cell")
        lengths = a.cell.lengths()
        lines.append(f"cell         {lengths[0]:.2f} x {lengths[1]:.2f} Å")
        pbc = list(a.get_pbc())
        spans = ", ".join(f"{name} {lengths[index]:.1f} Å"
                          for index, name in enumerate("xyz") if pbc[index])
        lines.append(f"periodic     {sum(pbc)}D — {spans}")
        if info["structure_type"] == "twisted_bilayer":
            # The commensurate cell is the answer to "can I get a unit
            # cell out of this": tiling it reproduces the moire exactly,
            # with no strain imposed on either layer, so it goes straight
            # into a plane-wave code at this size.
            lines.append("             tiles exactly — the moiré is the cell, "
                         "not an approximant")
        for message in report.errors[:3]:
            lines.append(f"  {message}")

        self.txt_info.configure(state="normal")
        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("1.0", "\n".join(lines))
        self.txt_info.configure(state="disabled")

    def _update_tmd_info(self, a) -> None:
        """Readout for a dichalcogenide.

        Shares no rows with the carbon one beyond the atom count: there
        are no rings to count and no Euler budget to check, and what
        matters instead is coordination, stoichiometry and -- for a tube
        -- how hard the roll had to strain the sandwich.
        """
        report = tmd_geometry_report(a)
        info = a.info
        lines = [
            f"atoms        {len(a):>6d}",
            f"formula      {a.get_chemical_formula():>9s}",
            f"material     {info['material']:>9s}",
            f"phase        {info['phase']:>9s}",
            f"coordination {info['coordination']:>18s}",
        ]
        if info.get("stacking", "n/a") != "n/a":
            lines.append(f"stacking     {info['stacking']:>9s} "
                         f"({info['n_layers']} layers)")
        if all(a.get_pbc()):
            lengths = a.cell.lengths()
            lines.append(f"periodic     3D, a={lengths[0]:.3f} c={lengths[2]:.3f} Å")
        if "edge" in info:
            lines.append(f"edge         {info['edge']:>9s} / {info['termination']}")
            lines.append(f"width        {info['width_angstrom']:>6.1f} Å")
        # Per field, not per block: a coil has chiral indices and a roll
        # strain but no single radius or diameter, and keying the whole
        # group off one field is what produced the KeyError last time.
        if "chiral_indices" in info:
            n, m = info["chiral_indices"]
            lines.append(f"tube         ({n},{m}) {info['chirality']}")
        if "radius" in info:
            lines.append(f"radius       {info['radius']:>6.2f} Å")
        if "diameter" in info:
            lines.append(f"diameter     {info['diameter']:>6.2f} Å")
        if "tube_radius" in info:
            lines.append(f"tube radius  {info['tube_radius']:>6.2f} Å")
        if "coil_radius" in info:
            lines.append(f"coil radius  {info['coil_radius']:>6.1f} Å")
            lines.append(f"pitch        {info['pitch']:>6.1f} Å")
            lines.append(f"turns        {info['turns']:>6.2f} "
                         f"({info['periods']} periods)")
        if "junction_kind" in info:
            lines.append(f"junction     {info['junction_kind']:>9s}")
            lines.append(f"tube radius  {info['tube_radius']:>6.1f} Å")
            lines.append(f"arm length   {info['arm_length']:>6.1f} Å")
        if "junction_kind" in info or "schwarzite_kind" in info:
            if "schwarzite_kind" in info:
                lines.append(f"surface      {info['schwarzite_kind']:>9s}")
            lines.append(f"genus        {info['genus']:>6d}")
            counts = info["ring_counts"]
            lines.append("rings        " + ", ".join(
                f"{k}:{v}" for k, v in sorted(counts.items())))
            expected = 6 * info["euler"]
            flag = "ok" if info["ring_deficit"] == expected else "MISMATCH"
            lines.append(f"sum(6-n)     {info['ring_deficit']:>6d}  "
                         f"(6·χ = {expected}) {flag}")
            lines.append(f"parity       {info['parity']:>9s} "
                         f"({info['odd_rings']} odd rings)")
            lines.append(f"antiphase    {info['antiphase_fraction']:>5.1%} "
                         f"({info['antiphase_bonds']} bonds M–M or X–X)")
            lines.append(f"M–X spread   {info['bond_deviation_p95']:>5.1%} p95, "
                         f"{info['bond_deviation_max']:.1%} worst")
        if "roll_strain" in info:
            lines.append(f"roll strain  {info['roll_strain']:>5.1%}")
        if "bend_strain" in info:
            lines.append(f"bend strain  {info['bend_strain']:>5.1%}")
            lines.append(f"total strain {info['total_strain']:>5.1%}")
        lines += [
            "",
            "geometry",
            (f"  M–X    {report['bond_min']:.3f}–{report['bond_max']:.3f} Å"
             f"  (ideal {report['bond_ideal']:.3f})"),
            (f"  metal coord   {report['metal_coordination_min']}–"
             f"{report['metal_coordination_max']}"),
            (f"  chalcogen     {report['chalcogen_coordination_min']}–"
             f"{report['chalcogen_coordination_max']}"),
            f"  X/M ratio     {report['stoichiometry']:.3f}",
        ]
        if info.get("structure_type") in ("tmd_schwarzite", "tmd_junction"):
            # Distance-based coordination is wrong on a saddle: the 2.4 Å
            # bond's cutoff reaches 3.0 Å and sweeps up non-bonded
            # neighbours. The builder kept the real bond graph, so use it.
            lines += [
                "",
                (f"  metal coord   {info['graph_metal_coordination'][0]}–"
                 f"{info['graph_metal_coordination'][1]} (bond graph)"),
                (f"  chalcogen     {info['graph_chalcogen_coordination'][0]}–"
                 f"{info['graph_chalcogen_coordination'][1]} (bond graph)"),
            ]
            from ..tmd.curved import schwarzite_quality

            verdict, why = schwarzite_quality(a)
            lines += ["", f"verdict      {verdict.upper()}", f"  {why}"]
            self.txt_info.configure(state="normal")
            self.txt_info.delete("1.0", "end")
            self.txt_info.insert("1.0", "\n".join(lines))
            self.txt_info.configure(state="disabled")
            return

        # A deliberately terminated ribbon is off-stoichiometry on purpose.
        stoichiometric = info.get("termination", "mixed") == "mixed"
        verdict, why = tmd_quality(report, expect_stoichiometric=stoichiometric,
                                   structure_type=info.get("structure_type"))
        lines += ["", f"verdict      {verdict.upper()}", f"  {why}"]
        if "phase_note" in info:
            lines += ["", f"  {info['phase_note']}"]

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

    def on_export_cell(self) -> Path | None:
        """Save the structure as a periodic unit cell for a DFT code.

        A separate button from the render bundle because it answers a
        different question. The bundle is for looking at; this is for
        computing with, and the conversion it applies -- vacuum on every
        direction the structure does not repeat in, ``pbc`` true on all
        three -- is what a plane-wave code and a periodic viewer both
        require and what neither can infer.
        """
        if self.atoms is None:
            self._show_error("Nothing to export", "Build a structure first.")
            return None
        path = filedialog.asksaveasfilename(
            title="Save unit cell",
            defaultextension=".cif",
            filetypes=[("Crystallographic Information File", "*.cif"),
                       ("All files", "*.*")],
            initialfile=f"{self.var_mode_kind.get().replace(' ', '_')}.cif",
        )
        if not path:
            return None
        # 3D for the file, whatever the viewer is showing: CIF has no way
        # to say "periodic along z only", so a reader given a 0D or 1D
        # flag either rejects it or invents its own answer. The vacuum in
        # the open directions is what makes that honest, and it is there.
        converted = to_unit_cell(self.atoms, mark="3D")
        written = write_cif(converted, Path(path))
        report = cell_report(converted)
        self._describe_cell(report)
        self._set_status(f"Saved {written.name}")
        return written

    def _describe_cell(self, report: dict) -> None:
        """Show the cell and whether its vacuum is enough to trust."""
        a, b, c = report["lengths"]
        separation = report["image_separation"]
        if separation is None:
            note = "bulk crystal — no vacuum direction to converge"
            colour = OK_GREEN
        elif report["converged"]:
            shown = ">= 20" if separation >= 20.0 else f"{separation:.1f}"
            note = f"images {shown} Å apart across vacuum"
            colour = OK_GREEN
        else:
            note = (f"images only {separation:.1f} Å apart — below "
                    f"{MIN_IMAGE_SEPARATION:.0f} Å they interact")
            colour = WARN_AMBER
        self.lbl_cell.config(
            text=(f"{report['periodicity']} cell {a:.2f} × {b:.2f} × {c:.2f} Å; "
                  f"{note}."),
            foreground=colour)

    # --------------------------------------------------------------- blender
    def _check_blender(self) -> None:
        if self.blender_exe and Path(self.blender_exe).exists():
            return  # a path the user picked by hand wins over auto-detection
        self.blender_exe = find_blender()
        if self.blender_exe:
            self.lbl_blender.config(text=f"Found: {self.blender_exe}")
        elif has_bpy():
            self.lbl_blender.config(
                text="Using the installed bpy module — no separate Blender "
                     "application needed."
            )
        else:
            self.lbl_blender.config(
                text="Blender not found automatically — use “Locate Blender…”, "
                     "or `pip install bpy` to render without installing "
                     "Blender at all."
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
        if not self.blender_exe and not has_bpy():
            self._show_error(
                "Blender not found",
                "Could not find Blender automatically.\n\n"
                "Easiest fix: `pip install bpy`, which installs Blender as "
                "a Python module and needs no separate application.\n\n"
                "Click “Locate Blender…” and point at the executable (on "
                "Windows, usually\n"
                r"C:\Program Files\Blender Foundation\Blender 4.x\blender.exe)"
                ",\n\nor export the bundle and run it yourself:\n"
                "  blender -b -P nanocarbon_lab/blender/render_cnt.py -- "
                "--xyz <file>.xyz "
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

        # Inside the package, so it is found whether the GUI is running
        # from a checkout or from a pip install. It used to be resolved
        # against the repository root, which does not exist once
        # installed -- the button was dead for every installed copy.
        script = Path(__file__).resolve().parent.parent / "blender" / "render_cnt.py"
        if not script.exists():
            self._show_error(
                "Render script missing",
                f"Could not find {script}. Run the GUI from a full checkout of "
                "the project, or invoke Blender manually.",
            )
            return

        # Two ways to reach the same script. A Blender application takes
        # it with -b -P; the `bpy` PyPI module *is* Blender inside this
        # interpreter, so the script runs directly. Preferring the
        # application when both exist keeps the GUI honest about which
        # Blender produced the image.
        launcher = ([self.blender_exe, "-b", "-P", str(script)]
                    if self.blender_exe else [sys.executable, str(script)])
        cmd = [
            *launcher, "--",
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
