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

from ..builders import build_capped_cnt
from ..builders import fullerene_mesh as fm
from ..exports.xyz import write_render_bundle

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
        box = ttk.LabelFrame(parent, text="Geometry", padding=8)
        box.pack(fill="x")

        self.var_rings = tk.IntVar(value=8)
        self.var_freq = tk.IntVar(value=3)
        self.var_bond = tk.DoubleVar(value=1.42)
        self.var_bend = tk.DoubleVar(value=0.0)
        self.var_seed = tk.IntVar(value=0)
        self.var_shape = tk.StringVar(value="straight")
        self.var_waviness = tk.DoubleVar(value=0.7)
        self.var_max_strain = tk.DoubleVar(value=0.08)
        self.var_shape_points = tk.IntVar(value=9)

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
        ttk.Label(sbox, text="Thinner + longer tubes curve more at the same "
                             "strain — lower the frequency and raise the rings.",
                  foreground="#777", font=("TkDefaultFont", 8), wraplength=230,
                  justify="left").grid(row=8, column=0, columnspan=2, sticky="w",
                                       pady=(4, 0))

        # --- defects
        dbox = ttk.LabelFrame(parent, text="Defects", padding=8)
        dbox.pack(fill="x", pady=(8, 0))

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

        self.btn_build = ttk.Button(parent, text="Build structure", command=self.on_build)
        self.btn_build.pack(fill="x", pady=(12, 0), ipady=4)
        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.pack(fill="x", pady=(6, 0))
        self._update_radius_hint()
        self._update_strain_hint()

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

        kwargs = dict(
            n_body_rings=int(self.var_rings.get()),
            freq=int(self.var_freq.get()),
            bond=float(self.var_bond.get()),
            bend_angle=float(self.var_bend.get()),
            shape=self.var_shape.get(),
            waviness=float(self.var_waviness.get()),
            max_strain=float(self.var_max_strain.get()),
            shape_points=int(self.var_shape_points.get()),
            defects=self._current_defects(),
            seed=int(self.var_seed.get()),
        )

        def worker():
            try:
                self._queue.put(("done", build_capped_cnt(**kwargs)))
            except Exception as exc:  # surfaced in the UI, not the console
                self._queue.put(("error", (exc, traceback.format_exc())))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                self.progress.stop()
                self.btn_build.config(state="normal")
                self._busy = False
                if kind == "done":
                    self.atoms = payload
                    self.last_saved_stem = None
                    self._redraw()
                    self._update_info()
                    self._set_status("Build complete.")
                else:
                    exc, tb = payload
                    self._set_status(f"Build failed: {exc}")
                    messagebox.showerror("Build failed", f"{exc}\n\n{tb}")
        except queue.Empty:
            pass
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
        clash = g["n_close_contacts"]
        lines = [
            f"atoms        {len(a):>6d}",
            f"radius       {a.info['radius']:>6.2f} Å",
            f"length       {a.info['length']:>6.1f} Å",
            f"shape        {a.info['shape']:>6s}",
            f"path strain  {a.info['path_strain']:>5.1%}",
            "",
            "rings",
            rings_txt,
            f"  Euler sum  {deficit:>5d}  {'OK' if deficit == 12 else 'BROKEN'}",
            "",
            "geometry",
            f"  bond   {g['bond_min']:.3f}–{g['bond_max']:.3f} Å",
            f"  angle  {g['angle_min']:.1f}–{g['angle_max']:.1f}°",
            f"  contacts <2Å  {clash}  {'OK' if clash == 0 else 'CHECK'}",
        ]
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
