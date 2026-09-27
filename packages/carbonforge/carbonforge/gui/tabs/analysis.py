"""The "Resultados → Bandas y espectros" page: read finished runs back.

Bands (QE, SIESTA), vibrational spectra from ``dynmat.x`` with their normal
modes (click a band to animate it, from ``dynmat.axsf``), and total or
projected densities of states.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import numpy as np


class AnalysisTab:
    """Open band structures, spectra, modes and densities of states and plot them.

    Uses from the host: ``tk``, ``ttk``, ``root``, ``_show_error``.
    """

    # ------------------------------------------------------------------
    # Analysis tab
    # ------------------------------------------------------------------
    def _build_analysis_tab(self, parent) -> None:
        """Open a finished calculation and plot it."""
        tk, ttk = self.tk, self.ttk
        from matplotlib.backends.backend_tkagg import (  # noqa: WPS433
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure  # noqa: WPS433

        outer = ttk.Frame(parent, padding=8)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer, width=340)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        intro = ttk.Label(
            left,
            text=(
                "Abre la salida de un cálculo terminado, o usa «Abrir resultados» "
                "en Calcular → Trabajos."
            ),
            wraplength=310, justify="left", foreground="#444444",
        )
        intro.pack(fill="x", pady=(0, 8))

        bands_box = ttk.LabelFrame(left, text="Estructura de bandas", padding=6)
        bands_box.pack(fill="x")
        ttk.Label(
            bands_box,
            text="bands.dat, bands.dat.gnu (QE) o SystemLabel.bands (SIESTA)",
            wraplength=300, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w")
        ttk.Button(
            bands_box, text="Abrir bandas…", command=self._on_open_bands
        ).pack(fill="x", pady=(4, 0))

        ttk.Label(bands_box, text="Nivel de Fermi (eV, opcional)").pack(
            anchor="w", pady=(6, 0)
        )
        self.fermi_var = tk.StringVar(value="")
        ttk.Entry(bands_box, textvariable=self.fermi_var).pack(fill="x")
        ttk.Label(
            bands_box,
            text="Necesario si el archivo no lo trae (QE no lo incluye).",
            wraplength=300, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w")

        ttk.Label(bands_box, text="Etiquetas del camino (p.ej. G,M,K,G)").pack(
            anchor="w", pady=(6, 0)
        )
        self.labels_var = tk.StringVar(value="")
        ttk.Entry(bands_box, textvariable=self.labels_var).pack(fill="x")

        spectrum_box = ttk.LabelFrame(left, text="Espectro vibracional", padding=6)
        spectrum_box.pack(fill="x", pady=(10, 0))
        ttk.Label(
            spectrum_box,
            text="Salida de dynmat.x (normalmente dynmat.out)",
            wraplength=300, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w")
        ttk.Button(
            spectrum_box, text="Abrir espectro…", command=self._on_open_spectrum
        ).pack(fill="x", pady=(4, 0))

        self.spectrum_kind_var = tk.StringVar(value="raman")
        ttk.Label(spectrum_box, text="Tipo").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(
            spectrum_box, textvariable=self.spectrum_kind_var,
            values=["raman", "ir"], state="readonly",
        ).pack(fill="x")

        ttk.Label(spectrum_box, text="Anchura lorentziana (cm⁻¹)").pack(
            anchor="w", pady=(6, 0)
        )
        self.width_var = tk.StringVar(value="8.0")
        ttk.Entry(spectrum_box, textvariable=self.width_var).pack(fill="x")

        self.correct_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            spectrum_box,
            text="Corregir a intensidad experimental",
            variable=self.correct_var,
        ).pack(anchor="w", pady=(6, 0))
        ttk.Label(
            spectrum_box,
            text=("Aplica el factor de Bose y (ν_láser − ν)⁴. Sin marcar se "
                  "muestran las actividades tal cual salen del cálculo."),
            wraplength=300, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w")

        ttk.Label(spectrum_box, text="Láser (nm) / Temperatura (K)").pack(
            anchor="w", pady=(6, 0)
        )
        row = ttk.Frame(spectrum_box)
        row.pack(fill="x")
        self.laser_var = tk.StringVar(value="532")
        self.temperature_var = tk.StringVar(value="300")
        ttk.Entry(row, textvariable=self.laser_var, width=8).pack(side="left")
        ttk.Entry(row, textvariable=self.temperature_var, width=8).pack(
            side="left", padx=(4, 0)
        )

        ttk.Label(
            spectrum_box,
            text=("Con dynmat.axsf junto a dynmat.out, un clic en una banda "
                  "anima su modo."),
            wraplength=300, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w", pady=(4, 0))

        dos_box = ttk.LabelFrame(left, text="Densidad de estados", padding=6)
        dos_box.pack(fill="x", pady=(10, 0))
        ttk.Button(dos_box, text="Abrir DOS (dos.dat)…",
                   command=self._on_open_dos).pack(fill="x")
        ttk.Button(dos_box, text="Abrir PDOS (carpeta de projwfc.x)…",
                   command=self._on_open_pdos).pack(fill="x", pady=(4, 0))
        ttk.Label(
            dos_box,
            text="Usa el nivel de Fermi de arriba si lo escribes; si no, el del archivo.",
            wraplength=300, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w")

        self.save_plot_button = ttk.Button(
            left, text="Guardar figura…", command=self._on_save_analysis_png,
            state="disabled",
        )
        self.save_plot_button.pack(fill="x", pady=(10, 0))

        self.analysis_status_var = tk.StringVar(value="Ningún archivo abierto.")
        ttk.Label(
            left, textvariable=self.analysis_status_var, wraplength=310,
            justify="left", foreground="#0a6",
        ).pack(fill="x", pady=(8, 0))

        # --- right: figure + report -------------------------------------
        figure_box = ttk.LabelFrame(right, text="Figura", padding=4)
        figure_box.pack(fill="both", expand=True)

        self.analysis_figure = Figure(figsize=(6, 4.2), dpi=100)
        self.analysis_axes = self.analysis_figure.add_subplot(111)
        self.analysis_axes.set_title("Abre un archivo de resultados")
        self.analysis_canvas = FigureCanvasTkAgg(
            self.analysis_figure, master=figure_box
        )
        self.analysis_canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(
            self.analysis_canvas, figure_box, pack_toolbar=False
        )
        toolbar.update()
        toolbar.pack(fill="x")
        self.analysis_canvas.draw()
        self.analysis_canvas.mpl_connect("button_press_event", self._on_analysis_click)
        self._shown_spectrum = None
        self._qe_modes = None
        self._mode_window = None

        report_box = ttk.LabelFrame(right, text="Informe", padding=4)
        report_box.pack(fill="both", expand=False, pady=(8, 0))
        self.analysis_text = self.tk.Text(report_box, height=10, wrap="word")
        report_scroll = ttk.Scrollbar(
            report_box, orient="vertical", command=self.analysis_text.yview
        )
        self.analysis_text.configure(
            yscrollcommand=report_scroll.set, state="disabled"
        )
        self.analysis_text.pack(side="left", fill="both", expand=True)
        report_scroll.pack(side="right", fill="y")

    def _set_analysis_report(self, text: str) -> None:
        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", text)
        self.analysis_text.configure(state="disabled")

    def _parse_optional_float(self, raw: str, label: str) -> Optional[float]:
        text = raw.strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"'{label}' debe ser un número (recibido: {raw!r}).")

    def _on_open_bands(self, path=None) -> None:
        """Open a band file: ``path`` when given (a finished job), else ask."""
        from tkinter import filedialog

        from ...results.bands import (
            attach_path_labels,
            read_qe_bands,
            read_qe_bands_gnu,
            read_siesta_bands,
        )

        path = path or filedialog.askopenfilename(
            title="Abrir archivo de bandas",
            filetypes=[
                ("Todos los formatos", "*.dat *.gnu *.bands"),
                ("QE bands.dat", "*.dat"),
                ("QE gnu", "*.gnu"),
                ("SIESTA", "*.bands"),
                ("Cualquiera", "*"),
            ],
        )
        if not path:
            return

        try:
            file = Path(path)
            if file.suffix == ".bands":
                bands = read_siesta_bands(file)
            elif file.name.endswith(".gnu"):
                bands = read_qe_bands_gnu(file)
            else:
                bands = read_qe_bands(file)

            labels = [c.strip() for c in self.labels_var.get().split(",") if c.strip()]
            if len(labels) >= 2:
                attach_path_labels(bands, labels)

            reference = self._parse_optional_float(
                self.fermi_var.get(), "Nivel de Fermi"
            )
            if reference is None:
                reference = bands.fermi_energy
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return

        self._render_bands(bands, reference)

    def _render_bands(self, bands, reference) -> None:
        from ...results.bands import draw_bands_on_axes

        self.analysis_figure.clear()
        self.analysis_axes = self.analysis_figure.add_subplot(111)
        # Draw straight onto our embedded axes. Going through the pyplot-based
        # plot_bands here would create a figure pyplot then owns and leaks.
        draw_bands_on_axes(bands, self.analysis_axes, reference=reference)
        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw_idle()

        lines = [f"{bands.n_kpoints} puntos k × {bands.n_bands} bandas"]
        if reference is not None:
            lines.append(f"Referencia de energía: {reference:.4f} eV")
            gap = bands.band_gap(fermi=reference)
            if gap is None:
                lines.append(
                    "Gap: ninguno — las bandas cruzan la referencia (metálico)."
                )
            else:
                lines.append(f"Gap muestreado: {gap:.4f} eV")
                lines.append(
                    "Ojo: solo ve los puntos k del camino. Un extremo de banda "
                    "fuera de él no aparece."
                )
        else:
            lines.append(
                "El archivo no trae nivel de Fermi (QE no lo incluye). "
                "Escríbelo en el campo correspondiente para obtener el gap."
            )
        self._set_analysis_report("\n".join(lines))
        self.analysis_status_var.set("Bandas cargadas.")
        self.save_plot_button.configure(state="normal")

    def _on_open_spectrum(self, path=None) -> None:
        """Open a dynmat.x output: ``path`` when given (a finished job), else ask."""
        from tkinter import filedialog

        from ...results.spectra import read_dynmat

        path = path or filedialog.askopenfilename(
            title="Abrir salida de dynmat.x",
            filetypes=[("Salida de dynmat", "*.out"), ("Cualquiera", "*")],
        )
        if not path:
            return

        try:
            spectrum = read_dynmat(path)
            kind = self.spectrum_kind_var.get()
            width = float(self.width_var.get().strip().replace(",", ".") or 8.0)
            laser = temperature = None
            if self.correct_var.get():
                laser = self._parse_optional_float(self.laser_var.get(), "Láser")
                temperature = self._parse_optional_float(
                    self.temperature_var.get(), "Temperatura"
                )
            if kind == "raman" and not spectrum.has_raman:
                raise ValueError(
                    "Este cálculo no trae actividades Raman. Hace falta "
                    "lraman=.true. en ph.x (y pseudos norm-conserving)."
                )
            if kind == "ir" and not spectrum.has_ir:
                raise ValueError(
                    "Este cálculo no trae actividades IR. Hace falta "
                    "epsil=.true. en ph.x."
                )
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return

        self._render_spectrum(spectrum, kind, width, laser, temperature)
        self._load_qe_modes(Path(path), spectrum, kind)

    def _render_spectrum(self, spectrum, kind, width, laser, temperature) -> None:
        from ...results.spectra import draw_spectrum_on_axes

        self.analysis_figure.clear()
        self.analysis_axes = self.analysis_figure.add_subplot(111)
        draw_spectrum_on_axes(
            spectrum, self.analysis_axes, kind=kind, width_cm1=width,
            laser_wavelength_nm=laser, temperature_k=temperature,
        )
        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw_idle()

        self._set_analysis_report(spectrum.summary())
        self.analysis_status_var.set(f"Espectro {kind} cargado.")
        self.save_plot_button.configure(state="normal")

    def _on_save_analysis_png(self) -> None:
        from tkinter import filedialog, messagebox

        path = filedialog.asksaveasfilename(
            title="Guardar figura", defaultextension=".png",
            filetypes=[("Imagen PNG", "*.png")],
        )
        if not path:
            return
        try:
            self.analysis_figure.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return
        messagebox.showinfo("Figura guardada", str(path))

    # -- densities of states ---------------------------------------------

    def _on_open_dos(self, path=None) -> None:
        """Total DOS from ``dos.x`` (``path`` from a finished job, else ask)."""
        from tkinter import filedialog

        from ...results.dos import read_dos

        path = path or filedialog.askopenfilename(
            title="Abrir salida de dos.x", filetypes=[("DOS", "*.dat *.dos"), ("Cualquiera", "*")])
        if not path:
            return
        try:
            dos = read_dos(path)
            reference = self._parse_optional_float(self.fermi_var.get(), "Nivel de Fermi")
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return
        self._render_dos(dos, reference)

    def _on_open_pdos(self, directory=None) -> None:
        """Projected DOS: the folder where ``projwfc.x`` wrote its files."""
        from tkinter import filedialog

        from ...results.dos import read_pdos

        directory = directory or filedialog.askdirectory(title="Carpeta de projwfc.x")
        if not directory:
            return
        try:
            dos = read_pdos(directory)
            reference = self._parse_optional_float(self.fermi_var.get(), "Nivel de Fermi")
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return
        self._render_dos(dos, reference)

    def _render_dos(self, dos, reference) -> None:
        from ...results.dos import ProjectedDOS, draw_dos_on_axes

        self._shown_spectrum = None
        self.analysis_figure.clear()
        self.analysis_axes = self.analysis_figure.add_subplot(111)
        draw_dos_on_axes(dos, self.analysis_axes, reference=reference)
        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw_idle()
        level = reference if reference is not None else dos.fermi_energy
        if isinstance(dos, ProjectedDOS):
            text = dos.summary(fermi=level)
        else:
            lines = [f"{len(dos.energies)} puntos, {dos.energies.min():.2f} a "
                     f"{dos.energies.max():.2f} eV"]
            if level is None:
                lines.append("Sin nivel de Fermi: escríbelo arriba (está en pw.scf.out).")
            else:
                lines.append(f"Nivel de Fermi: {level:.4f} eV; DOS(E_F) = "
                             f"{dos.at_fermi(level):.3f} estados/eV")
                gap = dos.gap_estimate(fermi=level)
                lines.append("Sin gap en torno a E_F (metálico o semimetal)." if gap is None
                             else f"Gap estimado: {gap:.3f} eV (limitado por el ensanchado "
                                  "y la malla del nscf).")
            text = "\n".join(lines)
        self._set_analysis_report(text)
        self.analysis_status_var.set("Densidad de estados cargada.")
        self.save_plot_button.configure(state="normal")

    # -- QE normal modes ---------------------------------------------------

    def _load_qe_modes(self, dynmat_out: Path, spectrum, kind: str) -> None:
        """Pick up ``dynmat.axsf`` beside ``dynmat.out`` so peaks can be clicked."""
        from ...results.modes import read_axsf_modes

        self._shown_spectrum = (spectrum, kind)
        self._qe_modes = None
        axsf = Path(dynmat_out).with_name("dynmat.axsf")
        if not axsf.exists():
            return
        try:
            atoms, vectors = read_axsf_modes(axsf)
        except Exception as exc:
            self.analysis_status_var.set(f"No se pudieron leer los modos: {exc}")
            return
        if len(vectors) != len(spectrum.modes):
            self.analysis_status_var.set(
                f"dynmat.axsf tiene {len(vectors)} modos y dynmat.out {len(spectrum.modes)}: "
                "no se animan (¿son del mismo cálculo?).")
            return
        self._qe_modes = (atoms, vectors)
        self.analysis_status_var.set(f"Espectro {kind} cargado; clic en una banda para ver "
                                     "su modo.")

    def _on_analysis_click(self, event) -> None:
        from ...results.modes import nearest_mode

        if self._shown_spectrum is None or self._qe_modes is None or event.xdata is None \
                or event.inaxes is not self.analysis_axes:
            return
        spectrum, kind = self._shown_spectrum
        try:
            activities = spectrum.activities(kind)
        except (ValueError, KeyError):
            activities = None
        index = nearest_mode(spectrum.frequencies, activities, float(event.xdata))
        if index is not None:
            self._show_qe_mode(index)

    def _show_qe_mode(self, index: int) -> None:
        """Animate one QE mode in its own small window."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        from ...results.modes import mode_character, mode_frames, normalised, view_angles
        from ...topology.graph import build_bond_graph
        from ...viz.plot import _ELEMENT_COLORS, _ELEMENT_SIZES

        atoms, vectors = self._qe_modes
        spectrum, _ = self._shown_spectrum
        mode = spectrum.modes[index]
        vector = normalised(vectors[index])
        frames = mode_frames(atoms, vector)
        if self._mode_window is not None:
            try:
                self.root.after_cancel(self._mode_window["after"])
                self._mode_window["top"].destroy()
            except Exception:
                pass
        top = self.tk.Toplevel(self.root)
        top.title(f"Modo {mode.index}: {mode.frequency_cm1:.1f} cm-1")
        self.ttk.Label(top, text=mode_character(atoms, vector), wraplength=520).pack(
            fill="x", padx=6, pady=4)
        figure = Figure(figsize=(5.5, 5.0), dpi=100)
        ax = figure.add_subplot(projection="3d")
        canvas = FigureCanvasTkAgg(figure, master=top)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        symbols = atoms.get_chemical_symbols()
        scatter = ax.scatter(*frames[0].T, c=[_ELEMENT_COLORS.get(x, "#888888") for x in symbols],
                             s=[_ELEMENT_SIZES.get(x, 30) for x in symbols],
                             edgecolors="black", linewidths=0.3)
        finite = atoms.copy()
        finite.pbc = False
        bonds = list(build_bond_graph(finite).edges)
        lines = Line3DCollection([(frames[0][i], frames[0][j]) for i, j in bonds],
                                 colors="#555555", linewidths=0.6)
        ax.add_collection3d(lines)
        centre = atoms.get_positions().mean(axis=0)
        half = float(np.ptp(atoms.get_positions(), axis=0).max()) / 2 + 1.0
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre, strict=True):
            setter(c - half, c + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(*view_angles(atoms))
        ax.set_axis_off()
        state = {"top": top, "k": 0, "after": None}

        def step() -> None:
            positions = frames[state["k"] % len(frames)]
            scatter._offsets3d = tuple(positions.T)
            lines.set_segments([(positions[i], positions[j]) for i, j in bonds])
            canvas.draw_idle()
            state["k"] += 1
            state["after"] = self.root.after(60, step)

        def close() -> None:
            if state["after"] is not None:
                self.root.after_cancel(state["after"])
            top.destroy()
            self._mode_window = None

        top.protocol("WM_DELETE_WINDOW", close)
        self._mode_window = state
        step()
        self.analysis_status_var.set(f"Modo {mode.index}: {mode.frequency_cm1:.1f} cm-1.")
