"""Tkinter desktop application for carbonforge.

Layout: parameters on the left, a live 3D preview plus a validation summary
on the right. Structures are built on a worker thread so the window never
freezes on a large nanocoil, and results are marshalled back to the main
thread via ``root.after`` (Tk is not thread-safe).

Run with ``carbonforge-gui`` or ``python -m carbonforge.gui``.
"""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from typing import Any, Optional

from ase import Atoms

from .params import (
    CALCULATION_PARAMS,
    FUNCTIONALIZATION_PARAMS,
    apply_functionalization,
    MODIFIER_PARAMS,
    STRUCTURES,
    ParamSpec,
    apply_modifiers,
    build_structure,
    describe_structure,
    export_structure,
    validate_calculation,
)

_TK_MISSING_MSG = """
No se encontró Tkinter, que es lo que dibuja la ventana.

  • Windows / macOS: reinstala Python desde python.org marcando la opción
    "tcl/tk and IDLE" durante la instalación.
  • Ubuntu / Debian:  sudo apt install python3-tk
  • Fedora:           sudo dnf install python3-tkinter
  • Arch:             sudo pacman -S tk

Mientras tanto puedes seguir usando la línea de comandos, que no necesita
Tkinter:

  carbonforge cnt --n 6 --m 6 --length 10 --out salida --format both
""".strip()

_EXPORT_FORMATS = (
    ("qe", "Quantum ESPRESSO (pw.x / ph.x)"),
    ("siesta", "SIESTA (.fdf)"),
    ("lammps", "LAMMPS (data + input)"),
    ("xyz", "XYZ (visores: OVITO, VMD)"),
    ("cif", "CIF (VESTA)"),
)


class CarbonForgeApp:
    """Main application window."""

    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("carbonforge — generador de nanoestructuras de carbono")
        self.root.geometry("1180x760")
        self.root.minsize(940, 620)

        self.atoms: Optional[Atoms] = None
        self._param_vars: dict[str, Any] = {}
        self._modifier_vars: dict[str, Any] = {}
        self._calculation_vars: dict[str, Any] = {}
        self._functionalization_vars: dict[str, Any] = {}
        self._format_vars: dict[str, Any] = {}
        self._queue: queue.Queue = queue.Queue()
        self._busy = False

        self._build_layout()
        self._rebuild_param_fields()
        self._poll_queue()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        """Two tabs: building structures, and analysing finished runs."""
        ttk = self.ttk

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)

        build_tab = ttk.Frame(notebook)
        analyse_tab = ttk.Frame(notebook)
        notebook.add(build_tab, text="  Construir estructura  ")
        notebook.add(analyse_tab, text="  Analizar resultados  ")

        self._build_builder_tab(build_tab)
        self._build_analysis_tab(analyse_tab)

    def _build_builder_tab(self, parent) -> None:
        tk, ttk = self.tk, self.ttk

        outer = ttk.Frame(parent, padding=8)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer, width=380)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        # --- structure selector -----------------------------------------
        sel = ttk.LabelFrame(left, text="Tipo de estructura", padding=8)
        sel.pack(fill="x")

        self._structure_labels = {v.label: k for k, v in STRUCTURES.items()}
        self.structure_var = tk.StringVar(
            value=STRUCTURES["cnt"].label
        )
        combo = ttk.Combobox(
            sel,
            textvariable=self.structure_var,
            values=list(self._structure_labels),
            state="readonly",
        )
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._rebuild_param_fields())

        self.description_label = ttk.Label(
            sel, text="", wraplength=340, justify="left", foreground="#444444"
        )
        self.description_label.pack(fill="x", pady=(6, 0))

        # --- scrollable parameter area ----------------------------------
        params_box = ttk.LabelFrame(left, text="Parámetros", padding=4)
        params_box.pack(fill="both", expand=True, pady=(8, 0))

        canvas = tk.Canvas(params_box, highlightthickness=0, width=340)
        scroll = ttk.Scrollbar(params_box, orient="vertical", command=canvas.yview)
        self.params_frame = ttk.Frame(canvas)
        self.params_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.params_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # --- action buttons ---------------------------------------------
        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=(8, 0))

        self.build_button = ttk.Button(
            actions, text="Construir y previsualizar", command=self._on_build
        )
        self.build_button.pack(fill="x")

        self.export_button = ttk.Button(
            actions, text="Exportar…", command=self._on_export, state="disabled"
        )
        self.export_button.pack(fill="x", pady=(4, 0))

        self.png_button = ttk.Button(
            actions, text="Guardar imagen PNG…", command=self._on_save_png,
            state="disabled",
        )
        self.png_button.pack(fill="x", pady=(4, 0))

        self.status_var = tk.StringVar(value="Listo.")
        ttk.Label(
            left, textvariable=self.status_var, wraplength=340,
            justify="left", foreground="#0a6",
        ).pack(fill="x", pady=(8, 0))

        # --- right: preview + info --------------------------------------
        self._build_preview(right)

    def _build_preview(self, parent) -> None:
        ttk = self.ttk
        from matplotlib.backends.backend_tkagg import (  # noqa: WPS433
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure  # noqa: WPS433

        preview = ttk.LabelFrame(parent, text="Vista previa 3D", padding=4)
        preview.pack(fill="both", expand=True)

        self.figure = Figure(figsize=(6, 4.6), dpi=100)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.axes.set_title("Pulsa «Construir y previsualizar»")

        self.canvas = FigureCanvasTkAgg(self.figure, master=preview)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # Toolbar gives pan/zoom/rotate and its own save button for free.
        toolbar = NavigationToolbar2Tk(self.canvas, preview, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        self.canvas.draw()

        info = ttk.LabelFrame(parent, text="Resumen y validación", padding=4)
        info.pack(fill="both", expand=False, pady=(8, 0))

        self.info_text = self.tk.Text(info, height=11, wrap="word")
        info_scroll = ttk.Scrollbar(
            info, orient="vertical", command=self.info_text.yview
        )
        self.info_text.configure(yscrollcommand=info_scroll.set, state="disabled")
        self.info_text.pack(side="left", fill="both", expand=True)
        info_scroll.pack(side="right", fill="y")

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
                "carbonforge no ejecuta los cálculos. Cuando el tuyo termine, "
                "abre aquí el archivo de salida."
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

    def _on_open_bands(self) -> None:
        from tkinter import filedialog

        from ..results.bands import (
            attach_path_labels,
            read_qe_bands,
            read_qe_bands_gnu,
            read_siesta_bands,
        )

        path = filedialog.askopenfilename(
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
        from ..results.bands import draw_bands_on_axes

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

    def _on_open_spectrum(self) -> None:
        from tkinter import filedialog

        from ..results.spectra import read_dynmat

        path = filedialog.askopenfilename(
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

    def _render_spectrum(self, spectrum, kind, width, laser, temperature) -> None:
        from ..results.spectra import draw_spectrum_on_axes

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

    # ------------------------------------------------------------------
    # Dynamic parameter fields
    # ------------------------------------------------------------------
    def _current_structure_key(self) -> str:
        return self._structure_labels[self.structure_var.get()]

    def _add_field(self, parent, spec: ParamSpec, store: dict[str, Any]) -> None:
        tk, ttk = self.tk, self.ttk
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)

        if spec.kind == "bool":
            var = tk.BooleanVar(value=bool(spec.default))
            ttk.Checkbutton(row, text=spec.label, variable=var).pack(anchor="w")
        else:
            ttk.Label(row, text=spec.label).pack(anchor="w")
            var = tk.StringVar(value=str(spec.default))
            if spec.kind == "choice":
                ttk.Combobox(
                    row, textvariable=var, values=list(spec.choices or ()),
                    state="readonly",
                ).pack(fill="x")
            else:
                ttk.Entry(row, textvariable=var).pack(fill="x")

        if spec.help:
            ttk.Label(
                row, text=spec.help, wraplength=320, justify="left",
                foreground="#777777", font=("TkDefaultFont", 8),
            ).pack(anchor="w")
        store[spec.key] = var

    def _rebuild_param_fields(self) -> None:
        ttk = self.ttk
        for child in self.params_frame.winfo_children():
            child.destroy()
        self._param_vars.clear()
        self._modifier_vars.clear()
        self._calculation_vars.clear()
        self._functionalization_vars.clear()
        self._format_vars.clear()

        # Switching structure type invalidates whatever was built before;
        # otherwise "Exportar" would silently write the previous structure.
        self._discard_current_structure()

        spec = STRUCTURES[self._current_structure_key()]
        self.description_label.configure(text=spec.description)

        for param in spec.params:
            self._add_field(self.params_frame, param, self._param_vars)

        if spec.supports_modifiers:
            mods = ttk.LabelFrame(
                self.params_frame, text="Dopaje y defectos", padding=4
            )
            mods.pack(fill="x", pady=(10, 0))
            for param in MODIFIER_PARAMS:
                self._add_field(mods, param, self._modifier_vars)

            groups = ttk.LabelFrame(
                self.params_frame,
                text="Grupos funcionales y nitrógeno",
                padding=4,
            )
            groups.pack(fill="x", pady=(10, 0))
            for param in FUNCTIONALIZATION_PARAMS:
                self._add_field(groups, param, self._functionalization_vars)

        calc = ttk.LabelFrame(self.params_frame, text="Cálculo", padding=4)
        calc.pack(fill="x", pady=(10, 0))
        for param in CALCULATION_PARAMS:
            self._add_field(calc, param, self._calculation_vars)

        fmts = ttk.LabelFrame(self.params_frame, text="Formatos de salida", padding=4)
        fmts.pack(fill="x", pady=(10, 0))
        for key, label in _EXPORT_FORMATS:
            var = self.tk.BooleanVar(value=key in ("qe", "lammps"))
            ttk.Checkbutton(fmts, text=label, variable=var).pack(anchor="w")
            self._format_vars[key] = var

        self.force_var = self.tk.BooleanVar(value=False)
        ttk.Checkbutton(
            fmts,
            text="Exportar aunque falle la validación",
            variable=self.force_var,
        ).pack(anchor="w", pady=(4, 0))

    def _read_raw(self, store: dict[str, Any]) -> dict[str, Any]:
        return {key: var.get() for key, var in store.items()}

    def _discard_current_structure(self) -> None:
        """Drop the built structure and disable the actions that consume it."""
        self.atoms = None
        self.export_button.configure(state="disabled")
        self.png_button.configure(state="disabled")
        self.axes.clear()
        self.axes.set_title("Pulsa «Construir y previsualizar»")
        self.canvas.draw_idle()
        self._set_info("")
        self.status_var.set("Listo.")

    # ------------------------------------------------------------------
    # Build (threaded)
    # ------------------------------------------------------------------
    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.build_button.configure(state=state)
        if message:
            self.status_var.set(message)

    def _on_build(self) -> None:
        if self._busy:
            return
        kind = self._current_structure_key()
        raw_params = self._read_raw(self._param_vars)
        raw_mods = self._read_raw(self._modifier_vars)
        raw_groups = self._read_raw(self._functionalization_vars)
        supports_mods = STRUCTURES[kind].supports_modifiers

        self._set_busy(True, "Construyendo… (puede tardar en estructuras grandes)")

        def worker() -> None:
            try:
                atoms = build_structure(kind, raw_params)
                if supports_mods:
                    atoms = apply_modifiers(atoms, raw_mods)
                    # Groups go on after doping and defects: attaching first
                    # would decorate carbons a later vacancy removes.
                    atoms = apply_functionalization(
                        atoms, {**raw_mods, **raw_groups}
                    )
                self._queue.put(("built", atoms))
            except Exception as exc:  # surfaced to the user in a dialog
                self._queue.put(("error", (exc, traceback.format_exc())))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        """Drain worker results on the main thread (Tk is not thread-safe)."""
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "built":
                    self._on_built(payload)
                elif kind == "error":
                    exc, tb = payload
                    self._set_busy(False, "Error.")
                    self._show_error(exc, tb)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _on_built(self, atoms: Atoms) -> None:
        self.atoms = atoms
        self._set_busy(False, f"Estructura lista: {len(atoms)} átomos.")
        self.export_button.configure(state="normal")
        self.png_button.configure(state="normal")
        self._render(atoms)
        summary = describe_structure(atoms)
        # The structure can be geometrically perfect and still be a hopeless
        # request (Raman on a metal), so report both.
        try:
            physics = validate_calculation(
                atoms, self._read_raw(self._calculation_vars)
            )
        except Exception as exc:  # never let the report break the preview
            physics = f"No se pudo evaluar el cálculo: {exc}"
        self._set_info(f"{summary}\n\n--- Cálculo solicitado ---\n{physics}")

    def _render(self, atoms: Atoms) -> None:
        from ..viz.plot import draw_structure_on_axes

        self.axes.clear()
        # Bond inference is O(N^2); skip the wireframe on very large models so
        # the preview stays responsive. The atoms themselves still render.
        draw_structure_on_axes(atoms, self.axes, show_bonds=len(atoms) <= 4000)
        self.canvas.draw_idle()

    def _set_info(self, text: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _on_export(self) -> None:
        from tkinter import filedialog, messagebox

        if self.atoms is None:
            return
        formats = [key for key, var in self._format_vars.items() if var.get()]
        if not formats:
            messagebox.showwarning(
                "Sin formatos", "Marca al menos un formato de salida."
            )
            return

        outdir = filedialog.askdirectory(title="Carpeta de destino")
        if not outdir:
            return
        try:
            written = export_structure(
                self.atoms,
                Path(outdir),
                formats,
                force=bool(self.force_var.get()),
                calculation_values=self._read_raw(self._calculation_vars),
            )
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return

        listado = "\n".join(f"  • {p}" for p in written)
        self.status_var.set(f"Exportado: {len(written)} archivo(s).")
        messagebox.showinfo("Exportación completada", f"Archivos escritos:\n{listado}")

    def _on_save_png(self) -> None:
        from tkinter import filedialog, messagebox

        if self.atoms is None:
            return
        path = filedialog.asksaveasfilename(
            title="Guardar imagen",
            defaultextension=".png",
            filetypes=[("Imagen PNG", "*.png")],
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return
        self.status_var.set(f"Imagen guardada en {path}")
        messagebox.showinfo("Imagen guardada", str(path))

    # ------------------------------------------------------------------
    def _show_error(self, exc: Exception, tb: str) -> None:
        from tkinter import messagebox

        # Builders and validators raise ValueError with user-facing text; only
        # unexpected exception types warrant dumping a traceback.
        if isinstance(exc, (ValueError, IndexError, RuntimeError)):
            messagebox.showerror("No se pudo construir", str(exc))
        else:
            messagebox.showerror(
                "Error inesperado", f"{type(exc).__name__}: {exc}\n\n{tb}"
            )
        self._set_info(f"❌ {type(exc).__name__}: {exc}")


def main() -> int:
    """Launch the GUI. Returns a process exit code."""
    try:
        import tkinter as tk
    except ImportError:
        print(_TK_MISSING_MSG)
        return 1

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "Falta matplotlib, necesario para la vista previa 3D.\n"
            "Instálalo con:  pip install matplotlib"
        )
        return 1

    root = tk.Tk()
    CarbonForgeApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
