"""The "Analizar resultados" tab: read finished runs back (bands, Raman/IR spectra)."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional


class AnalysisTab:
    """Open band structures and spectra and plot them.

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

        from ...results.bands import (
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

    def _on_open_spectrum(self) -> None:
        from tkinter import filedialog

        from ...results.spectra import read_dynmat

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
