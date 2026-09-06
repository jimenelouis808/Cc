"""The desktop application.

Layout: a spectrum list on the left, a tabbed workspace on the right, a
status bar underneath. Every long operation — loading a folder, fitting,
analysing a batch — runs on a worker thread and posts its result back
through a queue that the main thread drains with ``after``. Tk is not
thread-safe, so no worker ever touches a widget.

The tabs follow the order of the analysis rather than the order of the
code, which is the order a user works in:

1. **Espectro** — load, preprocess, look at what preprocessing did.
2. **Deconvolución** — build a model, fit it, read the residual.
3. **Informe** — the full written analysis.
4. **Diámetros** — the RBM region and what it implies.
5. **Comparación** — the batch table, and export.
6. **Base de datos** — what the program believes and where it got it.

Run with ``ramancarbon-gui`` or ``python -m ramancarbon.gui``.
"""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.io import TEXT_SUFFIXES
from ..models.fitting import PeakSpec
from ..models.lineshapes import PROFILES
from .plots import (
    figure_for_report,
    plot_fit,
    plot_overlay,
    plot_rbm,
    plot_spectrum,
    plot_strain_doping,
)
from .state import (
    BASELINE_METHODS,
    NORMALISATIONS,
    Session,
    laser_choices,
    preset_choices,
)
from .theme import PAD, PALETTES, apply_theme, matplotlib_style

WINDOW_TITLE = "ramancarbon — análisis Raman de nanocarbonos"

_TK_MISSING_MSG = """
No se encontró Tkinter, que es lo que dibuja la ventana.

  • Windows / macOS: reinstala Python desde python.org marcando la opción
    "tcl/tk and IDLE" durante la instalación.
  • Ubuntu / Debian:  sudo apt install python3-tk
  • Fedora:           sudo dnf install python3-tkinter
  • Arch:             sudo pacman -S tk

Mientras tanto puedes usar la línea de comandos, que no necesita Tkinter:

  ramancarbon analizar espectro.txt --laser 532
  ramancarbon lote carpeta/ --laser 532 --csv resultados.csv
""".strip()


class RamanCarbonApp:
    """Main application window."""

    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.session = Session()
        self.queue: queue.Queue = queue.Queue()
        self.busy = False
        self.palette = PALETTES[self.session.palette_name]
        self.fonts = apply_theme(root, self.palette)

        root.title(WINDOW_TITLE)
        root.geometry("1420x900")
        root.minsize(1080, 720)

        self._peak_rows: list[dict] = []
        self._canvases: dict[str, Any] = {}
        self._figures: dict[str, Any] = {}

        self._build_header()
        self._build_body()
        self._build_status()
        self._refresh_spectrum_list()
        self._set_status("Carga un espectro (.txt, .csv, .dat…) para empezar.")
        self.root.after(120, self._drain_queue)

    # ==================================================================
    # chrome
    # ==================================================================
    def _build_header(self) -> None:
        ttk, tk = self.ttk, self.tk
        header = ttk.Frame(self.root, padding=(PAD["lg"], PAD["md"], PAD["lg"], PAD["sm"]))
        header.pack(fill="x")

        ttk.Label(header, text="ramancarbon", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="  espectroscopía Raman de nanomateriales de carbono",
            style="Muted.TLabel",
        ).pack(side="left", padx=(0, PAD["lg"]))

        ttk.Button(header, text="Tema", command=self._toggle_theme).pack(side="right")

        self.laser_var = tk.StringVar(value="532")
        ttk.Button(header, text="Aplicar a todos",
                   command=lambda: self._apply_laser(True)).pack(side="right", padx=PAD["xs"])
        ttk.Button(header, text="Aplicar",
                   command=lambda: self._apply_laser(False)).pack(side="right", padx=PAD["xs"])
        laser_box = ttk.Combobox(header, textvariable=self.laser_var, width=8,
                                 values=laser_choices())
        laser_box.pack(side="right", padx=PAD["xs"])
        ttk.Label(header, text="Láser (nm):").pack(side="right", padx=(PAD["md"], PAD["xs"]))

    def _build_body(self) -> None:
        ttk = self.ttk
        body = ttk.Frame(self.root, padding=(PAD["lg"], 0, PAD["lg"], PAD["sm"]))
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, width=270)
        sidebar.pack(side="left", fill="y", padx=(0, PAD["md"]))
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side="left", fill="both", expand=True)
        self._build_tab_spectrum()
        self._build_tab_deconvolution()
        self._build_tab_report()
        self._build_tab_diameters()
        self._build_tab_batch()
        self._build_tab_database()

    def _build_sidebar(self, parent) -> None:
        from .widgets import card, hint

        ttk = self.ttk
        outer, body = card(parent, "Espectros")
        outer.pack(fill="both", expand=True)

        buttons = ttk.Frame(body, style="Card.TFrame")
        buttons.pack(fill="x", pady=(0, PAD["sm"]))
        ttk.Button(buttons, text="Abrir…", command=self._open_files).pack(
            side="left", fill="x", expand=True, padx=(0, PAD["xs"]))
        ttk.Button(buttons, text="Carpeta…", command=self._open_folder).pack(
            side="left", fill="x", expand=True)

        self.spectrum_list = self.tk.Listbox(
            body,
            background=self.palette.surface,
            foreground=self.palette.text,
            selectbackground=self.palette.accent,
            selectforeground=self.palette.accent_text,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.palette.border,
            activestyle="none",
            font=self.fonts["body"],
        )
        self.spectrum_list.pack(fill="both", expand=True, pady=PAD["xs"])
        self.spectrum_list.bind("<<ListboxSelect>>", self._on_select_spectrum)

        row = ttk.Frame(body, style="Card.TFrame")
        row.pack(fill="x", pady=PAD["xs"])
        ttk.Button(row, text="Quitar", command=self._remove_spectrum).pack(
            side="left", fill="x", expand=True, padx=(0, PAD["xs"]))
        ttk.Button(row, text="Ejemplo", command=self._load_demo).pack(
            side="left", fill="x", expand=True)

        self.control_var = self.tk.BooleanVar(value=False)
        ttk.Checkbutton(
            body,
            text="Usar como control (referencia)",
            variable=self.control_var,
            command=self._toggle_control,
        ).pack(anchor="w", pady=PAD["xs"])
        hint(
            body,
            "El control es la misma muestra sin tratar, medida el mismo día. "
            "Comparar contra él elimina la deriva del equipo y es mucho más "
            "fiable que comparar contra valores de la literatura.",
            wrap=230,
        )

        ttk.Button(body, text="Analizar", style="Accent.TButton",
                   command=self._analyse_current).pack(fill="x", pady=(PAD["sm"], PAD["xs"]))
        ttk.Button(body, text="Analizar todos", command=self._analyse_all).pack(fill="x")

    def _build_status(self) -> None:
        ttk = self.ttk
        bar = ttk.Frame(self.root, style="Toolbar.TFrame",
                        padding=(PAD["lg"], PAD["sm"]))
        bar.pack(fill="x", side="bottom")
        self.status_var = self.tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.status_var, style="Status.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=140)
        self.progress.pack(side="right")

    # ==================================================================
    # tabs
    # ==================================================================
    def _make_canvas(self, parent, key: str, subplots: Callable):
        """Embed a matplotlib figure, remembering it under ``key``."""
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure
        import matplotlib

        with matplotlib.rc_context(matplotlib_style(self.palette)):
            figure = Figure(figsize=(7.6, 5.0), dpi=100)
            subplots(figure)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        widget = canvas.get_tk_widget()
        widget.configure(background=self.palette.surface, highlightthickness=0)
        widget.pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.configure(background=self.palette.surface_alt)
        toolbar.update()
        toolbar.pack(fill="x")
        self._canvases[key] = canvas
        self._figures[key] = figure
        return figure, canvas

    def _build_tab_spectrum(self) -> None:
        from .widgets import card, hint, labelled, separator

        ttk, tk = self.ttk, self.tk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Espectro  ")

        controls = ttk.Frame(tab, width=330)
        controls.pack(side="left", fill="y", padx=(0, PAD["md"]))
        controls.pack_propagate(False)

        outer, body = card(controls, "Preprocesado",
                           "El orden importa: primero se quitan los rayos "
                           "cósmicos, luego se suaviza y por último se resta "
                           "la línea base.")
        outer.pack(fill="both", expand=True)

        self.despike_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="Eliminar rayos cósmicos",
                        variable=self.despike_var).pack(anchor="w")
        self.resample_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, text="Remuestrear a eje uniforme",
                        variable=self.resample_var).pack(anchor="w")

        self.smooth_var = tk.IntVar(value=0)
        labelled(body, "Suavizado (pts)",
                 lambda p: ttk.Spinbox(p, from_=0, to=51, increment=2,
                                       textvariable=self.smooth_var, width=8))
        hint(body, "0 = sin suavizar. Suavizar ensancha las bandas y hace que "
                   "las incertidumbres del ajuste parezcan mejores de lo que son.",
             wrap=290)

        self.baseline_var = tk.StringVar(value=BASELINE_METHODS[0][1])
        labelled(body, "Línea base",
                 lambda p: ttk.Combobox(p, textvariable=self.baseline_var, width=26,
                                        state="readonly",
                                        values=[label for _, label in BASELINE_METHODS]))
        self.lam_var = tk.StringVar(value="1e7")
        labelled(body, "Rigidez (λ)",
                 lambda p: ttk.Entry(p, textvariable=self.lam_var, width=10))
        self.p_var = tk.StringVar(value="0.001")
        labelled(body, "Asimetría (p)",
                 lambda p: ttk.Entry(p, textvariable=self.p_var, width=10))
        hint(body, "λ más grande = línea base más rígida. Si la base se mete "
                   "en el valle entre D y G, súbela.", wrap=290)

        self.normalise_var = tk.StringVar(value=NORMALISATIONS[0][1])
        labelled(body, "Normalizar",
                 lambda p: ttk.Combobox(p, textvariable=self.normalise_var, width=26,
                                        state="readonly",
                                        values=[label for _, label in NORMALISATIONS]))
        hint(body, "La normalización no cambia ningún cociente de "
                   "intensidades; sirve solo para comparar formas.", wrap=290)

        separator(body)
        ttk.Button(body, text="Aplicar preprocesado", style="Accent.TButton",
                   command=self._apply_preprocess).pack(fill="x")

        plot_area, plot_body = card(tab, None)
        plot_area.pack(side="left", fill="both", expand=True)
        self._make_canvas(plot_body, "spectrum",
                          lambda f: f.add_subplot(111))

    def _build_tab_deconvolution(self) -> None:
        from .widgets import card, hint, labelled, separator, table

        ttk, tk = self.ttk, self.tk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Deconvolución  ")

        controls = ttk.Frame(tab, width=360)
        controls.pack(side="left", fill="y", padx=(0, PAD["md"]))
        controls.pack_propagate(False)

        outer, body = card(controls, "Modelo",
                           "Añadir componentes siempre mejora el R². Usa el "
                           "criterio de información para decidir cuántas hacen "
                           "falta de verdad.")
        outer.pack(fill="both", expand=True)

        self.preset_var = tk.StringVar(value=preset_choices()[1][1])
        labelled(body, "Preajuste",
                 lambda p: ttk.Combobox(p, textvariable=self.preset_var, width=30,
                                        state="readonly",
                                        values=[label for _, label in preset_choices()]))
        row = ttk.Frame(body, style="Card.TFrame")
        row.pack(fill="x", pady=PAD["xs"])
        ttk.Button(row, text="Cargar preajuste", command=self._load_preset).pack(
            side="left", fill="x", expand=True, padx=(0, PAD["xs"]))
        ttk.Button(row, text="Comparar modelos", command=self._compare_models).pack(
            side="left", fill="x", expand=True)

        self.window_low_var = tk.StringVar(value="1100")
        self.window_high_var = tk.StringVar(value="1750")
        window_row = ttk.Frame(body, style="Card.TFrame")
        window_row.pack(fill="x", pady=PAD["xs"])
        ttk.Label(window_row, text="Ventana", style="Card.TLabel", width=16).pack(side="left")
        ttk.Entry(window_row, textvariable=self.window_low_var, width=8).pack(side="left")
        ttk.Label(window_row, text=" – ", style="Card.TLabel").pack(side="left")
        ttk.Entry(window_row, textvariable=self.window_high_var, width=8).pack(side="left")
        ttk.Label(window_row, text=" cm⁻¹", style="Card.TLabel").pack(side="left")

        self.background_var = tk.StringVar(value="linear")
        labelled(body, "Fondo del ajuste",
                 lambda p: ttk.Combobox(p, textvariable=self.background_var, width=16,
                                        state="readonly",
                                        values=["none", "constant", "linear", "quadratic"]))

        separator(body)
        ttk.Label(body, text="Componentes", style="Heading.TLabel").pack(anchor="w")
        hint(body, "Doble clic sobre una celda para editarla. La posición, la "
                   "altura y la anchura son valores iniciales; el ajuste los "
                   "mueve dentro de sus límites.", wrap=320)
        self.peak_table = table(
            body, ["Banda", "Perfil", "Centro", "Altura", "FWHM"], height=8)
        self.peak_table.bind("<Double-1>", self._edit_peak_cell)

        row2 = ttk.Frame(body, style="Card.TFrame")
        row2.pack(fill="x", pady=PAD["xs"])
        ttk.Button(row2, text="Añadir", command=self._add_peak).pack(
            side="left", fill="x", expand=True, padx=(0, PAD["xs"]))
        ttk.Button(row2, text="Quitar", command=self._remove_peak).pack(
            side="left", fill="x", expand=True)

        ttk.Button(body, text="Ajustar", style="Accent.TButton",
                   command=self._fit_manual).pack(fill="x", pady=PAD["sm"])

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True)
        plot_area, plot_body = card(right, None)
        plot_area.pack(fill="both", expand=True)

        def build(figure):
            grid = figure.add_gridspec(2, 1, height_ratios=(3.0, 1.0), hspace=0.08)
            figure.add_subplot(grid[0])
            figure.add_subplot(grid[1])

        self._make_canvas(plot_body, "fit", build)

        summary, summary_body = card(right, None)
        summary.pack(fill="x", pady=(PAD["sm"], 0))
        from .widgets import scrolled_text

        self.fit_text = scrolled_text(summary_body, self.palette, self.fonts["mono"],
                                      height=9)

    def _build_tab_report(self) -> None:
        from .widgets import card, scrolled_text

        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Informe  ")

        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", pady=(0, PAD["sm"]))
        ttk.Button(toolbar, text="Guardar informe…",
                   command=self._save_report).pack(side="left", padx=(0, PAD["xs"]))
        ttk.Button(toolbar, text="Guardar figura…",
                   command=self._save_figure).pack(side="left")

        outer, body = card(tab, None)
        outer.pack(fill="both", expand=True)
        self.report_text = scrolled_text(body, self.palette, self.fonts["mono"], height=34)

    def _build_tab_diameters(self) -> None:
        from .widgets import card, scrolled_text

        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Diámetros  ")

        top, top_body = card(tab, None)
        top.pack(fill="both", expand=True)

        def build(figure):
            grid = figure.add_gridspec(1, 2, wspace=0.28)
            figure.add_subplot(grid[0])
            figure.add_subplot(grid[1])

        self._make_canvas(top_body, "diameters", build)

        bottom, bottom_body = card(tab, None)
        bottom.pack(fill="x", pady=(PAD["sm"], 0))
        self.diameter_text = scrolled_text(bottom_body, self.palette,
                                           self.fonts["mono"], height=12)

    def _build_tab_batch(self) -> None:
        from .widgets import card, hint, table

        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Comparación  ")

        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", pady=(0, PAD["sm"]))
        ttk.Button(toolbar, text="Exportar CSV…",
                   command=self._export_csv).pack(side="left", padx=(0, PAD["xs"]))
        ttk.Button(toolbar, text="Superponer espectros",
                   command=self._draw_overlay).pack(side="left")

        outer, body = card(tab, "Resultados del lote")
        outer.pack(fill="both", expand=True)
        hint(body, "Los cocientes de una fila solo son comparables con los de "
                   "otra si ambos se midieron con el mismo láser y con la misma "
                   "base (áreas o alturas). La columna «laser_nm» está ahí "
                   "precisamente para que se vea.", wrap=900)
        self.batch_table = table(body, ["nombre"], height=14)

        plot_area, plot_body = card(tab, None)
        plot_area.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self._make_canvas(plot_body, "overlay", lambda f: f.add_subplot(111))

    def _build_tab_database(self) -> None:
        from .widgets import card, scrolled_text, table

        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Base de datos  ")

        left = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True, padx=(0, PAD["md"]))
        outer, body = card(left, "Bandas",
                           "Posiciones a 2.33 eV (532 nm) y su dispersión. El "
                           "programa las corrige al láser que estés usando "
                           "antes de comparar nada.")
        outer.pack(fill="both", expand=True)
        self.band_table = table(
            body, ["Banda", "Posición", "Ventana", "Dispersión", "Confianza"], height=18)
        self.band_table.bind("<<TreeviewSelect>>", self._on_band_selected)

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True)
        detail, detail_body = card(right, "Detalle y fuente")
        detail.pack(fill="both", expand=True)
        self.band_text = scrolled_text(detail_body, self.palette, self.fonts["body"],
                                       height=20)
        self._fill_database_tab()

    def _fill_database_tab(self) -> None:
        from .widgets import fill_table

        db = self.session.db
        rows = []
        for band in sorted(db.bands.values(), key=lambda b: b.position):
            rows.append([
                band.key,
                f"{band.position:.0f}",
                f"{band.window[0]:.0f}–{band.window[1]:.0f}",
                f"{band.dispersion:+.0f} cm⁻¹/eV",
                band.confidence,
            ])
        fill_table(self.band_table,
                   ["Banda", "Posición", "Ventana", "Dispersión", "Confianza"], rows)

    def _on_band_selected(self, _event=None) -> None:
        from .widgets import set_text

        selection = self.band_table.selection()
        if not selection:
            return
        key = self.band_table.item(selection[0], "values")[0]
        band = self.session.db.bands.get(key)
        if band is None:
            return
        laser = self._current_laser()
        lines = [
            f"{band.key} — {band.name}",
            "",
            f"Posición de referencia : {band.position:.1f} cm⁻¹ (a 2.33 eV)",
            f"Dispersión             : {band.dispersion:+.1f} cm⁻¹/eV",
        ]
        if laser:
            from ..core.spectrum import laser_energy_ev

            ev = laser_energy_ev(laser)
            lo, hi = band.window_at(ev)
            lines += [
                f"A tu láser ({laser:g} nm)   : {band.position_at(ev):.1f} cm⁻¹",
                f"Ventana de búsqueda    : {lo:.0f}–{hi:.0f} cm⁻¹",
            ]
        lines += [
            f"FWHM típica            : {band.typical_fwhm[0]:g}–{band.typical_fwhm[1]:g} cm⁻¹",
            f"Perfil por defecto     : {band.default_profile}",
            f"Orden                  : {band.order}",
            f"Aparece en             : {', '.join(band.occurs_in) or '—'}",
            f"Confianza del valor    : {band.confidence}",
            "",
            "ORIGEN FÍSICO",
            band.origin,
            "",
            "NOTAS",
            band.notes,
            "",
            "FUENTE",
            band.source,
        ]
        set_text(self.band_text, "\n".join(lines))

    # ==================================================================
    # actions
    # ==================================================================
    def _current_laser(self) -> Optional[float]:
        item = self.session.active
        if item is not None and item.raw.laser_nm is not None:
            return item.raw.laser_nm
        try:
            return float(self.laser_var.get())
        except (TypeError, ValueError):
            return None

    def _open_files(self) -> None:
        from tkinter import filedialog

        patterns = " ".join(f"*{suffix}" for suffix in TEXT_SUFFIXES)
        paths = filedialog.askopenfilenames(
            title="Abrir espectros",
            filetypes=[("Espectros de texto", patterns), ("Todos los archivos", "*.*")],
        )
        if paths:
            self._load_paths([Path(p) for p in paths])

    def _open_folder(self) -> None:
        from tkinter import filedialog

        folder = filedialog.askdirectory(title="Abrir carpeta de espectros")
        if not folder:
            return
        root = Path(folder)
        files = sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
        )
        if not files:
            self._warn("Carpeta vacía",
                       f"No hay archivos con extensión {', '.join(TEXT_SUFFIXES)} "
                       f"en {root}.")
            return
        self._load_paths(files)

    def _load_paths(self, paths: list[Path]) -> None:
        laser = None
        try:
            laser = float(self.laser_var.get())
        except (TypeError, ValueError):
            pass
        added = self.session.load(paths, laser_nm=None)
        # Only fill in the laser where the file itself did not say.
        for item in self.session.spectra:
            if item.raw.laser_nm is None and laser:
                item.raw.laser_nm = laser
        self._refresh_spectrum_list()
        self._flush_messages()
        self._set_status(f"{added} espectro(s) cargado(s) de {len(paths)} archivo(s).")
        if added:
            self.session.current = len(self.session.spectra) - added
            self._select_index(self.session.current)

    def _load_demo(self) -> None:
        """Load a synthetic example so the interface can be tried immediately."""
        from ..examples.demo_data import demo_spectra

        for spectrum in demo_spectra():
            self.session.add(spectrum)
        self._refresh_spectrum_list()
        self._select_index(len(self.session.spectra) - 1)
        self._set_status(
            "Ejemplos sintéticos cargados. Son datos generados, no medidas "
            "reales: sirven para probar la interfaz."
        )

    def _refresh_spectrum_list(self) -> None:
        self.spectrum_list.delete(0, "end")
        for item in self.session.spectra:
            mark = "◆ " if item.is_control else "  "
            laser = f"{item.raw.laser_nm:g}nm" if item.raw.laser_nm else "¿nm?"
            self.spectrum_list.insert("end", f"{mark}{item.name}  [{laser}] {item.status}")
        if 0 <= self.session.current < len(self.session.spectra):
            self.spectrum_list.selection_clear(0, "end")
            self.spectrum_list.selection_set(self.session.current)

    def _select_index(self, index: int) -> None:
        if not 0 <= index < len(self.session.spectra):
            return
        self.session.current = index
        self._refresh_spectrum_list()
        self._on_select_spectrum()

    def _on_select_spectrum(self, _event=None) -> None:
        selection = self.spectrum_list.curselection()
        if selection:
            self.session.current = selection[0]
        item = self.session.active
        if item is None:
            return
        self.control_var.set(item.is_control)
        if item.raw.laser_nm:
            self.laser_var.set(f"{item.raw.laser_nm:g}")
        self._redraw_all()

    def _remove_spectrum(self) -> None:
        self.session.remove(self.session.current)
        self._refresh_spectrum_list()
        self._redraw_all()

    def _toggle_control(self) -> None:
        item = self.session.active
        if item is None:
            return
        if self.control_var.get():
            for other in self.session.spectra:
                other.is_control = False
        item.is_control = self.control_var.get()
        self._refresh_spectrum_list()

    def _apply_laser(self, everywhere: bool) -> None:
        text = self.laser_var.get().strip()
        try:
            value = float(text) if text else None
        except ValueError:
            self._warn("Valor no válido",
                       f"«{text}» no es una longitud de onda. Escribe un número "
                       "en nanómetros, por ejemplo 532.")
            return
        count = self.session.set_laser(value, all_spectra=everywhere)
        self._refresh_spectrum_list()
        self._set_status(
            f"Láser fijado a {value:g} nm en {count} espectro(s). Los resultados "
            "previos se han descartado porque dependen de la excitación."
        )
        self._redraw_all()

    def _settings_from_widgets(self) -> None:
        settings = self.session.preprocess_settings
        settings.despike = bool(self.despike_var.get())
        settings.resample = bool(self.resample_var.get())
        settings.smooth_window = int(self.smooth_var.get() or 0)
        settings.baseline_method = _key_for_label(BASELINE_METHODS, self.baseline_var.get())
        settings.normalise = _key_for_label(NORMALISATIONS, self.normalise_var.get())
        try:
            settings.baseline_lam = float(self.lam_var.get())
        except ValueError:
            settings.baseline_lam = 1e7
        try:
            settings.baseline_p = float(self.p_var.get())
        except ValueError:
            settings.baseline_p = 0.001

    def _apply_preprocess(self) -> None:
        if self.session.active is None:
            self._warn("Sin espectro", "Carga y selecciona un espectro primero.")
            return
        self._settings_from_widgets()
        self.session.preprocess_active()
        self._flush_messages()
        self._refresh_spectrum_list()
        self._draw_spectrum()
        self._set_status(
            "Preprocesado aplicado: " + self.session.preprocess_settings.describe()
        )

    # -- background work ------------------------------------------------
    def _run_async(self, work: Callable[[], Any], done: Callable[[Any], None],
                   message: str) -> None:
        """Run ``work`` on a thread; call ``done`` on the main thread after.

        Tk is not thread-safe, so the worker returns a value through the
        queue and every widget touch happens in :meth:`_drain_queue`.
        """
        if self.busy:
            self._set_status("Ya hay un cálculo en marcha; espera a que termine.")
            return
        self.busy = True
        self.progress.start(12)
        self._set_status(message)

        def target() -> None:
            try:
                result = work()
                self.queue.put(("ok", done, result))
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                self.queue.put(("error", done, (exc, traceback.format_exc())))

        threading.Thread(target=target, daemon=True).start()

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, done, payload = self.queue.get_nowait()
                self.busy = False
                self.progress.stop()
                if kind == "ok":
                    done(payload)
                else:
                    exc, tb = payload
                    self._error(exc, tb)
        except queue.Empty:
            pass
        self.root.after(120, self._drain_queue)

    def _analyse_current(self) -> None:
        if self.session.active is None:
            self._warn("Sin espectro", "Carga y selecciona un espectro primero.")
            return
        if self.session.active.raw.laser_nm is None:
            self._warn(
                "Falta el láser",
                "Indica la longitud de onda de excitación antes de analizar.\n\n"
                "Sin ella no se corrigen las posiciones por dispersión (la banda "
                "D se mueve 22 cm⁻¹ entre 532 y 633 nm), no se puede calcular el "
                "tamaño de cristalito ni la densidad de defectos, y los "
                "desplazamientos de D y 2D no son interpretables.",
            )
            return
        self._settings_from_widgets()
        self._run_async(
            self.session.analyse_active,
            self._after_analysis,
            "Analizando…",
        )

    def _analyse_all(self) -> None:
        if not self.session.spectra:
            self._warn("Sin espectros", "Carga algún espectro primero.")
            return
        self._settings_from_widgets()
        self._run_async(
            self.session.analyse_all,
            self._after_batch,
            f"Analizando {len(self.session.spectra)} espectros…",
        )

    def _after_analysis(self, _item) -> None:
        self._flush_messages()
        self._refresh_spectrum_list()
        self._redraw_all()
        item = self.session.active
        if item and item.result:
            self._set_status(
                f"{item.name}: {item.result.classification.label} "
                f"(confianza {item.result.classification.confidence})"
            )
            self.notebook.select(2)

    def _after_batch(self, counts) -> None:
        ok, bad = counts
        self._flush_messages()
        self._refresh_spectrum_list()
        self._redraw_all()
        self._set_status(f"Lote terminado: {ok} analizados, {bad} con error.")
        self.notebook.select(4)

    # -- deconvolution editor -------------------------------------------
    def _load_preset(self) -> None:
        from .widgets import fill_table

        if self.session.active is None:
            self._warn("Sin espectro", "Carga y selecciona un espectro primero.")
            return
        key = _key_for_label(preset_choices(), self.preset_var.get())
        try:
            specs = self.session.preset_specs(key)
        except ValueError as exc:
            self._warn("No se puede usar este preajuste", str(exc))
            return
        self._peak_rows = [_spec_to_row(spec) for spec in specs]
        centres = [spec.centre for spec in specs]
        widths = [spec.fwhm for spec in specs]
        if centres:
            self.window_low_var.set(f"{min(centres) - 2 * max(widths):.0f}")
            self.window_high_var.set(f"{max(centres) + 2 * max(widths):.0f}")
        fill_table(self.peak_table, ["Banda", "Perfil", "Centro", "Altura", "FWHM"],
                   [_row_values(row) for row in self._peak_rows])
        self._set_status(f"Preajuste cargado: {len(specs)} componentes.")

    def _add_peak(self) -> None:
        from .widgets import fill_table

        item = self.session.active
        centre = 1500.0
        height = 1.0
        if item is not None:
            spectrum = item.display
            peak = spectrum.max_in(*spectrum.range)
            if peak:
                centre, height = peak[0], peak[1]
        self._peak_rows.append(
            {"name": f"P{len(self._peak_rows) + 1}", "profile": "lorentzian",
             "centre": centre, "height": height, "fwhm": 30.0, "band": None}
        )
        fill_table(self.peak_table, ["Banda", "Perfil", "Centro", "Altura", "FWHM"],
                   [_row_values(row) for row in self._peak_rows])

    def _remove_peak(self) -> None:
        from .widgets import fill_table

        selection = self.peak_table.selection()
        if not selection:
            return
        index = self.peak_table.index(selection[0])
        if 0 <= index < len(self._peak_rows):
            del self._peak_rows[index]
        fill_table(self.peak_table, ["Banda", "Perfil", "Centro", "Altura", "FWHM"],
                   [_row_values(row) for row in self._peak_rows])

    def _edit_peak_cell(self, event) -> None:
        from tkinter import simpledialog
        from .widgets import fill_table

        row_id = self.peak_table.identify_row(event.y)
        column_id = self.peak_table.identify_column(event.x)
        if not row_id or not column_id:
            return
        index = self.peak_table.index(row_id)
        column = int(column_id.replace("#", "")) - 1
        if not 0 <= index < len(self._peak_rows):
            return
        keys = ["name", "profile", "centre", "height", "fwhm"]
        key = keys[column]
        current = self._peak_rows[index][key]
        prompt = {
            "name": "Nombre de la componente:",
            "profile": "Perfil (" + ", ".join(PROFILES) + "):",
            "centre": "Centro inicial (cm⁻¹):",
            "height": "Altura inicial:",
            "fwhm": "FWHM inicial (cm⁻¹):",
        }[key]
        answer = simpledialog.askstring("Editar componente", prompt,
                                        initialvalue=str(current), parent=self.root)
        if answer is None:
            return
        if key in {"centre", "height", "fwhm"}:
            try:
                self._peak_rows[index][key] = float(answer)
            except ValueError:
                self._warn("Valor no válido", f"«{answer}» no es un número.")
                return
        elif key == "profile":
            if answer not in PROFILES:
                self._warn("Perfil desconocido",
                           f"«{answer}» no existe. Usa uno de: {', '.join(PROFILES)}.")
                return
            self._peak_rows[index][key] = answer
        else:
            self._peak_rows[index][key] = answer
        fill_table(self.peak_table, ["Banda", "Perfil", "Centro", "Altura", "FWHM"],
                   [_row_values(row) for row in self._peak_rows])

    def _fit_manual(self) -> None:
        if self.session.active is None:
            self._warn("Sin espectro", "Carga y selecciona un espectro primero.")
            return
        if not self._peak_rows:
            self._warn("Modelo vacío",
                       "Carga un preajuste o añade componentes antes de ajustar.")
            return
        try:
            window = (float(self.window_low_var.get()), float(self.window_high_var.get()))
        except ValueError:
            self._warn("Ventana no válida", "Los límites deben ser números en cm⁻¹.")
            return
        specs = [_row_to_spec(row) for row in self._peak_rows]
        background = self.background_var.get()
        self._run_async(
            lambda: self.session.fit_manual(specs, window, background),
            self._after_manual_fit,
            "Ajustando…",
        )

    def _after_manual_fit(self, result) -> None:
        from .widgets import set_text

        self._flush_messages()
        if result is None:
            self._set_status("El ajuste no se ha podido completar; mira los avisos.")
            return
        self._draw_fit()
        set_text(self.fit_text, result.summary())
        self._set_status(f"Ajuste terminado: R² = {result.r_squared:.5f}")

    def _compare_models(self) -> None:
        item = self.session.active
        if item is None:
            self._warn("Sin espectro", "Carga y selecciona un espectro primero.")
            return
        target = item.processed or item.raw

        def work():
            from ..models.deconvolution import compare_models

            return compare_models(target, db=self.session.db)

        def done(comparison):
            from .widgets import set_text

            item.manual_fit = comparison.results[comparison.best]
            self._draw_fit()
            set_text(self.fit_text,
                     comparison.summary() + "\n\n" + item.manual_fit.summary())
            self._set_status(f"Mejor modelo: {comparison.best}")

        self._run_async(work, done, "Comparando modelos…")

    # ==================================================================
    # drawing
    # ==================================================================
    def _redraw_all(self) -> None:
        self._draw_spectrum()
        self._draw_fit()
        self._draw_report()
        self._draw_diameters()
        self._draw_batch()

    def _with_style(self, key: str, draw: Callable) -> None:
        import matplotlib

        figure = self._figures.get(key)
        if figure is None:
            return
        with matplotlib.rc_context(matplotlib_style(self.palette)):
            for ax in figure.axes:
                ax.clear()
            draw(figure)
        self._canvases[key].draw_idle()

    def _draw_spectrum(self) -> None:
        item = self.session.active

        def draw(figure):
            ax = figure.axes[0]
            if item is None:
                _placeholder(ax, "Carga un espectro para verlo aquí", self.palette)
                return
            diagnostics = item.diagnostics or {}
            plot_spectrum(
                ax,
                item.display,
                self.palette,
                raw=item.raw if item.processed is not None else None,
                baseline=diagnostics.get("baseline"),
                baseline_x=diagnostics.get("baseline_x"),
                peaks=item.result.peaks if item.result else None,
                title=item.name,
            )
            figure.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.12)

        self._with_style("spectrum", draw)

    def _draw_fit(self) -> None:
        item = self.session.active
        fit = None
        if item is not None:
            fit = item.manual_fit or (item.result.fit if item.result else None)

        def draw(figure):
            main, residual = figure.axes[0], figure.axes[1]
            if fit is None:
                _placeholder(main, "Carga un preajuste y pulsa «Ajustar»", self.palette)
                residual.set_axis_off()
                return
            plot_fit(main, fit, self.palette, residual_axes=residual,
                     title=f"Deconvolución — {item.name}")
            main.set_xticklabels([])
            main.set_xlabel("")
            figure.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.12)

        self._with_style("fit", draw)

    def _draw_report(self) -> None:
        from .widgets import set_text

        item = self.session.active
        if item is None or item.result is None:
            set_text(self.report_text,
                     "Pulsa «Analizar» para generar el informe completo.\n\n"
                     "El informe incluye la identificación del material con su "
                     "evidencia, las bandas asignadas, la deconvolución D–G con "
                     "comparación de modelos, los cocientes de intensidad, los "
                     "diámetros y los desplazamientos respecto a la referencia.")
            return
        set_text(self.report_text, item.result.report())

    def _draw_diameters(self) -> None:
        from .widgets import set_text

        item = self.session.active
        result = item.result if item else None

        def draw(figure):
            left, right = figure.axes[0], figure.axes[1]
            if result is None:
                _placeholder(left, "Analiza un espectro", self.palette)
                right.set_axis_off()
                return
            plot_rbm(left, result, self.palette)
            plot_strain_doping(right, result, self.palette)
            figure.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.14)

        self._with_style("diameters", draw)

        if result is None:
            set_text(self.diameter_text, "")
            return
        lines: list[str] = []
        if not result.rbm.covered:
            lines.append(result.rbm.note)
        elif not result.rbm.diameters:
            lines.append(result.rbm.note or "No se han detectado RBM.")
        for estimate in result.rbm.diameters:
            lines.append(f"RBM {estimate.input_value:7.1f} cm⁻¹  →  d = {estimate}")
            candidates = result.rbm.chiralities.get(estimate.input_value, [])
            if candidates:
                lines.append("    (n,m) compatibles: " +
                             ", ".join(str(c.label) for c in candidates[:6]))
        for pair in [p for p in result.rbm.wall_pairs if p.plausible][:4]:
            lines.append(str(pair))
        if result.g_split_diameter:
            lines.append(f"Desdoblamiento G  →  d = {result.g_split_diameter}")
            lines.extend("  ⚠ " + w for w in result.g_split_diameter.warnings)
        set_text(self.diameter_text, "\n".join(lines))

    def _draw_batch(self) -> None:
        from .widgets import fill_table

        columns, rows = self.session.results_table()
        if columns:
            fill_table(self.batch_table, columns, rows)

    def _draw_overlay(self) -> None:
        spectra = [item.display for item in self.session.spectra]

        def draw(figure):
            ax = figure.axes[0]
            if not spectra:
                _placeholder(ax, "Carga espectros para superponerlos", self.palette)
                return
            plot_overlay(ax, spectra, self.palette, offset=0.25)
            ax.set_title("Espectros superpuestos (desplazados verticalmente)")
            figure.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.13)

        self._with_style("overlay", draw)

    # ==================================================================
    # export
    # ==================================================================
    def _save_report(self) -> None:
        from tkinter import filedialog

        item = self.session.active
        if item is None or item.result is None:
            self._warn("Nada que guardar", "Analiza un espectro primero.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar informe",
            defaultextension=".txt",
            initialfile=f"{item.name}_informe.txt",
            filetypes=[("Texto", "*.txt"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        Path(path).write_text(item.result.report(), encoding="utf-8")
        self._set_status(f"Informe guardado en {path}")

    def _save_figure(self) -> None:
        from tkinter import filedialog
        import matplotlib

        item = self.session.active
        if item is None or item.result is None:
            self._warn("Nada que guardar", "Analiza un espectro primero.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar figura",
            defaultextension=".png",
            initialfile=f"{item.name}_figura.png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if not path:
            return
        with matplotlib.rc_context(matplotlib_style(self.palette)):
            figure = figure_for_report(item.result, self.palette)
            figure.savefig(path)
        self._set_status(f"Figura guardada en {path}")

    def _export_csv(self) -> None:
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            title="Exportar resultados",
            defaultextension=".csv",
            initialfile="resultados_raman.csv",
            filetypes=[("CSV", "*.csv"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            self.session.export_table(path)
        except ValueError as exc:
            self._warn("Nada que exportar", str(exc))
            return
        self._set_status(f"Resultados exportados a {path}")

    # ==================================================================
    # misc
    # ==================================================================
    def _toggle_theme(self) -> None:
        self.session.palette_name = "oscuro" if self.session.palette_name == "claro" else "claro"
        self._warn(
            "Cambio de tema",
            "El tema se aplicará completamente al reiniciar la aplicación.\n\n"
            "Las figuras sí cambian ahora mismo.",
        )
        self.palette = PALETTES[self.session.palette_name]
        self._redraw_all()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _flush_messages(self) -> None:
        """Move queued session messages into the status bar."""
        if not self.session.messages:
            return
        level, text = self.session.messages[-1]
        prefix = {"error": "✗ ", "warning": "⚠ ", "info": ""}.get(level, "")
        self._set_status(prefix + text)

    def _warn(self, title: str, message: str) -> None:
        from tkinter import messagebox

        messagebox.showwarning(title, message, parent=self.root)

    def _error(self, exc: Exception, tb: str) -> None:
        from tkinter import messagebox

        self._set_status(f"✗ {exc}")
        messagebox.showerror(
            "Error",
            f"{type(exc).__name__}: {exc}\n\n{tb[-1500:]}",
            parent=self.root,
        )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _placeholder(ax, text: str, palette) -> None:
    """Empty-state message on an otherwise blank axes."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            color=palette.text_muted, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _key_for_label(pairs, label: str) -> str:
    """Reverse-map a combo box label back to its key."""
    for key, text in pairs:
        if text == label:
            return key
    return pairs[0][0]


def _spec_to_row(spec: PeakSpec) -> dict:
    return {
        "name": spec.name,
        "profile": spec.profile,
        "centre": spec.centre,
        "height": spec.height,
        "fwhm": spec.fwhm,
        "band": spec.band,
        "centre_bounds": spec.centre_bounds,
        "fwhm_bounds": spec.fwhm_bounds,
    }


def _row_values(row: dict) -> list:
    return [
        row["name"],
        row["profile"],
        f"{row['centre']:.1f}",
        f"{row['height']:.4g}",
        f"{row['fwhm']:.1f}",
    ]


def _row_to_spec(row: dict) -> PeakSpec:
    return PeakSpec(
        name=row["name"],
        profile=row["profile"],
        centre=row["centre"],
        height=row["height"],
        fwhm=row["fwhm"],
        centre_bounds=row.get("centre_bounds"),
        fwhm_bounds=row.get("fwhm_bounds"),
        band=row.get("band"),
    )


def main() -> int:
    """Entry point for ``ramancarbon-gui``.

    Returns a process exit code rather than raising, so a missing Tkinter
    produces installation instructions instead of a traceback.
    """
    try:
        import tkinter as tk
    except ImportError:
        print(_TK_MISSING_MSG)
        return 1
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"No se pudo abrir la ventana: {exc}")
        print("\n¿Estás en una sesión sin entorno gráfico (SSH sin X11)?")
        print("La línea de comandos funciona sin pantalla:\n")
        print("  ramancarbon analizar espectro.txt --laser 532")
        return 1
    RamanCarbonApp(root)
    root.mainloop()
    return 0


__all__ = ["RamanCarbonApp", "main"]
