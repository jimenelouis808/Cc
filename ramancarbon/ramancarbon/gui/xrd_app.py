"""The diffraction section: phase identification and Rietveld refinement.

Laid out in the order the work is actually done, which is also the order
in which the answers become trustworthy: look at the pattern, identify the
phases, look at what could not be explained, and only then refine. The
refinement tab offers the staged automatic protocol and a full manual
parameter table side by side, because the automatic one is right most of
the time and the manual one is what you reach for when it is not.

All the logic is in :mod:`ramancarbon.gui.xrd_state`; this file is layout
and event wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import SectionApp, placeholder
from .theme import PAD
from .widgets import card, fill_table, hint, labelled, scrolled_text, set_text, table
from .xrd_state import PARAMETER_GROUPS, XRDSession

#: File types offered in the open dialog.
FILE_TYPES = (
    ("Difractogramas", "*.xy *.xye *.dat *.txt *.asc *.csv *.xrdml *.uxd"),
    ("Todos", "*.*"),
)


class XRDApp(SectionApp):
    """The DRX section of the suite."""

    def __init__(self, root, container, palette, fonts) -> None:
        super().__init__(root, container, palette, fonts)
        self.session = XRDSession()
        self._drawers = {
            "pattern": self._draw_pattern,
            "sticks": self._draw_sticks,
            "rietveld": self._draw_rietveld,
            "residual": self._draw_residual,
        }
        self._tab_canvases = {
            0: ("pattern",),
            1: ("sticks",),
            2: ("rietveld", "residual"),
            3: (),
        }
        self._build()
        self.set_status(
            "Carga un difractograma (.xy, .xye, .dat, .txt, .xrdml…) o pulsa "
            "Demo."
        )

    # ==================================================================
    # layout
    # ==================================================================
    def _build(self) -> None:
        ttk = self.ttk
        body = ttk.Frame(self.container, padding=(PAD["md"], PAD["sm"]))
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, width=290)
        sidebar.pack(side="left", fill="y", padx=(0, PAD["md"]))
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side="left", fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._build_tab_pattern()
        self._build_tab_phases()
        self._build_tab_refinement()
        self._build_tab_library()
        self.build_status(self.container)
        self.root.after(150, self.drain_queue)

    def _build_sidebar(self, parent) -> None:
        ttk, tk = self.ttk, self.tk

        outer, listbody = card(parent, "Difractogramas")
        outer.pack(fill="both", expand=True)
        self.pattern_list = tk.Listbox(
            listbody, exportselection=False, activestyle="none",
            background=self.palette.surface_alt, foreground=self.palette.text,
            selectbackground=self.palette.accent,
            selectforeground=self.palette.accent_text,
            highlightthickness=0, borderwidth=0, font=self.fonts["body"],
        )
        self.pattern_list.pack(fill="both", expand=True, pady=(0, PAD["sm"]))
        self.pattern_list.bind("<<ListboxSelect>>", self._on_select)

        row = ttk.Frame(listbody)
        row.pack(fill="x")
        ttk.Button(row, text="Abrir…", command=self._open_files).pack(
            side="left", padx=(0, PAD["xs"]))
        ttk.Button(row, text="Demo", command=self._load_demo).pack(
            side="left", padx=(0, PAD["xs"]))
        ttk.Button(row, text="Quitar", command=self._remove).pack(side="left")

        setup, setupbody = card(parent, "Equipo")
        setup.pack(fill="x", pady=(PAD["sm"], 0))
        self.anode_var = tk.StringVar(value="Cu")
        labelled(setupbody, "Ánodo", lambda p: ttk.Combobox(
            p, textvariable=self.anode_var, width=8, state="readonly",
            values=self.session.anode_choices()))
        self.kalpha2_var = tk.StringVar(value="0.5")
        labelled(setupbody, "Kα₂/Kα₁", lambda p: ttk.Entry(
            p, textvariable=self.kalpha2_var, width=8))
        self.counts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(setupbody, text="Los datos son cuentas (σ = √N)",
                        variable=self.counts_var).pack(anchor="w", pady=(PAD["xs"], 0))
        self.resolution_var = tk.StringVar(value="0.06")
        labelled(setupbody, "FWHM equipo (°)", lambda p: ttk.Entry(
            p, textvariable=self.resolution_var, width=8))
        hint(setupbody,
             "El Kα₂ va a 0.5 salvo que midas con monocromador o en "
             "sincrotrón, donde va a 0. La FWHM del equipo sale de medir un "
             "patrón de ensanchamiento (el silicio de la biblioteca sirve): "
             "un tamaño de cristalito citado contra una resolución supuesta "
             "es la suposición.",
             wrap=250)

        options, optionsbody = card(parent, "Análisis")
        options.pack(fill="x", pady=(PAD["sm"], 0))
        self.texture_var = tk.StringVar(value="")
        labelled(optionsbody, "Eje de textura", lambda p: ttk.Entry(
            p, textvariable=self.texture_var, width=10))
        self.max_phases_var = tk.StringVar(value="4")
        labelled(optionsbody, "Máx. fases", lambda p: ttk.Spinbox(
            p, from_=1, to=8, textvariable=self.max_phases_var, width=6))
        self.background_var = tk.StringVar(value="6")
        labelled(optionsbody, "Orden del fondo", lambda p: ttk.Spinbox(
            p, from_=1, to=20, textvariable=self.background_var, width=6))
        hint(optionsbody,
             "El eje de textura se escribe como «001» y hace falta para "
             "refinar orientación preferente. Sin él no se refina, y en un "
             "material laminar la textura acaba absorbida por el U_iso.",
             wrap=250)

        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=(PAD["sm"], 0))
        ttk.Button(actions, text="Identificar fases", style="Accent.TButton",
                   command=self._identify).pack(fill="x", pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Identificar y refinar",
                   command=lambda: self._identify(refine_after=True)).pack(
            fill="x", pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Guardar informe…",
                   command=self._save_report).pack(fill="x")

    def _build_tab_pattern(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Difractograma  ")
        outer, body = card(tab, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "pattern", lambda f: f.add_subplot(111))

        info, infobody = card(tab, "Picos detectados")
        info.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        hint(infobody,
             "Los picos SIN EXPLICAR no son un resto: son la fase que no "
             "esperabas. Si hay alguno intenso, descarga su CIF de la "
             "Crystallography Open Database y añádelo en la pestaña "
             "Biblioteca antes de refinar nada.",
             wrap=900)
        self.peak_table = table(
            infobody, ["2θ (°)", "d (Å)", "I", "FWHM (°)", "σ", "estado"], height=8
        )

    def _build_tab_phases(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Fases  ")
        outer, body = card(tab, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "sticks", lambda f: f.add_subplot(111))

        results, resultsbody = card(tab, "Fases identificadas")
        results.pack(fill="x", pady=(PAD["sm"], 0))
        self.phase_table = table(
            resultsbody,
            ["fase", "fórmula", "veredicto", "FOM", "% peso", "D (nm)"],
            height=6,
        )
        hint(resultsbody,
             "Las posiciones dependen solo de la red y son prueba fuerte. Las "
             "intensidades las estropea la orientación preferente en cualquier "
             "material laminar, así que un acuerdo de intensidades bajo suele "
             "ser textura y no una fase equivocada.",
             wrap=900)

    def _build_tab_refinement(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Rietveld  ")

        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", pady=(0, PAD["sm"]))
        ttk.Button(toolbar, text="Refinamiento automático", style="Accent.TButton",
                   command=self._auto_refine).pack(side="left", padx=(0, PAD["xs"]))
        ttk.Button(toolbar, text="Preparar manual",
                   command=self._prepare_manual).pack(side="left", padx=(0, PAD["xs"]))
        ttk.Button(toolbar, text="Refinar los libres",
                   command=self._refine_once).pack(side="left", padx=(0, PAD["sm"]))
        ttk.Label(toolbar, text="Liberar grupo:").pack(side="left", padx=(0, PAD["xs"]))
        self.group_var = self.tk.StringVar(value=PARAMETER_GROUPS[0][1])
        ttk.Combobox(toolbar, textvariable=self.group_var, width=26, state="readonly",
                     values=[label for _, label, _ in PARAMETER_GROUPS]).pack(
            side="left", padx=(0, PAD["xs"]))
        ttk.Button(toolbar, text="Añadir", command=self._free_group).pack(side="left")

        panes = ttk.Panedwindow(tab, orient="horizontal")
        panes.pack(fill="both", expand=True)

        left = ttk.Frame(panes)
        panes.add(left, weight=3)
        plot_card, plot_body = card(left, None)
        plot_card.pack(fill="both", expand=True)
        self.make_canvas(plot_body, "rietveld", lambda f: f.add_subplot(111))
        residual_card, residual_body = card(left, None)
        residual_card.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self.make_canvas(residual_body, "residual", lambda f: f.add_subplot(111),
                         figsize=(7.6, 2.4))

        right = ttk.Frame(panes)
        panes.add(right, weight=2)
        params, params_body = card(right, "Parámetros",
                                   "Doble clic para liberar o fijar; clic derecho "
                                   "para editar el valor.")
        params.pack(fill="both", expand=True)
        self.parameter_table = table(
            params_body, ["parámetro", "libre", "valor", "error", "grupo"], height=18
        )
        self.parameter_table.bind("<Double-1>", self._toggle_parameter)
        self.parameter_table.bind("<Button-3>", self._edit_parameter)
        hint(params_body,
             "El automático libera los parámetros por etapas, en el orden "
             "recomendado desde que existe el método. Soltarlos todos a la vez "
             "converge, da factores R plausibles y devuelve una estructura "
             "equivocada.",
             wrap=380)

    def _build_tab_library(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Biblioteca  ")

        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", pady=(0, PAD["sm"]))
        ttk.Button(toolbar, text="Añadir carpeta de CIF…",
                   command=self._add_cif_directory).pack(side="left", padx=(0, PAD["xs"]))
        ttk.Button(toolbar, text="Usar solo las marcadas",
                   command=self._pin_selected).pack(side="left", padx=(0, PAD["xs"]))
        ttk.Button(toolbar, text="Usar todas",
                   command=self._unpin).pack(side="left")

        outer, body = card(tab, "Fases de referencia")
        outer.pack(fill="both", expand=True)
        hint(body,
             "No hay búsqueda en línea en la COD, y es deliberado: un "
             "resultado que depende de la red no se reproduce, y los "
             "ordenadores de los equipos suelen estar sin conexión. Descarga "
             "el CIF de la fase que te falte desde crystallography.net, "
             "guárdalo en una carpeta y añádela aquí.",
             wrap=900)
        self.library_table = table(
            body, ["fase", "fórmula", "grupo espacial", "confianza", "origen"],
            height=16,
        )
        self.library_table.bind("<<TreeviewSelect>>", self._on_library_select)
        self.library_text = scrolled_text(body, self.palette, self.fonts["mono"],
                                          height=8)
        self._fill_library()

    # ==================================================================
    # events
    # ==================================================================
    def _settings_from_widgets(self) -> None:
        session = self.session
        session.anode = self.anode_var.get()
        try:
            session.kalpha2_ratio = max(0.0, float(self.kalpha2_var.get()))
        except ValueError:
            session.kalpha2_ratio = 0.5
        session.counts = bool(self.counts_var.get())
        try:
            session.instrument_fwhm = max(1e-4, float(self.resolution_var.get()))
        except ValueError:
            session.instrument_fwhm = 0.06
        try:
            session.max_phases = max(1, int(self.max_phases_var.get()))
        except ValueError:
            session.max_phases = 4
        try:
            session.background_order = max(1, int(self.background_var.get()))
        except ValueError:
            session.background_order = 6
        session.texture_axis = _parse_axis(self.texture_var.get())

    def _open_files(self) -> None:
        from tkinter import filedialog

        paths = filedialog.askopenfilenames(
            title="Abrir difractogramas", filetypes=FILE_TYPES, parent=self.root
        )
        if not paths:
            return
        self._settings_from_widgets()
        added = self.session.load([Path(p) for p in paths])
        self._refresh_list()
        self.flush_messages(self.session.messages)
        self.set_status(f"{added} difractograma(s) cargados.")

    def _load_demo(self) -> None:
        from ..examples.demo_data import xrd_demo_spectra

        for pattern in xrd_demo_spectra(seed=3):
            self.session.add_pattern(pattern)
        self._refresh_list()
        self.set_status(
            "Difractogramas de demostración cargados. Son CALCULADOS a partir "
            "de estructuras reales, no medidos."
        )

    def _remove(self) -> None:
        self.session.remove_current()
        self._refresh_list()
        self._redraw()

    def _refresh_list(self) -> None:
        self.pattern_list.delete(0, "end")
        for item in self.session.patterns:
            mark = "✓ " if item.analysed else "  "
            self.pattern_list.insert("end", mark + item.name)
        if 0 <= self.session.current < len(self.session.patterns):
            self.pattern_list.selection_clear(0, "end")
            self.pattern_list.selection_set(self.session.current)
        self._redraw()

    def _on_select(self, _event=None) -> None:
        selection = self.pattern_list.curselection()
        if not selection:
            return
        self.session.current = int(selection[0])
        self._redraw()

    def _identify(self, refine_after: bool = False) -> None:
        if self.session.item is None:
            self.warn("Sin datos", "Carga un difractograma primero.")
            return
        self._settings_from_widgets()

        def done(result) -> None:
            self.flush_messages(self.session.messages)
            self._refresh_list()
            self._redraw()
            if result is not None and result.search.accepted:
                names = ", ".join(m.crystal.name for m in result.search.accepted)
                self.set_status(f"Fases: {names}")
            else:
                self.set_status("Ninguna fase de la biblioteca explica el patrón.")

        self.run_async(
            lambda: self.session.analyse_current(refine_after=refine_after),
            done,
            "Identificando fases…" if not refine_after
            else "Identificando y refinando…",
        )

    def _auto_refine(self) -> None:
        if self.session.item is None:
            self.warn("Sin datos", "Carga un difractograma primero.")
            return
        self._settings_from_widgets()

        def done(result) -> None:
            self.flush_messages(self.session.messages)
            self._fill_parameters()
            self._redraw()
            if result is not None:
                self.set_status(
                    f"Rwp = {100 * result.r_wp:.2f} %, GOF = {result.gof:.3f}"
                )

        self.run_async(self.session.auto_refine_current, done,
                       "Refinamiento por etapas…")

    def _prepare_manual(self) -> None:
        self._settings_from_widgets()
        if self.session.prepare_manual() is None:
            self.flush_messages(self.session.messages)
            return
        self._fill_parameters()
        self.set_status(
            "Parámetros listos: escala y fondo libres. Libera el resto por "
            "grupos, en ese orden."
        )

    def _refine_once(self) -> None:
        if self.session.item is None or self.session.item.parameters is None:
            self.warn("Sin parámetros", "Pulsa «Preparar manual» primero.")
            return

        def done(result) -> None:
            self.flush_messages(self.session.messages)
            self._fill_parameters()
            self._redraw()
            if result is not None:
                self.set_status(
                    f"Rwp = {100 * result.r_wp:.2f} %, GOF = {result.gof:.3f}"
                )

        self.run_async(self.session.refine_current, done, "Refinando…")

    def _free_group(self) -> None:
        label = self.group_var.get()
        kind = next((k for k, text, _ in PARAMETER_GROUPS if text == label), None)
        item = self.session.item
        if kind is None or item is None or item.parameters is None:
            return
        for parameter in item.parameters:
            if parameter.kind == kind:
                parameter.free = True
        self._fill_parameters()
        self.set_status(f"Grupo «{label}» liberado.")

    def _toggle_parameter(self, _event=None) -> None:
        selection = self.parameter_table.selection()
        if not selection:
            return
        name = self.parameter_table.item(selection[0], "values")[0]
        self.session.toggle_parameter(name)
        self._fill_parameters()

    def _edit_parameter(self, event) -> None:
        from tkinter import simpledialog

        row = self.parameter_table.identify_row(event.y)
        if not row:
            return
        values = self.parameter_table.item(row, "values")
        name, current = values[0], values[2]
        answer = simpledialog.askstring(
            "Valor del parámetro", f"{name} =", initialvalue=current,
            parent=self.root,
        )
        if answer is None:
            return
        try:
            value = float(answer)
        except ValueError:
            self.warn("Valor inválido", f"«{answer}» no es un número.")
            return
        if self.session.set_parameter_value(name, value):
            self._fill_parameters()
        else:
            self.flush_messages(self.session.messages)

    def _fill_parameters(self) -> None:
        fill_table(
            self.parameter_table,
            ["parámetro", "libre", "valor", "error", "grupo"],
            self.session.parameter_rows(),
        )

    def _add_cif_directory(self) -> None:
        from tkinter import filedialog

        folder = filedialog.askdirectory(
            title="Carpeta con archivos CIF", parent=self.root
        )
        if not folder:
            return
        if folder not in self.session.cif_directories:
            self.session.cif_directories.append(folder)
        self._fill_library()
        self.set_status(f"Añadida la carpeta {folder}")

    def _pin_selected(self) -> None:
        chosen = [
            self.library_table.item(row, "values")[0]
            for row in self.library_table.selection()
        ]
        self.session.selected_phases = chosen
        self.set_status(
            f"Se buscarán solo {len(chosen)} fase(s)." if chosen
            else "No se ha marcado ninguna; se buscarán todas."
        )

    def _unpin(self) -> None:
        self.session.selected_phases = []
        self.library_table.selection_remove(*self.library_table.selection())
        self.set_status("Se buscarán todas las fases de la biblioteca.")

    def _fill_library(self) -> None:
        rows = []
        for entry in self.session.library():
            crystal = entry.crystal
            rows.append(
                (
                    crystal.name,
                    crystal.formula,
                    crystal.space_group,
                    crystal.confidence,
                    "incluida" if entry.bundled else str(entry.path.parent),
                )
            )
        fill_table(
            self.library_table,
            ["fase", "fórmula", "grupo espacial", "confianza", "origen"],
            rows,
        )

    def _on_library_select(self, _event=None) -> None:
        selection = self.library_table.selection()
        if not selection:
            return
        name = self.library_table.item(selection[0], "values")[0]
        crystal = next(
            (e.crystal for e in self.session.library() if e.crystal.name == name), None
        )
        if crystal is None:
            return
        text = crystal.describe()
        if crystal.notes:
            text += "\n\n" + crystal.notes
        set_text(self.library_text, text)

    def _save_report(self) -> None:
        from tkinter import filedialog

        item = self.session.item
        if item is None or item.result is None:
            self.warn("Sin análisis", "Identifica las fases primero.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar informe", defaultextension=".txt",
            initialfile=f"{item.name}_drx.txt", parent=self.root,
        )
        if not path:
            return
        Path(path).write_text(item.result.report(), encoding="utf-8")
        self.set_status(f"Informe guardado en {path}")

    # ==================================================================
    # drawing
    # ==================================================================
    def _on_tab_changed(self, _event=None) -> None:
        self.flush_dirty(self._visible())

    def _visible(self) -> tuple[str, ...]:
        try:
            index = self.notebook.index(self.notebook.select())
        except Exception:  # noqa: BLE001 - no tab selected yet
            return ()
        return self._tab_canvases.get(index, ())

    def _redraw(self) -> None:
        self.mark_dirty(*self._drawers)
        self.flush_dirty(self._visible())
        self._fill_tables()

    def _fill_tables(self) -> None:
        fill_table(
            self.peak_table,
            ["2θ (°)", "d (Å)", "I", "FWHM (°)", "σ", "estado"],
            self.session.peak_rows(),
        )
        fill_table(
            self.phase_table,
            ["fase", "fórmula", "veredicto", "FOM", "% peso", "D (nm)"],
            self.session.phase_rows(),
        )

    def _draw_pattern(self, figure) -> None:
        from .plots_xrd import plot_pattern

        ax = figure.add_subplot(111)
        item = self.session.item
        if item is None:
            placeholder(ax, "Carga un difractograma", self.palette)
            return
        result = item.result
        plot_pattern(
            ax, item.pattern, self.palette,
            peaks=result.peaks if result else None,
            unexplained=result.search.unexplained if result else None,
        )
        ax.set_title(item.name, fontsize=9)

    def _draw_sticks(self, figure) -> None:
        from .plots_xrd import plot_phase_sticks

        ax = figure.add_subplot(111)
        item = self.session.item
        if item is None or item.result is None:
            placeholder(ax, "Identifica las fases", self.palette)
            return
        plot_phase_sticks(ax, item.result, self.palette)

    def _draw_rietveld(self, figure) -> None:
        from .plots_xrd import plot_rietveld

        ax = figure.add_subplot(111)
        item = self.session.item
        if item is None or item.refinement is None:
            placeholder(ax, "Refina para ver el ajuste", self.palette)
            return
        plot_rietveld(ax, item.refinement, self.palette)

    def _draw_residual(self, figure) -> None:
        from .plots_xrd import plot_weighted_difference

        ax = figure.add_subplot(111)
        item = self.session.item
        if item is None or item.refinement is None:
            placeholder(ax, "", self.palette)
            return
        plot_weighted_difference(ax, item.refinement, self.palette)


def _parse_axis(text: str) -> Optional[tuple[int, int, int]]:
    """``"001"`` or ``"0 0 1"`` or ``"0,0,1"`` → ``(0, 0, 1)``."""
    cleaned = text.strip().replace(",", " ")
    if not cleaned:
        return None
    parts = cleaned.split()
    if len(parts) == 1 and len(parts[0]) == 3 and parts[0].lstrip("-").isdigit():
        parts = list(parts[0])
    if len(parts) != 3:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


__all__ = ["FILE_TYPES", "XRDApp"]
