"""The dichalcogenide section: layer count, phase, oxides and what else is there.

Separate from the carbon section because the physics is separate — nobody
needs the program deciding between "multi-walled nanotube" and "bilayer
MoS₂", since the user knows what they put under the objective — but
sharing its :class:`~ramancarbon.gui.state.Session`, so a spectrum loaded
in either section is available in both, and so are the preprocessing
settings. A file is a file, and a baseline is a baseline.

Three of the tabs exist because a real dichalcogenide sample is not one
phase.

**Óxidos e intercara** is often the answer rather than a footnote: an
MoSe₂ measured in air is usually MoO₃ + MoSe₂, and which of the three ways
the oxide got there — grown in, stored in, or made by the laser during
this measurement — decides whether the numbers are usable at all.

**Otras fases** runs the general phase catalogue over the same spectrum.
That search is deliberately not restricted to the metals the chalcogenide
contains, which is what the oxide search does, because the phases it
catches are the ones the narrow search cannot see by construction:
unreacted selenium at 237 cm⁻¹, sulfur at 473, an iron oxide from the
catalyst, the titania of the support. A precursor that did not react is
not in the lattice, and a composition measured by EDS will not tell you
which of the two it is.

**Biblioteca** is the catalogue itself, with every source and every
confidence, including the list of what was deliberately left out. A
library you cannot read is a library you have to trust.
"""

from __future__ import annotations

from pathlib import Path

from .base import SectionApp, placeholder
from .state import BASELINE_METHODS, NORMALISATIONS, key_for_label, label_for_key
from .theme import PAD
from .widgets import (
    card,
    fill_table,
    hint,
    labelled,
    scrollable_column,
    scrolled_text,
    set_text,
    table,
)


class TMDApp(SectionApp):
    """The Raman-TMD section of the suite."""

    def __init__(self, root, container, palette, fonts, session) -> None:
        super().__init__(root, container, palette, fonts)
        self.session = session
        self._drawers = {
            "tmd": self._draw_tmd,
            "modes": self._draw_modes,
            "oxides": self._draw_oxides,
        }
        self._tab_canvases = {
            0: ("tmd",),
            1: ("modes",),
            2: ("oxides",),
            3: (),
            4: (),
            5: (),
        }
        self._build()
        self._widgets_from_settings()
        self._fill_library()
        self.set_status(
            "Carga un espectro de dicalcogenuro y pulsa «Analizar como TMD». "
            "La biblioteca cubre S, Se y Te de Mo, W, Ti, Nb, Ta y Fe."
        )

    # ==================================================================
    # layout
    # ==================================================================
    def _build(self) -> None:
        ttk = self.ttk
        body = ttk.Frame(self.container, padding=(PAD["md"], PAD["sm"]))
        body.pack(fill="both", expand=True)

        # Scrollable: this column's cards add up to more than any window is
        # tall, and a fixed frame simply clips them — which is how the
        # analysis buttons ended up invisible below the fold.
        sidebar_column, sidebar = scrollable_column(body, width=300)
        sidebar_column.pack(side="left", fill="y", padx=(0, PAD["md"]))
        self._build_sidebar(sidebar)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side="left", fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._build_tab_spectrum()
        self._build_tab_modes()
        self._build_tab_oxides()
        self._build_tab_phases()
        self._build_tab_batch()
        self._build_tab_library()
        self.build_status(self.container)
        self.root.after(150, self.drain_queue)

    def _build_sidebar(self, parent) -> None:
        ttk, tk = self.ttk, self.tk

        outer, listbody = card(parent, "Espectros",
                               "Compartidos con la sección de carbono.")
        outer.pack(fill="both", expand=True)
        self.spectrum_list = tk.Listbox(
            listbody, exportselection=False, activestyle="none",
            background=self.palette.surface_alt, foreground=self.palette.text,
            selectbackground=self.palette.accent,
            selectforeground=self.palette.accent_text,
            highlightthickness=0, borderwidth=0, font=self.fonts["body"],
            height=7,
        )
        self.spectrum_list.pack(fill="both", expand=True, pady=(0, PAD["sm"]))
        self.spectrum_list.bind("<<ListboxSelect>>", self._on_select)

        row = ttk.Frame(listbody)
        row.pack(fill="x")
        ttk.Button(row, text="Abrir…", command=self._open_files).pack(
            side="left", padx=(0, PAD["xs"]))
        ttk.Button(row, text="Demo TMD", command=self._load_demo).pack(side="left")

        self._build_preprocess_card(parent)
        self._build_options_card(parent)

        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=(PAD["sm"], 0))
        ttk.Button(actions, text="Analizar como TMD", style="Accent.TButton",
                   command=self._analyse).pack(fill="x", pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Analizar todos",
                   command=self._analyse_all).pack(fill="x", pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Exportar tabla…",
                   command=self._export_table).pack(fill="x", pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Guardar informe…",
                   command=self._save_report).pack(fill="x")

    def _build_preprocess_card(self, parent) -> None:
        """The same preprocessing controls the carbon section has.

        They edit the *shared* settings object, so a baseline chosen here is
        the baseline the carbon section uses and the other way round. That
        is deliberate: the alternative is two spectra from one file
        processed two different ways, and a difference between the sections
        that nobody can explain.
        """
        ttk, tk = self.ttk, self.tk
        outer, body = card(parent, "Preprocesado",
                           "Compartido con la sección de carbono.")
        outer.pack(fill="x", pady=(PAD["sm"], 0))

        ttk.Button(body, text="Elegir parámetros automáticamente",
                   command=self._auto_preprocess).pack(fill="x", pady=(0, PAD["xs"]))

        self.despike_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="Eliminar rayos cósmicos",
                        variable=self.despike_var).pack(anchor="w")

        self.smooth_var = tk.IntVar(value=0)
        labelled(body, "Suavizado (pts)",
                 lambda p: ttk.Spinbox(p, from_=0, to=51, increment=2,
                                       textvariable=self.smooth_var, width=8))
        self.baseline_var = tk.StringVar(value=BASELINE_METHODS[0][1])
        labelled(body, "Línea base",
                 lambda p: ttk.Combobox(p, textvariable=self.baseline_var,
                                        width=24, state="readonly",
                                        values=[label for _, label in BASELINE_METHODS]))
        self.lam_var = tk.StringVar(value="1e7")
        labelled(body, "Rigidez (λ)",
                 lambda p: ttk.Entry(p, textvariable=self.lam_var, width=10))
        self.normalise_var = tk.StringVar(value=NORMALISATIONS[0][1])
        labelled(body, "Normalizar",
                 lambda p: ttk.Combobox(p, textvariable=self.normalise_var,
                                        width=24, state="readonly",
                                        values=[label for _, label in NORMALISATIONS]))
        hint(body,
             "No suavices para contar capas. Las bandas de un dicalcogenuro "
             "miden 2–6 cm⁻¹ y las fronteras entre números de capa están a "
             "2–3 cm⁻¹: suavizar ensancha las bandas, mueve el centro del "
             "ajuste y estropea justo la medida que has venido a hacer.",
             wrap=270)

    def _build_options_card(self, parent) -> None:
        ttk, tk = self.ttk, self.tk
        options, obody = card(parent, "Opciones de TMD")
        options.pack(fill="x", pady=(PAD["sm"], 0))

        self.material_var = tk.StringVar(value="(identificar)")
        labelled(obody, "Material",
                 lambda p: ttk.Combobox(p, textvariable=self.material_var,
                                        width=22, state="readonly",
                                        values=["(identificar)", *_material_keys()]))
        self.oxides_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(obody, text="Buscar óxidos del metal",
                        variable=self.oxides_var).pack(anchor="w")
        self.phases_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(obody, text="Buscar otras fases cristalinas",
                        variable=self.phases_var).pack(anchor="w")
        hint(obody,
             "Las capas se cuentan por la SEPARACIÓN E₂g–A₁g, no por "
             "posiciones absolutas: al ser una diferencia, cualquier error "
             "común de calibración se cancela. Solo funciona en los 2H de Mo "
             "y W, y de verdad solo en el MoS₂.   "
             "En WSe₂ los dos modos son casi degenerados; allí se cuenta por "
             "el modo B¹₂g. En TiS₂, TiSe₂, TaS₂ y WTe₂ no hay separación que "
             "contar, y en la pirita ni siquiera hay capas.   "
             "Para ver óxidos hay que medir hasta al menos 1050 cm⁻¹: sus "
             "líneas inequívocas (819 y 995 del MoO₃, 744 del MoO₂, 905 del "
             "SeO₂) están ahí arriba, y sin esa región «no hay óxido» no "
             "significa nada.",
             wrap=270)

    # -- tabs ----------------------------------------------------------
    def _build_tab_spectrum(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["sm"])
        self.notebook.add(tab, text="  Espectro  ")

        panes = ttk.Panedwindow(tab, orient="vertical")
        panes.pack(fill="both", expand=True)

        top = ttk.Frame(panes)
        panes.add(top, weight=3)
        plot_card, plot_body = card(top, None)
        plot_card.pack(fill="both", expand=True)
        self.make_canvas(plot_body, "tmd", lambda f: f.add_subplot(111))

        bottom = ttk.Frame(panes)
        panes.add(bottom, weight=2)
        outer, text_body = card(bottom, "Dicalcogenuro")
        outer.pack(fill="both", expand=True)
        self.tmd_text = scrolled_text(text_body, self.palette,
                                      self.fonts["mono"], height=12)

    def _build_tab_modes(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["sm"])
        self.notebook.add(tab, text="  Modos  ")

        panes = ttk.Panedwindow(tab, orient="vertical")
        panes.pack(fill="both", expand=True)

        top = ttk.Frame(panes)
        panes.add(top, weight=3)
        plot_card, plot_body = card(
            top, None,
            "Ajustado frente a catalogado. Una flecha larga es un "
            "desplazamiento real o una calibración mala; el informe dice cuál.")
        plot_card.pack(fill="both", expand=True)
        self.make_canvas(plot_body, "modes", lambda f: f.add_subplot(111))

        bottom = ttk.Frame(panes)
        panes.add(bottom, weight=2)
        outer, body = card(bottom, "Modos ajustados")
        outer.pack(fill="both", expand=True)
        self.modes_table = table(
            body,
            ["modo", "descripción", "catálogo", "ajustado", "Δ", "FWHM", "altura"],
            height=8,
        )
        hint(body,
             "La FWHM es el indicador de calidad cristalina que de verdad "
             "sirve: 2–6 cm⁻¹ es un cristal limpio, y por encima de 12 tienes "
             "dominios pequeños, material policristalino o daño. Comprueba "
             "antes que tu paso de muestreo resuelve eso: con 3 cm⁻¹ de paso, "
             "una anchura de 4 no significa nada.",
             wrap=560)

    def _build_tab_oxides(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["sm"])
        self.notebook.add(tab, text="  Óxidos  ")

        panes = ttk.Panedwindow(tab, orient="vertical")
        panes.pack(fill="both", expand=True)

        top = ttk.Frame(panes)
        panes.add(top, weight=3)
        plot_card, plot_body = card(
            top, None,
            "Líneas de catálogo de los óxidos que este calcogenuro puede dar. "
            "Las de trazo continuo son exclusivas: sin una de ellas no hay "
            "identificación.")
        plot_card.pack(fill="both", expand=True)
        self.make_canvas(plot_body, "oxides", lambda f: f.add_subplot(111))

        bottom = ttk.Frame(panes)
        panes.add(bottom, weight=2)
        oxide_card, oxide_body = card(bottom, "Óxidos e intercara")
        oxide_card.pack(fill="both", expand=True)
        self.oxide_text = scrolled_text(oxide_body, self.palette,
                                        self.fonts["mono"], height=12)

    def _build_tab_phases(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["sm"])
        self.notebook.add(tab, text="  Otras fases  ")

        outer, body = card(
            tab, "Fases cristalinas en la misma muestra",
            "El mismo catálogo que usa la sección de carbono, sin restringir "
            "al metal del calcogenuro.")
        outer.pack(fill="both", expand=True)
        self.phases_text = scrolled_text(body, self.palette,
                                         self.fonts["mono"], height=22)

    def _build_tab_batch(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["sm"])
        self.notebook.add(tab, text="  Lote  ")

        outer, body = card(
            tab, "Resultados",
            "Una fila por espectro analizado. «Exportar tabla…» la guarda "
            "en CSV tal cual se ve.")
        outer.pack(fill="both", expand=True)
        self.tmd_table = table(body, ["nombre"], height=16)
        hint(body,
             "El índice de oxidación es un COCIENTE DE INTENSIDADES, no una "
             "fracción en masa. La banda de 819 cm⁻¹ del α-MoO₃ es una de las "
             "líneas Raman más intensas de la química inorgánica y los modos "
             "de un dicalcogenuro fuera de resonancia no lo son: comparar la "
             "columna entre muestras medidas igual tiene sentido, convertirla "
             "en «porcentaje de óxido» no.",
             wrap=760)

    def _build_tab_library(self) -> None:
        ttk, tk = self.ttk, self.tk
        tab = ttk.Frame(self.notebook, padding=PAD["sm"])
        self.notebook.add(tab, text="  Biblioteca  ")

        panes = ttk.Panedwindow(tab, orient="horizontal")
        panes.pack(fill="both", expand=True)

        left = ttk.Frame(panes)
        panes.add(left, weight=1)
        outer, lbody = card(left, "Catálogo",
                            "Calcogenuros, óxidos y lo que falta a propósito.")
        outer.pack(fill="both", expand=True)
        self.library_list = tk.Listbox(
            lbody, exportselection=False, activestyle="none",
            background=self.palette.surface_alt, foreground=self.palette.text,
            selectbackground=self.palette.accent,
            selectforeground=self.palette.accent_text,
            highlightthickness=0, borderwidth=0, font=self.fonts["body"],
        )
        self.library_list.pack(fill="both", expand=True)
        self.library_list.bind("<<ListboxSelect>>", self._on_library_select)

        right = ttk.Frame(panes)
        panes.add(right, weight=2)
        detail, dbody = card(right, "Detalle y fuente")
        detail.pack(fill="both", expand=True)
        self.library_text = scrolled_text(dbody, self.palette,
                                          self.fonts["mono"], height=22)

    # ==================================================================
    # settings
    # ==================================================================
    def _settings_from_widgets(self) -> None:
        settings = self.session.preprocess_settings
        settings.despike = bool(self.despike_var.get())
        settings.smooth_window = int(self.smooth_var.get() or 0)
        settings.baseline_method = key_for_label(
            BASELINE_METHODS, self.baseline_var.get())
        settings.normalise = key_for_label(NORMALISATIONS, self.normalise_var.get())
        try:
            settings.baseline_lam = float(self.lam_var.get())
        except ValueError:
            settings.baseline_lam = 1e7

    def _widgets_from_settings(self) -> None:
        settings = self.session.preprocess_settings
        self.despike_var.set(settings.despike)
        self.smooth_var.set(settings.smooth_window)
        self.baseline_var.set(label_for_key(BASELINE_METHODS, settings.baseline_method))
        self.normalise_var.set(label_for_key(NORMALISATIONS, settings.normalise))
        self.lam_var.set(f"{settings.baseline_lam:g}")

    def _auto_preprocess(self) -> None:
        item = self.session.active
        if item is None:
            self.warn("Sin espectro", "Carga y selecciona un espectro primero.")
            return
        self._settings_from_widgets()
        reasons = self.session.preprocess_settings.apply_auto(item.raw)
        self._widgets_from_settings()
        self.set_status(
            "Parámetros elegidos del propio espectro: "
            + "; ".join(reasons.values())
            if reasons else "Parámetros elegidos del propio espectro."
        )

    # ==================================================================
    # events
    # ==================================================================
    def refresh(self) -> None:
        """Re-read the shared session; called when the section is shown."""
        self.spectrum_list.delete(0, "end")
        for item in self.session.spectra:
            mark = "✓ " if item.tmd_result is not None else "  "
            self.spectrum_list.insert("end", mark + item.raw.name)
        if 0 <= self.session.current < len(self.session.spectra):
            self.spectrum_list.selection_clear(0, "end")
            self.spectrum_list.selection_set(self.session.current)
        self._widgets_from_settings()
        self._redraw()

    def _on_select(self, _event=None) -> None:
        selection = self.spectrum_list.curselection()
        if not selection:
            return
        self.session.current = int(selection[0])
        self._redraw()

    def _on_tab_changed(self, _event=None) -> None:
        self.flush_dirty(self._visible())

    def _visible(self) -> tuple[str, ...]:
        try:
            index = self.notebook.index(self.notebook.select())
        except Exception:  # noqa: BLE001 - no tab selected yet
            return ()
        return self._tab_canvases.get(index, ())

    def _open_files(self) -> None:
        from tkinter import filedialog

        paths = filedialog.askopenfilenames(
            title="Abrir espectros",
            filetypes=(("Espectros", "*.txt *.csv *.dat *.asc"), ("Todos", "*.*")),
            parent=self.root,
        )
        if not paths:
            return
        added = self.session.load([Path(p) for p in paths])
        self.flush_messages(self.session.messages)
        self.refresh()
        self.set_status(f"{added} espectro(s) cargados.")

    def _load_demo(self) -> None:
        from ..examples.demo_data import tmd_demo_spectra
        from .state import LoadedSpectrum

        for spectrum in tmd_demo_spectra(seed=1):
            self.session.spectra.append(LoadedSpectrum(raw=spectrum))
        if self.session.current < 0:
            self.session.current = 0
        self.refresh()
        self.set_status(
            "Espectros TMD de demostración cargados, incluidas ocho "
            "heteroestructuras óxido/calcogenuro. Son SINTÉTICOS."
        )

    def _analyse(self) -> None:
        if self.session.active is None:
            self.warn("Sin espectro", "Carga y selecciona un espectro primero.")
            return
        self._settings_from_widgets()
        choice = self.material_var.get()
        material = None if choice.startswith("(") else choice
        want_phases = bool(self.phases_var.get())

        def work():
            result = self.session.analyse_tmd_active(material)
            if want_phases:
                self.session.find_tmd_phases_active()
            return result

        def done(result) -> None:
            self.flush_messages(self.session.messages)
            self.refresh()
            if result is not None:
                self.set_status(
                    f"{result.label} — {result.layers or 'capas indeterminadas'}, "
                    f"fase {result.phase}"
                )

        self.run_async(work, done, "Analizando como TMD…")

    def _analyse_all(self) -> None:
        self._settings_from_widgets()
        choice = self.material_var.get()
        material = None if choice.startswith("(") else choice
        want_phases = bool(self.phases_var.get())

        def work():
            saved = self.session.current
            count = 0
            for index in range(len(self.session.spectra)):
                self.session.current = index
                if self.session.analyse_tmd_active(material) is not None:
                    count += 1
                    if want_phases:
                        self.session.find_tmd_phases_active()
            self.session.current = saved
            return count

        def done(count) -> None:
            self.flush_messages(self.session.messages)
            self.refresh()
            self.set_status(f"{count} espectro(s) analizados como TMD.")

        self.run_async(work, done, "Analizando el lote como TMD…")

    def _export_table(self) -> None:
        from tkinter import filedialog

        columns, rows = self.session.tmd_table()
        if not columns:
            self.warn("Sin resultados", "Analiza al menos un espectro primero.")
            return
        path = filedialog.asksaveasfilename(
            title="Exportar tabla", defaultextension=".csv",
            initialfile="tmd.csv", parent=self.root,
        )
        if not path:
            return
        import csv

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        self.set_status(f"Tabla exportada a {path}")

    def _save_report(self) -> None:
        from tkinter import filedialog

        item = self.session.active
        if item is None or item.tmd_result is None:
            self.warn("Sin análisis", "Analiza un espectro primero.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar informe", defaultextension=".txt",
            initialfile=f"{item.raw.name}_tmd.txt", parent=self.root,
        )
        if not path:
            return
        text = item.tmd_result.summary()
        if item.tmd_phases is not None:
            text += "\n\n" + "=" * 68 + "\nOTRAS FASES CRISTALINAS\n" + "=" * 68
            text += "\n" + item.tmd_phases.summary()
        Path(path).write_text(text, encoding="utf-8")
        self.set_status(f"Informe guardado en {path}")

    # ==================================================================
    # library
    # ==================================================================
    def _fill_library(self) -> None:
        from ..analysis.heterostructure import load_oxides
        from ..analysis.tmd import load_tmd_database, tmd_materials

        payload, _ = load_tmd_database()
        self._library: list[tuple[str, object]] = []
        self.library_list.delete(0, "end")

        self.library_list.insert("end", "— CALCOGENUROS —")
        self._library.append(("header", None))
        for material in tmd_materials():
            self.library_list.insert("end", f"  {material.key}")
            self._library.append(("material", material))

        self.library_list.insert("end", "— ÓXIDOS —")
        self._library.append(("header", None))
        for oxide in load_oxides():
            self.library_list.insert("end", f"  {oxide.key}")
            self._library.append(("oxide", oxide))

        self.library_list.insert("end", "— AUSENTES A PROPÓSITO —")
        self._library.append(("header", None))
        for entry in payload.get("_missing", {}).get("entries", []):
            self.library_list.insert("end", f"  {entry['formula']}")
            self._library.append(("missing", entry))

        set_text(
            self.library_text,
            "Elige una entrada de la izquierda.\n\n"
            "La biblioteca cubre los calcogenuros de S, Se y Te de Mo, W, Ti, "
            "Nb, Ta y Fe, con sus óxidos. No cubre todo lo que existe, y la "
            "lista de AUSENTES dice qué falta y por qué: para entrar hace "
            "falta una fuente citable, porque una posición inventada no "
            "avisa — identifica mal, y con seguridad aparente.",
        )

    def _on_library_select(self, _event=None) -> None:
        selection = self.library_list.curselection()
        if not selection:
            return
        kind, entry = self._library[int(selection[0])]
        if kind == "header":
            return
        set_text(self.library_text, _describe(kind, entry))

    # ==================================================================
    # drawing
    # ==================================================================
    def _redraw(self) -> None:
        self.mark_dirty(*self._drawers)
        self.flush_dirty(self._visible())
        self._fill_texts()

    def _fill_texts(self) -> None:
        item = self.session.active
        result = item.tmd_result if item else None
        set_text(
            self.tmd_text,
            result.summary() if result else
            "Carga un espectro de dicalcogenuro y pulsa «Analizar como TMD».\n\n"
            "El espectro debe cubrir al menos 100–500 cm⁻¹, que es donde están "
            "los modos, y conviene un paso de muestreo fino: estas bandas son "
            "mucho más estrechas que las del carbono.\n\n"
            "Para el 1T-TaS₂ hace falta llegar a 40 cm⁻¹, porque lo que lo "
            "separa del 2H son sus modos de onda de densidad de carga a 63 y "
            "75 cm⁻¹. Si tu filtro no baja de 100, el programa lo identificará "
            "igual y te dirá que se está apoyando en bandas compartidas.",
        )
        oxides = result.oxides if result is not None else None
        set_text(
            self.oxide_text,
            oxides.summary() if oxides is not None else
            "Aquí aparecen los óxidos del metal que acompañen al calcogenuro "
            "(MoO₃, MoO₂, WO₃, TiO₂, Ta₂O₅, Nb₂O₅, Fe₂O₃, SeO₂…), el índice "
            "de oxidación y lo que el desplazamiento de los modos dice sobre "
            "la intercara.\n\n"
            "Raman NO ve topología: un MoO₃@MoSe₂ y una mezcla de polvos dan "
            "el mismo espectro puntual. Lo que sí se mide es la deformación "
            "del calcogenuro, que una intercara íntima produce y una mezcla no.",
        )
        phases = item.tmd_phases if item else None
        set_text(
            self.phases_text,
            phases.summary() if phases is not None else
            "Marca «Buscar otras fases cristalinas» y analiza.\n\n"
            "Esta búsqueda NO se limita al metal del calcogenuro, y por eso "
            "encuentra lo que la de óxidos no puede: selenio elemental a 237 "
            "cm⁻¹ y azufre a 473, que son precursor sin reaccionar y por tanto "
            "calcógeno que NO está en la red; óxidos del catalizador; el "
            "soporte de titania; sales del lavado.\n\n"
            "Un análisis elemental que dé la estequiometría correcta no "
            "distingue el selenio de la red del selenio segregado. Esto sí.",
        )
        self._fill_modes_table()
        columns, rows = self.session.tmd_table()
        if columns:
            fill_table(self.tmd_table, columns, rows)

    def _fill_modes_table(self) -> None:
        columns = ["modo", "descripción", "catálogo", "ajustado", "Δ", "FWHM", "altura"]
        item = self.session.active
        result = item.tmd_result if item else None
        if result is None or not result.positions:
            fill_table(self.modes_table, columns, [])
            return
        reference = _reference_modes(result.material)
        rows = []
        for key, position in sorted(result.positions.items(), key=lambda kv: kv[1]):
            mode = reference.get(key)
            catalogue = f"{mode.position:.1f}" if mode else "—"
            delta = f"{position - mode.position:+.1f}" if mode else "—"
            width = result.widths.get(key)
            height = ""
            if result.fit is not None:
                peak = next((p for p in result.fit.peaks if p.name == key), None)
                if peak is not None:
                    height = f"{peak.height:.0f}"
            rows.append([
                key,
                mode.label if mode else "",
                catalogue,
                f"{position:.2f}",
                delta,
                f"{width:.2f}" if width else "—",
                height,
            ])
        fill_table(self.modes_table, columns, rows)

    def _draw_tmd(self, figure) -> None:
        from .plots import plot_fit, plot_spectrum

        ax = figure.add_subplot(111)
        item = self.session.active
        if item is None:
            placeholder(ax, "Carga un espectro", self.palette)
            return
        result = item.tmd_result
        if result is None or result.fit is None:
            plot_spectrum(ax, item.raw, self.palette)
            ax.set_title(item.raw.name, fontsize=9)
            return
        plot_fit(ax, result.fit, self.palette)
        ax.set_title(
            f"{result.label} — {result.layers or '?'} capa(s), fase {result.phase}",
            fontsize=9,
        )

    def _draw_modes(self, figure) -> None:
        """Fitted positions against catalogued ones, on one axis.

        A lollipop rather than a spectrum: what matters here is the size and
        the SIGN of each displacement, and reading those off two overlaid
        traces is exactly the thing eyes are bad at.
        """
        ax = figure.add_subplot(111)
        item = self.session.active
        result = item.tmd_result if item else None
        if result is None or not result.positions:
            placeholder(ax, "Analiza un espectro para ver sus modos", self.palette)
            return
        reference = _reference_modes(result.material)
        keys = sorted(result.positions, key=lambda k: result.positions[k])
        for index, key in enumerate(keys):
            observed = result.positions[key]
            mode = reference.get(key)
            if mode is None:
                ax.plot([observed], [index], "o", color=self.palette.accent)
                continue
            ax.plot([mode.position, observed], [index, index], "-",
                    color=self.palette.text_muted, linewidth=1.2, zorder=1)
            ax.plot([mode.position], [index], "|", markersize=14,
                    color=self.palette.text_muted, zorder=2)
            ax.plot([observed], [index], "o", color=self.palette.accent, zorder=3)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys, fontsize=8)
        ax.set_xlabel("Desplazamiento Raman (cm⁻¹)")
        ax.set_title(
            f"{result.label}: barra = catálogo, punto = ajustado", fontsize=9)
        ax.margins(x=0.12, y=0.18)

    def _draw_oxides(self, figure) -> None:
        """The spectrum with the catalogued lines of the possible oxides.

        Only the oxides this chalcogenide can chemically produce are drawn.
        Showing every oxide in the library would find tungsten oxides in
        molybdenum samples by coincidence, which is the thing the analysis
        itself refuses to do.
        """
        from .plots import plot_spectrum

        ax = figure.add_subplot(111)
        item = self.session.active
        if item is None:
            placeholder(ax, "Carga un espectro", self.palette)
            return
        plot_spectrum(ax, item.display, self.palette, label_peaks=False)
        result = item.tmd_result
        if result is None or result.material is None:
            ax.set_title("Analiza para ver los óxidos posibles", fontsize=9)
            return
        from ..analysis.heterostructure import expected_oxides

        found = {o.key for o in (result.oxides.oxides_present if result.oxides else [])}
        colours = [self.palette.accent, self.palette.warning, self.palette.text_muted,
                   self.palette.text]
        low, high = item.display.range
        for index, oxide in enumerate(expected_oxides(result.material)):
            colour = colours[index % len(colours)]
            drawn = False
            for line in oxide.bands:
                if not low <= line <= high:
                    continue
                exclusive = any(abs(line - s) < 1e-6 for s in oxide.signature)
                ax.axvline(line, color=colour, alpha=0.9 if exclusive else 0.35,
                           linewidth=1.4 if exclusive else 0.8,
                           linestyle="-" if exclusive else ":",
                           label=oxide.formula if not drawn else None, zorder=0)
                drawn = True
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, [
                f"{name} ✓" if name in {o.formula for o in
                                        (result.oxides.oxides_present
                                         if result.oxides else [])} else name
                for name in labels
            ], fontsize=7, loc="upper right")
        ax.set_title(
            "Óxidos posibles de " + result.label
            + (f" — presentes: {', '.join(sorted(found))}" if found
               else " — ninguno corroborado"),
            fontsize=9,
        )


# ==================================================================
# helpers
# ==================================================================
def _material_keys() -> list[str]:
    from ..analysis.tmd import tmd_materials

    return [m.key for m in tmd_materials()]


def _reference_modes(key):
    from ..analysis.tmd import tmd_materials

    material = next((m for m in tmd_materials() if m.key == key), None)
    return dict(material.modes) if material else {}


def _describe(kind: str, entry) -> str:
    """One catalogue entry, written out with its source and its caveats."""
    if kind == "missing":
        return (
            f"{entry['formula']}\n\n"
            "NO está en la biblioteca.\n\n"
            f"{entry['reason']}\n\n"
            "Para entrar hace falta una fuente citable. Una posición "
            "inventada no avisa: identifica mal, y con seguridad aparente."
        )
    lines = [entry.label, "=" * len(entry.label), ""]
    if kind == "material":
        lines += [
            f"Fórmula      : {entry.formula}",
            f"Metal        : {entry.metal}     Calcógeno: {entry.chalcogen}",
            f"Politipo     : {entry.polytype or '—'}",
            f"Laminar      : {'sí' if entry.layered else 'NO'}",
            f"Confianza    : {entry.confidence}",
            "",
            "Modos catalogados:",
        ]
        for key, mode in sorted(entry.modes.items(), key=lambda kv: kv[1].position):
            mark = " *" if mode.discriminating else "  "
            lines.append(
                f"{mark} {key:<10s} {mode.position:7.1f} cm⁻¹  "
                f"[{mode.window[0]:.0f}–{mode.window[1]:.0f}]  {mode.label}"
            )
        lines += [
            "",
            "* modo discriminante: separa este material de sus vecinos del",
            "  catálogo. La identificación exige al menos uno.",
        ]
        if entry.counts_layers_by_separation:
            lines += ["", "Separación E₂g–A₁g por número de capas:"]
            for name, (low, high) in entry.separation_by_layers.items():
                lines.append(f"    {name:<6s} {low:5.1f} – {high:5.1f} cm⁻¹")
            lines.append(f"    fuente: {entry.separation_source}")
        else:
            lines += [
                "",
                "No cuenta capas por separación: o los dos modos están casi",
                "degenerados, o no son un par E₂g/A₁g, o el material no es",
                "laminar.",
            ]
    else:
        lines += [
            f"Fórmula      : {entry.formula}",
            f"Metal        : {entry.metal}",
            f"Sistema      : {entry.crystal_system}"
            + (f"   ({entry.space_group})" if entry.space_group else ""),
            f"Aspecto      : {entry.colour}",
            f"Confianza    : {entry.confidence}",
            "",
            "Bandas (cm⁻¹): " + ", ".join(f"{b:g}" for b in entry.bands),
            "Fuertes      : " + ", ".join(f"{b:g}" for b in entry.strong),
            "Exclusivas   : "
            + (", ".join(f"{b:g}" for b in entry.signature) or "ninguna catalogada"),
        ]
        if entry.min_fwhm is not None:
            lines.append(
                f"Anchura mín. : {entry.min_fwhm:g} cm⁻¹ (fase amorfa; es un "
                "límite INFERIOR)"
            )
    lines += ["", "Fuente:", f"  {entry.source}", "", "Notas:", ""]
    lines.append(entry.notes)
    return "\n".join(lines)


__all__ = ["TMDApp"]
