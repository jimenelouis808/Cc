"""The XPS section: survey, high-resolution regions and composition.

One session is one sample, and the tabs are the steps in the order they
have to happen. That order is not a matter of taste:

1. **Reference the axis.** Every binding energy below depends on it, and
   the reference has to be measured once and applied to all the spectra —
   a per-region reference is fitting the charging separately to each
   region, which is how a chemical shift gets invented.
2. **Identify.** The survey says what is there, and its unexplained peaks
   say what is there that nobody went looking for.
3. **Fit.** One region at a time, with the number of components chosen by
   the person and the windows taken from the literature. The section shows
   the residual and χ² beside the areas because a component count is a
   claim about chemistry and those two are the only things that can argue
   with it.
4. **Quantify.** Last, because it needs all of the above to be right.

The region tab is where the work happens, and it is built so that the
choice — how many components, which chemical states, what background — is
visible, adjustable and recorded, rather than being a default nobody sees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import SectionApp, placeholder
from .theme import PAD
from .widgets import card, fill_table, hint, labelled, scrolled_text, set_text, table, scrollable_column
from .xps_state import BACKGROUNDS, REFERENCES, XPSSession

FILE_TYPES = (
    ("Espectros XPS", "*.spe *.vms *.npl *.txt *.csv *.dat *.asc"),
    ("PHI MultiPak", "*.spe"),
    ("VAMAS ISO 14976", "*.vms *.npl"),
    ("Todos", "*.*"),
)


class XPSApp(SectionApp):
    """The photoelectron-spectroscopy section of the suite."""

    def __init__(self, root, container, palette, fonts) -> None:
        super().__init__(root, container, palette, fonts)
        self.session = XPSSession()
        self._comparison: list = []
        self._drawers = {
            "survey": self._draw_survey,
            "region": self._draw_region,
            "counts": self._draw_counts,
            "composition": self._draw_composition,
        }
        self._tab_canvases = {
            0: (),
            1: ("survey",),
            2: ("region", "counts"),
            3: ("composition",),
        }
        self._build()
        self.set_status("Carga espectros o pulsa Demo.")

    # ==================================================================
    # layout
    # ==================================================================
    def _build(self) -> None:
        ttk = self.ttk
        body = ttk.Frame(self.container, padding=(PAD["md"], PAD["sm"]))
        body.pack(fill="both", expand=True)

        # Scrollable: this column's cards add up to more than any
        # window is tall, and a fixed frame simply clips them.
        sidebar_column, sidebar = scrollable_column(body, width=320)
        sidebar_column.pack(side="left", fill="y", padx=(0, PAD["md"]))
        self._build_sidebar(sidebar)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side="left", fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._build_tab_summary()
        self._build_tab_survey()
        self._build_tab_region()
        self._build_tab_composition()
        self.build_status(self.container)
        self.root.after(150, self.drain_queue)

    def _build_sidebar(self, parent) -> None:
        ttk, tk = self.ttk, self.tk

        sample, sbody = card(parent, "Muestra y equipo")
        sample.pack(fill="x")
        self.name_var = tk.StringVar(value="muestra")
        labelled(sbody, "Nombre", lambda p: ttk.Entry(
            p, textvariable=self.name_var, width=14))
        self.source_var = tk.StringVar(value="")
        labelled(sbody, "Ánodo", lambda p: ttk.Combobox(
            p, textvariable=self.source_var, width=12,
            values=["", "Al", "Mg", "Ag La"]))
        self.pass_var = tk.StringVar(value="")
        labelled(sbody, "Energía de paso (eV)", lambda p: ttk.Entry(
            p, textvariable=self.pass_var, width=10))
        hint(sbody,
             "El ánodo hace falta para situar las líneas Auger: suponer "
             "aluminio cuando era magnesio las mueve 233 eV. La energía de "
             "paso fija el suelo de resolución, y sin ella ninguna anchura "
             "ajustada se puede juzgar.",
             wrap=280)

        reference, rbody = card(parent, "Referencia de carga")
        reference.pack(fill="x", pady=(PAD["sm"], 0))
        self.reference_var = tk.StringVar(value=REFERENCES[0][1])
        labelled(rbody, "Referencia", lambda p: ttk.Combobox(
            p, textvariable=self.reference_var, width=18, state="readonly",
            values=[label for _, label in REFERENCES]))
        self.reference_state_var = tk.StringVar(value="")
        labelled(rbody, "…o componente", lambda p: ttk.Entry(
            p, textvariable=self.reference_state_var, width=14))
        hint(rbody,
             "En una muestra HECHA de carbono, referenciar contra «el pico "
             "C 1s» es circular y la envolvente arrastra el centro medio eV. "
             "Escribe «C 1s:C-C sp2» para referenciar sobre una componente "
             "ajustada.",
             wrap=280)

        loader, lbody = card(parent, "Espectros")
        loader.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self.loaded_list = tk.Listbox(
            lbody, height=8, exportselection=False, activestyle="none",
            background=self.palette.surface_alt, foreground=self.palette.text,
            selectbackground=self.palette.accent,
            selectforeground=self.palette.accent_text,
            highlightthickness=0, borderwidth=0, font=self.fonts["small"],
        )
        self.loaded_list.pack(fill="both", expand=True, pady=(0, PAD["sm"]))
        for text, command in (
            ("Cargar espectros…", self._load),
            ("Importar modelo (tabla)…", self._import_model),
            ("Vaciar", self._clear),
        ):
            ttk.Button(lbody, text=text, command=command).pack(
                fill="x", pady=(PAD["xs"], 0))

        options, obody = card(parent, "Cuantificación")
        options.pack(fill="x", pady=(PAD["sm"], 0))
        self.transmission_var = tk.StringVar(value="potencia")
        labelled(obody, "Transmisión", lambda p: ttk.Combobox(
            p, textvariable=self.transmission_var, width=12, state="readonly",
            values=["potencia", "ninguna", "phi"]))
        self.exponent_var = tk.StringVar(value="-0.65")
        labelled(obody, "Exponente", lambda p: ttk.Entry(
            p, textvariable=self.exponent_var, width=10))
        hint(obody,
             "T ∝ KE^n es una propiedad de TU analizador. Entre un C 1s y un "
             "Fe 2p con aluminio, ignorarla es un 30 %.",
             wrap=280)

        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=(PAD["sm"], 0))
        ttk.Button(actions, text="Analizar todo", style="Accent.TButton",
                   command=self._analyse).pack(fill="x", pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Demo", command=self._load_demo).pack(
            fill="x", pady=(0, PAD["xs"]))
        for text, command in (
            ("Guardar informe…", self._save_report),
            ("Exportar tablas…", self._export_tables),
            ("Exportar VAMAS…", self._export_vamas),
        ):
            ttk.Button(actions, text=text, command=command).pack(
                fill="x", pady=(0, PAD["xs"]))

    def _build_tab_summary(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Resumen  ")
        outer, body = card(tab, "Espectros cargados")
        outer.pack(fill="x")
        self.spectra_table = table(body, ["espectro", "región", "detalle"],
                                   height=7)
        report, report_body = card(tab, "Informe")
        report.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self.report_text = scrolled_text(report_body, self.palette,
                                         self.fonts["mono"], height=24)

    def _build_tab_survey(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Survey  ")
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", pady=(0, PAD["sm"]))
        ttk.Button(toolbar, text="Identificar elementos",
                   command=self._run_survey).pack(side="left")
        hint(toolbar,
             "  Un elemento tiene que enseñar su línea principal, la "
             "componente fuerte de su doblete, y al menos un pico que no "
             "explique nadie más. Sin esa última regla, todo hierro «contiene» "
             "cobalto: su Auger LMM cae sobre el Co 2p con ánodo de aluminio.",
             wrap=720)
        outer, body = card(tab, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "survey", lambda f: f.add_subplot(111),
                         figsize=(8.2, 3.6))
        info, info_body = card(tab, "Elementos")
        info.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self.survey_table = table(info_body,
                                  ["elemento", "confianza", "líneas"], height=9)

    def _build_tab_region(self) -> None:
        ttk, tk = self.ttk, self.tk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Regiones  ")

        controls, cbody = card(tab, "Qué ajustar")
        controls.pack(fill="x")
        row = ttk.Frame(cbody)
        row.pack(fill="x")
        self.region_var = tk.StringVar(value="")
        self.region_box = ttk.Combobox(row, textvariable=self.region_var,
                                       width=14, state="readonly", values=[])
        self.region_box.pack(side="left")
        self.region_box.bind("<<ComboboxSelected>>", self._on_region_changed)
        self.count_var = tk.StringVar(value="3")
        ttk.Label(row, text="  nº de componentes ").pack(side="left")
        ttk.Spinbox(row, from_=1, to=8, width=4,
                    textvariable=self.count_var).pack(side="left")
        self.background_var = tk.StringVar(value=BACKGROUNDS[0][1])
        ttk.Label(row, text="   fondo ").pack(side="left")
        ttk.Combobox(row, textvariable=self.background_var, width=22,
                     state="readonly",
                     values=[label for _, label in BACKGROUNDS]).pack(side="left")
        self.link_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text=" ligar anchuras",
                        variable=self.link_var).pack(side="left", padx=(PAD["sm"], 0))
        self.satellite_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text=" satélites",
                        variable=self.satellite_var).pack(side="left")

        picker = ttk.Frame(cbody)
        picker.pack(fill="x", pady=(PAD["sm"], 0))
        ttk.Label(picker, text="Estados químicos (ninguno = que los elija la "
                               "propia región):").pack(anchor="w")
        self.state_list = tk.Listbox(
            picker, height=5, selectmode="extended", exportselection=False,
            activestyle="none", background=self.palette.surface_alt,
            foreground=self.palette.text, selectbackground=self.palette.accent,
            selectforeground=self.palette.accent_text, highlightthickness=0,
            borderwidth=0, font=self.fonts["small"],
        )
        self.state_list.pack(fill="x", pady=(PAD["xs"], 0))
        buttons = ttk.Frame(cbody)
        buttons.pack(fill="x", pady=(PAD["sm"], 0))
        ttk.Button(buttons, text="Ajustar región", style="Accent.TButton",
                   command=self._fit_region).pack(side="left")
        ttk.Button(buttons, text="Ajustar todas",
                   command=self._fit_all).pack(side="left", padx=(PAD["xs"], 0))
        ttk.Button(buttons, text="¿Cuántas componentes?",
                   command=self._compare_counts).pack(side="left",
                                                      padx=(PAD["xs"], 0))
        hint(cbody,
             "Un ajuste por mínimos cuadrados SIEMPRE devuelve tantos picos "
             "como se le den, y añadir uno siempre baja el residuo. Lo que "
             "decide es el χ² frente al ruido de conteo y si el residuo aún "
             "tiene estructura (Durbin-Watson lejos de 2).",
             wrap=820)

        outer, body = card(tab, None)
        outer.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self.make_canvas(body, "region", lambda f: f.add_subplot(111),
                         figsize=(8.2, 4.4))
        bottom = ttk.Frame(tab)
        bottom.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        left, left_body = card(bottom, "Componentes")
        left.pack(side="left", fill="both", expand=True)
        self.components_table = table(
            left_body,
            ["componente", "E_enlace (eV)", "FWHM (eV)", "área", "%", "nota"],
            height=8,
        )
        right, right_body = card(bottom, "Número de componentes")
        right.pack(side="left", fill="both", expand=True,
                   padx=(PAD["sm"], 0))
        self.make_canvas(right_body, "counts", lambda f: f.add_subplot(111),
                         figsize=(4.2, 2.6))
        self.verdict_text = scrolled_text(right_body, self.palette,
                                          self.fonts["small"], height=4)

    def _build_tab_composition(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Composición  ")
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", pady=(0, PAD["sm"]))
        ttk.Button(toolbar, text="Cuantificar",
                   command=self._quantify).pack(side="left")
        hint(toolbar,
             "  Es un porcentaje de LO DETECTADO: el hidrógeno no se ve, y un "
             "elemento sin ajustar no baja el de los demás, su parte se "
             "reparte. Que sume 100 % es la normalización.",
             wrap=720)
        outer, body = card(tab, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "composition",
                         lambda f: (f.add_subplot(121), f.add_subplot(122)),
                         figsize=(8.2, 3.4))
        info, info_body = card(tab, "Composición")
        info.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self.composition_table = table(
            info_body,
            ["elemento", "línea", "% atómico", "área", "RSF", "transmisión"],
            height=8,
        )

    # ==================================================================
    # events
    # ==================================================================
    def _settings_from_widgets(self) -> None:
        self.session.name = self.name_var.get() or "muestra"
        labels = {label: key for key, label in REFERENCES}
        self.session.reference = labels.get(self.reference_var.get(),
                                            "C1s_adventitious")
        text = self.reference_state_var.get().strip()
        self.session.reference_state = None
        if text and ":" in text:
            region, _, key = text.partition(":")
            self.session.reference_state = (region.strip(), key.strip())
        self.session.transmission = self.transmission_var.get()
        exponent = _number(self.exponent_var.get())
        if exponent is not None:
            self.session.exponent = exponent

    def _choice_from_widgets(self) -> Optional[str]:
        label = self.region_var.get()
        if not label:
            return None
        choice = self.session.choice_for(label)
        backgrounds = {text: key for key, text in BACKGROUNDS}
        choice.background = backgrounds.get(self.background_var.get(), "shirley")
        choice.link_widths = bool(self.link_var.get())
        choice.satellites = bool(self.satellite_var.get())
        selected = [self._state_keys[index]
                    for index in self.state_list.curselection()]
        choice.states = selected or None
        count = _number(self.count_var.get())
        choice.count = int(count) if count else None
        return label

    def _load(self) -> None:
        from tkinter import filedialog

        paths = filedialog.askopenfilenames(
            title="Abrir espectros XPS", filetypes=FILE_TYPES, parent=self.root
        )
        if not paths:
            return
        self._settings_from_widgets()
        options = {}
        if self.source_var.get():
            from ..xps.spectrum import source_energy

            try:
                options["photon_energy"] = source_energy(self.source_var.get())
            except ValueError as error:
                self.warn("Ánodo", str(error))
        pass_energy = _number(self.pass_var.get())
        if pass_energy:
            options["pass_energy"] = pass_energy
        total = sum(self.session.load(path, **options) for path in paths)
        self.flush_messages(self.session.messages)
        self.set_status(f"{total} espectro(s) cargados")
        self._refresh_loaded()

    def _load_demo(self) -> None:
        from ..examples.demo_data import make_xps_demo

        self.name_var.set("demo NCNT@FeSe")
        self.source_var.set("Al")
        self.pass_var.set("26")
        self.reference_state_var.set("C 1s:C-C sp2")
        self._settings_from_widgets()
        self.session.clear()
        self.session.spectra.extend(make_xps_demo("NCNT_FeSe", seed=1,
                                                  charge_shift=1.5))
        self.session.shifted = list(self.session.spectra)
        self.session.choice_for("C 1s").states = ["C-C sp2", "C-O", "C=O",
                                                  "O-C=O"]
        self._refresh_loaded()
        self.set_status(
            "Espectros de demostración cargados. Son SINTÉTICOS: la muestra "
            "carga 1.5 eV y la referencia tiene que quitarlos."
        )

    def _clear(self) -> None:
        self.session.clear()
        self._refresh_loaded()
        self._redraw()
        self.set_status("Sesión vaciada.")

    def _refresh_loaded(self) -> None:
        self.loaded_list.delete(0, "end")
        rows = self.session.spectrum_rows()
        if not rows:
            self.loaded_list.insert("end", "(ningún espectro)")
        else:
            for name, kind, detail in rows:
                self.loaded_list.insert("end", f"{kind}: {name}")
        fill_table(self.spectra_table, ["espectro", "región", "detalle"],
                   [(name, kind, detail) for name, kind, detail in rows])
        labels = sorted({self.session.region_label(item)
                         for item in self.session.regions})
        self.region_box.configure(values=labels)
        if labels and self.region_var.get() not in labels:
            self.region_var.set(labels[0])
        self._on_region_changed()

    def _on_region_changed(self, _event=None) -> None:
        label = self.region_var.get()
        self.state_list.delete(0, "end")
        self._state_keys: list[str] = []
        if not label:
            return
        choice = self.session.choice_for(label)
        for key, name in self.session.available_states(label):
            self._state_keys.append(key)
            self.state_list.insert("end", f"{name}  ({key})")
        for position, key in enumerate(self._state_keys):
            if choice.states and key in choice.states:
                self.state_list.selection_set(position)
        if choice.count:
            self.count_var.set(str(choice.count))
        result = self.session.fits.get(label)
        if result is not None:
            self.mark_dirty("region")
            self._fill_components(result)
            self.flush_dirty(self._visible())

    def _import_model(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Importar tabla de componentes",
            filetypes=(("Tablas", "*.csv *.tsv *.txt"), ("Todos", "*.*")),
            parent=self.root,
        )
        if not path:
            return
        label = self.session.import_model(path)
        self.flush_messages(self.session.messages)
        if label:
            self.region_var.set(label)
            self._on_region_changed()
            self.set_status(f"Modelo importado para {label}")

    def _run_survey(self) -> None:
        self._settings_from_widgets()
        if not self.session.spectra:
            self.warn("Sin datos", "Carga espectros primero.")
            return

        def done(_result) -> None:
            self.flush_messages(self.session.messages)
            fill_table(self.survey_table, ["elemento", "confianza", "líneas"],
                       self.session.survey_rows())
            self.mark_dirty("survey")
            self.flush_dirty(self._visible())
            if self.session.survey:
                self.set_status(
                    "Elementos: " + ", ".join(self.session.survey.symbols())
                )

        def work():
            self.session.apply_reference()
            return self.session.run_survey()

        self.run_async(work, done, "Identificando…")

    def _fit_region(self) -> None:
        self._settings_from_widgets()
        label = self._choice_from_widgets()
        if label is None:
            self.warn("Sin región", "Carga espectros de región primero.")
            return
        if self.session.calibration is None and self.session.spectra:
            self.session.apply_reference()

        def done(result) -> None:
            self.flush_messages(self.session.messages)
            if result is None:
                self.set_status(f"{label}: el ajuste no ha salido.")
                return
            self._fill_components(result)
            self.mark_dirty("region")
            self.flush_dirty(self._visible())
            self.set_status(
                f"{label}: χ²_red = {result.reduced_chi2:.2f}, "
                f"DW = {result.durbin_watson:.2f}"
                if result.durbin_watson else f"{label}: ajustada"
            )

        self.run_async(lambda: self.session.fit(label), done, "Ajustando…")

    def _fit_all(self) -> None:
        self._settings_from_widgets()
        self._choice_from_widgets()

        def work():
            self.session.apply_reference()
            return self.session.fit_all()

        def done(count) -> None:
            self.flush_messages(self.session.messages)
            self._refresh_loaded()
            result = self.session.fits.get(self.region_var.get())
            if result is not None:
                self._fill_components(result)
            self.mark_dirty("region", "composition")
            self.flush_dirty(self._visible())
            self.set_status(f"{count} región/regiones ajustadas.")

        self.run_async(work, done, "Ajustando todas…")

    def _compare_counts(self) -> None:
        self._settings_from_widgets()
        label = self._choice_from_widgets()
        if label is None:
            return

        def done(payload) -> None:
            comparisons, verdict = payload
            self._comparison = list(comparisons)
            set_text(self.verdict_text, verdict)
            self.mark_dirty("counts")
            self.flush_dirty(self._visible())

        self.run_async(lambda: self.session.compare(label), done, "Comparando…")

    def _quantify(self) -> None:
        self._settings_from_widgets()

        def done(result) -> None:
            self.flush_messages(self.session.messages)
            if result is None:
                return
            fill_table(
                self.composition_table,
                ["elemento", "línea", "% atómico", "área", "RSF", "transmisión"],
                [(item.element, item.line, f"{item.atomic_percent:.2f}",
                  f"{item.area:.4g}", f"{item.rsf:.2f}",
                  f"{item.transmission:.3f}")
                 for item in result.abundances],
            )
            self.mark_dirty("composition")
            self.flush_dirty(self._visible())
            self.set_status("Composición calculada.")

        self.run_async(self.session.quantify, done, "Cuantificando…")

    def _analyse(self) -> None:
        self._settings_from_widgets()
        self._choice_from_widgets()
        if not self.session.spectra:
            self.warn("Sin datos", "Carga espectros o pulsa Demo.")
            return

        def work():
            self.session.apply_reference()
            self.session.run_survey()
            self.session.fit_all()
            self.session.quantify()
            return self.session.report()

        def done(text) -> None:
            self.flush_messages(self.session.messages)
            self._refresh_loaded()
            fill_table(self.survey_table, ["elemento", "confianza", "líneas"],
                       self.session.survey_rows())
            if self.session.composition:
                fill_table(
                    self.composition_table,
                    ["elemento", "línea", "% atómico", "área", "RSF",
                     "transmisión"],
                    [(item.element, item.line, f"{item.atomic_percent:.2f}",
                      f"{item.area:.4g}", f"{item.rsf:.2f}",
                      f"{item.transmission:.3f}")
                     for item in self.session.composition.abundances],
                )
            result = self.session.fits.get(self.region_var.get())
            if result is not None:
                self._fill_components(result)
            set_text(self.report_text, text or "")
            self._redraw()
            self.set_status("Análisis completo.")

        self.run_async(work, done, "Analizando…")

    def _fill_components(self, result) -> None:
        fill_table(
            self.components_table,
            ["componente", "E_enlace (eV)", "FWHM (eV)", "área", "%", "nota"],
            [(item.label, f"{item.centre:.2f}", f"{item.true_fwhm:.2f}",
              f"{item.area:.4g}", f"{100 * item.area_fraction:.1f}",
              ("satélite" if item.satellite else item.justification[:60]))
             for item in result.components],
        )

    def _save_report(self) -> None:
        from tkinter import filedialog

        if not self.session.fits and not self.session.survey:
            self.warn("Sin análisis", "Pulsa Analizar todo primero.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar informe", defaultextension=".txt",
            initialfile=f"{self.session.name}_xps.txt", parent=self.root,
        )
        if not path:
            return
        Path(path).write_text(self.session.report(), encoding="utf-8")
        self.set_status(f"Informe guardado en {path}")

    def _export_tables(self) -> None:
        from tkinter import filedialog

        if not self.session.fits:
            self.warn("Sin ajustes", "Ajusta al menos una región.")
            return
        directory = filedialog.askdirectory(
            title="Carpeta para las tablas", parent=self.root)
        if not directory:
            return
        written = self.session.export_tables(directory)
        self.set_status(f"{len(written)} tablas escritas en {directory}")

    def _export_vamas(self) -> None:
        from tkinter import filedialog

        if not self.session.spectra:
            self.warn("Sin datos", "Carga espectros primero.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar en VAMAS", defaultextension=".vms",
            initialfile=f"{self.session.name}.vms", parent=self.root,
        )
        if not path:
            return
        self.session.export_vamas(path)
        self.set_status(
            f"Espectros guardados en {path} con el eje ya referenciado"
        )

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

    def _draw_survey(self, figure) -> None:
        from .plots_xps import plot_survey

        ax = figure.add_subplot(111)
        surveys = self.session.surveys
        if not surveys:
            placeholder(ax, "Sin barrido ancho cargado", self.figure_palette)
            return
        plot_survey(ax, surveys[0], self.figure_palette, self.session.survey)

    def _draw_region(self, figure) -> None:
        from .plots_xps import plot_region

        result = self.session.fits.get(self.region_var.get())
        if result is None:
            placeholder(figure.add_subplot(111),
                        "Elige una región y pulsa Ajustar", self.figure_palette)
            return
        plot_region(figure, result, self.figure_palette)

    def _draw_counts(self, figure) -> None:
        from .plots_xps import plot_count_comparison

        ax = figure.add_subplot(111)
        if not self._comparison:
            placeholder(ax, "Pulsa «¿Cuántas componentes?»", self.figure_palette)
            return
        plot_count_comparison(ax, self._comparison, self.figure_palette)

    def _draw_composition(self, figure) -> None:
        from .plots_xps import plot_composition, plot_states

        left = figure.add_subplot(121)
        right = figure.add_subplot(122)
        if self.session.composition is None:
            placeholder(left, "Pulsa Cuantificar", self.figure_palette)
        else:
            plot_composition(left, self.session.composition, self.figure_palette)
        if not self.session.fits:
            placeholder(right, "Sin regiones ajustadas", self.figure_palette)
        else:
            plot_states(right, list(self.session.fits.values()), self.figure_palette)


def _number(text: str) -> Optional[float]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


__all__ = ["XPSApp"]
