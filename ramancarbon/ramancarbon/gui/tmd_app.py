"""The dichalcogenide section: layer count, phase, and oxide content.

Separate from the carbon section because the physics is separate — nobody
needs the program deciding between "multi-walled nanotube" and "bilayer
MoS₂", since the user knows what they put under the objective — but
sharing its :class:`~ramancarbon.gui.state.Session`, so a spectrum loaded
in either section is available in both. A file is a file.

The oxide panel is here rather than in a corner of the report because for
real samples it is often the answer: an MoSe₂ measured in air is usually
MoO₃ + MoSe₂, and which of the three ways the oxide got there decides
whether the measurement is usable at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import SectionApp, placeholder
from .theme import PAD
from .widgets import card, fill_table, hint, scrolled_text, set_text, table


class TMDApp(SectionApp):
    """The Raman-TMD section of the suite."""

    def __init__(self, root, container, palette, fonts, session) -> None:
        super().__init__(root, container, palette, fonts)
        self.session = session
        self._drawers = {"tmd": self._draw_tmd}
        self._build()
        self.set_status(
            "Carga un espectro de MoS₂, WS₂, MoSe₂, WSe₂ o MoTe₂ y pulsa "
            "«Analizar como TMD»."
        )

    # ==================================================================
    # layout
    # ==================================================================
    def _build(self) -> None:
        ttk = self.ttk
        body = ttk.Frame(self.container, padding=(PAD["md"], PAD["sm"]))
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, width=280)
        sidebar.pack(side="left", fill="y", padx=(0, PAD["md"]))
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        plot_card, plot_body = card(right, None)
        plot_card.pack(fill="both", expand=True)
        self.make_canvas(plot_body, "tmd", lambda f: f.add_subplot(111))

        panes = ttk.Panedwindow(right, orient="horizontal")
        panes.pack(fill="both", expand=True, pady=(PAD["sm"], 0))

        left_pane = ttk.Frame(panes)
        panes.add(left_pane, weight=1)
        outer, text_body = card(left_pane, "Dicalcogenuro")
        outer.pack(fill="both", expand=True)
        self.tmd_text = scrolled_text(text_body, self.palette,
                                      self.fonts["mono"], height=14)

        right_pane = ttk.Frame(panes)
        panes.add(right_pane, weight=1)
        oxide_card, oxide_body = card(right_pane, "Óxidos e intercara")
        oxide_card.pack(fill="both", expand=True)
        self.oxide_text = scrolled_text(oxide_body, self.palette,
                                        self.fonts["mono"], height=14)

        results, results_body = card(right, "Resultados")
        results.pack(fill="x", pady=(PAD["sm"], 0))
        self.tmd_table = table(results_body, ["nombre"], height=6)
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
        )
        self.spectrum_list.pack(fill="both", expand=True, pady=(0, PAD["sm"]))
        self.spectrum_list.bind("<<ListboxSelect>>", self._on_select)

        row = ttk.Frame(listbody)
        row.pack(fill="x")
        ttk.Button(row, text="Abrir…", command=self._open_files).pack(
            side="left", padx=(0, PAD["xs"]))
        ttk.Button(row, text="Demo TMD", command=self._load_demo).pack(side="left")

        options, obody = card(parent, "Opciones")
        options.pack(fill="x", pady=(PAD["sm"], 0))
        ttk.Label(obody, text="Material:").pack(anchor="w")
        self.material_var = tk.StringVar(value="(identificar)")
        ttk.Combobox(
            obody, textvariable=self.material_var, width=22, state="readonly",
            values=["(identificar)", *_material_keys()],
        ).pack(fill="x", pady=(0, PAD["xs"]))
        self.oxides_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(obody, text="Buscar óxidos del metal",
                        variable=self.oxides_var).pack(anchor="w")
        hint(obody,
             "Las capas se cuentan por la SEPARACIÓN E₂g–A₁g, no por "
             "posiciones absolutas: al ser una diferencia, cualquier error "
             "común de calibración se cancela.   "
             "Ojo con la resolución: las bandas miden 2–6 cm⁻¹ y las fronteras "
             "entre capas están a 2–3 cm⁻¹.   "
             "En WSe₂ los dos modos son casi degenerados y el método no vale; "
             "allí se cuenta por el modo B¹₂g.   "
             "Para ver óxidos hay que medir hasta al menos 1050 cm⁻¹: sus "
             "líneas inequívocas (819 y 995 del MoO₃, 744 del MoO₂) están ahí "
             "arriba, y sin esa región «no hay óxido» no significa nada.",
             wrap=250)

        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=(PAD["sm"], 0))
        ttk.Button(actions, text="Analizar como TMD", style="Accent.TButton",
                   command=self._analyse).pack(fill="x", pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Analizar todos",
                   command=self._analyse_all).pack(fill="x", pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Guardar informe…",
                   command=self._save_report).pack(fill="x")

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
        self._redraw()

    def _on_select(self, _event=None) -> None:
        selection = self.spectrum_list.curselection()
        if not selection:
            return
        self.session.current = int(selection[0])
        self._redraw()

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
            "Espectros TMD de demostración cargados, incluidas tres "
            "heteroestructuras óxido/calcogenuro. Son SINTÉTICOS."
        )

    def _analyse(self) -> None:
        if self.session.active is None:
            self.warn("Sin espectro", "Carga y selecciona un espectro primero.")
            return
        choice = self.material_var.get()
        material = None if choice.startswith("(") else choice

        def done(result) -> None:
            self.flush_messages(self.session.messages)
            self.refresh()
            if result is not None:
                self.set_status(
                    f"{result.label} — {result.layers or 'capas indeterminadas'}, "
                    f"fase {result.phase}"
                )

        self.run_async(
            lambda: self.session.analyse_tmd_active(material), done,
            "Analizando como TMD…",
        )

    def _analyse_all(self) -> None:
        choice = self.material_var.get()
        material = None if choice.startswith("(") else choice

        def work():
            saved = self.session.current
            count = 0
            for index in range(len(self.session.spectra)):
                self.session.current = index
                if self.session.analyse_tmd_active(material) is not None:
                    count += 1
            self.session.current = saved
            return count

        def done(count) -> None:
            self.flush_messages(self.session.messages)
            self.refresh()
            self.set_status(f"{count} espectro(s) analizados como TMD.")

        self.run_async(work, done, "Analizando el lote como TMD…")

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
        Path(path).write_text(item.tmd_result.summary(), encoding="utf-8")
        self.set_status(f"Informe guardado en {path}")

    # ==================================================================
    # drawing
    # ==================================================================
    def _redraw(self) -> None:
        self.mark_dirty("tmd")
        self.flush_dirty()
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
            "mucho más estrechas que las del carbono.",
        )
        oxides = result.oxides if result is not None else None
        set_text(
            self.oxide_text,
            oxides.summary() if oxides is not None else
            "Aquí aparecen los óxidos del metal que acompañen al calcogenuro "
            "(MoO₃, MoO₂, WO₃…), el índice de oxidación y lo que el "
            "desplazamiento de los modos dice sobre la intercara.\n\n"
            "Raman NO ve topología: un MoO₃@MoSe₂ y una mezcla de polvos dan "
            "el mismo espectro puntual. Lo que sí se mide es la deformación "
            "del calcogenuro, que una intercara íntima produce y una mezcla no.",
        )
        columns, rows = self.session.tmd_table()
        if columns:
            fill_table(self.tmd_table, columns, rows)

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


def _material_keys() -> list[str]:
    from ..analysis.tmd import tmd_materials

    return [m.key for m in tmd_materials()]


__all__ = ["TMDApp"]
