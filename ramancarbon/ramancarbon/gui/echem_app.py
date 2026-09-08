"""The electrochemistry section: CV, GCD, EIS and electrocatalysis.

One session is one electrode, and the tabs are its measurements. That is
the arrangement the analysis wants: the mechanism classification gets
stronger the more measurements it can see, and the cross-checks between
them — two capacitances that disagree, two resistances that disagree — are
the most useful thing the section produces, and they need the measurements
side by side to exist at all.

The mechanism verdict is deliberately shown first and at the top of the
report, because it decides whether the numbers below it are the right ones
for the material.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..echem.curve import Electrode
from .base import SectionApp, placeholder
from .echem_state import EchemSession
from .theme import PAD
from .widgets import card, fill_table, hint, labelled, scrolled_text, set_text, table

FILE_TYPES = (
    ("Exportaciones de potenciostato", "*.txt *.csv *.dat *.mpt *.DTA *.dta *.asc"),
    ("Todos", "*.*"),
)


class EchemApp(SectionApp):
    """The electrochemistry section of the suite."""

    def __init__(self, root, container, palette, fonts) -> None:
        super().__init__(root, container, palette, fonts)
        self.session = EchemSession()
        self._drawers = {
            "cv": self._draw_cv,
            "rates": self._draw_rates,
            "gcd": self._draw_gcd,
            "nyquist": self._draw_nyquist,
            "bode": self._draw_bode,
            "tafel": self._draw_tafel,
        }
        self._tab_canvases = {
            0: (),
            1: ("cv", "rates"),
            2: ("gcd",),
            3: ("nyquist", "bode"),
            4: ("tafel",),
        }
        self._build()
        self.set_status("Carga una medida o pulsa Demo.")

    # ==================================================================
    # layout
    # ==================================================================
    def _build(self) -> None:
        ttk = self.ttk
        body = ttk.Frame(self.container, padding=(PAD["md"], PAD["sm"]))
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, width=300)
        sidebar.pack(side="left", fill="y", padx=(0, PAD["md"]))
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side="left", fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._build_tab_summary()
        self._build_tab_cv()
        self._build_tab_gcd()
        self._build_tab_eis()
        self._build_tab_catalysis()
        self.build_status(self.container)
        self.root.after(150, self.drain_queue)

    def _build_sidebar(self, parent) -> None:
        ttk, tk = self.ttk, self.tk

        electrode, ebody = card(parent, "Electrodo")
        electrode.pack(fill="x")
        self.name_var = tk.StringVar(value="muestra")
        labelled(ebody, "Nombre", lambda p: ttk.Entry(p, textvariable=self.name_var,
                                                      width=14))
        self.mass_var = tk.StringVar(value="")
        labelled(ebody, "Masa activa (mg)", lambda p: ttk.Entry(
            p, textvariable=self.mass_var, width=10))
        self.area_var = tk.StringVar(value="1.0")
        labelled(ebody, "Área (cm²)", lambda p: ttk.Entry(
            p, textvariable=self.area_var, width=10))
        self.reference_var = tk.StringVar(value="Ag/AgCl_3M")
        labelled(ebody, "Referencia", lambda p: ttk.Combobox(
            p, textvariable=self.reference_var, width=14, state="readonly",
            values=_reference_keys()))
        self.ph_var = tk.StringVar(value="")
        labelled(ebody, "pH", lambda p: ttk.Entry(p, textvariable=self.ph_var,
                                                  width=10))
        self.resistance_var = tk.StringVar(value="")
        labelled(ebody, "R no compensada (Ω)", lambda p: ttk.Entry(
            p, textvariable=self.resistance_var, width=10))
        hint(ebody,
             "La masa es la de material ACTIVO, no la del electrodo: el "
             "aglomerante y el carbón conductor suelen ser el 20 %.   "
             "Sin pH no se puede pasar a RHE y no hay sobrepotencial. Sin "
             "resistencia no hay corrección óhmica, y la pendiente de Tafel "
             "sale grande de más.",
             wrap=260)

        loader, lbody = card(parent, "Medidas")
        loader.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self.loaded_list = tk.Listbox(
            lbody, height=7, exportselection=False, activestyle="none",
            background=self.palette.surface_alt, foreground=self.palette.text,
            selectbackground=self.palette.accent,
            selectforeground=self.palette.accent_text,
            highlightthickness=0, borderwidth=0, font=self.fonts["small"],
        )
        self.loaded_list.pack(fill="both", expand=True, pady=(0, PAD["sm"]))

        self.scan_rate_var = tk.StringVar(value="20")
        labelled(lbody, "Velocidad (mV/s)", lambda p: ttk.Entry(
            p, textvariable=self.scan_rate_var, width=10))
        self.current_var = tk.StringVar(value="1.0")
        labelled(lbody, "Corriente GCD (mA)", lambda p: ttk.Entry(
            p, textvariable=self.current_var, width=10))

        for text, command in (
            ("Cargar CV…", lambda: self._load("cv")),
            ("Añadir a la serie…", lambda: self._load("rates")),
            ("Cargar GCD…", lambda: self._load("gcd")),
            ("Cargar EIS…", lambda: self._load("eis")),
            ("Cargar polarización…", lambda: self._load("lsv")),
        ):
            ttk.Button(lbody, text=text, command=command).pack(
                fill="x", pady=(PAD["xs"], 0))

        options, obody = card(parent, "Opciones")
        options.pack(fill="x", pady=(PAD["sm"], 0))
        self.reaction_var = tk.StringVar(value="OER")
        labelled(obody, "Reacción", lambda p: ttk.Combobox(
            p, textvariable=self.reaction_var, width=8, state="readonly",
            values=["OER", "HER"]))
        self.circuit_var = tk.StringVar(value="randles_cpe")
        labelled(obody, "Circuito", lambda p: ttk.Combobox(
            p, textvariable=self.circuit_var, width=16,
            values=self.session.circuit_choices()))
        self.non_faradaic_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            obody, text="La ventana del CV no tiene corriente faradaica",
            variable=self.non_faradaic_var,
        ).pack(anchor="w", pady=(PAD["xs"], 0))

        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=(PAD["sm"], 0))
        ttk.Button(actions, text="Analizar", style="Accent.TButton",
                   command=self._analyse).pack(fill="x", pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Demo", command=self._load_demo).pack(
            fill="x", pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Guardar informe…",
                   command=self._save_report).pack(fill="x")

    def _build_tab_summary(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Resumen  ")
        outer, body = card(tab, "Resultados")
        outer.pack(fill="x")
        self.summary_table = table(body, ["magnitud", "valor", "nota"], height=14)
        report, report_body = card(tab, "Informe")
        report.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self.report_text = scrolled_text(report_body, self.palette,
                                         self.fonts["mono"], height=22)

    def _build_tab_cv(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Voltamperometría  ")
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", pady=(0, PAD["sm"]))
        self.normalise_var = self.tk.BooleanVar(value=False)
        ttk.Checkbutton(
            toolbar, text="Dividir la corriente por la velocidad",
            variable=self.normalise_var, command=self._redraw,
        ).pack(side="left")
        hint(toolbar,
             "  Al dividir por ν, las curvas de un condensador ideal se "
             "superponen exactamente. Lo que no se superponga es la parte que "
             "no es capacitiva.",
             wrap=700)
        outer, body = card(tab, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "cv", lambda f: f.add_subplot(111))
        rates_card, rates_body = card(tab, None)
        rates_card.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self.make_canvas(rates_body, "rates",
                         lambda f: (f.add_subplot(121), f.add_subplot(122)),
                         figsize=(7.6, 3.0))

    def _build_tab_gcd(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Carga-descarga  ")
        outer, body = card(tab, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "gcd", lambda f: f.add_subplot(111))
        info, info_body = card(tab, "Ramas")
        info.pack(fill="x", pady=(PAD["sm"], 0))
        self.branch_table = table(
            info_body,
            ["rama", "duración (s)", "V inicio", "V fin", "I (mA)", "IR (mV)",
             "q (mC)", "R² lineal"],
            height=8,
        )

    def _build_tab_eis(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Impedancia  ")
        outer, body = card(tab, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "nyquist", lambda f: f.add_subplot(111),
                         figsize=(6.0, 5.0))
        bode_card, bode_body = card(tab, None)
        bode_card.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self.make_canvas(bode_body, "bode",
                         lambda f: (f.add_subplot(121), f.add_subplot(122)),
                         figsize=(7.6, 3.0))
        info, info_body = card(tab, "Parámetros del circuito")
        info.pack(fill="x", pady=(PAD["sm"], 0))
        self.circuit_table = table(info_body,
                                   ["parámetro", "valor", "incertidumbre"],
                                   height=7)
        hint(info_body,
             "Kramers-Kronig va ANTES que el circuito: un ajuste a datos que "
             "han derivado da parámetros sin significado y el χ² no lo delata. "
             "Y la Q de un CPE no son faradios.",
             wrap=900)

    def _build_tab_catalysis(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  HER / OER  ")
        outer, body = card(tab, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "tafel", lambda f: f.add_subplot(111))
        info, info_body = card(tab, "Actividad")
        info.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self.catalysis_text = scrolled_text(info_body, self.palette,
                                            self.fonts["mono"], height=12)

    # ==================================================================
    # events
    # ==================================================================
    def _electrode_from_widgets(self) -> Electrode:
        def number(variable) -> Optional[float]:
            text = variable.get().strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None

        return Electrode(
            mass_mg=number(self.mass_var),
            area_cm2=number(self.area_var),
            reference=self.reference_var.get(),
            ph=number(self.ph_var),
            resistance_ohm=number(self.resistance_var),
            label=self.name_var.get(),
        )

    def _settings_from_widgets(self) -> None:
        self.session.name = self.name_var.get() or "muestra"
        self.session.electrode = self._electrode_from_widgets()
        self.session.reaction = self.reaction_var.get()
        self.session.circuit = self.circuit_var.get()
        self.session.non_faradaic = bool(self.non_faradaic_var.get())
        self.session.retag_electrode()

    def _load(self, kind: str) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Abrir medida", filetypes=FILE_TYPES, parent=self.root
        )
        if not path:
            return
        self._settings_from_widgets()
        rate = _number(self.scan_rate_var.get())
        current = _number(self.current_var.get())
        ok = False
        if kind in ("cv", "rates"):
            ok = self.session.load_cv(
                path,
                scan_rate=(rate / 1000.0) if rate else None,
                into_series=kind == "rates",
            )
        elif kind == "gcd":
            ok = self.session.load_gcd(
                path, current=(current / 1000.0) if current else None
            )
        elif kind == "eis":
            ok = self.session.load_eis(path)
        elif kind == "lsv":
            ok = self.session.load_lsv(path)
        self.flush_messages(self.session.messages)
        if ok:
            self.set_status(f"Cargado {Path(path).name}")
        self._refresh_loaded()

    def _load_demo(self) -> None:
        from ..examples.demo_data import (
            cv_rate_series,
            make_cv_demo,
            make_eis_demo,
            make_gcd_demo,
            make_lsv_demo,
        )

        self.mass_var.set("2.0")
        self.area_var.set("1.0")
        self.ph_var.set("14")
        self.resistance_var.set("2.0")
        self.name_var.set("demo pseudocondensador")
        self.circuit_var.set("R0-(R1|Q1)-Q2")
        self._settings_from_widgets()
        self.session.add_curves(
            cv=make_cv_demo("pseudocondensador", seed=1),
            rate_series=cv_rate_series("pseudocondensador", seed=2),
            gcd=make_gcd_demo("pseudocondensador", seed=1),
            eis=make_eis_demo("R0-(R1|Q1)-Q2", seed=3),
            lsv=make_lsv_demo("OER", seed=1),
        )
        self._refresh_loaded()
        self.set_status(
            "Medidas de demostración cargadas. Son SINTÉTICAS, generadas de "
            "la física que se quiere probar."
        )

    def _refresh_loaded(self) -> None:
        self.loaded_list.delete(0, "end")
        loaded = self.session.loaded
        if not loaded:
            self.loaded_list.insert("end", "(ninguna medida)")
            return
        for kind, description in loaded.items():
            self.loaded_list.insert("end", f"{kind}: {description[:70]}")

    def _analyse(self) -> None:
        self._settings_from_widgets()
        if not self.session.has_data:
            self.warn("Sin datos", "Carga al menos una medida.")
            return

        def done(result) -> None:
            self.flush_messages(self.session.messages)
            self._fill_tables()
            self._redraw()
            if result is not None and result.storage is not None:
                self.set_status(
                    f"{result.storage.label} — informar en "
                    f"{result.storage.report_as}"
                )

        self.run_async(self.session.analyse, done, "Analizando…")

    def _save_report(self) -> None:
        from tkinter import filedialog

        if self.session.result is None:
            self.warn("Sin análisis", "Pulsa Analizar primero.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar informe", defaultextension=".txt",
            initialfile=f"{self.session.name}_electroquimica.txt", parent=self.root,
        )
        if not path:
            return
        Path(path).write_text(self.session.report(), encoding="utf-8")
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

    def _fill_tables(self) -> None:
        fill_table(self.summary_table, ["magnitud", "valor", "nota"],
                   self.session.summary_rows())
        fill_table(self.circuit_table, ["parámetro", "valor", "incertidumbre"],
                   self.session.circuit_rows())
        set_text(self.report_text, self.session.report())
        result = self.session.result
        rows = []
        if result is not None and result.gcd is not None:
            for branch in result.gcd.branches:
                rows.append(
                    (
                        branch.kind,
                        f"{branch.duration_s:.2f}",
                        f"{branch.v_start:+.4f}",
                        f"{branch.v_end:+.4f}",
                        f"{1e3 * branch.current_a:+.4g}",
                        f"{1e3 * branch.ir_drop_v:.1f}",
                        f"{1e3 * branch.charge_c:.4g}",
                        f"{branch.linearity:.5f}",
                    )
                )
        fill_table(
            self.branch_table,
            ["rama", "duración (s)", "V inicio", "V fin", "I (mA)", "IR (mV)",
             "q (mC)", "R² lineal"],
            rows,
        )
        set_text(
            self.catalysis_text,
            result.catalysis.summary() if result and result.catalysis
            else "Carga una curva de polarización y elige HER u OER.",
        )

    def _draw_cv(self, figure) -> None:
        from .plots_echem import plot_cv

        ax = figure.add_subplot(111)
        curves = [c for c in ([self.session.cv] if self.session.cv else [])]
        curves.extend(self.session.rate_series)
        if not curves:
            placeholder(ax, "Carga un voltamperograma", self.palette)
            return
        result = self.session.result
        plot_cv(
            ax, curves, self.palette,
            peaks=result.cv.peaks if result and result.cv else None,
            normalise_by_rate=bool(self.normalise_var.get()),
        )

    def _draw_rates(self, figure) -> None:
        from .plots_echem import plot_b_values, plot_rate_capacitance

        left = figure.add_subplot(121)
        right = figure.add_subplot(122)
        result = self.session.result
        if result is None or result.rates is None:
            placeholder(left, "Carga una serie de velocidades", self.palette)
            placeholder(right, "", self.palette)
            return
        plot_rate_capacitance(left, result.rates, self.palette)
        plot_b_values(right, result.rates, self.palette)

    def _draw_gcd(self, figure) -> None:
        from .plots_echem import plot_gcd

        ax = figure.add_subplot(111)
        if self.session.gcd is None:
            placeholder(ax, "Carga una curva de carga-descarga", self.palette)
            return
        result = self.session.result
        plot_gcd(ax, self.session.gcd, self.palette,
                 result.gcd if result else None)

    def _draw_nyquist(self, figure) -> None:
        from .plots_echem import plot_nyquist

        ax = figure.add_subplot(111)
        if self.session.eis is None:
            placeholder(ax, "Carga un espectro de impedancia", self.palette)
            return
        result = self.session.result
        plot_nyquist(ax, self.session.eis, self.palette,
                     result.eis if result else None)

    def _draw_bode(self, figure) -> None:
        from .plots_echem import plot_bode, plot_kk_residuals

        left = figure.add_subplot(121)
        right = figure.add_subplot(122)
        if self.session.eis is None:
            placeholder(left, "", self.palette)
            placeholder(right, "", self.palette)
            return
        result = self.session.result
        plot_bode(left, self.session.eis, self.palette, result.eis if result else None)
        if result is not None and result.eis is not None:
            plot_kk_residuals(right, result.eis, self.palette)
        else:
            placeholder(right, "Analiza para ver Kramers-Kronig", self.palette)

    def _draw_tafel(self, figure) -> None:
        from .plots_echem import plot_tafel

        ax = figure.add_subplot(111)
        result = self.session.result
        curve = self.session.lsv
        if result is None or result.catalysis is None or curve is None:
            placeholder(ax, "Carga una curva de polarización", self.palette)
            return
        catalysis = result.catalysis
        rhe, _ = curve.electrode.to_rhe(curve.potential)
        if rhe is None:
            placeholder(ax, "Hace falta el pH para pasar a RHE", self.palette)
            return
        corrected, _ = curve.electrode.ir_correct(rhe, curve.current)
        equilibrium = 1.23 if catalysis.reaction == "OER" else 0.0
        sign = 1.0 if catalysis.reaction == "OER" else -1.0
        area = curve.electrode.area_cm2 or 1.0
        plot_tafel(
            ax, catalysis, self.palette,
            overpotential=sign * (corrected - equilibrium),
            current_density=1e3 * curve.current / area,
        )


def _number(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _reference_keys() -> list[str]:
    from ..echem.curve import load_echem_database

    return sorted(load_echem_database()["reference_electrodes"])


__all__ = ["FILE_TYPES", "EchemApp"]
